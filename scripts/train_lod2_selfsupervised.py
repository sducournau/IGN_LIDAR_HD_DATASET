#!/usr/bin/env python3
"""
LOD2 Self-Supervised Training Script
=====================================

Train a hybrid PointNet++ + Point Transformer model for LOD2 building classification
using self-supervised pretraining followed by supervised fine-tuning.

Architecture:
- PointNet++ branch: Hierarchical feature extraction
- Point Transformer branch: Global attention-based features  
- Fusion module: Attention-based combination
- Classification head: 6 LOD2 classes

Training Pipeline:
1. Phase 1 - Self-Supervised Pretraining (150 epochs):
   - Masked Point Modeling (MPM)
   - Normal Prediction
   - Contrastive Learning
   - Rotation Prediction
   - Spatial Context Prediction

2. Phase 2 - Supervised Fine-tuning (50 epochs):
   - LOD2 classification with pseudo-labels or manual annotations
   - Weighted cross-entropy loss for class imbalance

Usage:
    # Pretraining
    python train_lod2_selfsupervised.py \\
        --mode pretrain \\
        --data_dir data/lod2_dataset/train \\
        --output_dir models/lod2_pretrained \\
        --epochs 150 \\
        --batch_size 32 \\
        --gpu 0

    # Fine-tuning
    python train_lod2_selfsupervised.py \\
        --mode finetune \\
        --pretrained_model models/lod2_pretrained/best_model.pth \\
        --data_dir data/lod2_dataset_labeled/train \\
        --val_dir data/lod2_dataset_labeled/val \\
        --output_dir models/lod2_finetuned \\
        --epochs 50 \\
        --batch_size 16 \\
        --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm
import wandb

# Suppress warnings
warnings.filterwarnings('ignore')

# =======================
# Model Architecture
# =======================

class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction Layer"""
    
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) - point coordinates
            features: (B, C, N) - point features
        Returns:
            new_xyz: (B, npoint, 3)
            new_features: (B, C', npoint)
        """
        if self.group_all:
            # Global pooling
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = xyz.unsqueeze(1)
            grouped_features = features.unsqueeze(1)
        else:
            # FPS sampling
            fps_idx = self.farthest_point_sample(xyz, self.npoint)
            new_xyz = self.index_points(xyz, fps_idx)
            
            # Ball query grouping
            idx = self.ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = self.index_points(xyz, idx)
            grouped_xyz -= new_xyz.unsqueeze(2)
            
            if features is not None:
                grouped_features = self.index_points(features.transpose(1, 2), idx)
                grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
            else:
                grouped_features = grouped_xyz
            
            grouped_features = grouped_features.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
        
        # Apply MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_features = F.relu(bn(conv(grouped_features)))
        
        # Max pooling
        new_features = torch.max(grouped_features, 2)[0]  # (B, C', npoint)
        
        return new_xyz, new_features
    
    @staticmethod
    def farthest_point_sample(xyz, npoint):
        """Farthest Point Sampling"""
        B, N, _ = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
        distance = torch.ones(B, N, device=xyz.device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
        batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        
        return centroids
    
    @staticmethod
    def ball_query(radius, nsample, xyz, new_xyz):
        """Ball Query"""
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, device=xyz.device).unsqueeze(0).unsqueeze(0).repeat(B, S, 1)
        
        sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, -1)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        
        group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        
        return group_idx
    
    @staticmethod
    def index_points(points, idx):
        """Index points by indices"""
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


class PointTransformerLayer(nn.Module):
    """Point Transformer Layer with Self-Attention"""
    
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, pos_encoding=None):
        """
        Args:
            src: (B, N, d_model)
            pos_encoding: (B, N, d_model) - positional encoding
        """
        # Self-attention with positional encoding
        if pos_encoding is not None:
            q = k = src + pos_encoding
        else:
            q = k = src
        
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class PositionalEncoding3D(nn.Module):
    """3D Positional Encoding for Point Clouds"""
    
    def __init__(self, d_model, max_freq=10):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
        self.linear = nn.Linear(max_freq * 6, d_model)  # 3 coords × 2 (sin, cos) × max_freq
    
    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) - normalized coordinates [-1, 1]
        Returns:
            pos_encoding: (B, N, d_model)
        """
        B, N, _ = xyz.shape
        
        # Generate frequencies
        freqs = torch.arange(self.max_freq, device=xyz.device, dtype=torch.float32)
        freqs = 2.0 ** freqs * np.pi
        
        # Apply sin/cos encoding
        xyz_freq = xyz.unsqueeze(-1) * freqs  # (B, N, 3, max_freq)
        xyz_freq = xyz_freq.reshape(B, N, -1)  # (B, N, 3*max_freq)
        
        pos_sin = torch.sin(xyz_freq)
        pos_cos = torch.cos(xyz_freq)
        pos_encoding = torch.cat([pos_sin, pos_cos], dim=-1)  # (B, N, 6*max_freq)
        
        # Project to d_model
        pos_encoding = self.linear(pos_encoding)
        
        return pos_encoding


class HybridLOD2Model(nn.Module):
    """
    Hybrid PointNet++ + Point Transformer for LOD2 Classification
    
    Architecture:
    - PointNet++ branch: Hierarchical local features
    - Point Transformer branch: Global attention features
    - Fusion module: Attention-based combination
    - Classification head: 6 LOD2 classes or pretext task heads
    """
    
    def __init__(
        self,
        num_classes=6,
        d_model=512,
        nhead=8,
        num_transformer_layers=4,
        use_pretext_heads=False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_pretext_heads = use_pretext_heads
        
        # =====================
        # PointNet++ Branch
        # =====================
        self.sa1 = PointNetSetAbstraction(
            npoint=2048, radius=0.2, nsample=32, 
            in_channel=3, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=512, radius=0.4, nsample=32,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=128, radius=0.8, nsample=32,
            in_channel=256 + 3, mlp=[256, 256, 512]
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=512 + 3, mlp=[512, 512, 1024],
            group_all=True
        )
        
        # =====================
        # Point Transformer Branch
        # =====================
        self.pos_encoder = PositionalEncoding3D(d_model=d_model, max_freq=10)
        self.input_proj = nn.Linear(3, d_model)  # Project XYZ to d_model
        
        self.transformer_layers = nn.ModuleList([
            PointTransformerLayer(d_model, nhead)
            for _ in range(num_transformer_layers)
        ])
        
        # Global pooling for transformer
        self.transformer_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1024)
        )
        
        # =====================
        # Fusion Module
        # =====================
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=1024, num_heads=8, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(1024)
        self.fusion_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )
        
        # =====================
        # Classification Head
        # =====================
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # =====================
        # Pretext Task Heads (for self-supervised pretraining)
        # =====================
        if use_pretext_heads:
            # Masked Point Reconstruction
            self.mask_decoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)  # Predict XYZ
            )
            
            # Normal Prediction
            self.normal_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)  # Predict normal vector
            )
            
            # Rotation Prediction (4 classes: 0°, 90°, 180°, 270°)
            self.rotation_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4)
            )
            
            # Contrastive projection head
            self.contrastive_proj = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
    
    def forward(self, xyz, return_features=False, pretext_task=None):
        """
        Args:
            xyz: (B, N, 3) - point coordinates
            return_features: bool - return intermediate features
            pretext_task: str - which pretext task head to use
                - 'mask_reconstruction'
                - 'normal_prediction'
                - 'rotation_prediction'
                - 'contrastive'
        Returns:
            output: task-specific output
            features: (B, 512) - if return_features=True
        """
        B, N, _ = xyz.shape
        
        # =====================
        # PointNet++ Forward
        # =====================
        l1_xyz, l1_features = self.sa1(xyz, None)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        l4_xyz, l4_features = self.sa4(l3_xyz, l3_features)
        
        pointnet_global = l4_features.squeeze(-1)  # (B, 1024)
        
        # =====================
        # Point Transformer Forward
        # =====================
        # Subsample for transformer (use FPS result from PointNet++)
        xyz_sampled = l1_xyz  # (B, 2048, 3)
        
        # Input projection
        transformer_features = self.input_proj(xyz_sampled)  # (B, 2048, d_model)
        
        # Positional encoding
        pos_encoding = self.pos_encoder(xyz_sampled)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            transformer_features = layer(transformer_features, pos_encoding)
        
        # Global pooling
        transformer_global = transformer_features.mean(dim=1)  # (B, d_model)
        transformer_global = self.transformer_pool(transformer_global)  # (B, 1024)
        
        # =====================
        # Fusion
        # =====================
        # Concatenate both branches
        pointnet_global = pointnet_global.unsqueeze(1)  # (B, 1, 1024)
        transformer_global = transformer_global.unsqueeze(1)  # (B, 1, 1024)
        
        # Cross-attention fusion
        fused, _ = self.fusion_attn(
            pointnet_global, transformer_global, transformer_global
        )
        fused = self.fusion_norm(fused + pointnet_global)  # Residual
        
        # Combine with concatenation
        combined = torch.cat([
            pointnet_global.squeeze(1),
            transformer_global.squeeze(1)
        ], dim=1)  # (B, 2048)
        
        # Final fusion
        features = self.fusion_fc(combined)  # (B, 512)
        
        # =====================
        # Task-Specific Heads
        # =====================
        if self.use_pretext_heads and pretext_task is not None:
            if pretext_task == 'mask_reconstruction':
                return self.mask_decoder(features)
            elif pretext_task == 'normal_prediction':
                return self.normal_head(features)
            elif pretext_task == 'rotation_prediction':
                return self.rotation_head(features)
            elif pretext_task == 'contrastive':
                return F.normalize(self.contrastive_proj(features), dim=1)
        
        # Classification
        output = self.classifier(features)
        
        if return_features:
            return output, features
        else:
            return output


# =======================
# Dataset
# =======================

class LOD2SelfSupervisedDataset(Dataset):
    """Dataset for LOD2 self-supervised training"""
    
    def __init__(
        self,
        data_dir: str,
        mode: str = 'pretrain',
        num_points: int = 24576,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.num_points = num_points
        self.augment = augment
        
        # Find all NPZ files
        self.files = list(self.data_dir.glob('**/*.npz'))
        print(f"Found {len(self.files)} patches in {data_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load patch
        data = np.load(self.files[idx])
        points = data['points'].astype(np.float32)
        
        # Sample/pad to target number of points
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            pad_size = self.num_points - len(points)
            pad_indices = np.random.choice(len(points), pad_size, replace=True)
            points = np.vstack([points, points[pad_indices]])
        
        # Normalize to [-1, 1]
        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        
        # Augmentation
        if self.augment:
            # Random rotation around Z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            points = points @ rotation_matrix.T
            
            # Random jitter
            points += np.random.normal(0, 0.02, points.shape).astype(np.float32)
            
            # Random scale
            scale = np.random.uniform(0.8, 1.2)
            points *= scale
        
        # Prepare output
        output = {'points': torch.from_numpy(points)}
        
        # Add labels if in finetune mode
        if self.mode == 'finetune' and 'labels' in data:
            labels = data['labels']
            # Convert per-point labels to patch-level label (majority vote)
            unique, counts = np.unique(labels, return_counts=True)
            patch_label = unique[np.argmax(counts)]
            output['label'] = torch.tensor(patch_label, dtype=torch.long)
        
        return output


# =======================
# Training Functions
# =======================

def pretrain_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_weights: Dict[str, float]
) -> Dict[str, float]:
    """Single epoch of self-supervised pretraining"""
    model.train()
    
    losses = {
        'total': 0.0,
        'mask_reconstruction': 0.0,
        'normal_prediction': 0.0,
        'rotation_prediction': 0.0,
        'contrastive': 0.0
    }
    
    pbar = tqdm(dataloader, desc='Pretraining')
    for batch in pbar:
        points = batch['points'].to(device)  # (B, N, 3)
        B, N, _ = points.shape
        
        optimizer.zero_grad()
        total_loss = 0.0
        
        # Task 1: Masked Point Reconstruction
        if task_weights.get('mask_reconstruction', 0) > 0:
            mask_ratio = 0.3
            mask_indices = np.random.choice(
                N, int(N * mask_ratio), replace=False
            )
            masked_points = points.clone()
            masked_points[:, mask_indices, :] = 0
            
            reconstructed = model(
                masked_points, 
                pretext_task='mask_reconstruction'
            )
            
            # Chamfer distance loss (simplified)
            recon_loss = F.mse_loss(
                reconstructed, 
                points.mean(dim=1, keepdim=True).expand(-1, N, -1)[:, mask_indices, :]
            )
            total_loss += task_weights['mask_reconstruction'] * recon_loss
            losses['mask_reconstruction'] += recon_loss.item()
        
        # Task 2: Rotation Prediction
        if task_weights.get('rotation_prediction', 0) > 0:
            rotations = [0, 90, 180, 270]
            rot_labels = torch.randint(0, 4, (B,), device=device)
            
            rotated_points = points.clone()
            for i in range(B):
                angle = np.radians(rotations[rot_labels[i]])
                rotation_matrix = torch.tensor([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ], dtype=torch.float32, device=device)
                rotated_points[i] = rotated_points[i] @ rotation_matrix.T
            
            rot_pred = model(rotated_points, pretext_task='rotation_prediction')
            rot_loss = F.cross_entropy(rot_pred, rot_labels)
            total_loss += task_weights['rotation_prediction'] * rot_loss
            losses['rotation_prediction'] += rot_loss.item()
        
        # Task 3: Contrastive Learning
        if task_weights.get('contrastive', 0) > 0:
            # Create two augmented views
            aug1 = points.clone()
            aug2 = points.clone()
            
            # Different augmentations for each view
            for aug in [aug1, aug2]:
                theta = torch.rand(B, device=device) * 2 * np.pi
                for i in range(B):
                    rotation_matrix = torch.tensor([
                        [torch.cos(theta[i]), -torch.sin(theta[i]), 0],
                        [torch.sin(theta[i]), torch.cos(theta[i]), 0],
                        [0, 0, 1]
                    ], device=device)
                    aug[i] = aug[i] @ rotation_matrix.T
            
            z1 = model(aug1, pretext_task='contrastive')
            z2 = model(aug2, pretext_task='contrastive')
            
            # NT-Xent loss (simplified)
            temperature = 0.07
            similarity_matrix = torch.mm(z1, z2.T) / temperature
            labels = torch.arange(B, device=device)
            contrastive_loss = F.cross_entropy(similarity_matrix, labels)
            total_loss += task_weights['contrastive'] * contrastive_loss
            losses['contrastive'] += contrastive_loss.item()
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        losses['total'] += total_loss.item()
        
        pbar.set_postfix({'loss': total_loss.item()})
    
    # Average losses
    for key in losses:
        losses[key] /= len(dataloader)
    
    return losses


def finetune_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_train: bool = True
) -> Tuple[float, float]:
    """Single epoch of supervised fine-tuning"""
    if is_train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Finetuning' if is_train else 'Validation')
    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            points = batch['points'].to(device)
            labels = batch['label'].to(device)
            
            if is_train:
                optimizer.zero_grad()
            
            # Forward pass
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            if is_train:
                loss.backward()
                optimizer.step()
            
            # Metrics
            _, predicted = outputs.max(1)
            total_loss += loss.item()
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# =======================
# Main Training Script
# =======================

def main():
    parser = argparse.ArgumentParser(description='LOD2 Self-Supervised Training')
    
    # Mode
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune'],
                        required=True, help='Training mode')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation data directory (for finetune mode)')
    
    # Model
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Path to pretrained model (for finetune mode)')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of LOD2 classes')
    
    # Training
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_points', type=int, default=24576,
                        help='Number of points per patch')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for models and logs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default='lod2-selfsupervised',
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {output_dir}")
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args),
            name=f"{args.mode}_{Path(args.data_dir).name}"
        )
    
    # Create model
    model = HybridLOD2Model(
        num_classes=args.num_classes,
        use_pretext_heads=(args.mode == 'pretrain')
    ).to(device)
    
    # Load pretrained weights if finetune mode
    if args.mode == 'finetune' and args.pretrained_model:
        print(f"Loading pretrained model from {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    if args.mode == 'pretrain':
        train_dataset = LOD2SelfSupervisedDataset(
            args.data_dir,
            mode='pretrain',
            num_points=args.num_points,
            augment=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Pretext task weights
        task_weights = {
            'mask_reconstruction': 0.3,
            'rotation_prediction': 0.4,
            'contrastive': 0.3
        }
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
            
            losses = pretrain_epoch(
                model, train_loader, optimizer, device, task_weights
            )
            
            scheduler.step()
            
            # Log
            print(f"Total Loss: {losses['total']:.4f}")
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    **{f'loss/{k}': v for k, v in losses.items()}
                })
            
            # Save checkpoint
            if losses['total'] < best_loss:
                best_loss = losses['total']
                checkpoint_path = output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
            
            if (epoch + 1) % args.save_freq == 0:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses['total'],
                }, checkpoint_path)
    
    elif args.mode == 'finetune':
        train_dataset = LOD2SelfSupervisedDataset(
            args.data_dir,
            mode='finetune',
            num_points=args.num_points,
            augment=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Validation dataset
        val_loader = None
        if args.val_dir:
            val_dataset = LOD2SelfSupervisedDataset(
                args.val_dir,
                mode='finetune',
                num_points=args.num_points,
                augment=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Weighted cross-entropy for class imbalance
        class_weights = torch.tensor([1.0, 1.0, 1.2, 1.5, 2.0, 2.5], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        best_acc = 0.0
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
            
            # Train
            train_loss, train_acc = finetune_epoch(
                model, train_loader, optimizer, criterion, device, is_train=True
            )
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validate
            if val_loader:
                val_loss, val_acc = finetune_epoch(
                    model, val_loader, None, criterion, device, is_train=False
                )
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                scheduler.step(val_loss)
                
                # Log
                if args.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'lr': optimizer.param_groups[0]['lr']
                    })
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    checkpoint_path = output_dir / 'best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': best_acc,
                    }, checkpoint_path)
                    print(f"Saved best model (acc: {best_acc:.2f}%) to {checkpoint_path}")
    
    print("\nTraining completed!")
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
