"""
Transfer Learning Module for IGN LiDAR HD Dataset Processing Library.

Provides transfer learning capabilities for leveraging pre-trained models
and domain adaptation techniques to accelerate model training and improve
performance on LiDAR-based building classification tasks.

Components:
    - FeatureExtractor: Pre-trained model-based feature extraction
    - DomainAdapter: Domain adaptation techniques for LiDAR data
    - ProgressiveUnfreezing: Layer-wise unfreezing strategy
    - TransferLearningPipeline: Complete transfer learning workflow

Author: imagodata
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings
import logging

import numpy as np

# Optional PyTorch support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. Transfer learning features disabled. "
        "Install with: pip install torch",
        UserWarning
    )

logger = logging.getLogger(__name__)


@dataclass
class TransferConfig:
    """Configuration for transfer learning."""
    
    freeze_backbone: bool = True
    """Whether to freeze pre-trained layers initially."""
    
    learning_rate: float = 1e-3
    """Initial learning rate."""
    
    warmup_epochs: int = 5
    """Number of epochs to warm up before unfreezing."""
    
    unfreeze_strategy: str = "progressive"
    """Strategy for unfreezing: 'progressive', 'full', 'layer_wise'."""
    
    num_unfreeze_stages: int = 3
    """Number of stages for progressive unfreezing."""
    
    use_domain_adaptation: bool = True
    """Whether to apply domain adaptation."""
    
    domain_adaptation_weight: float = 0.1
    """Weight for domain adaptation loss."""
    
    batch_size: int = 32
    """Batch size for training."""
    
    num_epochs: int = 50
    """Total number of training epochs."""
    
    patience: int = 10
    """Early stopping patience."""
    
    device: str = "cuda"
    """Computation device ('cuda', 'cpu', 'mps')."""


class FeatureExtractor:
    """
    Extract features from pre-trained models for transfer learning.
    
    Provides efficient feature extraction from intermediate layers of
    pre-trained neural networks for use as input to downstream classifiers.
    """
    
    def __init__(
        self,
        model: "nn.Module",
        layer_name: str,
        device: str = "cuda"
    ):
        """
        Initialize feature extractor.
        
        Args:
            model: PyTorch model to extract features from
            layer_name: Name of layer to extract features from
            device: Computation device ('cuda', 'cpu', 'mps')
            
        Raises:
            ValueError: If layer not found in model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for transfer learning")
        
        self.model = model
        self.layer_name = layer_name
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Verify layer exists
        if not hasattr(model, layer_name):
            raise ValueError(f"Layer '{layer_name}' not found in model")
        
        self.hook_handle = None
        self.features = None
        self._register_hook()
    
    def _register_hook(self) -> None:
        """Register forward hook to capture intermediate features."""
        layer = getattr(self.model, self.layer_name)
        
        def hook_fn(module, input, output):
            self.features = output.detach()
        
        self.hook_handle = layer.register_forward_hook(hook_fn)
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from input data.
        
        Args:
            data: Input array [batch_size, features]
            
        Returns:
            Extracted features [batch_size, output_features]
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        with torch.no_grad():
            data_tensor = torch.from_numpy(data).float().to(self.device)
            _ = self.model(data_tensor)
            features = self.features.cpu().numpy()
        
        return features
    
    def batch_extract(
        self,
        data: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract features in batches for memory efficiency.
        
        Args:
            data: Input array [num_samples, features]
            batch_size: Batch size for processing
            
        Returns:
            Extracted features [num_samples, output_features]
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        num_samples = len(data)
        all_features = []
        
        for i in range(0, num_samples, batch_size):
            batch = data[i:i + batch_size]
            batch_features = self.extract(batch)
            all_features.append(batch_features)
        
        return np.concatenate(all_features, axis=0)
    
    def __del__(self):
        """Clean up hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()


class DomainAdapter:
    """
    Domain adaptation techniques for LiDAR-specific data.
    
    Implements domain adaptation losses to align source and target
    domain distributions for improved cross-domain generalization.
    """
    
    def __init__(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        method: str = "mmd",
        device: str = "cuda"
    ):
        """
        Initialize domain adapter.
        
        Args:
            source_features: Source domain features [num_source, features]
            target_features: Target domain features [num_target, features]
            method: Domain adaptation method ('mmd', 'coral', 'wasserstein')
            device: Computation device
            
        Raises:
            ValueError: If method not supported
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for domain adaptation")
        
        self.device = device
        self.method = method
        
        if method not in ["mmd", "coral", "wasserstein"]:
            raise ValueError(f"Unsupported method: {method}")
        
        self.source_features = torch.from_numpy(
            source_features
        ).float().to(device)
        self.target_features = torch.from_numpy(
            target_features
        ).float().to(device)
    
    def compute_mmd_loss(
        self,
        source: "torch.Tensor",
        target: "torch.Tensor",
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        sigma: float = 1.0
    ) -> "torch.Tensor":
        """
        Compute Maximum Mean Discrepancy (MMD) loss.
        
        Args:
            source: Source features [batch_size, features]
            target: Target features [batch_size, features]
            kernel_mul: Kernel bandwidth multiplier
            kernel_num: Number of kernels
            sigma: Kernel bandwidth
            
        Returns:
            MMD loss scalar
        """
        # Compute kernel matrix
        def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, sigma=1.0):
            total = torch.cat([x, y], dim=0)
            total0 = total.unsqueeze(0)
            total1 = total.unsqueeze(1)
            l2_distance = torch.pow(total0 - total1, 2).sum(2)
            
            bandwidth_list = [sigma * (kernel_mul ** (i - kernel_num // 2))
                            for i in range(kernel_num)]
            
            kernel_val = [torch.exp(-l2_distance / bandwidth)
                         for bandwidth in bandwidth_list]
            
            return sum(kernel_val) / len(kernel_val)
        
        kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, sigma)
        
        # Compute MMD
        batch_size_s = source.shape[0]
        batch_size_t = target.shape[0]
        
        K_ss = kernels[:batch_size_s, :batch_size_s]
        K_tt = kernels[batch_size_s:, batch_size_s:]
        K_st = kernels[:batch_size_s, batch_size_s:]
        
        mmd = (K_ss.sum() / (batch_size_s ** 2) +
               K_tt.sum() / (batch_size_t ** 2) -
               2 * K_st.sum() / (batch_size_s * batch_size_t))
        
        return torch.max(mmd, torch.tensor(0.0))
    
    def compute_coral_loss(
        self,
        source: "torch.Tensor",
        target: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute CORAL (Correlation Alignment) loss.
        
        Args:
            source: Source features [batch_size, features]
            target: Target features [batch_size, features]
            
        Returns:
            CORAL loss scalar
        """
        # Compute covariance matrices
        source_centered = source - source.mean(dim=0)
        target_centered = target - target.mean(dim=0)
        
        source_cov = source_centered.T @ source_centered
        target_cov = target_centered.T @ target_centered
        
        # Compute Frobenius norm
        coral_loss = torch.norm(source_cov - target_cov, p="fro") ** 2
        
        return coral_loss
    
    def compute_loss(
        self,
        source: "torch.Tensor",
        target: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute domain adaptation loss.
        
        Args:
            source: Source batch features
            target: Target batch features
            
        Returns:
            Domain adaptation loss
        """
        if self.method == "mmd":
            return self.compute_mmd_loss(source, target)
        elif self.method == "coral":
            return self.compute_coral_loss(source, target)
        else:
            return self.compute_mmd_loss(source, target)


class ProgressiveUnfreezing:
    """
    Progressive layer unfreezing strategy for transfer learning.
    
    Gradually unfreezes layers during training to preserve pre-trained
    knowledge while adapting to the target task.
    """
    
    def __init__(
        self,
        model: "nn.Module",
        num_stages: int = 3,
        warmup_epochs: int = 5
    ):
        """
        Initialize progressive unfreezing.
        
        Args:
            model: PyTorch model
            num_stages: Number of unfreezing stages
            warmup_epochs: Epochs before first unfreeze
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        self.num_stages = num_stages
        self.warmup_epochs = warmup_epochs
        
        # Get all named parameters
        self.param_groups = self._partition_parameters()
        
        # Freeze all initially
        for param in model.parameters():
            param.requires_grad = False
    
    def _partition_parameters(self) -> List[List[str]]:
        """
        Partition model parameters into stages.
        
        Returns:
            List of parameter names grouped by stage
        """
        param_names = [name for name, _ in self.model.named_parameters()]
        
        # Reverse to freeze from back to front
        param_names.reverse()
        
        stage_size = len(param_names) // self.num_stages
        
        return [
            param_names[i * stage_size:(i + 1) * stage_size]
            for i in range(self.num_stages)
        ]
    
    def unfreeze_stage(self, stage: int) -> None:
        """
        Unfreeze parameters in a specific stage.
        
        Args:
            stage: Stage index (0 to num_stages-1)
        """
        if stage < 0 or stage >= self.num_stages:
            raise ValueError(f"Invalid stage: {stage}")
        
        for param_name in self.param_groups[stage]:
            param = self._get_parameter(param_name)
            if param is not None:
                param.requires_grad = True
    
    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _get_parameter(self, param_name: str) -> Optional["torch.nn.Parameter"]:
        """Get parameter by name."""
        try:
            return dict(self.model.named_parameters())[param_name]
        except KeyError:
            return None
    
    def get_frozen_params(self) -> List[str]:
        """Get list of frozen parameters."""
        return [
            name for name, param in self.model.named_parameters()
            if not param.requires_grad
        ]


class TransferLearningPipeline:
    """
    Complete transfer learning pipeline for LiDAR classification.
    
    Orchestrates transfer learning including feature extraction, domain
    adaptation, progressive unfreezing, and training.
    """
    
    def __init__(
        self,
        model: "nn.Module",
        config: Optional[TransferConfig] = None
    ):
        """
        Initialize transfer learning pipeline.
        
        Args:
            model: Pre-trained PyTorch model
            config: Transfer learning configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        self.config = config or TransferConfig()
        self.device = self.config.device
        
        self.model.to(self.device)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "domain_loss": [],
            "best_epoch": 0,
            "best_loss": float("inf")
        }
    
    def prepare_transfer_learning(self) -> None:
        """Prepare model for transfer learning."""
        if self.config.freeze_backbone:
            # Freeze all but last layer
            for name, param in self.model.named_parameters():
                if "output" not in name and "fc" not in name:
                    param.requires_grad = False
    
    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        source_features: Optional[np.ndarray] = None,
        domain_adapter: Optional[DomainAdapter] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Train model with transfer learning.
        
        Args:
            train_features: Training features [num_train, features]
            train_labels: Training labels [num_train]
            val_features: Validation features
            val_labels: Validation labels
            source_features: Source domain features for adaptation
            domain_adapter: Domain adapter instance
            callbacks: List of callback functions
            
        Returns:
            Training history dictionary
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.prepare_transfer_learning()
        
        # Create data loaders
        train_tensor = TensorDataset(
            torch.from_numpy(train_features).float(),
            torch.from_numpy(train_labels).long()
        )
        train_loader = DataLoader(
            train_tensor,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Initialize optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        # Progressive unfreezing
        unfreezer = None
        if self.config.unfreeze_strategy == "progressive":
            unfreezer = ProgressiveUnfreezing(
                self.model,
                self.config.num_unfreeze_stages,
                self.config.warmup_epochs
            )
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Progressive unfreezing
            if unfreezer is not None:
                if epoch > self.config.warmup_epochs:
                    stage = min(
                        (epoch - self.config.warmup_epochs) //
                        (self.config.num_epochs // self.config.num_unfreeze_stages),
                        self.config.num_unfreeze_stages - 1
                    )
                    unfreezer.unfreeze_stage(stage)
            
            # Training
            self.model.train()
            train_loss = 0.0
            domain_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # Domain adaptation
                if domain_adapter is not None and source_features is not None:
                    source_batch = torch.from_numpy(
                        source_features[np.random.choice(
                            len(source_features), len(batch_features)
                        )]
                    ).float().to(self.device)
                    da_loss = domain_adapter.compute_loss(
                        source_batch, batch_features
                    )
                    loss = loss + self.config.domain_adaptation_weight * da_loss
                    domain_loss += da_loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            self.history["train_loss"].append(train_loss / len(train_loader))
            if domain_adapter is not None:
                self.history["domain_loss"].append(
                    domain_loss / len(train_loader)
                )
            
            # Validation
            if val_features is not None and val_labels is not None:
                val_loss = self._evaluate(val_features, val_labels, criterion)
                self.history["val_loss"].append(val_loss)
                
                # Early stopping
                if val_loss < self.history["best_loss"]:
                    self.history["best_loss"] = val_loss
                    self.history["best_epoch"] = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if callbacks:
                for callback in callbacks:
                    callback(epoch, self.history)
        
        return self.history
    
    def _evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        criterion: "nn.Module"
    ) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            features_tensor = torch.from_numpy(features).float().to(self.device)
            labels_tensor = torch.from_numpy(labels).long().to(self.device)
            
            outputs = self.model(features_tensor)
            loss = criterion(outputs, labels_tensor)
            val_loss = loss.item()
        
        return val_loss


# Export public API
__all__ = [
    "TransferConfig",
    "FeatureExtractor",
    "DomainAdapter",
    "ProgressiveUnfreezing",
    "TransferLearningPipeline",
    "TORCH_AVAILABLE"
]
