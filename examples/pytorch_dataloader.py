#!/usr/bin/env python3
"""
PyTorch DataLoader pour le dataset IGN LIDAR HD

Exemple d'utilisation des patches précomputés pour entraînement
de modèles de segmentation 3D et extraction de bâtiments.

Usage:
    from pytorch_dataloader import IGNLiDARDataset
    
    dataset = IGNLiDARDataset('/path/to/patches/train')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json


class IGNLiDARDataset(Dataset):
    """
    Dataset PyTorch pour les patches IGN LIDAR HD.
    
    Retourne des features géométriques précomputées et les labels
    pour la segmentation 3D.
    """
    
    def __init__(self,
                 data_dir: Path,
                 feature_set: str = 'full',
                 transform=None):
        """
        Args:
            data_dir: Répertoire contenant les patches HDF5
            feature_set: 'full', 'geometric_only', ou 'minimal'
            transform: Transformations optionnelles
        """
        self.data_dir = Path(data_dir)
        self.feature_set = feature_set
        self.transform = transform
        
        # Trouver tous les fichiers de patches
        self.patch_files = sorted(list(self.data_dir.rglob("*.h5")))
        
        if len(self.patch_files) == 0:
            raise ValueError(f"Aucun patch trouvé dans {data_dir}")
        
        print(f"Dataset initialisé: {len(self.patch_files)} patches")
        
        # Mapping des classes ASPRS vers LOD2/LOD3
        self.asprs_to_lod2 = {
            0: 0,   # Never classified -> Other
            1: 0,   # Unassigned -> Other
            2: 1,   # Ground -> Ground
            3: 2,   # Low Vegetation -> Vegetation
            4: 2,   # Medium Vegetation -> Vegetation
            5: 2,   # High Vegetation -> Vegetation
            6: 3,   # Building -> Building
            9: 4,   # Water -> Water
            17: 5,  # Bridge -> Infrastructure
            64: 0,  # Other -> Other
        }
    
    def __len__(self) -> int:
        return len(self.patch_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Charger un patch et retourner les features + labels.
        
        Returns:
            Dict contenant:
            - 'features': Tensor [N, F] des features
            - 'labels': Tensor [N] des labels de segmentation
            - 'xyz': Tensor [N, 3] des coordonnées (optionnel)
        """
        patch_file = self.patch_files[idx]
        
        # Charger depuis HDF5
        with h5py.File(patch_file, 'r') as f:
            # Coordonnées (toujours chargées)
            xyz = torch.from_numpy(f['xyz_normalized'][:]).float()
            
            # Sélection des features selon feature_set
            if self.feature_set == 'full':
                # Toutes les features géométriques
                features_list = [
                    xyz,  # [N, 3]
                    torch.from_numpy(f['intensity'][:]).float().unsqueeze(1),
                    torch.from_numpy(f['return_number'][:]).float().unsqueeze(1),
                    torch.from_numpy(f['normals'][:]).float(),  # [N, 3]
                    torch.from_numpy(f['curvature'][:]).float().unsqueeze(1),
                    torch.from_numpy(
                        f['height_above_ground'][:]
                    ).float().unsqueeze(1),
                    torch.from_numpy(f['density'][:]).float().unsqueeze(1),
                    torch.from_numpy(f['roughness'][:]).float().unsqueeze(1),
                    torch.from_numpy(f['planarity'][:]).float().unsqueeze(1),
                    torch.from_numpy(
                        f['verticality'][:]
                    ).float().unsqueeze(1),
                ]
                features = torch.cat(features_list, dim=1)  # [N, 16]
                
            elif self.feature_set == 'geometric_only':
                # Seulement les features géométriques pures
                features_list = [
                    xyz,  # [N, 3]
                    torch.from_numpy(f['normals'][:]).float(),  # [N, 3]
                    torch.from_numpy(f['curvature'][:]).float().unsqueeze(1),
                    torch.from_numpy(
                        f['height_above_ground'][:]
                    ).float().unsqueeze(1),
                ]
                features = torch.cat(features_list, dim=1)  # [N, 8]
                
            else:  # minimal
                # Seulement XYZ et normales
                features = torch.cat([
                    xyz,
                    torch.from_numpy(f['normals'][:]).float()
                ], dim=1)  # [N, 6]
            
            # Labels (classification ASPRS)
            labels_asprs = f['classification'][:]
            
            # Convertir vers LOD2 (classes simplifiées)
            labels = np.zeros_like(labels_asprs)
            for asprs_code, lod2_code in self.asprs_to_lod2.items():
                labels[labels_asprs == asprs_code] = lod2_code
            
            labels = torch.from_numpy(labels).long()
        
        sample = {
            'features': features,
            'labels': labels,
            'xyz': xyz,
            'file': str(patch_file.name)
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculer les poids de classes pour gérer le déséquilibre.
        
        Returns:
            Tensor des poids pour chaque classe
        """
        class_counts = np.zeros(6)  # 6 classes LOD2
        
        print("Calcul des statistiques de classes...")
        for patch_file in self.patch_files[:100]:  # Échantillon
            with h5py.File(patch_file, 'r') as f:
                labels_asprs = f['classification'][:]
                
                for asprs_code, lod2_code in self.asprs_to_lod2.items():
                    count = np.sum(labels_asprs == asprs_code)
                    class_counts[lod2_code] += count
        
        # Calculer les poids inversement proportionnels
        total = class_counts.sum()
        class_weights = total / (len(class_counts) * class_counts + 1e-6)
        
        return torch.from_numpy(class_weights).float()


class RandomRotation:
    """Rotation aléatoire autour de l'axe Z."""
    
    def __call__(self, sample):
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Rotation des coordonnées
        sample['xyz'] = sample['xyz'] @ rotation_matrix.T
        
        # Rotation des features qui contiennent des vecteurs
        # (normales sont aux positions 3:6 dans feature_set='full')
        if sample['features'].shape[1] >= 6:
            normals = sample['features'][:, 3:6]
            sample['features'][:, 3:6] = normals @ rotation_matrix.T
        
        return sample


class RandomJitter:
    """Ajouter du bruit gaussien."""
    
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    
    def __call__(self, sample):
        noise = torch.randn_like(sample['xyz']) * self.sigma
        sample['xyz'] = sample['xyz'] + noise
        
        return sample


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

def example_usage():
    """Exemple d'utilisation du DataLoader."""
    
    # Chemins
    train_dir = Path("/mnt/c/Users/Simon/ign/ai_dataset/patches/train")
    val_dir = Path("/mnt/c/Users/Simon/ign/ai_dataset/patches/val")
    
    # Transformations pour l'augmentation
    from torchvision import transforms
    train_transform = transforms.Compose([
        RandomRotation(),
        RandomJitter(sigma=0.01)
    ])
    
    # Créer les datasets
    print("Création des datasets...")
    train_dataset = IGNLiDARDataset(
        train_dir,
        feature_set='full',
        transform=train_transform
    )
    
    val_dataset = IGNLiDARDataset(
        val_dir,
        feature_set='full',
        transform=None
    )
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} patches")
    print(f"Val: {len(val_dataset)} patches")
    
    # Calculer les poids de classes
    class_weights = train_dataset.get_class_weights()
    print(f"\nPoids de classes: {class_weights}")
    
    # Tester un batch
    print("\n🧪 Test d'un batch...")
    batch = next(iter(train_loader))
    
    print(f"Features shape: {batch['features'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"XYZ shape: {batch['xyz'].shape}")
    
    # Statistiques du batch
    print(f"\nStatistiques batch:")
    print(f"  - Nombre de points: {batch['features'].shape[1]}")
    print(f"  - Nombre de features: {batch['features'].shape[2]}")
    print(f"  - Classes présentes: {torch.unique(batch['labels'])}")
    
    # Exemple d'utilisation dans une boucle d'entraînement
    print("\n🔄 Simulation d'une époque d'entraînement...")
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Juste 3 batches pour l'exemple
            break
        
        features = batch['features']  # [B, N, F]
        labels = batch['labels']      # [B, N]
        
        print(f"Batch {i+1}: features {features.shape}, labels {labels.shape}")
        
        # ICI: votre modèle ferait
        # outputs = model(features)
        # loss = criterion(outputs, labels)
        # ...
    
    print("\n✅ DataLoader prêt pour l'entraînement!")
    
    return train_loader, val_loader, class_weights


def create_example_model():
    """
    Exemple de modèle simple PointNet-like pour la segmentation.
    """
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SimpleSegmentationModel(nn.Module):
        def __init__(self, input_features=16, num_classes=6):
            super().__init__()
            
            # Encodeur point-wise
            self.fc1 = nn.Linear(input_features, 64)
            self.fc2 = nn.Linear(64, 128)
            self.fc3 = nn.Linear(128, 256)
            
            # Pooling global
            # (max pooling sur tous les points)
            
            # Décodeur point-wise
            self.fc4 = nn.Linear(256 + 256, 256)
            self.fc5 = nn.Linear(256, 128)
            self.fc6 = nn.Linear(128, num_classes)
            
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(256)
            self.bn4 = nn.BatchNorm1d(256)
            self.bn5 = nn.BatchNorm1d(128)
            
            self.dropout = nn.Dropout(0.3)
        
        def forward(self, x):
            """
            Args:
                x: [B, N, F] features
            Returns:
                [B, N, C] logits per point
            """
            B, N, F = x.shape
            
            # Encodeur
            x = x.transpose(1, 2)  # [B, F, N]
            x = F.relu(self.bn1(self.fc1(x.transpose(1, 2)).transpose(1, 2)))
            x = F.relu(self.bn2(self.fc2(x.transpose(1, 2)).transpose(1, 2)))
            x = F.relu(self.bn3(self.fc3(x.transpose(1, 2)).transpose(1, 2)))
            
            # Global feature
            global_feat = torch.max(x, dim=2, keepdim=True)[0]  # [B, 256, 1]
            global_feat = global_feat.repeat(1, 1, N)  # [B, 256, N]
            
            # Concaténer avec features locales
            x = torch.cat([x, global_feat], dim=1)  # [B, 512, N]
            
            # Décodeur
            x = F.relu(self.bn4(self.fc4(x.transpose(1, 2)).transpose(1, 2)))
            x = F.relu(self.bn5(self.fc5(x.transpose(1, 2)).transpose(1, 2)))
            x = self.dropout(x)
            x = self.fc6(x.transpose(1, 2))  # [B, N, C]
            
            return x
    
    return SimpleSegmentationModel(input_features=16, num_classes=6)


if __name__ == "__main__":
    print("="*70)
    print("EXEMPLE D'UTILISATION DU DATALOADER IGN LIDAR HD")
    print("="*70)
    
    # Test du DataLoader
    try:
        train_loader, val_loader, class_weights = example_usage()
        
        # Créer un modèle exemple
        print("\n🤖 Création d'un modèle exemple...")
        model = create_example_model()
        print(f"Modèle créé: {sum(p.numel() for p in model.parameters()):,} "
              "paramètres")
        
        # Test forward pass
        batch = next(iter(train_loader))
        with torch.no_grad():
            outputs = model(batch['features'])
        
        print(f"\nTest forward pass:")
        print(f"  Input: {batch['features'].shape}")
        print(f"  Output: {outputs.shape}")
        print(f"  Expected: [batch_size, num_points, num_classes]")
        
        print("\n✅ Tout fonctionne! Prêt pour l'entraînement.")
        
    except Exception as e:
        print(f"\n⚠️  Erreur: {e}")
        print("Assurez-vous que les patches ont été créés avec "
              "create_training_patches.py")
