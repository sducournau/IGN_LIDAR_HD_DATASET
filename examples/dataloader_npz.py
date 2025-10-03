#!/usr/bin/env python3
"""
PyTorch DataLoader pour patches NPZ (LAZ enrichis)

Charge les patches créés depuis les LAZ enrichis avec toutes les features
géométriques précomputées.

Usage:
    from dataloader_npz import LiDARPatchDataset
    
    dataset = LiDARPatchDataset('/path/to/patches_lod2/train')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiDARPatchDataset(Dataset):
    """
    Dataset PyTorch pour patches NPZ créés depuis LAZ enrichis.
    
    Supporte LOD2 et LOD3.
    """
    
    def __init__(self,
                 data_dir: Path,
                 feature_set: str = 'full',
                 normalize: bool = True,
                 transform=None):
        """
        Args:
            data_dir: Répertoire contenant les patches NPZ
            feature_set: 'full', 'geometric', 'minimal'
            normalize: Normaliser les features
            transform: Transformations optionnelles
        """
        self.data_dir = Path(data_dir)
        self.feature_set = feature_set
        self.normalize = normalize
        self.transform = transform
        
        # Trouver tous les fichiers NPZ
        self.patch_files = sorted(list(self.data_dir.rglob("*.npz")))
        
        if len(self.patch_files) == 0:
            raise ValueError(f"Aucun patch trouvé dans {data_dir}")
        
        logger.info(f"Dataset initialisé: {len(self.patch_files)} patches")
        logger.info(f"Feature set: {feature_set}")
        
        # Statistiques pour normalisation
        if normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Calculer les stats de normalisation sur un échantillon."""
        logger.info("Calcul des statistiques de normalisation...")
        
        sample_size = min(100, len(self.patch_files))
        sample_indices = np.random.choice(
            len(self.patch_files), sample_size, replace=False
        )
        
        all_features = []
        
        for idx in sample_indices:
            data = np.load(self.patch_files[idx])
            features = self._extract_features(data, normalize=False)
            all_features.append(features.numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        
        self.feature_mean = torch.from_numpy(
            np.mean(all_features, axis=0)
        ).float()
        self.feature_std = torch.from_numpy(
            np.std(all_features, axis=0) + 1e-8
        ).float()
        
        logger.info(f"Stats calculées sur {sample_size} patches")
    
    def _extract_features(self, data: Dict, normalize: bool = True) -> torch.Tensor:
        """
        Extraire et assembler les features selon le feature_set.
        
        Args:
            data: Dict chargé depuis NPZ
            normalize: Appliquer la normalisation
            
        Returns:
            Tensor [N, F] des features
        """
        if self.feature_set == 'full':
            # Toutes les features (16 dimensions)
            features_list = [
                torch.from_numpy(data['xyz']).float(),  # [N, 3]
                torch.from_numpy(data['intensity'][:, None]).float(),  # [N, 1]
                torch.from_numpy(data['return_number'][:, None]).float(),  # [N, 1]
                torch.from_numpy(data['normals']).float(),  # [N, 3]
                torch.from_numpy(data['curvature'][:, None]).float(),  # [N, 1]
                torch.from_numpy(data['height_above_ground'][:, None]).float(),
                torch.from_numpy(data['density'][:, None]).float(),
                torch.from_numpy(data['roughness'][:, None]).float(),
                torch.from_numpy(data['planarity'][:, None]).float(),
                torch.from_numpy(data['verticality'][:, None]).float(),
            ]
            features = torch.cat(features_list, dim=1)  # [N, 16]
            
        elif self.feature_set == 'geometric':
            # Seulement features géométriques (10 dimensions)
            features_list = [
                torch.from_numpy(data['xyz']).float(),  # [N, 3]
                torch.from_numpy(data['normals']).float(),  # [N, 3]
                torch.from_numpy(data['curvature'][:, None]).float(),
                torch.from_numpy(data['height_above_ground'][:, None]).float(),
                torch.from_numpy(data['roughness'][:, None]).float(),
                torch.from_numpy(data['planarity'][:, None]).float(),
            ]
            features = torch.cat(features_list, dim=1)  # [N, 10]
            
        else:  # minimal
            # XYZ + normales seulement (6 dimensions)
            features = torch.cat([
                torch.from_numpy(data['xyz']).float(),
                torch.from_numpy(data['normals']).float()
            ], dim=1)  # [N, 6]
        
        # Normalisation
        if normalize and self.normalize and hasattr(self, 'feature_mean'):
            features = (features - self.feature_mean) / self.feature_std
        
        return features
    
    def __len__(self) -> int:
        return len(self.patch_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Charger un patch.
        
        Returns:
            Dict contenant:
            - 'features': Tensor [N, F] des features
            - 'labels': Tensor [N] des labels
            - 'xyz': Tensor [N, 3] des coordonnées
            - 'file': Nom du fichier
        """
        patch_file = self.patch_files[idx]
        
        # Charger le patch NPZ
        data = np.load(patch_file)
        
        # Extraire les features
        features = self._extract_features(data, normalize=self.normalize)
        
        # Labels
        labels = torch.from_numpy(data['labels']).long()
        
        # Coordonnées (non normalisées pour visualisation)
        xyz = torch.from_numpy(data['xyz']).float()
        
        sample = {
            'features': features,
            'labels': labels,
            'xyz': xyz,
            'file': str(patch_file.name)
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Calculer la distribution des classes."""
        logger.info("Calcul de la distribution des classes...")
        
        class_counts = {}
        sample_size = min(100, len(self.patch_files))
        
        for patch_file in self.patch_files[:sample_size]:
            data = np.load(patch_file)
            labels = data['labels']
            unique, counts = np.unique(labels, return_counts=True)
            
            for cls, count in zip(unique, counts):
                cls = int(cls)
                class_counts[cls] = class_counts.get(cls, 0) + count
        
        return class_counts
    
    def get_class_weights(self, num_classes: int = 6) -> torch.Tensor:
        """
        Calculer les poids de classes pour gérer le déséquilibre.
        
        Args:
            num_classes: Nombre de classes (6 pour LOD2, 30 pour LOD3)
            
        Returns:
            Tensor des poids pour chaque classe
        """
        class_counts = self.get_class_distribution()
        
        # Créer un tableau de comptages
        counts = np.zeros(num_classes)
        for cls, count in class_counts.items():
            if cls < num_classes:
                counts[cls] = count
        
        # Calculer les poids inversement proportionnels
        total = counts.sum()
        weights = total / (num_classes * (counts + 1))
        
        # Normaliser
        weights = weights / weights.sum() * num_classes
        
        logger.info(f"Poids de classes: {weights}")
        
        return torch.from_numpy(weights).float()


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
        
        # Rotation des coordonnées XYZ (premières 3 dimensions)
        sample['features'][:, :3] = sample['features'][:, :3] @ rotation_matrix.T
        sample['xyz'] = sample['xyz'] @ rotation_matrix.T
        
        # Rotation des normales (dimensions 5:8 pour 'full')
        if sample['features'].shape[1] >= 8:
            sample['features'][:, 5:8] = (
                sample['features'][:, 5:8] @ rotation_matrix.T
            )
        
        return sample


class RandomJitter:
    """Ajouter du bruit gaussien aux coordonnées."""
    
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    
    def __call__(self, sample):
        noise = torch.randn_like(sample['features'][:, :3]) * self.sigma
        sample['features'][:, :3] = sample['features'][:, :3] + noise
        sample['xyz'] = sample['xyz'] + noise
        
        return sample


class RandomScale:
    """Échelle aléatoire."""
    
    def __init__(self, scale_range=(0.95, 1.05)):
        self.scale_range = scale_range
    
    def __call__(self, sample):
        scale = np.random.uniform(*self.scale_range)
        sample['features'][:, :3] = sample['features'][:, :3] * scale
        sample['xyz'] = sample['xyz'] * scale
        
        return sample


def example_usage_lod2():
    """Exemple d'utilisation pour LOD2."""
    print("="*70)
    print("EXEMPLE D'UTILISATION - LOD2")
    print("="*70)
    
    # Chemins
    train_dir = Path("/mnt/c/Users/Simon/ign/ai_dataset_laz/patches_lod2/train")
    val_dir = Path("/mnt/c/Users/Simon/ign/ai_dataset_laz/patches_lod2/val")
    
    if not train_dir.exists():
        print(f"⚠️  Répertoire non trouvé: {train_dir}")
        print("   Exécutez d'abord: python workflow_laz_enriched.py")
        return
    
    # Transformations
    from torchvision import transforms
    train_transform = transforms.Compose([
        RandomRotation(),
        RandomJitter(sigma=0.01),
        RandomScale(scale_range=(0.95, 1.05))
    ])
    
    # Créer les datasets
    print("\n📚 Chargement datasets LOD2...")
    train_dataset = LiDARPatchDataset(
        train_dir,
        feature_set='full',
        normalize=True,
        transform=train_transform
    )
    
    val_dataset = LiDARPatchDataset(
        val_dir,
        feature_set='full',
        normalize=True,
        transform=None
    )
    
    print(f"Train: {len(train_dataset)} patches")
    print(f"Val: {len(val_dataset)} patches")
    
    # DataLoaders
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
        num_workers=4
    )
    
    # Poids de classes (LOD2 a 6 classes)
    class_weights = train_dataset.get_class_weights(num_classes=6)
    print(f"\n⚖️  Poids de classes LOD2: {class_weights}")
    
    # Test d'un batch
    print("\n🧪 Test d'un batch...")
    batch = next(iter(train_loader))
    
    print(f"Features shape: {batch['features'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"XYZ shape: {batch['xyz'].shape}")
    print(f"Classes présentes: {torch.unique(batch['labels']).tolist()}")
    
    # Distribution des classes dans le batch
    unique_labels, label_counts = torch.unique(
        batch['labels'], return_counts=True
    )
    print(f"\nDistribution dans le batch:")
    for label, count in zip(unique_labels, label_counts):
        print(f"  Classe {label.item()}: {count.item()} points")
    
    print("\n✅ DataLoader LOD2 prêt!")
    
    return train_loader, val_loader, class_weights


def example_usage_lod3():
    """Exemple d'utilisation pour LOD3."""
    print("\n" + "="*70)
    print("EXEMPLE D'UTILISATION - LOD3")
    print("="*70)
    
    train_dir = Path("/mnt/c/Users/Simon/ign/ai_dataset_laz/patches_lod3/train")
    val_dir = Path("/mnt/c/Users/Simon/ign/ai_dataset_laz/patches_lod3/val")
    
    if not train_dir.exists():
        print(f"⚠️  Répertoire non trouvé: {train_dir}")
        return None
    
    # Dataset LOD3
    train_dataset = LiDARPatchDataset(
        train_dir,
        feature_set='full',
        normalize=True
    )
    
    val_dataset = LiDARPatchDataset(
        val_dir,
        feature_set='full',
        normalize=True
    )
    
    print(f"Train: {len(train_dataset)} patches")
    print(f"Val: {len(val_dataset)} patches")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # LOD3 a plus de classes (~30)
    class_weights = train_dataset.get_class_weights(num_classes=30)
    
    print("\n✅ DataLoader LOD3 prêt!")
    
    return train_loader, val_loader, class_weights


if __name__ == "__main__":
    print("="*70)
    print("🎯 DATALOADER POUR PATCHES NPZ (LAZ ENRICHIS)")
    print("="*70)
    
    try:
        # Test LOD2
        train_loader_lod2, val_loader_lod2, weights_lod2 = example_usage_lod2()
        
        # Test LOD3
        results_lod3 = example_usage_lod3()
        
        print("\n" + "="*70)
        print("✅ DATALOADERS CONFIGURÉS!")
        print("="*70)
        print("\n💡 Utilisez ces dataloaders dans votre code d'entraînement:")
        print("   from dataloader_npz import LiDARPatchDataset")
        print("   dataset = LiDARPatchDataset('path/to/patches/train')")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("\n💡 Assurez-vous d'avoir exécuté workflow_laz_enriched.py")
