# 🤖 Modèle Hybride et Entraînement - Explication Complète

## 📚 Table des Matières

1. [Qu'est-ce qu'un modèle hybride ?](#quest-ce-quun-modèle-hybride-)
2. [Nombre d'époques d'entraînement recommandé](#nombre-dépoques-dentraînement-recommandé)
3. [Configuration de votre dataset](#configuration-de-votre-dataset)
4. [Exemple d'entraînement complet](#exemple-dentraînement-complet)

---

## Qu'est-ce qu'un modèle hybride ?

### 🎯 Définition

Un **modèle hybride** dans le contexte d'IGN LiDAR HD combine plusieurs architectures de deep learning pour tirer parti des forces de chacune. Au lieu de choisir une seule architecture (PointNet++, Transformer, etc.), un modèle hybride utilise **plusieurs branches en parallèle ou en série**.

### 🏗️ Architecture Hybride Typique

```
Input LiDAR Patch (16,384 points)
         │
         ├─────────────┬─────────────┬─────────────┐
         │             │             │             │
    PointNet++     Transformer    Octree-CNN   Sparse Conv
    (géométrie)    (attention)   (multi-échelle) (voxels)
         │             │             │             │
         └─────────────┴─────────────┴─────────────┘
                        │
                   Fusion Layer
                  (concatenation
                   ou attention)
                        │
                  Classification
                   (LOD2/LOD3)
```

### 💡 Avantages du Modèle Hybride

| Composante      | Force                     | Utilisation                                              |
| --------------- | ------------------------- | -------------------------------------------------------- |
| **PointNet++**  | Géométrie locale          | Extraction de features géométriques (normales, courbure) |
| **Transformer** | Relations longue-distance | Contexte global, attention spatiale                      |
| **Octree-CNN**  | Multi-échelle             | Hiérarchie d'échelles (toit, façade, détails)            |
| **Sparse Conv** | Efficacité                | Traitement rapide, features spatiales                    |

### 🎨 Types de Modèles Hybrides

#### 1️⃣ **Hybride Parallèle** (le plus courant)

```python
# Toutes les branches traitent les mêmes données en parallèle
pointnet_features = pointnet_branch(points)      # [B, 256]
transformer_features = transformer_branch(points) # [B, 256]
octree_features = octree_branch(points)          # [B, 256]

# Fusion des features
combined = torch.cat([pointnet_features,
                      transformer_features,
                      octree_features], dim=1)   # [B, 768]

# Classification finale
output = classifier(combined)                     # [B, num_classes]
```

#### 2️⃣ **Hybride Séquentiel**

```python
# Chaque branche traite la sortie de la précédente
features_1 = pointnet_branch(points)              # Features géométriques
features_2 = transformer_branch(features_1)       # Features contextuelles
output = classifier(features_2)                   # Classification
```

#### 3️⃣ **Hybride avec Attention**

```python
# Les branches communiquent via mécanisme d'attention
pointnet_features = pointnet_branch(points)
transformer_features = transformer_branch(points)

# Attention cross-branch
attended_features = cross_attention(pointnet_features,
                                    transformer_features)
output = classifier(attended_features)
```

---

## Nombre d'époques d'entraînement recommandé

### 📊 Recommandations par Type de Tâche

| Tâche                               | Époques Minimales | Époques Optimales | Époques Maximales | Early Stopping |
| ----------------------------------- | ----------------- | ----------------- | ----------------- | -------------- |
| **LOD2 Classification (Buildings)** | 50                | **100-150**       | 200               | Patience: 20   |
| **LOD3 Classification (Détaillée)** | 100               | **150-200**       | 300               | Patience: 30   |
| **Segmentation Sémantique**         | 80                | **120-180**       | 250               | Patience: 25   |
| **Détection de Végétation**         | 50                | **80-120**        | 150               | Patience: 15   |
| **Prototypage Rapide**              | 20                | **30-50**         | 80                | Patience: 10   |

### 🎯 Pour Votre Cas (LOD2 Urban Dense)

**Configuration recommandée:**

```python
epochs = 150                    # Nombre total d'époques
early_stopping_patience = 20    # Arrêt si pas d'amélioration pendant 20 époques
warmup_epochs = 10              # Montée en puissance du learning rate
lr_schedule = 'cosine'          # Cosine annealing pour LR decay
```

### 📈 Timeline d'Entraînement Typique

```
Époques 1-10:    Warmup - Learning rapide, loss descend vite
                 ▼ Validation accuracy: 60-70%

Époques 11-50:   Phase d'apprentissage principal
                 ▼ Validation accuracy: 70-85%

Époques 51-100:  Raffinement - amélioration progressive
                 ▼ Validation accuracy: 85-90%

Époques 101-150: Fine-tuning - convergence vers optimum
                 ▼ Validation accuracy: 90-94%

Époques 151+:    Overfitting possible - surveiller val_loss
                 ▼ Train accuracy > Val accuracy (signe d'overfitting)
```

### ⏱️ Temps d'Entraînement Estimé

**Pour un dataset de ~1000 patches (16,384 points/patch):**

| GPU          | Batch Size | Temps/Époque | Temps Total (150 époques) |
| ------------ | ---------- | ------------ | ------------------------- |
| **RTX 3060** | 8          | ~5 min       | ~12.5 heures              |
| **RTX 3080** | 16         | ~3 min       | ~7.5 heures               |
| **RTX 4090** | 32         | ~2 min       | ~5 heures                 |
| **A100**     | 64         | ~1.5 min     | ~3.75 heures              |

### 🛠️ Hyperparamètres Recommandés

```python
# Configuration d'entraînement pour modèle hybride LOD2
training_config = {
    # Époques
    'epochs': 150,
    'early_stopping_patience': 20,
    'warmup_epochs': 10,

    # Learning rate
    'initial_lr': 1e-3,          # Learning rate initial
    'min_lr': 1e-6,              # LR minimum (cosine annealing)
    'lr_scheduler': 'cosine',    # ou 'step', 'plateau'

    # Optimizer
    'optimizer': 'AdamW',
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),

    # Batch size
    'batch_size': 16,            # Ajuster selon votre GPU
    'accumulation_steps': 1,     # Gradient accumulation si GPU limité

    # Augmentation
    'augmentation_prob': 0.5,    # Probabilité d'appliquer augmentation
    'mixup_alpha': 0.2,          # Mixup pour régularisation

    # Validation
    'val_frequency': 1,          # Valider chaque époque
    'save_best_only': True,      # Sauvegarder seulement le meilleur modèle

    # Régularisation
    'dropout': 0.3,              # Dropout dans le classifier
    'label_smoothing': 0.1,      # Label smoothing pour éviter surconfiance
}
```

---

## Configuration de votre dataset

### 📦 Dataset Multi-Architecture (Hybride)

Votre commande actuelle génère des patches au format **NPZ** qui contiennent TOUTES les informations nécessaires pour un modèle hybride:

```python
# Contenu d'un patch NPZ
{
    # Données de base
    'xyz': np.ndarray,              # (N, 3) Coordonnées 3D normalisées
    'rgb': np.ndarray,              # (N, 3) Couleurs RGB [0-1]
    'nir': np.ndarray,              # (N, 1) Near-Infrared
    'ndvi': np.ndarray,             # (N, 1) Vegetation Index

    # Features géométriques (pour PointNet++)
    'normals': np.ndarray,          # (N, 3) Vecteurs normaux
    'curvature': np.ndarray,        # (N, 1) Courbure locale
    'planarity': np.ndarray,        # (N, 1) Planimétrie
    'verticality': np.ndarray,      # (N, 1) Verticalité
    'height': np.ndarray,           # (N, 1) Hauteur relative

    # Features radiométriques
    'intensity': np.ndarray,        # (N, 1) Intensité retour laser
    'return_number': np.ndarray,    # (N, 1) Numéro de retour

    # Metadata
    'labels': np.ndarray,           # (N,) Étiquettes par point
    'tile_name': str,               # Nom de la dalle d'origine
    'num_points': int,              # 16384
}
```

### 🎓 Loading Dataset pour Entraînement Hybride

```python
from ign_lidar.datasets import IGNLiDARMultiArchDataset
from torch.utils.data import DataLoader

# Dataset pour entraînement
train_dataset = IGNLiDARMultiArchDataset(
    data_dir='/mnt/c/Users/Simon/ign/training_patches_lod2_hybrid',
    architecture='hybrid',          # ⭐ MODE HYBRIDE
    num_points=16384,

    # Features à utiliser (toutes pour hybride)
    use_rgb=True,
    use_infrared=True,
    use_geometric=True,
    use_radiometric=True,
    use_contextual=True,

    # Normalisation
    normalize=True,
    normalize_rgb=True,
    standardize_features=True,

    # Augmentation (déjà fait lors de la génération, mais on peut en ajouter)
    augment=True,

    # Split
    split='train',
    train_ratio=0.8,
    val_ratio=0.1,
    random_seed=42,
)

# Dataset de validation
val_dataset = IGNLiDARMultiArchDataset(
    data_dir='/mnt/c/Users/Simon/ign/training_patches_lod2_hybrid',
    architecture='hybrid',
    split='val',
    augment=False,              # Pas d'augmentation pour validation
    # ... autres params identiques
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"Training patches: {len(train_dataset)}")
print(f"Validation patches: {len(val_dataset)}")
print(f"Test patches: {len(test_dataset)}")
```

---

## Exemple d'entraînement complet

### 🔥 Code d'Entraînement Hybride LOD2

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

class HybridLOD2Model(nn.Module):
    """
    Modèle hybride pour classification LOD2.
    Combine PointNet++, Transformer, et Sparse Conv.
    """
    def __init__(self, num_classes=10, num_points=16384):
        super().__init__()

        # Branche PointNet++ (géométrie locale)
        self.pointnet_branch = PointNetPlusPlus(
            in_channels=10,      # xyz + normals + features
            out_channels=256,
        )

        # Branche Transformer (attention globale)
        self.transformer_branch = PointTransformer(
            in_channels=10,
            out_channels=256,
            num_heads=8,
            num_layers=4,
        )

        # Branche Sparse Conv (features spatiales)
        self.sparse_branch = SparseConvNet(
            in_channels=10,
            out_channels=256,
            voxel_size=0.1,
        )

        # Fusion des branches
        self.fusion = nn.Sequential(
            nn.Linear(768, 512),    # 256 * 3 branches
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classifier final
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, batch):
        """
        Forward pass du modèle hybride.

        Args:
            batch: Dict contenant:
                - 'points': (B, N, 3) coordonnées
                - 'features': (B, N, F) features
        """
        # Extract data
        points = batch['points']        # (B, 16384, 3)
        features = batch['features']    # (B, 16384, F)

        # Concaténer points et features
        x = torch.cat([points, features], dim=-1)  # (B, N, 3+F)

        # Branches parallèles
        feat_pointnet = self.pointnet_branch(x)      # (B, 256)
        feat_transformer = self.transformer_branch(x) # (B, 256)
        feat_sparse = self.sparse_branch(x)          # (B, 256)

        # Fusion
        combined = torch.cat([
            feat_pointnet,
            feat_transformer,
            feat_sparse
        ], dim=1)  # (B, 768)

        fused = self.fusion(combined)  # (B, 256)

        # Classification
        logits = self.classifier(fused)  # (B, num_classes)

        return logits


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pour une époque."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward
        optimizer.zero_grad()
        logits = model(batch)

        # Loss
        labels = batch['labels']
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Valide le modèle."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            logits = model(batch)
            labels = batch['labels']

            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def main():
    """Fonction principale d'entraînement."""

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10  # Ajuster selon vos classes LOD2
    num_epochs = 150
    batch_size = 16
    initial_lr = 1e-3

    print(f"🚀 Starting training on {device}")
    print(f"📊 Configuration: {num_epochs} epochs, batch_size={batch_size}")

    # Dataset & DataLoader
    train_dataset = IGNLiDARMultiArchDataset(
        data_dir='/mnt/c/Users/Simon/ign/training_patches_lod2_hybrid',
        architecture='hybrid',
        split='train',
        augment=True,
    )

    val_dataset = IGNLiDARMultiArchDataset(
        data_dir='/mnt/c/Users/Simon/ign/training_patches_lod2_hybrid',
        architecture='hybrid',
        split='val',
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Model
    model = HybridLOD2Model(num_classes=num_classes).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Training loop
    best_val_acc = 0
    patience_counter = 0
    patience = 20

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print results
        print(f"\n📈 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'best_hybrid_lod2_model.pth')
            print(f"   ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            print(f"   Best validation accuracy: {best_val_acc:.2f}%")
            break

    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
```

---

## 📊 Résumé - Votre Configuration

### Votre Commande de Génération de Dataset

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/training_patches_lod2_hybrid" \
  processor.lod_level=LOD2 \
  processor.num_points=16384 \
  processor.augment=true \
  processor.num_augmentations=5 \
  features=full \
  output.format=npz
```

### Entraînement Recommandé

| Paramètre          | Valeur         | Justification                  |
| ------------------ | -------------- | ------------------------------ |
| **Époques**        | 150            | Convergence optimale pour LOD2 |
| **Early Stopping** | 20             | Évite overfitting              |
| **Batch Size**     | 16             | Bon compromis GPU/convergence  |
| **Learning Rate**  | 1e-3 → 1e-6    | Cosine annealing               |
| **Optimizer**      | AdamW          | Meilleur pour transformers     |
| **Augmentation**   | 5x (déjà fait) | Robustesse et généralisation   |

### ⏱️ Timeline Estimée

- **Génération dataset**: ~2-4 heures (selon nombre de dalles)
- **Entraînement 150 époques**: ~8-12 heures (RTX 3080)
- **Validation finale**: ~30 minutes
- **Total**: **~12-16 heures** pour avoir un modèle production-ready

---

## 🎯 Points Clés à Retenir

1. **Modèle Hybride** = Combiner plusieurs architectures (PointNet++, Transformer, Sparse Conv)
2. **150 époques** est optimal pour LOD2 classification
3. **Early stopping (patience=20)** pour éviter overfitting
4. Votre dataset NPZ contient **toutes les features** nécessaires
5. L'**augmentation 5x** est déjà faite lors de la génération des patches
6. Format **architecture-agnostic** = flexible pour tous types de modèles

---

## 📚 Ressources Complémentaires

- Documentation multi-architecture: `website/docs/features/multi-architecture.md`
- Dataset class: `ign_lidar/datasets/multi_arch_dataset.py`
- Exemple d'entraînement: Adaptez le code ci-dessus à vos besoins

Bon entraînement ! 🚀
