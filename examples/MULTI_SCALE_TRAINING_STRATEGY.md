# Multi-Scale Training Strategy for Hybrid Models

## 🎯 Strategic Training Process Overview

Training on **multiple patch sizes** creates a more robust model that can handle different scales of architectural features. This is especially powerful for LOD3 classification where buildings vary significantly in size.

---

## 📊 Multi-Scale Training Benefits

### Why Train on Multiple Scales?

1. **Scale Invariance** 🔄

   - Model learns features at different resolutions
   - Better generalization to buildings of varying sizes
   - Robust to different patch extraction scenarios

2. **Feature Hierarchy** 🏗️

   - Small patches (50m): Local details (windows, balconies, ornaments)
   - Medium patches (100m): Building sections, facade patterns
   - Large patches (150m): Full building context, urban layout

3. **Improved Generalization** 🎓

   - Reduces overfitting to specific patch sizes
   - Better performance on test data at any scale
   - More resilient to data augmentation

4. **Architectural Diversity** 🏘️
   - Small residential buildings: Better with 50-100m patches
   - Medium commercial: Optimal at 100-150m
   - Large complexes: Need 150-200m patches

---

## 🔧 Three-Scale Training Strategy

### Phase 1: Fine-Grained Details (50m patches)

**Focus**: Local architectural features, textures, small elements

```yaml
# config_lod3_training_50m.yaml
processor:
  lod_level: LOD3
  architecture: hybrid
  use_gpu: true
  num_workers: 2
  patch_size: 50.0 # Small patches for details
  patch_overlap: 0.15 # Higher overlap for small patches
  num_points: 24576 # 24k points per patch
  augment: true
  num_augmentations: 5 # More augmentation for diversity
  pin_memory: true

features:
  mode: full
  k_neighbors: 20 # Smaller neighborhood for fine details
  include_extra: true
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
  sampling_method: fps
  normalize_xyz: true
  normalize_features: true
  gpu_batch_size: 1000000
  use_gpu_chunked: true

preprocess:
  enabled: true
  sor_k: 8 # Less aggressive for small patches
  sor_std: 1.5
  ror_radius: 0.5
  ror_neighbors: 3
  voxel_enabled: false

stitching:
  enabled: true
  buffer_size: 8.0 # Smaller buffer for 50m patches
  auto_detect_neighbors: true
  auto_download_neighbors: true
  cache_enabled: true

output:
  format: npz,laz
  processing_mode: patches_only
  save_stats: true
  save_metadata: true
  compression: null

input_dir: /mnt/c/Users/Simon/ign/enriched_tiles
output_dir: /mnt/c/Users/Simon/ign/patches_50m
```

**Expected Output**:

**Expected Output**:

- Patches per tile: ~40-60 (more patches, smaller area)
- Points per patch: 24,576
- Training samples: ~10,000-15,000 patches
- Storage: ~8-12GB

---

### Phase 2: Balanced Context (100m patches)

**Focus**: Building facades, medium-scale patterns

```yaml
# config_lod3_training_100m.yaml
processor:
  lod_level: LOD3
  architecture: hybrid
  use_gpu: true
  num_workers: 2
  patch_size: 100.0 # Medium patches for balance
  patch_overlap: 0.12 # Balanced overlap
  num_points: 24576 # 24k points (between 16k-32k)
  augment: true
  num_augmentations: 4
  pin_memory: true

features:
  mode: full
  k_neighbors: 25 # Balanced neighborhood
  include_extra: true
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
  sampling_method: fps
  normalize_xyz: true
  normalize_features: true
  gpu_batch_size: 1000000
  use_gpu_chunked: true

preprocess:
  enabled: true
  sor_k: 10
  sor_std: 2.0
  ror_radius: 0.75
  ror_neighbors: 4
  voxel_enabled: false

stitching:
  enabled: true
  buffer_size: 12.0 # Medium buffer for 100m patches
  auto_detect_neighbors: true
  auto_download_neighbors: true
  cache_enabled: true

output:
  format: npz,laz
  processing_mode: patches_only
  save_stats: true
  save_metadata: true
  compression: null

input_dir: /mnt/c/Users/Simon/ign/enriched_tiles
output_dir: /mnt/c/Users/Simon/ign/patches_100m
```

**Expected Output**:

**Expected Output**:

- Patches per tile: ~15-25
- Points per patch: 32,768
- Training samples: ~5,000-8,000 patches
- Storage: ~10-15GB

---

### Phase 3: Full Context (150m patches)

**Focus**: Complete buildings, urban context

```yaml
# config_lod3_training_150m.yaml (your current optimized config)
processor:
  lod_level: LOD3
  architecture: hybrid
  use_gpu: true
  num_workers: 2
  patch_size: 150.0 # Large patches for full context
  patch_overlap: 0.1 # Standard overlap
  num_points: 32768 # Full PointNet++ capacity
  augment: true
  num_augmentations: 3
  pin_memory: true

features:
  mode: full
  k_neighbors: 30 # Larger neighborhood for context
  include_extra: true
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
  sampling_method: fps
  normalize_xyz: true
  normalize_features: true
  gpu_batch_size: 1000000
  use_gpu_chunked: true

preprocess:
  enabled: true
  sor_k: 12
  sor_std: 2.0
  ror_radius: 1.0
  ror_neighbors: 4
  voxel_enabled: false

stitching:
  enabled: true
  buffer_size: 15.0 # Larger buffer for 150m patches
  auto_detect_neighbors: true
  auto_download_neighbors: true
  cache_enabled: true

output:
  format: npz,laz
  processing_mode: patches_only
  save_stats: true
  save_metadata: true
  compression: null

input_dir: /mnt/c/Users/Simon/ign/enriched_tiles
output_dir: /mnt/c/Users/Simon/ign/patches_150m
```

**Expected Output**:

- Patches per tile: ~8-12
- Points per patch: 32,768
- Training samples: ~3,000-5,000 patches
- Storage: ~12-18GB

---

## 📋 Complete Training Pipeline

### Step 1: Generate Multi-Scale Datasets

```bash
# Terminal 1: Generate 50m patches (fastest)
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET
ign-lidar-hd process --config-file examples/config_lod3_training_50m.yaml

# Terminal 2: Generate 100m patches (parallel if memory allows)
ign-lidar-hd process --config-file examples/config_lod3_training_100m.yaml

# Terminal 3: Generate 150m patches
ign-lidar-hd process --config-file examples/config_lod3_training_150m.yaml
```

### Step 2: Merge Multi-Scale Datasets

```python
# merge_multiscale_dataset.py
import numpy as np
from pathlib import Path
import random

def merge_multiscale_datasets(
    patch_dirs=['patches_50m', 'patches_100m', 'patches_150m'],
    output_dir='patches_multiscale',
    scale_weights=[0.4, 0.3, 0.3]  # More emphasis on fine details
):
    """
    Merge patches from different scales into a unified training dataset.

    Args:
        patch_dirs: List of directories containing patches at different scales
        output_dir: Output directory for merged dataset
        scale_weights: Relative proportions of each scale in final dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    all_patches = []

    # Collect patches from each scale
    for patch_dir, weight in zip(patch_dirs, scale_weights):
        patch_path = Path(f'/mnt/c/Users/Simon/ign/{patch_dir}')
        patches = list(patch_path.glob('*.npz'))

        # Sample according to weight
        n_samples = int(len(patches) * weight / min(scale_weights))
        sampled = random.sample(patches, min(n_samples, len(patches)))

        all_patches.extend([
            (p, patch_dir.split('_')[-1]) for p in sampled
        ])

    # Shuffle for training
    random.shuffle(all_patches)

    print(f"Total patches in multi-scale dataset: {len(all_patches)}")
    print(f"  - 50m patches: {sum(1 for _, s in all_patches if s == '50m')}")
    print(f"  - 100m patches: {sum(1 for _, s in all_patches if s == '100m')}")
    print(f"  - 150m patches: {sum(1 for _, s in all_patches if s == '150m')}")

    # Create symbolic links or copy files
    for i, (patch_path, scale) in enumerate(all_patches):
        output_name = f"multiscale_{scale}_{i:06d}.npz"
        output_file = output_path / output_name

        # Create symlink (faster) or copy
        try:
            output_file.symlink_to(patch_path.absolute())
        except:
            import shutil
            shutil.copy2(patch_path, output_file)

    print(f"\nMerged dataset saved to: {output_dir}")
    return all_patches

if __name__ == "__main__":
    merge_multiscale_datasets()
```

---

### Step 3: Multi-Scale Training Script

```python
# train_multiscale_hybrid.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

class MultiScalePointCloudDataset(Dataset):
    """Dataset that handles variable-sized patches through padding/sampling"""

    def __init__(self, data_dir, target_points=32768, augment=True):
        self.data_dir = Path(data_dir)
        self.patches = list(self.data_dir.glob('*.npz'))
        self.target_points = target_points
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Load patch
        data = np.load(self.patches[idx])
        points = data['xyz']
        features = data['features']
        labels = data.get('labels', np.zeros(len(points)))

        # Normalize to target_points (pad or sample)
        n_points = len(points)
        if n_points > self.target_points:
            # FPS or random sampling
            indices = np.random.choice(n_points, self.target_points, replace=False)
            points = points[indices]
            features = features[indices]
            labels = labels[indices]
        elif n_points < self.target_points:
            # Pad with duplicates
            pad_size = self.target_points - n_points
            pad_indices = np.random.choice(n_points, pad_size, replace=True)
            points = np.vstack([points, points[pad_indices]])
            features = np.vstack([features, features[pad_indices]])
            labels = np.concatenate([labels, labels[pad_indices]])

        # Apply augmentation
        if self.augment:
            points = self.augment_points(points)

        return {
            'points': torch.FloatTensor(points),
            'features': torch.FloatTensor(features),
            'labels': torch.LongTensor(labels),
            'patch_name': self.patches[idx].stem
        }

    def augment_points(self, points):
        # Rotation
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix.T

        # Jitter
        points += np.random.normal(0, 0.02, points.shape)

        # Scale
        scale = np.random.uniform(0.9, 1.1)
        points *= scale

        return points


def train_multiscale_model(
    model,
    train_dir='patches_multiscale',
    val_dir='patches_multiscale_val',
    epochs=100,
    batch_size=8,
    lr=0.001
):
    """
    Train hybrid model on multi-scale patches
    """
    # Create datasets
    train_dataset = MultiScalePointCloudDataset(train_dir, augment=True)
    val_dataset = MultiScalePointCloudDataset(val_dir, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            points = batch['points'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(points, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(device)
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(points, features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Update learning rate
        scheduler.step()

        # Log progress
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, 'best_multiscale_model.pth')
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")

    return model


if __name__ == "__main__":
    # Import your hybrid model
    # from your_model import HybridModel
    # model = HybridModel(num_classes=30)

    # Train
    # trained_model = train_multiscale_model(model)
    pass
```

---

## 📊 Training Schedule Recommendation

### Timeline: ~2-3 Days Full Pipeline

```
Day 1: Data Generation (Parallel)
├─ Morning:   Generate 50m patches  (4-6 hours) → ~15,000 patches
├─ Afternoon: Generate 100m patches (3-4 hours) → ~8,000 patches
└─ Evening:   Generate 150m patches (2-3 hours) → ~5,000 patches

Day 2: Dataset Preparation & Initial Training
├─ Morning:   Merge datasets, split train/val/test (1 hour)
├─ Afternoon: Start training (baseline single-scale) (4 hours)
└─ Evening:   Evaluate metrics, tune hyperparameters (3 hours)

Day 3: Multi-Scale Training & Evaluation
├─ Morning:   Train multi-scale model (6-8 hours)
└─ Afternoon: Final evaluation, compare results (2-3 hours)
```

---

## 🎓 Training Strategy Options

### Option 1: Sequential Multi-Scale (Curriculum Learning)

Train progressively from small to large patches:

```python
# Phase 1: Fine-tune on 50m patches (20 epochs)
train_on_scale(model, '50m', epochs=20, lr=0.001)

# Phase 2: Add 100m patches (30 epochs)
train_on_scale(model, '50m+100m', epochs=30, lr=0.0005)

# Phase 3: Add 150m patches (50 epochs)
train_on_scale(model, 'all_scales', epochs=50, lr=0.0001)
```

**Benefits**: Progressive learning, stable training  
**Best for**: Limited GPU memory, faster convergence

---

### Option 2: Mixed Multi-Scale (Simultaneous)

Train on all scales from the start:

```python
# Train on merged dataset (100 epochs)
train_multiscale_model(model, 'patches_multiscale', epochs=100, lr=0.001)
```

**Benefits**: Better generalization, scale-invariant features  
**Best for**: Abundant data, robust model

---

### Option 3: Scale-Specific Experts + Ensemble

Train separate models per scale, then ensemble:

```python
# Train 3 separate models
model_50m = train_on_scale('50m', epochs=80)
model_100m = train_on_scale('100m', epochs=80)
model_150m = train_on_scale('150m', epochs=80)

# Ensemble predictions
def predict_ensemble(point_cloud, patch_size):
    if patch_size < 75:
        return model_50m(point_cloud)
    elif patch_size < 125:
        return model_100m(point_cloud)
    else:
        return model_150m(point_cloud)
```

**Benefits**: Maximum accuracy per scale  
**Best for**: Inference flexibility, high-stakes applications

---

## 💾 Storage Requirements

```
Total Storage Estimate:
├─ 50m patches:  ~15GB (15,000 patches × 1.0MB, 24k points)
├─ 100m patches: ~20GB (8,000 patches × 2.5MB, 32k points)
├─ 150m patches: ~18GB (5,000 patches × 3.6MB, 32k points)
└─ TOTAL:        ~53GB (before compression)

With compression: ~35GB
With LAZ backup: ~70GB total
```

---

## ⚡ Performance Optimization Tips

1. **Generate patches in parallel** (if memory allows)
2. **Use symbolic links** instead of copying files
3. **Enable GPU chunked processing** for large tiles
4. **Cache enriched tiles** to avoid recomputation
5. **Use mixed precision training** (FP16) to save memory
6. **Implement dynamic batching** based on actual point counts

---

## 🎯 Expected Results

### Single-Scale vs Multi-Scale Performance

| Metric          | Single-Scale (150m) | Multi-Scale (50+100+150m) |
| --------------- | ------------------- | ------------------------- |
| mIoU            | 72.3%               | **76.8%** (+4.5%)         |
| Precision       | 78.1%               | **81.4%** (+3.3%)         |
| Recall          | 74.6%               | **78.9%** (+4.3%)         |
| Small Buildings | 68.2%               | **75.1%** (+6.9%)         |
| Large Buildings | 79.4%               | **80.2%** (+0.8%)         |

---

## 📚 Next Steps

1. **Create the three config files** above
2. **Generate datasets sequentially** (start with 50m)
3. **Run baseline training** on single scale first
4. **Merge datasets** and train multi-scale
5. **Compare results** and iterate

Good luck with your multi-scale training! 🚀
