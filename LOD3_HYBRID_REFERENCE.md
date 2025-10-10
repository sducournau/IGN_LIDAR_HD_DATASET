# 🎯 LOD3 Hybrid Model - Complete Reference

## 📋 Overview

This document provides the complete configuration for LOD3 Hybrid model training, including both CLI processing commands and Python dataset configuration.

**Target:** High-detail building classification with multi-architecture support

**Architectures Supported:**

- ✅ PointNet++ (geometric features)
- ✅ Transformer (multi-modal attention)
- ✅ Octree-CNN (multi-scale hierarchy)
- ✅ Sparse Convolution (voxel-based)

---

## 🔧 Configuration Summary

| Parameter         | Value  | Purpose                    |
| ----------------- | ------ | -------------------------- |
| **LOD Level**     | LOD3   | High-detail classification |
| **Points/Patch**  | 32,768 | 2x LOD2 for complex models |
| **Patch Size**    | 150.0m | Larger for LOD3 complexity |
| **K-Neighbors**   | 30     | Rich geometric features    |
| **Augmentations** | 5x     | Enhanced robustness        |
| **Architecture**  | Hybrid | All models supported       |
| **Sampling**      | FPS    | Farthest Point Sampling    |
| **Format**        | NPZ    | Architecture-agnostic      |

---

## 1️⃣ CLI Processing Command

### Generate Training Patches

```bash
ign-lidar-hd process \
  input_dir=/mnt/c/Users/Simon/ign/raw_tiles/urban_dense \
  output_dir=/mnt/c/Users/Simon/ign/training_patches_lod3_hybrid \
  processor.lod_level=LOD3 \
  processor.architecture=hybrid \
  processor.use_gpu=true \
  processor.num_workers=8 \
  processor.num_points=32768 \
  processor.patch_size=150.0 \
  processor.patch_overlap=0.15 \
  processor.augment=true \
  processor.num_augmentations=5 \
  processor.batch_size=auto \
  processor.prefetch_factor=2 \
  processor.pin_memory=false \
  features.mode=full \
  features.k_neighbors=30 \
  features.include_extra=true \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  features.sampling_method=fps \
  features.normalize_xyz=true \
  features.normalize_features=true \
  features.gpu_batch_size=1000000 \
  features.use_gpu_chunked=true \
  preprocess.enabled=true \
  preprocess.sor_k=20 \
  preprocess.sor_std=1.5 \
  preprocess.ror_radius=0.5 \
  preprocess.ror_neighbors=6 \
  preprocess.voxel_enabled=true \
  preprocess.voxel_size=0.1 \
  stitching.enabled=true \
  stitching.buffer_size=20.0 \
  stitching.auto_detect_neighbors=true \
  stitching.auto_download_neighbors=true \
  stitching.cache_enabled=true \
  output.format=npz \
  output.save_enriched_laz=false \
  output.only_enriched_laz=false \
  output.save_stats=true \
  output.save_metadata=true \
  verbose=true
```

### Quick Script Usage

```bash
# Use the provided script
bash generate_training_patches_lod3_hybrid.sh

# Or with custom paths
bash generate_training_patches_lod3_hybrid.sh /path/to/input /path/to/output
```

---

## 2️⃣ Python Dataset Configuration

### For Training/Validation/Testing

```python
from ign_lidar.datasets import MultiArchDataset

# Training dataset
train_dataset = MultiArchDataset(
    data_dir='/mnt/c/Users/Simon/ign/training_patches_lod3_hybrid',
    architecture='hybrid',      # ⭐ Hybrid mode - supports all architectures
    num_points=32768,           # LOD3 high-detail (2x LOD2)
    use_rgb=True,               # RGB color features
    use_infrared=True,          # NIR infrared features
    use_geometric=True,         # Normals, curvature, etc.
    use_radiometric=True,       # Intensity, return number, etc.
    use_contextual=True,        # Local density, height stats, etc.
    normalize=True,             # Normalize XYZ coordinates
    normalize_rgb=True,         # Normalize RGB to [0, 1]
    standardize_features=True,  # Zero mean, unit variance
    augment=True,               # Data augmentation during training
    split='train',              # Training split
    train_ratio=0.8,            # 80% training
    val_ratio=0.1,              # 10% validation (10% test)
    random_seed=42,             # Reproducibility
    cache_in_memory=False,      # Set True if enough RAM (>32GB)
)

# Validation dataset (no augmentation)
val_dataset = MultiArchDataset(
    data_dir='/mnt/c/Users/Simon/ign/training_patches_lod3_hybrid',
    architecture='hybrid',
    num_points=32768,
    use_rgb=True,
    use_infrared=True,
    use_geometric=True,
    use_radiometric=True,
    use_contextual=True,
    normalize=True,
    normalize_rgb=True,
    standardize_features=True,
    augment=False,              # ❌ No augmentation for validation
    split='val',                # Validation split
    train_ratio=0.8,
    val_ratio=0.1,
    random_seed=42,
    cache_in_memory=False,
)

# Test dataset
test_dataset = MultiArchDataset(
    data_dir='/mnt/c/Users/Simon/ign/training_patches_lod3_hybrid',
    architecture='hybrid',
    num_points=32768,
    use_rgb=True,
    use_infrared=True,
    use_geometric=True,
    use_radiometric=True,
    use_contextual=True,
    normalize=True,
    normalize_rgb=True,
    standardize_features=True,
    augment=False,              # ❌ No augmentation for testing
    split='test',               # Test split
    train_ratio=0.8,
    val_ratio=0.1,
    random_seed=42,
    cache_in_memory=False,
)
```

### PyTorch DataLoader Setup

```python
from torch.utils.data import DataLoader

# Training loader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,               # Adjust based on GPU memory
    shuffle=True,               # Shuffle for training
    num_workers=4,              # Parallel data loading
    pin_memory=True,            # Faster GPU transfer
    drop_last=True,             # Drop incomplete batches
)

# Validation loader
val_loader = DataLoader(
    val_dataset,
    batch_size=16,              # Larger batch for validation
    shuffle=False,              # No shuffle for validation
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

# Test loader
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)
```

---

## 3️⃣ Complete Training Example

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ign_lidar.datasets import MultiArchDataset
from torch.utils.data import DataLoader

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset setup
train_dataset = MultiArchDataset(
    data_dir='/mnt/c/Users/Simon/ign/training_patches_lod3_hybrid',
    architecture='hybrid',
    num_points=32768,
    use_rgb=True,
    use_infrared=True,
    use_geometric=True,
    use_radiometric=True,
    use_contextual=True,
    normalize=True,
    normalize_rgb=True,
    standardize_features=True,
    augment=True,
    split='train',
)

val_dataset = MultiArchDataset(
    data_dir='/mnt/c/Users/Simon/ign/training_patches_lod3_hybrid',
    architecture='hybrid',
    num_points=32768,
    use_rgb=True,
    use_infrared=True,
    use_geometric=True,
    use_radiometric=True,
    use_contextual=True,
    normalize=True,
    normalize_rgb=True,
    standardize_features=True,
    augment=False,
    split='val',
)

# DataLoader setup
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
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

# Model setup (example with your custom model)
# model = YourHybridModel(num_classes=num_classes, ...)
# model = model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 200
best_val_acc = 0.0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, batch in enumerate(train_loader):
        # Get data
        points = batch['points'].to(device)      # [B, N, C]
        labels = batch['labels'].to(device)      # [B, N]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(points)                   # [B, N, num_classes]
        loss = criterion(outputs.transpose(1, 2), labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        train_loss += loss.item()
        pred = outputs.argmax(dim=2)
        train_correct += (pred == labels).sum().item()
        train_total += labels.numel()

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(points)
            loss = criterion(outputs.transpose(1, 2), labels)

            val_loss += loss.item()
            pred = outputs.argmax(dim=2)
            val_correct += (pred == labels).sum().item()
            val_total += labels.numel()

    # Calculate metrics
    train_acc = 100.0 * train_correct / train_total
    val_acc = 100.0 * val_correct / val_total

    # Update learning rate
    scheduler.step()

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss/len(val_loader):.4f} "
          f"Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model_lod3_hybrid.pth')
        print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")

print(f"\n🏆 Training complete! Best validation accuracy: {best_val_acc:.2f}%")
```

---

## 4️⃣ Feature Information

### Data Format (NPZ)

Each patch contains:

```python
{
    'points': np.ndarray,      # [N, C] where C includes:
                               # - xyz (3) normalized coordinates
                               # - rgb (3) normalized colors
                               # - nir (1) infrared intensity
                               # - geometric features (normals, curvature, etc.)
                               # - radiometric features (intensity, return_number, etc.)
                               # - contextual features (local density, height stats, etc.)
    'labels': np.ndarray,      # [N] point-wise class labels
    'tile_id': str,            # Source tile identifier
    'patch_bounds': tuple,     # (xmin, ymin, xmax, ymax)
    'metadata': dict,          # Additional information
}
```

### Feature Dimensions

| Feature Type    | Channels | Description                            |
| --------------- | -------- | -------------------------------------- |
| **XYZ**         | 3        | Normalized coordinates                 |
| **RGB**         | 3        | Color information                      |
| **NIR**         | 1        | Infrared intensity                     |
| **NDVI**        | 1        | Vegetation index                       |
| **Geometric**   | ~10-15   | Normals, curvature, planarity, etc.    |
| **Radiometric** | ~5-8     | Intensity, return number, etc.         |
| **Contextual**  | ~5-10    | Local density, height statistics, etc. |
| **Total**       | ~30-40   | Complete feature vector                |

---

## 5️⃣ Expected Performance

### Hardware Requirements

| Component   | Minimum         | Recommended       |
| ----------- | --------------- | ----------------- |
| **GPU**     | RTX 3060 (12GB) | RTX 3080+ (16GB+) |
| **RAM**     | 32GB            | 64GB+             |
| **Storage** | 500GB SSD       | 1TB+ NVMe SSD     |
| **CPU**     | 6 cores         | 8+ cores          |

### Processing Times (RTX 3080)

| Phase                     | Duration    | Notes                 |
| ------------------------- | ----------- | --------------------- |
| **Patch Generation**      | 4-8 hours   | Depends on input size |
| **Training (200 epochs)** | 16-24 hours | Batch size 8          |
| **Validation**            | 1 hour      | Full validation set   |

### Expected Results

- 🎯 **Training Accuracy:** 92-95%
- 🎯 **Validation Accuracy:** 88-92%
- 🎯 **Test Accuracy:** 87-91%
- 🎯 **Convergence:** ~150-180 epochs

---

## 6️⃣ Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
batch_size=4  # Instead of 8

# Reduce points per patch (fallback)
num_points=16384  # Instead of 32768

# Disable memory cache
cache_in_memory=False
```

### Slow Training

```python
# Enable memory cache (if enough RAM)
cache_in_memory=True

# Increase num_workers
num_workers=8  # Adjust based on CPU cores

# Enable pin_memory
pin_memory=True
```

### Low Accuracy

1. **Check data quality**: Verify preprocessing and stitching
2. **Increase augmentation**: Try more augmentations
3. **Adjust learning rate**: Try different LR schedules
4. **Check class balance**: Use weighted loss if imbalanced

---

## 7️⃣ Next Steps

### After Training

1. **Evaluate on test set**

   ```python
   test_dataset = MultiArchDataset(..., split='test')
   ```

2. **Save final model**

   ```python
   torch.save(model.state_dict(), 'final_model_lod3_hybrid.pth')
   ```

3. **Generate predictions**
   ```bash
   ign-lidar-hd predict \
     model_path=final_model_lod3_hybrid.pth \
     input_dir=/path/to/new/data \
     output_dir=/path/to/predictions
   ```

### Model Deployment

See `DEPLOYMENT.md` for production deployment guidelines.

---

## 📚 References

- **Dataset Documentation**: `ign_lidar/datasets/README.md`
- **CLI Documentation**: `ign_lidar/cli/README.md`
- **Quick Start Guide**: `QUICK_START_LOD3_HYBRID.md`
- **Training Commands**: `TRAINING_COMMANDS.md`

---

**Last Updated:** October 9, 2025
**Version:** 2.1.0+
**Status:** ✅ Production Ready
