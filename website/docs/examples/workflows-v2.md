---
sidebar_position: 5
title: Real-World Workflows (v2++)
description: Complete end-to-end workflows for common use cases
keywords: [workflows, examples, use-cases, production]
---

# Real-World Workflows (v2++)

Complete end-to-end workflows for common IGN LiDAR HD processing scenarios using version 2.2.0+.

:::tip Production-Ready Examples
These workflows are battle-tested on 50+ tiles and represent best practices for different use cases. Copy and adapt them to your needs!
:::

---

## 🏙️ Workflow 1: Urban Building Classification

**Goal:** Create a high-quality LOD3 dataset for urban building classification.

**Dataset characteristics:**

- Dense urban environment
- 5 building classes (LOD3)
- RGB + NIR + NDVI features
- GPU-accelerated processing
- Boundary-aware features

### Step 1: Download Urban Tiles

```bash
# Download 10km radius around Paris center
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 10000 \
  --max-concurrent 5 \
  data/raw_urban
```

**Expected output:**

- 20-50 tiles depending on area
- ~500MB-2GB per tile
- Total: ~10-50GB

### Step 2: Process with LOD3 Configuration

```bash
# Complete LOD3 training dataset
ign-lidar-hd process \
  experiment=config_lod3_training \
  input_dir=data/raw_urban \
  output_dir=data/urban_lod3_training \
  processor.use_gpu=true \
  processor.num_workers=8 \
  processor.num_augmentations=5 \
  features.k_neighbors=30 \
  stitching.buffer_size=20.0 \
  output.save_enriched_laz=true \
  output.save_stats=true \
  log_level=INFO
```

**What this does:**

- ✅ LOD3 classification (5 classes)
- ✅ 32,768 points per patch
- ✅ 5x geometric augmentation
- ✅ RGB + NIR + NDVI features
- ✅ Tile stitching eliminates edge artifacts
- ✅ GPU acceleration (10-20x faster)
- ✅ Saves enriched LAZ for visualization

**Processing time:**

- GPU: ~3-5 minutes per tile (50k points)
- CPU: ~30-60 minutes per tile

### Step 3: Verify Dataset Quality

```bash
# Comprehensive validation
ign-lidar-hd verify data/urban_lod3_training \
  --detailed \
  --output urban_validation_report.json

# Check stats
cat urban_validation_report.json | jq '.summary'
```

### Step 4: Convert Sample for Visualization

```bash
# Create QGIS-compatible files for manual inspection
ign-lidar-hd batch-convert \
  data/urban_lod3_training \
  --output data/urban_preview \
  --format qgis \
  --max-points 100000 \
  --batch-size 10
```

### Expected Output Structure

```text
data/
├── raw_urban/                      # Raw LAZ tiles
│   ├── LHD_FXX_0649_6863.laz
│   └── ...
├── urban_lod3_training/           # ML-ready dataset
│   ├── patches/                   # NPZ patches
│   │   ├── LHD_FXX_0649_6863_patch_0000.npz
│   │   ├── LHD_FXX_0649_6863_patch_0001.npz
│   │   └── ...
│   ├── enriched_laz/             # Enriched LAZ files
│   │   ├── LHD_FXX_0649_6863_enriched.laz
│   │   └── ...
│   └── stats/                     # Processing statistics
│       └── processing_stats.json
└── urban_preview/                 # QGIS visualization
    └── ...
```

---

## 🌲 Workflow 2: Forest Vegetation Analysis

**Goal:** Multi-modal dataset for vegetation/tree classification and NDVI analysis.

**Dataset characteristics:**

- Forest and vegetation
- RGB + Near-Infrared + NDVI
- Lower point density
- Focus on height features

### Complete Workflow

```bash
# 1. Download forest area (Fontainebleau example)
ign-lidar-hd download \
  --position 648000 5380000 \
  --radius 8000 \
  data/raw_forest

# 2. Process with vegetation features
ign-lidar-hd process \
  features=vegetation \
  input_dir=data/raw_forest \
  output_dir=data/forest_multimodal \
  processor.lod_level=LOD2 \
  processor.num_points=16384 \
  processor.augment=true \
  processor.num_augmentations=3 \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  features.k_neighbors=20 \
  output.format=all \
  output.save_enriched_laz=true

# 3. Verify multi-modal features
ign-lidar-hd verify data/forest_multimodal \
  --detailed \
  --output forest_validation.json

# 4. Check NDVI statistics
python -c "
import numpy as np
import glob

files = glob.glob('data/forest_multimodal/patches/*.npz')
ndvi_values = []

for f in files[:10]:  # Sample first 10
    data = np.load(f)
    if 'ndvi' in data:
        ndvi_values.extend(data['ndvi'].flatten())

print(f'NDVI range: [{min(ndvi_values):.3f}, {max(ndvi_values):.3f}]')
print(f'NDVI mean: {np.mean(ndvi_values):.3f}')
"
```

**Key features:**

- 📊 NDVI computed from RGB + NIR
- 🌿 Height-based vegetation features
- 🎨 Multi-modal: Geometry + RGB + NIR
- 📁 Multiple output formats (NPZ, HDF5, PyTorch)

---

## 🎓 Workflow 3: Research Dataset with PointNet++

**Goal:** High-quality dataset optimized for PointNet++ architecture training.

**Dataset characteristics:**

- Farthest Point Sampling (FPS)
- Normalized coordinates and features
- PyTorch format output
- Extensive augmentation

### Complete Workflow

```bash
# 1. Download diverse area (mixed urban/suburban)
ign-lidar-hd download \
  --bbox 640000 6850000 660000 6870000 \
  --max-concurrent 8 \
  data/raw_research

# 2. Process with PointNet++ optimizations
ign-lidar-hd process \
  experiment=pointnet_training \
  input_dir=data/raw_research \
  output_dir=data/pointnet_research \
  processor.num_points=16384 \
  processor.augment=true \
  processor.num_augmentations=10 \
  processor.use_gpu=true \
  features.sampling_method=fps \
  features.normalize_xyz=true \
  features.normalize_features=true \
  features.k_neighbors=16 \
  output.format=torch \
  output.save_stats=true \
  output.save_metadata=true \
  stitching.enabled=true

# 3. Verify dataset
ign-lidar-hd verify data/pointnet_research \
  --detailed \
  --output pointnet_validation.json

# 4. Split train/val/test
python -c "
import shutil
from pathlib import Path
import random

patches = list(Path('data/pointnet_research/patches').glob('*.pt'))
random.shuffle(patches)

total = len(patches)
train_end = int(total * 0.7)
val_end = int(total * 0.85)

splits = {
    'train': patches[:train_end],
    'val': patches[train_end:val_end],
    'test': patches[val_end:]
}

for split, files in splits.items():
    split_dir = Path(f'data/pointnet_research/{split}')
    split_dir.mkdir(exist_ok=True)
    for f in files:
        shutil.copy(f, split_dir / f.name)
    print(f'{split}: {len(files)} patches')
"
```

**PointNet++ specific features:**

- ✨ Farthest Point Sampling for better coverage
- 📏 Normalized coordinates (mean=0, std=1)
- ⚖️ Normalized features
- 🔥 PyTorch tensor format (`.pt`)
- 🔄 10x augmentation for robustness

---

## ⚡ Workflow 4: Fast Prototyping

**Goal:** Quick dataset generation for algorithm testing and prototyping.

**Dataset characteristics:**

- Minimal features
- Small area
- Fast CPU processing
- No augmentation

### Complete Workflow

```bash
# 1. Download small test area
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 2000 \
  data/raw_test

# 2. Fast processing
ign-lidar-hd process \
  processor=cpu_fast \
  features=minimal \
  input_dir=data/raw_test \
  output_dir=data/test_patches \
  processor.num_points=8192 \
  processor.augment=false \
  preprocess.enabled=true \
  stitching.enabled=false \
  output.save_enriched_laz=false

# 3. Quick verification
ign-lidar-hd verify data/test_patches

# 4. Test your algorithm
python your_algorithm.py --data data/test_patches
```

**Processing time:**

- ~5-10 minutes for 2km radius
- Perfect for CI/CD pipelines

---

## 🔬 Workflow 5: Boundary Artifacts Research

**Goal:** Generate seamless cross-tile dataset for boundary feature research.

**Dataset characteristics:**

- Automatic neighbor detection
- Auto-download missing tiles
- Large buffer zones
- Eliminates edge artifacts

### Complete Workflow

```bash
# 1. Download initial tiles
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 5000 \
  data/raw_boundary_test

# 2. Process with boundary-aware configuration
ign-lidar-hd process \
  experiment=boundary_aware_autodownload \
  input_dir=data/raw_boundary_test \
  output_dir=data/seamless_dataset \
  stitching.buffer_size=30.0 \
  stitching.auto_download_neighbors=true \
  stitching.cache_enabled=true \
  processor.use_gpu=true \
  features.k_neighbors=40 \
  output.save_enriched_laz=true

# 3. Verify no boundary artifacts
ign-lidar-hd verify data/seamless_dataset \
  --detailed \
  --features-only

# 4. Visualize tile boundaries in QGIS
ign-lidar-hd batch-convert \
  data/seamless_dataset/enriched_laz \
  --output data/boundary_visualization \
  --format qgis
```

**Key features:**

- 🔄 Auto-downloads neighbor tiles as needed
- 📊 30m buffer for feature computation
- 🧩 Seamless cross-tile features
- ✅ Validates boundary feature quality

---

## 📦 Workflow 6: Production Pipeline with Data Versioning

**Goal:** Production-ready pipeline with full traceability and versioning.

### Complete Workflow with DVC

```bash
# Initialize DVC for data versioning
cd data
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# 1. Download and version raw data
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 10000 \
  raw_tiles

dvc add raw_tiles
git add raw_tiles.dvc
git commit -m "Add raw LiDAR tiles v1.0"

# 2. Process with full configuration
ign-lidar-hd process \
  experiment=config_lod3_training \
  input_dir=raw_tiles \
  output_dir=training_dataset_v1 \
  processor.use_gpu=true \
  processor.num_workers=16 \
  processor.num_augmentations=5 \
  features.k_neighbors=30 \
  output.save_enriched_laz=true \
  output.save_stats=true \
  output.save_metadata=true

# 3. Version processed dataset
dvc add training_dataset_v1
git add training_dataset_v1.dvc
git commit -m "Add training dataset v1.0"

# 4. Generate dataset card
python -c "
import json
from pathlib import Path
import datetime

stats_file = 'training_dataset_v1/stats/processing_stats.json'
with open(stats_file) as f:
    stats = json.load(f)

card = {
    'dataset_name': 'IGN LiDAR Urban LOD3 v1.0',
    'creation_date': datetime.datetime.now().isoformat(),
    'num_patches': len(list(Path('training_dataset_v1/patches').glob('*.npz'))),
    'processing_config': stats.get('config', {}),
    'version': '1.0',
    'license': 'IGN Open License',
}

with open('training_dataset_v1/DATASET_CARD.json', 'w') as f:
    json.dump(card, f, indent=2)

print('Dataset card created!')
"

git add training_dataset_v1/DATASET_CARD.json
git commit -m "Add dataset card"

# 5. Push to remote storage
dvc remote add -d storage s3://my-lidar-datasets
dvc push
git push
```

**Benefits:**

- 📝 Full version control
- 🔄 Reproducible processing
- 📊 Complete metadata
- 🌐 Shareable datasets

---

## 🎯 Workflow 7: Enriched LAZ Only (No Patches)

**Goal:** Generate enriched LAZ files for visualization and manual analysis.

**Use cases:**

- QGIS/CloudCompare visualization
- Manual quality control
- External processing pipelines
- Feature debugging

### Complete Workflow

```bash
# 1. Download tiles
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 5000 \
  data/raw_tiles

# 2. Generate only enriched LAZ (no patches)
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/enriched_laz_only \
  output=enriched_only \
  processor.use_gpu=true \
  features=full \
  features.use_rgb=true \
  features.compute_ndvi=true \
  preprocess.enabled=true \
  stitching.enabled=true

# 3. Verify LAZ files
ign-lidar-hd verify data/enriched_laz_only \
  --detailed

# 4. Open in CloudCompare or QGIS for visualization
# The enriched LAZ files contain all computed features as extra dimensions
```

**Output LAZ dimensions:**

- XYZ coordinates
- Intensity
- Classification
- RGB colors
- Surface normals (nx, ny, nz)
- Curvature
- Planarity
- Verticality
- Horizontality
- And more...

---

## 💡 Tips for All Workflows

### Performance Optimization

```bash
# GPU acceleration (10-20x speedup)
processor.use_gpu=true

# Parallel processing
processor.num_workers=16  # Adjust based on CPU cores

# Memory management
processor.batch_size=auto  # Or specify manually
```

### Quality Control

```bash
# Always verify after processing
ign-lidar-hd verify <output_dir> --detailed

# Save processing stats
output.save_stats=true
output.save_metadata=true

# Enable verbose logging
log_level=DEBUG
```

### Storage Management

```bash
# Use compression
output.compression=gzip

# Skip enriched LAZ if not needed
output.save_enriched_laz=false

# Choose appropriate output format
output.format=npz  # Smallest, default
output.format=torch  # PyTorch-ready (requires torch)
output.format=hdf5  # Large datasets, with compression
output.format=laz  # Visualization in CloudCompare/QGIS
output.format=hdf5,laz  # Both training and visualization (v2.2.0+)
```

### Configuration Management

```bash
# Save configuration for reproducibility
ign-lidar-hd process \
  experiment=my_config \
  ... \
  --cfg job > config_used.yaml

# Use config files
ign-lidar-hd process \
  --config-path=/path/to/configs \
  --config-name=production
```

---

## 📚 Next Steps

- [CLI Command Reference](./cli-commands) - Complete command documentation
- [Configuration Examples](./config-reference) - YAML configuration examples
- [API Reference](../api/processor) - Python API documentation
- [GPU Acceleration](../guides/gpu-acceleration) - GPU setup and optimization

---

## 🆘 Troubleshooting Workflows

### Out of Memory

```bash
# Reduce memory usage
processor=memory_constrained
processor.num_workers=2
processor.batch_size=1
processor.num_points=8192
```

### Slow Processing

```bash
# Enable GPU
processor=gpu

# Or reduce features
features=minimal
preprocess.enabled=false
```

### Missing RGB/NIR

```bash
# Disable if not needed
features.use_rgb=false
features.use_infrared=false
features.compute_ndvi=false
```

### Boundary Artifacts

```bash
# Enable tile stitching
stitching.enabled=true
stitching.buffer_size=20.0
stitching.auto_download_neighbors=true
```
