---
sidebar_position: 3
title: Configuration Files Reference
---

# Configuration Files Reference

Quick reference for all Hydra configuration files in the library.

---

## 📁 Directory Structure

```
ign_lidar/configs/
├── config.yaml                     # Root configuration
├── processor/                      # Processing backend configs
│   ├── default.yaml
│   ├── gpu.yaml
│   ├── cpu_fast.yaml
│   └── memory_constrained.yaml
├── features/                       # Feature computation configs
│   ├── full.yaml
│   ├── minimal.yaml
│   ├── pointnet.yaml
│   ├── buildings.yaml
│   └── vegetation.yaml
├── preprocess/                     # Preprocessing configs
│   └── default.yaml
├── stitching/                      # Tile stitching configs
│   └── enhanced.yaml
├── output/                         # Output format configs
│   └── default.yaml
└── experiment/                     # Complete experiment presets
    ├── config_lod3_training.yaml
    ├── pointnet_training.yaml
    ├── buildings_lod2.yaml
    ├── buildings_lod3.yaml
    ├── boundary_aware_autodownload.yaml
    ├── boundary_aware_offline.yaml
    ├── fast.yaml
    ├── semantic_sota.yaml
    └── vegetation_ndvi.yaml
```

---

## 🎯 Processor Configs

### `processor/default.yaml`

**CPU-based processing with sensible defaults**

```yaml
lod_level: LOD2 # LOD2 or LOD3
architecture: pointnet++ # pointnet++, octree, transformer, sparse_conv, hybrid, multi
use_gpu: false
num_workers: 4
patch_size: 150.0 # Size in meters
patch_overlap: 0.1 # 10% overlap
num_points: 16384 # Points per patch
augment: false
num_augmentations: 3
batch_size: auto # 'auto' or specific int
prefetch_factor: 2
pin_memory: false
```

**Use case:** Default CPU processing, development, testing

### `processor/gpu.yaml`

**GPU-accelerated processing with CUDA**

```yaml
lod_level: LOD2
architecture: pointnet++
use_gpu: true
num_workers: 8 # More workers for GPU pipeline
patch_size: 150.0
patch_overlap: 0.1
num_points: 16384
augment: false
num_augmentations: 3
batch_size: auto
prefetch_factor: 4 # Higher prefetch for GPU
pin_memory: true # Pin memory for faster GPU transfer
```

**Use case:** Production processing, large datasets, GPU acceleration

### `processor/cpu_fast.yaml`

**Quick CPU processing with minimal features**

```yaml
lod_level: LOD2
architecture: pointnet++
use_gpu: false
num_workers: 4
patch_size: 100.0 # Smaller patches
patch_overlap: 0.05 # Less overlap
num_points: 8192 # Fewer points
augment: false
num_augmentations: 0
batch_size: auto
prefetch_factor: 2
pin_memory: false
```

**Use case:** Quick testing, prototyping, small datasets

### `processor/memory_constrained.yaml`

**Low memory usage for limited RAM systems**

```yaml
lod_level: LOD2
architecture: pointnet++
use_gpu: false
num_workers: 2 # Fewer workers
patch_size: 100.0
patch_overlap: 0.05
num_points: 8192 # Fewer points
augment: false
num_augmentations: 0
batch_size: 1 # Process one at a time
prefetch_factor: 1
pin_memory: false
```

**Use case:** Laptops, systems with &lt;8GB RAM, memory-constrained environments

---

## 🎨 Feature Configs

### `features/full.yaml`

**Complete feature set for high-quality results**

```yaml
mode: full
k_neighbors: 20
include_extra: true # Height stats, verticality, etc.
use_rgb: true # IGN orthophotos
use_infrared: false # NIR (slower)
compute_ndvi: false # NDVI (requires infrared)
sampling_method: random
normalize_xyz: false
normalize_features: false
gpu_batch_size: 1000000
use_gpu_chunked: true
```

**Features computed:**

- Normals (nx, ny, nz)
- Curvature
- Planarity
- Linearity
- Sphericity
- Omnivariance
- Anisotropy
- Eigenentropy
- Sum of eigenvalues
- Change of curvature
- RGB colors (if available)
- Height statistics
- Verticality

**Use case:** High-quality datasets, production, research

### `features/minimal.yaml`

**Basic geometric features only**

```yaml
mode: basic
k_neighbors: 10
include_extra: false
use_rgb: false
use_infrared: false
compute_ndvi: false
sampling_method: random
normalize_xyz: false
normalize_features: false
gpu_batch_size: 1000000
use_gpu_chunked: true
```

**Features computed:**

- Normals (nx, ny, nz)
- Basic geometric features

**Use case:** Fast processing, testing, debugging

### `features/pointnet.yaml`

**Optimized for PointNet++ architecture**

```yaml
mode: full
k_neighbors: 10 # PointNet++ learns neighborhoods
include_extra: true
use_rgb: true
use_infrared: false
compute_ndvi: false
sampling_method: fps # Farthest Point Sampling
normalize_xyz: true # Critical for PointNet++
normalize_features: true # Standardize all features
gpu_batch_size: 1000000
use_gpu_chunked: true
```

**Use case:** PointNet++ training, neural network applications

### `features/buildings.yaml`

**Building-specific features**

```yaml
mode: full
k_neighbors: 20
include_extra: true
use_rgb: true
use_infrared: false
compute_ndvi: false
sampling_method: random
normalize_xyz: false
normalize_features: false
gpu_batch_size: 1000000
use_gpu_chunked: true
```

**Emphasis on:**

- Planarity (for flat surfaces)
- Verticality (for walls)
- Height statistics
- Sharp edges

**Use case:** Building detection, LOD classification

### `features/vegetation.yaml`

**Vegetation analysis with NDVI**

```yaml
mode: full
k_neighbors: 30 # More neighbors for vegetation
include_extra: true
use_rgb: true
use_infrared: true # Enable infrared
compute_ndvi: true # Compute NDVI
sampling_method: random
normalize_xyz: false
normalize_features: false
gpu_batch_size: 1000000
use_gpu_chunked: true
```

**Use case:** Forest analysis, vegetation mapping, NDVI studies

---

## 🔧 Preprocessing Configs

### `preprocess/default.yaml`

**Standard preprocessing pipeline**

```yaml
enabled: true
sor_k: 12 # Statistical Outlier Removal - neighbors
sor_std: 2.0 # Standard deviation threshold
ror_radius: 1.0 # Radius Outlier Removal - radius
ror_neighbors: 3 # Minimum neighbors in radius
voxel_enabled: true # Enable voxel downsampling
voxel_size: 0.25 # Voxel size in meters
```

**Steps:**

1. Statistical Outlier Removal (SOR)
2. Radius Outlier Removal (ROR)
3. Voxel downsampling (optional)

**Use case:** Standard quality, production workflows

---

## 🧩 Stitching Configs

### `stitching/enhanced.yaml`

**Boundary-aware processing with tile stitching**

```yaml
enabled: true
buffer_size: 20.0 # Buffer zone at tile boundaries (meters)
auto_detect_neighbors: true # Automatically detect neighboring tiles
auto_download_neighbors: true # Download missing neighbors from IGN
cache_enabled: true # Cache downloaded tiles
```

**Features:**

- Seamless features across tile boundaries
- Automatic neighbor detection
- Auto-download from IGN WFS
- Eliminates edge artifacts

**Use case:** Multi-tile processing, large areas, production quality

---

## 📦 Output Configs

### `output/default.yaml`

**Default output configuration**

```yaml
format: npz # npz, torch, hdf5
save_enriched_laz: false # Save LAZ with computed features
only_enriched_laz: false # Only save LAZ (no patches)
save_stats: true # Save processing statistics
save_metadata: true # Save patch metadata
compression: null # Compression level (null, 1-9)
```

**Output files:**

- Patches in specified format (NPZ/PyTorch/HDF5)
- Statistics JSON
- Metadata JSON
- Optional enriched LAZ

**Use case:** Standard ML training datasets

---

## 🧪 Experiment Presets

### `experiment/config_lod3_training.yaml`

**LOD3 hybrid model training dataset**

```yaml
# @package _global_
defaults:
  - override /processor: gpu
  - override /features: full
  - override /preprocess: default
  - override /stitching: enhanced
  - override /output: default

processor:
  lod_level: LOD3
  architecture: hybrid
  use_gpu: true
  num_workers: 1
  patch_size: 150.0
  patch_overlap: 0.1
  num_points: 32768 # High density for LOD3
  augment: true
  num_augmentations: 3

features:
  mode: full
  k_neighbors: 30
  include_extra: true
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
  sampling_method: fps # Farthest Point Sampling
  normalize_xyz: true
  normalize_features: true

preprocess:
  enabled: true
  sor_k: 12
  sor_std: 2.0
  voxel_enabled: true
  voxel_size: 0.25

stitching:
  enabled: true
  buffer_size: 20.0
  auto_detect_neighbors: true
  auto_download_neighbors: true

output:
  format: npz
  save_enriched_laz: false
  save_stats: true
  save_metadata: true
```

**Command:**

```bash
ign-lidar-hd \
  experiment=config_lod3_training \
  input_dir=data/tiles \
  output_dir=data/lod3_training
```

**Use case:** Training LOD3 building classification models

### `experiment/pointnet_training.yaml`

**PointNet++ optimized training dataset**

```yaml
# @package _global_
defaults:
  - override /processor: gpu
  - override /features: pointnet
  - override /preprocess: default
  - override /stitching: enhanced
  - override /output: default

processor:
  lod_level: LOD2
  num_points: 16384 # Standard for PointNet++
  augment: true
  num_augmentations: 5
  use_gpu: true

features:
  mode: full
  k_neighbors: 10 # PointNet++ learns neighborhoods
  include_extra: true
  use_rgb: true
  use_infrared: false
  sampling_method: fps # Critical for PointNet++
  normalize_xyz: true # Essential for PointNet++
  normalize_features: true

preprocess:
  enabled: true

stitching:
  enabled: true
  buffer_size: 10.0

output:
  format: torch # PyTorch format
  save_stats: true
```

**Command:**

```bash
ign-lidar-hd \
  experiment=pointnet_training \
  input_dir=data/tiles \
  output_dir=data/pointnet
```

**Use case:** PointNet++ neural network training

### `experiment/boundary_aware_autodownload.yaml`

**Automatic neighbor download for seamless processing**

```yaml
# @package _global_
defaults:
  - override /processor: default
  - override /features: full
  - override /preprocess: default
  - override /stitching: enhanced
  - override /output: default

stitching:
  enabled: true
  buffer_size: 20.0
  auto_detect_neighbors: true
  auto_download_neighbors: true # Download missing neighbors
  cache_enabled: true
```

**Command:**

```bash
ign-lidar-hd \
  experiment=boundary_aware_autodownload \
  input_dir=data/tiles \
  output_dir=data/seamless
```

**Use case:** Large-scale processing, automatic tile management

### `experiment/fast.yaml`

**Quick processing for testing**

```yaml
# @package _global_
defaults:
  - override /processor: cpu_fast
  - override /features: minimal
  - override /preprocess: default
  - override /output: default

processor:
  num_points: 4096 # Very few points

features:
  k_neighbors: 10
  use_rgb: false

preprocess:
  enabled: false # Skip preprocessing

stitching:
  enabled: false # Skip stitching

output:
  save_stats: false
  save_metadata: false
```

**Command:**

```bash
ign-lidar-hd \
  experiment=fast \
  input_dir=data/test \
  output_dir=data/quick_test
```

**Use case:** Quick validation, pipeline testing

---

## 🎛️ Override Examples

### Single Parameter

```bash
# Change number of points
ign-lidar-hd \
  processor.num_points=32768 \
  input_dir=data/tiles \
  output_dir=data/output
```

### Multiple Parameters

```bash
# Customize processing
ign-lidar-hd \
  processor=gpu \
  processor.num_points=65536 \
  processor.num_workers=16 \
  features.k_neighbors=40 \
  input_dir=data/tiles \
  output_dir=data/output
```

### Mix Preset + Overrides

```bash
# Start with preset, then customize
ign-lidar-hd \
  experiment=pointnet_training \
  processor.num_points=32768 \
  features.k_neighbors=20 \
  output.format=hdf5 \
  input_dir=data/tiles \
  output_dir=data/output
```

---

## 📖 Related Documentation

- [Hydra CLI Guide](/guides/hydra-cli) - Complete CLI reference
- [Configuration System](/guides/configuration-system) - Deep dive
- [Command Examples](/examples/hydra-commands) - Practical examples
- [Official Hydra Docs](https://hydra.cc) - Hydra framework documentation
