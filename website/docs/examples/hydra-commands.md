---
sidebar_position: 2
title: Hydra CLI Command Examples
---

# Hydra CLI Command Examples (v2++)

Comprehensive examples of using the Hydra-based CLI for various workflows in version 2.1.2+.

:::tip Modern CLI Architecture
Version 2.0+ combines **Hydra's powerful configuration system** with **Click's intuitive commands**. The `process` command is the main entry point using Hydra for flexible configuration management.
:::

---

## 🚀 Basic Commands

### 1. Minimal Processing

```bash
# Simplest command - uses all defaults from config.yaml
ign-lidar-hd process \
  input_dir=/path/to/raw_tiles \
  output_dir=/path/to/output
```

**What it does:**

- Uses `processor/default.yaml` (CPU, LOD2, 16K points)
- Uses `features/full.yaml` (all geometric features)
- Uses `preprocess/default.yaml` (outlier removal)
- Generates NPZ patches ready for training

### 2. Verify Configuration

```bash
# Print configuration without processing
ign-lidar-hd process \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output \
  --cfg job
```

**Output:** Full composed configuration in YAML format

### 3. Show Available Presets

```bash
# Display all available configurations
ign-lidar-hd info
```

---

## 🎯 Processor Configurations

### CPU Processing (Default)

```bash
ign-lidar-hd process \
  processor=default \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Config values:**

- `use_gpu: false`
- `num_workers: 4`
- `num_points: 16384`
- `lod_level: LOD2`

### GPU Accelerated

```bash
ign-lidar-hd process \
  processor=gpu \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Config values:**

- `use_gpu: true`
- `num_workers: 8`
- `pin_memory: true`
- `prefetch_factor: 4`

### Fast Processing

```bash
ign-lidar-hd process \
  processor=cpu_fast \
  features=minimal \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Config values:**

- Minimal preprocessing
- Fewer points (8192)
- No augmentation
- **Best for:** Quick testing

### Memory Constrained

```bash
ign-lidar-hd process \
  processor=memory_constrained \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Config values:**

- `num_workers: 2`
- `batch_size: 1`
- `num_points: 8192`
- **Best for:** Systems with limited RAM

---

## 🎨 Feature Configurations

### Full Features (Default)

```bash
ign-lidar-hd process \
  features=full \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Includes:**

- All geometric features (normals, curvature, planarity, etc.)
- RGB from IGN orthophotos
- Height statistics
- Verticality

### Minimal Features

```bash
ign-lidar-hd process \
  features=minimal \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Includes:**

- Basic geometric features only
- No RGB
- No NDVI
- **Best for:** Fast processing, testing

### PointNet++ Optimized

```bash
ign-lidar-hd process \
  features=pointnet \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Config values:**

- `sampling_method: fps` (Farthest Point Sampling)
- `normalize_xyz: true`
- `normalize_features: true`
- `k_neighbors: 10`

### Building Features

```bash
ign-lidar-hd process \
  features=buildings \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Optimized for:**

- Building detection
- Architectural features
- Facade analysis

### Vegetation Features

```bash
ign-lidar-hd process \
  features=vegetation \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Includes:**

- NDVI computation
- Infrared values
- Height statistics
- **Best for:** Forest/vegetation analysis

---

## 🧪 Experiment Presets

### LOD3 Training Dataset

```bash
ign-lidar-hd process \
  experiment=config_lod3_training \
  input_dir=data/urban_tiles \
  output_dir=data/lod3_training
```

**Complete configuration:**

```yaml
processor:
  lod_level: LOD3
  architecture: hybrid
  use_gpu: true
  num_points: 32768
  augment: true
  num_augmentations: 3

features:
  mode: full
  k_neighbors: 30
  use_rgb: true
  compute_ndvi: true
  sampling_method: fps

stitching:
  enabled: true
  buffer_size: 20.0
  auto_download_neighbors: true
```

### PointNet++ Training

```bash
ign-lidar-hd process \
  experiment=pointnet_training \
  input_dir=data/tiles \
  output_dir=data/pointnet_dataset
```

**Key features:**

- 16K points with FPS sampling
- Normalized coordinates
- Torch format output
- Augmentation enabled

### LOD2 Building Detection

```bash
ign-lidar-hd process \
  experiment=buildings_lod2 \
  input_dir=data/tiles \
  output_dir=data/buildings_lod2
```

**Optimized for:**

- Simpler building classification
- Faster processing
- CPU-friendly

### Boundary-Aware with Auto-Download

```bash
ign-lidar-hd process \
  experiment=boundary_aware_autodownload \
  input_dir=data/tiles \
  output_dir=data/seamless
```

**Features:**

- Automatic neighbor detection
- Auto-download from IGN WFS
- 20m buffer for boundary features
- Eliminates edge artifacts

---

## ⚙️ Configuration Overrides

### Override Single Values

```bash
# Change number of points
ign-lidar-hd process \
  experiment=pointnet_training \
  processor.num_points=32768 \
  input_dir=data/tiles \
  output_dir=data/output

# Disable GPU
ign-lidar-hd process \
  processor=gpu \
  processor.use_gpu=false \
  input_dir=data/tiles \
  output_dir=data/output

# Change LOD level
ign-lidar-hd process \
  processor.lod_level=LOD3 \
  input_dir=data/tiles \
  output_dir=data/output
```

### Override Multiple Values

```bash
ign-lidar-hd process \
  experiment=pointnet_training \
  processor.num_points=65536 \
  processor.num_workers=16 \
  features.k_neighbors=50 \
  features.use_rgb=false \
  output.format=torch \
  input_dir=data/tiles \
  output_dir=data/output
```

### Override Nested Values

```bash
ign-lidar-hd process \
  processor=gpu \
  input_dir=data/tiles \
  output_dir=data/output \
  preprocess.sor_k=20 \
  preprocess.sor_std=3.0 \
  preprocess.voxel_size=0.5 \
  stitching.buffer_size=30.0
```

---

## 🔄 Multi-Run (Parameter Sweeps)

### Sweep Over Point Counts

```bash
ign-lidar-hd process \
  --multirun \
  processor=gpu \
  processor.num_points=4096,8192,16384,32768 \
  input_dir=data/tiles \
  output_dir=data/sweep_points
```

**Creates:**

- `data/sweep_points/0/` - 4096 points
- `data/sweep_points/1/` - 8192 points
- `data/sweep_points/2/` - 16384 points
- `data/sweep_points/3/` - 32768 points

### Sweep Over K-Neighbors

```bash
ign-lidar-hd process \
  --multirun \
  features.k_neighbors=10,20,30,40,50 \
  input_dir=data/tiles \
  output_dir=data/sweep_knn
```

### Multi-Dimensional Sweep

```bash
ign-lidar-hd process \
  --multirun \
  processor.num_points=8192,16384 \
  features.k_neighbors=20,30 \
  processor.augment=true,false \
  input_dir=data/tiles \
  output_dir=data/grid_search
```

**Creates:** 2 × 2 × 2 = 8 runs

---

## 📁 Custom Config Files

### Using Custom Config

Create `my_experiment.yaml` in `ign_lidar/configs/experiment/`:

```yaml
# @package _global_

defaults:
  - override /processor: gpu
  - override /features: full

processor:
  lod_level: LOD3
  num_points: 65536
  augment: true
  num_augmentations: 10

features:
  k_neighbors: 40
  use_rgb: true
  compute_ndvi: true
  sampling_method: fps
```

Run with:

```bash
ign-lidar-hd process \
  experiment=my_experiment \
  input_dir=data/tiles \
  output_dir=data/custom
```

### Using External Config Path

```bash
ign-lidar-hd process \
  --config-path=/path/to/my/configs \
  --config-name=experiment_config \
  input_dir=data/tiles \
  output_dir=data/output
```

---

## 🎯 Output Formats

### NPZ Format (Default)

```bash
ign-lidar-hd process \
  output.format=npz \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Output:** NumPy compressed archives (`.npz`)

### PyTorch Format

```bash
ign-lidar-hd process \
  output.format=torch \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Output:** PyTorch tensors (`.pt`)

### HDF5 Format

```bash
ign-lidar-hd process \
  output.format=hdf5 \
  input_dir=data/tiles \
  output_dir=data/patches
```

**Output:** HDF5 files (`.h5`)

### Enriched LAZ Only

```bash
ign-lidar-hd process \
  output.save_enriched_laz=true \
  output.only_enriched_laz=true \
  input_dir=data/tiles \
  output_dir=data/enriched
```

**Output:** LAZ files with computed features as extra dimensions (for visualization in CloudCompare)

---

## 📊 Advanced Workflows

### Complete Production Pipeline

```bash
# Full-featured production dataset with GPU
ign-lidar-hd process \
  experiment=config_lod3_training \
  processor.num_workers=16 \
  processor.num_augmentations=5 \
  features.k_neighbors=40 \
  stitching.buffer_size=30.0 \
  output.save_stats=true \
  output.save_metadata=true \
  input_dir=/data/ign/urban_area \
  output_dir=/data/training/lod3_full
```

### Quick Validation Run

```bash
# Fast validation on small dataset
ign-lidar-hd process \
  processor=cpu_fast \
  features=minimal \
  preprocess.enabled=false \
  stitching.enabled=false \
  output.save_stats=false \
  input_dir=data/test_tile \
  output_dir=data/quick_test
```

### Research Experiment

```bash
# High-quality dataset for publication
ign-lidar-hd process \
  processor=gpu \
  processor.num_points=65536 \
  processor.augment=true \
  processor.num_augmentations=10 \
  features=full \
  features.k_neighbors=50 \
  preprocess.sor_k=20 \
  preprocess.sor_std=2.5 \
  stitching.enabled=true \
  stitching.buffer_size=50.0 \
  output.save_enriched_laz=true \
  output.save_stats=true \
  input_dir=data/research_area \
  output_dir=data/publication_dataset
```

---

## 🔍 Debugging and Validation

### Print Configuration

```bash
# Show full composed configuration
ign-lidar-hd process \
  experiment=config_lod3_training \
  input_dir=data/tiles \
  output_dir=data/output \
  --cfg job
```

### Print Available Options

```bash
# Show available config groups
ign-lidar-hd --help
```

### Verbose Logging

```bash
ign-lidar-hd process \
  input_dir=data/tiles \
  output_dir=data/output \
  verbose=true \
  log_level=DEBUG
```

---

## 📚 Additional Resources

- [Configuration System Guide](/guides/configuration-system) - Deep dive into Hydra configs
- [Hydra CLI Guide](/guides/hydra-cli) - Complete CLI reference
- [API Reference](/api/configuration) - Configuration schema documentation
- [Hydra Documentation](https://hydra.cc) - Official Hydra docs
