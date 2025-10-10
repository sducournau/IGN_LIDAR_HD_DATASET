---
sidebar_position: 6
title: Système de Configuration
---

<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->
<!-- Ce fichier est un modèle qui nécessite une traduction manuelle. -->
<!-- Veuillez traduire le contenu ci-dessous en conservant : -->
<!-- - Le frontmatter (métadonnées en haut) -->
<!-- - Les blocs de code (traduire uniquement les commentaires) -->
<!-- - Les liens et chemins de fichiers -->
<!-- - La structure Markdown -->



# Configuration System

Deep dive into the Hydra-based configuration system in v2.0+.

---

## 🎯 Overview

The v2.0 configuration system uses [Hydra](https://hydra.cc) for powerful, composable, type-safe configuration management.

### Key Concepts

- **Hierarchical Composition**: Build complex configs from simple components
- **Type Safety**: Runtime validation of configuration values
- **Presets**: Pre-configured workflows (fast/balanced/quality/ultra)
- **Overrides**: Change any parameter from command line
- **Config Groups**: Organize related configurations
- **Defaults**: Sensible defaults for all parameters

---

## 📁 Configuration Structure

```
ign_lidar/configs/
├── config.yaml                 # Root configuration
├── preset/                     # Workflow presets
│   ├── fast.yaml
│   ├── balanced.yaml
│   ├── quality.yaml
│   └── ultra.yaml
├── processor/                  # Processing backend
│   ├── cpu.yaml
│   └── gpu.yaml
├── features/                   # Feature computation
│   ├── basic.yaml
│   ├── standard.yaml
│   └── full.yaml
├── preprocess/                 # Preprocessing
│   ├── minimal.yaml
│   ├── standard.yaml
│   └── aggressive.yaml
├── dataset/                    # Dataset generation
│   ├── pointnet.yaml
│   ├── octree.yaml
│   └── transformer.yaml
└── stitching/                  # Tile stitching
    ├── none.yaml
    ├── features.yaml
    └── full.yaml
```

---

## 🔧 Root Configuration

**config.yaml** - The main configuration file:

```yaml
defaults:
  - preset: balanced # Default preset
  - processor: cpu # Default processor
  - features: standard # Default features
  - preprocess: standard # Default preprocessing
  - dataset: pointnet # Default dataset format
  - stitching: none # Default stitching
  - _self_ # Allow overrides

# Core parameters (required)
input_dir: ??? # Must be specified
output_dir: ??? # Must be specified

# Output mode
output: patches # Options: patches, both, enriched_only

# Performance
num_workers: 4 # Number of parallel workers
batch_size: 1 # Tiles per batch
max_memory_gb: 8 # Maximum RAM usage

# Logging
verbose: false # Verbose output
log_level: INFO # Logging level
show_progress: true # Show progress bars

# Hydra configuration
hydra:
  run:
    dir: ${output_dir}/.hydra # Hydra output directory
  job:
    chdir: false # Don't change working directory
```

### Required Parameters

These **must** be provided (marked with `???`):

```yaml
input_dir: ??? # Path to RAW LAZ files
output_dir: ??? # Output directory
```

**Usage:**

```bash
# Provide required parameters
ign-lidar-hd process input_dir=data/ output_dir=output/
```

---

## 🎨 Preset Configurations

Presets are complete workflow configurations optimized for different use cases.

### 1. Fast Preset

**preset/fast.yaml:**

```yaml
# @package _global_

# Override defaults for fast processing
defaults:
  - override /processor: cpu
  - override /features: basic
  - override /preprocess: minimal

# Fast-specific parameters
features:
  k_neighbors: 20 # Fewer neighbors
  radius: 1.5 # Smaller radius
  use_rgb: false # Skip RGB
  compute_ndvi: false # Skip NDVI
  boundary_aware: false # No boundary processing

preprocess:
  remove_outliers: true
  outlier_method: statistical
  outlier_nb_neighbors: 10 # Fewer neighbors
  normalize_ground: false # Skip normalization

dataset:
  patch_size: 50.0
  points_per_patch: 2048 # Fewer points

# Performance
num_workers: 2
```

### 2. Balanced Preset (⭐ Recommended)

**preset/balanced.yaml:**

```yaml
# @package _global_

defaults:
  - override /processor: cpu
  - override /features: standard
  - override /preprocess: standard

features:
  k_neighbors: 30
  radius: 2.0
  use_rgb: true # Use RGB if available
  compute_ndvi: false
  boundary_aware: false

preprocess:
  remove_outliers: true
  outlier_method: statistical
  outlier_nb_neighbors: 20
  outlier_std_ratio: 2.0
  normalize_ground: true
  ground_resolution: 1.0

dataset:
  patch_size: 50.0
  points_per_patch: 4096

num_workers: 4
```

### 3. Quality Preset

**preset/quality.yaml:**

```yaml
# @package _global_

defaults:
  - override /processor: gpu # GPU for better performance
  - override /features: full
  - override /preprocess: aggressive

features:
  k_neighbors: 50 # More neighbors
  radius: 3.0 # Larger radius
  use_rgb: true
  compute_ndvi: true # Compute NDVI
  use_infrared: true # Use infrared
  boundary_aware: false

preprocess:
  remove_outliers: true
  outlier_method: statistical
  outlier_nb_neighbors: 30
  outlier_std_ratio: 1.5 # More aggressive
  normalize_ground: true
  ground_resolution: 0.5 # Finer resolution
  filter_classes: true

dataset:
  patch_size: 50.0
  points_per_patch: 8192 # More points per patch

num_workers: 8
output: both # Save both LAZ and patches
```

### 4. Ultra Preset

**preset/ultra.yaml:**

```yaml
# @package _global_

defaults:
  - override /processor: gpu
  - override /features: full
  - override /preprocess: aggressive
  - override /stitching: full # Enable full stitching

features:
  k_neighbors: 100 # Maximum neighbors
  radius: 5.0 # Large radius
  use_rgb: true
  compute_ndvi: true
  use_infrared: true
  boundary_aware: true # Enable boundary-aware
  boundary_buffer: 15.0 # Large buffer

preprocess:
  remove_outliers: true
  outlier_method: statistical
  outlier_nb_neighbors: 50
  outlier_std_ratio: 1.0 # Very aggressive
  normalize_ground: true
  ground_resolution: 0.25 # Very fine resolution
  filter_classes: true

dataset:
  patch_size: 50.0
  points_per_patch: 16384 # Maximum points

stitching:
  tile_overlap: 10.0 # Tile overlap
  merge_method: weighted # Weighted merging

num_workers: 12
output: both
```

---

## 💻 Processor Configurations

### CPU Processor

**processor/cpu.yaml:**

```yaml
# CPU processing configuration
_target_: cpu

# CPU-specific parameters
use_gpu: false
num_threads: null # Use all available
chunk_size: 500000 # Points per chunk
parallel_backend: threading # or multiprocessing
```

### GPU Processor

**processor/gpu.yaml:**

```yaml
# GPU processing configuration
_target_: gpu

# GPU parameters
use_gpu: true
gpu_id: 0 # GPU device ID
gpu_memory_fraction: 0.9 # Fraction of GPU memory
gpu_chunk_size: 1000000 # Points per GPU chunk

# Fallback
cpu_fallback: true # Use CPU if GPU fails
```

---

## 🔬 Feature Configurations

### Basic Features

**features/basic.yaml:**

```yaml
# Basic geometric features only
compute_geometric: true
compute_curvature: false
use_rgb: false
compute_ndvi: false
use_infrared: false
boundary_aware: false

# Quality
k_neighbors: 20
radius: 1.5

# Performance
use_gpu: false
```

### Standard Features

**features/standard.yaml:**

```yaml
# Standard feature set
compute_geometric: true
compute_curvature: true
use_rgb: true # If available
compute_ndvi: false
use_infrared: false
boundary_aware: false

# Quality
k_neighbors: 30
radius: 2.0

# Performance
use_gpu: false
auto_params: false
```

### Full Features

**features/full.yaml:**

```yaml
# Complete feature set
compute_geometric: true
compute_curvature: true
compute_intensity_stats: true
use_rgb: true
compute_ndvi: true
use_infrared: true
boundary_aware: false

# Quality
k_neighbors: 50
radius: 3.0

# Performance
use_gpu: true
auto_params: true
```

---

## 🔧 Preprocessing Configurations

### Minimal Preprocessing

**preprocess/minimal.yaml:**

```yaml
# Minimal preprocessing (outliers only)
remove_outliers: true
outlier_method: statistical
outlier_nb_neighbors: 10
outlier_std_ratio: 3.0

normalize_ground: false
filter_classes: false
```

### Standard Preprocessing

**preprocess/standard.yaml:**

```yaml
# Standard preprocessing
remove_outliers: true
outlier_method: statistical
outlier_nb_neighbors: 20
outlier_std_ratio: 2.0

normalize_ground: true
ground_resolution: 1.0
ground_max_distance: 5.0

filter_classes: false
```

### Aggressive Preprocessing

**preprocess/aggressive.yaml:**

```yaml
# Aggressive preprocessing
remove_outliers: true
outlier_method: statistical
outlier_nb_neighbors: 30
outlier_std_ratio: 1.5 # More aggressive

normalize_ground: true
ground_resolution: 0.5 # Finer
ground_max_distance: 3.0 # Stricter

filter_classes: true
keep_classes: [2, 3, 4, 5, 6] # Ground, vegetation, building
```

---

## 🗂️ Dataset Configurations

### PointNet++ Format

**dataset/pointnet.yaml:**

```yaml
# PointNet++ dataset format
architecture: pointnet++
patch_size: 50.0
points_per_patch: 4096
overlap: 0.0
normalize_points: true
center_points: true
augment: true
```

### Octree Format

**dataset/octree.yaml:**

```yaml
# Octree dataset format
architecture: octree
patch_size: 50.0
points_per_patch: 8192 # More points for octree
octree_depth: 7
overlap: 0.0
normalize_points: true
```

### Transformer Format

**dataset/transformer.yaml:**

```yaml
# Transformer dataset format
architecture: transformer
patch_size: 50.0
points_per_patch: 2048 # Fewer points (attention complexity)
overlap: 0.0
normalize_points: true
positional_encoding: learned
```

---

## 🧩 Stitching Configurations

### No Stitching

**stitching/none.yaml:**

```yaml
# Independent tile processing
enabled: false
tile_overlap: 0.0
```

### Feature Stitching

**stitching/features.yaml:**

```yaml
# Boundary-aware features only
enabled: true
mode: features
tile_overlap: 0.0
boundary_buffer: 10.0
merge_method: average
```

### Full Stitching

**stitching/full.yaml:**

```yaml
# Complete stitching
enabled: true
mode: full
tile_overlap: 10.0
boundary_buffer: 15.0
merge_method: weighted
create_merged_dataset: true
```

---

## 🎭 Using Configurations

### 1. Command-Line Selection

```bash
# Select preset
ign-lidar-hd process preset=quality

# Select multiple config groups
ign-lidar-hd process \
  preset=quality \
  processor=gpu \
  features=full
```

### 2. Parameter Overrides

```bash
# Override individual parameters
ign-lidar-hd process \
  preset=balanced \
  features.k_neighbors=50 \
  num_workers=8
```

### 3. Custom Config Files

**my_config.yaml:**

```yaml
defaults:
  - preset: balanced
  - processor: gpu
  - _self_

input_dir: "/data/project1/"
output_dir: "/output/project1/"

features:
  use_rgb: true
  compute_ndvi: true
  k_neighbors: 40

num_workers: 12
```

**Usage:**

```bash
ign-lidar-hd process --config-name my_config
```

### 4. Config Composition

**base.yaml:**

```yaml
# Base configuration
num_workers: 4
verbose: false
log_level: INFO
```

**urban.yaml:**

```yaml
defaults:
  - base
  - preset: balanced
  - _self_

# Urban-specific settings
preprocess:
  filter_classes: true
  keep_classes: [6] # Buildings only

features:
  use_rgb: true
```

**Usage:**

```bash
ign-lidar-hd process --config-name urban input_dir=urban/ output_dir=out/
```

---

## 🔍 Configuration Introspection

### View Current Configuration

```bash
# Show resolved configuration
ign-lidar-hd process \
  preset=balanced \
  processor=gpu \
  --cfg job
```

### View Configuration Groups

```bash
# List available presets
ign-lidar-hd process --help | grep preset

# List available processors
ign-lidar-hd process --help | grep processor
```

### Validate Configuration

```bash
# Check configuration without running
ign-lidar-hd process \
  --config-name my_config \
  --cfg job
```

---

## 🎓 Advanced Patterns

### 1. Environment Variables

```yaml
# Use environment variables
input_dir: ${oc.env:DATA_DIR}
output_dir: ${oc.env:OUTPUT_DIR}

# With defaults
num_workers: ${oc.env:NUM_WORKERS,4}
```

**Usage:**

```bash
export DATA_DIR=/data/project/
export OUTPUT_DIR=/output/project/
ign-lidar-hd process
```

### 2. Conditional Configuration

```yaml
# Conditional GPU usage
features:
  use_gpu: ${oc.env:CUDA_VISIBLE_DEVICES,false}
  gpu_chunk_size: ${oc.select:features.use_gpu,1000000,null}
```

### 3. Config Inheritance

**base_research.yaml:**

```yaml
# Base research configuration
defaults:
  - preset: quality
  - processor: gpu

num_workers: 12
output: both
```

**experiment_1.yaml:**

```yaml
# Experiment 1: Different k_neighbors
defaults:
  - base_research
  - _self_

features:
  k_neighbors: 30
```

**experiment_2.yaml:**

```yaml
# Experiment 2: Boundary-aware
defaults:
  - base_research
  - _self_

features:
  k_neighbors: 50
  boundary_aware: true
```

### 4. Parameter Sweeps

```bash
# Run multiple experiments
ign-lidar-hd process \
  --config-name base_research \
  --multirun \
  features.k_neighbors=20,30,40,50
```

Creates:

- `output/multirun/0/` - k_neighbors=20
- `output/multirun/1/` - k_neighbors=30
- `output/multirun/2/` - k_neighbors=40
- `output/multirun/3/` - k_neighbors=50

---

## 📊 Configuration Best Practices

### 1. Use Presets as Starting Points

```bash
# Start with preset, then customize
ign-lidar-hd process \
  preset=balanced \
  features.use_rgb=true \
  num_workers=8
```

### 2. Create Project-Specific Configs

```yaml
# project_config.yaml
defaults:
  - preset: balanced
  - _self_

input_dir: "/project/data/"
output_dir: "/project/output/"

# Project-specific overrides
features:
  use_rgb: true
  compute_ndvi: true
```

### 3. Use Config Groups for Reusability

```
configs/
├── base.yaml
├── urban/
│   ├── buildings.yaml
│   └── vegetation.yaml
└── rural/
    └── agriculture.yaml
```

### 4. Document Your Configs

```yaml
# my_config.yaml
# Purpose: Process urban tiles for building classification
# Created: 2025-10-08
# Author: Your Name

defaults:
  - preset: quality
  - _self_

# Buildings only
preprocess:
  filter_classes: true
  keep_classes: [6]
```

### 5. Version Control Your Configs

```bash
# Track configs in git
git add configs/
git commit -m "Add production config"
```

---

## 🐛 Troubleshooting

### Issue: Missing Required Parameter

**Error:**

```
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: input_dir
```

**Solution:**

```bash
# Always provide required parameters
ign-lidar-hd process input_dir=data/ output_dir=output/
```

### Issue: Config Override Not Working

**Problem:** Override doesn't take effect

**Solution:**

Ensure `_self_` is in defaults list:

```yaml
defaults:
  - preset: balanced
  - _self_ # Must be last!
```

### Issue: Type Validation Error

**Error:**

```
ValidationError: Expected int, got str for 'num_workers'
```

**Solution:**

```bash
# Use correct type
ign-lidar-hd process num_workers=8          # Correct (int)
ign-lidar-hd process num_workers="8"        # Wrong (string)
```

### Issue: Config File Not Found

**Error:**

```
ConfigNotFound: Cannot find config 'my_config'
```

**Solution:**

```bash
# Ensure file is in configs/ directory
ls configs/my_config.yaml

# Or specify full path
ign-lidar-hd process \
  --config-path /full/path/to/configs \
  --config-name my_config
```

---

## 📚 Next Steps

- **[Hydra CLI Guide](/guides/hydra-cli)** - Learn the CLI
- **[Quick Start](/guides/quick-start)** - Get started quickly
- **[Complete Workflow](/guides/complete-workflow)** - End-to-end examples
- **[API Reference](/api/configuration)** - Configuration API

---

## 🔗 External Resources

- **Hydra Documentation**: [https://hydra.cc/](https://hydra.cc/)
- **OmegaConf**: [https://omegaconf.readthedocs.io/](https://omegaconf.readthedocs.io/)
- **Configuration Patterns**: [https://hydra.cc/docs/patterns/overview/](https://hydra.cc/docs/patterns/overview/)

---

**Master the configuration system to build reproducible, flexible workflows!** 🚀
