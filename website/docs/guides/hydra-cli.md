---
sidebar_position: 5
title: Hydra CLI Guide
---

# Hydra CLI Guide

Complete guide to the modern Hydra-based command-line interface in v2.0+.

---

## 🎯 What is Hydra CLI?

The **Hydra CLI** is a modern, configuration-based command-line interface built on [Hydra](https://hydra.cc), Facebook Research's framework for elegant configuration management.

### Why Hydra?

- ✅ **Hierarchical Configs**: Compose complex configurations from simple building blocks
- ✅ **Type Safety**: Configuration validation at runtime
- ✅ **Presets**: Pre-configured workflows for common use cases
- ✅ **Overrides**: Change any parameter from command line
- ✅ **Reproducibility**: Complete configuration tracking in output
- ✅ **Experiment Management**: Built-in support for parameter sweeps

### vs. Legacy CLI

| Feature           | Legacy CLI           | Hydra CLI                       |
| ----------------- | -------------------- | ------------------------------- |
| **Syntax**        | `--arg value`        | `arg=value`                     |
| **Configuration** | Command-line only    | Files + overrides               |
| **Presets**       | ❌ None              | ✅ 4 presets                    |
| **Composition**   | ❌ No                | ✅ Yes (hierarchical)           |
| **Validation**    | Basic                | Type-safe                       |
| **Tracking**      | Manual               | Automatic                       |
| **Status**        | Supported for legacy | ⭐ Recommended for new projects |

---

## 🚀 Quick Start

### Basic Usage

```bash
# Simplest command - uses all defaults
ign-lidar-hd process input_dir=data/ output_dir=output/
```

### With Preset

```bash
# Use a preset for common workflows
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  preset=balanced
```

### With Overrides

```bash
# Override specific parameters
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  preset=balanced \
  processor=gpu \
  features.use_rgb=true \
  num_workers=8
```

---

## 📋 Command Structure

### General Syntax

```bash
ign-lidar-hd <command> [key=value] [key.nested=value] ...
```

### Available Commands

| Command    | Purpose                         | Example                                |
| ---------- | ------------------------------- | -------------------------------------- |
| `process`  | Main processing pipeline        | `ign-lidar-hd process input_dir=data/` |
| `download` | Download IGN tiles              | `ign-lidar-hd download bbox="..."`     |
| `verify`   | Verify data integrity           | `ign-lidar-hd verify input_dir=data/`  |
| `enrich`   | Legacy enrichment (v1.x compat) | `ign-lidar-hd enrich --input-dir ...`  |
| `patch`    | Legacy patching (v1.x compat)   | `ign-lidar-hd patch --input-dir ...`   |

---

## 🎨 Presets

Presets are pre-configured workflows optimized for different use cases.

### Available Presets

#### 1. **fast** ⚡ - Quick Processing

**Use Case:** Rapid prototyping, testing, quick results

```bash
ign-lidar-hd process input_dir=data/ preset=fast
```

**Configuration:**

- Features: Basic geometric features only
- Preprocessing: Minimal (outlier removal only)
- Quality: Lower (fewer neighbors)
- Speed: **5-10 minutes per tile**

**Best For:**

- Testing pipelines
- Quick data exploration
- Development iterations

---

#### 2. **balanced** ⚖️ - Standard Quality (⭐ Recommended)

**Use Case:** Production workflows, general ML training

```bash
ign-lidar-hd process input_dir=data/ preset=balanced
```

**Configuration:**

- Features: Standard geometric + RGB (if available)
- Preprocessing: Standard (outliers + ground normalization)
- Quality: Medium (optimal neighbors)
- Speed: **15-20 minutes per tile**

**Best For:**

- Production ML datasets
- Standard research projects
- Most use cases

---

#### 3. **quality** 💎 - High Quality

**Use Case:** High-quality datasets, research publications

```bash
ign-lidar-hd process input_dir=data/ preset=quality
```

**Configuration:**

- Features: Full geometric + RGB + NDVI + infrared
- Preprocessing: Aggressive (all cleaning steps)
- Quality: High (more neighbors)
- Speed: **30-45 minutes per tile**

**Best For:**

- Publication-quality datasets
- Detailed analysis
- Critical applications

---

#### 4. **ultra** 🚀 - Maximum Quality

**Use Case:** Seamless large-area processing, research

```bash
ign-lidar-hd process input_dir=data/ preset=ultra
```

**Configuration:**

- Features: All features + boundary-aware processing
- Preprocessing: Aggressive + tile buffering
- Quality: Maximum (cross-tile features)
- Speed: **60+ minutes per tile**

**Best For:**

- Seamless multi-tile datasets
- Eliminating edge artifacts
- High-precision requirements

---

## ⚙️ Configuration Parameters

### Core Parameters

#### Input/Output

```bash
# Required
input_dir=path/to/input/        # Directory with RAW LAZ files
output_dir=path/to/output/      # Output directory

# Optional
output=patches                   # Output mode: patches, both, enriched_only
```

#### Preset Selection

```bash
preset=balanced                  # Choose workflow preset
# Options: fast, balanced, quality, ultra
```

#### Processor Selection

```bash
processor=cpu                    # Processing backend
# Options: cpu, gpu
```

### Feature Parameters

```bash
# Feature preset
features=standard                # Feature set
# Options: basic, standard, full

# RGB/Color
features.use_rgb=true            # Use RGB colors (if available)
features.compute_ndvi=true       # Compute NDVI (vegetation index)
features.use_infrared=true       # Use infrared intensity

# Boundary-aware
features.boundary_aware=false    # Enable cross-tile features
features.boundary_buffer=10.0    # Buffer distance (meters)

# Quality
features.k_neighbors=30          # Number of neighbors for features
features.radius=2.0              # Search radius (meters)

# GPU
features.use_gpu=false           # Use GPU acceleration
features.gpu_chunk_size=1000000  # Points per GPU chunk
```

### Preprocessing Parameters

```bash
# Preprocessing preset
preprocess=standard              # Preprocessing level
# Options: minimal, standard, aggressive

# Outlier removal
preprocess.remove_outliers=true
preprocess.outlier_method=statistical
preprocess.outlier_nb_neighbors=20
preprocess.outlier_std_ratio=2.0

# Ground normalization
preprocess.normalize_ground=true
preprocess.ground_resolution=1.0
preprocess.ground_max_distance=5.0

# Classification filtering
preprocess.filter_classes=true
preprocess.keep_classes=[2,3,4,5,6]  # Ground, vegetation, building
```

### Dataset Parameters

```bash
# Patch generation
dataset.patch_size=50.0          # Patch size (meters)
dataset.points_per_patch=4096    # Target points per patch
dataset.overlap=0.0              # Patch overlap (meters)

# Architecture
dataset.architecture=pointnet++  # Target ML architecture
# Options: pointnet++, octree, transformer, sparse_conv

# Labels
dataset.lod_level=2              # LOD classification level
# Options: 2, 3
```

### Stitching Parameters

```bash
# Tile stitching
stitching=none                   # Stitching mode
# Options: none, features, full

stitching.tile_overlap=0.0       # Tile overlap for stitching
stitching.merge_method=average   # Feature merging method
```

### Performance Parameters

```bash
# Parallelization
num_workers=4                    # Number of parallel workers
batch_size=1                     # Tiles per batch

# Memory
max_memory_gb=8                  # Maximum RAM usage
chunk_size=1000000              # Points per processing chunk

# GPU
gpu_memory_fraction=0.9         # Fraction of GPU memory to use
```

### Logging Parameters

```bash
# Verbosity
verbose=false                    # Verbose output
log_level=INFO                   # Logging level
# Options: DEBUG, INFO, WARNING, ERROR

# Progress
show_progress=true               # Show progress bars
```

---

## 🔧 Parameter Overrides

### Syntax

```bash
# Top-level parameter
ign-lidar-hd process param=value

# Nested parameter (use dot notation)
ign-lidar-hd process param.nested=value

# Multiple overrides
ign-lidar-hd process param1=value1 param2=value2 nested.param=value3
```

### Examples

#### Override Preset Values

```bash
# Start with balanced preset, but use GPU
ign-lidar-hd process \
  preset=balanced \
  processor=gpu
```

#### Override Multiple Parameters

```bash
# Customize feature computation
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  preset=quality \
  features.use_rgb=true \
  features.compute_ndvi=true \
  features.k_neighbors=50
```

#### Override Preprocessing

```bash
# More aggressive outlier removal
ign-lidar-hd process \
  preset=balanced \
  preprocess.outlier_std_ratio=1.5 \
  preprocess.outlier_nb_neighbors=30
```

---

## 📁 Configuration Files

For complex configurations, use YAML files instead of long command lines.

### Basic Config File

**config.yaml:**

```yaml
# Defaults
defaults:
  - preset: balanced
  - processor: gpu
  - _self_

# Paths
input_dir: "/data/urban/"
output_dir: "/output/urban/"

# Features
features:
  use_rgb: true
  compute_ndvi: true
  k_neighbors: 40

# Performance
num_workers: 8
verbose: true
```

**Usage:**

```bash
ign-lidar-hd process --config-name config
```

### Advanced Config with Overrides

**research_config.yaml:**

```yaml
defaults:
  - preset: quality
  - processor: gpu
  - features: full
  - preprocess: aggressive
  - _self_

input_dir: "/research/lidar/raw/"
output_dir: "/research/lidar/processed/"

# Boundary-aware processing
features:
  boundary_aware: true
  boundary_buffer: 15.0
  use_rgb: true
  compute_ndvi: true
  use_infrared: true

# Multi-tile stitching
stitching: full

# High performance
num_workers: 12
gpu_memory_fraction: 0.95

# Output both enriched LAZ and patches
output: both
```

**Usage:**

```bash
# Use config as-is
ign-lidar-hd process --config-name research_config

# Or override specific values
ign-lidar-hd process --config-name research_config num_workers=16
```

### Config Directory Structure

```
project/
├── configs/
│   ├── config.yaml              # Default config
│   ├── fast_test.yaml           # Fast testing config
│   ├── production.yaml          # Production config
│   └── research.yaml            # Research config
└── run.py                       # Your processing script
```

---

## 🎭 Common Workflows

### 1. Quick Test Run

```bash
# Fast processing for testing
ign-lidar-hd process \
  input_dir=test_data/ \
  output_dir=test_output/ \
  preset=fast \
  num_workers=2
```

### 2. Production ML Dataset

```bash
# Balanced quality for training
ign-lidar-hd process \
  input_dir=raw_tiles/ \
  output_dir=training_data/ \
  preset=balanced \
  processor=gpu \
  features.use_rgb=true \
  num_workers=8
```

### 3. Visualization Workflow

```bash
# Generate enriched LAZ for CloudCompare
ign-lidar-hd process \
  input_dir=raw/ \
  output_dir=viz/ \
  output=enriched_only \
  preset=quality \
  features.use_rgb=true \
  features.compute_ndvi=true
```

### 4. Seamless Multi-Tile Dataset

```bash
# Boundary-aware processing with stitching
ign-lidar-hd process \
  input_dir=tiles/ \
  output_dir=seamless/ \
  preset=ultra \
  processor=gpu \
  features.boundary_aware=true \
  stitching=full \
  num_workers=6
```

### 5. Architecture-Specific Processing

```bash
# Generate patches for Octree network
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=octree_patches/ \
  preset=balanced \
  dataset.architecture=octree \
  dataset.points_per_patch=8192
```

### 6. Research Quality Dataset

```bash
# Maximum quality with all features
ign-lidar-hd process \
  input_dir=research_area/ \
  output_dir=research_output/ \
  preset=quality \
  processor=gpu \
  output=both \
  features.use_rgb=true \
  features.compute_ndvi=true \
  features.use_infrared=true \
  preprocess=aggressive \
  num_workers=12
```

---

## 🔍 Help & Introspection

### View Available Parameters

```bash
# Show all available parameters
ign-lidar-hd process --help
```

### View Preset Configuration

```bash
# See what a preset configures
ign-lidar-hd process preset=balanced --cfg job
```

### View Final Configuration

```bash
# See resolved configuration (after overrides)
ign-lidar-hd process \
  preset=balanced \
  processor=gpu \
  --cfg job
```

### Validate Configuration

```bash
# Check configuration without running
ign-lidar-hd process \
  input_dir=data/ \
  preset=balanced \
  --cfg job
```

---

## 🎓 Advanced Features

### 1. Multi-Run Experiments

Run multiple configurations automatically:

```bash
# Sweep over multiple parameter values
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  preset=balanced \
  --multirun \
  features.k_neighbors=20,30,40,50
```

Creates separate runs for each value:

- `output/multirun/0/` - k_neighbors=20
- `output/multirun/1/` - k_neighbors=30
- `output/multirun/2/` - k_neighbors=40
- `output/multirun/3/` - k_neighbors=50

### 2. Experiment Tracking

Hydra automatically saves complete configuration:

```
output/
├── .hydra/
│   ├── config.yaml              # Full resolved config
│   ├── hydra.yaml               # Hydra runtime config
│   └── overrides.yaml           # Command-line overrides
├── enriched/
├── patches/
└── metadata.json
```

### 3. Config Composition

**Base config (configs/base.yaml):**

```yaml
input_dir: ??? # Must be provided
output_dir: ???

num_workers: 4
verbose: false
```

**Urban config (configs/urban.yaml):**

```yaml
defaults:
  - base
  - preset: balanced
  - _self_

preprocess:
  filter_classes: true
  keep_classes: [6] # Buildings only
```

**Usage:**

```bash
ign-lidar-hd process \
  --config-name urban \
  input_dir=urban_data/ \
  output_dir=urban_output/
```

### 4. Conditional Configuration

```yaml
# Enable GPU features only if GPU available
features:
  use_gpu: ${oc.env:CUDA_VISIBLE_DEVICES,false}
  gpu_chunk_size: ${oc.select:features.use_gpu,1000000,null}
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "Missing required parameter"

**Error:**

```
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: input_dir
```

**Fix:**

```bash
# Always provide required parameters
ign-lidar-hd process input_dir=data/ output_dir=output/
```

#### Issue: "Config not found"

**Error:**

```
hydra.errors.ConfigNotFound: Cannot find config 'my_config'
```

**Fix:**

```bash
# Ensure config file is in configs/ directory
# Or specify path:
ign-lidar-hd process --config-path /full/path/to/configs --config-name my_config
```

#### Issue: "Invalid value for parameter"

**Error:**

```
ValidationError: Invalid value for 'preset': unknown_preset
```

**Fix:**

```bash
# Use valid preset names: fast, balanced, quality, ultra
ign-lidar-hd process preset=balanced
```

#### Issue: "Override not working"

**Error:** Parameter doesn't change despite override

**Fix:**

```bash
# Ensure _self_ is in defaults to allow overrides
# In config.yaml:
defaults:
  - preset: balanced
  - _self_  # Must be last!
```

---

## 📊 Comparison: Legacy vs Hydra CLI

### Legacy CLI (Still Supported)

```bash
# v1.x style (still works!)
ign-lidar-hd enrich \
  --input-dir data/raw/ \
  --output output/enriched/ \
  --use-rgb \
  --compute-ndvi \
  --use-gpu \
  --num-workers 4

ign-lidar-hd patch \
  --input-dir output/enriched/ \
  --output output/patches/ \
  --patch-size 50
```

### Hydra CLI (v2.0+)

```bash
# Modern single-step workflow
ign-lidar-hd process \
  input_dir=data/raw/ \
  output_dir=output/ \
  preset=balanced \
  processor=gpu \
  features.use_rgb=true \
  features.compute_ndvi=true \
  num_workers=4 \
  dataset.patch_size=50
```

### Advantages of Hydra CLI

| Feature                  | Legacy CLI | Hydra CLI |
| ------------------------ | ---------- | --------- |
| Single-step workflow     | ❌         | ✅        |
| Configuration files      | ❌         | ✅        |
| Presets                  | ❌         | ✅        |
| Hierarchical composition | ❌         | ✅        |
| Type validation          | ❌         | ✅        |
| Experiment tracking      | ❌         | ✅        |
| Multi-run sweeps         | ❌         | ✅        |

---

## 📚 Next Steps

- **[Configuration System Guide](/guides/configuration-system)** - Deep dive into Hydra configs
- **[Complete Workflow](/guides/complete-workflow)** - End-to-end examples
- **[Quick Start](/guides/quick-start)** - Get started quickly
- **[Migration Guide](/guides/migration-v1-to-v2)** - Upgrade from v1.x

---

## 🔗 External Resources

- **Hydra Documentation**: https://hydra.cc/
- **Hydra Tutorials**: https://hydra.cc/docs/tutorials/intro/
- **OmegaConf**: https://omegaconf.readthedocs.io/

---

**Master the Hydra CLI to unlock the full power of v2.0!** 🚀
