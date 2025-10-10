---
slug: /
sidebar_position: 1
title: IGN LiDAR HD Processing Library
---

# IGN LiDAR HD Processing Library

**Version 2.2.0** | Python 3.8+ | MIT License

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

:::tip Major Update: v2.0 Architecture Overhaul!
Complete redesign with **modular architecture**, **Hydra configuration system**, and **unified pipeline**! Existing users, see the [Migration Guide](/guides/migration-v1-to-v2) to upgrade from v1.x.
:::

## 🎉 Latest Release: v2.2.0

### 🎨 Multi-Format Output Support

Version 2.2.0 introduces powerful multi-format output capabilities and fixes critical HDF5 issues:

**New in v2.2.0:**

- � **Multi-Format Output**: Save patches in multiple formats simultaneously (`hdf5,laz`, `npz,torch`, etc.)
- 🗂️ **Complete Format Support**: NPZ, HDF5 (fixed!), PyTorch (.pt), and LAZ patches
- 🔄 **HDF5 to LAZ Converter**: New tool to convert HDF5 patches to LAZ for visualization
- 🎯 **Hybrid Architecture Formatter**: Single-file format for ensemble/hybrid models
- 🐛 **HDF5 Bug Fix**: Critical fix - HDF5 format now properly generates files
- 🔍 **Format Validation**: Automatic validation with clear error messages

**Previous Releases:**

- **v2.1.2** - Documentation updates
- **v2.1.1** - Bug fixes for planarity and boundary features

## 📺 Video Demo

<div align="center">
  <a href="https://www.youtube.com/watch?v=ksBWEhkVqQI" target="_blank">
    <img src="https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/v1.6.3/website/static/img/aerial.png?raw=true" alt="IGN LiDAR HD Processing Demo" width="800" />
  </a>
  <p><em>Learn how to process LiDAR data for machine learning applications</em></p>
</div>

---

## 🔥 What's New in v2.2.0

### Multi-Format Output

Save your patches in multiple formats at once:

```yaml
output:
  format: hdf5,laz # Both HDF5 and LAZ simultaneously
```

Supported formats:

- **NPZ** - NumPy compressed (default, fast)
- **HDF5** - Hierarchical data with gzip compression (now working!)
- **PyTorch** - Direct `.pt` tensor files (requires PyTorch)
- **LAZ** - Point cloud format for visualization (CloudCompare, QGIS, etc.)

### New Tools

- **HDF5 to LAZ Converter**: `scripts/convert_hdf5_to_laz.py` - Convert patches for visualization
- **Hybrid Formatter**: Comprehensive single-file format for ensemble models

See the [v2.2.0 Release Notes](/release-notes/v2.2.0) for complete details.

### 🚀 v2.0 Architecture Overhaul

Version 2.0 represented a **major redesign** of the entire library:

**Key Features in v2.0.2:**

- 🔧 **Enhanced Stability**: Improved error handling and memory management
- 🐛 **Bug Fixes**: Resolved edge cases in boundary-aware processing and tile stitching
- ⚡ **Performance**: Optimized processing pipeline and reduced memory footprint

**Highlights from v2.0.1:**

- 🎯 **Enriched LAZ Only Mode**: Generate enriched LAZ files without patches for visualization workflows
- 🔧 **Automatic Corruption Recovery**: Detects and recovers from corrupted LAZ files automatically

**Major Features in v2.0:**

- 🏗️ **Modular Architecture**: Clean separation into `core`, `features`, `preprocessing`, `io`, and `config` modules
- ⚡ **Hydra CLI**: Modern configuration-based CLI with hierarchical configs and presets
- 🔄 **Unified Pipeline**: Single-step RAW→Patches workflow (no more multi-step processing!)
- 🌐 **Boundary-Aware Features**: Cross-tile processing eliminates edge artifacts
- 🧩 **Tile Stitching**: Multi-tile dataset workflows with automatic neighbor detection
- 🤖 **Multi-Architecture Support**: PointNet++, Octree, Transformer, and Sparse Convolutional networks

:::tip Migrating from v1.x?

The v2.0 architecture requires some changes to your workflows. See the [**Migration Guide**](/guides/migration-v1-to-v2) for:

- Command migration (legacy CLI still works!)
- Import path updates
- New Hydra CLI usage
- Configuration system guide

:::

:::info Legacy CLI Still Supported

Don't worry! The old CLI (`ign-lidar-hd enrich`, etc.) still works for backward compatibility. Try the new Hydra CLI when you're ready:

```bash
# New Hydra CLI - powerful and flexible
ign-lidar-hd process input_dir=data/ output_dir=output/ preset=balanced
```

:::

**Verified Performance:**

- ✅ CPU: 90k-110k points/sec (50k point test)
- ✅ GPU: 100% utilization, 40% VRAM usage
- ✅ Complete pipeline: 17M points in 3-4 minutes

📖 [Optimization Details](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/VECTORIZED_OPTIMIZATION.md) | [GPU Guide](/guides/gpu-acceleration)

---

## Previous Updates

### v1.7.4 - GPU Acceleration

- 🚀 **RAPIDS cuML Support**: 12-20x speedup with full GPU acceleration
- ⚡ **Hybrid GPU Mode**: 6-8x speedup with CuPy (no cuML required)
- 🔧 **Three Performance Tiers**: CPU (60 min), Hybrid (7-10 min), Full GPU (3-5 min)
- 📚 **Enhanced Documentation**: Complete GPU setup guides in English and French

### v1.7.3 - Infrared Augmentation

- 🌿 **NIR Values**: Near-Infrared from IGN IRC orthophotos
- 📊 **NDVI-Ready**: Enables vegetation index calculation
- 🎨 **Multi-Modal**: Geometry + RGB + NIR for ML
- 💾 **Smart Caching**: Efficient disk/GPU caching

### v1.7.1 - Auto-Parameter Analysis

- 🤖 **Automatic Tile Analysis**: Determines optimal processing parameters
- 🎯 **Adaptive Processing**: Custom settings per tile based on characteristics
- ⚡ **Zero Manual Tuning**: Eliminates guesswork for urban/rural/mixed tiles

---

## Getting Started

Welcome to the **IGN LiDAR HD Processing Library** documentation!

Transform French LiDAR data into machine learning-ready datasets for building classification with this comprehensive Python toolkit. 🏗️

:::tip Why use this library?

- **🎯 Specialized for French LiDAR**: Optimized for IGN's LiDAR HD format
- **⚡ Production-ready**: Battle-tested with 50+ tiles
- **🚀 GPU-accelerated**: Optional CUDA support for 12-20x faster processing
- **🌈 Rich feature extraction**: 28+ geometric and color features
- **🌿 Multi-modal**: Geometry + RGB + Infrared support
- **📦 Hydra Configuration**: Hierarchical YAML configs with powerful overrides
- **🔧 Flexible**: Modern CLI + Python API

:::

### Quick Start

Install the library:

```bash
pip install ign-lidar-hd
```

#### Modern CLI (v2.0+)

The v2.0+ CLI combines Hydra's configuration power with intuitive Click commands:

```bash
# Download tiles from IGN servers
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 5000 \
  data/raw_tiles

# Process tiles to create ML-ready patches
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/patches

# Verify dataset quality
ign-lidar-hd verify data/patches

# Show configuration info
ign-lidar-hd info

# Convert patches for QGIS visualization
ign-lidar-hd batch-convert \
  data/patches \
  --output data/qgis \
  --format qgis
```

#### Advanced Processing Examples

```bash
# LOD3 training with GPU acceleration
ign-lidar-hd process \
  experiment=config_lod3_training \
  input_dir=data/raw_tiles \
  output_dir=data/patches \
  processor.use_gpu=true

# Custom configuration with overrides
ign-lidar-hd process \
  processor=gpu \
  features=full \
  input_dir=data/raw_tiles \
  output_dir=data/patches \
  processor.num_points=32768 \
  features.k_neighbors=30 \
  features.use_rgb=true \
  features.compute_ndvi=true \
  stitching.enabled=true

# Generate only enriched LAZ (no patches)
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/enriched \
  output=enriched_only
```

#### Legacy CLI (Backward Compatible)

```bash
# Legacy CLI still works for v1.x compatibility
ign-lidar-hd enrich \
  --input-dir data/raw_tiles \
  --output data/enriched \
  --auto-params \
  --preprocess \
  --add-rgb
```

📖 Continue to [Installation](/installation/quick-start) for detailed setup instructions and [Hydra CLI Guide](/guides/hydra-cli) for advanced usage.

---

## 🎯 Hydra Configuration System

The v2.0 architecture introduces a powerful Hydra-based configuration system for flexible, reproducible workflows.

### Key Features

- **📁 Hierarchical Configs**: Compose complex configurations from simple building blocks
- **🔧 Config Groups**: Organized presets for processor, features, experiments
- **⚙️ Command-line Overrides**: Change any parameter without editing files
- **✅ Type Safety**: Configuration validation at runtime
- **🔄 Experiment Presets**: Pre-configured workflows for common tasks

### Configuration Structure

```
ign_lidar/configs/
├── config.yaml                     # Root configuration
├── processor/                      # Processing backend configs
│   ├── default.yaml               # CPU baseline (LOD2, 16K points)
│   ├── gpu.yaml                   # GPU accelerated
│   ├── cpu_fast.yaml              # Quick processing
│   └── memory_constrained.yaml    # Low memory systems
├── features/                       # Feature computation configs
│   ├── full.yaml                  # All features enabled
│   ├── minimal.yaml               # Basic geometric features
│   ├── pointnet.yaml              # PointNet++ optimized
│   ├── buildings.yaml             # Building-specific features
│   └── vegetation.yaml            # Vegetation analysis
├── experiment/                     # Complete workflow presets
│   ├── config_lod3_training.yaml  # LOD3 hybrid model training
│   ├── pointnet_training.yaml     # PointNet++ training
│   ├── buildings_lod2.yaml        # LOD2 building detection
│   ├── buildings_lod3.yaml        # LOD3 building detection
│   └── boundary_aware_autodownload.yaml  # Auto-download neighbors
├── preprocess/                     # Preprocessing configs
│   └── default.yaml               # Outlier removal, voxelization
├── stitching/                      # Tile stitching configs
│   └── enhanced.yaml              # Boundary-aware processing
└── output/                         # Output format configs
    └── default.yaml               # NPZ format with metadata
```

### Example Configurations

#### Basic Processing

```bash
# Minimal command - uses all defaults from config.yaml
ign-lidar-hd \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output
```

**Uses:** `config.yaml` defaults (CPU, LOD2, 16K points, random sampling)

#### GPU Processing

```bash
# Override processor group to use GPU
ign-lidar-hd \
  processor=gpu \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output
```

**Config:** `processor/gpu.yaml`

```yaml
use_gpu: true
num_workers: 8
pin_memory: true
prefetch_factor: 4
```

#### LOD3 Training Dataset

```bash
# Use complete experiment preset
ign-lidar-hd \
  experiment=config_lod3_training \
  input_dir=/path/to/tiles \
  output_dir=/path/to/patches
```

**Config:** `experiment/config_lod3_training.yaml`

```yaml
processor:
  lod_level: LOD3
  architecture: hybrid
  num_points: 32768
  use_gpu: true
  augment: true

features:
  mode: full
  k_neighbors: 30
  use_rgb: true
  compute_ndvi: true
  sampling_method: fps # Farthest Point Sampling
  normalize_xyz: true

stitching:
  enabled: true
  buffer_size: 20.0
  auto_download_neighbors: true
```

#### Custom Overrides

```bash
# Mix preset with custom overrides
ign-lidar-hd \
  experiment=pointnet_training \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output \
  processor.num_points=65536 \
  features.k_neighbors=50 \
  processor.augment=false \
  output.format=torch
```

### Available Experiment Presets

| Preset                          | Use Case                               | Key Settings                                   |
| ------------------------------- | -------------------------------------- | ---------------------------------------------- |
| **config_lod3_training**        | LOD3 hybrid model training             | GPU, 32K points, FPS, full features, NDVI      |
| **pointnet_training**           | PointNet++ training                    | GPU, 16K points, FPS, normalized features      |
| **buildings_lod2**              | LOD2 building detection                | CPU, building features, standard preprocessing |
| **buildings_lod3**              | LOD3 building detection                | GPU, hybrid arch, enhanced features            |
| **boundary_aware_autodownload** | Seamless tile boundaries               | Auto-download neighbors, 20m buffer            |
| **fast**                        | Quick prototyping                      | Minimal features, no preprocessing             |
| **semantic_sota**               | State-of-the-art semantic segmentation | Full features, aggressive preprocessing        |

📖 See [Configuration System Guide](/guides/configuration-system) for detailed documentation.

---

## 💡 Common Usage Examples

### 1. Quick Prototyping (CPU)

```bash
# Fast processing for testing
ign-lidar-hd \
  processor=cpu_fast \
  features=minimal \
  input_dir=data/test_tile \
  output_dir=data/test_output
```

**Best for:** Testing pipeline, debugging, small datasets

### 2. Production Training Dataset (GPU)

```bash
# Full quality with GPU acceleration
ign-lidar-hd \
  experiment=config_lod3_training \
  input_dir=data/urban_tiles \
  output_dir=data/training_patches \
  processor.num_workers=8 \
  stitching.buffer_size=30.0
```

**Best for:** Training deep learning models, production datasets

### 3. Multi-tile Boundary-Aware Processing

```bash
# Automatic neighbor detection and download
ign-lidar-hd \
  experiment=boundary_aware_autodownload \
  input_dir=data/tiles \
  output_dir=data/seamless_patches
```

**Best for:** Large-scale processing, eliminating boundary artifacts

### 4. PointNet++ Training Dataset

```bash
# Optimized for PointNet++ architecture
ign-lidar-hd \
  experiment=pointnet_training \
  input_dir=data/raw \
  output_dir=data/pointnet_patches \
  processor.num_points=16384 \
  features.sampling_method=fps \
  output.format=torch
```

**Best for:** PointNet++ models, PyTorch training

### 5. Memory-Constrained System

```bash
# Low memory usage
ign-lidar-hd \
  processor=memory_constrained \
  features=minimal \
  input_dir=data/tiles \
  output_dir=data/patches \
  processor.num_workers=2 \
  processor.batch_size=1
```

**Best for:** Systems with limited RAM, laptop processing

### 6. Custom Configuration File

Create `my_config.yaml`:

```yaml
# @package _global_
defaults:
  - override /processor: gpu
  - override /features: full

processor:
  lod_level: LOD3
  num_points: 32768
  augment: true
  num_augmentations: 5

features:
  k_neighbors: 40
  use_rgb: true
  compute_ndvi: true

input_dir: /mnt/data/lidar_tiles
output_dir: /mnt/data/training_dataset
```

Run with:

```bash
ign-lidar-hd \
  --config-path=/path/to/configs \
  --config-name=my_config
```

### 7. Parameter Sweep (Hyperparameter Search)

```bash
# Test multiple point counts
ign-lidar-hd \
  --multirun \
  processor=gpu \
  processor.num_points=4096,8192,16384,32768 \
  input_dir=data/tiles \
  output_dir=data/multirun
```

**Results in:** `data/multirun/0/`, `data/multirun/1/`, etc.

### 8. Enriched LAZ Only (No Patches)

```bash
# Generate enriched LAZ for visualization
ign-lidar-hd \
  processor=default \
  features=full \
  input_dir=data/tiles \
  output_dir=data/enriched \
  output.save_enriched_laz=true \
  output.only_enriched_laz=true
```

**Output:** LAZ files with computed features as extra dimensions

---

## Features

### Core Capabilities

- **🗺️ IGN Data Integration**: Direct download from IGN WFS service
- **🎨 RGB Augmentation**: Add true color from IGN aerial photos
- **🌿 Infrared Augmentation**: Add NIR for vegetation analysis (NDVI-ready)
- **📊 Rich Features**: 28+ geometric features (normals, curvature, planarity, etc.)
- **🏠 Building Classification**: LoD0/LoD1/LoD2/LoD3 classification
- **🚀 GPU Acceleration**: 12-20x speedup with RAPIDS cuML
- **🔧 Artifact Mitigation**: Statistical + radius outlier removal
- **🤖 Auto-Parameters**: Automatic tile analysis and optimization

### Processing Modes

| Mode           | Speed                        | Requirements            | Use Case                    |
| -------------- | ---------------------------- | ----------------------- | --------------------------- |
| **CPU**        | Baseline (60 min/tile)       | Python 3.8+             | Development, small datasets |
| **Hybrid GPU** | 6-8x faster (7-10 min/tile)  | NVIDIA GPU, CuPy        | Good balance                |
| **Full GPU**   | 12-20x faster (3-5 min/tile) | NVIDIA GPU, RAPIDS cuML | Production, large datasets  |

### Output Formats

- **LAZ 1.4**: Extended attributes (28+ features) - **Recommended**
- **LAZ 1.2**: CloudCompare compatible (RGB + basic features)
- **QGIS Layers**: Separate styled layers for visualization
- **Statistics**: JSON metrics for quality tracking

---

## Documentation Structure

📚 **Installation**

- [Quick Start](/installation/quick-start) - Get up and running in 5 minutes
- [GPU Setup](/installation/gpu-setup) - RAPIDS cuML configuration

⚡ **Guides**

- [Hydra CLI Guide](/guides/hydra-cli) - Modern configuration-based CLI
- [Configuration System](/guides/configuration-system) - Deep dive into Hydra configs
- [Unified Pipeline](/guides/unified-pipeline) - End-to-end workflows
- [Migration v1 to v2](/guides/migration-v1-to-v2) - Upgrade guide
- [GPU Acceleration](/guides/gpu-acceleration) - Performance optimization

🎨 **Features**

- [Boundary-Aware Processing](/features/boundary-features) - Seamless tile stitching
- [RGB Augmentation](/features/rgb-augmentation) - Add true color
- [Infrared Augmentation](/features/infrared-augmentation) - NIR and NDVI
- [LoD3 Classification](/features/lod3-classification) - Building detection
- [Multi-Architecture Support](/features/multi-arch-datasets) - PointNet++, Octree, Transformer

🏗️ **Architecture**

- [System Architecture](/architecture) - Modular design overview
- [Core Components](/api/core) - Processor, features, preprocessing
- [Config Schema](/api/config) - Configuration data structures

🔧 **API Reference**

- [CLI Commands](/api/cli) - Command-line interface
- [Python API](/api/features) - Programmatic usage
- [Configuration](/api/configuration) - YAML config reference

---

## Performance

With v1.7.5 vectorization optimization:

| Points | CPU  | GPU (cuML) | Speedup      |
| ------ | ---- | ---------- | ------------ |
| 1M     | 10s  | &lt;1s     | 15-20x       |
| 5M     | 50s  | 3s         | 100-150x     |
| 17M    | 180s | 30s        | **100-200x** |

Real-world example (17M point tile):

- Preprocessing: ~2 minutes
- Features: ~30 seconds (vectorized!)
- RGB augmentation: ~30 seconds
- Infrared augmentation: ~30 seconds
- **Total: 3-4 minutes** (was hours before optimization!)

---

## Community

- 🐛 [Report Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💡 [Feature Requests](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 📖 [Contribute](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/CONTRIBUTING.md)

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/LICENSE) file for details.

---

## Next Steps

Ready to dive in? Start with the [Quick Start Guide](/installation/quick-start) to install the library and process your first tile!

For GPU acceleration (recommended for production), check out the [GPU Setup Guide](/installation/gpu-setup).
