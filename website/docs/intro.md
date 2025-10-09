---
slug: /
sidebar_position: 1
title: IGN LiDAR HD Processing Library
---

# IGN LiDAR HD Processing Library

**Version 2.0.2** | Python 3.8+ | MIT License

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

:::tip Major Update: v2.0 Architecture Overhaul!
Complete redesign with **modular architecture**, **Hydra CLI**, and **unified pipeline**! Existing users, see the [Migration Guide](/guides/migration-v1-to-v2) to upgrade from v1.x.
:::

## 📺 Video Demo

<div align="center">
  <a href="https://www.youtube.com/watch?v=ksBWEhkVqQI" target="_blank">
    <img src="https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/v1.6.3/website/static/img/aerial.png?raw=true" alt="IGN LiDAR HD Processing Demo" width="800" />
  </a>
  <p><em>Learn how to process LiDAR data for machine learning applications</em></p>
</div>

---

## 🎉 Latest Release: v2.0.2

### 🚀 Complete Architecture Overhaul

Version 2.0 represents a **major redesign** of the entire library with significant improvements:

**New in v2.0.2:**

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
- **📦 Pipeline ready**: YAML config, smart caching, resumability
- **🔧 Flexible**: CLI tools + Python API

:::

### Quick Start

Install the library:

```bash
pip install ign-lidar-hd
```

Process your first tile:

```bash
ign-lidar-hd enrich \
  --input-dir data/raw_tiles \
  --output data/enriched \
  --auto-params \
  --preprocess \
  --add-rgb \
  --add-infrared
```

With GPU acceleration:

```bash
# Install GPU support (one-time setup)
./install_cuml.sh

# Process with GPU
ign-lidar-hd enrich \
  --input-dir data/raw_tiles \
  --output data/enriched \
  --auto-params \
  --preprocess \
  --use-gpu \
  --add-rgb \
  --add-infrared
```

📖 Continue to [Installation](/installation/quick-start) for detailed setup instructions.

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

- [GPU Acceleration](/guides/gpu-acceleration) - Performance optimization
- [Basic Usage](/guides/basic-usage) - Common workflows
- [Advanced Usage](/guides/advanced-usage) - Power user features

🎨 **Features**

- [RGB Augmentation](/features/rgb-augmentation) - Add true color
- [Infrared Augmentation](/features/infrared-augmentation) - NIR and NDVI
- [Auto Parameters](/features/auto-params) - Automatic optimization
- [LoD3 Classification](/features/lod3-classification) - Building detection

🔧 **API Reference**

- [CLI Commands](/api/cli) - Command-line interface
- [Python API](/api/features) - Programmatic usage
- [Configuration](/api/configuration) - YAML pipelines

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
