---
slug: /
sidebar_position: 1
title: IGN LiDAR HD Processing Library
---

# IGN LiDAR HD Processing Library

**Version 2.4.1** | Python 3.8+ | MIT License

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform French IGN LiDAR HD point clouds into ML-ready datasets for building classification. Features GPU acceleration, rich geometric features, RGB/NIR augmentation, intelligent skip system, and memory-optimized configurations for all system specs.

---

## 🚀 Quick Start

```bash
# Install
pip install ign-lidar-hd

# Download sample data
ign-lidar-hd download --bbox 2.3,48.8,2.4,48.9 --output data/ --max-tiles 5

# Process with default settings
ign-lidar-hd process input_dir=data/ output_dir=patches/
```

For GPU acceleration (12-20x speedup), see the [GPU Setup Guide](/installation/gpu-setup).

---

## ✨ Key Features

- **🗺️ IGN Integration** - Direct download from IGN WFS service
- **🎨 Multi-modal Data** - Geometry + RGB + Infrared (NDVI-ready)
- **🏗️ Building Classification** - LOD2/LOD3 schemas (15-30+ classes)
- **📊 Rich Features** - 28+ geometric features (normals, curvature, planarity, etc.)
- **🚀 GPU Acceleration** - 12-20x speedup with RAPIDS cuML
- **⚡ Intelligent Skip** - ~1800x faster on re-runs, automatic recovery
- **⚙️ YAML Configuration** - Reproducible workflows with example configs

---

## 📋 Processing Modes

Choose the right mode for your workflow:

### Patches Only (Default)

ML-ready patches for training:

```bash
ign-lidar-hd process input_dir=data/ output_dir=patches/
```

### Both Patches & Enriched LAZ

Training + GIS visualization:

```bash
ign-lidar-hd process input_dir=data/ output_dir=both/ output.processing_mode=both
```

### Enriched LAZ Only

Fast GIS workflow:

```bash
ign-lidar-hd process input_dir=data/ output_dir=enriched/ output.processing_mode=enriched_only
```

:::tip
For ML applications, patches provide cleaner geometric features than LAZ tiles (no boundary artifacts).
:::

---

## 📁 Configuration Examples

Production-ready configs in the `examples/` directory:

```bash
# GPU-accelerated processing
ign-lidar-hd process --config-file examples/config_gpu_processing.yaml \
  input_dir=data/raw output_dir=data/enriched

# ML training with augmentation
ign-lidar-hd process --config-file examples/config_training_dataset.yaml \
  input_dir=data/raw output_dir=data/patches

# Preview configuration
ign-lidar-hd process --config-file examples/config_training_dataset.yaml --show-config
```

See [example configs directory](/examples/config-files) for complete workflows.

---

## 📦 Output Formats

- **NPZ** - NumPy format (recommended for ML)
- **HDF5** - Hierarchical data format
- **PyTorch** - `.pt` files for PyTorch training
- **LAZ** - Point cloud visualization (CloudCompare, QGIS)
- **Multi-format** - Combine formats: `npz,laz`, `hdf5,torch`

---

## 🎯 What's New

### v2.4.0 (2025-10-12) - Latest Release

- **🔧 Enhanced Feature Validation**: All geometric features guaranteed within [0, 1] range
- **✅ Production-Ready ML Stability**: Eliminates NaN/Inf values, improved convergence
- **⚡ Zero Overhead**: Less than 1% performance impact from validation
- **🎯 Complete Feature Parity**: Consistent results across CPU/GPU/boundary paths
- **📊 Robustness Guarantees**: Eigenvalue clamping, density normalization

[Read the full release notes →](/release-notes/v2.4.0)

### Previous Releases

**v2.3.0** - Processing modes, YAML configs, intelligent skip system  
**v2.2.1** - Spatial consistency fixes, enhanced augmentation  
**v2.0.0** - Modular architecture, boundary-aware features

---

## � Common Workflows

### Download and Process

```bash
# Download tiles for a region
ign-lidar-hd download --bbox 2.3,48.8,2.4,48.9 --output data/ --max-tiles 10

# Process with GPU acceleration
ign-lidar-hd process input_dir=data/ output_dir=patches/ processor.use_gpu=true
```

### LOD3 with RGB and NDVI

```bash
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=patches/ \
  processor.lod_level=LOD3 \
  features.use_rgb=true \
  features.compute_ndvi=true
```

### Python API

```python
from ign_lidar import LiDARProcessor, IGNLiDARDownloader

# Download tiles
downloader = IGNLiDARDownloader("downloads/")
tiles = downloader.download_by_bbox(bbox=(2.3, 48.8, 2.4, 48.9), max_tiles=5)

# Process with custom settings
processor = LiDARProcessor(lod_level="LOD3", use_gpu=True)
patches = processor.process_directory("data/", "patches/", num_workers=4)

# PyTorch integration
from torch.utils.data import DataLoader
from ign_lidar import LiDARPatchDataset

dataset = LiDARPatchDataset("patches/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 📦 Output Structure

**NPZ File Contents:**

```python
{
    'points': np.ndarray,        # [N, 3] XYZ coordinates
    'normals': np.ndarray,       # [N, 3] surface normals
    'curvature': np.ndarray,     # [N] principal curvature
    'labels': np.ndarray,        # [N] building class labels
    # Optional: RGB, infrared, geometric features
}
```

**Directory Structure:**

```txt
output_dir/
├── tile1_patch_0001.npz
├── tile1_patch_0002.npz
├── tile1_enriched.laz          # if processing_mode="both"
└── metadata.json
```

---

## � Learn More

### Getting Started

- [Installation Guide](/installation/quick-start)
- [GPU Setup](/installation/gpu-setup)
- [Quick Start Tutorial](/guides/quick-start)

### Advanced Features

- [Boundary-Aware Processing](/features/boundary-features)
- [RGB & NIR Augmentation](/features/rgb-augmentation)
- [GPU Acceleration](/guides/gpu-acceleration)
- [Multi-Architecture Support](/features/multi-arch-datasets)

### Reference

- [CLI Commands](/api/cli)
- [Python API](/api/features)
- [Configuration Reference](/api/configuration)

---

## 🤝 Support

- 🐛 [Report Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💡 [Feature Requests](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions)
- 📖 [Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET)

---

## 📄 License

MIT License - See [LICENSE](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/LICENSE) for details.
