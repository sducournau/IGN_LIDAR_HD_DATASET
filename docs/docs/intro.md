---
slug: /
sidebar_position: 1
title: IGN LiDAR HD Processing Library
---

# IGN LiDAR HD Processing Library

**Version 2.3.0** | Python 3.8+ | MIT License

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform French IGN LiDAR HD point clouds into machine learning-ready datasets for building classification. Features GPU acceleration, rich geometric features, RGB/NIR augmentation, and flexible YAML-based configuration.

:::tip New in v2.3.0
**Processing modes** and **YAML configuration files** make workflows clearer and more flexible! Check out the [example configs](#example-configurations) and [processing modes guide](#processing-modes).
:::

---

## 🚀 Quick Start

### Installation

```bash
# Standard installation (CPU)
pip install ign-lidar-hd

# Verify installation
ign-lidar-hd --version
```

For GPU acceleration (6-20x speedup), see the [GPU Setup Guide](/installation/gpu-setup).

### Basic Usage

```bash
# Download sample data
ign-lidar-hd download \
  --bbox 2.3,48.8,2.4,48.9 \
  --output data/ \
  --max-tiles 5

# Process with default settings
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=patches/

# Or use example configuration
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=data/ \
  output_dir=patches/
```

### Python API

```python
from ign_lidar import LiDARProcessor

# Initialize processor
processor = LiDARProcessor(lod_level="LOD2")

# Process single tile
patches = processor.process_tile("data.laz", "output/")

# Batch processing
patches = processor.process_directory("data/", "output/", num_workers=4)
```

---

## 📋 Processing Modes

Version 2.3.0 introduces three clear processing modes:

### Mode 1: Patches Only (Default)

Creates ML-ready patches for training:

```bash
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/patches \
  output.processing_mode=patches_only
```

**Use case:** Machine learning model training

### Mode 2: Both Patches & Enriched LAZ

Creates both ML patches and enriched LAZ files:

```bash
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/both \
  output.processing_mode=both
```

**Use case:** Training + GIS visualization

### Mode 3: Enriched LAZ Only

Creates only enriched LAZ files with features:

```bash
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/enriched \
  output.processing_mode=enriched_only
```

**Use case:** Fast GIS workflow, visualization

:::warning Geometric Feature Artifacts in LAZ Files
When generating enriched LAZ tile files, geometric features (normals, curvature, planarity, etc.) may show artifacts at tile boundaries due to the nature of the source data. These artifacts are inherent to tile-based processing and **do not appear in patch exports**, which provide the best results for machine learning applications. For optimal quality, use `patches_only` or `both` modes.
:::

---

## 📁 Example Configurations

Four production-ready configs are in the `examples/` directory:

### 1. GPU Processing

**File:** `config_gpu_processing.yaml`

GPU-accelerated LAZ enrichment:

```bash
ign-lidar-hd process \
  --config-file examples/config_gpu_processing.yaml \
  input_dir=data/raw \
  output_dir=data/enriched
```

### 2. Training Dataset

**File:** `config_training_dataset.yaml`

ML training with augmentation:

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=data/raw \
  output_dir=data/patches
```

### 3. Quick Enrich

**File:** `config_quick_enrich.yaml`

Fast LAZ feature enrichment:

```bash
ign-lidar-hd process \
  --config-file examples/config_quick_enrich.yaml \
  input_dir=data/raw \
  output_dir=data/enriched
```

### 4. Complete Workflow

**File:** `config_complete.yaml`

Both patches and enriched LAZ:

```bash
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  input_dir=data/raw \
  output_dir=data/both
```

### Preview Configuration

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  --show-config \
  input_dir=data/raw
```

---

## ✨ Key Features

### Core Capabilities

- **🗺️ IGN Integration** - Direct download from IGN WFS service
- **🎨 Multi-modal Data** - Geometry + RGB + Infrared (NDVI-ready)
- **🏗️ Building Classification** - LOD2/LOD3 schemas (15-30+ classes)
- **📊 Rich Features** - 28+ geometric features (normals, curvature, planarity, etc.)
- **🚀 GPU Acceleration** - 12-20x speedup with RAPIDS cuML
- **📦 Multiple Formats** - NPZ, HDF5, PyTorch, LAZ
- **⚙️ YAML Configuration** - Reproducible workflows

### Performance Tiers

| Mode         | Speed                  | Requirements            |
| ------------ | ---------------------- | ----------------------- |
| **CPU**      | Baseline (60 min/tile) | Python 3.8+             |
| **Hybrid**   | 6-8x faster            | NVIDIA GPU, CuPy        |
| **Full GPU** | 12-20x faster          | NVIDIA GPU, RAPIDS cuML |

### Output Formats

- **NPZ** - Default NumPy format (recommended for ML)
- **HDF5** - Hierarchical data format
- **PyTorch** - `.pt` files for PyTorch training
- **LAZ** - Point cloud visualization (CloudCompare, QGIS) - may show boundary artifacts in tile mode
- **Multi-format** - Save in multiple formats: `hdf5,laz`, `npz,torch`

:::tip Best Results for Machine Learning
For machine learning applications, NPZ/HDF5/PyTorch patch formats provide cleaner geometric features than enriched LAZ tiles, as patches eliminate tile boundary artifacts.
:::

---

## 📖 What's New

### v2.3.0 - Processing Modes & Custom Configurations

**Key Improvements:**

- **Explicit Processing Modes**: Clear, intuitive modes replace boolean flags

  - `processing_mode="patches_only"` (default) - ML training
  - `processing_mode="both"` - Patches + enriched LAZ
  - `processing_mode="enriched_only"` - LAZ enrichment only

- **Custom Config Files**: Load complete workflows from YAML

  - Four production-ready examples in `examples/` directory
  - `--config-file` / `-c` option for easy loading
  - `--show-config` to preview merged configuration

- **Smart Precedence**: Package defaults < Custom file < CLI overrides

**Migration Example:**

```python
# Old API (deprecated)
processor = LiDARTileProcessor(
    save_enriched_laz=True,
    only_enriched_laz=True
)

# New API (recommended)
processor = LiDARTileProcessor(
    processing_mode="enriched_only"
)
```

### v2.2.1 - Critical Augmentation Fix

- Fixed spatial consistency bug in augmented patches
- Enhanced pipeline: extract once, augment individually
- ⚠️ Datasets with augmentation before v2.2.1 should be regenerated

### v2.0 - Architecture Overhaul

- Modular architecture with clean separation
- Hydra-based CLI with hierarchical configs
- Unified RAW→Patches workflow
- Boundary-aware features and tile stitching
- Multi-architecture support (PointNet++, Octree, Transformer)

---

## 📚 Documentation Structure

### Getting Started

- [Installation](/installation/quick-start) - Setup in 5 minutes
- [GPU Setup](/installation/gpu-setup) - RAPIDS cuML configuration
- [Quick Start](/guides/quick-start) - First steps

### Guides

- [Hydra CLI Guide](/guides/hydra-cli) - Configuration-based CLI
- [Processing Modes](/guides/processing-modes) - Choose the right mode
- [GPU Acceleration](/guides/gpu-acceleration) - Performance optimization
- [Migration v1→v2](/guides/migration-v1-to-v2) - Upgrade guide

### Features

- [Boundary-Aware Processing](/features/boundary-features) - Seamless stitching
- [RGB Augmentation](/features/rgb-augmentation) - Add true color
- [Infrared Augmentation](/features/infrared-augmentation) - NIR and NDVI
- [Multi-Architecture Support](/features/multi-arch-datasets) - Various network types

### API Reference

- [CLI Commands](/api/cli) - Command-line interface
- [Python API](/api/features) - Programmatic usage
- [Configuration](/api/configuration) - YAML config reference

---

## 💡 Usage Examples

### Download Tiles

```bash
# By bounding box
ign-lidar-hd download \
  --bbox 2.3,48.8,2.4,48.9 \
  --output data/ \
  --max-tiles 10

# By position and radius
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 5000 \
  data/raw_tiles
```

### Process with Defaults

```bash
# Simplest command - uses package defaults
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/patches
```

### GPU-Accelerated Processing

```bash
# Enable GPU acceleration
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/patches \
  processor.use_gpu=true \
  processor.num_workers=8
```

### LOD3 with RGB and NDVI

```bash
# Full-featured processing
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/patches \
  processor.lod_level=LOD3 \
  processor.num_points=32768 \
  features.use_rgb=true \
  features.compute_ndvi=true \
  processor.use_gpu=true
```

### Multi-Format Output

```bash
# Save in multiple formats
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/patches \
  output.format=npz,laz
```

### Python API Examples

```python
from ign_lidar import LiDARProcessor, IGNLiDARDownloader

# Download tiles
downloader = IGNLiDARDownloader("downloads/")
tiles = downloader.download_by_bbox(
    bbox=(2.3, 48.8, 2.4, 48.9),
    max_tiles=5
)

# Process with custom settings
processor = LiDARProcessor(
    lod_level="LOD3",
    patch_size=150.0,
    num_points=16384,
    use_gpu=True
)

patches = processor.process_directory(
    "data/",
    "patches/",
    num_workers=4
)

# PyTorch integration
from torch.utils.data import DataLoader
from ign_lidar import LiDARPatchDataset

dataset = LiDARPatchDataset("patches/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 📦 Output Structure

### NPZ File Contents

```python
{
    'points': np.ndarray,        # [N, 3] XYZ coordinates
    'normals': np.ndarray,       # [N, 3] surface normals
    'curvature': np.ndarray,     # [N] principal curvature
    'intensity': np.ndarray,     # [N] normalized intensity
    'planarity': np.ndarray,     # [N] planarity measure
    'verticality': np.ndarray,   # [N] verticality measure
    'density': np.ndarray,       # [N] local point density
    'labels': np.ndarray,        # [N] building class labels
    # Optional with augmentation:
    'red': np.ndarray,           # [N] RGB red channel
    'green': np.ndarray,         # [N] RGB green channel
    'blue': np.ndarray,          # [N] RGB blue channel
    'infrared': np.ndarray,      # [N] NIR values
}
```

### Directory Structure

```
output_dir/
├── tile1_patch_0001.npz
├── tile1_patch_0002.npz
├── tile1_enriched.laz          # if processing_mode="both"
├── tile2_patch_0001.npz
└── metadata.json                # dataset statistics
```

---

## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET
cd IGN_LIDAR_HD_DATASET

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black ign_lidar/
flake8 ign_lidar/
```

---

## 📋 Requirements

**Core Dependencies:**

- Python 3.8+
- NumPy >= 1.21.0
- laspy >= 2.3.0
- scikit-learn >= 1.0.0
- tqdm >= 4.60.0

**Optional GPU Acceleration:**

- CUDA >= 12.0
- CuPy >= 12.0.0
- RAPIDS cuML >= 24.10

---

## 🤝 Community

- 🐛 [Report Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💡 [Feature Requests](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 📖 [Contributing Guide](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/CONTRIBUTING.md)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/LICENSE) file for details.

---

## Next Steps

Ready to get started?

1. [Install the library](/installation/quick-start)
2. [Set up GPU acceleration](/installation/gpu-setup) (optional, recommended)
3. [Process your first tile](/guides/quick-start)
4. [Explore example configurations](/examples/config-files)
5. [Train a model](/tutorials/training-ml-models)

For questions and support, visit our [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues).
