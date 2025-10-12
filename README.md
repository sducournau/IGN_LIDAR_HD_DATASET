<div align="center">

# IGN LiDAR HD Processing Library

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ign-lidar-hd)](https://pypi.org/project/ign-lidar-hd/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

**Version 2.4.0** | [📚 Full Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

![LoD3 Building Model](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/docs/static/img/lod3.png?raw=true)

**Transform IGN LiDAR HD point clouds into ML-ready datasets for building classification**

[Quick Start](#-quick-start) • [Features](#-key-features) • [Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/) • [Examples](#-usage-examples)

</div>

---

## 📊 Overview

A comprehensive Python library for processing French IGN LiDAR HD data into machine learning-ready datasets. Features include GPU acceleration, rich geometric features, RGB/NIR augmentation, and flexible YAML-based configuration.

**Key Capabilities:**

- 🚀 **GPU Acceleration**: 6-20x speedup with RAPIDS cuML
- 🎨 **Multi-modal Data**: Geometry + RGB + Infrared (NDVI-ready)
- 🏗️ **Building Classification**: LOD2/LOD3 schemas with 15-30+ classes
- 📦 **Flexible Output**: NPZ, HDF5, PyTorch, LAZ formats
- ⚙️ **YAML Configuration**: Reproducible workflows with example configs

---

## ✨ What's New in v2.4.0

**Enhanced Geometric Feature Validation:**

- 🔧 **Feature Robustness**: All geometric features now guaranteed within valid ranges [0, 1]
- 🎯 **Eigenvalue Clamping**: Prevents negative eigenvalues from numerical artifacts
- 📊 **Density Normalization**: Capped at 1000 points/m³ for ML stability
- ✅ **Boundary Feature Parity**: Complete feature set across all computation paths
- 🔄 **Formula Standardization**: Consistent λ0 normalization (Weinmann et al.)
- 📈 **Zero Overhead**: <1% performance impact from validation
- 🛡️ **Production Ready**: Eliminates out-of-range warnings in all scenarios

**Previous Highlights (v2.3.x):**

**Input Data Preservation & RGB Bug Fix:**

- 🎨 **Preserve RGB/NIR/NDVI from Input LAZ**: Automatically detects and preserves RGB, NIR, and NDVI from input files
- 🐛 **CRITICAL RGB Bug Fix**: Fixed coordinate mismatch in augmented patches - RGB now applied at tile level before extraction
- ⚡ **3x Faster RGB Processing**: Fetch RGB once per tile instead of per patch
- 📊 **Patch Metadata**: Added `_patch_center` and `_patch_bounds` for debugging and validation
- ✅ **Comprehensive Testing**: RGB consistency verified across all augmentation types

**v2.3.1 - Memory Optimization & System Compatibility:**

- 🧠 Memory-optimized configurations for 8GB-32GB+ systems
- 📊 Automatic worker scaling based on memory pressure detection
- ⚙️ Sequential processing mode for minimal memory footprint
- 📖 Comprehensive memory optimization guide (`examples/MEMORY_OPTIMIZATION.md`)
- 🔧 Three configuration profiles: Original (32GB+), Optimized (16-24GB), Sequential (8-16GB)

**v2.3.0 - Processing Modes & Custom Configurations:**

- 🧠 Memory-optimized configurations for 8GB-32GB+ systems
- 📊 Automatic worker scaling based on memory pressure detection
- ⚙️ Sequential processing mode for minimal memory footprint
- 📖 Comprehensive memory optimization guide (`examples/MEMORY_OPTIMIZATION.md`)
- 🔧 Three configuration profiles: Original (32GB+), Optimized (16-24GB), Sequential (8-16GB)

**v2.3.0 - Processing Modes & Custom Configurations:**

- Clear processing modes: `patches_only`, `both`, `enriched_only`
- YAML config files in `examples/` directory for common workflows
- CLI parameter overrides with `--config-file` and `--show-config`

📖 [Full Release History](CHANGELOG.md)

---

## 🚀 Quick Start

### Installation

```bash
# Standard installation (CPU)
pip install ign-lidar-hd

# Optional: GPU acceleration (6-20x speedup)
./install_cuml.sh  # or follow GPU_SETUP.md
```

### Basic Usage

```bash
# Download sample data
ign-lidar-hd download --bbox 2.3,48.8,2.4,48.9 --output data/ --max-tiles 5

# Enrich with features (GPU accelerated if available)
ign-lidar-hd enrich --input-dir data/ --output enriched/ --use-gpu

# Create training patches
ign-lidar-hd patch --input-dir enriched/ --output patches/ --lod-level LOD2
```

### Python API

```python
from ign_lidar import LiDARProcessor

# Initialize and process
processor = LiDARProcessor(lod_level="LOD2")
patches = processor.process_tile("data.laz", "output/")
```

---

## 📋 Key Features

### Core Processing

- **Pure LiDAR** - Geometric analysis without RGB dependencies
- **Multi-level Classification** - LOD2 (15 classes) and LOD3 (30+ classes)
- **Rich Features** - Normals, curvature, planarity, verticality, density, wall/roof scores
- **Augmentation** - Optional RGB from orthophotos, NIR for NDVI
- **Auto-parameters** - Intelligent tile analysis for optimal settings

### Performance

- **GPU Acceleration** - RAPIDS cuML support (6-20x faster)
- **Parallel Processing** - Multi-worker with automatic CPU detection
- **Memory Optimized** - Per-chunk architecture, 50-60% reduction
- **Smart Skip** - Resume interrupted workflows automatically

### Flexibility

- **Processing Modes** - Three clear modes: patches only, both, or LAZ only
- **YAML Configs** - Declarative workflows with example templates
- **Multiple Formats** - NPZ, HDF5, PyTorch, LAZ (single or multi-format)
- **CLI & API** - Command-line tool and Python library

---

## 💡 Usage Examples

### Mode 1: Create Training Patches (Default)

```bash
# Using example config
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  input_dir=data/raw \
  output_dir=data/patches

# Or with CLI parameters
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/patches \
  output.processing_mode=patches_only
```

### Mode 2: Both Patches & Enriched LAZ

```bash
ign-lidar-hd process \
  --config-file examples/config_complete.yaml \
  input_dir=data/raw \
  output_dir=data/both
```

### Mode 3: LAZ Enrichment Only

```bash
ign-lidar-hd process \
  --config-file examples/config_quick_enrich.yaml \
  input_dir=data/raw \
  output_dir=data/enriched
```

> **⚠️ Note on Enriched LAZ Files:** When generating enriched LAZ tile files, geometric features (normals, curvature, planarity, etc.) may show artifacts at tile boundaries due to the nature of the source data. These artifacts are inherent to tile-based processing and **do not appear in patch exports**, which provide the best results for machine learning applications. For optimal quality, use `patches_only` or `both` modes.

### GPU-Accelerated Processing

```bash
ign-lidar-hd process \
  --config-file examples/config_gpu_processing.yaml \
  input_dir=data/raw \
  output_dir=data/output
```

### Preview Configuration

```bash
ign-lidar-hd process \
  --config-file examples/config_training_dataset.yaml \
  --show-config \
  input_dir=data/raw
```

### Python API Examples

```python
from ign_lidar import LiDARProcessor, IGNLiDARDownloader

# Download tiles
downloader = IGNLiDARDownloader("downloads/")
tiles = downloader.download_by_bbox(bbox=(2.3, 48.8, 2.4, 48.9), max_tiles=5)

# Process with custom config
processor = LiDARProcessor(
    lod_level="LOD3",
    patch_size=150.0,
    num_points=16384,
    use_gpu=True
)

# Single tile
patches = processor.process_tile("input.laz", "output/")

# Batch processing
patches = processor.process_directory("input_dir/", "output_dir/", num_workers=4)

# PyTorch integration
from torch.utils.data import DataLoader
dataset = LiDARPatchDataset("patches/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 📦 Output Format

### NPZ Structure

Each patch is saved as NPZ with:

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
    # Facultative features:
    'wall_score': np.ndarray,    # [N] wall likelihood (planarity * verticality)
    'roof_score': np.ndarray,    # [N] roof likelihood (planarity * horizontality)
    # Optional with augmentation:
    'red': np.ndarray,           # [N] RGB red
    'green': np.ndarray,         # [N] RGB green
    'blue': np.ndarray,          # [N] RGB blue
    'infrared': np.ndarray,      # [N] NIR values
}
```

### Available Formats

- **NPZ** - Default NumPy format (recommended for ML)
- **HDF5** - Hierarchical data format
- **PyTorch** - `.pt` files for PyTorch
- **LAZ** - Point cloud format for visualization (may show boundary artifacts in tile mode)
- **Multi-format** - Save in multiple formats: `hdf5,laz`, `npz,torch`

> **💡 Tip:** For machine learning applications, NPZ/HDF5/PyTorch patch formats provide cleaner geometric features than enriched LAZ tiles.

---

## 📚 Documentation

### Quick Links

- [📖 Full Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- [🚀 Installation Guide](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/installation/quick-start)
- [⚡ GPU Setup](GPU_SETUP.md)
- [🎯 Quick Reference](QUICK_REFERENCE.md)
- [🗺️ QGIS Integration](docs/guides/QUICK_START_QGIS.md)

### Examples & Workflows

- `examples/` - Python usage examples
- `examples/*.yaml` - Configuration templates
- [PyTorch Integration](examples/pytorch_dataloader.py)
- [Parallel Processing](examples/parallel_processing_example.py)

### Architecture & API

- [System Architecture](docs/architecture.md)
- [CLI Reference](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/api/cli)
- [Python API](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/api/features)
- [Configuration Schema](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/api/configuration)

---

## 🛠️ Development

```bash
# Clone and install in development mode
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET
cd IGN_LIDAR_HD_DATASET
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black ign_lidar/
```

---

## 📋 Requirements

**Core:**

- Python 3.8+
- NumPy >= 1.21.0
- laspy >= 2.3.0
- scikit-learn >= 1.0.0

**Optional GPU Acceleration:**

- CUDA >= 12.0
- CuPy >= 12.0.0
- RAPIDS cuML >= 24.10 (recommended)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Support & Contributing

- 🐛 [Report Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💡 [Feature Requests](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 📖 [Contributing Guide](CONTRIBUTING.md)

---

## 📝 Cite Me

If you use this library in your research or projects, please cite:

```bibtex
@software{ign_lidar_hd_dataset,
  author       = {Simon Ducournau},
  title        = {IGN LiDAR HD Processing Library},
  year         = {2025},
  publisher    = {ImagoData},
  url          = {https://github.com/sducournau/IGN_LIDAR_HD_DATASET},
  version      = {2.3.0}
}
```

**Project maintained by:** [ImagoData](https://github.com/sducournau)

---

<div align="center">

**Made with ❤️ for the LiDAR and Machine Learning communities**

[⬆ Back to top](#ign-lidar-hd-processing-library)

</div>
