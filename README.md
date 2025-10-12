<div align="center">

# IGN LiDAR HD Processing Library

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ign-lidar-hd)](https://pypi.org/project/ign-lidar-hd/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

**Version 2.4.2** | [📚 Full Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

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

## ✨ What's New in v2.4.2

### Complete GPU Acceleration

- 🚀 **Full GPU Implementation**: All advanced features in "full" mode now GPU-accelerated
- ⚡ **5-10x Speedup**: Massive performance boost for large point clouds (>10M points)
- 🎯 **GPU Eigenvalue Features**: Accelerated eigenvalue decomposition, entropy, omnivariance
- 🏗️ **GPU Architectural Features**: Edge strength, corner likelihood, overhang detection
- 📊 **GPU Density Features**: Accelerated density computation and neighborhood analysis
- 🔄 **Seamless Fallback**: Automatic GPU/CPU switching with zero API changes
- ✅ **Complete Compatibility**: Same output quality, same interface, better performance

### Key Benefits

- 🎯 **ML Model Stability**: No more NaN/Inf values, improved convergence
- ⚡ **Zero Breaking Changes**: Drop-in upgrade from v2.3.x
- 📊 **Consistent Features**: Same results across CPU/GPU/boundary processing
- 🚀 **Production Ready**: Enterprise-grade reliability and deterministic behavior

### Recent Highlights (v2.3.x)

**Input Data Preservation & RGB Enhancement:**

- 🎨 Preserve RGB/NIR/NDVI from input LAZ files automatically
- 🐛 Fixed critical RGB coordinate mismatch in augmented patches
- ⚡ 3x faster RGB processing (tile-level fetching)
- 📊 Added patch metadata for debugging and validation

**Memory Optimization:**

- 🧠 Support for 8GB-32GB+ systems with optimized configurations
- 📊 Automatic worker scaling based on memory pressure
- ⚙️ Sequential processing mode for minimal footprint
- Three configuration profiles for different system specs

**Processing Modes:**

- Clear modes: `patches_only`, `both`, `enriched_only`
- YAML configuration files with example templates
- CLI parameter overrides with `--config-file`

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

- **🎯 Complete Feature Export** - All 35-45 computed geometric features saved to disk (v2.4.2+)
- **🏗️ Multi-level Classification** - LOD2 (12 features), LOD3 (38 features), Full (43+ features) modes
- **📊 Rich Geometry** - Normals, curvature, eigenvalues, shape descriptors, architectural features, building scores
- **🎨 Optional Augmentation** - RGB from orthophotos, NIR, NDVI for vegetation analysis
- **⚙️ Auto-parameters** - Intelligent tile analysis for optimal settings
- **📝 Feature Tracking** - Metadata includes feature names and counts for reproducibility

### Performance

- **🚀 GPU Acceleration** - RAPIDS cuML support (6-20x faster)
- **⚡ Parallel Processing** - Multi-worker with automatic CPU detection
- **🧠 Memory Optimized** - Chunked processing, 50-60% reduction
- **💾 Smart Skip** - Resume interrupted workflows automatically (~1800x faster)

### Flexibility

- **📁 Processing Modes** - Three clear modes: patches only, both, or LAZ only
- **📋 YAML Configs** - Declarative workflows with example templates
- **📦 Multiple Formats** - NPZ, HDF5, PyTorch, LAZ (single or multi-format)
- **🔧 CLI & API** - Command-line tool and Python library

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

## 🎓 Feature Modes (LOD2 vs LOD3 vs Full)

### LOD2 Mode (12 features) - Fast Training

**Best for:** Basic building classification, quick prototyping, baseline models

**Features:** XYZ, normal_z, planarity, linearity, height, verticality, RGB, NDVI

**Performance:** ~15s per 1M points (CPU), fast convergence

### LOD3 Mode (38 features) - Detailed Modeling

**Best for:** Architectural modeling, fine structure detection, research

**Additional Features:** Complete normals (3), eigenvalues (5), curvature (2), shape descriptors (6), height features (3), building scores (3), density features (5), architectural features (4)

**Performance:** ~45s per 1M points (CPU), best accuracy

### Full Mode (43+ features) - Complete Feature Set

**Best for:** Research, feature analysis, maximum information extraction

**All Features:** Everything from LOD3 plus additional height variants (z_absolute, z_from_ground, z_from_median), distance_to_center, local_roughness, horizontality

**Performance:** ~50s per 1M points (CPU), complete geometric description

**Output Format:**

- NPZ/HDF5/PyTorch: Full feature matrix with all features
- LAZ: All features as extra dimensions for GIS tools
- Metadata: `feature_names` and `num_features` for tracking

📖 See [Feature Modes Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/features/feature-modes) for complete details.

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

- `examples/` - Python usage examples and configuration templates
- `examples/config_lod2_simplified_features.yaml` - Fast LOD2 training (12 features)
- `examples/config_lod3_full_features.yaml` - Detailed LOD3 modeling (38 features)
- `examples/config_complete.yaml` - Full mode with all 43+ features
- `examples/config_multiscale_hybrid.yaml` - Multi-scale adaptive features
- [PyTorch Integration](examples/pytorch_dataloader.py)
- [Parallel Processing](examples/parallel_processing_example.py)

### Architecture & API

- [System Architecture](docs/FEATURE_SYSTEM_ARCHITECTURE.md)
- [Geometric Features Reference](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/features/geometric-features)
- [Feature Modes Guide](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/features/feature-modes)
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
  version      = {2.4.2}
}
```

**Project maintained by:** [ImagoData](https://github.com/sducournau)

---

<div align="center">

**Made with ❤️ for the LiDAR and Machine Learning communities**

[⬆ Back to top](#ign-lidar-hd-processing-library)

</div>
