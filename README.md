<div align="center">

# IGN LiDAR HD Processing Library

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ign-lidar-hd)](https://pypi.org/project/ign-lidar-hd/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

**Version 2.5.3** | [📚 Full Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

![LoD3 Building Model](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/docs/static/img/lod3.png?raw=true)

**Transform IGN LiDAR HD point clouds into ML-ready datasets for building classification**

[Quick Start](#-quick-start) • [What's New](#-whats-new-in-v253) • [Features](#-key-features) • [Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/) • [Examples](#-usage-examples)

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

## ✨ What's New in v2.5.4

### 🆕 Optional Reclassification in Main Pipeline

**v2.5.4 adds reclassification as an optional feature in the main processing pipeline!**

You can now enable optimized ground truth reclassification directly in your processing config:

```yaml
processor:
  reclassification:
    enabled: true # Optional - disabled by default
    acceleration_mode: "auto" # CPU, GPU, or GPU+cuML
    use_geometric_rules: true
```

**Benefits:**

- ✅ **Flexible**: Enable/disable without separate runs
- ✅ **Fast**: GPU-accelerated spatial indexing
- ✅ **Accurate**: Ground truth from BD TOPO®
- ✅ **Backward compatible**: Existing configs work unchanged

📖 See [`docs/RECLASSIFICATION_INTEGRATION.md`](docs/RECLASSIFICATION_INTEGRATION.md) and [`docs/RECLASSIFICATION_QUICKSTART.md`](docs/RECLASSIFICATION_QUICKSTART.md) for details

---

## ✨ What's New in v2.5.3

### 🔧 Critical Fix: Ground Truth Classification

**v2.5.3 fixes critical issues with BD TOPO® ground truth classification.**

#### What Was Fixed

Ground truth classification from IGN BD TOPO® wasn't working - no points were being classified to roads, cemeteries, power lines, etc.

**Root Causes:**

- Incorrect class imports (`MultiSourceDataFetcher` → `DataFetcher`)
- Missing BD TOPO feature parameters (cemeteries, power_lines, sports)
- Missing buffer parameters (road_width_fallback, etc.)
- Wrong method call (`fetch_data()` → `fetch_all()`)

**Impact:** Ground truth now works correctly for all ASPRS codes:

- ✅ ASPRS 11: Roads
- ✅ ASPRS 40: Parking
- ✅ ASPRS 41: Sports Facilities
- ✅ ASPRS 42: Cemeteries
- ✅ ASPRS 43: Power Lines

#### What Was Added

**New BD TOPO® Configuration Directory** (`ign_lidar/configs/data_sources/`)

Pre-configured Hydra configs for different use cases:

- `default.yaml` - General purpose with core features
- `asprs_full.yaml` - Complete ASPRS classification
- `lod2_buildings.yaml` - Building-focused for LOD2
- `lod3_architecture.yaml` - Architectural focus for LOD3
- `disabled.yaml` - Pure geometric features

**Usage:**

```yaml
defaults:
  - data_sources: asprs_full # or lod2_buildings, lod3_architecture
  - _self_
```

📖 See `ign_lidar/configs/data_sources/README.md` for complete documentation

---

### 📦 Previous Updates (v2.5.0-2.5.2)

**v2.5.0 represented a complete internal modernization while maintaining 100% backward compatibility!**

#### Unified Feature System ✨

- **FeatureOrchestrator**: New unified class replaces FeatureManager + FeatureComputer
- **Simpler API**: One class handles all feature computation with automatic strategy selection
- **Better organized**: Clear separation of concerns with strategy pattern
- **Fully compatible**: All existing code works without changes

#### Improved Code Quality

- **67% reduction** in feature orchestration code complexity
- **Optimized error messages** and validation throughout
- **Complete type hints** for better IDE support
- **Modular architecture** for easier maintenance and extension

#### Migration Made Easy

- **Zero breaking changes**: Your v1.x code continues to work
- **Deprecation warnings**: Clear guidance for future-proofing your code
- **Migration guide**: Step-by-step instructions in [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Backward compatible**: Legacy APIs will be maintained through v2.x series

```python
# NEW (v2.0) - Recommended unified API
from ign_lidar import LiDARProcessor

processor = LiDARProcessor(
    config_path="config.yaml",
    feature_mode="lod3"  # Clearer mode specification
)

# Access unified orchestrator
orchestrator = processor.feature_orchestrator
print(f"Feature mode: {orchestrator.mode}")
print(f"Has RGB: {orchestrator.has_rgb}")
print(f"Available features: {orchestrator.get_feature_list('lod3')}")

# OLD (v1.x) - Still works with deprecation warnings
# feature_manager = processor.feature_manager  # Deprecated but functional
# feature_computer = processor.feature_computer  # Deprecated but functional
```

**Why upgrade?**

- Future-proof your code for v3.0
- Access to new features and improvements
- Better performance and error handling
- Professional, maintainable codebase

📖 See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for complete upgrade instructions  
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

## 🎓 Feature Modes

IGN LiDAR HD supports multiple feature computation modes optimized for different use cases:

### Minimal Mode (4 features) - Ultra-Fast

**Best for:** Quick processing, classification updates, minimal computation

**Features:** normal_z, planarity, height_above_ground, density

**Performance:** ⚡⚡⚡⚡⚡ Fastest (~5s per 1M points)

### LOD2 Mode (12 features) - Fast Training

**Best for:** Basic building classification, quick prototyping, baseline models

**Features:** XYZ (3), normal_z, planarity, linearity, height, verticality, RGB (3), NDVI

**Performance:** ~15s per 1M points (CPU), fast convergence

### LOD3 Mode (37 features) - Detailed Modeling

**Best for:** Architectural modeling, fine structure detection, research

**Features:** Complete normals (3), eigenvalues (5), curvature (2), shape descriptors (6), height features (2), building scores (3), density features (4), architectural features (4), spectral (5)

**Performance:** ~45s per 1M points (CPU), best accuracy

### Full Mode (37+ features) - Complete Feature Set

**Best for:** Research, feature analysis, maximum information extraction

**All Features:** All LOD3 features plus any additional computed features

**Performance:** ~50s per 1M points (CPU), complete geometric description

**Usage:**

```yaml
features:
  mode: minimal # or lod2, lod3, full, custom
  k_neighbors: 10
```

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

- **[📖 Complete Documentation Index](DOCUMENTATION.md)** - Full documentation navigation
- [📖 Full Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/) - Online documentation site
- [🚀 Installation Guide](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/installation/quick-start)
- [📋 Testing Guide](TESTING.md) - Test suite and development testing
- [📝 Changelog](CHANGELOG.md) - Version history and release notes

### User Guides

Located in **[docs/guides/](docs/guides/)**:

- [ASPRS Classification Guide](docs/guides/ASPRS_CLASSIFICATION_GUIDE.md) - Complete ASPRS standards
- [Building Classification Guide](docs/guides/BUILDING_CLASSIFICATION_QUICK_REFERENCE.md) - Building class reference
- [Vegetation Classification Guide](docs/guides/VEGETATION_CLASSIFICATION_GUIDE.md) - Vegetation analysis

### Examples & Configuration

Located in **[examples/](examples/)**:

- [Example Configurations](examples/) - YAML configuration templates
- [Versailles Configs](examples/) - LOD2, LOD3, and ASPRS examples
- [Architectural Analysis](examples/ARCHITECTURAL_STYLES_README.md) - Style detection
- [Multi-scale Training](examples/MULTI_SCALE_TRAINING_STRATEGY.md) - Advanced training

### Architecture & API

- [System Architecture](docs/architecture/) - Design documentation
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
@software{ign_lidar_hd,
  author       = {Ducournau, Simon},
  title        = {IGN LiDAR HD Processing Library},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/sducournau/IGN_LIDAR_HD_DATASET},
  version      = {2.5.3}
}
```

**Project maintained by:** [ImagoData](https://github.com/sducournau)

---

<div align="center">

**Made with ❤️ for the LiDAR and Machine Learning communities**

[⬆ Back to top](#ign-lidar-hd-processing-library)

</div>
