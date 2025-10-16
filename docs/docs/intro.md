---
slug: /
sidebar_position: 1
title: IGN LiDAR HD Processing Library
---

# IGN LiDAR HD Processing Library

**Version 2.5.3** | Python 3.8+ | MIT License

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform French IGN LiDAR HD point clouds into ML-ready datasets for building classification. Features GPU acceleration (6-20x speedup), rich geometric features (all 35-45+ computed features exported), RGB/NIR augmentation, LAZ data quality tools, and memory-optimized configurations for all system specs.

---

## 🎯 What's New

### v2.5.3 (2025-10-16) - Latest Release

### 🔧 Critical Fix: Ground Truth Classification

This release fixes critical issues with BD TOPO® ground truth classification that prevented points from being classified to roads, cemeteries, power lines, and other infrastructure features.

**Fixed:**

- ✅ Ground truth classification from BD TOPO® now works correctly
- ✅ ASPRS mode classification (was incorrectly using LOD3 mapping)
- ✅ Fixed DataFetcher integration with all BD TOPO features
- ✅ Added data_sources configuration directory for multi-source integration
- ✅ Roads (ASPRS 11), cemeteries (ASPRS 42), power lines (ASPRS 43), and sports (ASPRS 41) now classified correctly

**Impact:** All ground truth classifications now work correctly across ASPRS, LOD2, and LOD3 modes.

### v2.5.1 (2025-10-15)

### Maintenance & Documentation Updates

- 📦 **Version Update**: Maintenance release with documentation improvements and harmonization
- 📚 **Documentation**: Updated version references across all documentation files (README, docusaurus intro pages)
- 🔧 **Configuration**: Updated version in conda recipe and package configuration files
- ⚠️ **Deprecation Notices**: Updated deprecation timelines for consistency

### v2.5.0 (2025-10-14)

### System Consolidation & Modernization

- 🎯 **Unified Feature System**: New `FeatureOrchestrator` replaces `FeatureManager` + `FeatureComputer`
- 🏗️ **Strategy Pattern Architecture**: Clean separation of CPU/GPU/Chunked/Boundary-aware processing
- � **Enhanced Type Hints**: Complete type annotations throughout codebase for better IDE support
- 📊 **Improved Error Messages**: Clear, actionable error messages with validation details
- � **Better Documentation**: Updated API reference with comprehensive examples
- ✅ **100% Backward Compatible**: All existing code works without modification

### Key Improvements

- � **Automatic Strategy Selection**: Intelligent selection based on configuration and data
- 🧩 **Modular Design**: Better separation of concerns across modules
- 📈 **Enhanced Validation**: Improved configuration parameter checking
- 🔧 **Resource Management**: Proper initialization and cleanup
- ⚡ **Production Ready**: Robust error handling and validation

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

## 🎯 Key Features

- **📥 IGN Download**: Download HD LiDAR tiles from IGN French national geoportal
- **🎨 RGB Enhancement**: Fetch RGB colors from IGN orthophotos
- **📡 NIR Enhancement**: Fetch near-infrared channel from IGN IRC orthophotos
- **� NDVI Computation**: Compute vegetation indices from RGB + NIR
- **⚙️ Feature Engineering**: Compute geometric features (normals, curvature, height, planarity, etc.)
- **🏗️ LOD Classification**: Building-focused LOD2/LOD3 classification taxonomy
- **🏛️ Architectural Styles**: Encode regional/historical architectural characteristics
- **🧮 Axonometric Views**: Generate multiple viewpoint representations for 3D geometry
- **🌐 WFS Ground Truth**: Fetch building/vegetation polygons from IGN BD TOPO® WFS service
- **📦 Multiple Output Formats**: NPZ, PyTorch, TensorFlow, HDF5, LAZ
- **�️ Multi-Scale Training**: Generate datasets at multiple patch sizes (50m, 100m, 150m)

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
ign-lidar-hd process input_dir=data/ output_dir=patches/ processor.use_gpu=true

# Validate and fix LAZ data quality (new in v2.4.4)
python scripts/fix_enriched_laz.py enriched/tile.laz --fix
```

:::tip
For ML applications, patches provide cleaner geometric features than LAZ tiles (no boundary artifacts).
:::

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

## 📚 Learn More

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
  version      = {2.5.3}
}
```

**Project maintained by:** [ImagoData](https://github.com/sducournau)

---

<div align="center">

**Made with ❤️ for the LiDAR and Machine Learning communities**

[⬆ Back to top](#ign-lidar-hd-processing-library)

</div>
