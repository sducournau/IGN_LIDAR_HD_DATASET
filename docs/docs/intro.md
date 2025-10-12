---
slug: /
sidebar_p### v2.4.4 (2025-10-12) - Latest Release
### LAZ Data Quality Tools & Validation

- 🛠️ **Post-Processing Tools**: New `fix_enriched_laz.py` script for automated LAZ file correction
- 🔍 **Data Quality Detection**: Identifies NDVI calculation errors, eigenvalue outliers, and derived feature corruption
- 📊 **Diagnostic Reports**: Comprehensive analysis with root cause identification and impact assessment
- ✅ **Automated Fixes**: Caps eigenvalues, recomputes derived features, validates results
- 📈 **Enhanced Validation**: Improved NIR data checks and error handling in enrichment pipeline

### Key Fixes

- 🐛 **NDVI Calculation**: Fixed all values = -1.0 when NIR data is missing/corrupted
- 🔢 **Eigenvalue Outliers**: Addressed extreme values (>10,000) causing ML training instability
- 📉 **Derived Features**: Corrected cascading corruption in change_curvature, omnivariance, etc.
- 🏷️ **Duplicate LAZ Fields**: Fixed duplicate field warnings when processing pre-enriched LAZ files
- ⚡ **Production Ready**: Robust validation and error handling for real-world data quality issuese: IGN LiDAR HD Processing Library
---

# IGN LiDAR HD Processing Library

**Version 2.4.4** | Python 3.8+ | MIT License

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform French IGN LiDAR HD point clouds into ML-ready datasets for building classification. Features GPU acceleration (6-20x speedup), rich geometric features (all 35-45+ computed features exported), RGB/NIR augmentation, LAZ data quality tools, and memory-optimized configurations for all system specs.

---

## 🎯 What's New

### v2.4.4 (2025-10-12) - Latest Release

### LAZ Data Quality Tools & Validation

- � **Post-Processing Tools**: New `fix_enriched_laz.py` script for automated LAZ file correction
- 🔍 **Data Quality Detection**: Identifies NDVI calculation errors, eigenvalue outliers, and derived feature corruption
- 📊 **Diagnostic Reports**: Comprehensive analysis with root cause identification and impact assessment
- ✅ **Automated Fixes**: Caps eigenvalues, recomputes derived features, validates results
- 📈 **Enhanced Validation**: Improved NIR data checks and error handling in enrichment pipeline

### Key Fixes

- 🐛 **NDVI Calculation**: Fixed all values = -1.0 when NIR data is missing/corrupted
- 🔢 **Eigenvalue Outliers**: Addressed extreme values (>10,000) causing ML training instability
- � **Derived Features**: Corrected cascading corruption in change_curvature, omnivariance, etc.
- ⚡ **Production Ready**: Robust validation and error handling for real-world data quality issues

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

## ✨ Key Features

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
  version      = {2.4.2}
}
```

**Project maintained by:** [ImagoData](https://github.com/sducournau)

---

<div align="center">

**Made with ❤️ for the LiDAR and Machine Learning communities**

[⬆ Back to top](#ign-lidar-hd-processing-library)

</div>
