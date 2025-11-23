---
slug: /
sidebar_position: 1
title: IGN LiDAR HD Processing Library
---

# IGN LiDAR HD Processing Library

**Version 3.5.0** | Python 3.8+ | MIT License | [GitHub](https://github.com/sducournau/IGN_LIDAR_HD_DATASET)

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Transform French IGN LiDAR HD point clouds into ML-ready datasets for building classification.**

Production-ready library featuring GPU acceleration (16× speedup), rich geometric features (45+), RGB/NIR augmentation, RTM spatial indexing, memory-optimized processing, and advanced facade detection with 94-97% classification accuracy.

---

## 🎯 What's New in v3.5.0

### Latest Stable Release (November 2025)

Package harmonization and documentation consolidation release.

:::info Documentation Update
**Version 3.5.0** harmonizes all version references, consolidates documentation, and improves package consistency across all files.
:::

### 🌟 Key Highlights

- ✅ **Production Ready** - Stable, tested, optimized for real-world workflows
- 🚀 **FAISS GPU Boost** - Dynamic VRAM detection enables 10-50× faster k-NN on large datasets
- 💾 **Smart Memory** - Automatic FP16 precision for 100M+ points on 16GB GPUs
- 🔴 **Critical Bug Fix** - BD TOPO reclassification corrected (+20-30% accuracy)
- 🎯 **95% Fewer Artifacts** - Unified filtering eliminates boundary issues
- 🏗️ **Enhanced Detection** - +30-40% improvement in facade point capture

### 📊 Performance Improvements

| Metric                  | Before       | After        | Improvement          |
| ----------------------- | ------------ | ------------ | -------------------- |
| k-NN queries (large)    | 30-90s CPU   | 5-15s GPU    | **10-50× faster** 🔥 |
| FAISS memory limit      | 15M fixed    | 100M+ adapt. | **6× capacity** 🚀   |
| Building classification | 70-75%       | 94-97%       | **+20-30%** ✅       |
| DTM file lookup         | Sequential   | RTM indexed  | **10× faster** ⚡    |
| Planarity artifacts     | 100-200/tile | 5-10/tile    | **95% reduction** 📉 |
| Processing speed        | Baseline     | Optimized    | **40-50% faster** ⏱️ |
| Facade detection        | Standard     | Enhanced     | **+30-40%** 📈       |
| Memory stability        | OOM issues   | Auto-chunked | **100% stable** ✅   |

### 🆕 Major Features

**FAISS GPU Memory Optimization** (v3.4.1) 🚀

- Dynamic VRAM detection replaces hardcoded 15M point limit
- Automatic Float16 (FP16) precision for datasets >50M points (cuts memory in half)
- Smart memory calculation: query results + index storage + temp memory
- Adaptive threshold: 80% of detected VRAM limit
- **Impact on RTX 4080 SUPER (16GB)**: 72M point dataset now runs on GPU (was CPU-only)
- Expected speedup: **10-50× faster** k-NN queries (5-15s vs 30-90s)
- Supports up to **100M+ points on 16GB GPUs** with FP16 precision
- Dynamic temp memory: scales with VRAM (4GB for 16GB GPU, 2GB for 8GB GPU)

**Unified Feature Filtering** (v3.1.0)

- Generic API for planarity, linearity, horizontality filtering
- Adaptive spatial smoothing with variance detection
- Eliminates 95% of boundary artifacts (100-200 → 5-10 per tile)
- 100% elimination of NaN/Inf warnings
- ~60% code reduction through unified implementation

**Performance Enhancements** (v3.3.3)

- RTM spatial indexing for instant DTM file discovery
- Intelligent gap filling with nearest-neighbor interpolation
- Automatic memory-optimized chunking (2M-5M points)
- Building cluster IDs for instance segmentation
- Enhanced facade detection with +30-40% improvement

**Rules Framework** (v3.2.1)

- Extensible plugin architecture for custom classification rules
- 7 confidence calculation methods (binary, linear, sigmoid, gaussian, etc.)
- Hierarchical execution with 4 strategies
- Complete documentation with 15+ visual diagrams
- Type-safe design with comprehensive validation

### 🔄 Quick Upgrade

```bash
# Upgrade to latest version
pip install --upgrade ign-lidar-hd

# Verify installation
ign-lidar-hd --version  # Should show 3.5.0
```

:::tip Important for BD TOPO Users
If using BD TOPO classification, consider reprocessing your data to benefit from the critical v3.3.4 bug fix that improves accuracy by 20-30%.
:::

### 📖 Release Notes

- 📦 **[v3.5.0 (Current)](./release-notes/v3.5.0)** - Package harmonization & documentation **← NEW**
- 🚀 **[v3.4.1](./release-notes/v3.4.1)** - FAISS GPU memory optimization
- 🎯 **[v3.4.0](./release-notes/v3.4.0)** - GPU optimizations & road classification
- 📍 **[v3.3.5](./release-notes/v3.3.5)** - Maintenance release
- 🔴 **[v3.3.4 (Critical)](./release-notes/v3.3.4)** - BD TOPO priority fix **← IMPORTANT**
- ⚡ **[v3.3.3](./release-notes/v3.3.3)** - Performance optimizations
- 🎲 **[v3.2.1](./release-notes/v3.2.1)** - Rules framework
- ✨ **[v3.1.0](./release-notes/v3.1.0)** - Unified feature filtering
- 🎯 **[v3.0.6](./release-notes/v3.0.6)** - Planarity filtering
- 🚀 **[v3.0.0](./release-notes/v3.0.0)** - Major architectural refactor

---

## 🚀 Quick Start

### Installation

```bash
# Standard installation (CPU)
pip install ign-lidar-hd

# Optional: GPU acceleration (6-20x speedup)
pip install cupy-cuda12x  # or cupy-cuda11x for CUDA 11.x
conda install -c rapidsai -c conda-forge -c nvidia cuml cuspatial
```

### Basic Usage

```bash
# Download sample data
ign-lidar-hd download --bbox 2.3,48.8,2.4,48.9 --output data/ --max-tiles 5

# Process with features (GPU accelerated if available)
ign-lidar-hd process input_dir=data/ output_dir=output/
```

### Python API

```python
from ign_lidar import LiDARProcessor

# Initialize with configuration
processor = LiDARProcessor(config_path="config.yaml")

# Process a single tile
patches = processor.process_tile("data/tile.laz", "output/")
```

---

## 🎯 Key Features

### Core Capabilities

- **📥 IGN Download** - Download HD LiDAR tiles from IGN French national geoportal
- **🎨 RGB Enhancement** - Fetch RGB colors from IGN orthophotos
- **📡 NIR Enhancement** - Fetch near-infrared channel from IGN IRC orthophotos
- **🌿 NDVI Computation** - Compute vegetation indices from RGB + NIR
- **⚙️ Feature Engineering** - Compute 45+ geometric features (normals, curvature, height, planarity, etc.)
- **🏗️ LOD Classification** - Building-focused LOD2/LOD3 classification taxonomy
- **🧮 Axonometric Views** - Generate multiple viewpoint representations
- **🌐 WFS Ground Truth** - Fetch building/vegetation polygons from IGN BD TOPO®
- **📦 Multiple Formats** - NPZ, PyTorch, TensorFlow, HDF5, LAZ

### Performance & Optimization

- **🚀 GPU Acceleration** - RAPIDS cuML support (6-20× faster)
- **⚡ Parallel Processing** - Multi-worker with automatic CPU detection
- **🧠 Memory Optimized** - Chunked processing, 50-60% memory reduction
- **💾 Smart Skip** - Resume interrupted workflows automatically
- **🗺️ RTM Spatial Indexing** - 10× faster DTM file lookups
- **📊 Automatic Chunking** - Prevents OOM crashes on large tiles

### Flexibility

- **📁 Processing Modes** - Patches only, both patches+LAZ, or LAZ only
- **📋 YAML Configs** - Declarative workflows with example templates
- **🔧 CLI & API** - Command-line tool and Python library
- **🎲 Rules Framework** - Extensible classification rules engine
- **🏛️ Architectural Styles** - Encode regional/historical characteristics

---

## 📋 Processing Modes

### Patches Only (Default)

ML-ready patches for training:

```bash
ign-lidar-hd process input_dir=data/ output_dir=patches/
```

### Both Patches & Enriched LAZ

Training + GIS visualization:

```bash
ign-lidar-hd process input_dir=data/ output_dir=both/ processor.processing_mode=both
```

### Enriched LAZ Only

Fast GIS workflow:

```bash
ign-lidar-hd process input_dir=data/ output_dir=enriched/ processor.processing_mode=enriched_only
```

:::tip Best Practice
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

See [configuration examples](/reference/config-examples) for complete workflows.

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
- [GPU Setup Guide](/installation/gpu-setup)
- [Quick Start Tutorial](/guides/quick-start)
- [Basic Usage Guide](/guides/basic-usage)

### Core Features

- [Feature Computation](/features/geometric-features)
- [Classification Systems](/reference/classification-workflow)
- [Rules Framework](/features/rules-framework)
- [Ground Truth Integration](/features/ground-truth-classification)

### Advanced Topics

- [RGB & NIR Augmentation](/features/rgb-augmentation)
- [GPU Acceleration](/guides/gpu-acceleration)
- [Boundary-Aware Processing](/features/boundary-aware)
- [Multi-Architecture Support](/features/multi-architecture)

### Reference

- [CLI Commands](/api/cli)
- [Python API](/api/features)
- [Configuration Reference](/api/configuration)
- [Architecture](/architecture)

---

## 📄 License

MIT License - see [LICENSE](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/LICENSE) file for details.

---

## 🤝 Support & Contributing

- 🐛 [Report Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💡 [Feature Requests](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 📖 [Full Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- 📧 [Contact Author](mailto:simon.ducournau@gmail.com)

---

## 📝 Citation

If you use this library in your research or projects, please cite:

```bibtex
@software{ign_lidar_hd_dataset,
  author       = {Simon Ducournau},
  title        = {IGN LiDAR HD Processing Library},
  year         = {2025},
  publisher    = {ImagoData},
  url          = {https://github.com/sducournau/IGN_LIDAR_HD_DATASET},
  version      = {3.5.0}
}
```

---

**Project maintained by:** [ImagoData](https://github.com/sducournau)

**Made with ❤️ for the LiDAR and Machine Learning communities**

[⬆ Back to top](#ign-lidar-hd-processing-library)
