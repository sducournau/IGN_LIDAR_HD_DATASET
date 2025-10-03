<div align="center">

# IGN LiDAR HD Processing Library

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ign-lidar-hd)](https://pypi.org/project/ign-lidar-hd/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

**Version 1.6.4** | [🇫🇷 French](README.fr.md) | [📚 Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)

</div>

<img src="https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/website/static/img/lod3.png?raw=true" alt="LoD3 Building Model Icon">

A comprehensive Python library for processing IGN (Institut National de l'Information Géographique et Forestière) LiDAR HD data into machine learning-ready datasets for Building Level of Detail (LOD) classification tasks.

## ✨ What's New in v1.6.4

📺 **Enhanced Documentation** - Updated README with embedded YouTube player for better video experience  
🎨 **Improved Presentation** - Better visual integration of demo content  
🔧 **Minor Updates** - Documentation improvements and refinements

### Previous Updates (v1.6.2)

🔧 **Critical GPU Feature Fix** - Corrected eigenvalue normalization in GPU implementation (now matches CPU)  
🛡️ **Robust Feature Computation** - Added degenerate case filtering and outlier-resistant curvature  
🎯 **Enhanced Quality** - Radius search support and comprehensive validation suite  
⚠️ **Breaking Change**: GPU feature values changed for users of GPU acceleration

[📖 Analysis & Fixes](GEOMETRIC_FEATURES_ANALYSIS.md) | [✅ Implementation](IMPLEMENTATION_SUMMARY.md)

### Previous Updates (v1.6.0)

🎯 **Enhanced Data Augmentation** - Augmentation now happens during ENRICH phase (before feature computation) for better feature-geometry consistency  
🎨 **RGB CloudCompare Fix** - Perfect RGB color display with corrected 16-bit scaling  
⚡ **40% Processing Trade-off** - Slightly longer processing time but significantly better training data quality  
📚 **Improved Documentation** - Comprehensive guides and examples for all features

[📖 Full Release Notes](RELEASE_v1.6.0_SUMMARY.md) | [🔄 Migration Guide](AUGMENTATION_IMPROVEMENT.md)

## 📺 Video Demo

<div align="center">
  <iframe width="800" height="450" src="https://www.youtube.com/embed/ksBWEhkVqQI" title="IGN LiDAR HD Processing Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  <p><em>Learn how to process LiDAR data for machine learning applications</em></p>
</div>

## 📊 Project Overview

This library transforms raw IGN LiDAR HD point clouds into structured datasets ready for machine learning applications. Built specifically for building classification tasks, it handles the complete pipeline from data acquisition to training-ready patches.

### 🔄 **Processing Workflow**

```mermaid
flowchart TD
    A[IGN LiDAR HD Data] --> B[Download Tiles]
    B --> C[Enrich with Features]
    C --> D[Create Training Patches]
    D --> E[ML-Ready Dataset]

    B --> B1[Smart Skip Detection]
    C --> C1[GPU/CPU Processing]
    C --> C2[Geometric Features]
    D --> D1[Data Augmentation]
    D --> D2[LOD Classification]

    style A fill:#e1f5fe
    style E fill:#e8f5e8
    style B1 fill:#fff3e0
    style C1 fill:#fff3e0
    style D1 fill:#fff3e0
```

**📈 Project Stats:**

- 🏗️ **14 core modules** - Comprehensive processing toolkit
- 📝 **10 example scripts** - From basic usage to advanced workflows
- 🧪 **Comprehensive test suite** - Ensuring reliability and performance
- 🌍 **50+ curated tiles** - Covering diverse French territories
- ⚡ **GPU & CPU support** - Flexible computation backends
- 🔄 **Smart resumability** - Never reprocess existing data

---

## 🚀 Quick Start

### Installation

```bash
# Standard installation (CPU only)
pip install ign-lidar-hd

# With RGB augmentation support
pip install ign-lidar-hd[rgb]

# With all non-GPU features
pip install ign-lidar-hd[all]

# GPU acceleration (requires NVIDIA GPU + CUDA)
# Install base package first, then add GPU support separately:
pip install ign-lidar-hd
pip install cupy-cuda11x  # For CUDA 11.x
# OR
pip install cupy-cuda12x  # For CUDA 12.x

# Advanced GPU with RAPIDS cuML (best performance, conda recommended)
pip install ign-lidar-hd
pip install cupy-cuda12x  # Match your CUDA version
conda install -c rapidsai -c conda-forge -c nvidia cuml
# Or via pip (may require more configuration):
# pip install cuml-cu11  # For CUDA 11.x
# pip install cuml-cu12  # For CUDA 12.x
```

**GPU Requirements** (optional):

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0 or higher
- CuPy package matching your CUDA version
- Optional: RAPIDS cuML for advanced GPU-accelerated algorithms
- Expected speedup: 5-6x faster than CPU (CuPy), up to 10x with RAPIDS

📖 **GPU Documentation:**

- 📘 [Complete GPU Guide](website/docs/gpu-guide.md) - Full documentation
- 🚀 [Quick Start](GPU_QUICKSTART.md) - Get started in 30 seconds
- 📊 [Performance Benchmarks](GPU_COMPLETE.md#performance-expectations) - Expected speedups

### Basic Usage

```python
from ign_lidar import LiDARProcessor

# Initialize processor
processor = LiDARProcessor(lod_level="LOD2")

# Process a single tile
patches = processor.process_tile("data.laz", "output/")

# Process multiple files
patches = processor.process_directory("data/", "output/", num_workers=4)
```

### Command Line Interface

```bash
# Download tiles
ign-lidar-hd download --bbox -2.0,47.0,-1.0,48.0 --output tiles/ --max-tiles 10

# Enrich LAZ files with geometric features
ign-lidar-hd enrich --input-dir tiles/ --output enriched/ --num-workers 4

# Enrich with geometric features
ign-lidar-hd enrich --input-dir tiles/ --output enriched/

# Enrich with GPU acceleration (requires CuPy)
# Automatically falls back to CPU if GPU unavailable
ign-lidar-hd enrich --input-dir tiles/ --output enriched/ --use-gpu

# Enrich with RGB augmentation from IGN orthophotos
ign-lidar-hd enrich --input-dir tiles/ --output enriched/ --add-rgb --rgb-cache-dir cache/

# Create training patches
ign-lidar-hd patch --input-dir enriched/ --output patches/ --lod-level LOD2

# 🆕 Run complete workflow with YAML configuration
ign-lidar-hd pipeline config.yaml
```

### 🆕 Pipeline Configuration (Recommended)

Use YAML configuration files for reproducible workflows:

```bash
# Create example configuration
ign-lidar-hd pipeline my_config.yaml --create-example full

# Edit configuration (my_config.yaml)
# Then run complete pipeline
ign-lidar-hd pipeline my_config.yaml
```

**Example YAML configuration:**

```yaml
global:
  num_workers: 4

download:
  bbox: "2.3, 48.8, 2.4, 48.9"
  output: "data/raw"
  max_tiles: 10

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "building"
  add_rgb: true
  rgb_cache_dir: "cache/orthophotos"
  use_gpu: true

patch:
  input_dir: "data/enriched"
  output: "data/patches"
  lod_level: "LOD2"
  num_points: 16384
  augment: true
```

**Benefits:**

- ✅ **Reproducible** - Version control your workflows
- ✅ **Declarative** - Define what you want, not how
- ✅ **Flexible** - Run only the stages you need
- ✅ **Shareable** - Easy team collaboration

## 📋 Key Features

### 🏗️ **Core Processing Capabilities**

- **LiDAR-only processing**: Pure geometric analysis without RGB dependencies
- **RGB augmentation**: Optional color enrichment from IGN BD ORTHO® orthophotos
- **Multi-level classification**: Support for LOD2 (15 classes) and LOD3 (30+ classes)
- **Rich feature extraction**: Surface normals, curvature, planarity, verticality, local density
- **Architectural style inference**: Automatic building style classification
- **Patch-based processing**: Configurable 150m × 150m patches with overlap control

### ⚡ **Performance & Optimization**

- **GPU acceleration**: CUDA-accelerated feature computation with automatic CPU fallback
- **Parallel processing**: Multi-worker support with automatic CPU core detection
- **Memory optimization**: Chunked processing for large datasets
- **Smart skip detection**: ⏭️ Automatically skip existing files and resume interrupted workflows
- **Batch operations**: Process hundreds of tiles efficiently
- **🆕 Improved augmentation**: Features computed on augmented geometry for consistency

### 🔧 **Workflow Automation**

- **Pipeline configuration**: 🆕 YAML-based declarative workflows for reproducibility
- **Integrated downloader**: IGN WFS tile discovery and batch downloading
- **Format flexibility**: Choose between LAZ 1.4 (full features) or QGIS-compatible output
- **🆕 Enhanced augmentation**: Geometric transformations applied before feature computation for better data quality
- **Unified CLI**: Single `ign-lidar-hd` command with intuitive subcommands
- **Idempotent operations**: Safe to restart - never reprocesses existing data

### 🌍 **Geographic Intelligence**

- **Strategic locations**: Pre-configured urban, coastal, and rural area processing
- **Bounding box filtering**: Spatial subsetting for targeted analysis
- **Coordinate system handling**: Automatic Lambert93 to WGS84 transformations
- **Tile management**: Curated collection of 50+ test tiles across France

## 🏗️ Library Architecture

### 🎯 **Component Architecture**

```mermaid
graph TB
    subgraph "Core Processing"
        P[processor.py<br/>🔧 Main Engine]
        F[features.py<br/>⚡ Feature Extraction]
        GPU[features_gpu.py<br/>🖥️ GPU Acceleration]
    end

    subgraph "Data Management"
        D[downloader.py<br/>📥 IGN WFS Integration]
        TL[tile_list.py<br/>📂 Tile Management]
        SL[strategic_locations.py<br/>�️ Geographic Zones]
        MD[metadata.py<br/>📊 Dataset Metadata]
    end

    subgraph "Classification & Styles"
        C[classes.py<br/>🏢 LOD2/LOD3 Schemas]
        AS[architectural_styles.py<br/>🎨 Style Inference]
    end

    subgraph "Integration & Config"
        CLI[cli.py<br/>🖱️ Command Interface]
        CFG[config.py<br/>⚙️ Configuration]
        QGIS[qgis_converter.py<br/>🔄 QGIS Compatibility]
        U[utils.py<br/>🛠️ Core Utilities]
    end

    CLI --> P
    CLI --> D
    P --> F
    P --> GPU
    P --> C
    F --> AS
    D --> TL
    D --> SL
    P --> MD

    style P fill:#e3f2fd
    style F fill:#e8f5e8
    style D fill:#fff3e0
    style CLI fill:#f3e5f5
```

### 📋 **Module Responsibilities**

| Module                       | Purpose                | Key Features                                               |
| ---------------------------- | ---------------------- | ---------------------------------------------------------- |
| 🔧 `processor.py`            | Main processing engine | Patch creation, LOD classification, workflow orchestration |
| 📥 `downloader.py`           | IGN WFS integration    | Tile discovery, batch download, smart skip detection       |
| ⚡ `features.py`             | Feature extraction     | Normals, curvature, geometric properties                   |
| �️ `features_gpu.py`         | GPU acceleration       | CUDA-optimized feature computation                         |
| 🏢 `classes.py`              | Classification schemas | LOD2/LOD3 building taxonomies                              |
| 🎨 `architectural_styles.py` | Style inference        | Building architecture classification                       |

### 🔄 **Example Workflows**

```text
examples/
├── 🚀 basic_usage.py           # Getting started
├── 🏙️ example_urban_simple.py  # Urban processing
├── ⚡ parallel_processing_example.py # Performance
├── 🔄 full_workflow_example.py # End-to-end pipeline
├── 🎨 multistyle_processing.py # Architecture analysis
├── 🧠 pytorch_dataloader.py    # ML integration
├── 🆕 pipeline_example.py      # YAML pipeline usage
├── 🆕 enrich_with_rgb.py       # RGB augmentation
└── workflows/               # Production pipelines

config_examples/
├── 🆕 pipeline_full.yaml       # Complete workflow
├── 🆕 pipeline_enrich.yaml     # Enrich-only
└── 🆕 pipeline_patch.yaml      # Patch-only
```

## ⚙️ CLI Commands

The package provides a unified `ign-lidar-hd` command with four subcommands:

### 🔗 **CLI Workflow Chain**

```mermaid
sequenceDiagram
    participant User
    participant CLI as ign-lidar-hd
    participant D as Downloader
    participant E as Enricher
    participant P as Processor

    User->>CLI: download --bbox ...
    CLI->>D: Initialize downloader
    D->>D: Fetch available tiles
    D->>D: Smart skip check
    D-->>CLI: Downloaded tiles
    CLI-->>User: ✓ Tiles ready

    User->>CLI: enrich --input-dir ...
    CLI->>E: Initialize enricher
    E->>E: Compute geometric features
    E->>E: Optional RGB augmentation
    E->>E: GPU/CPU processing
    E-->>CLI: Enriched LAZ files
    CLI-->>User: ✓ Features computed

    User->>CLI: patch --input-dir ...
    CLI->>P: Initialize processor
    P->>P: Create training patches
    P->>P: Apply augmentations
    P-->>CLI: ML-ready dataset
    CLI-->>User: ✓ Dataset ready

    Note over User,CLI: 🆕 Or use pipeline command
    User->>CLI: pipeline config.yaml
    CLI->>CLI: Load YAML config
    CLI->>D: Execute download stage
    CLI->>E: Execute enrich stage
    CLI->>P: Execute patch stage
    CLI-->>User: ✓ Complete workflow
```

### 🆕 Pipeline Command (Recommended)

Execute complete workflows using YAML configuration:

```bash
# Create example configuration
ign-lidar-hd pipeline my_config.yaml --create-example full

# Run configured pipeline
ign-lidar-hd pipeline my_config.yaml
```

See [Pipeline Configuration Guide](config_examples/) for detailed examples.

### Download Command

Download LiDAR tiles from IGN:

```bash
ign-lidar-hd download \
  --bbox lon_min,lat_min,lon_max,lat_max \
  --output tiles/ \
  --max-tiles 50
```

### Enrich Command

Enrich LAZ files with geometric features and optional RGB:

```bash
# CPU version (automatically skips existing enriched files)
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --num-workers 4 \
  --k-neighbors 10

# 🆕 With RGB augmentation from IGN orthophotos
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --add-rgb \
  --rgb-cache-dir cache/orthophotos

# Force re-enrichment (ignore existing files)
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --force

# GPU version (requires CUDA)
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --use-gpu
```

> 💡 **Smart Skip**: By default, the enrich command skips files that have already been enriched, making it safe to resume interrupted operations.

### Patch Command

Create training patches from enriched LAZ files:

```bash
# Automatically skips tiles with existing patches
ign-lidar-hd patch \
  --input-dir enriched/ \
  --output patches/ \
  --lod-level LOD2 \
  --patch-size 150.0 \
  --num-workers 4 \
  --num-augmentations 3

# Force reprocessing (ignore existing patches)
ign-lidar-hd patch \
  --input-dir enriched/ \
  --output patches/ \
  --force
```

> 💡 **Smart Skip**: The patch command automatically detects existing patches and skips reprocessing, allowing you to resume interrupted batch jobs.

## 🔧 Configuration

### LOD Levels

- **LOD2**: Simplified building models (15 classes)
- **LOD3**: Detailed building models (30 classes)

### Processing Options

```python
processor = LiDARProcessor(
    lod_level="LOD2",           # LOD2 or LOD3
    augment=True,               # 🆕 Enable enhanced augmentation
    num_augmentations=3,        # Augmentations per tile (not per patch!)
    patch_size=150.0,          # Patch size in meters
    patch_overlap=0.1,         # 10% overlap
    bbox=[xmin, ymin, xmax, ymax]  # Spatial filter
)
```

> **🆕 v1.6.0**: Data augmentation now happens during the ENRICH phase (before feature computation) instead of the PATCH phase. This ensures geometric features (normals, curvature, planarity) are computed on augmented geometry for better feature-geometry consistency and improved model training quality. See `AUGMENTATION_IMPROVEMENT.md` for details.

## 📊 Output Format

### 📁 **Data Structure Overview**

```mermaid
graph TB
    subgraph "Raw Input"
        LAZ[LAZ Point Cloud<br/>XYZ + Intensity<br/>Classification]
    end

    subgraph "Enriched Data"
        ELAZ[Enhanced LAZ<br/>+ 30 Features<br/>+ Building Labels]
    end

    subgraph "ML Dataset"
        NPZ[NPZ Patches<br/>16K points each<br/>Ready for Training]
    end

    subgraph "NPZ Contents"
        COORD[Coordinates<br/>X, Y, Z]
        GEOM[Geometric Features<br/>Normals, Curvature]
        SEMANTIC[Semantic Features<br/>Planarity, Verticality]
        META[Metadata<br/>Intensity, Return#]
        LABELS[Building Labels<br/>LOD2/LOD3 Classes]
    end

    LAZ --> ELAZ
    ELAZ --> NPZ
    NPZ --> COORD
    NPZ --> GEOM
    NPZ --> SEMANTIC
    NPZ --> META
    NPZ --> LABELS

    style LAZ fill:#ffebee
    style ELAZ fill:#e3f2fd
    style NPZ fill:#e8f5e8
```

### 🔢 **NPZ File Structure**

Each patch is saved as an NPZ file containing:

```python
{
    'points': np.ndarray,          # [N, 3] XYZ coordinates
    'normals': np.ndarray,         # [N, 3] surface normals
    'curvature': np.ndarray,       # [N] principal curvature
    'intensity': np.ndarray,       # [N] normalized intensity
    'return_number': np.ndarray,   # [N] return number
    'height': np.ndarray,          # [N] height above ground
    'planarity': np.ndarray,       # [N] planarity measure
    'verticality': np.ndarray,     # [N] verticality measure
    'horizontality': np.ndarray,   # [N] horizontality measure
    'density': np.ndarray,         # [N] local point density
    'labels': np.ndarray,          # [N] building class labels
}
```

### 📏 **Data Dimensions**

| Component  | Shape   | Data Type | Description                |
| ---------- | ------- | --------- | -------------------------- |
| `points`   | [N, 3]  | `float32` | 3D coordinates (X, Y, Z)   |
| `normals`  | [N, 3]  | `float32` | Surface normal vectors     |
| `features` | [N, 27] | `float32` | Geometric feature matrix   |
| `labels`   | [N]     | `uint8`   | Building component classes |
| `metadata` | [4]     | `object`  | Patch info (bbox, tile_id) |

> **📦 Typical patch**: 16,384 points, ~2.5MB compressed, ~8MB in memory

## 🌍 Batch Download

```python
from ign_lidar import IGNLiDARDownloader

# Initialize downloader
downloader = IGNLiDARDownloader("downloads/")

# Download tiles by bounding box (WGS84)
tiles = downloader.download_by_bbox(
    bbox=(-2.0, 47.0, -1.0, 48.0),  # West France
    max_tiles=10
)

# Download specific tiles
tile_names = ["LHD_FXX_0186_6834_PTS_C_LAMB93_IGN69"]
downloader.download_tiles(tile_names)
```

## 📝 Examples

### Urban Processing

```python
# High-detail urban processing
processor = LiDARProcessor(lod_level="LOD3", num_augmentations=5)
patches = processor.process_tile("urban_area.laz", "output/urban/")
```

### Rural Processing

```python
# Simplified rural processing
processor = LiDARProcessor(lod_level="LOD2", num_augmentations=2)
patches = processor.process_tile("rural_area.laz", "output/rural/")
```

### Batch Processing

```python
from ign_lidar import WORKING_TILES, get_tiles_by_environment

# Get coastal tiles
coastal_tiles = get_tiles_by_environment("coastal")

# Process all coastal areas
for tile_info in coastal_tiles:
    patches = processor.process_tile(
        f"data/{tile_info['tile_name']}.laz",
        f"output/coastal/{tile_info['tile_name']}/"
    )
```

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/your-username/ign-lidar-hd-downloader
cd ign-lidar-hd-downloader
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black ign_lidar/
flake8 ign_lidar/
```

## 📚 Documentation & Resources

### 📖 **Complete Documentation Hub**

For comprehensive documentation, see the **[Documentation Hub](docs/README.md)**:

- **[📖 User Guides](docs/guides/)** - Quick start guides, QGIS integration, troubleshooting
- **[⚡ Features](docs/features/)** - Smart skip detection, format preferences, workflow optimization
- **[🔧 Technical Reference](docs/reference/)** - Memory optimization, performance tuning
- **[📦 Archive](docs/archive/)** - Bug fixes history, release notes, migration guides

### 🚀 **Essential Quick Links**

- **[🎯 Quick Reference Card](QUICK_REFERENCE.md)** - Fast reference for all commands
- **[⚡ Smart Skip Features](docs/features/SMART_SKIP_SUMMARY.md)** - Resume workflows efficiently
- **[🗺️ QGIS Integration](docs/guides/QUICK_START_QGIS.md)** - GIS compatibility guide
- **[⚙️ Memory Optimization](docs/reference/MEMORY_OPTIMIZATION.md)** - Performance tuning
- **[📋 Output Formats](docs/features/OUTPUT_FORMAT_PREFERENCES.md)** - LAZ 1.4 vs QGIS formats

### 💡 **Examples & Workflows**

- **[Basic Usage](examples/basic_usage.py)** - Simple processing examples
- **[Urban Processing](examples/example_urban_simple.py)** - City-specific workflows
- **[Parallel Processing](examples/parallel_processing_example.py)** - Multi-worker optimization
- **[Full Workflow](examples/full_workflow_example.py)** - End-to-end pipeline
- **[🆕 Pipeline Configuration](examples/pipeline_example.py)** - YAML-based workflows
- **[🆕 RGB Augmentation](examples/enrich_with_rgb.py)** - Orthophoto integration
- **[PyTorch Integration](examples/pytorch_dataloader.py)** - ML training setup

### 🚀 Coming Soon: Interactive Documentation

We're working on a comprehensive [Docusaurus documentation site](DOCUSAURUS_PLAN.md) that will include:

- 🌐 Multi-language support (English & French)
- 🔍 Full-text search
- 📱 Mobile-responsive design
- 📖 Interactive tutorials
- 🔗 Auto-generated API reference
- 💡 Live code examples

See the [Docusaurus Plan](DOCUSAURUS_PLAN.md) for details.

## 📚 API Reference

### Core Classes

- **`LiDARProcessor`**: Main processing engine
- **`IGNLiDARDownloader`**: Batch download functionality
- **`LOD2_CLASSES`**, **`LOD3_CLASSES`**: Classification taxonomies

### Utility Functions

- **`compute_normals()`**: Surface normal computation
- **`compute_curvature()`**: Principal curvature calculation
- **`extract_geometric_features()`**: Comprehensive feature extraction
- **`get_tiles_by_environment()`**: Filter tiles by environment type

## 🔗 Requirements

- Python 3.8+
- NumPy >= 1.21.0
- laspy >= 2.3.0
- scikit-learn >= 1.0.0
- tqdm >= 4.60.0
- requests >= 2.25.0
- PyYAML >= 6.0 (for pipeline configuration)
- Pillow >= 9.0.0 (for RGB augmentation)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please use the [GitHub Issues](https://github.com/your-username/ign-lidar-hd-downloader/issues) page.
