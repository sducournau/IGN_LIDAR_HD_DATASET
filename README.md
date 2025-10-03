# IGN LiDAR HD Processing Library

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for processing IGN (Institut National de l'Information Géographique et Forestière) LiDAR HD data into machine learning-ready datasets for Building Level of Detail (LOD) classification tasks.

## 🚀 Quick Start

### Installation

```bash
pip install ign-lidar-hd
```

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
ign-lidar-process download --bbox -2.0,47.0,-1.0,48.0 --output tiles/ --max-tiles 10

# Enrich LAZ files with geometric features
ign-lidar-process enrich --input-dir tiles/ --output enriched/ --num-workers 4

# Enrich with GPU acceleration
ign-lidar-process enrich --input-dir tiles/ --output enriched/ --use-gpu

# Create training patches
ign-lidar-process process --input-dir enriched/ --output patches/ --lod-level LOD2

# Full pipeline example
ign-lidar-process download --bbox -2.0,47.0,-1.0,48.0 --output tiles/
ign-lidar-process enrich --input-dir tiles/ --output enriched/ --num-workers 4
ign-lidar-process process --input-dir enriched/ --output patches/ --lod-level LOD2
```

## 📋 Features

- **LiDAR-only processing**: No RGB dependency, works purely with geometric data
- **Multi-level classification**: Supports LOD2 (15 classes) and LOD3 (30 classes)
- **Rich feature extraction**: Normals, curvature, planarity, verticality, density
- **Patch-based processing**: 150m × 150m patches with configurable overlap
- **Data augmentation**: Rotation, jitter, scaling, and dropout
- **Parallel processing**: Multi-worker support for large datasets
- **Batch downloading**: Integrated IGN WFS tile discovery and download
- **GPU acceleration**: Optional GPU support for faster feature computation
- **Unified CLI**: Single command-line tool with subcommands for complete workflow
- **Smart skip detection**: ⏭️ Automatically skip existing downloads, enriched files, and patches
- **Idempotent operations**: Resume interrupted workflows without re-processing existing data
- **Format preferences**: Choose between full-feature LAZ 1.4 or QGIS-compatible format

## �️ CLI Commands

The package provides a unified `ign-lidar-process` command with three subcommands:

### Download Command

Download LiDAR tiles from IGN:

```bash
ign-lidar-process download \
  --bbox lon_min,lat_min,lon_max,lat_max \
  --output tiles/ \
  --max-tiles 50
```

### Enrich Command

Enrich LAZ files with geometric features:

```bash
# CPU version (automatically skips existing enriched files)
ign-lidar-process enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --num-workers 4 \
  --k-neighbors 10

# Force re-enrichment (ignore existing files)
ign-lidar-process enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --force

# GPU version (requires CUDA)
ign-lidar-process enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --use-gpu
```

> 💡 **Smart Skip**: By default, the enrich command skips files that have already been enriched, making it safe to resume interrupted operations.

### Process Command

Create training patches from enriched LAZ files:

```bash
# Automatically skips tiles with existing patches
ign-lidar-process process \
  --input-dir enriched/ \
  --output patches/ \
  --lod-level LOD2 \
  --patch-size 150.0 \
  --num-workers 4 \
  --num-augmentations 3

# Force reprocessing (ignore existing patches)
ign-lidar-process process \
  --input-dir enriched/ \
  --output patches/ \
  --force
```

> 💡 **Smart Skip**: The process command automatically detects existing patches and skips reprocessing, allowing you to resume interrupted batch jobs.

## �🔧 Configuration

### LOD Levels

- **LOD2**: Simplified building models (15 classes)
- **LOD3**: Detailed building models (30 classes)

### Processing Options

```python
processor = LiDARProcessor(
    lod_level="LOD2",           # LOD2 or LOD3
    augment=True,               # Enable augmentation
    num_augmentations=3,        # Augmentations per patch
    patch_size=150.0,          # Patch size in meters
    patch_overlap=0.1,         # 10% overlap
    bbox=[xmin, ymin, xmax, ymax]  # Spatial filter
)
```

## 📊 Output Format

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

## 📚 Documentation

### Current Documentation

For comprehensive documentation, see the [Documentation Hub](docs/README.md):

- 📖 **[User Guides](docs/guides/)** - Quick start guides and tutorials
- ⚡ **[Features](docs/features/)** - Smart skip detection, format preferences
- 🔧 **[Technical Reference](docs/reference/)** - Memory optimization, API details
- 📦 **[Archive](docs/archive/)** - Historical bug fixes and release notes

### Quick Links

- **[Enrichment Guide](ENRICHMENT_GUIDE.md)** - Complete guide to LAZ enrichment with examples
- **[Smart Skip Features](docs/features/SMART_SKIP_SUMMARY.md)** - Avoid redundant downloads, enrichment, and processing
- **[QGIS Quick Start](docs/guides/QUICK_START_QGIS.md)** - Getting started with QGIS integration
- **[Memory Optimization](docs/reference/MEMORY_OPTIMIZATION.md)** - Performance tuning guide
- **[Output Format Preferences](docs/features/OUTPUT_FORMAT_PREFERENCES.md)** - LAZ 1.4 vs QGIS formats
- **[Quick Reference Card](QUICK_REFERENCE.md)** - Fast reference for common commands

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

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please use the [GitHub Issues](https://github.com/your-username/ign-lidar-hd-downloader/issues) page.
