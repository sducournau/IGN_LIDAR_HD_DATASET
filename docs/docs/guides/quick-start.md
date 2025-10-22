---
sidebar_position: 2
title: Quick Start Guide
description: Get up and running with IGN LiDAR HD in 5 minutes
keywords: [quickstart, workflow, tutorial, examples]
---

# Quick Start Guide

Get up and running with IGN LiDAR HD Processing Library in 5 minutes! This guide walks you through your first complete workflow from download to analysis.

:::info Prerequisites
Make sure you have IGN LiDAR HD installed. If not, see the [Installation Guide](../installation/quick-start) first.
:::

---

## 🚀 Your First Workflow (v2++ Modern CLI)

Let's process LiDAR data in 2 simple steps with the unified v2++ pipeline: Download → Process

### Step 1: Download LiDAR Tiles

Download tiles from IGN servers using Lambert93 coordinates:

```bash
# Download tiles within 5km radius of Paris center
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 5000 \
  data/raw_tiles
```

**What this does:**

- Queries IGN WFS service for available tiles
- Downloads tiles within 5km of the specified position
- Saves LAZ files to `data/raw_tiles/`
- Skips already downloaded tiles

:::info Coordinate Systems
The `download` command uses **Lambert93 coordinates** (EPSG:2154), not WGS84.

Example positions (Lambert93):

- Paris center: `650000 6860000`
- Marseille: `893000 6250000`
- Lyon: `842000 6518000`

You can also use:

- `--bbox XMIN YMIN XMAX YMAX` for rectangular areas
- `--location paris_center` for predefined locations
  :::

### Step 2: Process to ML-Ready Patches

The v2++ **unified pipeline** handles everything in one command:

```bash
# Complete processing: features + patches in one step
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/patches
```

**What this does:**

- Computes geometric features (normals, curvature, planarity, etc.)
- Creates 150m × 150m training patches
- Samples 16,384 points per patch
- Uses CPU by default (add `processor=gpu` for GPU acceleration)
- Saves as compressed NPZ files

**Output Structure:**

```text
data/patches/
├── LHD_FXX_0649_6863_patch_0000.npz
├── LHD_FXX_0649_6863_patch_0001.npz
├── LHD_FXX_0649_6863_patch_0002.npz
└── ...
```

Each NPZ file contains:

- `points`: [N, 3] XYZ coordinates
- `normals`: [N, 3] surface normals
- `features`: [N, 27+] geometric features
- `labels`: [N] building class labels

---

## 🎯 Advanced Processing Options

### GPU-Accelerated Processing

```bash
# 10-20x faster with GPU
ign-lidar-hd process \
  processor=gpu \
  input_dir=data/raw_tiles \
  output_dir=data/patches
```

### LOD3 Training Dataset

```bash
# Complete LOD3 configuration (5 building classes)
ign-lidar-hd process \
  experiment=config_lod3_training \
  input_dir=data/raw_tiles \
  output_dir=data/lod3_patches
```

**Includes:**

- 32,768 points per patch
- 3x geometric augmentation
- RGB + NIR + NDVI features
- Tile stitching (eliminates edge artifacts)
- Auto-download neighbor tiles

### RGB + Infrared + NDVI

```bash
# Multi-modal features for vegetation analysis
ign-lidar-hd process \
  features=vegetation \
  input_dir=data/raw_tiles \
  output_dir=data/multimodal_patches \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true
```

### Generate Only Enriched LAZ (No Patches)

```bash
# For visualization in QGIS/CloudCompare
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/enriched_laz \
  output=enriched_only
```

### Custom Configuration

```bash
# Fine-tune every parameter
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/custom_patches \
  processor.lod_level=LOD3 \
  processor.num_points=32768 \
  processor.patch_size=200.0 \
  processor.augment=true \
  processor.num_augmentations=5 \
  features.k_neighbors=30 \
  features.use_rgb=true \
  features.sampling_method=fps \
  stitching.enabled=true \
  output.save_enriched_laz=true
```

---

## ✅ Verify Your Dataset

```bash
# Check data quality
ign-lidar-hd verify data/patches

# Detailed statistics
ign-lidar-hd verify data/patches --detailed

# Generate JSON report
ign-lidar-hd verify data/patches --output report.json
```

---

## 🎯 Complete Workflow with YAML

For production workflows, use YAML configuration files for reproducibility:

### Create Configuration

```bash
ign-lidar-hd pipeline my_workflow.yaml --create-example full
```

This creates a YAML configuration file. For detailed configuration examples, see [Configuration Examples](../reference/config-examples).

### Quick Example

input_dir: "data/enriched"
output: "data/patches"
lod_level: "LOD2"
patch_size: 150.0
num_points: 16384
augment: true
num_augmentations: 3

````

### Run Pipeline

```bash
ign-lidar-hd pipeline my_workflow.yaml
````

**Benefits:**

- ✅ Reproducible workflows
- ✅ Version control friendly
- ✅ Easy team collaboration
- ✅ Run only specific stages
- ✅ Clear configuration documentation

---

## 🐍 Python API

For programmatic control, use the Python API:

```python
from ign_lidar import LiDARProcessor

# Initialize processor
processor = LiDARProcessor(
    lod_level="LOD2",
    augment=True,
    num_augmentations=3,
    use_gpu=True
)

# Process a single tile
patches = processor.process_tile(
    input_file="data/raw/tile.laz",
    output_dir="data/patches"
)

print(f"Generated {len(patches)} training patches")

# Or process entire directory
num_patches = processor.process_directory(
    input_dir="data/raw",
    output_dir="data/patches",
    num_workers=4
)

print(f"Total patches generated: {num_patches}")
```

---

## 🎓 Understanding LOD Levels

Choose the right Level of Detail for your task:

### LOD2 (15 Classes)

Simplified building models - good for general classification:

**Classes:**

- Ground, vegetation, road, railway
- Building parts: wall, roof, balcony, window, door
- Urban furniture, power lines, etc.

**Use Cases:**

- Building detection and segmentation
- Urban planning
- 3D city modeling (basic)

```python
processor = LiDARProcessor(lod_level="LOD2")
```

### LOD3 (30+ Classes)

Detailed building models - for architectural analysis:

**Additional Classes:**

- Detailed roof types (flat, gabled, hipped, etc.)
- Architectural elements (columns, cornices, ornaments)
- Building materials
- Precise architectural styles

**Use Cases:**

- Architectural heritage documentation
- Detailed 3D reconstruction
- Building condition assessment

```python
processor = LiDARProcessor(lod_level="LOD3")
```

---

## ⚡ Performance Tips

### 1. Use GPU Acceleration

```bash
# 5-10x faster feature computation
ign-lidar-hd enrich --use-gpu --input-dir tiles/ --output enriched/
```

### 2. Parallel Processing

```bash
# Use multiple CPU cores
ign-lidar-hd enrich --num-workers 8 --input-dir tiles/ --output enriched/
```

### 3. Smart Resumability

All commands automatically skip existing files:

```bash
# Safe to interrupt and resume
ign-lidar-hd enrich --input-dir tiles/ --output enriched/
# Press Ctrl+C anytime
# Run again - continues where it left off
```

### 4. RGB Caching

When using RGB augmentation, cache orthophotos for reuse:

```bash
ign-lidar-hd enrich \
  --add-rgb \
  --rgb-cache-dir cache/orthophotos \
  --input-dir tiles/ \
  --output enriched/
```

---

## 🔍 Verify Your Data

### Check Enriched Files

```python
import laspy

# Load enriched LAZ file
las = laspy.read("data/enriched/tile.laz")

# Check dimensions
print("Available dimensions:", las.point_format.dimension_names)

# Should include:
# - X, Y, Z (coordinates)
# - normal_x, normal_y, normal_z
# - curvature
# - planarity, verticality
# - intensity, return_number
# - RGB (if using --add-rgb)
```

### Check NPZ Patches

```python
import numpy as np

# Load patch
data = np.load("data/patches/tile_patch_0.npz")

# Check contents
print("Keys:", list(data.keys()))
print("Points shape:", data['points'].shape)
print("Labels shape:", data['labels'].shape)

# Verify point count
assert data['points'].shape[0] == 16384  # Default num_points
```

---

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import cupy as cp; print('CUDA available:', cp.is_available())"
```

If CUDA is not available:

- Ensure NVIDIA GPU drivers are installed
- Install correct CuPy version for your CUDA toolkit
- Library automatically falls back to CPU

### Out of Memory

For large tiles (>10M points):

```python
# Reduce patch size or point count
processor = LiDARProcessor(
    patch_size=100.0,      # Smaller patches (default: 150.0)
    num_points=8192,       # Fewer points (default: 16384)
)
```

### Slow Processing

1. Enable GPU acceleration: `--use-gpu`
2. Increase workers: `--num-workers 8`
3. Use 'core' mode instead of 'full': `--mode core`

---

## 📚 Next Steps

### Learn More

- 📖 [Feature Modes](../features/feature-modes.md) - Deep dive into all features
- ⚡ [GPU Guide](../gpu/overview.md) - GPU acceleration details
- 🔧 [Configuration Guide](../features/pipeline-configuration.md) - Advanced workflows
- 🎨 [RGB Augmentation](../features/rgb-augmentation.md) - Color enrichment

### Examples

- [Basic Usage](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/basic_usage.py)
- [Pipeline Configuration](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/pipeline_example.py)
- [GPU Processing](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/processor_gpu_usage.py)
- [RGB Augmentation](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/enrich_with_rgb.py)

### Get Help

- 🐛 [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues) - Report bugs
- 💬 [Discussions](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions) - Ask questions
- 📧 Email: simon.ducournau@gmail.com

---

**Ready to process your first dataset?** 🚀

```bash
# Download and process in one go
ign-lidar-hd download --bbox 2.3,48.8,2.4,48.9 --output raw/ --max-tiles 5
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --use-gpu
ign-lidar-hd patch --input-dir enriched/ --output patches/ --augment
```

Happy processing! 🎉
