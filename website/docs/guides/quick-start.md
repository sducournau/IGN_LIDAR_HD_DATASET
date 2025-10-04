---
sidebar_position: 2
title: Quick Start Guide
description: Get up and running with IGN LiDAR HD in 5 minutes
keywords: [quickstart, workflow, tutorial, examples]
---

# Quick Start Guide

Get up and running with IGN LiDAR HD Processing Library in 5 minutes! This guide walks you through your first complete workflow from download to analysis.

:::info Prerequisites
Make sure you have IGN LiDAR HD installed. If not, see the [Installation Guide](../installation/quick-start.md) first.
:::

---

## 🚀 Your First Workflow

Let's process LiDAR data in 3 simple steps: Download → Enrich → Create Patches

### Step 1: Download LiDAR Tiles

Download tiles from IGN servers using geographic coordinates:

```bash
ign-lidar-hd download \
  --bbox 2.3,48.8,2.4,48.9 \
  --output data/raw \
  --max-tiles 5
```

**What this does:**

- Queries IGN WFS service for available tiles
- Downloads up to 5 tiles in the specified bounding box (Paris area)
- Saves LAZ files to `data/raw/`
- Skips already downloaded tiles

:::info Bounding Box Format
`--bbox lon_min,lat_min,lon_max,lat_max` (WGS84 coordinates)

Example areas:

- Paris: `2.3,48.8,2.4,48.9`
- Marseille: `5.3,43.2,5.4,43.3`
- Lyon: `4.8,45.7,4.9,45.8`
  :::

### Step 2: Enrich with Features

Add geometric features and optional RGB colors:

```bash
ign-lidar-hd enrich \
  --input-dir data/raw \
  --output data/enriched \
  --mode full \
  --use-gpu
```

**What this does:**

- Computes geometric features (normals, curvature, planarity)
- Adds all additional features in 'full' mode
- Uses GPU acceleration if available (falls back to CPU)
- **Creates augmented versions** (enabled by default: 1 original + 3 augmented per tile)
- Skips already enriched tiles

:::info Data Augmentation (Enabled by Default)
By default, the enrich command creates **4 versions** of each tile:

- `tile_name.laz` (original)
- `tile_name_aug1.laz` (augmented version 1)
- `tile_name_aug2.laz` (augmented version 2)
- `tile_name_aug3.laz` (augmented version 3)

Each augmented version applies random rotation, jitter, scaling, and dropout before feature computation.

To disable: add `--no-augment`  
To change count: add `--num-augmentations N`
:::

**Features Added:**

- Surface normals (3D vectors)
- Curvature (principal curvature)
- Planarity, verticality, horizontality
- Local point density
- Building classification labels

:::tip Add RGB Colors
Add `--add-rgb --rgb-cache-dir cache/` to enrich with colors from IGN orthophotos!
:::

### Step 3: Create Training Patches

Generate machine learning-ready patches:

```bash
# Note: Augmentation happens during ENRICH phase (enabled by default)
# The tiles in data/enriched already contain augmented versions
ign-lidar-hd patch \
  --input-dir data/enriched \
  --output data/patches \
  --lod-level LOD2 \
  --num-points 16384
```

**What this does:**

- Creates 150m × 150m patches from enriched tiles
- Samples 16,384 points per patch
- Processes both original and augmented tiles (created during enrich)
- Saves as compressed NPZ files

**Output Structure:**

```text
data/patches/
├── tile_0501_6320_patch_0.npz
├── tile_0501_6320_patch_1.npz
├── tile_0501_6320_patch_2.npz
└── ...
```

Each NPZ file contains:

- `points`: [N, 3] XYZ coordinates
- `normals`: [N, 3] surface normals
- `features`: [N, 27] geometric features
- `labels`: [N] building class labels

---

## 🎯 Complete Workflow with YAML

For production workflows, use YAML configuration files for reproducibility:

### Create Configuration

```bash
ign-lidar-hd pipeline my_workflow.yaml --create-example full
```

This creates a YAML configuration file. For detailed configuration examples, see [Configuration Examples](../reference/config-examples.md).

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

- 📖 [Features Guide](features/overview.md) - Deep dive into all features
- ⚡ [GPU Guide](gpu/overview.md) - GPU acceleration details
- 🔧 [Configuration Guide](features/pipeline-configuration.md) - Advanced workflows
- 🎨 [RGB Augmentation](features/rgb-augmentation.md) - Color enrichment

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
