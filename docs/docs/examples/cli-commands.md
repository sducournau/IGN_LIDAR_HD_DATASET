---
sidebar_position: 1
title: CLI Command Reference
description: Complete reference for IGN LiDAR HD v2++ command-line interface
keywords: [cli, commands, examples, reference]
---

# CLI Command Reference (v2++)

Complete command-line interface reference for IGN LiDAR HD Processing Library version 2.1.2+.

:::tip What's New in v2++
Version 2.0+ introduced a **modern CLI** combining Hydra's powerful configuration system with intuitive Click commands. All commands support both configuration files and command-line overrides!
:::

---

## 📋 Command Overview

| Command                                   | Purpose                   | Common Use Cases         |
| ----------------------------------------- | ------------------------- | ------------------------ |
| [`download`](#download-command)           | Download tiles from IGN   | Initial data acquisition |
| [`process`](#process-command)             | Main processing pipeline  | Create ML-ready datasets |
| [`verify`](#verify-command)               | Validate data quality     | QA/QC workflows          |
| [`info`](#info-command)                   | Show configuration info   | Debug configurations     |
| [`batch-convert`](#batch-convert-command) | Convert for visualization | QGIS/CloudCompare        |

---

## 🔧 Global Options

Available for all commands:

```bash
ign-lidar-hd <command> --verbose    # Enable detailed logging
ign-lidar-hd <command> --help       # Show command help
```

---

## 📥 Download Command

Download LiDAR tiles from IGN servers.

### Basic Usage

```bash
ign-lidar-hd download [OPTIONS] OUTPUT_DIR
```

### By Position and Radius

Download tiles around a specific location (Lambert93 coordinates):

```bash
# Download tiles within 5km of Paris center
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 5000 \
  data/raw_tiles

# Larger area (10km radius)
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 10000 \
  --max-concurrent 5 \
  data/raw_tiles
```

### By Bounding Box

Download tiles within a bounding box:

```bash
# Define area with Lambert93 coordinates
ign-lidar-hd download \
  --bbox 649000 6859000 651000 6861000 \
  data/raw_tiles
```

### By Strategic Location

Use predefined strategic locations:

```bash
# List available locations
ign-lidar-hd download --list-locations

# Download using location name
ign-lidar-hd download \
  --location paris_center \
  data/raw_tiles

# Other locations: marseille_port, lyon_confluence, etc.
```

### Options

| Option                       | Description                    | Default |
| ---------------------------- | ------------------------------ | ------- |
| `--position X Y`             | Center coordinates (Lambert93) | -       |
| `--radius R`                 | Radius in meters               | 5000    |
| `--bbox XMIN YMIN XMAX YMAX` | Bounding box (Lambert93)       | -       |
| `--location NAME`            | Strategic location name        | -       |
| `--max-concurrent N`         | Parallel downloads             | 3       |
| `--force`                    | Redownload existing files      | false   |
| `--list-locations`           | Show available locations       | -       |

---

## ⚙️ Process Command

Main processing pipeline - transforms raw tiles into ML-ready datasets.

### Basic Usage

```bash
ign-lidar-hd process [HYDRA_OVERRIDES...]
```

:::info Hydra Configuration System
The `process` command uses **Hydra** for configuration. You can:

- Use config files from `ign_lidar/configs/`
- Override any parameter with `key=value` syntax
- Combine presets with custom overrides
  :::

### Minimal Processing

```bash
# Simplest command - uses all defaults
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/patches
```

**Defaults:**

- CPU processing
- LOD2 classification
- 16,384 points per patch
- Full geometric features
- NPZ output format

### GPU-Accelerated Processing

```bash
# Enable GPU for 10-20x speedup
ign-lidar-hd process \
  processor=gpu \
  input_dir=data/raw_tiles \
  output_dir=data/patches
```

### LOD3 Training Dataset

```bash
# Complete LOD3 training configuration
ign-lidar-hd process \
  experiment=config_lod3_training \
  input_dir=data/raw_tiles \
  output_dir=data/lod3_patches
```

**Includes:**

- LOD3 classification (5 classes)
- Hybrid architecture support
- 32,768 points per patch
- 3x augmentation
- RGB + NIR + NDVI features
- Tile stitching with 20m buffer
- Auto-download neighbors

### LOD2 Building Classification

```bash
# Simpler LOD2 classification (2 classes)
ign-lidar-hd process \
  experiment=buildings_lod2 \
  input_dir=data/raw_tiles \
  output_dir=data/lod2_patches
```

### PointNet++ Optimized

```bash
# Optimized for PointNet++ architecture
ign-lidar-hd process \
  experiment=pointnet_training \
  input_dir=data/raw_tiles \
  output_dir=data/pointnet_dataset
```

**Features:**

- Farthest Point Sampling (FPS)
- Normalized coordinates
- Normalized features
- PyTorch format output

### RGB + Infrared + NDVI

```bash
# Multi-modal features for vegetation analysis
ign-lidar-hd process \
  features=vegetation \
  input_dir=data/raw_tiles \
  output_dir=data/vegetation_patches \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true
```

### Boundary-Aware Processing

```bash
# Eliminate edge artifacts with tile stitching
ign-lidar-hd process \
  experiment=boundary_aware_autodownload \
  input_dir=data/raw_tiles \
  output_dir=data/seamless_patches
```

**Features:**

- Automatic neighbor detection
- Auto-download missing neighbors
- 20m buffer for boundary features
- Seamless cross-tile features

### Generate Only Enriched LAZ

```bash
# Skip patch generation, only create enriched LAZ files
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/enriched_laz \
  output=enriched_only
```

**Use cases:**

- Visualization in QGIS/CloudCompare
- Manual inspection
- External processing pipelines

### Custom Configuration

```bash
# Fine-grained control with overrides
ign-lidar-hd process \
  processor=gpu \
  features=full \
  input_dir=data/raw_tiles \
  output_dir=data/custom_patches \
  processor.lod_level=LOD3 \
  processor.num_points=32768 \
  processor.patch_size=200.0 \
  processor.patch_overlap=0.20 \
  processor.augment=true \
  processor.num_augmentations=5 \
  features.k_neighbors=30 \
  features.use_rgb=true \
  features.compute_ndvi=true \
  features.sampling_method=fps \
  features.normalize_xyz=true \
  stitching.enabled=true \
  stitching.buffer_size=15.0 \
  output.format=all \
  output.save_enriched_laz=true
```

### Memory-Constrained Processing

```bash
# For systems with limited RAM
ign-lidar-hd process \
  processor=memory_constrained \
  input_dir=data/raw_tiles \
  output_dir=data/patches
```

**Configuration:**

- 2 workers
- Batch size: 1
- 8,192 points per patch
- Reduced memory footprint

### Fast Prototyping

```bash
# Quick testing with minimal features
ign-lidar-hd process \
  processor=cpu_fast \
  features=minimal \
  input_dir=data/raw_tiles \
  output_dir=data/test_patches
```

### Verify Configuration Before Running

```bash
# Print configuration without processing
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/patches \
  --cfg job
```

### Common Overrides

```bash
# Change number of points
processor.num_points=32768

# Change LOD level
processor.lod_level=LOD3

# Enable/disable GPU
processor.use_gpu=true

# Change patch size
processor.patch_size=200.0

# Enable augmentation
processor.augment=true
processor.num_augmentations=5

# Enable tile stitching
stitching.enabled=true
stitching.buffer_size=15.0

# Change output format
output.format=torch      # Options: npz, torch, hdf5, all
output.save_enriched_laz=true

# RGB and infrared
features.use_rgb=true
features.use_infrared=true
features.compute_ndvi=true

# Feature computation
features.k_neighbors=30
features.sampling_method=fps  # Options: random, fps

# Normalization
features.normalize_xyz=true
features.normalize_features=true

# Preprocessing
preprocess=aggressive    # Options: default, aggressive, minimal
```

---

## ✅ Verify Command

Validate LiDAR data quality and features.

### Basic Usage

```bash
ign-lidar-hd verify [OPTIONS] INPUT_PATH
```

### Verify Single File

```bash
# Check a single LAZ/NPZ file
ign-lidar-hd verify data/patches/tile_0501_6320_patch_0.npz

# With detailed statistics
ign-lidar-hd verify \
  --detailed \
  data/patches/tile_0501_6320_patch_0.npz
```

### Verify Directory

```bash
# Verify all files in a directory
ign-lidar-hd verify data/patches

# Generate JSON report
ign-lidar-hd verify \
  data/patches \
  --output verification_report.json

# Skip RGB/NIR checks (faster)
ign-lidar-hd verify \
  --features-only \
  data/patches
```

### Options

| Option            | Description              | Default |
| ----------------- | ------------------------ | ------- |
| `--output FILE`   | Save report to JSON file | -       |
| `--detailed`      | Show detailed statistics | false   |
| `--features-only` | Skip RGB/NIR validation  | false   |

### What It Checks

- ✅ File integrity
- ✅ Expected features present
- ✅ Data ranges valid
- ✅ No NaN/Inf values
- ✅ Point counts
- ✅ Feature dimensions
- ✅ RGB/NIR channels (if enabled)

---

## ℹ️ Info Command

Display configuration and preset information.

### Basic Usage

```bash
ign-lidar-hd info
```

### Output Example

```text
IGN LiDAR HD v2.1.2
Configuration Directory: /path/to/ign_lidar/configs

Available Presets:
  Processors:
    - default (CPU, LOD2, 16K points)
    - gpu (GPU acceleration)
    - cpu_fast (Quick testing)
    - memory_constrained (Low memory)

  Features:
    - full (All geometric features + RGB)
    - minimal (Basic features only)
    - pointnet (PointNet++ optimized)
    - vegetation (RGB + NIR + NDVI)

  Experiments:
    - config_lod3_training (Complete LOD3 setup)
    - buildings_lod2 (Simple building classification)
    - pointnet_training (PointNet++ dataset)
    - boundary_aware_autodownload (Seamless tiles)
```

---

## 🔄 Batch-Convert Command

Convert patches to QGIS-compatible or other formats.

### Basic Usage

```bash
ign-lidar-hd batch-convert [OPTIONS] INPUT_DIR
```

### Convert for QGIS

```bash
# Simplify LAZ files for QGIS visualization
ign-lidar-hd batch-convert \
  data/patches \
  --output data/qgis \
  --format qgis \
  --max-points 100000
```

### Convert to LAS

```bash
# Convert LAZ to LAS format
ign-lidar-hd batch-convert \
  data/patches \
  --output data/las_files \
  --format las
```

### Convert to CSV

```bash
# Export to CSV for analysis
ign-lidar-hd batch-convert \
  data/patches \
  --output data/csv_files \
  --format csv \
  --max-points 50000
```

### Parallel Processing

```bash
# Process 20 files in parallel
ign-lidar-hd batch-convert \
  data/patches \
  --output data/converted \
  --batch-size 20 \
  --force
```

### Options

| Option           | Description                    | Default              |
| ---------------- | ------------------------------ | -------------------- |
| `--output DIR`   | Output directory               | `{input}_converted/` |
| `--format FMT`   | Output format (las, csv, qgis) | las                  |
| `--batch-size N` | Parallel workers               | 10                   |
| `--max-points N` | Max points per file            | 100000               |
| `--force`        | Overwrite existing files       | false                |

---

## 📝 Real-World Workflows

### Workflow 1: Complete Urban Dataset

```bash
# 1. Download urban area
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 5000 \
  data/raw_tiles

# 2. Process with GPU and full features
ign-lidar-hd process \
  experiment=config_lod3_training \
  processor.use_gpu=true \
  input_dir=data/raw_tiles \
  output_dir=data/lod3_dataset

# 3. Verify quality
ign-lidar-hd verify data/lod3_dataset --detailed

# 4. Convert sample for visualization
ign-lidar-hd batch-convert \
  data/lod3_dataset \
  --output data/qgis_preview \
  --format qgis
```

### Workflow 2: Quick Testing

```bash
# 1. Download small area
ign-lidar-hd download \
  --position 650000 6860000 \
  --radius 2000 \
  data/test_tiles

# 2. Fast processing
ign-lidar-hd process \
  processor=cpu_fast \
  features=minimal \
  input_dir=data/test_tiles \
  output_dir=data/test_patches

# 3. Quick check
ign-lidar-hd verify data/test_patches
```

### Workflow 3: Vegetation Analysis

```bash
# Process with multi-modal features
ign-lidar-hd process \
  features=vegetation \
  input_dir=data/forest_tiles \
  output_dir=data/vegetation_dataset \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  processor.lod_level=LOD2
```

### Workflow 4: Boundary-Aware Dataset

```bash
# Seamless cross-tile processing
ign-lidar-hd process \
  experiment=boundary_aware_autodownload \
  input_dir=data/tiles \
  output_dir=data/seamless_dataset \
  stitching.buffer_size=20.0
```

---

## 🔍 Troubleshooting

### Check Configuration

```bash
# View full configuration
ign-lidar-hd process input_dir=. output_dir=. --cfg job
```

### Enable Verbose Logging

```bash
# See detailed processing logs
ign-lidar-hd process \
  --verbose \
  input_dir=data/tiles \
  output_dir=data/output \
  log_level=DEBUG
```

### Test Single Tile

```bash
# Process just one tile for testing
ign-lidar-hd process \
  input_dir=data/single_tile \
  output_dir=data/test \
  processor.num_workers=1
```

### Memory Issues

```bash
# Reduce memory usage
ign-lidar-hd process \
  processor=memory_constrained \
  processor.num_points=8192 \
  processor.batch_size=1 \
  input_dir=data/tiles \
  output_dir=data/output
```

---

## 📚 Additional Resources

- [Configuration Examples](./config-reference) - Detailed config file examples
- [Hydra Commands](./hydra-commands) - Advanced Hydra usage
- [Quick Start Guide](../guides/quick-start) - Beginner tutorials
- [API Reference](../api/processor) - Python API documentation

---

## 💡 Tips

:::tip Configuration Best Practices

1. **Start with presets**: Use `experiment=` configs as starting points
2. **Override incrementally**: Test one change at a time
3. **Verify configs**: Use `--cfg job` to check before processing
4. **Save configs**: Document working configurations for reproducibility
   :::

:::tip Performance Optimization

1. **GPU acceleration**: 10-20x faster with `processor=gpu`
2. **Parallel downloads**: Use `--max-concurrent 5` for downloads
3. **Batch processing**: Increase `processor.num_workers` for CPU
4. **Memory tuning**: Adjust `processor.batch_size` based on RAM
   :::

:::tip Output Management

1. **Format options**: Use `output.format=all` to generate multiple formats
2. **Save enriched LAZ**: Enable `output.save_enriched_laz=true` for visualization
3. **Metadata**: Keep `output.save_metadata=true` for traceability
4. **Compression**: Use `output.compression=gzip` to save disk space
   :::
