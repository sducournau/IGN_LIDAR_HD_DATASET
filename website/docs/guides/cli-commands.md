---
sidebar_position: 2
title: CLI Commands
description: Complete reference for command-line interface commands
keywords: [cli, commands, reference, terminal]
---

# CLI Commands Reference

Complete reference for all command-line interface commands in the IGN LiDAR HD Processing Library.

## Command Structure

All commands follow this structure:

```bash
ign-lidar-hd COMMAND [options]

# Or using the installed command (if in PATH)
ign-lidar-hd COMMAND [options]
```

## Available Commands

- [`download`](#download) - Download LiDAR tiles from IGN servers
- [`enrich`](#enrich) - Add building features to LAZ files
- [`patch`](#patch) - Extract patches from enriched tiles (renamed from `process`)
- [`process`](#process-deprecated) - ⚠️ Deprecated alias for `patch`

## download

Download LiDAR tiles for a specified area.

### Syntax

```bash
ign-lidar-hd download \
  --bbox MIN_LON,MIN_LAT,MAX_LON,MAX_LAT \
  --output OUTPUT_DIR \
  [--max-tiles MAX_TILES] \
  [--force]
```

### Parameters

| Parameter     | Type                    | Required | Description                                     |
| ------------- | ----------------------- | -------- | ----------------------------------------------- |
| `--bbox`      | float,float,float,float | Yes      | Bounding box as min_lon,min_lat,max_lon,max_lat |
| `--output`    | string                  | Yes      | Output directory for downloaded tiles           |
| `--max-tiles` | integer                 | No       | Maximum number of tiles to download             |
| `--force`     | flag                    | No       | Force re-download existing tiles                |

### Examples

```bash
# Download tiles for Paris center (up to 10 tiles)
ign-lidar-hd download \
  --bbox 2.25,48.82,2.42,48.90 \
  --output /data/raw_tiles/ \
  --max-tiles 10

# Download all available tiles in area
ign-lidar-hd download \
  --bbox 2.25,48.82,2.42,48.90 \
  --output /data/raw_tiles/

# Force re-download existing tiles
ign-lidar-hd download \
  --bbox 2.25,48.82,2.42,48.90 \
  --output /data/raw_tiles/ \
  --force
```

### Output

Downloads LAZ files named with IGN conventions:

```
raw_tiles/
├── LIDARHD_FXX_0123_4567_LA93_IGN69_2020.laz
├── LIDARHD_FXX_0124_4567_LA93_IGN69_2020.laz
└── ...
```

### Notes

- Coordinates must be in WGS84 (longitude/latitude)
- Valid range for France: longitude 1-8°, latitude 42-51°
- Files are typically 200-300 MB each
- [Smart skip detection](../features/smart-skip.md) avoids re-downloading existing files

## enrich

Add building component features to LiDAR point clouds.

### Syntax

```bash
ign-lidar-hd enrich \
  --input-dir INPUT_DIR \
  --output OUTPUT_DIR \
  --mode MODE \
  [--num-workers WORKERS] \
  [--force]
```

### Parameters

| Parameter       | Type    | Required | Description                                     |
| --------------- | ------- | -------- | ----------------------------------------------- |
| `--input-dir`   | string  | Yes      | Directory containing raw LAZ tiles              |
| `--output`      | string  | Yes      | Output directory for enriched tiles             |
| `--mode`        | string  | Yes      | Feature extraction mode (currently: `building`) |
| `--num-workers` | integer | No       | Number of parallel workers (default: 4)         |
| `--force`       | flag    | No       | Force re-enrichment of existing files           |

### Examples

```bash
# Enrich tiles with building features
ign-lidar-hd enrich \
  --input-dir /data/raw_tiles/ \
  --output /data/enriched_tiles/ \
  --mode building

# Use 8 parallel workers
ign-lidar-hd enrich \
  --input-dir /data/raw_tiles/ \
  --output /data/enriched_tiles/ \
  --mode building \
  --num-workers 8

# Force re-enrichment
ign-lidar-hd enrich \
  --input-dir /data/raw_tiles/ \
  --output /data/enriched_tiles/ \
  --mode building \
  --force
```

### Output

Creates enriched LAZ files with additional point attributes:

```
enriched_tiles/
├── LIDARHD_FXX_0123_4567_LA93_IGN69_2020.laz  # +30 geometric features
├── LIDARHD_FXX_0124_4567_LA93_IGN69_2020.laz
└── ...
```

### Features Added

The enrichment process adds 30+ geometric features per point:

- **Normal vectors** (nx, ny, nz)
- **Curvature** (mean, gaussian)
- **Planarity** and **sphericity**
- **Verticality** and **eigenvalues**
- **Density** measures
- **Height** statistics
- And more...

### Notes

- Only `building` mode is currently supported
- Processing time: ~2-5 minutes per tile (depends on point density)
- Memory usage: ~2-4 GB per worker
- [Smart skip detection](../features/smart-skip.md) avoids re-enriching existing files

## patch

Extract machine learning patches from enriched tiles with optional RGB augmentation.

### Syntax

```bash
ign-lidar-hd patch \
  --input INPUT_PATH \
  --output OUTPUT_DIR \
  --lod-level LOD_LEVEL \
  [--patch-size PATCH_SIZE] \
  [--num-workers WORKERS] \
  [--include-rgb] \
  [--rgb-cache-dir CACHE_DIR] \
  [--force]
```

### Parameters

| Parameter         | Type    | Required | Description                              |
| ----------------- | ------- | -------- | ---------------------------------------- |
| `--input`         | string  | Yes      | Path to enriched LAZ file or directory   |
| `--output`        | string  | Yes      | Output directory for patches             |
| `--lod-level`     | string  | Yes      | Classification level: `LOD2` or `LOD3`   |
| `--patch-size`    | float   | No       | Patch size in meters (default: 10.0)     |
| `--num-workers`   | integer | No       | Number of parallel workers (default: 4)  |
| `--include-rgb`   | flag    | No       | Add RGB colors from IGN orthophotos      |
| `--rgb-cache-dir` | string  | No       | Cache directory for orthophoto downloads |
| `--force`         | flag    | No       | Force reprocessing existing patches      |

### Examples

```bash
# Create patches for LOD2 (geometry only)
ign-lidar-hd patch \
  --input /data/enriched_tiles/tile.laz \
  --output /data/patches/ \
  --lod-level LOD2

# Create patches with RGB augmentation from IGN orthophotos
ign-lidar-hd patch \
  --input /data/enriched_tiles/ \
  --output /data/patches/ \
  --lod-level LOD2 \
  --include-rgb \
  --rgb-cache-dir /data/cache/

# Process entire directory for LOD3 with RGB
ign-lidar-hd patch \
  --input /data/enriched_tiles/ \
  --output /data/patches/ \
  --lod-level LOD3 \
  --patch-size 15.0 \
  --num-workers 6 \
  --include-rgb

# Force reprocessing with RGB
ign-lidar-hd patch \
  --input /data/enriched_tiles/ \
  --output /data/patches/ \
  --lod-level LOD2 \
  --include-rgb \
  --force
```

### Output

Creates NPZ patch files organized by source tile:

```
patches/
├── tile_0123_4567/
│   ├── patch_0001.npz
│   ├── patch_0002.npz
│   └── ...
├── tile_0124_4567/
│   ├── patch_0001.npz
│   └── ...
```

### Patch Contents

Each NPZ file contains:

- `points`: Point coordinates (N×3 array)
- `features`: Geometric features (N×30+ array)
- `labels`: Building component labels (N×1 array)
- `rgb`: RGB colors (N×3 array, normalized 0-1) - **only if `--include-rgb` is used**
- `metadata`: Patch information (dict)

### RGB Augmentation

When using `--include-rgb`, the library automatically:

1. Fetches orthophotos from IGN BD ORTHO® service (20cm resolution)
2. Maps each 3D point to its corresponding 2D orthophoto pixel
3. Extracts RGB colors and normalizes them to [0, 1] range
4. Caches downloaded orthophotos for performance

**Benefits:**

- Multi-modal learning (geometry + photometry)
- Enhanced ML model accuracy
- Better visualization capabilities
- Automatic - no manual orthophoto downloads needed

**Requirements:**

```bash
pip install requests Pillow
```

See the [RGB Augmentation Guide](../features/rgb-augmentation.md) for detailed information.

### Classification Levels

**LOD2 (15 classes)**: Basic building components

- Wall, Roof, Ground, Vegetation, Window, Door, etc.

**LOD3 (30 classes)**: Detailed building components

- All LOD2 classes plus roof details, facade elements, etc.

### Notes

- Patch size affects the number of points per patch
- Smaller patches = more patches, fewer points each
- Larger patches = fewer patches, more points each
- Processing time: ~1-3 minutes per tile (geometry only), ~2-5 minutes with RGB
- RGB augmentation adds ~196KB per patch (16384 points × 3 × 4 bytes)
- [Smart skip detection](../features/smart-skip.md) avoids reprocessing existing patches

## process (Deprecated)

:::warning Deprecated Command
The `process` command has been renamed to `patch` for clarity. While `process` still works for backwards compatibility, it will be removed in a future major version. Please use `patch` instead.
:::

### Migration

Simply replace `process` with `patch` in your commands:

```bash
# Old (deprecated)
ign-lidar-hd process --input tiles/ --output patches/

# New (recommended)
ign-lidar-hd patch --input tiles/ --output patches/
```

All parameters and functionality remain identical. See the [`patch` command documentation](#patch) above.

## Global Options

### Logging

Control output verbosity:

```bash
# Default logging
ign-lidar-hd command [args]

# Verbose output (debug level)
ign-lidar-hd command [args] --verbose

# Quiet mode (errors only)
ign-lidar-hd command [args] --quiet
```

### Help

Get help for any command:

```bash
# General help
ign-lidar-hd --help

# Command-specific help
ign-lidar-hd download --help
ign-lidar-hd enrich --help
ign-lidar-hd process --help
```

## Common Workflows

### Full Pipeline

Complete processing workflow:

```bash
# 1. Download
ign-lidar-hd download \
  --bbox 2.25,48.82,2.42,48.90 \
  --output raw_tiles/ \
  --max-tiles 5

# 2. Enrich
ign-lidar-hd enrich \
  --input-dir raw_tiles/ \
  --output enriched_tiles/ \
  --mode building \
  --num-workers 4

# 3. Process
ign-lidar-hd process \
  --input enriched_tiles/ \
  --output patches/ \
  --lod-level LOD2 \
  --num-workers 4
```

### Resume Interrupted Work

Thanks to [smart skip detection](../features/smart-skip.md), you can safely re-run commands:

```bash
# If download was interrupted, just re-run
ign-lidar-hd download --bbox ... --output raw_tiles/
# Will skip existing files and download only missing ones

# Same for processing
ign-lidar-hd process --input enriched/ --output patches/ --lod-level LOD2
# Will skip tiles that already have patches
```

### Force Reprocessing

Override smart skip when needed:

```bash
# Force re-download
ign-lidar-hd download --bbox ... --output raw_tiles/ --force

# Force re-enrichment
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --mode building --force

# Force reprocessing
ign-lidar-hd process --input enriched/ --output patches/ --lod-level LOD2 --force
```

## Performance Tips

### Worker Configuration

Choose worker count based on your system:

```bash
# For 8-core CPU with 16GB RAM
--num-workers 4

# For 16-core CPU with 32GB RAM
--num-workers 8

# For systems with limited memory
--num-workers 2
```

### Memory Management

Monitor memory usage:

```bash
# Check memory during processing
htop

# If memory is limited, reduce workers
ign-lidar-hd process --input tiles/ --output patches/ --num-workers 1
```

See the [Memory Optimization Guide](../reference/memory-optimization.md) for detailed strategies.

## Troubleshooting

### Command Not Found

If `ign-lidar-hd` doesn't work:

```bash
# Check if package is installed
pip list | grep ign-lidar

# Reinstall if needed
pip install -e .

# Try the installed command name
ign-lidar-hd --help
```

### Permission Errors

```bash
# Check directory permissions
ls -la /path/to/output/

# Create directories if needed
mkdir -p /path/to/output/
```

### Network Issues

```bash
# Test connectivity for downloads
ping geoservices.ign.fr

# Check firewall/proxy settings
curl -I https://geoservices.ign.fr/
```

### Processing Errors

```bash
# Verify LAZ file integrity
lasinfo tile.laz

# Check available disk space
df -h /path/to/output/

# Reduce workers if getting memory errors
--num-workers 1
```

## See Also

- [Basic Usage Guide](basic-usage.md) - Step-by-step workflow tutorial
- [Smart Skip Features](../features/smart-skip.md) - Automatic skip detection
- [Memory Optimization](../reference/memory-optimization.md) - Performance tuning
- [Python API Reference](../api/processor.md) - Programmatic usage
