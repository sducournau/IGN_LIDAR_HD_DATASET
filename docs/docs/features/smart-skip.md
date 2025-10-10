---
sidebar_position: 1
title: Smart Skip Detection
description: Automatically skip existing downloads, enriched files, and patches
keywords: [skip, idempotent, resume, workflow]
---

Smart Skip Detection prevents redundant operations by automatically detecting and skipping existing files during downloads, enrichment, and processing workflows.

## Overview

This feature adds intelligent skip detection to all workflows:

- **Download Skip** - Avoid re-downloading existing tiles
- **Enrichment Skip** - Skip files that are already enriched
- **Processing Skip** - Skip tiles with existing patches

## Key Benefits

### ⚡ Time Savings

- **Downloads**: Skip re-downloading tiles (~60 min saved on 50 tiles)
- **Processing**: Skip reprocessing tiles (~90 min saved on 50 tiles)
- **Total**: ~150 minutes saved on typical workflow

### 💾 Resource Savings

- **Bandwidth**: Avoid downloading duplicate large files (12+ GB on 50 tiles)
- **Disk Space**: Avoid creating duplicate patches
- **CPU/Memory**: Avoid redundant feature computation

### 🔄 Workflow Improvements

- **Resume Capability**: Easily resume after interruptions
- **Incremental Builds**: Add new data to existing datasets
- **Idempotent Operations**: Safe to run commands multiple times

## Smart Download Skip

Automatically skips existing tiles during download:

```bash
# Downloads only missing tiles
ign-lidar-hd download \
  --bbox 2.0,48.8,2.5,49.0 \
  --output tiles/

# Output shows what's skipped vs downloaded
⏭️  tile_001.laz already exists (245 MB), skipping
Downloading tile_002.laz...
✅ Downloaded tile_002.laz (238 MB)

📊 Download Summary:
  Total tiles requested: 10
  ✅ Successfully downloaded: 7
  ⏭️  Skipped (already present): 2
  ❌ Failed: 1
```

### Force Re-download

Use `--force` flag to override skip behavior:

```bash
# Force re-download all tiles
ign-lidar-hd download \
  --bbox 2.0,48.8,2.5,49.0 \
  --output tiles/ \
  --force
```

## Smart Enrichment Skip

Automatically skips LAZ files that are already enriched:

```bash
# Enriches only files without building features
ign-lidar-hd enrich \
  --input-dir /path/to/raw_tiles/ \
  --output /path/to/enriched_tiles/ \
  --mode full

# Shows progress and skip statistics
[1/20] Processing: tile_001.laz
  ✅ Enriched: 1.2M points in 15.3s
[2/20] ⏭️  tile_002.laz: Already enriched, skipping
[3/20] Processing: tile_003.laz
  ✅ Enriched: 980K points in 12.1s
```

### Force Re-enrichment

Use `--force` flag to re-enrich files:

```bash
# Force re-enrichment of all files
ign-lidar-hd enrich \
  --input-dir /path/to/raw_tiles/ \
  --output /path/to/enriched_tiles/ \
  --mode full \
  --force
```

## Smart Processing Skip

Automatically skips tiles with existing patches:

```bash
# Processes only tiles without patches
ign-lidar-hd process \
  --input enriched_tiles/ \
  --output patches/ \
  --lod-level LOD2

# Shows detailed skip/process statistics
[1/20] Processing: tile_001.laz
  ✅ Completed: 48 patches in 23.5s
[2/20] ⏭️  tile_002.laz: 52 patches exist, skipping
[3/20] Processing: tile_003.laz
  ✅ Completed: 45 patches in 21.2s

📊 Processing Summary:
  Total tiles: 20
  ✅ Processed: 15
  ⏭️  Skipped: 5
  📦 Total patches created: 712
```

### Force Reprocessing

Use `--force` flag to reprocess all tiles:

```bash
# Force reprocess all tiles
ign-lidar-hd process \
  --input enriched_tiles/ \
  --output patches/ \
  --lod-level LOD2 \
  --force
```

## Common Use Cases

### 1. Resume After Interruption

```bash
# Start big job
ign-lidar-hd download --bbox ... --output tiles/ --max-tiles 100
ign-lidar-hd process --input tiles/ --output patches/

# System crashes at tile 45...

# Resume - automatically skips completed work
ign-lidar-hd download --bbox ... --output tiles/ --max-tiles 100  # Skips 45 tiles
ign-lidar-hd process --input tiles/ --output patches/          # Skips 45 tiles
```

### 2. Incremental Dataset Building

```bash
# Week 1: Download Paris
ign-lidar-hd download --bbox 2.0,48.8,2.5,49.0 --output france_tiles/
ign-lidar-hd process --input france_tiles/ --output france_patches/

# Week 2: Add Lyon (some overlap)
ign-lidar-hd download --bbox 4.7,45.6,5.0,45.9 --output france_tiles/
# Skips any overlapping tiles
ign-lidar-hd process --input france_tiles/ --output france_patches/
# Skips Paris tiles, processes only Lyon tiles
```

### 3. Batch Processing with Mixed Status

```bash
# Process a directory with mixed completion status
ign-lidar-hd process --input mixed_tiles/ --output patches/

# Output shows what's done vs what needs processing
⏭️  Tiles with existing patches: 15
✅ New tiles processed: 8
❌ Failed tiles: 2
```

## Python API

### Download with Skip Control

```python
from ign_lidar import IGNLiDARDownloader
from pathlib import Path

downloader = IGNLiDARDownloader(Path("tiles/"))

# Skip existing by default
results = downloader.batch_download(tile_list)

# Force re-download
results = downloader.batch_download(tile_list, skip_existing=False)

# Check individual results
success, was_skipped = downloader.download_tile(filename)
if was_skipped:
    print(f"Skipped {filename} (already exists)")
elif success:
    print(f"Downloaded {filename}")
else:
    print(f"Failed to download {filename}")
```

### Processing with Skip Control

```python
from ign_lidar import LiDARProcessor
from pathlib import Path

processor = LiDARProcessor(lod_level='LOD2')

# Skip existing patches by default
patches = processor.process_directory(
    Path("enriched_tiles/"),
    Path("patches/")
)

# Force reprocessing
patches = processor.process_directory(
    Path("enriched_tiles/"),
    Path("patches/"),
    skip_existing=False
)
```

## Performance Impact

### Skip Check Performance

- **File existence check**: ~0.001-0.01s per file
- **Patch directory check**: ~0.01-0.05s per tile
- **Enrichment check**: ~0.02-0.1s per file

### Time Comparison

```
100 tiles, 50% already processed:

Without Skip Detection:
  Download: 100 tiles × 45s = 75 min
  Process: 100 tiles × 35s = 58 min
  Total: 133 minutes

With Skip Detection:
  Download: 50 skipped (0.5 min) + 50 new (37.5 min) = 38 min
  Process: 50 skipped (0.4 min) + 50 new (29 min) = 29.4 min
  Total: 67.4 minutes

Time saved: 65.6 minutes (49% reduction)
```

## Configuration

Smart skip detection is **enabled by default** for all operations. You can control it via:

### CLI Flags

```bash
# Default: Skip existing
ign-lidar-hd command [args]

# Force override: Process everything
ign-lidar-hd command [args] --force
```

### Python Parameters

```python
# Default: skip_existing=True
processor.process_tile(file, output_dir)

# Override: skip_existing=False
processor.process_tile(file, output_dir, skip_existing=False)
```

## Skip Detection Logic

### Download Skip

- Checks if LAZ file exists in output directory
- Compares file size (skips if > 1MB, indicating complete download)
- Logs skip reason and file size

### Enrichment Skip

- Checks if output file already exists
- Validates that file contains building features
- Skips if features are already present

### Processing Skip

- Checks if patch directory exists for the tile
- Counts existing .npz patches
- Skips if patches already exist (non-zero count)

## Troubleshooting

### Files Not Being Skipped

Check that file paths and naming are consistent:

```bash
# Verify file naming patterns
ls -la tiles/
ls -la patches/
```

### Unexpected Skips

Use verbose logging to see skip decisions:

```bash
# Enable debug logging
ign-lidar-hd process --input tiles/ --output patches/ --verbose
```

### Force Reprocessing

When you need to reprocess everything:

```bash
# Method 1: Use --force flag
ign-lidar-hd process --input tiles/ --output patches/ --force

# Method 2: Clear output directory
rm -rf patches/*
ign-lidar-hd process --input tiles/ --output patches/
```

## See Also

- [Basic Usage Guide](../guides/basic-usage)
- [CLI Commands Reference](../guides/cli-commands)
- [Memory Optimization](../reference/memory-optimization)
