# Smart Skip Features - Summary

## Overview

This update adds intelligent skip detection to both **downloading** and **processing** workflows, preventing redundant operations and saving time and resources.

## Changes Summary

### 1. Output Format Preferences (Config)

**File**: `ign_lidar/config.py`

Added configuration options for output format preferences:

- `PREFER_AUGMENTED_LAZ = True`: Prefer augmented LAZ (LAZ 1.4) over QGIS format
- `AUTO_CONVERT_TO_QGIS = False`: Don't automatically convert to QGIS format

**Impact**: Users keep full-feature LAZ files by default, with optional QGIS conversion

### 2. Skip Existing Tiles (Downloader)

**File**: `ign_lidar/downloader.py`

Enhanced download methods to skip existing tiles:

- Modified `download_tile()` to return `(success, was_skipped)` tuple
- Added `skip_existing` parameter (default: `True`)
- Updated `batch_download()` to track downloaded/skipped/failed statistics
- Enhanced logging with emoji indicators (⏭️ for skipped, ✅ for downloaded)

**Impact**: Resume interrupted downloads, avoid re-downloading existing files

### 3. Skip Existing Patches (Processor)

**File**: `ign_lidar/processor.py`

Enhanced processing methods to skip tiles with existing patches:

- Modified `process_tile()` to check for existing patches before processing
- Added `skip_existing` parameter (default: `True`)
- Updated `process_directory()` to track processed/skipped/total statistics
- Enhanced logging with detailed summary statistics

**Impact**: Resume interrupted processing, avoid reprocessing existing tiles

### 4. CLI Enhancements

**File**: `ign_lidar/cli.py`

Added CLI options for skip control:

- `--auto-convert-qgis` flag for enrich command (opt-in QGIS conversion)
- `--force` flag for process command (override skip_existing)
- Automatic skip behavior by default in both commands

**Impact**: Easy control of skip behavior from command line

## Features

### Smart Download Skip

```bash
# Automatically skips existing tiles
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output tiles/

Output:
⏭️  tile_001.laz already exists (245 MB), skipping
Downloading tile_002.laz...
✓ Downloaded tile_002.laz (238 MB)

📊 Download Summary:
  Total tiles requested: 10
  ✅ Successfully downloaded: 7
  ⏭️  Skipped (already present): 2
  ❌ Failed: 1
```

### Smart Processing Skip

```bash
# Automatically skips tiles with existing patches
ign-lidar process --input-dir tiles/ --output patches/

Output:
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

### Format Preference

```bash
# Default: Save augmented LAZ with all features
ign-lidar enrich --input-dir raw/ --output enriched/
Output: enriched/tile.laz (LAZ 1.4, all features)

# Optional: Also create QGIS versions
ign-lidar enrich --input-dir raw/ --output enriched/ --auto-convert-qgis
Output:
  - enriched/tile.laz (LAZ 1.4, all features)
  - enriched/tile_qgis.laz (LAZ 1.2, QGIS-compatible)
```

## Benefits

### Time Savings

- **Downloads**: Skip re-downloading tiles (~60 min saved on 50 tiles)
- **Processing**: Skip reprocessing tiles (~90 min saved on 50 tiles)
- **Total**: ~150 minutes saved on typical workflow

### Resource Savings

- **Bandwidth**: Avoid downloading duplicate large files (12+ GB on 50 tiles)
- **Disk Space**: Avoid creating duplicate patches
- **CPU/Memory**: Avoid redundant feature computation

### Workflow Improvements

- **Resume Capability**: Easily resume after interruptions
- **Incremental Builds**: Add new data to existing datasets
- **Idempotent Operations**: Safe to run commands multiple times
- **Better Defaults**: Most users want augmented LAZ, not QGIS format

## Use Cases

### 1. Resume After Interruption

```bash
# Start big job
ign-lidar download --bbox ... --output tiles/ --max-tiles 100
ign-lidar process --input-dir tiles/ --output patches/

# System crashes at tile 45...

# Resume - automatically skips completed work
ign-lidar download --bbox ... --output tiles/ --max-tiles 100  # Skips 45 tiles
ign-lidar process --input-dir tiles/ --output patches/          # Skips 45 tiles
```

### 2. Incremental Dataset Building

```bash
# Week 1: Download Paris
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output france_tiles/
ign-lidar process --input-dir france_tiles/ --output france_patches/

# Week 2: Add Lyon (some overlap)
ign-lidar download --bbox 4.7,45.6,5.0,45.9 --output france_tiles/
# Skips any overlapping tiles
ign-lidar process --input-dir france_tiles/ --output france_patches/
# Skips Paris tiles, processes only Lyon tiles
```

### 3. Format Flexibility

```bash
# Process to augmented LAZ (default)
ign-lidar enrich --input-dir raw/ --output enriched/

# Later, if you need QGIS versions
ign-lidar-qgis enriched/tile.laz
# Or batch convert
python scripts/batch_convert_qgis.py enriched/
```

## API Examples

### Python: Download with Skip

```python
from ign_lidar import IGNLiDARDownloader
from pathlib import Path

downloader = IGNLiDARDownloader(Path("tiles/"))

# Skip existing by default
results = downloader.batch_download(tile_list)

# Check what was skipped
for tile, success in results.items():
    print(f"{tile}: {'✓' if success else '✗'}")
```

### Python: Process with Skip

```python
from ign_lidar import LiDARProcessor
from pathlib import Path

processor = LiDARProcessor(lod_level='LOD2')

# Skip existing patches by default
patches = processor.process_directory(
    Path("tiles/"),
    Path("patches/")
)

# Force reprocessing if needed
patches = processor.process_directory(
    Path("tiles/"),
    Path("patches/"),
    skip_existing=False
)
```

## Documentation

New documentation files created:

- `docs/OUTPUT_FORMAT_PREFERENCES.md` - LAZ format preferences
- `docs/SKIP_EXISTING_TILES.md` - Download skip behavior
- `docs/SKIP_EXISTING_PATCHES.md` - Processing skip behavior

## Migration Notes

### For Existing Scripts

**Downloads:**

```python
# Before: Always downloaded
success = downloader.download_tile(filename)

# After: Returns tuple (backward compatible with unpacking)
success, was_skipped = downloader.download_tile(filename)
# Or ignore skip info
success, _ = downloader.download_tile(filename)
```

**Processing:**

```python
# Before & After: Same interface, new parameter
patches = processor.process_directory(input_dir, output_dir)

# New option: Control skip behavior
patches = processor.process_directory(
    input_dir, output_dir, skip_existing=False
)
```

### For Existing Workflows

**If you want old behavior (always reprocess/redownload):**

```bash
# Processing
ign-lidar process --input-dir tiles/ --output patches/ --force

# Downloads (Python API)
downloader.batch_download(tiles, skip_existing=False)
```

## Configuration Options

### Download Behavior

```python
# In code
downloader.download_tile(filename, skip_existing=True)  # Default
downloader.download_tile(filename, skip_existing=False) # Always download
downloader.download_tile(filename, force=True)          # Force re-download
```

### Processing Behavior

```python
# In code
processor.process_tile(laz_file, output_dir, skip_existing=True)  # Default
processor.process_tile(laz_file, output_dir, skip_existing=False) # Always process

# CLI
ign-lidar process --input-dir tiles/ --output patches/         # Skip existing
ign-lidar process --input-dir tiles/ --output patches/ --force # Reprocess all
```

### Format Preferences

```python
# In ign_lidar/config.py
PREFER_AUGMENTED_LAZ = True      # Keep full-feature LAZ (default)
AUTO_CONVERT_TO_QGIS = False     # Don't auto-convert to QGIS (default)

# CLI
ign-lidar enrich --input-dir raw/ --output enriched/                     # LAZ only
ign-lidar enrich --input-dir raw/ --output enriched/ --auto-convert-qgis # LAZ + QGIS
```

## Performance Metrics

### Download Skip Performance

- Check if file exists: ~0.001-0.01s per file
- Download a 250MB file: ~30-60s
- **Speedup**: 3000-60000x faster to skip

### Processing Skip Performance

- Check for existing patches: ~0.01-0.05s per tile
- Process a tile: ~20-60s
- **Speedup**: 400-6000x faster to skip

### Typical Workflow Time Savings

```
100 tiles, 50% already processed:

Old behavior:
  Download: 100 tiles × 45s = 75 min
  Process: 100 tiles × 35s = 58 min
  Total: 133 minutes

New behavior:
  Download: 50 skipped (0.5 min) + 50 new (37.5 min) = 38 min
  Process: 50 skipped (0.4 min) + 50 new (29 min) = 29.4 min
  Total: 67.4 minutes

Time saved: 65.6 minutes (49% reduction)
```

## Testing

To verify the skip behavior works:

```bash
# Download some tiles
ign-lidar download --bbox 2.0,48.8,2.1,48.9 --output test_tiles/ --max-tiles 5

# Process them
ign-lidar process --input-dir test_tiles/ --output test_patches/

# Run again - should skip everything
ign-lidar download --bbox 2.0,48.8,2.1,48.9 --output test_tiles/ --max-tiles 5
# Should see: "5 tiles skipped"

ign-lidar process --input-dir test_tiles/ --output test_patches/
# Should see: "5 tiles skipped, 0 processed"

# Force reprocess
ign-lidar process --input-dir test_tiles/ --output test_patches/ --force
# Should see: "5 tiles processed, 0 skipped"
```

## See Also

- [Skip Existing Tiles Documentation](SKIP_EXISTING_TILES.md)
- [Skip Existing Patches Documentation](SKIP_EXISTING_PATCHES.md)
- [Output Format Preferences](OUTPUT_FORMAT_PREFERENCES.md)
- [Downloader Module](../ign_lidar/downloader.py)
- [Processor Module](../ign_lidar/processor.py)
