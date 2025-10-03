# Smart Download: Skip Existing Tiles

## Summary

The IGN LIDAR HD downloader now **automatically skips tiles that are already present** in the output directory by default. This prevents redundant downloads and saves time and bandwidth.

## What Changed

### New Default Behavior

- ✅ **Checks if tile already exists before downloading**
- ✅ **Skips download if file is already present** (with file size info)
- ✅ **Provides detailed statistics** showing downloaded, skipped, and failed tiles
- ✅ **Works in both sequential and parallel download modes**

### Enhanced Features

- **Smart Skip Detection**: Checks output directory for existing LAZ files
- **File Validation**: Verifies file exists and reports file size
- **Statistics Tracking**: Counts downloaded, skipped, and failed tiles separately
- **Detailed Logging**: Clear emoji indicators (⏭️ for skipped, ✅ for downloaded)
- **Optional Force Mode**: Can still force re-download if needed

## CLI Usage

### Basic Download (Default - Skip Existing)

```bash
# Automatically skips existing tiles
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output tiles/
```

Output example:

```
⏭️  HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz already exists (245 MB), skipping
Downloading HD_LIDARHD_FXX_0651_6860_PTS_C_LAMB93_IGN69.laz...
✓ Downloaded HD_LIDARHD_FXX_0651_6860_PTS_C_LAMB93_IGN69.laz (238 MB)

======================================================================
📊 Download Summary:
  Total tiles requested: 10
  ✅ Successfully downloaded: 7
  ⏭️  Skipped (already present): 2
  ❌ Failed: 1
======================================================================
```

### Force Re-download (Override Skip)

If you need to re-download existing files:

```python
from ign_lidar import IGNLiDARDownloader
from pathlib import Path

downloader = IGNLiDARDownloader(Path("tiles/"))

# Force re-download even if file exists
downloader.download_tile(
    "HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz",
    force=True  # Override skip_existing
)
```

### Disable Skip Behavior

```python
# Download all tiles, even if they exist (not recommended)
results = downloader.batch_download(
    tile_list=tiles,
    skip_existing=False  # Download everything
)
```

## API Changes

### `download_tile()` Method

**Before:**

```python
def download_tile(self, filename: str, force: bool = False) -> bool:
    """Returns: True if successful"""
```

**After:**

```python
def download_tile(self, filename: str, force: bool = False,
                  skip_existing: bool = True) -> Tuple[bool, bool]:
    """
    Returns:
        Tuple of (success, was_skipped)
        - success: True if download successful or file already exists
        - was_skipped: True if download was skipped (file already present)
    """
```

### `batch_download()` Method

**New Parameter:**

```python
def batch_download(self, tile_list: List[str],
                   skip_existing: bool = True,  # New parameter
                   ...):
```

**Enhanced Statistics:**
The method now tracks and reports:

- `downloaded`: Number of tiles actually downloaded
- `skipped`: Number of tiles skipped (already present)
- `failed`: Number of tiles that failed to download

## Benefits

### 1. **Resume Downloads**

If a download is interrupted, simply run the command again:

```bash
# First run - downloads 100 tiles
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output tiles/ --max-tiles 100

# Interrupted! Only got 45 tiles...

# Second run - only downloads remaining 55 tiles
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output tiles/ --max-tiles 100
# Automatically skips the 45 already downloaded
```

### 2. **Save Bandwidth**

No need to re-download large tiles you already have:

```
⏭️  Skipping 15 tiles (3.5 GB) - already downloaded
✅ Downloading 5 new tiles (1.2 GB)
```

### 3. **Incremental Updates**

Download new tiles as coverage expands:

```bash
# Download initial dataset
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output tiles/

# Later, download expanded area (overlaps with existing)
ign-lidar download --bbox 1.8,48.6,2.7,49.2 --output tiles/
# Skips tiles already in tiles/, downloads only new ones
```

### 4. **Safe Re-runs**

Accidentally run the same command twice? No problem:

```bash
# First run
python workflow.py --download-tiles

# Oops, ran it again
python workflow.py --download-tiles
# All tiles skipped - no duplicate downloads!
```

## Use Cases

### Workflow 1: Resume After Network Failure

```bash
# Download interrupted at tile 23/100
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output tiles/ --max-tiles 100

# Check what you have
ls tiles/*.laz | wc -l
# 23 files

# Resume - skips 23 existing, downloads remaining 77
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output tiles/ --max-tiles 100
```

### Workflow 2: Download Different Regions to Same Directory

```bash
# Download Paris area
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output france_tiles/

# Download Lyon area (some tiles may overlap)
ign-lidar download --bbox 4.7,45.6,5.0,45.9 --output france_tiles/

# Download Marseille area
ign-lidar download --bbox 5.2,43.2,5.5,43.5 --output france_tiles/

# Result: One directory with all unique tiles, no duplicates
```

### Workflow 3: Incremental Dataset Building

```python
from ign_lidar import IGNLiDARDownloader
from pathlib import Path

downloader = IGNLiDARDownloader(Path("dataset/raw_tiles"))

# Week 1: Download strategic locations
for location in ['paris', 'lyon', 'marseille']:
    tiles = get_strategic_tiles(location)
    downloader.batch_download(tiles)

# Week 2: Add more cities (automatic skip of existing)
for location in ['toulouse', 'nice', 'nantes']:
    tiles = get_strategic_tiles(location)
    downloader.batch_download(tiles)  # Skips any overlaps

# Week 3: Fill gaps
all_tiles = get_all_tiles_in_bbox(large_bbox)
downloader.batch_download(all_tiles)  # Only downloads new tiles
```

## Statistics Output

The download summary now provides detailed breakdown:

```
======================================================================
📊 Download Summary:
  Total tiles requested: 50
  ✅ Successfully downloaded: 35
  ⏭️  Skipped (already present): 12
  ❌ Failed: 3
======================================================================
```

This helps you understand:

- **Downloaded**: New tiles acquired in this run
- **Skipped**: Tiles you already had (saved bandwidth)
- **Failed**: Tiles that need attention (network issues, unavailable, etc.)

## File Validation

The skip detection includes basic file validation:

- ✅ Checks if file exists
- ✅ Reports file size in MB
- ✅ Logs skip action clearly

Example log:

```
⏭️  HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz already exists (245 MB), skipping
```

## Configuration

### Global Configuration (config.py)

You can set default behavior in your code:

```python
from ign_lidar import IGNLiDARDownloader

# Create downloader
downloader = IGNLiDARDownloader(output_dir="tiles/")

# All batch_download calls will skip existing by default
results = downloader.batch_download(tile_list)

# Override per call if needed
results = downloader.batch_download(tile_list, skip_existing=False)
```

### Per-Download Control

Control behavior for individual downloads:

```python
# Normal - skip existing
success, skipped = downloader.download_tile("tile.laz")

# Force re-download
success, skipped = downloader.download_tile("tile.laz", force=True)

# Disable skip checking
success, skipped = downloader.download_tile(
    "tile.laz",
    skip_existing=False
)
```

## Migration Guide

### Existing Code Compatibility

**Single Downloads:**
If you're checking return values, update to handle tuple:

```python
# Before
success = downloader.download_tile(filename)
if success:
    print("Downloaded!")

# After (backward compatible - tuple unpacking)
success, was_skipped = downloader.download_tile(filename)
if success:
    if was_skipped:
        print("Already had it!")
    else:
        print("Downloaded!")

# Or ignore skip info
success, _ = downloader.download_tile(filename)
```

**Batch Downloads:**
No changes needed - dict of results still works the same:

```python
# Still works exactly the same
results = downloader.batch_download(tile_list)
success_count = sum(1 for s in results.values() if s)
```

## Performance

### Time Savings Example

```
Scenario: Re-running download of 100 tiles (50 already present)

Without skip detection:
  - Downloads all 100 tiles again: ~120 minutes
  - Bandwidth used: 25 GB

With skip detection (NEW):
  - Checks 50 existing: ~5 seconds
  - Downloads 50 new tiles: ~60 minutes
  - Bandwidth saved: 12.5 GB
  - Time saved: ~60 minutes ⏱️
```

## Best Practices

1. **Use Default Behavior**: Let skip_existing=True save you time
2. **Monitor Statistics**: Check the summary to see what was skipped
3. **Force Only When Needed**: Use `force=True` only for corrupted files
4. **Incremental Downloads**: Build datasets gradually, run downloads multiple times
5. **Single Output Directory**: Keep related tiles in one directory for efficient skipping

## Troubleshooting

### "All tiles skipped but I need to re-download"

```python
# Force re-download specific tile
downloader.download_tile(filename, force=True)

# Or manually delete and re-run
import os
os.remove("tiles/tile_name.laz")
downloader.download_tile("tile_name.laz")
```

### "Want to see which tiles exist before downloading"

```python
from pathlib import Path

output_dir = Path("tiles/")
existing = set(f.name for f in output_dir.glob("*.laz"))
print(f"Already have {len(existing)} tiles")

# Filter tile_list to show what will be downloaded
to_download = [t for t in tile_list if t not in existing]
print(f"Will download {len(to_download)} new tiles")
```

## See Also

- [Downloader Module](../ign_lidar/downloader.py)
- [Batch Download Examples](../examples/download_strategic_dataset.py)
- [IGN LIDAR HD Documentation](https://geoservices.ign.fr/lidarhd)
