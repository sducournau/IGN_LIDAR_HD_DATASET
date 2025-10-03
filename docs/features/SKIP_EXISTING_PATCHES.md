# Smart Processing: Skip Existing Patches

## Summary

The IGN LIDAR HD processor now **automatically skips tiles that already have patches** in the output directory by default. This prevents redundant preprocessing and saves significant computation time.

## What Changed

### New Default Behavior

- ✅ **Checks if patches from a tile already exist before processing**
- ✅ **Skips preprocessing if patches are found** (with count info)
- ✅ **Provides detailed statistics** showing processed, skipped tiles, and total patches
- ✅ **Works in both sequential and parallel processing modes**

### Enhanced Features

- **Smart Skip Detection**: Checks for `{tile_stem}_patch_*.npz` files in output directory
- **Patch Count Display**: Shows how many patches already exist for skipped tiles
- **Statistics Tracking**: Counts processed, skipped tiles, and total patches separately
- **Detailed Logging**: Clear emoji indicators (⏭️ for skipped, ✅ for processed)
- **Optional Force Mode**: Can force reprocessing with `--force` flag

## CLI Usage

### Basic Processing (Default - Skip Existing)

```bash
# Automatically skips tiles that already have patches
ign-lidar process --input-dir tiles/ --output patches/
```

Output example:

```
Found 20 LAZ files
Configuration: LOD=LOD2 | k=auto | patch_size=150.0m | augment=True

🔄 Processing sequentially
======================================================================
[1/20] Processing: HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz
  📊 Loaded 25,432,198 points | Classes: 8
  🔧 Computing features | k=20 | mode=core
  ✅ Completed: 48 patches in 23.5s (1,081,796 pts/s)

[2/20] ⏭️  HD_LIDARHD_FXX_0651_6860_PTS_C_LAMB93_IGN69.laz: 52 patches exist, skipping

[3/20] Processing: HD_LIDARHD_FXX_0652_6860_PTS_C_LAMB93_IGN69.laz
  📊 Loaded 23,891,045 points | Classes: 7
  🔧 Computing features | k=20 | mode=core
  ✅ Completed: 45 patches in 21.2s (1,126,933 pts/s)

======================================================================
📊 Processing Summary:
  Total tiles: 20
  ✅ Processed: 15
  ⏭️  Skipped: 5
  📦 Total patches created: 712
======================================================================
```

### Force Reprocessing

```bash
# Force reprocessing even if patches exist
ign-lidar process --input-dir tiles/ --output patches/ --force
```

### Python API

```python
from ign_lidar import LiDARProcessor
from pathlib import Path

processor = LiDARProcessor(lod_level='LOD2')

# Normal - skip existing patches
patches = processor.process_directory(
    input_dir=Path("tiles/"),
    output_dir=Path("patches/")
)

# Force reprocessing
patches = processor.process_directory(
    input_dir=Path("tiles/"),
    output_dir=Path("patches/"),
    skip_existing=False  # Reprocess everything
)

# Single file with skip control
patches = processor.process_tile(
    laz_file=Path("tiles/tile_0001.laz"),
    output_dir=Path("patches/"),
    skip_existing=True  # Default
)
```

## Benefits

### 1. **Resume Interrupted Processing**

If processing is interrupted, simply run the command again:

```bash
# First run - processes 100 tiles, gets to tile 45
ign-lidar process --input-dir tiles/ --output patches/

# Interrupted! Only processed 45 tiles...

# Second run - automatically skips 45 processed, continues with remaining 55
ign-lidar process --input-dir tiles/ --output patches/
```

### 2. **Save Computation Time**

No need to reprocess tiles you've already completed:

```
⏭️  Skipping 12 tiles (35 minutes saved)
✅ Processing 8 new tiles (18 minutes)
```

### 3. **Incremental Dataset Building**

Add new tiles to existing dataset:

```bash
# Week 1: Process initial tiles
ign-lidar process --input-dir batch1/ --output dataset/patches/

# Week 2: Add more tiles to same output directory
cp batch2/*.laz batch1/
ign-lidar process --input-dir batch1/ --output dataset/patches/
# Skips batch1 tiles (already have patches), processes only batch2 tiles
```

### 4. **Experiment with Augmentation**

Change augmentation settings without reprocessing:

```bash
# Initial processing with augmentation
ign-lidar process --input-dir tiles/ --output patches/ --num-augmentations 3

# Later: Add more augmentations (would need to clear first)
# Or process different tiles
```

## How Skip Detection Works

The processor checks for patches by looking for files matching the pattern:

```
{tile_stem}_patch_*.npz
```

For example, for tile `HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz`, it checks for:

- `HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69_patch_0000.npz`
- `HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69_patch_0001.npz`
- `HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69_patch_0000_aug_0.npz`
- ... and so on

If **any patches exist** for a tile, the entire tile is skipped.

## Use Cases

### Workflow 1: Resume After System Failure

```bash
# Start processing large dataset
ign-lidar process --input-dir large_dataset/ --output patches/ --num-workers 4

# System crashes at tile 145/500...

# Check what you have
ls patches/*.npz | wc -l
# 6,912 patches from 145 tiles

# Resume - automatically skips 145 processed tiles
ign-lidar process --input-dir large_dataset/ --output patches/ --num-workers 4
# Continues with remaining 355 tiles
```

### Workflow 2: Process Multiple Batches to Same Output

```bash
# Download and process Paris tiles
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output tiles/ --max-tiles 50
ign-lidar process --input-dir tiles/ --output paris_dataset/

# Later: Add Lyon tiles to same dataset
ign-lidar download --bbox 4.7,45.6,5.0,45.9 --output tiles/ --max-tiles 50
ign-lidar process --input-dir tiles/ --output paris_dataset/
# Skips Paris tiles (already have patches), processes only Lyon tiles
```

### Workflow 3: Partial Reprocessing

```python
from pathlib import Path
import shutil

output_dir = Path("patches/")

# Remove patches from specific tile to reprocess it
tile_name = "HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69"
for patch in output_dir.glob(f"{tile_name}_patch_*.npz"):
    patch.unlink()

# Now reprocess - will skip all except deleted tile
from ign_lidar import LiDARProcessor
processor = LiDARProcessor()
processor.process_directory(Path("tiles/"), output_dir)
```

### Workflow 4: Change Processing Parameters

```bash
# Initial processing with default parameters
ign-lidar process --input-dir tiles/ --output patches_v1/

# Want different patch size? Process to new directory
ign-lidar process --input-dir tiles/ --output patches_v2/ --patch-size 200.0

# Want to force reprocess with new parameters in same dir?
# Option 1: Use --force (overwrites existing patches)
ign-lidar process --input-dir tiles/ --output patches_v1/ --patch-size 200.0 --force

# Option 2: Clear patches and reprocess
rm patches_v1/*.npz
ign-lidar process --input-dir tiles/ --output patches_v1/ --patch-size 200.0
```

## Statistics Output

The processing summary provides detailed breakdown:

```
======================================================================
📊 Processing Summary:
  Total tiles: 50
  ✅ Processed: 35
  ⏭️  Skipped: 15
  📦 Total patches created: 1,680
======================================================================
```

This helps you understand:

- **Processed**: Tiles that were processed in this run (created new patches)
- **Skipped**: Tiles with existing patches (saved computation time)
- **Total patches**: New patches created (doesn't count existing)

## Configuration

### Per-Call Control

```python
from ign_lidar import LiDARProcessor
from pathlib import Path

processor = LiDARProcessor(lod_level='LOD2')

# Default behavior - skip existing
patches = processor.process_directory(
    Path("tiles/"),
    Path("patches/")
)

# Override - force reprocessing
patches = processor.process_directory(
    Path("tiles/"),
    Path("patches/"),
    skip_existing=False
)

# Single tile processing
patches = processor.process_tile(
    Path("tiles/tile.laz"),
    Path("patches/"),
    skip_existing=True  # or False
)
```

### Command Line

```bash
# Skip existing (default)
ign-lidar process --input-dir tiles/ --output patches/

# Force reprocessing
ign-lidar process --input-dir tiles/ --output patches/ --force
```

## Performance Impact

### Time Savings Example

```
Scenario: Re-running processing of 100 tiles (50 already processed)

Without skip detection:
  - Reprocesses all 100 tiles: ~180 minutes
  - Overwrites 50 existing patch sets

With skip detection (NEW):
  - Checks 50 existing: ~2 seconds
  - Processes 50 new tiles: ~90 minutes
  - Time saved: ~90 minutes ⏱️
```

### Skip Check Performance

The skip check is very fast:

- **Glob pattern matching**: ~0.01-0.05 seconds per tile
- **Negligible overhead** compared to full processing (20-60 seconds per tile)

## Best Practices

1. **Use Default Behavior**: Let `skip_existing=True` save you time
2. **Monitor Statistics**: Check the summary to see what was skipped
3. **Force Only When Needed**: Use `--force` only when parameters changed
4. **Incremental Builds**: Process new tiles into existing dataset directories
5. **Separate Experiments**: Use different output dirs for different parameter sets

## Troubleshooting

### "All tiles skipped but I need to reprocess"

**Option 1: Use --force flag**

```bash
ign-lidar process --input-dir tiles/ --output patches/ --force
```

**Option 2: Delete existing patches**

```bash
# Delete all patches
rm patches/*.npz

# Or delete specific tile's patches
rm patches/HD_LIDARHD_FXX_0650_6860_*_patch_*.npz

# Then reprocess
ign-lidar process --input-dir tiles/ --output patches/
```

**Option 3: Process to new directory**

```bash
# Keep old patches, create new ones
ign-lidar process --input-dir tiles/ --output patches_new/
```

### "Want to see which tiles will be skipped"

```python
from pathlib import Path

tile_dir = Path("tiles/")
patch_dir = Path("patches/")

tiles = list(tile_dir.glob("*.laz"))
print(f"Total tiles: {len(tiles)}")

skipped = []
to_process = []

for tile in tiles:
    tile_stem = tile.stem
    existing = list(patch_dir.glob(f"{tile_stem}_patch_*.npz"))
    if existing:
        skipped.append(tile.name)
    else:
        to_process.append(tile.name)

print(f"Will skip: {len(skipped)} tiles")
print(f"Will process: {len(to_process)} tiles")
```

### "Changed parameters, want to reprocess only affected tiles"

If you changed parameters that affect output, you should:

1. **Process to new directory** (recommended):

   ```bash
   ign-lidar process --input-dir tiles/ --output patches_new_params/
   ```

2. **Clear and reprocess**:

   ```bash
   rm patches/*.npz
   ign-lidar process --input-dir tiles/ --output patches/
   ```

3. **Use --force**:
   ```bash
   ign-lidar process --input-dir tiles/ --output patches/ --force
   ```

## Migration Guide

### Existing Workflows

**Before:** Processing always reprocessed everything

```bash
ign-lidar process --input-dir tiles/ --output patches/
# Would reprocess all tiles every time
```

**After:** Processing skips existing by default

```bash
ign-lidar process --input-dir tiles/ --output patches/
# Skips tiles with existing patches

# If you want old behavior (reprocess everything):
ign-lidar process --input-dir tiles/ --output patches/ --force
```

### Code Changes

**Single Tile Processing:**

```python
# Still works the same way
patches = processor.process_tile(laz_file, output_dir)

# Can now control skip behavior
patches = processor.process_tile(
    laz_file, output_dir, skip_existing=False
)
```

**Directory Processing:**

```python
# Still works the same way
patches = processor.process_directory(input_dir, output_dir)

# Can now control skip behavior
patches = processor.process_directory(
    input_dir, output_dir, skip_existing=False
)
```

## See Also

- [Processor Module](../ign_lidar/processor.py)
- [Processing Examples](../examples/full_workflow_example.py)
- [Parallel Processing](../examples/parallel_processing_example.py)
- [Skip Existing Downloads](SKIP_EXISTING_TILES.md)
