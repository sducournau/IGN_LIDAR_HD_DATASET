# Skip Existing Enriched Files

## Summary

The enrichment process now **automatically skips LAZ files that have already been enriched** in the output directory by default. This prevents redundant feature computation and saves significant processing time.

## What Changed

### New Default Behavior

- ✅ **Checks if enriched output file exists before processing**
- ✅ **Skips enrichment if file found** (with file size info)
- ✅ **Provides detailed statistics** showing enriched, skipped, and failed files
- ✅ **Works in both sequential and parallel processing modes**

### Enhanced Features

- **Smart Skip Detection**: Checks for output LAZ file before processing
- **File Validation**: Verifies file exists and reports file size
- **Statistics Tracking**: Counts enriched, skipped, and failed files separately
- **Detailed Logging**: Clear emoji indicators (⏭️ for skipped, ✅ for enriched)
- **Optional Force Mode**: Can force re-enrichment with `--force` flag

## CLI Usage

### Basic Enrichment (Default - Skip Existing)

```bash
# Automatically skips tiles that are already enriched
ign-lidar enrich --input-dir raw_tiles/ --output enriched/ --mode building
```

Output example:

```
Found 20 LAZ files
Maximum file size: 282 MB
Copying metadata files...

Processing HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz...
  ✓ Saved to enriched/HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz

⏭️  HD_LIDARHD_FXX_0651_6860_PTS_C_LAMB93_IGN69.laz already enriched (238 MB), skipping

Processing HD_LIDARHD_FXX_0652_6860_PTS_C_LAMB93_IGN69.laz...
  ✓ Saved to enriched/HD_LIDARHD_FXX_0652_6860_PTS_C_LAMB93_IGN69.laz

======================================================================
📊 Enrichment Summary:
  Total files: 20
  ✅ Enriched: 15
  ⏭️  Skipped: 5
  ❌ Failed: 0
======================================================================
```

### Force Re-enrichment

```bash
# Force re-enrichment even if files exist
ign-lidar enrich --input-dir raw_tiles/ --output enriched/ --mode building --force
```

### Python API

```python
from ign_lidar.cli import _enrich_single_file
from pathlib import Path

# Normal - skip existing (requires tuple argument)
result = _enrich_single_file((
    Path("raw_tiles/tile.laz"),
    Path("enriched/tile.laz"),
    k_neighbors=10,
    use_gpu=False,
    mode='building',
    skip_existing=True  # Default
))

# Force re-enrichment
result = _enrich_single_file((
    Path("raw_tiles/tile.laz"),
    Path("enriched/tile.laz"),
    k_neighbors=10,
    use_gpu=False,
    mode='building',
    skip_existing=False  # Override
))
```

## Benefits

### 1. **Resume Interrupted Enrichment**

If enrichment is interrupted, simply run the command again:

```bash
# First run - enriches 100 tiles, gets to tile 45
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building --num-workers 4

# Interrupted! Only enriched 45 tiles...

# Second run - automatically skips 45 enriched, continues with remaining 55
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building --num-workers 4
```

### 2. **Save Computation Time**

No need to recompute features for tiles you've already processed:

```
⏭️  Skipping 12 tiles (45 minutes saved)
✅ Enriching 8 new tiles (30 minutes)
```

### 3. **Incremental Dataset Building**

Add new tiles to existing enriched dataset:

```bash
# Week 1: Enrich initial tiles
ign-lidar enrich --input-dir batch1/ --output enriched/

# Week 2: Add more tiles
cp batch2/*.laz batch1/
ign-lidar enrich --input-dir batch1/ --output enriched/
# Skips batch1 tiles (already enriched), processes only batch2 tiles
```

### 4. **Safe Re-runs**

Accidentally run the same command twice? No problem:

```bash
# First run
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building

# Oops, ran it again
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building
# All tiles skipped - no duplicate work!
```

## How Skip Detection Works

The enrichment process checks if the output file exists before processing:

```
Input:  raw_tiles/HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz
Output: enriched/HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz

If output exists: Skip processing ⏭️
If output missing: Process and enrich ✅
```

Directory structure is preserved:

```
raw_tiles/
  paris/tile_001.laz
  lyon/tile_002.laz

enriched/
  paris/tile_001.laz  ← Checks here
  lyon/tile_002.laz   ← And here
```

## Use Cases

### Workflow 1: Resume After System Failure

```bash
# Start enriching large dataset
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building --num-workers 4

# System crashes at tile 78/200...

# Check what you have
ls enriched/*.laz | wc -l
# 78 files

# Resume - automatically skips 78 enriched tiles
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building --num-workers 4
# Continues with remaining 122 tiles
```

### Workflow 2: Process Multiple Batches to Same Output

```bash
# Download and enrich Paris tiles
ign-lidar download --bbox 2.0,48.8,2.5,49.0 --output tiles/ --max-tiles 50
ign-lidar enrich --input-dir tiles/ --output enriched/ --mode building

# Later: Add Lyon tiles to same dataset
ign-lidar download --bbox 4.7,45.6,5.0,45.9 --output tiles/ --max-tiles 50
ign-lidar enrich --input-dir tiles/ --output enriched/ --mode building
# Skips Paris tiles (already enriched), enriches only Lyon tiles
```

### Workflow 3: Partial Re-enrichment

```bash
# Remove specific enriched file to reprocess it
rm enriched/HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz

# Reprocess - will skip all except deleted file
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building
```

### Workflow 4: Change Processing Parameters

```bash
# Initial enrichment with core mode
ign-lidar enrich --input-dir raw/ --output enriched_core/ --mode core

# Want building mode? Process to new directory
ign-lidar enrich --input-dir raw/ --output enriched_building/ --mode building

# Want to force reprocess with new mode in same dir?
# Option 1: Use --force (overwrites existing files)
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building --force

# Option 2: Clear enriched files and reprocess
rm enriched/*.laz
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building
```

## Statistics Output

The enrichment summary provides detailed breakdown:

```
======================================================================
📊 Enrichment Summary:
  Total files: 50
  ✅ Enriched: 35
  ⏭️  Skipped: 15
  ❌ Failed: 0
======================================================================
```

This helps you understand:

- **Enriched**: Files that were processed in this run (created new enriched LAZ)
- **Skipped**: Files already enriched (saved computation time)
- **Failed**: Files that encountered errors (need attention)

## Configuration

### Command Line Control

```bash
# Skip existing (default)
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building

# Force re-enrichment
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building --force
```

### Processing Modes

```bash
# Core mode (faster, basic features)
ign-lidar enrich --input-dir raw/ --output enriched/ --mode core

# Building mode (slower, full features including building-specific)
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building
```

Both modes support skip detection by default.

## Performance Impact

### Time Savings Example

```
Scenario: Re-running enrichment of 100 tiles (50 already enriched)

Without skip detection:
  - Re-enriches all 100 tiles: ~200 minutes (building mode)
  - Overwrites 50 existing files

With skip detection (NEW):
  - Checks 50 existing: ~2 seconds
  - Enriches 50 new tiles: ~100 minutes
  - Time saved: ~100 minutes ⏱️
```

### Skip Check Performance

The skip check is very fast:

- **File existence check**: ~0.001-0.01 seconds per file
- **Negligible overhead** compared to enrichment (1-5 minutes per tile in building mode)

## Memory Management

Skip detection works alongside memory optimization:

```bash
# Large files with skip detection
ign-lidar enrich --input-dir large_tiles/ --output enriched/ \
  --mode building --num-workers 2

Output:
⏭️  tile_001.laz (300 MB) already enriched, skipping
Processing tile_002.laz (285 MB)...
⚠️  Using sequential batching for large files to prevent memory issues
  ✓ Saved to enriched/tile_002.laz
```

## Best Practices

1. **Use Default Behavior**: Let `skip_existing=True` save you time
2. **Monitor Statistics**: Check the summary to see what was skipped
3. **Force Only When Needed**: Use `--force` only when parameters changed
4. **Separate Experiments**: Use different output dirs for different modes
5. **Incremental Processing**: Add new raw tiles and re-run enrichment

## Troubleshooting

### "All tiles skipped but I need to re-enrich"

**Option 1: Use --force flag**

```bash
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building --force
```

**Option 2: Delete existing enriched files**

```bash
# Delete all enriched files
rm enriched/*.laz

# Or delete specific file
rm enriched/HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69.laz

# Then re-run enrichment
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building
```

**Option 3: Enrich to new directory**

```bash
# Keep old enriched files, create new ones
ign-lidar enrich --input-dir raw/ --output enriched_new/ --mode building
```

### "Want to see which tiles will be skipped"

```python
from pathlib import Path

raw_dir = Path("raw_tiles/")
enriched_dir = Path("enriched/")

raw_files = list(raw_dir.glob("*.laz"))
print(f"Total raw tiles: {len(raw_files)}")

skipped = []
to_enrich = []

for raw_file in raw_files:
    enriched_file = enriched_dir / raw_file.name
    if enriched_file.exists():
        skipped.append(raw_file.name)
    else:
        to_enrich.append(raw_file.name)

print(f"Will skip: {len(skipped)} tiles (already enriched)")
print(f"Will enrich: {len(to_enrich)} tiles")
```

### "Changed mode, want to re-enrich only affected tiles"

If you changed from core to building mode, you should force re-enrichment:

```bash
# Option 1: Process to new directory (recommended)
ign-lidar enrich --input-dir raw/ --output enriched_building/ --mode building

# Option 2: Force re-enrich in same directory
ign-lidar enrich --input-dir raw/ --output enriched/ --mode building --force
```

## Comparison with Other Skip Features

| Feature              | Download Skip      | Processing Skip       | **Enrichment Skip**    |
| -------------------- | ------------------ | --------------------- | ---------------------- |
| **What it checks**   | LAZ file in output | Patches in output     | Enriched LAZ in output |
| **Skip condition**   | File exists        | Any patch exists      | Enriched file exists   |
| **Performance gain** | ~45s per tile      | ~35s per tile         | ~2-5 min per tile      |
| **Use case**         | Resume downloads   | Resume patch creation | Resume enrichment      |

All three work together for complete workflow resumption:

```bash
# Download (skips existing)
ign-lidar download --bbox ... --output tiles/

# Enrich (skips existing)  ← NEW!
ign-lidar enrich --input-dir tiles/ --output enriched/ --mode building

# Process (skips existing)
ign-lidar process --input-dir enriched/ --output patches/
```

## See Also

- [Skip Existing Tiles (Download)](SKIP_EXISTING_TILES.md)
- [Skip Existing Patches (Processing)](SKIP_EXISTING_PATCHES.md)
- [Output Format Preferences](OUTPUT_FORMAT_PREFERENCES.md)
- [Smart Skip Features Summary](SMART_SKIP_FEATURES_SUMMARY.md)
- [CLI Module](../ign_lidar/cli.py)
