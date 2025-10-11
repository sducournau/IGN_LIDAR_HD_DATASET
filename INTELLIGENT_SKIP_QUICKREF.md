# Intelligent Skip - Quick Reference

## TL;DR

The intelligent skip system automatically detects and skips processing of tiles that have already been processed, with deep validation and partial skip support.

## Quick Examples

### Skip Completely (Default)

```bash
# First run
ign-lidar-hd process --input data/ --output output/ --mode both
# Processes all tiles

# Second run - automatically skips
ign-lidar-hd process --input data/ --output output/ --mode both
# ⏭️ All tiles skipped (~1800x faster)
```

### Partial Skip (Smart Recovery)

```bash
# Generate patches only
ign-lidar-hd process --input data/ --output output/ --mode patches_only

# Later, add enriched LAZ without regenerating patches
ign-lidar-hd process --input data/ --output output/ --mode both
# ⏭️ Patches: skipped
# 🔄 Enriched LAZ: processing (~2x faster)
```

### Force Reprocess

```python
from ign_lidar.core.processor import LiDARProcessor

processor = LiDARProcessor(processing_mode='both')
result = processor.process_tile(
    laz_file, output_dir,
    skip_existing=False  # Force reprocess
)
```

## Skip Decision Matrix

| Enriched LAZ | Patches    | Action                       | Reason        |
| ------------ | ---------- | ---------------------------- | ------------- |
| ✅ Valid     | ✅ Valid   | ⏭️ **SKIP ALL**              | Both complete |
| ✅ Valid     | ❌ Missing | 🔄 **Process patches only**  | Partial       |
| ❌ Missing   | ✅ Valid   | 🔄 **Process enriched only** | Partial       |
| ❌ Invalid   | ✅ Valid   | 🔄 **Process enriched only** | Recovery      |
| ✅ Valid     | ❌ Invalid | 🔄 **Process patches only**  | Recovery      |
| ❌ Missing   | ❌ Missing | 🔄 **Process all**           | Fresh         |

## Validation Checks

### Patches

- ✅ File exists & size > 1KB
- ✅ Loadable (not corrupted)
- ✅ Has coords/points & labels
- ✅ Arrays not empty
- ✅ Dimensions match

### Enriched LAZ

- ✅ File exists & size > 1KB
- ✅ Loadable (not corrupted)
- ✅ Core features: normals, curvature, height
- ✅ Optional features match config:
  - RGB (if `include_rgb=True`)
  - NIR (if `include_infrared=True`)
  - NDVI (if `compute_ndvi=True`)
  - Geometric (if `include_extra_features=True`)

## Skip Messages

| Message                                | Meaning                      |
| -------------------------------------- | ---------------------------- |
| ⏭️ Both enriched LAZ and patches exist | Complete skip                |
| ⏭️ Valid enriched LAZ exists           | Skip (enriched_only mode)    |
| ⏭️ Patches exist and valid             | Skip (patches_only mode)     |
| 🔄 Enriched exists but no patches      | Process patches              |
| 🔄 Patches exist but no enriched       | Process enriched LAZ         |
| 🔄 Enriched LAZ invalid                | Reprocess (missing features) |
| 🔄 Corrupted patches detected          | Reprocess (file errors)      |
| 🔄 No outputs found                    | Process all                  |

## Performance

| Scenario                    | Time      | vs Fresh         |
| --------------------------- | --------- | ---------------- |
| Fresh processing            | 180s/tile | 1x               |
| Complete skip               | 0.1s/tile | **1800x faster** |
| Partial skip (add enriched) | 90s/tile  | **2x faster**    |
| Partial skip (add patches)  | 108s/tile | **1.7x faster**  |

## Common Use Cases

### 1. Resume After Crash

```bash
# Crashed at tile 50/100
ign-lidar-hd process --input data/ --output output/
# ⏭️ Tiles 1-50 skipped automatically
# 🔄 Tiles 51-100 processed
```

### 2. Configuration Changes

```bash
# Change patch size (patches need regeneration)
ign-lidar-hd process --input data/ --output output/ \
  --patch-size 200  # was 150
# 🔄 Patches: regenerated (different size)
# ⏭️ Enriched LAZ: skipped (still valid)
```

### 3. Add New Tiles

```bash
# Add tiles to existing output directory
ign-lidar-hd process --input data/ --output output/
# ⏭️ Existing tiles: skipped
# 🔄 New tiles: processed
```

### 4. Debug Single Tile

```python
# Test specific tile without reprocessing all
processor.process_tile(
    Path("data/problem_tile.laz"),
    Path("output/"),
    skip_existing=False  # Force reprocess this one
)
```

## Troubleshooting

### Tiles Not Skipping?

**Check:**

1. Output directory path correct?
2. Architecture/format changed?
3. Validation failing? (check logs)

**Debug:**

```python
checker = processor.skip_checker
should_skip, info = checker.should_skip_tile(
    tile_path, output_dir,
    save_enriched=True,
    include_rgb=True,
    # ... match config ...
)
print(f"Skip: {should_skip}")
print(f"Reason: {info['reason']}")
```

### Partial Skip Not Working?

**Common Issue**: Feature mismatch

```
🔄 tile.laz: Enriched LAZ invalid (Missing features: ['ndvi'])
```

**Fix**: Match configuration

```python
processor = LiDARProcessor(
    compute_ndvi=True,  # Add missing feature
    # ... other params ...
)
```

## Configuration

### Processing Modes

```python
# Only patches (default, fastest for training)
processor = LiDARProcessor(processing_mode='patches_only')

# Only enriched LAZ (fastest for GIS)
processor = LiDARProcessor(processing_mode='enriched_only')

# Both (enables partial skip)
processor = LiDARProcessor(processing_mode='both')
```

### Skip Control

```python
# Enable skip (default)
result = processor.process_tile(laz_file, output_dir, skip_existing=True)

# Disable skip (force reprocess)
result = processor.process_tile(laz_file, output_dir, skip_existing=False)
```

### Skip Checker Options

```python
from ign_lidar.core.skip_checker import PatchSkipChecker

checker = PatchSkipChecker(
    output_format='npz',           # Match your format
    architecture='pointnet++',      # Match your architecture
    augment=True,                   # Augmentation enabled?
    num_augmentations=3,            # How many versions?
    validate_content=True,          # Deep validation (recommended)
    min_file_size=1024,             # Minimum valid file size
    only_enriched_laz=False,        # Check enriched only?
)
```

## API Quick Reference

### Check Skip Decision

```python
should_skip, info = checker.should_skip_tile(
    tile_path=Path("tile.laz"),
    output_dir=Path("output/"),
    save_enriched=True,
    include_rgb=True,
    include_infrared=False,
    compute_ndvi=False,
    include_extra_features=True
)
```

### Format Skip Message

```python
msg = checker.format_skip_message(tile_path, info)
print(msg)
# ⏭️ tile.laz: Both enriched LAZ (125.3 MB) and 24 patches exist, skipping
```

### Validate Enriched LAZ

```python
is_valid, info = checker._validate_enriched_laz(
    enriched_path,
    include_rgb=True,
    include_infrared=False,
    compute_ndvi=False,
    include_extra_features=True
)
```

## Best Practices

### ✅ DO

- Use `skip_existing=True` (default) in production
- Use `processing_mode='both'` for partial skip benefits
- Check logs for skip reasons
- Validate configuration matches expected features

### ❌ DON'T

- Use `skip_existing=False` unless debugging
- Change output directory unnecessarily
- Ignore validation warnings in logs

## See Also

- Full documentation: [`docs/INTELLIGENT_SKIP.md`](docs/INTELLIGENT_SKIP.md)
- Implementation details: [`INTELLIGENT_SKIP_IMPLEMENTATION.md`](INTELLIGENT_SKIP_IMPLEMENTATION.md)
- Skip checker source: [`ign_lidar/core/skip_checker.py`](ign_lidar/core/skip_checker.py)
