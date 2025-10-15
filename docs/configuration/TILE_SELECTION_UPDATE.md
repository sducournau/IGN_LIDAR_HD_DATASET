# Tile Selection Update: Duplicate Prevention

**Date**: October 16, 2025  
**Purpose**: Prevent duplicate tiles across ASPRS, LOD2, and LOD3 selections

## Problem

Previously, the same tiles could be selected for multiple classification levels (ASPRS, LOD2, LOD3), leading to:

- Data leakage between training sets
- Inefficient use of available tiles
- Potential overfitting on specific geographic areas

## Solution

Updated `scripts/select_optimal_tiles.py` to ensure **zero overlap** between tile selections:

### 1. Sequential Selection with Exclusion

```python
# Selection order: ASPRS → LOD2 → LOD3
selection_order = ['asprs', 'lod2', 'lod3']
already_selected = set()  # Track selected tiles

for level in selection_order:
    # Filter out already selected tiles
    available_tiles = [t for t in tiles if t['file_name'] not in already_selected]

    # Select from available pool
    selected_tiles = select_tiles_by_strategy(available_tiles, count, strategy)

    # Mark as selected
    for tile in selected_tiles:
        already_selected.add(tile['file_name'])
```

### 2. Automatic Duplicate Detection

```python
# Verify no duplicates after selection
all_tiles = asprs_tiles + lod2_tiles + lod3_tiles
unique_tiles = set(all_tiles)
has_duplicates = len(all_tiles) != len(unique_tiles)

# Log warning if duplicates found
if has_duplicates:
    logger.warning("⚠️  WARNING: Found duplicates!")
```

### 3. Enhanced Reporting

New summary includes:

- `duplicate_detection.has_duplicates`: Boolean flag
- `duplicate_detection.total_selections`: Total tiles selected
- `duplicate_detection.unique_tiles`: Number of unique tiles
- `duplicate_detection.duplicates_count`: Number of duplicates (should be 0)

## Usage

### Basic Usage

```bash
python scripts/select_optimal_tiles.py \
    --input /mnt/d/ign/unified_dataset \
    --analysis /mnt/d/ign/analysis_report.json \
    --output /mnt/d/ign/tile_selections \
    --asprs-count 100 \
    --lod2-count 80 \
    --lod3-count 60 \
    --strategy best
```

### Expected Output

```
Selecting 100 tiles for ASPRS...
  Available tiles (excluding duplicates): 500 / 500
  Selected 100 unique tiles for ASPRS

Selecting 80 tiles for LOD2...
  Available tiles (excluding duplicates): 400 / 500
  Selected 80 unique tiles for LOD2

Selecting 60 tiles for LOD3...
  Available tiles (excluding duplicates): 320 / 500
  Selected 60 unique tiles for LOD3

✓ No duplicate tiles found across all levels

============================================================
Selection Complete!
============================================================
Tile counts:
  ASPRS: 100 tiles
  LOD2:  80 tiles
  LOD3:  60 tiles
  Total: 240 tiles
  Unique: 240 tiles

Duplicate check: ✓ PASS
============================================================
```

## Selection Strategy

### 1. Priority Order

**ASPRS First** (100 tiles)

- Largest selection
- Gets first pick of best tiles
- Requires diverse classes for multi-class classification

**LOD2 Second** (80 tiles)

- Medium selection
- Focuses on building-rich areas
- Excludes ASPRS tiles

**LOD3 Last** (60 tiles)

- Smallest selection
- Focuses on detailed architecture
- Excludes ASPRS + LOD2 tiles

### 2. Why This Order?

1. **ASPRS needs diversity**: Multi-class classification (ground, buildings, vegetation, roads, etc.) requires broad geographic coverage
2. **LOD2 needs buildings**: Can still get good building-rich areas after ASPRS selection
3. **LOD3 needs detail**: Smallest selection, can work with remaining high-quality tiles

### 3. Alternative Order (if needed)

To prioritize LOD3 (detailed architecture), change order:

```python
selection_order = ['lod3', 'lod2', 'asprs']  # LOD3 gets first pick
```

## File Outputs

### 1. Tile Lists (one per level)

**`asprs_selected_tiles.txt`**

```
LHD_FXX_0759_6274_PTS_C_LAMB93_IGN69.laz
LHD_FXX_0760_6275_PTS_C_LAMB93_IGN69.laz
...
```

**`lod2_selected_tiles.txt`**

```
LHD_FXX_0761_6276_PTS_C_LAMB93_IGN69.laz
LHD_FXX_0762_6277_PTS_C_LAMB93_IGN69.laz
...
```

**`lod3_selected_tiles.txt`**

```
LHD_FXX_0763_6278_PTS_C_LAMB93_IGN69.laz
LHD_FXX_0764_6279_PTS_C_LAMB93_IGN69.laz
...
```

### 2. Selection Details (one per level)

**`asprs_selection_details.json`**

```json
{
  "level": "asprs",
  "requested_count": 100,
  "selected_count": 100,
  "strategy": "best",
  "tiles": [
    {
      "file_name": "LHD_FXX_0759_6274_PTS_C_LAMB93_IGN69.laz",
      "success": true,
      "point_count": 8543210,
      "density_pts_m2": 12.5,
      "classification_distribution": {...}
    },
    ...
  ]
}
```

### 3. Selection Summary

**`selection_summary.json`**

```json
{
  "input_dataset": "/mnt/d/ign/unified_dataset",
  "analysis_report": "/mnt/d/ign/analysis_report.json",
  "strategy": "best",
  "duplicate_detection": {
    "has_duplicates": false,
    "total_selections": 240,
    "unique_tiles": 240,
    "duplicates_count": 0
  },
  "selection": {
    "asprs": {
      "requested": 100,
      "selected": 100,
      "tile_names": [...]
    },
    "lod2": {
      "requested": 80,
      "selected": 80,
      "tile_names": [...]
    },
    "lod3": {
      "requested": 60,
      "selected": 60,
      "tile_names": [...]
    }
  }
}
```

## Verification

### Check for Duplicates Manually

```bash
# Concatenate all tile lists
cat asprs_selected_tiles.txt lod2_selected_tiles.txt lod3_selected_tiles.txt > all_tiles.txt

# Check for duplicates
sort all_tiles.txt | uniq -d

# Should return nothing if no duplicates
```

### Count Unique Tiles

```bash
# Count total lines
wc -l all_tiles.txt

# Count unique lines
sort all_tiles.txt | uniq | wc -l

# Should be the same
```

### Python Verification

```python
import json

# Load summary
with open('selection_summary.json') as f:
    summary = json.load(f)

# Check duplicate detection
dup_check = summary['duplicate_detection']
print(f"Has duplicates: {dup_check['has_duplicates']}")
print(f"Total selections: {dup_check['total_selections']}")
print(f"Unique tiles: {dup_check['unique_tiles']}")
print(f"Duplicates: {dup_check['duplicates_count']}")

# Should show: has_duplicates=False, duplicates_count=0
```

## Troubleshooting

### Issue 1: Not Enough Tiles

**Symptoms**:

```
Selecting 80 tiles for LOD2...
  Available tiles (excluding duplicates): 20 / 500
  Selected 20 unique tiles for LOD2
```

**Cause**: Too many tiles requested for ASPRS, leaving few for LOD2/LOD3

**Solution**: Reduce counts or change selection order

```bash
# Reduce ASPRS count
--asprs-count 80 --lod2-count 80 --lod3-count 60

# OR change selection order in code
selection_order = ['lod3', 'lod2', 'asprs']
```

### Issue 2: Duplicates Still Found

**Symptoms**:

```
✗ Duplicate check: FAIL
WARNING: Found 5 duplicate tiles!
```

**Cause**: Bug in duplicate detection logic

**Solution**: Check the `already_selected` set is working correctly

```python
# Add debug logging
logger.debug(f"Already selected: {len(already_selected)} tiles")
logger.debug(f"Selecting from: {len(available_tiles)} available tiles")
```

### Issue 3: Low Quality Tiles Selected

**Symptoms**: LOD3 gets poor quality tiles because best tiles taken by ASPRS

**Solution**:

1. Change selection order to prioritize LOD3
2. Use 'diverse' strategy instead of 'best'
3. Increase tile pool size

```bash
# Use diverse strategy to spread quality across levels
--strategy diverse
```

## Integration with Pipeline

### 1. Run Analysis

```bash
python scripts/analyze_unified_dataset.py \
    --input /mnt/d/ign/unified_dataset \
    --output /mnt/d/ign/analysis_report.json
```

### 2. Select Tiles (NEW - with duplicate prevention)

```bash
python scripts/select_optimal_tiles.py \
    --input /mnt/d/ign/unified_dataset \
    --analysis /mnt/d/ign/analysis_report.json \
    --output /mnt/d/ign/tile_selections \
    --asprs-count 100 \
    --lod2-count 80 \
    --lod3-count 60 \
    --strategy best
```

### 3. Copy Selected Tiles

```bash
# ASPRS
mkdir -p /mnt/d/ign/selected_tiles/asprs/tiles
while read tile; do
    cp "/mnt/d/ign/unified_dataset/asprs/tiles/$tile" \
       "/mnt/d/ign/selected_tiles/asprs/tiles/"
done < /mnt/d/ign/tile_selections/asprs_selected_tiles.txt

# LOD2
mkdir -p /mnt/d/ign/selected_tiles/lod2/tiles
while read tile; do
    cp "/mnt/d/ign/unified_dataset/lod2/tiles/$tile" \
       "/mnt/d/ign/selected_tiles/lod2/tiles/"
done < /mnt/d/ign/tile_selections/lod2_selected_tiles.txt

# LOD3
mkdir -p /mnt/d/ign/selected_tiles/lod3/tiles
while read tile; do
    cp "/mnt/d/ign/unified_dataset/lod3/tiles/$tile" \
       "/mnt/d/ign/selected_tiles/lod3/tiles/"
done < /mnt/d/ign/tile_selections/lod3_selected_tiles.txt
```

### 4. Verify No Duplicates

```bash
# Quick check
python -c "
import json
with open('/mnt/d/ign/tile_selections/selection_summary.json') as f:
    summary = json.load(f)
    dup_check = summary['duplicate_detection']
    if not dup_check['has_duplicates']:
        print('✓ No duplicates - safe to proceed')
    else:
        print(f'✗ Found {dup_check[\"duplicates_count\"]} duplicates!')
        exit(1)
"
```

### 5. Run Preprocessing

```bash
# Now safe to run preprocessing on each level
ign-lidar-hd process --config config_asprs_preprocessing.yaml
ign-lidar-hd process --config config_lod2_preprocessing.yaml
ign-lidar-hd process --config config_lod3_preprocessing.yaml
```

## Benefits

✅ **Zero data leakage** - Each level uses completely different tiles  
✅ **Efficient tile usage** - No wasted processing on duplicates  
✅ **Better generalization** - Models train on diverse geographic areas  
✅ **Automatic validation** - Built-in duplicate detection and reporting  
✅ **Clear tracking** - Summary shows exactly which tiles go where  
✅ **Flexible strategy** - Easy to adjust selection order or counts

## Migration from Old System

If you have existing tile selections with potential duplicates:

### 1. Check Current Selections

```bash
# Find duplicates in current selection
cat /mnt/d/ign/selected_tiles/*/tiles/*.laz | \
    xargs -n1 basename | \
    sort | uniq -d
```

### 2. Re-run Selection

```bash
# Delete old selections
rm -rf /mnt/d/ign/selected_tiles/*

# Run new selection (with duplicate prevention)
python scripts/select_optimal_tiles.py \
    --input /mnt/d/ign/unified_dataset \
    --analysis /mnt/d/ign/analysis_report.json \
    --output /mnt/d/ign/tile_selections \
    --asprs-count 100 \
    --lod2-count 80 \
    --lod3-count 60 \
    --strategy best

# Copy new selections (see Integration section above)
```

### 3. Verify Clean State

```bash
# Verify no duplicates in new setup
python -c "
import json
with open('/mnt/d/ign/tile_selections/selection_summary.json') as f:
    summary = json.load(f)
    print(f'Duplicate check: {\"PASS\" if not summary[\"duplicate_detection\"][\"has_duplicates\"] else \"FAIL\"}')
"
```

## Summary

The updated tile selection system ensures:

1. **No overlapping tiles** across ASPRS, LOD2, and LOD3
2. **Automatic duplicate detection** and reporting
3. **Clear priority order** with configurable counts
4. **Comprehensive logging** and validation
5. **Easy integration** with existing pipeline

All tile selections are now guaranteed to be unique, preventing data leakage and ensuring proper model evaluation.
