# Quick Reference: ASPRS Classes in Your Tiles

## Standard ASPRS Classes Found

| Code   | Name              | Tile 1    | Tile 2    | Description               |
| ------ | ----------------- | --------- | --------- | ------------------------- |
| 1      | Unclassified      | 3.67%     | 5.34%     | Points not yet classified |
| 2      | Ground            | 43.79%    | 51.51%    | Ground surface            |
| 3      | Low Vegetation    | 1.30%     | 1.08%     | Vegetation < 0.5m         |
| 4      | Medium Vegetation | 2.78%     | 1.99%     | Vegetation 0.5-2.0m       |
| 5      | High Vegetation   | 37.21%    | 20.75%    | Vegetation > 2.0m         |
| 6      | Building          | 11.23%    | 19.28%    | Building structures       |
| 9      | Water             | -         | 0.00%     | Water surfaces            |
| 17     | Bridge Deck       | -         | 0.00%     | Bridge structures         |
| **67** | **Unknown**       | **0.02%** | **0.05%** | **⚠️ NON-STANDARD**       |

## BD TOPO® Extended Classes (Not Yet Applied)

These will be added during enrichment when data sources are enabled:

| Code | Name            | Source   | Description          |
| ---- | --------------- | -------- | -------------------- |
| 10   | Rail            | BD TOPO® | Railway tracks       |
| 11   | Road Surface    | BD TOPO® | Road surfaces        |
| 40   | Parking         | BD TOPO® | Parking areas        |
| 41   | Sports Facility | BD TOPO® | Sports facilities    |
| 42   | Cemetery        | BD TOPO® | Cemeteries           |
| 43   | Power Line      | BD TOPO® | Power lines          |
| 44   | Agriculture     | RPG      | Agricultural parcels |

## Configuration Status

```yaml
# Current configuration (enrichment_asprs_full.yaml)
preprocess:
  normalize_classification: true # ✅ ACTIVE
  strict_class_normalization: false # Only fix known issues

data_sources:
  bd_topo:
    enabled: true # ❌ Currently disabled in your run
  rpg:
    enabled: true # ❌ Currently disabled in your run
  cadastre:
    enabled: true # ❌ Currently disabled in your run
```

## Class 67 Fix

**Problem**: Class 67 is not part of ASPRS LAS 1.4 specification

**Solution**: Automatically remapped to Class 1 (Unclassified)

**Impact**:

- Tile 1: 4,380 points (0.02%)
- Tile 2: 8,032 points (0.05%)

**Status**: ✅ Fixed automatically during tile loading

## Verification Commands

Check input tiles:

```bash
python check_current_classes.py
```

Check individual tile:

```bash
python check_laz_classification.py /path/to/tile.laz
```

After processing:

```bash
python check_laz_classification.py /mnt/d/ign/preprocessed/asprs/enriched_tiles/*.laz
```

## Expected Output After Processing

**Before**: Classes [1, 2, 3, 4, 5, 6, 67]
**After**: Classes [1, 2, 3, 4, 5, 6] ✅

Class 67 points merged into Class 1 (Unclassified)
