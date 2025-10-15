# RGB Normalization Fix - All White Point Clouds

## Problem

When preprocessing ASPRS tiles with RGB augmentation enabled, the saved LAZ files had RGB colors but all points appeared white in visualization tools.

## Root Cause

The issue was a **normalization mismatch** between RGB fetching and serialization:

1. **RGB Fetcher** (`IGNOrthophotoFetcher.augment_points_with_rgb`):

   - Returns RGB values in range **[0, 255]** (uint8)

2. **Feature Orchestrator** (`FeatureOrchestrator._add_rgb_features`):

   - Was storing RGB directly without normalization
   - RGB stored in features dict in **[0, 255]** range

3. **LAZ Serialization** (`save_enriched_tile_laz`):
   - Expects RGB in range **[0, 1]** (normalized float32)
   - Multiplies by 65535.0 to convert to uint16 for LAZ format
   - When given [0, 255] values: `255 * 65535.0 = 16,711,425`
   - This overflows uint16 (max 65535), resulting in white (65535, 65535, 65535)

## Fix Applied

### File: `ign_lidar/features/orchestrator.py`

#### RGB Fix (line ~697)

```python
# BEFORE (incorrect):
rgb = self.rgb_fetcher.augment_points_with_rgb(points)
if rgb is not None:
    all_features['rgb'] = rgb  # [0, 255] range - WRONG!

# AFTER (correct):
rgb = self.rgb_fetcher.augment_points_with_rgb(points)
if rgb is not None:
    # Normalize RGB from [0, 255] to [0, 1] for consistency
    all_features['rgb'] = rgb.astype(np.float32) / 255.0
```

#### NIR Fix (line ~736)

Similar fix applied for NIR (near-infrared) values:

```python
# BEFORE (incorrect):
nir = self.infrared_fetcher.augment_points_with_infrared(points)
if nir is not None:
    all_features['nir'] = nir  # [0, 255] range - WRONG!

# AFTER (correct):
nir = self.infrared_fetcher.augment_points_with_infrared(points)
if nir is not None:
    # Normalize NIR from [0, 255] to [0, 1] for consistency
    all_features['nir'] = nir.astype(np.float32) / 255.0
```

## Verification

After applying the fix, RGB values should be correctly normalized:

1. **Feature Storage**: RGB/NIR stored as float32 in [0, 1] range
2. **LAZ Serialization**: `value * 65535.0` produces correct uint16 values
3. **Visualization**: Point clouds display with correct colors

## Impact

- **Fixed**: RGB and NIR augmentation from IGN orthophotos
- **Affected Files**:
  - `ign_lidar/features/orchestrator.py` (2 changes - RGB and NIR)
  - `ign_lidar/cli/commands/ground_truth.py` (2 changes + numpy import)
  - `ign_lidar/cli/commands/update_classification.py` (2 changes)
- **Already Correct**:
  - `ign_lidar/core/modules/enrichment.py` (already had normalization)
- **Required Action**: Reprocess any tiles that were preprocessed with the bug
  - Old tiles will have all-white RGB
  - New tiles will have correct RGB colors## Testing

To verify the fix works:

```bash
# Run preprocessing on a test tile
ign-lidar-hd process \
    --config configs/multiscale/config_unified_asprs_preprocessing.yaml

# Open output LAZ in CloudCompare/QGIS and verify RGB colors are correct
# Should see actual terrain colors, not all white
```

## Related Files

- `ign_lidar/preprocessing/rgb_augmentation.py` - RGB fetcher (returns [0, 255])
- `ign_lidar/preprocessing/infrared_augmentation.py` - NIR fetcher (returns [0, 255])
- `ign_lidar/core/modules/serialization.py` - LAZ writer (expects [0, 1])
- `ign_lidar/features/orchestrator.py` - **Fixed here** (now normalizes to [0, 1])

## Date

October 15, 2025
