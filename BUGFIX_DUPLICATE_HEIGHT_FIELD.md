# Bug Fix: Duplicate 'height' Field in LAZ Patches

## Problem

When saving patch LAZ files, the system was generating warnings:

```
⚠️  Could not add feature 'height' to LAZ: field 'height' occurs more than once
⚠️  Could not add NDVI to LAZ: field 'height' occurs more than once
```

This occurred hundreds of times (once per patch), indicating that certain features were being added multiple times to the LAZ extra dimensions.

## Root Cause

The issue had two related causes:

1. **Processing enriched LAZ files**: When an already-enriched LAZ file is used as input (e.g., `LHD_FXX_0635_6857_PTS_C_LAMB93_IGN69_enriched.laz`), it already contains extra dimensions like 'height', 'ndvi', etc. When loading this file, these extra dimensions are loaded into the `original_patch` dictionary.

2. **Duplicate feature addition**: When saving patches as LAZ files, the code had multiple sections that could add the same feature:
   - First loop: Iterates through ALL features in `original_patch` and adds them as extra dimensions
   - Explicit sections: Then explicitly tries to add 'height', 'ndvi', normals, etc. again

This resulted in attempting to add the same extra dimension twice, causing laspy to raise an exception.

## Solution

Modified both `serialization.py` and `processor.py` to track which extra dimensions have already been added using a `set`:

### Changes in `serialization.py`:

1. **Added dimension tracking**: Introduced `added_dimensions = set()` to track which fields have been added
2. **Updated function signatures**: Added `added_dimensions` parameter to helper functions:
   - `_add_geometric_features(las, original_patch, added_dimensions)`
   - `_add_height_features(las, original_patch, added_dimensions)`
   - `_add_radiometric_features(las, original_patch, point_format, added_dimensions)`
   - `_add_return_number(las, original_patch, added_dimensions)`
3. **Duplicate checking**: Before adding any extra dimension, check if it's already in `added_dimensions`
4. **Tracking updates**: After successfully adding a dimension, add it to the set

### Changes in `processor.py`:

Applied the same pattern in the `_save_patch_as_laz` method:

1. Introduced `added_dimensions = set()` tracking
2. Added dimension checking before each `add_extra_dim` call
3. Updated the set after successful additions

## Code Pattern

The fix follows this pattern throughout:

```python
# Before (would fail if dimension already exists)
las.add_extra_dim(laspy.ExtraBytesParams(name=feat_name, type=np.float32))
setattr(las, feat_name, data)

# After (checks and tracks)
if feat_name not in added_dimensions:
    las.add_extra_dim(laspy.ExtraBytesParams(name=feat_name, type=np.float32))
    setattr(las, feat_name, data)
    added_dimensions.add(feat_name)
```

## Impact

- ✅ Eliminates duplicate field warnings
- ✅ Ensures each feature is added exactly once
- ✅ Handles both fresh LAZ files and already-enriched LAZ files correctly
- ✅ No functional changes - same features are saved, just without duplicates
- ✅ Improves processing efficiency by avoiding exception handling overhead

## Testing

To verify the fix works:

1. Run processing on an already-enriched LAZ file
2. Check that no duplicate field warnings appear
3. Verify that patch LAZ files contain all expected features
4. Confirm that features are not missing or corrupted

## Files Modified

- `ign_lidar/core/modules/serialization.py`
- `ign_lidar/core/processor.py`

## Date

2025-10-12
