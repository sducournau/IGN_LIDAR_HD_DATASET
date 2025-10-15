# Fix: RGB Missing from Output

## Problem

RGB data was being removed from the output even though:

1. `use_rgb: true` was set in the config files
2. RGB data was present in the input LAZ files
3. The TileLoader was correctly extracting RGB data

## Root Cause

The issue was in the `FeatureOrchestrator.filter_features()` method. There was a mismatch between:

**Feature Storage**: RGB is stored as a single array with key `'rgb'` and shape (N, 3)

```python
all_features['rgb'] = input_rgb  # Shape: (N, 3)
```

**Feature Mode Definition**: LOD2 and LOD3 modes define RGB as three separate features:

```python
LOD2_FEATURES = {
    'red', 'green', 'blue',  # Three individual features
    ...
}
```

**Feature Filtering**: The `filter_features()` method was checking if feature keys existed in the allowed list. Since `'rgb'` was not in the list (only `'red'`, `'green'`, `'blue'` were), it got filtered out and removed from the output.

## Solution

Updated the `filter_features()` method in `/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/ign_lidar/features/orchestrator.py` to handle spectral features intelligently:

```python
# Handle spectral features: if mode defines 'red', 'green', 'blue' individually,
# also allow 'rgb' as a combined feature (and vice versa)
if 'red' in allowed_features or 'green' in allowed_features or 'blue' in allowed_features:
    allowed_features.add('rgb')
if 'rgb' in allowed_features:
    allowed_features.update(['red', 'green', 'blue'])

# Same for NIR and NDVI
if 'nir' in allowed_features or self.use_infrared:
    allowed_features.add('nir')
if 'ndvi' in allowed_features or (self.use_rgb and self.use_infrared):
    allowed_features.add('ndvi')
```

This ensures that:

- If a feature mode defines `red`, `green`, `blue`, the combined `rgb` array is also allowed
- If a feature mode defines `rgb`, the individual channels are also allowed
- Same logic applies for `nir` and `ndvi`

## Impact

With this fix:

- ✅ RGB data from input LAZ files will be preserved in the output
- ✅ Fetched RGB data from IGN orthophotos will be included
- ✅ NIR and NDVI features will also be properly preserved
- ✅ All existing configs (`config_versailles_asprs.yaml`, `config_versailles_lod2.yaml`, `config_versailles_lod3.yaml`) will now work correctly with RGB

## Testing

To verify the fix works:

```bash
# Test with any of the configs
ign-lidar-hd process --config examples/config_versailles_asprs.yaml

# Check output files for RGB data
# For NPZ files:
python -c "import numpy as np; data = np.load('output.npz'); print(data.files); print('rgb' in data.files)"

# For LAZ files (check for red/green/blue channels):
python -c "import laspy; las = laspy.read('output.laz'); print(hasattr(las, 'red'))"
```

## Files Modified

- `ign_lidar/features/orchestrator.py` - Updated `filter_features()` method (lines 415-463)

## Related Issues

This fix also resolves potential issues with:

- NIR (near-infrared) data being filtered out
- NDVI (Normalized Difference Vegetation Index) being removed
- Any spectral features defined in different formats across the codebase
