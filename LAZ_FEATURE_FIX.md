# LAZ Feature Missing - Issue Analysis & Fix

## Issue Description

When using `output=both` or `output=enriched_only` mode (especially with `multi` architecture), the enriched LAZ files are created but **missing all computed features** (normals, curvature, planarity, etc.).

## Root Causes Identified

### 1. Silent Failure in Feature Addition

**Location**: `ign_lidar/core/processor.py` lines 1659-1735

**Problem**: The entire feature-adding code is wrapped in a try-except block that only logs a warning if ANY exception occurs:

```python
try:
    # Add all features...
    new_las.add_extra_dim(...)
    new_las.write(enriched_path)
    # ...
except Exception as e:
    logger.warning(f"  ⚠️  Failed to save enriched LAZ: {e}")
```

This means:

- If ANY exception occurs during feature addition, the LAZ is written WITHOUT features
- Only a vague warning is logged
- No traceback or detailed error information
- Processing continues as if nothing happened

### 2. Potential Data Type Mismatches

Features computed by GPU/CPU processing might have different data types (float64, float32, etc.), and laspy `ExtraBytesParams` expects specific types. Without explicit type conversion, this can cause silent failures.

### 3. Size Mismatch Issues

After preprocessing (outlier removal, filtering), the point arrays might have different sizes than the computed features. The code didn't validate sizes before adding features.

### 4. Lack of Verification

After writing the LAZ file, there was no verification that:

- The file was actually written
- Features were successfully persisted to disk
- The file can be read back with features intact

## The Fix

### Changes Made to `ign_lidar/core/processor.py`

1. **Added Data Validation**

   ```python
   # Validate data sizes before adding features
   expected_size = len(points)
   if len(normals) != expected_size:
       raise ValueError(f"Normals size mismatch: {len(normals)} != {expected_size}")
   ```

2. **Explicit Type Conversion**

   ```python
   # Force float32 for all features
   new_las.normal_x = normals[:, 0].astype(np.float32)
   new_las.curvature = curvature.astype(np.float32)
   ```

3. **Per-Feature Size Validation**

   ```python
   # Check each feature individually
   for feature_name, feature_values in geo_features.items():
       if len(feature_values) != expected_size:
           logger.warning(f"⚠️  Skipping feature {feature_name}: size mismatch")
           continue
   ```

4. **Post-Write Verification**

   ```python
   # Re-read file to verify features were saved
   verify_las = laspy.read(str(enriched_path))
   verify_extra_dims = verify_las.point_format.extra_dimension_names

   if len(verify_extra_dims) == 0:
       logger.error("❌ CRITICAL: LAZ written but contains NO extra dimensions!")
   ```

5. **Enhanced Error Logging**

   ```python
   except Exception as e:
       logger.error(f"❌ Failed to save enriched LAZ: {e}")
       logger.error(f"   Error type: {type(e).__name__}")
       logger.error(f"   File: {enriched_path}")
       import traceback
       logger.error(f"   Traceback:\n{traceback.format_exc()}")
   ```

6. **Debug Logging**
   - Added debug messages for each feature group added
   - Logs total count of extra dimensions
   - Lists all extra dimension names

## Diagnostic Tool

Created `debug_laz_features.py` to check LAZ files for features:

```bash
# Check a single file
python debug_laz_features.py output/enriched/tile_1234_5678_enriched.laz

# Check all files in a directory
python debug_laz_features.py output/enriched/
```

This tool:

- Lists all standard dimensions (X, Y, Z, intensity, RGB, etc.)
- Lists all extra dimensions (computed features)
- Shows value ranges for each feature
- Identifies missing expected features
- Provides detailed diagnostics

## Testing the Fix

### 1. Test with Single Architecture

```bash
ign-lidar-hd process \
  input_dir=data/raw/ \
  output_dir=output/test_single/ \
  output=both \
  processor.use_gpu=false
```

### 2. Test with Multi Architecture

```bash
ign-lidar-hd process \
  input_dir=data/raw/ \
  output_dir=output/test_multi/ \
  output=both \
  processor.architecture=multi \
  processor.use_gpu=false
```

### 3. Verify Features

```bash
# Check enriched LAZ files
python debug_laz_features.py output/test_single/enriched/

# Expected features:
# - normal_x, normal_y, normal_z
# - curvature
# - height
# - planarity, linearity, sphericity, verticality (if include_extra=true)
# - RGB (if available)
# - NIR, NDVI (if available)
```

## Expected Behavior After Fix

1. **All features should be present** in enriched LAZ files
2. **Detailed error messages** if something goes wrong
3. **Verification warnings** if features fail to persist
4. **No silent failures**

## Common Issues and Solutions

### Issue: "Size mismatch" warnings

**Cause**: Preprocessing filters points, but features were computed on original data

**Solution**: Ensure features are computed AFTER preprocessing (already implemented)

### Issue: "No geometric features (geo_features is None)"

**Cause**: `include_extra_features=False` in config, or features dropped due to artifacts

**Solution**:

```bash
# Enable extra features
ign-lidar-hd process \
  features.include_extra=true \
  ...
```

### Issue: LAZ file has no RGB despite RGB augmentation

**Cause**: Point format doesn't support RGB, or RGB wasn't computed

**Solution**: The code now automatically selects RGB-compatible point format

## Architecture-Specific Notes

### Multi-Architecture Mode

When using `architecture=multi`, enriched LAZ is only saved for the **original version** (version_idx=0), not for augmented versions. This is intentional because:

- Augmented versions have different point counts (dropout)
- Enriched LAZ is meant for visualization of the original tile
- Patches are saved for all versions (original + augmented)

### Hybrid Mode

Hybrid mode uses `HybridFormatter` which combines all architectures into a single comprehensive format. Enriched LAZ behavior is the same as multi mode.

## Related Files

- `ign_lidar/core/processor.py` - Main processing logic (FIXED)
- `ign_lidar/io/formatters/` - Architecture formatters
- `debug_laz_features.py` - Diagnostic tool (NEW)

## Verification Checklist

After running with the fix:

- [ ] Check log output for "✓ Enriched LAZ saved" message
- [ ] Verify no "❌ CRITICAL" errors about missing dimensions
- [ ] Run `debug_laz_features.py` on output files
- [ ] Open enriched LAZ in CloudCompare and check for extra scalar fields
- [ ] Verify at least 5+ extra dimensions (normals, curvature, height + geo features)

## Next Steps

1. **Test the fix** with your actual data
2. **Run diagnostic tool** on existing enriched LAZ files to confirm the issue
3. **Re-process** any tiles that have missing features
4. **Report** if issues persist (with full error logs)
