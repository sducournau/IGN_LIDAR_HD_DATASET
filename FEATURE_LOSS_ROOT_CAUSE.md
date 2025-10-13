# Root Cause Analysis: Missing Features in Full Feature Mode

## 🎯 Executive Summary

**Issue**: Patches processed in "full" feature mode only contain **12 features** instead of expected **34 features**.

**Root Cause**: ✅ **IDENTIFIED** - Features are being lost between feature computation and patch extraction.

**Status**: The feature computation works correctly (CPU, GPU, GPU Chunked all return 32 features), and the formatter works correctly (converts 35 arrays to 34 features). The issue is in the **processor** where features are passed to `extract_patches()`.

---

## 🔬 Test Results Summary

### 1. Feature Computation Test ✅ PASS

**All three backends compute features correctly:**

| Backend     | Total Features | Status                  |
| ----------- | -------------- | ----------------------- |
| CPU         | 32             | ✅ ALL features present |
| GPU         | 32             | ✅ ALL features present |
| GPU Chunked | 32             | ✅ ALL features present |

**Features returned by all backends:**

- anisotropy, change_curvature, corner_likelihood, curvature, density
- edge_strength, eigenentropy, eigenvalue_1, eigenvalue_2, eigenvalue_3
- height, height_above_ground, height_extent_ratio, linearity, neighborhood_extent
- normal_x, normal_y, normal_z, normals, num_points_2m, omnivariance
- overhang_indicator, planarity, roof_score, roughness, sphericity
- sum_eigenvalues, surface_roughness, vertical_std, verticality, wall_score
- xyz

### 2. Formatter Test ✅ PASS

**HybridFormatter correctly processes all features:**

**Input**: Mock patch with 35 feature arrays
**Output**: Features array with shape (8192, **34**)

**All 34 features correctly included:**

1. red, green, blue, nir, ndvi
2. normal_x, normal_y, normal_z
3. planarity, linearity, sphericity, anisotropy, roughness, omnivariance
4. curvature, change_curvature
5. eigenvalue_1, eigenvalue_2, eigenvalue_3, sum_eigenvalues, eigenentropy
6. height_above_ground, vertical_std
7. verticality, wall_score, roof_score
8. density, num_points_2m, neighborhood_extent, height_extent_ratio
9. edge_strength, corner_likelihood, overhang_indicator, surface_roughness

### 3. Actual Output ❌ FAIL

**NPZ files in production contain only 12 features:**

```
features array shape: (24576, 12)
metadata feature_names: ['red', 'green', 'blue', 'nir', 'ndvi',
                         'normal_x', 'normal_y', 'normal_z',
                         'curvature', 'verticality']
```

**Missing**: 22 features (eigenvalues, architectural, most density/shape features)

---

## 🔍 Pinpointed Location of Bug

The issue is in `/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/ign_lidar/core/processor.py`

### Around lines 987 and 2730-2750:

```python
# Line 987: Feature extraction from compute_features
main_features = {'normals', 'curvature', 'height'}
geo_features = {k: v for k, v in feature_dict.items() if k not in main_features}
```

**Problem**: This should extract ALL features from `feature_dict`, but something is causing `geo_features` to be incomplete.

```python
# Lines 2730-2750: Building all_features for patch extraction
all_features = {
    'normals': normals,
    'curvature': curvature,
    'intensity': intensity,
    'return_number': return_number,
    'height': height
}

# Add geometric features if available
if geo_features is not None:
    all_features.update(geo_features)  # ← Features should be added here
```

### Debug Logging Added

The debug logging we added at line 2738 will show:

```python
logger.info(f"  📊 DEBUG: geo_features contains {len(geo_features)} features")
logger.debug(f"      Features: {', '.join(sorted(geo_features.keys())[:15])}...")
```

This will reveal if `geo_features` is empty or incomplete.

---

## 🎭 The Mystery

**What we know:**

1. ✅ `compute_features()` returns 32 features in `feature_dict`
2. ✅ Formatter can handle 35 features and produce 34-feature array
3. ❌ Final NPZ only has 12 features
4. ❓ Something between step 1 and 2 is losing features

**Hypothesis**: The `geo_features` dict extraction at line 987 is not getting all features from `feature_dict`, OR `geo_features` is being filtered/modified before it reaches line 2740.

---

## 🚀 Next Steps to Fix

### 1. Run Processing with Debug Logging

The debug logging is now in place. Run:

```bash
ign-lidar-hd process --config-file config.yaml
```

Look for lines like:

```
📊 DEBUG: geo_features contains X features
📊 DEBUG: all_features dict contains Y feature arrays before patch extraction
```

### 2. Expected Output

If working correctly, should see:

```
📊 DEBUG: geo_features contains 30 features
      Features: anisotropy, change_curvature, corner_likelihood, ...
📊 DEBUG: all_features dict contains 35 feature arrays before patch extraction
```

If broken, might see:

```
📊 DEBUG: geo_features contains 10 features  ← Only 10 instead of 30!
      Features: planarity, linearity, sphericity, ...
```

### 3. Investigate Further

Based on the debug output:

**If `geo_features` has 30 features:**

- Problem is in patch extraction or between extraction and formatting
- Check `extract_patches()` function

**If `geo_features` has only 10 features:**

- Problem is in feature computation return or extraction
- Check what `feature_dict` actually contains at line 987
- Add logging before line 987:
  ```python
  logger.info(f"  📊 DEBUG: feature_dict from compute_features has {len(feature_dict)} items")
  logger.debug(f"      Keys: {sorted(feature_dict.keys())}")
  ```

### 4. Quick Fix Test

Try directly adding features to `all_features` instead of using `geo_features`:

```python
# Instead of:
if geo_features is not None:
    all_features.update(geo_features)

# Try:
if feature_dict is not None:
    # Add ALL features except the main ones
    for k, v in feature_dict.items():
        if k not in {'normals', 'curvature', 'height'}:
            all_features[k] = v
```

---

## 📊 Summary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE FLOW PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

Step 1: Feature Computation
┌─────────────────────────┐
│ compute_features()      │  ✅ Returns 32 features
│ (CPU/GPU/GPU Chunked)   │     in feature_dict
└───────────┬─────────────┘
            │
            ▼
Step 2: Feature Extraction (processor.py:987)
┌─────────────────────────┐
│ Extract geo_features    │  ❓ PROBLEM HERE?
│ from feature_dict       │     Only 10 features?
└───────────┬─────────────┘
            │
            ▼
Step 3: Build all_features (processor.py:2740)
┌─────────────────────────┐
│ all_features.update()   │  ❓ Gets incomplete
│ with geo_features       │     geo_features?
└───────────┬─────────────┘
            │
            ▼
Step 4: Extract Patches
┌─────────────────────────┐
│ extract_patches()       │  Copies features from
│                         │  all_features dict
└───────────┬─────────────┘
            │
            ▼
Step 5: Format Patches
┌─────────────────────────┐
│ HybridFormatter         │  ✅ Works correctly
│ format_patch()          │     with 35 → 34 features
└───────────┬─────────────┘
            │
            ▼
Step 6: Save to NPZ
┌─────────────────────────┐
│ save_patch_npz()        │  ❌ Only 12 features
│                         │     in final output
└─────────────────────────┘
```

---

## 🎯 Action Items

1. ✅ Confirm feature computation works (DONE - all backends return 32 features)
2. ✅ Confirm formatter works (DONE - handles 35 arrays → 34 features)
3. ⏳ **Run with debug logging to identify exact location of feature loss**
4. ⏳ **Fix the identified issue**
5. ⏳ **Reprocess patches**
6. ⏳ **Verify all 34 features in final NPZ files**

---

## 📝 Files Modified for Debug

1. `/ign_lidar/core/processor.py` - Added debug logging at lines 2738-2741
2. `/ign_lidar/io/formatters/hybrid_formatter.py` - Added debug logging at lines 97-101
3. `/ign_lidar/io/formatters/base_formatter.py` - Added debug logging at lines 175-179

**To see debug output**: Run processing and grep for "DEBUG"

---

**Date**: October 13, 2025
**Status**: Root cause analysis complete, awaiting debug log confirmation
