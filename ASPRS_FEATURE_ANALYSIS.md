# ASPRS Mode Feature Analysis

## Investigation: Missing Features in Output LAZ Files

**Date:** October 25, 2025  
**Issue:** Computed features may be missing from ASPRS mode output files  
**Hypothesis:** Features might be dropped during validation/filtering due to false positives or invalid values

---

## 🔍 Investigation Summary

### Architecture Overview

The feature computation and saving pipeline follows this flow:

```
1. Tile Loading (TileLoader)
   ↓
2. Feature Computation (FeatureOrchestrator)
   ↓
3. Feature Filtering (by FeatureMode)
   ↓
4. Feature Validation (optional)
   ↓
5. LAZ File Saving (save_enriched_tile_laz)
```

---

## 📊 Feature Mode Configuration

### ASPRS_CLASSES Mode Features (from feature_modes.py)

Expected features in ASPRS mode (~19 features):

```python
ASPRS_FEATURES = {
    # Coordinates (3)
    'xyz',

    # Normals (3)
    'normal_x', 'normal_y', 'normal_z',

    # Shape descriptors (3)
    'planarity',
    'sphericity',
    'curvature',

    # Height features (2)
    'height_above_ground',
    'height',

    # Building detection (2)
    'verticality',
    'horizontality',

    # Density (1)
    'density',

    # Spectral (5 - if available)
    'red', 'green', 'blue',
    'nir',
    'ndvi',
}
```

**Location:** `ign_lidar/features/feature_modes.py:251-280`

---

## 🔎 Critical Code Paths

### 1. Feature Filtering (orchestrator.py)

**Line 1133-1139:**

```python
# Enforce feature mode (filter to only allowed features)
if self.feature_mode != FeatureMode.FULL:
    all_features = self.filter_features(all_features, self.feature_mode)
    logger.debug(
        f"  🔽 Filtered to {len(all_features)} features for mode {self.feature_mode.value}"
    )
```

**filter_features() method (Lines 953-999):**

```python
def filter_features(
    self,
    features: Dict[str, np.ndarray],
    mode: Optional[FeatureMode] = None
) -> Dict[str, np.ndarray]:
    """
    Filter features dict to only include features for given mode.
    """
    if mode is None:
        mode = self.feature_mode

    # Get allowed features for this mode
    allowed_features = set(self.get_feature_list(mode))

    # Always keep core features regardless of mode
    core_features = {'normals', 'curvature', 'height', 'intensity', 'return_number'}
    allowed_features.update(core_features)

    # Handle spectral features
    if 'red' in allowed_features or 'green' in allowed_features or 'blue' in allowed_features:
        allowed_features.add('rgb')
    if 'rgb' in allowed_features:
        allowed_features.update(['red', 'green', 'blue'])

    # Same for NIR and NDVI
    if 'nir' in allowed_features or self.use_infrared:
        allowed_features.add('nir')
    if 'ndvi' in allowed_features or (self.use_rgb and self.use_infrared):
        allowed_features.add('ndvi')

    # Filter
    filtered = {
        k: v for k, v in features.items()
        if k in allowed_features or k.startswith('enriched_')
    }

    return filtered
```

### 2. Validation System (Feature Validator)

**Location:** `ign_lidar/core/classification/feature_validator.py`

The system has validation for false positives:

```python
def filter_ground_truth_false_positives(
    self,
    labels: np.ndarray,
    ground_truth_mask: np.ndarray,
    features: Dict[str, np.ndarray],
    ground_truth_types: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter false positives from ground truth classification.

    This method removes ground truth labels that are inconsistent with
    observed features, preventing propagation of errors.
    """
```

**This affects CLASSIFICATION LABELS, not computed features!**

### 3. Artifact Checking (Ground Truth)

**Location:** `ign_lidar/core/classification/ground_truth_artifact_checker.py`

```python
def filter_artifacts(
    self,
    features: Dict[str, np.ndarray],
    reports: Optional[Dict[str, ArtifactReport]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Filter out points with artifacts from features.

    Sets artifact points to NaN (does not remove features entirely)
    """
```

**This sets invalid POINT VALUES to NaN, not removing entire features!**

### 4. LAZ File Saving

**Location:** `ign_lidar/core/classification/io/serializers.py:493-696`

**Key section (Lines 633-695):**

```python
# 🔍 DEBUG: Log features being processed
logger.info(f"  🔍 DEBUG: Processing {len(features)} features for LAZ export")
logger.debug(f"  🔍 Feature names: {list(features.keys())[:20]}...")

for feat_name, feat_data in features.items():
    if feat_name in ['points', 'classification', 'intensity', 'return_number']:
        continue  # Skip standard fields already set

    if feat_name in added_dimensions:
        continue  # Skip duplicates

    try:
        # Handle normals specially - split into 3 dimensions
        if feat_name == 'normals' and feat_data.ndim == 2 and feat_data.shape[1] == 3:
            for i, axis in enumerate(['x', 'y', 'z']):
                dim_name = f'normal_{axis}'
                if dim_name not in added_dimensions:
                    las.add_extra_dim(laspy.ExtraBytesParams(
                        name=dim_name,
                        type=np.float32,
                        description=f"Normal vector {axis} component"
                    ))
                    setattr(las, dim_name, feat_data[:, i].astype(np.float32))
                    added_dimensions.add(dim_name)
            continue

        # Skip multi-dimensional features that aren't normals
        if feat_data.ndim > 1:
            logger.debug(f"Skipping multi-dimensional feature '{feat_name}' (shape: {feat_data.shape})")
            continue

        # Add extra dimension
        truncated_name = truncate_name(feat_name)
        las.add_extra_dim(laspy.ExtraBytesParams(
            name=truncated_name,
            type=dtype,
            description=description
        ))
        setattr(las, truncated_name, feat_data.astype(dtype))
        added_dimensions.add(truncated_name)

    except Exception as e:
        logger.warning(f"Could not add feature '{feat_name}' to LAZ: {e}")
```

---

## 🐛 Potential Issues Identified

### Issue 1: Feature Mode Filtering ⚠️ **HIGH PRIORITY**

**Problem:** The `filter_features()` method in `orchestrator.py` removes features NOT in the ASPRS_FEATURES set.

**Impact:**

- If a computed feature name doesn't match the exact name in `ASPRS_FEATURES`, it gets dropped
- Features computed but not in the mode definition are silently removed

**Example:**

```python
# If compute returns 'normal_z' but mode expects 'normals'
# OR if there's a name mismatch, feature is dropped
```

**Evidence:** Line 1135 in orchestrator.py:

```python
if self.feature_mode != FeatureMode.FULL:
    all_features = self.filter_features(all_features, self.feature_mode)
```

**Solution:** Verify feature name consistency between:

1. `ASPRS_FEATURES` set in `feature_modes.py`
2. Actual feature names returned by geometric/density/architectural compute functions
3. Feature names saved to LAZ

### Issue 2: Multi-dimensional Features Skipped

**Problem:** Line 674 in serializers.py:

```python
# Skip multi-dimensional features that aren't normals
if feat_data.ndim > 1:
    logger.debug(f"Skipping multi-dimensional feature '{feat_name}' (shape: {feat_data.shape})")
    continue
```

**Impact:** Any 2D or 3D features (except normals) are NOT saved to LAZ.

**Affected features:**

- Any matrix-based features
- Eigenvalue arrays if not flattened

### Issue 3: Silent Exception Handling

**Problem:** Line 692 in serializers.py:

```python
except Exception as e:
    logger.warning(f"Could not add feature '{feat_name}' to LAZ: {e}")
```

**Impact:** Features that fail to save are logged as warnings but processing continues. User might not notice missing features.

### Issue 4: RGB/NIR Removal Before Save

**Problem:** Line 2190 in processor.py:

```python
features_to_save = {k: v for k, v in all_features_v.items()
                   if k not in ['rgb', 'nir', 'input_rgb', 'input_nir', 'points']}
```

**Impact:** RGB and NIR are explicitly removed from features dict before saving (they're saved separately as standard LAZ fields). This is **correct behavior**, but could be confusing if monitoring feature count.

---

## ✅ What's Working Correctly

1. **Validation does NOT drop features** - It only affects classification labels or sets point values to NaN
2. **Artifact checking does NOT remove features** - It only marks invalid points with NaN
3. **Core features are always kept** - `normals`, `curvature`, `height`, etc. are added to allowed set regardless of mode
4. **Enriched features are preserved** - Features starting with `enriched_` are always kept

---

## 🔧 Debugging Strategy

### Step 1: Check Feature Filtering

Add logging BEFORE and AFTER filter_features():

```python
# In orchestrator.py, around line 1133
logger.info(f"  📊 Features BEFORE filtering: {list(all_features.keys())}")
if self.feature_mode != FeatureMode.FULL:
    all_features = self.filter_features(all_features, self.feature_mode)
    logger.info(f"  📊 Features AFTER filtering: {list(all_features.keys())}")
    dropped = set(all_features_original.keys()) - set(all_features.keys())
    if dropped:
        logger.warning(f"  ⚠️  Dropped features: {dropped}")
```

### Step 2: Check Save Function Input

Add logging at save_enriched_tile_laz() entry:

```python
# In serializers.py, line ~636
logger.info(f"  🔍 DEBUG: Received {len(features)} features to save")
logger.info(f"  🔍 Feature names: {sorted(features.keys())}")
for name, data in features.items():
    logger.debug(f"    - {name}: shape={data.shape}, dtype={data.dtype}, ndim={data.ndim}")
```

### Step 3: Check Save Function Output

Count features actually saved:

```python
# In serializers.py, at end of save function (line ~695)
logger.info(f"  ✓ Saved enriched tile: {save_path.name}")
logger.info(f"     Points: {len(points):,}")
logger.info(f"     Extra dimensions added: {len(added_dimensions)}")
logger.info(f"     Extra dimension names: {sorted(added_dimensions)}")
```

### Step 4: Verify with Check Script

Use existing check script:

```bash
python scripts/check_laz_features_v3.py <output_file.laz>
```

This will show:

- Standard dimensions
- Extra dimensions (features)
- Expected vs actual features
- Missing features

---

## 📝 Recommendations

### Immediate Actions

1. **Add Debug Logging:**

   - Log feature names at each pipeline stage
   - Log dropped features with reasons
   - Make warnings more visible

2. **Verify Feature Name Consistency:**

   ```python
   # Check if all computed features are in ASPRS_FEATURES
   computed_names = set(computed_features.keys())
   expected_names = set(ASPRS_FEATURES)
   missing = expected_names - computed_names
   extra = computed_names - expected_names
   ```

3. **Check Configuration:**
   - Verify `feature_mode: asprs_classes` in config
   - Verify all required compute flags are enabled

### Long-term Improvements

1. **Strict Mode for Feature Filtering:**

   ```python
   if strict_mode and dropped_features:
       raise ValueError(f"Features dropped: {dropped_features}")
   ```

2. **Feature Manifest:**

   ```python
   # Save manifest of computed features
   manifest = {
       'computed': list(all_features.keys()),
       'expected': list(ASPRS_FEATURES),
       'saved': list(added_dimensions),
       'mode': str(self.feature_mode)
   }
   ```

3. **Better Error Messages:**
   - Don't silently continue on save failures
   - Provide actionable error messages
   - Suggest fixes for common issues

---

## 🎯 Expected Behavior

In **ASPRS_CLASSES** mode, the output LAZ should contain:

### Standard Fields:

- X, Y, Z (coordinates)
- Classification (ASPRS codes)
- Intensity
- Return Number
- RGB (if available)
- NIR (if available)

### Extra Dimensions (Features):

- `normal_x`, `normal_y`, `normal_z` (3)
- `planarity` (1)
- `sphericity` (1)
- `curvature` (1)
- `height` (1)
- `height_above_ground` (1)
- `verticality` (1)
- `horizontality` (1)
- `density` (1)
- `ndvi` (1) - if RGB+NIR available

**Total extra dimensions expected: 12-13 features**

---

## 🔬 Test Case

Create a minimal test to isolate the issue:

```python
#!/usr/bin/env python3
"""Test ASPRS feature filtering and saving."""
import numpy as np
from pathlib import Path
from ign_lidar.features.feature_modes import ASPRS_FEATURES, FeatureMode
from ign_lidar.features.orchestrator import FeatureOrchestrator
from ign_lidar.core.classification.io import save_enriched_tile_laz

# Create synthetic features
n_points = 1000
features = {
    'normal_x': np.random.rand(n_points),
    'normal_y': np.random.rand(n_points),
    'normal_z': np.random.rand(n_points),
    'planarity': np.random.rand(n_points),
    'sphericity': np.random.rand(n_points),
    'curvature': np.random.rand(n_points),
    'height': np.random.rand(n_points) * 10,
    'height_above_ground': np.random.rand(n_points) * 5,
    'verticality': np.random.rand(n_points),
    'horizontality': np.random.rand(n_points),
    'density': np.random.rand(n_points) * 100,
}

print(f"Created {len(features)} features")
print(f"Feature names: {sorted(features.keys())}")

# Test filtering
config = {'features': {'mode': 'asprs_classes'}}
orchestrator = FeatureOrchestrator(config, feature_mode=FeatureMode.ASPRS_CLASSES)

filtered = orchestrator.filter_features(features, FeatureMode.ASPRS_CLASSES)

print(f"\nAfter filtering: {len(filtered)} features")
print(f"Filtered names: {sorted(filtered.keys())}")

dropped = set(features.keys()) - set(filtered.keys())
if dropped:
    print(f"\n⚠️ DROPPED: {dropped}")

# Test saving
points = np.random.rand(n_points, 3) * 100
classification = np.random.randint(0, 10, n_points, dtype=np.uint8)
intensity = np.random.rand(n_points)
return_number = np.ones(n_points, dtype=np.float32)

save_path = Path('/tmp/test_asprs_features.laz')
save_enriched_tile_laz(
    save_path=save_path,
    points=points,
    classification=classification,
    intensity=intensity,
    return_number=return_number,
    features=filtered,
    original_las=None,
    header=None,
    input_rgb=None,
    input_nir=None
)

# Verify
import laspy
las = laspy.read(str(save_path))
extra_dims = list(las.point_format.extra_dimension_names)

print(f"\nSaved extra dimensions: {len(extra_dims)}")
print(f"Extra dim names: {sorted(extra_dims)}")

missing = set(filtered.keys()) - set(extra_dims)
if missing:
    print(f"\n❌ NOT SAVED: {missing}")
else:
    print(f"\n✅ All features saved successfully!")
```

---

## 📚 References

**Key Files:**

- `ign_lidar/features/feature_modes.py` - Feature mode definitions
- `ign_lidar/features/orchestrator.py` - Feature computation and filtering
- `ign_lidar/core/classification/io/serializers.py` - LAZ saving logic
- `ign_lidar/core/processor.py` - Main processing pipeline
- `scripts/check_laz_features_v3.py` - Feature verification tool

**Related Documentation:**

- `docs/FEATURE_AUDIT_REPORT.md` - Complete feature audit
- `examples/README_ASPRS_OPTIMIZED.md` - ASPRS configuration guide
- `docs/docs/examples/asprs-classification-example.md` - ASPRS examples

---

**Status:** Analysis complete - ready for debugging  
**Next Step:** Run test case and add logging to identify exact point where features are dropped
