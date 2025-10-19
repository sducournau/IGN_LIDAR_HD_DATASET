# Ground Truth Artifact Detection Guide

**Date:** October 19, 2025  
**Version:** 5.0  
**Status:** Production Ready

## Overview

This guide explains how to ensure **correct ground truth classification** by detecting and filtering artifacts in features before they are used for classification.

## 🎯 Key Principle

**NEVER use a feature with artifacts for ground truth classification.**

If a feature contains NaN, Inf, or out-of-range values, it will produce **incorrect classifications**. This system validates all features before use and filters out problematic data points.

---

## 📋 Artifact Detection System

### 1. **What are Artifacts?**

Artifacts are **invalid or corrupted feature values** that include:

- **NaN (Not a Number)** - Missing or undefined values
- **Inf (Infinity)** - Division by zero or numerical overflow
- **Out-of-range values** - Values outside expected physical limits
- **Constant values** - All values are identical (indicates computation failure)
- **Low diversity** - Too few unique values (data corruption)

### 2. **Why Artifacts are Dangerous**

❌ **Using artifacts leads to:**

- Incorrect classifications (e.g., water labeled as buildings)
- False positives (e.g., tree canopy as roads)
- Missing classifications (e.g., buildings marked as unclassified)
- Inconsistent results across tiles

✅ **Filtering artifacts ensures:**

- Accurate classifications
- Reliable ground truth
- Consistent results
- Trustworthy training data

---

## 🔧 Using the Artifact Checker

### Quick Start

```python
from ign_lidar.core.modules.ground_truth_artifact_checker import (
    GroundTruthArtifactChecker,
    validate_features_before_classification
)

# Prepare features dictionary
features = {
    'height': height_array,
    'planarity': planarity_array,
    'curvature': curvature_array,
    'ndvi': ndvi_array,
    'normals': normals_array
}

# Validate features before classification
is_valid, reports = validate_features_before_classification(
    features=features,
    strict=False,  # Allow small artifact ratios
    log_results=True  # Print detailed report
)

if not is_valid:
    print("⚠️ Features contain too many artifacts!")
    print("Ground truth classification may be unreliable.")
else:
    print("✓ Features are clean - safe to use for classification")
```

### Advanced Usage

#### 1. Check Individual Features

```python
checker = GroundTruthArtifactChecker()

# Check a single feature
report = checker.check_feature('ndvi', ndvi_array)

print(report)
# Output:
# ✓ ndvi: OK (range=[-0.234, 0.876], mean=0.412)
# OR
# ⚠️ ndvi: ARTIFACTS DETECTED
#     - 124 NaN values (0.7%)
#     - 45 values above expected maximum 1.0 (actual max: 1.234)
```

#### 2. Check All Features

```python
reports = checker.check_all_features(features)

for feature_name, report in reports.items():
    print(report)
```

#### 3. Filter Artifacts Automatically

```python
# Remove artifact points from features
clean_features, artifact_masks = checker.filter_artifacts(features)

# Now clean_features has NaN where artifacts were detected
# artifact_masks tells you which points were affected
```

#### 4. Custom Artifact Thresholds

```python
config = {
    'max_artifact_ratio': 0.05,  # Allow max 5% artifacts
    'min_std_threshold': 1e-5,   # Detect constant values
    'feature_ranges': {
        'custom_feature': (0.0, 10.0)  # Custom range
    }
}

checker = GroundTruthArtifactChecker(config=config)
```

---

## 📊 Expected Feature Ranges

The artifact checker validates features against **expected physical ranges**:

### Geometric Features (normalized 0-1)

| Feature       | Min | Max | Purpose                       |
| ------------- | --- | --- | ----------------------------- |
| `linearity`   | 0.0 | 1.0 | 1D structures (edges, cables) |
| `planarity`   | 0.0 | 1.0 | 2D structures (roofs, walls)  |
| `sphericity`  | 0.0 | 1.0 | 3D structures (vegetation)    |
| `anisotropy`  | 0.0 | 1.0 | Directionality                |
| `verticality` | 0.0 | 1.0 | How vertical (walls)          |
| `curvature`   | 0.0 | 1.0 | Surface curvature             |

### Height Features

| Feature               | Min   | Max   | Purpose            |
| --------------------- | ----- | ----- | ------------------ |
| `height`              | -10.0 | 200.0 | Absolute height    |
| `height_above_ground` | 0.0   | 200.0 | Relative height    |
| `z_from_ground`       | 0.0   | 200.0 | Height from ground |

### Spectral Features

| Feature    | Min  | Max   | Purpose                    |
| ---------- | ---- | ----- | -------------------------- |
| `ndvi`     | -1.0 | 1.0   | Vegetation index           |
| `nir`      | 0.0  | 1.0   | Near-infrared (normalized) |
| `red`      | 0    | 65535 | Red channel (16-bit)       |
| `green`    | 0    | 65535 | Green channel (16-bit)     |
| `blue`     | 0    | 65535 | Blue channel (16-bit)      |
| `infrared` | 0    | 65535 | Infrared channel (16-bit)  |

---

## 🔍 Integration with Ground Truth Classification

### Recommended Workflow

```python
from ign_lidar.core.modules.ground_truth_artifact_checker import (
    GroundTruthArtifactChecker
)
from ign_lidar.core.modules.feature_validator import FeatureValidator
from ign_lidar.core.modules.ground_truth_refinement import GroundTruthRefiner

# Step 1: Check for artifacts
checker = GroundTruthArtifactChecker()
reports = checker.check_all_features(features)

# Step 2: Filter artifacts
clean_features, artifact_masks = checker.filter_artifacts(features)

# Step 3: Validate ground truth with clean features
validator = FeatureValidator()
validated_labels, confidences, valid_mask = validator.validate_ground_truth(
    labels=ground_truth_labels,
    ground_truth_types=ground_truth_types,
    features=clean_features  # Use clean features!
)

# Step 4: Refine ground truth (water, roads, etc.)
refiner = GroundTruthRefiner()
refined_labels, stats = refiner.refine_all(
    labels=validated_labels,
    points=points,
    ground_truth_features=ground_truth_features,
    features=clean_features  # Use clean features!
)
```

### Integration in Advanced Classification

The artifact checker is automatically used when enabled in configuration:

```yaml
# config.yaml
processor:
  use_optimized_ground_truth: true
  ground_truth_refinement: true
  validate_features: true # Enable artifact checking
  handle_nan_values: true # Handle NaN gracefully

features:
  validate_features: true # Validate feature ranges
```

---

## 📈 Artifact Detection in Action

### Example Output

```
================================================================================
ARTIFACT DETECTION SUMMARY
================================================================================
Total features checked: 8
Clean features: 6
Features with artifacts: 2

✓ Clean Features:
  ✓ planarity: OK (range=[0.123, 0.987], mean=0.654)
  ✓ curvature: OK (range=[0.001, 0.234], mean=0.045)
  ✓ height: OK (range=[0.234, 45.678], mean=12.345)
  ✓ ndvi: OK (range=[-0.123, 0.876], mean=0.412)
  ✓ nir: OK (range=[0.123, 0.876], mean=0.534)
  ✓ intensity: OK (range=[1234, 45678], mean=23456)

⚠️ Features with Artifacts:
  ⚠️ verticality: ARTIFACTS DETECTED
    - 234 NaN values (1.3%)
    - Nearly constant values (std=2.34e-07 < 1.00e-06)

  ⚠️ normals: ARTIFACTS DETECTED
    - 45 Inf values (0.3%)
    - 89 values outside expected range

================================================================================
```

### Handling Artifacts

When artifacts are detected:

1. **Option 1: Filter Points** (Recommended)

   ```python
   clean_features, artifact_masks = checker.filter_artifacts(features)
   # Use clean_features for classification
   ```

2. **Option 2: Exclude Feature**

   ```python
   # Don't use the feature with artifacts
   if reports['verticality'].has_artifacts:
       features.pop('verticality')
   ```

3. **Option 3: Recompute Feature**
   ```python
   # If possible, recompute with different parameters
   verticality = compute_verticality(normals, method='robust')
   ```

---

## 🎓 Best Practices

### ✅ DO

1. **Always validate features** before ground truth classification
2. **Filter artifact points** before using features
3. **Log artifact reports** for debugging
4. **Use strict mode** for critical applications
5. **Monitor artifact ratios** across tiles

### ❌ DON'T

1. **Don't ignore artifact warnings** - they indicate real problems
2. **Don't use features with >10% artifacts** - results will be unreliable
3. **Don't skip validation** to save time - it's fast (<1s per tile)
4. **Don't mix clean and dirty features** - filter all or none

---

## 🔬 Artifact Detection Examples

### Case 1: Water Classification

```python
# Ground truth says "water" but features don't match
features = {
    'height': np.array([15.0]),     # 15m high - not water!
    'planarity': np.array([0.45]),  # Not flat
    'normals': np.array([[0, 0, 0.3]])  # Not horizontal
}

# Artifact checker will flag this
report = checker.check_feature('height', features['height'])
# ⚠️ height: 15m exceeds water height limit (0.3m)

# Feature validator will reject this
validator = FeatureValidator()
validated, conf, valid = validator.validate_ground_truth(
    labels=np.array([9]),  # ASPRS_WATER
    ground_truth_types=np.array(['water']),
    features=features
)
# Result: valid[0] = False (rejected)
```

### Case 2: Road with Tree Canopy

```python
# Ground truth says "road" but features show vegetation
features = {
    'height': np.array([8.0]),      # 8m high - tree canopy!
    'ndvi': np.array([0.75]),       # High NDVI - vegetation
    'curvature': np.array([0.35])   # High curvature - not flat
}

# Feature validator detects tree canopy
validated, conf, valid = validator.validate_ground_truth(
    labels=np.array([11]),  # ASPRS_ROAD
    ground_truth_types=np.array(['road']),
    features=features
)
# Result: validated[0] = 5 (ASPRS_HIGH_VEGETATION)
```

### Case 3: Roof Vegetation

```python
# Ground truth says "building" but features show vegetation
features = {
    'height': np.array([12.0]),     # On roof
    'ndvi': np.array([0.65]),       # High NDVI - vegetation
    'curvature': np.array([0.25]),  # Complex surface
    'planarity': np.array([0.45])   # Not planar
}

# Feature validator detects roof vegetation
validated, conf, valid = validator.validate_ground_truth(
    labels=np.array([6]),  # ASPRS_BUILDING
    ground_truth_types=np.array(['building']),
    features=features
)
# Result: validated[0] = 5 (ASPRS_HIGH_VEGETATION)
```

---

## 📦 Module Integration

Add to your imports:

```python
# In your processing script
from ign_lidar.core.modules.ground_truth_artifact_checker import (
    GroundTruthArtifactChecker,
    validate_features_before_classification,
    ArtifactReport
)
```

Update module exports:

```python
# In ign_lidar/core/modules/__init__.py
from .ground_truth_artifact_checker import (
    GroundTruthArtifactChecker,
    validate_features_before_classification,
    ArtifactReport
)

__all__ = [
    ...,
    'GroundTruthArtifactChecker',
    'validate_features_before_classification',
    'ArtifactReport'
]
```

---

## 📊 Performance

- **Artifact checking overhead:** ~0.1-0.3s per tile (18M points)
- **Memory overhead:** Minimal (~50MB for masks)
- **CPU usage:** Negligible (vectorized NumPy operations)

**Recommendation:** Always enable artifact checking - the overhead is negligible compared to the accuracy improvements.

---

## 🎯 Summary

**For ground truth classification:**

1. ✅ **Check features for artifacts** before use
2. ✅ **Filter artifact points** or exclude bad features
3. ✅ **Validate ground truth** with clean features
4. ✅ **Refine classifications** using feature signatures
5. ✅ **Monitor artifact ratios** across tiles

**Result:** Accurate, reliable ground truth classifications without artifacts.

---

## 📚 Related Documentation

- [Ground Truth Refinement Guide](ground-truth-refinement.md)
- [Feature Validation Guide](../phase1_archive/CLASSIFICATION_VEGETATION_AUDIT_2025.md)
- [ASPRS Features Guide](../../ASPRS_FEATURES_QUICK_REFERENCE.md)

---

**Questions?** Check the logs for detailed artifact reports or contact the development team.
