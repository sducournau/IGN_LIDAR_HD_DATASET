# Feature Validation and Artifact Detection

## Overview

The IGN LiDAR HD v2.0 processing pipeline now includes automatic validation and artifact detection for geometric features computed at tile boundaries. This ensures that only high-quality, reliable features are used for downstream processing.

## Problem Statement

When processing LiDAR tiles with boundary-aware feature computation, several types of artifacts can occur:

1. **Dash/Line Patterns**: Scan line artifacts causing artificially high linearity
2. **Discontinuous Planes**: Sharp transitions in planarity at tile boundaries
3. **Invalid Values**: NaN, Inf, or out-of-range values
4. **Constant Values**: Artificial uniformity indicating computation errors

## Validation Strategy

### 1. Invalid Value Detection

```python
# Check for NaN and Inf values
has_nan = np.any(np.isnan(boundary_values))
has_inf = np.any(np.isinf(boundary_values))
```

### 2. Artifact Pattern Detection

#### Linearity Artifacts (Scan Lines)

- **Pattern**: High mean (>0.8) with low variance (<0.1)
- **Cause**: Scan pattern artifacts at tile boundaries
- **Action**: Drop linearity feature

```python
if mean_val > 0.8 and std_val < 0.1:
    drop_feature('linearity')
```

#### Planarity Artifacts (Discontinuities)

- **Low Variance (<0.05)**: Artificial constant values
- **High Variance (>0.4)**: Sharp discontinuities at boundaries
- **Action**: Drop planarity feature

```python
if std_val < 0.05 or std_val > 0.4:
    drop_feature('planarity')
```

#### Verticality Artifacts (Bimodal Extremes)

- **Pattern**: >95% of values at extremes (0 or 1)
- **Cause**: Classification edge effects
- **Action**: Drop verticality feature

```python
if (high_extreme + low_extreme) > 0.95:
    drop_feature('verticality')
```

### 3. Range Validation

All geometric features should be in valid range [0, 1]:

```python
if not np.all((boundary_values >= 0) & (boundary_values <= 1)):
    drop_feature(fname)
```

## Implementation

### BoundaryAwareFeatureComputer

The validation is integrated into the `compute_features()` method:

```python
# After computing all features
if num_boundary > 0:
    results = self._validate_features(results, boundary_mask)
```

### Feature Dropping

When artifacts are detected:

1. **Warning Logged**: Details about the artifact pattern
2. **Feature Removed**: Dropped from results dictionary
3. **Processing Continues**: With remaining valid features

### Graceful Degradation

If all geometric features are dropped:

```python
if not geo_features:
    geo_features = None
    logger.warning("All geometric features dropped due to artifacts")
```

Processing continues with:

- Normals (always computed)
- Curvature (always computed)
- Height (always computed)
- RGB/NIR (if available)

## Example Output

```
[INFO] Feature computation complete: 7 feature types computed
[INFO] Validating features for artifacts...
[WARNING] ⚠️  Feature 'linearity' shows line artifact pattern (mean=0.853, std=0.067) - dropping
[WARNING] ⚠️  Feature 'planarity' shows discontinuity pattern (std=0.457) - dropping
[INFO] 🔍 Feature validation: dropping 2 problematic features: ['linearity', 'planarity']
[INFO] ✓ Boundary-aware features computed (1124198 boundary points affected)
```

## Benefits

1. **Quality Assurance**: Only reliable features used for training
2. **Automatic Detection**: No manual inspection required
3. **Graceful Degradation**: Processing continues with valid features
4. **Transparency**: Clear logging of validation decisions

## Statistics Thresholds

| Feature     | Artifact Type | Threshold          | Action |
| ----------- | ------------- | ------------------ | ------ |
| Linearity   | Scan lines    | mean>0.8, std<0.1  | Drop   |
| Planarity   | Constant      | std<0.05           | Drop   |
| Planarity   | Discontinuity | std>0.4            | Drop   |
| Verticality | Bimodal       | extreme_ratio>0.95 | Drop   |
| All         | Invalid       | NaN or Inf         | Drop   |
| All         | Out of range  | value<0 or value>1 | Drop   |

## Future Enhancements

1. **Adaptive Thresholds**: Learn thresholds from clean data
2. **Feature Repair**: Attempt to fix minor artifacts instead of dropping
3. **Cross-Validation**: Compare with non-boundary regions
4. **ML-Based Detection**: Train classifier to detect subtle artifacts

## Related Files

- `/ign_lidar/features/features_boundary.py`: Validation implementation
- `/ign_lidar/core/processor.py`: Integration with processing pipeline
- `/tests/test_boundary_features.py`: Validation tests

## References

- Sprint 3 - Tile Stitching (Phase 3.2)
- Boundary-Aware Feature Computation Module
- IGN LiDAR HD v2.0 Documentation

---

**Date**: October 9, 2025  
**Status**: ✅ Implemented  
**Version**: v2.0
