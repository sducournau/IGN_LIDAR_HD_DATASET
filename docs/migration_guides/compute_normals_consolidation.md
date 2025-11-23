
# Migration Guide: compute_normals() Consolidation

## Summary
Multiple duplicate implementations of `compute_normals()` have been consolidated
into a single canonical implementation in `ign_lidar.features.compute.normals`.

## Changes

### Before (v3.5.x and earlier)
```python
# Multiple ways to compute normals (all valid)
from ign_lidar.features.feature_computer import FeatureComputer
computer = FeatureComputer()
normals = computer.compute_normals(points, k=30)

# OR
from ign_lidar.features.gpu_processor import GPUProcessor
processor = GPUProcessor()
normals = processor.compute_normals(points)

# OR
from ign_lidar.features.utils import validate_normals
validate_normals(normals)
```

### After (v3.6.0+)
```python
# Single canonical API
from ign_lidar.features.compute import compute_normals

# Standard computation
normals, eigenvalues = compute_normals(points, k_neighbors=30)

# Fast variant (optimized)
from ign_lidar.features.compute import compute_normals_fast
normals = compute_normals_fast(points, k_neighbors=30)

# Accurate variant (quality over speed)
from ign_lidar.features.compute import compute_normals_accurate
normals = compute_normals_accurate(points, k_neighbors=30)

# Validation
from ign_lidar.features.compute.utils import validate_normals
validate_normals(normals)
```

## Backward Compatibility

Old imports will continue to work in v3.6.x with deprecation warnings:

```python
# These will work but emit DeprecationWarning
from ign_lidar.features.feature_computer import FeatureComputer
computer = FeatureComputer()
normals = computer.compute_normals(points, k=30)  # ⚠️ DeprecationWarning
```

**Removal timeline:**
- v3.6.0: Deprecation warnings added
- v3.7.0-3.9.0: Warnings continue
- v4.0.0: Old implementations removed

## Why This Change?

1. **Code duplication**: 7 different implementations (~350 lines duplicated)
2. **Maintenance burden**: Bug fixes had to be applied 7 times
3. **Inconsistency risk**: Different implementations could diverge
4. **Performance**: Single optimized implementation is faster

## Benefits

- ✅ Single source of truth for normal computation
- ✅ Easier to maintain and test
- ✅ Better performance (unified optimizations)
- ✅ Clearer API for new users

## Need Help?

Open an issue: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
