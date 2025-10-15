# Migration Guide: Features Core Module (v2.5.2)

**Version**: 2.5.2  
**Date**: October 16, 2025  
**Phase**: Phase 1 Consolidation Complete

---

## Overview

Version 2.5.2 introduces the new `ign_lidar.features.core` module, providing canonical implementations for all feature computation functions. This guide helps you migrate to the new architecture while maintaining full backward compatibility.

**TL;DR**: Your existing code works without changes! This guide shows you how to use the new, improved APIs for new code.

---

## What Changed?

### New Features Core Module

We've created a centralized module for all feature computation:

```
ign_lidar/features/core/
├── __init__.py          # Public API
├── normals.py          # Normal vector computation
├── curvature.py        # Curvature features
├── eigenvalues.py      # Eigenvalue-based features
├── density.py          # Density features
├── architectural.py    # Building-specific features
└── utils.py            # Shared utilities
```

### Benefits

✅ **Single Source of Truth**: Each feature has one canonical implementation  
✅ **Better Documentation**: Comprehensive docstrings with examples  
✅ **Type Hints**: Full type annotations for IDE support  
✅ **Consistent APIs**: Uniform function signatures across all modes  
✅ **Easier Testing**: Centralized test suite  
✅ **GPU Support**: Optional GPU acceleration built-in

---

## Quick Start

### For Existing Users

**No action required!** Your existing code continues to work:

```python
# This still works (v2.5.1 and earlier)
from ign_lidar.features.features import compute_normals

points = np.random.randn(1000, 3)
normals = compute_normals(points, k=20)
```

### For New Code (Recommended)

Use the new core module for better features:

```python
# New way (v2.5.2+)
from ign_lidar.features.core import compute_normals

points = np.random.randn(1000, 3)
normals, eigenvectors = compute_normals(
    points,
    k_neighbors=20,
    use_gpu=False
)
```

---

## Migration Examples

### 1. Normal Computation

#### Old API

```python
from ign_lidar.features.features import compute_normals

# Old signature
normals = compute_normals(points, k=20)
# Returns: np.ndarray (N, 3) - just normals
```

#### New API

```python
from ign_lidar.features.core import compute_normals

# New signature with more options
normals, eigenvectors = compute_normals(
    points,                    # np.ndarray (N, 3)
    k_neighbors=20,           # Changed from 'k' to 'k_neighbors'
    use_gpu=False,            # Optional GPU acceleration
    return_eigenvectors=True  # Get eigenvectors too
)

# Returns:
# - normals: np.ndarray (N, 3)
# - eigenvectors: np.ndarray (N, 3, 3) or None
```

**Benefits**:

- More descriptive parameter names
- Optional GPU acceleration
- Can get eigenvectors for advanced analysis
- Better error handling

**Migration Steps**:

1. Change `k` to `k_neighbors`
2. Unpack two return values (or ignore second one)
3. Optionally enable GPU with `use_gpu=True`

### 2. Curvature Features

#### Old API

```python
from ign_lidar.features.features import compute_curvature

# Old signature
curvature = compute_curvature(points, k=20)
# Returns: np.ndarray (N,) - just mean curvature
```

#### New API

```python
from ign_lidar.features.core import compute_curvature

# New signature with multiple curvature types
curvature = compute_curvature(
    points,
    k_neighbors=20,
    curvature_type='mean'  # 'mean', 'gaussian', or 'both'
)

# For multiple types:
mean_curv, gaussian_curv = compute_curvature(
    points,
    k_neighbors=20,
    curvature_type='both'
)
```

**Benefits**:

- Choose specific curvature type
- Get both mean and Gaussian curvature
- Better mathematical accuracy
- 98% test coverage

### 3. Eigenvalue Features

#### Old API

```python
from ign_lidar.features.features import compute_eigenvalue_features

# Old signature
features = compute_eigenvalue_features(points, k=20)
# Returns: dict with some features
```

#### New API

```python
from ign_lidar.features.core import compute_eigenvalue_features

# New signature with all features
features = compute_eigenvalue_features(
    points,
    k_neighbors=20,
    compute_all=True  # Get all eigenvalue-based features
)

# Returns: dict with keys:
# - 'planarity': float array
# - 'sphericity': float array
# - 'linearity': float array
# - 'anisotropy': float array
# - 'eigenentropy': float array
# - 'sum_eigenvalues': float array
# - 'omnivariance': float array
# - 'change_curvature': float array
```

**Benefits**:

- Complete set of geometric features
- Consistent naming (planarity, not planar_score)
- All features in one call
- Based on published research

### 4. Density Features

#### Old API

```python
from ign_lidar.features.features import compute_density_features

# Old signature
features = compute_density_features(
    points,
    tree,  # Had to pre-build tree
    k=20
)
```

#### New API

```python
from ign_lidar.features.core import compute_density

# New signature - tree built internally
density = compute_density(
    points,
    k_neighbors=20,
    radius=None  # Optional radius-based search
)

# Returns: np.ndarray (N,) - point density values
```

**Benefits**:

- No need to pre-build k-d tree
- Simpler API
- Radius-based search option
- Better memory efficiency

### 5. Architectural Features

#### Old API

```python
# No canonical implementation existed!
# Had to use building-specific code
```

#### New API ✨

```python
from ign_lidar.features.core import (
    compute_wall_likelihood,
    compute_roof_likelihood,
    compute_facade_score,
    compute_building_regularity
)

# Wall detection (vertical + planar surfaces)
wall_score = compute_wall_likelihood(
    verticality,  # From eigenvalue features
    planarity
)

# Roof detection (horizontal + planar surfaces)
roof_score = compute_roof_likelihood(
    horizontality,
    planarity
)

# Facade scoring (composite feature)
facade = compute_facade_score(
    verticality,
    height_above_ground,
    planarity
)

# Building regularity (structured geometry)
regularity = compute_building_regularity(
    planarity,
    linearity,
    eigenvalues
)
```

**Benefits**:

- New canonical implementations
- Based on published research (Weinmann et al.)
- More robust than ad-hoc methods
- Consistent across all detection modes

---

## Complete Migration Example

### Before (v2.5.1)

```python
import numpy as np
from ign_lidar.features.features import (
    compute_normals,
    compute_curvature,
    compute_eigenvalue_features,
    compute_density_features
)
from sklearn.neighbors import KDTree

# Load points
points = load_point_cloud('data.laz')

# Build tree manually
tree = KDTree(points)

# Compute features
normals = compute_normals(points, k=20)
curvature = compute_curvature(points, k=20)
eig_features = compute_eigenvalue_features(points, k=20)
density = compute_density_features(points, tree, k=20)

# Extract specific features
planarity = eig_features.get('planarity', np.zeros(len(points)))
verticality = eig_features.get('verticality', np.zeros(len(points)))
```

### After (v2.5.2+)

```python
import numpy as np
from ign_lidar.features.core import (
    compute_normals,
    compute_curvature,
    compute_eigenvalue_features,
    compute_density,
    compute_wall_likelihood,
    compute_roof_likelihood
)

# Load points
points = load_point_cloud('data.laz')

# Compute features (tree built automatically)
normals, eigenvectors = compute_normals(points, k_neighbors=20)
mean_curv, gauss_curv = compute_curvature(points, k_neighbors=20, curvature_type='both')
eig_features = compute_eigenvalue_features(points, k_neighbors=20, compute_all=True)
density = compute_density(points, k_neighbors=20)

# Extract features (consistent keys)
planarity = eig_features['planarity']
verticality = eig_features['verticality']
horizontality = eig_features['horizontality']

# Use architectural features
wall_score = compute_wall_likelihood(verticality, planarity)
roof_score = compute_roof_likelihood(horizontality, planarity)
```

**Changes**:

- ✅ No manual tree building
- ✅ Consistent parameter names (`k_neighbors`)
- ✅ More features available
- ✅ Better type hints and documentation
- ✅ Optional GPU acceleration

---

## API Reference

### Core Module Imports

```python
# All-in-one import
from ign_lidar.features import core

# Individual function imports
from ign_lidar.features.core import (
    compute_normals,
    compute_curvature,
    compute_eigenvalue_features,
    compute_density,
    compute_wall_likelihood,
    compute_roof_likelihood,
    compute_facade_score,
    compute_building_regularity,
    compute_corner_likelihood
)

# Utility imports
from ign_lidar.features.core.utils import (
    build_kdtree,
    compute_covariance_matrix,
    has_gpu
)
```

### Function Signatures

#### compute_normals

```python
def compute_normals(
    points: np.ndarray,
    k_neighbors: int = 10,
    use_gpu: bool = False,
    return_eigenvectors: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute surface normals using PCA on k-nearest neighbors.

    Args:
        points: Point cloud (N, 3)
        k_neighbors: Number of neighbors (default: 10)
        use_gpu: Use GPU acceleration if available (default: False)
        return_eigenvectors: Return eigenvectors too (default: False)

    Returns:
        - normals: Normal vectors (N, 3)
        - eigenvectors: Eigenvectors (N, 3, 3) or None
    """
```

#### compute_curvature

```python
def compute_curvature(
    points: np.ndarray,
    k_neighbors: int = 10,
    curvature_type: str = 'mean'
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute surface curvature features.

    Args:
        points: Point cloud (N, 3)
        k_neighbors: Number of neighbors (default: 10)
        curvature_type: 'mean', 'gaussian', or 'both' (default: 'mean')

    Returns:
        - If 'mean' or 'gaussian': curvature values (N,)
        - If 'both': tuple of (mean_curvature, gaussian_curvature)
    """
```

#### compute_eigenvalue_features

```python
def compute_eigenvalue_features(
    points: np.ndarray,
    k_neighbors: int = 10,
    compute_all: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute eigenvalue-based geometric features.

    Args:
        points: Point cloud (N, 3)
        k_neighbors: Number of neighbors (default: 10)
        compute_all: Compute all features (default: True)

    Returns:
        Dictionary with keys:
        - planarity: Planarity feature (N,)
        - sphericity: Sphericity feature (N,)
        - linearity: Linearity feature (N,)
        - anisotropy: Anisotropy feature (N,)
        - eigenentropy: Eigenentropy (N,)
        - sum_eigenvalues: Sum of eigenvalues (N,)
        - omnivariance: Omnivariance (N,)
        - change_curvature: Surface change (N,)
        - verticality: Verticality score (N,)
        - horizontality: Horizontality score (N,)
    """
```

#### compute_density

```python
def compute_density(
    points: np.ndarray,
    k_neighbors: int = 10,
    radius: Optional[float] = None
) -> np.ndarray:
    """
    Compute local point density.

    Args:
        points: Point cloud (N, 3)
        k_neighbors: Number of neighbors (default: 10)
        radius: Search radius (optional, uses k-NN if None)

    Returns:
        Point density values (N,)
    """
```

### Architectural Features

```python
def compute_wall_likelihood(
    verticality: np.ndarray,
    planarity: np.ndarray
) -> np.ndarray:
    """Wall likelihood: sqrt(verticality * planarity)"""

def compute_roof_likelihood(
    horizontality: np.ndarray,
    planarity: np.ndarray
) -> np.ndarray:
    """Roof likelihood: sqrt(horizontality * planarity)"""

def compute_facade_score(
    verticality: np.ndarray,
    height: np.ndarray,
    planarity: np.ndarray
) -> np.ndarray:
    """Facade score: verticality + height + planarity"""

def compute_building_regularity(
    planarity: np.ndarray,
    linearity: np.ndarray,
    eigenvalues: np.ndarray
) -> np.ndarray:
    """Building regularity from geometric features"""
```

---

## GPU Acceleration

### Checking GPU Availability

```python
from ign_lidar.features.core.utils import has_gpu

if has_gpu():
    print("GPU available - can use GPU acceleration")
    use_gpu = True
else:
    print("GPU not available - using CPU")
    use_gpu = False
```

### Using GPU Acceleration

```python
from ign_lidar.features.core import compute_normals

# Enable GPU acceleration
normals, _ = compute_normals(
    points,
    k_neighbors=20,
    use_gpu=True  # Will fall back to CPU if GPU unavailable
)
```

**Note**: GPU acceleration requires:

- CUDA-capable GPU
- CuPy library installed (`pip install cupy-cuda11x`)
- RAPIDS cuML (optional, for advanced features)

---

## Backward Compatibility

### Old APIs Still Work

All existing code continues to function:

```python
# All of these still work
from ign_lidar.features.features import (
    compute_normals,           # ✅ Works
    compute_curvature,         # ✅ Works
    compute_eigenvalue_features,  # ✅ Works
    compute_density_features,  # ✅ Works
    compute_verticality        # ✅ Works (bug fixed!)
)

# Old function signatures accepted
normals = compute_normals(points, k=20)  # ✅ Works
```

### Wrapper Functions

Old APIs are now lightweight wrappers:

```python
# In features.py
def compute_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute surface normals (backward-compatible wrapper).

    Note:
        This is a wrapper around the core implementation.
        For new code, use: ign_lidar.features.core.compute_normals
    """
    normals, _ = core_compute_normals(points, k_neighbors=k, use_gpu=False)
    return normals
```

---

## Deprecation Timeline

### v2.5.2 (Current)

- ✅ New `features/core` module available
- ✅ Old APIs maintained with backward compatibility
- ✅ No deprecation warnings yet

### v2.6.0 (November 2025 - Planned)

- ⚠️ Deprecation warnings added to old APIs
- 📚 Documentation updated to recommend new APIs
- 🔄 Old APIs still functional

### v3.0.0 (Q1 2026 - Planned)

- ❌ Old wrapper functions removed
- ❌ Deprecated memory modules removed
- 📖 Migration guide provided
- ⏱️ 6-month notice period

---

## Best Practices

### 1. For New Code

✅ **DO**: Use `features.core` imports

```python
from ign_lidar.features.core import compute_normals
```

❌ **DON'T**: Use old imports

```python
from ign_lidar.features.features import compute_normals
```

### 2. For Existing Code

✅ **DO**: Leave working code alone (if it works, don't fix it)

✅ **DO**: Migrate incrementally when making changes

❌ **DON'T**: Rush to rewrite everything at once

### 3. For Libraries

✅ **DO**: Support both APIs during transition period

```python
try:
    from ign_lidar.features.core import compute_normals
except ImportError:
    from ign_lidar.features.features import compute_normals
```

### 4. For Tests

✅ **DO**: Write new tests using core module

✅ **DO**: Update existing tests gradually

---

## Common Pitfalls

### 1. Parameter Name Change

**Problem**:

```python
# This will fail with TypeError
from ign_lidar.features.core import compute_normals
normals, _ = compute_normals(points, k=20)  # ❌ 'k' not recognized
```

**Solution**:

```python
normals, _ = compute_normals(points, k_neighbors=20)  # ✅ Correct
```

### 2. Return Value Unpacking

**Problem**:

```python
# This will fail with ValueError (too many values to unpack)
from ign_lidar.features.core import compute_normals
normals = compute_normals(points, k_neighbors=20)  # ❌ Returns 2 values
```

**Solution**:

```python
normals, eigenvectors = compute_normals(points, k_neighbors=20)  # ✅ Unpack both
# Or ignore second value
normals, _ = compute_normals(points, k_neighbors=20)  # ✅ Also correct
```

### 3. Dictionary Key Changes

**Problem**:

```python
features = compute_eigenvalue_features(points, k_neighbors=20)
planarity = features.get('planar_score')  # ❌ Old key name, returns None
```

**Solution**:

```python
planarity = features['planarity']  # ✅ New consistent naming
```

---

## Migration Checklist

### For Application Developers

- [ ] Review code for feature computation usage
- [ ] Identify files using old `features.features` imports
- [ ] Plan gradual migration (don't rush!)
- [ ] Update imports file-by-file as you make changes
- [ ] Run tests after each migration
- [ ] Update documentation with new examples

### For Library Developers

- [ ] Support both old and new APIs
- [ ] Add feature detection for available APIs
- [ ] Update examples to show new API
- [ ] Provide migration guide for users
- [ ] Plan deprecation timeline

### For Contributors

- [ ] Use `features.core` for all new code
- [ ] Add tests for new features (80%+ coverage)
- [ ] Follow type hints and documentation patterns
- [ ] Update relevant documentation
- [ ] Add examples in docstrings

---

## Getting Help

### Documentation

- **API Reference**: See docstrings in `features/core/` modules
- **Examples**: Check `examples/` directory
- **Tests**: Look at `tests/test_core_*.py` for usage examples

### Reporting Issues

If you encounter problems:

1. Check if using correct parameter names (`k_neighbors` not `k`)
2. Verify return value unpacking (functions return tuples)
3. Check dictionary keys (consistent naming now)
4. Review this migration guide
5. Open an issue on GitHub with minimal reproducible example

### Community

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share experiences
- Pull Requests: Contribute improvements

---

## Summary

### Key Takeaways

✅ **No Breaking Changes**: Your existing code works without modification

✅ **New Core Module**: Use `ign_lidar.features.core` for new code

✅ **Better APIs**: Consistent naming, more features, GPU support

✅ **Gradual Migration**: Update code incrementally, not all at once

✅ **6-Month Timeline**: Old APIs removed in v3.0.0 (Q1 2026)

### Quick Reference

| Old API              | New API         | Status         |
| -------------------- | --------------- | -------------- |
| `features.features`  | `features.core` | Both work      |
| `k` parameter        | `k_neighbors`   | Both work      |
| Single return value  | Tuple return    | Update code    |
| Manual tree building | Automatic       | Simplified     |
| Limited features     | Complete set    | More available |

### Next Steps

1. **Read**: Review relevant sections of this guide
2. **Experiment**: Try new APIs in a test script
3. **Plan**: Identify files to migrate gradually
4. **Migrate**: Update code file-by-file
5. **Test**: Verify behavior after each change
6. **Enjoy**: Benefit from better APIs and features!

---

**Document Version**: 1.0  
**Last Updated**: October 16, 2025  
**Applies To**: ign_lidar_hd v2.5.2+

For questions or feedback, please open an issue on GitHub.
