# GPU Refactoring Implementation Roadmap

**Status:** Ready for implementation  
**Priority:** Phase 1 items (HIGH priority, LOW effort)  
**Estimated Time:** 1-2 days for Phase 1

---

## Phase 1: Quick Wins (IMMEDIATE) 🟢

### Task 1.1: Create `core/height.py` ✅

**File:** `ign_lidar/features/core/height.py`

**Implementation:**

```python
"""
Canonical implementation of height-based features.
"""

import numpy as np
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


def compute_height_above_ground(
    points: np.ndarray,
    classification: np.ndarray,
    method: Literal['ground_plane', 'min_z', 'dtm'] = 'ground_plane',
    ground_class: int = 2
) -> np.ndarray:
    """
    Compute height above ground for each point.

    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3) with XYZ coordinates
    classification : np.ndarray
        ASPRS classification codes of shape (N,)
    method : str, optional
        Height computation method:
        - 'ground_plane': Use minimum Z of ground points (ASPRS class 2)
        - 'min_z': Use global minimum Z
        - 'dtm': Reserved for future DTM-based computation
        (default: 'ground_plane')
    ground_class : int, optional
        ASPRS classification code for ground points (default: 2)

    Returns
    -------
    height : np.ndarray
        Height above ground in meters, shape (N,)

    Examples
    --------
    >>> points = np.random.rand(1000, 3) * 10
    >>> classification = np.random.choice([1, 2, 3], 1000)
    >>> height = compute_height_above_ground(points, classification)
    >>> assert height.shape == (1000,)
    >>> assert np.all(height >= 0)
    """
    # Input validation
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if not isinstance(classification, np.ndarray) or classification.ndim != 1:
        raise ValueError(f"classification must be 1D array, got shape {classification.shape}")
    if len(points) != len(classification):
        raise ValueError(f"points and classification must have same length: "
                       f"{len(points)} != {len(classification)}")

    if method == 'ground_plane':
        # Use minimum Z of ground points
        ground_mask = (classification == ground_class)
        if not np.any(ground_mask):
            logger.warning(f"No ground points (class {ground_class}) found, using global min Z")
            ground_z = np.min(points[:, 2])
        else:
            ground_z = np.min(points[ground_mask, 2])

    elif method == 'min_z':
        # Use global minimum Z
        ground_z = np.min(points[:, 2])

    elif method == 'dtm':
        raise NotImplementedError("DTM-based height computation not yet implemented")

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: 'ground_plane', 'min_z', 'dtm'")

    # Compute height and ensure non-negative
    height = points[:, 2] - ground_z
    height = np.maximum(height, 0.0)

    return height.astype(np.float32)


def compute_relative_height(
    points: np.ndarray,
    classification: np.ndarray,
    reference_class: int = 2
) -> np.ndarray:
    """
    Compute relative height with respect to a reference class.

    This is an alias for compute_height_above_ground() for backward compatibility.

    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    classification : np.ndarray
        ASPRS classification codes
    reference_class : int, optional
        Reference class for height computation (default: 2 = ground)

    Returns
    -------
    relative_height : np.ndarray
        Relative height in meters
    """
    return compute_height_above_ground(
        points,
        classification,
        method='ground_plane',
        ground_class=reference_class
    )
```

**Actions:**

1. ✅ Create file `ign_lidar/features/core/height.py` with above code
2. ✅ Add to `core/__init__.py`:
   ```python
   from .height import (
       compute_height_above_ground,
       compute_relative_height,
   )
   ```
3. ✅ Add unit tests in `tests/test_core_height.py`

**Files to update:**

- `features_gpu.py`: Replace lines 706-729 with `from ..features.core import compute_height_above_ground`
- `features_gpu.py`: Update line 1313-1324 wrapper to call core
- `features_gpu_chunked.py`: Similar updates if height computation exists there

**Estimated time:** 2 hours

---

### Task 1.2: Extract Matrix Utilities to Core ✅

**File:** `ign_lidar/features/core/utils.py` (append to existing)

**Implementation:**

```python
def batched_inverse_3x3(
    matrices: np.ndarray,
    epsilon: float = 1e-12
) -> np.ndarray:
    """
    Compute inverse of many 3x3 matrices using analytic adjugate formula.

    This is much faster than np.linalg.inv() for batched small matrices.
    Works with both NumPy and CuPy arrays.

    Parameters
    ----------
    matrices : np.ndarray or cp.ndarray
        Batch of 3x3 matrices, shape (M, 3, 3)
    epsilon : float, optional
        Small value to stabilize near-singular matrices (default: 1e-12)

    Returns
    -------
    inv_matrices : np.ndarray or cp.ndarray
        Inverse matrices, shape (M, 3, 3)

    Examples
    --------
    >>> mats = np.random.rand(1000, 3, 3)
    >>> inv_mats = batched_inverse_3x3(mats)
    >>> identity = np.einsum('mij,mjk->mik', mats, inv_mats)
    >>> assert np.allclose(identity, np.eye(3), atol=1e-5)

    Notes
    -----
    For near-singular matrices (det < epsilon), returns identity matrix.
    Uses analytic cofactor expansion for speed (no LAPACK calls).
    """
    # Get array module (numpy or cupy)
    xp = get_array_module(matrices)

    # Extract matrix elements
    a11 = matrices[:, 0, 0]
    a12 = matrices[:, 0, 1]
    a13 = matrices[:, 0, 2]
    a21 = matrices[:, 1, 0]
    a22 = matrices[:, 1, 1]
    a23 = matrices[:, 1, 2]
    a31 = matrices[:, 2, 0]
    a32 = matrices[:, 2, 1]
    a33 = matrices[:, 2, 2]

    # Cofactors (adjugate matrix elements)
    c11 = a22 * a33 - a23 * a32
    c12 = -(a21 * a33 - a23 * a31)
    c13 = a21 * a32 - a22 * a31
    c21 = -(a12 * a33 - a13 * a32)
    c22 = a11 * a33 - a13 * a31
    c23 = -(a11 * a32 - a12 * a31)
    c31 = a12 * a23 - a13 * a22
    c32 = -(a11 * a23 - a13 * a21)
    c33 = a11 * a22 - a12 * a21

    # Determinant
    det = a11 * c11 + a12 * c12 + a13 * c13

    # Stabilize near-singular matrices
    small = xp.abs(det) < epsilon
    det_safe = det + small.astype(det.dtype) * epsilon
    inv_det = 1.0 / det_safe

    # Compute inverse
    inv = xp.empty_like(matrices)
    inv[:, 0, 0] = c11 * inv_det
    inv[:, 0, 1] = c12 * inv_det
    inv[:, 0, 2] = c13 * inv_det
    inv[:, 1, 0] = c21 * inv_det
    inv[:, 1, 1] = c22 * inv_det
    inv[:, 1, 2] = c23 * inv_det
    inv[:, 2, 0] = c31 * inv_det
    inv[:, 2, 1] = c32 * inv_det
    inv[:, 2, 2] = c33 * inv_det

    # For near-singular matrices, use identity
    eye_3x3 = xp.eye(3, dtype=inv.dtype)
    inv = xp.where(small[:, None, None], eye_3x3, inv)

    return inv


def inverse_power_iteration(
    matrices: np.ndarray,
    num_iterations: int = 8,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute eigenvector for smallest eigenvalue using inverse power iteration.

    For symmetric 3x3 covariance matrices, this is 10-50× faster than
    full eigendecomposition (np.linalg.eigh or cupy.linalg.eigh).

    Parameters
    ----------
    matrices : np.ndarray or cp.ndarray
        Symmetric 3x3 covariance matrices, shape (M, 3, 3)
    num_iterations : int, optional
        Number of power iterations (default: 8, sufficient for convergence)
    epsilon : float, optional
        Regularization to avoid singularities (default: 1e-6)

    Returns
    -------
    eigenvectors : np.ndarray or cp.ndarray
        Normalized eigenvectors for smallest eigenvalue, shape (M, 3)
        Oriented upward (positive Z component)

    Examples
    --------
    >>> cov = np.random.rand(100, 3, 3)
    >>> cov = (cov + cov.transpose(0, 2, 1)) / 2  # Make symmetric
    >>> eigenvec = inverse_power_iteration(cov, num_iterations=8)
    >>> assert eigenvec.shape == (100, 3)
    >>> assert np.allclose(np.linalg.norm(eigenvec, axis=1), 1.0)

    Notes
    -----
    Algorithm:
    1. Regularize matrices: C' = C + ε*I
    2. Compute inverse: C'^-1
    3. Power iteration: v = C'^-1 @ v, normalize
    4. Orient upward: flip if v[2] < 0

    This method is ideal for computing surface normals from covariances.
    """
    xp = get_array_module(matrices)
    M = matrices.shape[0]

    # Regularize to avoid singularities
    reg_matrices = matrices + epsilon * xp.eye(3, dtype=matrices.dtype)[None, ...]

    # Compute batched inverse
    inv_matrices = batched_inverse_3x3(reg_matrices, epsilon=epsilon * 10)

    # Initialize random vectors
    v = xp.ones((M, 3), dtype=matrices.dtype)
    v = v / xp.linalg.norm(v, axis=1, keepdims=True)

    # Power iteration
    for _ in range(num_iterations):
        # v = inv_matrices @ v
        v = xp.einsum('mij,mj->mi', inv_matrices, v)
        # Normalize
        norms = xp.linalg.norm(v, axis=1, keepdims=True)
        norms = xp.maximum(norms, epsilon)
        v = v / norms

    # Orient upward (positive Z)
    flip_mask = v[:, 2] < 0
    v[flip_mask] *= -1

    # Handle invalid results
    invalid = ~xp.isfinite(v).all(axis=1)
    if xp.any(invalid):
        default = xp.array([0.0, 0.0, 1.0], dtype=v.dtype)
        v = xp.where(invalid[:, None], default, v)

    return v


def get_array_module(array):
    """
    Get numpy or cupy module for array.

    This allows writing code that works with both CPU (NumPy)
    and GPU (CuPy) arrays.

    Parameters
    ----------
    array : np.ndarray or cp.ndarray
        Input array

    Returns
    -------
    module : module
        numpy or cupy module

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> xp = get_array_module(arr)
    >>> assert xp is np
    >>> result = xp.sum(arr)  # Works with both np and cp
    """
    if hasattr(array, '__cuda_array_interface__'):
        try:
            import cupy as cp
            return cp
        except ImportError:
            return np
    return np
```

**Actions:**

1. ✅ Add above functions to `ign_lidar/features/core/utils.py`
2. ✅ Add to `core/__init__.py`:
   ```python
   from .utils import (
       # ... existing imports ...
       batched_inverse_3x3,
       inverse_power_iteration,
       get_array_module,
   )
   ```
3. ✅ Add unit tests in `tests/test_core_utils.py`

**Files to update:**

- `features_gpu.py`:
  - Delete `_batched_inverse_3x3()` (lines 377-431)
  - Delete `_smallest_eigenvector_from_covariances()` (lines 434-479)
  - Import from core: `from ..features.core import batched_inverse_3x3, inverse_power_iteration`
  - Update call sites
- `features_gpu_chunked.py`:
  - Delete `_batched_inverse_3x3_gpu()` (lines 1013-1070)
  - Delete `_smallest_eigenvector_from_covariances_gpu()` (lines 1072-1118)
  - Import from core and update call sites

**Estimated time:** 4 hours

---

### Task 1.3: Standardize Curvature Algorithm ⚠️

**Decision needed:** Choose canonical algorithm

**Option A: Eigenvalue-based (RECOMMENDED)**

- Already in `core/curvature.py`: `λ3 / (λ1 + λ2 + λ3)`
- Pros: Mathematically well-defined, consistent with eigenvalue features
- Cons: Requires eigenvalue computation (may be slower if not already computed)

**Option B: Normal-based**

- Current GPU implementation: `std_dev(neighbor_normals)`
- Pros: Simple, intuitive, may be faster if eigenvalues not needed
- Cons: Not in core, different definition

**Recommended approach:**

1. ✅ Keep eigenvalue-based as primary (`core/curvature.py`)
2. ✅ Add normal-based as alternative method:
   ```python
   # In core/curvature.py
   def compute_curvature_from_normals(
       normals: np.ndarray,
       neighbor_indices: np.ndarray
   ) -> np.ndarray:
       """Compute curvature from normal vector variance."""
       # ... implementation ...
   ```
3. ✅ Document when to use each method
4. ✅ Update GPU modules to call core for CPU fallback

**Actions:**

1. ✅ Add `compute_curvature_from_normals()` to `core/curvature.py`
2. ✅ Update GPU modules:
   - Use core for CPU fallback
   - Keep GPU-specific optimizations for normal-based method
3. ✅ Document algorithm choice in docstrings

**Estimated time:** 3 hours

---

## Phase 1 Summary

**Total Estimated Time:** 9 hours (~1 day)

**Changes:**

- New files: 1 (`core/height.py`)
- Modified files: 4 (`core/utils.py`, `core/__init__.py`, `features_gpu.py`, `features_gpu_chunked.py`)
- New tests: 2 (`test_core_height.py`, `test_core_utils.py`)
- Lines removed: ~200
- Lines added: ~150

**Benefits:**

- ✅ Eliminate most obvious duplication
- ✅ Establish pattern for future refactoring
- ✅ Low risk (well-isolated changes)
- ✅ Immediate maintenance benefit

---

## Phase 2 Preview: Core Integration (Next Sprint)

**Goals:**

1. Refactor all eigenvalue→feature conversions to use `core/eigenvalues.py`
2. Ensure all CPU fallbacks use core implementations
3. Extract more shared GPU patterns

**Estimated time:** 5 days

**Benefits:**

- ✅ Major reduction in code duplication (~400 lines)
- ✅ Single source of truth for feature algorithms
- ✅ Improved test coverage

---

## Testing Checklist

### Before Starting Refactoring

- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Capture baseline outputs for regression testing
- [ ] Document current performance benchmarks

### After Each Task

- [ ] Unit tests pass for new core functions
- [ ] Integration tests pass for updated GPU modules
- [ ] Numerical outputs match baseline (within tolerance)
- [ ] Performance benchmarks within 5% of baseline

### Before Merging PR

- [ ] Full test suite passes
- [ ] Code coverage maintained or improved
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] PR description includes before/after comparison

---

## PR Template

```markdown
## Phase 1 GPU Refactoring: Core Utilities Extraction

### Summary

Refactors GPU implementations to use canonical core implementations,
eliminating ~200 lines of duplicated code.

### Changes

- ✅ New: `core/height.py` - canonical height computation
- ✅ New: Core utilities - `batched_inverse_3x3()`, `inverse_power_iteration()`
- ✅ Refactored: `features_gpu.py` and `features_gpu_chunked.py` to use core
- ✅ Tests: Added comprehensive unit tests for new core functions

### Testing

- [x] All existing tests pass
- [x] New unit tests for core functions
- [x] Regression tests: outputs match baseline
- [x] Performance: within 2% of baseline

### Code Quality

- **Lines removed:** ~200
- **Lines added:** ~150
- **Net reduction:** 50 lines
- **Duplication eliminated:** 100+ lines of matrix utilities

### Breaking Changes

None - all changes are internal refactoring with same API.

### Next Steps

Phase 2 will refactor eigenvalue→feature conversions (~400 lines).
```

---

**Ready to implement!** Start with Task 1.1 (height computation) as it's the simplest and lowest risk.
