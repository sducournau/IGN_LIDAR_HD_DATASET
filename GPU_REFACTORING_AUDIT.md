# GPU Implementation Refactoring Audit

**Date:** October 18, 2025  
**Scope:** Analysis of `features_gpu.py` and `features_gpu_chunked.py` for code duplication and refactoring opportunities using core implementations

---

## Executive Summary

This audit identifies significant code duplication between GPU implementations (`features_gpu.py`, `features_gpu_chunked.py`) and the canonical core implementations in `ign_lidar/features/core/`. The analysis reveals multiple opportunities for refactoring to:

1. **Reduce code duplication** (estimated 30-40% reduction in GPU module lines)
2. **Improve maintainability** (single source of truth for algorithms)
3. **Enhance consistency** (unified API across CPU/GPU paths)
4. **Simplify testing** (test core logic once, not per implementation)

---

## Current Architecture Overview

### Module Structure

```
ign_lidar/features/
├── core/                          # Canonical implementations (NEW, well-tested)
│   ├── __init__.py               # Unified API
│   ├── normals.py                # Standard normal computation
│   ├── curvature.py              # Curvature features
│   ├── eigenvalues.py            # Eigenvalue-based features
│   ├── density.py                # Density features
│   ├── architectural.py          # Architectural features
│   └── utils.py                  # Shared utilities
│
├── features_gpu.py               # GPU acceleration (CuPy/cuML)
└── features_gpu_chunked.py       # GPU with chunked processing
```

### Import Analysis

**features_gpu.py:**

```python
from ..features.core import (
    compute_normals as core_compute_normals,
    compute_curvature as core_compute_curvature,
    compute_eigenvalue_features as core_compute_eigenvalue_features,
    compute_density_features as core_compute_density_features,
    compute_verticality as core_compute_verticality,
    extract_geometric_features as core_extract_geometric_features,
)
```

✅ **Good:** Core imports present  
⚠️ **Problem:** Many imported but not consistently used (CPU fallbacks only)

**features_gpu_chunked.py:**

```python
from ..features.core import (
    compute_eigenvalue_features as core_compute_eigenvalue_features,
    compute_density_features as core_compute_density_features,
)
```

✅ **Good:** Some core usage  
⚠️ **Problem:** Incomplete - should use more core functions

---

## Detailed Duplication Analysis

### 1. ✅ **Eigenvalue Features** - WELL REFACTORED

**Core Implementation:** `core/eigenvalues.py::compute_eigenvalue_features()`

- ✅ Canonical algorithm in one place
- ✅ Well-documented with clear mathematical formulas
- ✅ Complete feature set: linearity, planarity, sphericity, anisotropy, eigenentropy, omnivariance

**GPU Usage:**

- ✅ `features_gpu.py` uses core implementation for CPU fallback (line 52 import)
- ✅ `features_gpu_chunked.py` uses core implementation (line 42 import)
- ✅ GPU implementations handle only GPU-specific optimizations (batching, transfers)

**Verdict:** ✅ **EXCELLENT** - This is the model for other features

---

### 2. ❌ **Height Above Ground** - COMPLETELY DUPLICATED

**Current State:**

- ❌ `features_gpu.py::compute_height_above_ground()` (lines 706-729) - **DUPLICATED LOGIC**
- ❌ Module-level wrapper function (lines 1313-1324) - **DUPLICATED WRAPPER**
- ❌ **NO CORE IMPLEMENTATION FOUND** in `core/` directory

**Code Duplication:**

```python
# features_gpu.py - line 706
def compute_height_above_ground(self, points, classification):
    ground_mask = (classification == 2)
    if not np.any(ground_mask):
        ground_z = np.min(points[:, 2])
    else:
        ground_z = np.min(points[ground_mask, 2])
    height = points[:, 2] - ground_z
    return np.maximum(height, 0)
```

**Recommendation:**

1. ✅ Create `core/height.py` with canonical implementation
2. ✅ Both GPU modules should import and use core version
3. ✅ GPU optimization only for large-scale parallelization (if beneficial)

**Priority:** 🔴 **HIGH** - Simple logic, easy refactor, immediate benefit

---

### 3. ⚠️ **Normals Computation** - PARTIALLY DUPLICATED

**Core Implementation:** `core/normals.py::compute_normals()`

- ✅ Standard CPU implementation with sklearn
- ✅ Returns both normals AND eigenvalues
- ⚠️ No GPU-specific optimizations

**GPU Implementations:**

**A. features_gpu.py:**

```python
# Line 210: compute_normals() - Uses cuML when available
def _compute_normals_gpu(self, points, k):
    # GPU-specific: cuML NearestNeighbors, CuPy arrays
    # Custom batched PCA with SVD optimization
    # Falls back to core_compute_normals for CPU
```

- ✅ Uses core for CPU fallback (line 220)
- ✅ GPU path is genuinely GPU-specific (cuML, batched operations)

**B. features_gpu_chunked.py:**

```python
# Line 520: compute_normals_chunked() - Chunked processing
def compute_normals_chunked(self, points, k):
    # Per-chunk KDTree strategy (lines 622-710)
    # Global KDTree strategy (lines 729-803)
    # Custom PCA implementation
```

- ⚠️ Does NOT use core implementation
- ⚠️ Has its own PCA logic (inverse power iteration)
- ❌ Could benefit from core imports

**Recommendation:**

1. ✅ Keep GPU-specific paths (cuML, chunking logic)
2. ⚠️ Consider extracting PCA logic to `core/normals.py::compute_pca_normals_batch()`
3. ⚠️ Ensure CPU fallback uses core consistently

**Priority:** 🟡 **MEDIUM** - GPU code is specialized, but PCA could be unified

---

### 4. ⚠️ **Curvature Computation** - PARTIALLY REFACTORED

**Core Implementation:** `core/curvature.py::compute_curvature()`

- ✅ Takes eigenvalues as input
- ✅ Multiple methods: 'standard', 'normalized', 'gaussian'
- ✅ Well-documented

**GPU Implementations:**

**A. features_gpu.py:**

```python
# Line 573: compute_curvature() - GPU-accelerated
def compute_curvature(self, points, normals, k):
    # Rebuilds KNN (could reuse from normals)
    # Custom curvature calculation from neighbor normals
    # CPU fallback to parallel KDTree
```

- ❌ Does NOT use `core/curvature.py`
- ⚠️ Different algorithm (neighbor normal variance vs eigenvalue ratio)
- ❌ Could at least use core for CPU fallback

**B. features_gpu_chunked.py:**

```python
# Line 1491: compute_curvature_chunked() - Chunked GPU curvature
def compute_curvature_chunked(self, points, normals, k):
    # Per-chunk strategy (lines 1591-1672)
    # Global KDTree strategy (lines 1499-1583)
    # Same algorithm as features_gpu.py
```

- ❌ Duplicates features_gpu.py logic
- ❌ Does NOT use core implementation

**Issue:** Different curvature algorithms across modules!

- Core: `λ3 / (λ1 + λ2 + λ3)` (standard eigenvalue-based)
- GPU: `std_dev(neighbor_normals)` (normal variation)

**Recommendation:**

1. 🔴 **CRITICAL:** Decide on canonical curvature algorithm
2. ✅ If eigenvalue-based: use `core/curvature.py` for CPU fallback
3. ✅ If normal-based: move algorithm to `core/curvature.py::compute_curvature_from_normals()`
4. ✅ Document which method to use when

**Priority:** 🔴 **HIGH** - Algorithm inconsistency is a correctness issue

---

### 5. ⚠️ **Verticality** - GOOD REFACTORING EXAMPLE

**Core Implementation:** `core/eigenvalues.py::compute_verticality()`

```python
def compute_verticality(normals: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(normals[:, 2])
```

**GPU Usage:**

- ✅ `features_gpu.py::compute_verticality()` (line 736-754) uses core for CPU path
- ✅ GPU path is minimal optimization (CuPy array operations)
- ✅ Wrapper function marked deprecated, redirects to core (lines 1356-1376)

**Verdict:** ✅ **EXCELLENT** - Good example of refactoring

---

### 6. ❌ **Geometric Features** - HEAVY DUPLICATION

**Core Implementation:** `core/eigenvalues.py::compute_eigenvalue_features()`

- ✅ Complete feature set
- ✅ Well-tested

**GPU Implementations:**

**A. features_gpu.py:**

```python
# Line 835: compute_geometric_features()
# Line 855: _compute_essential_geometric_features_optimized()
# Line 932: _compute_essential_geometric_features_cpu()
# Line 983: _compute_batch_eigenvalue_features_gpu()
# Line 1058: _compute_essential_geometric_features()
# Line 1137: _compute_batch_eigenvalue_features()
```

- ⚠️ Multiple implementations of same features
- ⚠️ Some use core (line 855), others don't
- ❌ Heavy code duplication

**B. features_gpu_chunked.py:**

```python
# Line 1809: compute_architectural_features()
# Line 1938: _compute_geometric_features_from_neighbors()
# Line 2090: compute_density_features()
```

- ⚠️ Partial core usage
- ❌ Much duplicated logic

**Recommendation:**

1. ✅ Consolidate to use `core/eigenvalues.py::compute_eigenvalue_features()`
2. ✅ GPU code should only handle:
   - Neighbor queries (KNN)
   - Batch transfers
   - Eigenvalue computation on GPU
   - Call core for feature derivation
3. ✅ Remove all duplicated eigenvalue → feature logic

**Priority:** 🔴 **HIGH** - Large duplication, maintenance burden

---

### 7. ⚠️ **Density Features** - PARTIAL REFACTORING

**Core Implementation:** `core/density.py::compute_density_features()`

- ✅ Complete density feature suite
- ✅ Well-documented

**GPU Usage:**

- ✅ Both modules import `core_compute_density_features`
- ⚠️ But have custom implementations for GPU paths
- ⚠️ Could use core more consistently

**Recommendation:**

1. ✅ Use core for CPU fallback
2. ⚠️ Evaluate if GPU-specific optimizations are necessary
3. ✅ If so, extract to core as `compute_density_features_batch()`

**Priority:** 🟡 **MEDIUM** - Working, but could be cleaner

---

## Specific Code Duplication Examples

### Example 1: Analytic 3x3 Matrix Inverse (CRITICAL DUPLICATION)

**Duplicated in:**

- `features_gpu.py::_batched_inverse_3x3()` (lines 377-431)
- `features_gpu_chunked.py::_batched_inverse_3x3_gpu()` (lines 1013-1070)

**EXACT SAME CODE:**

```python
def _batched_inverse_3x3(self, mats):
    a11 = mats[:, 0, 0]
    a12 = mats[:, 0, 1]
    # ... 60+ lines of identical code ...
    return inv
```

**Recommendation:**

1. ✅ Move to `core/utils.py::batched_inverse_3x3()`
2. ✅ Support both NumPy and CuPy arrays
3. ✅ Both GPU modules import from core

**Priority:** 🔴 **CRITICAL** - Exact duplication, 60+ lines

---

### Example 2: Inverse Power Iteration (CRITICAL DUPLICATION)

**Duplicated in:**

- `features_gpu.py::_smallest_eigenvector_from_covariances()` (lines 434-479)
- `features_gpu_chunked.py::_smallest_eigenvector_from_covariances_gpu()` (lines 1072-1118)

**NEAR-IDENTICAL CODE:**

```python
def _smallest_eigenvector_from_covariances(self, cov_matrices, num_iters=8):
    # Regularize
    cov = cov_matrices + reg * cp.eye(3, ...)
    # Compute inverse
    inv_cov = self._batched_inverse_3x3(cov)
    # Power iteration
    for _ in range(num_iters):
        v = inv_cov @ v[..., None]
        # ... normalization ...
    return v
```

**Recommendation:**

1. ✅ Move to `core/utils.py::inverse_power_iteration()`
2. ✅ Support both NumPy and CuPy
3. ✅ Parameterize iteration count

**Priority:** 🔴 **CRITICAL** - Core algorithm, duplicated

---

### Example 3: Eigenvalue-to-Feature Conversion (HIGH DUPLICATION)

**Scattered across:**

- `features_gpu.py::_compute_batch_eigenvalue_features_gpu()` (lines 983-1055)
- `features_gpu.py::_compute_batch_eigenvalue_features()` (lines 1137-1225)
- `features_gpu_chunked.py::_compute_minimal_eigenvalue_features()` (lines 2505-2630)

**All implement:**

```python
λ0, λ1, λ2 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
sum_λ = λ0 + λ1 + λ2
planarity = (λ1 - λ2) / (sum_λ + eps)
linearity = (λ0 - λ1) / (sum_λ + eps)
# ... etc ...
```

**Core already has this:** `core/eigenvalues.py::compute_eigenvalue_features()`

**Recommendation:**

1. ✅ Delete all GPU-side eigenvalue→feature logic
2. ✅ After computing eigenvalues on GPU, transfer to CPU
3. ✅ Call `core/eigenvalues.py::compute_eigenvalue_features()`
4. ⚠️ OR: Make core version GPU-compatible (accept CuPy arrays)

**Priority:** 🔴 **HIGH** - Major duplication, correctness risk

---

## Refactoring Recommendations

### Phase 1: Quick Wins (1-2 days) 🟢

#### 1.1 Extract Height Above Ground

- **Create:** `core/height.py`
- **Function:** `compute_height_above_ground(points, classification, method='ground_plane')`
- **Benefit:** Eliminate duplication in GPU modules
- **Effort:** 2 hours
- **Risk:** ✅ Low (simple logic)

#### 1.2 Move Matrix Utilities to Core

- **Create:** `core/utils.py::batched_inverse_3x3()` (or add if exists)
- **Create:** `core/utils.py::inverse_power_iteration()`
- **Benefit:** Eliminate 100+ lines of duplication
- **Effort:** 4 hours
- **Risk:** ✅ Low (pure math, well-tested)

#### 1.3 Standardize Curvature Algorithm

- **Decision:** Choose canonical algorithm (eigenvalue-based or normal-based)
- **Document:** When to use each method
- **Update:** All implementations to be consistent
- **Effort:** 3 hours
- **Risk:** ⚠️ Medium (algorithm change may affect results)

**Phase 1 Total:** ~9 hours, eliminates ~200 lines of duplication

---

### Phase 2: Core Integration (3-5 days) 🟡

#### 2.1 GPU-Compatible Core Utilities

**Goal:** Make core functions work with both NumPy and CuPy arrays

**Approach:**

```python
# core/utils.py
def get_array_module(arr):
    """Get numpy or cupy module for array."""
    if hasattr(arr, '__cuda_array_interface__'):
        import cupy as cp
        return cp
    return np

def batched_inverse_3x3(mats):
    """Compute 3x3 inverse - works with NumPy or CuPy."""
    xp = get_array_module(mats)
    # ... use xp instead of np/cp ...
```

**Benefits:**

- ✅ Single implementation for CPU and GPU
- ✅ Core functions usable in GPU context
- ✅ Easier testing

**Effort:** 2 days
**Risk:** ⚠️ Medium (needs careful testing)

#### 2.2 Refactor Geometric Features

- **Goal:** All eigenvalue→feature conversions use core
- **Delete:** GPU-side feature derivation code
- **Keep:** GPU-side eigenvalue computation (genuinely GPU-specific)
- **Effort:** 2 days
- **Risk:** ⚠️ Medium (many call sites)

#### 2.3 Unify CPU Fallback Paths

- **Goal:** All CPU fallbacks use core implementations
- **Update:** Replace custom CPU code with core imports
- **Effort:** 1 day
- **Risk:** ✅ Low (core is well-tested)

**Phase 2 Total:** ~5 days, major architecture improvement

---

### Phase 3: Advanced Optimization (1-2 weeks) 🔴

#### 3.1 Extract Common GPU Patterns

- **Create:** `core/gpu_utils.py` for common GPU operations
- **Patterns:** Chunked processing, stream management, memory pooling
- **Benefit:** Reduce code between features_gpu.py and features_gpu_chunked.py
- **Effort:** 1 week
- **Risk:** 🔴 High (complex, affects performance)

#### 3.2 Unified Feature API

- **Goal:** Single API across CPU/GPU/chunked
- **Design:**

  ```python
  from ign_lidar.features.core import compute_all_features

  features = compute_all_features(
      points,
      classification,
      backend='auto',  # 'cpu', 'gpu', 'gpu_chunked'
      **kwargs
  )
  ```

- **Effort:** 1 week
- **Risk:** 🔴 High (API redesign)

**Phase 3 Total:** ~2 weeks, major refactoring

---

## Priority Matrix

| Feature              | Duplication | Impact  | Effort  | Priority  |
| -------------------- | ----------- | ------- | ------- | --------- |
| Height Above Ground  | 🔴 High     | 🟢 Low  | 🟢 Low  | 🔴 HIGH   |
| Matrix Inverse Utils | 🔴 High     | 🟢 Low  | 🟢 Low  | 🔴 HIGH   |
| Curvature Algorithm  | 🔴 High     | 🔴 High | 🟡 Med  | 🔴 HIGH   |
| Eigenvalue Features  | 🔴 High     | 🟡 Med  | 🟡 Med  | 🔴 HIGH   |
| Normals PCA Logic    | 🟡 Med      | 🟡 Med  | 🟡 Med  | 🟡 MEDIUM |
| Density Features     | 🟡 Med      | 🟢 Low  | 🟢 Low  | 🟡 MEDIUM |
| GPU Core Compat      | 🟢 Low      | 🔴 High | 🔴 High | 🟡 MEDIUM |
| Unified API          | 🟢 Low      | 🔴 High | 🔴 High | 🟢 LOW    |

---

## Testing Strategy

### Regression Testing

1. ✅ Before refactoring: Capture outputs of current implementations
2. ✅ After refactoring: Verify outputs match (within numerical tolerance)
3. ✅ Use existing test suite as baseline

### Performance Testing

1. ✅ Benchmark before/after for each change
2. ✅ Acceptable threshold: <5% performance regression
3. ✅ Target: Maintain or improve performance

### Integration Testing

1. ✅ Test CPU, GPU, and chunked paths
2. ✅ Test with real datasets (small, medium, large)
3. ✅ Verify end-to-end pipeline still works

---

## API Compatibility Concerns

### Breaking Changes (Need Deprecation Warnings)

#### 1. Curvature Algorithm Change

**If we unify to eigenvalue-based:**

```python
# OLD (normal-based): std_dev of neighbor normals
# NEW (eigenvalue-based): λ3 / (λ1 + λ2 + λ3)
```

**Impact:** Results will differ for existing users
**Mitigation:**

- Add `method='legacy'` parameter for backward compat
- Deprecate legacy method over 2-3 releases

#### 2. Feature Name Changes

**Current:** Inconsistent naming (`change_curvature` vs `change_of_curvature`)
**Proposed:** Standardize to core names
**Mitigation:**

- Provide aliases for old names
- Deprecation warnings
- Update documentation

### Non-Breaking Changes (Safe)

✅ Internal implementation changes (same API, same results)
✅ Performance improvements
✅ Bug fixes
✅ Adding new features

---

## Metrics & Success Criteria

### Code Quality Metrics

**Before Refactoring:**

- Total lines in GPU modules: ~4,800 lines
- Duplicated logic: ~1,200 lines (25%)
- Core imports used: ~40%

**After Refactoring (Phase 1+2):**

- Target reduction: -30% lines in GPU modules (~1,440 lines removed)
- Duplicated logic: <5%
- Core imports used: ~80%

### Maintainability Metrics

**Before:**

- Feature implementations: 3× (CPU, GPU, GPU-chunked)
- Testing burden: 3× test suites
- Bug fix propagation: Manual across 3 modules

**After:**

- Feature implementations: 1× (core) + GPU optimizations
- Testing burden: 1× core tests + GPU adapter tests
- Bug fix propagation: Automatic (core changes propagate)

---

## Risks & Mitigation

### Risk 1: Performance Regression

**Likelihood:** Medium  
**Impact:** High  
**Mitigation:**

- Benchmark every change
- Keep GPU-specific optimizations (don't force everything through core)
- Profile before/after

### Risk 2: Numerical Differences

**Likelihood:** Medium  
**Impact:** Medium  
**Mitigation:**

- Use strict numerical tolerance tests
- Document algorithm changes
- Provide legacy compatibility mode

### Risk 3: Breaking User Code

**Likelihood:** Low  
**Impact:** High  
**Mitigation:**

- Deprecation warnings (not immediate removal)
- Maintain API compatibility where possible
- Clear migration guide

### Risk 4: Incomplete Refactoring

**Likelihood:** High  
**Impact:** Medium  
**Mitigation:**

- Phased approach (don't try everything at once)
- Focus on high-priority items first
- Document what's left for future

---

## Recommended Action Plan

### Immediate (This Week)

1. ✅ Create `core/height.py` with `compute_height_above_ground()`
2. ✅ Move `batched_inverse_3x3()` to `core/utils.py`
3. ✅ Move `inverse_power_iteration()` to `core/utils.py`
4. ✅ Update GPU modules to use new core utilities

**Deliverable:** PR #1 - Core utilities extraction (~200 lines removed)

### Short-Term (Next 2 Weeks)

1. ✅ Decide canonical curvature algorithm
2. ✅ Refactor all curvature implementations to use core
3. ✅ Refactor eigenvalue→feature conversions to use core
4. ✅ Ensure all CPU fallbacks use core

**Deliverable:** PR #2 - Feature computation unification (~400 lines removed)

### Medium-Term (Next Month)

1. ✅ Make core utilities GPU-compatible (NumPy/CuPy agnostic)
2. ✅ Extract common GPU patterns
3. ✅ Comprehensive testing and benchmarking

**Deliverable:** PR #3 - GPU-compatible core (~600 lines removed)

### Long-Term (Next Quarter)

1. ⚠️ Consider unified feature API
2. ⚠️ Advanced GPU optimizations
3. ⚠️ Documentation and migration guide

**Deliverable:** PR #4 - Unified architecture (optional)

---

## Conclusion

The GPU implementations have significant code duplication that can be reduced by leveraging the well-designed core module. The recommended phased approach:

1. **Phase 1 (Quick Wins):** Extract obvious utilities → ~200 lines removed
2. **Phase 2 (Core Integration):** Unify feature computation → ~400 lines removed
3. **Phase 3 (Advanced):** GPU-compatible core → ~600 lines removed

**Total Estimated Reduction:** ~1,200 lines (25% of GPU modules)
**Estimated Effort:** 2-3 weeks full-time
**Risk Level:** Medium (with proper testing)
**Benefit:** High (maintainability, consistency, correctness)

**Recommendation:** ✅ **Proceed with Phase 1 immediately**, then evaluate Phase 2 based on results.

---

## Appendix: Core Module Coverage

### Already Well-Implemented in Core ✅

- Eigenvalue features (linearity, planarity, sphericity, anisotropy, etc.)
- Curvature computation (eigenvalue-based)
- Density features
- Architectural features
- Verticality computation

### Missing from Core (Should Add) ⚠️

- Height above ground computation
- 3x3 matrix inverse (batched)
- Inverse power iteration for eigenvectors
- Normal-based curvature (if we want to keep it)

### GPU-Specific (Keep in GPU Modules) ✅

- cuML integration (NearestNeighbors, PCA)
- CUDA stream management
- Memory pooling
- Chunked processing logic
- GPU memory management

---

**End of Audit Report**
