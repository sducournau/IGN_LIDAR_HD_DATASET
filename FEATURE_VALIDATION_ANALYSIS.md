# Feature Calculation Analysis & Validation Report

**Date**: October 12, 2025  
**Context**: Post-sphericity fix validation  
**Scope**: All geometric feature calculations across all computation modules

---

## Overview

After fixing the sphericity out-of-range issue in `features_boundary.py`, this report analyzes all feature calculations to ensure consistency and robustness across the codebase.

---

## Feature Calculation Methods

The codebase has **4 feature computation paths**:

1. **Boundary-aware** (`features_boundary.py`) - For tile stitching
2. **GPU** (`features_gpu.py`) - GPU-accelerated computation
3. **CPU radius-based** (`features.py` - loop) - Radius search with k-NN fallback
4. **CPU k-NN** (`features.py` - vectorized) - Fast k-NN computation

---

## Analysis by Feature

### ✅ **SPHERICITY** - FIXED

**Formula**: `λ2 / λ0` (or `λ3 / λ1` in boundary)  
**Expected Range**: [0, 1]

| Module     | Status         | Validation                                |
| ---------- | -------------- | ----------------------------------------- |
| Boundary   | ✅ **FIXED**   | Eigenvalue clamping + result clipping     |
| GPU        | ✅ OK          | NaN/Inf checking + zero invalid features  |
| CPU radius | ⚠️ **PARTIAL** | No clamping, relies on safe division only |
| CPU k-NN   | ✅ OK          | NaN/Inf checking + zero invalid features  |

---

### ⚠️ **PLANARITY** - NEEDS ATTENTION

**Formula**: `(λ1 - λ2) / λ0`  
**Expected Range**: [0, 1]

| Module     | Status             | Issue                                             |
| ---------- | ------------------ | ------------------------------------------------- |
| Boundary   | ✅ **FIXED**       | Eigenvalue clamping + result clipping             |
| GPU        | ⚠️ **NO CLIPPING** | Can produce values > 1 if λ1 < λ2 (sorting error) |
| CPU radius | ⚠️ **NO CLIPPING** | Can produce values > 1 if λ1 < λ2                 |
| CPU k-NN   | ⚠️ **NO CLIPPING** | Can produce values > 1 if λ1 < λ2                 |

**Potential Issue**: If eigenvalues aren't properly sorted or have numerical issues, `(λ1 - λ2)` could be negative, producing:

- Negative planarity values
- When divided by very small λ0, can exceed 1.0

---

### ⚠️ **LINEARITY** - NEEDS ATTENTION

**Formula**: `(λ0 - λ1) / λ0`  
**Expected Range**: [0, 1]

| Module     | Status             | Issue                                         |
| ---------- | ------------------ | --------------------------------------------- |
| Boundary   | ✅ **FIXED**       | Eigenvalue clamping + result clipping         |
| GPU        | ⚠️ **NO CLIPPING** | Theoretically OK but no explicit bounds check |
| CPU radius | ⚠️ **NO CLIPPING** | Same as GPU                                   |
| CPU k-NN   | ⚠️ **NO CLIPPING** | Same as GPU                                   |

**Note**: Should always be in [0,1] by definition, but numerical errors can cause issues.

---

### ⚠️ **ANISOTROPY** - NEEDS ATTENTION

**Formula**: `(λ0 - λ2) / λ0`  
**Expected Range**: [0, 1]

| Module     | Status              | Issue                                         |
| ---------- | ------------------- | --------------------------------------------- |
| Boundary   | ❌ **NOT COMPUTED** | Missing from boundary features!               |
| GPU        | ⚠️ **NO CLIPPING**  | Theoretically OK but no explicit bounds check |
| CPU radius | ⚠️ **NO CLIPPING**  | Same as GPU                                   |
| CPU k-NN   | ⚠️ **NO CLIPPING**  | Same as GPU                                   |

**Issues**:

1. **Missing in boundary computation** - inconsistency!
2. No explicit clamping in GPU/CPU
3. Can exceed 1.0 if λ2 becomes negative (numerical artifact)

---

### ⚠️ **ROUGHNESS** - NEEDS ATTENTION

**Formula**: `λ2 / (λ0 + λ1 + λ2)`  
**Expected Range**: [0, 1/3] theoretically, but can be [0, 1] with degenerate cases

| Module     | Status              | Issue                           |
| ---------- | ------------------- | ------------------------------- |
| Boundary   | ❌ **NOT COMPUTED** | Missing from boundary features! |
| GPU        | ⚠️ **NO CLIPPING**  | No explicit validation          |
| CPU radius | ⚠️ **NO CLIPPING**  | No explicit validation          |
| CPU k-NN   | ⚠️ **NO CLIPPING**  | No explicit validation          |

**Issues**:

1. **Missing in boundary computation** - inconsistency!
2. Can produce values > 1/3 if eigenvalues are degenerate
3. Negative eigenvalues would cause negative roughness

---

### ⚠️ **DENSITY** - NEEDS ATTENTION

**Formula**: `1 / mean_distance` or `N / volume`  
**Expected Range**: [0, ∞) - unbounded!

| Module     | Status              | Issue                                        |
| ---------- | ------------------- | -------------------------------------------- |
| Boundary   | ❌ **NOT COMPUTED** | Missing from boundary features!              |
| GPU        | ⚠️ **UNBOUNDED**    | No max limit, can be huge for dense clusters |
| CPU radius | ⚠️ **UNBOUNDED**    | Same as GPU                                  |
| CPU k-NN   | ⚠️ **UNBOUNDED**    | Same as GPU                                  |

**Issues**:

1. **Missing in boundary computation**
2. Unbounded values can cause issues in ML models
3. No normalization or max cap
4. Very dense point clusters (duplicates) can produce extreme values

---

### ✅ **VERTICALITY** - OK

**Formula**: `1 - |nz|` where nz is normal's Z component  
**Expected Range**: [0, 1]

| Module     | Status | Notes                                   |
| ---------- | ------ | --------------------------------------- |
| Boundary   | ✅ OK  | Computed from normals, guaranteed [0,1] |
| GPU        | ✅ OK  | Same formula, same guarantees           |
| CPU radius | ✅ OK  | Same formula, same guarantees           |
| CPU k-NN   | ✅ OK  | Same formula, same guarantees           |

**Status**: ✅ **No issues** - mathematically guaranteed to be in [0,1]

---

### ✅ **HORIZONTALITY** - OK

**Formula**: `|nz|` where nz is normal's Z component  
**Expected Range**: [0, 1]

| Module     | Status          | Notes                      |
| ---------- | --------------- | -------------------------- |
| Boundary   | ❌ NOT COMPUTED | Not in boundary features   |
| GPU        | ✅ OK           | Computed, guaranteed [0,1] |
| CPU radius | ✅ OK           | Same                       |
| CPU k-NN   | ✅ OK           | Same                       |

**Status**: ✅ **No issues** - mathematically guaranteed to be in [0,1]  
**Note**: Missing in boundary features but not critical

---

### ✅ **CURVATURE** - FIXED

**Formula**: `λ2 / (λ0 + λ1 + λ2)`  
**Expected Range**: [0, 1]

| Module   | Status             | Notes                                 |
| -------- | ------------------ | ------------------------------------- |
| Boundary | ✅ **FIXED**       | Eigenvalue clamping + result clipping |
| GPU      | ⚠️ **NO CLIPPING** | Relies on validation only             |
| CPU      | ⚠️ **NO CLIPPING** | Same as GPU                           |

---

## Critical Issues Found

### 🔴 **HIGH PRIORITY**

1. **Missing features in boundary computation**:

   - ❌ Anisotropy
   - ❌ Roughness
   - ❌ Density
   - ❌ Horizontality

   This creates **inconsistency** when boundary-aware features are used.

2. **No clamping in CPU radius-based features**:

   - Loop version in `features.py` (lines 580-620) has no clamping
   - Could produce out-of-range values like sphericity did

3. **Density is unbounded**:
   - Can produce extremely large values (thousands+)
   - May cause numerical instability in downstream ML models
   - Should be normalized or capped

### 🟡 **MEDIUM PRIORITY**

4. **GPU/CPU vectorized lack explicit clipping**:

   - Rely on validation masks (`valid_features`) to zero invalid
   - But don't clip to [0,1] for edge cases that pass validation
   - Could have values slightly > 1 due to floating point errors

5. **Eigenvalue sorting assumption**:
   - Code assumes `np.linalg.eigvalsh()` + sort produces λ0 ≥ λ1 ≥ λ2
   - Numerical errors could violate this
   - No explicit validation that λ0 ≥ λ1 ≥ λ2

---

## Recommendations

### **Immediate Actions**

1. ✅ **Apply same fix to CPU radius-based features** (loop version):

   ```python
   # Clamp eigenvalues to non-negative
   λ0, λ1, λ2 = max(λ0, 0), max(λ1, 0), max(λ2, 0)

   # Compute features
   linearity[i] = np.clip((λ0 - λ1) / λ0_safe, 0.0, 1.0)
   planarity[i] = np.clip((λ1 - λ2) / λ0_safe, 0.0, 1.0)
   sphericity[i] = np.clip(λ2 / λ0_safe, 0.0, 1.0)
   anisotropy[i] = np.clip((λ0 - λ2) / λ0_safe, 0.0, 1.0)
   roughness[i] = np.clip(λ2 / sum_λ, 0.0, 1.0)
   ```

2. ✅ **Add clipping to GPU features**:

   ```python
   # After computing features, before validation
   linearity = np.clip(linearity, 0.0, 1.0)
   planarity = np.clip(planarity, 0.0, 1.0)
   sphericity = np.clip(sphericity, 0.0, 1.0)
   anisotropy = np.clip(anisotropy, 0.0, 1.0)
   roughness = np.clip(roughness, 0.0, 1.0)
   ```

3. ✅ **Normalize or cap density**:

   ```python
   # Option 1: Cap at reasonable maximum
   density = np.clip(density, 0.0, 1000.0)

   # Option 2: Normalize to [0, 1] via log scale
   density = np.clip(np.log1p(density) / 10.0, 0.0, 1.0)
   ```

4. ✅ **Add missing features to boundary computation**:
   - Add anisotropy, roughness, density calculations
   - Ensures feature consistency across all code paths

### **Long-term Improvements**

5. **Centralize feature validation**:

   - Create a single `validate_geometric_features()` function
   - Call from all modules to ensure consistency
   - Reduces code duplication

6. **Add comprehensive unit tests**:

   - Test all features with edge cases
   - Verify ranges for all computation paths
   - Catch regressions early

7. **Document expected ranges**:
   - Add clear docstrings for each feature
   - Specify theoretical and practical ranges
   - Note when features can be unbounded

---

## Consistency Issues

### Formula Inconsistencies

**Current state**:

- GPU/Boundary use: `sphericity = λ2 / λ0` (λ0 normalization)
- CPU k-NN uses: `sphericity = λ2 / sum_λ` (sum normalization)

**Impact**: Different numerical values for same point cloud!

**Recommendation**: Standardize on **λ0 normalization** (Weinmann et al.) across all modules.

---

## Testing Strategy

Create comprehensive test suite:

```python
def test_feature_ranges():
    """Test all features stay in valid ranges for edge cases"""
    # Test cases:
    # 1. Negative eigenvalues
    # 2. Near-zero eigenvalues
    # 3. Equal eigenvalues (sphere)
    # 4. Degenerate (line, plane)
    # 5. Real LiDAR data

    for module in [boundary, gpu, cpu_radius, cpu_knn]:
        features = module.compute_features(test_data)

        assert features['sphericity'].min() >= 0.0
        assert features['sphericity'].max() <= 1.0
        assert features['planarity'].min() >= 0.0
        assert features['planarity'].max() <= 1.0
        # ... etc for all features
```

---

## Summary

| Feature     | Boundary   | GPU          | CPU Radius   | CPU k-NN     | Action Needed               |
| ----------- | ---------- | ------------ | ------------ | ------------ | --------------------------- |
| Sphericity  | ✅ Fixed   | ✅ OK        | ⚠️ Partial   | ✅ OK        | Add clipping to CPU radius  |
| Planarity   | ✅ Fixed   | ⚠️ No clip   | ⚠️ No clip   | ⚠️ No clip   | Add clipping everywhere     |
| Linearity   | ✅ Fixed   | ⚠️ No clip   | ⚠️ No clip   | ⚠️ No clip   | Add clipping everywhere     |
| Anisotropy  | ❌ Missing | ⚠️ No clip   | ⚠️ No clip   | ⚠️ No clip   | Add to boundary + clip      |
| Roughness   | ❌ Missing | ⚠️ No clip   | ⚠️ No clip   | ⚠️ No clip   | Add to boundary + clip      |
| Density     | ❌ Missing | ⚠️ Unbounded | ⚠️ Unbounded | ⚠️ Unbounded | Add to boundary + normalize |
| Verticality | ✅ OK      | ✅ OK        | ✅ OK        | ✅ OK        | ✅ None                     |
| Curvature   | ✅ Fixed   | ⚠️ No clip   | ⚠️ No clip   | N/A          | Add clipping                |

**Overall Status**: 🟡 **Multiple improvements needed for robustness**

---

## Next Steps

1. ✅ Implement clamping fixes for all geometric features
2. ✅ Add missing features to boundary computation
3. ✅ Normalize/cap density feature
4. ✅ Create comprehensive test suite
5. ✅ Run full pipeline validation
6. 📝 Update documentation with feature ranges
