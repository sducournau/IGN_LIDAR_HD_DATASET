# Feature Validation Fix Implementation Plan

**Date**: October 12, 2025  
**Author**: IGN LiDAR HD Team  
**Sprint**: Feature Robustness & Consistency  
**Priority**: HIGH

---

## Executive Summary

This plan addresses critical issues found in geometric feature calculations across all computation modules. The fixes will ensure:

- ✅ All features stay within valid ranges
- ✅ Consistent feature computation across all code paths
- ✅ Robust handling of numerical edge cases
- ✅ Complete feature sets for boundary-aware computation

**Estimated Time**: 4-6 hours  
**Risk Level**: LOW (backward compatible, additive changes)  
**Testing Required**: YES (comprehensive)

---

## Phase 1: Critical Fixes (HIGH PRIORITY)

### 1.1 Fix CPU Radius-Based Features (Loop Version)

**File**: `ign_lidar/features/features.py`  
**Lines**: ~580-620  
**Issue**: No eigenvalue validation or result clamping  
**Risk**: Can produce out-of-range values like sphericity did

**Implementation**:

```python
# BEFORE (line ~580-595)
λ0, λ1, λ2 = eigenvals[0], eigenvals[1], eigenvals[2]
sum_λ = λ0 + λ1 + λ2 + 1e-8
λ0_safe = λ0 + 1e-8

linearity[i] = (λ0 - λ1) / λ0_safe
planarity[i] = (λ1 - λ2) / λ0_safe
sphericity[i] = λ2 / λ0_safe
anisotropy[i] = (λ0 - λ2) / λ0_safe
roughness[i] = λ2 / sum_λ

# AFTER (proposed fix)
λ0, λ1, λ2 = eigenvals[0], eigenvals[1], eigenvals[2]

# Clamp eigenvalues to non-negative (handle numerical artifacts)
λ0 = max(λ0, 0.0)
λ1 = max(λ1, 0.0)
λ2 = max(λ2, 0.0)

sum_λ = λ0 + λ1 + λ2 + 1e-8
λ0_safe = λ0 + 1e-8

# Compute features with clamping to valid range
linearity[i] = np.clip((λ0 - λ1) / λ0_safe, 0.0, 1.0)
planarity[i] = np.clip((λ1 - λ2) / λ0_safe, 0.0, 1.0)
sphericity[i] = np.clip(λ2 / λ0_safe, 0.0, 1.0)
anisotropy[i] = np.clip((λ0 - λ2) / λ0_safe, 0.0, 1.0)
roughness[i] = np.clip(λ2 / sum_λ, 0.0, 1.0)
```

**Test Cases**:

- Negative eigenvalues
- Near-zero eigenvalues
- Equal eigenvalues (sphere)
- Real LiDAR data

---

### 1.2 Add Explicit Clamping to GPU Features

**File**: `ign_lidar/features/features_gpu.py`  
**Lines**: ~394 (after feature computation, before validation)  
**Issue**: Relies only on validation masks, no explicit [0,1] bounds  
**Risk**: Floating point errors can produce values slightly > 1

**Implementation**:

```python
# BEFORE (line ~390-398)
linearity = ((λ0 - λ1) / λ0_safe).astype(np.float32)
planarity = ((λ1 - λ2) / λ0_safe).astype(np.float32)
sphericity = (λ2 / λ0_safe).astype(np.float32)
anisotropy = ((λ0 - λ2) / λ0_safe).astype(np.float32)
roughness = (λ2 / sum_λ).astype(np.float32)

# AFTER (proposed fix)
# Clamp eigenvalues first
λ0 = np.maximum(λ0, 0.0)
λ1 = np.maximum(λ1, 0.0)
λ2 = np.maximum(λ2, 0.0)

# Recompute safe divisors
λ0_safe = λ0 + 1e-8
sum_λ = λ0 + λ1 + λ2 + 1e-8

# Compute and clamp features
linearity = np.clip((λ0 - λ1) / λ0_safe, 0.0, 1.0).astype(np.float32)
planarity = np.clip((λ1 - λ2) / λ0_safe, 0.0, 1.0).astype(np.float32)
sphericity = np.clip(λ2 / λ0_safe, 0.0, 1.0).astype(np.float32)
anisotropy = np.clip((λ0 - λ2) / λ0_safe, 0.0, 1.0).astype(np.float32)
roughness = np.clip(λ2 / sum_λ, 0.0, 1.0).astype(np.float32)
```

**Also apply to**:

- Chunked GPU computation (line ~820)
- Any other GPU feature computation paths

---

### 1.3 Add Explicit Clamping to CPU k-NN Features

**File**: `ign_lidar/features/features.py`  
**Lines**: ~1030  
**Issue**: Same as GPU - relies only on validation  
**Risk**: Same floating point edge cases

**Implementation**:

```python
# BEFORE (line ~1020-1032)
linearity = ((λ0 - λ1) / sum_λ).astype(np.float32)
planarity = ((λ1 - λ2) / sum_λ).astype(np.float32)
sphericity = (λ2 / sum_λ).astype(np.float32)
anisotropy = ((λ0 - λ2) / λ0_safe).astype(np.float32)
roughness = (λ2 / sum_λ).astype(np.float32)

# AFTER (proposed fix)
# Clamp eigenvalues first
eigenvalues_sorted = np.maximum(eigenvalues_sorted, 0.0)

λ0 = eigenvalues_sorted[:, 0]
λ1 = eigenvalues_sorted[:, 1]
λ2 = eigenvalues_sorted[:, 2]

λ0_safe = λ0 + 1e-8
sum_λ = λ0 + λ1 + λ2 + 1e-8

# Compute and clamp features
linearity = np.clip((λ0 - λ1) / sum_λ, 0.0, 1.0).astype(np.float32)
planarity = np.clip((λ1 - λ2) / sum_λ, 0.0, 1.0).astype(np.float32)
sphericity = np.clip(λ2 / sum_λ, 0.0, 1.0).astype(np.float32)
anisotropy = np.clip((λ0 - λ2) / λ0_safe, 0.0, 1.0).astype(np.float32)
roughness = np.clip(λ2 / sum_λ, 0.0, 1.0).astype(np.float32)
```

---

### 1.4 Normalize/Cap Density Feature

**Files**: All feature computation modules  
**Issue**: Unbounded values can be huge (1000+)  
**Risk**: Numerical instability in ML models

**Option A: Cap at Maximum** (RECOMMENDED - simpler)

```python
# Cap density at reasonable maximum (e.g., 1000 points/m³)
density = np.clip(density, 0.0, 1000.0).astype(np.float32)
```

**Option B: Log Normalization** (better for ML but changes scale)

```python
# Normalize via log scale to [0, 1]
density = np.clip(np.log1p(density) / 10.0, 0.0, 1.0).astype(np.float32)
```

**Recommendation**: Use **Option A** for backward compatibility. Add config flag for Option B if needed.

**Apply to**:

- `features.py` (both radius and k-NN versions)
- `features_gpu.py`
- `features_gpu_chunked.py`

---

## Phase 2: Consistency Fixes (HIGH PRIORITY)

### 2.1 Add Missing Features to Boundary Computation

**File**: `ign_lidar/features/features_boundary.py`  
**Lines**: Add to `compute_features()` method  
**Issue**: Missing anisotropy, roughness, density, horizontality  
**Impact**: Feature inconsistency when tile stitching enabled

**Implementation**:

Add to `_compute_planarity_features()` return dict:

```python
def _compute_planarity_features(
    self,
    eigenvalues: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute planarity, linearity, sphericity, anisotropy, and roughness.
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]

    # Clamp eigenvalues
    lambda1 = np.maximum(lambda1, 0.0)
    lambda2 = np.maximum(lambda2, 0.0)
    lambda3 = np.maximum(lambda3, 0.0)

    eps = 1e-10
    sum_lambda = lambda1 + lambda2 + lambda3 + eps

    # Compute all geometric features
    planarity = np.clip(
        np.where(lambda1 > eps, (lambda2 - lambda3) / lambda1, 0.0),
        0.0, 1.0
    )
    linearity = np.clip(
        np.where(lambda1 > eps, (lambda1 - lambda2) / lambda1, 0.0),
        0.0, 1.0
    )
    sphericity = np.clip(
        np.where(lambda1 > eps, lambda3 / lambda1, 0.0),
        0.0, 1.0
    )

    # NEW: Add anisotropy
    anisotropy = np.clip(
        np.where(lambda1 > eps, (lambda1 - lambda3) / lambda1, 0.0),
        0.0, 1.0
    )

    # NEW: Add roughness
    roughness = np.clip(lambda3 / sum_lambda, 0.0, 1.0)

    return {
        'planarity': planarity.astype(np.float32),
        'linearity': linearity.astype(np.float32),
        'sphericity': sphericity.astype(np.float32),
        'anisotropy': anisotropy.astype(np.float32),  # NEW
        'roughness': roughness.astype(np.float32),    # NEW
    }
```

Add new method `_compute_density()`:

```python
def _compute_density(
    self,
    neighbor_indices: np.ndarray,
    radius: float
) -> np.ndarray:
    """
    Compute local point density.

    Args:
        neighbor_indices: (N, K) Neighbor indices
        radius: Search radius

    Returns:
        (N,) Density values (points per cubic meter)
    """
    # Number of neighbors per point
    num_neighbors = np.array([len(neighbors) for neighbors in neighbor_indices])

    # Density = points / volume
    volume = (4/3) * np.pi * (radius ** 3) + 1e-8
    density = num_neighbors / volume

    # Cap at reasonable maximum
    density = np.clip(density, 0.0, 1000.0)

    return density.astype(np.float32)
```

Add to `_compute_verticality()` method:

```python
def _compute_verticality_and_horizontality(
    self,
    normals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute verticality and horizontality from normals.

    Returns:
        verticality: (N,) values [0,1] (1=vertical)
        horizontality: (N,) values [0,1] (1=horizontal)
    """
    verticality = (1.0 - np.abs(normals[:, 2])).astype(np.float32)
    horizontality = np.abs(normals[:, 2]).astype(np.float32)

    return verticality, horizontality
```

Update `compute_features()` to return all features:

```python
# In compute_features() method, add:
features['anisotropy'] = planarity_features['anisotropy']
features['roughness'] = planarity_features['roughness']
features['density'] = self._compute_density(neighbor_indices, self.k_neighbors)
verticality, horizontality = self._compute_verticality_and_horizontality(normals)
features['verticality'] = verticality
features['horizontality'] = horizontality
```

---

### 2.2 Standardize Feature Formulas

**Issue**: Inconsistent normalization (λ0 vs sum_λ)  
**Current**:

- GPU/Boundary: Use λ0 normalization
- CPU k-NN: Uses sum_λ normalization

**Decision Required**: Which to standardize on?

**Recommendation**: Use **λ0 normalization** (Weinmann et al.)

- More common in literature
- Already used in 2 of 3 implementations
- Ensures `linearity + planarity + sphericity ≤ 1`

**Change**: Update CPU k-NN version (line ~1030) to use λ0 normalization:

```python
# BEFORE
linearity = ((λ0 - λ1) / sum_λ).astype(np.float32)
planarity = ((λ1 - λ2) / sum_λ).astype(np.float32)
sphericity = (λ2 / sum_λ).astype(np.float32)

# AFTER
linearity = np.clip((λ0 - λ1) / λ0_safe, 0.0, 1.0).astype(np.float32)
planarity = np.clip((λ1 - λ2) / λ0_safe, 0.0, 1.0).astype(np.float32)
sphericity = np.clip(λ2 / λ0_safe, 0.0, 1.0).astype(np.float32)
```

**Note**: Roughness keeps sum_λ normalization (different feature)

---

## Phase 3: Testing & Validation (CRITICAL)

### 3.1 Create Comprehensive Test Suite

**File**: `tests/test_feature_validation_comprehensive.py`

**Test Coverage**:

```python
class TestFeatureValidation:
    """Comprehensive feature validation across all modules."""

    def test_eigenvalue_edge_cases(self):
        """Test all modules with problematic eigenvalues."""
        test_cases = [
            "negative_eigenvalues",
            "near_zero_eigenvalues",
            "equal_eigenvalues_sphere",
            "line_configuration",
            "plane_configuration",
            "mixed_normal_and_edge"
        ]

        for module in [boundary, gpu, cpu_radius, cpu_knn]:
            for test_case in test_cases:
                features = module.compute_features(test_case)
                self._validate_ranges(features)

    def test_feature_consistency(self):
        """Ensure all modules compute similar features."""
        points = load_real_lidar_sample()

        f_boundary = boundary.compute_features(points)
        f_gpu = gpu.compute_features(points)
        f_cpu = cpu.compute_features(points)

        # Should have same feature keys
        assert set(f_boundary.keys()) == set(f_gpu.keys())

        # Should have similar values (allow small numerical diff)
        for key in f_boundary.keys():
            np.testing.assert_allclose(
                f_boundary[key],
                f_gpu[key],
                rtol=1e-3,
                err_msg=f"Feature {key} differs between modules"
            )

    def test_feature_ranges(self):
        """Verify all features stay in valid ranges."""
        for module in [boundary, gpu, cpu]:
            features = module.compute_features(test_data)

            # Geometric features: [0, 1]
            for feat in ['linearity', 'planarity', 'sphericity',
                        'anisotropy', 'roughness', 'verticality']:
                assert features[feat].min() >= 0.0, f"{feat} < 0"
                assert features[feat].max() <= 1.0, f"{feat} > 1"
                assert not np.any(np.isnan(features[feat])), f"{feat} has NaN"
                assert not np.any(np.isinf(features[feat])), f"{feat} has Inf"

            # Density: [0, max_cap]
            assert features['density'].min() >= 0.0
            assert features['density'].max() <= 1000.0

            # Curvature: [0, 1]
            assert features['curvature'].min() >= 0.0
            assert features['curvature'].max() <= 1.0

    def test_real_lidar_data(self):
        """Test with actual IGN LiDAR HD data."""
        # Load real tile
        points = load_ign_tile("test_tile.laz")

        # Process through full pipeline
        features = process_with_all_modules(points)

        # Validate
        self._validate_ranges(features)
        self._check_for_artifacts(features)
```

### 3.2 Create Edge Case Test Data

**File**: `tests/fixtures/edge_case_eigenvalues.npz`

Generate test data with known edge cases:

```python
import numpy as np

# Edge case eigenvalue sets
edge_cases = {
    'negative': np.array([
        [1.0, 0.5, -0.001],
        [1.0, -0.1, -0.2],
        [-0.5, -0.8, -1.0]
    ]),

    'near_zero': np.array([
        [1e-12, 1e-13, 1e-14],
        [1e-11, 5e-12, 3e-12],
        [0.0, 0.0, 0.0]
    ]),

    'sphere': np.array([
        [1.0, 0.99, 0.98],
        [0.5, 0.5, 0.5],
        [0.333, 0.333, 0.333]
    ]),

    'line': np.array([
        [10.0, 0.01, 0.001],
        [100.0, 1e-10, 1e-15],
        [5.0, 0.0, 0.0]
    ]),

    'plane': np.array([
        [10.0, 9.0, 0.1],
        [5.0, 4.8, 0.01],
        [1.0, 0.99, 0.001]
    ]),

    'real_lidar': np.array([
        [0.234, 0.123, 0.012],   # Ground
        [0.456, 0.034, 0.002],   # Edge
        [0.089, 0.087, 0.085],   # Vegetation
        [1.234, 0.001, 0.0001]   # Power line
    ])
}

np.savez('edge_case_eigenvalues.npz', **edge_cases)
```

---

## Phase 4: Documentation & Validation

### 4.1 Update Feature Documentation

**Files to update**:

- `docs/docs/features/geometric-features.md`
- `docs/docs/api/features-api.md`
- Docstrings in all feature modules

**Add**:

- Expected range for each feature
- Formula used
- When feature might be invalid (set to 0)
- Difference between λ0 and sum_λ normalization

Example:

```markdown
### Sphericity

**Formula**: `λ₂ / λ₀` (smallest / largest eigenvalue)

**Range**: [0, 1]

- 0 = Linear or planar structure
- 1 = Spherical structure (equal eigenvalues)

**Use Cases**:

- Vegetation detection (high sphericity)
- Noise detection
- 3D structure classification

**Notes**:

- Values are clamped to [0,1] to handle numerical artifacts
- Invalid neighborhoods (< 3 points) produce sphericity = 0
```

### 4.2 Add Validation to Pipeline

**File**: `ign_lidar/core/processor.py`

Add feature validation after computation:

```python
def _validate_computed_features(self, features: Dict[str, np.ndarray]) -> None:
    """
    Validate computed features for common issues.

    Logs warnings if features have unexpected values.
    """
    for feat_name, feat_values in features.items():
        # Check for NaN/Inf
        if np.any(np.isnan(feat_values)):
            num_nan = np.sum(np.isnan(feat_values))
            logger.warning(
                f"Feature '{feat_name}' has {num_nan} NaN values "
                f"({100*num_nan/len(feat_values):.2f}%)"
            )

        if np.any(np.isinf(feat_values)):
            num_inf = np.sum(np.isinf(feat_values))
            logger.warning(
                f"Feature '{feat_name}' has {num_inf} Inf values"
            )

        # Check expected ranges
        expected_ranges = {
            'linearity': (0, 1),
            'planarity': (0, 1),
            'sphericity': (0, 1),
            'anisotropy': (0, 1),
            'roughness': (0, 1),
            'verticality': (0, 1),
            'horizontality': (0, 1),
            'curvature': (0, 1),
            'density': (0, 1000)
        }

        if feat_name in expected_ranges:
            min_val, max_val = expected_ranges[feat_name]
            actual_min = feat_values.min()
            actual_max = feat_values.max()

            if actual_min < min_val or actual_max > max_val:
                logger.warning(
                    f"Feature '{feat_name}' out of expected range "
                    f"[{min_val}, {max_val}]: "
                    f"actual=[{actual_min:.4f}, {actual_max:.4f}]"
                )
```

---

## Phase 5: Rollout & Verification

### 5.1 Implementation Order

**Priority Order**:

1. ✅ Phase 1.1: Fix CPU radius features (CRITICAL)
2. ✅ Phase 1.2: Fix GPU features (HIGH)
3. ✅ Phase 1.3: Fix CPU k-NN features (HIGH)
4. ✅ Phase 1.4: Normalize density (HIGH)
5. ✅ Phase 2.1: Add missing boundary features (HIGH)
6. ✅ Phase 2.2: Standardize formulas (MEDIUM)
7. ✅ Phase 3: Testing (CRITICAL - before merge)
8. ✅ Phase 4: Documentation (MEDIUM)

### 5.2 Testing Checklist

Before merging:

- [ ] All unit tests pass
- [ ] Edge case tests pass
- [ ] Real LiDAR data tests pass
- [ ] Feature consistency tests pass
- [ ] No regression in existing features
- [ ] Documentation updated
- [ ] Performance benchmarks (should be ~same)

### 5.3 Deployment Strategy

**Step 1: Branch & Test**

```bash
git checkout -b feature/robust-feature-validation
# Implement all fixes
python -m pytest tests/test_feature_validation_comprehensive.py -v
```

**Step 2: Integration Test**

```bash
# Run full pipeline on test data
ign-lidar-hd process --config-file tests/config_test.yaml

# Verify output
ign-lidar-hd verify --input output/test_tile.npz
```

**Step 3: Real Data Validation**

```bash
# Process actual IGN tile
ign-lidar-hd process --config-file config_enriched_only.yaml

# Should see NO warnings about out-of-range features
```

**Step 4: Merge**

```bash
git add .
git commit -m "feat: Add robust validation to all geometric features

- Add eigenvalue clamping to prevent negative values
- Add result clipping to ensure [0,1] ranges
- Add missing features to boundary computation
- Standardize formulas across all modules
- Cap density feature at 1000
- Comprehensive test suite for edge cases

Fixes #XXX"
git push origin feature/robust-feature-validation
```

---

## Risk Assessment

| Risk                    | Probability | Impact | Mitigation                                |
| ----------------------- | ----------- | ------ | ----------------------------------------- |
| Breaking existing code  | LOW         | HIGH   | Comprehensive tests + backward compatible |
| Performance degradation | LOW         | MEDIUM | Clamping is O(N), minimal overhead        |
| Changed feature values  | MEDIUM      | MEDIUM | Document changes, provide migration guide |
| Test suite gaps         | MEDIUM      | HIGH   | Code review + real data validation        |
| Integration issues      | LOW         | MEDIUM | Staged rollout + rollback plan            |

---

## Success Criteria

1. ✅ **No more out-of-range warnings** in production pipeline
2. ✅ **All tests pass** (unit, integration, edge cases)
3. ✅ **Feature consistency** across all computation paths
4. ✅ **Performance maintained** (<5% overhead)
5. ✅ **Documentation complete** and accurate
6. ✅ **Zero regressions** in existing features

---

## Timeline

| Phase                    | Duration      | Dependencies        |
| ------------------------ | ------------- | ------------------- |
| Phase 1 (Critical Fixes) | 2 hours       | None                |
| Phase 2 (Consistency)    | 1-2 hours     | Phase 1 complete    |
| Phase 3 (Testing)        | 1-2 hours     | Phase 1-2 complete  |
| Phase 4 (Documentation)  | 1 hour        | Phase 1-3 complete  |
| Phase 5 (Rollout)        | 30 min        | All phases complete |
| **Total**                | **4-6 hours** |                     |

---

## Follow-Up Items

After initial fixes:

1. **Centralize validation logic** (refactor)

   - Create `FeatureValidator` class
   - Single source of truth for ranges
   - Reduces code duplication

2. **Add feature quality metrics**

   - Log feature statistics per tile
   - Track percentage of valid features
   - Detect anomalies early

3. **Performance optimization**

   - Profile clamping overhead
   - Consider vectorization improvements
   - GPU-accelerated validation

4. **ML model impact analysis**
   - Retrain models with fixed features
   - Compare performance metrics
   - Update benchmarks

---

## Approval Required

**Before Implementation**:

- [ ] Technical lead approval
- [ ] Testing strategy review
- [ ] Resource allocation

**Before Merge**:

- [ ] Code review passed
- [ ] All tests passing
- [ ] Documentation reviewed
- [ ] Performance validated

---

## Notes

- All changes are **backward compatible**
- Existing NPZ files remain valid
- Feature values may change slightly (within ε)
- No breaking API changes
- Safe to deploy to production

---

## Questions & Decisions

**Q1**: Should we change sum_λ normalization to λ0?  
**A**: YES - for consistency with boundary features

**Q2**: How to handle density normalization?  
**A**: Cap at 1000 (Option A) - simpler and backward compatible

**Q3**: Add new features to existing NPZ files?  
**A**: NO - only affect new processing

**Q4**: Version bump required?  
**A**: YES - Minor version bump (v2.X.0 → v2.Y.0) for new features

---

**END OF PLAN**
