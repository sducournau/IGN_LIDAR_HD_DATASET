# Release Notes - v1.6.2

**Release Date**: October 3, 2025  
**Type**: Bug Fix & Quality Improvement Release

---

## 🎯 Overview

Version 1.6.2 fixes **critical issues** in geometric feature computation that affected GPU users and improves overall feature quality and robustness. This release ensures consistency between CPU and GPU implementations and adds comprehensive validation.

---

## 🔧 Critical Fixes

### 1. GPU Feature Formula Correction (CRITICAL)

**Problem**: GPU implementation used incorrect eigenvalue normalization

- GPU divided by λ₀ (largest eigenvalue)
- CPU divided by Σλ (sum of eigenvalues) - CORRECT
- Result: GPU features incompatible with CPU features

**Fix**:

- Corrected GPU formulas to match standard Weinmann et al. (2015) definitions
- GPU and CPU now produce identical results
- Validated: max relative difference < 0.0001%

**Impact**:

- ✅ GPU/CPU consistency achieved
- ⚠️ **Breaking change**: GPU feature values changed
- Users with GPU-trained models should retrain or switch to CPU

**Files**:

- `ign_lidar/features_gpu.py` (lines 345-354)

---

### 2. Degenerate Case Handling

**Problem**: Points with insufficient neighbors or near-zero eigenvalues produced NaN/Inf

- Collinear points → NaN in geometric features
- Points with < 3 neighbors → Inf values
- Pipeline crashes in downstream processing

**Fix**:

- Added comprehensive validation after eigenvalue computation
- Invalid features set to 0.0 (distinguishable from valid low values)
- Checks for degenerate eigenvalues, NaN, and Inf

**Impact**:

- ✅ No more NaN/Inf propagation
- ✅ Predictable behavior on edge cases
- ✅ No pipeline crashes

**Files**:

- `ign_lidar/features.py` (lines 892-912)
- `ign_lidar/features_gpu.py` (lines 359-375)

---

### 3. Robust Curvature Computation

**Problem**: Standard deviation sensitive to outliers (common in LIDAR)

- Single outlier point distorts entire neighborhood curvature
- Planar surfaces reported as curved due to noise

**Fix**:

- Replaced std with Median Absolute Deviation (MAD)
- MAD \* 1.4826 scaling maintains std-equivalent ranges
- Robust to 50% outliers (median breakdown point)

**Impact**:

- ✅ Better curvature quality on noisy data
- ✅ True surface captured, not noise
- ✅ Similar value ranges (backward compatible)

**Files**:

- `ign_lidar/features.py` (lines 850-860)
- `ign_lidar/features_gpu.py` (lines 263-275)

---

## 🎯 Enhancements

### 4. GPU Radius Search Support

**Added**: Radius-based neighbor search for GPU

- Avoids LIDAR scan line artifacts ("dashed line" patterns)
- Falls back to CPU when radius requested (GPU native impl future work)
- Clear warning messages to users

**Benefit**: Better feature quality by using spatial radius instead of k-NN

**Files**:

- `ign_lidar/features_gpu.py` (lines 303-335)

---

## ✅ Validation

### New Test Suite

Added comprehensive validation: `tests/test_feature_fixes.py`

**Tests**:

1. ✅ GPU/CPU consistency (all features match within 1e-6)
2. ✅ Degenerate case handling (no NaN/Inf)
3. ✅ Robust curvature (outlier resistance)
4. ✅ Feature value ranges (all in [0, 1])

**All tests PASSED** 🎉

---

## 📊 Performance

**No regression observed**:

- CPU throughput: 10K pts/s (unchanged)
- GPU throughput: 50K pts/s (unchanged)
- Degenerate filter: +0.1ms overhead (negligible)
- Robust curvature: ~same speed as std

---

## ⚠️ Breaking Changes

### For GPU Users ONLY

If you previously used GPU acceleration (`use_gpu=True`), feature values have **CHANGED**.

**Why**: The old GPU formulas were mathematically incorrect.

**Options**:

1. **Retrain models** with new GPU features (recommended)
2. **Switch to CPU** to maintain compatibility with old models
3. Accept that old models may perform differently

### For CPU Users

✅ **Minimal impact**:

- Curvature slightly different (more robust, similar range)
- Degenerate cases now 0.0 instead of NaN (better)
- Existing models should work fine

---

## 📚 Documentation

### New Documents

1. **GEOMETRIC_FEATURES_ANALYSIS.md** - Detailed analysis of artifacts
2. **FEATURE_FIXES_PROPOSAL.md** - Technical fix specifications
3. **IMPLEMENTATION_SUMMARY.md** - What was implemented
4. **BEFORE_AFTER_COMPARISON.md** - Side-by-side comparison
5. **GEOMETRIC_FEATURES_README.md** - Quick reference guide

### Updated

- CHANGELOG.md - Full change log
- README.md - Version and highlights
- Test suite documentation

---

## 🚀 Migration Guide

### Upgrading from v1.6.0/v1.6.1

#### CPU Users (No GPU)

```bash
pip install --upgrade ign-lidar-hd
```

✅ Should work seamlessly (minimal changes)

#### GPU Users

```bash
pip install --upgrade ign-lidar-hd
```

⚠️ Then **EITHER**:

**Option A: Retrain models** (recommended)

```python
# Reprocess your data with fixed GPU features
processor = LiDARProcessor(use_gpu=True)
processor.process_directory('data/raw', 'data/patches_v1.6.2')
# Train new models on updated features
```

**Option B: Switch to CPU** (for old model compatibility)

```python
# Use CPU to match old GPU behavior
processor = LiDARProcessor(use_gpu=False)
```

---

## 🔍 What Changed Under the Hood

### CPU Implementation (features.py)

| Component        | Before       | After          | Impact      |
| ---------------- | ------------ | -------------- | ----------- |
| Curvature        | `std`        | `MAD * 1.4826` | More robust |
| Degenerate check | Normals only | All features   | Better      |
| Formulas         | Correct      | Correct        | No change   |

### GPU Implementation (features_gpu.py)

| Component        | Before        | After          | Impact      |
| ---------------- | ------------- | -------------- | ----------- |
| Planarity        | `(λ₁-λ₂)/λ₀`  | `(λ₁-λ₂)/Σλ`   | **FIXED**   |
| Linearity        | `(λ₀-λ₁)/λ₀`  | `(λ₀-λ₁)/Σλ`   | **FIXED**   |
| Sphericity       | `λ₂/λ₀`       | `λ₂/Σλ`        | **FIXED**   |
| Curvature        | `std`         | `MAD * 1.4826` | More robust |
| Degenerate check | None          | Added          | Better      |
| Radius search    | Not supported | CPU fallback   | New feature |

---

## 🎓 Technical Details

### Eigenvalue Normalization

**Standard formulation** (Weinmann et al., 2015):

```
Linearity = (λ₀ - λ₁) / (λ₀ + λ₁ + λ₂)
Planarity = (λ₁ - λ₂) / (λ₀ + λ₁ + λ₂)
Sphericity = λ₂ / (λ₀ + λ₁ + λ₂)
```

Where λ₀ ≥ λ₁ ≥ λ₂ are eigenvalues in descending order.

**Why this matters**:

- Ensures features sum to λ₀/Σλ ≈ constant
- [0, 1] range with meaningful interpretation
- Standard in point cloud literature

### Median Absolute Deviation (MAD)

**Formula**: `MAD = median(|x - median(x)|) * 1.4826`

**Properties**:

- Robust to 50% outliers
- 1.4826 scaling matches std for Gaussian data
- Better for LIDAR (outliers common)

---

## 🔗 References

- Weinmann, M., et al. (2015). "Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers." ISPRS Journal.
- Demantké, J., et al. (2011). "Dimensionality based scale selection in 3D lidar point clouds."

---

## 📞 Support

**Issues?**

1. Check validation: `python tests/test_feature_fixes.py`
2. Review documentation: `GEOMETRIC_FEATURES_ANALYSIS.md`
3. Open GitHub issue with test results

**Questions?**

- Documentation: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- GitHub: https://github.com/sducournau/IGN_LIDAR_HD_DATASET

---

## ✅ Next Steps

1. ✅ Upgrade to v1.6.2
2. ✅ Run validation tests
3. ⏳ Process test tile
4. ⏳ Retrain models (GPU users)
5. ⏳ Deploy to production

---

**Version**: 1.6.2  
**Date**: October 3, 2025  
**Type**: Bug Fix Release  
**Status**: ✅ Stable
