# Fix Summary: LAZ Tile Augmentation Correspondence Issue

**Date**: October 10, 2025  
**Issue**: LAZ tiles in output don't correspond across their different augmentation versions  
**Status**: ✓ Root cause identified, ✓ Fix implemented and tested, ⚠️ Processor refactor needed

---

## Problem Identified

The augmentation correspondence issue in `C:\Users\Simon\ign\patch_1st_training\urban_dense` was caused by:

**Root Cause**: Augmentation is applied to the full tile BEFORE patch extraction, causing patches to be extracted from DIFFERENT spatial regions after rotation.

### Evidence

```
Patch 0 Comparison:
  Original:  X=[-0.6989, 0.7111], Y=[-0.7011, 0.7049]
  Augmented: X=[-0.1844, 0.0956], Y=[-0.9815, 0.6315]

  X Overlap: 19.9% ❌
  Y Overlap: 94.8%

  Classification Distribution:
    Class  0: 15,369 → 2,688  (DIFFERENT OBJECTS!)
    Class 23: 10,208 → 23,692 (DIFFERENT OBJECTS!)
```

---

## Solution Implemented

### 1. Created New Function: `augment_patch()`

**Location**: `ign_lidar/preprocessing/utils.py`

```python
def augment_patch(patch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Apply augmentation to a single patch while maintaining correspondence.

    Ensures augmented versions correspond to the same spatial region.
    """
```

**Features**:

- Applies rotation, jitter, scaling, dropout to individual patches
- Maintains spatial correspondence
- Properly handles all feature types (normals, RGB, NIR, geometric features)
- Tested and verified ✓

### 2. Created Verification Tools

**`scripts/verify_augmentation_correspondence.py`**:

- Checks spatial overlap between original and augmented patches
- Analyzes classification distribution similarity
- Identifies correspondence issues

**`tests/test_patch_augmentation_correspondence.py`**:

- Unit test for new `augment_patch()` function
- Confirms label distributions remain similar (0.8% difference)
- All tests pass ✓

### 3. Documentation

Created comprehensive documentation:

- `AUGMENTATION_CORRESPONDENCE_FIX.md` - Problem analysis and evidence
- `IMPLEMENTATION_PLAN_PATCH_AUGMENTATION.md` - Step-by-step implementation guide

---

## What Works Now

✅ **Patch-level augmentation function**: Fully implemented and tested  
✅ **Verification tools**: Can detect correspondence issues  
✅ **Test suite**: Confirms new approach maintains correspondence  
✅ **Documentation**: Complete problem analysis and solution

## What Still Needs to Be Done

### Critical: Refactor Processor

The main change needed is in `ign_lidar/core/processor.py`:

**Change this** (lines ~1450-2070):

```python
# WRONG: Augment full tile, then extract patches
for version_idx in range(num_versions):
    if version_idx > 0:
        points = augment_raw_points(points, ...)  # Changes spatial distribution
    patches = extract_patches(points, ...)        # Wrong locations!
```

**To this**:

```python
# CORRECT: Extract patches, then augment each patch
patches = extract_patches(points, ...)
for patch_idx, patch in enumerate(patches):
    save_patch(patch, version="original", spatial_idx=patch_idx)
    for aug_idx in range(num_augmentations):
        aug_patch = augment_patch(patch)
        save_patch(aug_patch, version=f"aug_{aug_idx}", spatial_idx=patch_idx)
```

### Estimated Time

- Processor refactor: 2-3 hours
- Configuration updates: 30 min
- CLI updates: 30 min
- Documentation: 1 hour
- Testing: 1-2 hours

**Total**: 5-7 hours

---

## Migration Path

Users with existing data generated using the old approach will need to:

1. **Backup existing data**:

   ```bash
   mv output/patches output/patches_old
   ```

2. **Reprocess with fixed code**:

   ```bash
   ign-lidar process \
     --input data/raw_tiles \
     --output output/patches \
     --augmentation-mode patch
   ```

3. **Verify correspondence**:
   ```bash
   python scripts/verify_augmentation_correspondence.py
   # Should show: ✓ All patches correspond correctly
   ```

---

## Files Modified

### New Files

- `ign_lidar/preprocessing/utils.py` - Added `augment_patch()` function
- `scripts/verify_augmentation_correspondence.py` - Verification tool
- `tests/test_patch_augmentation_correspondence.py` - Unit test
- `AUGMENTATION_CORRESPONDENCE_FIX.md` - Problem documentation
- `IMPLEMENTATION_PLAN_PATCH_AUGMENTATION.md` - Implementation guide

### Modified Files

- `ign_lidar/preprocessing/__init__.py` - Export `augment_patch`

### To Be Modified

- `ign_lidar/core/processor.py` - Refactor augmentation loop (TODO)
- `ign_lidar/config/config.yaml` - Add augmentation config (TODO)
- `docs/docs/guides/basic-usage.md` - Update documentation (TODO)

---

## Testing Results

### Patch-Level Augmentation Test

```
✓ Point counts correct (dropout: 8.6%)
✓ Points are different (augmentation applied)
✓ Label distribution similar (0.8% difference)
✓ All features present
✓ TEST PASSED
```

### Original Issue Verification

```
❌ X overlap: 19.9% (expected >90%)
❌ Different classification distribution
❌ Patches don't correspond
```

---

## Recommendations

1. **Immediate**: Complete processor refactor (Phase 1c in implementation plan)
2. **Short-term**: Reprocess training data with fixed code
3. **Long-term**: Consider making patch-level augmentation the default

## Contact

For questions about this fix, see:

- `IMPLEMENTATION_PLAN_PATCH_AUGMENTATION.md` for detailed steps
- `AUGMENTATION_CORRESPONDENCE_FIX.md` for technical details
- `scripts/verify_augmentation_correspondence.py` to test your data
