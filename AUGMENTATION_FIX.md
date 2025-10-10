# Augmentation Fix: Same Patches at Different Augmentation Versions

## Problem

When augmentation was enabled, the **same patch indices** represented **different spatial areas** across augmentation versions. For example:

- `urban_dense_patch_0000.npz` (original)
- `urban_dense_patch_0000_aug_0.npz` (augmented v1)
- `urban_dense_patch_0000_aug_1.npz` (augmented v2)

These patches had the same filename but contained **completely different geographical regions**.

## Root Cause

The original pipeline worked as follows:

1. Load raw LiDAR data
2. **FOR EACH augmentation version:**
   - Apply augmentation (rotation, jitter, scaling, dropout) to the **entire tile**
   - Extract patches using spatial grid
   - Save patches

The problem: When you **rotate/jitter/scale the entire point cloud**, the spatial grid changes, causing patches to be extracted from **different locations**.

### Example:

```
Original tile (0-360 degrees orientation):
┌──────────────────┐
│ A  │ B  │ C  │ D │  <- Patches 0, 1, 2, 3
└──────────────────┘

After 45° rotation + jitter:
┌──────────────────┐
│ E  │ F  │ G  │ H │  <- Different spatial areas!
└──────────────────┘
```

## Solution

The fix changes the pipeline to:

1. Load raw LiDAR data
2. **Extract patches ONCE** from original data (defines spatial locations)
3. **FOR EACH patch:**
   - Version 0: Original patch
   - Version 1-N: Apply augmentation to THIS SPECIFIC PATCH

### Key Changes

#### 1. Extract Patches Once (`processor.py`)

```python
# OLD: Extract patches PER augmentation version
for version_idx in range(num_versions):
    augmented_data = augment_raw_points(original_data)  # Tile-level!
    patches_v = extract_patches(augmented_data)  # Different regions!

# NEW: Extract patches ONCE, then augment each patch
base_patches = extract_patches(original_data)  # Spatial regions defined

for patch in base_patches:
    # Original version
    all_patches.append(patch)

    # Augmented versions of SAME spatial region
    for aug_idx in range(num_augmentations):
        augmented_patch = augment_raw_points(patch)  # Patch-level!
        all_patches.append(augmented_patch)
```

#### 2. Enhanced Augmentation Function (`preprocessing/utils.py`)

```python
def augment_raw_points(..., return_mask: bool = False):
    """
    Args:
        return_mask: If True, return dropout mask to align labels

    Returns:
        If return_mask=False: (points, intensity, return_number,
                               classification, rgb, nir, ndvi)
        If return_mask=True:  (...same..., keep_mask)
    """
```

The `return_mask` parameter ensures that labels stay aligned with points after random dropout.

#### 3. Patch Metadata Tracking

```python
# Each patch now has metadata to track versions
patch['_version'] = 'original'  # or 'aug_0', 'aug_1', ...
patch['_patch_idx'] = patch_idx  # Base patch index

# Filenames properly reflect relationship:
# urban_dense_patch_0000.npz        <- Original
# urban_dense_patch_0000_aug_0.npz  <- Augmented v1 (SAME REGION)
# urban_dense_patch_0000_aug_1.npz  <- Augmented v2 (SAME REGION)
```

## Benefits

✅ **Spatial Consistency**: All versions of patch_0000 represent the **same geographical area**

✅ **Training Quality**: Models learn augmentation invariance correctly

✅ **Reproducibility**: Patch indices have consistent meaning across versions

✅ **Validation**: Can compare original vs augmented patches of same region

✅ **Debugging**: Easy to visualize "before/after" augmentation effects

## Verification

To verify the fix works:

```python
import numpy as np

# Load original and augmented versions
original = np.load('urban_dense_patch_0000.npz')
aug_0 = np.load('urban_dense_patch_0000_aug_0.npz')
aug_1 = np.load('urban_dense_patch_0000_aug_1.npz')

# Check point counts (should be similar, within dropout range)
print(f"Original: {len(original['points'])} points")
print(f"Aug 0: {len(aug_0['points'])} points")
print(f"Aug 1: {len(aug_1['points'])} points")

# Check label distributions (should be similar)
from collections import Counter
print(f"Original classes: {Counter(original['labels'])}")
print(f"Aug 0 classes: {Counter(aug_0['labels'])}")
print(f"Aug 1 classes: {Counter(aug_1['labels'])}")

# Check spatial extent (should overlap significantly after unrotating)
print(f"Original bbox: {original['points'].min(0), original['points'].max(0)}")
print(f"Aug 0 bbox: {aug_0['points'].min(0), aug_0['points'].max(0)}")
```

## Migration

Existing datasets created with the **old augmentation pipeline** should be **regenerated** to ensure spatial consistency. The old data had patches with misaligned augmentation versions.

## Technical Details

### Augmentation Transformations (per patch)

1. **Rotation**: Random Z-axis rotation (0-360°)
2. **Jitter**: Gaussian noise σ=0.1m
3. **Scaling**: Uniform (0.95-1.05)
4. **Dropout**: Random (5-15% points removed)

All transformations now applied to individual patches, not the entire tile.

### Label Alignment

The `return_mask` parameter ensures dropout mask is returned so labels can be filtered consistently:

```python
(aug_points, ..., _, keep_mask) = augment_raw_points(..., return_mask=True)
aug_labels = original_labels[keep_mask]  # Keep labels aligned!
```

## Related Files Modified

- `/ign_lidar/core/processor.py`: Changed patch extraction pipeline
- `/ign_lidar/preprocessing/utils.py`: Added `return_mask` parameter
- Documentation: This file

## Date

Fixed: October 10, 2025
