# Visual Explanation: Augmentation Correspondence Issue

## The Problem: Tile-Level Augmentation

```
CURRENT APPROACH (WRONG):

Step 1: Load Tile
┌─────────────────┐
│  A   B   C   D  │  Original Tile (100x100m)
│  E   F   G   H  │  Letters = different spatial regions
│  I   J   K   L  │
│  M   N   O   P  │
└─────────────────┘

Step 2: Apply Augmentation (Rotation + Dropout)
       ┌─────────────────┐
       │  L   H   D   P  │  Rotated Tile
       │  K   G   C   O  │  (90° clockwise)
       │  J   F   B   N  │
       │  I   E   A   M  │
       └─────────────────┘

Step 3: Extract Patches (Same Grid Positions)
┌───┬───┬───┬───┐         ┌───┬───┬───┬───┐
│ A │ B │ C │ D │         │ L │ H │ D │ P │
├───┼───┼───┼───┤         ├───┼───┼───┼───┤
│ E │ F │ G │ H │   →     │ K │ G │ C │ O │
├───┼───┼───┼───┤         ├───┼───┼───┼───┤
│ I │ J │ K │ L │         │ J │ F │ B │ N │
├───┼───┼───┼───┤         ├───┼───┼───┼───┤
│ M │ N │ O │ P │         │ I │ E │ A │ M │
└───┴───┴───┴───┘         └───┴───┴───┴───┘
Original Patches          Augmented Patches

RESULT: ❌ Patch 0,0 contains region A → region L (DIFFERENT!)
        ❌ Patch 1,1 contains region F → region G (DIFFERENT!)
        ❌ No correspondence!
```

## The Solution: Patch-Level Augmentation

```
NEW APPROACH (CORRECT):

Step 1: Load Tile
┌─────────────────┐
│  A   B   C   D  │  Original Tile
│  E   F   G   H  │
│  I   J   K   L  │
│  M   N   O   P  │
└─────────────────┘

Step 2: Extract Patches FIRST
┌───┬───┬───┬───┐
│ A │ B │ C │ D │  Patch 0,0 = Region A
├───┼───┼───┼───┤  Patch 1,1 = Region F
│ E │ F │ G │ H │  Patch 2,3 = Region L
├───┼───┼───┼───┤  etc.
│ I │ J │ K │ L │
├───┼───┼───┼───┤
│ M │ N │ O │ P │
└───┴───┴───┴───┘

Step 3: Apply Augmentation TO EACH PATCH

Patch 0,0 (Region A):
┌───┐  augment_patch()  ┌───┐
│ A │  ────────────→     │ A'│  Same region, rotated/jittered
└───┘                    └───┘

Patch 1,1 (Region F):
┌───┐  augment_patch()  ┌───┐
│ F │  ────────────→     │ F'│  Same region, rotated/jittered
└───┘                    └───┘

RESULT: ✓ Patch 0,0 original = Region A
        ✓ Patch 0,0 augmented = Region A' (SAME region, transformed)
        ✓ Perfect correspondence!
```

## Key Difference

### Tile-Level Augmentation (Wrong)

- **Augment** → **Extract** → Different spatial regions
- Patch_0000 original ≠ Patch_0000 augmented (spatially)
- Classification distribution completely different
- Model learns on unrelated data

### Patch-Level Augmentation (Correct)

- **Extract** → **Augment** → Same spatial regions
- Patch_0000 original = Patch_0000 augmented (spatially)
- Classification distribution similar (±dropout)
- Model learns same objects under transformation

## Evidence from Real Data

```
urban_dense/LHD_FXX_0649_6863...patch_0000:

Tile-Level (Current):
┌────────────────────┐     ┌────────────────────┐
│  Original Patch    │     │  Augmented Patch   │
│  X: [-0.70, 0.71]  │     │  X: [-0.18, 0.10]  │ ← Different!
│  Y: [-0.70, 0.70]  │     │  Y: [-0.98, 0.63]  │ ← Different!
│                    │     │                    │
│  Class 0: 15,369   │     │  Class 0: 2,688    │ ← Very different!
│  Class 23: 10,208  │     │  Class 23: 23,692  │ ← Very different!
└────────────────────┘     └────────────────────┘
      ❌ Only 19.9% spatial overlap

Patch-Level (Fixed):
┌────────────────────┐     ┌────────────────────┐
│  Original Patch    │     │  Augmented Patch   │
│  X: [-9.97, 10.00] │     │  X: [-9.64, 9.84]  │ ← Similar!
│  Y: [-9.86, 9.97]  │     │  Y: [-9.72, 9.72]  │ ← Similar!
│                    │     │                    │
│  Class 0: 122      │     │  Class 0: 115      │ ← Similar!
│  Class 1: 109      │     │  Class 1: 96       │ ← Similar!
└────────────────────┘     └────────────────────┘
      ✓ 0.8% label distribution difference
```

## Implementation Change

### In processor.py

```python
# ❌ WRONG (Current):
for version_idx in range(num_versions):
    if version_idx > 0:
        # Rotate entire tile
        points = augment_raw_points(points, ...)

    # Extract patches from rotated tile
    patches = extract_patches(points, ...)
    # → Patches from different locations!


# ✓ CORRECT (Fixed):
# Extract patches from original tile ONCE
patches = extract_patches(points, ...)

# For each patch, create augmented versions
for patch_idx, patch in enumerate(patches):
    # Save original
    save_patch(patch, version="original", idx=patch_idx)

    # Create augmented versions of THIS patch
    for aug_idx in range(num_augmentations):
        aug_patch = augment_patch(patch)
        save_patch(aug_patch, version=f"aug_{aug_idx}", idx=patch_idx)
    # → All versions of patch_idx correspond!
```

## Benefits

1. **Spatial Correspondence**: Augmented patches represent the same region
2. **Label Consistency**: Classification distribution remains similar
3. **Training Quality**: Model learns to recognize same objects under transformation
4. **Debugging**: Can visually compare original vs augmented
5. **Data Integrity**: Spatial context preserved

## Files to Review

- `IMPLEMENTATION_PLAN_PATCH_AUGMENTATION.md` - Step-by-step implementation
- `FIX_SUMMARY.md` - Complete overview
- `scripts/verify_augmentation_correspondence.py` - Tool to verify your data
