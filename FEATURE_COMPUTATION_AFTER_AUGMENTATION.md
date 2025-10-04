# How Features Are Computed After Augmentation

## Overview

The augmentation implementation ensures that **geometric features are computed on the augmented geometry**, not on the original geometry. This is the key innovation of v1.6.0 and ensures feature-geometry consistency.

## Processing Flow

### Step-by-Step Process

```
1. Load original LAZ file
   ↓
2. Apply preprocessing (if enabled)
   ↓
3. Create augmented versions (augment_raw_points)
   ↓
4. FOR EACH VERSION (original + augmented):
   ↓
   4a. Compute features on THIS version's geometry
   ↓
   4b. Save to separate LAZ file
```

## Detailed Implementation

### 1. Data Augmentation (Before Features)

**Function:** `augment_raw_points()` in `ign_lidar/utils.py`

**Input:** Raw point coordinates + attributes

```python
points: [N, 3]           # X, Y, Z coordinates
intensity: [N]           # Intensity values
return_number: [N]       # Return numbers
classification: [N]      # Classification codes
```

**Transformations Applied (in order):**

1. **Rotation** (Z-axis, 0-360°)

   ```python
   angle = random(0, 2π)
   rotation_matrix = [[cos θ, -sin θ, 0],
                      [sin θ,  cos θ, 0],
                      [0,      0,     1]]
   points_aug = points @ rotation_matrix.T
   ```

2. **Jitter** (Gaussian noise, σ=0.1m)

   ```python
   jitter = normal(μ=0, σ=0.1, shape=[N, 3])
   points_aug += jitter
   ```

3. **Scaling** (0.95-1.05)

   ```python
   scale = random(0.95, 1.05)
   points_aug *= scale
   ```

4. **Dropout** (5-15% of points)
   ```python
   dropout_ratio = random(0.05, 0.15)
   keep_mask = random([N]) > dropout_ratio
   points_aug = points_aug[keep_mask]
   # Also apply mask to intensity, return_number, classification
   ```

**Output:** Augmented geometry

```python
points_augmented: [M, 3]  # M < N due to dropout
intensity_augmented: [M]
return_number_augmented: [M]
classification_augmented: [M]
```

### 2. Feature Computation (On Augmented Geometry)

**Function:** `compute_all_features_with_gpu()` or `compute_all_features_optimized()`

**For EACH version** (original + augmented), features are computed from scratch:

```python
# Version loop in _enrich_single_file()
for version_data in versions_to_process:
    points_ver = version_data['points']  # Augmented or original
    classification_ver = version_data['classification']

    # Compute features ON THIS VERSION'S GEOMETRY
    normals, curvature, height_above_ground, geometric_features = \
        compute_all_features_with_gpu(
            points_ver,           # Uses augmented points!
            classification_ver,
            k=k_neighbors,
            use_gpu=use_gpu,
            radius=radius
        )
```

### 3. Features Computed

**Core Features (always computed):**

1. **Surface Normals** `[N, 3]`

   - Computed from augmented point neighborhoods
   - Uses k-NN or radius search on augmented coordinates
   - PCA on local neighborhoods

2. **Curvature** `[N]`

   - Principal curvature from eigenvalues
   - Computed on augmented geometry

3. **Height Above Ground** `[N]`

   - Ground classification from augmented points
   - Height relative to augmented ground surface

4. **Geometric Features** (dict):
   - **Planarity** - How flat the local surface is
   - **Sphericity** - How sphere-like the local geometry is
   - **Linearity** - How linear the local structure is
   - **Omnivariance** - Geometric spread measure
   - **Anisotropy** - Directional variation
   - **Eigenentropy** - Entropy of eigenvalues
   - **Change of curvature** - Local curvature variation

**Building Features (mode='full'):**

5. **Verticality** `[N]`

   - Computed from augmented normals
   - How vertical the surface is

6. **Wall Score** `[N]`

   - Based on augmented normals and height

7. **Roof Score** `[N]`

   - Based on augmented normals, height, and curvature

8. **Local Density** `[N]`
   - Number of points within radius (on augmented geometry)

### 4. File Output

Each version is saved to a separate LAZ file:

```
output/
├── tile_name.laz          # Original geometry + features computed on original
├── tile_name_aug1.laz     # Augmented geometry 1 + features computed on aug1
├── tile_name_aug2.laz     # Augmented geometry 2 + features computed on aug2
└── tile_name_aug3.laz     # Augmented geometry 3 + features computed on aug3
```

## Why This Matters

### ✅ Correct Approach (v1.6.0+)

```
Original Points → Augment Geometry → Compute Features → Save
                  (rotate, jitter)   (on augmented)

Result: Features match the geometry ✓
- Normals point in direction of augmented surfaces
- Curvature reflects augmented surface shape
- Planarity describes augmented local neighborhoods
```

### ❌ Wrong Approach (pre-v1.6.0)

```
Original Points → Compute Features → Augment Geometry → Save
                  (on original)      (rotate, jitter)

Result: Features DON'T match the geometry ✗
- Normals point in wrong directions after rotation
- Curvature describes wrong surface
- Planarity doesn't match actual augmented neighborhoods
```

## Example

### Original Point Cloud

```
Points: [(0, 0, 0), (1, 0, 0), (0, 1, 0)]  # L-shaped
Normal at (0,0,0): pointing up (0, 0, 1)
```

### After Augmentation (90° rotation)

```
Points: [(0, 0, 0), (0, 1, 0), (-1, 0, 0)]  # Still L-shaped, rotated
```

**If features computed BEFORE rotation (WRONG):**

```
Normal: (0, 0, 1)  # Still points up
But geometry is rotated! ❌ MISMATCH
```

**If features computed AFTER rotation (CORRECT):**

```
Normal: Recomputed on rotated geometry
New normal: (0, 0, 1)  # Points up relative to NEW geometry ✓ CORRECT
```

## Code Path Summary

```python
# In _enrich_single_file() worker function

# 1. Load and preprocess original data
points, classification = load_laz(laz_path)
if preprocess:
    points = apply_preprocessing(points)

# 2. Create versions (original + augmented)
versions = []
versions.append({'points': points, 'suffix': ''})  # Original

for i in range(num_augmentations):
    # Apply augmentation transformations
    aug_points = augment_raw_points(points, ...)
    versions.append({'points': aug_points, 'suffix': f'_aug{i+1}'})

# 3. Process EACH version independently
for version in versions:
    points_ver = version['points']

    # Compute features ON THIS VERSION'S GEOMETRY
    normals, curvature, ... = compute_features(points_ver)

    # Save with version-specific filename
    save_laz(
        points=points_ver,        # Augmented geometry
        normals=normals,          # Features from augmented geometry
        curvature=curvature,      # Features from augmented geometry
        filename=f"tile{version['suffix']}.laz"
    )
```

## Performance Impact

**Per tile with 3 augmentations:**

- **Storage:** 4× (1 original + 3 augmented files)
- **Processing time:** ~4× (features computed 4 times)
- **Memory:** Same (processes one version at a time)

**Benefit:**

- ✅ 4× more training data with diverse geometry
- ✅ Features correctly describe the geometry
- ✅ Better model generalization and accuracy

## Key Functions

| Function                          | Purpose                         | Input                   | Output                              |
| --------------------------------- | ------------------------------- | ----------------------- | ----------------------------------- |
| `augment_raw_points()`            | Apply geometric transformations | Raw points + attributes | Augmented points + attributes       |
| `compute_all_features_with_gpu()` | Compute all features            | Augmented points        | Features (normals, curvature, etc.) |
| `_enrich_single_file()`           | Main worker                     | LAZ file path           | Multiple augmented LAZ files        |

## Verification

To verify features are computed correctly on augmented geometry:

```python
# Load original and augmented versions
original = laspy.read("tile.laz")
augmented = laspy.read("tile_aug1.laz")

# Points should be different (rotated, jittered, scaled, dropout)
assert not np.allclose(original.xyz, augmented.xyz)

# Features should also be different (computed on different geometry)
assert not np.allclose(original.normal_x, augmented.normal_x)
assert not np.allclose(original.curvature, augmented.curvature)

# But classification codes should be similar (after dropout)
# (classification is preserved through augmentation)
```

## Summary

**The augmentation implementation ensures feature-geometry consistency by:**

1. ✅ Applying geometric transformations FIRST
2. ✅ Computing features AFTER transformation (for each version)
3. ✅ Saving each version with its own correct features
4. ✅ Processing versions independently in a loop
5. ✅ Using the same feature computation pipeline for all versions

This approach guarantees that normals, curvature, planarity, and all other geometric features accurately describe the augmented geometry, leading to better training data quality and model performance.

---

**Date:** October 4, 2025  
**Version:** v1.6.0+  
**Status:** ✅ Implemented and Functional
