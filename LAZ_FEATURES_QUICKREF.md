# LAZ Features Quick Reference

## Available Features in Enhanced LAZ Files

### ✅ Standard Fields (Always Present)

```
x, y, z              - LAMB93 coordinates + IGN69
intensity            - Laser return intensity
classification       - Point labels
red, green, blue     - RGB colors
return_number        - Return number
```

### ✅ Extra Dimensions (When Computed)

**Normals** (3D vectors)

```
normal_x, normal_y, normal_z - Unit normal vectors
```

**Geometric** (0-1 range except curvature)

```
planarity            - Planar surface measure
linearity            - Linear structure measure
sphericity           - Spherical structure measure
verticality          - Vertical orientation measure
curvature            - Local surface curvature
height               - Normalized height
```

**Radiometric** (0-1 range)

```
nir                  - Near-infrared reflectance
ndvi                 - Vegetation index (-1 to +1)
```

## CloudCompare Quick Commands

### View Features

```
1. Edit → Scalar Fields → Set Active → Choose feature
2. Edit → Colors → Set Color Scale
3. Adjust color ramp in Properties
```

### View Normals

```
Shift+N - Toggle normals display
Properties → Normals → Adjust length
```

### Common Filters

```
# Planar surfaces (walls, roofs)
Tools → Segmentation → Label → planarity > 0.8

# Vegetation
Tools → Segmentation → Label → ndvi > 0.3

# Vertical surfaces
Tools → Segmentation → Label → verticality > 0.9
```

## Python Quick Access

```python
import laspy
import numpy as np

# Read LAZ
las = laspy.read('file.laz')

# Standard fields
xyz = np.c_[las.x, las.y, las.z]
rgb = np.c_[las.red, las.green, las.blue] / 65535

# Features
normals = np.c_[las.normal_x, las.normal_y, las.normal_z]
planarity = las.planarity
ndvi = las.ndvi

# List all
print(las.point_format.extra_dimension_names)
```

## Feature Value Interpretation

| Feature     | Low (0-0.3)     | Medium (0.3-0.7) | High (0.7-1.0)   |
| ----------- | --------------- | ---------------- | ---------------- |
| Planarity   | Rough/scattered | Mixed            | Smooth planes    |
| Linearity   | Volume/area     | Mixed            | Lines/edges      |
| Sphericity  | Planar/linear   | Mixed            | Spherical        |
| Verticality | Horizontal      | Diagonal         | Vertical         |
| NDVI        | Bare soil/water | Low vegetation   | Dense vegetation |

## Typical Values by Object Type

| Object          | Planarity | Verticality | NDVI    | Notes                    |
| --------------- | --------- | ----------- | ------- | ------------------------ |
| Building facade | 0.8-0.95  | 0.85-1.0    | <0.2    | High planar, vertical    |
| Roof            | 0.7-0.9   | 0.0-0.3     | <0.2    | Planar, horizontal       |
| Trees           | 0.2-0.4   | 0.3-0.6     | 0.5-0.9 | Low planarity, high NDVI |
| Ground          | 0.5-0.8   | 0.0-0.2     | 0.1-0.4 | Variable by terrain      |
| Power lines     | 0.1-0.3   | 0.4-0.7     | <0.1    | High linearity           |

## File Locations

**Recovery script:** `scripts/convert_npz_to_laz_with_coords.py`  
**Verification:** `scripts/verify_laz_features.py`  
**Full guide:** `LAZ_FEATURES_GUIDE.md`  
**Recovery summary:** `LAZ_RECOVERY_SUMMARY.md`

## Regenerate with Features

```bash
# From NPZ files
python scripts/convert_npz_to_laz_with_coords.py \
    input_npz_dir/ \
    output_laz_dir/ \
    --overwrite

# Check results
python scripts/verify_laz_features.py output_laz_dir/
```

## Common Issues

**Q: Features missing in CloudCompare?**  
A: Check `Edit → Scalar Fields → Set Active`

**Q: How to verify features exist?**  
A: Run `python scripts/verify_laz_features.py file.laz`

**Q: File too large?**  
A: Ensure file extension is `.laz` (not `.las`) for compression

**Q: Need more features?**  
A: Check NPZ file has them: `np.load('file.npz').keys()`

## Tips

- Use **planarity + verticality** to find building facades
- Use **NDVI > 0.3** for vegetation detection
- Use **curvature** to find edges and corners
- Combine multiple features for better classification
- Export filtered points as new LAZ from CloudCompare

---

_Last updated: October 10, 2025_
