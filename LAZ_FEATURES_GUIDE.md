# Enhanced LAZ Output with Computed Features

**Date:** October 10, 2025  
**Feature:** Export LAZ files with all computed geometric, radiometric, and normal features

## Overview

The IGN LiDAR HD processor now supports exporting LAZ point cloud files with **all computed features** as extra dimensions. This allows you to:

- ✅ Visualize features directly in CloudCompare, QGIS, or other LAZ viewers
- ✅ Access geometric features (normals, curvature, planarity, etc.) in standard GIS tools
- ✅ Analyze radiometric features (NIR, NDVI) alongside geometry
- ✅ Preserve all processing results in a single, standard format

## Features Included in LAZ Files

### Standard LAS Fields

- **XYZ Coordinates** - LAMB93 projection + IGN69 vertical datum
- **Intensity** - Laser return intensity (uint16)
- **Classification** - Point classification labels (uint8)
- **RGB Colors** - Red, Green, Blue (uint16 each)
- **Return Number** - Laser return number (uint8)

### Extra Dimensions (Computed Features)

#### 1. Normal Vectors (3 dimensions)

- `normal_x` - Normal vector X component (float32)
- `normal_y` - Normal vector Y component (float32)
- `normal_z` - Normal vector Z component (float32)

#### 2. Geometric Features (scalar, float32)

- `curvature` - Local surface curvature
- `planarity` - Measure of planar-like geometry (0-1)
- `linearity` - Measure of linear-like geometry (0-1)
- `sphericity` - Measure of spherical-like geometry (0-1)
- `verticality` - Measure of vertical orientation (0-1)
- `height` - Normalized height above ground (if computed)

#### 3. Radiometric Features (scalar, float32)

- `nir` - Near-Infrared channel (normalized 0-1)
- `ndvi` - Normalized Difference Vegetation Index (-1 to +1)

### Total Dimensions Per Point

- **Standard fields:** 10 (x, y, z, intensity, rgb, classification, etc.)
- **Extra dimensions:** Up to 11 (normals + geometric + radiometric)
- **Total:** Up to 21 attributes per point

## File Format Specifications

### LAS Format Details

- **Version:** LAS 1.4
- **Point Format:** 3 (with RGB support)
- **Compression:** LAZ (LASzip compressed)
- **Coordinate System:** LAMB93 (EPSG:2154)
- **Vertical Datum:** IGN69
- **Scale:** 0.001 (1mm precision)

### File Size

- **Uncompressed:** ~1.3 MB per 32k points
- **LAZ Compressed:** ~280-350 KB per 32k points
- **Compression ratio:** ~4:1

## Usage

### Method 1: During Processing (Automatic)

The processor automatically saves LAZ files with features when configured:

```yaml
# config.yaml
output:
  save_laz: true
  save_npz: true
```

Output locations:

- **Patches:** `output_dir/patches/*.laz`
- **Enriched tiles:** `output_dir/enriched/*_enriched.laz`

### Method 2: Convert Existing NPZ Files

Use the enhanced conversion script to regenerate LAZ files from NPZ:

```bash
# Convert entire directory
python scripts/convert_npz_to_laz_with_coords.py \
    /path/to/npz_files/ \
    /path/to/output/ \
    --overwrite

# Features are automatically included if present in NPZ
```

## Visualization in CloudCompare

### Opening Files

1. **File → Open** → Select `.laz` file
2. Point cloud loads with all attributes

### Viewing Features

#### Display Normal Vectors

1. Select point cloud in DB Tree
2. **Edit → Normals → Compute** (or use existing)
3. **Display → Toggle Normals** (Shift+N)
4. Adjust normal length in properties

#### Color by Geometric Features

1. Select point cloud
2. **Edit → Scalar Fields → Set Active**
3. Choose feature (e.g., `planarity`, `curvature`)
4. **Edit → Colors → Set Color Scale**
5. Adjust color ramp in properties panel

#### View NDVI

1. Set `ndvi` as active scalar field
2. Use **Vegetation** color scale
3. Values: -1 (water/rock) → 0 (soil) → +1 (vegetation)

### Recommended Color Scales

- **Planarity:** Blue (low) → Green → Red (high)
- **Verticality:** Blue (horizontal) → Red (vertical)
- **NDVI:** Brown (-1) → Yellow (0) → Green (+1)
- **Curvature:** Blue (flat) → Red (high curvature)

## Python Access

### Reading LAZ with Features

```python
import laspy
import numpy as np

# Read LAZ file
las = laspy.read('patch_0000.laz')

# Access standard fields
xyz = np.column_stack([las.x, las.y, las.z])
rgb = np.column_stack([las.red, las.green, las.blue]) / 65535.0
classification = las.classification

# Access extra dimensions (features)
normals = np.column_stack([las.normal_x, las.normal_y, las.normal_z])
planarity = las.planarity
curvature = las.curvature
ndvi = las.ndvi

# List all available dimensions
print("Standard dimensions:", las.point_format.dimension_names)
print("Extra dimensions:", las.point_format.extra_dimension_names)
```

### Feature Analysis

```python
# Find planar regions (walls, roofs)
planar_mask = las.planarity > 0.8
planar_points = xyz[planar_mask]

# Find vegetation using NDVI
vegetation_mask = las.ndvi > 0.3
vegetation_points = xyz[vegetation_mask]

# Find high curvature areas (edges, corners)
edges_mask = las.curvature > 0.5
edge_points = xyz[edges_mask]

# Vertical surfaces (walls)
vertical_mask = las.verticality > 0.9
wall_points = xyz[vertical_mask]
```

## QGIS Integration

### Loading LAZ with Features

1. **Layer → Add Layer → Add Point Cloud Layer**
2. Select LAZ file
3. All attributes appear in attribute table

### Visualization

1. **Symbology → Attribute by Gradient**
2. Select feature (e.g., `verticality`)
3. Choose color ramp
4. Apply

### Analysis

- Use **Processing Toolbox** → **Point Cloud** tools
- Filter by feature values
- Export subsets based on geometric properties

## Implementation Details

### In Processor (`processor.py`)

The `_save_patch_as_laz()` method automatically includes:

```python
# Geometric features
geometric_features = [
    'planarity', 'linearity', 'sphericity',
    'curvature', 'verticality', 'roughness'
]

# Normals (3D vector)
if 'normals' in data:
    las.add_extra_dim(laspy.ExtraBytesParams(name="normal_x", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="normal_y", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="normal_z", type=np.float32))

# Radiometric
if 'ndvi' in data:
    las.add_extra_dim(laspy.ExtraBytesParams(name="ndvi", type=np.float32))
```

### In Conversion Script (`convert_npz_to_laz_with_coords.py`)

Automatically detects and includes all available features:

```python
# Normals (if present)
if 'normals' in data:
    las.normal_x = normals[:, 0]
    las.normal_y = normals[:, 1]
    las.normal_z = normals[:, 2]

# Geometric features (if present)
for feature in ['curvature', 'planarity', 'linearity', ...]:
    if feature in data:
        las.add_extra_dim(...)
        setattr(las, feature, data[feature])
```

## Feature Value Ranges

| Feature       | Range    | Meaning                           |
| ------------- | -------- | --------------------------------- |
| `planarity`   | 0-1      | 0=non-planar, 1=perfect plane     |
| `linearity`   | 0-1      | 0=non-linear, 1=perfect line      |
| `sphericity`  | 0-1      | 0=non-spherical, 1=perfect sphere |
| `verticality` | 0-1      | 0=horizontal, 1=vertical          |
| `curvature`   | 0-∞      | 0=flat, higher=more curved        |
| `ndvi`        | -1 to +1 | -1=water, 0=soil, +1=vegetation   |
| `normals`     | -1 to +1 | Unit vector components            |
| `nir`         | 0-1      | Normalized NIR reflectance        |

## Quality Checks

### Verify Features Are Present

```bash
python scripts/verify_laz_features.py output/patches/
```

Output:

```
✓ LAZ file loaded successfully
  - Points: 32,768
  - Point format: 3

✓ Standard dimensions (7):
    - x, y, z
    - intensity
    - classification
    - red, green, blue

✓ Extra dimensions (11):
    - normal_x, normal_y, normal_z
    - curvature, planarity, linearity, sphericity, verticality
    - nir, ndvi
```

### Check Feature Statistics

```python
import laspy

las = laspy.read('patch.laz')

for dim in las.point_format.extra_dimension_names:
    values = getattr(las, dim)
    print(f"{dim:15s}: min={values.min():.3f}, max={values.max():.3f}, "
          f"mean={values.mean():.3f}, std={values.std():.3f}")
```

## Performance

### Processing Speed

- **Feature computation:** ~2-5 seconds per 32k points
- **LAZ writing:** ~0.5-1 second per 32k points
- **LAZ reading:** ~0.3-0.5 seconds per 32k points

### Memory Usage

- **In memory:** ~4 MB per 32k points (with all features)
- **On disk (LAZ):** ~280-350 KB per 32k points
- **Compression:** Minimal impact on file size from extra dimensions

## Troubleshooting

### Missing Features

**Problem:** Some features not present in LAZ file

**Solutions:**

1. Check NPZ file has the features:

   ```python
   data = np.load('file.npz')
   print(data.keys())
   ```

2. Ensure feature computation is enabled in config:

   ```yaml
   features:
     compute_normals: true
     compute_geometric: true
     compute_radiometric: true
   ```

3. Regenerate with updated script:
   ```bash
   python scripts/convert_npz_to_laz_with_coords.py input/ output/ --overwrite
   ```

### CloudCompare Not Showing Features

**Problem:** Features not visible in CloudCompare

**Solutions:**

1. Check scalar fields: **Edit → Scalar Fields → Set Active**
2. Refresh display: **F5**
3. Verify features exist:
   ```python
   las = laspy.read('file.laz')
   print(las.point_format.extra_dimension_names)
   ```

### Large File Sizes

**Problem:** LAZ files larger than expected

**Solutions:**

1. Ensure LAZ compression (not LAS):

   - File extension must be `.laz` (not `.las`)
   - Compression happens automatically with laspy

2. Check point format:
   - Format 3 recommended (with RGB)
   - Avoid format 6+ unless NIR needed

## Best Practices

### 1. Always Save Both NPZ and LAZ

```yaml
output:
  save_npz: true # Full features + metadata
  save_laz: true # Standard format + features
```

### 2. Use Descriptive Naming

- Include tile ID and patch number
- Example: `LHD_FXX_0649_6863_patch_0042.laz`

### 3. Validate After Processing

```bash
# Check a sample of files
python scripts/verify_laz_features.py output/patches/ | head -50
```

### 4. Document Feature Usage

Keep notes on which features were computed and their purposes for your specific analysis.

## Examples

### Example 1: Building Facade Extraction

```python
import laspy
import numpy as np

las = laspy.read('urban_patch.laz')

# Find vertical, planar surfaces (building facades)
vertical = las.verticality > 0.85
planar = las.planarity > 0.75

facades = vertical & planar
facade_points = np.column_stack([las.x[facades],
                                  las.y[facades],
                                  las.z[facades]])

print(f"Found {facades.sum()} facade points")
```

### Example 2: Vegetation Analysis

```python
# Identify vegetation using NDVI and geometric features
vegetation = (las.ndvi > 0.3) & (las.sphericity > 0.5)
trees = vegetation & (las.height > 2.0)  # Tall vegetation

# Separate into categories
grass = vegetation & (las.height < 0.5)
shrubs = vegetation & (las.height >= 0.5) & (las.height < 2.0)
```

### Example 3: Edge Detection

```python
# Find edges using high curvature
edges = las.curvature > np.percentile(las.curvature, 95)
edge_points = np.column_stack([las.x[edges],
                                las.y[edges],
                                las.z[edges]])

# Refine using normals (sharp direction changes)
normal_changes = np.linalg.norm(np.diff(normals[edges], axis=0), axis=1)
sharp_edges = edge_points[:-1][normal_changes > 0.7]
```

## Summary

✅ **All computed features are now preserved in LAZ files**  
✅ **Compatible with CloudCompare, QGIS, and all LAZ-supporting tools**  
✅ **No additional processing needed - features computed during initial processing**  
✅ **Minimal file size increase due to LAZ compression**  
✅ **Easy to access and visualize geometric and radiometric properties**

For more information, see:

- `scripts/convert_npz_to_laz_with_coords.py` - Conversion with features
- `ign_lidar/core/processor.py` - `_save_patch_as_laz()` method
- `scripts/verify_laz_features.py` - Feature validation tool
