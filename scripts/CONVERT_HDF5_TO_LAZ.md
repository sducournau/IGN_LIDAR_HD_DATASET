# HDF5 to LAZ Conversion Tool

This script converts HDF5 patch files back to LAZ point cloud format for visualization in tools like CloudCompare, QGIS, or other LiDAR software.

## Quick Start

### Convert Single File

```bash
# Basic conversion (output.laz auto-generated)
python scripts/convert_hdf5_to_laz.py patch_0001.h5

# Specify output file
python scripts/convert_hdf5_to_laz.py patch_0001.h5 output.laz
```

### Convert Directory

```bash
# Convert all HDF5 files in a directory
python scripts/convert_hdf5_to_laz.py patches/ output_laz/

# In-place conversion (output to same directory)
python scripts/convert_hdf5_to_laz.py patches/
```

### Inspect HDF5 Structure

```bash
# View the structure of an HDF5 file without converting
python scripts/convert_hdf5_to_laz.py --inspect patch_0001.h5
```

## Usage

```bash
python convert_hdf5_to_laz.py [-h] [--inspect] [-q] input [output]

positional arguments:
  input           Input HDF5 file or directory containing HDF5 files
  output          Output LAZ file or directory (optional)

optional arguments:
  -h, --help      show this help message and exit
  --inspect       Inspect HDF5 file structure without conversion
  -q, --quiet     Suppress detailed output
```

## Features

### Supported Data Fields

The script automatically converts all available fields from HDF5 to LAZ:

**Standard LAS Fields:**

- XYZ coordinates (required)
- Intensity
- RGB colors
- Classification
- Return number
- Number of returns

**Extra Dimensions:**
The script preserves computed features as LAZ extra dimensions:

- Surface normals (normal_x, normal_y, normal_z)
- Geometric features (curvature, planarity, verticality, etc.)
- Contextual features (density, height, eigenvalues)
- Custom feature arrays

### HDF5 Key Variations

The converter handles multiple naming conventions:

- **Coordinates**: `points`, `coords`, `xyz`, `coordinates`, `point_cloud`
- **Intensity**: `intensity`, `intensities`, `Intensity`
- **RGB**: `rgb`, `colors`, `RGB`, `color`
- **Classification**: `classification`, `classifications`, `labels`, `label`, `class`
- **Returns**: `return_number`, `return_num`, `returns`
- **Features**: `features`, `normals`, `curvature`, `planarity`, etc.

### Data Normalization

The script handles various data ranges automatically:

- **RGB Values:**
  - [0, 1] normalized → [0, 65535] uint16
  - [0, 255] byte → [0, 65535] uint16
  - Already uint16 → preserved
- **Intensity:**

  - [0, 1] normalized → [0, 65535] uint16
  - Already uint16 → preserved

- **Features:**
  - Multi-dimensional arrays → separate LAZ extra dimensions
  - Normals → `normal_x`, `normal_y`, `normal_z`
  - Generic features → `feature_0`, `feature_1`, etc.

## Examples

### Example 1: Basic Conversion

```bash
# Convert a single patch
python scripts/convert_hdf5_to_laz.py data/patches/tile_0123_4567_patch_0001.h5

# Output: data/patches/tile_0123_4567_patch_0001.laz
```

### Example 2: Batch Conversion

```bash
# Convert all patches from a tile
python scripts/convert_hdf5_to_laz.py data/patches/tile_0123_4567/ data/laz/

# Result:
# ✅ Successfully converted: 245/245 files
```

### Example 3: Inspect Before Converting

```bash
# Check what's inside an HDF5 file
python scripts/convert_hdf5_to_laz.py --inspect data/patches/sample.h5

# Output:
# 📋 Top-level keys:
#   - points
#   - normals
#   - curvature
#   - rgb
#   - classification
#
# 📊 Datasets:
#   - points
#     Shape: (16384, 3), Type: float32, Size: 49,152
#   - normals
#     Shape: (16384, 3), Type: float32, Size: 49,152
#   ...
```

### Example 4: Quiet Mode

```bash
# Suppress detailed output
python scripts/convert_hdf5_to_laz.py -q patches/ laz_output/

# Output:
# CONVERSION SUMMARY
# ✅ Successfully converted: 100/100 files
```

## Error Handling

### Metadata-Only Files

Some HDF5 files may only contain metadata without point cloud data:

```
⚠️  This appears to be a metadata-only HDF5 file.
    These files only contain patch metadata without actual point cloud data.

Cannot convert metadata-only HDF5 files to LAZ format.
```

**Solution:** These files are informational only. Use full HDF5 files with point data.

### Missing Required Fields

```
HDF5 file must contain coordinate data.
Expected one of: ['points', 'coords', 'xyz', 'coordinates', 'point_cloud']
Found keys: ['metadata', 'features']
```

**Solution:** Ensure your HDF5 files include XYZ coordinate data.

### Invalid Data Shapes

```
Coordinate data must be (N, 3), got (16384,)
```

**Solution:** The script attempts to reshape flattened arrays automatically. If this fails, check your HDF5 file structure.

## Output Format

The generated LAZ files use:

- **LAS Version:** 1.4
- **Point Format:** 3 (includes XYZ, Intensity, RGB, Classification)
- **Scale:** 0.001 (1mm precision)
- **Compression:** LAZ compressed
- **Extra Dimensions:** All available features from HDF5

## Viewing Converted Files

### CloudCompare

```bash
# Open in CloudCompare
cloudcompare.CloudCompare output.laz

# View features as scalar fields:
# - Edit > Scalar Fields > [select feature]
# - Tools > Colors > Height Ramp
```

### QGIS

```python
# Load in QGIS (Python console)
from qgis.core import QgsVectorLayer
layer = QgsVectorLayer('output.laz', 'point_cloud', 'ogr')
QgsProject.instance().addMapLayer(layer)
```

### Python (laspy)

```python
import laspy

# Load converted LAZ
las = laspy.read('output.laz')

# Access standard fields
print(f"Points: {len(las.points):,}")
print(f"Has RGB: {las.header.point_format.id in [2, 3, 5, 7, 8, 10]}")

# Access extra dimensions
print(f"Extra dimensions: {las.point_format.extra_dimension_names}")

# Extract features
if 'normal_x' in las.point_format.dimension_names:
    normals = np.column_stack([las.normal_x, las.normal_y, las.normal_z])
    print(f"Normals shape: {normals.shape}")
```

## Performance

Conversion speed depends on:

- Number of points per patch (typically 16,384 for standard patches)
- Number of features/extra dimensions
- Disk I/O speed

**Typical Performance:**

- Single patch (16k points): ~0.5-1 second
- Directory (100 patches): ~1-2 minutes
- Large dataset (1000+ patches): ~10-20 minutes

## Comparison: HDF5 vs NPZ Conversion

| Feature              | HDF5                           | NPZ                  |
| -------------------- | ------------------------------ | -------------------- |
| **File Format**      | Binary HDF5                    | NumPy compressed     |
| **Read Speed**       | Fast (streaming)               | Fast (memory-mapped) |
| **File Size**        | Smaller (with compression)     | Larger               |
| **Metadata**         | Rich (attributes, groups)      | Limited (keys only)  |
| **Interoperability** | High (many tools support HDF5) | Python-specific      |
| **Random Access**    | Yes (efficient)                | No (load all)        |

**When to use HDF5:**

- Large datasets requiring efficient random access
- Need for rich metadata and hierarchical structure
- Sharing with non-Python tools
- Memory-constrained environments

**When to use NPZ:**

- Python-only workflows
- Simple flat data structures
- Quick prototyping
- No need for metadata

## Related Scripts

- **convert_npz_to_laz.py**: Convert NPZ patches to LAZ
- **convert_npz_to_laz.md**: NPZ conversion documentation
- **verify_npz_fix.py**: Verify NPZ file integrity

## Requirements

```bash
pip install numpy laspy h5py
```

Or use the package environment:

```bash
conda activate ign-lidar-hd
# All dependencies already installed
```

## Troubleshooting

### Import Error: h5py not found

```bash
pip install h5py
# or
conda install h5py -c conda-forge
```

### LAZ Write Error

If you get compression errors:

```bash
pip install laspy[lazrs]
# or
pip install lazrs
```

### Memory Error (Large Files)

For very large HDF5 files:

```python
# Process in chunks (not yet implemented)
# Contact maintainers for chunked conversion support
```

## Contributing

Found a bug or have suggestions?

- Open an issue: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- Submit a PR: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/pulls

## License

MIT License - See LICENSE file for details.

---

**Author:** IGN LiDAR HD Team  
**Date:** October 10, 2025  
**Version:** 1.0.0
