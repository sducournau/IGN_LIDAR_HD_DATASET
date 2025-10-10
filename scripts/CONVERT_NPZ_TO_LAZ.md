# convert_npz_to_laz.py - Usage Guide

## Description

Converts NPZ patch files back to LAZ point cloud format for visualization and further processing.

## Features

- ✅ Converts full NPZ files with point cloud data to LAZ
- ✅ Detects and handles metadata-only NPZ files gracefully
- ✅ Batch directory conversion with summary
- ✅ Preserves point coordinates, intensity, RGB, and classification
- ✅ Automatic output path generation

## Requirements

```bash
pip install numpy laspy
```

## Usage

### Single File Conversion

```bash
# Auto-generate output filename (.npz -> .laz)
python scripts/convert_npz_to_laz.py patch_0001.npz

# Specify output filename
python scripts/convert_npz_to_laz.py patch_0001.npz output.laz
```

### Directory Conversion

```bash
# Convert all NPZ files in directory (output in same directory)
python scripts/convert_npz_to_laz.py patches/

# Convert to different output directory
python scripts/convert_npz_to_laz.py input_patches/ output_laz/
```

## NPZ File Requirements

The script requires **full NPZ files** containing point cloud data with at least one of these keys:

- `points` - Point coordinates [N, 3]
- `coords` - Coordinate array [N, 3]

### Optional Data Fields

If present, these fields will be included in the LAZ file:

- `intensity` - Point intensities
- `rgb` - RGB colors [N, 3]
- `classification` or `labels` - Point classifications
- `return_number` - LiDAR return numbers
- `number_of_returns` - Number of returns per pulse

## Metadata-Only NPZ Files

The script automatically detects **metadata-only NPZ files** (files containing only the `metadata` key without actual point cloud data).

### Single File Behavior

When encountering a metadata-only file, the script will:

1. Display a warning
2. Show the metadata contents
3. Explain why conversion cannot proceed

Example output:

```
⚠️  This appears to be a metadata-only NPZ file.
    These files only contain patch metadata without actual point cloud data.

Metadata contents:
  num_points: 32768
  has_rgb: True
  has_nir: True
  has_ndvi: True
  has_normals: True
  has_labels: True
  bbox_min: [-75.0, -75.0, 26.61]
  bbox_max: [74.75, 74.5, 65.30]
  centroid: [7.30, 9.31, 41.00]

❌ Cannot convert metadata-only NPZ files to LAZ format.
   These files were likely created with save_metadata=True but without actual point data.
```

### Directory Batch Behavior

When processing a directory with metadata-only files:

- Metadata-only files are automatically skipped with a warning
- Full NPZ files are converted normally
- A summary is displayed at the end

Example summary:

```
============================================================
CONVERSION SUMMARY
============================================================
✅ Successfully converted: 45/146 files
⚠️  Metadata-only files skipped: 101
   (These files only contain metadata without point cloud data)
============================================================
```

## Output LAZ Format

Generated LAZ files include:

- **Point Format:** 3 (includes XYZ, intensity, RGB, classification)
- **Version:** 1.4
- **Precision:** 1mm (0.001 scale)
- **Compression:** Automatic LAZ compression

## Examples

### Example 1: Convert Training Patches

```bash
# Check if files are full NPZ or metadata-only
python scripts/convert_npz_to_laz.py /path/to/patches/patch_0000.npz

# Convert entire training set
python scripts/convert_npz_to_laz.py /path/to/training/patches/ /path/to/output/laz/
```

### Example 2: Visualization Workflow

```bash
# Convert a subset for visualization
python scripts/convert_npz_to_laz.py \
  /mnt/c/Users/Simon/ign/full_patches/urban_dense/ \
  /mnt/c/Users/Simon/ign/visualization/

# Then open in CloudCompare, QGIS, etc.
```

## Troubleshooting

### "NPZ file must contain either 'points' or 'coords' field"

**Cause:** The NPZ file is metadata-only or corrupted.

**Solution:**

1. Check file size (metadata-only files are ~450-500 bytes)
2. Verify NPZ contents:

```python
import numpy as np
data = np.load('patch.npz', allow_pickle=True)
print(list(data.keys()))
```

3. If metadata-only, re-run processing to create full NPZ files

### "Object arrays cannot be loaded when allow_pickle=False"

**Cause:** NPZ contains pickled objects (now fixed in the script)

**Solution:** Update to the latest version of the script which uses `allow_pickle=True`

### Memory Issues with Large Batches

**Solution:** Process in smaller batches:

```bash
# Process first 100 files
cd patches && ls *.npz | head -100 | xargs -I {} python ../scripts/convert_npz_to_laz.py {}
```

## Performance

Typical conversion speed:

- Small patches (4K points): ~0.1 seconds
- Medium patches (16K points): ~0.3 seconds
- Large patches (65K points): ~1 second

Batch processing 1000 patches: ~5-10 minutes

## See Also

- [NPZ_FILE_TYPES.md](../NPZ_FILE_TYPES.md) - Understanding full vs metadata-only NPZ files
- [TRAINING_COMMANDS.md](../TRAINING_COMMANDS.md) - Generating full NPZ training patches
- LAZ format specification: https://laszip.org/
