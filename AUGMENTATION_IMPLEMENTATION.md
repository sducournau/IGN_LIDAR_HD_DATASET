# Data Augmentation Implementation - Complete

## Overview

The `--augment` and `--num-augmentations` CLI parameters are now **fully functional** in the `ign-lidar-hd enrich` command. This implements geometric data augmentation at the ENRICH phase, before feature computation, ensuring feature-geometry consistency.

**⚡ ENABLED BY DEFAULT**: As of implementation, augmentation is **enabled by default** with 3 augmented versions per tile. Each tile produces 4 files total: 1 original + 3 augmented versions. Use `--no-augment` to disable if you only want original tiles.

## What Was Implemented

### 1. **Worker Function Signature Updated**

- Added `augment` and `num_augmentations` parameters to `_enrich_single_file()` worker function
- Parameters are now properly unpacked from the args tuple

### 2. **Augmentation Logic Added**

The worker now:

1. Stores intensity and return_number from the original LAZ file
2. Creates multiple versions (original + N augmented) using `augment_raw_points()` from `utils.py`
3. Processes each version in a loop:
   - Applies geometric transformations (rotation, jitter, scaling, dropout)
   - Computes features on the augmented geometry
   - Saves each version with a suffix (`_aug1`, `_aug2`, etc.)

### 3. **File Naming Convention**

- Original: `tile_name.laz`
- Augmented versions: `tile_name_aug1.laz`, `tile_name_aug2.laz`, etc.

### 4. **Integration with cmd_enrich**

- Augmentation settings are extracted from CLI args
- Settings are logged for transparency
- Worker args tuple updated to include augmentation parameters

## Usage

```bash
# Default behavior (augmentation ENABLED with 3 versions)
ign-lidar-hd enrich \
  --input-dir /path/to/raw_tiles \
  --output /path/to/enriched \
  --mode building
# This creates 4 files per tile: original + 3 augmented versions

# Explicitly specify augmentation (same as default)
ign-lidar-hd enrich \
  --input-dir /path/to/raw_tiles \
  --output /path/to/enriched \
  --mode building \
  --augment \
  --num-augmentations 3

# Disable augmentation (only original tiles)
ign-lidar-hd enrich \
  --input-dir /path/to/raw_tiles \
  --output /path/to/enriched \
  --mode building \
  --no-augment

# Custom number of augmented versions
ign-lidar-hd enrich \
  --input-dir /path/to/raw_tiles \
  --output /path/to/enriched \
  --mode building \
  --augment \
  --num-augmentations 5
```

## Augmentation Transformations

Each augmented version applies:

1. **Random rotation** around Z-axis (0-360°) - preserves vertical structures
2. **Random jitter** - Gaussian noise (σ=0.1m) - simulates sensor noise
3. **Random scaling** (0.95-1.05) - simulates distance variations
4. **Random dropout** (5-15%) - simulates occlusion and missing data

## Benefits

### ✅ Feature-Geometry Consistency

- Features (normals, curvature, planarity, etc.) computed on augmented geometry
- No mismatch between coordinates and derived features
- Better model training quality

### ✅ Efficiency

- All augmented versions created in a single pass
- Shares preprocessing (artifact mitigation) across versions
- Memory-efficient: processes one version at a time

### ✅ Compatibility

- Works with all modes: `core` and `full` (building)
- Compatible with RGB augmentation (`--add-rgb`)
- Compatible with preprocessing (`--preprocess`)
- Works with GPU acceleration (`--use-gpu`)

## Example Output

For a single input tile with `--num-augmentations 3`:

```
Input:
  raw_tiles/Tile_0001.laz

Output:
  enriched/Tile_0001.laz          # Original
  enriched/Tile_0001_aug1.laz     # Augmented version 1
  enriched/Tile_0001_aug2.laz     # Augmented version 2
  enriched/Tile_0001_aug3.laz     # Augmented version 3
```

## Implementation Details

### Version Processing Loop

```python
# Create versions list (original + augmented)
versions_to_process = []
if augment and num_augmentations > 0:
    # Version 0: Original
    versions_to_process.append({
        'suffix': '',
        'points': points,
        'classification': classification,
        ...
    })

    # Versions 1-N: Augmented
    for aug_idx in range(num_augmentations):
        aug_points, ... = augment_raw_points(...)
        versions_to_process.append({
            'suffix': f'_aug{aug_idx + 1}',
            'points': aug_points,
            ...
        })

# Process each version
for version_data in versions_to_process:
    # Compute features on versioned geometry
    normals, curvature, ... = compute_all_features_with_gpu(
        version_data['points'],
        version_data['classification'],
        ...
    )

    # Save with versioned filename
    output_path_ver = output_path.parent / (
        output_path.stem + version_data['suffix'] + output_path.suffix
    )
    las_out.write(output_path_ver, do_compress=True)
```

### Integration with Other Features

- **Preprocessing**: Applied BEFORE augmentation (shared across all versions)
- **RGB augmentation**: Applied to EACH version independently
- **GPU acceleration**: Used for feature computation on each version
- **Auto-params**: Tile analysis shared across versions

## Testing

To test the implementation:

```bash
# Process a small test dataset
ign-lidar-hd enrich \
  --input-dir test_tiles \
  --output test_output \
  --mode core \
  --augment \
  --num-augmentations 2 \
  --num-workers 1

# Check output
ls -lh test_output/
# Should see: tile.laz, tile_aug1.laz, tile_aug2.laz
```

## Performance Considerations

### Memory Usage

- Each version is processed sequentially
- Memory is freed after each version is saved
- Similar memory footprint to processing N separate files

### Processing Time

- Linear scaling: 1 original + N augmented = (N+1) × processing time
- Example: 3 augmentations → 4× the time of processing original only

### Recommendations

- Use `--num-workers` to parallelize across different tiles
- For large datasets, consider processing in batches
- Augmented versions can be skipped with `--no-augment` for quick testing

## Future Enhancements

Possible improvements:

1. **Configurable augmentation parameters** - Allow users to specify rotation range, jitter sigma, etc.
2. **Selective augmentation** - Only augment certain classification codes
3. **Augmentation statistics** - Log transformation parameters for reproducibility
4. **Batch augmentation mode** - Process multiple tiles with same augmentation seed

## Conclusion

The augmentation implementation is complete and functional. The CLI parameters `--augment` and `--num-augmentations` now control the creation of geometrically augmented versions of each tile, with features properly computed on the augmented geometry for improved model training.

---

_Implementation Date: October 4, 2025_
_Status: ✅ Complete and Functional_
