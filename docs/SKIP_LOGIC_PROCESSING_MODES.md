# Skip Logic for Processing Modes

## Overview

The IGN LiDAR HD processor supports three processing modes with intelligent skip detection that validates outputs to ensure completeness before skipping tiles.

## Processing Modes

### 1. `enriched_only` Mode

- **Purpose**: Only generate enriched LAZ tiles with computed features
- **Output**: Enriched LAZ files directly in `output_dir`
- **Use Case**: When you only need full tiles with features, no ML patches

**Output Structure:**

```
output_dir/
├── TILE_NAME_enriched.laz
├── TILE_NAME_enriched.laz
└── ...
```

**Skip Logic:**

1. ✅ Check if `TILE_NAME_enriched.laz` exists in `output_dir`
2. ✅ Validate file size (> 1KB)
3. ✅ **Validate content** - checks for:
   - Core geometric features (normals, curvature, height)
   - RGB channels (if `use_rgb: true`)
   - NIR channel (if `use_infrared: true`)
   - NDVI (if `compute_ndvi: true`)
   - Extra geometric features (if `include_extra_features: true`)
   - Classification data (if BD TOPO enabled)
   - Forest attributes (if BD Forêt enabled)
   - Agriculture data (if RPG enabled)
   - Cadastre parcels (if Cadastre enabled)
4. ✅ Skip only if **all required features are present**
5. ⚠️ Reprocess if features are missing

**Skip Messages:**

- `⏭️ Valid enriched LAZ exists (123.4 MB, 25 features), skipping`
- `🔄 Enriched LAZ incomplete (missing: ['ndvi', 'classification']), reprocessing`
- `🔄 No enriched LAZ found, processing`

---

### 2. `both` Mode

- **Purpose**: Generate both enriched LAZ tiles AND ML-ready patches
- **Output**: Enriched LAZ in subdirectory + patches in output_dir
- **Use Case**: When you need both full tiles and ML datasets

**Output Structure:**

```
output_dir/
├── enriched/
│   ├── TILE_NAME_enriched.laz
│   └── ...
├── TILE_NAME_hybrid_patch_0000.npz
├── TILE_NAME_hybrid_patch_0001.npz
└── ...
```

**Skip Logic:**

1. ✅ Check if enriched LAZ exists in `enriched/` subdirectory
2. ✅ Validate enriched LAZ content (same as `enriched_only` mode)
3. ✅ Check if patches exist in `output_dir`
4. ✅ Validate patches are not corrupted
5. ✅ Skip only if **BOTH enriched LAZ AND patches are valid**
6. ⚠️ Reprocess if either is missing or invalid

**Skip Messages:**

- `⏭️ Both enriched LAZ (123.4 MB) and 45 patches exist, skipping`
- `🔄 Enriched LAZ exists but patches missing, processing patches only`
- `🔄 45 patches exist but enriched LAZ missing, processing enriched LAZ only`
- `🔄 No outputs found, full processing`

---

### 3. `patches_only` Mode

- **Purpose**: Only generate ML-ready patch files
- **Output**: Patch files in output_dir
- **Use Case**: When you only need ML datasets, not full tiles

**Output Structure:**

```
output_dir/
├── TILE_NAME_hybrid_patch_0000.npz
├── TILE_NAME_hybrid_patch_0001.npz
├── TILE_NAME_hybrid_patch_0000_aug1.npz  # if augmentation enabled
└── ...
```

**Skip Logic:**

1. ✅ Find all patches for tile (supports multiple formats: `.npz`, `.h5`, `.pt`, `.laz`)
2. ✅ Validate each patch file:
   - Check file size (> 1KB)
   - Load and verify structure (coords/points, labels, features)
   - Check for corruption
3. ✅ Count valid vs corrupted patches
4. ✅ Skip if all patches are valid and none corrupted
5. ⚠️ Reprocess if corrupted patches detected

**Skip Messages:**

- `⏭️ 45 valid patches exist, skipping`
- `🔄 Corrupted patches detected (3 of 45), reprocessing`
- `🔄 No patches found, processing`

---

## Configuration

### Enable Skip Logic

```yaml
processor:
  skip_existing: true # Enable intelligent skip detection
  processing_mode: enriched_only # or 'both' or 'patches_only'
```

### Disable Skip Logic (Force Reprocessing)

```yaml
processor:
  skip_existing: false # Process all tiles regardless of existing outputs
```

---

## Metadata-Based Skip Detection

In addition to output validation, the processor uses **metadata tracking** to detect configuration changes:

### Metadata File

Each processed tile gets a metadata JSON file:

```
output_dir/.metadata/
└── TILE_NAME.json
```

### Metadata Contents

```json
{
  "tile_name": "LHD_FXX_0558_6413_PTS_O_LAMB93_IGN69",
  "processing_time": 45.2,
  "timestamp": "2025-10-16T10:30:00",
  "config_hash": "abc123...",
  "output_files": {
    "enriched_laz": {
      "path": "/path/to/enriched.laz",
      "size_bytes": 123456789
    },
    "patches": {
      "count": 45,
      "format": "npz"
    }
  }
}
```

### Configuration Change Detection

The processor computes a hash of relevant configuration parameters:

- Feature settings (k_neighbors, feature_mode, etc.)
- RGB/NIR/NDVI settings
- Classification settings
- Preprocessing settings
- Output format

**If configuration changed:**

- ⚠️ Reprocess tile even if outputs exist
- 📝 Log: "Reprocessing: Configuration changed since last processing"

**If configuration unchanged:**

- ✅ Skip based on output validation
- 📝 Log: "Already processed with same config, skipping"

---

## Implementation Details

### Key Files

- `ign_lidar/core/skip_checker.py` - Output validation logic
- `ign_lidar/core/processing_metadata.py` - Config change detection
- `ign_lidar/core/processor.py` - Main processing flow

### Skip Checker Initialization

```python
self.skip_checker = PatchSkipChecker(
    output_format=self.output_format,
    architecture=self.architecture,
    num_augmentations=3,
    augment=True,
    validate_content=True,  # Enable content validation
    min_file_size=1024,     # 1KB minimum
    only_enriched_laz=(self.processing_mode == "enriched_only"),
)
```

### Processing Flow

1. **Check metadata** → If config changed, reprocess
2. **Check outputs** → Validate existence and content
3. **Decide skip/process** → Based on validation results
4. **Log decision** → Clear message about why skipping or processing
5. **Save metadata** → After successful processing

---

## Bug Fixes (October 16, 2025)

### Issue 1: Simple File Existence Check

**Problem:** In `enriched_only` mode, the skip checker was only checking if files existed, not validating their content. This caused tiles with incomplete features to be skipped.

**Fix:** Added content validation to `enriched_only` mode using `_validate_enriched_laz()` to ensure all required features are present before skipping.

**Impact:** Now correctly detects and reprocesses tiles missing features like NDVI, classification, etc.

### Issue 2: Output Path Mismatch

**Problem:** The processor was saving enriched LAZ to `output_dir/enriched/` subdirectory in `enriched_only` mode, but the skip checker was looking for files directly in `output_dir/`.

**Fix:** Changed output path in `enriched_only` mode to save directly to `output_dir/TILE_NAME_enriched.laz` (matching skip checker expectations).

**Impact:** Skip logic now correctly finds and validates enriched LAZ files in all modes.

---

## Testing

### Test Skip Logic

```python
# Test enriched_only mode
python -m ign_lidar.main \
    processor.processing_mode=enriched_only \
    processor.skip_existing=true \
    input_dir=/path/to/tiles \
    output_dir=/path/to/output

# Expected: First run processes all tiles
# Expected: Second run skips tiles with complete features
# Expected: Reprocesses tiles with missing features

# Test both mode
python -m ign_lidar.main \
    processor.processing_mode=both \
    processor.skip_existing=true \
    input_dir=/path/to/tiles \
    output_dir=/path/to/output

# Expected: Skips only if BOTH enriched LAZ and patches exist

# Test patches_only mode
python -m ign_lidar.main \
    processor.processing_mode=patches_only \
    processor.skip_existing=true \
    input_dir=/path/to/tiles \
    output_dir=/path/to/output

# Expected: Skips only if all patches are valid
```

### Force Reprocessing

```bash
# Option 1: Disable skip_existing
python -m ign_lidar.main processor.skip_existing=false

# Option 2: Delete outputs
rm -rf /path/to/output/*

# Option 3: Delete metadata
rm -rf /path/to/output/.metadata/
```

---

## Best Practices

1. **Use `skip_existing=true` for large datasets** - Saves time on reruns
2. **Monitor skip messages** - Understand why tiles are being skipped/processed
3. **Check logs for "missing features"** - Indicates incomplete outputs
4. **Delete `.metadata/` if you change config significantly** - Forces full reprocessing
5. **Use `enriched_only` for tile-based workflows** - Faster than `both`
6. **Use `patches_only` for ML-only workflows** - Skips tile saving overhead
7. **Use `both` when you need maximum flexibility** - Keeps all outputs

---

## Troubleshooting

### Tiles Not Being Skipped

- Check if `skip_existing=true` in config
- Check if output files exist in correct location
- Check logs for validation errors
- Check if config changed (metadata hash mismatch)

### Tiles Being Skipped When They Shouldn't

- Check if features are actually missing
- Delete `.metadata/` to force revalidation
- Check file size (files < 1KB considered invalid)
- Use `skip_existing=false` to force reprocessing

### Performance Issues

- Content validation adds ~1-2 seconds per tile
- Trade-off: validation time vs reprocessing time
- For very large datasets, consider disabling validation temporarily

---

## Future Improvements

1. **Parallel validation** - Validate multiple tiles simultaneously
2. **Cached validation results** - Store validation results to avoid re-checking
3. **Partial reprocessing** - Only compute missing features instead of full reprocessing
4. **Progress persistence** - Resume interrupted processing runs
5. **Smart batch validation** - Sample-based validation for very large datasets
