# Output Format Preferences: Augmented LAZ vs QGIS

## Summary

The IGN LIDAR HD downloader now **prefers augmented LAZ files by default** instead of automatically converting to QGIS-compatible format. This change gives you more control over output formats.

## What Changed

### Default Behavior (New)

- ✅ **Enriched LAZ files are saved in LAZ 1.4 format** (point format 6+)
- ✅ **All extra dimensions are preserved** (normals, curvature, height features, etc.)
- ✅ **No automatic conversion to QGIS format**
- ✅ Better for ML workflows and advanced analysis

### Previous Behavior

- Files were sometimes automatically converted to QGIS format
- This resulted in loss of some extra dimensions

## Configuration

Two new configuration options have been added to `ign_lidar/config.py`:

```python
# Prefer augmented LAZ files (LAZ 1.4, point format 6+)
PREFER_AUGMENTED_LAZ = True

# Auto-convert to QGIS format after enrichment
# Only applies when PREFER_AUGMENTED_LAZ is False
AUTO_CONVERT_TO_QGIS = False
```

## CLI Usage

### Basic Enrichment (Default)

```bash
# Creates augmented LAZ files only
ign-lidar enrich --input-dir raw_tiles/ --output enriched/
```

Output:

- `enriched/tile_0001.laz` (LAZ 1.4, all features preserved)

### With Optional QGIS Conversion

```bash
# Creates both augmented LAZ AND QGIS-compatible versions
ign-lidar enrich --input-dir raw_tiles/ --output enriched/ --auto-convert-qgis
```

Output:

- `enriched/tile_0001.laz` (LAZ 1.4, all features)
- `enriched/tile_0001_qgis.laz` (LAZ 1.2, QGIS-compatible)

### Manual QGIS Conversion (Recommended)

If you need QGIS versions later, you can always convert manually:

```bash
# Convert a single file
ign-lidar-qgis enriched/tile_0001.laz

# Batch convert all files
python scripts/batch_convert_qgis.py enriched/
```

## Format Comparison

| Feature            | Augmented LAZ       | QGIS LAZ        |
| ------------------ | ------------------- | --------------- |
| LAZ Version        | 1.4                 | 1.2             |
| Point Format       | 6+                  | 3               |
| RGB                | ✅                  | ✅              |
| Classification     | ✅                  | ✅              |
| Intensity          | ✅                  | ✅              |
| Normals (X,Y,Z)    | ✅                  | ⚠️ (3 dims max) |
| Curvature          | ✅                  | ❌              |
| Height Features    | ✅                  | ❌              |
| Geometric Features | ✅                  | ❌              |
| Building Features  | ✅                  | ❌              |
| QGIS Compatible    | ⚠️ (newer versions) | ✅              |
| CloudCompare       | ✅                  | ✅              |
| Python (laspy)     | ✅                  | ✅              |

## Recommendations

### For ML Workflows

✅ **Use augmented LAZ** (default)

- Preserves all computed features
- Better for training models
- Full feature set available

### For QGIS Visualization

Choose one option:

1. **Manual conversion** (recommended):

   ```bash
   ign-lidar-qgis enriched/tile.laz
   ```

2. **Auto-convert during enrichment**:

   ```bash
   ign-lidar enrich --input-dir raw/ --output enriched/ --auto-convert-qgis
   ```

3. **Use modern QGIS** (v3.28+):
   - Recent QGIS versions can open LAZ 1.4 files directly
   - May not display all extra dimensions in the UI

### For CloudCompare

✅ **Use augmented LAZ** (default)

- CloudCompare supports LAZ 1.4 with all extra dimensions
- Best visualization of all features

## Migration Guide

If you have existing workflows that expect automatic QGIS conversion:

### Option 1: Update Your Workflow (Recommended)

```bash
# Before
ign-lidar enrich --input-dir raw/ --output enriched/

# After - explicitly request QGIS conversion
ign-lidar enrich --input-dir raw/ --output enriched/ --auto-convert-qgis
```

### Option 2: Change Configuration

Edit `ign_lidar/config.py`:

```python
AUTO_CONVERT_TO_QGIS = True  # Enable automatic conversion
```

### Option 3: Manual Batch Conversion

Keep default behavior and convert as needed:

```bash
# Enrich files
ign-lidar enrich --input-dir raw/ --output enriched/

# Later, convert for QGIS
python scripts/batch_convert_qgis.py enriched/
```

## Benefits

1. **Better Default**: Most users want full feature preservation
2. **Explicit Control**: QGIS conversion is now opt-in
3. **Disk Space**: Avoid creating duplicate files unless needed
4. **Performance**: Skip conversion step when not needed
5. **Flexibility**: Easy to convert later if requirements change

## See Also

- [QGIS Troubleshooting Guide](QGIS_TROUBLESHOOTING.md)
- [QGIS Compatibility Fix](QGIS_COMPATIBILITY_FIX.md)
- [Quick Start QGIS](../QUICK_START_QGIS.md)
