---
sidebar_position: 2
title: Format Preferences
description: Configure output formats for LAZ files and QGIS compatibility
keywords: [format, laz, qgis, configuration, preferences]
---

features/format-preferences.md

# Sortie Format Preferences

Configure how enriched LAZ files are saved to balance feature completeness with software compatibility.

## Vue d'ensemble

The library supports two output format strategies:

1. **Augmented LAZ (LAZ 1.4)** - Full feature preservation (default)
2. **QGIS-Compatible LAZ (LAZ 1.2)** - Maximum compatibility (optional)

## Default Behavior

By default, the library preserves all geometric features using LAZ 1.4 format:

```bash
# Default: Creates augmented LAZ with all features
ign-lidar-hd enrich \
  --input-dir raw_tiles/ \
  --output enriched_tiles/ \
  --mode full
```

**Output**: `enriched_tiles/tile.laz` (LAZ 1.4 with 30+ features)

## Configuration Options

### Prefer Augmented LAZ (Default)

**Setting**: `PREFER_AUGMENTED_LAZ = True`

- **Format**: LAZ 1.4
- **Features**: All 30+ geometric attributes preserved
- **Compatibility**: Modern LiDAR software (CloudCompare, FME, etc.)
- **File Size**: Slightly larger due to extended attributes

**Benefits**:

- Complete feature set available
- No data loss during processing
- Future-proof format

### QGIS Compatibility Mode

**Setting**: `AUTO_CONVERT_TO_QGIS = False` (manual conversion)

For QGIS compatibility, use the conversion utility:

```bash
# Method 1: Convert after enrichment
ign-lidar-hd enrich --input-dir raw/ --output enriched/
python scripts/batch_convert_qgis.py enriched/

# Method 2: Use QGIS converter directly
python -m ign_lidar.qgis_converter enriched/tile.laz
```

**Output**:

- `enriched/tile.laz` (LAZ 1.4, full features)
- `enriched/tile_qgis.laz` (LAZ 1.2, QGIS-compatible)

## Format Comparison

| Feature            | Augmented LAZ   | QGIS LAZ            |
| ------------------ | --------------- | ------------------- |
| Format Version     | LAZ 1.4         | LAZ 1.2             |
| Geometric Features | 30+ attributes  | Selected attributes |
| File Size          | ~310MB          | ~305MB              |
| QGIS Compatibility | Modern versions | All versions        |
| CloudCompare       | ✅ Full support | ✅ Full support     |
| FME                | ✅ Full support | ✅ Full support     |
| Custom Analysis    | ✅ All features | ⚠️ Limited features |

## Available Features by Format

### Augmented LAZ (Full Feature Set)

**Geometric Features**:

- `normal_x`, `normal_y`, `normal_z` - Surface normals
- `curvature` - Surface curvature
- `planarity`, `linearity`, `sphericity` - Shape descriptors
- `roughness` - Surface texture
- `anisotropy` - Directional variation

**Building-Specific Features**:

- `verticality` - Wall orientation score
- `wall_score`, `roof_score` - Component probabilities
- `height_above_ground` - Normalized elevation
- `density` - Local point density
- `vertical_std` - Height variation

**Advanced Features**:

- `eigenvalue_1`, `eigenvalue_2`, `eigenvalue_3` - Principal components
- `num_points_2m` - Neighborhood size
- `neighborhood_extent` - Spatial extent
- `height_extent_ratio` - Geometric ratios

### QGIS LAZ (Compatible Subset)

**Core Features** (guaranteed compatibility):

- `normal_x`, `normal_y`, `normal_z`
- `curvature`
- `planarity`, `linearity`, `sphericity`
- `height_above_ground`
- `wall_score`, `roof_score`
- `verticality`

**Excluded Features** (compatibility limitations):

- Advanced eigenvalue decomposition
- Complex neighborhood statistics
- Extended geometric ratios

## When to Use Each Format

### Use Augmented LAZ When:

- **Research applications** requiring full feature set
- **Custom analysis** with machine learning models
- **Advanced visualization** in CloudCompare or specialized tools
- **Future processing** where feature completeness matters
- **Archival storage** for long-term data preservation

### Use QGIS LAZ When:

- **GIS workflows** primarily using QGIS
- **Visualization** focused on basic geometric properties
- **Sharing data** with users who have older software
- **Web mapping** applications with format constraints
- **Teaching/demos** where compatibility is critical

## Configuration Examples

### Research/Development Workflow

```bash
# Full feature extraction for ML research
ign-lidar-hd enrich \
  --input-dir raw_tiles/ \
  --output research_data/ \
  --mode full

# Keep all 30+ features in LAZ 1.4 format
# Use for: scikit-learn, PyTorch, custom analysis
```

### GIS/Visualization Workflow

```bash
# Basique enrichment
ign-lidar-hd enrich \
  --input-dir raw_tiles/ \
  --output enriched_tiles/ \
  --mode full

# Convert for QGIS
python scripts/batch_convert_qgis.py enriched_tiles/

# Result: Both formats available
ls enriched_tiles/
# tile.laz          (full features)
# tile_qgis.laz     (QGIS-compatible)
```

### Production Workflow

```bash
# Process with smart skip detection
ign-lidar-hd enrich \
  --input-dir large_dataset/ \
  --output processed/ \
  --mode full

# Selectively convert tiles needing QGIS compatibility
python -m ign_lidar.qgis_converter processed/priority_tiles/
```

## Programmatic Configuration

### Python API

```python
from ign_lidar import LiDARProcessor, config

# Option 1: Use configuration
config.PREFER_AUGMENTED_LAZ = True
config.AUTO_CONVERT_TO_QGIS = False

processor = LiDARProcessor()
processor.enrich_file('input.laz', 'output/')

# Option 2: Direct format control
processor = LiDARProcessor(
    output_format='augmented'  # or 'qgis'
)
```

### Configuration File

Create `config/local_settings.py`:

```python
# Format preferences
PREFER_AUGMENTED_LAZ = True
AUTO_CONVERT_TO_QGIS = False

# Traitement preferences
DEFAULT_MODE = 'full'
DEFAULT_WORKERS = 4

# Sortie preferences
PRESERVE_DIRECTORY_STRUCTURE = True
COPY_METADATA_FILES = True
```

## Fichier Naming Conventions

### Standard Output

```
enriched_tiles/
├── LIDARHD_FXX_0123_4567_LA93_IGN69_2020.laz    # Augmented LAZ
└── metadata/
    └── LIDARHD_FXX_0123_4567_LA93_IGN69_2020.json
```

### With QGIS Conversion

```
enriched_tiles/
├── LIDARHD_FXX_0123_4567_LA93_IGN69_2020.laz         # Augmented LAZ
├── LIDARHD_FXX_0123_4567_LA93_IGN69_2020_qgis.laz    # QGIS-compatible
└── metadata/
    └── LIDARHD_FXX_0123_4567_LA93_IGN69_2020.json
```

## Quality Verification

### Check Format Version

```python
import laspy

# Check file format
las = laspy.read('enriched_tile.laz')
print(f"LAZ version: {las.header.version}")
print(f"Point format: {las.point_format.id}")
print(f"Extra dimensions: {las.point_format.extra_dimension_names}")
```

### Validate QGIS Compatibility

```bash
# Test QGIS compatibility
python scripts/validation/test_qgis_compatibility.py enriched_tile_qgis.laz

# Expected output:
# ✅ LAZ 1.2 format detected
# ✅ Compatible point format (ID 6)
# ✅ Standard dimensions present
# ✅ Compatible extra dimensions: 8 found
# ✅ File should load properly in QGIS
```

## Dépannage

### QGIS Won't Load File

**Issue**: "Unsupported file format" in QGIS

**Solution**: Convert to QGIS format

```bash
python -m ign_lidar.qgis_converter problematic_file.laz
# Creates: problematic_file_qgis.laz
```

### Missing Features in Analysis

**Issue**: Can't find expected attributes

**Cause**: Using QGIS-compatible format instead of full format

**Solution**: Use augmented LAZ for analysis

```python
# Load full feature set
las = laspy.read('tile.laz')  # Not tile_qgis.laz
features = las.point_format.extra_dimension_names
print(f"Available features: {len(features)}")
```

### Large File Sizes

**Issue**: Files larger than expected

**Cause**: Full feature set increases file size

**Solution**:

1. Use QGIS format for storage-constrained applications
2. Compress with higher LAZ compression levels
3. Filter to essential features only

## Performance Considerations

### Traitement Speed

- **Augmented LAZ**: Faster (no conversion overhead)
- **QGIS LAZ**: Slower (requires attribute filtering/conversion)

### Storage Requirements

- **Augmented LAZ**: ~5-10% larger files
- **QGIS LAZ**: Standard LAZ file sizes
- **Both formats**: 2x storage if keeping both

### Memory Usage

- **Augmented LAZ**: Higher memory during processing
- **QGIS LAZ**: Lower memory (fewer attributes)

## Voir Aussi

- [QGIS Integration Guide](../guides/qgis-integration.md) - Using files in QGIS
- [Smart Skip Features](smart-skip.md) - Avoid reprocessing files
- [CLI Commands](../guides/cli-commands.md) - Command-line options
