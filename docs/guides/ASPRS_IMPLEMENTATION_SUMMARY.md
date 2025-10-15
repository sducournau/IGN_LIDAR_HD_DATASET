# ASPRS LAS 1.4 Classification Implementation Summary

## Overview

Implemented comprehensive ASPRS LAS 1.4 classification system with extended codes for French topographic data (IGN BD TOPO®).

## Changes Made

### 1. New Module: `ign_lidar/asprs_classes.py`

**Purpose**: Complete implementation of ASPRS LAS 1.4 classification specification with French extensions.

**Features**:

- `ASPRSClass` enum with all standard codes (0-31) and extended codes (32-255)
- Classification mode support: `asprs_standard`, `asprs_extended`, `lod2`, `lod3`
- Helper functions for BD TOPO® nature attribute mapping:
  - `get_classification_for_building(nature, mode)` - Building classification
  - `get_classification_for_road(nature, mode)` - Road type classification
  - `get_classification_for_vegetation(nature, height, mode)` - Vegetation classification
  - `get_classification_for_water(nature, mode)` - Water body classification
  - `get_class_name(code)` - Human-readable class names
  - `get_class_color(code)` - Visualization colors

**Standard ASPRS Classifications** (0-31):

- Ground, vegetation (low/medium/high), buildings, water, roads, rails
- Infrastructure: bridges, transmission towers, wires
- Noise: low points, high noise, temporal exclusion

**Extended Classifications** (32-255):

- **Roads (32-49)**: Motorways, primary/secondary roads, cycleways, parking, bridges, tunnels
- **Buildings (50-69)**: Residential, commercial, industrial, religious, public, agricultural, sports, historic
- **Building Elements**: Roofs, walls, facades, chimneys, balconies
- **Vegetation (70-79)**: Trees, bushes, grass, hedges, forests, vineyards, orchards
- **Water (80-89)**: Rivers, lakes, ponds, canals, fountains, swimming pools
- **Infrastructure (90-109)**: Railways, power lines, antennas, street lights, traffic signs, fences
- **Urban Furniture (110-119)**: Benches, bins, shelters, bollards, barriers
- **Terrain (120-129)**: Bare terrain, gravel, sand, rock, cliffs, quarries
- **Vehicles (130-139)**: Cars, trucks, buses, trains, boats, aircraft

### 2. Updated: `ign_lidar/__init__.py`

**Added exports**:

```python
from .asprs_classes import (
    ASPRSClass,
    ASPRS_CLASS_NAMES,
    ClassificationMode,
    get_classification_for_building,
    get_classification_for_road,
    get_classification_for_vegetation,
    get_classification_for_water,
    get_class_name,
    get_class_color,
)
```

### 3. Updated: `ign_lidar/core/processor.py`

**Fixed RGB preservation issue**:

- Added `'las'` and `'header'` to `original_data` dictionary
- Ensures RGB/NIR data from enriched tiles is properly passed to LAZ save function
- RGB data now correctly written to output LAZ files in point format 7 (RGB) or 8 (RGB+NIR)

**Before**:

```python
original_data = {
    'points': points,
    'intensity': intensity,
    'return_number': return_number,
    'classification': classification,
    'input_rgb': input_rgb,
    'input_nir': input_nir,
    # Missing: 'las' and 'header'
}
```

**After**:

```python
original_data = {
    'points': points,
    'intensity': intensity,
    'return_number': return_number,
    'classification': classification,
    'input_rgb': input_rgb,
    'input_nir': input_nir,
    'input_ndvi': input_ndvi,
    'enriched_features': enriched_features,
    'las': tile_data.get('las'),  # ✅ Added
    'header': tile_data.get('header')  # ✅ Added
}
```

### 4. Updated: `ign_lidar/configs/experiment/classify_enriched_tiles.yaml`

**Added classification mode configuration**:

```yaml
ground_truth:
  # Classification mode - ASPRS LAS 1.4 standard
  classification_mode: asprs_extended # Options: asprs_standard, asprs_extended, lod2, lod3

  # Preserve BD TOPO® nature attributes for detailed classification
  preserve_building_nature: true
  preserve_road_nature: true
  preserve_water_nature: true

  # Additional features
  fetch_railways: true # Fetch railway tracks
```

### 5. New Documentation: `ASPRS_CLASSIFICATION_GUIDE.md`

**Comprehensive guide including**:

- Complete ASPRS LAS 1.4 classification tables
- Extended classification codes for French topography
- Configuration examples
- Python API usage examples
- Best practices and recommendations

## Key Features

### Classification Modes

1. **ASPRS Standard** (`asprs_standard`)

   - Uses only codes 0-31 (ASPRS reserved)
   - Maximum compatibility with other LiDAR software
   - Example: All buildings → 6, all roads → 11

2. **ASPRS Extended** (`asprs_extended`)

   - Uses codes 0-255
   - Detailed French infrastructure classification
   - Example: Residential building → 50, Motorway → 32

3. **LOD2** (`lod2`)

   - 15 building-focused classes
   - Training schema for LOD2 reconstruction
   - Includes walls, roof types, context

4. **LOD3** (`lod3`)
   - 30 detailed building classes
   - Training schema for LOD3 reconstruction
   - Includes windows, doors, architectural details

### BD TOPO® Integration

Automatic mapping of BD TOPO® `nature` attributes to ASPRS codes:

**Buildings**:

- "Résidentiel" → 50 (Residential Building)
- "Commercial" → 51 (Commercial Building)
- "Religieux" → 53 (Religious Building)
- etc.

**Roads**:

- "Autoroute" → 32 (Motorway)
- "Route à 2 chaussées" → 33 (Primary Road)
- "Piste cyclable" → 39 (Cycleway)
- etc.

**Vegetation**:

- "Forêt" → 74 (Forest)
- "Vigne" → 75 (Vineyard)
- "Verger" → 76 (Orchard)
- etc.

**Water**:

- "Cours d'eau" → 80 (River)
- "Lac" → 81 (Lake)
- "Canal" → 83 (Canal)
- etc.

## Usage

### Command Line

```bash
# Process with ASPRS extended classification
ign-lidar-hd process experiment=classify_enriched_tiles

# The config file specifies:
# ground_truth.classification_mode: asprs_extended
```

### Python API

```python
from ign_lidar import (
    ASPRSClass,
    ClassificationMode,
    get_classification_for_building,
    get_classification_for_road,
)

# Get classification codes
building_code = get_classification_for_building(
    nature="Résidentiel",
    mode=ClassificationMode.ASPRS_EXTENDED
)  # Returns 50

road_code = get_classification_for_road(
    nature="Autoroute",
    mode=ClassificationMode.ASPRS_EXTENDED
)  # Returns 32

# Standard mode for compatibility
building_std = get_classification_for_building(
    nature="Résidentiel",
    mode=ClassificationMode.ASPRS_STANDARD
)  # Returns 6 (standard BUILDING code)
```

## Benefits

1. **ASPRS LAS 1.4 Compliance**: Follows official specification for maximum compatibility
2. **French Topography Support**: Extended codes tailored for IGN BD TOPO® data
3. **Flexible Classification**: Choose between standard, extended, or training modes
4. **Automatic Mapping**: BD TOPO® nature attributes → ASPRS codes
5. **RGB/NIR Preservation**: Fixed bug ensuring spectral data is preserved in output
6. **Detailed Infrastructure**: Roads, buildings, water bodies with type classification
7. **Comprehensive Documentation**: Complete guide with examples and best practices

## Files Modified

1. ✅ `ign_lidar/asprs_classes.py` - NEW (528 lines)
2. ✅ `ign_lidar/__init__.py` - Updated exports
3. ✅ `ign_lidar/core/processor.py` - Fixed RGB preservation bug
4. ✅ `ign_lidar/configs/experiment/classify_enriched_tiles.yaml` - Added classification_mode
5. ✅ `ASPRS_CLASSIFICATION_GUIDE.md` - NEW comprehensive guide (300+ lines)

## Testing Recommendations

1. **Verify RGB preservation**:

   ```python
   import laspy
   las = laspy.read("enriched_output.laz")
   print(f"Point format: {las.header.point_format}")  # Should be 7 or 8
   print(f"Has RGB: {hasattr(las, 'red')}")  # Should be True
   ```

2. **Check classification distribution**:

   ```python
   import numpy as np
   from ign_lidar import get_class_name

   unique_classes, counts = np.unique(las.classification, return_counts=True)
   for cls, count in zip(unique_classes, counts):
       print(f"{cls:3d} - {get_class_name(cls):30s}: {count:10,} points")
   ```

3. **Validate ASPRS compliance**:
   - Check that codes are within valid range (0-255)
   - Verify standard codes (0-31) match ASPRS specification
   - Confirm extended codes (32-255) are properly mapped

## Next Steps

To use the new ASPRS classification system:

1. Run processing with updated config:

   ```bash
   ign-lidar-hd process experiment=classify_enriched_tiles
   ```

2. Verify output LAZ files have:

   - Correct ASPRS classification codes
   - Preserved RGB/NIR data (point format 7 or 8)
   - Extra dimensions with computed features

3. Review classification distribution to ensure expected results

4. Adjust `classification_mode` and nature preservation settings as needed

## References

- ASPRS LAS Specification 1.4 - R15
- IGN BD TOPO® Documentation
- IGN LiDAR HD Dataset Documentation

## Version

- Implementation date: October 15, 2025
- Library version: 2.5.1+
- Author: Simon Ducournau (imagodata)
