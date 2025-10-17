---
sidebar_position: 1
title: ASPRS Classification Reference
description: Complete ASPRS LAS 1.4 classification codes and BD TOPO® extensions
---

# ASPRS Classification Reference

Comprehensive guide to ASPRS LAS 1.4 classification codes and IGN-specific extensions for French topographic data.

---

## 🎯 Overview

This library implements the **ASPRS LAS Specification 1.4** classification system with extensions for BD TOPO® integration, providing:

- **Standard ASPRS classes** (0-31) - Official LAS 1.4 specification
- **Extended classes** (32-255) - Custom classes for BD TOPO® features
- **IGN-specific fixes** - Handle non-standard classes (Class 67)
- **Automatic remapping** - Convert between classification schemes

**Reference**: [ASPRS LAS 1.4 Specification R15](https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf)

---

## 📊 Standard ASPRS Classes (0-31)

### Core Classification Codes

| Code   | Name                      | Description                           | Typical Use         |
| ------ | ------------------------- | ------------------------------------- | ------------------- |
| **0**  | Created, Never Classified | Points with no classification applied | Initial state       |
| **1**  | Unclassified              | Points not yet classified             | Default for unknown |
| **2**  | Ground                    | Bare earth surface                    | DTM generation      |
| **3**  | Low Vegetation            | Vegetation < 0.5m height              | Grass, low shrubs   |
| **4**  | Medium Vegetation         | Vegetation 0.5-2.0m                   | Bushes, hedges      |
| **5**  | High Vegetation           | Vegetation > 2.0m                     | Trees, forests      |
| **6**  | Building                  | Building structures                   | Roofs, walls        |
| **7**  | Low Point (Noise)         | Noise or outliers                     | Below ground        |
| **8**  | Reserved                  | Model Key-point (deprecated)          | -                   |
| **9**  | Water                     | Water surfaces                        | Rivers, lakes, sea  |
| **10** | Rail                      | Railway tracks and infrastructure     | Train lines         |
| **11** | Road Surface              | Road surfaces                         | Streets, highways   |
| **12** | Reserved                  | Wire - Guard (deprecated)             | -                   |
| **13** | Wire - Guard              | Shield wire                           | Power lines         |
| **14** | Wire - Conductor          | Phase wire                            | Power lines         |
| **15** | Transmission Tower        | Power transmission structures         | Pylons              |
| **16** | Wire-structure Connector  | Insulators                            | Power lines         |
| **17** | Bridge Deck               | Bridge surfaces                       | Elevated structures |
| **18** | High Noise                | Noise or outliers                     | Above expected      |
| **19** | Overhead Structure        | Elevated non-bridge structures        | Canopies            |
| **20** | Ignored Ground            | Breakline proximity                   | DTM exclusion       |
| **21** | Snow                      | Snow cover                            | Seasonal            |
| **22** | Temporal Exclusion        | Temporary features                    | Time-specific       |

### Reserved Codes

- **23-31**: Reserved for future ASPRS use

---

## 🇫🇷 BD TOPO® Extended Classes (32-255)

### Road Infrastructure (32-49)

| Code   | Name           | BD TOPO® Attribute          | Description         |
| ------ | -------------- | --------------------------- | ------------------- |
| **32** | Motorway       | nature='Autoroute'          | Highway/motorway    |
| **33** | Primary Road   | nature='Départementale'     | Main roads          |
| **34** | Secondary Road | nature='Route à 1 chaussée' | Secondary roads     |
| **35** | Tertiary Road  | nature='Route empierrée'    | Minor roads         |
| **36** | Residential    | nature='Chemin'             | Residential streets |
| **37** | Service Road   | pos_sol='0'                 | Service roads       |
| **38** | Pedestrian     | nature='Sentier'            | Pedestrian paths    |
| **39** | Cycleway       | nature='Piste cyclable'     | Bike paths          |
| **40** | Parking        | nature='Parking'            | Parking areas       |
| **41** | Road Bridge    | pos_sol='1'                 | Elevated roads      |
| **42** | Road Tunnel    | pos_sol='-1'                | Underground roads   |
| **43** | Roundabout     | nature='Rond-point'         | Traffic circles     |

### Building Types (50-69)

| Code   | Name                  | BD TOPO® Attribute     | Description           |
| ------ | --------------------- | ---------------------- | --------------------- |
| **50** | Residential Building  | nature='Indifférencié' | Houses, apartments    |
| **51** | Commercial Building   | nature='Commercial'    | Shops, offices        |
| **52** | Industrial Building   | nature='Industriel'    | Factories, warehouses |
| **53** | Religious Building    | nature='Religieux'     | Churches, temples     |
| **54** | Public Building       | nature='Administratif' | Government buildings  |
| **55** | Agricultural Building | nature='Agricole'      | Barns, silos          |
| **56** | Sports Building       | nature='Sportif'       | Gyms, stadiums        |
| **57** | Historic Building     | nature='Remarquable'   | Monuments, heritage   |
| **58** | Building Roof         | -                      | Roof surfaces         |
| **59** | Building Wall         | -                      | Wall surfaces         |
| **60** | Building Facade       | -                      | Facade elements       |
| **61** | Chimney               | -                      | Chimneys              |
| **62** | Balcony               | -                      | Balconies, terraces   |

### Vegetation Types (70-79)

| Code   | Name     | BD TOPO® Layer     | Description      |
| ------ | -------- | ------------------ | ---------------- |
| **70** | Tree     | zone_de_vegetation | Individual trees |
| **71** | Bush     | zone_de_vegetation | Shrubs, bushes   |
| **72** | Grass    | zone_de_vegetation | Grassland        |
| **73** | Hedge    | zone_de_vegetation | Hedgerows        |
| **74** | Forest   | zone_de_vegetation | Forest areas     |
| **75** | Vineyard | RPG                | Vineyards        |
| **76** | Orchard  | RPG                | Orchards         |

### Water Features (80-89)

| Code   | Name          | BD TOPO® Layer         | Description |
| ------ | ------------- | ---------------------- | ----------- |
| **80** | River         | surface_hydrographique | Rivers      |
| **81** | Lake          | surface_hydrographique | Lakes       |
| **82** | Pond          | surface_hydrographique | Ponds       |
| **83** | Canal         | surface_hydrographique | Canals      |
| **84** | Reservoir     | reservoir              | Water tanks |
| **85** | Swimming Pool | -                      | Pools       |

### Special Features (90-99)

| Code   | Name            | BD TOPO® Layer   | Description          |
| ------ | --------------- | ---------------- | -------------------- |
| **90** | Cemetery        | cimetiere        | Cemeteries           |
| **91** | Sports Facility | terrain_de_sport | Sports grounds       |
| **92** | Power Line      | ligne_electrique | Power line corridors |
| **93** | Wind Turbine    | -                | Wind turbines        |
| **94** | Solar Panel     | -                | Solar installations  |

---

## ⚠️ IGN-Specific Issues

### Class 67 Fix

**Problem**: Class 67 appears in some IGN LiDAR HD tiles but is **not part of the ASPRS specification**.

**Solution**: Automatically remapped to Class 1 (Unclassified) during preprocessing.

```yaml
preprocess:
  normalize_classification: true # Enable automatic fix
  strict_class_normalization: false # Only fix known issues
```

**Impact Example**:

```
Tile: LHD_FXX_0650_6860.laz
- Class 67 points: 4,380 (0.02%)
- Remapped to: Class 1 (Unclassified)
- Status: ✅ Fixed automatically
```

### Known IGN Classes

| Original Class | Remapped To      | Status     | Notes                  |
| -------------- | ---------------- | ---------- | ---------------------- |
| 67             | 1 (Unclassified) | ✅ Handled | Non-standard IGN class |
| 64             | 1 (Unclassified) | ✅ Handled | Rare, non-standard     |
| 65             | 1 (Unclassified) | ✅ Handled | Rare, non-standard     |

---

## 🔄 Classification Mappings

### BD TOPO® → ASPRS Mapping

Automatic mapping from BD TOPO® features to ASPRS classes:

```python
BD_TOPO_TO_ASPRS = {
    # Buildings
    'batiment': 6,                    # ASPRS Building

    # Roads
    'troncon_de_route': 11,          # ASPRS Road Surface

    # Railways
    'troncon_de_voie_ferree': 10,    # ASPRS Rail

    # Water
    'surface_hydrographique': 9,      # ASPRS Water
    'reservoir': 9,                   # ASPRS Water

    # Vegetation
    'zone_de_vegetation': 5,          # ASPRS High Vegetation

    # Infrastructure
    'ligne_electrique': 14,           # ASPRS Wire - Conductor
    'terrain_de_sport': 6,            # ASPRS Building
    'cimetiere': 6,                   # ASPRS Building
}
```

### ASPRS → LOD2 Mapping

Conversion from ASPRS to LOD2 building-focused classes:

```python
ASPRS_TO_LOD2 = {
    0: 14,   # Never classified → other
    1: 14,   # Unclassified → other
    2: 9,    # Ground → ground
    3: 10,   # Low Vegetation → vegetation_low
    4: 10,   # Medium Vegetation → vegetation_low
    5: 11,   # High Vegetation → vegetation_high
    6: 0,    # Building → wall (requires refinement)
    7: 10,   # Low Point → vegetation_low
    9: 12,   # Water → water
    10: 14,  # Rail → other
    11: 14,  # Road Surface → other
    67: 14,  # Unknown → other
}
```

### ASPRS → LOD3 Mapping

Conversion from ASPRS to LOD3 detailed building classes:

```python
ASPRS_TO_LOD3 = {
    0: 29,   # Never classified → other
    1: 29,   # Unclassified → other
    2: 23,   # Ground → ground
    3: 24,   # Low Vegetation → vegetation_low
    4: 24,   # Medium Vegetation → vegetation_low
    5: 25,   # High Vegetation → vegetation_high
    6: 0,    # Building → wall_plain (requires refinement)
    7: 24,   # Low Point → vegetation_low
    9: 26,   # Water → water
    10: 29,  # Rail → other
    11: 23,  # Road Surface → ground
    67: 29,  # Unknown → other
}
```

---

## ⚙️ Configuration

### Enable ASPRS Classification

```yaml
# config.yaml (V5)
defaults:
  - base/processor
  - base/features
  - base/data_sources
  - base/output
  - base/monitoring
  - _self_

# Classification mode
classification:
  mode: asprs # Use ASPRS classification

  # Class normalization
  normalize_classes: true
  strict_normalization: false # Only fix known issues

  # Extended classes
  use_extended_classes: true # Enable BD TOPO® extensions (32-255)

  # Class-specific parameters
  planarity_ground: 0.95
  planarity_building: 0.85
  planarity_vegetation: 0.5
  planarity_road: 0.75

data_sources:
  bd_topo:
    enabled: true
    features:
      - buildings # → Class 6
      - roads # → Class 11
      - railways # → Class 10
      - water # → Class 9
      - vegetation # → Class 3-5 (by height)
      - power_lines # → Class 14
```

### Preset Configuration

```bash
# Use ASPRS classification preset
ign-lidar-hd process \
  --config-name asprs_classification \
  input_dir=data/tiles/ \
  output_dir=output/asprs/
```

---

## 💻 Python API

### Check Classification Codes

```python
from ign_lidar.asprs_classes import ASPRSClass, ASPRS_CLASS_NAMES
import laspy

# Load tile
las = laspy.read("tile.laz")
classes = las.classification

# Get unique classes
unique_classes = np.unique(classes)
print("Classes found:")
for cls in unique_classes:
    name = ASPRS_CLASS_NAMES.get(cls, "Unknown")
    count = np.sum(classes == cls)
    percent = 100 * count / len(classes)
    print(f"  {cls:3d}: {name:25s} - {count:8d} pts ({percent:5.2f}%)")
```

### Remap Non-Standard Classes

```python
from ign_lidar.asprs_classes import remap_non_standard_classes

# Fix Class 67 and other non-standard classes
classes_fixed = remap_non_standard_classes(classes)

# Verify fix
assert 67 not in classes_fixed
print(f"Remapped {np.sum(classes == 67)} Class 67 points to Class 1")
```

### Apply BD TOPO® Classification

```python
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.asprs_classes import BD_TOPO_TO_ASPRS

# Fetch BD TOPO® data
fetcher = IGNGroundTruthFetcher()
ground_truth = fetcher.fetch_all_features(
    bbox=tile_bbox,
    include_buildings=True,
    include_roads=True,
    include_water=True
)

# Classify points with ASPRS codes
labels = fetcher.label_points_with_ground_truth(
    points=points,
    ground_truth_features=ground_truth,
    class_mapping=BD_TOPO_TO_ASPRS  # Use ASPRS codes
)

# Verify ASPRS compliance
assert all(0 <= cls <= 255 for cls in labels)
print(f"All {len(labels)} points have valid ASPRS classes")
```

### Export Classification Report

```python
from ign_lidar.asprs_classes import generate_classification_report

# Generate detailed report
report = generate_classification_report(
    las_file="tile.laz",
    output_format="markdown"
)

print(report)
```

**Output**:

```markdown
# Classification Report: tile.laz

## Summary

- Total points: 18,234,567
- Unique classes: 7
- ASPRS compliant: ✅ Yes

## Class Distribution

| Code | Name              | Count     | Percentage |
| ---- | ----------------- | --------- | ---------- |
| 1    | Unclassified      | 668,890   | 3.67%      |
| 2    | Ground            | 7,982,345 | 43.79%     |
| 3    | Low Vegetation    | 237,049   | 1.30%      |
| 4    | Medium Vegetation | 506,921   | 2.78%      |
| 5    | High Vegetation   | 6,784,123 | 37.21%     |
| 6    | Building          | 2,047,207 | 11.23%     |
| 9    | Water             | 8,032     | 0.04%      |

## Issues

- ⚠️ No Class 67 (fixed during preprocessing)
```

---

## 📊 Class Statistics

### Typical Distribution in IGN LiDAR HD

Based on analysis of French urban and rural tiles:

| Class        | Urban Areas | Rural Areas | Forest Areas |
| ------------ | ----------- | ----------- | ------------ |
| Ground (2)   | 35-45%      | 45-55%      | 15-25%       |
| Low Veg (3)  | 1-3%        | 3-5%        | 1-2%         |
| Med Veg (4)  | 2-4%        | 4-8%        | 2-4%         |
| High Veg (5) | 15-25%      | 25-40%      | 60-75%       |
| Building (6) | 15-25%      | 5-10%       | 1-3%         |
| Road (11)    | 5-10%       | 2-5%        | 0.5-1%       |
| Water (9)    | 0-2%        | 0-5%        | 0-1%         |

### Quality Indicators

Good classification quality:

- Ground (2): > 30%
- High Vegetation (5): 10-60% (depends on area)
- Building (6): 5-30% (urban areas)
- Unclassified (1): < 5%
- Noise (7, 18): < 1%

---

## 🔍 Validation

### Verify ASPRS Compliance

```python
from ign_lidar.asprs_classes import validate_asprs_compliance
import laspy

# Load and validate
las = laspy.read("tile.laz")
is_valid, issues = validate_asprs_compliance(las)

if is_valid:
    print("✅ ASPRS LAS 1.4 compliant")
else:
    print("❌ Issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

### Check for Non-Standard Classes

```bash
# CLI validation
ign-lidar-hd verify \
  tile.laz \
  --check-asprs-compliance \
  --report-non-standard-classes
```

---

## 📚 See Also

- [LOD Classification Reference](./lod-classification.md)
- [BD TOPO Integration](./bd-topo-integration.md)
- [Ground Truth Classification](../features/ground-truth-classification.md)
- [Classification Workflow](./classification-workflow.md)

---

## 💡 Best Practices

1. **Always normalize classes** - Enable `normalize_classification` to fix Class 67
2. **Validate before processing** - Check for non-standard classes
3. **Use extended classes wisely** - Only when BD TOPO® data is available
4. **Document custom classes** - If adding new codes beyond 99
5. **Respect ASPRS spec** - Don't overwrite reserved codes (23-31)
6. **Test classification** - Verify output with QGIS or CloudCompare
7. **Export metadata** - Include class descriptions in output files

---

**Standard**: ASPRS LAS 1.4 R15  
**Updated**: October 17, 2025 - V5 Configuration
