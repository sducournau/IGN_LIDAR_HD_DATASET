---
sidebar_position: 8
title: Classification Taxonomy
description: Understanding LOD2 and LOD3 class schemas and ASPRS mapping
---

# Classification Taxonomy

## Overview

This library uses a **building-focused classification schema** that differs from the standard ASPRS classification. Point cloud classifications are automatically remapped from ASPRS codes to LOD2 or LOD3 taxonomies designed for detailed architectural analysis.

:::warning Important
When you see buildings classified as **Class 0**, this is **CORRECT**! The LOD2 schema maps buildings to class 0 (`wall`) as the base category for detailed building analysis.
:::

## LOD2 Classification Schema (15 Classes)

LOD2 is designed for building structure analysis with basic architectural elements.

### Building Elements

| Class ID | Class Name   | Description                       | ASPRS Source |
| -------- | ------------ | --------------------------------- | ------------ |
| **0**    | `wall`       | Building walls and main structure | 6 (Building) |
| 1        | `roof_flat`  | Flat roof surfaces                | —            |
| 2        | `roof_gable` | Gable/pitched roof                | —            |
| 3        | `roof_hip`   | Hip roof                          | —            |
| 4        | `chimney`    | Chimney structures                | —            |
| 5        | `dormer`     | Dormer windows                    | —            |
| 6        | `balcony`    | Balconies and terraces            | —            |
| 7        | `overhang`   | Overhanging elements              | —            |
| 8        | `foundation` | Building foundation               | —            |

### Context Elements (Non-Building)

| Class ID | Class Name        | Description              | ASPRS Source          |
| -------- | ----------------- | ------------------------ | --------------------- |
| **9**    | `ground`          | Ground surface, roads    | 2 (Ground), 11 (Road) |
| **10**   | `vegetation_low`  | Low vegetation, shrubs   | 3, 4 (Low/Med Veg)    |
| **11**   | `vegetation_high` | Trees, high vegetation   | 5 (High Vegetation)   |
| **12**   | `water`           | Water bodies             | 9 (Water)             |
| 13       | `vehicle`         | Vehicles, mobile objects | 17 (Bridge Deck)      |
| **14**   | `other`           | Unclassified objects     | 0, 1, 8, 10-16, 18    |

## LOD3 Classification Schema (30 Classes)

LOD3 provides detailed architectural classification including facade elements.

### Structural Elements

| Class ID | Class Name          | Description                 |
| -------- | ------------------- | --------------------------- |
| **0**    | `wall_plain`        | Plain wall without openings |
| 1        | `wall_with_windows` | Wall with window openings   |
| 2        | `wall_with_door`    | Wall with door openings     |
| 8        | `chimney`           | Chimney structures          |
| 19       | `pillar`            | Structural pillars          |
| 20       | `cornice`           | Decorative cornices         |
| 21       | `foundation`        | Building foundation         |

### Roof Elements

| Class ID | Class Name     | Description        |
| -------- | -------------- | ------------------ |
| 3        | `roof_flat`    | Flat roof surfaces |
| 4        | `roof_gable`   | Gable/pitched roof |
| 5        | `roof_hip`     | Hip roof           |
| 6        | `roof_mansard` | Mansard roof       |
| 7        | `roof_gambrel` | Gambrel roof       |
| 9        | `dormer_gable` | Gable dormer       |
| 10       | `dormer_shed`  | Shed dormer        |
| 11       | `skylight`     | Roof skylights     |
| 12       | `roof_edge`    | Roof edges/gutters |

### Openings

| Class ID | Class Name        | Description      |
| -------- | ----------------- | ---------------- |
| 13       | `window`          | Standard windows |
| 14       | `door`            | Entry doors      |
| 15       | `garage_door`     | Garage doors     |
| 22       | `basement_window` | Basement windows |

### Facade Elements

| Class ID | Class Name   | Description          |
| -------- | ------------ | -------------------- |
| 16       | `balcony`    | Balconies            |
| 17       | `balustrade` | Balustrades/railings |
| 18       | `overhang`   | Overhanging elements |

### Context Elements

| Class ID | Class Name         | Description        |
| -------- | ------------------ | ------------------ |
| 23       | `ground`           | Ground surface     |
| 24       | `vegetation_low`   | Low vegetation     |
| 25       | `vegetation_high`  | Trees              |
| 26       | `water`            | Water bodies       |
| 27       | `vehicle`          | Vehicles           |
| 28       | `street_furniture` | Urban furniture    |
| 29       | `other`            | Other/unclassified |

## ASPRS to LOD Mapping

### How It Works

When processing LiDAR files, the pipeline automatically remaps ASPRS standard classifications to LOD schemas:

````python
# Example: ASPRS class 6 (Building) → LOD2 class 0 (wall)
original_classification = 6  # ASPRS Building
lod2_classification = 0      # LOD2 Wall

# Example: ASPRS class 2 (Ground) → LOD2 class 9 (ground)
original_classification = 2  # ASPRS Ground
lod2_classification = 9      # LOD2 Ground
```python
# Output shows LOD2 classifications
````

### ASPRS to LOD2 Mapping Table

| ASPRS Code | ASPRS Name          | LOD2 Class | LOD2 Name           |
| ---------- | ------------------- | ---------- | ------------------- |
| 0          | Never classified    | 14         | other               |
| 1          | Unclassified        | 14         | other               |
| **2**      | **Ground**          | **9**      | **ground**          |
| 3          | Low Vegetation      | 10         | vegetation_low      |
| 4          | Medium Vegetation   | 10         | vegetation_low      |
| **5**      | **High Vegetation** | **11**     | **vegetation_high** |
| **6**      | **Building**        | **0**      | **wall**            |
| 7          | Low Point (noise)   | 10         | vegetation_low      |
| 8          | Model Key-point     | 14         | other               |
| **9**      | **Water**           | **12**     | **water**           |
| 10         | Rail                | 14         | other               |
| 11         | Road Surface        | 14         | other               |
| 17         | Bridge Deck         | 13         | vehicle             |
| 18         | High Noise          | 11         | vegetation_high     |

### ASPRS to LOD3 Mapping Table

| ASPRS Code | ASPRS Name          | LOD3 Class | LOD3 Name           |
| ---------- | ------------------- | ---------- | ------------------- |
| 0          | Never classified    | 29         | other               |
| 1          | Unclassified        | 29         | other               |
| **2**      | **Ground**          | **23**     | **ground**          |
| 3          | Low Vegetation      | 24         | vegetation_low      |
| 4          | Medium Vegetation   | 24         | vegetation_low      |
| **5**      | **High Vegetation** | **25**     | **vegetation_high** |
| **6**      | **Building**        | **0**      | **wall_plain**      |
| 7          | Low Point (noise)   | 24         | vegetation_low      |
| 8          | Model Key-point     | 29         | other               |
| **9**      | **Water**           | **26**     | **water**           |
| 10         | Rail                | 29         | other               |
| 11         | Road Surface        | 23         | ground              |
| 17         | Bridge Deck         | 27         | vehicle             |
| 18         | High Noise          | 25         | vegetation_high     |

## Understanding Your Classifications

### Example Output Analysis

When you inspect an enriched LAZ file, you might see:

```
Classification distribution in the file:
============================================================
Class   0:  2,644,762 points ( 12.30%) → WALLS (building structure)
Class   9:  9,024,921 points ( 41.96%) → GROUND (terrain)
Class  10:    292,956 points (  1.36%) → VEGETATION LOW (shrubs)
Class  11:  7,953,872 points ( 36.98%) → VEGETATION HIGH (trees)
Class  12:        674 points (  0.00%) → WATER
Class  14:  1,591,015 points (  7.40%) → OTHER (unclassified)
```

**This is the expected behavior!** Buildings are classified as Class 0 (`wall`) in the LOD2 schema.

### Why This Approach?

The LOD-focused taxonomy provides several advantages:

1. **Building-Centric Analysis**: Optimized for architectural feature detection
2. **Hierarchical Structure**: Buildings are subdivided into structural components
3. **ML Training**: Better class balance for machine learning models
4. **Semantic Richness**: More meaningful categories for urban analysis

## Checking Classifications

### Using Python

```python
import laspy
import numpy as np
from ign_lidar.classes import LOD2_CLASSES

# Load LAZ file
las = laspy.read("your_file.laz")

# Get classification distribution
unique, counts = np.unique(las.classification, return_counts=True)

# Print with LOD2 names
class_names = {v: k for k, v in LOD2_CLASSES.items()}
for cls, count in zip(unique, counts):
    name = class_names.get(cls, "unknown")
    percentage = (count / len(las.classification)) * 100
    print(f"Class {cls:3d} ({name:20s}): {count:10,} ({percentage:6.2f}%)")
```

### Using CloudCompare

1. Open the enriched LAZ file in CloudCompare
2. Select the point cloud in the DB Tree
3. Go to **Edit > Scalar Fields > Classification**
4. The classification values correspond to LOD2/LOD3 schema, **not ASPRS**

## Configuration Options

### Disabling Remapping (Use ASPRS Classes)

If you need to preserve original ASPRS classifications, modify the configuration:

```yaml
processor:
  lod_level: null # Disable LOD remapping
  preserve_asprs_classes: true
```

### Custom Class Mapping

You can define custom mappings in `ign_lidar/classes.py`:

```python
# Custom mapping example
CUSTOM_MAPPING = {
    6: 100,  # Map buildings to class 100 instead of 0
    2: 101,  # Map ground to class 101 instead of 9
    # ... etc
}
```

## Implementation Details

The remapping occurs in `ign_lidar/core/processor.py`:

```python
# Remap ASPRS classifications to LOD schema
labels_v = np.array([
    self.class_mapping.get(c, self.default_class)
    for c in classification_v
], dtype=np.uint8)
```

The mapping dictionaries are defined in `ign_lidar/classes.py`:

- `LOD2_CLASSES`: LOD2 taxonomy (15 classes)
- `LOD3_CLASSES`: LOD3 taxonomy (30 classes)
- `ASPRS_TO_LOD2`: ASPRS → LOD2 mapping
- `ASPRS_TO_LOD3`: ASPRS → LOD3 mapping

## Related Documentation

- [LOD3 Classification](../features/lod3-classification.md) - Level of Detail concepts
- [Enriched LAZ Output](../features/enriched-laz-only.md) - Understanding output files
- [API Reference](../api/classes.md) - Python API for class definitions

## References

- **ASPRS LAS Specification 1.4**: Standard LiDAR classification codes
- **CityGML LOD Specification**: Level of Detail standards for 3D city models
- **OGC 3D Portrayal Service**: 3D geospatial visualization standards
