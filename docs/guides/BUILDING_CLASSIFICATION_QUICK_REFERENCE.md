# Building Classification with Geometric Attributes - Quick Reference

## 🎯 Quick Start

### Import Functions

```python
from ign_lidar.core.modules.classification_refinement import (
    refine_building_classification,
    classify_lod2_building_elements,
    classify_lod3_building_elements,
    RefinementConfig
)

from ign_lidar.features import (
    compute_horizontality,
    compute_facade_score,
    compute_roof_plane_score,
    compute_opening_likelihood,
    compute_edge_strength,
    compute_structural_element_score
)
```

---

## 📐 Key Geometric Attributes

| Attribute         | Formula                     | Range  | Usage                         |
| ----------------- | --------------------------- | ------ | ----------------------------- |
| **Verticality**   | `1 - \|normal_z\|`          | [0, 1] | Wall detection (1=vertical)   |
| **Horizontality** | `\|normal_z\|`              | [0, 1] | Roof detection (1=horizontal) |
| **Wall Score**    | `planarity × verticality`   | [0, 1] | Combined wall metric          |
| **Roof Score**    | `planarity × horizontality` | [0, 1] | Combined roof metric          |
| **Facade Score**  | Multi-component             | [0, 1] | Facade likelihood             |
| **Edge Strength** | `(λ1 - λ2) / λ1`            | [0, 1] | Building edges/corners        |

---

## 🏗️ Classification Thresholds

### ASPRS Building Detection

```python
config = RefinementConfig()

# Wall detection
config.VERTICALITY_WALL_MIN = 0.7       # Minimum verticality
config.PLANARITY_BUILDING_MIN = 0.5     # Minimum planarity
config.WALL_SCORE_MIN = 0.35            # Wall score threshold

# Roof detection
config.HORIZONTALITY_ROOF_MIN = 0.85    # Minimum horizontality
config.ROOF_PLANARITY_MIN = 0.7         # Minimum planarity
config.ROOF_SCORE_MIN = 0.5             # Roof score threshold

# Structure detection
config.ANISOTROPY_BUILDING_MIN = 0.5    # Organized structure
config.LINEARITY_EDGE_MIN = 0.4         # Edge detection
```

### LOD2 Element Thresholds

| Element         | Verticality | Horizontality | Planarity | Other                 |
| --------------- | ----------- | ------------- | --------- | --------------------- |
| **Wall**        | > 0.70      | -             | > 0.50    | -                     |
| **Flat Roof**   | -           | > 0.95        | > 0.70    | -                     |
| **Sloped Roof** | -           | 0.70-0.95     | > 0.60    | -                     |
| **Chimney**     | > 0.85      | -             | > 0.60    | height > roof + 0.5m  |
| **Dormer**      | > 0.75      | -             | > 0.50    | at_roof_level         |
| **Balcony**     | -           | > 0.85        | > 0.70    | 2m < height < 0.8×max |

### LOD3 Element Thresholds

| Element       | Key Features                 | Thresholds                             |
| ------------- | ---------------------------- | -------------------------------------- |
| **Window**    | Linearity, Low Planarity     | linearity > 0.5, planarity < 0.4       |
| **Door**      | Linearity, Low Height        | linearity > 0.5, height < 3m           |
| **Pillar**    | Verticality, Linearity       | verticality > 0.9, linearity > 0.7     |
| **Roof Edge** | Linearity, At Roof           | linearity > 0.6, at_roof_level         |
| **Cornice**   | Horizontality, High Position | horizontality > 0.6, height > 0.85×max |

---

## 💻 Code Examples

### 1. Enhanced ASPRS Building Classification

```python
import numpy as np
from ign_lidar.core.modules.classification_refinement import (
    refine_building_classification, RefinementConfig
)

# Setup configuration
config = RefinementConfig()
config.VERTICALITY_WALL_MIN = 0.7
config.ROOF_SCORE_MIN = 0.5

# Refine classification
refined_labels, num_changed = refine_building_classification(
    labels=current_labels,
    height=height_array,
    planarity=planarity_array,
    verticality=verticality_array,
    normals=normals_array,              # [N, 3] for horizontality
    linearity=linearity_array,          # Edge detection
    anisotropy=anisotropy_array,        # Structure detection
    wall_score=wall_score_array,        # Pre-computed wall scores
    roof_score=roof_score_array,        # Pre-computed roof scores
    config=config
)

print(f"Refined {num_changed:,} building points")
```

### 2. LOD2 Building Element Classification

```python
from ign_lidar.core.modules.classification_refinement import (
    classify_lod2_building_elements
)

# Classify building elements
lod2_labels = classify_lod2_building_elements(
    points=xyz_coordinates,      # [N, 3]
    labels=initial_labels,       # [N] current classification
    normals=normals_array,       # [N, 3] surface normals
    planarity=planarity_array,   # [N] planarity values
    height=height_array,         # [N] height above ground
    linearity=linearity_array,   # [N] optional edge detection
    curvature=curvature_array    # [N] optional detail detection
)

# Check distribution
from ign_lidar.classes import LOD2_CLASSES
unique, counts = np.unique(lod2_labels, return_counts=True)
for class_id, count in zip(unique, counts):
    class_name = [k for k, v in LOD2_CLASSES.items() if v == class_id][0]
    print(f"{class_name}: {count:,} points")
```

### 3. LOD3 Detailed Classification

```python
from ign_lidar.core.modules.classification_refinement import (
    classify_lod3_building_elements
)

# Detailed architectural classification
lod3_labels = classify_lod3_building_elements(
    points=xyz_coordinates,
    labels=initial_labels,
    normals=normals_array,
    planarity=planarity_array,
    linearity=linearity_array,
    height=height_array,
    curvature=curvature_array,   # Detail detection
    anisotropy=anisotropy_array, # Structure detection
    intensity=intensity_array    # Material/glass detection
)

# Count windows, doors, etc.
num_windows = (lod3_labels == 13).sum()
num_doors = (lod3_labels == 14).sum()
num_pillars = (lod3_labels == 19).sum()
print(f"Detected: {num_windows} windows, {num_doors} doors, {num_pillars} pillars")
```

### 4. Compute New Geometric Features

```python
from ign_lidar.features import (
    compute_horizontality,
    compute_facade_score,
    compute_roof_plane_score,
    compute_opening_likelihood,
    compute_edge_strength,
    compute_structural_element_score
)

# Compute horizontality (for roof detection)
horizontality = compute_horizontality(normals)

# Compute facade score (vertical planar elevated)
facade_score = compute_facade_score(
    normals=normals,
    planarity=planarity,
    verticality=verticality,
    height=height,
    min_height=2.5  # Minimum facade height
)

# Compute roof plane scores
flat_roof, sloped_roof, steep_roof = compute_roof_plane_score(
    normals=normals,
    planarity=planarity,
    height=height,
    min_roof_height=3.0
)

# Detect openings (windows/doors)
opening_likelihood = compute_opening_likelihood(
    planarity=planarity,
    linearity=linearity,
    verticality=verticality,
    intensity=intensity  # Optional: glass detection
)

# Detect edges (corners, roof edges)
edge_strength = compute_edge_strength(eigenvalues)

# Detect structural elements (pillars, columns)
structural_score = compute_structural_element_score(
    linearity=linearity,
    verticality=verticality,
    anisotropy=anisotropy,
    height=height
)
```

---

## 🔍 Feature Descriptions

### Core Features

```python
# Verticality: 1 for walls, 0 for roofs
verticality = 1.0 - np.abs(normals[:, 2])

# Horizontality: 1 for roofs, 0 for walls
horizontality = np.abs(normals[:, 2])

# Wall Score: Combined wall metric
wall_score = planarity * verticality

# Roof Score: Combined roof metric
roof_score = planarity * horizontality
```

### Roof Angle Classification

```python
# Flat roof: slope < 15°
horizontality > 0.966  # cos(15°)

# Sloped roof: 15° - 45°
0.707 < horizontality <= 0.966  # cos(45°) to cos(15°)

# Steep roof: 45° - 70°
0.342 < horizontality <= 0.707  # cos(70°) to cos(45°)
```

### Edge Detection

```python
# From eigenvalues (λ1 >= λ2 >= λ3)
edge_strength = (lambda1 - lambda2) / (lambda1 + epsilon)

# Strong edges: corners, roof edges, structural transitions
is_edge = edge_strength > 0.5
```

---

## 📊 LOD2 Class IDs

```python
LOD2_CLASSES = {
    'wall': 0,
    'roof_flat': 1,
    'roof_gable': 2,
    'roof_hip': 3,
    'chimney': 4,
    'dormer': 5,
    'balcony': 6,
    'overhang': 7,
    'foundation': 8,
    'ground': 9,
    'vegetation_low': 10,
    'vegetation_high': 11,
    'water': 12,
    'vehicle': 13,
    'other': 14,
}
```

---

## 📊 LOD3 Class IDs

```python
LOD3_CLASSES = {
    # Walls
    'wall_plain': 0,
    'wall_with_windows': 1,
    'wall_with_door': 2,

    # Roofs
    'roof_flat': 3,
    'roof_gable': 4,
    'roof_hip': 5,
    'roof_mansard': 6,
    'roof_gambrel': 7,

    # Roof details
    'chimney': 8,
    'dormer_gable': 9,
    'dormer_shed': 10,
    'skylight': 11,
    'roof_edge': 12,

    # Openings
    'window': 13,
    'door': 14,
    'garage_door': 15,

    # Facade
    'balcony': 16,
    'balustrade': 17,
    'overhang': 18,
    'pillar': 19,
    'cornice': 20,

    # Foundation
    'foundation': 21,

    # Context
    'ground': 23,
    'vegetation_low': 24,
    'vegetation_high': 25,
    'water': 26,
    'vehicle': 27,
    'street_furniture': 28,
    'other': 29,
}
```

---

## ⚡ Performance Tips

### 1. Pre-compute Combined Scores

```python
# Compute once, use many times
wall_score = planarity * verticality
roof_score = planarity * horizontality
```

### 2. Filter by Height First

```python
# Reduce candidates before geometric analysis
building_candidates = height > 2.5
# Then apply geometric filters on subset
```

### 3. Use Appropriate Feature Set

```python
# LOD2: ~17 features (faster)
# LOD3: ~43 features (more detailed)

from ign_lidar.features import LOD2_FEATURES, LOD3_FEATURES
```

### 4. Adjust Thresholds for Your Data

```python
# Tune thresholds based on your point cloud characteristics
config.VERTICALITY_WALL_MIN = 0.65  # Lower for noisy data
config.PLANARITY_BUILDING_MIN = 0.45  # Lower for old buildings
```

---

## 🐛 Troubleshooting

### Problem: Too Few Buildings Detected

**Solution**: Lower thresholds

```python
config.VERTICALITY_WALL_MIN = 0.6  # From 0.7
config.WALL_SCORE_MIN = 0.3  # From 0.35
```

### Problem: Vegetation Classified as Buildings

**Solution**: Increase planarity requirement

```python
config.PLANARITY_BUILDING_MIN = 0.6  # From 0.5
# Also use NDVI if available
```

### Problem: Missing Roof Detection

**Solution**: Adjust roof thresholds

```python
config.ROOF_SCORE_MIN = 0.4  # From 0.5
config.HORIZONTALITY_ROOF_MIN = 0.80  # From 0.85
```

### Problem: Windows Not Detected

**Solution**: Check linearity and planarity

```python
# Windows need: low planarity + high linearity
opening_candidates = (
    (planarity < 0.4) &  # Opening (not solid)
    (linearity > 0.5) &  # Strong edges
    (verticality > 0.6)  # On walls
)
```

---

## 📈 Validation

### Check Classification Quality

```python
# Distribution analysis
unique, counts = np.unique(labels, return_counts=True)
total = counts.sum()
for class_id, count in zip(unique, counts):
    pct = (count / total) * 100
    print(f"Class {class_id}: {count:,} ({pct:.1f}%)")

# Check geometric consistency
walls = labels == 0
print(f"Wall verticality: {verticality[walls].mean():.2f}")
print(f"Wall planarity: {planarity[walls].mean():.2f}")

roofs = (labels >= 1) & (labels <= 3)
print(f"Roof horizontality: {horizontality[roofs].mean():.2f}")
print(f"Roof planarity: {planarity[roofs].mean():.2f}")
```

---

## 📚 References

- **Full Documentation**: `BUILDING_CLASSIFICATION_IMPROVEMENTS.md`
- **Feature Modes**: `ign_lidar/features/feature_modes.py`
- **Classification Classes**: `ign_lidar/classes.py`
- **ASPRS Classes**: `ign_lidar/asprs_classes.py`

---

**Version**: 1.0  
**Date**: October 15, 2025  
**Status**: Production Ready ✅
