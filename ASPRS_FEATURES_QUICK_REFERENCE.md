# Quick Reference: ASPRS Feature Requirements

**Fast lookup table for features needed by each classification type**

---

## Feature Matrix

| Feature     | Water | Road | Vegetation | Building |
| ----------- | :---: | :--: | :--------: | :------: |
| height      |   ✓   |  ✓   |     ✓      |    ✓     |
| planarity   |   ✓   |  ✓   |     ✓      |    ✓     |
| curvature   |   ✓   |  ✓   |     ✓      |          |
| normals     |   ✓   |  ✓   |            |          |
| ndvi        |       |  ✓   |     ✓      |    ✓     |
| sphericity  |       |      |     ✓      |          |
| roughness   |       |      |     ✓      |          |
| verticality |       |      |            |    ✓     |

---

## By Classification Type

### Water (Class 9)

```python
['height', 'planarity', 'curvature', 'normals']  # 4 features
```

### Road (Class 11)

```python
['height', 'planarity', 'curvature', 'normals', 'ndvi']  # 5 features
```

### Vegetation (Classes 3, 4, 5)

```python
['ndvi', 'height', 'curvature', 'planarity', 'sphericity', 'roughness']  # 6 features
```

### Building (Class 6)

```python
['height', 'planarity', 'verticality', 'ndvi']  # 4 features
```

---

## All Features (8 unique)

```python
ALL_CLASSIFICATION_FEATURES = [
    'height',        # Z above ground (meters)
    'planarity',     # Flatness [0-1]
    'curvature',     # Surface curvature [0-∞]
    'normals',       # Normal vectors [N, 3]
    'ndvi',          # Vegetation index [-1, 1]
    'sphericity',    # Organic shape [0-1]
    'roughness',     # Surface irregularity [0-∞]
    'verticality',   # Wall-like [0-1]
]
```

---

## Code Usage

```python
from ign_lidar.asprs_classes import (
    WATER_FEATURES,
    ROAD_FEATURES,
    VEGETATION_FEATURES,
    BUILDING_FEATURES,
    get_required_features_for_class,
    validate_features
)

# Get features for a class
veg_features = get_required_features_for_class(3)  # Low vegetation

# Validate features before processing
is_valid, missing = validate_features(my_features, VEGETATION_FEATURES)
if not is_valid:
    print(f"Missing: {missing}")
```

---

## See Also

- **Detailed Guide:** [ASPRS_FEATURE_REQUIREMENTS.md](ASPRS_FEATURE_REQUIREMENTS.md)
- **Implementation:** [ASPRS_FEATURES_UPDATE_SUMMARY.md](ASPRS_FEATURES_UPDATE_SUMMARY.md)
- **Module:** [ign_lidar/asprs_classes.py](ign_lidar/asprs_classes.py)
- **Refinement:** [ign_lidar/core/modules/ground_truth_refinement.py](ign_lidar/core/modules/ground_truth_refinement.py)
