# ASPRS Classification Feature Requirements

**Complete feature specifications for ground truth refinement and classification**

---

## Overview

The IGN LiDAR HD processing pipeline requires specific geometric and spectral features for accurate classification and ground truth refinement. This document defines all required features for each ASPRS classification type.

**Last Updated:** October 19, 2025  
**Module:** `ign_lidar.asprs_classes`  
**Related:** `ign_lidar.core.modules.ground_truth_refinement`

---

## Feature Definitions

### Core Features (8 total)

| Feature       | Type     | Range        | Description                                                                |
| ------------- | -------- | ------------ | -------------------------------------------------------------------------- |
| `height`      | float    | -10 to 100 m | Z coordinate above ground (Z - DTM elevation)                              |
| `planarity`   | float    | 0 to 1       | Local surface flatness. High = flat. Eigenvalue-based: (λ2 - λ3) / λ1      |
| `curvature`   | float    | 0 to ∞       | Surface curvature from local fitting. Typically 0-0.1 for natural surfaces |
| `normals`     | float[3] | -1 to 1      | Surface normal vectors (nx, ny, nz). PCA-based estimation                  |
| `ndvi`        | float    | -1 to 1      | Normalized Difference Vegetation Index. (NIR - Red) / (NIR + Red)          |
| `sphericity`  | float    | 0 to 1       | Shape sphericity. Eigenvalue-based: λ3 / λ1. High = spherical/organic      |
| `roughness`   | float    | 0 to ∞       | Surface roughness. Std dev of distances to plane. Typically 0-0.5          |
| `verticality` | float    | 0 to 1       | Wall-like measure. \|normal_z\| for vertical surfaces                      |

---

## Feature Requirements by Classification Type

### 1. Water (ASPRS Class 9)

**Required Features:** `['height', 'planarity', 'curvature', 'normals']`

**Validation Criteria:**

- Very low height (near ground, < 0.3 m)
- Very high planarity (flat surface, > 0.90)
- Very low curvature (smooth, < 0.02)
- Horizontal normals (pointing up, normal_z > 0.95)

**Purpose:** Ensure water classifications are on flat, horizontal surfaces at ground level (reject bridges over water).

---

### 2. Roads (ASPRS Class 11)

**Required Features:** `['height', 'planarity', 'curvature', 'normals', 'ndvi']`

**Validation Criteria:**

- Low height (near ground, -0.5 to 2.0 m)
- High planarity (flat surface, > 0.85)
- Low curvature (smooth pavement, < 0.05)
- Horizontal normals (normal_z > 0.90)
- Low NDVI (not vegetation, < 0.15)

**Purpose:** Validate road surfaces and detect tree canopy overhanging roads (NDVI-based override to vegetation).

---

### 3. Vegetation (ASPRS Classes 3, 4, 5)

**Required Features:** `['ndvi', 'height', 'curvature', 'planarity', 'sphericity', 'roughness']`

**Multi-Feature Confidence Scoring (Weighted):**

- **NDVI:** 40% (primary vegetation indicator)
- **Curvature:** 20% (complex surfaces - branches, leaves)
- **Sphericity:** 20% (organic, irregular shapes)
- **Planarity (inverse):** 10% (non-flat surfaces)
- **Roughness:** 10% (surface irregularity)

**Height-Based Sub-Classification:**

- Low vegetation (3): < 0.5 m
- Medium vegetation (4): 0.5 - 2.0 m
- High vegetation (5): > 2.0 m

**Purpose:** Pure feature-based vegetation detection. BD TOPO vegetation disabled due to polygon misalignment issues.

---

### 4. Buildings (ASPRS Class 6)

**Required Features:** `['height', 'planarity', 'verticality', 'ndvi']`

**Validation Criteria:**

- Minimum height (> 1.5 m)
- Moderate planarity (roof surfaces, > 0.65)
- High verticality (walls, > 0.6 for facades)
- Low NDVI (not vegetation, < 0.20)

**Special Processing:**

- Building polygons expanded by 0.5 m buffer
- Validates points within expanded polygons using features

**Purpose:** Capture all building points including edges missed by BD TOPO polygons.

---

## Feature Computation Pipeline

### Source Module: `ign_lidar.features`

**Computation Order:**

1. **height**: Z - DTM elevation (per-point)
2. **normals**: PCA on k-nearest neighbors (typically k=20)
3. **planarity**: Eigenvalue ratio from PCA
4. **curvature**: Mean curvature from local surface fitting
5. **sphericity**: Eigenvalue ratio λ3 / λ1
6. **roughness**: Std dev of distances to fitted plane
7. **verticality**: |normal_z| < threshold for vertical surfaces
8. **ndvi**: (NIR - Red) / (NIR + Red) from RGB approximation

**Performance:**

- All features computed in single pass
- GPU acceleration available
- No redundant computations

---

## Code Usage

### Getting Required Features

```python
from ign_lidar.asprs_classes import (
    get_required_features_for_class,
    get_all_required_features,
    WATER_FEATURES,
    VEGETATION_FEATURES
)

# Get features for specific class
water_features = get_required_features_for_class(9)
# Returns: ['height', 'planarity', 'curvature', 'normals']

# Get features for vegetation
veg_features = get_required_features_for_class(3)
# Returns: ['ndvi', 'height', 'curvature', 'planarity', 'sphericity', 'roughness']

# Get all features needed
all_features = get_all_required_features()
# Returns: ['height', 'planarity', 'curvature', 'normals', 'ndvi', 'sphericity', 'roughness', 'verticality']
```

### Feature Validation

```python
from ign_lidar.asprs_classes import validate_features, VEGETATION_FEATURES

# Check if features are available
features = {
    'ndvi': ndvi_array,
    'height': height_array,
    'curvature': curvature_array,
    # ... other features
}

is_valid, missing = validate_features(features, VEGETATION_FEATURES)
if not is_valid:
    print(f"Missing features: {missing}")
```

### Feature Descriptions

```python
from ign_lidar.asprs_classes import get_feature_description, get_feature_range

# Get feature info
desc = get_feature_description('ndvi')
# Returns: "Normalized Difference Vegetation Index [-1, 1]. (NIR - Red) / (NIR + Red). High = vegetation."

min_val, max_val = get_feature_range('height')
# Returns: (-10.0, 100.0)
```

---

## Configuration

### Enable All Required Features

In your configuration file (e.g., `config_asprs_bdtopo_cadastre_optimized.yaml`):

```yaml
features:
  height: true # Required for all classes
  planarity: true # Required for water, roads, buildings
  curvature: true # Required for water, roads, vegetation
  normals: true # Required for water, roads
  ndvi: true # Required for roads, vegetation, buildings
  sphericity: true # Required for vegetation
  roughness: true # Required for vegetation
  verticality: true # Required for buildings
```

### Ground Truth Refinement

```yaml
strtree:
  enable_refinement: true

  # Feature-based vegetation (BD TOPO vegetation disabled)
  vegetation: false # Use computed features instead of BD TOPO polygons
```

---

## Implementation Details

### Module Structure

```
ign_lidar/
├── asprs_classes.py                    # Feature definitions & utilities (THIS FILE)
├── features/                           # Feature computation
│   ├── height.py
│   ├── planarity.py
│   ├── curvature.py
│   ├── normals.py
│   ├── ndvi.py
│   ├── sphericity.py
│   ├── roughness.py
│   └── verticality.py
└── core/
    └── modules/
        └── ground_truth_refinement.py  # Uses features for refinement
```

### Feature Constants

```python
# From ign_lidar.asprs_classes

WATER_FEATURES = ['height', 'planarity', 'curvature', 'normals']
ROAD_FEATURES = ['height', 'planarity', 'curvature', 'normals', 'ndvi']
VEGETATION_FEATURES = ['ndvi', 'height', 'curvature', 'planarity', 'sphericity', 'roughness']
BUILDING_FEATURES = ['height', 'planarity', 'verticality', 'ndvi']

ALL_CLASSIFICATION_FEATURES = [
    'height', 'planarity', 'curvature', 'normals',
    'ndvi', 'sphericity', 'roughness', 'verticality'
]
```

---

## Performance Considerations

### Memory Usage

| Feature     | Memory (1M points) | Computation     |
| ----------- | ------------------ | --------------- |
| height      | 4 MB               | O(n)            |
| planarity   | 4 MB               | O(n × k)        |
| curvature   | 4 MB               | O(n × k)        |
| normals     | 12 MB              | O(n × k)        |
| ndvi        | 4 MB               | O(n)            |
| sphericity  | 4 MB               | O(n × k)        |
| roughness   | 4 MB               | O(n × k)        |
| verticality | 4 MB               | O(n)            |
| **TOTAL**   | **44 MB**          | **Single pass** |

Where k = number of neighbors (typically 20)

### GPU Acceleration

All features support GPU acceleration for large point clouds:

- 10-30× speedup on typical tiles
- Automatic fallback to CPU if GPU unavailable

---

## Testing

### Verify Feature Availability

```python
import numpy as np
from ign_lidar.asprs_classes import validate_features, ALL_CLASSIFICATION_FEATURES

# Simulate feature computation
features = {
    'height': np.random.rand(1000),
    'planarity': np.random.rand(1000),
    'curvature': np.random.rand(1000),
    'normals': np.random.rand(1000, 3),
    'ndvi': np.random.rand(1000),
    'sphericity': np.random.rand(1000),
    'roughness': np.random.rand(1000),
    'verticality': np.random.rand(1000),
}

is_valid, missing = validate_features(features, ALL_CLASSIFICATION_FEATURES)
assert is_valid, f"Missing features: {missing}"
```

---

## Version History

- **v5.2.2** (Oct 19, 2025): Complete feature documentation in `asprs_classes.py`
- **v5.2.1** (Oct 19, 2025): Enhanced vegetation with sphericity + roughness
- **v5.2.0** (Oct 19, 2025): Initial ground truth refinement implementation

---

## See Also

- [Ground Truth Refinement Guide](docs/guides/ground-truth-refinement.md)
- [Vegetation Feature-Based Update](VEGETATION_FEATURE_BASED_UPDATE.md)
- [CHANGELOG](CHANGELOG.md)
- [Module: asprs_classes.py](ign_lidar/asprs_classes.py)
- [Module: ground_truth_refinement.py](ign_lidar/core/modules/ground_truth_refinement.py)
