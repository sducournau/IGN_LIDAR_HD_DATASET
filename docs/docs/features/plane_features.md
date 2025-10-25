# Plane-Based Features for Building Classification

## Overview

The IGN LiDAR HD Dataset library now includes **plane-based feature extraction** for enhanced building classification and architectural element detection. This feature enables ML models to learn from planar structures (roofs, walls, facades) in point clouds.

## What Are Plane Features?

Plane features describe the geometric relationship between individual points and detected planar surfaces in the point cloud. For building classification, planes correspond to:

- **Horizontal planes**: Flat roofs, terraces, floors
- **Vertical planes**: Walls, facades
- **Inclined planes**: Sloped roofs, pitched surfaces

## Available Plane Features

Each point can have up to **8 plane-based features**:

| Feature               | Type    | Range         | Description                                         |
| --------------------- | ------- | ------------- | --------------------------------------------------- |
| `plane_id`            | int32   | [-1, ∞)       | ID of nearest plane (-1 if not assigned)            |
| `plane_type`          | int8    | {-1, 0, 1, 2} | Type: 0=horizontal, 1=vertical, 2=inclined, -1=none |
| `distance_to_plane`   | float32 | [0, ∞)        | Perpendicular distance to plane surface (meters)    |
| `plane_area`          | float32 | [0, ∞)        | Area of containing plane (m²)                       |
| `plane_orientation`   | float32 | [0, 90]       | Angle of plane normal from horizontal (degrees)     |
| `plane_planarity`     | float32 | [0, 1]        | Planarity score of plane (1 = perfectly flat)       |
| `position_on_plane_u` | float32 | [0, 1]        | Normalized U coordinate on plane                    |
| `position_on_plane_v` | float32 | [0, 1]        | Normalized V coordinate on plane                    |

## Usage

### 1. Enable in Configuration

```yaml
# config.yaml
features:
  mode: "lod3" # LOD3 mode includes plane features automatically

  plane_detection:
    enabled: true
    horizontal_angle_max: 10.0 # degrees
    vertical_angle_min: 75.0 # degrees
    min_points_per_plane: 50 # minimum points
    max_assignment_distance: 0.5 # meters
```

### 2. Process Tiles

```bash
ign-lidar-hd process -c config_plane_features.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

### 3. Use in Python

```python
from ign_lidar import LiDARProcessor
from ign_lidar.features import get_feature_config

# Configure with plane features
config = get_feature_config(mode="lod3")  # Includes planes
print(f"Features: {config.num_features}")  # ~51 features

# Process
processor = LiDARProcessor(config_path="config_plane_features.yaml")
processor.process_tile("tile.laz")
```

### 4. Standalone Plane Detection

```python
from ign_lidar.core.classification.plane_detection import (
    PlaneDetector, PlaneFeatureExtractor
)
import numpy as np

# Your point cloud data
points = ...  # [N, 3]
normals = ...  # [N, 3]
planarity = ...  # [N]

# Detect planes and extract features
detector = PlaneDetector(
    horizontal_angle_max=10.0,
    vertical_angle_min=75.0
)
extractor = PlaneFeatureExtractor(detector)

features = extractor.detect_and_assign_planes(
    points, normals, planarity
)

# Access features
plane_ids = features['plane_id']  # [-1, 0, 1, 2, ...]
plane_types = features['plane_type']  # [1, 1, 0, 0, ...] (walls, roofs)
distances = features['distance_to_plane']  # [0.05, 0.12, ...]

# Statistics
stats = extractor.get_plane_statistics()
print(f"Detected {stats['n_planes']} planes")
print(f"  Horizontal: {stats['n_horizontal']}")
print(f"  Vertical: {stats['n_vertical']}")
print(f"  Inclined: {stats['n_inclined']}")
```

## Feature Modes

Plane features are included in:

- **LOD3_FULL** (`lod3`): Complete feature set including planes (~51 features)
- **PLANES** (`planes`): Plane features only (~8 features)
- **FULL** (`full`): All available features
- **CUSTOM** (`custom`): User-defined selection

```python
# Example custom mode with planes
custom_features = {
    'xyz',
    'normal_z',
    'planarity',
    'verticality',
    # Plane features
    'plane_id',
    'plane_type',
    'distance_to_plane',
    'plane_orientation',
}

config = get_feature_config(
    mode="custom",
    custom_features=custom_features
)
```

## Use Cases

### 1. Building Classification

```python
# Train classifier with plane features
X_train = patches['features']  # Shape: [N, 51] with plane features
y_train = patches['labels']

# Plane features help distinguish:
# - Walls (vertical planes, plane_type=1)
# - Roofs (horizontal/inclined planes, plane_type=0/2)
# - Architectural elements (small planes, low plane_area)
```

### 2. Facade Detection

```python
# Extract facade points using plane features
is_facade = (
    (features['plane_type'] == 1) &  # Vertical
    (features['plane_area'] > 10.0) &  # Large plane
    (features['plane_planarity'] > 0.8)  # High planarity
)

facade_points = points[is_facade]
```

### 3. Roof Type Classification

```python
# Classify roof types
is_flat_roof = (
    (features['plane_type'] == 0) &  # Horizontal
    (features['plane_orientation'] < 10.0)  # Nearly flat
)

is_pitched_roof = (
    (features['plane_type'] == 2) &  # Inclined
    (features['plane_orientation'] > 15.0) &
    (features['plane_orientation'] < 70.0)
)
```

### 4. LOD3 Reconstruction

```python
# Group points by plane for LOD3 modeling
for plane_id in np.unique(features['plane_id']):
    if plane_id < 0:
        continue  # Skip unassigned points

    plane_mask = (features['plane_id'] == plane_id)
    plane_points = points[plane_mask]
    plane_type = features['plane_type'][plane_mask][0]

    # Reconstruct plane surface
    if plane_type == 0:  # Horizontal
        reconstruct_roof(plane_points)
    elif plane_type == 1:  # Vertical
        reconstruct_wall(plane_points)
    elif plane_type == 2:  # Inclined
        reconstruct_pitched_roof(plane_points)
```

## Configuration Parameters

### Plane Detection

```yaml
plane_detection:
  # Angle thresholds (degrees from horizontal)
  horizontal_angle_max: 10.0 # Max angle for horizontal planes
  vertical_angle_min: 75.0 # Min angle for vertical planes
  inclined_angle_min: 15.0 # Min angle for inclined planes
  inclined_angle_max: 70.0 # Max angle for inclined planes

  # Quality thresholds
  horizontal_planarity_min: 0.75 # Min planarity for horizontal
  vertical_planarity_min: 0.65 # Min planarity for vertical
  inclined_planarity_min: 0.70 # Min planarity for inclined

  # Size constraints
  min_points_per_plane: 50 # Min points to form valid plane
  max_plane_distance: 0.15 # Max distance for inliers (m)

  # Assignment
  max_assignment_distance: 0.5 # Max distance to assign point to plane (m)
```

## Performance

### Computational Cost

Plane detection adds ~10-15% overhead:

- **Plane detection:** ~2-5 seconds per 1M points
- **Feature assignment:** ~1-2 seconds per 1M points
- **Total overhead:** ~3-7 seconds per 1M points

### Memory Usage

- **Plane features:** 8 features × 4 bytes = 32 bytes per point
- **Example:** 1M points = 32 MB additional memory

### GPU Acceleration

Plane detection currently runs on CPU. GPU acceleration planned for future release.

## Output Formats

### NPZ (Training Patches)

```python
# Load NPZ patch
patch = np.load('patch_0000.npz')

# Access plane features
plane_id = patch['plane_id']  # [N]
plane_type = patch['plane_type']  # [N]
distance_to_plane = patch['distance_to_plane']  # [N]
# ... other features
```

### LAZ (Enriched Point Cloud)

Plane features can be saved as extra dimensions in LAZ files:

```yaml
output:
  laz_output:
    include_plane_features: true # Add plane features to LAZ
```

View in CloudCompare or QGIS:

- `plane_id`: Integer attribute
- `plane_type`: Integer attribute (0/1/2)
- `distance_to_plane`: Float attribute
- etc.

## Examples

See `examples/` directory:

- **config_plane_features.yaml**: Complete configuration with plane features
- **demo_plane_detection.py**: Standalone plane detection example
- **demo_building_segmentation.py**: Building segmentation with planes

## Testing

Run plane feature tests:

```bash
# Unit tests
pytest tests/test_plane_features.py -v

# Integration tests
pytest tests/test_integration_planes.py -v
```

## Troubleshooting

### No Planes Detected

**Problem:** All points have `plane_id = -1`

**Solutions:**

1. Check planarity values: `np.mean(planarity)` should be > 0.5 for buildings
2. Lower `min_points_per_plane` threshold
3. Increase `max_plane_distance` tolerance
4. Verify normals are computed correctly

### Too Many Planes Detected

**Problem:** Many small fragmented planes

**Solutions:**

1. Increase `min_points_per_plane` (default: 50)
2. Increase planarity thresholds
3. Enable spatial coherence checking
4. Use larger search radius for normal computation

### Points Not Assigned to Planes

**Problem:** `distance_to_plane = inf` for many points

**Solutions:**

1. Increase `max_assignment_distance` (default: 0.5m)
2. Lower planarity thresholds to detect more planes
3. Check if points are on boundaries (expected behavior)

## API Reference

### PlaneDetector

```python
from ign_lidar.core.classification.plane_detection import PlaneDetector

detector = PlaneDetector(
    horizontal_angle_max=10.0,
    vertical_angle_min=75.0,
    inclined_angle_min=15.0,
    inclined_angle_max=70.0,
    min_points_per_plane=50,
    max_plane_distance=0.15
)

# Detect all plane types
planes_dict = detector.detect_all_planes(
    points, normals, planarity, height
)

# Access by type
horizontal_planes = planes_dict[PlaneType.HORIZONTAL]
vertical_planes = planes_dict[PlaneType.VERTICAL]
inclined_planes = planes_dict[PlaneType.INCLINED]
```

### PlaneFeatureExtractor

```python
from ign_lidar.core.classification.plane_detection import PlaneFeatureExtractor

extractor = PlaneFeatureExtractor(detector)

# Extract features
features = extractor.detect_and_assign_planes(
    points, normals, planarity, height,
    max_assignment_distance=0.5
)

# Get statistics
stats = extractor.get_plane_statistics()
```

## Citation

If you use plane-based features in your research, please cite:

```bibtex
@software{ign_lidar_hd_planes,
  title={IGN LiDAR HD Dataset - Plane-Based Features},
  author={IGN LiDAR HD Development Team},
  year={2025},
  url={https://github.com/sducournau/IGN_LIDAR_HD_DATASET}
}
```

## Related Documentation

- [Feature Computation Guide](../guides/features.md)
- [Building Classification Guide](../guides/classification.md)
- [API Reference](../api/plane_detection.md)
- [LOD3 Reconstruction](../guides/lod3_reconstruction.md)

## Version History

- **v3.1.0** (October 2025): Initial release of plane-based features
  - 8 plane features per point
  - Horizontal/vertical/inclined plane detection
  - Integration with LOD3 feature mode
  - Example configurations and tests
