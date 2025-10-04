---
sidebar_position: 1
title: Features API
description: Core feature extraction and processing functions
keywords: [api, features, building, classification, geometric]
---

# Features API Reference

The Features API provides comprehensive tools for extracting geometric and semantic features from LiDAR point clouds.

## Core Classes

### FeatureExtractor

Main class for feature extraction operations.

```python
from ign_lidar import FeatureExtractor

extractor = FeatureExtractor(
    building_threshold=0.5,
    min_points_per_building=100,
    use_gpu=True
)
```

#### Methods

##### `extract_building_features(points, labels)`

Extracts geometric features for building classification.

**Parameters:**

- `points` (numpy.ndarray): Point cloud data (N×3)
- `labels` (numpy.ndarray): Classification labels
- `neighborhood_size` (int, optional): Search radius for feature computation

**Returns:**

- `dict`: Dictionary containing extracted features

**Example:**

```python
features = extractor.extract_building_features(
    points=point_cloud,
    labels=classifications,
    neighborhood_size=1.0
)
```

##### `compute_geometric_features(points)`

Computes basic geometric features for each point.

**Parameters:**

- `points` (numpy.ndarray): Input point coordinates

**Returns:**

- `numpy.ndarray`: Feature array (N×F where F is number of features)

### BuildingClassifier

Advanced classification for building components.

```python
from ign_lidar import BuildingClassifier

classifier = BuildingClassifier(
    model_type="random_forest",
    use_height_features=True,
    enable_planarity=True
)
```

#### Methods

##### `classify_components(points, features)`

Classifies building components (roof, wall, ground).

**Parameters:**

- `points` (numpy.ndarray): Point coordinates
- `features` (dict): Extracted features from FeatureExtractor

**Returns:**

- `numpy.ndarray`: Component labels (0=ground, 1=wall, 2=roof)

##### `refine_classification(labels, points)`

Post-processes classification results for better accuracy.

**Parameters:**

- `labels` (numpy.ndarray): Initial classification
- `points` (numpy.ndarray): Point coordinates

**Returns:**

- `numpy.ndarray`: Refined classification labels

## Feature Types

### Geometric Features

| Feature               | Description                  | Range   |
| --------------------- | ---------------------------- | ------- |
| `planarity`           | Measure of local planarity   | [0, 1]  |
| `linearity`           | Linear structure indicator   | [0, 1]  |
| `sphericity`          | 3D structure compactness     | [0, 1]  |
| `height_above_ground` | Normalized height            | [0, ∞]  |
| `normal_z`            | Z-component of normal vector | [-1, 1] |

### Architectural Features

| Feature              | Description             | Application             |
| -------------------- | ----------------------- | ----------------------- |
| `edge_strength`      | Building edge detection | Wall/roof boundaries    |
| `corner_likelihood`  | Corner probability      | Building corners        |
| `surface_roughness`  | Texture measure         | Material classification |
| `overhang_indicator` | Overhang detection      | Complex geometries      |

## Configuration

### Feature Extraction Settings

```python
config = {
    "geometric_features": {
        "planarity": True,
        "linearity": True,
        "sphericity": True,
        "normal_vectors": True
    },
    "architectural_features": {
        "edge_detection": True,
        "corner_detection": True,
        "surface_analysis": True
    },
    "computation": {
        "neighborhood_size": 1.0,
        "min_neighbors": 10,
        "max_neighbors": 100
    }
}

extractor = FeatureExtractor(config=config)
```

### GPU Acceleration

Enable GPU processing for faster feature extraction:

```python
extractor = FeatureExtractor(
    use_gpu=True,
    gpu_memory_fraction=0.7,
    batch_size=50000
)
```

## Error Handling

```python
try:
    features = extractor.extract_building_features(points, labels)
except InsufficientPointsError:
    print("Not enough points for feature extraction")
except GPUMemoryError:
    print("GPU memory insufficient, falling back to CPU")
    extractor.use_gpu = False
    features = extractor.extract_building_features(points, labels)
```

## Performance Optimization

### Memory Management

```python
# Process large datasets in chunks
def process_large_dataset(large_points):
    chunk_size = 100000
    all_features = []

    for i in range(0, len(large_points), chunk_size):
        chunk = large_points[i:i+chunk_size]
        features = extractor.extract_building_features(chunk)
        all_features.append(features)

    return combine_features(all_features)
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parallel_feature_extraction(point_chunks):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(extractor.extract_building_features, chunk)
            for chunk in point_chunks
        ]
        results = [future.result() for future in futures]
    return results
```

## Examples

### Basic Feature Extraction

```python
import numpy as np
from ign_lidar import FeatureExtractor

# Load point cloud
points = np.load('building_points.npy')
labels = np.load('building_labels.npy')

# Initialize extractor
extractor = FeatureExtractor()

# Extract features
features = extractor.extract_building_features(points, labels)

# Access specific features
planarity = features['planarity']
height_features = features['height_above_ground']
```

### Advanced Classification Pipeline

```python
from ign_lidar import FeatureExtractor, BuildingClassifier

# Setup processing pipeline
extractor = FeatureExtractor(use_gpu=True)
classifier = BuildingClassifier(model_type="gradient_boosting")

# Process point cloud
features = extractor.extract_building_features(points, initial_labels)
refined_labels = classifier.classify_components(points, features)
final_labels = classifier.refine_classification(refined_labels, points)
```

## Related Documentation

- [Processor API](./processor.md)
- [RGB Augmentation API](./rgb-augmentation.md)
- [GPU Integration Guide](../guides/gpu-acceleration.md)
- [Performance Optimization](../guides/performance.md)
