---
sidebar_position: 1
---

# Custom Features

Learn how to create and integrate custom feature extractors for specialized LiDAR processing tasks.

## Overview

The IGN LiDAR HD library provides a flexible framework for implementing custom feature extraction algorithms. This tutorial shows how to create your own feature extractors.

## Creating Custom Features

### Basic Feature Extractor

Create a custom feature class by inheriting from the base FeatureExtractor:

```python
from ign_lidar.features import FeatureExtractor
import numpy as np

class CustomFeature(FeatureExtractor):
    def __init__(self, radius=2.0):
        super().__init__(name="custom_feature", radius=radius)

    def extract(self, points, neighborhoods):
        """Extract custom feature from point neighborhoods."""
        features = []

        for neighbors in neighborhoods:
            # Implement your custom feature calculation
            feature_value = self._calculate_feature(neighbors)
            features.append(feature_value)

        return np.array(features)

    def _calculate_feature(self, neighbors):
        """Implement your custom calculation here."""
        # Example: distance variance
        if len(neighbors) < 3:
            return 0.0

        distances = np.linalg.norm(neighbors - neighbors.mean(axis=0), axis=1)
        return np.var(distances)
```

### Advanced Feature with Parameters

Create more sophisticated features with configurable parameters:

```python
class AdvancedFeature(FeatureExtractor):
    def __init__(self, radius=2.0, min_points=10, weight_function="inverse"):
        super().__init__(name="advanced_feature", radius=radius)
        self.min_points = min_points
        self.weight_function = weight_function

    def extract(self, points, neighborhoods):
        features = []

        for i, neighbors in enumerate(neighborhoods):
            if len(neighbors) < self.min_points:
                features.append(0.0)
                continue

            # Apply weighting based on distance
            center = points[i]
            distances = np.linalg.norm(neighbors - center, axis=1)
            weights = self._compute_weights(distances)

            # Weighted feature calculation
            weighted_values = self._compute_weighted_values(neighbors, weights)
            features.append(weighted_values)

        return np.array(features)

    def _compute_weights(self, distances):
        if self.weight_function == "inverse":
            return 1.0 / (distances + 1e-6)
        elif self.weight_function == "gaussian":
            sigma = self.radius / 3.0
            return np.exp(-distances**2 / (2 * sigma**2))
        else:
            return np.ones_like(distances)
```

## Registering Custom Features

### Single Feature Registration

Register your custom feature with the processor:

```python
from ign_lidar import Processor, Config

# Create processor with custom features
processor = Processor()

# Register custom feature
custom_feature = CustomFeature(radius=3.0)
processor.register_feature(custom_feature)

# Use in processing
config = Config(
    feature_types=["height_above_ground", "custom_feature"],
    feature_radius=3.0
)

result = processor.process_tile("input.las", config=config)
```

### Batch Feature Registration

Register multiple custom features:

```python
# Create multiple custom features
features = [
    CustomFeature(radius=2.0),
    AdvancedFeature(radius=3.0, min_points=15),
    CustomFeature(radius=1.0)  # Different parameters
]

# Register all features
for feature in features:
    processor.register_feature(feature)

# Configure processing
config = Config(
    feature_types=["custom_feature", "advanced_feature"],
    enable_gpu=True  # GPU acceleration for custom features
)
```

## Feature Combination

### Combining Multiple Features

Create composite features that combine multiple calculations:

```python
class CompositeFeature(FeatureExtractor):
    def __init__(self, radius=2.0):
        super().__init__(name="composite_feature", radius=radius)

        # Initialize sub-features
        self.geometric_feature = CustomFeature(radius)
        self.intensity_feature = IntensityFeature(radius)

    def extract(self, points, neighborhoods):
        # Extract individual features
        geometric = self.geometric_feature.extract(points, neighborhoods)
        intensity = self.intensity_feature.extract(points, neighborhoods)

        # Combine features (example: weighted sum)
        weights = [0.7, 0.3]
        combined = weights[0] * geometric + weights[1] * intensity

        return combined
```

## Performance Optimization

### GPU-Accelerated Features

Implement GPU acceleration for custom features:

```python
try:
    import cupy as cp

    class GPUFeature(FeatureExtractor):
        def __init__(self, radius=2.0):
            super().__init__(name="gpu_feature", radius=radius)
            self.use_gpu = cp.cuda.is_available()

        def extract(self, points, neighborhoods):
            if self.use_gpu:
                return self._extract_gpu(points, neighborhoods)
            else:
                return self._extract_cpu(points, neighborhoods)

        def _extract_gpu(self, points, neighborhoods):
            # GPU implementation using CuPy
            gpu_points = cp.asarray(points)
            # ... GPU-accelerated calculations
            return cp.asnumpy(result)

        def _extract_cpu(self, points, neighborhoods):
            # Fallback CPU implementation
            pass

except ImportError:
    print("GPU acceleration not available")
```

## Testing Custom Features

### Unit Testing

Create tests for your custom features:

```python
import unittest
from ign_lidar.test_utils import generate_test_points

class TestCustomFeature(unittest.TestCase):
    def setUp(self):
        self.feature = CustomFeature(radius=2.0)
        self.test_points = generate_test_points(1000)

    def test_feature_shape(self):
        """Test that feature output has correct shape."""
        neighborhoods = self.feature.get_neighborhoods(self.test_points)
        features = self.feature.extract(self.test_points, neighborhoods)

        self.assertEqual(len(features), len(self.test_points))

    def test_feature_range(self):
        """Test that feature values are in expected range."""
        neighborhoods = self.feature.get_neighborhoods(self.test_points)
        features = self.feature.extract(self.test_points, neighborhoods)

        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(np.isfinite(features)))
```

## Best Practices

1. **Normalization**: Always normalize feature values to [0,1] or [-1,1] range
2. **Error Handling**: Handle edge cases (empty neighborhoods, NaN values)
3. **Documentation**: Provide clear docstrings explaining feature meaning
4. **Testing**: Write comprehensive unit tests
5. **Performance**: Consider GPU implementation for computationally intensive features
