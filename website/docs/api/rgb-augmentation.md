---
sidebar_position: 2
title: RGB Augmentation API
description: API for integrating orthophoto color data with LiDAR point clouds
keywords: [api, rgb, color, orthophoto, augmentation]
---

# RGB Augmentation API Reference

The RGB Augmentation API provides tools for integrating IGN orthophoto data with LiDAR point clouds to create color-enhanced datasets.

## Core Classes

### RGBProcessor

Main class for RGB augmentation operations.

```python
from ign_lidar import RGBProcessor

processor = RGBProcessor(
    interpolation_method='bilinear',
    quality_threshold=0.8,
    enable_caching=True
)
```

## Methods

### `augment_point_cloud(points, orthophoto_path)`

Adds RGB values to point cloud from orthophoto.

**Parameters:**

- `points` (numpy.ndarray): Point coordinates (N×3)
- `orthophoto_path` (str): Path to orthophoto file

**Returns:**

- `numpy.ndarray`: RGB values (N×3) as uint8

### `batch_augmentation(tile_list, ortho_dir)`

Process multiple tiles with RGB augmentation.

**Parameters:**

- `tile_list` (list): List of tile paths
- `ortho_dir` (str): Directory containing orthophotos

**Returns:**

- `dict`: Results dictionary with augmented data

## Configuration Options

### Interpolation Methods

- `nearest`: Fastest, pixel-exact colors
- `bilinear`: Smooth color transitions
- `bicubic`: Highest quality interpolation

### Quality Control

```python
processor = RGBProcessor(
    quality_threshold=0.9,  # Color accuracy threshold
    validate_coordinates=True,  # Check point-photo alignment
    handle_missing_data=True   # Fill gaps gracefully
)
```

## Error Handling

```python
try:
    rgb_data = processor.augment_point_cloud(points, orthophoto)
except OrthophotoNotFoundError:
    print("Orthophoto file not accessible")
except CoordinateMismatchError:
    print("Point cloud and orthophoto coordinates don't align")
except InsufficientOverlapError:
    print("Not enough overlap between data sources")
```

## Performance Optimization

### GPU Acceleration

```python
processor = RGBProcessor(
    use_gpu=True,
    gpu_batch_size=50000,
    enable_gpu_caching=True
)
```

### Memory Management

```python
# Process large datasets efficiently
def process_large_dataset(points, orthophoto):
    chunk_size = 100000
    rgb_results = []

    for i in range(0, len(points), chunk_size):
        chunk = points[i:i+chunk_size]
        rgb_chunk = processor.augment_point_cloud(chunk, orthophoto)
        rgb_results.append(rgb_chunk)

    return np.concatenate(rgb_results)
```

## Examples

### Basic RGB Augmentation

```python
import numpy as np
from ign_lidar import RGBProcessor

# Initialize processor
processor = RGBProcessor()

# Load point cloud
points = np.load('building_points.npy')

# Add RGB data
rgb_colors = processor.augment_point_cloud(
    points,
    'orthophoto.tif'
)

# Save enhanced point cloud
enhanced_points = np.column_stack([points, rgb_colors])
```

### Batch Processing with Quality Control

```python
processor = RGBProcessor(
    quality_threshold=0.85,
    validate_coordinates=True
)

results = processor.batch_augmentation(
    tile_list=['tile1.las', 'tile2.las'],
    ortho_dir='/path/to/orthophotos/'
)
```

## Related Documentation

- [Features API](./features.md)
- [Processor API](./processor.md)
- [GPU Acceleration Guide](../guides/gpu-acceleration.md)
