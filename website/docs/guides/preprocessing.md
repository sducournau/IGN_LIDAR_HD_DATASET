---
sidebar_position: 7
title: Preprocessing Guide
description: Point cloud preprocessing for artifact mitigation
keywords: [preprocessing, artifacts, SOR, ROR, voxel, outlier removal]
---

# Preprocessing for Artifact Mitigation

Learn how to use point cloud preprocessing to reduce LiDAR scan line artifacts and improve geometric feature quality.

## Overview

LiDAR point clouds often contain various artifacts that negatively impact geometric feature computation:

- **Scan line patterns** - "Dashed line" appearance in planarity/curvature maps
- **Measurement noise** - Outliers from atmospheric conditions, sensor errors
- **Isolated points** - Stray points from birds, insects, or reflection errors
- **Density variations** - Uneven point distribution affecting feature consistency

The preprocessing module addresses these issues through three complementary techniques:

1. **Statistical Outlier Removal (SOR)** - Remove outliers based on neighbor distance statistics
2. **Radius Outlier Removal (ROR)** - Remove isolated points without sufficient neighbors
3. **Voxel Downsampling** - Homogenize point density (optional)

## Quick Start

### Enable with Defaults

```bash
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --preprocess
```

This applies:

- SOR with k=12 neighbors, std_multiplier=2.0
- ROR with radius=1.0m, min_neighbors=4
- No voxel downsampling

### Building Mode with Preprocessing

```bash
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --mode building \
  --preprocess \
  --num-workers 4
```

### With RGB and Preprocessing

```bash
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --mode building \
  --preprocess \
  --add-rgb \
  --rgb-cache-dir cache/
```

## Preprocessing Techniques

### Statistical Outlier Removal (SOR)

Removes points whose mean distance to k-nearest neighbors is outside the statistical norm.

**How it works:**

1. For each point, compute mean distance to k nearest neighbors
2. Calculate global mean and standard deviation of these distances
3. Remove points exceeding: `threshold = mean + std_multiplier × std`

**Parameters:**

- `--sor-k` - Number of neighbors (default: 12)
  - Higher values: more robust but slower
  - Lower values: faster but may miss some outliers
- `--sor-std` - Standard deviation multiplier (default: 2.0)
  - Higher values: more lenient (keep more points)
  - Lower values: stricter filtering (remove more points)

**Example:**

```bash
# Conservative (keep more points)
--preprocess --sor-k 15 --sor-std 3.0

# Aggressive (remove more outliers)
--preprocess --sor-k 10 --sor-std 1.5
```

### Radius Outlier Removal (ROR)

Removes points that don't have enough neighbors within a specified radius.

**How it works:**

1. For each point, count neighbors within search radius
2. Remove points with fewer than min_neighbors

**Parameters:**

- `--ror-radius` - Search radius in meters (default: 1.0)
  - Urban: 0.5-1.0m (denser point clouds)
  - Rural: 1.0-2.0m (sparser point clouds)
- `--ror-neighbors` - Minimum required neighbors (default: 4)
  - Higher values: stricter isolation detection
  - Lower values: more lenient

**Example:**

```bash
# Urban areas (dense clouds)
--preprocess --ror-radius 0.8 --ror-neighbors 5

# Rural areas (sparse clouds)
--preprocess --ror-radius 1.5 --ror-neighbors 3
```

### Voxel Downsampling

Divides space into voxels (3D grid cells) and reduces points to one representative per voxel.

**How it works:**

1. Divide space into voxel grid of size `voxel_size`
2. For each voxel with points, compute centroid
3. Replace all points in voxel with single centroid point

**Parameters:**

- `--voxel-size` - Voxel size in meters (optional, e.g., 0.5)
  - Smaller values: preserve more detail (0.2-0.5m)
  - Larger values: aggressive downsampling (0.5-1.0m)

**Example:**

```bash
# Moderate downsampling
--preprocess --voxel-size 0.5

# Aggressive memory reduction
--preprocess --voxel-size 1.0
```

**When to use:**

- Very dense point clouds (>20M points/tile)
- Memory-constrained systems
- When processing speed is critical
- When uniform density is desired

## Recommended Presets

### Conservative (Default)

Preserve maximum detail while removing clear outliers:

```bash
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --preprocess \
  --sor-k 12 \
  --sor-std 2.0 \
  --ror-radius 1.0 \
  --ror-neighbors 4
```

**Use for:**

- High-quality datasets
- Detailed building extraction
- When preserving fine features

### Balanced

Good quality improvement with reasonable processing time:

```bash
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --preprocess \
  --sor-k 12 \
  --sor-std 2.0 \
  --ror-radius 1.0 \
  --ror-neighbors 4 \
  --voxel-size 0.5
```

**Use for:**

- General-purpose processing
- Large batches of tiles
- Moderate memory constraints

### Aggressive

Maximum artifact removal and memory reduction:

```bash
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --preprocess \
  --sor-k 10 \
  --sor-std 1.5 \
  --ror-radius 0.8 \
  --ror-neighbors 5 \
  --voxel-size 0.3
```

**Use for:**

- Noisy datasets
- Memory-limited systems
- When speed > detail
- Initial exploratory analysis

### Urban Scenes

Optimized for dense urban environments:

```bash
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --preprocess \
  --sor-k 15 \
  --sor-std 2.5 \
  --ror-radius 0.8 \
  --ror-neighbors 5
```

### Natural Scenes

Optimized for vegetation and rural areas:

```bash
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --preprocess \
  --sor-k 12 \
  --sor-std 3.0 \
  --ror-radius 1.5 \
  --ror-neighbors 3
```

## Python API

### Using LiDARProcessor

```python
from ign_lidar import LiDARProcessor

# Enable preprocessing with defaults
processor = LiDARProcessor(
    lod_level='LOD2',
    include_extra_features=True,
    preprocess=True
)

# Custom preprocessing configuration
preprocess_config = {
    'sor': {
        'enable': True,
        'k': 15,
        'std_multiplier': 2.5
    },
    'ror': {
        'enable': True,
        'radius': 1.0,
        'min_neighbors': 4
    },
    'voxel': {
        'enable': False
    }
}

processor = LiDARProcessor(
    lod_level='LOD2',
    include_extra_features=True,
    preprocess=True,
    preprocess_config=preprocess_config
)

# Process tiles
processor.process_tile('tile.laz', 'output/')
```

### Direct Preprocessing Functions

```python
from ign_lidar.preprocessing import (
    statistical_outlier_removal,
    radius_outlier_removal,
    voxel_downsample,
    preprocess_point_cloud
)
import numpy as np

# Load your points (N×3 array)
points = np.load('points.npy')

# Apply individual filters
filtered_sor, mask_sor = statistical_outlier_removal(
    points, k=12, std_multiplier=2.0
)

filtered_ror, mask_ror = radius_outlier_removal(
    points, radius=1.0, min_neighbors=4
)

downsampled, voxel_indices = voxel_downsample(
    points, voxel_size=0.5, method='centroid'
)

# Or use the complete pipeline
processed, stats = preprocess_point_cloud(points, config=None)

print(f"Original: {stats['original_points']:,} points")
print(f"Final: {stats['final_points']:,} points")
print(f"Reduction: {stats['reduction_ratio']:.1%}")
print(f"Time: {stats['processing_time_ms']:.0f}ms")
```

## Performance Impact

### Processing Time

| Configuration     | Overhead      | Typical Tile Time |
| ----------------- | ------------- | ----------------- |
| No preprocessing  | 0% (baseline) | 2-5 minutes       |
| SOR only          | +5-10%        | 2.2-5.5 minutes   |
| SOR + ROR         | +10-20%       | 2.5-6 minutes     |
| SOR + ROR + Voxel | +15-30%       | 2.8-6.5 minutes   |

### Memory Usage

| Configuration     | Memory Impact               |
| ----------------- | --------------------------- |
| No preprocessing  | Baseline                    |
| SOR/ROR           | +10% (KDTree overhead)      |
| With voxel (0.5m) | -30% to -50% (fewer points) |
| With voxel (1.0m) | -50% to -70% (fewer points) |

### Quality Improvement

Based on comprehensive artifact analysis:

| Metric                 | Improvement      |
| ---------------------- | ---------------- |
| Scan line artifacts    | 60-80% reduction |
| Surface normal quality | 40-60% cleaner   |
| Edge discontinuities   | 30-50% smoother  |
| Degenerate features    | 20-40% fewer     |

## Best Practices

### When to Use Preprocessing

✅ **Use preprocessing when:**

- You observe scan line patterns in feature visualizations
- Point clouds have visible noise or outliers
- Building edges appear jagged or noisy
- Training ML models (cleaner features = better results)
- Processing large batches (voxel helps with memory)

❌ **Skip preprocessing when:**

- Point clouds are already clean (manual inspection)
- You need absolute maximum detail preservation
- Processing time is critical constraint
- Working with very small/sparse clouds (less than 100k points)

### Parameter Tuning Tips

1. **Start with defaults** - They work well for most IGN LiDAR HD data
2. **Visualize before/after** - Use CloudCompare or similar tools
3. **Check reduction ratio** - Aim for less than 10% point reduction for conservative filtering
4. **Monitor processing time** - Balance quality vs. speed for your use case
5. **Test on representative tiles** - Don't tune on outliers

### Troubleshooting

**Problem: Too many points removed (>20% reduction)**

Solution:

- Increase `--sor-std` to 3.0 or higher
- Increase `--sor-k` to 15-20
- Increase `--ror-radius` to 1.5-2.0
- Decrease `--ror-neighbors` to 3

**Problem: Still see scan line artifacts**

Solution:

- Decrease `--sor-std` to 1.5
- Decrease `--sor-k` to 10
- Decrease `--ror-radius` to 0.8
- Increase `--ror-neighbors` to 5-6
- Add `--voxel-size 0.5`

**Problem: Processing too slow**

Solution:

- Disable voxel downsampling
- Increase `--sor-k` to reduce KDTree rebuilds
- Reduce `--num-workers` to avoid memory thrashing
- Consider processing in batches

**Problem: Out of memory errors**

Solution:

- Add `--voxel-size 0.5` or smaller
- Reduce `--num-workers`
- Process tiles individually (`--input` instead of `--input-dir`)
- Increase system swap space

## Examples

### Example 1: Basic Usage

```bash
# Process a single tile with preprocessing
ign-lidar-hd enrich \
  --input raw_tiles/tile_001.laz \
  --output enriched/ \
  --mode building \
  --preprocess
```

### Example 2: Batch Processing

```bash
# Process all tiles with conservative preprocessing
ign-lidar-hd enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --mode building \
  --preprocess \
  --sor-k 15 \
  --sor-std 3.0 \
  --num-workers 4
```

### Example 3: Memory-Constrained System

```bash
# Use voxel downsampling to reduce memory usage
ign-lidar-hd enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --mode building \
  --preprocess \
  --voxel-size 0.5 \
  --num-workers 2
```

### Example 4: High-Quality Building Extraction

```bash
# Conservative preprocessing + RGB for best quality
ign-lidar-hd enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --mode building \
  --preprocess \
  --sor-k 15 \
  --sor-std 2.5 \
  --ror-radius 1.0 \
  --ror-neighbors 4 \
  --add-rgb \
  --rgb-cache-dir cache/ \
  --num-workers 2
```

## Related Documentation

- [Artifact Analysis Report](../../../artifacts.md) - Detailed analysis of LiDAR artifacts
- [Implementation Guide](../../../PHASE1_SPRINT1_COMPLETE.md) - Technical implementation details
- [Integration Details](../../../PHASE1_SPRINT2_COMPLETE.md) - CLI and processor integration
- [CLI Commands](cli-commands.md) - Complete CLI reference
- [Python API](../api/processor.md) - Processor API documentation

## References

- PDAL Statistical Outlier Removal: https://pdal.io/stages/filters.outlier.html
- PDAL Radius Outlier Removal: https://pdal.io/stages/filters.radiusoutlier.html
- PCL Voxel Grid: https://pointclouds.org/documentation/classpcl_1_1_voxel_grid.html
