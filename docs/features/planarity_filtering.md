# Planarity Artifact Reduction Guide

## Overview

This guide explains how to use the new planarity filtering functionality to reduce line/dash artifacts in planarity features.

## Problem Description

**Artifacts in planarity features** typically appear as lines or dashes along object boundaries. These occur when:

1. **Neighborhood boundary crossing**: The k-NN neighborhood spans across different surfaces (e.g., wall→air, ground→building)
2. **Sparse regions**: Insufficient neighbors lead to unstable covariance matrices
3. **Numerical issues**: NaN/Inf values from division by very small eigenvalues

### Example of Artifacts

```
Normal planarity:     Planarity with artifacts:
┌──────────┐          ┌──────────┐
│ 0.9 0.9  │          │ 0.9 0.9  │
│ 0.9 0.9  │          │ 0.9 0.2  │ ← Artifact line
│ 0.9 0.9  │          │ 0.9 0.9  │
└──────────┘          └──────────┘
```

## Solutions

### Solution 1: Post-Processing with Spatial Filtering (Recommended)

Use `smooth_planarity_spatial()` to apply adaptive spatial filtering **after** feature computation.

#### Basic Usage

```python
from ign_lidar.features.compute import (
    compute_planarity,
    smooth_planarity_spatial
)
import numpy as np

# 1. Compute planarity (standard way)
planarity = compute_planarity(eigenvalues)

# 2. Apply spatial smoothing to reduce artifacts
smoothed_planarity, stats = smooth_planarity_spatial(
    planarity=planarity,
    points=points,
    k_neighbors=15,           # Number of neighbors for filtering
    std_threshold=0.3,        # Variance threshold for artifact detection
    min_valid_neighbors=5     # Min neighbors for interpolation
)

# 3. Check statistics
print(f"Artifacts fixed: {stats['n_artifacts_fixed']}")
print(f"NaN/Inf fixed: {stats['n_nan_fixed']}")
print(f"Unchanged: {stats['n_unchanged']}")
```

#### Advanced Usage with Validation

```python
from ign_lidar.features.compute import (
    smooth_planarity_spatial,
    validate_planarity
)

# Step 1: Validate planarity (handles NaN/Inf, clips range)
validated_planarity, val_stats = validate_planarity(
    planarity,
    clip_outliers=True,
    sigma=3.0  # Clip beyond 3 std dev
)

print(f"NaN values: {val_stats['n_nan']}")
print(f"Inf values: {val_stats['n_inf']}")
print(f"Out of range: {val_stats['n_out_of_range']}")

# Step 2: Apply spatial smoothing
smoothed_planarity, smooth_stats = smooth_planarity_spatial(
    validated_planarity,
    points,
    k_neighbors=20,       # Larger neighborhood for smoother results
    std_threshold=0.25,   # Lower threshold = more aggressive filtering
)
```

## Integration with LiDARProcessor

### Option A: Add to Feature Orchestrator (Recommended)

Modify your feature computation pipeline to automatically apply filtering:

```python
# In your processing code
from ign_lidar import LiDARProcessor

processor = LiDARProcessor(config_path="config.yaml")

# Add filtering option to config
processor.config.features.planarity_filtering = {
    "enabled": True,
    "k_neighbors": 15,
    "std_threshold": 0.3
}
```

### Option B: Manual Integration

Apply filtering after feature computation:

```python
from ign_lidar.features.compute import smooth_planarity_spatial

# After computing features
features = processor.compute_features(points)

# Filter planarity
if "planarity" in features:
    features["planarity"], stats = smooth_planarity_spatial(
        features["planarity"],
        points,
        k_neighbors=15
    )

    if stats["n_artifacts_fixed"] > 0:
        logger.info(
            f"Reduced {stats['n_artifacts_fixed']} planarity artifacts"
        )
```

## Parameter Tuning

### `k_neighbors`

Controls the size of the spatial neighborhood used for filtering.

- **Small values (5-10)**: Preserves local detail, less aggressive smoothing
- **Medium values (15-20)**: Balanced (recommended for most cases)
- **Large values (25-50)**: More aggressive smoothing, may over-smooth

```python
# Conservative (preserves detail)
smoothed, _ = smooth_planarity_spatial(planarity, points, k_neighbors=10)

# Balanced (default)
smoothed, _ = smooth_planarity_spatial(planarity, points, k_neighbors=15)

# Aggressive (strong smoothing)
smoothed, _ = smooth_planarity_spatial(planarity, points, k_neighbors=30)
```

### `std_threshold`

Controls artifact detection sensitivity based on neighborhood variance.

- **Low values (0.1-0.2)**: Aggressive artifact detection, more smoothing
- **Medium values (0.3-0.4)**: Balanced (recommended)
- **High values (0.5+)**: Conservative, only fixes obvious artifacts

```python
# Aggressive (more artifacts detected)
smoothed, _ = smooth_planarity_spatial(
    planarity, points, std_threshold=0.2
)

# Conservative (fewer artifacts detected)
smoothed, _ = smooth_planarity_spatial(
    planarity, points, std_threshold=0.5
)
```

### `min_valid_neighbors`

Minimum number of valid neighbors required for interpolation of NaN/Inf values.

- **Default: 5** (works well for most cases)
- **Increase for dense point clouds** (7-10)
- **Decrease for sparse data** (3-4)

## Performance Considerations

### Memory Usage

Spatial filtering requires building a KD-tree, which uses additional memory:

```python
# Memory estimate: ~50-100 bytes per point
n_points = 1_000_000
memory_mb = n_points * 75 / 1024 / 1024  # ~71 MB
```

### Computational Cost

Time complexity: **O(N _ k _ log(N))**

- N = number of points
- k = k_neighbors
- log(N) = KD-tree query cost

```python
# Benchmark (approximate)
# 100k points, k=15: ~0.5-1.0 seconds
# 1M points, k=15: ~5-10 seconds
# 10M points, k=15: ~50-100 seconds
```

### Optimization Tips

1. **Use smaller k for large datasets**
2. **Apply filtering only where needed** (e.g., building facades)
3. **Cache KD-tree if filtering multiple features**

```python
from scipy.spatial import cKDTree

# Build tree once, reuse for multiple features
tree = cKDTree(points)

# Then modify smooth_planarity_spatial to accept pre-built tree
# (requires code modification)
```

## When to Use Filtering

### ✅ Use filtering when:

- You observe line/dash artifacts in visualization
- Ground truth refinement reports many NaN/Inf warnings
- Classification struggles near object boundaries
- Working with building facades/edges

### ❌ Don't use filtering when:

- Planarity already looks clean
- You need maximum computational speed
- Working with very sparse point clouds (< 10 pts/m²)

## Visualization

### Before/After Comparison

```python
import matplotlib.pyplot as plt

# Before filtering
plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], c=planarity, cmap='viridis', s=1)
plt.title('Original Planarity')
plt.colorbar()

# After filtering
smoothed, _ = smooth_planarity_spatial(planarity, points)
plt.subplot(1, 2, 2)
plt.scatter(points[:, 0], points[:, 1], c=smoothed, cmap='viridis', s=1)
plt.title('Filtered Planarity')
plt.colorbar()

plt.show()
```

## Troubleshooting

### Problem: Filtering is too aggressive

**Solution**: Increase `std_threshold` or decrease `k_neighbors`

```python
smoothed, _ = smooth_planarity_spatial(
    planarity, points,
    std_threshold=0.5,  # More conservative
    k_neighbors=10       # Smaller neighborhood
)
```

### Problem: Artifacts still present

**Solution**: Decrease `std_threshold` or increase `k_neighbors`

```python
smoothed, _ = smooth_planarity_spatial(
    planarity, points,
    std_threshold=0.2,  # More aggressive
    k_neighbors=25       # Larger neighborhood
)
```

### Problem: NaN values remain after filtering

**Solution**: Check `min_valid_neighbors` and data quality

```python
# Debug NaN issues
mask_nan = ~np.isfinite(smoothed)
if np.any(mask_nan):
    print(f"Remaining NaN: {np.sum(mask_nan)}")

    # Check neighbor counts
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    distances, _ = tree.query(points[mask_nan], k=15)
    print(f"Avg neighbor distance: {distances.mean():.3f}")
```

## References

- **Module**: `ign_lidar/features/compute/planarity_filter.py`
- **Tests**: `tests/test_planarity_filtering.py`
- **Related**: `ign_lidar/core/classification/ground_truth_refinement.py` (uses validation)

## Future Enhancements

Potential improvements being considered:

1. **GPU acceleration** for large-scale filtering
2. **Adaptive k_neighbors** based on local point density
3. **Edge-preserving filtering** (bilateral filtering)
4. **Automatic parameter selection** based on data characteristics

---

**Version**: 3.0.6  
**Author**: Simon Ducournau  
**Date**: October 30, 2025
