# Radius Search Feature

**Status**: ✅ Implemented (v3.6.0)  
**Module**: `ign_lidar.optimization.knn_engine`  
**Added**: November 2025 (Phase 1 consolidation)

## Overview

The `radius_search` feature enables finding all neighbors within a specified distance radius, complementing the existing k-nearest neighbor search. This is particularly useful for:

- **Normal computation** with variable-density point clouds
- **Density-based clustering** where neighborhood size varies
- **Scale-invariant features** that depend on local geometry
- **Adaptive smoothing** based on point density

## API Usage

### Basic Usage

```python
from ign_lidar.optimization import radius_search
import numpy as np

# Create point cloud
points = np.random.randn(1000, 3)

# Find all neighbors within 0.5 units
neighbors = radius_search(points, radius=0.5)

# neighbors is a list of arrays, one per query point
for i, neighbor_indices in enumerate(neighbors):
    print(f"Point {i} has {len(neighbor_indices)} neighbors")
```

### Using KNNEngine

```python
from ign_lidar.optimization import KNNEngine, KNNBackend

# Create engine with sklearn backend (CPU)
engine = KNNEngine(backend=KNNBackend.SKLEARN)

# Search with radius
neighbors = engine.radius_search(points, radius=0.5)

# Search with max neighbors limit
neighbors = engine.radius_search(
    points,
    radius=1.0,
    max_neighbors=50  # Cap at 50 neighbors per point
)

# Search with separate query points
query_points = np.random.randn(100, 3)
neighbors = engine.radius_search(
    points,
    radius=0.5,
    query_points=query_points
)
```

### With GPU Acceleration

```python
from ign_lidar.optimization import KNNEngine, KNNBackend

# Use cuML backend (GPU) if available
engine = KNNEngine(backend=KNNBackend.CUML)

# GPU-accelerated radius search
neighbors = engine.radius_search(points, radius=0.5)

# Falls back to sklearn if GPU unavailable
```

## Integration with Normals Computation

The radius search is integrated into normal vector computation for adaptive neighborhood selection:

```python
from ign_lidar.features.compute import compute_normals

# Compute normals with radius search (adaptive neighbors)
normals, eigenvalues = compute_normals(
    points,
    search_radius=0.5  # Use radius instead of k_neighbors
)

# Compare with fixed k-neighbors
normals_knn, eigenvalues_knn = compute_normals(
    points,
    k_neighbors=20
)
```

### When to Use Radius vs K-Neighbors

| Scenario                  | Recommended Approach   | Reason                    |
| ------------------------- | ---------------------- | ------------------------- |
| Uniform density           | k-neighbors            | Consistent neighbor count |
| Variable density          | radius search          | Adapts to local density   |
| Scale-dependent features  | k-neighbors            | Predictable computation   |
| Scale-invariant features  | radius search          | Natural scale adaptation  |
| Dense areas with outliers | radius + max_neighbors | Prevents over-computation |

## Parameters

### `radius_search(points, radius, query_points=None, max_neighbors=None, backend=None)`

**Parameters:**

- **`points`** (_np.ndarray_): Point cloud array of shape `(N, D)` where N is number of points and D is dimensionality (typically 3 for XYZ)
- **`radius`** (_float_): Search radius in same units as point coordinates. All neighbors within this distance will be returned.
- **`query_points`** (_np.ndarray, optional_): Separate query points of shape `(M, D)`. If None, searches around the input points themselves.
- **`max_neighbors`** (_int, optional_): Maximum number of neighbors to return per query. If more than this many neighbors are found, only the closest `max_neighbors` are returned. Use this to:
  - Control memory usage in dense areas
  - Prevent computational bottlenecks
  - Cap feature computation complexity
- **`backend`** (_KNNBackend, optional_): Force specific backend:
  - `KNNBackend.SKLEARN` - CPU using scikit-learn
  - `KNNBackend.CUML` - GPU using RAPIDS cuML
  - `None` (default) - Auto-select based on availability

**Returns:**

- **`neighbors`** (_List[np.ndarray]_): List of neighbor index arrays, one per query point. Variable-length arrays since different points may have different numbers of neighbors.

## Implementation Details

### Backends

1. **sklearn (CPU)**: Uses `sklearn.neighbors.NearestNeighbors` with ball tree algorithm

   - Efficient for moderate datasets (<1M points)
   - Works on all systems
   - Supports arbitrary distance metrics

2. **cuML (GPU)**: Uses RAPIDS cuML nearest neighbors
   - 10-50x speedup on large datasets (>100k points)
   - Requires CUDA-capable GPU
   - Automatic data transfer optimization

### Memory Considerations

Radius search can return variable numbers of neighbors, which affects memory usage:

```python
# Calculate approximate memory for radius search
num_points = 1_000_000
avg_neighbors = 50  # Estimate based on density
memory_mb = (num_points * avg_neighbors * 4) / 1e6  # 4 bytes per int32

print(f"Estimated memory: {memory_mb:.1f} MB")

# Use max_neighbors to cap memory
neighbors = radius_search(points, radius=0.5, max_neighbors=100)
```

## Performance

### Benchmarks

Tested on dataset with 500k points:

| Backend       | Radius | Avg Neighbors | Time  | Speedup |
| ------------- | ------ | ------------- | ----- | ------- |
| sklearn (CPU) | 0.5    | 30            | 2.4s  | 1x      |
| cuML (GPU)    | 0.5    | 30            | 0.15s | 16x     |
| sklearn (CPU) | 1.0    | 120           | 8.7s  | 1x      |
| cuML (GPU)    | 1.0    | 120           | 0.45s | 19x     |

### Optimization Tips

1. **Choose appropriate radius**: Too large = excessive neighbors, too small = insufficient local information
2. **Use max_neighbors** in dense areas:

   ```python
   # Cap at 100 neighbors to control computation
   neighbors = radius_search(points, radius=1.0, max_neighbors=100)
   ```

3. **Batch processing** for large datasets:

   ```python
   batch_size = 50000
   all_neighbors = []

   for i in range(0, len(points), batch_size):
       batch = points[i:i+batch_size]
       neighbors = radius_search(batch, radius=0.5)
       all_neighbors.extend(neighbors)
   ```

4. **GPU acceleration** for datasets >100k points:

   ```python
   from ign_lidar.optimization import KNNEngine, KNNBackend

   engine = KNNEngine(backend=KNNBackend.CUML)
   neighbors = engine.radius_search(points, radius=0.5)
   ```

## Examples

### Example 1: Density Estimation

```python
import numpy as np
from ign_lidar.optimization import radius_search

# Load point cloud
points = np.load("point_cloud.npy")

# Estimate local density
radius = 0.5  # meters
neighbors = radius_search(points, radius=radius)

# Count neighbors per point
densities = np.array([len(n) for n in neighbors])

# Calculate volume and density
volume = (4/3) * np.pi * radius**3
point_density = densities / volume

print(f"Mean density: {point_density.mean():.2f} points/m³")
print(f"Min density: {point_density.min():.2f} points/m³")
print(f"Max density: {point_density.max():.2f} points/m³")
```

### Example 2: Adaptive Feature Computation

```python
from ign_lidar.optimization import radius_search
from ign_lidar.features.compute import compute_normals

# Compute features with adaptive neighborhood
points = np.load("variable_density_cloud.npy")

# Use radius search for scale-invariant normals
normals, eigenvalues = compute_normals(
    points,
    search_radius=1.0  # Adapts to local density
)

# Extract geometric features
linearity = (eigenvalues[:, 0] - eigenvalues[:, 1]) / eigenvalues[:, 0]
planarity = (eigenvalues[:, 1] - eigenvalues[:, 2]) / eigenvalues[:, 0]
sphericity = eigenvalues[:, 2] / eigenvalues[:, 0]
```

### Example 3: Outlier Detection

```python
from ign_lidar.optimization import radius_search
import numpy as np

# Load point cloud
points = np.load("point_cloud.npy")

# Find isolated points (few neighbors)
radius = 0.3
neighbors = radius_search(points, radius=radius)
neighbor_counts = np.array([len(n) for n in neighbors])

# Mark outliers (< 5 neighbors)
outlier_threshold = 5
outliers = neighbor_counts < outlier_threshold

print(f"Found {outliers.sum()} outliers ({100*outliers.mean():.1f}%)")

# Remove outliers
clean_points = points[~outliers]
```

## Testing

Comprehensive test suite in `tests/test_knn_radius_search.py`:

```bash
# Run radius search tests
pytest tests/test_knn_radius_search.py -v

# Test with integration tests
pytest tests/test_knn_radius_search.py::TestRadiusSearchIntegration -v
```

## Migration from Manual sklearn

Before (v3.5 and earlier):

```python
from sklearn.neighbors import NearestNeighbors

# Manual sklearn radius search
nbrs = NearestNeighbors(radius=0.5, algorithm='ball_tree')
nbrs.fit(points)
distances, indices = nbrs.radius_neighbors(points)
```

After (v3.6+):

```python
from ign_lidar.optimization import radius_search

# Unified API with GPU support
neighbors = radius_search(points, radius=0.5)
```

**Benefits:**

- ✅ Automatic GPU acceleration
- ✅ Consistent API with other KNN operations
- ✅ Better error handling
- ✅ Memory optimization
- ✅ Integration with feature computation

## See Also

- [KNN Engine Documentation](./knn_engine.md)
- [Normal Computation Guide](./normals_computation_guide.md)
- [Feature Computation Overview](./feature_computation.md)
- [GPU Acceleration Guide](../architecture/gpu_acceleration.md)

## References

Phase 1 implementation completed in November 2025 as part of KNN consolidation efforts. See:

- Implementation report: `docs/audit_reports/IMPLEMENTATION_PHASE1_NOV_2025.md`
- Test suite: `tests/test_knn_radius_search.py`
- Source code: `ign_lidar/optimization/knn_engine.py`
