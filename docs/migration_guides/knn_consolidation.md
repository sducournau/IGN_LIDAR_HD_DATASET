# Migration Guide: Unified KNN Engine

## Overview

Multiple KNN implementations have been consolidated into `KNNEngine` for
consistent performance and easier maintenance.

## Before (v3.5.x - Multiple Implementations)

```python
# Different KNN approaches scattered across codebase
from ign_lidar.optimization.gpu_accelerated_ops import compute_knn_gpu
from ign_lidar.io.formatters.hybrid_formatter import find_neighbors

# Different APIs, different behavior
indices, distances = compute_knn_gpu(points, k=30)
neighbors = find_neighbors(points, radius=3.0)
```

## After (v3.6.0+ - Unified API)

```python
from ign_lidar.optimization import KNNEngine

# Create engine once
knn_engine = KNNEngine(
    backend='auto',  # Auto-select: FAISS-GPU, cuML, or sklearn
    use_gpu=True
)

# Build index
knn_engine.build_index(points)

# Search (consistent API)
distances, indices = knn_engine.search(points, k=30)

# Radius search
distances, indices = knn_engine.radius_search(points, radius=3.0)

# Lazy GPU transfers (Phase 2 optimization)
distances_gpu, indices_gpu = knn_engine.search(
    points, 
    k=30, 
    return_gpu=True  # Keep results on GPU
)
```

## Key Features

1. **Auto-Selection**: Automatically picks best backend (FAISS-GPU > cuML > sklearn)
2. **Consistent API**: Same interface regardless of backend
3. **GPU Optimization**: Lazy transfers reduce CPU↔GPU bottlenecks
4. **Fallback**: Graceful CPU fallback if GPU unavailable
5. **Performance**: Optimized for large point clouds

## Migration Checklist

- [ ] Replace custom KNN calls with `KNNEngine`
- [ ] Update to unified search API
- [ ] Enable `return_gpu=True` for GPU pipelines
- [ ] Remove duplicate KNN implementations
- [ ] Test performance benchmarks

## Performance Tips

```python
# ✅ Good: Keep data on GPU
knn_engine.build_index(points_gpu)
distances, indices = knn_engine.search(points_gpu, k=30, return_gpu=True)
features = compute_features_gpu(points_gpu, indices)  # No transfer!

# ❌ Bad: Unnecessary transfers
knn_engine.build_index(points_gpu)
distances, indices = knn_engine.search(points_gpu, k=30)  # Transfer to CPU
features = compute_features_gpu(cp.asarray(points), cp.asarray(indices))  # Transfer back
```

## Timeline

- **v3.6.0**: Unified KNNEngine available, old APIs deprecated
- **v3.7.0-3.9.0**: Deprecation warnings
- **v4.0.0**: Old KNN implementations removed
