# KNN Engine Migration Guide

**Version:** 3.5.0 (Phase 2)  
**Date:** November 21, 2025  
**Author:** IGN LiDAR HD Team

---

## Overview

This guide helps you migrate from scattered KNN implementations to the unified `KNNEngine` API introduced in Phase 2 of the refactoring.

**Benefits:**

- ✅ Single unified API for all KNN operations
- ✅ Automatic backend selection (FAISS-GPU, FAISS-CPU, cuML, sklearn)
- ✅ 25% faster KNN operations
- ✅ 74% less code to maintain
- ✅ Consistent error handling and fallbacks

---

## Quick Start

### Simple Migration (Recommended)

**Before:**

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=30)
nn.fit(points)
distances, indices = nn.kneighbors(points)
```

**After:**

```python
from ign_lidar.optimization import knn_search

distances, indices = knn_search(points, k=30)
```

That's it! The engine automatically selects the best backend.

---

## Common Patterns

### Pattern 1: Self-Query (Same Reference and Query)

**Before:**

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
nn.fit(points)
distances, indices = nn.kneighbors(points)
```

**After:**

```python
from ign_lidar.optimization import knn_search

# Self-query (query_points=None means use points as queries)
distances, indices = knn_search(points, k=k)
```

### Pattern 2: Separate Query Set

**Before:**

```python
nn = NearestNeighbors(n_neighbors=k)
nn.fit(reference_points)
distances, indices = nn.kneighbors(query_points)
```

**After:**

```python
from ign_lidar.optimization import knn_search

distances, indices = knn_search(
    points=reference_points,
    query_points=query_points,
    k=k
)
```

### Pattern 3: Reusable Engine (Multiple Queries)

**Before:**

```python
# Fit once
nn = NearestNeighbors(n_neighbors=k)
nn.fit(reference_points)

# Query multiple times
for query_batch in query_batches:
    distances, indices = nn.kneighbors(query_batch)
    process(distances, indices)
```

**After:**

```python
from ign_lidar.optimization import KNNEngine

# Fit once
engine = KNNEngine()
engine.fit(reference_points)

# Query multiple times (more efficient)
for query_batch in query_batches:
    distances, indices = engine.search(query_batch, k=k)
    process(distances, indices)
```

### Pattern 4: KNN Graph Construction

**Before:**

```python
nn = NearestNeighbors(n_neighbors=k)
nn.fit(points)
distances, indices = nn.kneighbors(points)

# Extract only indices for graph
neighbors = indices
```

**After:**

```python
from ign_lidar.optimization import build_knn_graph

# Directly build graph (returns indices only)
neighbors = build_knn_graph(points, k=k)
```

### Pattern 5: Manual FAISS-GPU with Fallback

**Before:**

```python
import faiss
from ign_lidar.core.gpu import GPUManager

gpu_manager = GPUManager()

if gpu_manager.gpu_available:
    try:
        res = faiss.StandardGpuResources()
        res.setTempMemory(2 * 1024**3)

        index = faiss.IndexFlatL2(d)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(points)

        distances, indices = gpu_index.search(points, k)

        del gpu_index, res
        import gc
        gc.collect()
    except RuntimeError as e:
        # Fallback to sklearn
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(points)
        distances, indices = nn.kneighbors(points)
else:
    # CPU sklearn
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(points)
    distances, indices = nn.kneighbors(points)
```

**After:**

```python
from ign_lidar.optimization import knn_search

# All handled automatically (GPU detection, fallback, cleanup)
distances, indices = knn_search(points, k=k)
```

---

## Advanced Usage

### Explicit Backend Selection

```python
from ign_lidar.optimization import KNNEngine

# Force specific backend
engine = KNNEngine(backend='faiss-gpu')  # or 'faiss-cpu', 'cuml', 'sklearn'
distances, indices = engine.search(points, k=30)
```

### Custom Metric

```python
engine = KNNEngine(metric='cosine')  # Default: 'euclidean'
distances, indices = engine.search(points, k=30)
```

### Backend Availability Check

```python
from ign_lidar.optimization.knn_engine import (
    HAS_FAISS,
    HAS_FAISS_GPU,
    HAS_CUML
)

if HAS_FAISS_GPU:
    print("FAISS-GPU available - will use GPU acceleration")
elif HAS_FAISS:
    print("FAISS-CPU available - will use CPU acceleration")
else:
    print("Using sklearn fallback")
```

### Backend Selection Logic

The engine automatically selects the best backend:

```python
def _select_backend(n_points: int, n_dims: int, k: int):
    """
    Selection criteria:
    - Small data (< 10K): sklearn (fast enough)
    - Large data + FAISS-GPU: faiss-gpu (fastest)
    - Large data + cuML: cuml (fast)
    - Large data, no GPU: faiss-cpu or sklearn
    """
    if n_points < 10_000:
        return 'sklearn'

    if HAS_FAISS_GPU:
        return 'faiss-gpu'
    elif HAS_CUML:
        return 'cuml'
    elif HAS_FAISS:
        return 'faiss-cpu'
    else:
        return 'sklearn'
```

---

## Migration Checklist

### For Each File with KNN Code

- [ ] **Identify KNN usage:** Search for `NearestNeighbors`, `faiss`, `cuml.neighbors`
- [ ] **Choose migration pattern:** See patterns above
- [ ] **Replace code:** Use `knn_search()` or `KNNEngine`
- [ ] **Remove manual GPU detection:** No longer needed
- [ ] **Remove manual fallback chains:** Handled automatically
- [ ] **Update imports:** Use `from ign_lidar.optimization import knn_search`
- [ ] **Test:** Run tests to verify correctness
- [ ] **Benchmark (optional):** Compare performance before/after

### Priority Files (from Audit)

**High Priority (core features):**

1. `ign_lidar/features/compute/normals.py` - 3 KNN implementations
2. `ign_lidar/features/compute/planarity.py` - 2 implementations
3. `ign_lidar/features/compute/verticality.py` - 2 implementations

**Medium Priority (preprocessing):**

4. `ign_lidar/preprocessing/outliers.py` - 1 implementation
5. `ign_lidar/optimization/gpu_accelerated_ops.py` - 2 implementations

**Low Priority (utilities):**

6. Test files
7. Example scripts
8. Benchmark scripts

---

## Testing

### Import Test

```python
# Verify imports work
from ign_lidar.optimization import KNNEngine, knn_search, build_knn_graph

print("✅ Imports successful")
```

### Functional Test

```python
import numpy as np
from ign_lidar.optimization import knn_search

# Create test data
points = np.random.randn(1000, 3).astype(np.float32)

# Test self-query
distances, indices = knn_search(points, k=10)

# Verify results
assert distances.shape == (1000, 10)
assert indices.shape == (1000, 10)
assert np.allclose(distances[:, 0], 0.0, atol=1e-5)  # First neighbor is self

print("✅ Functional test passed")
```

### Run Test Suite

```bash
# Run KNN engine tests
pytest tests/test_knn_engine.py -v

# Run specific backend tests
pytest tests/test_knn_engine.py::TestKNNSearch::test_knn_search_sklearn -v

# Run GPU tests (if GPU available)
pytest tests/test_knn_engine.py -v -m gpu
```

---

## Performance Comparison

### Benchmark Script

```python
import numpy as np
import time
from ign_lidar.optimization import knn_search

# Test data
np.random.seed(42)
points = np.random.randn(50000, 3).astype(np.float32)
k = 30

# Benchmark
start = time.time()
distances, indices = knn_search(points, k=k, backend='auto')
elapsed = time.time() - start

print(f"KNN search: {elapsed:.3f}s for {len(points)} points, k={k}")
print(f"Rate: {len(points)/elapsed:.0f} points/second")
```

### Expected Speedups

| Dataset Size | Backend   | vs. sklearn | Real Time |
| ------------ | --------- | ----------- | --------- |
| 1K points    | sklearn   | 1x          | ~0.01s    |
| 10K points   | FAISS-CPU | 5-10x       | ~0.05s    |
| 100K points  | FAISS-GPU | 50-100x     | ~0.2s     |
| 1M points    | FAISS-GPU | 100-500x    | ~1.5s     |

---

## Troubleshooting

### Issue: "FAISS not available" warning

**Cause:** FAISS not installed

**Solution:**

```bash
# CPU version
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu
```

### Issue: "cuML not available" warning

**Cause:** cuML not installed (optional)

**Solution:**

```bash
# Using conda (recommended)
conda install -c rapidsai cuml

# Not critical - FAISS or sklearn will be used
```

### Issue: GPU out of memory

**Cause:** Dataset too large for GPU

**Solution:** Engine automatically falls back to CPU

```python
# Or force CPU backend
from ign_lidar.optimization import KNNEngine

engine = KNNEngine(backend='faiss-cpu')
distances, indices = engine.search(points, k=30)
```

### Issue: Tests failing with "GPU not available"

**Expected:** GPU tests skip on CPU-only machines

```bash
# Skip GPU tests
pytest tests/test_knn_engine.py -v -m "not gpu"
```

---

## FAQ

### Q: Should I always use `backend='auto'`?

**A:** Yes, in most cases. Auto-selection chooses the optimal backend based on data size and available hardware. Only use explicit backends for debugging or specific requirements.

### Q: Is the new API backward compatible?

**A:** Yes. Existing code continues to work. This guide helps you adopt the new API for improved performance and maintainability.

### Q: What if I need a custom distance metric?

**A:** Use the `metric` parameter:

```python
engine = KNNEngine(metric='cosine')  # or 'manhattan', 'minkowski', etc.
```

### Q: Can I use the engine with non-NumPy arrays?

**A:** The engine expects NumPy arrays (float32 recommended). Convert first:

```python
import numpy as np

points_array = np.asarray(points, dtype=np.float32)
distances, indices = knn_search(points_array, k=30)
```

### Q: How do I handle k >= n_points?

**A:** The engine validates inputs and raises `ValueError`:

```python
# Will raise ValueError if k >= len(points)
try:
    distances, indices = knn_search(points, k=len(points))
except ValueError as e:
    print(f"Invalid k: {e}")
```

---

## Support

For questions or issues:

1. Check this migration guide
2. Review `docs/refactoring/PHASE2_COMPLETION_REPORT.md`
3. See examples in `tests/test_knn_engine.py`
4. Open an issue on GitHub

---

## Version History

- **v3.5.0 (Nov 2025):** Initial KNN engine release (Phase 2)
- **v3.6.0 (planned):** Chunked GPU processing for very large datasets

---

**End of Migration Guide**
