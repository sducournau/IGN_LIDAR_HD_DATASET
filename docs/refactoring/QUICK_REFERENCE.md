# Quick Reference Guide - Refactoring v3.6.0

**Last Updated:** November 21, 2025  
**Status:** All 4 phases complete ✅

---

## 🚀 For Developers: New APIs

### GPU Memory Management (Phase 1)

```python
from ign_lidar.core.gpu_memory import get_gpu_memory_manager

# Get singleton instance
gpu_mem = get_gpu_memory_manager()

# Check and allocate memory
if gpu_mem.allocate(size_gb=2.5):
    result = gpu_process(data)
    gpu_mem.deallocate(2.5)
else:
    result = cpu_fallback(data)

# Monitor usage
print(f"GPU used: {gpu_mem.get_used_memory():.2f} GB")
print(f"GPU available: {gpu_mem.get_available_memory():.2f} GB")

# Cleanup
gpu_mem.free_cache()
```

### FAISS Index Management (Phase 1)

```python
from ign_lidar.optimization.faiss_utils import create_faiss_index

# Automatic configuration
index, res = create_faiss_index(
    n_dims=3,
    n_points=1_000_000,
    use_gpu=True,
    metric='L2'
)

# Use index
distances, indices = index.search(queries, k=30)
```

### KNN Search (Phase 2)

```python
from ign_lidar.optimization import knn_search

# One-line KNN (automatic backend selection)
distances, indices = knn_search(points, k=30)

# With specific backend
distances, indices = knn_search(
    points,
    k=30,
    backend='faiss_gpu'  # or 'faiss_cpu', 'cuml', 'sklearn'
)

# Reusable engine
from ign_lidar.optimization import KNNEngine

engine = KNNEngine(backend='auto')
engine.fit(reference_points)
distances, indices = engine.search(query_points, k=30)
```

---

## 📚 Key Documents

### Getting Started

- **Start here:** [`README.md`](README.md) - Complete overview
- **Audit report:** [`../audit_reports/CODEBASE_AUDIT_NOV2025.md`](../audit_reports/CODEBASE_AUDIT_NOV2025.md)
- **Final report:** [`PHASES_1_4_FINAL_REPORT.md`](PHASES_1_4_FINAL_REPORT.md)

### Migration Guides

- **GPU & FAISS:** [`MIGRATION_GUIDE_PHASE1.md`](MIGRATION_GUIDE_PHASE1.md)
- **KNN Engine:** [`KNN_ENGINE_MIGRATION_GUIDE.md`](KNN_ENGINE_MIGRATION_GUIDE.md)

### Phase Reports

- **Phase 1:** [`PHASE1_COMPLETION_REPORT.md`](PHASE1_COMPLETION_REPORT.md) - GPU consolidation
- **Phase 2:** [`PHASE2_COMPLETION_REPORT.md`](PHASE2_COMPLETION_REPORT.md) - KNN unification
- **Phase 3:** [`PHASE3_ANALYSIS.md`](PHASE3_ANALYSIS.md) - Feature simplification
- **Phase 4:** [`PHASE4_COMPLETION_REPORT.md`](PHASE4_COMPLETION_REPORT.md) - Code quality validation

---

## 🔄 Migration Checklist

### Migrating Old Code

**1. GPU Memory Checks**

```python
# OLD (scattered checks)
if torch.cuda.is_available():
    device = torch.device('cuda')
    available_mem = torch.cuda.get_device_properties(0).total_memory
    # ... manual memory management

# NEW (centralized)
from ign_lidar.core.gpu_memory import get_gpu_memory_manager

gpu_mem = get_gpu_memory_manager()
if gpu_mem.allocate(required_gb):
    # ... GPU processing
    gpu_mem.deallocate(required_gb)
```

**2. FAISS Initialization**

```python
# OLD (manual configuration)
import faiss
if use_gpu:
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, d)
else:
    index = faiss.IndexFlatL2(d)

# NEW (automatic)
from ign_lidar.optimization.faiss_utils import create_faiss_index

index, res = create_faiss_index(n_dims=d, use_gpu=True)
```

**3. KNN Queries**

```python
# OLD (sklearn)
from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(n_neighbors=k)
knn.fit(points)
distances, indices = knn.kneighbors(points)

# NEW (unified)
from ign_lidar.optimization import knn_search

distances, indices = knn_search(points, k=k)
```

**4. Feature Computation**

```python
# OLD (direct sklearn import)
from sklearn.neighbors import NearestNeighbors
# ... feature computation with sklearn

# NEW (unified KNN)
from ign_lidar.optimization import knn_search
# ... feature computation with knn_search()
```

---

## 📊 Performance Tips

### GPU Optimization

```python
# Check GPU availability first
from ign_lidar.core.gpu_memory import get_gpu_memory_manager

gpu_mem = get_gpu_memory_manager()
if not gpu_mem.is_available():
    # Fallback to CPU
    use_gpu = False

# For large datasets, check memory before processing
required_gb = estimate_memory_requirement(data)
if gpu_mem.allocate(required_gb):
    result = gpu_process(data)
    gpu_mem.deallocate(required_gb)
else:
    # Use chunked processing or CPU fallback
    result = chunked_process(data)
```

### KNN Backend Selection

```python
from ign_lidar.optimization import knn_search

# Let automatic selection choose best backend
distances, indices = knn_search(points, k=30, backend='auto')

# Manual selection for specific needs:
# - 'faiss_gpu': Best for large datasets (>1M points) with GPU
# - 'faiss_cpu': Good for medium datasets (100K-1M) without GPU
# - 'cuml': Best for medium GPU datasets
# - 'sklearn': Reliable fallback for small datasets
```

### Feature Computation

```python
from ign_lidar.optimization import knn_search

# Reuse KNN results when computing multiple features
distances, indices = knn_search(points, k=30)

# Use the same indices for multiple feature computations
normals = compute_normals_from_indices(points, indices)
curvature = compute_curvature_from_indices(points, indices)
eigenvalues = compute_eigenvalues_from_indices(points, indices)
```

---

## 🐛 Common Issues & Solutions

### Issue: GPU Out of Memory

```python
# Solution 1: Check memory before allocation
from ign_lidar.core.gpu_memory import get_gpu_memory_manager

gpu_mem = get_gpu_memory_manager()
required = estimate_memory(data)

if not gpu_mem.allocate(required):
    # Fallback or chunk data
    result = process_in_chunks(data)
else:
    result = gpu_process(data)
    gpu_mem.deallocate(required)

# Solution 2: Free cache before large operations
gpu_mem.free_cache()
result = gpu_process(data)
```

### Issue: Slow KNN Performance

```python
# Solution 1: Use explicit backend selection
from ign_lidar.optimization import knn_search

# For large datasets with GPU
distances, indices = knn_search(points, k=30, backend='faiss_gpu')

# Solution 2: Reuse KNN engine for multiple queries
from ign_lidar.optimization import KNNEngine

engine = KNNEngine(backend='faiss_gpu')
engine.fit(reference_points)

# Multiple queries reuse the fitted index
for query_batch in query_batches:
    distances, indices = engine.search(query_batch, k=30)
```

### Issue: ImportError After Refactoring

```python
# Old imports (deprecated but still work)
from ign_lidar.features.compute.features import compute_normals  # ⚠️ Deprecated

# New imports (recommended)
from ign_lidar.features.compute.normals import compute_normals  # ✅ Correct
from ign_lidar.optimization import knn_search  # ✅ New API
```

---

## 🧪 Testing

### Unit Tests

```bash
# Test GPU memory manager
pytest tests/test_gpu_memory_refactoring.py -v

# Test KNN engine
pytest tests/test_knn_engine.py -v

# Test feature computation (integration)
pytest tests/test_feature_*.py -v -m integration
```

### Benchmarking

```python
from ign_lidar.optimization import KNNEngine
import time

# Benchmark different backends
backends = ['faiss_gpu', 'faiss_cpu', 'cuml', 'sklearn']
points = np.random.randn(100_000, 3)

for backend in backends:
    engine = KNNEngine(backend=backend)

    start = time.time()
    engine.fit(points)
    distances, indices = engine.search(points, k=30)
    duration = time.time() - start

    print(f"{backend}: {duration:.3f}s")
```

---

## 📞 Help & Support

### Documentation

- **API docs:** See docstrings in code
- **Architecture:** See phase completion reports
- **Examples:** See migration guides

### Issues

- **GitHub Issues:** Label with `refactoring` or `performance`
- **Questions:** Reference specific phase report

### Updates

- **Version 3.6.0:** All 4 phases complete
- **Version 4.0.0:** Breaking changes planned (Q2 2026)
- **Changelog:** See `CHANGELOG.md`

---

## ✅ Checklist: Am I Using New APIs?

**Before running production code, verify:**

- [ ] No direct `sklearn.neighbors` imports in feature code
- [ ] GPU memory checks use `GPUMemoryManager`
- [ ] FAISS indices created via `create_faiss_index()`
- [ ] KNN queries use `knn_search()` or `KNNEngine`
- [ ] No manual GPU memory calculations
- [ ] Backward compatibility warnings addressed

---

**Quick Start Command:**

```python
# One-liner to verify new APIs are available
from ign_lidar.optimization import knn_search, KNNEngine
from ign_lidar.core.gpu_memory import get_gpu_memory_manager
from ign_lidar.optimization.faiss_utils import create_faiss_index

print("✅ All new APIs available!")
```

---

**Last Updated:** November 21, 2025  
**Version:** 3.6.0-dev  
**Status:** Production ready ✅
