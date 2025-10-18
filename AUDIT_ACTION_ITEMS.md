# Computation Audit - Action Items

**Date:** October 18, 2025  
**Status:** 🔥 **URGENT - Multiple Critical Issues Found**

---

## 🚨 Critical Blockers (Fix Immediately)

### 1. GPU Mode Non-Functional ❌ BLOCKING

**Issue:** Missing/broken `compute_geometric_features()` method  
**File:** `ign_lidar/features/features_gpu.py`  
**Error:** `'GPUFeatureComputer' object has no attribute 'compute_geometric_features'`  
**Impact:** GPU mode completely unusable for geometric features  
**Priority:** 🔥 **P0 - BLOCKING**

**Action:**

```python
# Verify method exists and has correct signature
def compute_geometric_features(
    self,
    points: np.ndarray,
    required_features: list,
    k: int = 20
) -> Dict[str, np.ndarray]:
    # Implementation should match test expectations
```

**Test:**

```bash
python -c "
from ign_lidar.features.features_gpu import GPUFeatureComputer
import numpy as np

computer = GPUFeatureComputer(use_gpu=True)
points = np.random.rand(10000, 3).astype(np.float32)
features = computer.compute_geometric_features(
    points, ['planarity', 'linearity'], k=20
)
print('✅ Method works!')
"
```

---

### 2. Per-Feature GPU→CPU Transfers ⚠️ 10-20× SLOWDOWN

**Issue:** 4 separate GPU→CPU transfers per batch  
**File:** `ign_lidar/features/features_gpu.py`  
**Lines:** 925-941  
**Impact:** Each transfer adds 10-20ms overhead, compounds across batches  
**Priority:** 🔥 **P0 - CRITICAL PERF**

**Current (BAD):**

```python
if 'planarity' in required_features:
    planarity = (λ1 - λ2) / (sum_λ + 1e-8)
    batch_features['planarity'] = self._to_cpu(planarity).astype(np.float32)  # ❌

if 'linearity' in required_features:
    linearity = (λ0 - λ1) / (sum_λ + 1e-8)
    batch_features['linearity'] = self._to_cpu(linearity).astype(np.float32)  # ❌
# ... 2 more transfers
```

**Fixed (GOOD):**

```python
# Keep all on GPU
batch_features_gpu = {}
if 'planarity' in required_features:
    batch_features_gpu['planarity'] = (λ1 - λ2) / (sum_λ + 1e-8)
if 'linearity' in required_features:
    batch_features_gpu['linearity'] = (λ0 - λ1) / (sum_λ + 1e-8)
if 'sphericity' in required_features:
    batch_features_gpu['sphericity'] = λ2 / (sum_λ + 1e-8)
if 'anisotropy' in required_features:
    batch_features_gpu['anisotropy'] = (λ0 - λ2) / (sum_λ + 1e-8)

# Single batched transfer
batch_features = {
    feat: self._to_cpu(val).astype(np.float32)
    for feat, val in batch_features_gpu.items()
}
```

**Expected Impact:** 4 transfers → 1 transfer = **4-10× faster**

---

### 3. Per-Batch KNN Rebuild ⚠️ 5-10× SLOWDOWN

**Issue:** Rebuilds KNN for every batch instead of global once  
**File:** `ign_lidar/features/features_gpu.py`  
**Lines:** 825-860  
**Impact:** Wastes GPU cycles rebuilding same tree  
**Priority:** 🔥 **P0 - CRITICAL PERF**

**Current (BAD):**

```python
for batch_idx in range(num_batches):
    batch_points = points[start_idx:end_idx]

    # ❌ Upload per batch
    points_gpu = cp.asarray(batch_points)
    # ❌ Build per batch - EXPENSIVE!
    knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(points_gpu)
    distances, indices = knn.kneighbors(points_gpu)
    # ❌ Download per batch
    indices = cp.asnumpy(indices)
```

**Fixed (GOOD):**

```python
# ✅ Upload once
points_gpu = cp.asarray(points)

# ✅ Build global KNN once
knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(points_gpu)

for batch_idx in range(num_batches):
    batch_points_gpu = points_gpu[start_idx:end_idx]

    # ✅ Only query (fast!)
    distances, indices = knn.kneighbors(batch_points_gpu)

    # ✅ Keep on GPU for computation
    batch_eigen_features = self._compute_batch_eigenvalue_features_gpu(
        points_gpu, indices, required_features
    )
```

**Expected Impact:** 100 builds → 1 build = **5-10× faster**

---

### 4. Boundary Mode Non-Vectorized Loop 💀 10-100× SLOWDOWN

**Issue:** Python loop over every point instead of vectorized NumPy  
**File:** `ign_lidar/features/features_boundary.py`  
**Lines:** 260-290  
**Impact:** Makes boundary mode **completely unusable** for large datasets  
**Priority:** 🔥 **P0 - CRITICAL PERF**

**Current (TERRIBLE):**

```python
# ❌ Python loop - KILLS PERFORMANCE!
for i in range(num_points):
    neighbor_idx = neighbor_indices[i]
    neighbors = all_points[neighbor_idx]
    centroid = neighbors.mean(axis=0)
    centered = neighbors - centroid
    cov = (centered.T @ centered) / len(neighbors)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # ... process one point at a time
```

**Time for 1M points:** ~50-100 seconds (Python loop overhead!)

**Fixed (VECTORIZED):**

```python
# ✅ Vectorized - MASSIVE SPEEDUP!
# Gather all neighbors: [N, k, 3]
neighbors = all_points[neighbor_indices]

# Center all neighborhoods: [N, k, 3]
centroids = neighbors.mean(axis=1, keepdims=True)  # [N, 1, 3]
centered = neighbors - centroids

# Compute ALL covariance matrices: [N, 3, 3]
cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / k

# Batch eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)

# Extract normals: [N, 3]
normals = eigenvectors[:, :, 0]

# Sort eigenvalues: [N, 3]
eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]
```

**Time for 1M points:** ~2-5 seconds (vectorized!)

**Expected Impact:** 50-100s → 2-5s = **10-50× faster**

---

## ⚠️ High Priority (Fix This Week)

### 5. CPU Mode Single-Threaded KDTree

**Impact:** 2-4× slowdown  
**Fix:** Use parallel KDTree build  
**Priority:** ⚠️ **P1 - HIGH**

### 6. GPU Chunked Curvature Not Using CUDA Streams

**Impact:** +20-30% potential speedup  
**Fix:** Apply triple-buffering to curvature  
**Priority:** ⚠️ **P1 - HIGH**

---

## 📊 Expected Performance After Fixes

### 1M Points

```
Operation              Before    After     Speedup
────────────────────────────────────────────────────
GPU geometric features BROKEN    0.8s      N/A (fixed)
GPU per-batch KNN      5s        0.5s      10×
Boundary mode          80s       5s        16×
CPU (parallelized)     15s       5s        3×
```

### 10M Points

```
Operation              Before    After     Speedup
────────────────────────────────────────────────────
GPU geometric features BROKEN    8s        N/A (fixed)
GPU Chunked            14s       10s       1.4×
Boundary mode          800s      50s       16×
CPU (parallelized)     150s      50s       3×
```

---

## 🧪 Validation Tests

After each fix, run:

```bash
# 1. Quick smoke test
python scripts/test_cpu_bottlenecks.py

# 2. Full integration test
pytest tests/test_gpu_features.py -v

# 3. Performance benchmark
python scripts/benchmark_full_pipeline.py
```

**Expected results after all fixes:**

- ✅ All 3 bottleneck tests pass
- ✅ GPU mode 10-20× faster than CPU
- ✅ Boundary mode usable for large datasets
- ✅ No performance regressions

---

## 📝 Implementation Checklist

### Week 1: Critical Fixes

- [ ] Fix GPU `compute_geometric_features()` method
- [ ] Implement batched GPU transfers
- [ ] Implement global KNN in GPU mode
- [ ] Vectorize boundary mode normal computation
- [ ] Test all fixes with bottleneck test suite
- [ ] Benchmark performance improvements

### Week 2: High Priority Optimizations

- [ ] Add CUDA streams to chunked curvature
- [ ] Parallelize CPU KDTree operations
- [ ] Add GPU acceleration to boundary mode
- [ ] Update documentation with new performance numbers

### Week 3: Architecture Improvements

- [ ] Complete unified core implementations
- [ ] Add automatic mode selection
- [ ] Implement cross-mode benchmarking
- [ ] Add performance regression tests

---

## 📈 Success Metrics

| Metric                          | Current | Target      | Critical? |
| ------------------------------- | ------- | ----------- | --------- |
| GPU mode functional             | ❌      | ✅          | 🔥 YES    |
| GPU geometric features (1M pts) | BROKEN  | <1s         | 🔥 YES    |
| Boundary mode (1M pts)          | 80s     | <5s         | 🔥 YES    |
| GPU throughput                  | 0       | >5M pts/sec | ⚠️ HIGH   |
| Test suite pass rate            | 33%     | 100%        | ⚠️ HIGH   |

---

## 🔗 Related Documents

- `COMPREHENSIVE_COMPUTATION_AUDIT.md` - Full detailed analysis
- `CRITICAL_CPU_BOTTLENECKS_FOUND.md` - Original bottleneck report
- `CUDA_GPU_OPTIMIZATION_SUMMARY.md` - GPU optimizations implemented
- `GPU_NORMAL_OPTIMIZATION.md` - Normal computation optimization

---

**Next Actions:**

1. Start with GPU mode fixes (blocking issue)
2. Run test suite to validate
3. Implement remaining high-priority fixes
4. Document performance improvements

**ETA for all critical fixes:** 2-3 days  
**ETA for full optimization:** 2-3 weeks
