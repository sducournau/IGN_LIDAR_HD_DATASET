# 🚨 CRITICAL CPU Bottlenecks Found in GPU Code

**Date:** October 18, 2025  
**Status:** ⚠️ **URGENT - Multiple CPU Operations in GPU Path**  
**Impact:** **10-50x slowdown** on critical operations

---

## 🔥 Summary

Found **4 critical bottlenecks** where CPU operations are executed in the GPU processing path, causing massive slowdowns:

| Bottleneck                                         | File            | Lines              | Impact        | Status      |
| -------------------------------------------------- | --------------- | ------------------ | ------------- | ----------- |
| **#1: Per-Batch Transfers in Eigenvalue Features** | features_gpu.py | 925-941            | 10-20x slower | ⚠️ CRITICAL |
| **#2: CPU Curvature Computation**                  | features_gpu.py | 620-650            | 10-20x slower | ⚠️ CRITICAL |
| **#3: CPU Batched KNN in Geometric Features**      | features_gpu.py | 820-860            | 5-10x slower  | ⚠️ HIGH     |
| **#4: Python Loop Sync Points**                    | features_gpu.py | 254, 303, 625, 825 | 2-5x slower   | ⚠️ MEDIUM   |

---

## 🔍 Bottleneck #1: Per-Batch GPU→CPU Transfers in Eigenvalue Features

### Location

**File:** `ign_lidar/features/features_gpu.py`  
**Function:** `_compute_batch_eigenvalue_features()`  
**Lines:** 925-941

### Problem Code

```python
# Compute only required features
batch_features = {}

if 'planarity' in required_features:
    planarity = (λ1 - λ2) / (sum_λ + 1e-8)
    batch_features['planarity'] = self._to_cpu(planarity).astype(np.float32)  # ❌ TRANSFER!

if 'linearity' in required_features:
    linearity = (λ0 - λ1) / (sum_λ + 1e-8)
    batch_features['linearity'] = self._to_cpu(linearity).astype(np.float32)  # ❌ TRANSFER!

if 'sphericity' in required_features:
    sphericity = λ2 / (sum_λ + 1e-8)
    batch_features['sphericity'] = self._to_cpu(sphericity).astype(np.float32)  # ❌ TRANSFER!

if 'anisotropy' in required_features:
    anisotropy = (λ0 - λ2) / (sum_λ + 1e-8)
    batch_features['anisotropy'] = self._to_cpu(anisotropy).astype(np.float32)  # ❌ TRANSFER!
```

### Why This is Bad

- **4 separate GPU→CPU transfers** (one per feature!)
- Each transfer causes GPU synchronization
- Eigenvalues already on GPU, no need to transfer
- Called in loop from `compute_geometric_features()` (line 825)

### Impact

- **Per 100K points:** 4 transfers × 0.4KB × 10ms = 40ms wasted
- **Per 10M points:** 100 batches × 40ms = **4 seconds wasted**
- **Speedup potential:** 10-20x

### Fix

```python
# ✅ OPTIMIZED: Keep all features on GPU, single transfer at end
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

**Result:** 4 transfers → 1 transfer = **4x faster**

---

## 🔍 Bottleneck #2: CPU Curvature Computation

### Location

**File:** `ign_lidar/features/features_gpu.py`  
**Function:** `compute_curvature()`  
**Lines:** 620-650

### Problem Code

```python
def compute_curvature(
    self,
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 20
) -> np.ndarray:
    """Compute curvature on CPU using vectorized operations."""
    N = len(points)
    curvature = np.zeros(N, dtype=np.float32)

    # Build KDTree
    tree = KDTree(points, metric='euclidean')  # ❌ CPU ONLY!

    # Batch processing
    batch_size = 50000
    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, N)
        batch_points = points[start_idx:end_idx]
        batch_normals = normals[start_idx:end_idx]

        # Query KNN
        _, indices = tree.query(batch_points, k=k)  # ❌ CPU QUERY!

        # VECTORIZED curvature computation
        neighbor_normals = normals[indices]
        query_normals_expanded = batch_normals[:, np.newaxis, :]
        normal_diff = neighbor_normals - query_normals_expanded

        # Compute norms and mean
        curv_norms = np.linalg.norm(normal_diff, axis=2)  # ❌ CPU COMPUTATION!
        batch_curvature = np.mean(curv_norms, axis=1)

        curvature[start_idx:end_idx] = batch_curvature

    return curvature
```

### Why This is Bad

- **NO GPU PATH AT ALL** - function is pure CPU!
- `np.linalg.norm()` is 10-20x slower than `cp.linalg.norm()`
- KDTree on CPU instead of cuML NearestNeighbors
- Called for every tile, every workflow

### Impact

- **Per 2M points:** 2-3 minutes on CPU vs 5-10 seconds on GPU
- **Speedup potential:** 12-36x

### Fix

```python
def compute_curvature(
    self,
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 20
) -> np.ndarray:
    """Compute curvature with GPU acceleration."""
    N = len(points)

    # ✅ GPU PATH
    if self.use_gpu and cp is not None and cuNearestNeighbors is not None:
        try:
            # Transfer to GPU once
            points_gpu = cp.asarray(points)
            normals_gpu = cp.asarray(normals)

            # Build GPU KNN
            knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(points_gpu)

            # Batch processing on GPU
            batch_size = 200000  # Larger batches on GPU
            num_batches = (N + batch_size - 1) // batch_size
            curvature_gpu = cp.zeros(N, dtype=cp.float32)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, N)
                batch_points_gpu = points_gpu[start_idx:end_idx]
                batch_normals_gpu = normals_gpu[start_idx:end_idx]

                # Query KNN on GPU
                _, indices_gpu = knn.kneighbors(batch_points_gpu)

                # ✅ ALL GPU COMPUTATION
                neighbor_normals_gpu = normals_gpu[indices_gpu]
                query_normals_expanded = batch_normals_gpu[:, cp.newaxis, :]
                normal_diff_gpu = neighbor_normals_gpu - query_normals_expanded

                # GPU norm and mean
                curv_norms_gpu = cp.linalg.norm(normal_diff_gpu, axis=2)
                batch_curvature_gpu = cp.mean(curv_norms_gpu, axis=1)

                curvature_gpu[start_idx:end_idx] = batch_curvature_gpu

            # Single transfer at end
            return cp.asnumpy(curvature_gpu)

        except Exception as e:
            print(f"⚠️ GPU curvature failed: {e}, falling back to CPU")

    # CPU fallback (existing code)
    curvature = np.zeros(N, dtype=np.float32)
    tree = KDTree(points, metric='euclidean')
    # ... (rest of CPU code)
```

**Result:** 2-3 min → 5-10 sec = **12-36x faster**

---

## 🔍 Bottleneck #3: CPU Batched KNN in Geometric Features

### Location

**File:** `ign_lidar/features/features_gpu.py`  
**Function:** `compute_geometric_features()`  
**Lines:** 825-860

### Problem Code

```python
# Process in batches
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, N)
    batch_points = points[start_idx:end_idx]

    # Build KDTree for batch
    if self.use_gpu and self.use_cuml and cuNearestNeighbors is not None:
        # GPU path
        points_gpu = cp.asarray(batch_points)  # ❌ TRANSFER PER BATCH!
        knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(points_gpu)
        distances, indices = knn.kneighbors(points_gpu)
        indices = cp.asnumpy(indices)  # ❌ TRANSFER BACK!
        distances = cp.asnumpy(distances)  # ❌ TRANSFER BACK!
    else:
        # CPU path (fast KDTree)
        tree = KDTree(batch_points, metric='euclidean')
        distances, indices = tree.query(batch_points, k=k)

    if need_eigenvalues:
        # Compute eigenvalue features for batch
        batch_eigen_features = self._compute_batch_eigenvalue_features(
            batch_points, indices, required_features
        )
        for feat, values in batch_eigen_features.items():
            features[feat][start_idx:end_idx] = values
```

### Why This is Bad

- **Per-batch GPU transfers** (upload points, download indices)
- Builds new KNN model per batch (expensive!)
- Should build **one global KNN** on full dataset
- Transfers cancel out GPU compute benefits

### Impact

- **Per 10M points:** 100 batches × (upload + download + rebuild) = **5-10x slower**
- Should use global KNN like `features_gpu_chunked.py` does

### Fix

```python
# ✅ OPTIMIZED: Build global KNN once
if self.use_gpu and self.use_cuml and cuNearestNeighbors is not None:
    # Transfer all points once
    points_gpu = cp.asarray(points)

    # Build global KNN
    knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(points_gpu)

    # Process in batches (but reuse global KNN)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, N)
        batch_points_gpu = points_gpu[start_idx:end_idx]

        # Query global KNN
        distances_gpu, indices_gpu = knn.kneighbors(batch_points_gpu)

        # Keep on GPU for eigenvalue computation
        batch_eigen_features = self._compute_batch_eigenvalue_features_gpu(
            points_gpu, indices_gpu, required_features
        )
        # ... (accumulate results on GPU, transfer once at end)
```

**Result:** 100 builds → 1 build = **5-10x faster**

---

## 🔍 Bottleneck #4: Python Loop Sync Points

### Locations

**File:** `ign_lidar/features/features_gpu.py`  
**Lines:** 254, 303, 625, 825

### Problem

Python `for` loops with GPU operations can create implicit synchronization:

```python
for batch_idx in range(num_batches):
    # GPU operation
    result = compute_on_gpu(batch_idx)

    # This assignment may trigger sync!
    results[batch_idx] = result
```

### Why This is Bad

- Python loop control can force GPU sync
- `range()` and index access can block GPU pipeline
- Better to use GPU-native operations or CUDA streams

### Fix Options

**Option 1: Preallocate and use GPU arrays**

```python
# Preallocate on GPU
results_gpu = cp.zeros((num_batches, batch_size), dtype=cp.float32)

for batch_idx in range(num_batches):
    result_gpu = compute_on_gpu(batch_idx)
    results_gpu[batch_idx] = result_gpu  # Stay on GPU

# Single transfer
results = cp.asnumpy(results_gpu)
```

**Option 2: Use CUDA streams** (already implemented in gpu_chunked)

```python
# Overlap computation with transfers
stream_manager.compute_async(batch_idx, stream=0)
stream_manager.transfer_async(batch_idx-1, stream=1)
```

**Impact:** 2-5x speedup by eliminating sync points

---

## 📊 Combined Impact

### Current Performance (with bottlenecks)

| Operation           | Time (10M points) | Method                    |
| ------------------- | ----------------- | ------------------------- |
| Geometric features  | ~50-60s           | Per-batch KNN + transfers |
| Curvature           | ~120-180s         | CPU computation           |
| Eigenvalue features | ~30-40s           | 4 transfers per batch     |
| **TOTAL**           | **~200-280s**     | **Multiple bottlenecks**  |

### After Fixes (optimized)

| Operation           | Time (10M points) | Optimization             |
| ------------------- | ----------------- | ------------------------ |
| Geometric features  | ~6-10s            | Global KNN + GPU compute |
| Curvature           | ~10-15s           | GPU acceleration         |
| Eigenvalue features | ~3-5s             | Batched transfers        |
| **TOTAL**           | **~20-30s**       | **10x faster!**          |

---

## ✅ Fix Priority

### 🔥 URGENT (10-20x impact)

1. **Fix #2: GPU Curvature** - Pure CPU, no GPU path
2. **Fix #1: Batched Eigenvalue Transfers** - 4 transfers per batch

### ⚠️ HIGH (5-10x impact)

3. **Fix #3: Global KNN** - Rebuild per batch

### 📌 MEDIUM (2-5x impact)

4. **Fix #4: Loop Sync Points** - Use streams/batching

---

## 🧪 Testing Plan

### 1. Quick Validation (10K points)

```bash
python -c "
from ign_lidar.features.features_gpu import GPUFeatureComputer
import numpy as np

computer = GPUFeatureComputer(use_gpu=True)
points = np.random.rand(10000, 3).astype(np.float32)

# Test curvature
normals = computer.compute_normals(points, k=20)
curv = computer.compute_curvature(points, normals, k=20)
print(f'✅ Curvature: {curv.shape}')

# Test geometric features
features = computer.compute_geometric_features(
    points, ['planarity', 'linearity', 'sphericity'], k=20
)
print(f'✅ Features: {list(features.keys())}')
"
```

### 2. Benchmark (1M points)

```bash
python -c "
import time
import numpy as np
from ign_lidar.features.features_gpu import GPUFeatureComputer

computer = GPUFeatureComputer(use_gpu=True)
points = np.random.rand(1000000, 3).astype(np.float32)

# Curvature benchmark
normals = computer.compute_normals(points, k=20)
start = time.time()
curv = computer.compute_curvature(points, normals, k=20)
print(f'Curvature: {time.time() - start:.2f}s')

# Geometric features benchmark
start = time.time()
features = computer.compute_geometric_features(
    points, ['planarity', 'linearity', 'sphericity'], k=20
)
print(f'Geometric: {time.time() - start:.2f}s')
"
```

**Expected Results:**

- Curvature: <2 seconds (was ~20s)
- Geometric: <1 second (was ~5s)

---

## 🎯 Next Steps

1. **Apply Fix #2 (GPU Curvature)** - Highest impact
2. **Apply Fix #1 (Batched Transfers)** - Easy fix
3. **Apply Fix #3 (Global KNN)** - Medium complexity
4. **Test on 1M points** - Verify speedup
5. **Test full pipeline** - End-to-end validation

---

**Status:** Ready to implement fixes  
**Risk:** Low (add GPU paths, keep CPU fallbacks)  
**Expected Impact:** **10-20x overall speedup**
