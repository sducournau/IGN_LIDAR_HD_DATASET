# Normal Computation Architecture

## Overview

This document describes the **canonical architecture** for normal vector computation in the IGN LiDAR HD library. Understanding this hierarchy is essential for developers working on feature computation.

**Last Updated**: November 23, 2025  
**Version**: 3.1.0+

---

## 🎯 Quick Start: What Should I Use?

### For Application Code (Recommended)

```python
from ign_lidar.features import FeatureOrchestrator

orchestrator = FeatureOrchestrator(use_gpu=True)
features = orchestrator.compute_features(
    points=points,
    feature_mode='lod2',
    k_neighbors=20
)

# Normals are in features['normals']
normals = features['normals']
eigenvalues = features['eigenvalues']
```

**Why?** `FeatureOrchestrator` automatically:

- Selects CPU vs GPU based on `use_gpu` flag
- Handles chunking for large datasets
- Manages memory efficiently
- Provides consistent API

### For CPU-Only Code

```python
from ign_lidar.features.compute.normals import compute_normals

normals, eigenvalues = compute_normals(
    points=points,
    k_neighbors=20,
    method='standard',  # or 'fast', 'accurate'
    return_eigenvalues=True
)
```

### For GPU Code

```python
from ign_lidar.features.gpu_processor import GPUProcessor

gpu_proc = GPUProcessor()
normals_gpu, eigenvalues_gpu = gpu_proc.compute_normals(
    points_gpu=points_gpu,
    k_neighbors=20
)
```

---

## 🏗️ Architecture Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ APPLICATION CODE (Your Code)                                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. FeatureOrchestrator                                       │
│    features/orchestrator.py                                  │
│    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│    ROLE: Entry point, CPU/GPU routing, mode selection       │
│    USE WHEN: Building applications, need automatic routing  │
└─────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ 2A. CPU Path             │  │ 2B. GPU Path             │
│                          │  │                          │
│ compute/normals.py       │  │ gpu_processor.py         │
│ ─────────────────────    │  │ ─────────────────────    │
│ CANONICAL CPU            │  │ GPU-OPTIMIZED            │
│ IMPLEMENTATION           │  │ IMPLEMENTATION           │
│                          │  │                          │
│ • Standard numpy/scipy   │  │ • CuPy arrays            │
│ • Multiple methods       │  │ • cuML algorithms        │
│ • Numba acceleration     │  │ • CUDA kernels           │
└──────────────────────────┘  └──────────────────────────┘
              │                          │
              ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ 3A. CPU Helpers          │  │ 3B. GPU Kernels          │
│                          │  │                          │
│ numba_accelerated.py     │  │ gpu_kernels.py           │
│ ─────────────────────    │  │ optimization/           │
│ LOW-LEVEL JIT            │  │ ─────────────────────    │
│                          │  │ FUSED CUDA KERNELS       │
│ • Covariance matrices    │  │                          │
│ • Eigenvector extraction │  │ • KNN + normals          │
│ • Pure NumPy fallback    │  │ • Eigendecomposition     │
└──────────────────────────┘  └──────────────────────────┘
```

---

## 📁 File Responsibilities

### Layer 1: Application Entry Point

#### `features/orchestrator.py` - **FeatureOrchestrator**

**Purpose**: Unified entry point for all feature computation  
**Responsibilities**:

- Route to CPU or GPU based on configuration
- Select appropriate computation strategy
- Handle feature mode (LOD2, LOD3, etc.)
- Manage memory and chunking

**When to use**:

- ✅ Building applications
- ✅ Need automatic CPU/GPU selection
- ✅ Processing full feature sets
- ✅ Production code

**When NOT to use**:

- ❌ Low-level optimization work
- ❌ Need direct control over algorithms
- ❌ Writing unit tests for specific implementations

---

### Layer 2A: CPU Canonical Implementation

#### `features/compute/normals.py` - **compute_normals()**

**Purpose**: Canonical CPU implementation for normal computation  
**Responsibilities**:

- Standard numpy/scipy implementation
- Multiple computation methods (fast, standard, accurate)
- Fallback when Numba unavailable
- KNN integration via unified engine

**Key functions**:

```python
def compute_normals(
    points: np.ndarray,
    k_neighbors: int = 20,
    search_radius: Optional[float] = None,
    method: str = 'standard',
    return_eigenvalues: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute normals using standard CPU implementation.

    Methods:
    - 'fast': k=10, quick results
    - 'standard': Use provided k_neighbors
    - 'accurate': k=50, best quality
    """
```

**When to use**:

- ✅ CPU-only environments
- ✅ Need specific computation method
- ✅ Testing CPU implementation
- ✅ Fallback from GPU

**When NOT to use**:

- ❌ GPU available and preferred
- ❌ Need full feature computation (use Orchestrator)

---

### Layer 2B: GPU Optimized Implementation

#### `features/gpu_processor.py` - **GPUProcessor**

**Purpose**: GPU-accelerated feature computation with chunking  
**Responsibilities**:

- CuPy/cuML based GPU implementation
- Automatic chunking for large datasets
- Memory management
- CUDA kernel integration

**Key functions**:

```python
class GPUProcessor:
    def compute_normals(
        self,
        points_gpu: cp.ndarray,
        k_neighbors: int = 20,
        return_eigenvalues: bool = True
    ) -> Tuple[cp.ndarray, Optional[cp.ndarray]]:
        """
        GPU-accelerated normal computation.

        Uses:
        - cuML for KNN
        - Custom CUDA kernels for covariance
        - CuPy for eigendecomposition
        """
```

**When to use**:

- ✅ GPU available
- ✅ Large datasets (> 1M points)
- ✅ Performance critical
- ✅ Already have CuPy arrays

**When NOT to use**:

- ❌ Small datasets (< 100k points) - CPU is fine
- ❌ GPU not available
- ❌ Need CPU portability

---

### Layer 3A: CPU Low-Level Helpers

#### `features/numba_accelerated.py`

**Purpose**: JIT-compiled helper functions for CPU performance  
**Responsibilities**:

- Numba @jit decorators for speedup
- Covariance matrix computation
- Eigenvector extraction
- Pure NumPy fallback when Numba unavailable

**Key functions**:

```python
@jit(nopython=True, parallel=True, cache=True)
def compute_covariance_matrices_numba(
    points: np.ndarray,
    indices: np.ndarray,
    k: int
) -> np.ndarray:
    """JIT-compiled covariance computation."""

def compute_normals_from_eigenvectors(
    points: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray
) -> np.ndarray:
    """Extract normals from eigenvectors (dispatcher)."""
```

**When to use**:

- ⚠️ **DO NOT call directly from application code**
- ✅ Called by `compute/normals.py` internally
- ✅ Writing low-level optimization

**When NOT to use**:

- ❌ Application code (use layer 2A instead)
- ❌ GPU code (use layer 3B)

---

### Layer 3B: GPU Low-Level Kernels

#### `optimization/gpu_kernels.py`

**Purpose**: Fused CUDA kernels for GPU performance  
**Responsibilities**:

- Custom CUDA kernels
- Kernel fusion (KNN + normals + eigenvalues)
- Memory-efficient GPU operations

**Key functions**:

```python
def compute_normals_and_eigenvalues(
    points_gpu: cp.ndarray,
    k_neighbors: int = 20
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Fused kernel: KNN + covariance + eigen + normals.

    Advantages:
    - Single GPU kernel launch
    - Reduced memory transfers
    - 30-40% faster than separate operations
    """
```

**When to use**:

- ⚠️ **DO NOT call directly from application code**
- ✅ Called by `gpu_processor.py` internally
- ✅ Writing GPU optimizations

---

## 🔄 Call Flow Examples

### Example 1: Application Using Orchestrator

```python
from ign_lidar.features import FeatureOrchestrator

# User code
orchestrator = FeatureOrchestrator(use_gpu=True)
features = orchestrator.compute_features(points, feature_mode='lod2')

# Internal flow:
# FeatureOrchestrator.compute_features()
#   ↓ (checks use_gpu=True, GPU available)
#   ↓ routes to GPU path
#   ↓
# gpu_processor.py::GPUProcessor.compute_normals()
#   ↓ (uploads points to GPU)
#   ↓ (checks dataset size)
#   ↓ (decides: batch or chunked)
#   ↓
# gpu_kernels.py::compute_normals_and_eigenvalues()
#   ↓ (single fused CUDA kernel)
#   ↓ (returns CuPy arrays)
#   ↓
# gpu_processor.py (downloads results)
#   ↓ (returns NumPy arrays)
#   ↓
# FeatureOrchestrator (packages into feature dict)
#   ↓
# User receives: features['normals'], features['eigenvalues']
```

### Example 2: Direct CPU Usage

```python
from ign_lidar.features.compute.normals import compute_normals

# User code
normals, eigenvalues = compute_normals(points, k_neighbors=20)

# Internal flow:
# compute/normals.py::compute_normals()
#   ↓ (validates inputs)
#   ↓ (builds KNN graph using optimization.knn_search)
#   ↓
# numba_accelerated.py::compute_covariance_matrices_numba()
#   ↓ (JIT-compiled covariance computation)
#   ↓ (or NumPy fallback if Numba unavailable)
#   ↓
# numpy.linalg.eigh() (eigendecomposition)
#   ↓
# numba_accelerated.py::compute_normals_from_eigenvectors()
#   ↓ (extract normals from smallest eigenvector)
#   ↓
# User receives: normals, eigenvalues
```

### Example 3: GPU Direct Usage

```python
import cupy as cp
from ign_lidar.features.gpu_processor import GPUProcessor

# User code
points_gpu = cp.asarray(points)
gpu_proc = GPUProcessor()
normals_gpu, eigenvalues_gpu = gpu_proc.compute_normals(points_gpu)

# Internal flow:
# gpu_processor.py::GPUProcessor.compute_normals()
#   ↓ (points already on GPU)
#   ↓ (estimates memory requirements)
#   ↓ (decides strategy: batch vs chunked)
#   ↓
# [If batch mode]
# gpu_kernels.py::compute_normals_and_eigenvalues()
#   ↓ (single fused kernel)
#   ↓
# [If chunked mode]
# Loop over chunks:
#   gpu_kernels.py::compute_normals_and_eigenvalues(chunk)
#   ↓
# Concatenate results
#   ↓
# User receives: normals_gpu (CuPy), eigenvalues_gpu (CuPy)
```

---

## ⚠️ Common Mistakes

### ❌ WRONG: Calling Low-Level Functions Directly

```python
# DON'T DO THIS!
from ign_lidar.features.numba_accelerated import compute_covariance_matrices_numba

# This is a low-level helper, not meant for direct use
cov_matrices = compute_covariance_matrices_numba(points, indices, k)
```

**Why it's wrong**:

- Missing KNN computation
- No input validation
- No error handling
- Breaks abstraction

**Do this instead**:

```python
from ign_lidar.features.compute.normals import compute_normals

normals, eigenvalues = compute_normals(points, k_neighbors=20)
```

### ❌ WRONG: Mixing CPU and GPU Arrays

```python
# DON'T DO THIS!
import cupy as cp
from ign_lidar.features.compute.normals import compute_normals  # CPU function

points_gpu = cp.asarray(points)
normals, eigenvalues = compute_normals(points_gpu)  # ERROR!
```

**Why it's wrong**:

- CPU function expects NumPy arrays
- Type mismatch causes errors

**Do this instead**:

```python
# Option 1: Use GPU path
from ign_lidar.features.gpu_processor import GPUProcessor

points_gpu = cp.asarray(points)
gpu_proc = GPUProcessor()
normals_gpu, eigenvalues_gpu = gpu_proc.compute_normals(points_gpu)

# Option 2: Use orchestrator (automatic routing)
from ign_lidar.features import FeatureOrchestrator

orchestrator = FeatureOrchestrator(use_gpu=True)
features = orchestrator.compute_features(points)  # NumPy input
```

### ❌ WRONG: Reimplementing Normal Computation

```python
# DON'T DO THIS!
def my_compute_normals(points):
    """Custom normal computation"""
    # ... reimplemented logic ...
    return normals
```

**Why it's wrong**:

- Duplicates existing functionality
- Missing optimizations
- No GPU support
- Harder to maintain

**Do this instead**:

```python
# Extend or improve existing implementation
from ign_lidar.features.compute import normals

# If you need custom behavior, add it to the existing module
# and submit a PR
```

---

## 🚀 Performance Guidelines

### When to Use Each Layer

| Dataset Size | CPU Available | GPU Available | Recommended Approach                 |
| ------------ | ------------- | ------------- | ------------------------------------ |
| < 100k       | ✓             | ✗             | `compute/normals.py` (CPU)           |
| < 100k       | ✓             | ✓             | `compute/normals.py` (CPU fine)      |
| 100k - 1M    | ✓             | ✗             | `compute/normals.py` (CPU)           |
| 100k - 1M    | ✓             | ✓             | `FeatureOrchestrator` (GPU)          |
| 1M - 10M     | ✓             | ✗             | `compute/normals.py` (CPU slow)      |
| 1M - 10M     | ✓             | ✓             | `gpu_processor.py` (GPU batch)       |
| > 10M        | ✓             | ✗             | `compute/normals.py` (CPU very slow) |
| > 10M        | ✓             | ✓             | `gpu_processor.py` (GPU chunked)     |

### Method Selection (CPU)

```python
# Fast mode: k=10, quick results (~30% faster)
normals, _ = compute_normals(points, method='fast', return_eigenvalues=False)

# Standard mode: balanced (recommended)
normals, eigenvalues = compute_normals(points, k_neighbors=20)

# Accurate mode: k=50, best quality (~50% slower)
normals, eigenvalues = compute_normals(points, method='accurate')
```

### GPU Strategy Selection (Automatic)

```python
# GPUProcessor automatically selects strategy
gpu_proc = GPUProcessor()

# < 10M points: Uses batch mode (single GPU call)
normals, eigenvalues = gpu_proc.compute_normals(small_dataset_gpu)

# > 10M points: Uses chunked mode (multiple GPU calls)
normals, eigenvalues = gpu_proc.compute_normals(large_dataset_gpu)
```

---

## 🔧 Extension Guidelines

### Adding a New Normal Computation Method

1. **Add to CPU canonical implementation**:

   ```python
   # features/compute/normals.py

   def compute_normals(points, method='standard', ...):
       if method == 'my_new_method':
           return _compute_normals_my_method(points, ...)
       # ... existing methods ...
   ```

2. **Add GPU version if applicable**:

   ```python
   # features/gpu_processor.py

   def compute_normals(self, points_gpu, method='standard', ...):
       if method == 'my_new_method':
           return self._compute_normals_my_method_gpu(points_gpu, ...)
       # ... existing methods ...
   ```

3. **Update tests**:

   ```python
   # tests/test_normals.py

   def test_my_new_method():
       normals, eigenvalues = compute_normals(
           points, method='my_new_method'
       )
       assert normals.shape == (len(points), 3)
   ```

4. **Update documentation**: Add method to this document!

---

## 📚 Related Documentation

- [Feature Computation Overview](../features/README.md)
- [GPU Acceleration Guide](../guides/gpu-acceleration.md)
- [Performance Optimization](../guides/performance.md)
- [API Reference](../api/features.md)

---

## 🐛 Troubleshooting

### "ImportError: No module named cupy"

**Problem**: Trying to use GPU features without CuPy installed.

**Solution**:

```bash
# Install CuPy for CUDA 11.x
pip install cupy-cuda11x

# Or for CUDA 12.x
pip install cupy-cuda12x
```

### "GPU available but using CPU"

**Problem**: GPU detected but not being used.

**Solution**: Check `use_gpu` flag:

```python
orchestrator = FeatureOrchestrator(use_gpu=True)  # Explicitly enable
```

### "CUDA out of memory"

**Problem**: Dataset too large for GPU memory.

**Solution**: Use chunked processing:

```python
from ign_lidar.optimization import auto_chunk_size

chunk_size = auto_chunk_size(points.shape, target_memory_usage=0.7)
gpu_proc = GPUProcessor(chunk_size=chunk_size)
```

---

**Questions or Issues?** See [CONTRIBUTING.md](../../CONTRIBUTING.md) or open an issue on GitHub.
