# GPU Stack Architecture

**Version:** 3.3.0  
**Last Updated:** November 23, 2025  
**Status:** Production

---

## Overview

The IGN LiDAR HD library provides a comprehensive GPU acceleration stack for point cloud processing. This document describes the architecture, module hierarchy, and best practices for GPU operations.

## Module Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
│  (LiDARProcessor, FeatureOrchestrator, Classifiers)             │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────────┐
│                    Strategy Layer                                │
│  • strategy_gpu.py         (full GPU processing)                │
│  • strategy_gpu_chunked.py (chunked GPU processing)             │
│  • strategy_cpu.py         (CPU fallback)                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────────┐
│                  Optimization Layer                              │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  ground_truth_   │  │   knn_engine.py  │  │  gpu_cache/   │ │
│  │  classifier.py   │  │  (unified KNN)   │  │               │ │
│  │  (GPU labeling)  │  │                  │  │ • arrays.py   │ │
│  │                  │  │  • FAISS-GPU     │  │ • transfer.py │ │
│  │                  │  │  • cuML          │  │               │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────────┐
│                      Core Layer                                  │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │    gpu.py        │  │  gpu_profiler.py │  │ gpu_memory.py │ │
│  │  (GPUManager)    │  │  (Profiling)     │  │ (System Mgmt) │ │
│  │                  │  │                  │  │               │ │
│  │  • Singleton     │  │  • CUDA events   │  │ • VRAM track  │ │
│  │  • Init/cleanup  │  │  • Bottleneck    │  │ • Allocation  │ │
│  │  • Device info   │  │    analysis      │  │ • Cleanup     │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────────┐
│                    Hardware Layer                                │
│  CUDA Runtime, CuPy, cuML, cuSpatial, FAISS-GPU                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. `core/gpu.py` - GPUManager

**Purpose:** System-level GPU management and initialization.

**Responsibilities:**

- Device detection and selection
- CUDA context management
- GPU availability checks
- Singleton pattern for global GPU state

**Key Classes:**

- `GPUManager`: Singleton for GPU lifecycle management

**Usage:**

```python
from ign_lidar.core.gpu import GPUManager

gpu = GPUManager()  # Singleton instance
if gpu.is_available():
    device_info = gpu.get_device_info()
    print(f"GPU: {device_info['name']}, VRAM: {device_info['total_memory_gb']:.1f}GB")
```

---

### 2. `core/gpu_profiler.py` - Profiling

**Purpose:** Performance profiling and bottleneck analysis.

**Responsibilities:**

- CUDA event-based timing (sub-millisecond precision)
- Transfer vs compute analysis
- Bottleneck detection
- Performance reporting

**Key Classes:**

- `GPUProfiler`: Main profiler with CUDA events support
- `ProfileEntry`: Individual profiling entry
- `ProfilingStats`: Aggregated statistics

**Usage:**

```python
from ign_lidar.core.gpu_profiler import create_profiler

profiler = create_profiler(enabled=True, use_cuda_events=True)

with profiler.profile('feature_computation', transfer='upload', size_mb=120):
    features_gpu = compute_features_gpu(points_gpu)

# Bottleneck analysis
analysis = profiler.get_bottleneck_analysis()
if analysis['bottleneck'] == 'transfer':
    print(f"⚠️ Transfer bottleneck: {analysis['transfer_pct']:.1f}%")
```

---

### 3. `core/gpu_memory.py` - System Memory Manager

**Purpose:** System-level VRAM tracking and allocation management.

**Responsibilities:**

- VRAM usage monitoring
- Allocation tracking across processes
- Memory cleanup coordination
- Out-of-memory prevention

**Key Classes:**

- `GPUMemoryManager`: System-level memory tracker

**Usage:**

```python
from ign_lidar.core.gpu_memory import get_gpu_memory_manager

mem_mgr = get_gpu_memory_manager()
mem_info = mem_mgr.get_memory_info()

print(f"Used: {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB")
print(f"Available: {mem_info['available_gb']:.1f}GB")
```

---

## Optimization Modules

### 4. `optimization/gpu_cache/` - Caching & Transfer

**Purpose:** Application-level GPU memory optimization.

#### 4.1 `gpu_cache/arrays.py` - GPUArrayCache

**Responsibilities:**

- Smart caching of frequently accessed arrays
- Least Frequently Used (LFU) eviction
- In-place slice updates
- Access pattern tracking

**Usage:**

```python
from ign_lidar.optimization.gpu_cache import GPUArrayCache

cache = GPUArrayCache(max_size_gb=8.0)

# First access: uploads to GPU
normals_gpu = cache.get_or_upload('normals', normals_cpu)

# Second access: cached (no upload!)
normals_gpu = cache.get_or_upload('normals', normals_cpu)

# Update slice in-place
cache.update_slice('normals', start_idx=0, end_idx=1000, data=new_data_gpu)

# Statistics
stats = cache.get_stats()
print(f"Cache utilization: {stats['utilization_pct']:.1f}%")
```

#### 4.2 `gpu_cache/transfer.py` - Transfer Optimization

**Responsibilities:**

- Transfer logging and analysis
- Pre-allocated memory pools
- Array reuse across tiles
- Chunk size optimization

**Key Classes:**

- `TransferOptimizer`: Transfer pattern analysis
- `GPUMemoryPool`: Pre-allocated array pool (60-80% speedup)

**Usage:**

```python
from ign_lidar.optimization.gpu_cache import GPUMemoryPool, TransferOptimizer

# Memory pool for reuse
pool = GPUMemoryPool(max_arrays=20, max_size_gb=4.0)
arr = pool.get_array(shape=(10000, 3), dtype=cp.float32)
# ... use array ...
pool.return_array(arr)  # Return for reuse

# Transfer tracking
optimizer = TransferOptimizer()
optimizer.log_upload(points.nbytes, "points")
optimizer.print_summary()
```

---

### 5. `optimization/ground_truth_classifier.py`

**Purpose:** GPU-accelerated ground truth labeling with building footprints.

**Responsibilities:**

- Chunked spatial joins (cuSpatial)
- Building footprint matching
- Large-scale point labeling
- Memory-efficient processing

**Key Classes:**

- `GPUGroundTruthClassifier`: GPU labeling engine

**Usage:**

```python
from ign_lidar.optimization.ground_truth_classifier import GPUGroundTruthClassifier

classifier = GPUGroundTruthClassifier(chunk_size=5_000_000)
labels = classifier.label_points(
    points=points_cpu,
    buildings_gdf=buildings_gdf,
    use_chunking=True
)
```

---

### 6. `optimization/knn_engine.py`

**Purpose:** Unified KNN interface with GPU/CPU backends.

**Responsibilities:**

- FAISS-GPU acceleration
- cuML KNN fallback
- CPU fallback (scikit-learn)
- Automatic backend selection

**Usage:**

```python
from ign_lidar.optimization import knn_search, KNNEngine

# Simple API
indices, distances = knn_search(
    points=points,
    k=30,
    use_gpu=True,
    backend='auto'  # Chooses best available
)

# Advanced API
engine = KNNEngine(backend='faiss-gpu')
engine.fit(points)
indices, distances = engine.search(query_points, k=30)
```

---

## Decision Matrix

### When to Use Which Module?

| Task                        | Module                                 | Reason                           |
| --------------------------- | -------------------------------------- | -------------------------------- |
| Check GPU availability      | `core.gpu.GPUManager`                  | System-level detection           |
| Profile GPU performance     | `core.gpu_profiler`                    | CUDA events, bottleneck analysis |
| Track VRAM usage            | `core.gpu_memory`                      | System-wide memory monitoring    |
| Cache frequent arrays       | `optimization.gpu_cache.GPUArrayCache` | Avoid redundant uploads          |
| Reuse arrays across tiles   | `optimization.gpu_cache.GPUMemoryPool` | Reduce allocation overhead       |
| Label points with buildings | `optimization.ground_truth_classifier` | Spatial join acceleration        |
| KNN search                  | `optimization.knn_engine`              | Unified API, automatic backend   |
| Feature computation         | `features.strategy_gpu`                | Complete feature pipeline        |

---

## Best Practices

### 1. GPU Initialization

```python
from ign_lidar.core.gpu import GPUManager

# Always check availability first
gpu = GPUManager()
if not gpu.is_available():
    logger.warning("GPU not available, falling back to CPU")
    use_gpu = False
```

### 2. Memory Management

```python
from ign_lidar.core.gpu_memory import get_gpu_memory_manager
from ign_lidar.optimization.gpu_cache import GPUArrayCache

mem_mgr = get_gpu_memory_manager()
cache = GPUArrayCache(max_size_gb=mem_mgr.get_memory_info()['available_gb'] * 0.7)

# Always check before large allocations
if mem_mgr.get_memory_info()['available_gb'] < 2.0:
    logger.warning("Low VRAM, reducing chunk size")
    chunk_size //= 2
```

### 3. Profiling

```python
from ign_lidar.core.gpu_profiler import create_profiler

profiler = create_profiler(enabled=True, use_cuda_events=True)

with profiler.profile('total_pipeline'):
    with profiler.profile('knn_search', transfer='upload', size_mb=points.nbytes/1024**2):
        indices = knn_search_gpu(points)

    with profiler.profile('feature_compute'):
        features = compute_features_gpu(points, indices)

    with profiler.profile('download', transfer='download', size_mb=features.nbytes/1024**2):
        features_cpu = cp.asnumpy(features)

# Analyze bottlenecks
analysis = profiler.get_bottleneck_analysis()
profiler.print_report()
```

### 4. Caching Strategy

```python
from ign_lidar.optimization.gpu_cache import GPUArrayCache

cache = GPUArrayCache(max_size_gb=8.0)

# Cache arrays that are accessed multiple times
for tile in tiles:
    points = load_tile(tile)

    # Upload once, reuse for all feature computations
    points_gpu = cache.get_or_upload(f'points_{tile.id}', points)

    normals = compute_normals(points_gpu)
    curvature = compute_curvature(points_gpu, normals)
    # ... points_gpu stays on GPU
```

### 5. Error Handling

```python
import cupy as cp
from ign_lidar.core.error_handler import GPUMemoryError

try:
    features = compute_features_gpu(points)
except cp.cuda.memory.OutOfMemoryError:
    logger.warning("GPU OOM, falling back to CPU")
    features = compute_features_cpu(points)
except Exception as e:
    logger.error(f"GPU error: {e}")
    raise
```

---

## Migration from v3.2

### Deprecated Imports (Still Work with Warnings)

```python
# ❌ Old (deprecated, will break in v4.0)
from ign_lidar.optimization.gpu_profiler import GPUProfiler
from ign_lidar.optimization.gpu import GPUGroundTruthClassifier
from ign_lidar.optimization.gpu_memory import GPUArrayCache

# ✅ New (recommended v3.3+)
from ign_lidar.core.gpu_profiler import GPUProfiler
from ign_lidar.optimization.ground_truth_classifier import GPUGroundTruthClassifier
from ign_lidar.optimization.gpu_cache import GPUArrayCache
```

---

## Performance Tips

### 1. Minimize Transfers

- Use `GPUArrayCache` for frequently accessed arrays
- Keep intermediate results on GPU when possible
- Batch operations to reduce round-trips

### 2. Use Memory Pools

- `GPUMemoryPool` reduces allocation overhead by 60-80%
- Essential for multi-tile batch processing
- Pre-allocates common array shapes

### 3. Profile Everything

- Use `gpu_profiler` to identify bottlenecks
- Target transfer vs compute imbalance
- Optimize the slowest operations first

### 4. Chunked Processing

- Use `strategy_gpu_chunked.py` for large datasets
- Automatic chunk size optimization based on VRAM
- Prevents OOM errors

---

## Troubleshooting

### GPU Not Detected

```bash
# Check CuPy installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

# Verify CUDA version
nvidia-smi
```

### Out of Memory

```python
from ign_lidar.optimization.gpu_cache.transfer import optimize_chunk_size_for_vram

optimal_size = optimize_chunk_size_for_vram(
    num_points=total_points,
    available_vram_gb=gpu.get_device_info()['available_memory_gb'],
    safety_factor=0.75
)
```

### Performance Issues

```python
# Enable profiling to find bottleneck
profiler = create_profiler(enabled=True, use_cuda_events=True)
# ... run pipeline ...
analysis = profiler.get_bottleneck_analysis()

if analysis['bottleneck'] == 'transfer':
    # Use more caching
    cache = GPUArrayCache(max_size_gb=12.0)  # Increase cache
elif analysis['bottleneck'] == 'compute':
    # Optimize compute kernels or reduce feature count
    pass
```

---

## See Also

- [Feature Computation Guide](../guides/feature-computation.md)
- [GPU Configuration](../guides/gpu-configuration.md)
- [Performance Optimization](../guides/performance-optimization.md)
- [v3.3 Migration Guide](../migration_guides/v3_to_v4_gpu.md)

---

**Questions or issues?** Open an issue at [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
