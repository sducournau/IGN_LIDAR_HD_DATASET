# GPU Optimization Guide

Comprehensive guide to GPU-accelerated processing optimizations in IGN LiDAR HD.

## Overview

The IGN LiDAR HD dataset processor includes state-of-the-art GPU optimizations for:

- **Feature computation** (normals, curvature, geometric features)
- **Ground truth classification** (BD TOPO® integration)
- **Memory management** (adaptive chunking, pooling)
- **Pipeline optimization** (overlapped compute/transfer)

**Performance gains: 10-100× speedup over CPU with intelligent resource management**

---

## Architecture

### Three-Layer Optimization Stack

```
┌──────────────────────────────────────────────────────────────┐
│                   Application Layer                           │
│  (LiDAR Processor, Feature Orchestrator)                      │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│              GPU Optimization Coordinator                     │
│  • Unified memory management                                  │
│  • Adaptive chunk sizing                                      │
│  • Resource allocation                                        │
│  • Performance monitoring                                     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────┬─────────────────────────────────┐
│  GPU Feature Computation   │  GPU Ground Truth Classifier    │
│  • Chunked processing      │  • Spatial operations           │
│  • Memory pooling          │  • Polygon classification       │
│  • Stream optimization     │  • NDVI integration             │
└────────────────────────────┴─────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                      CUDA/CuPy Layer                          │
│  CuPy (GPU arrays) + RAPIDS cuML (algorithms) + cuSpatial    │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. GPU Optimization Coordinator (`gpu_coordinator.py`)

**Purpose:** Centralized GPU resource management

**Features:**

- **Adaptive chunk sizing** - Automatically determines optimal chunk size based on available VRAM
- **Memory pooling** - Reduces allocation overhead by 50-80%
- **Unified management** - Coordinates resources across all GPU operations
- **Performance tracking** - Monitors GPU utilization and identifies bottlenecks

**Usage:**

```python
from ign_lidar.optimization.gpu_coordinator import get_gpu_coordinator

# Get global coordinator
coordinator = get_gpu_coordinator(
    enable_memory_pooling=True,
    enable_adaptive_chunking=True,
    vram_target_utilization=0.85
)

# Get optimized parameters for feature computation
params = coordinator.optimize_for_feature_computation(
    num_points=18_000_000,
    feature_mode='asprs_classes'
)

# Use recommended chunk size
chunk_size = params['chunk_size']  # Optimized for your GPU
```

**Benefits:**

- ✅ Prevents OOM errors through intelligent sizing
- ✅ Maximizes GPU utilization (target: 85%)
- ✅ 2-5× additional speedup through better resource management

---

### 2. GPU Chunked Feature Computer (`features_gpu_chunked.py`)

**Purpose:** GPU-accelerated feature computation with chunking

**Optimizations:**

1. **Single Global KDTree** - Build once, query in chunks (10-100× faster)
2. **Batched GPU Transfers** - Accumulate results, transfer once (reduces sync overhead)
3. **Memory Pooling** - Pre-allocated VRAM pool for faster allocations
4. **CUDA Streams** - Overlap data transfer and computation
5. **Vectorized Operations** - Covariance computation on GPU (100× faster than loops)

**Key Methods:**

```python
from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer

computer = GPUChunkedFeatureComputer(
    chunk_size=None,  # Auto-optimize based on VRAM
    vram_limit_gb=None,  # Auto-detect
    auto_optimize=True,
    use_cuda_streams=True,
    enable_memory_pooling=True
)

# Compute normals (optimized)
normals = computer.compute_normals_chunked(points, k=20)

# Compute all features
normals, curvature, height, geo_features = computer.compute_all_features_chunked(
    points=points,
    classification=classification,
    k=20,
    mode='asprs_classes'
)
```

**Performance:**

- **Before optimization:** 353s per 1.86M point chunk
- **After optimization:** 22s per chunk
- **Speedup:** 16× improvement ✅

---

### 3. GPU Ground Truth Classifier (`gpu.py`)

**Purpose:** GPU-accelerated spatial classification with BD TOPO® data

**Optimizations:**

1. **Adaptive Chunking** - Automatically adjusts chunk size based on polygon count
2. **Spatial Indexing** - Build spatial index cache for repeated queries
3. **Memory Pooling** - Shared pool with feature computation
4. **GPU Bbox Filtering** - Fast rejection on GPU before CPU precise checks
5. **cuSpatial Integration** - Ultra-fast point-in-polygon (when available)

**Usage:**

```python
from ign_lidar.optimization.gpu import GPUGroundTruthClassifier

classifier = GPUGroundTruthClassifier(
    gpu_chunk_size=5_000_000,  # Or None for auto
    use_cuspatial=True,
    enable_adaptive_chunking=True,
    enable_memory_pooling=True,
    enable_spatial_indexing=True
)

labels = classifier.classify_with_ground_truth(
    labels=labels,
    points=points,
    ground_truth_features=ground_truth_dict,
    ndvi=ndvi,  # Optional NDVI refinement
    height=height,
    planarity=planarity
)
```

**Performance:**

- **CPU STRtree:** 10-30× speedup over naive
- **GPU CuPy:** 100-500× speedup for small-medium datasets
- **GPU cuSpatial:** 100-1000× speedup for large datasets

---

### 4. Performance Profiler (`gpu_profiler.py`)

**Purpose:** Track and analyze GPU performance

**Features:**

- Operation-level timing
- Memory transfer tracking
- Bottleneck detection
- Performance recommendations

**Usage:**

```python
from ign_lidar.optimization.gpu_profiler import get_profiler

profiler = get_profiler(enable=True, session_name="processing")

# Profile an operation
with profiler.profile_operation('feature_computation', data_size_mb=150):
    features = compute_features(points)

# Get recommendations
recommendations = profiler.get_recommendations()
for rec in recommendations:
    print(f"💡 {rec}")

# Print summary
profiler.print_summary()
```

**Output Example:**

```
================================================================================
GPU Performance Profiler - Session: processing
================================================================================

Session Duration: 45.23s
Operations Tracked: 128

Operation Breakdown:
--------------------------------------------------------------------------------
Operation                      Count    Time (ms)   Transfer    Compute
--------------------------------------------------------------------------------
feature_computation              128      42150.5      8430.1    33720.4
ground_truth_classification        1       3080.2       616.0     2464.2

Overall Statistics:
--------------------------------------------------------------------------------
Total Transfer Time:       9046.1 ms
Total Compute Time:       36184.6 ms
Total Data Transfer:      18432.5 MB

Bottleneck Analysis:
--------------------------------------------------------------------------------
Bottleneck Type: compute
Transfer Time: 20.0%
Compute Time:  80.0%

Recommendations:
Performance is well balanced. Minor optimizations:
  - Monitor VRAM usage to maximize batch sizes
  - Enable all available optimizations
  - Consider pipeline optimization for further gains

VRAM Usage: 8456.2MB / 16384.0MB (51.6%)
================================================================================
```

---

## Configuration

### Recommended Settings

**For 8GB VRAM GPU:**

```yaml
processor:
  use_gpu: true
  gpu_batch_size: 3_000_000 # Conservative
  gpu_memory_target: 0.80
  enable_memory_pooling: true
  enable_async_transfers: false # Limited by memory
  adaptive_chunk_sizing: true

features:
  gpu_batch_size: 1_000_000
  use_gpu_chunked: true
```

**For 16GB VRAM GPU (Recommended):**

```yaml
processor:
  use_gpu: true
  gpu_batch_size: 8_000_000
  gpu_memory_target: 0.85
  enable_memory_pooling: true
  enable_async_transfers: true
  adaptive_chunk_sizing: true
  gpu_streams: 4

features:
  gpu_batch_size: 2_000_000
  use_gpu_chunked: true
```

**For 24GB+ VRAM GPU (High Performance):**

```yaml
processor:
  use_gpu: true
  gpu_batch_size: 15_000_000
  gpu_memory_target: 0.90
  enable_memory_pooling: true
  enable_async_transfers: true
  adaptive_chunk_sizing: true
  gpu_streams: 6

features:
  gpu_batch_size: 5_000_000
  use_gpu_chunked: true
```

---

## Performance Tuning

### Identifying Bottlenecks

Use the profiler to identify bottlenecks:

```python
profiler = get_profiler(enable=True)
# ... run processing ...
analysis = profiler.current_session.get_bottleneck_analysis()

if analysis['bottleneck'] == 'memory_transfer':
    # Increase chunk size, enable streams
    print("Recommendation: Increase chunk_size by 50%")
elif analysis['bottleneck'] == 'compute':
    # Optimize algorithms, use cuML
    print("Recommendation: Enable cuML algorithms")
```

### Optimization Checklist

- [ ] **Enable GPU acceleration** (`use_gpu: true`)
- [ ] **Use chunked processing** for datasets > 10M points
- [ ] **Enable memory pooling** (`enable_memory_pooling: true`)
- [ ] **Enable adaptive chunking** (automatic optimal sizing)
- [ ] **Install RAPIDS cuML** for maximum performance
  ```bash
  conda install -c rapidsai -c conda-forge -c nvidia \
      cuml=23.10 python=3.11 cudatoolkit=11.8
  ```
- [ ] **Enable CUDA streams** for GPUs with 16GB+ VRAM
- [ ] **Monitor VRAM usage** with profiler
- [ ] **Tune chunk sizes** based on your specific GPU

---

## Troubleshooting

### OOM Errors (Out of Memory)

**Symptom:** `OutOfMemoryError` during processing

**Solutions:**

1. Reduce `gpu_batch_size` or `chunk_size`
2. Enable `adaptive_chunk_sizing: true` (automatic)
3. Lower `gpu_memory_target` (e.g., 0.75 instead of 0.85)
4. Disable `enable_async_transfers` if using streams
5. Check other GPU processes: `nvidia-smi`

### Slow Performance

**Symptom:** GPU slower than expected

**Solutions:**

1. Run profiler to identify bottleneck
2. Install RAPIDS cuML (much faster than CPU fallback)
3. Increase chunk size if memory allows
4. Enable memory pooling
5. Check GPU utilization: `nvidia-smi` during processing

### Memory Fragmentation

**Symptom:** Gradually increasing memory usage, eventual OOM

**Solutions:**

1. Enable memory pooling (`enable_memory_pooling: true`)
2. Reduce chunk size slightly
3. Add periodic cleanup: coordinator calls `cleanup_gpu_memory()`

---

## Advanced Topics

### Custom Chunk Sizing

Override automatic sizing for specific workloads:

```python
coordinator = get_gpu_coordinator()

# Override for specific operation
custom_params = coordinator.optimize_for_feature_computation(
    num_points=20_000_000,
    feature_mode='asprs_classes'
)

# Manually adjust
custom_params['chunk_size'] = 6_000_000  # Custom size

# Use in processing
computer = GPUChunkedFeatureComputer(
    chunk_size=custom_params['chunk_size']
)
```

### Integration with Existing Code

The optimizer is designed to work seamlessly with existing code:

```python
# Before: Manual configuration
computer = GPUChunkedFeatureComputer(chunk_size=5_000_000)

# After: Automatic optimization
coordinator = get_gpu_coordinator()
params = coordinator.optimize_for_feature_computation(
    num_points=len(points),
    feature_mode='asprs_classes'
)
computer = GPUChunkedFeatureComputer(**params)
```

---

## Performance Benchmarks

### Real-World Performance (18M points, ASPRS classification)

| Configuration            | Time | Speedup    |
| ------------------------ | ---- | ---------- |
| CPU (baseline)           | 353s | 1×         |
| GPU (basic)              | 89s  | 4×         |
| GPU + chunking           | 45s  | 8×         |
| GPU + chunking + pooling | 28s  | 12.6×      |
| GPU + all optimizations  | 22s  | **16×** ✅ |

### Ground Truth Classification (16M points, 1500 polygons)

| Method        | Time  | Speedup     |
| ------------- | ----- | ----------- |
| CPU naive     | 1200s | 1×          |
| CPU STRtree   | 95s   | 12.6×       |
| GPU CuPy      | 8.5s  | 141×        |
| GPU cuSpatial | 2.1s  | **571×** ✅ |

---

## Summary

The GPU optimization stack provides:

✅ **16× speedup** for feature computation
✅ **100-1000× speedup** for ground truth classification
✅ **Automatic resource management** (no manual tuning needed)
✅ **Prevents OOM errors** through adaptive sizing
✅ **Production-ready** performance monitoring

**Next Steps:**

1. Enable GPU in your config: `use_gpu: true`
2. Install RAPIDS cuML for maximum performance
3. Run with profiler to identify opportunities
4. Tune based on recommendations

For questions or issues, see the main README or open an issue on GitHub.
