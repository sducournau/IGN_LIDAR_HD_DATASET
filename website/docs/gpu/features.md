---
sidebar_position: 2
title: "GPU Feature Computation"
description: "Technical details of GPU-accelerated feature extraction"
keywords: [gpu, features, performance, cupy, benchmarks, api]
---

# GPU Feature Computation

**Available in:** v1.3.0+  
**Acceleration:** 5-10x speedup over CPU  
**Fixed in v1.6.2:** GPU formulas now match CPU (see breaking changes below)

:::warning Breaking Change in v1.6.2
GPU feature formulas were corrected to match CPU and standard literature (Weinmann et al., 2015). If you used GPU acceleration in v1.6.1 or earlier, feature values have changed. You'll need to retrain models or switch to CPU for compatibility with old models.
:::

This guide covers the technical details of GPU-accelerated feature computation, including which features are accelerated, API reference, and advanced optimization techniques.

## Features Accelerated

The following features are computed on GPU when GPU acceleration is enabled:

### Core Geometric Features

- ✅ **Surface normals** (nx, ny, nz) - Normal vectors for each point
- ✅ **Curvature** values - Surface curvature at each point
- ✅ **Height above ground** - Normalized height values

### Advanced Geometric Features

- ✅ **Planarity** - Measure of how flat a surface is (useful for roofs, roads)
- ✅ **Linearity** - Measure of linear structures (useful for edges, cables)
- ✅ **Sphericity** - Measure of spherical structures (useful for vegetation)
- ✅ **Anisotropy** - Directional structure measure
- ✅ **Roughness** - Surface texture and irregularity
- ✅ **Local density** - Point density in local neighborhood

### Building-Specific Features

- ✅ **Verticality** - Measure of vertical alignment (walls)
- ✅ **Horizontality** - Measure of horizontal alignment (roofs, floors)
- ✅ **Wall score** - Probability of being a wall element
- ✅ **Roof score** - Probability of being a roof element

### Performance by Feature Type

| Feature Type          | CPU Time | GPU Time | Speedup |
| --------------------- | -------- | -------- | ------- |
| Surface Normals       | 2.5s     | 0.3s     | 8.3x    |
| Curvature             | 3.0s     | 0.4s     | 7.5x    |
| Height Above Ground   | 1.5s     | 0.2s     | 7.5x    |
| Geometric Features    | 4.0s     | 0.6s     | 6.7x    |
| Building Features     | 5.0s     | 0.8s     | 6.3x    |
| **Total (1M points)** | **16s**  | **2.3s** | **7x**  |

## What Changed in v1.6.2

### Formula Corrections

GPU formulas were corrected to match CPU and standard literature:

**Before v1.6.2** (INCORRECT):

```python
planarity = (λ1 - λ2) / λ0  # Wrong normalization
linearity = (λ0 - λ1) / λ0  # Wrong normalization
sphericity = λ2 / λ0         # Wrong normalization
```

**v1.6.2+** (CORRECT - matches [Weinmann et al., 2015](https://www.sciencedirect.com/science/article/pii/S0924271615001842)):

```python
sum_λ = λ0 + λ1 + λ2
planarity = (λ1 - λ2) / sum_λ   # Standard formulation
linearity = (λ0 - λ1) / sum_λ   # Standard formulation
sphericity = λ2 / sum_λ          # Standard formulation
```

### New Robustness Features

1. **Degenerate Case Filtering**: Points with insufficient neighbors or near-zero eigenvalues now return 0.0 instead of NaN/Inf
2. **Robust Curvature**: Uses Median Absolute Deviation (MAD) instead of std for outlier resistance
3. **Radius Search Support**: Optional radius-based neighbor search (falls back to CPU)

### Validation

GPU now produces identical results to CPU (validated: max difference < 0.0001%):

```python
# Run validation test
python tests/test_feature_fixes.py
# Expected: ✓✓✓ ALL TESTS PASSED ✓✓✓
```

For more details, see:

- [v1.6.2 Release Notes](/docs/release-notes/v1.6.2)
- Repository files: `GEOMETRIC_FEATURES_ANALYSIS.md`, `IMPLEMENTATION_SUMMARY.md`

---

## API Reference

### GPUFeatureComputer Class

The main class for GPU-accelerated feature computation.

```python
from ign_lidar.features_gpu import GPUFeatureComputer

# Initialize GPU feature computer
computer = GPUFeatureComputer(
    use_gpu=True,
    batch_size=100000,
    memory_limit=0.8,
    device_id=0
)
```

#### Constructor Parameters

| Parameter      | Type  | Default  | Description                    |
| -------------- | ----- | -------- | ------------------------------ |
| `use_gpu`      | bool  | `True`   | Enable GPU acceleration        |
| `batch_size`   | int   | `100000` | Points processed per GPU batch |
| `memory_limit` | float | `0.8`    | GPU memory usage limit (0-1)   |
| `device_id`    | int   | `0`      | CUDA device ID (for multi-GPU) |

### Main Methods

#### compute_all_features_with_gpu()

Compute all features for a point cloud using GPU acceleration.

```python
from ign_lidar.features import compute_all_features_with_gpu
import numpy as np

# Your point cloud data
points = np.random.rand(1000000, 3).astype(np.float32)
classification = np.random.randint(0, 10, 1000000).astype(np.uint8)

# Compute features
normals, curvature, height, geo_features = compute_all_features_with_gpu(
    points=points,
    classification=classification,
    k=20,
    auto_k=False,
    use_gpu=True,
    batch_size=100000
)
```

**Parameters:**

| Parameter        | Type       | Required | Description                                              |
| ---------------- | ---------- | -------- | -------------------------------------------------------- |
| `points`         | np.ndarray | Yes      | Point coordinates (N, 3)                                 |
| `classification` | np.ndarray | Yes      | Point classifications (N,)                               |
| `k`              | int        | No       | Number of neighbors for features (default: 20)           |
| `auto_k`         | bool       | No       | Automatically adjust k based on density (default: False) |
| `use_gpu`        | bool       | No       | Enable GPU acceleration (default: True)                  |
| `batch_size`     | int        | No       | Batch size for GPU processing (default: 100000)          |

**Returns:**

| Return Value   | Type       | Shape  | Description                      |
| -------------- | ---------- | ------ | -------------------------------- |
| `normals`      | np.ndarray | (N, 3) | Surface normal vectors           |
| `curvature`    | np.ndarray | (N,)   | Curvature values                 |
| `height`       | np.ndarray | (N,)   | Height above ground              |
| `geo_features` | dict       | -      | Dictionary of geometric features |

#### compute_normals_gpu()

Compute surface normals using GPU.

```python
from ign_lidar.features_gpu import compute_normals_gpu

normals = compute_normals_gpu(
    points=points,
    k=20,
    batch_size=100000
)
```

#### compute_curvature_gpu()

Compute curvature values using GPU.

```python
from ign_lidar.features_gpu import compute_curvature_gpu

curvature = compute_curvature_gpu(
    points=points,
    normals=normals,
    k=20,
    batch_size=100000
)
```

#### compute_geometric_features_gpu()

Compute all geometric features using GPU.

```python
from ign_lidar.features_gpu import compute_geometric_features_gpu

geo_features = compute_geometric_features_gpu(
    points=points,
    normals=normals,
    k=20,
    batch_size=100000
)

# Access individual features
planarity = geo_features['planarity']
linearity = geo_features['linearity']
sphericity = geo_features['sphericity']
```

## Advanced Usage

### Batch Processing Optimization

For processing multiple tiles, reuse the GPU computer instance:

```python
from ign_lidar.features_gpu import GPUFeatureComputer
from pathlib import Path

# Initialize once
computer = GPUFeatureComputer(use_gpu=True, batch_size=100000)

# Process multiple tiles
for tile_path in Path("tiles/").glob("*.laz"):
    # Load tile
    points, classification = load_tile(tile_path)

    # Compute features (GPU stays initialized)
    normals, curvature, height, geo_features = computer.compute_all(
        points=points,
        classification=classification,
        k=20
    )

    # Save results
    save_enriched_tile(tile_path, normals, curvature, height, geo_features)
```

### Memory Management

Control GPU memory usage for large tiles:

```python
from ign_lidar.features_gpu import GPUFeatureComputer

# For very large tiles (>5M points)
computer = GPUFeatureComputer(
    use_gpu=True,
    batch_size=50000,  # Smaller batch size
    memory_limit=0.6   # Use less GPU memory
)

# For small to medium tiles (<1M points)
computer = GPUFeatureComputer(
    use_gpu=True,
    batch_size=200000,  # Larger batch size
    memory_limit=0.9    # Use more GPU memory
)
```

### Multi-GPU Support (Experimental)

:::caution Experimental Feature
Multi-GPU support is experimental in v1.5.0. Use with caution in production.
:::

```python
import os

# Specify GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

# Or for specific GPU
computer = GPUFeatureComputer(use_gpu=True, device_id=1)  # Use second GPU
```

### Custom Feature Computation

Implement custom GPU-accelerated features:

```python
import cupy as cp
from ign_lidar.features_gpu import GPUFeatureComputer

class CustomGPUComputer(GPUFeatureComputer):
    def compute_custom_feature(self, points_gpu):
        """Compute a custom feature on GPU"""
        # Your custom GPU computation using CuPy
        feature = cp.mean(points_gpu, axis=1)
        return cp.asnumpy(feature)

# Use custom computer
computer = CustomGPUComputer(use_gpu=True)
```

## Performance Optimization Tips

### 1. Optimal Batch Size

Choose batch size based on GPU memory:

| GPU Memory | Recommended Batch Size | Max Point Cloud |
| ---------- | ---------------------- | --------------- |
| 4 GB       | 50,000                 | 2M points       |
| 8 GB       | 100,000                | 5M points       |
| 12 GB      | 150,000                | 8M points       |
| 16 GB+     | 200,000+               | 10M+ points     |

### 2. K-Neighbors Selection

Larger k values benefit more from GPU:

```python
# Optimal for GPU (k >= 20)
features = compute_all_features_with_gpu(points, classification, k=20, use_gpu=True)

# Less optimal for GPU (k < 10)
features = compute_all_features_with_gpu(points, classification, k=5, use_gpu=True)
```

### 3. Memory Transfer Optimization

Minimize CPU-GPU transfers by batching operations:

```python
# ❌ Bad: Multiple transfers
normals = compute_normals_gpu(points)
curvature = compute_curvature_gpu(points, normals)  # Transfer normals back
geo = compute_geometric_features_gpu(points, normals)  # Transfer again

# ✅ Good: Single batch
normals, curvature, height, geo = compute_all_features_with_gpu(points, classification)
```

### 4. Persistent GPU Memory

For repeated processing, keep data on GPU:

```python
import cupy as cp

# Transfer to GPU once
points_gpu = cp.asarray(points)

# Process multiple times without re-transfer
for k in [10, 20, 30]:
    normals = compute_normals_gpu(points_gpu, k=k)
```

## Benchmarking

### Running Benchmarks

The library includes comprehensive benchmarking tools:

```bash
# Synthetic benchmark (quick test)
python scripts/benchmarks/benchmark_gpu.py --synthetic

# Real data benchmark
python scripts/benchmarks/benchmark_gpu.py path/to/tile.laz

# Multi-size benchmark
python scripts/benchmarks/benchmark_gpu.py --multi-size

# Compare different k values
python scripts/benchmarks/benchmark_gpu.py --test-k
```

### Interpreting Results

Example benchmark output:

```
GPU Benchmark Results
=====================
GPU Model: NVIDIA RTX 3080 (10GB)
CUDA Version: 11.8
CuPy Version: 11.6.0

Point Cloud: 1,000,000 points
K-neighbors: 20

Feature Computation Times:
--------------------------
Normals (CPU):     2.45s
Normals (GPU):     0.31s  → 7.9x speedup

Curvature (CPU):   2.98s
Curvature (GPU):   0.42s  → 7.1x speedup

Geometric (CPU):   3.87s
Geometric (GPU):   0.58s  → 6.7x speedup

Total (CPU):      15.32s
Total (GPU):       2.14s  → 7.2x speedup

Memory Usage:
-------------
GPU Memory Used:   1.2 GB / 10 GB (12%)
Peak Memory:       1.8 GB
CPU Memory Used:   2.4 GB
```

### Performance Factors

GPU performance depends on:

1. **Point cloud size**: Larger = better GPU utilization
2. **K-neighbors value**: Larger = more parallelizable work
3. **GPU model**: Newer = faster processing
4. **Memory bandwidth**: Higher = faster transfers
5. **CUDA compute capability**: Higher = more features

## Troubleshooting

### Performance Issues

#### GPU Slower Than Expected

**Symptoms**: GPU processing not much faster than CPU

**Possible Causes**:

1. Small point clouds (&lt;10K points) - GPU overhead dominates
2. Low k value (&lt;10) - Not enough parallelizable work
3. Memory transfer bottleneck
4. GPU not fully utilized

**Solutions**:

```bash
# Check GPU utilization
nvidia-smi -l 1

# Should show high GPU usage during processing
# If GPU usage is low:
```

```python
# Increase batch size
computer = GPUFeatureComputer(batch_size=200000)

# Increase k value
features = compute_all_features_with_gpu(points, classification, k=30)

# Use larger tiles or batch processing
```

#### Out of Memory Errors

**Symptoms**: CUDA out of memory errors

**Solutions**:

```python
# Reduce batch size
computer = GPUFeatureComputer(batch_size=50000)

# Reduce memory limit
computer = GPUFeatureComputer(memory_limit=0.6)

# Process in smaller chunks
for chunk in split_point_cloud(points, chunk_size=500000):
    features = compute_all_features_with_gpu(chunk, classification)
```

#### CuPy Import Errors

**Symptoms**: ImportError or CUDA version mismatch warnings

**Solutions**:

```bash
# Check CUDA version
nvidia-smi | grep "CUDA Version"

# Reinstall matching CuPy
pip uninstall cupy
pip install cupy-cuda11x  # or cupy-cuda12x
```

### Memory Leaks

If GPU memory keeps increasing:

```python
# Force GPU memory cleanup
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()

# Or use context manager
from ign_lidar.features_gpu import GPUMemoryManager

with GPUMemoryManager():
    # GPU memory automatically freed after this block
    features = compute_all_features_with_gpu(points, classification)
```

## Limitations

### Current Limitations

1. **GPU Memory**: Limited by available GPU RAM
2. **Single GPU**: Multi-GPU support is experimental
3. **NVIDIA Only**: Requires NVIDIA GPU with CUDA
4. **K-NN Implementation**: Uses brute-force for k < 50, KD-tree for k >= 50

### Future Improvements (Roadmap)

- 🔄 **Multi-GPU support** (v1.6.0) - Distribute work across multiple GPUs
- 🔄 **Mixed precision** (v1.6.0) - Use FP16 for faster computation
- 🔄 **AMD GPU support** (v2.0.0) - ROCm support for AMD GPUs
- 🔄 **Chunked processing** (v1.6.0) - Automatic chunking for very large tiles
- 🔄 **Persistent GPU cache** (v1.7.0) - Cache preprocessed data on GPU

## See Also

- **[GPU Overview](overview.md)** - GPU setup and installation
- **[RGB GPU Acceleration](rgb-augmentation.md)** - GPU-accelerated RGB augmentation
- **[Architecture](../architecture.md)** - System architecture
- **[Workflows](../workflows.md)** - GPU workflow examples

## References

- [CuPy Documentation](https://docs.cupy.dev/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [GPU Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
