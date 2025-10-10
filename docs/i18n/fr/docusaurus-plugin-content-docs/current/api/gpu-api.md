---
sidebar_position: 5
title: API GPU
description: Référence API d'accélération GPU pour le traitement LiDAR haute performance
keywords: [gpu, api, cuda, acceleration, pytorch, tensor]
---

<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->
<!-- Ce fichier est un modèle qui nécessite une traduction manuelle. -->
<!-- Veuillez traduire le contenu ci-dessous en conservant : -->
<!-- - Le frontmatter (métadonnées en haut) -->
<!-- - Les blocs de code (traduire uniquement les commentaires) -->
<!-- - Les liens et chemins de fichiers -->
<!-- - La structure Markdown -->

# GPU API Reference

Complete API documentation for GPU-accelerated LiDAR processing components.

## Core GPU Classes

### GPUProcessor

Main GPU-accelerated processor for LiDAR data.

```python
from ign_lidar.gpu import GPUProcessor

processor = GPUProcessor(
    device="cuda:0",
    batch_size=10000,
    memory_fraction=0.8
)
```

#### Parameters

- **device** (`str`): GPU device identifier (`"cuda:0"`, `"cuda:1"`, etc.)
- **batch_size** (`int`): Number of points processed per batch
- **memory_fraction** (`float`): Fraction of GPU memory to use (0.1-1.0)
- **mixed_precision** (`bool`): Enable automatic mixed precision (FP16/FP32)
- **compile_models** (`bool`): Use PyTorch 2.0 model compilation

#### Methods

##### `process_points(points, features=None)`

Process LiDAR points on GPU.

```python
import torch
from ign_lidar.gpu import GPUProcessor

processor = GPUProcessor(device="cuda:0")

# Input tensor (N, 3) for XYZ coordinates
points = torch.tensor([[x1, y1, z1], [x2, y2, z2], ...], device="cuda:0")

# Process points
result = processor.process_points(
    points=points,
    features=["buildings", "vegetation"]
)

# Result structure
{
    'classifications': torch.Tensor,  # (N,) class labels
    'features': torch.Tensor,        # (N, F) feature vectors
    'confidence': torch.Tensor,      # (N,) confidence scores
    'processing_time': float         # GPU processing time
}
```

##### `extract_buildings_gpu(points, **kwargs)`

GPU-accelerated building extraction.

```python
buildings = processor.extract_buildings_gpu(
    points=points,
    min_points=100,
    height_threshold=2.0,
    planarity_threshold=0.1,
    return_meshes=True
)

# Returns
{
    'building_points': torch.Tensor,    # (M, 3) building points
    'building_ids': torch.Tensor,       # (M,) building instance IDs
    'meshes': List[torch.Tensor],       # 3D building meshes
    'properties': Dict                  # Building properties
}
```

##### `rgb_augmentation_gpu(points, orthophoto, **kwargs)`

GPU-accelerated RGB augmentation.

```python
# Load orthophoto as tensor
orthophoto = torch.from_numpy(orthophoto_array).cuda()

augmented = processor.rgb_augmentation_gpu(
    points=points,
    orthophoto=orthophoto,
    interpolation="bilinear",
    batch_size=50000
)

# Returns points with RGB colors
{
    'points': torch.Tensor,          # (N, 3) XYZ coordinates
    'colors': torch.Tensor,          # (N, 3) RGB colors [0-255]
    'interpolation_quality': torch.Tensor  # (N,) quality scores
}
```

### GPUMemoryManager

Manage GPU memory allocation and optimization.

```python
from ign_lidar.gpu import GPUMemoryManager

memory_manager = GPUMemoryManager(device="cuda:0")

# Get memory information
info = memory_manager.get_memory_info()
print(f"Available: {info['available']:.2f} GB")
print(f"Total: {info['total']:.2f} GB")

# Optimize batch size based on available memory
optimal_batch = memory_manager.get_optimal_batch_size(
    point_features=7,  # XYZ + RGB + intensity + classification
    model_memory_mb=500
)
```

#### Methods

##### `allocate_tensor_memory(size, dtype=torch.float32)`

Pre-allocate GPU tensors for better memory management.

```python
# Pre-allocate memory for large point clouds
memory_pool = memory_manager.allocate_tensor_memory(
    size=(1000000, 3),  # 1M points, XYZ
    dtype=torch.float32
)

# Use pre-allocated memory
points_tensor = memory_pool[:actual_size]
```

##### `clear_cache()`

Clear GPU memory cache.

```python
# Clear PyTorch cache
memory_manager.clear_cache()

# Get memory freed
freed_mb = memory_manager.get_freed_memory()
print(f"Freed {freed_mb:.1f} MB")
```

### GPUFeatureExtractor

Extract geometric features on GPU.

```python
from ign_lidar.gpu import GPUFeatureExtractor

extractor = GPUFeatureExtractor(
    device="cuda:0",
    neighborhood_size=50,
    feature_types=["eigenvalues", "normals", "curvature"]
)

features = extractor.extract_features(points)
```

#### Supported Features

##### Eigenvalue-based Features

```python
# Extract eigenvalue features
eigenvalue_features = extractor.extract_eigenvalues(
    points=points,
    k_neighbors=20,
    search_radius=1.0
)

# Returns
{
    'eigenvalues': torch.Tensor,     # (N, 3) λ0, λ1, λ2
    'linearity': torch.Tensor,       # (N,) (λ0 - λ1) / λ0
    'planarity': torch.Tensor,       # (N,) (λ1 - λ2) / λ0
    'sphericity': torch.Tensor,      # (N,) λ2 / λ0
    'anisotropy': torch.Tensor,      # (N,) (λ0 - λ2) / λ0
    'eigenvectors': torch.Tensor     # (N, 3, 3) eigenvectors
}
```

##### Normal Estimation

```python
# Compute point normals
normals = extractor.estimate_normals(
    points=points,
    k_neighbors=20,
    orient_normals=True,
    viewpoint=[0, 0, 10]  # Camera/sensor position
)

# Returns (N, 3) normal vectors
```

##### Curvature Computation

```python
# Calculate surface curvature
curvature = extractor.compute_curvature(
    points=points,
    normals=normals,
    method="mean"  # "mean", "gaussian", "principal"
)

# Returns (N,) curvature values
```

## GPU Utilities

### Tensor Operations

```python
from ign_lidar.gpu.utils import (
    points_to_tensor,
    tensor_to_points,
    batch_process,
    knn_search_gpu
)

# Convert numpy points to GPU tensor
points_np = np.array([[x, y, z], ...])
points_gpu = points_to_tensor(points_np, device="cuda:0")

# Batch processing for large datasets
results = batch_process(
    data=large_point_cloud,
    process_func=processor.extract_buildings_gpu,
    batch_size=100000,
    device="cuda:0"
)

# GPU-accelerated k-nearest neighbors
neighbors, distances = knn_search_gpu(
    query_points=points_gpu,
    reference_points=reference_gpu,
    k=20
)
```

### Performance Monitoring

```python
from ign_lidar.gpu.profiling import GPUProfiler, benchmark_gpu

# Profile GPU operations
with GPUProfiler() as profiler:
    result = processor.process_points(points)

profiler.print_summary()
# Output:
# GPU Utilization: 85.3%
# Memory Usage: 6.2/8.0 GB
# Processing Time: 2.35s
# Throughput: 425K points/sec

# Benchmark different configurations
benchmark_results = benchmark_gpu(
    point_cloud=test_points,
    batch_sizes=[5000, 10000, 20000],
    devices=["cuda:0", "cuda:1"]
)
```

## Multi-GPU Support

### DataParallel Processing

```python
from ign_lidar.gpu import MultiGPUProcessor

# Initialize multi-GPU processor
multi_processor = MultiGPUProcessor(
    devices=["cuda:0", "cuda:1"],
    strategy="data_parallel"
)

# Process with multiple GPUs
results = multi_processor.process_batch(
    point_clouds=batch_of_tiles,
    features=["buildings", "vegetation"]
)
```

### Distributed Processing

```python
import torch.distributed as dist
from ign_lidar.gpu.distributed import DistributedProcessor

# Initialize distributed processing
dist.init_process_group(backend="nccl")
processor = DistributedProcessor(
    local_rank=0,
    world_size=4
)

# Distributed feature extraction
features = processor.extract_features_distributed(
    points=points,
    feature_types=["geometric", "radiometric"]
)
```

## Advanced GPU Operations

### Custom CUDA Kernels

```python
from ign_lidar.gpu.kernels import (
    voxelize_cuda,
    ground_segmentation_cuda,
    noise_removal_cuda
)

# Voxelize point cloud on GPU
voxel_grid = voxelize_cuda(
    points=points_gpu,
    voxel_size=0.1,
    max_points_per_voxel=100
)

# GPU ground segmentation
ground_mask = ground_segmentation_cuda(
    points=points_gpu,
    cloth_resolution=0.5,
    iterations=500
)

# GPU noise removal
clean_points = noise_removal_cuda(
    points=points_gpu,
    std_ratio=2.0,
    nb_neighbors=20
)
```

### Mesh Generation

```python
from ign_lidar.gpu.mesh import GPUMeshGenerator

mesh_generator = GPUMeshGenerator(device="cuda:0")

# Generate building meshes
meshes = mesh_generator.generate_building_meshes(
    building_points=building_points,
    method="poisson",
    octree_depth=9
)

# Export meshes
for i, mesh in enumerate(meshes):
    mesh_generator.save_mesh(mesh, f"building_{i}.ply")
```

## Configuration Classes

### GPUConfig

```python
from ign_lidar.gpu import GPUConfig

config = GPUConfig(
    # Device settings
    device="cuda:0",
    devices=["cuda:0", "cuda:1"],  # Multi-GPU
    fallback_to_cpu=True,

    # Memory management
    memory_fraction=0.8,
    pin_memory=True,
    empty_cache_every=100,

    # Performance optimization
    mixed_precision=True,
    compile_models=True,
    benchmark_cudnn=True,

    # Processing parameters
    batch_size="auto",
    num_workers=4,
    prefetch_factor=2,

    # Feature extraction
    neighborhood_size=50,
    feature_cache_size=1000000
)

# Use configuration
processor = GPUProcessor(config=config)
```

### Optimization Settings

```python
# Performance tuning
config.set_optimization_level("aggressive")
# Sets:
# - mixed_precision=True
# - compile_models=True
# - benchmark_cudnn=True
# - memory_fraction=0.9

# Memory-optimized settings
config.set_optimization_level("memory_efficient")
# Sets:
# - memory_fraction=0.6
# - batch_size=5000
# - gradient_checkpointing=True
```

## Error Handling

### GPU-Specific Exceptions

```python
from ign_lidar.gpu.exceptions import (
    GPUNotAvailableError,
    CUDAOutOfMemoryError,
    GPUComputeError
)

try:
    result = processor.process_points(points)
except CUDAOutOfMemoryError as e:
    print(f"GPU out of memory: {e}")
    # Reduce batch size and retry
    processor.batch_size = processor.batch_size // 2
    result = processor.process_points(points)

except GPUComputeError as e:
    print(f"GPU computation failed: {e}")
    # Fall back to CPU processing
    cpu_processor = CPUProcessor()
    result = cpu_processor.process_points(points.cpu())
```

### Automatic Fallback

```python
from ign_lidar.gpu import AdaptiveProcessor

# Processor that automatically handles GPU/CPU fallback
processor = AdaptiveProcessor(
    prefer_gpu=True,
    fallback_to_cpu=True,
    retry_on_oom=True
)

# Automatically handles device failures
result = processor.process_points(points)
```

## Integration Examples

### With Existing Workflows

```python
from ign_lidar import Processor
from ign_lidar.gpu import enable_gpu_acceleration

# Enable GPU acceleration for existing processor
processor = Processor()
enable_gpu_acceleration(
    processor,
    device="cuda:0",
    batch_size=20000
)

# Process as normal - GPU acceleration is transparent
result = processor.process_tile("input.las")
```

### Custom GPU Pipeline

```python
class CustomGPUPipeline:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.feature_extractor = GPUFeatureExtractor(device=device)
        self.classifier = GPUClassifier(device=device)
        self.mesh_generator = GPUMeshGenerator(device=device)

    def process(self, points_tensor):
        # Extract features on GPU
        features = self.feature_extractor.extract_features(points_tensor)

        # Classify points
        classifications = self.classifier.classify(features)

        # Generate meshes for buildings
        building_mask = classifications == BuildingClass.BUILDING
        building_points = points_tensor[building_mask]
        meshes = self.mesh_generator.generate_meshes(building_points)

        return {
            'classifications': classifications,
            'features': features,
            'meshes': meshes
        }

# Usage
pipeline = CustomGPUPipeline(device="cuda:0")
result = pipeline.process(points_gpu)
```

## Best Practices

### Memory Management

```python
# Use context managers for automatic cleanup
from ign_lidar.gpu import gpu_context

with gpu_context(device="cuda:0", memory_fraction=0.8) as ctx:
    processor = GPUProcessor(device=ctx.device)
    result = processor.process_points(points)
# GPU memory automatically cleaned up
```

### Performance Optimization

```python
# Optimal batch size calculation
def calculate_optimal_batch_size(gpu_memory_gb, point_features=7):
    bytes_per_point = point_features * 4  # float32
    safety_factor = 0.8
    max_points = int(gpu_memory_gb * 1e9 * safety_factor / bytes_per_point)
    return min(max_points, 100000)  # Cap at 100k points

batch_size = calculate_optimal_batch_size(8.0)  # 8GB GPU
```

### Error Recovery

```python
def robust_gpu_processing(points, max_retries=3):
    for attempt in range(max_retries):
        try:
            return processor.process_points(points)
        except CUDAOutOfMemoryError:
            if attempt < max_retries - 1:
                # Reduce batch size and clear cache
                processor.batch_size //= 2
                torch.cuda.empty_cache()
                continue
            else:
                # Final fallback to CPU
                return cpu_processor.process_points(points.cpu())
```

## Related Documentation

- [GPU Setup Guide](../installation/gpu-setup)
- [Performance Guide](../guides/performance)
- [GPU Acceleration Guide](../guides/gpu-acceleration)
- [Processor API](./processor)
