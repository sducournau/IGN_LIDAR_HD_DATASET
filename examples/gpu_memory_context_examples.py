"""
GPU Memory Context Manager - Usage Examples

Demonstrates the new memory_context() feature added in v3.5.3 for
automatic GPU memory lifecycle management.

Author: LiDAR Trainer Agent
Date: November 24, 2025
Version: 3.5.3
"""

import numpy as np
from ign_lidar.core.gpu import GPUManager


def example_basic_usage():
    """Basic usage of memory context manager."""
    print("=" * 60)
    print("Example 1: Basic Memory Context Usage")
    print("=" * 60)
    
    gpu = GPUManager()
    
    # Generate sample data
    points = np.random.rand(100000, 3).astype(np.float32)
    
    # Use context manager for automatic memory management
    with gpu.memory_context("basic feature computation"):
        if gpu.gpu_available:
            import cupy as cp
            
            # Upload to GPU
            points_gpu = cp.asarray(points)
            
            # Compute features
            normals_gpu = compute_simple_normals(points_gpu)
            
            # Download results
            normals = cp.asnumpy(normals_gpu)
        else:
            # CPU fallback
            normals = compute_simple_normals_cpu(points)
    
    # Memory automatically cleaned up here
    print(f"✅ Computed normals with shape: {normals.shape}")
    print()


def example_batch_transfers_with_context():
    """Combining memory context with batch transfers."""
    print("=" * 60)
    print("Example 2: Memory Context + Batch Transfers")
    print("=" * 60)
    
    gpu = GPUManager()
    
    if not gpu.gpu_available:
        print("⚠️  GPU not available, skipping GPU example")
        return
    
    import cupy as cp
    
    # Sample data
    points = np.random.rand(50000, 3).astype(np.float32)
    features = np.random.rand(50000, 10).astype(np.float32)
    labels = np.random.randint(0, 5, 50000, dtype=np.int32)
    
    with gpu.memory_context("batch transfer example"):
        # ✅ Batch upload - single transfer
        points_gpu, features_gpu, labels_gpu = gpu.batch_upload(
            points, features, labels
        )
        
        # Process on GPU
        result_gpu = cp.mean(features_gpu[labels_gpu == 1], axis=0)
        
        # ✅ Batch download - single transfer
        result = gpu.batch_download(result_gpu)[0]
    
    # Memory cleaned up automatically
    print(f"✅ Computed mean features: {result[:5]}")
    print()


def example_exception_handling():
    """Memory context handles exceptions gracefully."""
    print("=" * 60)
    print("Example 3: Exception Handling")
    print("=" * 60)
    
    gpu = GPUManager()
    
    if not gpu.gpu_available:
        print("⚠️  GPU not available, skipping GPU example")
        return
    
    import cupy as cp
    
    try:
        with gpu.memory_context("operation with error"):
            # Allocate GPU memory
            data_gpu = cp.arange(1000, dtype=cp.float32)
            
            # Simulate an error
            raise ValueError("Simulated error during processing")
            
    except ValueError as e:
        print(f"⚠️  Caught error: {e}")
        print("✅ Memory was still cleaned up properly")
    
    # GPU is still functional after exception
    test_array = cp.array([1, 2, 3])
    print(f"✅ GPU still works: mean = {cp.mean(test_array)}")
    print()


def example_nested_contexts():
    """Nested memory contexts for complex operations."""
    print("=" * 60)
    print("Example 4: Nested Memory Contexts")
    print("=" * 60)
    
    gpu = GPUManager()
    
    if not gpu.gpu_available:
        print("⚠️  GPU not available, skipping GPU example")
        return
    
    import cupy as cp
    
    points = np.random.rand(10000, 3).astype(np.float32)
    
    with gpu.memory_context("outer: full pipeline"):
        points_gpu = cp.asarray(points)
        
        # Nested context for specific operation
        with gpu.memory_context("inner: normal computation"):
            normals_gpu = compute_simple_normals(points_gpu)
            normals = cp.asnumpy(normals_gpu)
        
        # Continue with outer context
        distances_gpu = cp.linalg.norm(points_gpu, axis=1)
        distances = cp.asnumpy(distances_gpu)
    
    print(f"✅ Computed normals: {normals.shape}")
    print(f"✅ Computed distances: {distances.shape}")
    print()


def example_real_world_feature_pipeline():
    """Real-world example: feature computation pipeline."""
    print("=" * 60)
    print("Example 5: Real-World Feature Pipeline")
    print("=" * 60)
    
    gpu = GPUManager()
    
    # Simulated LiDAR point cloud
    n_points = 100000
    points = np.random.rand(n_points, 3).astype(np.float32)
    
    # Feature computation with memory management
    with gpu.memory_context("LiDAR feature extraction"):
        if gpu.gpu_available:
            import cupy as cp
            
            # Upload data
            points_gpu = cp.asarray(points)
            
            # Compute multiple features
            features = {}
            
            # Geometric features
            features['normals'] = cp.asnumpy(compute_simple_normals(points_gpu))
            features['curvature'] = cp.asnumpy(compute_simple_curvature(points_gpu))
            features['planarity'] = cp.asnumpy(compute_planarity(points_gpu))
            
            print(f"✅ GPU Feature extraction complete")
        else:
            # CPU fallback
            features = {
                'normals': compute_simple_normals_cpu(points),
                'curvature': compute_simple_curvature_cpu(points),
                'planarity': compute_planarity_cpu(points)
            }
            print(f"✅ CPU Feature extraction complete")
    
    # Memory automatically managed
    for name, feat in features.items():
        print(f"   {name}: {feat.shape}")
    print()


# Helper functions (simplified for examples)
def compute_simple_normals(points_gpu):
    """Simplified normal computation on GPU."""
    import cupy as cp
    # Simplified: just normalize point vectors
    norms = cp.linalg.norm(points_gpu, axis=1, keepdims=True)
    return points_gpu / (norms + 1e-10)


def compute_simple_curvature(points_gpu):
    """Simplified curvature computation on GPU."""
    import cupy as cp
    # Simplified: variance of coordinates
    return cp.var(points_gpu, axis=1)


def compute_planarity(points_gpu):
    """Simplified planarity computation on GPU."""
    import cupy as cp
    # Simplified: ratio of min/max variance
    variances = cp.var(points_gpu, axis=0)
    return cp.min(variances) / (cp.max(variances) + 1e-10) * cp.ones(len(points_gpu))


def compute_simple_normals_cpu(points):
    """CPU fallback for normals."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / (norms + 1e-10)


def compute_simple_curvature_cpu(points):
    """CPU fallback for curvature."""
    return np.var(points, axis=1)


def compute_planarity_cpu(points):
    """CPU fallback for planarity."""
    variances = np.var(points, axis=0)
    return np.min(variances) / (np.max(variances) + 1e-10) * np.ones(len(points))


def main():
    """Run all examples."""
    print("\n" + "🔷" * 30)
    print("GPU Memory Context Manager Examples (v3.5.3)")
    print("🔷" * 30 + "\n")
    
    # Check GPU availability
    gpu = GPUManager()
    print(f"GPU Status: {gpu}")
    print()
    
    # Run examples
    example_basic_usage()
    example_batch_transfers_with_context()
    example_exception_handling()
    example_nested_contexts()
    example_real_world_feature_pipeline()
    
    print("=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
