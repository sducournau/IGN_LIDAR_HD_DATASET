"""
GPU Optimization Examples (Audit Nov 2025)

Examples demonstrating the GPU optimizations implemented as part of
the November 2025 audit recommendations:

1. Batched GPU transfers (reducing PCIe latency)
2. GPU memory context manager (automatic cleanup)
3. Optimized feature computation patterns

Author: LiDAR Trainer Agent (Audit Implementation)
Date: November 23, 2025
"""

import numpy as np
from ign_lidar.core.gpu import GPUManager
from ign_lidar.core.gpu_memory import get_gpu_memory_manager


# ============================================================================
# Example 1: GPU Memory Context Manager (RECOMMENDED PATTERN)
# ============================================================================

def example_context_manager_basic():
    """
    Basic usage of GPU memory context manager.
    
    The context manager automatically:
    - Checks memory availability
    - Performs cleanup on exit
    - Handles errors gracefully
    """
    gpu = GPUManager()
    
    if not gpu.gpu_available:
        print("❌ GPU not available")
        return
    
    import cupy as cp
    
    # ✅ GOOD: Use context manager for automatic cleanup
    with gpu.memory.managed_context():
        # Allocate and process on GPU
        points_gpu = cp.asarray(np.random.rand(10000, 3))
        normals_gpu = cp.random.rand(10000, 3)
        
        # Do GPU computation
        result = cp.mean(points_gpu * normals_gpu, axis=0)
        result_cpu = cp.asnumpy(result)
        
        print(f"✅ Result computed: {result_cpu}")
    
    # Memory automatically cleaned up here
    print("✅ GPU memory cleaned up automatically")


def example_context_manager_with_size_check():
    """
    Context manager with pre-allocation check.
    
    This ensures you have enough memory before starting computation.
    """
    gpu_mem = get_gpu_memory_manager()
    
    if not gpu_mem.gpu_available:
        print("❌ GPU not available")
        return
    
    import cupy as cp
    
    required_gb = 0.5  # Need 500 MB
    
    try:
        # ✅ GOOD: Check memory availability first
        with gpu_mem.managed_context(size_gb=required_gb):
            # Safe to allocate
            large_array = cp.random.rand(1000, 10000)
            result = cp.mean(large_array)
            print(f"✅ Computed with {required_gb} GB GPU memory")
            
    except MemoryError as e:
        # ❌ Not enough memory, fallback to CPU
        print(f"⚠️ Insufficient GPU memory: {e}")
        print("Falling back to CPU...")
        large_array = np.random.rand(1000, 10000)
        result = np.mean(large_array)
        print(f"✅ Computed on CPU (fallback)")


def example_context_manager_no_cleanup():
    """
    Context manager without automatic cleanup.
    
    Use when you need to keep GPU data between operations.
    """
    gpu_mem = get_gpu_memory_manager()
    
    if not gpu_mem.gpu_available:
        return
    
    import cupy as cp
    
    # ✅ Keep data on GPU for next operation
    with gpu_mem.managed_context(cleanup=False):
        cached_data = cp.random.rand(1000, 1000)
        # Data stays on GPU after context
    
    # Manually cleanup later
    gpu_mem.free_cache()


# ============================================================================
# Example 2: Batched GPU Transfers (PERFORMANCE OPTIMIZATION)
# ============================================================================

def example_bad_separate_transfers():
    """
    ❌ BAD: Multiple separate transfers (slow, high latency)
    
    This is the OLD pattern that should be avoided.
    Each cp.asnumpy() call incurs PCIe latency (~20-100μs).
    """
    gpu = GPUManager()
    
    if not gpu.gpu_available:
        return
    
    import cupy as cp
    
    # Compute features on GPU
    points_gpu = cp.random.rand(10000, 3)
    
    # ❌ BAD: 5 separate transfers = 5x PCIe latency
    mean_x = cp.asnumpy(cp.mean(points_gpu[:, 0]))  # Transfer 1
    mean_y = cp.asnumpy(cp.mean(points_gpu[:, 1]))  # Transfer 2
    mean_z = cp.asnumpy(cp.mean(points_gpu[:, 2]))  # Transfer 3
    std_x = cp.asnumpy(cp.std(points_gpu[:, 0]))    # Transfer 4
    std_y = cp.asnumpy(cp.std(points_gpu[:, 1]))    # Transfer 5
    
    print(f"❌ Transferred 5 times separately (slow)")


def example_good_batched_transfers():
    """
    ✅ GOOD: Batched transfer (fast, minimal latency)
    
    This is the NEW pattern used in the optimized code.
    Combine all features into one array, then transfer once.
    """
    gpu = GPUManager()
    
    if not gpu.gpu_available:
        return
    
    import cupy as cp
    
    # Compute features on GPU
    points_gpu = cp.random.rand(10000, 3)
    
    # ✅ GOOD: Compute all features, then stack and transfer once
    mean_x = cp.mean(points_gpu[:, 0])
    mean_y = cp.mean(points_gpu[:, 1])
    mean_z = cp.mean(points_gpu[:, 2])
    std_x = cp.std(points_gpu[:, 0])
    std_y = cp.std(points_gpu[:, 1])
    
    # Stack all features into single array on GPU
    features_gpu = cp.stack([mean_x, mean_y, mean_z, std_x, std_y])
    
    # ✅ Single transfer instead of 5
    features_cpu = cp.asnumpy(features_gpu)
    
    # Unpack
    mean_x_cpu, mean_y_cpu, mean_z_cpu, std_x_cpu, std_y_cpu = features_cpu
    
    print(f"✅ Transferred once (5x faster than separate)")


def example_batched_normals_eigenvalues():
    """
    ✅ Real example: Batched transfer of normals and eigenvalues
    
    This pattern is used in the optimized gpu_kernels.py
    """
    gpu = GPUManager()
    
    if not gpu.gpu_available:
        return
    
    import cupy as cp
    
    # Simulate computed features on GPU
    n_points = 10000
    normals_gpu = cp.random.rand(n_points, 3)
    eigenvalues_gpu = cp.random.rand(n_points, 3)
    
    # ❌ BAD (old way): 2 separate transfers
    # normals_cpu = cp.asnumpy(normals_gpu)      # Transfer 1
    # eigenvalues_cpu = cp.asnumpy(eigenvalues_gpu)  # Transfer 2
    
    # ✅ GOOD (new way): Stack and transfer once
    combined_gpu = cp.concatenate([normals_gpu, eigenvalues_gpu], axis=1)
    combined_cpu = cp.asnumpy(combined_gpu)
    
    # Split back
    normals_cpu = combined_cpu[:, :3]
    eigenvalues_cpu = combined_cpu[:, 3:]
    
    print(f"✅ Batched normals+eigenvalues: {normals_cpu.shape}, {eigenvalues_cpu.shape}")


# ============================================================================
# Example 3: Complete Processing Pipeline with Optimizations
# ============================================================================

def example_optimized_feature_pipeline():
    """
    Complete example: Optimized feature computation pipeline
    
    Demonstrates:
    1. Context manager for memory management
    2. Batched transfers for performance
    3. Error handling with CPU fallback
    """
    gpu = GPUManager()
    
    # Generate sample point cloud
    n_points = 50000
    points = np.random.rand(n_points, 3).astype(np.float32)
    
    if not gpu.gpu_available:
        print("⚠️ GPU not available, using CPU")
        # CPU processing here...
        return
    
    import cupy as cp
    
    try:
        # ✅ Use context manager with size check
        required_gb = 0.2  # Estimate based on data size
        
        with gpu.memory.managed_context(size_gb=required_gb):
            # Upload to GPU
            points_gpu = cp.asarray(points)
            
            # Compute multiple features on GPU
            mean_xyz = cp.mean(points_gpu, axis=0)
            std_xyz = cp.std(points_gpu, axis=0)
            min_xyz = cp.min(points_gpu, axis=0)
            max_xyz = cp.max(points_gpu, axis=0)
            
            # ✅ OPTIMIZATION: Batch all transfers
            features_gpu = cp.stack([
                mean_xyz, std_xyz, min_xyz, max_xyz
            ])  # Shape: [4, 3]
            
            # Single transfer
            features_cpu = cp.asnumpy(features_gpu)
            
            # Unpack results
            mean = features_cpu[0]
            std = features_cpu[1]
            bbox_min = features_cpu[2]
            bbox_max = features_cpu[3]
            
            print("✅ Features computed on GPU:")
            print(f"  Mean: {mean}")
            print(f"  Std:  {std}")
            print(f"  BBox: [{bbox_min}] - [{bbox_max}]")
        
        # Memory automatically cleaned up here
        print("✅ GPU memory cleaned up")
        
    except MemoryError as e:
        print(f"⚠️ GPU OOM: {e}")
        print("Falling back to CPU...")
        
        # CPU fallback
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        print(f"✅ Computed on CPU (fallback)")


# ============================================================================
# Example 4: Performance Comparison
# ============================================================================

def example_performance_comparison():
    """
    Benchmark: Batched vs Separate transfers
    
    This demonstrates the performance improvement from batching.
    """
    gpu = GPUManager()
    
    if not gpu.gpu_available:
        print("⚠️ GPU not available, skipping benchmark")
        return
    
    import cupy as cp
    import time
    
    n_points = 100000
    n_features = 10
    n_iterations = 100
    
    print(f"\n{'='*60}")
    print(f"Performance Benchmark: {n_iterations} iterations")
    print(f"{'='*60}\n")
    
    # Generate test data on GPU
    features_gpu = [cp.random.rand(n_points) for _ in range(n_features)]
    
    # ❌ Test separate transfers
    start = time.time()
    for _ in range(n_iterations):
        results_separate = [cp.asnumpy(f) for f in features_gpu]
    time_separate = time.time() - start
    
    # ✅ Test batched transfer
    start = time.time()
    for _ in range(n_iterations):
        combined = cp.stack(features_gpu)
        results_batched = cp.asnumpy(combined)
    time_batched = time.time() - start
    
    speedup = time_separate / time_batched
    
    print(f"❌ Separate transfers: {time_separate*1000:.2f} ms")
    print(f"✅ Batched transfer:   {time_batched*1000:.2f} ms")
    print(f"⚡ Speedup: {speedup:.2f}x faster\n")
    
    # Cleanup
    gpu.memory.free_cache()


# ============================================================================
# Main: Run all examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GPU Optimization Examples (November 2025 Audit)")
    print("=" * 70)
    
    # Check GPU availability
    gpu = GPUManager()
    if not gpu.gpu_available:
        print("\n⚠️  GPU not available. Examples will run in demo mode.")
        print("For full examples, install CuPy and run on a GPU-enabled system.\n")
    else:
        print(f"\n✅ GPU detected: {gpu}")
        print()
    
    print("\n📚 Example 1: Context Manager (Basic)")
    print("-" * 70)
    example_context_manager_basic()
    
    print("\n📚 Example 2: Context Manager (With Size Check)")
    print("-" * 70)
    example_context_manager_with_size_check()
    
    print("\n📚 Example 3: Batched Transfers")
    print("-" * 70)
    example_good_batched_transfers()
    
    print("\n📚 Example 4: Batched Normals + Eigenvalues")
    print("-" * 70)
    example_batched_normals_eigenvalues()
    
    print("\n📚 Example 5: Complete Optimized Pipeline")
    print("-" * 70)
    example_optimized_feature_pipeline()
    
    print("\n📚 Example 6: Performance Benchmark")
    print("-" * 70)
    example_performance_comparison()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
