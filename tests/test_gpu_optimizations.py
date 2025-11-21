"""
GPU Optimization Tests

Tests for GPU-accelerated preprocessing and I/O functions.
Validates correctness and measures performance improvements.

Run with:
    pytest tests/test_gpu_optimizations.py -v
    pytest tests/test_gpu_optimizations.py -v -m gpu  # GPU tests only
    pytest tests/test_gpu_optimizations.py -v -m benchmark  # Benchmark tests
"""

import pytest
import numpy as np
import time
from pathlib import Path

# Check if GPU is available
try:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_points():
    """Generate sample point cloud for testing."""
    np.random.seed(42)
    return np.random.randn(10_000, 3).astype(np.float32)  # Reduced: 100k -> 10k


@pytest.fixture
def large_points():
    """Generate large point cloud for benchmarking."""
    np.random.seed(42)
    return np.random.randn(100_000, 3).astype(np.float32)  # Reduced: 1M -> 100k


# ============================================================================
# Test: Statistical Outlier Removal (SOR)
# ============================================================================

@pytest.mark.gpu
def test_sor_gpu_availability(sample_points):
    """Test that GPU SOR is available and can be imported."""
    from ign_lidar.preprocessing import statistical_outlier_removal, GPU_AVAILABLE as PREPROC_GPU
    
    assert PREPROC_GPU, "GPU should be available in preprocessing module"


@pytest.mark.gpu
def test_sor_gpu_correctness(sample_points):
    """Verify GPU SOR produces similar results to CPU version."""
    from ign_lidar.preprocessing import statistical_outlier_removal
    
    # CPU version
    cpu_result, cpu_mask = statistical_outlier_removal(
        sample_points, k=12, std_multiplier=2.0, use_gpu=False
    )
    
    # GPU version
    gpu_result, gpu_mask = statistical_outlier_removal(
        sample_points, k=12, std_multiplier=2.0, use_gpu=True
    )
    
    # Verify shapes match
    assert cpu_result.shape == gpu_result.shape, "CPU and GPU results should have same shape"
    assert cpu_mask.shape == gpu_mask.shape, "CPU and GPU masks should have same shape"
    
    # Verify results are similar (allow small numerical differences)
    assert np.allclose(cpu_result, gpu_result, rtol=1e-3, atol=1e-5), \
        "CPU and GPU results should be similar"
    
    # Verify masks match
    mask_agreement = np.sum(cpu_mask == gpu_mask) / len(cpu_mask)
    assert mask_agreement > 0.95, \
        f"CPU and GPU masks should agree on >95% of points (got {mask_agreement:.1%})"


@pytest.mark.gpu
@pytest.mark.benchmark
def test_sor_gpu_speedup(large_points):
    """Measure GPU speedup for SOR."""
    from ign_lidar.preprocessing import statistical_outlier_removal
    
    # CPU version
    start = time.time()
    cpu_result, _ = statistical_outlier_removal(large_points, k=12, use_gpu=False)
    cpu_time = time.time() - start
    
    # GPU version
    start = time.time()
    gpu_result, _ = statistical_outlier_removal(large_points, k=12, use_gpu=True)
    gpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    
    print(f"\n{'='*60}")
    print(f"SOR Performance (N={len(large_points):,} points)")
    print(f"{'='*60}")
    print(f"CPU time:  {cpu_time:.3f}s")
    print(f"GPU time:  {gpu_time:.3f}s")
    print(f"Speedup:   {speedup:.1f}x")
    print(f"{'='*60}")
    
    # Expect at least 3x speedup (conservative, should be 10-15x)
    assert speedup > 3.0, f"Expected >3x speedup, got {speedup:.1f}x"


# ============================================================================
# Test: Radius Outlier Removal (ROR)
# ============================================================================

@pytest.mark.gpu
def test_ror_gpu_correctness(sample_points):
    """Verify GPU ROR produces similar results to CPU version."""
    from ign_lidar.preprocessing import radius_outlier_removal
    
    # CPU version
    cpu_result, cpu_mask = radius_outlier_removal(
        sample_points, radius=1.0, min_neighbors=4, use_gpu=False
    )
    
    # GPU version
    gpu_result, gpu_mask = radius_outlier_removal(
        sample_points, radius=1.0, min_neighbors=4, use_gpu=True
    )
    
    # Verify shapes match
    assert cpu_result.shape == gpu_result.shape
    assert cpu_mask.shape == gpu_mask.shape
    
    # Verify masks match
    mask_agreement = np.sum(cpu_mask == gpu_mask) / len(cpu_mask)
    assert mask_agreement > 0.95, \
        f"CPU and GPU masks should agree on >95% of points (got {mask_agreement:.1%})"


@pytest.mark.gpu
@pytest.mark.benchmark
def test_ror_gpu_speedup(large_points):
    """Measure GPU speedup for ROR."""
    from ign_lidar.preprocessing import radius_outlier_removal
    
    # CPU version
    start = time.time()
    cpu_result, _ = radius_outlier_removal(large_points, radius=1.0, use_gpu=False)
    cpu_time = time.time() - start
    
    # GPU version
    start = time.time()
    gpu_result, _ = radius_outlier_removal(large_points, radius=1.0, use_gpu=True)
    gpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    
    print(f"\n{'='*60}")
    print(f"ROR Performance (N={len(large_points):,} points)")
    print(f"{'='*60}")
    print(f"CPU time:  {cpu_time:.3f}s")
    print(f"GPU time:  {gpu_time:.3f}s")
    print(f"Speedup:   {speedup:.1f}x")
    print(f"{'='*60}")
    
    # Expect at least 3x speedup
    assert speedup > 3.0, f"Expected >3x speedup, got {speedup:.1f}x"


# ============================================================================
# Test: GPU Wrapper
# ============================================================================

@pytest.mark.gpu
def test_gpu_wrapper_decorator():
    """Test GPU wrapper decorator functionality."""
    from ign_lidar.optimization.gpu_wrapper import gpu_accelerated, check_gpu_available
    
    # Check GPU is available
    assert check_gpu_available(), "GPU should be available for wrapper tests"
    
    # Define test functions
    @gpu_accelerated(cpu_fallback=True)
    def test_function(x, y):
        """CPU version."""
        return x + y
    
    def test_function_gpu(x, y):
        """GPU version."""
        import cupy as cp
        x_gpu = cp.asarray(x)
        y_gpu = cp.asarray(y)
        result = x_gpu + y_gpu
        return cp.asnumpy(result)
    
    # Make GPU function available in test_function's namespace
    test_function.__globals__['test_function_gpu'] = test_function_gpu
    
    # Test CPU path
    result_cpu = test_function(np.array([1, 2, 3]), np.array([4, 5, 6]), use_gpu=False)
    assert np.array_equal(result_cpu, np.array([5, 7, 9]))
    
    # Test GPU path
    result_gpu = test_function(np.array([1, 2, 3]), np.array([4, 5, 6]), use_gpu=True)
    assert np.array_equal(result_gpu, np.array([5, 7, 9]))


@pytest.mark.gpu
def test_gpu_context_manager():
    """Test GPU context manager."""
    from ign_lidar.optimization.gpu_wrapper import GPUContext
    
    with GPUContext() as gpu:
        assert gpu.available, "GPU should be available"
        
        # Test CPU->GPU transfer
        cpu_array = np.array([1, 2, 3, 4, 5])
        gpu_array = gpu.to_gpu(cpu_array)
        
        # Verify it's on GPU
        assert hasattr(gpu_array, 'device'), "Array should be on GPU"
        
        # Test GPU->CPU transfer
        result = gpu.to_cpu(gpu_array)
        assert np.array_equal(result, cpu_array), "Round-trip should preserve data"


# ============================================================================
# Test: KNN Graph Construction
# ============================================================================

@pytest.mark.gpu
def test_knn_graph_gpu_correctness(sample_points):
    """Verify GPU KNN graph produces similar results to CPU."""
    from ign_lidar.io.formatters.multi_arch_formatter import MultiArchitectureFormatter
    
    formatter = MultiArchitectureFormatter()
    
    # CPU version
    edges_cpu, dist_cpu = formatter._build_knn_graph(sample_points[:500], k=16, use_gpu=False)
    
    # GPU version
    edges_gpu, dist_gpu = formatter._build_knn_graph(sample_points[:500], k=16, use_gpu=True)
    
    # Verify shapes
    assert edges_cpu.shape == edges_gpu.shape
    assert dist_cpu.shape == dist_gpu.shape
    
    # Verify edges are similar (neighbors may differ slightly due to numerical precision)
    edge_agreement = np.sum(edges_cpu == edges_gpu) / edges_cpu.size
    assert edge_agreement > 0.90, \
        f"CPU and GPU graphs should agree on >90% of edges (got {edge_agreement:.1%})"


@pytest.mark.gpu
@pytest.mark.benchmark
def test_knn_graph_gpu_speedup(large_points):
    """Measure GPU speedup for KNN graph construction."""
    from ign_lidar.io.formatters.multi_arch_formatter import MultiArchitectureFormatter
    
    formatter = MultiArchitectureFormatter()
    points_subset = large_points[:10_000]  # Use 10k points (reduced for memory)
    
    # CPU version
    start = time.time()
    edges_cpu, _ = formatter._build_knn_graph(points_subset, k=32, use_gpu=False)
    cpu_time = time.time() - start
    
    # GPU version
    start = time.time()
    edges_gpu, _ = formatter._build_knn_graph(points_subset, k=32, use_gpu=True)
    gpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    
    print(f"\n{'='*60}")
    print(f"KNN Graph Performance (N={len(points_subset):,} points, k=32)")
    print(f"{'='*60}")
    print(f"CPU time:  {cpu_time:.3f}s")
    print(f"GPU time:  {gpu_time:.3f}s")
    print(f"Speedup:   {speedup:.1f}x")
    print(f"{'='*60}")
    
    # Expect at least 5x speedup
    assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.1f}x"


# ============================================================================
# Test: Consistency Across Use Cases
# ============================================================================

@pytest.mark.gpu
@pytest.mark.parametrize("use_gpu", [False, True])
def test_preprocessing_pipeline_consistency(sample_points, use_gpu):
    """Ensure preprocessing pipeline works with both CPU and GPU."""
    from ign_lidar.preprocessing import (
        statistical_outlier_removal,
        radius_outlier_removal
    )
    
    # Run full preprocessing pipeline
    points_sor, _ = statistical_outlier_removal(
        sample_points, k=12, std_multiplier=2.0, use_gpu=use_gpu
    )
    
    points_ror, _ = radius_outlier_removal(
        points_sor, radius=1.0, min_neighbors=4, use_gpu=use_gpu
    )
    
    # Verify we got valid results
    assert len(points_ror) > 0, "Preprocessing should return some points"
    assert points_ror.shape[1] == 3, "Should maintain 3D coordinates"
    assert np.isfinite(points_ror).all(), "All coordinates should be finite"


# ============================================================================
# Summary Report
# ============================================================================

@pytest.mark.gpu
@pytest.mark.benchmark
def test_overall_performance_summary(large_points):
    """Generate comprehensive performance summary."""
    from ign_lidar.preprocessing import (
        statistical_outlier_removal,
        radius_outlier_removal
    )
    
    print("\n" + "="*80)
    print("GPU OPTIMIZATION PERFORMANCE SUMMARY")
    print("="*80)
    
    # Test SOR
    start = time.time()
    statistical_outlier_removal(large_points, k=12, use_gpu=False)
    sor_cpu = time.time() - start
    
    start = time.time()
    statistical_outlier_removal(large_points, k=12, use_gpu=True)
    sor_gpu = time.time() - start
    
    # Test ROR
    start = time.time()
    radius_outlier_removal(large_points, radius=1.0, use_gpu=False)
    ror_cpu = time.time() - start
    
    start = time.time()
    radius_outlier_removal(large_points, radius=1.0, use_gpu=True)
    ror_gpu = time.time() - start
    
    print(f"\nDataset: {len(large_points):,} points")
    print(f"\n{'Operation':<30} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10}")
    print("-" * 80)
    print(f"{'Statistical Outlier Removal':<30} {sor_cpu:>10.3f}s  {sor_gpu:>10.3f}s  {sor_cpu/sor_gpu:>8.1f}x")
    print(f"{'Radius Outlier Removal':<30} {ror_cpu:>10.3f}s  {ror_gpu:>10.3f}s  {ror_cpu/ror_gpu:>8.1f}x")
    
    total_cpu = sor_cpu + ror_cpu
    total_gpu = sor_gpu + ror_gpu
    
    print("-" * 80)
    print(f"{'TOTAL PREPROCESSING':<30} {total_cpu:>10.3f}s  {total_gpu:>10.3f}s  {total_cpu/total_gpu:>8.1f}x")
    print("=" * 80)
    
    print(f"\n✅ Overall speedup: {total_cpu/total_gpu:.1f}x faster with GPU")
    print(f"✅ Time saved: {total_cpu - total_gpu:.1f}s ({(1 - total_gpu/total_cpu)*100:.1f}% reduction)")
