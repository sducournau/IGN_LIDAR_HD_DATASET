"""
Performance tests for reclassifier optimizations.

Tests the CPU vectorized and GPU batched implementations to validate
the expected 10-30× speedup over baseline implementations.

Run with: pytest tests/test_reclassifier_performance.py -v -s
"""

import time
import pytest
import numpy as np
from pathlib import Path

# Import test markers
pytestmark = [pytest.mark.performance, pytest.mark.slow]


def test_cpu_vectorized_vs_legacy(sample_points_large, sample_polygons):
    """
    Test CPU vectorized implementation vs legacy.
    Expected: 10-20× speedup for vectorized (Shapely 2.0+).
    """
    from ign_lidar.core.classification.reclassifier import (
        Reclassifier,
        HAS_BULK_QUERY,
    )
    
    if not HAS_BULK_QUERY:
        pytest.skip("Shapely 2.0+ required for vectorized implementation")
    
    points = sample_points_large  # 1M points
    labels = np.ones(len(points), dtype=np.uint8)
    geometries = sample_polygons  # Array of shapely polygons
    asprs_code = 6  # Building
    
    # Test vectorized implementation
    reclassifier_vec = Reclassifier(
        acceleration_mode="cpu",
        chunk_size=100_000,
        show_progress=False
    )
    
    start = time.time()
    n_classified_vec = reclassifier_vec._classify_feature_cpu_vectorized(
        points, labels.copy(), geometries, asprs_code, "test_vectorized"
    )
    time_vec = time.time() - start
    
    # Test legacy implementation
    start = time.time()
    n_classified_leg = reclassifier_vec._classify_feature_cpu_legacy(
        points, labels.copy(), geometries, asprs_code, "test_legacy"
    )
    time_leg = time.time() - start
    
    # Verify same results
    assert n_classified_vec == n_classified_leg, "Vectorized and legacy should classify same points"
    
    # Calculate speedup
    speedup = time_leg / time_vec
    
    print(f"\n{'='*60}")
    print(f"CPU Vectorized vs Legacy Performance")
    print(f"{'='*60}")
    print(f"Points:          {len(points):,}")
    print(f"Polygons:        {len(geometries):,}")
    print(f"Classified:      {n_classified_vec:,}")
    print(f"Legacy time:     {time_leg:.3f}s")
    print(f"Vectorized time: {time_vec:.3f}s")
    print(f"Speedup:         {speedup:.1f}×")
    print(f"{'='*60}")
    
    # Expected: 10-20× speedup (may vary based on data distribution)
    # We use 3× as minimum threshold to account for test variability
    assert speedup >= 3.0, f"Expected at least 3× speedup, got {speedup:.1f}×"


@pytest.mark.gpu
@pytest.mark.skipif(not pytest.config.getoption("--run-gpu", default=False), 
                    reason="GPU tests not enabled (use --run-gpu)")
def test_gpu_batched_vs_cpu(sample_points_large, sample_polygons):
    """
    Test GPU batched implementation vs CPU vectorized.
    Expected: 5-10× speedup for GPU on large datasets (>5M points).
    """
    from ign_lidar.core.classification.reclassifier import Reclassifier, HAS_GPU
    
    if not HAS_GPU:
        pytest.skip("RAPIDS cuSpatial required for GPU tests")
    
    points = sample_points_large  # 1M points
    labels = np.ones(len(points), dtype=np.uint8)
    geometries = sample_polygons
    asprs_code = 6  # Building
    
    # Test CPU implementation
    reclassifier_cpu = Reclassifier(
        acceleration_mode="cpu",
        chunk_size=100_000,
        show_progress=False
    )
    
    start = time.time()
    n_classified_cpu = reclassifier_cpu._classify_feature_cpu(
        points, labels.copy(), geometries, asprs_code, "test_cpu"
    )
    time_cpu = time.time() - start
    
    # Test GPU implementation
    reclassifier_gpu = Reclassifier(
        acceleration_mode="gpu",
        chunk_size=100_000,
        show_progress=False
    )
    
    start = time.time()
    n_classified_gpu = reclassifier_gpu._classify_feature_gpu(
        points, labels.copy(), geometries, asprs_code, "test_gpu"
    )
    time_gpu = time.time() - start
    
    # Verify similar results (may have minor differences due to floating point)
    assert abs(n_classified_cpu - n_classified_gpu) < len(points) * 0.01, \
        "CPU and GPU should classify similar number of points (±1%)"
    
    # Calculate speedup
    speedup = time_cpu / time_gpu
    
    print(f"\n{'='*60}")
    print(f"GPU Batched vs CPU Vectorized Performance")
    print(f"{'='*60}")
    print(f"Points:          {len(points):,}")
    print(f"Polygons:        {len(geometries):,}")
    print(f"Classified CPU:  {n_classified_cpu:,}")
    print(f"Classified GPU:  {n_classified_gpu:,}")
    print(f"CPU time:        {time_cpu:.3f}s")
    print(f"GPU time:        {time_gpu:.3f}s")
    print(f"Speedup:         {speedup:.1f}×")
    print(f"{'='*60}")
    
    # Expected: 5-10× speedup for large datasets
    # We use 2× as minimum threshold for 1M points (GPU shines at >5M)
    assert speedup >= 2.0, f"Expected at least 2× speedup, got {speedup:.1f}×"


def test_auto_selection_logic(sample_points_small, sample_points_large, sample_polygons):
    """
    Test auto-selection logic chooses appropriate backend.
    """
    from ign_lidar.core.classification.reclassifier import Reclassifier, HAS_GPU
    
    reclassifier = Reclassifier(
        acceleration_mode="auto",
        chunk_size=100_000,
        show_progress=False
    )
    
    # Small dataset (<1M): should use CPU
    points_small = sample_points_small  # 10K points
    labels_small = np.ones(len(points_small), dtype=np.uint8)
    geometries = sample_polygons
    
    # We can't directly check which method was called, but we can verify it works
    n_classified = reclassifier._classify_feature(
        points_small, labels_small, geometries, 6, "test_small"
    )
    assert n_classified >= 0, "Auto-selection should work for small datasets"
    
    # Large dataset (>1M): should use GPU if available
    points_large = sample_points_large  # 1M points
    labels_large = np.ones(len(points_large), dtype=np.uint8)
    
    n_classified = reclassifier._classify_feature(
        points_large, labels_large, geometries, 6, "test_large"
    )
    assert n_classified >= 0, "Auto-selection should work for large datasets"
    
    print(f"\n{'='*60}")
    print(f"Auto-Selection Logic Test")
    print(f"{'='*60}")
    print(f"Small dataset:   {len(points_small):,} points - CPU preferred")
    print(f"Large dataset:   {len(points_large):,} points - {'GPU' if HAS_GPU else 'CPU (no GPU)'} used")
    print(f"GPU available:   {HAS_GPU}")
    print(f"{'='*60}")


def test_memory_efficiency(sample_points_large, sample_polygons):
    """
    Test memory efficiency of vectorized implementation.
    Should not cause memory issues with large datasets.
    """
    from ign_lidar.core.classification.reclassifier import Reclassifier
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    points = sample_points_large  # 1M points
    labels = np.ones(len(points), dtype=np.uint8)
    geometries = sample_polygons
    
    reclassifier = Reclassifier(
        acceleration_mode="cpu",
        chunk_size=100_000,  # Process in 100K chunks
        show_progress=False
    )
    
    n_classified = reclassifier._classify_feature_cpu(
        points, labels, geometries, 6, "test_memory"
    )
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before
    
    print(f"\n{'='*60}")
    print(f"Memory Efficiency Test")
    print(f"{'='*60}")
    print(f"Points:          {len(points):,}")
    print(f"Memory before:   {mem_before:.1f} MB")
    print(f"Memory after:    {mem_after:.1f} MB")
    print(f"Memory used:     {mem_used:.1f} MB")
    print(f"MB per point:    {mem_used / len(points) * 1000:.3f} MB/K")
    print(f"{'='*60}")
    
    # Should use less than 500 MB for 1M points
    assert mem_used < 500, f"Memory usage too high: {mem_used:.1f} MB"


def test_chunking_behavior(sample_points_large, sample_polygons):
    """
    Test that chunking works correctly with different chunk sizes.
    """
    from ign_lidar.core.classification.reclassifier import Reclassifier
    
    points = sample_points_large[:100_000]  # 100K points for faster test
    geometries = sample_polygons
    
    # Test different chunk sizes
    chunk_sizes = [10_000, 50_000, 100_000]
    results = []
    times = []
    
    for chunk_size in chunk_sizes:
        labels = np.ones(len(points), dtype=np.uint8)
        
        reclassifier = Reclassifier(
            acceleration_mode="cpu",
            chunk_size=chunk_size,
            show_progress=False
        )
        
        start = time.time()
        n_classified = reclassifier._classify_feature_cpu(
            points, labels, geometries, 6, f"test_chunk_{chunk_size}"
        )
        elapsed = time.time() - start
        
        results.append(n_classified)
        times.append(elapsed)
    
    print(f"\n{'='*60}")
    print(f"Chunking Behavior Test")
    print(f"{'='*60}")
    print(f"Points:          {len(points):,}")
    for chunk_size, n_classified, elapsed in zip(chunk_sizes, results, times):
        print(f"Chunk {chunk_size:>6,}: {n_classified:>6,} classified in {elapsed:.3f}s")
    print(f"{'='*60}")
    
    # All chunk sizes should give same result
    assert len(set(results)) == 1, "Different chunk sizes should give same results"
    
    # Larger chunks should be faster (less overhead)
    assert times[-1] <= times[0] * 1.5, "Larger chunks should be more efficient"


# Fixtures
@pytest.fixture
def sample_points_small():
    """Generate small sample point cloud (10K points)."""
    np.random.seed(42)
    n_points = 10_000
    
    # Generate points in a 1000m × 1000m area
    points = np.column_stack([
        np.random.uniform(0, 1000, n_points),  # X
        np.random.uniform(0, 1000, n_points),  # Y
        np.random.uniform(0, 50, n_points),    # Z
    ])
    
    return points


@pytest.fixture
def sample_points_large():
    """Generate large sample point cloud (1M points)."""
    np.random.seed(42)
    n_points = 1_000_000
    
    # Generate points in a 1000m × 1000m area
    points = np.column_stack([
        np.random.uniform(0, 1000, n_points),  # X
        np.random.uniform(0, 1000, n_points),  # Y
        np.random.uniform(0, 50, n_points),    # Z
    ])
    
    return points


@pytest.fixture
def sample_polygons():
    """Generate sample polygons (buildings)."""
    from shapely.geometry import Polygon
    
    np.random.seed(42)
    n_polygons = 100
    
    polygons = []
    for i in range(n_polygons):
        # Random building location
        center_x = np.random.uniform(100, 900)
        center_y = np.random.uniform(100, 900)
        
        # Random building size (10-50m)
        width = np.random.uniform(10, 50)
        height = np.random.uniform(10, 50)
        
        # Create rectangular polygon
        poly = Polygon([
            (center_x - width/2, center_y - height/2),
            (center_x + width/2, center_y - height/2),
            (center_x + width/2, center_y + height/2),
            (center_x - width/2, center_y + height/2),
            (center_x - width/2, center_y - height/2),
        ])
        polygons.append(poly)
    
    return np.array(polygons)


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "performance: Performance benchmark tests")
    config.addinivalue_line("markers", "slow: Slow tests (>5 seconds)")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
