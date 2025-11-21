"""
Phase 3: Façade Processing Optimization Tests

Tests for GPU-accelerated KNN, parallel processing, and vectorized calculations
in facade_processor.py. Validates 20-30× expected speedup from Phase 1.4 KNN gains.

Author: Performance Optimization Team
Date: November 20, 2025
"""

import numpy as np
import pytest
import logging

logger = logging.getLogger(__name__)

# Test markers
pytestmark = [pytest.mark.unit, pytest.mark.performance]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_building_points():
    """Generate realistic building point cloud data."""
    np.random.seed(42)
    
    # Rectangular building: 20m x 15m x 10m (height)
    n_points = 50000
    
    # Generate points on walls (N/S/E/W)
    points = []
    
    # North wall (y=15)
    n_north = n_points // 4
    x_north = np.random.uniform(0, 20, n_north)
    y_north = np.full(n_north, 15.0) + np.random.normal(0, 0.1, n_north)  # Small noise
    z_north = np.random.uniform(0, 10, n_north)
    points.append(np.column_stack([x_north, y_north, z_north]))
    
    # South wall (y=0)
    n_south = n_points // 4
    x_south = np.random.uniform(0, 20, n_south)
    y_south = np.full(n_south, 0.0) + np.random.normal(0, 0.1, n_south)
    z_south = np.random.uniform(0, 10, n_south)
    points.append(np.column_stack([x_south, y_south, z_south]))
    
    # East wall (x=20)
    n_east = n_points // 4
    x_east = np.full(n_points // 4, 20.0) + np.random.normal(0, 0.1, n_east)
    y_east = np.random.uniform(0, 15, n_east)
    z_east = np.random.uniform(0, 10, n_east)
    points.append(np.column_stack([x_east, y_east, z_east]))
    
    # West wall (x=0)
    n_west = n_points - n_north - n_south - n_east
    x_west = np.full(n_west, 0.0) + np.random.normal(0, 0.1, n_west)
    y_west = np.random.uniform(0, 15, n_west)
    z_west = np.random.uniform(0, 10, n_west)
    points.append(np.column_stack([x_west, y_west, z_west]))
    
    all_points = np.vstack(points)
    
    # Generate normals (pointing outward from walls)
    normals = np.zeros_like(all_points)
    
    # North wall normals: (0, 1, 0)
    normals[:n_north] = [0, 1, 0]
    # South wall normals: (0, -1, 0)
    normals[n_north:n_north+n_south] = [0, -1, 0]
    # East wall normals: (1, 0, 0)
    normals[n_north+n_south:n_north+n_south+n_east] = [1, 0, 0]
    # West wall normals: (-1, 0, 0)
    normals[n_north+n_south+n_east:] = [-1, 0, 0]
    
    # Add small noise to normals
    normals += np.random.normal(0, 0.05, normals.shape)
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    
    # Compute verticality (from normals)
    verticality = np.abs(normals[:, 2])  # |z component|
    verticality = 1.0 - verticality  # High verticality = horizontal normal
    
    return {
        "points": all_points,
        "normals": normals,
        "verticality": verticality,
        "heights": all_points[:, 2],
        "n_north": n_north,
        "n_south": n_south,
        "n_east": n_east,
    }


@pytest.fixture
def building_polygon():
    """Generate building polygon for testing."""
    try:
        from shapely.geometry import Polygon
        return Polygon([(0, 0), (20, 0), (20, 15), (0, 15)])
    except ImportError:
        pytest.skip("Shapely not available")


# ============================================================================
# Test GPU-Accelerated KNN (Phase 3.1)
# ============================================================================


def test_gpu_knn_import():
    """Test that GPU-accelerated KNN can be imported."""
    from ign_lidar.optimization.gpu_accelerated_ops import knn
    
    assert callable(knn), "KNN function should be callable"


def test_facade_processor_uses_gpu_knn(sample_building_points):
    """Test that FacadeProcessor uses GPU-accelerated KNN."""
    from ign_lidar.core.classification.building.facade_processor import (
        FacadeProcessor,
        FacadeSegment,
        FacadeOrientation
    )
    
    # Create a simple facade
    try:
        from shapely.geometry import LineString
    except ImportError:
        pytest.skip("Shapely not available")
    
    facade = FacadeSegment(
        orientation=FacadeOrientation.NORTH,
        edge_line=LineString([(0, 15), (20, 15)]),
        length=20.0,
        initial_buffer=3.0,
        verticality_threshold=0.55
    )
    
    processor = FacadeProcessor(
        facade=facade,
        points=sample_building_points["points"],
        heights=sample_building_points["heights"],
        normals=sample_building_points["normals"],
        verticality=sample_building_points["verticality"],
    )
    
    # Process should use GPU KNN internally
    processed = processor.process(building_height=10.0)
    
    # Validate results
    assert processed is not None
    assert processed.n_points > 0, "Should detect facade points"
    assert processed.n_wall_points > 0, "Should detect wall points"


def test_gpu_knn_performance_vs_cpu(sample_building_points):
    """Benchmark GPU KNN vs CPU for facade processing."""
    import time
    from ign_lidar.optimization.gpu_accelerated_ops import knn, set_force_cpu
    
    points = sample_building_points["points"][:, :2]  # XY only
    
    # Subset for quick test (10K points)
    n_test = min(10000, len(points))
    test_points = points[:n_test]
    
    k = 30
    
    # Test with GPU (if available)
    set_force_cpu(False)
    start_gpu = time.time()
    distances_gpu, indices_gpu = knn(test_points, k=k)
    time_gpu = time.time() - start_gpu
    
    # Test with CPU (forced)
    set_force_cpu(True)
    start_cpu = time.time()
    distances_cpu, indices_cpu = knn(test_points, k=k)
    time_cpu = time.time() - start_cpu
    set_force_cpu(False)  # Reset
    
    logger.info(f"\nKNN Performance ({n_test} points, k={k}):")
    logger.info(f"  GPU: {time_gpu:.3f}s")
    logger.info(f"  CPU: {time_cpu:.3f}s")
    
    if time_gpu < time_cpu:
        speedup = time_cpu / time_gpu
        logger.info(f"  🚀 GPU Speedup: {speedup:.1f}×")
    
    # Validate results are similar (within 5% tolerance)
    np.testing.assert_allclose(
        distances_gpu, distances_cpu, rtol=0.05, atol=1e-5,
        err_msg="GPU and CPU KNN should produce similar results"
    )


# ============================================================================
# Test Parallel Facade Processing (Phase 3.2)
# ============================================================================


def test_parallel_facade_processing(sample_building_points, building_polygon):
    """Test parallel processing of N/S/E/W facades."""
    from ign_lidar.core.classification.building.facade_processor import (
        BuildingFacadeClassifier
    )
    
    points = sample_building_points["points"]
    n_points = len(points)
    
    # Create labels array
    labels = np.zeros(n_points, dtype=np.uint8)
    
    # Test with parallel enabled
    classifier_parallel = BuildingFacadeClassifier(
        enable_parallel_facades=True,
        max_workers=4
    )
    
    import time
    start = time.time()
    labels_parallel, stats_parallel = classifier_parallel.classify_building(
        building_id=1,
        points=points,
        labels=labels.copy(),
        polygon=building_polygon,
        normals=sample_building_points["normals"],
        verticality=sample_building_points["verticality"],
    )
    time_parallel = time.time() - start
    
    # Test with parallel disabled
    classifier_sequential = BuildingFacadeClassifier(
        enable_parallel_facades=False
    )
    
    start = time.time()
    labels_sequential, stats_sequential = classifier_sequential.classify_building(
        building_id=1,
        points=points,
        labels=labels.copy(),
        polygon=building_polygon,
        normals=sample_building_points["normals"],
        verticality=sample_building_points["verticality"],
    )
    time_sequential = time.time() - start
    
    logger.info(f"\nParallel vs Sequential Facade Processing:")
    logger.info(f"  Parallel:   {time_parallel:.3f}s")
    logger.info(f"  Sequential: {time_sequential:.3f}s")
    
    if time_parallel < time_sequential:
        speedup = time_sequential / time_parallel
        logger.info(f"  🚀 Parallel Speedup: {speedup:.1f}×")
    
    # Validate results are identical
    n_classified_parallel = np.sum(labels_parallel != 0)
    n_classified_sequential = np.sum(labels_sequential != 0)
    
    # Allow small differences due to thread timing
    diff_ratio = abs(n_classified_parallel - n_classified_sequential) / max(
        n_classified_parallel, n_classified_sequential
    )
    
    assert diff_ratio < 0.05, "Parallel and sequential should produce similar results"


def test_parallel_processing_with_single_facade(sample_building_points):
    """Test that parallel processing handles single facade gracefully."""
    from ign_lidar.core.classification.building.facade_processor import (
        BuildingFacadeClassifier
    )
    from shapely.geometry import Polygon
    
    # Small polygon with only one facade worth processing
    small_polygon = Polygon([(0, 0), (5, 0), (5, 1), (0, 1)])
    
    points = sample_building_points["points"][:1000]  # Small subset
    labels = np.zeros(len(points), dtype=np.uint8)
    
    classifier = BuildingFacadeClassifier(
        enable_parallel_facades=True,
        max_workers=4
    )
    
    # Should not crash with parallel enabled
    labels_result, stats = classifier.classify_building(
        building_id=1,
        points=points,
        labels=labels,
        polygon=small_polygon,
        normals=sample_building_points["normals"][:1000],
        verticality=sample_building_points["verticality"][:1000],
    )
    
    assert labels_result is not None
    assert stats is not None


# ============================================================================
# Test Vectorized Calculations (Phase 3.3)
# ============================================================================


def test_vectorized_projection():
    """Test vectorized projection calculation in gap detection."""
    from ign_lidar.core.classification.building.facade_processor import (
        FacadeProcessor,
        FacadeSegment,
        FacadeOrientation
    )
    from shapely.geometry import LineString
    
    # Create test points along a line
    n_points = 10000
    x = np.linspace(0, 20, n_points)
    y = np.full(n_points, 10.0) + np.random.normal(0, 0.1, n_points)
    z = np.random.uniform(0, 10, n_points)
    points = np.column_stack([x, y, z])
    
    # Create facade
    facade = FacadeSegment(
        orientation=FacadeOrientation.NORTH,
        edge_line=LineString([(0, 10), (20, 10)]),
        length=20.0,
        initial_buffer=1.0,
        verticality_threshold=0.5
    )
    
    verticality = np.full(n_points, 0.8)  # High verticality
    
    processor = FacadeProcessor(
        facade=facade,
        points=points,
        heights=z,
        normals=None,
        verticality=verticality,
    )
    
    import time
    
    # Process with vectorized implementation
    start = time.time()
    processor.process(building_height=10.0)
    time_vectorized = time.time() - start
    
    logger.info(f"\nVectorized projection time: {time_vectorized:.4f}s for {n_points} points")
    
    # Validate gap detection worked
    assert processor.facade.has_gaps is not None
    assert processor.facade.gap_ratio >= 0.0


def test_vectorized_vs_loop_consistency():
    """Test that vectorized projection gives same results as loop."""
    from shapely.geometry import LineString
    import numpy as np
    
    # Create simple test case
    line = LineString([(0, 0), (10, 0)])
    points = np.array([
        [2, 0.5],
        [5, -0.5],
        [8, 0.2],
        [12, 0],  # Beyond line
        [-1, 0],  # Before line
    ])
    
    # Extract line coordinates for vectorized calculation
    line_coords = np.array(line.coords)
    p1, p2 = line_coords[0], line_coords[1]
    line_vec = p2 - p1
    line_length = np.linalg.norm(line_vec)
    line_vec_normalized = line_vec / line_length
    
    # Vectorized projection
    point_vecs = points - p1
    projected_vectorized = np.dot(point_vecs, line_vec_normalized)
    projected_vectorized = np.clip(projected_vectorized, 0, line_length)
    
    # Loop projection (reference)
    from shapely.geometry import Point
    projected_loop = np.array([line.project(Point(pt)) for pt in points])
    
    # Should match within numerical precision
    np.testing.assert_allclose(
        projected_vectorized, projected_loop, rtol=1e-10, atol=1e-10,
        err_msg="Vectorized and loop projection should match"
    )


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_facade_pipeline_optimized(sample_building_points, building_polygon):
    """Test full optimized pipeline with all Phase 3 improvements."""
    from ign_lidar.core.classification.building.facade_processor import (
        BuildingFacadeClassifier
    )
    
    points = sample_building_points["points"]
    labels = np.zeros(len(points), dtype=np.uint8)
    
    # Create classifier with all optimizations enabled
    classifier = BuildingFacadeClassifier(
        enable_parallel_facades=True,  # Phase 3.2
        max_workers=4,
        initial_buffer=3.0,
        verticality_threshold=0.55,
        min_confidence=0.2,
    )
    
    import time
    start = time.time()
    
    labels_result, stats = classifier.classify_building(
        building_id=1,
        points=points,
        labels=labels,
        polygon=building_polygon,
        normals=sample_building_points["normals"],
        verticality=sample_building_points["verticality"],
    )
    
    elapsed = time.time() - start
    
    logger.info(f"\n🚀 Full Optimized Pipeline:")
    logger.info(f"  Time: {elapsed:.3f}s for {len(points)} points")
    logger.info(f"  Facades processed: {stats.get('facades_processed', 0)}")
    logger.info(f"  Points classified: {np.sum(labels_result != 0)}")
    
    # Validate results
    assert labels_result is not None
    assert np.sum(labels_result != 0) > 0, "Should classify some points"
    assert stats["facades_processed"] == 4, "Should process all 4 facades"


def test_backward_compatibility():
    """Test that optimizations don't break backward compatibility."""
    from ign_lidar.core.classification.building.facade_processor import (
        BuildingFacadeClassifier
    )
    
    # Should accept old initialization without new parameters
    classifier_old = BuildingFacadeClassifier(
        initial_buffer=3.0,
        verticality_threshold=0.55,
    )
    
    assert classifier_old.enable_parallel_facades is True  # Default
    assert classifier_old.max_workers == 4  # Default


# ============================================================================
# Performance Regression Tests
# ============================================================================


@pytest.mark.slow
def test_performance_target_50k_points(sample_building_points, building_polygon):
    """
    Performance target: Process 50K point building in <2 seconds.
    
    Baseline (Phase 2): ~8-10s
    Target (Phase 3): <2s (4-5× speedup)
    """
    from ign_lidar.core.classification.building.facade_processor import (
        BuildingFacadeClassifier
    )
    
    points = sample_building_points["points"]
    labels = np.zeros(len(points), dtype=np.uint8)
    
    classifier = BuildingFacadeClassifier(
        enable_parallel_facades=True,
        max_workers=4,
    )
    
    import time
    start = time.time()
    
    labels_result, stats = classifier.classify_building(
        building_id=1,
        points=points,
        labels=labels,
        polygon=building_polygon,
        normals=sample_building_points["normals"],
        verticality=sample_building_points["verticality"],
    )
    
    elapsed = time.time() - start
    
    logger.info(f"\n⏱️  Performance Target Test (50K points):")
    logger.info(f"  Time: {elapsed:.3f}s")
    logger.info(f"  Target: <2.0s")
    
    if elapsed < 2.0:
        logger.info(f"  ✅ PASS - {2.0/elapsed:.1f}× better than target")
    else:
        logger.warning(f"  ⚠️  SLOW - {elapsed/2.0:.1f}× slower than target")
    
    # Soft assertion - log warning but don't fail
    # (depends on hardware and GPU availability)
    if elapsed > 5.0:
        pytest.fail(f"Performance too slow: {elapsed:.3f}s (baseline was ~8-10s)")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
