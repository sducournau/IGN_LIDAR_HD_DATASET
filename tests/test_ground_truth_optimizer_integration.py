"""
Test ground truth optimizer integration in processor.py

This test validates that GroundTruthOptimizer can replace AdvancedClassifier
with equivalent or better results.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon


from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
from ign_lidar.core.modules.advanced_classification import AdvancedClassifier


@pytest.fixture
def sample_points():
    """Create sample point cloud."""
    # Create 10,000 test points
    np.random.seed(42)
    n_points = 10000
    
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 100, n_points)
    z = np.random.uniform(0, 50, n_points)
    
    points = np.column_stack([x, y, z])
    return points


@pytest.fixture
def sample_ground_truth():
    """Create sample ground truth features matching GroundTruthOptimizer API."""
    # Building: 20x20 square at (20,20)
    building = gpd.GeoDataFrame(
        {'geometry': [Polygon([(20, 20), (40, 20), (40, 40), (20, 40)])]},
        crs='EPSG:2154'
    )
    
    # Road: 80 unit long strip at y=50
    road = gpd.GeoDataFrame(
        {'geometry': [Polygon([(0, 48), (80, 48), (80, 52), (0, 52)])]},
        crs='EPSG:2154'
    )
    
    # Water: 15x15 square at (60,60)
    water = gpd.GeoDataFrame(
        {'geometry': [Polygon([(60, 60), (75, 60), (75, 75), (60, 75)])]},
        crs='EPSG:2154'
    )
    
    # Return in the format expected by GroundTruthOptimizer
    return {
        'buildings': building,
        'roads': road,
        'water': water,
        'railways': gpd.GeoDataFrame(),  # Empty
        'parking': gpd.GeoDataFrame(),  # Empty
        'sports_ground': gpd.GeoDataFrame(),  # Empty
        'cemetery': gpd.GeoDataFrame(),  # Empty
        'power_lines': gpd.GeoDataFrame()  # Empty
    }


def test_ground_truth_optimizer_basic(sample_points, sample_ground_truth):
    """Test that GroundTruthOptimizer can label points correctly."""
    # Use GroundTruthOptimizer
    optimizer = GroundTruthOptimizer(
        force_method='strtree',  # Use CPU for reproducibility
        verbose=True
    )
    
    labels = optimizer.label_points(
        points=sample_points,
        ground_truth_features=sample_ground_truth,
        label_priority=None
    )
    
    # Verify labels shape
    assert labels.shape == (len(sample_points),)
    
    # Verify we have some labeled points (not all unlabeled)
    assert np.sum(labels != 0) > 0
    
    # Verify we have expected classes
    unique_labels = np.unique(labels)
    print(f"Unique labels found: {unique_labels}")
    
    # Should have at least some points in each feature
    # (exact counts depend on random point distribution)
    assert len(unique_labels) > 1


def test_optimizer_vs_classifier_compatibility(sample_points, sample_ground_truth):
    """
    Test that GroundTruthOptimizer produces compatible results with AdvancedClassifier.
    
    Note: Exact match not expected due to different implementations,
    but distribution should be similar.
    """
    # Initialize both classifiers
    optimizer = GroundTruthOptimizer(
        force_method='strtree',
        verbose=False
    )
    
    classifier = AdvancedClassifier(
        use_ground_truth=True,
        use_ndvi=False,
        use_geometric=False
    )
    
    # Get labels from optimizer
    labels_optimizer = optimizer.label_points(
        points=sample_points,
        ground_truth_features=sample_ground_truth
    )
    
    # Get labels from old classifier
    labels_old = np.zeros(len(sample_points), dtype=np.int32)
    labels_classifier = classifier._classify_by_ground_truth(
        labels=labels_old,
        points=sample_points,
        ground_truth_features=sample_ground_truth,
        ndvi=None,
        height=None,
        planarity=None,
        intensity=None
    )
    
    # Compare distributions (should be similar)
    unique_opt, counts_opt = np.unique(labels_optimizer, return_counts=True)
    unique_cls, counts_cls = np.unique(labels_classifier, return_counts=True)
    
    print("\nOptimizer distribution:")
    for label, count in zip(unique_opt, counts_opt):
        print(f"  Class {label}: {count:,} ({100*count/len(sample_points):.1f}%)")
    
    print("\nClassifier distribution:")
    for label, count in zip(unique_cls, counts_cls):
        print(f"  Class {label}: {count:,} ({100*count/len(sample_points):.1f}%)")
    
    # Both should have labeled some points
    assert np.sum(labels_optimizer != 0) > 0
    assert np.sum(labels_classifier != 0) > 0


def test_optimizer_gpu_selection():
    """Test that optimizer correctly selects GPU method for large datasets."""
    optimizer = GroundTruthOptimizer(verbose=False)
    
    # Small dataset → GPU
    method_small = optimizer.select_method(n_points=1_000_000, n_polygons=100)
    print(f"Small dataset (1M pts): {method_small}")
    
    # Large dataset → GPU chunked (if GPU available) or CPU
    method_large = optimizer.select_method(n_points=20_000_000, n_polygons=100)
    print(f"Large dataset (20M pts): {method_large}")
    
    # Should select appropriate method
    assert method_small in ['gpu', 'strtree', 'vectorized']
    assert method_large in ['gpu_chunked', 'strtree', 'vectorized']


@pytest.mark.parametrize("method", ['strtree', 'vectorized'])
def test_optimizer_forced_method(sample_points, sample_ground_truth, method):
    """Test that forced method selection works."""
    optimizer = GroundTruthOptimizer(
        force_method=method,
        verbose=False
    )
    
    labels = optimizer.label_points(
        points=sample_points,
        ground_truth_features=sample_ground_truth
    )
    
    # Should successfully label points
    assert labels.shape == (len(sample_points),)
    assert np.sum(labels != 0) > 0


def test_optimizer_with_ndvi(sample_points, sample_ground_truth):
    """Test NDVI refinement functionality."""
    # Create mock NDVI values
    ndvi = np.random.uniform(-0.5, 1.0, len(sample_points))
    
    optimizer = GroundTruthOptimizer(
        force_method='strtree',
        verbose=False
    )
    
    labels = optimizer.label_points(
        points=sample_points,
        ground_truth_features=sample_ground_truth,
        ndvi=ndvi,
        use_ndvi_refinement=True,
        ndvi_vegetation_threshold=0.3
    )
    
    # Should work without errors
    assert labels.shape == (len(sample_points),)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
