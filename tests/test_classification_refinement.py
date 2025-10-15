"""
Tests for classification refinement module
"""

import pytest
import numpy as np
from ign_lidar.core.modules.classification_refinement import (
    refine_classification,
    refine_vegetation_classification,
    refine_building_classification,
    refine_ground_classification,
    detect_vehicles,
    RefinementConfig
)


# LOD2 class constants
LOD2_WALL = 0
LOD2_GROUND = 9
LOD2_VEG_LOW = 10
LOD2_VEG_HIGH = 11
LOD2_WATER = 12
LOD2_VEHICLE = 13
LOD2_OTHER = 14


def test_refine_vegetation_with_ndvi():
    """Test vegetation refinement using NDVI values."""
    # Create test data
    n_points = 1000
    labels = np.full(n_points, LOD2_OTHER, dtype=np.uint8)
    
    # Create NDVI gradient: low to high
    ndvi = np.linspace(-0.2, 0.8, n_points)
    
    # Create height gradient: short to tall
    height = np.linspace(0.1, 5.0, n_points)
    
    # Refine
    refined, n_changed = refine_vegetation_classification(labels, ndvi, height)
    
    # Verify high vegetation detected (high NDVI + tall)
    high_veg_mask = (ndvi > 0.5) & (height > 1.5)
    assert np.all(refined[high_veg_mask] == LOD2_VEG_HIGH), "High vegetation not detected"
    
    # Verify low vegetation detected (moderate NDVI + short)
    low_veg_mask = (ndvi > 0.3) & (ndvi < 0.6) & (height < 2.0)
    assert np.sum(refined[low_veg_mask] == LOD2_VEG_LOW) > 0, "Low vegetation not detected"
    
    # Verify some changes were made
    assert n_changed > 0, "No refinements applied"
    print(f"✓ Vegetation refinement: {n_changed} points changed")


def test_refine_buildings_with_geometry():
    """Test building refinement using geometric features."""
    n_points = 500
    labels = np.full(n_points, LOD2_OTHER, dtype=np.uint8)
    
    # Create tall, vertical, planar points (building-like)
    height = np.random.uniform(3.0, 10.0, n_points)
    verticality = np.random.uniform(0.7, 0.95, n_points)
    planarity = np.random.uniform(0.5, 0.9, n_points)
    
    # Refine
    refined, n_changed = refine_building_classification(
        labels, height, planarity, verticality
    )
    
    # Verify buildings detected
    building_mask = (height > 2.5) & (verticality > 0.7) & (planarity > 0.5)
    assert np.sum(refined[building_mask] == LOD2_WALL) > 0, "Buildings not detected"
    assert n_changed > 0, "No refinements applied"
    print(f"✓ Building refinement: {n_changed} points changed")


def test_refine_buildings_with_ground_truth():
    """Test building refinement with ground truth override."""
    n_points = 500
    labels = np.full(n_points, LOD2_OTHER, dtype=np.uint8)
    
    # Create ground truth mask (first 250 points are buildings)
    gt_mask = np.zeros(n_points, dtype=bool)
    gt_mask[:250] = True
    
    # Simple geometric features
    height = np.random.uniform(1.0, 10.0, n_points)
    verticality = np.random.uniform(0.3, 0.9, n_points)
    planarity = np.random.uniform(0.3, 0.8, n_points)
    
    # Refine
    refined, n_changed = refine_building_classification(
        labels, height, planarity, verticality, ground_truth_mask=gt_mask
    )
    
    # Verify ground truth applied
    assert np.all(refined[gt_mask] == LOD2_WALL), "Ground truth not applied"
    assert n_changed >= 250, "Not all ground truth points updated"
    print(f"✓ Ground truth override: {n_changed} points changed")


def test_refine_ground():
    """Test ground refinement using planarity and height."""
    n_points = 300
    labels = np.full(n_points, LOD2_OTHER, dtype=np.uint8)
    
    # Create flat, low points (ground-like)
    height = np.random.uniform(0.0, 0.25, n_points)
    planarity = np.random.uniform(0.75, 0.95, n_points)
    
    # Refine
    refined, n_changed = refine_ground_classification(labels, height, planarity)
    
    # Verify ground detected
    ground_mask = (height < 0.3) & (planarity > 0.7)
    assert np.sum(refined[ground_mask] == LOD2_GROUND) > 0, "Ground not detected"
    assert n_changed > 0, "No refinements applied"
    print(f"✓ Ground refinement: {n_changed} points changed")


def test_detect_vehicles():
    """Test vehicle detection using height."""
    n_points = 400
    labels = np.full(n_points, LOD2_OTHER, dtype=np.uint8)
    
    # Create medium-height points (vehicle-like)
    height = np.random.uniform(0.8, 2.5, n_points)
    density = np.random.uniform(10, 50, n_points)
    
    # Detect vehicles
    refined, n_changed = detect_vehicles(labels, height, density)
    
    # Verify vehicles detected
    vehicle_mask = (height > 0.5) & (height < 3.0)
    assert np.sum(refined[vehicle_mask] == LOD2_VEHICLE) > 0, "Vehicles not detected"
    assert n_changed > 0, "No detections made"
    print(f"✓ Vehicle detection: {n_changed} points changed")


def test_full_refinement_pipeline():
    """Test complete refinement pipeline with multiple feature types."""
    n_points = 1000
    
    # Create mixed classification (some buildings, some vegetation, some other)
    labels = np.random.choice([LOD2_WALL, LOD2_VEG_LOW, LOD2_OTHER], n_points)
    
    # Create realistic features
    features = {
        'ndvi': np.random.uniform(-0.2, 0.8, n_points),
        'height': np.random.uniform(0.0, 15.0, n_points),
        'planarity': np.random.uniform(0.2, 0.95, n_points),
        'verticality': np.random.uniform(0.1, 0.95, n_points),
        'density': np.random.uniform(5, 100, n_points),
    }
    
    # Add some ground truth
    gt_building_mask = np.zeros(n_points, dtype=bool)
    gt_building_mask[100:200] = True  # 100 building points
    
    ground_truth_data = {
        'building_mask': gt_building_mask
    }
    
    # Apply refinement
    refined, stats = refine_classification(
        labels=labels,
        features=features,
        ground_truth_data=ground_truth_data,
        lod_level='LOD2'
    )
    
    # Verify refinements were made
    assert len(stats) > 0, "No refinement stats returned"
    total_changes = sum(stats.values())
    assert total_changes > 0, "No points refined"
    
    # Verify ground truth buildings applied
    assert np.all(refined[gt_building_mask] == LOD2_WALL), "Ground truth not applied"
    
    print(f"✓ Full pipeline: {total_changes} total changes")
    print(f"  Stats: {stats}")


def test_refinement_config():
    """Test custom refinement configuration."""
    config = RefinementConfig()
    
    # Verify default values
    assert config.NDVI_VEGETATION_MIN == 0.3
    assert config.LOW_VEG_HEIGHT_MAX == 2.0
    assert config.BUILDING_HEIGHT_MIN == 2.5
    
    # Test with custom config
    config.NDVI_VEGETATION_MIN = 0.4
    config.BUILDING_HEIGHT_MIN = 3.0
    
    n_points = 500
    labels = np.full(n_points, LOD2_OTHER, dtype=np.uint8)
    ndvi = np.random.uniform(0.35, 0.45, n_points)  # Between old and new threshold
    height = np.random.uniform(0.5, 3.0, n_points)
    
    # With default config, some vegetation should be detected
    refined_default, n_default = refine_vegetation_classification(
        labels.copy(), ndvi, height, RefinementConfig()
    )
    
    # With stricter config, less vegetation detected
    refined_strict, n_strict = refine_vegetation_classification(
        labels.copy(), ndvi, height, config
    )
    
    assert n_strict <= n_default, "Stricter config should refine fewer points"
    print(f"✓ Config test: default={n_default}, strict={n_strict}")


def test_no_features_available():
    """Test refinement with no features available."""
    n_points = 100
    labels = np.full(n_points, LOD2_OTHER, dtype=np.uint8)
    
    # Empty features
    features = {}
    
    # Should return unchanged labels
    refined, stats = refine_classification(
        labels=labels,
        features=features,
        lod_level='LOD2'
    )
    
    assert np.array_equal(refined, labels), "Labels changed with no features"
    assert len(stats) == 0 or sum(stats.values()) == 0, "Stats reported changes"
    print("✓ No-features test passed")


def test_lod3_not_implemented():
    """Test that LOD3 refinement returns original labels (not yet implemented)."""
    n_points = 100
    labels = np.full(n_points, 0, dtype=np.uint8)
    features = {'ndvi': np.random.uniform(0, 1, n_points)}
    
    refined, stats = refine_classification(
        labels=labels,
        features=features,
        lod_level='LOD3'
    )
    
    assert np.array_equal(refined, labels), "LOD3 changed labels (not implemented)"
    print("✓ LOD3 not-implemented test passed")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Testing Classification Refinement Module")
    print("="*70 + "\n")
    
    test_refine_vegetation_with_ndvi()
    test_refine_buildings_with_geometry()
    test_refine_buildings_with_ground_truth()
    test_refine_ground()
    test_detect_vehicles()
    test_full_refinement_pipeline()
    test_refinement_config()
    test_no_features_available()
    test_lod3_not_implemented()
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70 + "\n")
