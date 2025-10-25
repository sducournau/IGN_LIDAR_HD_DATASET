"""
Test Plane Feature Extraction

This test validates the PlaneFeatureExtractor functionality.
"""

import pytest
import numpy as np
from ign_lidar.core.classification.plane_detection import (
    PlaneDetector,
    PlaneFeatureExtractor,
    PlaneType
)


@pytest.fixture
def synthetic_building():
    """Create synthetic building with roof and walls."""
    # Create a simple box-shaped building
    # Ground level at z=0, roof at z=10
    
    # Walls (vertical points)
    n_wall_points = 1000
    wall_points = []
    
    # North wall (y=10)
    north_wall = np.column_stack([
        np.random.uniform(0, 10, n_wall_points // 4),
        np.full(n_wall_points // 4, 10.0),
        np.random.uniform(0, 10, n_wall_points // 4),
    ])
    wall_points.append(north_wall)
    
    # South wall (y=0)
    south_wall = np.column_stack([
        np.random.uniform(0, 10, n_wall_points // 4),
        np.full(n_wall_points // 4, 0.0),
        np.random.uniform(0, 10, n_wall_points // 4),
    ])
    wall_points.append(south_wall)
    
    # East wall (x=10)
    east_wall = np.column_stack([
        np.full(n_wall_points // 4, 10.0),
        np.random.uniform(0, 10, n_wall_points // 4),
        np.random.uniform(0, 10, n_wall_points // 4),
    ])
    wall_points.append(east_wall)
    
    # West wall (x=0)
    west_wall = np.column_stack([
        np.full(n_wall_points // 4, 0.0),
        np.random.uniform(0, 10, n_wall_points // 4),
        np.random.uniform(0, 10, n_wall_points // 4),
    ])
    wall_points.append(west_wall)
    
    walls = np.vstack(wall_points)
    
    # Roof (horizontal points at z=10)
    n_roof_points = 500
    roof = np.column_stack([
        np.random.uniform(0, 10, n_roof_points),
        np.random.uniform(0, 10, n_roof_points),
        np.full(n_roof_points, 10.0) + np.random.normal(0, 0.1, n_roof_points),
    ])
    
    # Combine
    points = np.vstack([walls, roof])
    
    # Generate normals
    normals = np.zeros_like(points)
    
    # Wall normals (pointing outward)
    normals[:n_wall_points // 4, :] = [0, 1, 0]  # North
    normals[n_wall_points // 4:n_wall_points // 2, :] = [0, -1, 0]  # South
    normals[n_wall_points // 2:3 * n_wall_points // 4, :] = [1, 0, 0]  # East
    normals[3 * n_wall_points // 4:n_wall_points, :] = [-1, 0, 0]  # West
    
    # Roof normals (pointing up)
    normals[n_wall_points:, :] = [0, 0, 1]
    
    # Planarity (high for both walls and roof)
    planarity = np.full(len(points), 0.9)
    
    # Height above ground
    height = points[:, 2].copy()
    
    return points, normals, planarity, height


def test_plane_feature_extractor_basic(synthetic_building):
    """Test basic plane feature extraction."""
    points, normals, planarity, height = synthetic_building
    
    # Create detector and extractor
    detector = PlaneDetector(
        horizontal_angle_max=10.0,
        vertical_angle_min=75.0,
        min_points_per_plane=50
    )
    extractor = PlaneFeatureExtractor(detector)
    
    # Extract features
    features = extractor.detect_and_assign_planes(
        points, normals, planarity, height
    )
    
    # Validate feature keys
    expected_keys = {
        'plane_id',
        'plane_type',
        'distance_to_plane',
        'plane_area',
        'plane_orientation',
        'plane_planarity',
        'position_on_plane_u',
        'position_on_plane_v',
    }
    assert set(features.keys()) == expected_keys
    
    # Validate shapes
    n_points = len(points)
    for key, array in features.items():
        assert len(array) == n_points, f"Feature {key} has wrong length"
    
    # Check that some planes were detected
    n_assigned = (features['plane_id'] >= 0).sum()
    assert n_assigned > 0, "No points assigned to planes"
    
    # Check plane types
    plane_types = np.unique(features['plane_type'][features['plane_id'] >= 0])
    assert len(plane_types) > 0, "No plane types detected"
    assert 0 in plane_types or 1 in plane_types, "Should detect horizontal or vertical planes"


def test_plane_detection_statistics(synthetic_building):
    """Test plane statistics computation."""
    points, normals, planarity, height = synthetic_building
    
    detector = PlaneDetector()
    extractor = PlaneFeatureExtractor(detector)
    
    # Extract features
    features = extractor.detect_and_assign_planes(
        points, normals, planarity, height
    )
    
    # Get statistics
    stats = extractor.get_plane_statistics()
    
    # Validate statistics
    assert stats['n_planes'] > 0, "Should detect at least one plane"
    assert stats['total_area'] > 0, "Total area should be positive"
    assert 0 <= stats['avg_planarity'] <= 1, "Planarity should be in [0,1]"
    
    # Should detect both horizontal and vertical planes
    assert stats['n_horizontal'] > 0 or stats['n_vertical'] > 0, \
        "Should detect horizontal or vertical planes"


def test_plane_feature_values(synthetic_building):
    """Test that plane feature values are in expected ranges."""
    points, normals, planarity, height = synthetic_building
    
    detector = PlaneDetector()
    extractor = PlaneFeatureExtractor(detector)
    
    features = extractor.detect_and_assign_planes(
        points, normals, planarity, height
    )
    
    # Filter to assigned points
    assigned = features['plane_id'] >= 0
    
    if not np.any(assigned):
        pytest.skip("No planes detected in synthetic data")
    
    # Plane type should be 0, 1, or 2
    assert np.all(np.isin(features['plane_type'][assigned], [0, 1, 2])), \
        "Plane type should be 0 (horizontal), 1 (vertical), or 2 (inclined)"
    
    # Distance should be non-negative and reasonable
    assert np.all(features['distance_to_plane'][assigned] >= 0), \
        "Distance to plane should be non-negative"
    assert np.all(features['distance_to_plane'][assigned] < 10), \
        "Distance to plane should be reasonable"
    
    # Plane area should be positive
    assert np.all(features['plane_area'][assigned] > 0), \
        "Plane area should be positive"
    
    # Orientation should be in [0, 90] degrees
    assert np.all(features['plane_orientation'][assigned] >= 0), \
        "Plane orientation should be >= 0"
    assert np.all(features['plane_orientation'][assigned] <= 90), \
        "Plane orientation should be <= 90 degrees"
    
    # Planarity should be in [0, 1]
    assert np.all(features['plane_planarity'][assigned] >= 0), \
        "Plane planarity should be >= 0"
    assert np.all(features['plane_planarity'][assigned] <= 1), \
        "Plane planarity should be <= 1"
    
    # UV coordinates should be in [0, 1]
    assert np.all(features['position_on_plane_u'][assigned] >= 0), \
        "U coordinate should be >= 0"
    assert np.all(features['position_on_plane_u'][assigned] <= 1), \
        "U coordinate should be <= 1"
    assert np.all(features['position_on_plane_v'][assigned] >= 0), \
        "V coordinate should be >= 0"
    assert np.all(features['position_on_plane_v'][assigned] <= 1), \
        "V coordinate should be <= 1"


def test_no_planes_detected():
    """Test behavior when no planes are detected."""
    # Create random noise points (no planar structures)
    n_points = 100
    points = np.random.rand(n_points, 3) * 10
    normals = np.random.rand(n_points, 3)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    planarity = np.random.rand(n_points) * 0.3  # Low planarity
    
    detector = PlaneDetector(min_points_per_plane=50)
    extractor = PlaneFeatureExtractor(detector)
    
    features = extractor.detect_and_assign_planes(
        points, normals, planarity
    )
    
    # All points should have plane_id = -1
    assert np.all(features['plane_id'] == -1), \
        "All points should be unassigned when no planes detected"
    
    # Statistics should show no planes
    stats = extractor.get_plane_statistics()
    assert stats['n_planes'] == 0, "Should report 0 planes"


if __name__ == '__main__':
    # Run tests manually
    building = synthetic_building()
    
    print("Testing plane feature extraction...")
    test_plane_feature_extractor_basic(building)
    print("✓ Basic extraction test passed")
    
    test_plane_detection_statistics(building)
    print("✓ Statistics test passed")
    
    test_plane_feature_values(building)
    print("✓ Feature value range test passed")
    
    test_no_planes_detected()
    print("✓ No planes detected test passed")
    
    print("\n✅ All tests passed!")
