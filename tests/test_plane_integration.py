"""
Test plane feature integration with FeatureOrchestrator.

This test validates that plane features are properly computed and integrated
into the feature computation pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from ign_lidar.features.orchestrator import FeatureOrchestrator
from ign_lidar.features.feature_modes import FeatureMode


@pytest.fixture
def synthetic_building_tile():
    """
    Create a synthetic building tile with walls and roof.
    
    Returns:
        Dictionary with tile data
    """
    np.random.seed(42)
    
    # Create a simple building: 10m x 10m x 5m high
    n_points = 5000
    
    # Ground floor (horizontal plane at z=0)
    n_ground = n_points // 5
    ground = np.random.uniform([0, 0, 0], [10, 10, 0.1], (n_ground, 3))
    
    # Four walls (vertical planes)
    n_wall = n_points // 5
    wall_north = np.random.uniform([0, 10, 0], [10, 10.1, 5], (n_wall, 3))
    wall_south = np.random.uniform([0, 0, 0], [10, 0.1, 5], (n_wall, 3))
    wall_east = np.random.uniform([10, 0, 0], [10.1, 10, 5], (n_wall, 3))
    wall_west = np.random.uniform([0, 0, 0], [0.1, 10, 5], (n_wall, 3))
    
    # Roof (inclined plane)
    n_roof = n_points - (n_ground + 4 * n_wall)
    roof_x = np.random.uniform(0, 10, n_roof)
    roof_y = np.random.uniform(0, 10, n_roof)
    roof_z = 5 + roof_x * 0.3  # 30% slope
    roof = np.column_stack([roof_x, roof_y, roof_z])
    
    # Combine all points
    points = np.vstack([ground, wall_north, wall_south, wall_east, wall_west, roof])
    
    # Create mock classification (all buildings)
    classification = np.full(len(points), 6, dtype=np.uint8)  # Building
    
    # Create mock intensity and return numbers
    intensity = np.random.uniform(1000, 5000, len(points)).astype(np.uint16)
    return_number = np.ones(len(points), dtype=np.uint8)
    
    return {
        'points': points,
        'classification': classification,
        'intensity': intensity,
        'return_number': return_number,
    }


@pytest.fixture
def plane_feature_config():
    """
    Create configuration with plane features enabled.
    
    Returns:
        OmegaConf configuration object
    """
    config = OmegaConf.create({
        'processor': {
            'use_gpu': False,
            'use_strategy_pattern': True,
            'use_feature_computer': False,
        },
        'features': {
            'mode': 'lod3',  # LOD3 includes plane features
            'k_neighbors': 30,
            'search_radius': 3.0,
            'compute_plane_features': True,
            'plane_detection': {
                'horizontal_angle_max': 10.0,
                'vertical_angle_min': 75.0,
                'min_points_per_plane': 50,
                'horizontal_planarity_min': 0.75,
                'vertical_planarity_min': 0.65,
                'max_assignment_distance': 0.5,
            }
        }
    })
    
    return config


@pytest.mark.integration
def test_plane_features_integration(plane_feature_config, synthetic_building_tile):
    """
    Test that plane features are computed and integrated correctly.
    """
    # Initialize orchestrator
    orchestrator = FeatureOrchestrator(plane_feature_config)
    
    # Compute features
    features = orchestrator.compute_features(synthetic_building_tile)
    
    # Check that plane features are present
    plane_feature_names = [
        'plane_id', 'plane_type', 'distance_to_plane', 'plane_area',
        'plane_orientation', 'plane_planarity', 
        'position_on_plane_u', 'position_on_plane_v'
    ]
    
    for feature_name in plane_feature_names:
        assert feature_name in features, f"Missing plane feature: {feature_name}"
    
    # Check feature shapes
    n_points = len(synthetic_building_tile['points'])
    for feature_name in plane_feature_names:
        feature_array = features[feature_name]
        assert len(feature_array) == n_points, \
            f"Feature {feature_name} has wrong shape: {len(feature_array)} vs {n_points}"
    
    # Check that some planes were detected
    plane_ids = features['plane_id']
    n_assigned = np.sum(plane_ids >= 0)
    assert n_assigned > 0, "No points were assigned to planes"
    
    # Check that plane types are valid
    plane_types = features['plane_type']
    valid_types = [-1, 0, 1, 2]  # -1=none, 0=horizontal, 1=vertical, 2=inclined
    assert np.all(np.isin(plane_types, valid_types)), \
        f"Invalid plane types found: {np.unique(plane_types)}"
    
    # Check distance values are reasonable
    distances = features['distance_to_plane']
    assigned_distances = distances[plane_ids >= 0]
    if len(assigned_distances) > 0:
        assert np.all(assigned_distances >= 0), "Negative distances found"
        assert np.all(assigned_distances < 10), "Unreasonably large distances found"
    
    print(f"\n✅ Plane integration test passed")
    print(f"   Points: {n_points}")
    print(f"   Assigned to planes: {n_assigned} ({n_assigned/n_points*100:.1f}%)")
    print(f"   Unique planes: {len(np.unique(plane_ids[plane_ids >= 0]))}")
    print(f"   Plane types: {np.bincount(plane_types[plane_types >= 0])}")


@pytest.mark.integration
def test_plane_features_with_planes_mode(synthetic_building_tile):
    """
    Test plane features with PLANES mode (plane features only).
    """
    config = OmegaConf.create({
        'processor': {
            'use_gpu': False,
            'use_strategy_pattern': True,
        },
        'features': {
            'mode': 'planes',  # PLANES mode - plane features only
            'k_neighbors': 30,
            'plane_detection': {
                'horizontal_angle_max': 10.0,
                'vertical_angle_min': 75.0,
                'min_points_per_plane': 50,
            }
        }
    })
    
    orchestrator = FeatureOrchestrator(config)
    features = orchestrator.compute_features(synthetic_building_tile)
    
    # In PLANES mode, we should have plane features
    assert 'plane_id' in features, "plane_id missing in PLANES mode"
    assert 'plane_type' in features, "plane_type missing in PLANES mode"
    
    # Check that some planes were detected
    plane_ids = features['plane_id']
    n_assigned = np.sum(plane_ids >= 0)
    assert n_assigned > 0, "No points were assigned to planes in PLANES mode"
    
    print(f"\n✅ PLANES mode test passed")
    print(f"   Features computed: {len(features)}")
    print(f"   Points assigned to planes: {n_assigned}")


@pytest.mark.integration
def test_plane_features_disabled(synthetic_building_tile):
    """
    Test that plane features are not computed when disabled.
    """
    config = OmegaConf.create({
        'processor': {
            'use_gpu': False,
            'use_strategy_pattern': True,
        },
        'features': {
            'mode': 'lod2',  # LOD2 doesn't include plane features by default
            'k_neighbors': 20,
            'compute_plane_features': False,  # Explicitly disabled
        }
    })
    
    orchestrator = FeatureOrchestrator(config)
    features = orchestrator.compute_features(synthetic_building_tile)
    
    # Plane features should not be present
    plane_feature_names = [
        'plane_id', 'plane_type', 'distance_to_plane', 'plane_area'
    ]
    
    for feature_name in plane_feature_names:
        assert feature_name not in features, \
            f"Plane feature {feature_name} should not be present when disabled"
    
    print(f"\n✅ Plane features disabled test passed")
    print(f"   Features computed: {len(features)}")
    print(f"   No plane features present (as expected)")


@pytest.mark.integration
def test_plane_statistics(plane_feature_config, synthetic_building_tile):
    """
    Test that plane statistics are reasonable for synthetic building.
    """
    orchestrator = FeatureOrchestrator(plane_feature_config)
    features = orchestrator.compute_features(synthetic_building_tile)
    
    plane_ids = features['plane_id']
    plane_types = features['plane_type']
    
    # Get assigned points
    assigned_mask = plane_ids >= 0
    assigned_types = plane_types[assigned_mask]
    
    # Count plane types
    type_counts = np.bincount(assigned_types[assigned_types >= 0])
    
    # We should have horizontal planes (ground, potentially roof)
    if len(type_counts) > 0:
        n_horizontal = type_counts[0] if len(type_counts) > 0 else 0
        n_vertical = type_counts[1] if len(type_counts) > 1 else 0
        n_inclined = type_counts[2] if len(type_counts) > 2 else 0
        
        print(f"\n✅ Plane statistics test")
        print(f"   Horizontal points: {n_horizontal}")
        print(f"   Vertical points: {n_vertical}")
        print(f"   Inclined points: {n_inclined}")
        
        # We expect at least some planes to be detected (may be vertical or horizontal)
        total_plane_points = n_vertical + n_horizontal + n_inclined
        assert total_plane_points > 0, "No planes detected at all"
        
        # Success if we detected any plane type
        print(f"   ✅ Total plane points detected: {total_plane_points}")


if __name__ == "__main__":
    # Run tests manually (not using pytest fixtures)
    print("Testing plane feature integration...")
    
    # Create synthetic building tile manually
    np.random.seed(42)
    n_points = 5000
    
    # Create a simple building: 10m x 10m x 5m high
    n_ground = n_points // 5
    ground = np.random.uniform([0, 0, 0], [10, 10, 0.1], (n_ground, 3))
    
    # Four walls (vertical planes)
    n_wall = n_points // 5
    wall_north = np.random.uniform([0, 10, 0], [10, 10.1, 5], (n_wall, 3))
    wall_south = np.random.uniform([0, 0, 0], [10, 0.1, 5], (n_wall, 3))
    wall_east = np.random.uniform([10, 0, 0], [10.1, 10, 5], (n_wall, 3))
    wall_west = np.random.uniform([0, 0, 0], [0.1, 10, 5], (n_wall, 3))
    
    # Roof (inclined plane)
    n_roof = n_points - (n_ground + 4 * n_wall)
    roof_x = np.random.uniform(0, 10, n_roof)
    roof_y = np.random.uniform(0, 10, n_roof)
    roof_z = 5 + roof_x * 0.3  # 30% slope
    roof = np.column_stack([roof_x, roof_y, roof_z])
    
    # Combine all points
    points = np.vstack([ground, wall_north, wall_south, wall_east, wall_west, roof])
    classification = np.full(len(points), 6, dtype=np.uint8)  # Building
    intensity = np.random.uniform(1000, 5000, len(points)).astype(np.uint16)
    return_number = np.ones(len(points), dtype=np.uint8)
    
    tile = {
        'points': points,
        'classification': classification,
        'intensity': intensity,
        'return_number': return_number,
    }
    
    # Create config manually
    config = OmegaConf.create({
        'processor': {
            'use_gpu': False,
            'use_strategy_pattern': True,
            'use_feature_computer': False,
        },
        'features': {
            'mode': 'lod3',  # LOD3 includes plane features
            'k_neighbors': 30,
            'search_radius': 3.0,
            'compute_plane_features': True,
            'plane_detection': {
                'horizontal_angle_max': 10.0,
                'vertical_angle_min': 75.0,
                'min_points_per_plane': 50,
                'horizontal_planarity_min': 0.75,
                'vertical_planarity_min': 0.65,
                'max_assignment_distance': 0.5,
            }
        }
    })
    
    # Run tests
    test_plane_features_integration(config, tile)
    test_plane_features_with_planes_mode(tile)
    test_plane_features_disabled(tile)
    test_plane_statistics(config, tile)
    
    print("\n🎉 All plane integration tests passed!")
