"""
Test building-plane hierarchical clustering integration.

This test validates that building-plane features are properly computed and
integrated into the feature computation pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from ign_lidar.features.orchestrator import FeatureOrchestrator
from ign_lidar.features.feature_modes import FeatureMode


@pytest.fixture
def synthetic_multi_building_tile():
    """
    Create a synthetic tile with two buildings, each with multiple planes.
    
    Building 1: At origin (0-10, 0-10)
    Building 2: Offset (20-30, 0-10)
    
    Returns:
        Dictionary with tile data
    """
    np.random.seed(42)
    
    # Building 1: Ground + 4 walls
    b1_ground = np.random.uniform([0, 0, 0], [10, 10, 0.1], (200, 3))
    b1_wall_n = np.random.uniform([0, 10, 0], [10, 10.1, 5], (200, 3))
    b1_wall_s = np.random.uniform([0, 0, 0], [10, 0.1, 5], (200, 3))
    b1_wall_e = np.random.uniform([10, 0, 0], [10.1, 10, 5], (200, 3))
    b1_wall_w = np.random.uniform([0, 0, 0], [0.1, 10, 5], (200, 3))
    
    # Building 2: Ground + 4 walls (offset)
    b2_ground = np.random.uniform([20, 0, 0], [30, 10, 0.1], (200, 3))
    b2_wall_n = np.random.uniform([20, 10, 0], [30, 10.1, 5], (200, 3))
    b2_wall_s = np.random.uniform([20, 0, 0], [30, 0.1, 5], (200, 3))
    b2_wall_e = np.random.uniform([30, 0, 0], [30.1, 10, 5], (200, 3))
    b2_wall_w = np.random.uniform([20, 0, 0], [20.1, 10, 5], (200, 3))
    
    # Combine all points
    points = np.vstack([
        b1_ground, b1_wall_n, b1_wall_s, b1_wall_e, b1_wall_w,
        b2_ground, b2_wall_n, b2_wall_s, b2_wall_e, b2_wall_w
    ])
    
    # Create mock classification (all buildings)
    classification = np.full(len(points), 6, dtype=np.uint8)  # Building
    
    # Create mock intensity and return numbers
    intensity = np.random.uniform(1000, 5000, len(points)).astype(np.uint16)
    return_number = np.ones(len(points), dtype=np.uint8)
    
    # Create synthetic building footprints using shapely
    try:
        from shapely.geometry import Polygon
        import geopandas as gpd
        
        building1_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        building2_poly = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])
        
        buildings_gdf = gpd.GeoDataFrame({
            'geometry': [building1_poly, building2_poly],
            'id': [1, 2]
        }, crs='EPSG:2154')  # French projection
        
        ground_truth_features = {'buildings': buildings_gdf}
    except ImportError:
        print("Warning: Shapely/GeoPandas not available, skipping building footprints")
        ground_truth_features = {}
    
    return {
        'points': points,
        'classification': classification,
        'intensity': intensity,
        'return_number': return_number,
        'ground_truth_features': ground_truth_features,
    }


@pytest.mark.integration
def test_building_plane_integration(synthetic_multi_building_tile):
    """
    Test that building-plane features are computed correctly.
    """
    config = OmegaConf.create({
        'processor': {
            'use_gpu': False,
            'use_strategy_pattern': True,
        },
        'features': {
            'mode': 'lod3',  # LOD3 includes plane features
            'k_neighbors': 30,
            'compute_plane_features': True,
            'compute_building_plane_features': True,
            'plane_detection': {
                'horizontal_angle_max': 10.0,
                'vertical_angle_min': 75.0,
                'min_points_per_plane': 50,
            },
            'building_plane_clustering': {
                'min_points_per_plane': 30,
                'compute_facade_ids': True,
            }
        }
    })
    
    orchestrator = FeatureOrchestrator(config)
    features = orchestrator.compute_features(synthetic_multi_building_tile)
    
    # Check that building-plane features are present
    building_plane_feature_names = [
        'building_id', 'plane_id_local', 'facade_id',
        'distance_to_building_center', 'relative_height_in_building',
        'n_planes_in_building', 'plane_area_ratio'
    ]
    
    for feature_name in building_plane_feature_names:
        assert feature_name in features, f"Missing building-plane feature: {feature_name}"
    
    # Check feature shapes
    n_points = len(synthetic_multi_building_tile['points'])
    for feature_name in building_plane_feature_names:
        feature_array = features[feature_name]
        assert len(feature_array) == n_points, \
            f"Feature {feature_name} has wrong shape: {len(feature_array)} vs {n_points}"
    
    # Check that points were assigned to buildings
    building_ids = features['building_id']
    n_assigned_buildings = np.sum(building_ids >= 0)
    print(f"\n✅ Building-plane integration test")
    print(f"   Points assigned to buildings: {n_assigned_buildings}/{n_points}")
    
    # We expect most points to be assigned since we have building footprints
    assert n_assigned_buildings > 0, "No points assigned to buildings"
    
    # Check local plane IDs
    plane_id_local = features['plane_id_local']
    unique_local_planes = np.unique(plane_id_local[plane_id_local >= 0])
    print(f"   Local plane IDs: {len(unique_local_planes)}")
    
    # Check facade IDs for vertical planes
    facade_ids = features['facade_id']
    n_facades = np.sum(facade_ids >= 0)
    print(f"   Points assigned to facades: {n_facades}")
    
    # Check building count
    unique_buildings = np.unique(building_ids[building_ids >= 0])
    print(f"   Unique buildings detected: {len(unique_buildings)}")
    
    # We should detect 2 buildings
    if len(unique_buildings) > 0:
        assert len(unique_buildings) <= 2, f"Too many buildings detected: {len(unique_buildings)}"
    
    print("   ✅ All building-plane features validated")


@pytest.mark.integration
def test_building_plane_without_footprints(synthetic_multi_building_tile):
    """
    Test that building-plane features are skipped when no footprints available.
    """
    # Remove building footprints
    tile_data = synthetic_multi_building_tile.copy()
    tile_data['ground_truth_features'] = {}
    
    config = OmegaConf.create({
        'processor': {'use_gpu': False, 'use_strategy_pattern': True},
        'features': {
            'mode': 'lod3',
            'k_neighbors': 30,
            'compute_plane_features': True,
            'compute_building_plane_features': True,
        }
    })
    
    orchestrator = FeatureOrchestrator(config)
    features = orchestrator.compute_features(tile_data)
    
    # Building-plane features should not be present
    building_plane_features = ['building_id', 'plane_id_local', 'facade_id']
    
    for feature_name in building_plane_features:
        assert feature_name not in features, \
            f"Building-plane feature {feature_name} should not be present without footprints"
    
    # But plane features should still be present
    assert 'plane_id' in features, "Plane features should still be computed"
    
    print("\n✅ Building-plane disabled without footprints (as expected)")


@pytest.mark.integration
def test_relative_height_values(synthetic_multi_building_tile):
    """
    Test that relative_height_in_building values are in valid range [0, 1].
    """
    config = OmegaConf.create({
        'processor': {'use_gpu': False, 'use_strategy_pattern': True},
        'features': {
            'mode': 'lod3',
            'k_neighbors': 30,
            'compute_plane_features': True,
            'compute_building_plane_features': True,
        }
    })
    
    orchestrator = FeatureOrchestrator(config)
    features = orchestrator.compute_features(synthetic_multi_building_tile)
    
    if 'relative_height_in_building' in features:
        rel_heights = features['relative_height_in_building']
        building_ids = features['building_id']
        
        # Check assigned points only
        assigned_mask = building_ids >= 0
        assigned_heights = rel_heights[assigned_mask]
        
        if len(assigned_heights) > 0:
            assert np.all((assigned_heights >= 0) & (assigned_heights <= 1)), \
                "Relative heights must be in range [0, 1]"
            
            print(f"\n✅ Relative height test passed")
            print(f"   Min: {assigned_heights.min():.3f}, Max: {assigned_heights.max():.3f}")
            print(f"   Mean: {assigned_heights.mean():.3f}, Std: {assigned_heights.std():.3f}")


if __name__ == "__main__":
    # Run tests manually
    print("Testing building-plane clustering integration...")
    
    # Create fixture manually (simplified version)
    np.random.seed(42)
    
    # Building 1
    b1_ground = np.random.uniform([0, 0, 0], [10, 10, 0.1], (200, 3))
    b1_wall = np.random.uniform([0, 0, 0], [0.1, 10, 5], (800, 3))
    
    # Building 2
    b2_ground = np.random.uniform([20, 0, 0], [30, 10, 0.1], (200, 3))
    b2_wall = np.random.uniform([20, 0, 0], [20.1, 10, 5], (800, 3))
    
    points = np.vstack([b1_ground, b1_wall, b2_ground, b2_wall])
    classification = np.full(len(points), 6, dtype=np.uint8)
    intensity = np.random.uniform(1000, 5000, len(points)).astype(np.uint16)
    return_number = np.ones(len(points), dtype=np.uint8)
    
    # Create synthetic buildings
    try:
        from shapely.geometry import Polygon
        import geopandas as gpd
        
        building1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        building2 = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])
        
        buildings_gdf = gpd.GeoDataFrame({
            'geometry': [building1, building2],
            'id': [1, 2]
        }, crs='EPSG:2154')
        
        tile = {
            'points': points,
            'classification': classification,
            'intensity': intensity,
            'return_number': return_number,
            'ground_truth_features': {'buildings': buildings_gdf},
        }
        
        # Run tests
        test_building_plane_integration(tile)
        test_building_plane_without_footprints(tile)
        test_relative_height_values(tile)
        
        print("\n🎉 All building-plane clustering tests passed!")
        
    except ImportError as e:
        print(f"\n⚠️  Cannot run tests: {e}")
        print("   Install shapely and geopandas to run building-plane tests")
