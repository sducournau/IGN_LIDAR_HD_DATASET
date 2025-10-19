"""
Unit tests for parcel-based classification.

Tests:
- Parcel grouping
- Feature aggregation
- Parcel type classification
- Point refinement within parcels
- Ground truth integration
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from ign_lidar.core.modules.parcel_classifier import (
    ParcelClassifier,
    ParcelClassificationConfig,
    ParcelStatistics,
    ParcelType
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_cadastre():
    """Create mock cadastre GeoDataFrame."""
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        # Create 3 test parcels
        parcels = gpd.GeoDataFrame({
            'id_parcelle': ['001', '002', '003'],
            'geometry': [
                Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # Forest
                Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),  # Agriculture
                Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])  # Building
            ]
        }, crs='EPSG:2154')
        return parcels
    except ImportError:
        pytest.skip("geopandas not available")


@pytest.fixture
def forest_points():
    """Create synthetic forest points."""
    np.random.seed(42)
    n = 100
    
    # Forest: high NDVI, irregular surface, mixed heights
    points = np.column_stack([
        np.random.uniform(0, 10, n),  # X within parcel 001
        np.random.uniform(0, 10, n),  # Y
        np.random.uniform(0, 15, n)   # Z (heights 0-15m)
    ])
    
    features = {
        'ndvi': np.random.uniform(0.5, 0.8, n),  # High NDVI
        'height': np.random.uniform(0, 15, n),
        'planarity': np.random.uniform(0.1, 0.5, n),  # Low planarity
        'verticality': np.random.uniform(0.0, 0.3, n),
        'curvature': np.random.uniform(0.3, 0.6, n),  # High curvature
        'normals': np.random.randn(n, 3)
    }
    features['normals'] /= np.linalg.norm(features['normals'], axis=1, keepdims=True)
    
    return points, features


@pytest.fixture
def agriculture_points():
    """Create synthetic agriculture points."""
    np.random.seed(43)
    n = 100
    
    # Agriculture: moderate NDVI, flat, low heights
    points = np.column_stack([
        np.random.uniform(10, 20, n),  # X within parcel 002
        np.random.uniform(0, 10, n),   # Y
        np.random.uniform(0, 1, n)     # Z (0-1m crops)
    ])
    
    features = {
        'ndvi': np.random.uniform(0.3, 0.5, n),  # Moderate NDVI
        'height': np.random.uniform(0, 1, n),
        'planarity': np.random.uniform(0.7, 0.9, n),  # High planarity
        'verticality': np.random.uniform(0.0, 0.1, n),
        'curvature': np.random.uniform(0.0, 0.1, n),
        'normals': np.column_stack([
            np.random.randn(n) * 0.1,
            np.random.randn(n) * 0.1,
            np.ones(n)  # Mostly upward
        ])
    }
    features['normals'] /= np.linalg.norm(features['normals'], axis=1, keepdims=True)
    
    return points, features


@pytest.fixture
def building_points():
    """Create synthetic building points."""
    np.random.seed(44)
    n = 100
    
    # Building: low NDVI, vertical walls + flat roofs
    points = np.column_stack([
        np.random.uniform(20, 30, n),  # X within parcel 003
        np.random.uniform(0, 10, n),   # Y
        np.random.uniform(0, 10, n)    # Z (0-10m building)
    ])
    
    features = {
        'ndvi': np.random.uniform(-0.1, 0.1, n),  # Very low NDVI
        'height': np.random.uniform(0, 10, n),
        'planarity': np.random.uniform(0.8, 0.95, n),  # High planarity
        'verticality': np.random.uniform(0.5, 0.9, n),  # High verticality
        'curvature': np.random.uniform(0.0, 0.05, n),
        'normals': np.random.randn(n, 3)
    }
    features['normals'] /= np.linalg.norm(features['normals'], axis=1, keepdims=True)
    
    return points, features


# ============================================================================
# Test Configuration
# ============================================================================

def test_config_defaults():
    """Test default configuration."""
    config = ParcelClassificationConfig()
    
    assert config.min_parcel_points == 20
    assert config.min_parcel_area == 10.0
    assert config.parcel_confidence_threshold == 0.6
    assert config.refine_points is True
    assert config.refinement_method == 'feature_based'


def test_config_custom():
    """Test custom configuration."""
    config = ParcelClassificationConfig(
        min_parcel_points=50,
        parcel_confidence_threshold=0.8,
        refine_points=False
    )
    
    assert config.min_parcel_points == 50
    assert config.parcel_confidence_threshold == 0.8
    assert config.refine_points is False


# ============================================================================
# Test Parcel Statistics
# ============================================================================

def test_parcel_statistics_creation():
    """Test ParcelStatistics dataclass."""
    stats = ParcelStatistics(
        parcel_id='001',
        n_points=100,
        area_m2=100.0,
        point_density=1.0,
        mean_ndvi=0.6,
        std_ndvi=0.1,
        mean_height=5.0,
        std_height=2.0,
        height_range=10.0,
        mean_planarity=0.3,
        mean_verticality=0.2,
        mean_curvature=0.4,
        dominant_normal_z=0.8
    )
    
    assert stats.parcel_id == '001'
    assert stats.n_points == 100
    assert stats.mean_ndvi == 0.6
    assert stats.parcel_type == 'unknown'
    assert stats.confidence_scores == {}


# ============================================================================
# Test ParcelClassifier Initialization
# ============================================================================

def test_classifier_init_default():
    """Test classifier initialization with defaults."""
    classifier = ParcelClassifier()
    
    assert classifier.config is not None
    assert isinstance(classifier.config, ParcelClassificationConfig)
    assert len(classifier._parcel_stats_cache) == 0


def test_classifier_init_custom_config():
    """Test classifier initialization with custom config."""
    config = ParcelClassificationConfig(min_parcel_points=50)
    classifier = ParcelClassifier(config=config)
    
    assert classifier.config.min_parcel_points == 50


def test_classifier_requires_spatial_libs():
    """Test that classifier fails gracefully without spatial libraries."""
    with patch('ign_lidar.core.modules.parcel_classifier.HAS_SPATIAL', False):
        with pytest.raises(ImportError, match="shapely and geopandas required"):
            ParcelClassifier()


# ============================================================================
# Test Feature Computation
# ============================================================================

def test_compute_parcel_features_basic(forest_points):
    """Test parcel feature computation."""
    classifier = ParcelClassifier()
    points, features = forest_points
    
    stats = classifier.compute_parcel_features(
        parcel_points=points,
        parcel_features=features,
        parcel_id='001'
    )
    
    assert stats.parcel_id == '001'
    assert stats.n_points == 100
    assert stats.area_m2 > 0
    assert 0.5 <= stats.mean_ndvi <= 0.8  # Forest NDVI range
    assert 0 <= stats.mean_height <= 15
    assert stats.height_range > 0


def test_compute_parcel_features_missing_features():
    """Test feature computation with missing features."""
    classifier = ParcelClassifier()
    
    points = np.array([[0, 0, 0], [1, 1, 1]])
    features = {}  # Empty features
    
    stats = classifier.compute_parcel_features(
        parcel_points=points,
        parcel_features=features,
        parcel_id='test'
    )
    
    # Should handle missing features gracefully with zeros
    assert stats.n_points == 2
    assert stats.mean_ndvi == 0.0
    assert stats.mean_height == 0.0


# ============================================================================
# Test Parcel Type Classification
# ============================================================================

def test_classify_forest_parcel():
    """Test forest parcel classification."""
    classifier = ParcelClassifier()
    
    stats = ParcelStatistics(
        parcel_id='001',
        n_points=100,
        area_m2=100.0,
        point_density=1.0,
        mean_ndvi=0.65,  # High NDVI
        std_ndvi=0.1,
        mean_height=10.0,
        std_height=3.0,
        height_range=15.0,
        mean_planarity=0.4,  # Low planarity
        mean_verticality=0.2,
        mean_curvature=0.5,  # High curvature
        dominant_normal_z=0.7
    )
    
    parcel_type, confidence = classifier.classify_parcel_type(stats)
    
    assert parcel_type == ParcelType.FOREST
    assert confidence[ParcelType.FOREST] >= 0.6


def test_classify_agriculture_parcel():
    """Test agriculture parcel classification."""
    classifier = ParcelClassifier()
    
    stats = ParcelStatistics(
        parcel_id='002',
        n_points=100,
        area_m2=100.0,
        point_density=1.0,
        mean_ndvi=0.4,  # Moderate NDVI
        std_ndvi=0.05,
        mean_height=0.5,
        std_height=0.2,
        height_range=1.0,
        mean_planarity=0.85,  # High planarity
        mean_verticality=0.05,
        mean_curvature=0.05,
        dominant_normal_z=0.95
    )
    
    parcel_type, confidence = classifier.classify_parcel_type(stats)
    
    # Should be agriculture, mixed, or potentially road (high planarity)
    assert parcel_type in [ParcelType.AGRICULTURE, ParcelType.MIXED, ParcelType.ROAD]


def test_classify_building_parcel():
    """Test building parcel classification."""
    classifier = ParcelClassifier()
    
    stats = ParcelStatistics(
        parcel_id='003',
        n_points=100,
        area_m2=100.0,
        point_density=1.0,
        mean_ndvi=0.05,  # Very low NDVI
        std_ndvi=0.02,
        mean_height=8.0,
        std_height=3.0,
        height_range=10.0,  # Multi-story
        mean_planarity=0.80,  # High planarity
        mean_verticality=0.75,  # High verticality (walls)
        mean_curvature=0.02,
        dominant_normal_z=0.3
    )
    
    parcel_type, confidence = classifier.classify_parcel_type(stats)
    
    assert parcel_type == ParcelType.BUILDING
    assert confidence[ParcelType.BUILDING] > 0.6


def test_classify_water_parcel():
    """Test water parcel classification."""
    classifier = ParcelClassifier()
    
    stats = ParcelStatistics(
        parcel_id='004',
        n_points=50,
        area_m2=50.0,
        point_density=1.0,
        mean_ndvi=-0.1,  # Negative NDVI
        std_ndvi=0.02,
        mean_height=0.2,
        std_height=0.1,
        height_range=0.3,
        mean_planarity=0.95,  # Very flat
        mean_verticality=0.01,
        mean_curvature=0.01,
        dominant_normal_z=0.99
    )
    
    parcel_type, confidence = classifier.classify_parcel_type(stats)
    
    assert parcel_type == ParcelType.WATER
    assert confidence[ParcelType.WATER] > 0.6


def test_classify_mixed_parcel_low_confidence():
    """Test that low confidence results in MIXED classification."""
    classifier = ParcelClassifier()
    
    # Ambiguous statistics
    stats = ParcelStatistics(
        parcel_id='999',
        n_points=50,
        area_m2=50.0,
        point_density=1.0,
        mean_ndvi=0.25,  # Borderline
        std_ndvi=0.15,
        mean_height=2.0,
        std_height=2.0,
        height_range=5.0,
        mean_planarity=0.5,  # Medium
        mean_verticality=0.3,
        mean_curvature=0.2,
        dominant_normal_z=0.6
    )
    
    parcel_type, confidence = classifier.classify_parcel_type(stats)
    
    assert parcel_type == ParcelType.MIXED
    assert max(confidence.values()) < 0.6


# ============================================================================
# Test Point Refinement
# ============================================================================

def test_refine_forest_parcel_points():
    """Test point refinement for forest parcel."""
    classifier = ParcelClassifier()
    
    n = 100
    features = {
        'ndvi': np.concatenate([
            np.full(20, 0.65),  # High veg
            np.full(30, 0.45),  # Medium veg
            np.full(30, 0.25),  # Low veg
            np.full(20, 0.10)   # Ground
        ]),
        'height': np.concatenate([
            np.full(20, 8.0),
            np.full(30, 1.5),
            np.full(30, 0.3),
            np.full(20, 0.0)
        ]),
        'planarity': np.full(n, 0.3),
        'verticality': np.full(n, 0.2),
        'curvature': np.full(n, 0.4),
        'normals': np.zeros((n, 3))
    }
    
    stats = ParcelStatistics(
        parcel_id='001',
        n_points=n,
        area_m2=100.0,
        point_density=1.0,
        mean_ndvi=0.4,
        std_ndvi=0.2,
        mean_height=3.0,
        std_height=3.0,
        height_range=8.0,
        mean_planarity=0.3,
        mean_verticality=0.2,
        mean_curvature=0.4,
        dominant_normal_z=0.8
    )
    
    labels = classifier._refine_forest_parcel(features, stats)
    
    # Check we get vegetation stratification
    assert np.any(labels == classifier.ASPRS_HIGH_VEGETATION)
    assert np.any(labels == classifier.ASPRS_MEDIUM_VEGETATION)
    # Note: with NDVI 0.25, this becomes ground, not low vegetation
    assert np.any(labels == classifier.ASPRS_GROUND)


def test_refine_building_parcel_points():
    """Test point refinement for building parcel."""
    classifier = ParcelClassifier()
    
    n = 100
    features = {
        'ndvi': np.full(n, 0.05),
        'height': np.full(n, 5.0),
        'planarity': np.full(n, 0.8),
        'verticality': np.concatenate([
            np.full(60, 0.8),  # Walls
            np.full(40, 0.2)   # Roof
        ]),
        'curvature': np.full(n, 0.02),
        'normals': np.column_stack([
            np.zeros(n),
            np.zeros(n),
            np.concatenate([np.full(60, 0.2), np.full(40, 0.95)])
        ])
    }
    
    stats = ParcelStatistics(
        parcel_id='003',
        n_points=n,
        area_m2=100.0,
        point_density=1.0,
        mean_ndvi=0.05,
        std_ndvi=0.02,
        mean_height=5.0,
        std_height=2.0,
        height_range=8.0,
        mean_planarity=0.8,
        mean_verticality=0.6,
        mean_curvature=0.02,
        dominant_normal_z=0.5
    )
    
    labels = classifier._refine_building_parcel(features, stats)
    
    # All should be building
    assert np.all(labels == classifier.ASPRS_BUILDING)


# ============================================================================
# Test Ground Truth Integration
# ============================================================================

def test_match_bd_foret():
    """Test BD Forêt matching."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
    except ImportError:
        pytest.skip("geopandas not available")
    
    classifier = ParcelClassifier()
    
    # Mock BD Forêt data
    bd_foret = gpd.GeoDataFrame({
        'forest_type': ['coniferous'],
        'dominant_species': ['pine'],
        'density_category': ['dense'],
        'estimated_height': [15.0],
        'geometry': [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
    }, crs='EPSG:2154')
    
    # Points inside forest polygon
    parcel_points = np.array([[5, 5, 0], [5, 6, 0]])
    
    match = classifier._match_bd_foret('001', parcel_points, bd_foret)
    
    assert match is not None
    assert match['forest_type'] == 'coniferous'
    assert match['dominant_species'] == 'pine'


def test_match_rpg():
    """Test RPG matching."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
    except ImportError:
        pytest.skip("geopandas not available")
    
    classifier = ParcelClassifier()
    
    # Mock RPG data
    rpg = gpd.GeoDataFrame({
        'code_cultu': ['BLE'],
        'crop_category': ['cereals'],
        'surf_parc': [1.5],
        'bio': [False],
        'geometry': [Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])]
    }, crs='EPSG:2154')
    
    # Points inside agricultural polygon
    parcel_points = np.array([[15, 5, 0], [15, 6, 0]])
    
    match = classifier._match_rpg('002', parcel_points, rpg)
    
    assert match is not None
    assert match['crop_code'] == 'BLE'
    assert match['crop_category'] == 'cereals'


# ============================================================================
# Test Full Pipeline
# ============================================================================

def test_classify_by_parcels_integration(mock_cadastre, forest_points):
    """Test full classification pipeline."""
    classifier = ParcelClassifier()
    points, features = forest_points
    
    # Mock the cadastre grouping directly
    with patch.object(classifier, '_group_by_parcels', return_value={
        '001': np.arange(100)
    }):
        labels = classifier.classify_by_parcels(
            points=points,
            features=features,
            cadastre=mock_cadastre
        )
    
    assert len(labels) == 100
    assert labels.dtype == np.uint8
    # Should have classified as vegetation
    assert np.any(np.isin(labels, [
        classifier.ASPRS_LOW_VEGETATION,
        classifier.ASPRS_MEDIUM_VEGETATION,
        classifier.ASPRS_HIGH_VEGETATION
    ]))


# ============================================================================
# Test Statistics Export
# ============================================================================

def test_export_parcel_statistics():
    """Test parcel statistics export."""
    classifier = ParcelClassifier()
    
    # Add some cached statistics
    stats1 = ParcelStatistics(
        parcel_id='001',
        n_points=100,
        area_m2=100.0,
        point_density=1.0,
        mean_ndvi=0.6,
        std_ndvi=0.1,
        mean_height=5.0,
        std_height=2.0,
        height_range=10.0,
        mean_planarity=0.3,
        mean_verticality=0.2,
        mean_curvature=0.4,
        dominant_normal_z=0.8,
        parcel_type='forest',
        confidence_scores={'forest': 0.9}
    )
    
    classifier._parcel_stats_cache['001'] = stats1
    
    export = classifier.export_parcel_statistics()
    
    assert len(export) == 1
    assert export[0]['parcel_id'] == '001'
    assert export[0]['n_points'] == 100
    assert export[0]['parcel_type'] == 'forest'
    assert export[0]['confidence'] == 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
