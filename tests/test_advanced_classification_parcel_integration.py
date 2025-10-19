"""
Integration tests for parcel classification with advanced classifier.

Tests the integration of ParcelClassifier into AdvancedClassifier.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from ign_lidar.core.modules.advanced_classification import AdvancedClassifier


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_cadastre():
    """Create mock cadastre GeoDataFrame."""
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        parcels = gpd.GeoDataFrame({
            'id_parcelle': ['001', '002'],
            'geometry': [
                Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
            ]
        }, crs='EPSG:2154')
        return parcels
    except ImportError:
        pytest.skip("geopandas not available")


@pytest.fixture
def sample_points():
    """Create sample point cloud."""
    np.random.seed(42)
    n = 100
    
    points = np.column_stack([
        np.random.uniform(0, 20, n),
        np.random.uniform(0, 10, n),
        np.random.uniform(0, 10, n)
    ])
    
    return points


@pytest.fixture
def sample_features(sample_points):
    """Create sample features."""
    n = len(sample_points)
    
    features = {
        'ndvi': np.random.uniform(0.2, 0.7, n),
        'height': np.random.uniform(0, 10, n),
        'planarity': np.random.uniform(0.3, 0.9, n),
        'verticality': np.random.uniform(0.0, 0.6, n),
        'curvature': np.random.uniform(0.0, 0.5, n),
        'normals': np.random.randn(n, 3)
    }
    features['normals'] /= np.linalg.norm(features['normals'], axis=1, keepdims=True)
    
    return features


# ============================================================================
# Test Initialization
# ============================================================================

def test_advanced_classifier_init_without_parcel():
    """Test that advanced classifier initializes without parcel classification."""
    classifier = AdvancedClassifier(use_parcel_classification=False)
    
    assert classifier.use_parcel_classification is False
    assert classifier.parcel_classifier is None


def test_advanced_classifier_init_with_parcel():
    """Test that advanced classifier initializes with parcel classification."""
    classifier = AdvancedClassifier(
        use_parcel_classification=True,
        parcel_classification_config={'min_parcel_points': 50}
    )
    
    # Should either be enabled or disabled with warning if module not available
    if classifier.use_parcel_classification:
        assert classifier.parcel_classifier is not None
    else:
        assert classifier.parcel_classifier is None


# ============================================================================
# Test Classification Without Parcels
# ============================================================================

def test_classify_without_parcel_classification(sample_points, sample_features):
    """Test classification without parcel-based clustering."""
    classifier = AdvancedClassifier(
        use_parcel_classification=False,
        use_ground_truth=False,
        use_ndvi=True,
        use_geometric=True
    )
    
    labels = classifier.classify_points(
        points=sample_points,
        ndvi=sample_features['ndvi'],
        height=sample_features['height'],
        normals=sample_features['normals'],
        planarity=sample_features['planarity'],
        curvature=sample_features['curvature']
    )
    
    assert len(labels) == len(sample_points)
    assert labels.dtype == np.uint8
    # Should have some classified points
    assert np.any(labels != classifier.ASPRS_UNCLASSIFIED)


# ============================================================================
# Test Classification With Parcels
# ============================================================================

def test_classify_with_parcel_classification_no_cadastre(sample_points, sample_features):
    """Test that parcel classification skips gracefully without cadastre."""
    classifier = AdvancedClassifier(
        use_parcel_classification=True,
        use_ground_truth=False,
        use_ndvi=True,
        use_geometric=True
    )
    
    # No ground_truth_features provided
    labels = classifier.classify_points(
        points=sample_points,
        ndvi=sample_features['ndvi'],
        height=sample_features['height'],
        normals=sample_features['normals'],
        planarity=sample_features['planarity'],
        curvature=sample_features['curvature']
    )
    
    assert len(labels) == len(sample_points)
    assert labels.dtype == np.uint8


def test_classify_with_parcel_classification_with_cadastre(
    sample_points, sample_features, mock_cadastre
):
    """Test classification with parcel-based clustering."""
    classifier = AdvancedClassifier(
        use_parcel_classification=True,
        use_ground_truth=False,
        use_ndvi=True,
        use_geometric=True,
        parcel_classification_config={'min_parcel_points': 10}
    )
    
    # Skip test if parcel classifier not available
    if not classifier.use_parcel_classification:
        pytest.skip("Parcel classifier not available")
    
    ground_truth_features = {
        'cadastre': mock_cadastre
    }
    
    labels = classifier.classify_points(
        points=sample_points,
        ground_truth_features=ground_truth_features,
        ndvi=sample_features['ndvi'],
        height=sample_features['height'],
        normals=sample_features['normals'],
        planarity=sample_features['planarity'],
        verticality=sample_features['verticality'],
        curvature=sample_features['curvature']
    )
    
    assert len(labels) == len(sample_points)
    assert labels.dtype == np.uint8
    # Should have some classified points
    assert np.any(labels != classifier.ASPRS_UNCLASSIFIED)


def test_classify_with_parcel_and_ground_truth(
    sample_points, sample_features, mock_cadastre
):
    """Test that parcel classification and ground truth work together."""
    classifier = AdvancedClassifier(
        use_parcel_classification=True,
        use_ground_truth=True,  # Both enabled
        use_ndvi=True,
        use_geometric=True
    )
    
    # Skip test if parcel classifier not available
    if not classifier.use_parcel_classification:
        pytest.skip("Parcel classifier not available")
    
    ground_truth_features = {
        'cadastre': mock_cadastre,
        # Could add more ground truth here (buildings, roads, etc.)
    }
    
    labels = classifier.classify_points(
        points=sample_points,
        ground_truth_features=ground_truth_features,
        ndvi=sample_features['ndvi'],
        height=sample_features['height'],
        normals=sample_features['normals'],
        planarity=sample_features['planarity'],
        verticality=sample_features['verticality'],
        curvature=sample_features['curvature']
    )
    
    assert len(labels) == len(sample_points)
    assert labels.dtype == np.uint8


# ============================================================================
# Test Configuration Validation
# ============================================================================

def test_parcel_config_validation():
    """Test that parcel configuration is properly validated."""
    # Valid configuration
    config = {
        'min_parcel_points': 20,
        'parcel_confidence_threshold': 0.7,
        'refine_points': True
    }
    
    classifier = AdvancedClassifier(
        use_parcel_classification=True,
        parcel_classification_config=config
    )
    
    if classifier.use_parcel_classification:
        assert classifier.parcel_classifier is not None


def test_parcel_classification_error_handling(sample_points, sample_features):
    """Test that errors in parcel classification are handled gracefully."""
    classifier = AdvancedClassifier(
        use_parcel_classification=True,
        use_ground_truth=False,
        use_ndvi=True
    )
    
    # Skip test if parcel classifier not available
    if not classifier.use_parcel_classification:
        pytest.skip("Parcel classifier not available")
    
    # Mock cadastre that will cause an error
    bad_cadastre = Mock()
    bad_cadastre.__len__ = Mock(return_value=1)
    bad_cadastre.get = Mock(side_effect=Exception("Mock error"))
    
    ground_truth_features = {
        'cadastre': bad_cadastre
    }
    
    # Should not crash, just log warning and continue
    labels = classifier.classify_points(
        points=sample_points,
        ground_truth_features=ground_truth_features,
        ndvi=sample_features['ndvi'],
        height=sample_features['height']
    )
    
    assert len(labels) == len(sample_points)


# ============================================================================
# Test Parcel Stats Export
# ============================================================================

def test_parcel_stats_export(sample_points, sample_features, mock_cadastre):
    """Test that parcel statistics can be exported."""
    classifier = AdvancedClassifier(
        use_parcel_classification=True,
        use_ground_truth=False
    )
    
    # Skip test if parcel classifier not available
    if not classifier.use_parcel_classification:
        pytest.skip("Parcel classifier not available")
    
    ground_truth_features = {
        'cadastre': mock_cadastre
    }
    
    # Run classification
    labels = classifier.classify_points(
        points=sample_points,
        ground_truth_features=ground_truth_features,
        ndvi=sample_features['ndvi'],
        height=sample_features['height'],
        planarity=sample_features['planarity'],
        verticality=sample_features['verticality'],
        curvature=sample_features['curvature'],
        normals=sample_features['normals']
    )
    
    # Check if we can export statistics
    if hasattr(classifier.parcel_classifier, 'export_parcel_statistics'):
        stats = classifier.parcel_classifier.export_parcel_statistics()
        assert isinstance(stats, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
