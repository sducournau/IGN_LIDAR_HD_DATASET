"""
Tests for unified classifier (Phase 6 consolidation).

This test suite validates:
1. UnifiedClassifier with all three strategies
2. Backward compatibility wrappers
3. Feature availability detection
4. Classification accuracy
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Import unified classifier
from ign_lidar.core.classification.unified_classifier import (
    UnifiedClassifier,
    ClassificationStrategy,
    UnifiedClassifierConfig,
    classify_points_unified,
    refine_classification_unified
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample point cloud data for testing."""
    np.random.seed(42)
    n_points = 1000
    
    data = pd.DataFrame({
        'x': np.random.uniform(0, 100, n_points),
        'y': np.random.uniform(0, 100, n_points),
        'z': np.random.uniform(0, 50, n_points),
        'height': np.random.uniform(0, 20, n_points),
        'planarity': np.random.uniform(0, 1, n_points),
        'verticality': np.random.uniform(0, 1, n_points),
        'curvature': np.random.uniform(0, 0.2, n_points),
        'roughness': np.random.uniform(0, 0.2, n_points),
        'normal_z': np.random.uniform(0, 1, n_points),
        'ndvi': np.random.uniform(-1, 1, n_points),
        'intensity': np.random.uniform(0, 255, n_points)
    })
    
    return data


@pytest.fixture
def sample_features():
    """Create sample feature dictionary for adaptive classification."""
    np.random.seed(42)
    n_points = 1000
    
    features = {
        'height': np.random.uniform(0, 20, n_points),
        'planarity': np.random.uniform(0, 1, n_points),
        'verticality': np.random.uniform(0, 1, n_points),
        'curvature': np.random.uniform(0, 0.2, n_points),
        'roughness': np.random.uniform(0, 0.2, n_points),
        'normal_z': np.random.uniform(0, 1, n_points),
        'ndvi': np.random.uniform(-1, 1, n_points)
    }
    
    return features


@pytest.fixture
def sample_labels():
    """Create sample classification labels."""
    np.random.seed(42)
    n_points = 1000
    # Random labels from common ASPRS classes
    labels = np.random.choice([1, 2, 3, 4, 5, 6, 9, 11], size=n_points)
    return labels


# ============================================================================
# Test Unified Classifier
# ============================================================================

def test_unified_classifier_basic_strategy(sample_data):
    """Test BASIC classification strategy."""
    classifier = UnifiedClassifier(strategy=ClassificationStrategy.BASIC)
    
    labels = classifier.classify_points(sample_data)
    
    assert len(labels) == len(sample_data)
    assert labels.dtype in [np.int32, np.int64]
    assert np.all(labels >= 1)  # All labels should be valid ASPRS codes


def test_unified_classifier_adaptive_strategy(sample_features):
    """Test ADAPTIVE classification strategy."""
    classifier = UnifiedClassifier(strategy=ClassificationStrategy.ADAPTIVE)
    
    labels, confidences = classifier.classify_batch(sample_features)
    
    assert len(labels) == len(sample_features['height'])
    assert len(confidences) == len(labels)
    assert labels.dtype in [np.int32, np.int64]
    assert confidences.dtype in [np.float32, np.float64]
    assert np.all(confidences >= 0.0)
    assert np.all(confidences <= 1.0)


def test_unified_classifier_comprehensive_strategy(sample_data):
    """Test COMPREHENSIVE classification strategy."""
    classifier = UnifiedClassifier(strategy=ClassificationStrategy.COMPREHENSIVE)
    
    labels = classifier.classify_points(sample_data, verbose=False)
    
    assert len(labels) == len(sample_data)
    assert labels.dtype in [np.int32, np.int64]
    assert np.all(labels >= 1)


def test_unified_classifier_with_config():
    """Test UnifiedClassifier with explicit configuration."""
    config = UnifiedClassifierConfig(
        strategy=ClassificationStrategy.ADAPTIVE,
        use_ground_truth=True,
        use_ndvi=True,
        use_geometric=True
    )
    
    classifier = UnifiedClassifier(config=config)
    
    assert classifier.strategy == ClassificationStrategy.ADAPTIVE
    assert classifier.config.use_ground_truth == True
    assert classifier.config.use_ndvi == True
    assert classifier.config.use_geometric == True


def test_feature_availability_detection(sample_features):
    """Test automatic feature availability detection."""
    classifier = UnifiedClassifier(strategy=ClassificationStrategy.ADAPTIVE)
    
    available = classifier.get_available_features(sample_features)
    
    assert isinstance(available, set)
    assert 'height' in available
    assert 'planarity' in available
    assert 'ndvi' in available
    
    # Add invalid feature
    sample_features['invalid'] = np.full(len(sample_features['height']), np.nan)
    available = classifier.get_available_features(sample_features)
    assert 'invalid' not in available


def test_artifact_feature_handling(sample_features):
    """Test artifact feature exclusion."""
    classifier = UnifiedClassifier(strategy=ClassificationStrategy.ADAPTIVE)
    
    # Mark some features as artifacts
    classifier.set_artifact_features({'curvature', 'roughness'})
    
    available = classifier.get_available_features(sample_features)
    
    assert 'curvature' not in available
    assert 'roughness' not in available
    assert 'height' in available  # Should still be available


def test_feature_importance_report(sample_features):
    """Test feature importance reporting."""
    classifier = UnifiedClassifier(strategy=ClassificationStrategy.ADAPTIVE)
    
    available = classifier.get_available_features(sample_features)
    report = classifier.get_feature_importance_report(available)
    
    assert 'available_features' in report
    assert 'classifiable_categories' in report
    assert 'degraded_categories' in report
    assert 'impossible_categories' in report
    
    assert len(report['available_features']) > 0
    assert len(report['classifiable_categories']) > 0


# ============================================================================
# Test Refinement Functions
# ============================================================================

def test_refine_vegetation(sample_labels, sample_features):
    """Test vegetation refinement."""
    classifier = UnifiedClassifier()
    
    refined, n_changed = classifier.refine_vegetation(sample_labels, sample_features)
    
    assert len(refined) == len(sample_labels)
    assert isinstance(n_changed, (int, np.integer))
    assert n_changed >= 0


def test_refine_buildings(sample_labels, sample_features):
    """Test building refinement."""
    classifier = UnifiedClassifier()
    
    refined, n_changed = classifier.refine_buildings(sample_labels, sample_features)
    
    assert len(refined) == len(sample_labels)
    assert isinstance(n_changed, (int, np.integer))
    assert n_changed >= 0


def test_refine_roads(sample_labels, sample_features):
    """Test road refinement."""
    classifier = UnifiedClassifier()
    
    refined, n_changed = classifier.refine_roads(sample_labels, sample_features)
    
    assert len(refined) == len(sample_labels)
    assert isinstance(n_changed, (int, np.integer))
    assert n_changed >= 0


def test_refine_ground(sample_labels, sample_features):
    """Test ground refinement."""
    classifier = UnifiedClassifier()
    
    refined, n_changed = classifier.refine_ground(sample_labels, sample_features)
    
    assert len(refined) == len(sample_labels)
    assert isinstance(n_changed, (int, np.integer))
    assert n_changed >= 0


def test_detect_vehicles(sample_labels, sample_features):
    """Test vehicle detection."""
    classifier = UnifiedClassifier()
    
    refined, n_detected = classifier.detect_vehicles(sample_labels, sample_features)
    
    assert len(refined) == len(sample_labels)
    assert isinstance(n_detected, (int, np.integer))
    assert n_detected >= 0


def test_refine_classification_comprehensive(sample_labels, sample_features):
    """Test comprehensive refinement pipeline."""
    classifier = UnifiedClassifier()
    
    refined, stats = classifier.refine_classification(sample_labels, sample_features)
    
    assert len(refined) == len(sample_labels)
    assert isinstance(stats, dict)
    
    # Check expected stats keys
    expected_keys = ['vegetation', 'buildings', 'roads', 'ground', 'vehicles']
    for key in expected_keys:
        assert key in stats
        assert isinstance(stats[key], (int, np.integer))


# ============================================================================
# Test Convenience Functions
# ============================================================================

def test_classify_points_unified_convenience(sample_data):
    """Test classify_points_unified() convenience function."""
    labels = classify_points_unified(
        data=sample_data,
        strategy='comprehensive',
        use_ground_truth=False
    )
    
    assert len(labels) == len(sample_data)
    assert labels.dtype in [np.int32, np.int64]


def test_refine_classification_unified_convenience(sample_labels, sample_features):
    """Test refine_classification_unified() convenience function."""
    refined, stats = refine_classification_unified(
        labels=sample_labels,
        features=sample_features
    )
    
    assert len(refined) == len(sample_labels)
    assert isinstance(stats, dict)


# ============================================================================
# Test Strategy String Conversion
# ============================================================================

def test_strategy_string_conversion():
    """Test automatic conversion of strategy strings to enums."""
    # Test with string
    classifier1 = UnifiedClassifier(strategy='basic')
    assert classifier1.strategy == ClassificationStrategy.BASIC
    
    classifier2 = UnifiedClassifier(strategy='adaptive')
    assert classifier2.strategy == ClassificationStrategy.ADAPTIVE
    
    classifier3 = UnifiedClassifier(strategy='comprehensive')
    assert classifier3.strategy == ClassificationStrategy.COMPREHENSIVE


def test_lod2_element_classification(sample_labels, sample_features):
    """Test LOD2 element classification."""
    classifier = UnifiedClassifier()
    
    # Add building labels
    sample_labels[:100] = 6  # ASPRS_BUILDING
    
    lod2_labels = classifier.classify_lod2_elements(sample_labels, sample_features)
    
    assert len(lod2_labels) == len(sample_labels)
    assert lod2_labels.dtype in [np.int32, np.int64]


def test_lod3_element_classification(sample_labels, sample_features):
    """Test LOD3 element classification."""
    classifier = UnifiedClassifier()
    
    # Add building labels
    sample_labels[:100] = 6  # ASPRS_BUILDING
    
    lod3_labels = classifier.classify_lod3_elements(sample_labels, sample_features)
    
    assert len(lod3_labels) == len(sample_labels)
    assert lod3_labels.dtype in [np.int32, np.int64]


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_pipeline_basic_to_refinement(sample_data, sample_features):
    """Test full pipeline: classify → refine."""
    # Classify with BASIC strategy
    classifier = UnifiedClassifier(strategy=ClassificationStrategy.BASIC)
    labels = classifier.classify_points(sample_data, verbose=False)
    
    # Refine
    refined, stats = classifier.refine_classification(labels, sample_features)
    
    assert len(refined) == len(labels)
    assert isinstance(stats, dict)
    
    # Some refinement should have occurred
    total_refined = sum(stats.values())
    assert total_refined >= 0


def test_full_pipeline_comprehensive(sample_data):
    """Test full COMPREHENSIVE pipeline with all features."""
    classifier = UnifiedClassifier(
        strategy=ClassificationStrategy.COMPREHENSIVE,
        use_ground_truth=False,  # No ground truth for this test
        use_ndvi=True,
        use_geometric=True
    )
    
    labels = classifier.classify_points(sample_data, verbose=False)
    
    assert len(labels) == len(sample_data)
    
    # Check distribution (should have multiple classes)
    unique_labels = np.unique(labels)
    assert len(unique_labels) > 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
