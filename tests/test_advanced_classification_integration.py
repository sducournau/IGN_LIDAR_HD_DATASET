"""
Integration test for feature validation in advanced classification.

Tests that the AdvancedClassifier correctly integrates with FeatureValidator.
"""

import pytest
import numpy as np
from ign_lidar.core.modules.advanced_classification import AdvancedClassifier


class TestFeatureValidationIntegration:
    """Test AdvancedClassifier with feature validation."""
    
    @pytest.fixture
    def classifier_without_validation(self):
        """Create classifier without feature validation."""
        return AdvancedClassifier(
            use_ground_truth=False,
            use_ndvi=True,
            use_geometric=True,
            use_feature_validation=False
        )
    
    @pytest.fixture
    def classifier_with_validation(self):
        """Create classifier with feature validation."""
        return AdvancedClassifier(
            use_ground_truth=False,
            use_ndvi=True,
            use_geometric=True,
            use_feature_validation=True
        )
    
    @pytest.fixture
    def sample_points(self):
        """Create sample point cloud."""
        n_points = 100
        
        # Create synthetic point cloud
        points = np.column_stack([
            np.random.uniform(0, 100, n_points),  # X
            np.random.uniform(0, 100, n_points),  # Y
            np.random.uniform(0, 10, n_points)    # Z
        ])
        
        return points
    
    @pytest.fixture
    def vegetation_features(self):
        """Create vegetation-like features."""
        n_points = 100
        return {
            'ndvi': np.random.uniform(0.5, 0.8, n_points),  # High NDVI
            'height': np.random.uniform(3.0, 10.0, n_points),  # Tree height
            'curvature': np.random.uniform(0.25, 0.45, n_points),  # High curvature
            'planarity': np.random.uniform(0.2, 0.6, n_points),  # Low planarity
            'normals': np.random.randn(n_points, 3),
            'intensity': np.random.uniform(0, 255, n_points)
        }
    
    def test_classifier_initialization_without_validation(self, classifier_without_validation):
        """Test that classifier initializes without validation."""
        assert classifier_without_validation.use_feature_validation is False
        assert classifier_without_validation.feature_validator is None
    
    def test_classifier_initialization_with_validation(self, classifier_with_validation):
        """Test that classifier initializes with validation."""
        assert classifier_with_validation.use_feature_validation is True
        assert classifier_with_validation.feature_validator is not None
    
    def test_multi_level_ndvi_classification(
        self, classifier_with_validation, sample_points, vegetation_features
    ):
        """Test multi-level NDVI classification."""
        # Classify points
        labels = classifier_with_validation.classify_points(
            points=sample_points,
            ndvi=vegetation_features['ndvi'],
            height=vegetation_features['height'],
            curvature=vegetation_features['curvature'],
            planarity=vegetation_features['planarity']
        )
        
        # Should classify as vegetation (3, 4, or 5)
        vegetation_mask = (labels >= 3) & (labels <= 5)
        assert np.sum(vegetation_mask) > 50  # At least half should be vegetation
    
    def test_dense_forest_classification(self, classifier_with_validation, sample_points):
        """Test dense forest classification (NDVI >= 0.60)."""
        features = {
            'ndvi': np.full(100, 0.70),  # Dense forest NDVI
            'height': np.random.uniform(5.0, 15.0, 100),  # Tree height
            'curvature': np.random.uniform(0.30, 0.50, 100),  # High curvature
            'planarity': np.random.uniform(0.2, 0.5, 100)  # Low planarity
        }
        
        labels = classifier_with_validation.classify_points(
            points=sample_points,
            ndvi=features['ndvi'],
            height=features['height'],
            curvature=features['curvature'],
            planarity=features['planarity']
        )
        
        # Should be mostly high vegetation (5)
        assert np.sum(labels == 5) > 70
    
    def test_sparse_vegetation_classification(self, classifier_with_validation, sample_points):
        """Test sparse vegetation classification (NDVI >= 0.20)."""
        features = {
            'ndvi': np.full(100, 0.25),  # Sparse vegetation NDVI
            'height': np.random.uniform(0.2, 0.8, 100),  # Low height
            'curvature': np.random.uniform(0.10, 0.20, 100),
            'planarity': np.random.uniform(0.5, 0.8, 100)
        }
        
        labels = classifier_with_validation.classify_points(
            points=sample_points,
            ndvi=features['ndvi'],
            height=features['height'],
            curvature=features['curvature'],
            planarity=features['planarity']
        )
        
        # Should be mostly low vegetation (3)
        assert np.sum(labels == 3) > 60
    
    def test_non_vegetation_classification(self, classifier_with_validation, sample_points):
        """Test non-vegetation classification (low NDVI)."""
        features = {
            'ndvi': np.full(100, 0.10),  # Very low NDVI
            'height': np.random.uniform(0.0, 1.0, 100),
            'curvature': np.random.uniform(0.0, 0.05, 100),  # Low curvature
            'planarity': np.random.uniform(0.85, 0.95, 100)  # High planarity
        }
        
        labels = classifier_with_validation.classify_points(
            points=sample_points,
            ndvi=features['ndvi'],
            height=features['height'],
            curvature=features['curvature'],
            planarity=features['planarity']
        )
        
        # Should NOT be vegetation
        vegetation_mask = (labels >= 3) & (labels <= 5)
        assert np.sum(vegetation_mask) < 20  # Less than 20% vegetation
    
    def test_feature_validation_logging(self, classifier_with_validation, sample_points, caplog):
        """Test that feature validation logs information."""
        import logging
        caplog.set_level(logging.INFO)
        
        features = {
            'ndvi': np.random.uniform(0.3, 0.7, 100),
            'height': np.random.uniform(1.0, 8.0, 100),
            'curvature': np.random.uniform(0.1, 0.4, 100),
            'planarity': np.random.uniform(0.3, 0.7, 100)
        }
        
        labels = classifier_with_validation.classify_points(
            points=sample_points,
            **features
        )
        
        # Check logs
        log_text = caplog.text
        assert "multi-level" in log_text.lower() or "NDVI" in log_text
    
    def test_backward_compatibility(self, classifier_without_validation, sample_points):
        """Test that old behavior works without validation."""
        features = {
            'ndvi': np.random.uniform(0.4, 0.7, 100),
            'height': np.random.uniform(1.0, 10.0, 100)
        }
        
        # Should work without errors
        labels = classifier_without_validation.classify_points(
            points=sample_points,
            **features
        )
        
        assert len(labels) == 100
        assert labels.dtype == np.uint8


class TestMultiLevelNDVI:
    """Test multi-level NDVI thresholds."""
    
    def test_ndvi_thresholds_coverage(self):
        """Test that NDVI thresholds cover expected ranges."""
        classifier = AdvancedClassifier(
            use_feature_validation=False,  # Disable validation for pure NDVI testing
            use_ground_truth=False
        )
        
        # Create test data across NDVI range
        n_points = 50
        points = np.zeros((n_points, 3))
        
        # Test each threshold level
        ndvi_levels = [0.70, 0.55, 0.45, 0.35, 0.25, 0.10]
        expected_veg = [True, True, True, True, True, False]
        
        for ndvi_val, should_be_veg in zip(ndvi_levels, expected_veg):
            ndvi = np.full(n_points, ndvi_val)
            height = np.full(n_points, 1.5)  # Medium height
            curvature = np.full(n_points, 0.30 if ndvi_val >= 0.2 else 0.05)
            planarity = np.full(n_points, 0.40 if ndvi_val >= 0.2 else 0.90)
            
            labels = classifier.classify_points(
                points=points,
                ndvi=ndvi,
                height=height,
                curvature=curvature,
                planarity=planarity
            )
            
            is_veg = np.any((labels >= 3) & (labels <= 5))
            assert is_veg == should_be_veg, f"NDVI {ndvi_val} failed: got {is_veg}, expected {should_be_veg}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--log-cli-level=INFO'])
