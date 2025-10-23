"""
Comprehensive tests for rules validation module.

Tests validation utilities for feature checking, quality assessment,
and point cloud validation.
"""

import pytest
import numpy as np
from typing import Dict

from ign_lidar.core.classification.rules.validation import (
    FeatureRequirements,
    validate_features,
    validate_feature_shape,
    check_feature_quality,
    check_all_feature_quality,
    validate_feature_ranges,
    validate_points_array,
    get_feature_statistics
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_features():
    """Create sample feature dictionary"""
    return {
        'height': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'intensity': np.array([100, 150, 200, 250, 300]),
        'planarity': np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    }


@pytest.fixture
def features_with_nan():
    """Features containing NaN values"""
    return {
        'height': np.array([1.0, np.nan, 3.0, 4.0, 5.0]),
        'intensity': np.array([100, 150, np.nan, 250, 300])
    }


@pytest.fixture
def features_with_inf():
    """Features containing infinite values"""
    return {
        'height': np.array([1.0, 2.0, np.inf, 4.0, 5.0]),
        'intensity': np.array([100, -np.inf, 200, 250, 300])
    }


@pytest.fixture
def valid_points():
    """Valid point cloud array"""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ])


# ============================================================================
# Test FeatureRequirements
# ============================================================================

class TestFeatureRequirements:
    """Test FeatureRequirements dataclass"""
    
    def test_basic_creation(self):
        """Test creating FeatureRequirements with basic parameters"""
        req = FeatureRequirements(
            required={'height', 'intensity'},
            optional={'planarity'},
            min_quality=0.8,
            allow_nan=False
        )
        
        assert req.required == {'height', 'intensity'}
        assert req.optional == {'planarity'}
        assert req.min_quality == 0.8
        assert req.allow_nan is False
    
    def test_overlap_handling(self):
        """Test that overlapping required/optional features are handled"""
        req = FeatureRequirements(
            required={'height', 'intensity'},
            optional={'height', 'planarity'}  # height in both
        )
        
        # height should be in required, not optional
        assert 'height' in req.required
        assert 'height' not in req.optional
        assert 'planarity' in req.optional
    
    def test_list_to_set_conversion(self):
        """Test automatic conversion of lists to sets"""
        req = FeatureRequirements(
            required=['height', 'intensity'],  # list instead of set
            optional=['planarity']
        )
        
        assert isinstance(req.required, set)
        assert isinstance(req.optional, set)
        assert req.required == {'height', 'intensity'}


# ============================================================================
# Test validate_features
# ============================================================================

class TestValidateFeatures:
    """Test validate_features function"""
    
    def test_valid_features_pass(self, sample_features):
        """Test that valid features pass validation"""
        req = FeatureRequirements(
            required={'height', 'intensity'},
            optional={'planarity'}
        )
        
        # Should not raise any exception
        validate_features(sample_features, req)
    
    def test_missing_required_raises(self, sample_features):
        """Test that missing required features raise ValueError"""
        req = FeatureRequirements(
            required={'height', 'intensity', 'missing_feature'}
        )
        
        with pytest.raises(ValueError, match="Missing required features"):
            validate_features(sample_features, req)
    
    def test_nan_values_not_allowed(self, features_with_nan):
        """Test that NaN values raise error when not allowed"""
        req = FeatureRequirements(
            required={'height'},
            allow_nan=False
        )
        
        with pytest.raises(ValueError, match="contains .* NaN values"):
            validate_features(features_with_nan, req)
    
    def test_nan_values_allowed(self, features_with_nan):
        """Test that NaN values pass when allowed"""
        req = FeatureRequirements(
            required={'height'},
            allow_nan=True
        )
        
        # Should not raise
        validate_features(features_with_nan, req)
    
    def test_infinite_values_raise(self, features_with_inf):
        """Test that infinite values always raise error"""
        req = FeatureRequirements(
            required={'height'}
        )
        
        with pytest.raises(ValueError, match="contains .* infinite values"):
            validate_features(features_with_inf, req)
    
    def test_shape_mismatch_raises(self):
        """Test that mismatched feature shapes raise error"""
        features = {
            'height': np.array([1.0, 2.0, 3.0]),
            'intensity': np.array([100, 150])  # Different length!
        }
        req = FeatureRequirements(required={'height', 'intensity'})
        
        with pytest.raises(ValueError, match="has .* values, expected"):
            validate_features(features, req)
    
    def test_n_points_inference(self, sample_features):
        """Test that n_points is correctly inferred"""
        req = FeatureRequirements(required={'height'})
        
        # Should infer n_points=5 from sample_features
        validate_features(sample_features, req, n_points=None)
    
    def test_empty_features_raises(self):
        """Test that empty features dict raises error"""
        req = FeatureRequirements(required=set())
        
        with pytest.raises(ValueError, match="Cannot infer n_points"):
            validate_features({}, req, n_points=None)


# ============================================================================
# Test validate_feature_shape
# ============================================================================

class TestValidateFeatureShape:
    """Test validate_feature_shape function"""
    
    def test_correct_1d_shape(self, sample_features):
        """Test validation of correct 1D shape"""
        # Should not raise
        validate_feature_shape(sample_features, (5,))
    
    def test_incorrect_1d_shape(self, sample_features):
        """Test that incorrect 1D shape raises error"""
        with pytest.raises(ValueError, match="has shape .*, expected"):
            validate_feature_shape(sample_features, (10,))
    
    def test_2d_shape_validation(self):
        """Test validation of 2D arrays"""
        features = {
            'normals': np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        }
        
        # Should pass
        validate_feature_shape(features, (3, 3))
        
        # Should fail
        with pytest.raises(ValueError):
            validate_feature_shape(features, (3, 4))
    
    def test_specific_features_only(self, sample_features):
        """Test validation of specific features only"""
        # Only check 'height', ignore others
        validate_feature_shape(
            sample_features,
            (5,),
            feature_names=['height']
        )
    
    def test_missing_feature_ignored(self, sample_features):
        """Test that missing features are silently ignored"""
        # Should not raise even though 'missing' doesn't exist
        validate_feature_shape(
            sample_features,
            (5,),
            feature_names=['missing']
        )


# ============================================================================
# Test check_feature_quality
# ============================================================================

class TestCheckFeatureQuality:
    """Test check_feature_quality function"""
    
    def test_perfect_quality(self, sample_features):
        """Test feature with 100% valid values"""
        quality = check_feature_quality(sample_features, 'height')
        assert quality == 1.0
    
    def test_partial_quality(self):
        """Test feature with partial quality"""
        features = {
            'height': np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        }
        
        quality = check_feature_quality(features, 'height', min_quality=0.0)
        assert quality == 0.6  # 3/5 valid values
    
    def test_below_threshold_raises(self):
        """Test that quality below threshold raises error"""
        features = {
            'height': np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        }
        
        with pytest.raises(ValueError, match="quality .* is below minimum"):
            check_feature_quality(features, 'height', min_quality=0.5)
    
    def test_infinite_values_affect_quality(self):
        """Test that infinite values reduce quality"""
        features = {
            'height': np.array([1.0, np.inf, 3.0, -np.inf, 5.0])
        }
        
        quality = check_feature_quality(features, 'height', min_quality=0.0)
        assert quality == 0.6  # 3/5 finite values
    
    def test_missing_feature_raises(self, sample_features):
        """Test that missing feature raises error"""
        with pytest.raises(ValueError, match="not found"):
            check_feature_quality(sample_features, 'missing_feature')
    
    def test_empty_array(self):
        """Test quality of empty array"""
        features = {'height': np.array([])}
        quality = check_feature_quality(features, 'height', min_quality=0.0)
        assert quality == 0.0


# ============================================================================
# Test check_all_feature_quality
# ============================================================================

class TestCheckAllFeatureQuality:
    """Test check_all_feature_quality function"""
    
    def test_all_perfect_quality(self, sample_features):
        """Test when all features have perfect quality"""
        quality_dict = check_all_feature_quality(sample_features, min_quality=0.0)
        
        assert len(quality_dict) == 3
        assert all(q == 1.0 for q in quality_dict.values())
    
    def test_mixed_quality(self):
        """Test features with varying quality"""
        features = {
            'height': np.array([1.0, 2.0, 3.0]),
            'intensity': np.array([100, np.nan, 200])
        }
        
        quality_dict = check_all_feature_quality(features, min_quality=0.0)
        
        assert quality_dict['height'] == 1.0
        assert abs(quality_dict['intensity'] - 2/3) < 0.01
    
    def test_any_below_threshold_raises(self):
        """Test that any feature below threshold raises error"""
        features = {
            'height': np.array([1.0, 2.0, 3.0]),
            'intensity': np.array([np.nan, np.nan, 200])
        }
        
        with pytest.raises(ValueError):
            check_all_feature_quality(features, min_quality=0.5)


# ============================================================================
# Test validate_feature_ranges
# ============================================================================

class TestValidateFeatureRanges:
    """Test validate_feature_ranges function"""
    
    def test_values_in_range_pass(self, sample_features):
        """Test that values in range pass validation"""
        ranges = {
            'height': (0.0, 10.0),
            'intensity': (0, 500)
        }
        
        # Should not raise in non-strict mode
        validate_feature_ranges(sample_features, ranges, strict=False)
    
    def test_values_below_min_strict(self):
        """Test that values below minimum raise in strict mode"""
        features = {'height': np.array([1.0, 2.0, -5.0])}
        ranges = {'height': (0.0, 10.0)}
        
        with pytest.raises(ValueError, match="below minimum"):
            validate_feature_ranges(features, ranges, strict=True)
    
    def test_values_above_max_strict(self):
        """Test that values above maximum raise in strict mode"""
        features = {'height': np.array([1.0, 2.0, 15.0])}
        ranges = {'height': (0.0, 10.0)}
        
        with pytest.raises(ValueError, match="above maximum"):
            validate_feature_ranges(features, ranges, strict=True)
    
    def test_non_strict_mode_warns_only(self, sample_features):
        """Test that non-strict mode only warns"""
        ranges = {'height': (0.0, 3.0)}  # max too low
        
        # Should not raise in non-strict mode
        validate_feature_ranges(sample_features, ranges, strict=False)
    
    def test_missing_feature_ignored(self, sample_features):
        """Test that missing features in ranges are ignored"""
        ranges = {'missing_feature': (0.0, 10.0)}
        
        # Should not raise
        validate_feature_ranges(sample_features, ranges, strict=True)


# ============================================================================
# Test validate_points_array
# ============================================================================

class TestValidatePointsArray:
    """Test validate_points_array function"""
    
    def test_valid_points_pass(self, valid_points):
        """Test that valid points array passes"""
        validate_points_array(valid_points, min_points=1, expected_dims=3)
    
    def test_not_numpy_array_raises(self):
        """Test that non-numpy array raises error"""
        with pytest.raises(ValueError, match="must be numpy array"):
            validate_points_array([[0, 0, 0], [1, 1, 1]])
    
    def test_not_2d_raises(self):
        """Test that non-2D array raises error"""
        points = np.array([0, 1, 2, 3, 4])  # 1D
        
        with pytest.raises(ValueError, match="must be 2D array"):
            validate_points_array(points)
    
    def test_too_few_points_raises(self, valid_points):
        """Test that too few points raises error"""
        with pytest.raises(ValueError, match="Need at least"):
            validate_points_array(valid_points, min_points=10)
    
    def test_wrong_dimensions_raises(self, valid_points):
        """Test that wrong number of dimensions raises error"""
        with pytest.raises(ValueError, match="Expected .* dimensions"):
            validate_points_array(valid_points, expected_dims=2)
    
    def test_nan_values_raise(self):
        """Test that NaN values in points raise error"""
        points = np.array([[0, 0, 0], [1, np.nan, 1]])
        
        with pytest.raises(ValueError, match="contains NaN"):
            validate_points_array(points)
    
    def test_infinite_values_raise(self):
        """Test that infinite values in points raise error"""
        points = np.array([[0, 0, 0], [np.inf, 1, 1]])
        
        with pytest.raises(ValueError, match="contains infinite"):
            validate_points_array(points)


# ============================================================================
# Test get_feature_statistics
# ============================================================================

class TestGetFeatureStatistics:
    """Test get_feature_statistics function"""
    
    def test_basic_statistics(self, sample_features):
        """Test computation of basic statistics"""
        stats = get_feature_statistics(sample_features)
        
        assert len(stats) == 3
        assert 'height' in stats
        assert 'intensity' in stats
        assert 'planarity' in stats
        
        # Check height statistics
        height_stats = stats['height']
        assert height_stats['mean'] == 3.0
        assert height_stats['min'] == 1.0
        assert height_stats['max'] == 5.0
        assert height_stats['quality'] == 1.0
        assert height_stats['n_valid'] == 5
        assert height_stats['n_total'] == 5
    
    def test_statistics_with_nan(self):
        """Test statistics computation with NaN values"""
        features = {
            'height': np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        }
        
        stats = get_feature_statistics(features)
        height_stats = stats['height']
        
        # Statistics should be computed only on valid values
        assert height_stats['mean'] == 3.25  # (1+3+4+5)/4
        assert height_stats['quality'] == 0.8  # 4/5 valid
        assert height_stats['n_valid'] == 4
        assert height_stats['n_total'] == 5
    
    def test_all_invalid_values(self):
        """Test statistics when all values are invalid"""
        features = {
            'height': np.array([np.nan, np.nan, np.inf])
        }
        
        stats = get_feature_statistics(features)
        height_stats = stats['height']
        
        assert height_stats['mean'] == 0.0
        assert height_stats['std'] == 0.0
        assert height_stats['quality'] == 0.0
        assert height_stats['n_valid'] == 0
        assert height_stats['n_total'] == 3
    
    def test_median_computation(self, sample_features):
        """Test median computation"""
        stats = get_feature_statistics(sample_features)
        
        assert stats['height']['median'] == 3.0
        assert stats['intensity']['median'] == 200.0
    
    def test_empty_features(self):
        """Test with empty features dictionary"""
        stats = get_feature_statistics({})
        assert stats == {}


# ============================================================================
# Integration Tests
# ============================================================================

class TestValidationIntegration:
    """Test integration of validation functions"""
    
    def test_complete_validation_workflow(self):
        """Test complete validation workflow"""
        # Create features
        features = {
            'height': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'intensity': np.array([100, 150, 200, 250, 300]),
            'planarity': np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        }
        
        # Define requirements
        req = FeatureRequirements(
            required={'height', 'intensity'},
            optional={'planarity'},
            min_quality=0.8
        )
        
        # Validate features
        validate_features(features, req)
        
        # Check quality
        quality = check_all_feature_quality(features, min_quality=0.8)
        assert all(q >= 0.8 for q in quality.values())
        
        # Validate ranges
        ranges = {
            'height': (0.0, 10.0),
            'intensity': (0, 500)
        }
        validate_feature_ranges(features, ranges, strict=True)
        
        # Get statistics
        stats = get_feature_statistics(features)
        assert len(stats) == 3
    
    def test_validation_failure_propagation(self):
        """Test that validation failures propagate correctly"""
        features = {
            'height': np.array([1.0, np.nan, 3.0]),
            'intensity': np.array([100, 150, 200, 250])  # Wrong length!
        }
        
        req = FeatureRequirements(
            required={'height', 'intensity'},
            allow_nan=False
        )
        
        # Should fail on NaN or length mismatch
        with pytest.raises(ValueError):
            validate_features(features, req)
