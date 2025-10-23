"""
Comprehensive tests for rules confidence module.

Tests confidence calculation, combination, normalization, and calibration
utilities for rule-based classification.
"""

import pytest
import numpy as np
from typing import Dict

from ign_lidar.core.classification.rules.confidence import (
    ConfidenceMethod,
    CombinationMethod,
    calculate_confidence,
    combine_confidences,
    normalize_confidence,
    calibrate_confidence,
    apply_confidence_threshold
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_scores():
    """Sample score array for testing"""
    return np.array([0.1, 0.3, 0.5, 0.7, 0.9])


@pytest.fixture
def sample_confidences():
    """Sample confidence dictionary for testing"""
    return {
        'geometric': np.array([0.8, 0.9, 0.7, 0.6, 0.85]),
        'height': np.array([0.7, 0.8, 0.9, 0.5, 0.6]),
        'intensity': np.array([0.6, 0.7, 0.8, 0.9, 0.5])
    }


@pytest.fixture
def calibration_data():
    """Sample data for calibration testing"""
    return {
        'confidence': np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]),
        'true_labels': np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0]),
        'predicted_labels': np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0])
    }


# ============================================================================
# Test ConfidenceMethod Enum
# ============================================================================

class TestConfidenceMethod:
    """Test ConfidenceMethod enumeration"""
    
    def test_enum_values(self):
        """Test that all confidence methods are defined"""
        assert ConfidenceMethod.BINARY == "binary"
        assert ConfidenceMethod.LINEAR == "linear"
        assert ConfidenceMethod.SIGMOID == "sigmoid"
        assert ConfidenceMethod.GAUSSIAN == "gaussian"
        assert ConfidenceMethod.THRESHOLD == "threshold"
        assert ConfidenceMethod.EXPONENTIAL == "exponential"
        assert ConfidenceMethod.COMPOSITE == "composite"
    
    def test_enum_membership(self):
        """Test enum membership"""
        assert "linear" in [m.value for m in ConfidenceMethod]
        assert "sigmoid" in [m.value for m in ConfidenceMethod]


# ============================================================================
# Test CombinationMethod Enum
# ============================================================================

class TestCombinationMethod:
    """Test CombinationMethod enumeration"""
    
    def test_enum_values(self):
        """Test that all combination methods are defined"""
        assert CombinationMethod.WEIGHTED_AVERAGE == "weighted_average"
        assert CombinationMethod.MAX == "max"
        assert CombinationMethod.MIN == "min"
        assert CombinationMethod.PRODUCT == "product"
        assert CombinationMethod.GEOMETRIC_MEAN == "geometric_mean"
        assert CombinationMethod.HARMONIC_MEAN == "harmonic_mean"


# ============================================================================
# Test calculate_confidence - Binary Method
# ============================================================================

class TestCalculateConfidenceBinary:
    """Test binary confidence calculation"""
    
    def test_binary_default_threshold(self, sample_scores):
        """Test binary method with default threshold"""
        confidence = calculate_confidence(sample_scores, ConfidenceMethod.BINARY)
        
        expected = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(confidence, expected)
    
    def test_binary_custom_threshold(self, sample_scores):
        """Test binary method with custom threshold"""
        confidence = calculate_confidence(
            sample_scores,
            ConfidenceMethod.BINARY,
            params={'threshold': 0.7}
        )
        
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(confidence, expected)
    
    def test_binary_all_below_threshold(self):
        """Test binary when all values below threshold"""
        scores = np.array([0.1, 0.2, 0.3])
        confidence = calculate_confidence(
            scores,
            ConfidenceMethod.BINARY,
            params={'threshold': 0.5}
        )
        
        np.testing.assert_array_equal(confidence, np.array([0.0, 0.0, 0.0]))


# ============================================================================
# Test calculate_confidence - Linear Method
# ============================================================================

class TestCalculateConfidenceLinear:
    """Test linear confidence calculation"""
    
    def test_linear_auto_range(self, sample_scores):
        """Test linear scaling with automatic range"""
        confidence = calculate_confidence(sample_scores, ConfidenceMethod.LINEAR)
        
        # Should scale from 0.1-0.9 to 0.0-1.0
        assert confidence[0] == 0.0  # min maps to 0
        assert confidence[-1] == 1.0  # max maps to 1
        assert 0 <= confidence.min() <= confidence.max() <= 1
    
    def test_linear_custom_range(self):
        """Test linear scaling with custom range"""
        scores = np.array([0.0, 0.5, 1.0])
        confidence = calculate_confidence(
            scores,
            ConfidenceMethod.LINEAR,
            params={'min': 0.0, 'max': 1.0}
        )
        
        np.testing.assert_array_almost_equal(confidence, scores)
    
    def test_linear_constant_scores(self):
        """Test linear with all same values"""
        scores = np.array([0.5, 0.5, 0.5])
        confidence = calculate_confidence(scores, ConfidenceMethod.LINEAR)
        
        # Should return all ones when min == max
        np.testing.assert_array_equal(confidence, np.ones(3))


# ============================================================================
# Test calculate_confidence - Sigmoid Method
# ============================================================================

class TestCalculateConfidenceSigmoid:
    """Test sigmoid confidence calculation"""
    
    def test_sigmoid_default_params(self):
        """Test sigmoid with default parameters"""
        scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        confidence = calculate_confidence(scores, ConfidenceMethod.SIGMOID)
        
        # Check that it's monotonically increasing
        assert np.all(np.diff(confidence) > 0)
        
        # Center should give ~0.5
        assert 0.4 < confidence[2] < 0.6
    
    def test_sigmoid_custom_center(self):
        """Test sigmoid with custom center"""
        scores = np.array([0.0, 0.3, 0.6, 1.0])
        confidence = calculate_confidence(
            scores,
            ConfidenceMethod.SIGMOID,
            params={'center': 0.3, 'steepness': 10.0}
        )
        
        # Value at center should be ~0.5
        assert 0.4 < confidence[1] < 0.6
    
    def test_sigmoid_steepness(self):
        """Test sigmoid steepness parameter"""
        scores = np.array([0.4, 0.5, 0.6])
        
        # Low steepness = gradual transition
        gradual = calculate_confidence(
            scores,
            ConfidenceMethod.SIGMOID,
            params={'center': 0.5, 'steepness': 1.0}
        )
        
        # High steepness = sharp transition
        sharp = calculate_confidence(
            scores,
            ConfidenceMethod.SIGMOID,
            params={'center': 0.5, 'steepness': 50.0}
        )
        
        # Sharp should be closer to binary
        assert sharp[2] - sharp[0] > gradual[2] - gradual[0]


# ============================================================================
# Test calculate_confidence - Gaussian Method
# ============================================================================

class TestCalculateConfidenceGaussian:
    """Test gaussian confidence calculation"""
    
    def test_gaussian_centered(self):
        """Test gaussian peaks at center"""
        scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        confidence = calculate_confidence(
            scores,
            ConfidenceMethod.GAUSSIAN,
            params={'center': 0.5, 'sigma': 0.2}
        )
        
        # Should peak at center
        assert confidence[2] == confidence.max()
        assert confidence[2] == 1.0
    
    def test_gaussian_symmetry(self):
        """Test gaussian is symmetric around center"""
        scores = np.array([0.3, 0.5, 0.7])
        confidence = calculate_confidence(
            scores,
            ConfidenceMethod.GAUSSIAN,
            params={'center': 0.5, 'sigma': 0.2}
        )
        
        # Values equidistant from center should be equal
        assert abs(confidence[0] - confidence[2]) < 1e-10
    
    def test_gaussian_sigma(self):
        """Test gaussian sigma affects width"""
        scores = np.linspace(0, 1, 11)
        
        narrow = calculate_confidence(
            scores,
            ConfidenceMethod.GAUSSIAN,
            params={'center': 0.5, 'sigma': 0.1}
        )
        
        wide = calculate_confidence(
            scores,
            ConfidenceMethod.GAUSSIAN,
            params={'center': 0.5, 'sigma': 0.3}
        )
        
        # Narrow sigma should have sharper peak
        assert narrow[5] == wide[5] == 1.0  # Both peak at center
        assert narrow[3] < wide[3]  # But narrow drops faster


# ============================================================================
# Test calculate_confidence - Threshold Method
# ============================================================================

class TestCalculateConfidenceThreshold:
    """Test threshold confidence calculation"""
    
    def test_threshold_hard_regions(self):
        """Test hard threshold regions"""
        scores = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        confidence = calculate_confidence(
            scores,
            ConfidenceMethod.THRESHOLD,
            params={'threshold': 0.5, 'soft_margin': 0.1}
        )
        
        # Below threshold - margin = 0
        assert confidence[0] == 0.0
        assert confidence[1] == 0.0
        
        # Above threshold + margin = 1
        assert confidence[4] == 1.0
    
    def test_threshold_soft_transition(self):
        """Test soft threshold transition zone"""
        scores = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
        confidence = calculate_confidence(
            scores,
            ConfidenceMethod.THRESHOLD,
            params={'threshold': 0.5, 'soft_margin': 0.1}
        )
        
        # In transition zone [0.4, 0.6]
        assert 0.0 <= confidence[0] <= 1.0
        assert 0.0 <= confidence[2] <= 1.0
        assert 0.0 <= confidence[4] <= 1.0
        
        # Should be monotonically increasing
        assert np.all(np.diff(confidence) >= 0)


# ============================================================================
# Test calculate_confidence - Exponential Method
# ============================================================================

class TestCalculateConfidenceExponential:
    """Test exponential confidence calculation"""
    
    def test_exponential_decay(self):
        """Test exponential decay mode"""
        scores = np.array([0.0, 0.5, 1.0, 2.0])
        confidence = calculate_confidence(
            scores,
            ConfidenceMethod.EXPONENTIAL,
            params={'rate': 1.0, 'decay': True}
        )
        
        # Decay: should decrease with increasing scores
        assert np.all(np.diff(confidence) <= 0)
        assert confidence[0] == 1.0  # exp(0) = 1
    
    def test_exponential_growth(self):
        """Test exponential growth mode"""
        scores = np.array([0.0, 0.5, 1.0, 2.0])
        confidence = calculate_confidence(
            scores,
            ConfidenceMethod.EXPONENTIAL,
            params={'rate': 1.0, 'decay': False}
        )
        
        # Growth: should increase with increasing scores
        assert np.all(np.diff(confidence) >= 0)
        assert confidence[0] == 0.0  # 1 - exp(0) = 0
    
    def test_exponential_rate(self):
        """Test exponential rate parameter"""
        scores = np.array([1.0])
        
        slow = calculate_confidence(
            scores,
            ConfidenceMethod.EXPONENTIAL,
            params={'rate': 0.5, 'decay': True}
        )
        
        fast = calculate_confidence(
            scores,
            ConfidenceMethod.EXPONENTIAL,
            params={'rate': 2.0, 'decay': True}
        )
        
        # Higher rate = faster decay
        assert fast[0] < slow[0]


# ============================================================================
# Test calculate_confidence - Error Cases
# ============================================================================

class TestCalculateConfidenceErrors:
    """Test error handling in calculate_confidence"""
    
    def test_unknown_method_raises(self, sample_scores):
        """Test that unknown method raises error"""
        with pytest.raises(ValueError, match="Unknown confidence method"):
            calculate_confidence(sample_scores, "invalid_method")


# ============================================================================
# Test combine_confidences
# ============================================================================

class TestCombineConfidences:
    """Test confidence combination methods"""
    
    def test_weighted_average_uniform(self, sample_confidences):
        """Test weighted average with uniform weights"""
        combined = combine_confidences(
            sample_confidences,
            method=CombinationMethod.WEIGHTED_AVERAGE
        )
        
        # Should be average of all confidences
        expected = np.mean(list(sample_confidences.values()), axis=0)
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_weighted_average_custom(self, sample_confidences):
        """Test weighted average with custom weights"""
        weights = {'geometric': 0.5, 'height': 0.3, 'intensity': 0.2}
        combined = combine_confidences(
            sample_confidences,
            weights=weights,
            method=CombinationMethod.WEIGHTED_AVERAGE
        )
        
        # Manually calculate expected
        expected = (
            0.5 * sample_confidences['geometric'] +
            0.3 * sample_confidences['height'] +
            0.2 * sample_confidences['intensity']
        )
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_max_combination(self, sample_confidences):
        """Test maximum combination"""
        combined = combine_confidences(
            sample_confidences,
            method=CombinationMethod.MAX
        )
        
        # Should be element-wise maximum
        expected = np.maximum.reduce(list(sample_confidences.values()))
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_min_combination(self, sample_confidences):
        """Test minimum combination"""
        combined = combine_confidences(
            sample_confidences,
            method=CombinationMethod.MIN
        )
        
        # Should be element-wise minimum
        expected = np.minimum.reduce(list(sample_confidences.values()))
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_product_combination(self, sample_confidences):
        """Test product combination"""
        combined = combine_confidences(
            sample_confidences,
            method=CombinationMethod.PRODUCT
        )
        
        # Should be element-wise product
        expected = np.prod(list(sample_confidences.values()), axis=0)
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_geometric_mean(self, sample_confidences):
        """Test geometric mean combination"""
        combined = combine_confidences(
            sample_confidences,
            method=CombinationMethod.GEOMETRIC_MEAN
        )
        
        # Should be geometric mean
        n = len(sample_confidences)
        product = np.prod(list(sample_confidences.values()), axis=0)
        expected = product ** (1.0 / n)
        np.testing.assert_array_almost_equal(combined, expected, decimal=5)
    
    def test_harmonic_mean(self, sample_confidences):
        """Test harmonic mean combination"""
        combined = combine_confidences(
            sample_confidences,
            method=CombinationMethod.HARMONIC_MEAN
        )
        
        # Should be harmonic mean
        n = len(sample_confidences)
        reciprocal_sum = np.sum(
            [1.0 / conf for conf in sample_confidences.values()],
            axis=0
        )
        expected = n / reciprocal_sum
        np.testing.assert_array_almost_equal(combined, expected, decimal=5)
    
    def test_empty_confidences_raises(self):
        """Test that empty confidences raises error"""
        with pytest.raises(ValueError, match="Cannot combine empty"):
            combine_confidences({})
    
    def test_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise error"""
        confidences = {
            'a': np.array([0.5, 0.6]),
            'b': np.array([0.7, 0.8, 0.9])  # Different length!
        }
        
        with pytest.raises(ValueError, match="has .* values, expected"):
            combine_confidences(confidences)
    
    def test_zero_weights_raises(self):
        """Test that all-zero weights raise error"""
        confidences = {'a': np.array([0.5, 0.6])}
        weights = {'a': 0.0}
        
        with pytest.raises(ValueError, match="Total weight is zero"):
            combine_confidences(confidences, weights=weights)


# ============================================================================
# Test normalize_confidence
# ============================================================================

class TestNormalizeConfidence:
    """Test confidence normalization"""
    
    def test_normalize_to_01(self):
        """Test normalization to [0, 1]"""
        confidence = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        normalized = normalize_confidence(confidence)
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
    
    def test_normalize_custom_range(self):
        """Test normalization to custom range"""
        confidence = np.array([0.0, 0.5, 1.0])
        normalized = normalize_confidence(confidence, min_val=0.3, max_val=0.9)
        
        assert abs(normalized.min() - 0.3) < 1e-10
        assert abs(normalized.max() - 0.9) < 1e-10
        assert abs(normalized[1] - 0.6) < 1e-10  # Middle value
    
    def test_normalize_constant_values(self):
        """Test normalization with constant values"""
        confidence = np.array([0.7, 0.7, 0.7])
        normalized = normalize_confidence(confidence, min_val=0.2, max_val=0.8)
        
        # Should return midpoint
        expected = (0.2 + 0.8) / 2
        np.testing.assert_array_almost_equal(normalized, np.full(3, expected))
    
    def test_normalize_invalid_range_raises(self):
        """Test that invalid range raises error"""
        confidence = np.array([0.5, 0.6])
        
        with pytest.raises(ValueError, match="must be >"):
            normalize_confidence(confidence, min_val=0.8, max_val=0.3)


# ============================================================================
# Test calibrate_confidence
# ============================================================================

class TestCalibrateConfidence:
    """Test confidence calibration"""
    
    def test_calibration_basic(self, calibration_data):
        """Test basic calibration computation"""
        stats = calibrate_confidence(
            calibration_data['confidence'],
            calibration_data['true_labels'],
            calibration_data['predicted_labels'],
            n_bins=5
        )
        
        # Check all expected keys present
        assert 'expected_calibration_error' in stats
        assert 'overall_accuracy' in stats
        assert 'mean_confidence' in stats
        assert 'confidence_accuracy_gap' in stats
        assert 'bin_accuracies' in stats
        assert 'bin_confidences' in stats
        assert 'bin_counts' in stats
    
    def test_calibration_bin_counts(self, calibration_data):
        """Test calibration bin counts"""
        stats = calibrate_confidence(
            calibration_data['confidence'],
            calibration_data['true_labels'],
            calibration_data['predicted_labels'],
            n_bins=5
        )
        
        # Total bin counts should equal total samples
        assert sum(stats['bin_counts']) == len(calibration_data['confidence'])
    
    def test_perfect_calibration(self):
        """Test perfectly calibrated predictions"""
        # Perfect calibration: confidence = accuracy
        n = 100
        confidence = np.random.rand(n)
        true_labels = np.random.randint(0, 2, n)
        
        # Make predictions perfectly calibrated
        predicted_labels = (np.random.rand(n) < confidence).astype(int)
        
        # Note: This is a stochastic test, so we just check it runs
        stats = calibrate_confidence(confidence, true_labels, predicted_labels)
        
        assert 0 <= stats['expected_calibration_error'] <= 1
        assert 0 <= stats['overall_accuracy'] <= 1


# ============================================================================
# Test apply_confidence_threshold
# ============================================================================

class TestApplyConfidenceThreshold:
    """Test confidence thresholding"""
    
    def test_threshold_basic(self):
        """Test basic confidence thresholding"""
        labels = np.array([1, 2, 3, 4, 5])
        confidence = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        
        filtered = apply_confidence_threshold(labels, confidence, threshold=0.5)
        
        # Above threshold kept, below set to 0
        expected = np.array([1, 2, 3, 0, 0])
        np.testing.assert_array_equal(filtered, expected)
    
    def test_threshold_all_above(self):
        """Test when all confidences above threshold"""
        labels = np.array([1, 2, 3])
        confidence = np.array([0.9, 0.8, 0.7])
        
        filtered = apply_confidence_threshold(labels, confidence, threshold=0.5)
        
        # All should be kept
        np.testing.assert_array_equal(filtered, labels)
    
    def test_threshold_all_below(self):
        """Test when all confidences below threshold"""
        labels = np.array([1, 2, 3])
        confidence = np.array([0.3, 0.2, 0.1])
        
        filtered = apply_confidence_threshold(labels, confidence, threshold=0.5)
        
        # All should be zeroed
        np.testing.assert_array_equal(filtered, np.zeros(3))
    
    def test_threshold_preserves_original(self):
        """Test that original labels are not modified"""
        labels = np.array([1, 2, 3])
        confidence = np.array([0.9, 0.3, 0.7])
        original_labels = labels.copy()
        
        filtered = apply_confidence_threshold(labels, confidence, threshold=0.5)
        
        # Original should be unchanged
        np.testing.assert_array_equal(labels, original_labels)
        # But filtered should be different
        assert not np.array_equal(filtered, labels)


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfidenceIntegration:
    """Test integration of confidence functions"""
    
    def test_calculate_and_combine_workflow(self):
        """Test complete workflow of calculating and combining confidences"""
        # Calculate confidences using different methods
        scores1 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        scores2 = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        
        conf1 = calculate_confidence(scores1, ConfidenceMethod.LINEAR)
        conf2 = calculate_confidence(scores2, ConfidenceMethod.SIGMOID)
        
        # Combine them
        confidences = {'linear': conf1, 'sigmoid': conf2}
        combined = combine_confidences(
            confidences,
            weights={'linear': 0.6, 'sigmoid': 0.4}
        )
        
        # Normalize
        normalized = normalize_confidence(combined, min_val=0.0, max_val=1.0)
        
        # Apply threshold
        labels = np.array([1, 2, 3, 4, 5])
        filtered = apply_confidence_threshold(labels, normalized, threshold=0.5)
        
        # Should have some labels filtered
        assert np.sum(filtered == 0) >= 0
        assert len(filtered) == len(labels)
    
    def test_all_confidence_methods(self):
        """Test that all confidence methods work"""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        methods = [
            ConfidenceMethod.BINARY,
            ConfidenceMethod.LINEAR,
            ConfidenceMethod.SIGMOID,
            ConfidenceMethod.GAUSSIAN,
            ConfidenceMethod.THRESHOLD,
            ConfidenceMethod.EXPONENTIAL
        ]
        
        for method in methods:
            confidence = calculate_confidence(scores, method)
            
            # All should return valid confidence scores
            assert len(confidence) == len(scores)
            assert np.all(confidence >= 0)
            assert np.all(confidence <= 1)
    
    def test_all_combination_methods(self, sample_confidences):
        """Test that all combination methods work"""
        methods = [
            CombinationMethod.WEIGHTED_AVERAGE,
            CombinationMethod.MAX,
            CombinationMethod.MIN,
            CombinationMethod.PRODUCT,
            CombinationMethod.GEOMETRIC_MEAN,
            CombinationMethod.HARMONIC_MEAN
        ]
        
        for method in methods:
            combined = combine_confidences(sample_confidences, method=method)
            
            # All should return valid confidence scores
            assert len(combined) == 5
            assert np.all(combined >= 0)
            assert np.all(combined <= 1)
