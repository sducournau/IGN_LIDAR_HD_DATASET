"""
Tests for BaseClassifier interface (v3.2+).

Tests the unified classifier interface including:
- ClassificationResult dataclass
- BaseClassifier abstract class
- Input validation
- Result statistics

Author: IGN LiDAR HD Team
Date: October 25, 2025
"""

import numpy as np
import pytest

from ign_lidar.core.classification import BaseClassifier, ClassificationResult


class TestClassificationResult:
    """Test ClassificationResult dataclass."""

    def test_create_result(self):
        """Test creating a classification result."""
        labels = np.array([1, 2, 3, 1, 2])
        confidence = np.array([0.9, 0.8, 0.7, 0.95, 0.85])

        result = ClassificationResult(
            labels=labels, confidence=confidence, metadata={"method": "test"}
        )

        assert len(result.labels) == 5
        assert len(result.confidence) == 5
        assert result.metadata["method"] == "test"

    def test_result_without_confidence(self):
        """Test result without confidence scores."""
        labels = np.array([1, 2, 3])

        result = ClassificationResult(labels=labels)

        assert result.confidence is None
        assert len(result.labels) == 3

    def test_result_validation_labels_not_array(self):
        """Test validation fails if labels is not array."""
        with pytest.raises(TypeError, match="labels must be numpy array"):
            ClassificationResult(labels=[1, 2, 3])

    def test_result_validation_labels_wrong_shape(self):
        """Test validation fails if labels has wrong shape."""
        labels = np.array([[1, 2], [3, 4]])  # 2D

        with pytest.raises(ValueError, match="labels must be 1D array"):
            ClassificationResult(labels=labels)

    def test_result_validation_confidence_mismatch(self):
        """Test validation fails if confidence shape doesn't match labels."""
        labels = np.array([1, 2, 3])
        confidence = np.array([0.9, 0.8])  # Too short

        with pytest.raises(ValueError, match="confidence shape .* must match"):
            ClassificationResult(labels=labels, confidence=confidence)

    def test_get_statistics_basic(self):
        """Test get_statistics method."""
        labels = np.array([1, 1, 2, 2, 3])
        result = ClassificationResult(labels=labels)

        stats = result.get_statistics()

        assert stats["total_points"] == 5
        assert stats["num_classes"] == 3
        assert stats["class_distribution"] == {1: 2, 2: 2, 3: 1}
        assert stats["class_percentages"][1] == 40.0
        assert stats["class_percentages"][2] == 40.0
        assert stats["class_percentages"][3] == 20.0

    def test_get_statistics_with_confidence(self):
        """Test statistics include confidence if available."""
        labels = np.array([1, 2, 3])
        confidence = np.array([0.9, 0.4, 0.7])

        result = ClassificationResult(labels=labels, confidence=confidence)
        stats = result.get_statistics()

        assert "avg_confidence" in stats
        assert abs(stats["avg_confidence"] - 0.666667) < 0.001
        assert stats["min_confidence"] == 0.4
        assert stats["low_confidence_count"] == 1  # confidence < 0.5

    def test_filter_by_confidence(self):
        """Test filtering points by confidence."""
        labels = np.array([1, 2, 3, 4, 5])
        confidence = np.array([0.9, 0.8, 0.3, 0.7, 0.2])

        result = ClassificationResult(labels=labels, confidence=confidence)
        high_conf = result.filter_by_confidence(threshold=0.5)

        assert len(high_conf.labels) == 3
        np.testing.assert_array_equal(high_conf.labels, [1, 2, 4])
        np.testing.assert_array_equal(high_conf.confidence, [0.9, 0.8, 0.7])

    def test_filter_by_confidence_no_confidence(self):
        """Test error when filtering without confidence scores."""
        labels = np.array([1, 2, 3])
        result = ClassificationResult(labels=labels)

        with pytest.raises(ValueError, match="no confidence scores available"):
            result.filter_by_confidence(0.5)

    def test_get_class_mask(self):
        """Test getting mask for a specific class."""
        labels = np.array([1, 2, 1, 3, 2, 1])
        result = ClassificationResult(labels=labels)

        mask_1 = result.get_class_mask(1)
        mask_2 = result.get_class_mask(2)

        np.testing.assert_array_equal(mask_1, [True, False, True, False, False, True])
        np.testing.assert_array_equal(mask_2, [False, True, False, False, True, False])

        assert np.sum(mask_1) == 3
        assert np.sum(mask_2) == 2


class TestBaseClassifierInterface:
    """Test BaseClassifier abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseClassifier cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseClassifier()

    def test_must_implement_classify(self):
        """Test that subclasses must implement classify()."""

        class IncompleteClassifier(BaseClassifier):
            pass

        with pytest.raises(TypeError):
            IncompleteClassifier()


class TestConcreteClassifier:
    """Test a concrete classifier implementation."""

    @pytest.fixture
    def simple_classifier(self):
        """Fixture providing a simple test classifier."""

        class SimpleClassifier(BaseClassifier):
            def classify(self, points, features, ground_truth=None, **kwargs):
                self.validate_inputs(points, features)

                # Simple classification: class based on height
                height = features.get("height", points[:, 2])
                labels = np.where(height > 5, 6, 2)  # Building or Ground
                confidence = np.ones(len(points)) * 0.8

                return ClassificationResult(
                    labels=labels,
                    confidence=confidence,
                    metadata={"method": "height_threshold"},
                )

        return SimpleClassifier()

    def test_classify_basic(self, simple_classifier):
        """Test basic classification."""
        points = np.random.rand(100, 3) * 10
        features = {"height": points[:, 2]}

        result = simple_classifier.classify(points, features)

        assert isinstance(result, ClassificationResult)
        assert len(result.labels) == 100
        assert len(result.confidence) == 100
        assert result.metadata["method"] == "height_threshold"

    def test_validate_inputs_valid(self, simple_classifier):
        """Test input validation passes with valid data."""
        points = np.random.rand(50, 3)
        features = {"height": np.random.rand(50), "planarity": np.random.rand(50)}

        # Should not raise
        simple_classifier.validate_inputs(points, features)

    def test_validate_inputs_invalid_points_type(self, simple_classifier):
        """Test validation fails with non-array points."""
        points = [[1, 2, 3], [4, 5, 6]]  # List, not array
        features = {"height": np.array([1, 2])}

        with pytest.raises(ValueError, match="points must be numpy array"):
            simple_classifier.validate_inputs(points, features)

    def test_validate_inputs_invalid_points_shape(self, simple_classifier):
        """Test validation fails with wrong point shape."""
        points = np.random.rand(50, 4)  # Should be [N, 3]
        features = {"height": np.random.rand(50)}

        with pytest.raises(ValueError, match="points must have shape"):
            simple_classifier.validate_inputs(points, features)

    def test_validate_inputs_empty_points(self, simple_classifier):
        """Test validation fails with empty points."""
        points = np.empty((0, 3))
        features = {}

        with pytest.raises(ValueError, match="points array is empty"):
            simple_classifier.validate_inputs(points, features)

    def test_validate_inputs_invalid_features_type(self, simple_classifier):
        """Test validation fails with non-dict features."""
        points = np.random.rand(50, 3)
        features = [1, 2, 3]  # List, not dict

        with pytest.raises(ValueError, match="features must be dictionary"):
            simple_classifier.validate_inputs(points, features)

    def test_validate_inputs_empty_features(self, simple_classifier):
        """Test validation fails with empty features."""
        points = np.random.rand(50, 3)
        features = {}

        with pytest.raises(ValueError, match="features dictionary is empty"):
            simple_classifier.validate_inputs(points, features)

    def test_validate_inputs_feature_wrong_length(self, simple_classifier):
        """Test validation fails when feature length doesn't match points."""
        points = np.random.rand(50, 3)
        features = {"height": np.random.rand(30)}  # Wrong length

        with pytest.raises(ValueError, match="has 30 values, expected 50"):
            simple_classifier.validate_inputs(points, features)

    def test_validate_inputs_feature_non_numeric(self, simple_classifier):
        """Test validation fails with non-numeric features."""
        points = np.random.rand(50, 3)
        features = {"height": np.array(["a", "b"] * 25, dtype=object)}

        with pytest.raises(ValueError, match="must be numeric"):
            simple_classifier.validate_inputs(points, features)

    def test_validate_inputs_feature_has_nan(self, simple_classifier):
        """Test validation fails with NaN in features."""
        points = np.random.rand(50, 3)
        features = {"height": np.array([np.nan] * 50)}

        with pytest.raises(ValueError, match="contains .* invalid values"):
            simple_classifier.validate_inputs(points, features)

    def test_validate_inputs_feature_has_inf(self, simple_classifier):
        """Test validation fails with Inf in features."""
        points = np.random.rand(50, 3)
        features = {"height": np.array([np.inf] * 50)}

        with pytest.raises(ValueError, match="contains .* invalid values"):
            simple_classifier.validate_inputs(points, features)

    def test_classifier_repr(self, simple_classifier):
        """Test classifier string representation."""
        repr_str = repr(simple_classifier)
        assert "SimpleClassifier" in repr_str


@pytest.fixture
def sample_points():
    """Fixture providing sample points."""
    return np.random.rand(100, 3) * 10


@pytest.fixture
def sample_features(sample_points):
    """Fixture providing sample features."""
    return {
        "height": sample_points[:, 2],
        "planarity": np.random.rand(len(sample_points)),
        "verticality": np.random.rand(len(sample_points)),
    }


def test_full_classification_pipeline(sample_points, sample_features):
    """Test a full classification pipeline."""

    class PipelineClassifier(BaseClassifier):
        def classify(self, points, features, ground_truth=None, **kwargs):
            self.validate_inputs(points, features)

            # Multi-criteria classification
            height = features["height"]
            planarity = features["planarity"]

            labels = np.ones(len(points), dtype=int)  # Default: unclassified
            labels[height < 0.5] = 2  # Ground
            labels[(height >= 0.5) & (height < 2)] = 3  # Low veg
            labels[(height >= 2) & (planarity > 0.8)] = 6  # Building
            labels[(height >= 2) & (planarity <= 0.8)] = 5  # High veg

            confidence = np.random.rand(len(points)) * 0.5 + 0.5

            return ClassificationResult(
                labels=labels,
                confidence=confidence,
                metadata={"criteria": "height_planarity"},
            )

    classifier = PipelineClassifier()
    result = classifier.classify(sample_points, sample_features)

    # Verify result
    assert len(result.labels) == len(sample_points)
    stats = result.get_statistics()
    assert stats["num_classes"] >= 1
    assert stats["total_points"] == len(sample_points)

    # Filter high confidence
    high_conf = result.filter_by_confidence(0.7)
    assert len(high_conf.labels) < len(result.labels)
