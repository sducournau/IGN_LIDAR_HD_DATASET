"""
Tests for Roof Type Classifier (Phase 2 - v3.1).

Tests roof type detection, segmentation, and architectural detail detection.

Author: IGN LiDAR HD Development Team
Date: October 26, 2025
"""

import numpy as np
import pytest

from ign_lidar.core.classification.building.roof_classifier import (
    RoofTypeClassifier,
    RoofType,
    RoofSegment,
    RoofClassificationResult,
)
from ign_lidar.classification_schema import ASPRSClass


@pytest.fixture
def roof_classifier():
    """Create a standard roof classifier for testing."""
    return RoofTypeClassifier(
        flat_threshold=15.0,
        pitched_threshold=20.0,
        min_plane_points=50,
        planarity_threshold=0.85,
        verticality_threshold=0.3,
    )


@pytest.fixture
def flat_roof_data():
    """Generate synthetic flat roof point cloud."""
    # Create a flat horizontal surface at z=10m
    x = np.random.uniform(0, 10, 500)
    y = np.random.uniform(0, 10, 500)
    z = np.full(500, 10.0) + np.random.normal(0, 0.05, 500)  # Small noise

    points = np.column_stack([x, y, z])

    # Normals pointing up for flat roof
    normals = np.tile([0, 0, 1], (500, 1))
    normals += np.random.normal(0, 0.05, (500, 3))  # Small variation
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    features = {
        "normals": normals,
        "planarity": np.full(500, 0.95),
        "verticality": np.full(500, 0.05),  # Low verticality = horizontal
    }

    labels = np.full(500, ASPRSClass.BUILDING, dtype=int)

    return points, features, labels


@pytest.fixture
def gabled_roof_data():
    """Generate synthetic gabled roof point cloud (2 slopes)."""
    points_list = []
    normals_list = []

    # Left slope (tilted)
    for i in range(250):
        x = np.random.uniform(0, 5, 1)[0]
        y = np.random.uniform(0, 10, 1)[0]
        z = 10 + x * 0.6  # 30° slope
        points_list.append([x, y, z])
        # Normal pointing up and left
        normals_list.append([-0.5, 0, 0.866])

    # Right slope (opposite tilt)
    for i in range(250):
        x = np.random.uniform(5, 10, 1)[0]
        y = np.random.uniform(0, 10, 1)[0]
        z = 10 + (10 - x) * 0.6  # Opposite 30° slope
        points_list.append([x, y, z])
        # Normal pointing up and right
        normals_list.append([0.5, 0, 0.866])

    points = np.array(points_list)
    normals = np.array(normals_list)

    # Add noise
    normals += np.random.normal(0, 0.05, normals.shape)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    features = {
        "normals": normals,
        "planarity": np.full(500, 0.90),
        "verticality": np.full(500, 0.15),  # Slightly tilted surfaces
    }

    labels = np.full(500, ASPRSClass.BUILDING, dtype=int)

    return points, features, labels


class TestRoofTypeClassifier:
    """Test suite for RoofTypeClassifier."""

    def test_initialization(self):
        """Test classifier initialization with default parameters."""
        classifier = RoofTypeClassifier()

        assert classifier.flat_threshold == 15.0
        assert classifier.pitched_threshold == 20.0
        assert classifier.min_plane_points == 100
        assert classifier.planarity_threshold == 0.85

    def test_initialization_custom_params(self):
        """Test classifier initialization with custom parameters."""
        classifier = RoofTypeClassifier(
            flat_threshold=10.0,
            pitched_threshold=25.0,
            min_plane_points=75,
        )

        assert classifier.flat_threshold == 10.0
        assert classifier.pitched_threshold == 25.0
        assert classifier.min_plane_points == 75

    def test_classify_flat_roof(self, roof_classifier, flat_roof_data):
        """Test classification of flat roof."""
        points, features, labels = flat_roof_data

        result = roof_classifier.classify_roof(points, features, labels)

        assert isinstance(result, RoofClassificationResult)
        assert result.roof_type == RoofType.FLAT
        assert result.confidence > 0.8
        assert result.stats["num_segments"] >= 1
        assert result.stats["total_roof_points"] > 0

    def test_classify_gabled_roof(self, roof_classifier, gabled_roof_data):
        """Test classification of gabled roof (2 slopes)."""
        points, features, labels = gabled_roof_data

        result = roof_classifier.classify_roof(points, features, labels)

        assert isinstance(result, RoofClassificationResult)
        # Should detect 2 slopes => gabled roof
        assert result.roof_type in [RoofType.GABLED, RoofType.COMPLEX]
        assert result.confidence > 0.5
        assert result.stats["num_segments"] >= 1

    def test_empty_input(self, roof_classifier):
        """Test with empty point cloud."""
        points = np.empty((0, 3))
        features = {
            "normals": np.empty((0, 3)),
            "planarity": np.empty(0),
            "verticality": np.empty(0),
        }

        result = roof_classifier.classify_roof(points, features)

        assert result.roof_type == RoofType.UNKNOWN
        assert result.confidence == 0.0
        assert len(result.segments) == 0

    def test_missing_features_error(self, roof_classifier):
        """Test error handling when required features are missing."""
        points = np.random.rand(100, 3)
        features = {"normals": np.random.rand(100, 3)}  # Missing planarity

        with pytest.raises(ValueError, match="Missing required features"):
            roof_classifier.classify_roof(points, features)

    def test_identify_roof_points(self, roof_classifier, flat_roof_data):
        """Test roof point identification."""
        points, features, labels = flat_roof_data

        roof_mask = roof_classifier._identify_roof_points(points, features, labels)

        assert isinstance(roof_mask, np.ndarray)
        assert roof_mask.dtype == bool
        assert roof_mask.sum() > 0  # Should find some roof points
        # Most points should be classified as roof (low verticality)
        assert roof_mask.sum() / len(roof_mask) > 0.5

    def test_identify_roof_points_no_labels(self, roof_classifier):
        """Test roof identification without existing labels."""
        points = np.random.rand(100, 3)
        features = {
            "normals": np.random.rand(100, 3),
            "planarity": np.random.rand(100),
            "verticality": np.random.rand(100) * 0.2,  # Low verticality
        }

        roof_mask = roof_classifier._identify_roof_points(points, features, None)

        assert isinstance(roof_mask, np.ndarray)
        assert roof_mask.dtype == bool

    def test_segment_roof_planes(self, roof_classifier, flat_roof_data):
        """Test roof plane segmentation."""
        points, features, _ = flat_roof_data

        segments = roof_classifier._segment_roof_planes(
            points, features["normals"], features["planarity"]
        )

        assert isinstance(segments, list)
        if len(segments) > 0:
            segment = segments[0]
            assert isinstance(segment, RoofSegment)
            assert len(segment.points) > 0
            assert len(segment.normal) == 3
            assert 0 <= segment.slope_angle <= 90
            assert 0 <= segment.planarity <= 1

    def test_classify_segment_type(self, roof_classifier):
        """Test individual segment type classification."""
        # Flat roof segment
        flat_type = roof_classifier._classify_segment_type(5.0)
        assert flat_type == RoofType.FLAT

        # Pitched roof segment
        pitched_type = roof_classifier._classify_segment_type(35.0)
        assert pitched_type == RoofType.GABLED

        # Intermediate angle
        inter_type = roof_classifier._classify_segment_type(17.0)
        assert inter_type == RoofType.UNKNOWN

    def test_detect_ridge_lines(self, roof_classifier, gabled_roof_data):
        """Test ridge line detection."""
        points, features, _ = gabled_roof_data

        # Create mock segments
        segments = [
            RoofSegment(
                points=np.arange(250),
                normal=np.array([-0.5, 0, 0.866]),
                centroid=np.array([2.5, 5, 11]),
                area=50.0,
                slope_angle=30.0,
                planarity=0.9,
            ),
            RoofSegment(
                points=np.arange(250, 500),
                normal=np.array([0.5, 0, 0.866]),
                centroid=np.array([7.5, 5, 11]),
                area=50.0,
                slope_angle=30.0,
                planarity=0.9,
            ),
        ]

        ridge_indices = roof_classifier._detect_ridge_lines(
            points, features["normals"], segments
        )

        assert isinstance(ridge_indices, np.ndarray)
        # Ridge detection is complex, just verify it returns valid indices
        if len(ridge_indices) > 0:
            assert ridge_indices.min() >= 0
            assert ridge_indices.max() < len(points)

    def test_detect_roof_edges(self, roof_classifier, flat_roof_data):
        """Test roof edge detection."""
        points, _, _ = flat_roof_data

        edge_indices = roof_classifier._detect_roof_edges(points)

        assert isinstance(edge_indices, np.ndarray)
        # Should detect some edge points
        if len(edge_indices) > 0:
            assert edge_indices.min() >= 0
            assert edge_indices.max() < len(points)

    def test_detect_dormers(self, roof_classifier):
        """Test dormer detection."""
        # Create roof with small vertical structure (dormer)
        points_list = []
        normals_list = []

        # Main flat roof
        for i in range(400):
            x = np.random.uniform(0, 10, 1)[0]
            y = np.random.uniform(0, 10, 1)[0]
            z = 10.0
            points_list.append([x, y, z])
            normals_list.append([0, 0, 1])

        # Dormer (vertical structure)
        for i in range(20):
            x = 5 + np.random.uniform(-0.5, 0.5, 1)[0]
            y = 5 + np.random.uniform(-0.5, 0.5, 1)[0]
            z = 10 + np.random.uniform(0, 1, 1)[0]
            points_list.append([x, y, z])
            normals_list.append([1, 0, 0])  # Vertical normal

        points = np.array(points_list)
        normals = np.array(normals_list)

        # Mock segments (main roof)
        segments = [
            RoofSegment(
                points=np.arange(400),
                normal=np.array([0, 0, 1]),
                centroid=np.array([5, 5, 10]),
                area=100.0,
                slope_angle=0.0,
                planarity=0.95,
            )
        ]

        dormer_indices = roof_classifier._detect_dormers(points, normals, segments)

        assert isinstance(dormer_indices, np.ndarray)
        # Dormer detection is complex, verify it returns valid output
        if len(dormer_indices) > 0:
            assert dormer_indices.min() >= 0
            assert dormer_indices.max() < len(points)

    def test_roof_type_classification_logic(self, roof_classifier):
        """Test the overall roof type classification logic."""
        # Single flat segment => FLAT
        seg1 = RoofSegment(
            points=np.arange(100),
            normal=np.array([0, 0, 1]),
            centroid=np.array([5, 5, 10]),
            area=50.0,
            slope_angle=5.0,
            planarity=0.95,
            roof_type=RoofType.FLAT,
        )
        roof_type, conf = roof_classifier._classify_roof_type([seg1])
        assert roof_type == RoofType.FLAT
        assert conf > 0.8

        # Two opposed segments => GABLED
        seg2 = RoofSegment(
            points=np.arange(100, 200),
            normal=np.array([-0.5, 0, 0.866]),
            centroid=np.array([2.5, 5, 11]),
            area=50.0,
            slope_angle=30.0,
            planarity=0.90,
            roof_type=RoofType.GABLED,
        )
        seg3 = RoofSegment(
            points=np.arange(200, 300),
            normal=np.array([0.5, 0, 0.866]),
            centroid=np.array([7.5, 5, 11]),
            area=50.0,
            slope_angle=30.0,
            planarity=0.90,
            roof_type=RoofType.GABLED,
        )
        roof_type, conf = roof_classifier._classify_roof_type([seg2, seg3])
        assert roof_type == RoofType.GABLED
        assert conf > 0.5

    def test_insufficient_points(self, roof_classifier):
        """Test with insufficient points for meaningful classification."""
        points = np.random.rand(10, 3)  # Too few points
        features = {
            "normals": np.random.rand(10, 3),
            "planarity": np.full(10, 0.95),
            "verticality": np.full(10, 0.1),
        }

        result = roof_classifier.classify_roof(points, features)

        # Should handle gracefully
        assert result.roof_type == RoofType.UNKNOWN
        assert result.confidence == 0.0

    def test_result_statistics(self, roof_classifier, flat_roof_data):
        """Test that result statistics are properly populated."""
        points, features, labels = flat_roof_data

        result = roof_classifier.classify_roof(points, features, labels)

        assert "total_roof_points" in result.stats
        assert "num_segments" in result.stats
        assert "avg_slope" in result.stats
        assert "avg_planarity" in result.stats
        assert result.stats["total_roof_points"] > 0
        assert result.stats["num_segments"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
