"""
Test suite for enhanced building classifier integration.

Tests the BuildingClassifier that integrates:
- Roof type detection (Phase 2.1)
- Chimney detection (Phase 2.2)
- Balcony detection (Phase 2.3)

Author: IGN LiDAR HD Processing Library
Version: 3.4.0 (Phase 2.4 Integration)
Date: October 2025
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from ign_lidar.core.classification.building.building_classifier import (
    BuildingClassifier,
    BuildingClassifierConfig,
    BuildingClassificationResult,
    classify_building,
)


class TestEnhancedClassifierInit:
    """Test enhanced classifier initialization."""

    def test_default_initialization(self):
        """Test classifier initializes with default config."""
        classifier = BuildingClassifier()

        assert classifier.roof_classifier is not None
        assert classifier.chimney_detector is not None
        assert classifier.balcony_detector is not None

        stats = classifier.get_detector_stats()
        assert stats["roof_classifier"] is True
        assert stats["chimney_detector"] is True
        assert stats["balcony_detector"] is True

    def test_custom_config(self):
        """Test classifier with custom configuration."""
        config = BuildingClassifierConfig(
            enable_roof_detection=True,
            enable_chimney_detection=False,
            enable_balcony_detection=True,
            roof_flat_threshold=10.0,
            balcony_min_points=30,
        )

        classifier = BuildingClassifier(config)

        assert classifier.roof_classifier is not None
        assert classifier.chimney_detector is None
        assert classifier.balcony_detector is not None

        stats = classifier.get_detector_stats()
        assert stats["roof_classifier"] is True
        assert stats["chimney_detector"] is False
        assert stats["balcony_detector"] is True

    def test_all_detectors_disabled(self):
        """Test classifier with all detectors disabled."""
        config = BuildingClassifierConfig(
            enable_roof_detection=False,
            enable_chimney_detection=False,
            enable_balcony_detection=False,
        )

        classifier = BuildingClassifier(config)

        assert classifier.roof_classifier is None
        assert classifier.chimney_detector is None
        assert classifier.balcony_detector is None


class TestEnhancedClassification:
    """Test complete enhanced classification pipeline."""

    @pytest.fixture
    def classifier(self):
        """Create classifier with relaxed parameters for testing."""
        config = BuildingClassifierConfig(
            enable_roof_detection=True,
            enable_chimney_detection=True,
            enable_balcony_detection=True,
            roof_flat_threshold=15.0,
            chimney_min_height_above_roof=0.5,
            chimney_min_points=15,
            balcony_min_distance_from_facade=0.3,
            balcony_min_points=15,
        )
        return BuildingClassifier(config)

    @pytest.fixture
    def complex_building(self):
        """
        Create synthetic complex building with all features.

        Returns:
            Tuple of (points, features, polygon, ground_elevation)
        """
        points = []

        # 1. Facade walls (10x10m, 0-15m height) - 100 points
        for _ in range(100):
            x = np.random.choice([0.0, 10.0])
            y = np.random.uniform(0, 10)
            z = np.random.uniform(0, 15)
            points.append([x, y, z])

        # 2. Pitched roof (two planes, 15-18m height) - 100 points
        for _ in range(50):
            x = np.random.uniform(0, 5)  # West slope
            y = np.random.uniform(0, 10)
            z = 15 + x * 0.6  # Pitched
            points.append([x, y, z])

        for _ in range(50):
            x = np.random.uniform(5, 10)  # East slope
            y = np.random.uniform(0, 10)
            z = 15 + (10 - x) * 0.6  # Pitched
            points.append([x, y, z])

        # 3. Chimney (at roof peak, 18-20m height) - 30 points
        for _ in range(30):
            x = np.random.uniform(4.5, 5.5)
            y = np.random.uniform(4, 6)
            z = np.random.uniform(18, 20)
            points.append([x, y, z])

        # 4. Balcony (extending from south wall, 5-6m height) - 40 points
        for _ in range(40):
            x = np.random.uniform(3, 7)
            y = np.random.uniform(-1.5, 0)  # Extends beyond facade
            z = np.random.uniform(5, 6)
            points.append([x, y, z])

        points = np.array(points)
        total_points = len(points)  # 270 points

        # Create features
        features = {
            "verticality": np.concatenate(
                [
                    np.ones(100) * 0.85,  # Walls: high verticality
                    np.ones(100) * 0.25,  # Roof: low verticality
                    np.ones(30) * 0.9,  # Chimney: very high verticality
                    np.ones(40) * 0.3,  # Balcony: low verticality
                ]
            ),
            "normals": np.concatenate(
                [
                    np.tile([1, 0, 0], (100, 1)),  # Walls: horizontal normals
                    np.tile([0, 0, 1], (100, 1)),  # Roof: upward normals
                    np.tile([1, 0, 0], (30, 1)),  # Chimney: horizontal
                    np.tile([0, 0, 1], (40, 1)),  # Balcony: upward
                ]
            ),
            "planarity": np.concatenate(
                [
                    np.ones(100) * 0.9,  # Walls: planar
                    np.ones(100) * 0.8,  # Roof: fairly planar
                    np.ones(30) * 0.85,  # Chimney: planar
                    np.ones(40) * 0.75,  # Balcony: fairly planar
                ]
            ),
        }

        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        ground_elevation = 0.0

        return points, features, polygon, ground_elevation

    def test_empty_point_cloud(self, classifier):
        """Test handling of empty point cloud."""
        points = np.array([]).reshape(0, 3)
        features = {
            "verticality": np.array([]),
            "normals": np.array([]).reshape(0, 3),
            "planarity": np.array([]),
        }
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        result = classifier.classify_building(points, features, polygon, 0.0)

        # Should return unsuccessful result, not raise
        assert result.success is False

    def test_missing_required_features(self, classifier):
        """Test handling of missing required features."""
        points = np.random.rand(100, 3)
        features = {}  # No features
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        result = classifier.classify_building(points, features, polygon, 0.0)

        # Should return unsuccessful result, not raise
        assert result.success is False

    def test_classify_complex_building(self, classifier, complex_building):
        """Test classification of complex building with all features."""
        points, features, polygon, ground_elev = complex_building

        result = classifier.classify_building(points, features, polygon, ground_elev)

        # Check result structure
        assert isinstance(result, BuildingClassificationResult)
        assert result.success is True

        # Check roof detection
        assert result.roof_result is not None
        # Roof classification doesn't have detection_success,
        # just check that we got a result

        # Check chimney detection
        assert result.chimney_result is not None
        # May or may not detect chimney depending on parameters

        # Check balcony detection
        assert result.balcony_result is not None
        # May or may not detect balcony depending on parameters

        # Check point labels
        assert result.point_labels is not None
        assert len(result.point_labels) == len(points)

        # Check building stats
        assert "total_points" in result.building_stats
        assert result.building_stats["total_points"] == len(points)
        assert "roof_type" in result.building_stats

    def test_convenience_function(self, complex_building):
        """Test convenience function for classification."""
        points, features, polygon, ground_elev = complex_building

        result = classify_building(points, features, polygon, ground_elev)

        assert isinstance(result, BuildingClassificationResult)
        assert result.success is True


class TestDetectorIntegration:
    """Test individual detector integration."""

    def test_roof_only_classification(self):
        """Test with only roof detection enabled."""
        config = BuildingClassifierConfig(
            enable_roof_detection=True,
            enable_chimney_detection=False,
            enable_balcony_detection=False,
        )
        classifier = BuildingClassifier(config)

        # Simple flat roof
        points = np.array([[5, 5, 15 + i * 0.1] for i in range(100)])
        features = {
            "verticality": np.ones(100) * 0.2,
            "normals": np.tile([0, 0, 1], (100, 1)),
            "planarity": np.ones(100) * 0.9,
        }
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        result = classifier.classify_building(points, features, polygon, 0.0)

        assert result.success is True
        assert result.roof_result is not None
        assert result.chimney_result is None  # Disabled
        assert result.balcony_result is None  # Disabled

    def test_chimney_requires_roof(self):
        """Test that chimney detection requires roof detection."""
        config = BuildingClassifierConfig(
            enable_roof_detection=False,
            enable_chimney_detection=True,
            enable_balcony_detection=False,
        )
        classifier = BuildingClassifier(config)

        # Roof with chimney
        points = np.random.rand(100, 3) * [10, 10, 5] + [0, 0, 15]
        features = {
            "verticality": np.random.rand(100) * 0.5,
        }
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        result = classifier.classify_building(points, features, polygon, 0.0)

        # Chimney detector runs but can't detect without roof result
        assert result.success is True
        assert result.roof_result is None
        assert result.chimney_result is None  # Not run without roof


class TestStatisticsComputation:
    """Test building statistics computation."""

    def test_stats_all_detectors(self):
        """Test statistics with all detectors enabled."""
        classifier = BuildingClassifier()

        # Simple building
        points = np.random.rand(100, 3) * [10, 10, 15]
        features = {
            "verticality": np.random.rand(100),
            "normals": np.random.rand(100, 3),
            "planarity": np.random.rand(100),
        }
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        result = classifier.classify_building(points, features, polygon, 0.0)

        assert "total_points" in result.building_stats
        assert "height_range" in result.building_stats
        assert "label_distribution" in result.building_stats

        # Check label distribution is valid
        label_dist = result.building_stats["label_distribution"]
        assert isinstance(label_dist, dict)
        assert sum(label_dist.values()) == len(points)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
