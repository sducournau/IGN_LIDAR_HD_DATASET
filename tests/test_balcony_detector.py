"""
Test suite for balcony and horizontal protrusion detection.

Tests the BalconyDetector class for Phase 2.3 implementation.
Validates detection of balconies, overhangs, and canopies.

Author: IGN LiDAR HD Processing Library
Version: 3.3.0 (Phase 2.3)
Date: January 2025
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from ign_lidar.core.classification.building.balcony_detector import (
    BalconyDetectionResult,
    BalconyDetector,
    ProtrusionSegment,
    ProtrusionType,
)


class TestBalconyDetectorInit:
    """Test BalconyDetector initialization."""

    def test_default_initialization(self):
        """Test detector initializes with default parameters."""
        detector = BalconyDetector()

        assert detector.min_distance_from_facade == 0.5
        assert detector.min_balcony_points == 25
        assert detector.max_balcony_depth == 3.0
        assert detector.min_height_above_ground == 2.0
        assert detector.max_height_from_roof == 2.0
        assert detector.dbscan_eps == 0.5
        assert detector.dbscan_min_samples == 15
        assert detector.confidence_threshold == 0.5

    def test_custom_initialization(self):
        """Test detector initializes with custom parameters."""
        detector = BalconyDetector(
            min_distance_from_facade=0.8,
            min_balcony_points=30,
            max_balcony_depth=2.5,
            min_height_above_ground=2.5,
            max_height_from_roof=1.5,
            dbscan_eps=0.4,
            dbscan_min_samples=20,
            confidence_threshold=0.6,
        )

        assert detector.min_distance_from_facade == 0.8
        assert detector.min_balcony_points == 30
        assert detector.max_balcony_depth == 2.5
        assert detector.min_height_above_ground == 2.5
        assert detector.max_height_from_roof == 1.5
        assert detector.dbscan_eps == 0.4
        assert detector.dbscan_min_samples == 20
        assert detector.confidence_threshold == 0.6


class TestBalconyDetection:
    """Test balcony detection functionality."""

    @pytest.fixture
    def detector(self):
        """Create standard detector instance with relaxed parameters for testing."""
        return BalconyDetector(
            min_distance_from_facade=0.3,  # Relaxed threshold
            min_balcony_points=15,  # Fewer points required
            max_balcony_depth=4.0,  # Allow deeper protrusions
            min_height_above_ground=1.0,  # Lower threshold
            dbscan_eps=0.8,  # Larger clustering radius
            dbscan_min_samples=10,  # Fewer samples for cluster
        )

    @pytest.fixture
    def simple_building_polygon(self):
        """Create simple rectangular building polygon."""
        return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    @pytest.fixture
    def building_with_balcony(self):
        """
        Create synthetic building with balcony on one facade.

        Returns:
            Tuple of (points, features, polygon, ground_elevation)
        """
        # Main building: 10x10m footprint, walls from z=0 to z=15m
        # Create points for building walls (300 points)
        wall_points = []

        # South wall (y=0)
        for _ in range(75):
            x = np.random.uniform(0, 10)
            y = 0.0
            z = np.random.uniform(0, 15)
            wall_points.append([x, y, z])

        # North wall with balcony (y=10)
        for _ in range(75):
            x = np.random.uniform(0, 10)
            y = 10.0
            z = np.random.uniform(0, 15)
            wall_points.append([x, y, z])

        # East and west walls
        for _ in range(75):
            x = np.random.choice([0.0, 10.0])
            y = np.random.uniform(0, 10)
            z = np.random.uniform(0, 15)
            wall_points.append([x, y, z])

        # Balcony: Protruding 1.5m from north wall, at height 5-6m
        balcony_points = []
        for _ in range(50):
            x = np.random.uniform(3, 7)  # 4m wide balcony centered
            y = np.random.uniform(10.0, 11.5)  # Protrudes 1.5m
            z = np.random.uniform(5.0, 6.0)  # At height 5-6m
            balcony_points.append([x, y, z])

        # Combine
        points = np.array(wall_points + balcony_points)
        total_points = len(points)  # Should be 300

        # Features - ensure same length as points
        features = {
            "verticality": np.concatenate(
                [
                    np.ones(225) * 0.8,  # Walls: high verticality
                    np.ones(50) * 0.3,  # Balcony floor: low verticality
                ]
            )
        }

        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        ground_elevation = 0.0

        return points, features, polygon, ground_elevation

    def test_empty_point_cloud(self, detector, simple_building_polygon):
        """Test handling of empty point cloud."""
        points = np.array([]).reshape(0, 3)
        features = {"verticality": np.array([])}

        result = detector.detect_protrusions(
            points, features, simple_building_polygon, 0.0
        )

        assert isinstance(result, BalconyDetectionResult)
        assert not result.detection_success
        assert result.num_balconies == 0

    def test_missing_verticality_feature(self, detector, simple_building_polygon):
        """Test handling of missing verticality feature."""
        points = np.random.rand(100, 3)
        features = {}  # No verticality

        result = detector.detect_protrusions(
            points, features, simple_building_polygon, 0.0
        )

        assert isinstance(result, BalconyDetectionResult)
        assert not result.detection_success
        assert result.num_balconies == 0

    def test_detect_balcony(self, detector, building_with_balcony):
        """Test detection of balcony on building facade."""
        points, features, polygon, ground_elev = building_with_balcony

        result = detector.detect_protrusions(points, features, polygon, ground_elev)

        # Should detect the balcony
        assert result.detection_success
        assert result.num_balconies >= 1 or result.num_canopies >= 1
        # Might be classified as balcony or canopy depending on geometry

    def test_no_protrusions(self, detector, simple_building_polygon):
        """Test building with no horizontal protrusions."""
        # Just wall points, no protrusions
        points = []
        for _ in range(200):
            x = np.random.choice([0.0, 10.0, np.random.uniform(0, 10)])
            y = np.random.choice([0.0, 10.0, np.random.uniform(0, 10)])
            z = np.random.uniform(0, 15)
            # Keep points close to building envelope
            if x not in [0.0, 10.0]:
                y = np.random.choice([0.0, 10.0])
            points.append([x, y, z])

        points = np.array(points)
        features = {"verticality": np.ones(200) * 0.8}

        result = detector.detect_protrusions(
            points, features, simple_building_polygon, 0.0
        )

        # Should not detect any protrusions
        assert result.num_balconies == 0
        assert result.num_overhangs == 0
        assert result.num_canopies == 0


class TestFacadeExtraction:
    """Test facade line extraction from building polygons."""

    @pytest.fixture
    def detector(self):
        return BalconyDetector()

    def test_extract_rectangular_facade(self, detector):
        """Test facade extraction from rectangular building."""
        polygon = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])

        facade_lines = detector._extract_facade_lines(polygon)

        assert len(facade_lines) == 4  # 4 sides
        # Check that lines exist and have length
        for line in facade_lines:
            assert line.length > 0

    def test_extract_complex_facade(self, detector):
        """Test facade extraction from complex building."""
        polygon = Polygon([(0, 0), (10, 0), (10, 5), (8, 5), (8, 8), (0, 8)])

        facade_lines = detector._extract_facade_lines(polygon)

        assert len(facade_lines) == 6  # 6 sides


class TestDistanceComputation:
    """Test distance from facade computation."""

    @pytest.fixture
    def detector(self):
        return BalconyDetector()

    def test_distance_from_single_facade(self, detector):
        """Test distance computation from single facade line."""
        from shapely.geometry import LineString

        # Facade at y=0, from x=0 to x=10
        facade_lines = [LineString([(0, 0), (10, 0)])]

        # Points at various distances from facade
        points = np.array(
            [
                [5, 0, 0],  # On facade
                [5, 1, 0],  # 1m from facade
                [5, 2, 0],  # 2m from facade
            ]
        )

        distances, _ = detector._compute_distance_from_facades(points, facade_lines)

        assert np.isclose(distances[0], 0.0, atol=0.01)
        assert np.isclose(distances[1], 1.0, atol=0.01)
        assert np.isclose(distances[2], 2.0, atol=0.01)


class TestCandidateDetection:
    """Test candidate protrusion point detection."""

    @pytest.fixture
    def detector(self):
        return BalconyDetector(
            min_distance_from_facade=0.5,
            max_balcony_depth=3.0,
            min_height_above_ground=2.0,
        )

    def test_detect_candidates_basic(self, detector):
        """Test basic candidate detection logic."""
        points = np.random.rand(100, 3)
        features = {"verticality": np.ones(100) * 0.3}

        # Distances: half beyond threshold, half within
        distances = np.concatenate(
            [
                np.random.uniform(0.6, 2.0, 50),  # Beyond facade
                np.random.uniform(0.0, 0.4, 50),  # Too close to facade
            ]
        )

        # Heights: all above ground
        heights = np.ones(100) * 5.0

        mask = detector._detect_candidates(points, features, distances, heights, None)

        # Should detect only the points beyond facade threshold
        assert np.sum(mask) <= 50
        assert np.sum(mask) > 0


class TestProtrusionClassification:
    """Test protrusion type classification."""

    @pytest.fixture
    def detector(self):
        return BalconyDetector()

    def test_classify_balcony_geometry(self, detector):
        """Test classification of balcony-like geometry."""
        # Balcony: 4m wide, 1.5m deep, at height 5m
        cluster_points = []
        for _ in range(40):
            x = np.random.uniform(0, 4)
            y = np.random.uniform(0, 1.5)
            z = np.random.uniform(4.8, 5.2)
            cluster_points.append([x, y, z])

        cluster_points = np.array(cluster_points)
        cluster_indices = np.arange(40)

        features = {"verticality": np.ones(40) * 0.4}
        distances = np.random.uniform(0.8, 2.3, 40)
        heights = np.ones(40) * 5.0
        facade_indices = np.zeros(40, dtype=int)

        prot = detector._classify_protrusion_cluster(
            cluster_indices,
            cluster_points,
            features,
            distances,
            heights,
            facade_indices,
        )

        assert prot is not None
        # Should classify as balcony or canopy
        assert prot.type in [
            ProtrusionType.BALCONY,
            ProtrusionType.CANOPY,
        ]

    def test_classify_overhang_geometry(self, detector):
        """Test classification of overhang-like geometry."""
        # Overhang: 2m wide, 0.8m deep, at high elevation (12m)
        cluster_points = []
        for _ in range(30):
            x = np.random.uniform(0, 2)
            y = np.random.uniform(0, 0.8)
            z = np.random.uniform(11.8, 12.2)
            cluster_points.append([x, y, z])

        cluster_points = np.array(cluster_points)
        cluster_indices = np.arange(30)

        features = {"verticality": np.ones(30) * 0.2}
        distances = np.random.uniform(0.5, 1.3, 30)
        heights = np.ones(30) * 12.0
        facade_indices = np.zeros(30, dtype=int)

        prot = detector._classify_protrusion_cluster(
            cluster_indices,
            cluster_points,
            features,
            distances,
            heights,
            facade_indices,
        )

        assert prot is not None
        assert prot.type == ProtrusionType.OVERHANG
        assert prot.confidence > 0.5


class TestResultDataclass:
    """Test BalconyDetectionResult dataclass."""

    def test_empty_result_creation(self):
        """Test creation of empty result."""
        detector = BalconyDetector()
        result = detector._empty_result()

        assert isinstance(result, BalconyDetectionResult)
        assert result.num_balconies == 0
        assert result.num_overhangs == 0
        assert result.num_canopies == 0
        assert not result.detection_success
        assert len(result.protrusions) == 0

    def test_result_with_detections(self):
        """Test result with actual detections."""
        prot1 = ProtrusionSegment(
            type=ProtrusionType.BALCONY,
            points_mask=np.zeros(100, dtype=bool),
            centroid=np.array([5, 11, 5]),
            distance_from_facade=1.0,
            max_distance_from_facade=1.5,
            height_above_ground=5.0,
            width=4.0,
            depth=1.5,
            area=6.0,
            verticality=0.3,
            point_count=40,
            facade_side=0,
            confidence=0.8,
        )

        result = BalconyDetectionResult(
            protrusions=[prot1],
            balcony_indices=np.arange(40),
            overhang_indices=np.array([], dtype=int),
            canopy_indices=np.array([], dtype=int),
            facade_lines=[],
            detection_success=True,
            num_balconies=1,
            num_overhangs=0,
            num_canopies=0,
        )

        assert result.detection_success
        assert result.num_balconies == 1
        assert len(result.protrusions) == 1
        assert result.protrusions[0].type == ProtrusionType.BALCONY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
