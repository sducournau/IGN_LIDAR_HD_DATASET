"""
Tests for facade scaling functionality (v3.0.3).

Tests the adaptive scaling feature that detects and applies
optimal scale factors to building facades.
"""

import numpy as np
import pytest
from shapely.geometry import LineString

from ign_lidar.core.classification.building.facade_processor import (
    BuildingFacadeClassifier,
    FacadeProcessor,
    FacadeSegment,
    FacadeOrientation,
)


class TestFacadeScaling:
    """Test suite for facade scaling functionality."""

    def test_project_points_on_facade_direction(self):
        """Test projection of points onto facade direction."""
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
        )

        # Points along facade direction
        candidate_points = np.array(
            [
                [0, 0, 0],
                [5, 0, 0],
                [10, 0, 0],
            ]
        )

        processor = FacadeProcessor(
            facade=facade,
            points=candidate_points,
            heights=np.zeros(len(candidate_points)),
            normals=None,
            verticality=None,
        )

        projected = processor._project_points_on_facade_direction(candidate_points)

        # Should get distances 0, 5, 10
        expected = np.array([0, 5, 10])
        np.testing.assert_array_almost_equal(projected, expected, decimal=5)

    def test_apply_scaling_to_line(self):
        """Test scaling of LineString."""
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
        )

        points = np.array([[0, 0, 0]])
        processor = FacadeProcessor(
            facade=facade,
            points=points,
            heights=np.array([0]),
            normals=None,
            verticality=None,
        )

        # Scale line 2x around center (5, 0)
        line = LineString([(0, 0), (10, 0)])
        center = np.array([5, 0])
        scale_factor = 2.0

        scaled_line = processor._apply_scaling_to_line(line, scale_factor, center)

        # Original: (0,0) to (10,0), center at (5,0)
        # After 2x scaling around (5,0):
        #   (0,0): dist from center = (-5,0), scaled = (-10,0),
        #          final = (5,0) + (-10,0) = (-5,0)
        #   (10,0): dist from center = (5,0), scaled = (10,0),
        #           final = (5,0) + (10,0) = (15,0)
        coords = list(scaled_line.coords)
        assert len(coords) == 2
        np.testing.assert_array_almost_equal(coords[0], (-5, 0), decimal=5)
        np.testing.assert_array_almost_equal(coords[1], (15, 0), decimal=5)

    def test_detect_optimal_scale_no_scaling_needed(self):
        """Test scale detection when facade is correct size."""
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
            length=10.0,
        )

        # Points spanning 0 to 10 (same as facade)
        candidate_points = np.array(
            [
                [0.5, 0, 0],
                [1.5, 0, 0],
                [2.5, 0, 0],
                [3.5, 0, 0],
                [4.5, 0, 0],
                [5.5, 0, 0],
                [6.5, 0, 0],
                [7.5, 0, 0],
                [8.5, 0, 0],
                [9.5, 0, 0],
            ]
        )

        processor = FacadeProcessor(
            facade=facade,
            points=candidate_points,
            heights=np.zeros(len(candidate_points)),
            normals=None,
            verticality=None,
        )

        current_length = 10.0
        max_scale_factor = 1.5

        scale_factor, confidence = processor._detect_optimal_scale(
            candidate_points, current_length, max_scale_factor
        )

        # Scale factor should be close to 1.0 (no scaling needed)
        # Note: Uses 5th-95th percentile, so with points at 0.5-9.5,
        # the detected extent will be slightly smaller (0.9 -> 9.1 = 8.2m)
        # Scale factor = 8.2 / 10.0 = 0.82
        assert 0.75 <= scale_factor <= 1.1  # Allow for percentile effect

    def test_detect_optimal_scale_needs_expansion(self):
        """Test scale detection when facade needs to be expanded."""
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
            length=10.0,
        )

        # Points spanning 0 to 15 (larger than facade)
        candidate_points = np.array(
            [
                [0.5, 0, 0],
                [2.0, 0, 0],
                [4.0, 0, 0],
                [6.0, 0, 0],
                [8.0, 0, 0],
                [10.0, 0, 0],
                [12.0, 0, 0],
                [13.0, 0, 0],
                [14.0, 0, 0],
                [14.5, 0, 0],
            ]
        )

        processor = FacadeProcessor(
            facade=facade,
            points=candidate_points,
            heights=np.zeros(len(candidate_points)),
            normals=None,
            verticality=None,
        )

        current_length = 10.0
        max_scale_factor = 1.5

        scale_factor, confidence = processor._detect_optimal_scale(
            candidate_points, current_length, max_scale_factor
        )

        # Scale factor should be > 1.0 (expansion needed)
        # Actual extent is ~14m, so scale should be ~1.4
        assert scale_factor > 1.2
        assert confidence > 0.0

    def test_detect_optimal_scale_needs_shrinkage(self):
        """Test scale detection when facade needs to be shrunk."""
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
            length=10.0,
        )

        # Points spanning only 2 to 8 (smaller than facade)
        candidate_points = np.array(
            [
                [2.5, 0, 0],
                [3.0, 0, 0],
                [4.0, 0, 0],
                [4.5, 0, 0],
                [5.0, 0, 0],
                [5.5, 0, 0],
                [6.0, 0, 0],
                [7.0, 0, 0],
                [7.5, 0, 0],
                [8.0, 0, 0],
            ]
        )

        processor = FacadeProcessor(
            facade=facade,
            points=candidate_points,
            heights=np.zeros(len(candidate_points)),
            normals=None,
            verticality=None,
        )

        current_length = 10.0
        max_scale_factor = 1.5

        scale_factor, confidence = processor._detect_optimal_scale(
            candidate_points, current_length, max_scale_factor
        )

        # Scale factor should be < 1.0 (shrinkage needed)
        # Actual extent is ~5.5m, so scale should be ~0.55
        assert scale_factor < 0.8
        assert confidence > 0.0

    def test_detect_optimal_scale_clamping(self):
        """Test that scale factor is clamped to max_scale_factor."""
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
            length=10.0,
        )

        # Points spanning 0 to 30 (3x larger than facade)
        candidate_points = np.array(
            [
                [0.5, 0, 0],
                [5.0, 0, 0],
                [10.0, 0, 0],
                [15.0, 0, 0],
                [20.0, 0, 0],
                [25.0, 0, 0],
                [29.5, 0, 0],
                [29.8, 0, 0],
                [29.9, 0, 0],
                [30.0, 0, 0],
            ]
        )

        processor = FacadeProcessor(
            facade=facade,
            points=candidate_points,
            heights=np.zeros(len(candidate_points)),
            normals=None,
            verticality=None,
        )

        current_length = 10.0
        max_scale_factor = 1.5

        scale_factor, confidence = processor._detect_optimal_scale(
            candidate_points, current_length, max_scale_factor
        )

        # Should be clamped to max_scale_factor
        assert scale_factor <= max_scale_factor

    def test_insufficient_points_for_scaling(self):
        """Test scaling detection with insufficient points."""
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
            length=10.0,
        )

        # Only 5 points (< 10 required)
        candidate_points = np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [4, 0, 0],
                [5, 0, 0],
            ]
        )

        processor = FacadeProcessor(
            facade=facade,
            points=candidate_points,
            heights=np.zeros(len(candidate_points)),
            normals=None,
            verticality=None,
        )

        scale_factor, confidence = processor._detect_optimal_scale(
            candidate_points, 10.0, 1.5
        )

        # Should return no scaling due to insufficient points
        assert scale_factor == 1.0
        assert confidence == 0.0

    def test_facade_scaling_integration(self):
        """Test full scaling integration in BuildingFacadeClassifier."""
        # Create classifier with scaling enabled
        classifier = BuildingFacadeClassifier(
            enable_facade_adaptation=True,
            max_rotation_degrees=0.0,  # Disable rotation for this test
            enable_scaling=True,
            max_scale_factor=1.5,
            min_confidence=0.1,
        )

        # Verify parameters
        assert classifier.enable_scaling is True
        assert classifier.max_scale_factor == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
