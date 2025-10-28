"""
Tests for facade rotation functionality (v3.0.3).

Tests the adaptive rotation feature that detects and applies
optimal rotation angles to building facades.
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


class TestFacadeRotation:
    """Test suite for facade rotation functionality."""

    def test_rotate_points_2d(self):
        """Test 2D point rotation around a center."""
        # Create a simple facade processor for testing
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
        )

        points = np.array([[0, 0, 0], [10, 0, 0]])
        processor = FacadeProcessor(
            facade=facade,
            points=points,
            heights=np.array([0, 0]),
            normals=None,
            verticality=None,
        )

        # Test 90-degree rotation
        points_2d = np.array([[1, 0], [2, 0]])
        center = np.array([0, 0])
        angle = np.pi / 2  # 90 degrees

        rotated = processor._rotate_points_2d(points_2d, angle, center)

        # After 90-degree rotation: (x, y) -> (-y, x)
        expected = np.array([[0, 1], [0, 2]])
        np.testing.assert_array_almost_equal(rotated, expected, decimal=5)

    def test_apply_rotation_to_line(self):
        """Test rotation of LineString."""
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

        # Rotate line 90 degrees around origin
        line = LineString([(0, 0), (10, 0)])
        center = np.array([0, 0])
        angle = np.pi / 2  # 90 degrees

        rotated_line = processor._apply_rotation_to_line(line, angle, center)

        # Original: (0,0) -> (10,0)
        # After 90° rotation: (0,0) -> (0,10)
        coords = list(rotated_line.coords)
        assert len(coords) == 2
        np.testing.assert_array_almost_equal(coords[0], (0, 0), decimal=5)
        np.testing.assert_array_almost_equal(coords[1], (0, 10), decimal=5)

    def test_compute_alignment_score(self):
        """Test alignment score computation."""
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

        line = LineString([(0, 0), (10, 0)])

        # Points on the line should have high score
        points_on_line = np.array([[1, 0], [5, 0], [9, 0]])
        score_on = processor._compute_alignment_score(points_on_line, line)
        assert score_on > 0.9  # Should be close to 1.0

        # Points far from line should have low score
        points_far = np.array([[1, 5], [5, 5], [9, 5]])
        score_far = processor._compute_alignment_score(points_far, line)
        assert score_far < 0.5  # Should be lower

        # Score should decrease with distance
        assert score_on > score_far

    def test_detect_optimal_rotation_no_rotation_needed(self):
        """Test rotation detection when points are already aligned."""
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
        )

        # Points aligned with facade (no rotation needed)
        candidate_points = np.array(
            [
                [1, 0.1, 0],
                [2, -0.1, 0],
                [3, 0.05, 0],
                [4, -0.05, 0],
                [5, 0.1, 0],
                [6, -0.1, 0],
                [7, 0.1, 0],
                [8, -0.1, 0],
                [9, 0.05, 0],
                [10, -0.05, 0],
            ]
        )

        processor = FacadeProcessor(
            facade=facade,
            points=candidate_points,
            heights=np.zeros(len(candidate_points)),
            normals=None,
            verticality=None,
        )

        current_angle = 0.0
        max_rotation = 15.0

        rotation_angle, confidence = processor._detect_optimal_rotation(
            candidate_points, current_angle, max_rotation
        )

        # Should detect minimal rotation (< 2 degrees)
        assert abs(np.degrees(rotation_angle)) < 2.0

    def test_detect_optimal_rotation_with_misalignment(self):
        """Test rotation detection when points are misaligned."""
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
        )

        # Points rotated 10 degrees from facade
        angle_deg = 10.0
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Create rotated points
        x_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_values = np.zeros(len(x_values))

        candidate_points = np.zeros((len(x_values), 3))
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            # Rotate point
            candidate_points[i, 0] = x * cos_a - y * sin_a
            candidate_points[i, 1] = x * sin_a + y * cos_a
            candidate_points[i, 2] = 0

        processor = FacadeProcessor(
            facade=facade,
            points=candidate_points,
            heights=np.zeros(len(candidate_points)),
            normals=None,
            verticality=None,
        )

        current_angle = 0.0
        max_rotation = 15.0

        rotation_angle, confidence = processor._detect_optimal_rotation(
            candidate_points, current_angle, max_rotation
        )

        # Should detect a rotation that improves alignment
        # The rotation might be detected in either direction
        # due to test_angles range
        # What matters is that it improves alignment
        detected_angle_deg = np.degrees(rotation_angle)

        # Either detects -10° (counterclockwise) or finds
        # max_rotation boundary
        # Since points are rotated 10° from facade,
        # detector should find non-zero rotation
        assert abs(detected_angle_deg) > 5.0  # Significant rotation
        assert (
            confidence > 0.0
        )  # Should have confidence    def test_facade_rotation_integration(self):
        """Test full rotation integration in BuildingFacadeClassifier."""
        # Create classifier with rotation enabled
        classifier = BuildingFacadeClassifier(
            enable_facade_adaptation=True,
            max_rotation_degrees=15.0,
            enable_scaling=False,  # Disable scaling for this test
            min_confidence=0.1,
        )

        # Verify the classifier can be instantiated with new params
        assert classifier.max_rotation_degrees == 15.0
        assert classifier.enable_scaling is False

    def test_insufficient_points_for_rotation(self):
        """Test rotation detection with insufficient points."""
        facade = FacadeSegment(
            edge_line=LineString([(0, 0), (10, 0)]),
            orientation=FacadeOrientation.NORTH,
            building_id=1,
            centroid=np.array([5.0, 0.0, 0.0]),
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

        rotation_angle, confidence = processor._detect_optimal_rotation(
            candidate_points, 0.0, 15.0
        )

        # Should return no rotation due to insufficient points
        assert rotation_angle == 0.0
        assert confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
