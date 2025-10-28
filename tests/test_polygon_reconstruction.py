"""
Tests for polygon reconstruction from adapted facades (v3.0.3).

Tests the reconstruction of building polygons from adapted facade geometry.
"""

import numpy as np
import pytest
from shapely.geometry import LineString

from ign_lidar.core.classification.building.facade_processor import (
    BuildingFacadeClassifier,
    FacadeSegment,
    FacadeOrientation,
)


class TestPolygonReconstruction:
    """Test suite for polygon reconstruction functionality."""

    def test_reconstruct_simple_rectangle(self):
        """Test reconstruction of a simple rectangular building."""
        # Create 4 facades for a 20x10m building
        facades = [
            FacadeSegment(
                edge_line=LineString([(0, 0), (20, 0)]),
                orientation=FacadeOrientation.SOUTH,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 10), (20, 10)]),
                orientation=FacadeOrientation.NORTH,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([(20, 0), (20, 10)]),
                orientation=FacadeOrientation.EAST,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 0), (0, 10)]),
                orientation=FacadeOrientation.WEST,
                building_id=1,
                is_adapted=False,
            ),
        ]

        classifier = BuildingFacadeClassifier()
        reconstructed = classifier._reconstruct_polygon_from_facades(facades)

        assert reconstructed is not None
        assert reconstructed.is_valid
        # Area should be approximately 20 * 10 = 200 m²
        assert 195 <= reconstructed.area <= 205

    def test_reconstruct_with_adapted_facades(self):
        """Test reconstruction with adapted facade geometry."""
        # Create 4 facades with adapted lines
        facades = [
            FacadeSegment(
                edge_line=LineString([(0, 0), (20, 0)]),
                orientation=FacadeOrientation.SOUTH,
                building_id=1,
                adjusted_edge_line=LineString([(0, -1), (20, -1)]),
                is_adapted=True,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 10), (20, 10)]),
                orientation=FacadeOrientation.NORTH,
                building_id=1,
                adjusted_edge_line=LineString([(0, 11), (20, 11)]),
                is_adapted=True,
            ),
            FacadeSegment(
                edge_line=LineString([(20, 0), (20, 10)]),
                orientation=FacadeOrientation.EAST,
                building_id=1,
                adjusted_edge_line=LineString([(21, 0), (21, 10)]),
                is_adapted=True,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 0), (0, 10)]),
                orientation=FacadeOrientation.WEST,
                building_id=1,
                adjusted_edge_line=LineString([(-1, 0), (-1, 10)]),
                is_adapted=True,
            ),
        ]

        classifier = BuildingFacadeClassifier()
        reconstructed = classifier._reconstruct_polygon_from_facades(facades)

        assert reconstructed is not None
        assert reconstructed.is_valid
        # Area should be larger due to expansion: ~22 * 12 = 264 m²
        assert 250 <= reconstructed.area <= 280

    def test_reconstruct_mixed_adapted_original(self):
        """Test reconstruction with mix of adapted and original facades."""
        # Only north and east facades are adapted
        facades = [
            FacadeSegment(
                edge_line=LineString([(0, 0), (20, 0)]),
                orientation=FacadeOrientation.SOUTH,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 10), (20, 10)]),
                orientation=FacadeOrientation.NORTH,
                building_id=1,
                adjusted_edge_line=LineString([(0, 11), (20, 11)]),
                is_adapted=True,
            ),
            FacadeSegment(
                edge_line=LineString([(20, 0), (20, 10)]),
                orientation=FacadeOrientation.EAST,
                building_id=1,
                adjusted_edge_line=LineString([(21, 0), (21, 10)]),
                is_adapted=True,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 0), (0, 10)]),
                orientation=FacadeOrientation.WEST,
                building_id=1,
                is_adapted=False,
            ),
        ]

        classifier = BuildingFacadeClassifier()
        reconstructed = classifier._reconstruct_polygon_from_facades(facades)

        assert reconstructed is not None
        assert reconstructed.is_valid
        # Area should be between original (200) and fully adapted (264)
        assert 200 <= reconstructed.area <= 280

    def test_reconstruct_rotated_facades(self):
        """Test reconstruction with rotated facades."""
        # Facades rotated 5 degrees
        angle_deg = 5.0
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotate the building corners
        def rotate_point(x, y):
            return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

        sw = rotate_point(0, 0)
        se = rotate_point(20, 0)
        ne = rotate_point(20, 10)
        nw = rotate_point(0, 10)

        facades = [
            FacadeSegment(
                edge_line=LineString([sw, se]),
                orientation=FacadeOrientation.SOUTH,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([nw, ne]),
                orientation=FacadeOrientation.NORTH,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([se, ne]),
                orientation=FacadeOrientation.EAST,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([sw, nw]),
                orientation=FacadeOrientation.WEST,
                building_id=1,
                is_adapted=False,
            ),
        ]

        classifier = BuildingFacadeClassifier()
        reconstructed = classifier._reconstruct_polygon_from_facades(facades)

        assert reconstructed is not None
        assert reconstructed.is_valid
        # Area should still be approximately 20 * 10 = 200 m²
        assert 195 <= reconstructed.area <= 205

    def test_reconstruct_insufficient_facades(self):
        """Test reconstruction with fewer than 4 facades."""
        # Only 3 facades
        facades = [
            FacadeSegment(
                edge_line=LineString([(0, 0), (20, 0)]),
                orientation=FacadeOrientation.SOUTH,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 10), (20, 10)]),
                orientation=FacadeOrientation.NORTH,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([(20, 0), (20, 10)]),
                orientation=FacadeOrientation.EAST,
                building_id=1,
                is_adapted=False,
            ),
        ]

        classifier = BuildingFacadeClassifier()
        reconstructed = classifier._reconstruct_polygon_from_facades(facades)

        # Should return None with insufficient facades
        assert reconstructed is None

    def test_reconstruct_scaled_facades(self):
        """Test reconstruction with scaled facades."""
        # All facades scaled to 1.5x length
        facades = [
            FacadeSegment(
                edge_line=LineString([(0, 0), (20, 0)]),
                orientation=FacadeOrientation.SOUTH,
                building_id=1,
                adjusted_edge_line=LineString([(-5, 0), (25, 0)]),
                is_adapted=True,
                is_scaled=True,
                scale_factor=1.5,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 10), (20, 10)]),
                orientation=FacadeOrientation.NORTH,
                building_id=1,
                adjusted_edge_line=LineString([(-5, 10), (25, 10)]),
                is_adapted=True,
                is_scaled=True,
                scale_factor=1.5,
            ),
            FacadeSegment(
                edge_line=LineString([(20, 0), (20, 10)]),
                orientation=FacadeOrientation.EAST,
                building_id=1,
                adjusted_edge_line=LineString([(25, -2.5), (25, 12.5)]),
                is_adapted=True,
                is_scaled=True,
                scale_factor=1.5,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 0), (0, 10)]),
                orientation=FacadeOrientation.WEST,
                building_id=1,
                adjusted_edge_line=LineString([(-5, -2.5), (-5, 12.5)]),
                is_adapted=True,
                is_scaled=True,
                scale_factor=1.5,
            ),
        ]

        classifier = BuildingFacadeClassifier()
        reconstructed = classifier._reconstruct_polygon_from_facades(facades)

        assert reconstructed is not None
        assert reconstructed.is_valid
        # Area: width scaled 20→30, height scaled 10→15
        # But reconstruction uses actual line intersections
        # Actual area ~30 * 10 = 300 m² (y coords still at 0 and 10)
        assert 280 <= reconstructed.area <= 320

    def test_reconstruct_with_rotation_and_scaling(self):
        """Test reconstruction with both rotation and scaling."""
        # Complex case: rotation + scaling
        angle_deg = 10.0
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotate and scale
        def transform_point(x, y, scale=1.0):
            x_scaled = x * scale
            y_scaled = y * scale
            return (
                x_scaled * cos_a - y_scaled * sin_a,
                x_scaled * sin_a + y_scaled * cos_a,
            )

        sw = transform_point(0, 0, 1.2)
        se = transform_point(20, 0, 1.2)
        ne = transform_point(20, 10, 1.2)
        nw = transform_point(0, 10, 1.2)

        facades = [
            FacadeSegment(
                edge_line=LineString([(0, 0), (20, 0)]),
                orientation=FacadeOrientation.SOUTH,
                building_id=1,
                adjusted_edge_line=LineString([sw, se]),
                is_adapted=True,
                is_rotated=True,
                is_scaled=True,
                rotation_angle=angle_rad,
                scale_factor=1.2,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 10), (20, 10)]),
                orientation=FacadeOrientation.NORTH,
                building_id=1,
                adjusted_edge_line=LineString([nw, ne]),
                is_adapted=True,
                is_rotated=True,
                is_scaled=True,
                rotation_angle=angle_rad,
                scale_factor=1.2,
            ),
            FacadeSegment(
                edge_line=LineString([(20, 0), (20, 10)]),
                orientation=FacadeOrientation.EAST,
                building_id=1,
                adjusted_edge_line=LineString([se, ne]),
                is_adapted=True,
                is_rotated=True,
                is_scaled=True,
                rotation_angle=angle_rad,
                scale_factor=1.2,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 0), (0, 10)]),
                orientation=FacadeOrientation.WEST,
                building_id=1,
                adjusted_edge_line=LineString([sw, nw]),
                is_adapted=True,
                is_rotated=True,
                is_scaled=True,
                rotation_angle=angle_rad,
                scale_factor=1.2,
            ),
        ]

        classifier = BuildingFacadeClassifier()
        reconstructed = classifier._reconstruct_polygon_from_facades(facades)

        assert reconstructed is not None
        assert reconstructed.is_valid
        # Area should be scaled: 20*10*1.2² = 288 m²
        assert 270 <= reconstructed.area <= 310

    def test_reconstruct_area_validation(self):
        """Test that reconstructed polygon area is reasonable."""
        # Original building: 10x10m
        original_area = 100.0

        facades = [
            FacadeSegment(
                edge_line=LineString([(0, 0), (10, 0)]),
                orientation=FacadeOrientation.SOUTH,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 10), (10, 10)]),
                orientation=FacadeOrientation.NORTH,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([(10, 0), (10, 10)]),
                orientation=FacadeOrientation.EAST,
                building_id=1,
                is_adapted=False,
            ),
            FacadeSegment(
                edge_line=LineString([(0, 0), (0, 10)]),
                orientation=FacadeOrientation.WEST,
                building_id=1,
                is_adapted=False,
            ),
        ]

        classifier = BuildingFacadeClassifier()
        reconstructed = classifier._reconstruct_polygon_from_facades(facades)

        assert reconstructed is not None
        # Area should not change by more than 10% for non-adapted facades
        area_ratio = reconstructed.area / original_area
        assert 0.9 <= area_ratio <= 1.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
