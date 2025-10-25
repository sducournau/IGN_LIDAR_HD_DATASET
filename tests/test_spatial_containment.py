"""Test spatial containment functionality in ASPRS class rules."""

import pytest
import numpy as np
from ign_lidar.core.classification.asprs_class_rules import (
    ASPRSClassRulesEngine,
    WaterDetectionConfig,
)

# Skip if shapely not available
shapely = pytest.importorskip("shapely")
geopandas = pytest.importorskip("geopandas")

from shapely.geometry import Polygon, Point


class TestSpatialContainment:
    """Test the _check_spatial_containment helper method."""

    def test_basic_containment(self):
        """Test basic point-in-polygon containment."""
        # Create test data
        engine = ASPRSClassRulesEngine()

        # Create points: some inside, some outside a square
        points = np.array([
            [5.0, 5.0, 0.0],    # Inside
            [15.0, 15.0, 0.0],  # Outside
            [7.5, 7.5, 0.0],    # Inside
            [25.0, 25.0, 0.0],  # Outside
        ])

        # All points are candidates initially
        mask = np.ones(len(points), dtype=bool)

        # Create polygon (10x10 square centered at origin)
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        polygons = geopandas.GeoDataFrame(
            geometry=[polygon],
            crs="EPSG:2154"
        )

        # Check containment
        result = engine._check_spatial_containment(
            points, mask, polygons, buffer_m=0.0
        )

        # Verify: only points at (5,5) and (7.5,7.5) should be contained
        assert result[0] == True, "Point (5,5) should be inside"
        assert result[1] == False, "Point (15,15) should be outside"
        assert result[2] == True, "Point (7.5,7.5) should be inside"
        assert result[3] == False, "Point (25,25) should be outside"
        assert result.sum() == 2, "Exactly 2 points should be contained"

    def test_buffer_expansion(self):
        """Test that buffer_m expands the polygon."""
        engine = ASPRSClassRulesEngine()

        # Create point just outside a square
        points = np.array([
            [10.5, 5.0, 0.0],  # 0.5m outside right edge
        ])
        mask = np.ones(len(points), dtype=bool)

        # Create 10x10 square
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        polygons = geopandas.GeoDataFrame(
            geometry=[polygon],
            crs="EPSG:2154"
        )

        # Without buffer: should be outside
        result_no_buffer = engine._check_spatial_containment(
            points, mask, polygons, buffer_m=0.0
        )
        assert result_no_buffer[0] == False, \
            "Point should be outside without buffer"

        # With 1m buffer: should be inside
        result_with_buffer = engine._check_spatial_containment(
            points, mask, polygons, buffer_m=1.0
        )
        assert result_with_buffer[0] == True, \
            "Point should be inside with 1m buffer"

    def test_strtree_optimization(self):
        """Test that STRtree is used for multiple polygons."""
        engine = ASPRSClassRulesEngine()

        # Create many points
        points = np.array([
            [5.0, 5.0, 0.0],
            [15.0, 15.0, 0.0],
            [25.0, 25.0, 0.0],
        ])
        mask = np.ones(len(points), dtype=bool)

        # Create multiple polygons (triggers STRtree)
        polygons_list = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
        ]
        polygons = geopandas.GeoDataFrame(
            geometry=polygons_list,
            crs="EPSG:2154"
        )

        # Check containment with STRtree
        result = engine._check_spatial_containment(
            points, mask, polygons, buffer_m=0.0, use_strtree=True
        )

        # Points at (5,5) and (25,25) should be in respective polygons
        assert result[0] == True, "First point in first polygon"
        assert result[1] == False, "Second point outside both"
        assert result[2] == True, "Third point in second polygon"
        assert result.sum() == 2

    def test_mask_filtering(self):
        """Test that input mask filters which points are checked."""
        engine = ASPRSClassRulesEngine()

        points = np.array([
            [5.0, 5.0, 0.0],    # Inside polygon
            [15.0, 15.0, 0.0],  # Outside polygon
        ])

        # Only check first point
        mask = np.array([True, False])

        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        polygons = geopandas.GeoDataFrame(
            geometry=[polygon],
            crs="EPSG:2154"
        )

        result = engine._check_spatial_containment(
            points, mask, polygons, buffer_m=0.0
        )

        # First point checked and contained
        assert result[0] == True
        # Second point not checked, stays False
        assert result[1] == False
        assert result.sum() == 1

    def test_empty_polygons(self):
        """Test handling of empty polygon list."""
        engine = ASPRSClassRulesEngine()

        points = np.array([[5.0, 5.0, 0.0]])
        mask = np.ones(len(points), dtype=bool)

        # No polygons provided
        result = engine._check_spatial_containment(
            points, mask, None, buffer_m=0.0
        )

        # Mask should be unchanged
        assert np.array_equal(result, mask)

    def test_no_candidates(self):
        """Test handling when no points are candidates."""
        engine = ASPRSClassRulesEngine()

        points = np.array([[5.0, 5.0, 0.0]])
        mask = np.zeros(len(points), dtype=bool)  # No candidates

        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        polygons = geopandas.GeoDataFrame(
            geometry=[polygon],
            crs="EPSG:2154"
        )

        result = engine._check_spatial_containment(
            points, mask, polygons, buffer_m=0.0
        )

        # Result should still be all False
        assert result.sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
