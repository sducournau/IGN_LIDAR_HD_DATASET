"""
Test suite for reclassification improvements - November 1, 2025.

Tests for:
1. Improved building facade capture (with overhangs)
2. Strict road ground-level enforcement
3. Enhanced vegetation detection with NDVI
"""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Polygon

from ign_lidar.classification_schema import ASPRSClass
from ign_lidar.core.classification.ground_truth_refinement import (
    GroundTruthRefiner,
    GroundTruthRefinementConfig,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return GroundTruthRefinementConfig()


@pytest.fixture
def refiner(config):
    """Create GroundTruthRefiner instance."""
    return GroundTruthRefiner(config)


@pytest.fixture
def building_polygon():
    """Create a simple building polygon (10x10m)."""
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    return gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:2154")


class TestBuildingFacadeCapture:
    """Test improved facade capture with height stratification."""

    def test_low_facade_capture(self, refiner, building_polygon):
        """Test capture of low facade points (foundations, low walls)."""
        # Create points: 2 inside building, 2 at edges (facades)
        points = np.array(
            [
                [5.0, 5.0, 0.0],  # Inside, low
                [5.0, 5.0, 2.0],  # Inside, high (roof)
                [0.1, 5.0, 0.0],  # Edge, very low (foundation)
                [0.1, 5.0, 0.5],  # Edge, low (facade)
            ]
        )

        labels = np.array(
            [
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
            ]
        )

        # Height values
        height = np.array([0.2, 3.0, 0.15, 0.4])  # Low facades: 0.15m, 0.4m

        # Verticality (facades are vertical)
        verticality = np.array([0.2, 0.1, 0.45, 0.50])

        # Planarity (roofs are planar)
        planarity = np.array([0.3, 0.85, 0.2, 0.3])

        # NDVI (not vegetation)
        ndvi = np.array([0.1, 0.1, 0.1, 0.1])

        refined, stats = refiner.refine_building_with_expanded_polygons(
            labels=labels,
            points=points,
            building_polygons=building_polygon,
            height=height,
            planarity=planarity,
            verticality=verticality,
            ndvi=ndvi,
        )

        # Check that low facades were captured (0.15m and 0.4m heights)
        assert stats["facades_captured"] >= 2, "Should capture low facades"
        assert (
            np.sum(refined == ASPRSClass.BUILDING) >= 2
        ), "Should classify facades as building"

    def test_overhang_detection(self, refiner, building_polygon):
        """Test detection of roof overhangs extending beyond footprint."""
        # Create points just outside building polygon (overhangs)
        points = np.array(
            [
                [5.0, 5.0, 4.0],  # Inside, high
                [-0.3, 5.0, 3.5],  # Outside, high (overhang)
                [10.3, 5.0, 3.8],  # Outside, high (overhang)
            ]
        )

        labels = np.array(
            [
                ASPRSClass.BUILDING,
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
            ]
        )

        # Height values (elevated)
        height = np.array([4.0, 3.5, 3.8])

        # Mixed planarity/verticality (sloped overhangs)
        planarity = np.array([0.80, 0.55, 0.60])
        verticality = np.array([0.1, 0.45, 0.40])

        # NDVI (not vegetation)
        ndvi = np.array([0.1, 0.1, 0.1])

        refined, stats = refiner.refine_building_with_expanded_polygons(
            labels=labels,
            points=points,
            building_polygons=building_polygon,
            height=height,
            planarity=planarity,
            verticality=verticality,
            ndvi=ndvi,
        )

        # Check that overhangs were detected (if enabled)
        if refiner.config.OVERHANG_DETECTION_ENABLED:
            assert stats.get("overhangs_captured", 0) > 0, "Should detect overhangs"


class TestRoadGroundLevelEnforcement:
    """Test strict ground-level enforcement for roads."""

    def test_elevated_points_reclassified(self, refiner):
        """Test that elevated points above roads are reclassified."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )

        # All initially marked as road from ground truth
        labels = np.array(
            [
                ASPRSClass.ROAD_SURFACE,
                ASPRSClass.ROAD_SURFACE,
                ASPRSClass.ROAD_SURFACE,
                ASPRSClass.ROAD_SURFACE,
            ]
        )

        road_mask = np.ones(4, dtype=bool)

        # Heights: ground level and elevated
        height = np.array([0.1, 0.2, 3.5, 8.0])  # Last 2 are elevated

        # Planarity (all flat)
        planarity = np.array([0.90, 0.88, 0.75, 0.60])

        # NDVI: low for roads, high for elevated vegetation
        ndvi = np.array([0.05, 0.08, 0.35, 0.60])

        refined, stats = refiner.refine_road_classification(
            labels=labels,
            points=points,
            road_mask=road_mask,
            height=height,
            planarity=planarity,
            ndvi=ndvi,
        )

        # Check that elevated points were reclassified
        assert refined[2] != ASPRSClass.ROAD_SURFACE, "Point at 3.5m should not be road"
        assert refined[3] != ASPRSClass.ROAD_SURFACE, "Point at 8m should not be road"

        # Check that they were reclassified as vegetation (due to NDVI)
        assert refined[3] in [
            ASPRSClass.HIGH_VEGETATION,
            ASPRSClass.MEDIUM_VEGETATION,
        ], "High elevated point should be vegetation"

        # Check stats
        assert (
            stats["elevated_to_vegetation"] > 0
        ), "Should have reclassified to vegetation"

    def test_ground_level_roads_preserved(self, refiner):
        """Test that true ground-level roads are preserved."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        labels = np.array(
            [
                ASPRSClass.ROAD_SURFACE,
                ASPRSClass.ROAD_SURFACE,
            ]
        )

        road_mask = np.ones(2, dtype=bool)

        # Heights: very low (true road surface)
        height = np.array([0.05, 0.15])

        # Planarity: high (flat road)
        planarity = np.array([0.92, 0.90])

        # NDVI: low (asphalt)
        ndvi = np.array([0.05, 0.08])

        refined, stats = refiner.refine_road_classification(
            labels=labels,
            points=points,
            road_mask=road_mask,
            height=height,
            planarity=planarity,
            ndvi=ndvi,
        )

        # All should remain as road
        assert np.all(
            refined == ASPRSClass.ROAD_SURFACE
        ), "Ground-level roads should be preserved"
        assert stats["road_validated"] == 2, "Should validate all ground-level points"


class TestVegetationNDVI:
    """Test enhanced vegetation detection with NDVI."""

    def test_vegetation_confidence_scoring(self, refiner):
        """Test multi-feature vegetation confidence scoring."""
        labels = np.array(
            [
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
            ]
        )

        # NDVI: vegetation has high NDVI (increased to ensure detection)
        ndvi = np.array([0.75, 0.70, 0.65, 0.10])

        # Height: different vegetation levels
        height = np.array([0.3, 1.5, 6.0, 0.2])

        # Curvature: vegetation has high curvature
        curvature = np.array([0.12, 0.15, 0.15, 0.02])

        # Planarity: vegetation has low planarity
        planarity = np.array([0.25, 0.20, 0.15, 0.85])

        refined, stats = refiner.refine_vegetation_with_features(
            labels=labels,
            ndvi=ndvi,
            height=height,
            curvature=curvature,
            planarity=planarity,
        )

        # Check that high-NDVI points were classified as vegetation
        # (First 3 points should have high confidence scores > 0.65)
        assert (
            refined[0] == ASPRSClass.LOW_VEGETATION
        ), "High NDVI + low height = low veg"
        assert (
            refined[1] == ASPRSClass.MEDIUM_VEGETATION
        ), "High NDVI + medium height = medium veg"
        assert (
            refined[2] == ASPRSClass.HIGH_VEGETATION
        ), "High NDVI + high height = high veg"

        # Low NDVI point should remain unclassified
        assert (
            refined[3] == ASPRSClass.UNCLASSIFIED
        ), "Low NDVI should not be vegetation"

    def test_nan_handling_robustness(self, refiner):
        """Test robust handling of NaN/Inf in features."""
        labels = np.array(
            [
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
            ]
        )

        # NDVI with NaN
        ndvi = np.array([0.60, np.nan, 0.55])

        # Height with Inf
        height = np.array([1.0, 2.0, np.inf])

        # Planarity with NaN
        planarity = np.array([0.30, np.nan, 0.25])

        # Should not crash and should handle gracefully
        refined, stats = refiner.refine_vegetation_with_features(
            labels=labels,
            ndvi=ndvi,
            height=height,
            planarity=planarity,
        )

        # Check that NaN/Inf didn't cause issues
        assert np.all(np.isfinite(refined)), "Classification should not produce NaN/Inf"
        assert stats["vegetation_added"] >= 0, "Should produce valid stats"


class TestFacadeRecovery:
    """Test aggressive facade recovery pass."""

    def test_recover_missing_low_walls(self, refiner, building_polygon):
        """Test recovery of very low walls and foundations."""
        points = np.array(
            [
                [0.1, 5.0, 0.0],  # Near building, very low
                [0.2, 5.0, 0.0],  # Near building, low
                [50.0, 50.0, 0.0],  # Far from building
            ]
        )

        labels = np.array(
            [
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
                ASPRSClass.UNCLASSIFIED,
            ]
        )

        # Heights: very low
        height = np.array([0.15, 0.3, 0.2])

        # Verticality: moderate
        verticality = np.array([0.35, 0.40, 0.30])

        refined, stats = refiner.recover_missing_facades(
            labels=labels,
            points=points,
            building_polygons=building_polygon,
            height=height,
            verticality=verticality,
        )

        # Check that near-building points were recovered
        assert refined[0] == ASPRSClass.BUILDING, "Should recover point near building"
        assert refined[1] == ASPRSClass.BUILDING, "Should recover point near building"

        # Far point should remain unclassified
        assert (
            refined[2] == ASPRSClass.UNCLASSIFIED
        ), "Should not classify point far from building"

        # Check stats
        assert stats["facades_recovered"] >= 2, "Should recover at least 2 facades"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
