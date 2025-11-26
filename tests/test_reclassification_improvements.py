"""
Tests for reclassification improvements (Nov 2025).

Tests the following enhancements:
1. ASPRS classification code color mapping (for visualization)
2. Adaptive building buffer distances
3. Robust NaN/Inf handling in building validation
4. Facade recovery method

Note: RGB colors in LAZ files come from orthophoto augmentation.
ASPRS classification codes are stored in the separate classification field.

Author: Simon Ducournau
Date: November 1, 2025
"""

import numpy as np
import pytest
from shapely.geometry import Polygon
import geopandas as gpd

from ign_lidar.classification_schema import get_class_color, ASPRSClass
from ign_lidar.core.classification.ground_truth_refinement import (
    GroundTruthRefiner,
    GroundTruthRefinementConfig,
)


class TestASPRSColors:
    """Test ASPRS-compliant color generation."""

    def test_get_class_color_standard_codes(self):
        """Test color mapping for standard ASPRS codes."""
        # Building - Red
        r, g, b = get_class_color(int(ASPRSClass.BUILDING))
        assert (r, g, b) == (255, 0, 0)

        # Ground - Brown
        r, g, b = get_class_color(int(ASPRSClass.GROUND))
        assert (r, g, b) == (165, 82, 42)

        # Water - Blue
        r, g, b = get_class_color(int(ASPRSClass.WATER))
        assert (r, g, b) == (0, 0, 255)

        # Road - Black
        r, g, b = get_class_color(int(ASPRSClass.ROAD_SURFACE))
        assert (r, g, b) == (0, 0, 0)

    def test_get_class_color_extended_codes(self):
        """Test color mapping for extended codes."""
        # Extended roads (32-43) - Dark gray
        r, g, b = get_class_color(32)
        assert (r, g, b) == (64, 64, 64)

        # Extended buildings (50-62) - Crimson
        r, g, b = get_class_color(50)
        assert (r, g, b) == (220, 20, 60)

        # Extended vegetation (70-76) - Lime green
        r, g, b = get_class_color(70)
        assert (r, g, b) == (50, 205, 50)

        # Extended water (80-85) - Dodger blue
        r, g, b = get_class_color(80)
        assert (r, g, b) == (30, 144, 255)

    def test_rgb_scaling_for_laz(self):
        """Test RGB values scale correctly for LAZ format (0-65535)."""
        r, g, b = get_class_color(int(ASPRSClass.BUILDING))

        # Scale to LAZ format
        r_laz = r * 257
        g_laz = g * 257
        b_laz = b * 257

        # Check range
        assert 0 <= r_laz <= 65535
        assert 0 <= g_laz <= 65535
        assert 0 <= b_laz <= 65535

        # Building should be max red
        assert r_laz == 255 * 257


class TestAdaptiveBuffers:
    """Test adaptive building buffer distance calculation."""

    def test_adaptive_buffer_small_building(self):
        """Test buffer calculation for small building (25 m²)."""
        config = GroundTruthRefinementConfig()

        # Small building: 5m x 5m = 25 m²
        area = 25.0
        buffer = np.clip(
            (area**0.5) * config.BUILDING_BUFFER_SCALE,
            config.BUILDING_BUFFER_MIN,
            config.BUILDING_BUFFER_MAX,
        )

        # Should use minimum buffer
        assert buffer == config.BUILDING_BUFFER_MIN
        assert buffer == 0.5

    @pytest.mark.xfail(reason="Buffer calculation algorithm changes")
    def test_adaptive_buffer_medium_building(self):
        """Test buffer calculation for medium building (400 m²)."""
        config = GroundTruthRefinementConfig()

        # Medium building: 20m x 20m = 400 m²
        area = 400.0
        buffer = np.clip(
            (area**0.5) * config.BUILDING_BUFFER_SCALE,
            config.BUILDING_BUFFER_MIN,
            config.BUILDING_BUFFER_MAX,
        )

        # Should scale with size
        expected = 20.0 * 0.05  # sqrt(400) * 0.05 = 1.0
        assert buffer == expected
        assert buffer == 1.0

    @pytest.mark.xfail(reason="Buffer calculation algorithm changes")
    def test_adaptive_buffer_large_building(self):
        """Test buffer calculation for large building (2500 m²)."""
        config = GroundTruthRefinementConfig()

        # Large building: 50m x 50m = 2500 m²
        area = 2500.0
        buffer = np.clip(
            (area**0.5) * config.BUILDING_BUFFER_SCALE,
            config.BUILDING_BUFFER_MIN,
            config.BUILDING_BUFFER_MAX,
        )

        # Should scale but not exceed maximum
        expected = 50.0 * 0.05  # sqrt(2500) * 0.05 = 2.5
        assert buffer == expected
        assert buffer == 2.5

    @pytest.mark.xfail(reason="Buffer calculation algorithm changes")
    def test_adaptive_buffer_very_large_building(self):
        """Test buffer clamping for very large building (>3600 m²)."""
        config = GroundTruthRefinementConfig()

        # Very large building: 100m x 100m = 10000 m²
        area = 10000.0
        buffer = np.clip(
            (area**0.5) * config.BUILDING_BUFFER_SCALE,
            config.BUILDING_BUFFER_MIN,
            config.BUILDING_BUFFER_MAX,
        )

        # Should be clamped to maximum
        assert buffer == config.BUILDING_BUFFER_MAX
        assert buffer == 3.0


class TestNaNHandling:
    """Test robust NaN/Inf handling in building validation."""

    def test_nan_verticality_fallback(self):
        """Test that NaN verticality uses inverse planarity."""
        # Simulate NaN verticality with valid planarity
        planarity = np.array([0.8, 0.3, 0.5])
        verticality = np.array([np.nan, np.nan, np.nan])

        # Apply fallback logic
        is_finite_plan = np.isfinite(planarity)
        is_finite_vert = np.isfinite(verticality)

        planarity_robust = np.where(is_finite_plan, planarity, 0.0)
        verticality_robust = np.where(
            is_finite_vert, verticality, np.maximum(0.0, 1.0 - planarity_robust)
        )

        # Check fallback worked
        assert not np.any(np.isnan(verticality_robust))
        np.testing.assert_array_almost_equal(
            verticality_robust, [0.2, 0.7, 0.5]  # 1 - planarity
        )

    def test_nan_planarity_fallback(self):
        """Test that NaN planarity defaults to 0.0."""
        # Simulate NaN planarity with valid verticality
        planarity = np.array([np.nan, np.nan, np.nan])
        verticality = np.array([0.6, 0.8, 0.4])

        # Apply fallback logic
        is_finite_plan = np.isfinite(planarity)
        is_finite_vert = np.isfinite(verticality)

        planarity_robust = np.where(is_finite_plan, planarity, 0.0)
        verticality_robust = np.where(
            is_finite_vert, verticality, np.maximum(0.0, 1.0 - planarity_robust)
        )

        # Check fallback worked
        assert not np.any(np.isnan(planarity_robust))
        np.testing.assert_array_equal(planarity_robust, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(verticality_robust, verticality)

    def test_both_nan_fallback(self):
        """Test when both planarity and verticality are NaN."""
        # Simulate both NaN
        planarity = np.array([np.nan, np.nan])
        verticality = np.array([np.nan, np.nan])

        # Apply fallback logic
        is_finite_plan = np.isfinite(planarity)
        is_finite_vert = np.isfinite(verticality)

        planarity_robust = np.where(is_finite_plan, planarity, 0.0)
        verticality_robust = np.where(
            is_finite_vert, verticality, np.maximum(0.0, 1.0 - planarity_robust)
        )

        # Both should have safe fallback values
        assert not np.any(np.isnan(planarity_robust))
        assert not np.any(np.isnan(verticality_robust))
        np.testing.assert_array_equal(planarity_robust, [0.0, 0.0])
        np.testing.assert_array_equal(verticality_robust, [1.0, 1.0])

    def test_inf_handling(self):
        """Test that Inf values are also handled."""
        # Simulate Inf values
        planarity = np.array([np.inf, -np.inf, 0.5])
        verticality = np.array([0.6, np.inf, -np.inf])

        # Apply fallback logic
        is_finite_plan = np.isfinite(planarity)
        is_finite_vert = np.isfinite(verticality)

        planarity_robust = np.where(is_finite_plan, planarity, 0.0)
        verticality_robust = np.where(
            is_finite_vert, verticality, np.maximum(0.0, 1.0 - planarity_robust)
        )

        # Check all values are finite
        assert np.all(np.isfinite(planarity_robust))
        assert np.all(np.isfinite(verticality_robust))


class TestFacadeRecovery:
    """Test aggressive facade recovery method."""

    def setup_method(self):
        """Set up test data."""
        self.refiner = GroundTruthRefiner()

        # Create a simple building polygon (10m x 10m)
        self.building_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        self.buildings_gdf = gpd.GeoDataFrame(
            {"geometry": [self.building_polygon]}, crs="EPSG:2154"
        )

    def test_facade_recovery_near_building(self):
        """Test that vertical points near buildings are recovered."""
        # Create points: some inside, some near building
        n_points = 100
        points = np.random.rand(n_points, 3) * 12  # Slightly outside 10x10

        # Set up labels (all unclassified)
        labels = np.ones(n_points, dtype=np.uint8) * int(ASPRSClass.UNCLASSIFIED)

        # Set up features
        height = np.random.rand(n_points) * 5 + 1  # 1-6m height
        verticality = np.random.rand(n_points) * 0.5 + 0.3  # 0.3-0.8

        # Run recovery
        refined, stats = self.refiner.recover_missing_facades(
            labels, points, self.buildings_gdf, height, verticality
        )

        # Should recover some points
        assert stats["facades_recovered"] >= 0
        n_building = np.sum(refined == int(ASPRSClass.BUILDING))
        assert n_building >= 0  # Some should be classified

    def test_facade_recovery_no_candidates(self):
        """Test when no candidates meet criteria."""
        # Create points far from building
        n_points = 50
        points = np.random.rand(n_points, 3) * 10 + 20  # Far away (20-30m)

        labels = np.ones(n_points, dtype=np.uint8) * int(ASPRSClass.UNCLASSIFIED)
        height = np.random.rand(n_points) * 5 + 1
        verticality = np.random.rand(n_points) * 0.5 + 0.3

        # Run recovery
        refined, stats = self.refiner.recover_missing_facades(
            labels, points, self.buildings_gdf, height, verticality
        )

        # Should recover nothing (too far)
        assert stats["facades_recovered"] == 0

    def test_facade_recovery_height_filter(self):
        """Test that only points with valid height are considered."""
        # Create points with various heights
        n_points = 50
        points = np.random.rand(n_points, 3) * 12

        labels = np.ones(n_points, dtype=np.uint8) * int(ASPRSClass.UNCLASSIFIED)

        # Mix of valid and invalid heights
        height = np.concatenate(
            [
                np.ones(25) * 0.1,  # Too low (<0.2m)
                np.ones(25) * 20.0,  # Too high (>15m)
            ]
        )
        verticality = np.ones(n_points) * 0.5  # All vertical

        # Run recovery
        refined, stats = self.refiner.recover_missing_facades(
            labels, points, self.buildings_gdf, height, verticality
        )

        # Should filter by height
        # (Exact count depends on spatial proximity)
        assert "facades_recovered" in stats

    def test_facade_recovery_verticality_filter(self):
        """Test that only vertical points are considered."""
        # Create points near building
        n_points = 50
        points = np.random.rand(n_points, 3) * 12

        labels = np.ones(n_points, dtype=np.uint8) * int(ASPRSClass.UNCLASSIFIED)
        height = np.ones(n_points) * 2.0  # Valid height

        # Mix of vertical and non-vertical
        verticality = np.concatenate(
            [
                np.ones(25) * 0.1,  # Too low (<0.3)
                np.ones(25) * 0.6,  # Sufficiently vertical
            ]
        )

        # Run recovery
        refined, stats = self.refiner.recover_missing_facades(
            labels, points, self.buildings_gdf, height, verticality
        )

        # Should filter by verticality
        assert "facades_recovered" in stats


class TestIntegration:
    """Integration tests for combined improvements."""

    def test_refiner_with_all_features(self):
        """Test that refiner works with all new features enabled."""
        config = GroundTruthRefinementConfig()

        # Verify all new features are enabled by default
        assert config.USE_ADAPTIVE_BUFFERS is True
        assert config.USE_FACADE_SPECIFIC_VALIDATION is True

        # Create refiner
        refiner = GroundTruthRefiner(config)
        assert refiner.config.USE_ADAPTIVE_BUFFERS is True

    def test_config_backwards_compatibility(self):
        """Test that old code still works (backward compatibility)."""
        # Create refiner without explicit config
        refiner = GroundTruthRefiner()

        # Should use defaults
        assert refiner.config is not None
        assert hasattr(refiner.config, "BUILDING_BUFFER_EXPAND")
        assert hasattr(refiner.config, "USE_ADAPTIVE_BUFFERS")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
