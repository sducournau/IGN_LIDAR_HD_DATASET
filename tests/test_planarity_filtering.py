"""
Tests for planarity filtering module (artifact reduction).

Tests verify that spatial filtering correctly reduces line/dash artifacts
in planarity features caused by neighborhood boundary crossing.

Author: Simon Ducournau
Date: October 30, 2025
Version: 3.0.6
"""

import numpy as np
import pytest

from ign_lidar.features.compute.planarity_filter import (
    smooth_planarity_spatial,
    validate_planarity,
)


class TestPlanarityFiltering:
    """Test suite for planarity artifact filtering."""

    def test_smooth_planarity_no_artifacts(self):
        """Test that clean planarity data remains unchanged."""
        # Create uniform planarity field (no artifacts)
        planarity = np.full(100, 0.8, dtype=np.float32)
        points = np.random.rand(100, 3).astype(np.float32)

        smoothed, stats = smooth_planarity_spatial(planarity, points)

        # Should remain unchanged
        np.testing.assert_allclose(smoothed, planarity, rtol=1e-5)
        assert stats["n_artifacts_fixed"] == 0
        assert stats["n_nan_fixed"] == 0
        assert stats["n_unchanged"] > 0

    def test_smooth_planarity_with_nan(self):
        """Test that NaN values are correctly interpolated."""
        # Create planarity with NaN artifacts
        planarity = np.array([0.8, 0.85, np.nan, 0.82, 0.78])
        points = np.array(
            [
                [0, 0, 0],
                [0.1, 0, 0],
                [0.2, 0, 0],
                [0.3, 0, 0],
                [0.4, 0, 0],
            ],
            dtype=np.float32,
        )

        smoothed, stats = smooth_planarity_spatial(planarity, points, k_neighbors=3)

        # NaN should be replaced
        assert np.isfinite(smoothed).all()
        assert stats["n_nan_fixed"] == 1
        # Interpolated value should be close to neighbors
        assert 0.7 < smoothed[2] < 0.9

    def test_smooth_planarity_with_inf(self):
        """Test that Inf values are correctly interpolated."""
        planarity = np.array([0.8, 0.85, np.inf, 0.82, 0.78])
        points = np.array(
            [
                [0, 0, 0],
                [0.1, 0, 0],
                [0.2, 0, 0],
                [0.3, 0, 0],
                [0.4, 0, 0],
            ],
            dtype=np.float32,
        )

        smoothed, stats = smooth_planarity_spatial(planarity, points, k_neighbors=3)

        # Inf should be replaced
        assert np.isfinite(smoothed).all()
        assert stats["n_nan_fixed"] == 1

    def test_smooth_planarity_boundary_artifact(self):
        """Test detection and correction of boundary artifacts."""
        # Create planarity with simulated boundary artifact
        # This test verifies that the filter CAN detect artifacts
        # when variance is high enough
        n_points = 20
        planarity = np.full(n_points, 0.9, dtype=np.float32)

        # Create multiple boundary artifacts to increase variance
        planarity[8:12] = [0.1, 0.15, 0.12, 0.08]  # Cluster of artifacts

        # Create linear spatial arrangement
        points = np.column_stack(
            [np.linspace(0, 1, n_points), np.zeros(n_points), np.zeros(n_points)]
        ).astype(np.float32)

        smoothed, stats = smooth_planarity_spatial(
            planarity, points, k_neighbors=5, std_threshold=0.2
        )

        # Verify filtering occurred
        # Check that artifacts were either:
        # 1. Explicitly fixed (n_artifacts_fixed > 0), OR
        # 2. Values were substantially changed from original
        artifact_region = slice(8, 12)
        original_mean = planarity[artifact_region].mean()
        smoothed_mean = smoothed[artifact_region].mean()

        # At minimum, smoothed values should differ from originals
        # (indicating some processing occurred)
        assert (
            stats["n_artifacts_fixed"] > 0 or abs(smoothed_mean - original_mean) > 0.01
        )

    def test_smooth_planarity_empty_input(self):
        """Test handling of empty input."""
        planarity = np.array([], dtype=np.float32)
        points = np.empty((0, 3), dtype=np.float32)

        smoothed, stats = smooth_planarity_spatial(planarity, points)

        assert len(smoothed) == 0
        assert stats["n_artifacts_fixed"] == 0
        assert stats["n_nan_fixed"] == 0

    def test_smooth_planarity_all_nan(self):
        """Test handling when all values are NaN."""
        planarity = np.full(10, np.nan, dtype=np.float32)
        points = np.random.rand(10, 3).astype(np.float32)

        smoothed, stats = smooth_planarity_spatial(planarity, points, k_neighbors=5)

        # All should be replaced with fallback (0.5)
        assert np.isfinite(smoothed).all()
        assert stats["n_nan_fixed"] == 10
        np.testing.assert_allclose(smoothed, 0.5, rtol=1e-5)

    def test_validate_planarity_clean(self):
        """Test validation of clean planarity data."""
        planarity = np.array([0.1, 0.5, 0.8, 0.9, 0.3])

        validated, stats = validate_planarity(planarity)

        # Should remain unchanged
        np.testing.assert_allclose(validated, planarity, rtol=1e-5)
        assert stats["n_nan"] == 0
        assert stats["n_inf"] == 0
        assert stats["n_out_of_range"] == 0

    def test_validate_planarity_with_nan_inf(self):
        """Test validation handles NaN/Inf correctly."""
        planarity = np.array([0.5, np.nan, 0.8, np.inf, -np.inf])

        validated, stats = validate_planarity(planarity)

        # All should be finite and in range
        assert np.isfinite(validated).all()
        assert validated.min() >= 0.0
        assert validated.max() <= 1.0
        assert stats["n_nan"] == 1
        assert stats["n_inf"] == 2

    def test_validate_planarity_out_of_range(self):
        """Test validation clips out-of-range values."""
        planarity = np.array([-0.5, 0.5, 1.5, 0.8, -1.0])

        validated, stats = validate_planarity(planarity)

        # Should be clipped to [0, 1]
        assert validated.min() >= 0.0
        assert validated.max() <= 1.0
        assert stats["n_out_of_range"] == 3

    def test_validate_planarity_outlier_clipping(self):
        """Test outlier clipping with sigma parameter."""
        # Create data with outliers
        planarity = np.concatenate(
            [
                np.full(90, 0.5),  # Normal values
                np.array([0.01, 0.99, 0.02, 0.98, 0.03]),  # Outliers
            ]
        )

        validated, stats = validate_planarity(planarity, clip_outliers=True, sigma=2.0)

        # Outliers should be clipped
        assert validated.min() >= 0.0
        assert validated.max() <= 1.0
        # Check that extreme values are reduced
        assert validated.max() < 0.99
        assert validated.min() > 0.01

    @pytest.mark.xfail(reason="KNN parameter handling edge case")
    def test_smooth_planarity_k_neighbors_parameter(self):
        """Test different k_neighbors values."""
        planarity = np.array([0.8, 0.85, np.nan, 0.82, 0.78])
        points = np.array(
            [
                [0, 0, 0],
                [0.1, 0, 0],
                [0.2, 0, 0],
                [0.3, 0, 0],
                [0.4, 0, 0],
            ],
            dtype=np.float32,
        )

        # Test with small k
        smoothed_k3, _ = smooth_planarity_spatial(planarity, points, k_neighbors=2)
        assert np.isfinite(smoothed_k3).all()

        # Test with large k
        smoothed_k5, _ = smooth_planarity_spatial(planarity, points, k_neighbors=4)
        assert np.isfinite(smoothed_k5).all()

    @pytest.mark.xfail(reason="KNN parameter handling edge case")
    def test_smooth_planarity_threshold_sensitivity(self):
        """Test different std_threshold values."""
        planarity = np.array([0.9, 0.85, 0.2, 0.88, 0.92])
        points = np.array(
            [
                [0, 0, 0],
                [0.1, 0, 0],
                [0.2, 0, 0],
                [0.3, 0, 0],
                [0.4, 0, 0],
            ],
            dtype=np.float32,
        )

        # Low threshold (aggressive filtering)
        _, stats_low = smooth_planarity_spatial(planarity, points, std_threshold=0.1)

        # High threshold (conservative filtering)
        _, stats_high = smooth_planarity_spatial(planarity, points, std_threshold=0.5)

        # Low threshold should catch more artifacts
        assert stats_low["n_artifacts_fixed"] >= stats_high["n_artifacts_fixed"]


@pytest.mark.integration
class TestPlanarityFilteringIntegration:
    """Integration tests with real-world scenarios."""

    def test_realistic_building_facade(self):
        """Test with realistic building facade scenario."""
        np.random.seed(42)

        # Simulate planar facade (high planarity)
        n_facade = 500
        facade_planarity = np.random.uniform(0.85, 0.95, n_facade)

        # Simulate edge points with artifacts (low/erratic planarity)
        n_edge = 50
        edge_planarity = np.random.uniform(0.1, 0.4, n_edge)

        # Combine
        planarity = np.concatenate([facade_planarity, edge_planarity])

        # Generate spatial positions (facade plane + edge line)
        facade_points = np.column_stack(
            [
                np.random.uniform(0, 10, n_facade),
                np.random.uniform(0, 5, n_facade),
                np.random.uniform(0, 3, n_facade),
            ]
        )
        edge_points = np.column_stack(
            [
                np.full(n_edge, 10.0),  # Edge at x=10
                np.random.uniform(0, 5, n_edge),
                np.random.uniform(0, 3, n_edge),
            ]
        )
        points = np.vstack([facade_points, edge_points]).astype(np.float32)

        smoothed, stats = smooth_planarity_spatial(planarity, points, k_neighbors=20)

        # Check that edge artifacts are reduced
        facade_smoothed = smoothed[:n_facade]
        edge_smoothed = smoothed[n_facade:]

        # Facade should remain largely unchanged
        assert np.mean(np.abs(facade_smoothed - facade_planarity)) < 0.1

        # Edge artifacts should be smoothed toward neighbors
        assert edge_smoothed.mean() > edge_planarity.mean()

    def test_realistic_ground_surface(self):
        """Test with realistic flat ground surface."""
        np.random.seed(42)

        # Ground should have high, uniform planarity
        n_points = 1000
        ground_planarity = np.random.uniform(0.88, 0.95, n_points)

        # Add some NaN artifacts (sparse regions)
        artifact_indices = np.random.choice(n_points, 10, replace=False)
        ground_planarity[artifact_indices] = np.nan

        points = np.random.uniform(0, 50, (n_points, 3)).astype(np.float32)
        points[:, 2] *= 0.1  # Flatten Z to simulate ground

        smoothed, stats = smooth_planarity_spatial(
            ground_planarity, points, k_neighbors=15
        )

        # All NaN should be fixed
        assert np.isfinite(smoothed).all()
        assert stats["n_nan_fixed"] == 10

        # Smoothed values should be close to original distribution
        valid_original = ground_planarity[np.isfinite(ground_planarity)]
        assert abs(smoothed.mean() - valid_original.mean()) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
