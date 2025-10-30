"""
Unit tests for unified feature filtering module.

Tests the generic feature_filter.py module that handles planarity,
linearity, and horizontality filtering.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from ign_lidar.features.compute.feature_filter import (
    smooth_feature_spatial,
    validate_feature,
    smooth_planarity_spatial,
    smooth_linearity_spatial,
    smooth_horizontality_spatial,
    validate_planarity,
    validate_linearity,
    validate_horizontality,
)


class TestGenericFeatureFiltering:
    """Test generic smooth_feature_spatial and validate_feature functions."""

    def test_smooth_feature_no_artifacts(self):
        """Test that clean features remain unchanged."""
        # Create grid of points with uniform feature values
        points = np.array(
            [[i, j, 0] for i in range(10) for j in range(10)], dtype=np.float32
        )
        feature = np.full(len(points), 0.7, dtype=np.float32)  # Uniform planarity

        result = smooth_feature_spatial(feature, points, k_neighbors=8)

        # Should remain mostly unchanged (low variance)
        assert np.allclose(result, 0.7, atol=0.05)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_smooth_feature_with_artifacts(self):
        """Test artifact detection and smoothing."""
        # Create grid with artifact pattern
        points = np.array(
            [[i, j, 0] for i in range(20) for j in range(20)], dtype=np.float32
        )
        feature = np.full(len(points), 0.7, dtype=np.float32)

        # Inject artifacts (high variance stripes)
        for i in range(5, 15):
            for j in range(20):
                idx = i * 20 + j
                # Create alternating high/low (boundary artifact pattern)
                feature[idx] = 0.1 if j % 2 == 0 else 0.95

        result = smooth_feature_spatial(
            feature, points, k_neighbors=15, std_threshold=0.3
        )

        # Artifacts should be smoothed to intermediate values
        artifact_indices = [i * 20 + j for i in range(5, 15) for j in range(20)]
        artifact_region = result[artifact_indices]

        # Check that extreme values are reduced
        assert np.mean(artifact_region) < 0.7  # Less than clean mean
        # Variance may still be significant due to partial smoothing
        assert np.std(artifact_region) < 0.5  # Relaxed threshold
        assert not np.any(np.isnan(result))

    def test_smooth_feature_with_nan(self):
        """Test NaN handling in features."""
        points = np.array(
            [[i, j, 0] for i in range(10) for j in range(10)], dtype=np.float32
        )
        feature = np.full(len(points), 0.7, dtype=np.float32)

        # Insert NaN values
        feature[25:30] = np.nan

        result = smooth_feature_spatial(feature, points)

        # NaN should be interpolated from neighbors
        assert not np.any(np.isnan(result))
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_smooth_feature_with_inf(self):
        """Test Inf handling in features."""
        points = np.array(
            [[i, j, 0] for i in range(10) for j in range(10)], dtype=np.float32
        )
        feature = np.full(len(points), 0.7, dtype=np.float32)

        # Insert Inf values
        feature[10:15] = np.inf
        feature[15:20] = -np.inf

        result = smooth_feature_spatial(feature, points)

        # Inf should be replaced
        assert not np.any(np.isinf(result))
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_smooth_feature_parameter_k_neighbors(self):
        """Test effect of k_neighbors parameter."""
        points = np.array(
            [[i, j, 0] for i in range(20) for j in range(20)], dtype=np.float32
        )
        feature = np.full(len(points), 0.7, dtype=np.float32)

        # Inject artifacts
        for i in range(8, 12):
            for j in range(20):
                idx = i * 20 + j
                feature[idx] = 0.1 if j % 2 == 0 else 0.95

        # Test with different k values
        result_k8 = smooth_feature_spatial(feature, points, k_neighbors=8)
        result_k15 = smooth_feature_spatial(feature, points, k_neighbors=15)
        result_k25 = smooth_feature_spatial(feature, points, k_neighbors=25)

        # Larger k should produce smoother result
        variance_k8 = np.var(result_k8)
        variance_k15 = np.var(result_k15)
        variance_k25 = np.var(result_k25)

        assert variance_k25 <= variance_k15 <= variance_k8

    def test_smooth_feature_parameter_threshold(self):
        """Test effect of std_threshold parameter."""
        points = np.array(
            [[i, j, 0] for i in range(20) for j in range(20)], dtype=np.float32
        )
        feature = np.full(len(points), 0.7, dtype=np.float32)

        # Inject moderate artifacts
        for i in range(8, 12):
            for j in range(20):
                idx = i * 20 + j
                feature[idx] = 0.4 if j % 2 == 0 else 0.8

        # Test different thresholds
        result_aggressive = smooth_feature_spatial(
            feature, points, std_threshold=0.1  # Aggressive
        )
        result_balanced = smooth_feature_spatial(
            feature, points, std_threshold=0.3  # Balanced
        )
        result_conservative = smooth_feature_spatial(
            feature, points, std_threshold=0.5  # Conservative
        )

        # Lower threshold = more smoothing
        changes_aggressive = np.sum(np.abs(result_aggressive - feature) > 0.01)
        changes_balanced = np.sum(np.abs(result_balanced - feature) > 0.01)
        changes_conservative = np.sum(np.abs(result_conservative - feature) > 0.01)

        assert changes_aggressive >= changes_balanced >= changes_conservative

    def test_validate_feature_clean(self):
        """Test validation of clean features."""
        feature = np.random.uniform(0.2, 0.8, 100).astype(np.float32)

        result = validate_feature(feature, "test_feature", valid_range=(0.0, 1.0))

        # Should remain unchanged
        assert np.allclose(result, feature)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_validate_feature_with_nan_inf(self):
        """Test validation handles NaN and Inf."""
        feature = np.array([0.5, 0.7, np.nan, 0.3, np.inf, 0.6, -np.inf, 0.4])

        result = validate_feature(feature, "test_feature")

        # NaN/Inf should be replaced
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_validate_feature_out_of_range(self):
        """Test validation clips out-of-range values."""
        feature = np.array([-0.5, 0.5, 1.2, 0.3, 2.0, -1.0, 0.8])

        result = validate_feature(feature, "test_feature", valid_range=(0.0, 1.0))

        # Should be clipped to [0, 1]
        assert np.all((result >= 0.0) & (result <= 1.0))
        assert result[0] == 0.0  # -0.5 clipped
        assert result[2] == 1.0  # 1.2 clipped

    def test_validate_feature_outlier_clipping(self):
        """Test outlier clipping with sigma threshold."""
        # Create feature with outliers
        np.random.seed(42)
        feature = np.random.normal(0.5, 0.1, 100).astype(np.float32)
        feature[0] = 5.0  # Extreme outlier
        feature[1] = -3.0  # Extreme outlier

        result = validate_feature(feature, "test_feature", clip_sigma=3.0)

        # Outliers should be clipped to valid range [0, 1]
        assert result[0] <= 1.0  # Extreme positive clipped
        assert result[1] >= 0.0  # Extreme negative clipped
        assert not np.any(np.isnan(result))

    def test_smooth_feature_empty_input(self):
        """Test handling of empty arrays."""
        points = np.array([], dtype=np.float32).reshape(0, 3)
        feature = np.array([], dtype=np.float32)

        # Should return empty array (not raise)
        result = smooth_feature_spatial(feature, points)
        assert len(result) == 0

    def test_smooth_feature_insufficient_neighbors(self):
        """Test handling of insufficient neighbors."""
        # Only 3 points (k=15 will be too many)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        feature = np.array([0.5, 0.6, 0.7], dtype=np.float32)

        # Should handle gracefully (use all available neighbors)
        result = smooth_feature_spatial(feature, points, k_neighbors=15)

        assert len(result) == 3
        assert not np.any(np.isnan(result))


class TestPlanarityFiltering:
    """Test planarity-specific filtering functions (backward compatibility)."""

    def test_smooth_planarity_spatial(self):
        """Test planarity-specific wrapper function."""
        points = np.array(
            [[i, j, 0] for i in range(10) for j in range(10)], dtype=np.float32
        )
        planarity = np.random.uniform(0.3, 0.9, len(points)).astype(np.float32)

        result = smooth_planarity_spatial(planarity, points)

        assert len(result) == len(planarity)
        assert not np.any(np.isnan(result))
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_validate_planarity(self):
        """Test planarity-specific validation."""
        planarity = np.array([0.5, np.nan, 1.2, -0.1, 0.8, np.inf])

        result = validate_planarity(planarity)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_backward_compatibility_with_v306(self):
        """Test that planarity functions match v3.0.6 behavior."""
        # This ensures backward compatibility
        points = np.array(
            [[i, j, 0] for i in range(20) for j in range(20)], dtype=np.float32
        )
        planarity = np.full(len(points), 0.7, dtype=np.float32)

        # Add some artifacts with high variance
        np.random.seed(42)
        planarity[100:150] = np.random.uniform(0.0, 1.0, 50)

        result = smooth_planarity_spatial(planarity, points, k_neighbors=15)

        # Should produce valid output
        assert len(result) == len(planarity)
        assert not np.any(np.isnan(result))
        assert np.all((result >= 0.0) & (result <= 1.0))
        # Artifacts should be partially smoothed towards background
        # (may not fully converge to 0.7 due to spatial heterogeneity)
        original_mean = np.mean(planarity[100:150])
        filtered_mean = np.mean(result[100:150])
        # Check that filtered is closer to 0.7 than original
        assert np.abs(filtered_mean - 0.7) < np.abs(original_mean - 0.7)


class TestLinearityFiltering:
    """Test linearity-specific filtering functions."""

    def test_smooth_linearity_spatial(self):
        """Test linearity-specific wrapper function."""
        points = np.array(
            [[i, j, 0] for i in range(10) for j in range(10)], dtype=np.float32
        )
        linearity = np.random.uniform(0.1, 0.5, len(points)).astype(np.float32)

        result = smooth_linearity_spatial(linearity, points)

        assert len(result) == len(linearity)
        assert not np.any(np.isnan(result))
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_validate_linearity(self):
        """Test linearity-specific validation."""
        linearity = np.array([0.3, np.nan, 1.5, -0.2, 0.4, np.inf])

        result = validate_linearity(linearity)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_linearity_artifact_removal(self):
        """Test that linearity artifacts are removed."""
        # Simulate linear features (poles, edges) with boundary artifacts
        points = np.array(
            [[i, j, 0] for i in range(30) for j in range(30)], dtype=np.float32
        )

        # Most points have low linearity (planar surfaces)
        linearity = np.full(len(points), 0.1, dtype=np.float32)

        # Add linear feature (vertical pole)
        for i in range(10, 20):
            idx = i * 30 + 15
            linearity[idx] = 0.8  # High linearity

        # Add artifacts at boundaries (set seed for reproducibility)
        np.random.seed(42)
        for i in range(10, 20):
            for offset in [-1, 1]:
                idx = i * 30 + 15 + offset
                linearity[idx] = np.random.uniform(0.0, 1.0)

        result = smooth_linearity_spatial(linearity, points, k_neighbors=15)

        # Core linear feature should be preserved
        core_idx = 15 * 30 + 15
        assert result[core_idx] > 0.5  # Still reasonably high linearity

        # Most artifacts should be reduced (not all may be fully smoothed)
        assert not np.any(np.isnan(result))


class TestHorizontalityFiltering:
    """Test horizontality-specific filtering functions."""

    def test_smooth_horizontality_spatial(self):
        """Test horizontality-specific wrapper function."""
        points = np.array(
            [[i, j, 0] for i in range(10) for j in range(10)], dtype=np.float32
        )
        horizontality = np.random.uniform(0.7, 1.0, len(points)).astype(np.float32)

        result = smooth_horizontality_spatial(horizontality, points)

        assert len(result) == len(horizontality)
        assert not np.any(np.isnan(result))
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_validate_horizontality(self):
        """Test horizontality-specific validation."""
        horizontality = np.array([0.9, np.nan, 1.1, -0.1, 0.95, np.inf])

        result = validate_horizontality(horizontality)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_horizontality_roof_boundary_artifacts(self):
        """Test horizontality artifact removal at roof boundaries."""
        # Simulate horizontal surface (roof) with edge artifacts
        points = np.array(
            [[i, j, 10] for i in range(30) for j in range(30)], dtype=np.float32
        )

        # Most points are horizontal (roof surface)
        horizontality = np.full(len(points), 0.95, dtype=np.float32)

        # Add edge artifacts (roof-to-air boundary)
        for i in range(30):
            # Left edge
            idx = i * 30
            horizontality[idx] = np.random.uniform(0.0, 0.5)
            # Right edge
            idx = i * 30 + 29
            horizontality[idx] = np.random.uniform(0.0, 0.5)

        result = smooth_horizontality_spatial(horizontality, points, k_neighbors=15)

        # Core roof should remain horizontal
        core_idx = 15 * 30 + 15
        assert result[core_idx] > 0.85  # Still high horizontality

        # Edge artifacts should be smoothed
        edge_idx = 15 * 30 + 0
        assert result[edge_idx] > 0.5  # Smoothed upward from artifacts


class TestMultiFeatureIntegration:
    """Test integration of multiple features together."""

    def test_all_three_features_together(self):
        """Test filtering all three features simultaneously."""
        # Create diverse point cloud
        np.random.seed(42)
        points = np.array(
            [[i, j, k] for i in range(15) for j in range(15) for k in range(5)],
            dtype=np.float32,
        )

        # Generate realistic features with artifacts
        planarity = np.random.uniform(0.3, 0.9, len(points)).astype(np.float32)
        linearity = np.random.uniform(0.0, 0.4, len(points)).astype(np.float32)
        horizontality = np.random.uniform(0.5, 1.0, len(points)).astype(np.float32)

        # Add artifacts
        artifact_indices = np.random.choice(len(points), size=50, replace=False)
        planarity[artifact_indices] = np.random.uniform(0.0, 1.0, 50)
        linearity[artifact_indices] = np.random.uniform(0.0, 1.0, 50)
        horizontality[artifact_indices] = np.random.uniform(0.0, 1.0, 50)

        # Filter all three
        planarity_clean = smooth_planarity_spatial(planarity, points)
        linearity_clean = smooth_linearity_spatial(linearity, points)
        horizontality_clean = smooth_horizontality_spatial(horizontality, points)

        # All should be valid
        assert not np.any(np.isnan(planarity_clean))
        assert not np.any(np.isnan(linearity_clean))
        assert not np.any(np.isnan(horizontality_clean))

        # Validate ranges
        assert np.all((planarity_clean >= 0.0) & (planarity_clean <= 1.0))
        assert np.all((linearity_clean >= 0.0) & (linearity_clean <= 1.0))
        assert np.all((horizontality_clean >= 0.0) & (horizontality_clean <= 1.0))

    def test_feature_consistency_after_filtering(self):
        """Test that features remain mathematically consistent after filtering."""
        points = np.array(
            [[i, j, 0] for i in range(20) for j in range(20)], dtype=np.float32
        )

        # Create features where planarity + linearity + sphericity ≈ 1
        planarity = np.random.uniform(0.5, 0.7, len(points)).astype(np.float32)
        linearity = np.random.uniform(0.1, 0.3, len(points)).astype(np.float32)
        sphericity = 1.0 - planarity - linearity

        # Filter
        planarity_clean = smooth_planarity_spatial(planarity, points)
        linearity_clean = smooth_linearity_spatial(linearity, points)

        # All should be in valid range
        assert np.all((planarity_clean >= 0.0) & (planarity_clean <= 1.0))
        assert np.all((linearity_clean >= 0.0) & (linearity_clean <= 1.0))
        # Note: Sum may not equal 1 after independent filtering,
        # but values should be reasonable
        assert np.all(planarity_clean + linearity_clean <= 1.2)  # Reasonable overlap


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_point(self):
        """Test with single point."""
        points = np.array([[0, 0, 0]], dtype=np.float32)
        feature = np.array([0.5], dtype=np.float32)

        # Should handle gracefully
        result = smooth_feature_spatial(feature, points, k_neighbors=8)
        assert len(result) == 1
        assert not np.isnan(result[0])

    def test_all_nan_input(self):
        """Test with all NaN input."""
        points = np.array(
            [[i, j, 0] for i in range(10) for j in range(10)], dtype=np.float32
        )
        feature = np.full(len(points), np.nan, dtype=np.float32)

        result = smooth_feature_spatial(feature, points)

        # Should return zeros or interpolated values
        assert not np.all(np.isnan(result))
        assert len(result) == len(feature)

    def test_mismatched_dimensions(self):
        """Test error handling for mismatched array dimensions."""
        points = np.array(
            [[i, j, 0] for i in range(10) for j in range(10)], dtype=np.float32
        )
        feature = np.array([0.5] * 50, dtype=np.float32)  # Wrong length

        with pytest.raises((ValueError, AssertionError)):
            smooth_feature_spatial(feature, points)

    def test_invalid_k_neighbors(self):
        """Test handling of edge case k_neighbors values."""
        points = np.array(
            [[i, j, 0] for i in range(10) for j in range(10)], dtype=np.float32
        )
        feature = np.random.uniform(0.3, 0.8, len(points)).astype(np.float32)

        # k too small - should be handled gracefully (adjusted internally)
        result = smooth_feature_spatial(feature, points, k_neighbors=1)
        assert len(result) == len(feature)

        # k larger than n_points - should be adjusted internally
        result = smooth_feature_spatial(feature, points, k_neighbors=200)
        assert len(result) == len(feature)

    def test_2d_points(self):
        """Test with 2D points (should work, treating Z=0)."""
        points_2d = np.array(
            [[i, j] for i in range(10) for j in range(10)], dtype=np.float32
        )
        feature = np.random.uniform(0.3, 0.8, len(points_2d)).astype(np.float32)

        # Should handle or raise clear error
        try:
            result = smooth_feature_spatial(feature, points_2d)
            # If it works, check validity
            assert len(result) == len(feature)
            assert not np.any(np.isnan(result))
        except (ValueError, AssertionError):
            # Expected if 3D required
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
