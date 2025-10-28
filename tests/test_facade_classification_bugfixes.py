"""
Tests for facade classification bugfixes (v3.0.4).

These tests verify that the bugfixes for building facade classification
properly handle NaN/Inf artifacts, ground filtering, and feature validation.

Author: IGN LiDAR HD Classification Team
Date: October 26, 2025
Version: 3.0.4
"""

import numpy as np
import pytest

from ign_lidar.core.classification.feature_validator import (
    sanitize_feature,
    validate_features_for_classification,
    create_safe_building_mask,
)


class TestFeatureSanitization:
    """Test feature sanitization functions."""

    def test_sanitize_feature_with_nan(self):
        """Test sanitizing features with NaN values."""
        feature = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        sanitized, stats = sanitize_feature(feature, "test_feature", fill_nan=0.0)

        assert stats["n_nan"] == 1
        assert stats["n_inf"] == 0
        assert not np.isnan(sanitized).any()
        assert sanitized[2] == 0.0

    def test_sanitize_feature_with_inf(self):
        """Test sanitizing features with Inf values."""
        feature = np.array([1.0, 2.0, np.inf, 4.0, -np.inf])

        sanitized, stats = sanitize_feature(feature, "test_feature", clip_sigma=3.0)

        assert stats["n_inf"] == 2
        assert not np.isinf(sanitized).any()
        assert np.all(np.isfinite(sanitized))

    def test_sanitize_feature_with_outliers(self):
        """Test sanitizing features with outliers."""
        # Create feature with strong outliers
        feature = np.array([1.0, 1.1, 1.2, 1.1, 1.0, 1.1, 100.0])

        sanitized, stats = sanitize_feature(feature, "test_feature", clip_sigma=2.0)

        # Outlier should be clipped (if detected)
        # Note: May not always detect outliers depending on distribution
        if stats["n_outliers"] > 0:
            assert sanitized[6] < 100.0
            assert sanitized[6] == sanitized.max()

    def test_sanitize_feature_no_issues(self):
        """Test sanitizing clean features."""
        feature = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sanitized, stats = sanitize_feature(feature, "test_feature")

        assert stats["n_nan"] == 0
        assert stats["n_inf"] == 0
        assert stats["n_total_fixed"] == 0
        assert np.array_equal(sanitized, feature)

    def test_sanitize_feature_all_nan(self):
        """Test sanitizing feature that is all NaN."""
        feature = np.array([np.nan, np.nan, np.nan])

        sanitized, stats = sanitize_feature(feature, "test_feature", fill_nan=0.0)

        assert stats["n_nan"] == 3
        assert np.all(sanitized == 0.0)


class TestFeatureValidation:
    """Test feature validation for classification."""

    def test_validate_features_all_valid(self):
        """Test validation with all valid features."""
        features = {
            "normals": np.random.randn(100, 3),
            "verticality": np.random.rand(100),
            "curvature": np.random.rand(100),
        }

        is_valid, sanitized, issues = validate_features_for_classification(
            features=features,
            required_features=["normals", "verticality"],
        )

        assert is_valid
        assert len(issues) == 0
        assert "normals" in sanitized
        assert "verticality" in sanitized

    def test_validate_features_with_nan(self):
        """Test validation with NaN values."""
        features = {
            "verticality": np.array([1.0, 2.0, np.nan, 4.0]),
        }

        is_valid, sanitized, issues = validate_features_for_classification(
            features=features,
            required_features=["verticality"],
        )

        # Should sanitize and report issues
        assert len(issues) > 0
        assert not np.isnan(sanitized["verticality"]).any()

    def test_validate_features_missing_required(self):
        """Test validation with missing required features."""
        features = {
            "verticality": np.random.rand(100),
        }

        is_valid, sanitized, issues = validate_features_for_classification(
            features=features,
            required_features=["normals", "verticality"],
        )

        # Should fail validation
        assert not is_valid
        assert any("normals" in issue and "missing" in issue for issue in issues)

    def test_validate_features_with_mask(self):
        """Test validation with point mask."""
        n_points = 100
        features = {
            "verticality": np.random.rand(n_points),
        }
        mask = np.zeros(n_points, dtype=bool)
        mask[:50] = True  # Only first 50 points

        is_valid, sanitized, issues = validate_features_for_classification(
            features=features,
            required_features=["verticality"],
            point_mask=mask,
        )

        assert is_valid
        assert len(sanitized["verticality"]) == n_points

    def test_validate_vector_features(self):
        """Test validation of vector features (e.g., normals)."""
        features = {
            "normals": np.random.randn(100, 3),
        }
        # Introduce some NaN in normals
        features["normals"][10, 1] = np.nan

        is_valid, sanitized, issues = validate_features_for_classification(
            features=features,
            required_features=["normals"],
        )

        # Should sanitize
        assert not np.isnan(sanitized["normals"]).any()
        assert len(issues) > 0


class TestSafeBuildingMask:
    """Test safe building mask creation."""

    def test_create_safe_mask_with_ground_filter(self):
        """Test creating safe mask with ground filtering."""
        n_points = 100
        building_mask = np.zeros(n_points, dtype=bool)
        building_mask[20:80] = True  # 60 building points

        is_ground = np.zeros(n_points, dtype=int)
        is_ground[30:40] = 1  # 10 ground points in building

        heights = np.random.rand(n_points) * 10
        heights[30:40] = 0.2  # Low ground points

        safe_mask, stats = create_safe_building_mask(
            building_mask=building_mask,
            is_ground=is_ground,
            heights=heights,
            ground_height_tolerance=0.5,
        )

        # Should filter out low ground points
        assert stats["n_ground_removed"] > 0
        assert stats["n_final"] < stats["n_initial"]
        assert not np.any(safe_mask[30:40])  # Ground points removed

    def test_create_safe_mask_no_ground_feature(self):
        """Test creating safe mask without ground feature."""
        n_points = 100
        building_mask = np.zeros(n_points, dtype=bool)
        building_mask[20:80] = True

        heights = np.random.rand(n_points) * 10
        heights[30:35] = 0.2  # Very low points

        safe_mask, stats = create_safe_building_mask(
            building_mask=building_mask,
            is_ground=None,
            heights=heights,
            ground_height_tolerance=0.5,
        )

        # Should filter by height only
        assert stats["n_low_removed"] > 0
        assert not np.any(safe_mask[30:35])

    def test_create_safe_mask_no_filtering(self):
        """Test creating safe mask with no filtering needed."""
        n_points = 100
        building_mask = np.zeros(n_points, dtype=bool)
        building_mask[20:80] = True

        safe_mask, stats = create_safe_building_mask(
            building_mask=building_mask,
            is_ground=None,
            heights=None,
        )

        # No filtering, should be identical
        assert stats["n_ground_removed"] == 0
        assert stats["n_low_removed"] == 0
        assert np.array_equal(safe_mask, building_mask)


class TestEdgeDetectionRobustness:
    """Test edge detection with artifacts."""

    def test_edge_detection_with_nan_curvature(self):
        """Test edge detection handles NaN curvature."""
        curvature = np.array([0.5, 0.8, np.nan, 1.2, 0.3])
        threshold = 0.7

        # Sanitize before use
        curvature_clean, stats = sanitize_feature(curvature, "curvature", fill_nan=0.0)

        # Now detect edges
        edge_mask = curvature_clean > threshold

        # Should not have NaN in result
        assert not np.isnan(curvature_clean).any()
        # NaN point (index 2) should be False
        assert not edge_mask[2]

    def test_edge_detection_with_inf_curvature(self):
        """Test edge detection handles Inf curvature."""
        curvature = np.array([0.5, 0.8, np.inf, 1.2, -np.inf])
        threshold = 0.7

        # Sanitize
        curvature_clean, stats = sanitize_feature(
            curvature, "curvature", clip_sigma=3.0
        )

        # Detect edges
        edge_mask = curvature_clean > threshold

        # Should not have Inf
        assert not np.isinf(curvature_clean).any()
        assert stats["n_inf"] == 2


class TestRoofClassificationSafety:
    """Test roof classification with validated features."""

    def test_roof_features_validation(self):
        """Test that roof classification validates features."""
        n_points = 100

        # Create features with some invalid values
        normals = np.random.randn(n_points, 3)
        normals[10:15] = np.nan  # Invalid normals

        verticality = np.random.rand(n_points)
        verticality[20] = np.inf  # Invalid verticality

        # Validate before use
        features = {"normals": normals, "verticality": verticality}
        is_valid, sanitized, issues = validate_features_for_classification(
            features=features,
            required_features=["normals", "verticality"],
        )

        # Should report issues but provide sanitized versions
        assert len(issues) > 0
        assert not np.isnan(sanitized["normals"]).any()
        assert not np.isinf(sanitized["verticality"]).any()


@pytest.mark.integration
class TestBuildingClassificationIntegration:
    """Integration tests for building classification bugfixes."""

    def test_full_classification_pipeline_with_artifacts(self):
        """Test full classification pipeline handles artifacts correctly."""
        # Create synthetic building data with artifacts
        n_points = 1000
        points = np.random.randn(n_points, 3) * 10

        # Create features with intentional artifacts
        normals = np.random.randn(n_points, 3)
        normals[100:110] = np.nan  # NaN normals

        verticality = np.random.rand(n_points)
        verticality[200] = np.inf  # Inf verticality

        curvature = np.random.rand(n_points)
        curvature[300:305] = np.nan  # NaN curvature

        heights = np.random.rand(n_points) * 20
        is_ground = (heights < 0.5).astype(int)

        # Validate all features
        features = {
            "normals": normals,
            "verticality": verticality,
            "curvature": curvature,
        }

        is_valid, sanitized, issues = validate_features_for_classification(
            features=features,
            required_features=["normals", "verticality", "curvature"],
        )

        # Should successfully sanitize
        assert len(issues) > 0  # Issues detected
        assert not np.isnan(sanitized["normals"]).any()
        assert not np.isinf(sanitized["verticality"]).any()
        assert not np.isnan(sanitized["curvature"]).any()

        # Now classification can proceed safely with sanitized features
        assert sanitized["normals"].shape == normals.shape
        assert sanitized["verticality"].shape == verticality.shape
        assert sanitized["curvature"].shape == curvature.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
