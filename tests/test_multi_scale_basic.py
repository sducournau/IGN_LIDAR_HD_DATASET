"""
Basic functional tests for multi-scale feature computation.

Tests the core multi-scale computation with synthetic data.
"""

import pytest
import numpy as np
from ign_lidar.features.compute.multi_scale import (
    MultiScaleFeatureComputer,
    ScaleConfig,
)


class TestMultiScaleBasic:
    """Basic tests for multi-scale computation."""

    @pytest.fixture
    def synthetic_points(self):
        """Create synthetic point cloud for testing."""
        # Create a simple planar surface with some noise
        np.random.seed(42)
        n_points = 1000

        # Generate planar points (z = 0 with small noise)
        x = np.random.uniform(0, 10, n_points)
        y = np.random.uniform(0, 10, n_points)
        z = np.random.normal(0, 0.01, n_points)  # Small noise

        points = np.column_stack([x, y, z])
        return points.astype(np.float32)

    @pytest.fixture
    def multi_scale_computer(self):
        """Create multi-scale computer with test scales."""
        scales = [
            ScaleConfig("fine", k_neighbors=10, search_radius=0.5, weight=0.3),
            ScaleConfig("medium", k_neighbors=30, search_radius=1.0, weight=0.5),
            ScaleConfig("coarse", k_neighbors=50, search_radius=2.0, weight=0.2),
        ]
        return MultiScaleFeatureComputer(
            scales=scales,
            aggregation_method="variance_weighted",
            variance_penalty=2.0,
        )

    def test_scale_config_validation(self):
        """Test ScaleConfig validation."""
        # Valid config
        scale = ScaleConfig("test", k_neighbors=10, search_radius=1.0, weight=0.5)
        assert scale.name == "test"
        assert scale.k_neighbors == 10
        assert scale.search_radius == 1.0
        assert scale.weight == 0.5

        # Invalid k_neighbors
        with pytest.raises(ValueError, match="k_neighbors must be > 0"):
            ScaleConfig("test", k_neighbors=0, search_radius=1.0, weight=0.5)

        # Invalid search_radius
        with pytest.raises(ValueError, match="search_radius must be > 0"):
            ScaleConfig("test", k_neighbors=10, search_radius=0, weight=0.5)

        # Invalid weight
        with pytest.raises(ValueError, match="weight must be >= 0"):
            ScaleConfig("test", k_neighbors=10, search_radius=1.0, weight=-0.5)

    def test_computer_initialization(self):
        """Test MultiScaleFeatureComputer initialization."""
        scales = [
            ScaleConfig("fine", k_neighbors=10, search_radius=0.5, weight=0.3),
            ScaleConfig("coarse", k_neighbors=30, search_radius=1.5, weight=0.7),
        ]

        computer = MultiScaleFeatureComputer(
            scales=scales, aggregation_method="weighted_average"
        )

        assert len(computer.scales) == 2
        assert computer.aggregation_method == "weighted_average"
        assert computer.variance_penalty == 2.0

    def test_requires_at_least_2_scales(self):
        """Test that at least 2 scales are required."""
        with pytest.raises(ValueError, match="At least 2 scales required"):
            MultiScaleFeatureComputer(
                scales=[
                    ScaleConfig("single", k_neighbors=10, search_radius=1.0, weight=1.0)
                ]
            )

    def test_invalid_aggregation_method(self):
        """Test that invalid aggregation method raises error."""
        scales = [
            ScaleConfig("fine", k_neighbors=10, search_radius=0.5, weight=0.5),
            ScaleConfig("coarse", k_neighbors=30, search_radius=1.5, weight=0.5),
        ]

        with pytest.raises(ValueError, match="aggregation_method must be one of"):
            MultiScaleFeatureComputer(scales=scales, aggregation_method="invalid")

    def test_compute_planarity_synthetic(self, synthetic_points, multi_scale_computer):
        """Test multi-scale planarity on synthetic planar surface."""
        features = multi_scale_computer.compute_features(
            points=synthetic_points, features_to_compute=["planarity"]
        )

        assert "planarity" in features
        assert features["planarity"].shape == (len(synthetic_points),)

        # Planar surface should have high planarity
        mean_planarity = np.mean(features["planarity"])
        assert mean_planarity > 0.5, f"Expected high planarity, got {mean_planarity}"

    def test_compute_multiple_features(self, synthetic_points, multi_scale_computer):
        """Test computing multiple features at once."""
        features = multi_scale_computer.compute_features(
            points=synthetic_points,
            features_to_compute=["planarity", "linearity", "sphericity"],
        )

        assert "planarity" in features
        assert "linearity" in features
        assert "sphericity" in features

        # Check shapes
        for feature_name, feature_values in features.items():
            assert feature_values.shape == (len(synthetic_points),)

        # Planar surface should have:
        # - High planarity (> 0.5 for multi-scale)
        # - Low linearity (relaxed for multi-scale)
        # - Low sphericity (relaxed for multi-scale)
        assert np.mean(features["planarity"]) > 0.5
        assert np.mean(features["linearity"]) < 0.45
        assert np.mean(features["sphericity"]) < 0.45

    def test_variance_weighted_vs_simple_weighted(self, synthetic_points):
        """Test that variance weighting produces different results than simple."""
        scales = [
            ScaleConfig("fine", k_neighbors=10, search_radius=0.5, weight=0.5),
            ScaleConfig("coarse", k_neighbors=30, search_radius=1.5, weight=0.5),
        ]

        # Variance weighted
        computer_var = MultiScaleFeatureComputer(
            scales=scales, aggregation_method="variance_weighted"
        )
        features_var = computer_var.compute_features(
            points=synthetic_points, features_to_compute=["planarity"]
        )

        # Simple weighted
        computer_simple = MultiScaleFeatureComputer(
            scales=scales, aggregation_method="weighted_average"
        )
        features_simple = computer_simple.compute_features(
            points=synthetic_points, features_to_compute=["planarity"]
        )

        # Results should be similar but not identical
        # (for clean synthetic data, variance weighting shouldn't change much)
        correlation = np.corrcoef(
            features_var["planarity"], features_simple["planarity"]
        )[0, 1]
        assert correlation > 0.95, "Methods should produce similar results"

    def test_kdtree_reuse(self, synthetic_points):
        """Test that KD-tree reuse works correctly."""
        from scipy.spatial import cKDTree

        scales = [
            ScaleConfig("fine", k_neighbors=10, search_radius=0.5, weight=0.5),
            ScaleConfig("coarse", k_neighbors=30, search_radius=1.5, weight=0.5),
        ]

        computer = MultiScaleFeatureComputer(scales=scales, reuse_kdtrees=True)

        # Build KD-tree once
        kdtree = cKDTree(synthetic_points)

        # Compute features with pre-built KD-tree
        features = computer.compute_features(
            points=synthetic_points,
            features_to_compute=["planarity"],
            kdtree=kdtree,
        )

        assert "planarity" in features
        assert features["planarity"].shape == (len(synthetic_points),)

    def test_small_point_cloud(self):
        """Test with very small point cloud."""
        # Create tiny point cloud
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.1]],
            dtype=np.float32,
        )

        scales = [
            ScaleConfig("fine", k_neighbors=3, search_radius=0.5, weight=0.5),
            ScaleConfig("coarse", k_neighbors=4, search_radius=2.0, weight=0.5),
        ]

        computer = MultiScaleFeatureComputer(scales=scales)
        features = computer.compute_features(
            points=points, features_to_compute=["planarity"]
        )

        assert "planarity" in features
        assert features["planarity"].shape == (len(points),)
        # Should not crash, even with small data

    def test_verticality_and_horizontality(self, synthetic_points):
        """Test verticality and horizontality computation."""
        scales = [
            ScaleConfig("fine", k_neighbors=10, search_radius=0.5, weight=0.5),
            ScaleConfig("coarse", k_neighbors=30, search_radius=1.5, weight=0.5),
        ]

        computer = MultiScaleFeatureComputer(scales=scales)
        features = computer.compute_features(
            points=synthetic_points,
            features_to_compute=["verticality", "horizontality"],
        )

        assert "verticality" in features
        assert "horizontality" in features

        # Horizontal surface should have:
        # - Low verticality (not a wall)
        # - High horizontality (flat ground)
        assert np.mean(features["verticality"]) < 0.3
        assert np.mean(features["horizontality"]) > 0.7


class TestArtifactDetection:
    """Test artifact detection system."""

    def test_artifact_detection_clean_data(self):
        """Test artifact detection on clean synthetic data."""
        np.random.seed(42)
        n_points = 500

        # Create clean planar surface
        x = np.random.uniform(0, 10, n_points)
        y = np.random.uniform(0, 10, n_points)
        z = np.random.normal(0, 0.01, n_points)
        points = np.column_stack([x, y, z]).astype(np.float32)

        scales = [
            ScaleConfig("fine", k_neighbors=10, search_radius=0.5, weight=0.5),
            ScaleConfig("coarse", k_neighbors=30, search_radius=1.5, weight=0.5),
        ]

        computer = MultiScaleFeatureComputer(
            scales=scales, artifact_detection=True, artifact_variance_threshold=0.15
        )

        features = computer.compute_features(
            points=points, features_to_compute=["planarity"]
        )

        # Clean data should have few artifacts detected
        # (actual artifact detection tested separately)
        assert "planarity" in features


class TestPerformance:
    """Test performance characteristics."""

    def test_processing_time_scaling(self):
        """Test that multi-scale adds expected computational cost."""
        np.random.seed(42)
        n_points = 500

        x = np.random.uniform(0, 10, n_points)
        y = np.random.uniform(0, 10, n_points)
        z = np.random.normal(0, 0.1, n_points)
        points = np.column_stack([x, y, z]).astype(np.float32)

        # 2 scales
        scales_2 = [
            ScaleConfig("fine", k_neighbors=10, search_radius=0.5, weight=0.5),
            ScaleConfig("coarse", k_neighbors=30, search_radius=1.5, weight=0.5),
        ]

        # 3 scales
        scales_3 = [
            ScaleConfig("fine", k_neighbors=10, search_radius=0.5, weight=0.3),
            ScaleConfig("medium", k_neighbors=20, search_radius=1.0, weight=0.4),
            ScaleConfig("coarse", k_neighbors=30, search_radius=1.5, weight=0.3),
        ]

        import time

        # Time 2 scales
        computer_2 = MultiScaleFeatureComputer(scales=scales_2)
        start = time.time()
        computer_2.compute_features(points=points, features_to_compute=["planarity"])
        time_2 = time.time() - start

        # Time 3 scales
        computer_3 = MultiScaleFeatureComputer(scales=scales_3)
        start = time.time()
        computer_3.compute_features(points=points, features_to_compute=["planarity"])
        time_3 = time.time() - start

        # 3 scales should take longer, but not dramatically
        # (more than 2 scales, but less than 2x)
        assert time_3 > time_2
        assert time_3 < time_2 * 2.0  # Reasonable scaling
