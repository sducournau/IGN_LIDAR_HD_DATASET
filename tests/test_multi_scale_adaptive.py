"""
Tests for adaptive aggregation in multi-scale feature computation.

Tests the adaptive scale selection based on local geometry complexity.
"""

import numpy as np
import pytest
from ign_lidar.features.compute.multi_scale import (
    MultiScaleFeatureComputer,
    ScaleConfig,
)


@pytest.fixture
def adaptive_computer():
    """Create multi-scale computer with adaptive aggregation."""
    scales = [
        ScaleConfig(name="fine", k_neighbors=20, search_radius=1.0, weight=0.3),
        ScaleConfig(name="medium", k_neighbors=50, search_radius=2.5, weight=0.5),
        ScaleConfig(name="coarse", k_neighbors=100, search_radius=5.0, weight=0.2),
    ]

    return MultiScaleFeatureComputer(
        scales=scales,
        aggregation_method="adaptive",
        variance_penalty=2.0,
    )


@pytest.fixture
def synthetic_geometry():
    """
    Create synthetic point cloud with different geometry types.

    Returns:
        points: [N, 3] array with:
            - First 100 points: planar surface (low complexity)
            - Next 100 points: edge/corner (high complexity)
            - Last 100 points: noisy region (medium complexity)
    """
    np.random.seed(42)

    # Planar surface (Z=0)
    plane_points = np.random.rand(100, 3)
    plane_points[:, 2] = 0.01 * np.random.randn(100)  # Minimal Z noise

    # Edge/corner (L-shape)
    edge_points = np.random.rand(100, 3)
    edge_points[:50, 2] = 0  # Horizontal surface
    edge_points[50:, 0] = 0  # Vertical surface

    # Noisy region
    noise_points = np.random.rand(100, 3) * 2.0

    points = np.vstack([plane_points, edge_points, noise_points])

    return points


def test_adaptive_initialization(adaptive_computer):
    """Test that adaptive computer initializes correctly."""
    assert adaptive_computer.aggregation_method == "adaptive"
    assert len(adaptive_computer.scales) == 3
    assert adaptive_computer.variance_penalty == 2.0


def test_adaptive_aggregation_runs(adaptive_computer, synthetic_geometry):
    """Test that adaptive aggregation completes without errors."""
    points = synthetic_geometry

    # Compute features with adaptive aggregation
    features = adaptive_computer.compute_features(
        points=points,
        features_to_compute=["planarity", "linearity", "sphericity"],
    )

    # Check results
    assert "planarity" in features
    assert "linearity" in features
    assert "sphericity" in features

    assert len(features["planarity"]) == len(points)
    assert np.all(np.isfinite(features["planarity"]))


def test_adaptive_complexity_based_selection(adaptive_computer):
    """
    Test that adaptive aggregation selects scales based on complexity.

    Planar regions should prefer coarse scales (stable).
    Edge regions should prefer fine scales (detailed).
    """
    # Create simple test case
    np.random.seed(42)

    # Perfect plane (very low complexity)
    plane_points = np.zeros((50, 3))
    plane_points[:, :2] = np.random.rand(50, 2) * 10  # XY spread
    plane_points[:, 2] = 0.0  # Flat Z

    # Edge (high complexity)
    edge_points = np.zeros((50, 3))
    edge_points[:25, :2] = np.random.rand(25, 2) * 10
    edge_points[:25, 2] = 0.0  # Horizontal
    edge_points[25:, :2] = np.random.rand(25, 2) * 10
    edge_points[25:, 2] = 5.0  # Vertical jump

    points = np.vstack([plane_points, edge_points])

    # Compute features
    features = adaptive_computer.compute_features(
        points=points,
        features_to_compute=["planarity", "linearity"],
    )

    # Plane regions should have higher planarity
    plane_planarity = features["planarity"][:50]
    edge_planarity = features["planarity"][50:]

    # Statistical test: plane should be more planar on average
    assert np.mean(plane_planarity) > np.mean(edge_planarity)


def test_adaptive_variance_weighting(adaptive_computer, synthetic_geometry):
    """Test that adaptive aggregation down-weights high-variance scales."""
    points = synthetic_geometry

    # Compute with adaptive
    features_adaptive = adaptive_computer.compute_features(
        points=points,
        features_to_compute=["planarity", "verticality"],
    )

    # Create variance-weighted computer for comparison
    variance_computer = MultiScaleFeatureComputer(
        scales=adaptive_computer.scales,
        aggregation_method="variance_weighted",
        variance_penalty=2.0,
    )

    features_variance = variance_computer.compute_features(
        points=points,
        features_to_compute=["planarity", "verticality"],
    )

    # Both should produce similar results (both use variance weighting)
    # but adaptive adds complexity-based preferences

    # Should be correlated but not identical
    correlation = np.corrcoef(
        features_adaptive["planarity"], features_variance["planarity"]
    )[0, 1]
    assert correlation > 0.8  # Highly correlated
    assert correlation < 0.99  # But not identical


def test_adaptive_handles_missing_features():
    """Test adaptive aggregation when some scales don't have all features."""
    # Create scales
    scales = [
        ScaleConfig(name="fine", k_neighbors=10, search_radius=0.5, weight=0.5),
        ScaleConfig(name="coarse", k_neighbors=50, search_radius=3.0, weight=0.5),
    ]

    computer = MultiScaleFeatureComputer(
        scales=scales,
        aggregation_method="adaptive",
        variance_penalty=2.0,
    )

    # Create simple point cloud
    np.random.seed(42)
    points = np.random.rand(50, 3)

    # Compute features
    features = computer.compute_features(
        points=points,
        features_to_compute=["planarity", "linearity"],
    )

    # Should handle gracefully
    assert "planarity" in features
    assert len(features["planarity"]) == len(points)
    assert np.all(np.isfinite(features["planarity"]))


def test_adaptive_no_geometric_features(adaptive_computer):
    """
    Test adaptive falls back gracefully when no geometric features available.
    """
    np.random.seed(42)
    points = np.random.rand(50, 3)

    # Try to compute features that don't include planarity/linearity
    # (This should trigger fallback to uniform complexity)
    features = adaptive_computer.compute_features(
        points=points,
        features_to_compute=["verticality"],  # No planarity/linearity
    )

    # Should still work with uniform complexity
    assert "verticality" in features
    assert len(features["verticality"]) == len(points)


def test_adaptive_single_scale_fallback():
    """Test adaptive with only one scale (should raise error)."""
    scales = [
        ScaleConfig(name="single", k_neighbors=30, search_radius=3.0, weight=1.0),
    ]

    # Single scale should raise ValueError
    with pytest.raises(ValueError, match="At least 2 scales required"):
        MultiScaleFeatureComputer(
            scales=scales,
            aggregation_method="adaptive",
            variance_penalty=2.0,
        )


def test_adaptive_complexity_range(adaptive_computer):
    """Test that complexity scores are properly normalized [0, 1]."""
    np.random.seed(42)

    # Create extreme geometry
    points = np.random.rand(100, 3) * 100  # Large scale

    features = adaptive_computer.compute_features(
        points=points,
        features_to_compute=["planarity", "linearity"],
    )

    # Results should be valid probabilities
    assert np.all(features["planarity"] >= 0.0)
    assert np.all(features["planarity"] <= 1.0)
    assert np.all(features["linearity"] >= 0.0)
    assert np.all(features["linearity"] <= 1.0)


def test_adaptive_vs_weighted_performance(adaptive_computer, synthetic_geometry):
    """
    Compare adaptive vs simple weighted aggregation performance.

    Adaptive should be slightly slower but produce better results on
    complex geometry.
    """
    import time

    points = synthetic_geometry

    # Adaptive
    start = time.time()
    features_adaptive = adaptive_computer.compute_features(
        points=points,
        features_to_compute=["planarity", "linearity", "sphericity"],
    )
    time_adaptive = time.time() - start

    # Weighted average
    weighted_computer = MultiScaleFeatureComputer(
        scales=adaptive_computer.scales,
        aggregation_method="weighted_average",
        variance_penalty=2.0,
    )

    start = time.time()
    features_weighted = weighted_computer.compute_features(
        points=points,
        features_to_compute=["planarity", "linearity", "sphericity"],
    )
    time_weighted = time.time() - start

    # Adaptive should be slightly slower (more computation)
    # but not dramatically (< 5x for reasonable overhead)
    assert time_adaptive < time_weighted * 5.0

    # Both should produce valid results
    assert np.all(np.isfinite(features_adaptive["planarity"]))
    assert np.all(np.isfinite(features_weighted["planarity"]))


@pytest.mark.parametrize(
    "n_scales,expected_performance",
    [
        (2, "fast"),  # 2 scales
        (3, "medium"),  # 3 scales
        (4, "slower"),  # 4 scales
    ],
)
def test_adaptive_scalability(n_scales, expected_performance):
    """Test that adaptive aggregation scales with number of scales."""
    # Create scales
    scales = []
    for i in range(n_scales):
        scales.append(
            ScaleConfig(
                name=f"scale_{i}",
                k_neighbors=20 + i * 30,
                search_radius=1.0 + i * 2.0,
                weight=1.0 / n_scales,
            )
        )

    computer = MultiScaleFeatureComputer(
        scales=scales,
        aggregation_method="adaptive",
        variance_penalty=2.0,
    )

    np.random.seed(42)
    points = np.random.rand(100, 3)

    # Should complete regardless of scale count
    features = computer.compute_features(
        points=points,
        features_to_compute=["planarity"],
    )

    assert "planarity" in features
    assert len(features["planarity"]) == len(points)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
