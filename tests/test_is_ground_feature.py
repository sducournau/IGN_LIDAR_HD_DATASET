"""
Tests for is_ground feature computation with DTM augmentation support.

Author: IGN LiDAR HD Development Team
Date: October 25, 2025
"""

import numpy as np
import pytest

from ign_lidar.features.compute.is_ground import (
    compute_ground_density,
    compute_is_ground,
    compute_is_ground_with_stats,
    identify_ground_gaps,
)


class TestIsGroundFeature:
    """Test suite for is_ground feature computation."""

    def test_basic_is_ground(self):
        """Test basic is_ground computation without DTM augmentation."""
        # Create test data: 6 points with mixed classifications
        classification = np.array([2, 2, 6, 3, 2, 5])  # 3 ground, 3 non-ground

        is_ground = compute_is_ground(classification)

        # Check shape and dtype
        assert is_ground.shape == (6,)
        assert is_ground.dtype == np.int8

        # Check values
        expected = np.array([1, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(is_ground, expected)

    def test_is_ground_with_synthetic(self):
        """Test is_ground with DTM-augmented synthetic points."""
        classification = np.array([2, 2, 6, 3, 2, 5])
        # Point 4 (index 4) is synthetic ground from DTM
        synthetic_flags = np.array([False, False, False, False, True, False])

        # Include synthetic points (default)
        is_ground = compute_is_ground(classification, synthetic_flags)
        expected = np.array([1, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(is_ground, expected)

        # Exclude synthetic points
        is_ground_no_syn = compute_is_ground(
            classification, synthetic_flags, include_synthetic=False
        )
        expected_no_syn = np.array([1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(is_ground_no_syn, expected_no_syn)

    def test_is_ground_with_stats(self):
        """Test is_ground computation with statistics."""
        classification = np.array([2, 2, 6, 3, 2, 5])
        synthetic_flags = np.array([False, False, False, False, True, False])

        is_ground, stats = compute_is_ground_with_stats(
            classification, synthetic_flags, verbose=False
        )

        # Check statistics
        assert stats["total_points"] == 6
        assert stats["natural_ground"] == 2
        assert stats["synthetic_ground"] == 1
        assert stats["total_ground"] == 3
        assert stats["non_ground"] == 3
        assert stats["ground_percentage"] == 50.0
        assert stats["synthetic_percentage"] == pytest.approx(33.33, rel=0.01)

    def test_is_ground_all_ground(self):
        """Test with all points being ground."""
        classification = np.full(100, 2)  # All ground

        is_ground = compute_is_ground(classification)

        assert np.all(is_ground == 1)
        assert np.sum(is_ground) == 100

    def test_is_ground_no_ground(self):
        """Test with no ground points."""
        classification = np.array([6, 6, 3, 5, 6, 3])  # No ground

        is_ground = compute_is_ground(classification)

        assert np.all(is_ground == 0)
        assert np.sum(is_ground) == 0

    def test_is_ground_custom_class(self):
        """Test with custom ground class."""
        # Use class 9 (water) as "ground"
        classification = np.array([9, 9, 6, 3, 9, 5])

        is_ground = compute_is_ground(classification, ground_class=9)

        expected = np.array([1, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(is_ground, expected)

    def test_is_ground_invalid_input(self):
        """Test error handling for invalid inputs."""
        # Wrong shape
        with pytest.raises(ValueError, match="must be 1D array"):
            compute_is_ground(np.array([[1, 2], [3, 4]]))

        # Mismatched lengths
        classification = np.array([2, 2, 6])
        synthetic_flags = np.array([False, False])
        with pytest.raises(ValueError, match="same length"):
            compute_is_ground(classification, synthetic_flags)

    def test_ground_density(self):
        """Test ground density computation."""
        # Create a 10x10 grid of points
        x = np.repeat(np.arange(10), 10)
        y = np.tile(np.arange(10), 10)
        z = np.zeros(100)
        points = np.column_stack([x, y, z])

        # Half are ground
        is_ground = np.zeros(100, dtype=np.int8)
        is_ground[:50] = 1

        density_map, mean_density = compute_ground_density(
            points, is_ground, grid_size=5.0
        )

        # Check that we got a density map
        assert density_map.shape[0] > 0
        assert density_map.shape[1] > 0
        assert mean_density > 0

    def test_identify_ground_gaps(self):
        """Test ground gap identification."""
        # Create a 10x10x1 point cloud
        x = np.repeat(np.arange(10), 10)
        y = np.tile(np.arange(10), 10)
        z = np.zeros(100)
        points = np.column_stack([x, y, z])

        # Create sparse ground coverage (only 20% ground)
        is_ground = np.zeros(100, dtype=np.int8)
        is_ground[:20] = 1

        gap_mask, gap_stats = identify_ground_gaps(
            points, is_ground, grid_size=5.0, min_density_threshold=0.5
        )

        # Check that gaps were identified
        assert isinstance(gap_mask, np.ndarray)
        assert gap_mask.shape == (100,)
        assert "n_gap_cells" in gap_stats
        assert "pct_gap" in gap_stats

    def test_is_ground_large_dataset(self):
        """Test with larger dataset for performance."""
        # 1 million points
        n_points = 1_000_000
        # 10% ground
        classification = np.full(n_points, 6)
        classification[: n_points // 10] = 2

        is_ground = compute_is_ground(classification)

        assert is_ground.shape == (n_points,)
        assert np.sum(is_ground) == n_points // 10

    def test_is_ground_with_stats_no_synthetic(self):
        """Test statistics without synthetic points."""
        classification = np.array([2, 2, 6, 3, 2, 5])

        is_ground, stats = compute_is_ground_with_stats(classification, verbose=False)

        assert stats["synthetic_ground"] == 0
        assert stats["natural_ground"] == 3
        assert stats["total_ground"] == 3


class TestIsGroundIntegration:
    """Integration tests with FeatureOrchestrator."""

    @pytest.mark.xfail(reason="Feature orchestrator API changes")
    def test_is_ground_in_orchestrator(self):
        """Test that is_ground feature can be computed via orchestrator."""
        from omegaconf import OmegaConf

        from ign_lidar.features.orchestrator import FeatureOrchestrator

        # Create minimal config
        config = OmegaConf.create(
            {
                "processor": {"use_gpu": False},
                "features": {
                    "mode": "minimal",
                    "k_neighbors": 20,
                    "compute_is_ground": True,
                },
            }
        )

        orchestrator = FeatureOrchestrator(config)

        # Create test tile data
        points = np.random.rand(100, 3) * 10
        classification = np.random.choice([2, 6, 3], 100)
        intensity = np.random.randint(0, 256, 100)
        return_number = np.ones(100, dtype=np.int8)

        tile_data = {
            "points": points,
            "classification": classification,
            "intensity": intensity,
            "return_number": return_number,
        }

        # Compute features
        features = orchestrator.compute_features(tile_data)

        # Check that is_ground was computed
        assert "is_ground" in features
        assert features["is_ground"].shape == (100,)
        assert features["is_ground"].dtype == np.int8
        assert np.all((features["is_ground"] == 0) | (features["is_ground"] == 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
