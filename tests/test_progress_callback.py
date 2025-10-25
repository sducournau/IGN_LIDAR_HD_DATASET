"""Test progress callback functionality in FeatureOrchestrator."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from ign_lidar.features.orchestrator import FeatureOrchestrator


class TestProgressCallback:
    """Test progress callback functionality."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing."""
        return OmegaConf.create(
            {
                "processor": {
                    "use_gpu": False,
                    "use_feature_computer": True,
                },
                "features": {
                    "mode": "minimal",
                    "k_neighbors": 30,
                    "use_rgb": False,
                    "use_infrared": False,
                    "search_radius": 3.0,
                },
            }
        )

    @pytest.fixture
    def sample_tile_data(self):
        """Generate sample tile data in expected format."""
        np.random.seed(42)
        n_points = 1000
        points = np.random.rand(n_points, 3) * 10.0
        classification = np.ones(n_points, dtype=int)
        intensity = np.random.randint(0, 255, n_points, dtype=np.uint16)
        return {
            "points": points,
            "classification": classification,
            "intensity": intensity,
        }

    def test_progress_callback_called(self, base_config, sample_tile_data):
        """Test that progress callback is actually called."""
        # Track callback invocations
        progress_updates = []

        def callback(progress, message):
            progress_updates.append({"progress": progress, "message": message})

        # Create orchestrator with callback
        orchestrator = FeatureOrchestrator(base_config, progress_callback=callback)

        # Compute features
        features = orchestrator.compute_features(sample_tile_data)

        # Verify callback was invoked
        assert (
            len(progress_updates) > 0
        ), "Progress callback should be called at least once"

        # Check progress values are reasonable
        for update in progress_updates:
            assert (
                0.0 <= update["progress"] <= 1.0
            ), f"Progress should be in [0, 1], got {update['progress']}"
            assert isinstance(update["message"], str), "Message should be string"
            assert len(update["message"]) > 0, "Message should not be empty"

    def test_progress_increases_monotonically(self, base_config, sample_tile_data):
        """Test that progress values increase over time."""
        progress_values = []

        def callback(progress, message):
            progress_values.append(progress)

        orchestrator = FeatureOrchestrator(base_config, progress_callback=callback)

        orchestrator.compute_features(sample_tile_data)

        # Check monotonic increase (allowing same value repeats)
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1], (
                f"Progress should not decrease: "
                f"{progress_values[i-1]} -> {progress_values[i]}"
            )

    def test_progress_reaches_completion(self, base_config, sample_tile_data):
        """Test that progress reaches 1.0 at completion."""
        progress_values = []

        def callback(progress, message):
            progress_values.append(progress)

        orchestrator = FeatureOrchestrator(base_config, progress_callback=callback)

        orchestrator.compute_features(sample_tile_data)

        # Last progress should be 1.0 (complete)
        assert len(progress_values) > 0, "Should have progress updates"
        assert progress_values[-1] == pytest.approx(
            1.0, abs=0.01
        ), f"Final progress should be 1.0, got {progress_values[-1]}"

    def test_callback_messages_informative(self, base_config, sample_tile_data):
        """Test that callback messages provide meaningful info."""
        messages = []

        def callback(progress, message):
            messages.append(message.lower())

        orchestrator = FeatureOrchestrator(base_config, progress_callback=callback)

        orchestrator.compute_features(sample_tile_data)

        # Check for expected message patterns
        messages_str = " ".join(messages)

        # Should mention feature computation stages
        expected_keywords = ["feature", "compute", "process"]
        found_keywords = [kw for kw in expected_keywords if kw in messages_str]

        assert len(found_keywords) > 0, (
            f"Messages should mention computation stages. " f"Got: {messages[:3]}"
        )

    def test_no_callback_works(self, base_config, sample_tile_data):
        """Test that orchestrator works without callback."""
        # Create without callback (should not crash)
        orchestrator = FeatureOrchestrator(base_config)

        # Compute features
        features = orchestrator.compute_features(sample_tile_data)

        # Verify features were computed
        assert len(features) > 0, "Should compute features"
        assert (
            "normals" in features or "curvature" in features
        ), "Should have geometric features"

    def test_callback_exception_handling(self, base_config, sample_tile_data):
        """Test that callback exceptions don't crash computation."""
        call_count = [0]

        def bad_callback(progress, message):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second call
                raise ValueError("Intentional callback error")

        orchestrator = FeatureOrchestrator(base_config, progress_callback=bad_callback)

        # Should complete despite callback error
        # (FeatureComputer should handle exceptions)
        features = orchestrator.compute_features(sample_tile_data)

        # Verify computation succeeded
        assert len(features) > 0, "Computation should succeed despite callback error"

    def test_callback_with_different_point_counts(self, base_config):
        """Test callback behavior with varying point cloud sizes."""
        progress_counts = {}

        def callback(progress, message):
            size = len(progress_counts)
            if size not in progress_counts:
                progress_counts[size] = 0
            progress_counts[size] += 1

        for n_points in [100, 500, 1000, 5000]:
            progress_counts.clear()
            tile_data = {
                "points": np.random.rand(n_points, 3) * 10.0,
                "classification": np.ones(n_points, dtype=int),
                "intensity": np.random.randint(0, 255, n_points, dtype=np.uint16),
            }

            orchestrator = FeatureOrchestrator(base_config, progress_callback=callback)

            orchestrator.compute_features(tile_data)

            # Should have at least some progress updates
            # (exact count may vary by implementation)
            total_updates = sum(progress_counts.values())
            assert (
                total_updates > 0
            ), f"Should have progress updates for {n_points} points"

    def test_callback_signature(self, base_config, sample_tile_data):
        """Test that callback receives correct argument types."""
        type_checks = []

        def callback(progress, message):
            type_checks.append(
                {
                    "progress_type": type(progress).__name__,
                    "message_type": type(message).__name__,
                    "progress_is_number": isinstance(progress, (int, float)),
                    "message_is_str": isinstance(message, str),
                }
            )

        orchestrator = FeatureOrchestrator(base_config, progress_callback=callback)

        orchestrator.compute_features(sample_tile_data)

        # Check all invocations had correct types
        for check in type_checks:
            assert check[
                "progress_is_number"
            ], f"Progress should be numeric, got {check['progress_type']}"
            assert check[
                "message_is_str"
            ], f"Message should be string, got {check['message_type']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
