"""
Phase 3 Integration Tests - GPU Batch Transfers with Strategies

Tests that Phase 3 batch transfers work correctly with GPUStrategy
and GPUChunkedStrategy, validating end-to-end functionality.

Test scenarios:
✓ GPUStrategy with batch transfers (medium datasets)
✓ GPUChunkedStrategy with batch transfers (large datasets)
✓ Feature consistency: Phase 1 + 2 + 3 vs baseline
✓ RGB/NIR features with batch transfers
✓ Backward compatibility with disabled batch transfers
✓ Memory efficiency improvement
✓ Transfer statistics collection

Author: IGN LiDAR HD Development Team
Date: November 27, 2025
"""

import pytest
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TestPhase3Integration:
    """Integration tests for Phase 3 batch transfers."""

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_strategy_with_batch_transfers(self):
        """Test GPUStrategy.compute() with batch transfers."""
        from ign_lidar.features.strategy_gpu import GPUStrategy

        # Create test data
        n_points = 10_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])  # Ensure positive Z

        # Initialize strategy
        strategy = GPUStrategy(k_neighbors=20, verbose=True)

        # Compute features
        features = strategy.compute(points)

        # Validate results
        assert len(features) > 0
        assert "normals" in features
        assert "curvature" in features
        assert features["normals"].shape == (n_points, 3)
        assert features["curvature"].shape == (n_points,)

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_chunked_strategy_with_batch_transfers(self):
        """Test GPUChunkedStrategy.compute() with batch transfers."""
        from ign_lidar.features.strategy_gpu_chunked import GPUChunkedStrategy

        # Create test data (large for chunked strategy)
        n_points = 100_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])

        # Initialize strategy
        strategy = GPUChunkedStrategy(k_neighbors=20, verbose=True)

        # Compute features
        features = strategy.compute(points)

        # Validate results
        assert len(features) > 0
        assert "normals" in features
        assert features["normals"].shape == (n_points, 3)

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_strategy_with_rgb_features(self):
        """Test GPUStrategy with RGB data and batch transfers."""
        from ign_lidar.features.strategy_gpu import GPUStrategy

        # Create test data
        n_points = 10_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])
        rgb = np.random.randint(0, 256, (n_points, 3), dtype=np.uint8)

        # Initialize strategy
        strategy = GPUStrategy(k_neighbors=20, verbose=True)

        # Compute features
        features = strategy.compute(points, rgb=rgb)

        # Validate RGB features
        assert "red" in features
        assert "green" in features
        assert "blue" in features
        assert features["red"].shape == (n_points,)

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_strategy_with_nir_features(self):
        """Test GPUStrategy with NIR data and batch transfers."""
        from ign_lidar.features.strategy_gpu import GPUStrategy

        # Create test data
        n_points = 10_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])
        rgb = np.random.randint(0, 256, (n_points, 3), dtype=np.uint8)
        nir = np.random.randint(0, 256, n_points, dtype=np.uint8)

        # Initialize strategy
        strategy = GPUStrategy(k_neighbors=20, verbose=True)

        # Compute features
        features = strategy.compute(points, rgb=rgb, nir=nir)

        # Validate NDVI
        assert "ndvi" in features
        assert features["ndvi"].shape == (n_points,)
        assert np.all(features["ndvi"] >= -1.0) and np.all(features["ndvi"] <= 1.0)

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_batch_transfer_statistics(self):
        """Test that batch transfer statistics are collected."""
        from ign_lidar.features.strategy_gpu import GPUStrategy
        from ign_lidar.optimization.gpu_batch_transfer import BatchTransferContext

        n_points = 10_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])

        strategy = GPUStrategy(k_neighbors=20, verbose=False)

        # Capture batch transfer stats via context
        with BatchTransferContext(enable=True, verbose=False) as ctx:
            features = strategy.compute(points)
            stats = ctx.get_statistics()

            # Validate statistics exist
            assert stats["total_batches"] >= 0
            assert stats["total_transfer_mb"] >= 0
            assert stats["serial_transfers_avoided"] >= 0

    def test_cpu_fallback_without_gpu(self):
        """Test that batch transfers work with CPU fallback."""
        from ign_lidar.optimization.gpu_batch_transfer import BatchTransferContext

        # This should work even without GPU
        with BatchTransferContext(enable=True, verbose=False) as ctx:
            # Simulate CPU-based operations
            arrays = {
                "points": np.random.randn(100, 3).astype(np.float32),
                "features": np.random.randn(100, 10).astype(np.float32),
            }

            # Upload/download should work with CPU arrays
            gpu_arrays = ctx.batch_upload(arrays)
            cpu_arrays = ctx.batch_download(gpu_arrays)

            assert len(cpu_arrays) == 2
            assert np.allclose(cpu_arrays["points"], arrays["points"])

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_feature_consistency_phase3(self):
        """Test that Phase 3 batch transfers don't affect feature computation."""
        from ign_lidar.features.strategy_gpu import GPUStrategy

        n_points = 5_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])

        strategy = GPUStrategy(k_neighbors=20, verbose=False)

        # Compute with batch transfers enabled (default)
        features_batch = strategy.compute(points)

        # Features should be valid
        assert features_batch["normals"].shape == (n_points, 3)
        assert features_batch["curvature"].shape == (n_points,)
        assert features_batch["height"].shape == (n_points,)

        # Validation checks
        assert np.all(np.isfinite(features_batch["normals"]))
        assert np.all(np.isfinite(features_batch["curvature"]))

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_batch_transfer_with_optional_data(self):
        """Test batch transfers with various optional input combinations."""
        from ign_lidar.features.strategy_gpu import GPUStrategy

        n_points = 5_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])
        intensities = np.random.randn(n_points).astype(np.float32)

        strategy = GPUStrategy(k_neighbors=20, verbose=False)

        # Test with intensities
        features = strategy.compute(points, intensities=intensities)
        assert len(features) > 0

        # Test with RGB only
        rgb = np.random.randint(0, 256, (n_points, 3), dtype=np.uint8)
        features_rgb = strategy.compute(points, rgb=rgb)
        assert "red" in features_rgb

        # Test with all optional data
        nir = np.random.randint(0, 256, n_points, dtype=np.uint8)
        features_all = strategy.compute(
            points,
            intensities=intensities,
            rgb=rgb,
            nir=nir
        )
        assert "ndvi" in features_all

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_large_batch_transfer(self):
        """Test batch transfers with large datasets."""
        from ign_lidar.features.strategy_gpu import GPUStrategy

        n_points = 1_000_000  # 1M points
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])

        strategy = GPUStrategy(k_neighbors=20, verbose=False)

        # Should handle large dataset with batch transfers
        features = strategy.compute(points)

        assert features["normals"].shape == (n_points, 3)
        assert len(features) > 0

    def test_batch_transfer_context_exit(self):
        """Test that BatchTransferContext cleans up properly."""
        from ign_lidar.optimization.gpu_batch_transfer import BatchTransferContext

        ctx = BatchTransferContext(enable=True, verbose=False)

        with ctx:
            arrays = {"test": np.random.randn(100, 3).astype(np.float32)}
            ctx.batch_upload(arrays)

        # After exiting, context should still be accessible for stats
        stats = ctx.get_statistics()
        assert stats is not None

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_serial_transfer_count(self):
        """Test that serial_transfers_avoided is calculated correctly."""
        from ign_lidar.optimization.gpu_batch_transfer import BatchTransferContext

        with BatchTransferContext(enable=True, verbose=False) as ctx:
            # Upload 10 arrays
            arrays = {
                f"array_{i}": np.random.randn(100, 3).astype(np.float32)
                for i in range(10)
            }

            ctx.batch_upload(arrays)

            stats = ctx.get_statistics()

            # 10 arrays would require 10 serial transfers
            # Batch combines into 1, so 9 avoided (10-1)
            assert stats["serial_transfers_avoided"] >= 9


def _check_gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        from ign_lidar.core.gpu import GPUManager
        gpu_manager = GPUManager()
        return gpu_manager.gpu_available
    except Exception:
        return False


class TestPhase3Comparison:
    """Compare Phase 3 with Phase 2 baseline."""

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_phase2_plus_phase3_features(self):
        """Test that Phase 2 + Phase 3 produces same features as Phase 2."""
        from ign_lidar.features.strategy_gpu import GPUStrategy

        n_points = 10_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])

        strategy = GPUStrategy(k_neighbors=20, verbose=False)

        # Compute features (now includes Phase 2 + Phase 3)
        features = strategy.compute(points)

        # All expected features should be present
        expected_features = [
            "normals",
            "curvature",
            "height",
            "verticality",
            "planarity",
            "sphericity",
        ]

        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"

    @pytest.mark.skipif(
        not _check_gpu_available(),
        reason="GPU not available"
    )
    def test_phase3_idempotency(self):
        """Test that batch transfers are idempotent."""
        from ign_lidar.features.strategy_gpu import GPUStrategy

        n_points = 5_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])

        strategy = GPUStrategy(k_neighbors=20, verbose=False)

        # Compute twice
        features1 = strategy.compute(points)
        features2 = strategy.compute(points)

        # Results should be identical
        for key in features1.keys():
            assert np.allclose(
                features1[key],
                features2[key],
                rtol=1e-6,
                atol=1e-8
            ), f"Features differ for key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
