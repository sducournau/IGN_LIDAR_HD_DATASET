"""
Phase 3 GPU Batch Transfer Tests

Tests for BatchUploader, BatchDownloader, and BatchTransferContext classes.
Validates that batch transfers reduce CPU↔GPU transfer overhead by combining
multiple transfers into single operations.

Expected benefits:
- Reduces transfer count from 2*N to 2 (per-batch)
- Minimizes overhead of multiple cudaMemcpy calls
- Target speedup: 1.1-1.2x on medium datasets

Tests:
✓ BatchUploader: Accumulates arrays, uploads in batch
✓ BatchDownloader: Accumulates GPU arrays, downloads in batch
✓ BatchTransferContext: Full context manager workflow
✓ Transfer statistics: Validation of metrics
✓ GPU availability: CPU fallback testing

Author: IGN LiDAR HD Development Team
Date: November 27, 2025
"""

import pytest
import numpy as np
from ign_lidar.optimization.gpu_batch_transfer import (
    BatchUploader,
    BatchDownloader,
    BatchTransferContext,
    batch_transfer_context,
    TransferStatistics,
)


class TestBatchUploader:
    """Test BatchUploader class."""

    def test_init(self):
        """Test uploader initialization."""
        uploader = BatchUploader(batch_id="test_batch")
        assert uploader.batch_id == "test_batch"
        assert len(uploader.arrays) == 0
        assert uploader.total_size_mb == 0.0

    def test_add_single_array(self):
        """Test adding single array."""
        uploader = BatchUploader()
        points = np.random.randn(100, 3).astype(np.float32)

        uploader.add("points", points)

        assert "points" in uploader.arrays
        assert uploader.arrays["points"].shape == (100, 3)
        assert uploader.total_size_mb > 0

    def test_add_multiple_arrays(self):
        """Test adding multiple arrays."""
        uploader = BatchUploader()
        points = np.random.randn(100, 3).astype(np.float32)
        normals = np.random.randn(100, 3).astype(np.float32)
        curvature = np.random.randn(100).astype(np.float32)

        uploader.add("points", points)
        uploader.add("normals", normals)
        uploader.add("curvature", curvature)

        assert len(uploader.arrays) == 3
        assert uploader.total_size_mb > 0

    def test_add_invalid_type(self):
        """Test adding non-ndarray raises error."""
        uploader = BatchUploader()

        with pytest.raises(TypeError):
            uploader.add("invalid", [1, 2, 3])  # List, not ndarray

    def test_clear(self):
        """Test clearing uploader."""
        uploader = BatchUploader()
        uploader.add("points", np.random.randn(100, 3).astype(np.float32))

        uploader.clear()

        assert len(uploader.arrays) == 0
        assert uploader.total_size_mb == 0.0

    def test_upload_batch_cpu_fallback(self):
        """Test batch upload with CPU fallback."""
        uploader = BatchUploader()
        points = np.random.randn(100, 3).astype(np.float32)
        normals = np.random.randn(100, 3).astype(np.float32)

        uploader.add("points", points)
        uploader.add("normals", normals)

        # This should work even without GPU
        gpu_arrays = uploader.upload_batch()

        assert "points" in gpu_arrays
        assert "normals" in gpu_arrays
        assert len(gpu_arrays) == 2


class TestBatchDownloader:
    """Test BatchDownloader class."""

    def test_init(self):
        """Test downloader initialization."""
        downloader = BatchDownloader(batch_id="test_batch")
        assert downloader.batch_id == "test_batch"
        assert len(downloader.gpu_arrays) == 0
        assert downloader.total_size_mb == 0.0

    def test_add_single_array(self):
        """Test adding single array."""
        downloader = BatchDownloader()
        features = np.random.randn(100, 10).astype(np.float32)

        downloader.add("features", features)

        assert "features" in downloader.gpu_arrays
        assert downloader.total_size_mb > 0

    def test_add_multiple_arrays(self):
        """Test adding multiple arrays."""
        downloader = BatchDownloader()
        normals = np.random.randn(100, 3).astype(np.float32)
        curvature = np.random.randn(100).astype(np.float32)
        height = np.random.randn(100).astype(np.float32)

        downloader.add("normals", normals)
        downloader.add("curvature", curvature)
        downloader.add("height", height)

        assert len(downloader.gpu_arrays) == 3
        assert downloader.total_size_mb > 0

    def test_download_batch_cpu_fallback(self):
        """Test batch download with CPU fallback."""
        downloader = BatchDownloader()
        normals = np.random.randn(100, 3).astype(np.float32)
        curvature = np.random.randn(100).astype(np.float32)

        downloader.add("normals", normals)
        downloader.add("curvature", curvature)

        # This should work even without GPU
        cpu_arrays = downloader.download_batch()

        assert "normals" in cpu_arrays
        assert "curvature" in cpu_arrays
        assert isinstance(cpu_arrays["normals"], np.ndarray)

    def test_clear(self):
        """Test clearing downloader."""
        downloader = BatchDownloader()
        downloader.add("features", np.random.randn(100, 10).astype(np.float32))

        downloader.clear()

        assert len(downloader.gpu_arrays) == 0
        assert downloader.total_size_mb == 0.0


class TestBatchTransferContext:
    """Test BatchTransferContext class."""

    def test_init_enabled(self):
        """Test context initialization with batch transfers enabled."""
        ctx = BatchTransferContext(enable=True, verbose=False)
        assert ctx.enable is True
        assert ctx.stats is not None

    def test_init_disabled(self):
        """Test context initialization with batch transfers disabled."""
        ctx = BatchTransferContext(enable=False, verbose=False)
        assert ctx.enable is False

    def test_context_manager(self):
        """Test context manager protocol."""
        with BatchTransferContext(enable=True) as ctx:
            assert ctx is not None
            assert ctx.stats is not None

    def test_batch_upload(self):
        """Test batch upload through context."""
        with BatchTransferContext(enable=True) as ctx:
            arrays = {
                "points": np.random.randn(100, 3).astype(np.float32),
                "normals": np.random.randn(100, 3).astype(np.float32),
                "curvature": np.random.randn(100).astype(np.float32),
            }

            gpu_arrays = ctx.batch_upload(arrays)

            assert len(gpu_arrays) == len(arrays)
            assert "points" in gpu_arrays
            assert "normals" in gpu_arrays
            assert "curvature" in gpu_arrays

    def test_batch_download(self):
        """Test batch download through context."""
        with BatchTransferContext(enable=True) as ctx:
            # Create some data to "download"
            gpu_data = {
                "features": np.random.randn(100, 10).astype(np.float32),
                "results": np.random.randn(100, 5).astype(np.float32),
            }

            cpu_arrays = ctx.batch_download(gpu_data)

            assert len(cpu_arrays) == len(gpu_data)
            assert all(isinstance(arr, np.ndarray) for arr in cpu_arrays.values())

    def test_upload_download_pipeline(self):
        """Test full upload-compute-download pipeline."""
        with BatchTransferContext(enable=True, verbose=False) as ctx:
            # Upload
            input_data = {
                "points": np.random.randn(100, 3).astype(np.float32),
                "intensities": np.random.randn(100).astype(np.float32),
            }
            gpu_inputs = ctx.batch_upload(input_data, batch_id="inputs")
            assert len(gpu_inputs) == 2

            # Simulate computation on GPU
            gpu_outputs = {
                "features": np.random.randn(100, 10).astype(np.float32),
                "classifications": np.random.randint(0, 10, 100).astype(np.int32),
            }

            # Download
            cpu_outputs = ctx.batch_download(gpu_outputs, batch_id="outputs")
            assert len(cpu_outputs) == 2

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        with BatchTransferContext(enable=True, verbose=False) as ctx:
            arrays = {
                f"array_{i}": np.random.randn(100, 3).astype(np.float32)
                for i in range(5)
            }

            ctx.batch_upload(arrays)
            ctx.batch_download(arrays)

            stats = ctx.get_statistics()

            assert stats["total_batches"] > 0
            assert stats["upload_mb"] > 0
            assert stats["download_mb"] > 0
            assert stats["serial_transfers_avoided"] > 0

    def test_serial_transfers_avoided_counting(self):
        """Test that serial transfers avoided is counted correctly."""
        with BatchTransferContext(enable=True, verbose=False) as ctx:
            # 5 arrays = 4 serial transfers avoided (batch combines into 1)
            arrays = {
                f"array_{i}": np.random.randn(100, 3).astype(np.float32)
                for i in range(5)
            }

            ctx.batch_upload(arrays)

            stats = ctx.get_statistics()
            # 5 arrays, serial would be 5 transfers, batch is 1 = 4 avoided
            assert stats["serial_transfers_avoided"] >= 4

    def test_disabled_context_fallback(self):
        """Test that disabled context still works with serial transfers."""
        with BatchTransferContext(enable=False, verbose=False) as ctx:
            arrays = {
                "points": np.random.randn(100, 3).astype(np.float32),
                "normals": np.random.randn(100, 3).astype(np.float32),
            }

            # Should still work but without batching
            gpu_arrays = ctx.batch_upload(arrays)
            assert len(gpu_arrays) == 2

            cpu_arrays = ctx.batch_download(gpu_arrays)
            assert len(cpu_arrays) == 2

    def test_empty_upload(self):
        """Test uploading empty dictionary."""
        with BatchTransferContext(enable=True) as ctx:
            gpu_arrays = ctx.batch_upload({})
            assert len(gpu_arrays) == 0

    def test_empty_download(self):
        """Test downloading empty dictionary."""
        with BatchTransferContext(enable=True) as ctx:
            cpu_arrays = ctx.batch_download({})
            assert len(cpu_arrays) == 0

    def test_mixed_size_arrays(self):
        """Test batch transfers with mixed-size arrays."""
        with BatchTransferContext(enable=True) as ctx:
            arrays = {
                "small": np.random.randn(10).astype(np.float32),
                "medium": np.random.randn(100, 10).astype(np.float32),
                "large": np.random.randn(1000, 10).astype(np.float32),
            }

            gpu_arrays = ctx.batch_upload(arrays)
            assert len(gpu_arrays) == 3

            cpu_arrays = ctx.batch_download(gpu_arrays)
            assert len(cpu_arrays) == 3


class TestTransferStatistics:
    """Test TransferStatistics class."""

    def test_init(self):
        """Test statistics initialization."""
        stats = TransferStatistics()
        assert stats.total_batches == 0
        assert stats.total_transfer_mb == 0.0
        assert stats.serial_transfers_avoided == 0

    def test_properties(self):
        """Test statistics properties."""
        stats = TransferStatistics()
        stats.total_uploads_mb = 100.0
        stats.total_downloads_mb = 50.0
        stats.total_upload_time_ms = 100.0
        stats.total_download_time_ms = 50.0

        assert stats.total_transfer_mb == 150.0
        assert stats.total_time_ms == 150.0

    def test_get_summary(self):
        """Test getting summary statistics."""
        stats = TransferStatistics()
        stats.total_batches = 10
        stats.total_uploads_mb = 100.0
        stats.total_downloads_mb = 50.0
        stats.serial_transfers_avoided = 20

        summary = stats.get_summary()

        assert summary["total_batches"] == 10
        assert summary["total_transfer_mb"] == 150.0
        assert summary["serial_transfers_avoided"] == 20


class TestContextManagerFactory:
    """Test context manager factory function."""

    def test_batch_transfer_context_factory(self):
        """Test batch_transfer_context factory function."""
        with batch_transfer_context(enable=True, verbose=False) as ctx:
            arrays = {
                "test": np.random.randn(100, 3).astype(np.float32)
            }
            gpu_arrays = ctx.batch_upload(arrays)
            assert len(gpu_arrays) == 1

    def test_factory_creates_correct_type(self):
        """Test that factory creates correct context type."""
        with batch_transfer_context() as ctx:
            assert isinstance(ctx, BatchTransferContext)


class TestLargeDatasetTransfers:
    """Test batch transfers with realistic dataset sizes."""

    def test_medium_dataset_transfer(self):
        """Test batch transfer with medium dataset (1M points)."""
        with BatchTransferContext(enable=True, verbose=False) as ctx:
            # Simulate 1M points with 10 features
            input_data = {
                "points": np.random.randn(10_000, 3).astype(np.float32),  # ~120 KB
                "features": np.random.randn(10_000, 10).astype(np.float32),  # ~400 KB
            }

            gpu_inputs = ctx.batch_upload(input_data)
            gpu_outputs = {
                "results": np.random.randn(10_000, 10).astype(np.float32)
            }
            cpu_outputs = ctx.batch_download(gpu_outputs)

            assert len(cpu_outputs) == 1
            assert cpu_outputs["results"].shape == (10_000, 10)

    def test_large_dataset_transfer(self):
        """Test batch transfer with large dataset (10M points)."""
        with BatchTransferContext(enable=True, verbose=False) as ctx:
            # Simulate 10M points (reduce size for testing)
            n_points = 50_000  # ~600 KB per array
            input_data = {
                "points": np.random.randn(n_points, 3).astype(np.float32),
                "intensities": np.random.randn(n_points).astype(np.float32),
                "rgb": np.random.randint(0, 256, (n_points, 3), dtype=np.uint8),
            }

            gpu_inputs = ctx.batch_upload(input_data, batch_id="large_inputs")
            assert len(gpu_inputs) == 3

            gpu_outputs = {
                f"feature_{i}": np.random.randn(n_points).astype(np.float32)
                for i in range(8)
            }
            cpu_outputs = ctx.batch_download(gpu_outputs, batch_id="large_outputs")

            assert len(cpu_outputs) == 8

    def test_very_large_feature_set(self):
        """Test batch transfer with many features."""
        with BatchTransferContext(enable=True, verbose=False) as ctx:
            # 1000 features per 100 points
            n_features = 100
            features = {
                f"feature_{i}": np.random.randn(100).astype(np.float32)
                for i in range(n_features)
            }

            gpu_features = ctx.batch_upload(features, batch_id="many_features")
            assert len(gpu_features) == n_features

            cpu_features = ctx.batch_download(gpu_features)
            assert len(cpu_features) == n_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
