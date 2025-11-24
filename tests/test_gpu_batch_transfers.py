"""
Tests for GPU batch transfer optimization in GPUManager.

Tests the batch upload/download methods added in v3.5.3 that reduce
PCIe transaction overhead for GPU operations.

Expected performance improvement: +10-30% for multiple array transfers.

Author: Consolidation Phase
Date: November 24, 2025
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ign_lidar.core.gpu import GPUManager


@pytest.fixture
def gpu_manager():
    """Get GPUManager instance."""
    return GPUManager()


@pytest.fixture
def sample_arrays():
    """Generate sample NumPy arrays for testing."""
    np.random.seed(42)
    return {
        'points': np.random.rand(1000, 3).astype(np.float32),
        'features': np.random.rand(1000, 10).astype(np.float32),
        'labels': np.random.randint(0, 10, 1000).astype(np.int32),
    }


class TestBatchUpload:
    """Test suite for batch_upload() method."""
    
    def test_batch_upload_exists(self, gpu_manager):
        """Test that batch_upload method exists."""
        assert hasattr(gpu_manager, 'batch_upload')
        assert callable(gpu_manager.batch_upload)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_batch_upload_single_array(self, gpu_manager, sample_arrays):
        """Test batch upload with a single array."""
        points = sample_arrays['points']
        
        result = gpu_manager.batch_upload(points)
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        
        points_gpu = result[0]
        cp = gpu_manager.get_cupy()
        assert isinstance(points_gpu, cp.ndarray)
        assert points_gpu.shape == points.shape
        assert cp.allclose(points_gpu, points)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_batch_upload_multiple_arrays(self, gpu_manager, sample_arrays):
        """Test batch upload with multiple arrays."""
        points = sample_arrays['points']
        features = sample_arrays['features']
        labels = sample_arrays['labels']
        
        result = gpu_manager.batch_upload(points, features, labels)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        points_gpu, features_gpu, labels_gpu = result
        cp = gpu_manager.get_cupy()
        
        # Verify all arrays uploaded correctly
        assert isinstance(points_gpu, cp.ndarray)
        assert isinstance(features_gpu, cp.ndarray)
        assert isinstance(labels_gpu, cp.ndarray)
        
        # Verify shapes preserved
        assert points_gpu.shape == points.shape
        assert features_gpu.shape == features.shape
        assert labels_gpu.shape == labels.shape
        
        # Verify data integrity
        assert cp.allclose(points_gpu, points)
        assert cp.allclose(features_gpu, features)
        assert cp.allclose(labels_gpu, labels)
    
    def test_batch_upload_raises_without_gpu(self):
        """Test that batch_upload raises error when GPU unavailable."""
        gpu_manager = GPUManager()
        
        if not gpu_manager.gpu_available:
            with pytest.raises(ImportError, match="GPU not available"):
                gpu_manager.batch_upload(np.array([1, 2, 3]))
        else:
            pytest.skip("GPU is available, cannot test error case")
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_batch_upload_preserves_order(self, gpu_manager):
        """Test that batch upload preserves array order."""
        arr1 = np.array([1, 2, 3], dtype=np.float32)
        arr2 = np.array([4, 5, 6], dtype=np.float32)
        arr3 = np.array([7, 8, 9], dtype=np.float32)
        
        result = gpu_manager.batch_upload(arr1, arr2, arr3)
        
        cp = gpu_manager.get_cupy()
        r1, r2, r3 = result
        
        assert cp.allclose(r1, arr1)
        assert cp.allclose(r2, arr2)
        assert cp.allclose(r3, arr3)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_batch_upload_different_dtypes(self, gpu_manager):
        """Test batch upload with different data types."""
        float_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        int_arr = np.array([1, 2, 3], dtype=np.int32)
        bool_arr = np.array([True, False, True], dtype=np.bool_)
        
        result = gpu_manager.batch_upload(float_arr, int_arr, bool_arr)
        
        assert len(result) == 3
        cp = gpu_manager.get_cupy()
        
        float_gpu, int_gpu, bool_gpu = result
        assert float_gpu.dtype == float_arr.dtype
        assert int_gpu.dtype == int_arr.dtype
        assert bool_gpu.dtype == bool_arr.dtype


class TestBatchDownload:
    """Test suite for batch_download() method."""
    
    def test_batch_download_exists(self, gpu_manager):
        """Test that batch_download method exists."""
        assert hasattr(gpu_manager, 'batch_download')
        assert callable(gpu_manager.batch_download)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_batch_download_single_array(self, gpu_manager, sample_arrays):
        """Test batch download with a single array."""
        points = sample_arrays['points']
        cp = gpu_manager.get_cupy()
        
        # Upload to GPU first
        points_gpu = cp.asarray(points)
        
        # Download using batch
        result = gpu_manager.batch_download(points_gpu)
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        
        points_cpu = result[0]
        assert isinstance(points_cpu, np.ndarray)
        assert points_cpu.shape == points.shape
        assert np.allclose(points_cpu, points)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_batch_download_multiple_arrays(self, gpu_manager, sample_arrays):
        """Test batch download with multiple arrays."""
        points = sample_arrays['points']
        features = sample_arrays['features']
        labels = sample_arrays['labels']
        
        cp = gpu_manager.get_cupy()
        
        # Upload to GPU
        points_gpu = cp.asarray(points)
        features_gpu = cp.asarray(features)
        labels_gpu = cp.asarray(labels)
        
        # Batch download
        result = gpu_manager.batch_download(points_gpu, features_gpu, labels_gpu)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        points_cpu, features_cpu, labels_cpu = result
        
        # Verify all arrays downloaded correctly
        assert isinstance(points_cpu, np.ndarray)
        assert isinstance(features_cpu, np.ndarray)
        assert isinstance(labels_cpu, np.ndarray)
        
        # Verify shapes preserved
        assert points_cpu.shape == points.shape
        assert features_cpu.shape == features.shape
        assert labels_cpu.shape == labels.shape
        
        # Verify data integrity
        assert np.allclose(points_cpu, points)
        assert np.allclose(features_cpu, features)
        assert np.allclose(labels_cpu, labels)
    
    def test_batch_download_raises_without_gpu(self):
        """Test that batch_download raises error when GPU unavailable."""
        gpu_manager = GPUManager()
        
        if not gpu_manager.gpu_available:
            with pytest.raises(ImportError, match="GPU not available"):
                gpu_manager.batch_download(MagicMock())
        else:
            pytest.skip("GPU is available, cannot test error case")
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_batch_download_preserves_order(self, gpu_manager):
        """Test that batch download preserves array order."""
        cp = gpu_manager.get_cupy()
        
        arr1_gpu = cp.array([1, 2, 3], dtype=cp.float32)
        arr2_gpu = cp.array([4, 5, 6], dtype=cp.float32)
        arr3_gpu = cp.array([7, 8, 9], dtype=cp.float32)
        
        result = gpu_manager.batch_download(arr1_gpu, arr2_gpu, arr3_gpu)
        
        r1, r2, r3 = result
        
        assert np.allclose(r1, [1, 2, 3])
        assert np.allclose(r2, [4, 5, 6])
        assert np.allclose(r3, [7, 8, 9])


class TestBatchTransferIntegration:
    """Integration tests for upload+download workflows."""
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_upload_download_roundtrip(self, gpu_manager, sample_arrays):
        """Test that upload followed by download returns original data."""
        points = sample_arrays['points']
        features = sample_arrays['features']
        labels = sample_arrays['labels']
        
        # Upload
        points_gpu, features_gpu, labels_gpu = gpu_manager.batch_upload(
            points, features, labels
        )
        
        # Download
        points_back, features_back, labels_back = gpu_manager.batch_download(
            points_gpu, features_gpu, labels_gpu
        )
        
        # Verify data integrity through roundtrip
        assert np.allclose(points_back, points)
        assert np.allclose(features_back, features)
        assert np.allclose(labels_back, labels)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_batch_transfers_with_computation(self, gpu_manager, sample_arrays):
        """Test batch transfers with GPU computation in between."""
        points = sample_arrays['points']
        features = sample_arrays['features']
        
        cp = gpu_manager.get_cupy()
        
        # Upload
        points_gpu, features_gpu = gpu_manager.batch_upload(points, features)
        
        # Perform some GPU computation
        means_gpu = cp.mean(points_gpu, axis=0)
        stds_gpu = cp.std(features_gpu, axis=0)
        
        # Download results
        means_cpu, stds_cpu = gpu_manager.batch_download(means_gpu, stds_gpu)
        
        # Verify computation correctness
        expected_means = np.mean(points, axis=0)
        expected_stds = np.std(features, axis=0)
        
        assert np.allclose(means_cpu, expected_means)
        assert np.allclose(stds_cpu, expected_stds)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_batch_transfer_memory_efficiency(self, gpu_manager):
        """Test that batch transfers don't leak memory."""
        cp = gpu_manager.get_cupy()
        
        # Get initial memory state
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        initial_used = mempool.used_bytes()
        
        # Perform multiple batch transfers
        for _ in range(10):
            arrays = [np.random.rand(1000, 10).astype(np.float32) for _ in range(3)]
            gpu_arrays = gpu_manager.batch_upload(*arrays)
            cpu_arrays = gpu_manager.batch_download(*gpu_arrays)
            
            # Explicit cleanup
            del gpu_arrays
            del cpu_arrays
        
        # Force cleanup
        mempool.free_all_blocks()
        final_used = mempool.used_bytes()
        
        # Memory usage should be similar (within 10 MB difference)
        memory_diff_mb = abs(final_used - initial_used) / (1024 ** 2)
        assert memory_diff_mb < 10, f"Memory leak detected: {memory_diff_mb:.2f} MB"


@pytest.mark.unit
class TestBatchTransferAPI:
    """Test API design and usability."""
    
    def test_batch_upload_returns_tuple(self, gpu_manager):
        """Test that batch_upload always returns tuple."""
        if not gpu_manager.gpu_available:
            pytest.skip("GPU not available")
        
        single_result = gpu_manager.batch_upload(np.array([1, 2, 3]))
        assert isinstance(single_result, tuple)
        assert len(single_result) == 1
        
        multi_result = gpu_manager.batch_upload(
            np.array([1, 2]), np.array([3, 4])
        )
        assert isinstance(multi_result, tuple)
        assert len(multi_result) == 2
    
    def test_batch_download_returns_tuple(self, gpu_manager):
        """Test that batch_download always returns tuple."""
        if not gpu_manager.gpu_available:
            pytest.skip("GPU not available")
        
        cp = gpu_manager.get_cupy()
        
        single_result = gpu_manager.batch_download(cp.array([1, 2, 3]))
        assert isinstance(single_result, tuple)
        assert len(single_result) == 1
        
        multi_result = gpu_manager.batch_download(
            cp.array([1, 2]), cp.array([3, 4])
        )
        assert isinstance(multi_result, tuple)
        assert len(multi_result) == 2
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_batch_upload_unpacking(self, gpu_manager):
        """Test that unpacking works correctly."""
        arrays = [np.array([i, i+1, i+2]) for i in range(5)]
        
        # Should be able to unpack directly
        a, b, c, d, e = gpu_manager.batch_upload(*arrays)
        
        cp = gpu_manager.get_cupy()
        assert isinstance(a, cp.ndarray)
        assert isinstance(e, cp.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
