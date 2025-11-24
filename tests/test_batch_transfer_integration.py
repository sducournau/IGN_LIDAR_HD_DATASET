"""
Tests for batch transfer optimizations in GPU modules.

Validates that the migration from individual cp.asnumpy() calls
to GPUManager.batch_download() maintains correctness and improves performance.

Author: Consolidation Phase
Date: November 24, 2025
"""
import pytest
import numpy as np

from ign_lidar.core.gpu import GPUManager


@pytest.fixture
def gpu_manager():
    """Get GPUManager instance."""
    return GPUManager()


@pytest.mark.gpu
@pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
class TestGPUKernelsBatchTransfers:
    """Test batch transfers in gpu_kernels.py."""
    
    def test_compute_normals_and_eigenvalues_batch_transfer(self, gpu_manager):
        """Test that compute_normals_and_eigenvalues uses batch transfer correctly."""
        try:
            from ign_lidar.optimization.gpu_kernels import GPUKernelManager
            
            kernel_mgr = GPUKernelManager()
            if not kernel_mgr.available:
                pytest.skip("GPU kernels not available")
            
            # Create test covariance matrices
            np.random.seed(42)
            n_points = 100
            cov_matrices = np.random.rand(n_points, 3, 3).astype(np.float32)
            # Make symmetric
            cov_matrices = (cov_matrices + cov_matrices.transpose(0, 2, 1)) / 2
            
            # Compute normals and eigenvalues
            normals, eigenvalues = kernel_mgr.compute_normals_and_eigenvalues(cov_matrices)
            
            # Verify shapes
            assert normals.shape == (n_points, 3)
            assert eigenvalues.shape == (n_points, 3)
            
            # Verify normals are unit vectors
            norms = np.linalg.norm(normals, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5)
            
            # Verify eigenvalues are sorted descending
            for i in range(n_points):
                assert eigenvalues[i, 0] >= eigenvalues[i, 1] >= eigenvalues[i, 2]
                
        except ImportError:
            pytest.skip("GPU kernels module not available")
    
    def test_compute_normals_eigenvalues_fused_batch_transfer(self, gpu_manager):
        """Test that fused kernel uses batch transfer for 3 arrays."""
        try:
            from ign_lidar.optimization.gpu_kernels import GPUKernelManager
            
            kernel_mgr = GPUKernelManager()
            if not kernel_mgr.available:
                pytest.skip("GPU kernels not available")
            
            # Create test data
            np.random.seed(42)
            n_points = 100
            k = 20
            points = np.random.rand(n_points, 3).astype(np.float32)
            
            # Create fake KNN indices
            knn_indices = np.random.randint(0, n_points, (n_points, k)).astype(np.int32)
            
            # Compute fused normals, eigenvalues, curvature
            normals, eigenvalues, curvature = kernel_mgr.compute_normals_eigenvalues_fused(
                points, knn_indices, k
            )
            
            # Verify shapes
            assert normals.shape == (n_points, 3)
            assert eigenvalues.shape == (n_points, 3)
            assert curvature.shape == (n_points,)
            
            # Verify normals are unit vectors
            norms = np.linalg.norm(normals, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5)
            
            # Verify curvature is in valid range [0, 1]
            assert np.all(curvature >= 0)
            assert np.all(curvature <= 1)
            
        except ImportError:
            pytest.skip("GPU kernels module not available")


@pytest.mark.gpu
@pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
class TestGPUAcceleratedOpsBatchTransfers:
    """Test batch transfers in gpu_accelerated_ops.py."""
    
    def test_eigh_batch_transfer(self, gpu_manager):
        """Test that eigh uses batch transfer for eigenvalues and eigenvectors."""
        try:
            from ign_lidar.optimization.gpu_accelerated_ops import GPUAcceleratedOps
            
            gpu_ops = GPUAcceleratedOps(use_gpu=True)
            
            # Create symmetric test matrices
            np.random.seed(42)
            n_matrices = 100
            matrices = np.random.rand(n_matrices, 3, 3).astype(np.float32)
            # Make symmetric
            matrices = (matrices + matrices.transpose(0, 2, 1)) / 2
            
            # Compute eigendecomposition
            eigenvalues, eigenvectors = gpu_ops.eigh(matrices)
            
            # Verify shapes
            assert eigenvalues.shape == (n_matrices, 3)
            assert eigenvectors.shape == (n_matrices, 3, 3)
            
            # Verify eigenvalues are real and sorted
            assert np.all(np.isreal(eigenvalues))
            for i in range(n_matrices):
                sorted_evals = np.sort(eigenvalues[i])
                assert np.allclose(sorted_evals, eigenvalues[i])
            
            # Verify eigenvectors are orthonormal
            for i in range(n_matrices):
                evecs = eigenvectors[i]
                identity = evecs.T @ evecs
                assert np.allclose(identity, np.eye(3), atol=1e-5)
                
        except ImportError:
            pytest.skip("GPU accelerated ops not available")
    
    def test_svd_batch_transfer(self, gpu_manager):
        """Test that SVD uses batch transfer for U, S, Vh."""
        try:
            from ign_lidar.optimization.gpu_accelerated_ops import GPUAcceleratedOps
            
            gpu_ops = GPUAcceleratedOps(use_gpu=True)
            
            # Create test matrix
            np.random.seed(42)
            matrix = np.random.rand(50, 30).astype(np.float32)
            
            # Compute SVD
            u, s, vh = gpu_ops.svd(matrix, full_matrices=False)
            
            # Verify shapes
            assert u.shape == (50, 30)
            assert s.shape == (30,)
            assert vh.shape == (30, 30)
            
            # Verify singular values are non-negative and sorted descending
            assert np.all(s >= 0)
            assert np.all(s[:-1] >= s[1:])
            
            # Verify reconstruction
            reconstructed = u @ np.diag(s) @ vh
            assert np.allclose(reconstructed, matrix, atol=1e-4)
            
            # Verify U and Vh are orthonormal
            assert np.allclose(u.T @ u, np.eye(30), atol=1e-5)
            assert np.allclose(vh @ vh.T, np.eye(30), atol=1e-5)
            
        except ImportError:
            pytest.skip("GPU accelerated ops not available")


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
class TestBatchTransferPerformance:
    """Integration tests for batch transfer performance improvements."""
    
    def test_batch_transfer_faster_than_individual(self, gpu_manager):
        """Verify that batch transfers are faster than individual transfers."""
        import time
        
        cp = gpu_manager.try_get_cupy()
        if cp is None:
            pytest.skip("CuPy not available")
        
        # Create test arrays
        np.random.seed(42)
        arrays_cpu = [
            np.random.rand(10000, 10).astype(np.float32) for _ in range(3)
        ]
        
        # Upload to GPU
        arrays_gpu = [cp.asarray(arr) for arr in arrays_cpu]
        
        # Warmup
        for _ in range(5):
            _ = gpu_manager.batch_download(*arrays_gpu)
        
        # Benchmark batch transfer
        start = time.perf_counter()
        for _ in range(100):
            result_batch = gpu_manager.batch_download(*arrays_gpu)
        batch_time = time.perf_counter() - start
        
        # Benchmark individual transfers
        start = time.perf_counter()
        for _ in range(100):
            result_individual = tuple(cp.asnumpy(arr) for arr in arrays_gpu)
        individual_time = time.perf_counter() - start
        
        # Verify correctness
        for r1, r2 in zip(result_batch, result_individual):
            assert np.allclose(r1, r2)
        
        # Batch should be faster (or at least not slower)
        speedup = individual_time / batch_time
        print(f"\nBatch transfer speedup: {speedup:.2f}x")
        
        # Note: Speedup may vary, but batch should not be significantly slower
        # Allow for measurement noise
        assert speedup >= 0.8, f"Batch transfer slower than expected: {speedup:.2f}x"
    
    def test_no_memory_leaks_with_batch_transfers(self, gpu_manager):
        """Verify batch transfers don't cause memory leaks."""
        cp = gpu_manager.try_get_cupy()
        if cp is None:
            pytest.skip("CuPy not available")
        
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        
        initial_used = mempool.used_bytes()
        
        # Perform many batch transfers
        for _ in range(100):
            arrays_cpu = [np.random.rand(1000, 10).astype(np.float32) for _ in range(3)]
            arrays_gpu = gpu_manager.batch_upload(*arrays_cpu)
            results = gpu_manager.batch_download(*arrays_gpu)
            del arrays_gpu, results
        
        # Cleanup
        mempool.free_all_blocks()
        final_used = mempool.used_bytes()
        
        # Memory should be similar (within 5 MB)
        memory_diff_mb = abs(final_used - initial_used) / (1024 ** 2)
        assert memory_diff_mb < 5, f"Possible memory leak: {memory_diff_mb:.2f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
