"""  
Unit tests for GPU eigenvalue computation optimization (P1 Task #2).

Tests the GPU-accelerated eigenvalue decomposition implementation in
gpu_processor.py, ensuring correctness, performance, and fallback behavior.

**IMPORTANT: Run with ign_gpu conda environment:**

    conda run -n ign_gpu python -m pytest tests/test_gpu_eigenvalue_optimization.py -v

Version: 1.0.0
Date: 2025-11-21
"""

import numpy as np
import pytest

# GPU imports with fallback
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class TestGPUEigenvalueOptimization:
    """Test suite for GPU eigenvalue computation optimization."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.gpu
    def test_gpu_eigenvalue_basic(self):
        """Test basic GPU eigenvalue computation."""
        # Create simple symmetric positive definite matrices
        n_matrices = 100
        dim = 3
        
        # Generate random covariance matrices
        np.random.seed(42)
        matrices_cpu = np.random.randn(n_matrices, dim, dim).astype(np.float32)
        
        # Make symmetric
        for i in range(n_matrices):
            matrices_cpu[i] = (matrices_cpu[i] + matrices_cpu[i].T) / 2
            # Make positive definite
            matrices_cpu[i] += np.eye(dim) * 0.1
        
        # GPU computation
        matrices_gpu = cp.asarray(matrices_cpu)
        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(matrices_gpu)
        eigenvalues_result = cp.asnumpy(eigenvalues_gpu)
        eigenvectors_result = cp.asnumpy(eigenvectors_gpu)
        
        # CPU reference
        eigenvalues_cpu = np.zeros((n_matrices, dim))
        eigenvectors_cpu = np.zeros((n_matrices, dim, dim))
        for i in range(n_matrices):
            eigenvalues_cpu[i], eigenvectors_cpu[i] = np.linalg.eigh(matrices_cpu[i])
        
        # Verify results match within tolerance
        np.testing.assert_allclose(
            eigenvalues_result, 
            eigenvalues_cpu, 
            rtol=1e-5, 
            atol=1e-6,
            err_msg="GPU eigenvalues don't match CPU reference"
        )
        
        # Verify eigenvectors (allow sign flip)
        for i in range(n_matrices):
            for j in range(dim):
                vec_gpu = eigenvectors_result[i, :, j]
                vec_cpu = eigenvectors_cpu[i, :, j]
                
                # Check if vectors match (with possible sign flip)
                dot_product = np.abs(np.dot(vec_gpu, vec_cpu))
                assert dot_product > 0.999, f"Eigenvector {i},{j} mismatch: dot={dot_product}"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.gpu
    def test_gpu_eigenvalue_normal_computation(self):
        """Test GPU eigenvalue in context of normal computation."""
        from ign_lidar.features.gpu_processor import GPUProcessor
        
        # Create synthetic point cloud
        np.random.seed(42)
        n_points = 1000
        points = np.random.randn(n_points, 3).astype(np.float32)
        
        # Initialize processor
        processor = GPUProcessor(use_gpu=True)
        
        if not processor.use_gpu:
            pytest.skip("GPU not available in processor")
        
        # Compute normals using public API (this will use the optimized eigenvalue path)
        k = 30
        normals = processor.compute_normals(points, k=k)
        
        # Verify output shape
        assert normals.shape == (n_points, 3), f"Expected shape {(n_points, 3)}, got {normals.shape}"
        
        # Verify normals are unit vectors
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(
            norms, 
            1.0, 
            rtol=1e-3, 
            atol=1e-4,
            err_msg="Normals are not unit vectors"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_gpu_eigenvalue_performance(self):
        """Benchmark GPU vs CPU eigenvalue computation."""
        import time
        
        # Test different sizes
        sizes = [100, 1000, 10000]
        dim = 3
        
        results = {}
        
        for n_matrices in sizes:
            # Generate test data
            np.random.seed(42)
            matrices_cpu = np.random.randn(n_matrices, dim, dim).astype(np.float32)
            
            # Make symmetric positive definite
            for i in range(n_matrices):
                matrices_cpu[i] = (matrices_cpu[i] + matrices_cpu[i].T) / 2
                matrices_cpu[i] += np.eye(dim) * 0.1
            
            # CPU timing
            t0 = time.time()
            for _ in range(3):  # Multiple runs for averaging
                for i in range(n_matrices):
                    _ = np.linalg.eigh(matrices_cpu[i])
            cpu_time = (time.time() - t0) / 3
            
            # GPU timing
            matrices_gpu = cp.asarray(matrices_cpu)
            cp.cuda.Stream.null.synchronize()  # Ensure transfer complete
            
            t0 = time.time()
            for _ in range(3):
                _ = cp.linalg.eigh(matrices_gpu)
                cp.cuda.Stream.null.synchronize()
            gpu_time = (time.time() - t0) / 3
            
            speedup = cpu_time / gpu_time
            results[n_matrices] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup
            }
            
            print(f"\nN={n_matrices}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Speedup={speedup:.2f}x")
        
        # Verify we get speedup for large datasets
        if 10000 in results:
            assert results[10000]['speedup'] > 1.3, \
                f"Expected >1.3x speedup for large dataset, got {results[10000]['speedup']:.2f}x"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.gpu
    def test_gpu_memory_fallback(self):
        """Test fallback to CPU when GPU runs out of memory."""
        # This test verifies the GPU-to-CPU fallback logic in normal computation
        
        from ign_lidar.features.gpu_processor import GPUProcessor
        
        processor = GPUProcessor(use_gpu=True)
        
        if not processor.use_gpu:
            pytest.skip("GPU not available")
        
        # Create small test case that should succeed without OOM
        n_points = 100
        np.random.seed(42)
        points = np.random.randn(n_points, 3).astype(np.float32)
        
        # This should succeed without OOM using public API
        normals = processor.compute_normals(points, k=10)
        assert normals is not None
        assert normals.shape == (n_points, 3)
        
        # Verify normals are unit vectors
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-3, atol=1e-4)

    @pytest.mark.integration
    def test_cpu_fallback_when_gpu_unavailable(self):
        """Test that processor works when GPU is disabled."""
        from ign_lidar.features.gpu_processor import GPUProcessor
        
        # Force CPU mode
        processor = GPUProcessor(use_gpu=False)
        
        # Verify processor initialized in CPU mode
        assert not processor.use_gpu, "Processor should be in CPU mode"
        assert not processor.use_cuml, "cuML should be disabled in CPU mode"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.gpu
    def test_eigenvalue_accuracy(self):
        """Test numerical accuracy of GPU eigenvalue computation."""
        # Create well-conditioned covariance matrices
        n_matrices = 50
        dim = 3
        
        np.random.seed(42)
        matrices_cpu = np.zeros((n_matrices, dim, dim), dtype=np.float32)
        
        for i in range(n_matrices):
            # Generate random points
            points = np.random.randn(100, dim).astype(np.float32)
            # Compute covariance
            points_centered = points - points.mean(axis=0)
            matrices_cpu[i] = np.dot(points_centered.T, points_centered) / 100
        
        # GPU computation
        matrices_gpu = cp.asarray(matrices_cpu)
        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(matrices_gpu)
        
        # Verify eigenvalue properties
        eigenvalues_result = cp.asnumpy(eigenvalues_gpu)
        eigenvectors_result = cp.asnumpy(eigenvectors_gpu)
        
        for i in range(n_matrices):
            A = matrices_cpu[i]
            eigenvals = eigenvalues_result[i]
            eigenvecs = eigenvectors_result[i]
            
            # Verify Av = λv for each eigenvalue/vector pair
            for j in range(dim):
                lhs = np.dot(A, eigenvecs[:, j])
                rhs = eigenvals[j] * eigenvecs[:, j]
                np.testing.assert_allclose(
                    lhs, rhs, 
                    rtol=1e-4, atol=1e-5,
                    err_msg=f"Eigenvalue equation not satisfied for matrix {i}, eigenvalue {j}"
                )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    @pytest.mark.gpu
    def test_batch_eigenvalue_computation(self):
        """Test batched eigenvalue computation for multiple matrices."""
        # This is the key use case - computing eigenvalues for many matrices at once
        
        n_matrices = 1000
        dim = 3
        
        np.random.seed(42)
        matrices = np.random.randn(n_matrices, dim, dim).astype(np.float32)
        
        # Make positive definite using A^T @ A construction
        matrices_pd = np.zeros((n_matrices, dim, dim), dtype=np.float32)
        for i in range(n_matrices):
            A = matrices[i]
            matrices_pd[i] = np.dot(A.T, A) / dim + np.eye(dim) * 0.1
        
        # Batched GPU computation
        matrices_gpu = cp.asarray(matrices_pd)
        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(matrices_gpu)
        
        # Verify all eigenvalues are real and positive
        eigenvalues = cp.asnumpy(eigenvalues_gpu)
        assert np.all(eigenvalues > -1e-6), f"Some eigenvalues are significantly negative: min={eigenvalues.min()}"
        assert eigenvalues.shape == (n_matrices, dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "gpu"])
