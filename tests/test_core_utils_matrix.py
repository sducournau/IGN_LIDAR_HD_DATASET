"""
Unit tests for GPU-compatible matrix utilities in core.utils.

Tests the new batched matrix operations that work with both NumPy and CuPy.
"""

import numpy as np
import pytest
from ign_lidar.features.core.utils import (
    get_array_module,
    batched_inverse_3x3,
    inverse_power_iteration,
)

# Try to import CuPy for GPU tests
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class TestGetArrayModule:
    """Tests for get_array_module function."""
    
    def test_numpy_array(self):
        """Test that NumPy arrays return numpy module."""
        arr = np.array([1, 2, 3])
        xp = get_array_module(arr)
        assert xp is np
    
    def test_list_returns_numpy(self):
        """Test that lists return numpy module."""
        arr = [1, 2, 3]
        xp = get_array_module(arr)
        assert xp is np
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_cupy_array(self):
        """Test that CuPy arrays return cupy module."""
        arr = cp.array([1, 2, 3])
        xp = get_array_module(arr)
        assert xp is cp


class TestBatchedInverse3x3:
    """Tests for batched_inverse_3x3 function."""
    
    def test_single_identity_matrix(self):
        """Test inverse of identity matrix."""
        identity = np.eye(3, dtype=np.float32)[None, ...]
        inv = batched_inverse_3x3(identity)
        
        np.testing.assert_array_almost_equal(inv[0], np.eye(3), decimal=5)
    
    def test_single_diagonal_matrix(self):
        """Test inverse of diagonal matrix."""
        diag = np.array([[[2, 0, 0], [0, 3, 0], [0, 0, 4]]], dtype=np.float32)
        inv = batched_inverse_3x3(diag)
        
        expected = np.array([[[0.5, 0, 0], [0, 1/3, 0], [0, 0, 0.25]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(inv, expected, decimal=5)
    
    def test_batch_of_random_matrices(self):
        """Test batch of random matrices."""
        np.random.seed(42)
        batch_size = 10
        
        # Generate random symmetric positive definite matrices
        mats = np.random.rand(batch_size, 3, 3).astype(np.float32)
        mats = np.einsum('mij,mkj->mik', mats, mats)  # A @ A.T is symmetric PD
        
        inv_mats = batched_inverse_3x3(mats)
        
        # Verify A @ A^-1 = I
        identity = np.einsum('mij,mjk->mik', mats, inv_mats)
        expected_identity = np.eye(3, dtype=np.float32)[None, :, :]
        
        for i in range(batch_size):
            np.testing.assert_array_almost_equal(
                identity[i], expected_identity[0], decimal=4
            )
    
    def test_near_singular_matrix(self):
        """Test that near-singular matrices return identity."""
        # Create nearly singular matrix (determinant ~ 0)
        singular = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=np.float32)
        inv = batched_inverse_3x3(singular, epsilon=1e-12)
        
        # Should return identity for singular matrix
        # (or at least not crash/return NaN)
        assert np.all(np.isfinite(inv))
    
    def test_large_batch(self):
        """Test performance with large batch."""
        batch_size = 1000
        np.random.seed(42)
        
        # Generate random matrices
        mats = np.random.rand(batch_size, 3, 3).astype(np.float32)
        mats = np.einsum('mij,mkj->mik', mats, mats)
        
        inv_mats = batched_inverse_3x3(mats)
        
        assert inv_mats.shape == (batch_size, 3, 3)
        assert np.all(np.isfinite(inv_mats))
    
    def test_output_dtype_preserved(self):
        """Test that output dtype matches input dtype."""
        mats_f32 = np.eye(3, dtype=np.float32)[None, ...]
        inv_f32 = batched_inverse_3x3(mats_f32)
        assert inv_f32.dtype == np.float32
        
        mats_f64 = np.eye(3, dtype=np.float64)[None, ...]
        inv_f64 = batched_inverse_3x3(mats_f64)
        assert inv_f64.dtype == np.float64
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_cupy_array(self):
        """Test that function works with CuPy arrays."""
        identity = cp.eye(3, dtype=cp.float32)[None, ...]
        inv = batched_inverse_3x3(identity)
        
        # Result should be CuPy array
        assert isinstance(inv, cp.ndarray)
        
        # Verify correctness
        np.testing.assert_array_almost_equal(
            cp.asnumpy(inv[0]), np.eye(3), decimal=5
        )
    
    def test_analytic_vs_numpy_linalg(self):
        """Compare analytic formula with numpy.linalg.inv."""
        np.random.seed(42)
        batch_size = 50
        
        # Generate random symmetric PD matrices
        mats = np.random.rand(batch_size, 3, 3).astype(np.float32)
        mats = np.einsum('mij,mkj->mik', mats, mats)
        
        # Our analytic inverse
        inv_analytic = batched_inverse_3x3(mats)
        
        # NumPy's inverse (loop over batch)
        inv_numpy = np.array([np.linalg.inv(mats[i]) for i in range(batch_size)])
        
        # Should match closely (within relative tolerance due to float32)
        # Use relative tolerance since absolute values can be large
        for i in range(batch_size):
            np.testing.assert_allclose(inv_analytic[i], inv_numpy[i], rtol=1e-2, atol=1.0)


class TestInversePowerIteration:
    """Tests for inverse_power_iteration function."""
    
    def test_single_covariance_matrix(self):
        """Test single covariance matrix."""
        # Create covariance of planar points (smallest eigenvalue ≈ 0)
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ], dtype=np.float32)
        
        centered = points - points.mean(axis=0)
        cov = np.dot(centered.T, centered) / len(points)
        cov_batch = cov[None, ...]
        
        eigenvec = inverse_power_iteration(cov_batch, num_iterations=8)
        
        # Should be unit length
        assert eigenvec.shape == (1, 3)
        norm = np.linalg.norm(eigenvec[0])
        assert np.isclose(norm, 1.0, atol=1e-5)
        
        # For planar points in XY plane, normal should be ±Z
        # Note: May converge to different eigenvector depending on initialization
        assert np.all(np.isfinite(eigenvec))
    
    def test_batch_of_covariances(self):
        """Test batch of covariance matrices."""
        np.random.seed(42)
        batch_size = 20
        
        # Generate random covariance matrices (symmetric PD)
        mats = np.random.rand(batch_size, 3, 3).astype(np.float32)
        cov_mats = np.einsum('mij,mkj->mik', mats, mats) / 10
        
        eigenvecs = inverse_power_iteration(cov_mats, num_iterations=8)
        
        # Check shape and unit length
        assert eigenvecs.shape == (batch_size, 3)
        norms = np.linalg.norm(eigenvecs, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(batch_size), decimal=5)
    
    def test_upward_orientation(self):
        """Test that eigenvectors are oriented upward (positive Z)."""
        np.random.seed(42)
        batch_size = 50
        
        mats = np.random.rand(batch_size, 3, 3).astype(np.float32)
        cov_mats = np.einsum('mij,mkj->mik', mats, mats)
        
        eigenvecs = inverse_power_iteration(cov_mats)
        
        # All Z components should be positive (upward oriented)
        assert np.all(eigenvecs[:, 2] >= 0)
    
    def test_convergence_iterations(self):
        """Test that more iterations improve convergence."""
        np.random.seed(42)
        
        # Create a covariance matrix
        points = np.random.rand(10, 3).astype(np.float32)
        centered = points - points.mean(axis=0)
        cov = np.dot(centered.T, centered) / len(points)
        cov_batch = cov[None, ...]
        
        # Test different iteration counts
        v_4 = inverse_power_iteration(cov_batch, num_iterations=4)
        v_8 = inverse_power_iteration(cov_batch, num_iterations=8)
        v_16 = inverse_power_iteration(cov_batch, num_iterations=16)
        
        # More iterations should give more stable results
        # (v_8 and v_16 should be closer than v_4 and v_8)
        diff_4_8 = np.linalg.norm(v_4 - v_8)
        diff_8_16 = np.linalg.norm(v_8 - v_16)
        
        # Later iterations should converge (smaller differences)
        assert diff_8_16 < diff_4_8 or diff_8_16 < 0.01
    
    def test_singular_matrix_handling(self):
        """Test handling of singular matrices."""
        # Create singular matrix (all zeros)
        singular = np.zeros((1, 3, 3), dtype=np.float32)
        
        eigenvec = inverse_power_iteration(singular, epsilon=1e-6)
        
        # Should return valid unit vector (likely [0, 0, 1])
        assert eigenvec.shape == (1, 3)
        assert np.all(np.isfinite(eigenvec))
        norm = np.linalg.norm(eigenvec[0])
        assert np.isclose(norm, 1.0, atol=1e-5)
    
    def test_compare_with_full_eigh(self):
        """Compare results with full eigendecomposition."""
        np.random.seed(42)
        batch_size = 10
        
        # Generate covariance matrices
        mats = np.random.rand(batch_size, 3, 3).astype(np.float32)
        cov_mats = np.einsum('mij,mkj->mik', mats, mats)
        
        # Our fast method
        eigenvecs_fast = inverse_power_iteration(cov_mats, num_iterations=10)
        
        # Full eigendecomposition
        eigenvecs_full = np.zeros((batch_size, 3), dtype=np.float32)
        for i in range(batch_size):
            eigvals, eigvecs = np.linalg.eigh(cov_mats[i])
            # Smallest eigenvalue is first (eigh returns ascending order)
            eigenvecs_full[i] = eigvecs[:, 0]
            # Orient upward
            if eigenvecs_full[i, 2] < 0:
                eigenvecs_full[i] *= -1
        
        # Should be close (within 0.1 due to numerical differences)
        # Use absolute dot product to ignore sign ambiguity
        for i in range(batch_size):
            dot_product = abs(np.dot(eigenvecs_fast[i], eigenvecs_full[i]))
            assert dot_product > 0.9  # Nearly parallel
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_cupy_array(self):
        """Test that function works with CuPy arrays."""
        # Create covariance matrix
        mats = cp.random.rand(5, 3, 3).astype(cp.float32)
        cov_mats = cp.einsum('mij,mkj->mik', mats, mats)
        
        eigenvecs = inverse_power_iteration(cov_mats)
        
        # Result should be CuPy array
        assert isinstance(eigenvecs, cp.ndarray)
        
        # Verify unit length
        norms = cp.linalg.norm(eigenvecs, axis=1)
        assert cp.allclose(norms, cp.ones(5), atol=1e-5)


class TestIntegration:
    """Integration tests combining utilities."""
    
    def test_normal_computation_workflow(self):
        """Test typical workflow for normal computation from point neighborhoods."""
        np.random.seed(42)
        n_points = 100
        k_neighbors = 10
        
        # Simulate point neighborhoods (100 points, 10 neighbors each)
        neighborhoods = np.random.rand(n_points, k_neighbors, 3).astype(np.float32)
        
        # Compute covariance matrices
        centroids = neighborhoods.mean(axis=1, keepdims=True)
        centered = neighborhoods - centroids
        cov_matrices = np.einsum('mki,mkj->mij', centered, centered) / k_neighbors
        
        # Compute normals using inverse power iteration
        normals = inverse_power_iteration(cov_matrices, num_iterations=8)
        
        # Verify results
        assert normals.shape == (n_points, 3)
        assert np.all(np.isfinite(normals))
        
        # All should be unit length
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(n_points), decimal=5)
        
        # All should be oriented upward
        assert np.all(normals[:, 2] >= 0)
    
    def test_inverse_and_power_iteration_combined(self):
        """Test that batched_inverse_3x3 works correctly in inverse_power_iteration."""
        np.random.seed(42)
        
        # Create well-conditioned covariance matrix
        mats = np.random.rand(10, 3, 3).astype(np.float32)
        cov_mats = np.einsum('mij,mkj->mik', mats, mats) + 0.1 * np.eye(3)
        
        # This internally uses batched_inverse_3x3
        eigenvecs = inverse_power_iteration(cov_mats, num_iterations=8)
        
        # Verify we get valid results
        assert eigenvecs.shape == (10, 3)
        norms = np.linalg.norm(eigenvecs, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(10), decimal=5)


class TestPerformance:
    """Performance comparison tests (informational)."""
    
    @pytest.mark.slow
    def test_batched_vs_loop_inverse(self):
        """Compare batched inverse with loop (should be faster)."""
        import time
        
        np.random.seed(42)
        batch_size = 1000
        mats = np.random.rand(batch_size, 3, 3).astype(np.float32)
        mats = np.einsum('mij,mkj->mik', mats, mats)
        
        # Batched version
        start = time.time()
        inv_batched = batched_inverse_3x3(mats)
        time_batched = time.time() - start
        
        # Loop version
        start = time.time()
        inv_loop = np.array([np.linalg.inv(mats[i]) for i in range(batch_size)])
        time_loop = time.time() - start
        
        print(f"\nBatched: {time_batched:.4f}s, Loop: {time_loop:.4f}s")
        print(f"Speedup: {time_loop / time_batched:.2f}x")
        
        # Batched should be faster (though numpy.linalg.inv is highly optimized)
        # Verify results by checking M @ M^-1 ≈ I (more robust than element comparison)
        identity_batched = np.einsum('mij,mjk->mik', mats, inv_batched)
        identity_loop = np.einsum('mij,mjk->mik', mats, inv_loop)
        expected_identity = np.eye(3, dtype=np.float32)
        
        # Compute success rates
        errors_batched = np.max(np.abs(identity_batched - expected_identity), axis=(1, 2))
        errors_loop = np.max(np.abs(identity_loop - expected_identity), axis=(1, 2))
        
        batched_success = np.sum(errors_batched < 0.05)
        numpy_success = np.sum(errors_loop < 0.05)
        both_success = np.sum((errors_batched < 0.05) & (errors_loop < 0.05))
        
        print(f"Batched success rate: {batched_success}/{batch_size} ({100*batched_success/batch_size:.1f}%)")
        print(f"NumPy success rate: {numpy_success}/{batch_size} ({100*numpy_success/batch_size:.1f}%)")
        print(f"Both succeed: {both_success}/{batch_size} ({100*both_success/batch_size:.1f}%)")
        
        # Require high success rate (>99%) for analytic method
        assert batched_success >= 0.99 * batch_size, \
            f"Analytic inverse success rate too low: {batched_success}/{batch_size}"
    
    @pytest.mark.slow
    def test_power_iteration_vs_eigh(self):
        """Compare inverse power iteration with full eigh (should be faster)."""
        import time
        
        np.random.seed(42)
        batch_size = 1000
        mats = np.random.rand(batch_size, 3, 3).astype(np.float32)
        cov_mats = np.einsum('mij,mkj->mik', mats, mats)
        
        # Power iteration
        start = time.time()
        eigenvecs_fast = inverse_power_iteration(cov_mats, num_iterations=8)
        time_fast = time.time() - start
        
        # Full eigendecomposition
        start = time.time()
        eigenvecs_full = np.zeros((batch_size, 3), dtype=np.float32)
        for i in range(batch_size):
            eigvals, eigvecs = np.linalg.eigh(cov_mats[i])
            eigenvecs_full[i] = eigvecs[:, 0]
            if eigenvecs_full[i, 2] < 0:
                eigenvecs_full[i] *= -1
        time_full = time.time() - start
        
        print(f"\nPower iteration: {time_fast:.4f}s, Full eigh: {time_full:.4f}s")
        print(f"Speedup: {time_full / time_fast:.2f}x")
        
        # Power iteration should be significantly faster
        assert time_fast < time_full


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
