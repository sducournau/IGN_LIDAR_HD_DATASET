"""
Test suite for Numba-accelerated computational kernels.

This module tests the Numba JIT-compiled functions for feature computation,
ensuring correctness, performance, and graceful fallback to NumPy.

Author: IGN LiDAR HD Team
Date: 2025-11-21
"""

import pytest
import numpy as np
import warnings
from typing import Tuple

# Import functions to test
from ign_lidar.features.numba_accelerated import (
    is_numba_available,
    compute_covariance_matrices,
    compute_covariance_matrices_numpy,
    compute_covariance_matrices_numba,
    compute_normals_from_eigenvectors,
    compute_normals_from_eigenvectors_numpy,
    compute_normals_from_eigenvectors_numba,
    compute_local_point_density,
    compute_local_point_density_numpy,
    compute_local_point_density_numba,
    get_numba_info,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_points_small():
    """Small point cloud for basic testing (100 points)."""
    np.random.seed(42)
    return np.random.rand(100, 3).astype(np.float32)


@pytest.fixture
def sample_points_medium():
    """Medium point cloud for performance testing (10K points)."""
    np.random.seed(42)
    return np.random.rand(10_000, 3).astype(np.float32)


@pytest.fixture
def sample_points_large():
    """Large point cloud for scalability testing (100K points)."""
    np.random.seed(42)
    return np.random.rand(100_000, 3).astype(np.float32)


@pytest.fixture
def sample_knn_indices_small():
    """KNN indices for small point cloud (k=30)."""
    np.random.seed(42)
    return np.random.randint(0, 100, (100, 30), dtype=np.int32)


@pytest.fixture
def sample_knn_indices_medium():
    """KNN indices for medium point cloud (k=30)."""
    np.random.seed(42)
    return np.random.randint(0, 10_000, (10_000, 30), dtype=np.int32)


@pytest.fixture
def sample_eigenvectors():
    """Sample eigenvector matrices for normal extraction."""
    np.random.seed(42)
    eigenvectors = np.random.rand(100, 3, 3).astype(np.float32)
    # Ensure they're orthonormal (approximate)
    for i in range(100):
        q, r = np.linalg.qr(eigenvectors[i])
        eigenvectors[i] = q
    return eigenvectors


# ============================================================================
# Test Numba Availability
# ============================================================================

class TestNumbaAvailability:
    """Test Numba availability detection."""
    
    def test_is_numba_available(self):
        """Test Numba availability check."""
        available = is_numba_available()
        assert isinstance(available, bool)
    
    def test_get_numba_info(self):
        """Test Numba information retrieval."""
        info = get_numba_info()
        
        assert isinstance(info, dict)
        assert 'available' in info
        assert 'version' in info
        assert isinstance(info['available'], bool)
        
        if info['available']:
            assert info['version'] is not None
            print(f"\nNumba version: {info['version']}")
            if info['threading_layer']:
                print(f"Threading layer: {info['threading_layer']}")
            if info['num_threads']:
                print(f"Num threads: {info['num_threads']}")


# ============================================================================
# Test Covariance Matrix Computation
# ============================================================================

class TestCovarianceMatrices:
    """Test covariance matrix computation with Numba and NumPy."""
    
    def test_numpy_implementation_basic(self, sample_points_small, sample_knn_indices_small):
        """Test NumPy covariance computation basic functionality."""
        k = 30
        cov = compute_covariance_matrices_numpy(
            sample_points_small, sample_knn_indices_small, k
        )
        
        assert cov.shape == (100, 3, 3)
        assert cov.dtype == np.float32
        
        # Check symmetry
        for i in range(min(10, len(cov))):  # Check first 10
            assert np.allclose(cov[i], cov[i].T, rtol=1e-5)
    
    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    def test_numba_implementation_basic(self, sample_points_small, sample_knn_indices_small):
        """Test Numba covariance computation basic functionality."""
        k = 30
        cov = compute_covariance_matrices_numba(
            sample_points_small, sample_knn_indices_small, k
        )
        
        assert cov.shape == (100, 3, 3)
        assert cov.dtype == np.float32
        
        # Check symmetry
        for i in range(min(10, len(cov))):
            assert np.allclose(cov[i], cov[i].T, rtol=1e-5)
    
    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    def test_numba_numpy_consistency(self, sample_points_small, sample_knn_indices_small):
        """Test that Numba and NumPy produce consistent results."""
        k = 30
        
        cov_numpy = compute_covariance_matrices_numpy(
            sample_points_small, sample_knn_indices_small, k
        )
        cov_numba = compute_covariance_matrices_numba(
            sample_points_small, sample_knn_indices_small, k
        )
        
        # Should be very close (within floating point precision)
        assert np.allclose(cov_numpy, cov_numba, rtol=1e-4, atol=1e-6)
        
        # Calculate max difference for reporting
        max_diff = np.max(np.abs(cov_numpy - cov_numba))
        print(f"\nMax difference between NumPy and Numba: {max_diff:.2e}")
        assert max_diff < 1e-4
    
    def test_automatic_selection_numpy_forced(self, sample_points_small, sample_knn_indices_small):
        """Test automatic selection with NumPy forced."""
        k = 30
        cov = compute_covariance_matrices(
            sample_points_small, sample_knn_indices_small, k, use_numba=False
        )
        
        assert cov.shape == (100, 3, 3)
        assert cov.dtype == np.float32
    
    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    def test_automatic_selection_numba_forced(self, sample_points_small, sample_knn_indices_small):
        """Test automatic selection with Numba forced."""
        k = 30
        cov = compute_covariance_matrices(
            sample_points_small, sample_knn_indices_small, k, use_numba=True
        )
        
        assert cov.shape == (100, 3, 3)
        assert cov.dtype == np.float32
    
    def test_automatic_selection_auto(self, sample_points_small, sample_knn_indices_small):
        """Test automatic selection with auto mode."""
        k = 30
        cov = compute_covariance_matrices(
            sample_points_small, sample_knn_indices_small, k, use_numba=None
        )
        
        assert cov.shape == (100, 3, 3)
        assert cov.dtype == np.float32


# ============================================================================
# Test Normal Extraction
# ============================================================================

class TestNormalExtraction:
    """Test normal extraction from eigenvectors."""
    
    def test_numpy_normal_extraction(self, sample_eigenvectors):
        """Test NumPy normal extraction."""
        normals = compute_normals_from_eigenvectors_numpy(sample_eigenvectors)
        
        assert normals.shape == (100, 3)
        assert normals.dtype == np.float32
        
        # Check that all normals are approximately unit vectors
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, rtol=0.1)
        
        # Check upward orientation (positive Z)
        assert np.all(normals[:, 2] >= 0)
    
    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    def test_numba_normal_extraction(self, sample_eigenvectors):
        """Test Numba normal extraction."""
        normals = compute_normals_from_eigenvectors_numba(sample_eigenvectors)
        
        assert normals.shape == (100, 3)
        assert normals.dtype == np.float32
        
        # Check upward orientation
        assert np.all(normals[:, 2] >= 0)
    
    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    def test_numba_numpy_normal_consistency(self, sample_eigenvectors):
        """Test that Numba and NumPy produce consistent normals."""
        normals_numpy = compute_normals_from_eigenvectors_numpy(sample_eigenvectors)
        normals_numba = compute_normals_from_eigenvectors_numba(sample_eigenvectors)
        
        # Should be identical (no approximations involved)
        assert np.allclose(normals_numpy, normals_numba, rtol=1e-6, atol=1e-8)
        
        max_diff = np.max(np.abs(normals_numpy - normals_numba))
        print(f"\nMax normal difference: {max_diff:.2e}")
        assert max_diff < 1e-6
    
    def test_automatic_normal_selection(self, sample_eigenvectors):
        """Test automatic normal extraction selection."""
        normals = compute_normals_from_eigenvectors(
            sample_eigenvectors, use_numba=None
        )
        
        assert normals.shape == (100, 3)
        assert normals.dtype == np.float32
        assert np.all(normals[:, 2] >= 0)


# ============================================================================
# Test Local Point Density
# ============================================================================

class TestLocalPointDensity:
    """Test local point density computation."""
    
    def test_numpy_density_basic(self, sample_points_small, sample_knn_indices_small):
        """Test NumPy density computation."""
        k = 30
        density = compute_local_point_density_numpy(
            sample_points_small, sample_knn_indices_small, k
        )
        
        assert density.shape == (100,)
        assert density.dtype == np.float32
        assert np.all(density >= 0)
        assert np.all(np.isfinite(density))
    
    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    def test_numba_density_basic(self, sample_points_small, sample_knn_indices_small):
        """Test Numba density computation."""
        k = 30
        density = compute_local_point_density_numba(
            sample_points_small, sample_knn_indices_small, k
        )
        
        assert density.shape == (100,)
        assert density.dtype == np.float32
        assert np.all(density >= 0)
        assert np.all(np.isfinite(density))
    
    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    def test_numba_numpy_density_consistency(self, sample_points_small, sample_knn_indices_small):
        """Test that Numba and NumPy produce consistent density values."""
        k = 30
        
        density_numpy = compute_local_point_density_numpy(
            sample_points_small, sample_knn_indices_small, k
        )
        density_numba = compute_local_point_density_numba(
            sample_points_small, sample_knn_indices_small, k
        )
        
        # Allow small differences due to floating point
        assert np.allclose(density_numpy, density_numba, rtol=1e-4, atol=1e-6)
        
        max_diff = np.max(np.abs(density_numpy - density_numba))
        print(f"\nMax density difference: {max_diff:.2e}")
        assert max_diff < 1e-3
    
    def test_automatic_density_selection(self, sample_points_small, sample_knn_indices_small):
        """Test automatic density computation selection."""
        k = 30
        density = compute_local_point_density(
            sample_points_small, sample_knn_indices_small, k, use_numba=None
        )
        
        assert density.shape == (100,)
        assert density.dtype == np.float32
        assert np.all(density >= 0)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_points(self):
        """Test with empty point array."""
        points = np.zeros((0, 3), dtype=np.float32)
        indices = np.zeros((0, 30), dtype=np.int32)
        
        cov = compute_covariance_matrices(points, indices, k=30)
        assert cov.shape == (0, 3, 3)
    
    def test_single_point(self):
        """Test with single point."""
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        indices = np.array([[0] * 30], dtype=np.int32)
        
        cov = compute_covariance_matrices(points, indices, k=30)
        assert cov.shape == (1, 3, 3)
    
    def test_degenerate_covariance(self):
        """Test with colinear points (degenerate covariance)."""
        # Points on a line
        points = np.array([
            [i, 0.0, 0.0] for i in range(100)
        ], dtype=np.float32)
        indices = np.tile(np.arange(30), (100, 1)).astype(np.int32)
        
        cov = compute_covariance_matrices(points, indices, k=30)
        
        # Covariance should be rank-deficient
        assert cov.shape == (100, 3, 3)
        # Check that at least one eigenvalue is near zero
        eigvals = np.linalg.eigvalsh(cov[0])
        assert np.min(eigvals) < 0.1
    
    def test_numba_unavailable_warning(self, sample_points_small, sample_knn_indices_small):
        """Test warning when Numba requested but unavailable."""
        if is_numba_available():
            pytest.skip("Numba is available, cannot test warning")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cov = compute_covariance_matrices(
                sample_points_small, sample_knn_indices_small, 
                k=30, use_numba=True
            )
            
            # Should have warned
            assert len(w) >= 1
            assert "Numba" in str(w[0].message)
            
            # Should still work (fallback to NumPy)
            assert cov.shape == (100, 3, 3)


# ============================================================================
# Performance Benchmarks (Optional - run with pytest -v -s)
# ============================================================================

class TestPerformance:
    """Performance benchmarks for Numba acceleration."""
    
    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    @pytest.mark.slow
    def test_covariance_performance_medium(self, sample_points_medium, sample_knn_indices_medium):
        """Benchmark covariance computation on medium dataset."""
        import time
        
        k = 30
        
        # NumPy timing
        start = time.time()
        cov_numpy = compute_covariance_matrices_numpy(
            sample_points_medium, sample_knn_indices_medium, k
        )
        numpy_time = time.time() - start
        
        # Numba timing (includes JIT compilation on first run)
        start = time.time()
        cov_numba = compute_covariance_matrices_numba(
            sample_points_medium, sample_knn_indices_medium, k
        )
        numba_time_first = time.time() - start
        
        # Numba timing (second run, JIT compiled)
        start = time.time()
        cov_numba = compute_covariance_matrices_numba(
            sample_points_medium, sample_knn_indices_medium, k
        )
        numba_time_cached = time.time() - start
        
        print(f"\n{'='*60}")
        print(f"Covariance Matrix Performance (10K points, k={k}):")
        print(f"{'='*60}")
        print(f"NumPy time:              {numpy_time:.3f}s")
        print(f"Numba time (first run):  {numba_time_first:.3f}s")
        print(f"Numba time (cached):     {numba_time_cached:.3f}s")
        print(f"Speedup (cached):        {numpy_time/numba_time_cached:.2f}x")
        print(f"{'='*60}")
        
        # Numba should be faster (after JIT compilation)
        # Note: Speedup may vary based on hardware and NumPy backend
        assert numba_time_cached < numpy_time or numpy_time < 0.1  # Very fast anyway
    
    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    @pytest.mark.slow
    def test_normals_performance_medium(self, sample_eigenvectors):
        """Benchmark normal extraction performance."""
        import time
        
        # Create larger eigenvector array
        np.random.seed(42)
        large_eigenvectors = np.random.rand(10_000, 3, 3).astype(np.float32)
        for i in range(10_000):
            q, r = np.linalg.qr(large_eigenvectors[i])
            large_eigenvectors[i] = q
        
        # NumPy timing
        start = time.time()
        normals_numpy = compute_normals_from_eigenvectors_numpy(large_eigenvectors)
        numpy_time = time.time() - start
        
        # Numba timing (cached)
        start = time.time()
        normals_numba = compute_normals_from_eigenvectors_numba(large_eigenvectors)
        numba_time = time.time() - start
        
        print(f"\n{'='*60}")
        print(f"Normal Extraction Performance (10K points):")
        print(f"{'='*60}")
        print(f"NumPy time:   {numpy_time:.3f}s")
        print(f"Numba time:   {numba_time:.3f}s")
        print(f"Speedup:      {numpy_time/numba_time:.2f}x")
        print(f"{'='*60}")


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with realistic workflows."""
    
    @pytest.mark.integration
    def test_full_normal_computation_workflow(self, sample_points_medium):
        """Test full workflow from points to normals."""
        from sklearn.neighbors import NearestNeighbors
        
        k = 30
        
        # Step 1: Find neighbors
        knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        knn.fit(sample_points_medium)
        _, indices = knn.kneighbors(sample_points_medium)
        indices = indices.astype(np.int32)
        
        # Step 2: Compute covariance matrices
        cov = compute_covariance_matrices(sample_points_medium, indices, k)
        
        # Step 3: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Step 4: Extract normals
        normals = compute_normals_from_eigenvectors(eigenvectors)
        
        # Validation
        assert normals.shape == (10_000, 3)
        assert np.all(normals[:, 2] >= 0)  # Upward oriented
        
        # Normals should be approximately unit vectors
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, rtol=0.1)
    
    @pytest.mark.integration
    def test_density_computation_workflow(self, sample_points_medium):
        """Test density computation workflow."""
        from sklearn.neighbors import NearestNeighbors
        
        k = 30
        
        # Find neighbors
        knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        knn.fit(sample_points_medium)
        _, indices = knn.kneighbors(sample_points_medium)
        indices = indices.astype(np.int32)
        
        # Compute density
        density = compute_local_point_density(sample_points_medium, indices, k)
        
        # Validation
        assert density.shape == (10_000,)
        assert np.all(density >= 0)
        assert np.all(np.isfinite(density))
        
        # Density should have reasonable range
        assert np.mean(density) > 0
        assert np.std(density) > 0
