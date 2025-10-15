"""
Tests for features/core/curvature.py module.
"""

import numpy as np
import pytest

from ign_lidar.features.core.curvature import (
    compute_curvature,
    compute_mean_curvature,
    compute_shape_index,
    compute_curvedness,
    compute_all_curvature_features,
)


class TestCurvatureFeatures:
    """Test suite for curvature feature computation."""
    
    def test_compute_curvature_standard(self):
        """Test standard curvature computation."""
        # Create test eigenvalues (λ1 >= λ2 >= λ3)
        eigenvalues = np.array([
            [1.0, 0.5, 0.1],
            [1.0, 0.8, 0.01],
            [1.0, 0.1, 0.05]
        ])
        
        curvature = compute_curvature(eigenvalues, method='standard')
        
        # Check shape
        assert curvature.shape == (3,)
        
        # Check range [0, 1]
        assert np.all(curvature >= 0)
        assert np.all(curvature <= 1)
        
        # Verify calculation: λ3 / (λ1 + λ2 + λ3)
        expected_0 = 0.1 / (1.0 + 0.5 + 0.1)
        np.testing.assert_allclose(curvature[0], expected_0, rtol=1e-5)
    
    def test_compute_curvature_normalized(self):
        """Test normalized curvature computation."""
        eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
        
        curvature = compute_curvature(eigenvalues, method='normalized')
        
        # Check shape
        assert curvature.shape == (2,)
        
        # Verify: λ3 / λ1
        expected = np.array([0.1 / 1.0, 0.01 / 1.0])
        np.testing.assert_allclose(curvature, expected, rtol=1e-5)
    
    def test_compute_curvature_gaussian(self):
        """Test Gaussian curvature computation."""
        eigenvalues = np.array([[1.0, 0.5, 0.1]])
        
        curvature = compute_curvature(eigenvalues, method='gaussian')
        
        # Verify: (λ2 * λ3) / (λ1^2)
        expected = (0.5 * 0.1) / (1.0 * 1.0)
        np.testing.assert_allclose(curvature[0], expected, rtol=1e-5)
    
    def test_curvature_input_validation(self):
        """Test input validation."""
        # Invalid input type
        with pytest.raises(ValueError, match="eigenvalues must be a numpy array"):
            compute_curvature([[1, 2, 3]])
        
        # Invalid shape
        with pytest.raises(ValueError, match="eigenvalues must have shape"):
            compute_curvature(np.random.rand(10, 2))
        
        # Unknown method
        eigenvalues = np.random.rand(10, 3)
        with pytest.raises(ValueError, match="Unknown curvature method"):
            compute_curvature(eigenvalues, method='unknown')
    
    def test_compute_mean_curvature(self):
        """Test mean curvature computation."""
        eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
        
        mean_curv = compute_mean_curvature(eigenvalues)
        
        # Check shape
        assert mean_curv.shape == (2,)
        
        # Verify: (λ2 + λ3) / λ1
        expected = np.array([(0.5 + 0.1) / 1.0, (0.8 + 0.01) / 1.0])
        np.testing.assert_allclose(mean_curv, expected, rtol=1e-5)
    
    def test_compute_shape_index(self):
        """Test shape index computation."""
        eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
        
        shape_idx = compute_shape_index(eigenvalues)
        
        # Check shape
        assert shape_idx.shape == (2,)
        
        # Check range [-1, 1]
        assert np.all(shape_idx >= -1)
        assert np.all(shape_idx <= 1)
    
    def test_compute_curvedness(self):
        """Test curvedness computation."""
        eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
        
        curvedness = compute_curvedness(eigenvalues)
        
        # Check shape
        assert curvedness.shape == (2,)
        
        # Curvedness should be positive
        assert np.all(curvedness >= 0)
    
    def test_compute_all_curvature_features(self):
        """Test computation of all curvature features."""
        eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.8, 0.01]])
        
        features = compute_all_curvature_features(eigenvalues)
        
        # Check that all expected features are present
        assert isinstance(features, dict)
        assert 'curvature' in features
        assert 'mean_curvature' in features
        assert 'shape_index' in features
        assert 'curvedness' in features
        
        # Check shapes
        for key, value in features.items():
            assert value.shape == (2,)
    
    def test_epsilon_handling(self):
        """Test that epsilon prevents division by zero."""
        # Create eigenvalues with very small values
        eigenvalues = np.array([[1e-15, 1e-16, 1e-17]])
        
        # Should not raise error or produce inf/nan
        curvature = compute_curvature(eigenvalues)
        assert np.all(np.isfinite(curvature))
        
        mean_curv = compute_mean_curvature(eigenvalues)
        assert np.all(np.isfinite(mean_curv))
    
    def test_flat_surface(self):
        """Test curvature for flat surface (high λ2, low λ3)."""
        # Flat surface: λ1 ≈ λ2 >> λ3
        eigenvalues = np.array([[1.0, 0.95, 0.01]])
        
        curvature = compute_curvature(eigenvalues)
        
        # Should have low curvature (flat)
        assert curvature[0] < 0.1
    
    def test_sharp_edge(self):
        """Test curvature for sharp edge (low λ2, low λ3)."""
        # Sharp edge: λ1 >> λ2 ≈ λ3
        eigenvalues = np.array([[1.0, 0.1, 0.05]])
        
        curvature = compute_curvature(eigenvalues)
        
        # Should have moderate curvature
        assert 0.01 < curvature[0] < 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
