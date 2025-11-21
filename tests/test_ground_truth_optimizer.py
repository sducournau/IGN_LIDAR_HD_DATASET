"""
Unit tests for unified GroundTruthOptimizer (Week 2 consolidation).

Tests the auto-selection logic and basic functionality.
"""

import pytest
import numpy as np

from ign_lidar.optimization import GroundTruthOptimizer

# Check for GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Pytest marker for GPU tests
requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU (CuPy) not available")


class TestGroundTruthOptimizer:
    """Test unified ground truth optimizer."""
    
    def test_initialization(self):
        """Test optimizer can be initialized."""
        optimizer = GroundTruthOptimizer(verbose=False)
        assert optimizer is not None
        assert hasattr(optimizer, 'select_method')
        assert hasattr(optimizer, 'label_points')
    
    def test_auto_selection_small_dataset(self):
        """Test auto-selection for small dataset (100K-1M points)."""
        optimizer = GroundTruthOptimizer(verbose=False)
        method = optimizer.select_method(n_points=500_000, n_polygons=100)
        
        if GPU_AVAILABLE:
            # With GPU: should use 'gpu' for medium datasets (100K-1M)
            assert method == 'gpu'
        else:
            # Without GPU: should use STRtree
            assert method in ['strtree', 'vectorized']
    
    def test_auto_selection_medium_dataset(self):
        """Test auto-selection for medium dataset (1-10M points)."""
        optimizer = GroundTruthOptimizer(verbose=False)
        method = optimizer.select_method(n_points=5_000_000, n_polygons=100)
        
        if GPU_AVAILABLE:
            # With GPU: should select 'gpu_chunked' for datasets >1M
            assert method == 'gpu_chunked'
        else:
            # Without GPU: should select STRtree
            assert method == 'strtree'
    
    def test_auto_selection_large_dataset(self):
        """Test auto-selection for large dataset (>10M points)."""
        optimizer = GroundTruthOptimizer(verbose=False)
        method = optimizer.select_method(n_points=15_000_000, n_polygons=100)
        
        if GPU_AVAILABLE:
            # With GPU: should select 'gpu_chunked' for large datasets
            assert method == 'gpu_chunked'
        else:
            # Without GPU: should select STRtree
            assert method == 'strtree'
    
    def test_force_method(self):
        """Test forcing a specific method."""
        optimizer = GroundTruthOptimizer(force_method='strtree', verbose=False)
        method = optimizer.select_method(n_points=15_000_000, n_polygons=100)
        
        # Should respect forced method
        assert method == 'strtree'
    
    @requires_gpu
    def test_force_gpu_method(self):
        """Test forcing GPU method."""
        optimizer = GroundTruthOptimizer(force_method='gpu', verbose=False)
        method = optimizer.select_method(n_points=500_000, n_polygons=100)
        
        # Should respect forced method
        assert method == 'gpu'
    
    def test_gpu_detection(self):
        """Test GPU availability detection."""
        optimizer = GroundTruthOptimizer(verbose=False)
        
        # Check GPU detection
        gpu_available = optimizer._check_gpu()
        assert isinstance(gpu_available, bool)
        
        if GPU_AVAILABLE:
            assert gpu_available is True
        else:
            assert gpu_available is False
    
    def test_hardware_detection_caching(self):
        """Test that hardware detection is cached."""
        # Create first optimizer
        optimizer1 = GroundTruthOptimizer(verbose=False)
        
        # Hardware detection should be cached in class variable
        assert GroundTruthOptimizer._gpu_available is not None
        
        # Create second optimizer - should use cached value
        optimizer2 = GroundTruthOptimizer(verbose=False)
        assert optimizer2._gpu_available == optimizer1._gpu_available


class TestMethodSelection:
    """Test method selection logic."""
    
    def test_selection_with_no_gpu(self):
        """Test selection when GPU is not available."""
        # Force no GPU
        original_value = GroundTruthOptimizer._gpu_available
        GroundTruthOptimizer._gpu_available = False
        
        try:
            optimizer = GroundTruthOptimizer(verbose=False)
            
            # Small dataset
            method = optimizer.select_method(n_points=500_000, n_polygons=100)
            assert method == 'strtree'
            
            # Large dataset
            method = optimizer.select_method(n_points=15_000_000, n_polygons=100)
            assert method == 'strtree'
        finally:
            # Restore original value
            GroundTruthOptimizer._gpu_available = original_value
    
    @requires_gpu
    def test_selection_with_gpu(self):
        """Test selection when GPU is available."""
        # Force GPU available
        original_value = GroundTruthOptimizer._gpu_available
        GroundTruthOptimizer._gpu_available = True
        
        try:
            optimizer = GroundTruthOptimizer(verbose=False)
            
            # Very small dataset (<100K) - should use STRtree (CPU faster due to transfer overhead)
            method = optimizer.select_method(n_points=50_000, n_polygons=100)
            assert method == 'strtree'
            
            # Medium dataset (100K-1M) - should use GPU
            method = optimizer.select_method(n_points=500_000, n_polygons=100)
            assert method == 'gpu'
            
            # Large dataset (1M-10M) - should use GPU chunked
            method = optimizer.select_method(n_points=5_000_000, n_polygons=100)
            assert method == 'gpu_chunked'
            
            # Very large dataset (>10M) - should use GPU chunked
            method = optimizer.select_method(n_points=15_000_000, n_polygons=100)
            assert method == 'gpu_chunked'
        finally:
            # Restore original value
            GroundTruthOptimizer._gpu_available = original_value


class TestOptimizerRepr:
    """Test string representation of optimizer."""
    
    def test_repr_cpu(self):
        """Test repr for CPU-only optimizer."""
        original_value = GroundTruthOptimizer._gpu_available
        GroundTruthOptimizer._gpu_available = False
        
        try:
            optimizer = GroundTruthOptimizer(verbose=False)
            repr_str = repr(optimizer)
            
            assert 'GroundTruthOptimizer' in repr_str
            assert 'CPU only' in repr_str
            assert 'auto-select' in repr_str
        finally:
            GroundTruthOptimizer._gpu_available = original_value
    
    @requires_gpu
    def test_repr_gpu(self):
        """Test repr for GPU-enabled optimizer."""
        original_value = GroundTruthOptimizer._gpu_available
        GroundTruthOptimizer._gpu_available = True
        
        try:
            optimizer = GroundTruthOptimizer(verbose=False)
            repr_str = repr(optimizer)
            
            assert 'GroundTruthOptimizer' in repr_str
        finally:
            GroundTruthOptimizer._gpu_available = original_value
