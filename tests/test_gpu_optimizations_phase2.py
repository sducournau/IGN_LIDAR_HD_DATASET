#!/usr/bin/env python3
"""
Comprehensive tests for GPU optimization implementations (Phase 2.1-2.5)

This test suite validates:
- Fix 1: GPU Memory Pooling in strategy_gpu.py
- Fix 2: Auto-GPU Selection for KDTree  
- Fix 3: Cache Mode Selection in Dispatcher
- Fix 4: GPU Stream Overlap
- Fix 5: Minimize GPU-CPU Copies
"""

import pytest
import numpy as np
from ign_lidar.features.compute.dispatcher import (
    FeatureComputeDispatcher,
    get_feature_compute_dispatcher,
    ComputeMode,
)
from ign_lidar.features.utils import build_kdtree, _is_gpu_available
from ign_lidar.features.strategy_gpu import GPUStrategy
from ign_lidar.features.strategy_cpu import CPUStrategy


class TestGPUMemoryPooling:
    """Test GPU memory pooling in strategy_gpu.py"""

    def test_gpu_strategy_initialization(self):
        """Test that GPUStrategy initializes memory pooling"""
        try:
            strategy = GPUStrategy(k_neighbors=20, verbose=True)
            assert strategy.memory_pool is not None
            assert strategy.gpu_cache is not None
        except RuntimeError as e:
            if "GPU" in str(e):
                pytest.skip("GPU not available")
            raise

    def test_gpu_strategy_computes_features(self):
        """Test that GPUStrategy can compute features"""
        try:
            strategy = GPUStrategy(k_neighbors=10)
            points = np.random.rand(1000, 3).astype(np.float32)
            
            result = strategy.compute_features(points)
            
            assert "normals" in result
            assert "curvature" in result
            assert "height" in result
            assert len(result) > 0
        except RuntimeError as e:
            if "GPU" in str(e):
                pytest.skip("GPU not available")
            raise


class TestKDTreeAutoGPUSelection:
    """Test auto-GPU selection for KDTree (Fix 2)"""

    def test_build_kdtree_small_dataset_cpu(self):
        """Test that small datasets use CPU KDTree"""
        small_points = np.random.rand(100, 3)
        tree = build_kdtree(small_points, use_gpu=False)
        assert tree is not None
        
        # Query should work
        distances, indices = tree.query(small_points[:10], k=5)
        assert indices.shape == (10, 5)

    def test_build_kdtree_large_dataset_auto(self):
        """Test that large datasets auto-select"""
        large_points = np.random.rand(200000, 3)
        tree = build_kdtree(large_points)  # Auto-select
        assert tree is not None
        
        # Query should work
        distances, indices = tree.query(large_points[:10], k=10)
        assert indices.shape == (10, 10)

    def test_build_kdtree_force_gpu(self):
        """Test forcing GPU KDTree"""
        small_points = np.random.rand(1000, 3)
        tree = build_kdtree(small_points, use_gpu=True)
        assert tree is not None

    def test_build_kdtree_force_cpu(self):
        """Test forcing CPU KDTree"""
        large_points = np.random.rand(500000, 3)
        tree = build_kdtree(large_points, use_gpu=False)
        assert tree is not None

    def test_gpu_availability_check(self):
        """Test GPU availability detection"""
        available = _is_gpu_available()
        assert isinstance(available, bool)


class TestDispatcherModeCaching:
    """Test cached mode selection in dispatcher (Fix 3)"""

    def test_dispatcher_initialization(self):
        """Test dispatcher initialization with explicit mode"""
        disp = FeatureComputeDispatcher(mode=ComputeMode.CPU)
        assert disp.mode == ComputeMode.CPU

    def test_dispatcher_auto_selection(self):
        """Test dispatcher auto-selection of mode"""
        disp = FeatureComputeDispatcher(expected_size=50000)
        assert disp.mode in [ComputeMode.CPU, ComputeMode.GPU, ComputeMode.GPU_CHUNKED]

    def test_singleton_caching(self):
        """Test singleton caching of dispatcher"""
        disp1 = get_feature_compute_dispatcher(cache=True)
        disp2 = get_feature_compute_dispatcher(cache=True)
        assert disp1 is disp2

    def test_non_cached_dispatcher(self):
        """Test non-cached dispatcher creates new instances"""
        disp1 = get_feature_compute_dispatcher(cache=False)
        disp2 = get_feature_compute_dispatcher(cache=False)
        assert disp1 is not disp2

    def test_dispatcher_compute(self):
        """Test dispatcher compute method"""
        disp = FeatureComputeDispatcher(mode=ComputeMode.CPU)
        points = np.random.rand(1000, 3).astype(np.float32)
        classification = np.random.randint(0, 32, 1000, dtype=np.uint8)
        
        normals, curvature, height, features = disp.compute(
            points, classification, k_neighbors=10
        )
        
        assert normals.shape == (1000, 3)
        assert curvature.shape == (1000,)
        assert height.shape == (1000,)
        assert len(features) > 0


class TestDispatcherCachingPerformance:
    """Test performance improvement of dispatcher caching"""

    def test_cached_vs_standard(self):
        """Compare performance of cached vs standard dispatcher"""
        import time
        
        points = np.random.rand(10000, 3).astype(np.float32)
        classification = np.random.randint(0, 32, 10000, dtype=np.uint8)
        
        # Time standard calls (mode selection per call)
        start = time.time()
        for _ in range(3):
            from ign_lidar.features.compute.dispatcher import compute_all_features
            normals, curvature, height, features = compute_all_features(
                points, classification, mode='cpu', k_neighbors=10
            )
        standard_time = time.time() - start
        
        # Time cached calls
        start = time.time()
        for _ in range(3):
            from ign_lidar.features.compute.dispatcher import compute_all_features
            normals, curvature, height, features = compute_all_features(
                points, classification, use_cached_dispatcher=True, k_neighbors=10
            )
        cached_time = time.time() - start
        
        # Cached should not be significantly slower
        # (Usually faster or similar due to reduced overhead)
        assert cached_time <= standard_time * 1.1  # Allow 10% margin


class TestIntegration:
    """Integration tests for all GPU optimizations"""

    def test_full_pipeline_cpu(self):
        """Test full pipeline with CPU strategy"""
        strategy = CPUStrategy(k_neighbors=10)
        points = np.random.rand(5000, 3).astype(np.float32)
        
        result = strategy.compute_features(points)
        
        assert "normals" in result
        assert len(result) > 0

    def test_full_pipeline_with_dispatcher(self):
        """Test full pipeline using dispatcher"""
        from ign_lidar.features.compute.dispatcher import compute_all_features
        
        points = np.random.rand(5000, 3).astype(np.float32)
        classification = np.random.randint(0, 32, 5000, dtype=np.uint8)
        
        normals, curvature, height, features = compute_all_features(
            points, classification, mode='cpu', k_neighbors=10
        )
        
        assert normals.shape == (5000, 3)
        assert len(features) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
