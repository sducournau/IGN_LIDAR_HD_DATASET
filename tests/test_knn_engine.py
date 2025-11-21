"""
Unit tests for unified KNN engine.

Tests the Phase 2 refactoring:
- KNNEngine multi-backend support
- Automatic backend selection
- All backends (FAISS-GPU, FAISS-CPU, cuML, sklearn)
- KNN graph building
- Convenience functions

Author: LiDAR Trainer Agent (Phase 2: KNN Consolidation)
Date: November 21, 2025
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestKNNEngineBasic:
    """Basic tests for KNN engine."""
    
    def test_imports(self):
        """Test that KNN engine can be imported."""
        from ign_lidar.optimization.knn_engine import (
            KNNEngine,
            KNNBackend,
            knn_search,
            build_knn_graph,
            HAS_FAISS,
            HAS_FAISS_GPU,
            HAS_CUML
        )
        
        assert KNNEngine is not None
        assert KNNBackend is not None
        assert callable(knn_search)
        assert callable(build_knn_graph)
        
        # Check availability flags
        assert isinstance(HAS_FAISS, bool)
        assert isinstance(HAS_FAISS_GPU, bool)
        assert isinstance(HAS_CUML, bool)
    
    def test_backend_enum(self):
        """Test KNNBackend enum."""
        from ign_lidar.optimization.knn_engine import KNNBackend
        
        assert KNNBackend.FAISS_GPU.value == "faiss-gpu"
        assert KNNBackend.FAISS_CPU.value == "faiss-cpu"
        assert KNNBackend.CUML.value == "cuml"
        assert KNNBackend.SKLEARN.value == "sklearn"
        assert KNNBackend.AUTO.value == "auto"
    
    def test_engine_initialization(self):
        """Test KNN engine initialization."""
        from ign_lidar.optimization.knn_engine import KNNEngine
        
        # Default initialization
        engine = KNNEngine()
        assert engine.backend_preference == 'auto'
        assert engine.metric == 'euclidean'
        
        # Custom backend
        engine = KNNEngine(backend='sklearn')
        assert engine.backend_preference == 'sklearn'
        
        # Custom metric
        engine = KNNEngine(metric='cosine')
        assert engine.metric == 'cosine'


class TestKNNSearch:
    """Tests for KNN search functionality."""
    
    def test_knn_search_sklearn(self):
        """Test KNN search with sklearn backend."""
        from ign_lidar.optimization.knn_engine import knn_search
        
        # Create test data
        np.random.seed(42)
        points = np.random.randn(1000, 3).astype(np.float32)
        
        # Self-query
        distances, indices = knn_search(points, k=10, backend='sklearn')
        
        # Verify shape
        assert distances.shape == (1000, 10)
        assert indices.shape == (1000, 10)
        
        # First neighbor should be self (distance ~0)
        assert np.allclose(distances[:, 0], 0.0, atol=1e-5)
        assert np.all(indices[:, 0] == np.arange(1000))
    
    def test_knn_search_separate_queries(self):
        """Test KNN search with separate query set."""
        from ign_lidar.optimization.knn_engine import knn_search
        
        np.random.seed(42)
        reference = np.random.randn(1000, 3).astype(np.float32)
        queries = np.random.randn(100, 3).astype(np.float32)
        
        distances, indices = knn_search(
            points=reference,
            query_points=queries,
            k=5,
            backend='sklearn'
        )
        
        # Verify shape
        assert distances.shape == (100, 5)
        assert indices.shape == (100, 5)
        
        # All distances should be >= 0
        assert np.all(distances >= 0)
        
        # All indices should be valid
        assert np.all(indices >= 0)
        assert np.all(indices < 1000)
    
    @pytest.mark.skipif(not pytest.importorskip("faiss", reason="FAISS not available"), reason="FAISS not available")
    def test_knn_search_faiss_cpu(self):
        """Test KNN search with FAISS-CPU backend."""
        from ign_lidar.optimization.knn_engine import knn_search, HAS_FAISS
        
        if not HAS_FAISS:
            pytest.skip("FAISS not available")
        
        np.random.seed(42)
        points = np.random.randn(1000, 3).astype(np.float32)
        
        distances, indices = knn_search(points, k=10, backend='faiss-cpu')
        
        assert distances.shape == (1000, 10)
        assert indices.shape == (1000, 10)
        assert np.allclose(distances[:, 0], 0.0, atol=1e-5)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not pytest.importorskip("faiss", reason="FAISS not available") or
        not hasattr(__import__("faiss"), "StandardGpuResources"),
        reason="FAISS-GPU not available"
    )
    def test_knn_search_faiss_gpu(self):
        """Test KNN search with FAISS-GPU backend (GPU required)."""
        from ign_lidar.optimization.knn_engine import knn_search, HAS_FAISS_GPU
        
        if not HAS_FAISS_GPU:
            pytest.skip("FAISS-GPU not available")
        
        np.random.seed(42)
        points = np.random.randn(1000, 3).astype(np.float32)
        
        distances, indices = knn_search(points, k=10, backend='faiss-gpu')
        
        assert distances.shape == (1000, 10)
        assert indices.shape == (1000, 10)
        assert np.allclose(distances[:, 0], 0.0, atol=1e-5)
    
    def test_knn_search_auto_backend(self):
        """Test automatic backend selection."""
        from ign_lidar.optimization.knn_engine import knn_search
        
        np.random.seed(42)
        
        # Small dataset - should use sklearn
        small_points = np.random.randn(100, 3).astype(np.float32)
        distances, indices = knn_search(small_points, k=5, backend='auto')
        assert distances.shape == (100, 5)
        
        # Larger dataset - backend depends on availability
        large_points = np.random.randn(10000, 3).astype(np.float32)
        distances, indices = knn_search(large_points, k=10, backend='auto')
        assert distances.shape == (10000, 10)


class TestKNNEngine:
    """Tests for KNNEngine class."""
    
    def test_engine_fit_query(self):
        """Test fit-then-query workflow."""
        from ign_lidar.optimization.knn_engine import KNNEngine
        
        np.random.seed(42)
        reference = np.random.randn(1000, 3).astype(np.float32)
        queries = np.random.randn(100, 3).astype(np.float32)
        
        engine = KNNEngine(backend='sklearn')
        engine.fit(reference)
        distances, indices = engine.search(queries, k=5)
        
        assert distances.shape == (100, 5)
        assert indices.shape == (100, 5)
    
    def test_engine_backend_selection(self):
        """Test backend selection logic."""
        from ign_lidar.optimization.knn_engine import KNNEngine, KNNBackend
        
        engine = KNNEngine()
        
        # Small dataset - sklearn
        backend = engine._select_backend(n_points=1000, n_dims=3, k=10)
        assert backend == KNNBackend.SKLEARN
        
        # Large dataset - depends on availability
        backend = engine._select_backend(n_points=200_000, n_dims=3, k=10)
        assert backend in [
            KNNBackend.FAISS_GPU,
            KNNBackend.FAISS_CPU,
            KNNBackend.CUML,
            KNNBackend.SKLEARN
        ]
    
    def test_engine_invalid_input(self):
        """Test error handling for invalid inputs."""
        from ign_lidar.optimization.knn_engine import KNNEngine
        
        engine = KNNEngine()
        
        # 1D array should fail
        with pytest.raises(ValueError):
            engine.search(np.array([1, 2, 3]), k=5)
        
        # k >= n_points should fail
        points = np.random.randn(10, 3)
        with pytest.raises(ValueError):
            engine.search(points, k=10)


class TestKNNGraph:
    """Tests for KNN graph building."""
    
    def test_build_knn_graph(self):
        """Test KNN graph construction."""
        from ign_lidar.optimization.knn_engine import build_knn_graph
        
        np.random.seed(42)
        points = np.random.randn(1000, 3).astype(np.float32)
        
        # Build graph
        neighbors = build_knn_graph(points, k=10, backend='sklearn')
        
        # Verify shape
        assert neighbors.shape == (1000, 10)
        
        # First neighbor should be self
        assert np.all(neighbors[:, 0] == np.arange(1000))
        
        # All indices should be valid
        assert np.all(neighbors >= 0)
        assert np.all(neighbors < 1000)
    
    def test_knn_graph_connectivity(self):
        """Test that KNN graph has proper connectivity."""
        from ign_lidar.optimization.knn_engine import build_knn_graph
        
        np.random.seed(42)
        points = np.random.randn(100, 3).astype(np.float32)
        
        neighbors = build_knn_graph(points, k=5, backend='sklearn')
        
        # Each point should have k neighbors
        assert neighbors.shape == (100, 5)
        
        # No duplicate neighbors (except self at position 0)
        for i in range(100):
            neighbor_set = set(neighbors[i, 1:])  # Exclude self
            assert len(neighbor_set) == 4  # k-1 unique neighbors


class TestBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_knn_engine_import_from_optimization(self):
        """Test that KNNEngine can be imported from optimization module."""
        # This will be available after updating __init__.py
        try:
            from ign_lidar.optimization import KNNEngine, knn_search
            assert KNNEngine is not None
            assert knn_search is not None
        except ImportError:
            pytest.skip("Optimization module not yet updated")


class TestPerformance:
    """Performance comparison tests (informational)."""
    
    @pytest.mark.slow
    def test_performance_comparison(self):
        """Compare performance of different backends (informational)."""
        from ign_lidar.optimization.knn_engine import knn_search, HAS_FAISS
        import time
        
        np.random.seed(42)
        points = np.random.randn(50000, 3).astype(np.float32)
        k = 30
        
        # Sklearn baseline
        start = time.time()
        distances_sklearn, _ = knn_search(points, k=k, backend='sklearn')
        time_sklearn = time.time() - start
        
        # FAISS-CPU (if available)
        if HAS_FAISS:
            start = time.time()
            distances_faiss, _ = knn_search(points, k=k, backend='faiss-cpu')
            time_faiss = time.time() - start
            
            speedup = time_sklearn / time_faiss
            print(f"\nFAISS-CPU speedup: {speedup:.2f}x")
            assert speedup > 1.5  # FAISS should be at least 1.5x faster


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
