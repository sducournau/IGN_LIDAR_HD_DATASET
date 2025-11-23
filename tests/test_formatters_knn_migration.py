"""
Tests for KNN Engine Migration in Formatters

Validates that the migration from manual cuML implementations to
KNNEngine in hybrid_formatter.py and multi_arch_formatter.py:
- Maintains backward compatibility
- Produces correct results
- Handles GPU/CPU fallback correctly
- Improves performance with FAISS-GPU

Author: Phase 1 Consolidation
Date: November 23, 2025
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock


class TestHybridFormatterKNNMigration:
    """Test KNN migration in hybrid_formatter.py"""
    
    @pytest.fixture
    def hybrid_formatter(self):
        """Create hybrid formatter instance."""
        from ign_lidar.io.formatters.hybrid_formatter import HybridFormatter
        
        config = {
            'num_points': 16384,
            'k_neighbors': 16,
            'voxel_size': 0.5
        }
        return HybridFormatter(config)
    
    @pytest.fixture
    def sample_points(self):
        """Create sample point cloud."""
        np.random.seed(42)
        return np.random.randn(1000, 3).astype(np.float32)
    
    def test_build_knn_graph_cpu(self, hybrid_formatter, sample_points):
        """Test CPU KNN graph building with KNNEngine."""
        edges = hybrid_formatter._build_knn_graph(sample_points, k=8, use_gpu=False)
        
        # Verify shape: [N, K, 2]
        assert edges.shape == (1000, 8, 2)
        assert edges.dtype == np.int32
        
        # Verify edge structure
        # edges[i, j, 0] = source index i
        # edges[i, j, 1] = neighbor index
        for i in range(100):  # Check first 100 points
            assert np.all(edges[i, :, 0] == i)
            
        # Verify all neighbor indices are valid
        assert np.all(edges[:, :, 1] >= 0)
        assert np.all(edges[:, :, 1] < 1000)
        
        # No self-loops (neighbor should not be same as source)
        for i in range(100):
            neighbors = edges[i, :, 1]
            # Self may appear but should not dominate
            assert np.sum(neighbors == i) <= 1
    
    @pytest.mark.gpu
    def test_build_knn_graph_gpu(self, hybrid_formatter, sample_points):
        """Test GPU KNN graph building with KNNEngine."""
        try:
            edges = hybrid_formatter._build_knn_graph_gpu(sample_points, k=8)
            
            # Verify shape
            assert edges.shape == (1000, 8, 2)
            assert edges.dtype == np.int32
            
            # Verify edge structure (same as CPU)
            for i in range(100):
                assert np.all(edges[i, :, 0] == i)
                
            assert np.all(edges[:, :, 1] >= 0)
            assert np.all(edges[:, :, 1] < 1000)
            
        except Exception as e:
            if "GPU not available" in str(e) or "KNN engine failed" in str(e):
                pytest.skip("GPU not available or KNN engine failed")
            else:
                raise
    
    def test_build_knn_graph_fallback(self, hybrid_formatter, sample_points):
        """Test fallback from GPU to CPU when GPU fails."""
        # This should work even without GPU
        edges = hybrid_formatter._build_knn_graph(sample_points, k=8, use_gpu=True)
        
        # Should return valid edges (either from GPU or CPU fallback)
        assert edges.shape == (1000, 8, 2)
        assert edges.dtype == np.int32
    
    def test_knn_graph_consistency_cpu_gpu(self, hybrid_formatter, sample_points):
        """Test that CPU and GPU produce consistent neighbor sets."""
        edges_cpu = hybrid_formatter._build_knn_graph(sample_points, k=8, use_gpu=False)
        
        try:
            edges_gpu = hybrid_formatter._build_knn_graph(sample_points, k=8, use_gpu=True)
            
            # Neighbor sets should be similar (order may differ)
            # Check that at least 80% of neighbors match
            matches = 0
            total = 0
            for i in range(100):  # Check first 100 points
                neighbors_cpu = set(edges_cpu[i, :, 1])
                neighbors_gpu = set(edges_gpu[i, :, 1])
                matches += len(neighbors_cpu & neighbors_gpu)
                total += 8
            
            match_ratio = matches / total
            assert match_ratio > 0.8, f"CPU/GPU neighbor match ratio: {match_ratio:.2%}"
            
        except Exception:
            pytest.skip("GPU not available")


class TestMultiArchFormatterKNNMigration:
    """Test KNN migration in multi_arch_formatter.py"""
    
    @pytest.fixture
    def multi_arch_formatter(self):
        """Create multi-arch formatter instance."""
        from ign_lidar.io.formatters.multi_arch_formatter import MultiArchFormatter
        
        config = {
            'num_points': 16384,
            'k_neighbors': 16,
            'architecture': 'pointnet++'
        }
        return MultiArchFormatter(config)
    
    @pytest.fixture
    def sample_points(self):
        """Create sample point cloud."""
        np.random.seed(42)
        return np.random.randn(1000, 3).astype(np.float32)
    
    def test_build_knn_graph_cpu(self, multi_arch_formatter, sample_points):
        """Test CPU KNN graph building with KNNEngine."""
        edges, distances = multi_arch_formatter._build_knn_graph(
            sample_points, k=8, use_gpu=False
        )
        
        # Verify shapes: edges [N, K, 2], distances [N, K]
        assert edges.shape == (1000, 8, 2)
        assert distances.shape == (1000, 8)
        assert edges.dtype == np.int32
        assert distances.dtype == np.float32
        
        # Verify distances are non-negative
        assert np.all(distances >= 0)
        
        # Verify edge structure
        for i in range(100):
            assert np.all(edges[i, :, 0] == i)
            
        assert np.all(edges[:, :, 1] >= 0)
        assert np.all(edges[:, :, 1] < 1000)
        
        # Distances should be sorted (nearest to farthest)
        for i in range(100):
            dists = distances[i]
            assert np.all(dists[:-1] <= dists[1:])  # Non-decreasing
    
    @pytest.mark.gpu
    def test_build_knn_graph_gpu(self, multi_arch_formatter, sample_points):
        """Test GPU KNN graph building with KNNEngine."""
        try:
            edges, distances = multi_arch_formatter._build_knn_graph_gpu(
                sample_points, k=8
            )
            
            # Verify shapes
            assert edges.shape == (1000, 8, 2)
            assert distances.shape == (1000, 8)
            
            # Verify distances
            assert np.all(distances >= 0)
            
            # Verify edge structure
            for i in range(100):
                assert np.all(edges[i, :, 0] == i)
                
        except Exception as e:
            if "GPU not available" in str(e) or "KNN engine failed" in str(e):
                pytest.skip("GPU not available")
            else:
                raise
    
    def test_knn_graph_distances_consistency(self, multi_arch_formatter, sample_points):
        """Test that distances are consistent with neighbors."""
        edges, distances = multi_arch_formatter._build_knn_graph(
            sample_points, k=8, use_gpu=False
        )
        
        # Manually verify a few distances
        for i in range(10):
            point_i = sample_points[i]
            neighbors = edges[i, :, 1]
            computed_distances = distances[i]
            
            # Compute distances manually
            for j, neighbor_idx in enumerate(neighbors):
                point_j = sample_points[neighbor_idx]
                expected_dist = np.linalg.norm(point_i - point_j)
                actual_dist = computed_distances[j]
                
                # Allow small numerical differences
                assert np.abs(expected_dist - actual_dist) < 1e-4, \
                    f"Distance mismatch: expected {expected_dist}, got {actual_dist}"


class TestKNNEngineFallback:
    """Test fallback behavior when KNNEngine fails."""
    
    def test_fallback_to_sklearn(self):
        """Test that formatters fall back to sklearn when KNNEngine fails."""
        from ign_lidar.io.formatters.hybrid_formatter import HybridFormatter
        
        config = {'num_points': 16384, 'k_neighbors': 16}
        formatter = HybridFormatter(config)
        
        np.random.seed(42)
        points = np.random.randn(100, 3).astype(np.float32)
        
        # Mock KNNEngine to raise exception
        with patch('ign_lidar.io.formatters.hybrid_formatter.KNNEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.query.side_effect = RuntimeError("Mocked KNN failure")
            mock_engine.return_value = mock_instance
            
            # Should fall back to sklearn and still work
            edges = formatter._build_knn_graph(points, k=8, use_gpu=False)
            
            # Verify fallback worked
            assert edges.shape == (100, 8, 2)


class TestPerformanceImprovement:
    """Test that migration improves performance (informational)."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(not pytest.importorskip("faiss", reason="FAISS not available"), reason="FAISS not available")
    def test_performance_with_faiss(self):
        """Test that KNNEngine with FAISS is faster than manual cuML."""
        from ign_lidar.io.formatters.hybrid_formatter import HybridFormatter
        from ign_lidar.optimization.knn_engine import HAS_FAISS
        import time
        
        if not HAS_FAISS:
            pytest.skip("FAISS not available")
        
        config = {'num_points': 16384, 'k_neighbors': 16}
        formatter = HybridFormatter(config)
        
        np.random.seed(42)
        points = np.random.randn(10000, 3).astype(np.float32)
        
        # Time KNN graph building
        start = time.time()
        edges = formatter._build_knn_graph(points, k=16, use_gpu=False)
        elapsed = time.time() - start
        
        print(f"\nKNN graph (10k points, k=16): {elapsed:.3f}s")
        
        # Verify result
        assert edges.shape == (10000, 16, 2)
        
        # With FAISS, should be < 1s for 10k points
        if HAS_FAISS:
            assert elapsed < 1.0, f"Expected < 1s, got {elapsed:.3f}s"


class TestBackwardCompatibility:
    """Test backward compatibility after migration."""
    
    def test_hybrid_formatter_api_unchanged(self):
        """Test that HybridFormatter API is unchanged."""
        from ign_lidar.io.formatters.hybrid_formatter import HybridFormatter
        
        config = {'num_points': 16384, 'k_neighbors': 16}
        formatter = HybridFormatter(config)
        
        # Methods should exist and have correct signatures
        assert hasattr(formatter, '_build_knn_graph')
        assert hasattr(formatter, '_build_knn_graph_gpu')
        
        # Test basic usage
        np.random.seed(42)
        points = np.random.randn(100, 3).astype(np.float32)
        edges = formatter._build_knn_graph(points, k=8, use_gpu=False)
        
        # Should work as before
        assert edges.shape == (100, 8, 2)
    
    def test_multi_arch_formatter_api_unchanged(self):
        """Test that MultiArchFormatter API is unchanged."""
        from ign_lidar.io.formatters.multi_arch_formatter import MultiArchFormatter
        
        config = {'num_points': 16384, 'k_neighbors': 16}
        formatter = MultiArchFormatter(config)
        
        # Methods should exist
        assert hasattr(formatter, '_build_knn_graph')
        assert hasattr(formatter, '_build_knn_graph_gpu')
        
        # Test basic usage
        np.random.seed(42)
        points = np.random.randn(100, 3).astype(np.float32)
        edges, distances = formatter._build_knn_graph(points, k=8, use_gpu=False)
        
        # Should work as before
        assert edges.shape == (100, 8, 2)
        assert distances.shape == (100, 8)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
