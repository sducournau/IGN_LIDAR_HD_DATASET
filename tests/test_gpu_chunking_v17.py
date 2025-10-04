"""
Tests for GPU Chunked Processing and Adaptive Memory Management
Version: 1.7.0
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path


class TestGPUChunkedFeatureComputer:
    """Test GPU chunked feature computation."""
    
    @pytest.fixture
    def sample_points(self):
        """Generate sample point cloud."""
        np.random.seed(42)
        n_points = 10_000
        points = np.random.randn(n_points, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])  # Positive Z
        return points
    
    @pytest.fixture
    def sample_classification(self, sample_points):
        """Generate sample classification."""
        n_points = len(sample_points)
        # Mix of ground (2) and building (6)
        classification = np.random.choice([2, 6], size=n_points)
        return classification.astype(np.uint8)
    
    def test_gpu_chunked_computer_initialization(self):
        """Test GPUChunkedFeatureComputer initialization."""
        try:
            from ign_lidar.features_gpu_chunked import (
                GPUChunkedFeatureComputer
            )
            
            computer = GPUChunkedFeatureComputer(
                chunk_size=5_000_000,
                vram_limit_gb=8.0,
                use_gpu=True
            )
            
            assert computer.chunk_size == 5_000_000
            assert computer.vram_limit_gb == 8.0
            
        except ImportError:
            pytest.skip("GPU modules not available")
    
    def test_compute_normals_chunked(self, sample_points):
        """Test chunked normal computation."""
        try:
            from ign_lidar.features_gpu_chunked import (
                GPUChunkedFeatureComputer
            )
            
            computer = GPUChunkedFeatureComputer(
                chunk_size=5_000,  # Small chunk for testing
                use_gpu=True
            )
            
            normals = computer.compute_normals_chunked(
                sample_points, k=10
            )
            
            # Check shape
            assert normals.shape == (len(sample_points), 3)
            
            # Check normalization
            norms = np.linalg.norm(normals, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5)
            
            # Check orientation (mostly upward)
            assert np.mean(normals[:, 2]) > 0
            
        except ImportError:
            pytest.skip("GPU modules not available")
    
    def test_compute_curvature_chunked(self, sample_points):
        """Test chunked curvature computation."""
        try:
            from ign_lidar.features_gpu_chunked import (
                GPUChunkedFeatureComputer
            )
            
            computer = GPUChunkedFeatureComputer(
                chunk_size=5_000,
                use_gpu=True
            )
            
            # First compute normals
            normals = computer.compute_normals_chunked(
                sample_points, k=10
            )
            
            # Then compute curvature
            curvature = computer.compute_curvature_chunked(
                sample_points, normals, k=10
            )
            
            # Check shape
            assert curvature.shape == (len(sample_points),)
            
            # Check range
            assert np.all(curvature >= 0)
            assert np.all(curvature < 10)  # Reasonable range
            
        except ImportError:
            pytest.skip("GPU modules not available")
    
    def test_compute_all_features_chunked(
        self, sample_points, sample_classification
    ):
        """Test all features with chunked processing."""
        try:
            from ign_lidar.features_gpu_chunked import (
                GPUChunkedFeatureComputer
            )
            
            computer = GPUChunkedFeatureComputer(
                chunk_size=5_000,
                use_gpu=True
            )
            
            normals, curvature, height, geo_features = \
                computer.compute_all_features_chunked(
                    sample_points,
                    sample_classification,
                    k=10
                )
            
            # Check shapes
            assert normals.shape == (len(sample_points), 3)
            assert curvature.shape == (len(sample_points),)
            assert height.shape == (len(sample_points),)
            
            # Check geometric features
            assert 'planarity' in geo_features
            assert 'sphericity' in geo_features
            assert 'linearity' in geo_features
            
        except ImportError:
            pytest.skip("GPU modules not available")
    
    def test_memory_cleanup(self, sample_points):
        """Test GPU memory is properly cleaned up."""
        try:
            from ign_lidar.features_gpu_chunked import (
                GPUChunkedFeatureComputer
            )
            import cupy as cp
            
            # Get initial memory usage
            mempool = cp.get_default_memory_pool()
            initial_used = mempool.used_bytes()
            
            # Process
            computer = GPUChunkedFeatureComputer(
                chunk_size=5_000,
                use_gpu=True
            )
            computer.compute_normals_chunked(sample_points, k=10)
            
            # Check memory cleaned up
            final_used = mempool.used_bytes()
            assert final_used <= initial_used * 1.1  # Allow 10% overhead
            
        except ImportError:
            pytest.skip("GPU modules not available")
    
    def test_fallback_to_cpu(self, sample_points):
        """Test automatic fallback to CPU if GPU fails."""
        try:
            from ign_lidar.features_gpu_chunked import (
                GPUChunkedFeatureComputer
            )
            
            # Force GPU unavailable
            computer = GPUChunkedFeatureComputer(
                chunk_size=5_000,
                use_gpu=False  # Explicit CPU
            )
            
            # Should still work
            normals = computer.compute_normals_chunked(
                sample_points, k=10
            )
            
            assert normals.shape == (len(sample_points), 3)
            
        except ImportError:
            pytest.skip("GPU modules not available")


class TestAdaptiveMemoryManager:
    """Test adaptive memory management."""
    
    def test_initialization(self):
        """Test AdaptiveMemoryManager initialization."""
        from ign_lidar.memory_manager import AdaptiveMemoryManager
        
        manager = AdaptiveMemoryManager(
            min_chunk_size=1_000_000,
            max_chunk_size=20_000_000,
            enable_gpu=True
        )
        
        assert manager.min_chunk_size == 1_000_000
        assert manager.max_chunk_size == 20_000_000
        assert manager.enable_gpu is True
    
    def test_get_current_memory_status(self):
        """Test memory status retrieval."""
        from ign_lidar.memory_manager import AdaptiveMemoryManager
        
        manager = AdaptiveMemoryManager()
        
        available_ram, swap_percent, vram_free = \
            manager.get_current_memory_status()
        
        # Should return reasonable values
        assert available_ram > 0
        assert 0 <= swap_percent <= 100
        assert vram_free >= 0
    
    def test_estimate_memory_needed(self):
        """Test memory estimation."""
        from ign_lidar.memory_manager import AdaptiveMemoryManager
        
        manager = AdaptiveMemoryManager()
        
        # Estimate for 10M points, full mode, no augmentation
        memory_gb = manager.estimate_memory_needed(
            num_points=10_000_000,
            mode='full',
            num_augmentations=0
        )
        
        # Should be reasonable (3-7 GB for 10M points)
        assert 2 < memory_gb < 10
        
        # With augmentation should be more
        memory_gb_aug = manager.estimate_memory_needed(
            num_points=10_000_000,
            mode='full',
            num_augmentations=2
        )
        
        assert memory_gb_aug > memory_gb
    
    def test_calculate_optimal_chunk_size(self):
        """Test optimal chunk size calculation."""
        from ign_lidar.memory_manager import AdaptiveMemoryManager
        
        manager = AdaptiveMemoryManager()
        
        # Small dataset - should not chunk
        chunk_size = manager.calculate_optimal_chunk_size(
            num_points=5_000_000,
            mode='full',
            num_augmentations=0
        )
        
        # Either no chunking or very large chunks
        assert chunk_size >= 5_000_000
        
        # Large dataset with augmentation - should chunk
        chunk_size_aug = manager.calculate_optimal_chunk_size(
            num_points=20_000_000,
            mode='full',
            num_augmentations=2
        )
        
        # Should use chunking
        assert chunk_size_aug < 20_000_000
        assert chunk_size_aug >= manager.min_chunk_size
    
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    def test_chunk_size_with_low_memory(
        self, mock_swap, mock_mem
    ):
        """Test chunk size with low available memory."""
        from ign_lidar.memory_manager import AdaptiveMemoryManager
        
        # Mock low memory
        mock_mem.return_value = Mock(
            total=8 * 1024**3,      # 8 GB total
            available=2 * 1024**3,  # 2 GB available
            percent=75.0
        )
        mock_swap.return_value = Mock(percent=30.0)
        
        manager = AdaptiveMemoryManager()
        
        chunk_size = manager.calculate_optimal_chunk_size(
            num_points=20_000_000,
            mode='full',
            num_augmentations=0
        )
        
        # Should use smaller chunks due to low memory
        assert chunk_size < 10_000_000
    
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    def test_chunk_size_with_high_swap(
        self, mock_swap, mock_mem
    ):
        """Test chunk size with high swap usage."""
        from ign_lidar.memory_manager import AdaptiveMemoryManager
        
        # Mock high swap usage
        mock_mem.return_value = Mock(
            total=16 * 1024**3,
            available=4 * 1024**3,
            percent=75.0
        )
        mock_swap.return_value = Mock(percent=75.0)  # High swap!
        
        manager = AdaptiveMemoryManager()
        
        chunk_size = manager.calculate_optimal_chunk_size(
            num_points=20_000_000,
            mode='full',
            num_augmentations=0
        )
        
        # Should use minimum chunk size
        assert chunk_size == manager.min_chunk_size
    
    def test_calculate_optimal_workers(self):
        """Test optimal worker calculation."""
        from ign_lidar.memory_manager import AdaptiveMemoryManager
        
        manager = AdaptiveMemoryManager()
        
        # Multiple small files
        workers = manager.calculate_optimal_workers(
            num_files=8,
            file_sizes_mb=[100, 120, 90, 110, 95, 105, 115, 100],
            mode='core'
        )
        
        # Should allow multiple workers
        assert 1 <= workers <= 8
        
        # Large files
        workers_large = manager.calculate_optimal_workers(
            num_files=4,
            file_sizes_mb=[600, 650, 580, 620],
            mode='full'
        )
        
        # Should limit workers for large files
        assert workers_large <= 3
    
    def test_get_optimal_config(self):
        """Test complete optimal configuration."""
        from ign_lidar.memory_manager import AdaptiveMemoryManager
        
        manager = AdaptiveMemoryManager()
        
        config = manager.get_optimal_config(
            num_points=17_000_000,
            num_augmentations=2,
            mode='full',
            num_files=1
        )
        
        # Check all fields present
        assert hasattr(config, 'chunk_size')
        assert hasattr(config, 'max_workers')
        assert hasattr(config, 'use_gpu')
        assert hasattr(config, 'available_ram_gb')
        assert hasattr(config, 'swap_usage_percent')
        
        # Check reasonable values
        assert config.chunk_size >= 1_000_000
        assert config.max_workers >= 1
        assert config.available_ram_gb > 0
    
    def test_monitor_during_processing(self):
        """Test memory monitoring during processing."""
        from ign_lidar.memory_manager import AdaptiveMemoryManager
        
        manager = AdaptiveMemoryManager()
        
        # Should return True if memory OK
        is_ok = manager.monitor_during_processing(
            warn_threshold_percent=85.0
        )
        
        assert isinstance(is_ok, bool)


class TestIntegration:
    """Integration tests for v1.7.0 features."""
    
    def test_gpu_chunking_integration(self, tmp_path):
        """Test GPU chunking integration with CLI."""
        pytest.skip("Integration test - requires real data")
        # TODO: Add integration test with real LAZ file
    
    def test_adaptive_memory_integration(self, tmp_path):
        """Test adaptive memory with enrichment."""
        pytest.skip("Integration test - requires real data")
        # TODO: Add integration test with enrichment command


class TestPerformance:
    """Performance tests for v1.7.0 features."""
    
    @pytest.mark.slow
    def test_gpu_vs_cpu_speed(self):
        """Compare GPU chunked vs CPU performance."""
        try:
            from ign_lidar.features_gpu_chunked import (
                GPUChunkedFeatureComputer
            )
            from ign_lidar.features_gpu import GPUFeatureComputer
            import time
            
            # Generate large dataset
            np.random.seed(42)
            points = np.random.randn(1_000_000, 3).astype(np.float32)
            
            # CPU timing
            cpu_computer = GPUFeatureComputer(use_gpu=False)
            start = time.time()
            normals_cpu = cpu_computer.compute_normals(points, k=10)
            cpu_time = time.time() - start
            
            # GPU chunked timing
            gpu_computer = GPUChunkedFeatureComputer(
                chunk_size=500_000,
                use_gpu=True
            )
            start = time.time()
            normals_gpu = gpu_computer.compute_normals_chunked(
                points, k=10
            )
            gpu_time = time.time() - start
            
            print(f"\nCPU time: {cpu_time:.2f}s")
            print(f"GPU time: {gpu_time:.2f}s")
            print(f"Speedup: {cpu_time / gpu_time:.1f}x")
            
            # GPU should be faster
            assert gpu_time < cpu_time
            
            # Results should be similar
            assert np.allclose(normals_cpu, normals_gpu, atol=1e-2)
            
        except ImportError:
            pytest.skip("GPU not available for performance test")
    
    @pytest.mark.slow
    def test_memory_usage_chunked(self):
        """Test memory usage with chunked processing."""
        try:
            from ign_lidar.features_gpu_chunked import (
                GPUChunkedFeatureComputer
            )
            import cupy as cp
            
            # Generate large dataset
            np.random.seed(42)
            points = np.random.randn(5_000_000, 3).astype(np.float32)
            
            mempool = cp.get_default_memory_pool()
            
            # Process with chunking
            computer = GPUChunkedFeatureComputer(
                chunk_size=1_000_000,
                use_gpu=True
            )
            
            initial_used = mempool.used_bytes()
            normals = computer.compute_normals_chunked(points, k=10)
            peak_used = mempool.used_bytes()
            
            # Clean up
            del normals
            computer._free_gpu_memory()
            final_used = mempool.used_bytes()
            
            print(f"\nInitial VRAM: {initial_used / (1024**3):.2f} GB")
            print(f"Peak VRAM: {peak_used / (1024**3):.2f} GB")
            print(f"Final VRAM: {final_used / (1024**3):.2f} GB")
            
            # Should clean up properly
            assert final_used <= peak_used
            
        except ImportError:
            pytest.skip("GPU not available for memory test")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
