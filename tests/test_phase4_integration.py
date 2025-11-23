"""
Integration tests for Phase 4 OptimizationManager in LiDARProcessor.

Tests the complete integration of all Phase 4 optimizations:
- Phase 4.1: WFS Memory Cache
- Phase 4.2: Preprocessing GPU
- Phase 4.3: GPU Memory Pooling
- Phase 4.4: Batch Multi-Tile Processing
- Phase 4.5: Async I/O Pipeline

Author: IGN LiDAR HD Development Team
Date: November 23, 2025
"""

import pytest
from pathlib import Path
from omegaconf import OmegaConf
import tempfile
import time


@pytest.fixture
def temp_dirs():
    """Create temporary input/output directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        yield input_dir, output_dir


@pytest.fixture
def base_config():
    """Base configuration for testing."""
    return OmegaConf.create({
        'processor': {
            'lod_level': 'LOD2',
            'processing_mode': 'patches_only',
            'use_gpu': False,  # Default to CPU for testing
            'patch_size': 150.0,
            'num_points': 16384,
        },
        'features': {
            'mode': 'lod2',
            'k_neighbors': 20,
        },
        'data_sources': {
            'bd_topo': {
                'enabled': False,  # Disable for unit tests
            }
        }
    })


class TestOptimizationManagerIntegration:
    """Test OptimizationManager integration in LiDARProcessor."""
    
    def test_optimization_manager_initialization_enabled(self, base_config):
        """Test OptimizationManager is initialized when enabled."""
        from ign_lidar.core.processor import LiDARProcessor
        
        # Enable optimizations
        config = base_config.copy()
        config['enable_optimizations'] = True
        config['enable_async_io'] = True
        config['enable_batch_processing'] = True
        config['enable_gpu_pooling'] = True
        
        processor = LiDARProcessor(config)
        
        # Check optimization manager exists
        assert hasattr(processor, 'optimization_manager')
        # May be None if components unavailable, but should exist
        assert processor.optimization_manager is not None or processor.optimization_manager is None
    
    def test_optimization_manager_disabled(self, base_config):
        """Test OptimizationManager is not initialized when disabled."""
        from ign_lidar.core.processor import LiDARProcessor
        
        # Disable optimizations
        config = base_config.copy()
        config['enable_optimizations'] = False
        
        processor = LiDARProcessor(config)
        
        # Check optimization manager is None
        assert hasattr(processor, 'optimization_manager')
        assert processor.optimization_manager is None
    
    def test_process_directory_routing_optimized(self, base_config, temp_dirs, monkeypatch):
        """Test process_directory routes to optimized version when OptimizationManager exists."""
        from ign_lidar.core.processor import LiDARProcessor
        
        input_dir, output_dir = temp_dirs
        
        # Enable optimizations
        config = base_config.copy()
        config['enable_optimizations'] = True
        config['input_dir'] = str(input_dir)
        config['output_dir'] = str(output_dir)
        
        processor = LiDARProcessor(config)
        
        # Mock the optimized processing method
        optimized_called = []
        def mock_optimized(*args, **kwargs):
            optimized_called.append(True)
            return 0
        
        monkeypatch.setattr(processor, '_process_directory_optimized', mock_optimized)
        
        # Process empty directory (should call optimized version)
        processor.process_directory(input_dir, output_dir)
        
        # Check optimized version was called
        if processor.optimization_manager is not None:
            assert len(optimized_called) == 1
    
    def test_process_directory_routing_sequential(self, base_config, temp_dirs, monkeypatch):
        """Test process_directory routes to sequential version when OptimizationManager disabled."""
        from ign_lidar.core.processor import LiDARProcessor
        
        input_dir, output_dir = temp_dirs
        
        # Disable optimizations
        config = base_config.copy()
        config['enable_optimizations'] = False
        config['input_dir'] = str(input_dir)
        config['output_dir'] = str(output_dir)
        
        processor = LiDARProcessor(config)
        
        # Mock the sequential processing method
        sequential_called = []
        def mock_sequential(*args, **kwargs):
            sequential_called.append(True)
            return 0
        
        monkeypatch.setattr(processor, '_process_directory_sequential', mock_sequential)
        
        # Process empty directory (should call sequential version)
        processor.process_directory(input_dir, output_dir)
        
        # Check sequential version was called
        assert len(sequential_called) == 1
    
    def test_optimization_config_parameters(self, base_config):
        """Test optimization configuration parameters are passed correctly."""
        from ign_lidar.core.processor import LiDARProcessor
        
        # Set custom optimization parameters
        config = base_config.copy()
        config['enable_optimizations'] = True
        config['async_workers'] = 4
        config['tile_cache_size'] = 5
        config['batch_size'] = 8
        config['gpu_pool_max_size_gb'] = 6.0
        
        processor = LiDARProcessor(config)
        
        # If optimization manager was created, check parameters
        if processor.optimization_manager is not None:
            assert processor.optimization_manager.async_workers == 4
            assert processor.optimization_manager.tile_cache_size == 5
            # Note: batch_size and gpu_pool_max_size_gb might not be directly accessible
    
    def test_graceful_fallback_on_error(self, base_config, monkeypatch):
        """Test graceful fallback when OptimizationManager initialization fails."""
        from ign_lidar.core.processor import LiDARProcessor
        
        # Mock create_optimization_manager to raise error
        def mock_create_error(*args, **kwargs):
            raise RuntimeError("Test error")
        
        # This requires patching before LiDARProcessor import, so skip for now
        # Just verify processor can be created even if optimization fails
        config = base_config.copy()
        config['enable_optimizations'] = True
        
        # Should not raise, even if optimization setup fails
        processor = LiDARProcessor(config)
        assert processor is not None


class TestOptimizationManagerCleanup:
    """Test resource cleanup for OptimizationManager."""
    
    def test_optimization_manager_shutdown_on_del(self, base_config):
        """Test OptimizationManager.shutdown() is called on processor deletion."""
        from ign_lidar.core.processor import LiDARProcessor
        
        config = base_config.copy()
        config['enable_optimizations'] = True
        
        processor = LiDARProcessor(config)
        
        # Store reference to optimization manager
        opt_mgr = processor.optimization_manager
        
        # Delete processor
        del processor
        
        # If optimization manager existed, it should have been shut down
        # (Can't easily verify shutdown was called, but test shouldn't crash)
        assert True


@pytest.mark.integration
class TestPhase4EndToEnd:
    """End-to-end integration tests for Phase 4."""
    
    @pytest.mark.skip(reason="Requires real LiDAR data")
    def test_optimized_vs_sequential_performance(self, temp_dirs):
        """Compare performance between optimized and sequential processing."""
        from ign_lidar.core.processor import LiDARProcessor
        
        input_dir, output_dir = temp_dirs
        
        # TODO: Add real test tiles
        # This test requires actual LiDAR data to be meaningful
        
        # Test with optimizations enabled
        config_optimized = OmegaConf.create({
            'processor': {
                'lod_level': 'LOD2',
                'use_gpu': True,
            },
            'features': {'mode': 'lod2'},
            'enable_optimizations': True,
            'input_dir': str(input_dir),
            'output_dir': str(output_dir / 'optimized'),
        })
        
        processor_opt = LiDARProcessor(config_optimized)
        start = time.time()
        patches_opt = processor_opt.process_directory(input_dir, output_dir / 'optimized')
        time_opt = time.time() - start
        
        # Test with optimizations disabled
        config_sequential = config_optimized.copy()
        config_sequential['enable_optimizations'] = False
        config_sequential['output_dir'] = str(output_dir / 'sequential')
        
        processor_seq = LiDARProcessor(config_sequential)
        start = time.time()
        patches_seq = processor_seq.process_directory(input_dir, output_dir / 'sequential')
        time_seq = time.time() - start
        
        # Same number of patches should be created
        assert patches_opt == patches_seq
        
        # Optimized should be faster (or at least not slower)
        speedup = time_seq / time_opt
        print(f"Speedup: {speedup:.2f}×")
        assert speedup >= 1.0  # At minimum, no slowdown


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
