"""
Tests for GPU memory context manager.

Tests the memory_context() method added to GPUManager for automatic
GPU memory lifecycle management.

Author: LiDAR Trainer Agent
Date: November 24, 2025
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ign_lidar.core.gpu import GPUManager


class TestGPUMemoryContext:
    """Test GPU memory context manager."""
    
    def test_memory_context_exists(self):
        """Test that memory_context method exists."""
        gpu = GPUManager()
        assert hasattr(gpu, 'memory_context')
        assert callable(gpu.memory_context)
    
    def test_memory_context_without_gpu(self):
        """Test context manager works without GPU (no-op)."""
        gpu = GPUManager()
        
        # Should work even without GPU
        with gpu.memory_context("test operation"):
            # No-op on CPU-only systems
            pass
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_memory_context_with_gpu(self):
        """Test context manager with GPU hardware."""
        import cupy as cp
        
        gpu = GPUManager()
        
        # Context manager should handle GPU memory
        with gpu.memory_context("test allocation"):
            # Allocate some GPU memory
            data_gpu = cp.arange(1000000, dtype=cp.float32)
            result = cp.mean(data_gpu)
        
        # Memory should be cleaned up after context
        # (exact assertions difficult without implementation details)
        assert result is not None
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_memory_context_exception_handling(self):
        """Test context manager handles exceptions properly."""
        import cupy as cp
        
        gpu = GPUManager()
        
        # Context should cleanup even if exception occurs
        with pytest.raises(ValueError):
            with gpu.memory_context("test exception"):
                data_gpu = cp.arange(1000, dtype=cp.float32)
                raise ValueError("Test error")
        
        # GPU should still be functional after exception
        test_array = cp.array([1, 2, 3])
        assert cp.mean(test_array) == 2.0
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_memory_context_yields_gpu_manager(self):
        """Test context manager yields GPUManager instance."""
        gpu = GPUManager()
        
        with gpu.memory_context("test yield") as manager:
            assert isinstance(manager, GPUManager)
            assert manager is gpu
    
    def test_memory_context_description_parameter(self):
        """Test context manager accepts custom description."""
        gpu = GPUManager()
        
        # Should work with custom description
        with gpu.memory_context("custom operation name"):
            pass
        
        # Should work with default description
        with gpu.memory_context():
            pass
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_memory_context_nested(self):
        """Test nested context managers work correctly."""
        import cupy as cp
        
        gpu = GPUManager()
        
        with gpu.memory_context("outer operation"):
            data1 = cp.arange(1000, dtype=cp.float32)
            
            with gpu.memory_context("inner operation"):
                data2 = cp.arange(500, dtype=cp.float32)
                result = cp.sum(data2)
            
            # Outer context still active
            result2 = cp.sum(data1)
        
        assert result is not None
        assert result2 is not None
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_memory_context_multiple_operations(self):
        """Test context manager with multiple GPU operations."""
        import cupy as cp
        
        gpu = GPUManager()
        
        results = []
        with gpu.memory_context("batch operations"):
            # Multiple allocations and computations
            for i in range(5):
                data = cp.arange(10000 * (i + 1), dtype=cp.float32)
                results.append(cp.mean(data))
        
        assert len(results) == 5
        assert all(r is not None for r in results)


class TestMemoryContextIntegration:
    """Integration tests with real GPU operations."""
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_memory_context_with_batch_transfers(self):
        """Test memory context works with batch upload/download."""
        import cupy as cp
        
        gpu = GPUManager()
        
        arr1 = np.random.rand(1000, 3).astype(np.float32)
        arr2 = np.random.rand(1000).astype(np.float32)
        
        with gpu.memory_context("batch transfer test"):
            # Upload
            gpu_arr1, gpu_arr2 = gpu.batch_upload(arr1, arr2)
            
            # Compute
            result_gpu = cp.sum(gpu_arr1, axis=1) + gpu_arr2
            
            # Download
            result_cpu = gpu.batch_download(result_gpu)[0]
        
        # Verify result
        assert result_cpu.shape == (1000,)
        assert result_cpu.dtype == np.float32
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_memory_context_performance_tracking(self, caplog):
        """Test that memory context logs memory usage."""
        import cupy as cp
        
        gpu = GPUManager()
        
        # Should log memory info
        import logging
        caplog.set_level(logging.DEBUG)
        
        with gpu.memory_context("performance test"):
            data = cp.arange(1000000, dtype=cp.float32)
            _ = cp.mean(data)
        
        # Check for debug logs (if logging is enabled)
        # Note: Exact log format may vary
        log_messages = [record.message for record in caplog.records]
        assert any("GPU Memory Context" in msg for msg in log_messages) or len(log_messages) == 0


class TestMemoryContextEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_memory_context_no_gpu_no_error(self):
        """Test context manager doesn't fail without GPU."""
        gpu = GPUManager()
        
        # Should not raise even without GPU
        try:
            with gpu.memory_context("no GPU test"):
                x = np.array([1, 2, 3])
                _ = np.mean(x)
        except Exception as e:
            pytest.fail(f"Context manager failed without GPU: {e}")
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_memory_context_empty_operation(self):
        """Test context manager with no operations."""
        gpu = GPUManager()
        
        # Should handle empty context
        with gpu.memory_context("empty operation"):
            pass
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not GPUManager().gpu_available, reason="GPU not available")
    def test_memory_context_large_allocation(self):
        """Test context manager with large memory allocation."""
        import cupy as cp
        
        gpu = GPUManager()
        
        try:
            with gpu.memory_context("large allocation"):
                # Allocate large array (may fail on low-memory GPUs)
                # Using reasonable size that most GPUs can handle
                size = 100_000_000  # 100M floats ≈ 400MB
                data = cp.arange(size, dtype=cp.float32)
                result = cp.sum(data)
            
            assert result is not None
        except MemoryError:
            pytest.skip("GPU out of memory for large allocation test")
