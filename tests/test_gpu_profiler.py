"""
Tests for GPU Profiler

Comprehensive test suite for the GPU performance profiling system.

Author: IGN LiDAR HD Development Team
Date: November 22, 2025
Version: 1.0.0
"""

import pytest
import numpy as np
import time
from typing import Dict, Any

from ign_lidar.core.gpu import GPUManager
from ign_lidar.core.gpu_profiler import GPUProfiler, ProfileEntry, ProfilingStats, create_profiler

# Check GPU availability
_gpu_manager = GPUManager()
GPU_AVAILABLE = _gpu_manager.gpu_available

if GPU_AVAILABLE:
    import cupy as cp


# =============================================================================
# Profiler Basic Functionality Tests
# =============================================================================

class TestGPUProfilerBasics:
    """Test basic profiling functionality."""
    
    def test_profiler_creation(self):
        """Test profiler can be created."""
        profiler = GPUProfiler()
        assert profiler is not None
        assert isinstance(profiler, GPUProfiler)
    
    def test_profiler_creation_disabled(self):
        """Test profiler can be created in disabled mode."""
        profiler = GPUProfiler(enabled=False)
        assert profiler is not None
        assert not profiler.enabled
    
    def test_factory_function(self):
        """Test factory function creates profiler correctly."""
        profiler = create_profiler(enabled=True, use_cuda_events=True, bottleneck_threshold=25.0)
        assert profiler is not None
        assert profiler.enabled == GPU_AVAILABLE  # Only enabled if GPU available
        assert profiler.use_cuda_events == GPU_AVAILABLE
    
    def test_profiler_reset(self):
        """Test profiler reset clears entries."""
        profiler = GPUProfiler()
        
        # Add dummy entry
        entry = ProfileEntry(
            operation_name="test",
            elapsed_ms=100.0,
            mem_allocated_mb=10.0,
            mem_freed_mb=0.0,
            start_time=time.time(),
            end_time=time.time()
        )
        profiler.entries.append(entry)
        
        assert len(profiler.entries) == 1
        
        # Reset
        profiler.reset()
        assert len(profiler.entries) == 0


# =============================================================================
# Profiling Context Manager Tests
# =============================================================================

class TestProfilingContextManager:
    """Test profiling context manager."""
    
    def test_profile_context_disabled(self):
        """Test profiling context when disabled."""
        profiler = GPUProfiler(enabled=False)
        
        with profiler.profile('test_operation'):
            # Do some work
            time.sleep(0.01)
        
        # No entries should be recorded
        assert len(profiler.entries) == 0
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_profile_context_enabled(self):
        """Test profiling context when enabled."""
        profiler = GPUProfiler(enabled=True)
        
        with profiler.profile('test_operation'):
            # Do some GPU work
            arr = cp.random.rand(1000, 3)
            _ = cp.sum(arr)
        
        # Entry should be recorded
        assert len(profiler.entries) == 1
        entry = profiler.entries[0]
        assert entry.operation_name == 'test_operation'
        assert entry.elapsed_ms > 0
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_profile_multiple_operations(self):
        """Test profiling multiple operations."""
        profiler = GPUProfiler(enabled=True)
        
        with profiler.profile('operation_1'):
            arr1 = cp.random.rand(100, 3)
        
        with profiler.profile('operation_2'):
            arr2 = cp.random.rand(200, 3)
        
        assert len(profiler.entries) == 2
        assert profiler.entries[0].operation_name == 'operation_1'
        assert profiler.entries[1].operation_name == 'operation_2'
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_profile_with_transfer(self):
        """Test profiling with transfer information."""
        profiler = GPUProfiler(enabled=True)
        
        arr_cpu = np.random.rand(1000, 3)
        size_mb = arr_cpu.nbytes / (1024**2)
        
        with profiler.profile('upload_points', transfer='upload', size_mb=size_mb):
            arr_gpu = cp.asarray(arr_cpu)
        
        assert len(profiler.entries) == 1
        entry = profiler.entries[0]
        assert entry.transfer_type == 'upload'
        assert entry.transfer_size_mb == size_mb


# =============================================================================
# Statistics and Reporting Tests
# =============================================================================

class TestProfilingStatistics:
    """Test profiling statistics generation."""
    
    def test_empty_stats(self):
        """Test statistics for empty profiler."""
        profiler = GPUProfiler()
        stats = profiler.get_stats()
        
        assert isinstance(stats, ProfilingStats)
        assert stats.total_time_ms == 0.0
        assert stats.num_operations == 0
        assert len(stats.bottlenecks) == 0
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_basic_stats(self):
        """Test basic statistics calculation."""
        profiler = GPUProfiler(enabled=True)
        
        # Profile some operations
        with profiler.profile('op1'):
            _ = cp.random.rand(100, 3)
        
        with profiler.profile('op2'):
            _ = cp.random.rand(200, 3)
        
        stats = profiler.get_stats()
        
        assert stats.num_operations == 2
        assert stats.total_time_ms > 0
        assert stats.avg_time_ms > 0
        assert stats.max_time_ms >= stats.min_time_ms
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_transfer_stats(self):
        """Test transfer statistics."""
        profiler = GPUProfiler(enabled=True)
        
        arr_cpu = np.random.rand(1000, 3)
        size_mb = arr_cpu.nbytes / (1024**2)
        
        # Upload
        with profiler.profile('upload', transfer='upload', size_mb=size_mb):
            arr_gpu = cp.asarray(arr_cpu)
        
        # Download
        with profiler.profile('download', transfer='download', size_mb=size_mb):
            _ = cp.asnumpy(arr_gpu)
        
        stats = profiler.get_stats()
        
        assert stats.upload_count == 1
        assert stats.download_count == 1
        assert stats.upload_mb > 0
        assert stats.download_mb > 0
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_bottleneck_detection(self):
        """Test bottleneck detection."""
        profiler = GPUProfiler(enabled=True, bottleneck_threshold_pct=30.0)
        
        # Create operations with different durations
        with profiler.profile('fast_op'):
            _ = cp.random.rand(10, 3)
        
        with profiler.profile('slow_op'):
            # Simulate slow operation
            arr = cp.random.rand(10000, 10000)
            _ = cp.sum(arr)
        
        with profiler.profile('fast_op'):
            _ = cp.random.rand(10, 3)
        
        stats = profiler.get_stats()
        
        # slow_op should be detected as bottleneck
        assert len(stats.bottlenecks) > 0
        assert stats.bottlenecks[0]['operation'] == 'slow_op'
        assert stats.bottlenecks[0]['percentage'] > 30.0
    
    def test_operation_summary(self):
        """Test operation summary generation."""
        profiler = GPUProfiler()
        
        # Add dummy entries
        for i in range(3):
            entry = ProfileEntry(
                operation_name="op1",
                elapsed_ms=100.0 + i * 10,
                mem_allocated_mb=10.0,
                mem_freed_mb=0.0,
                start_time=time.time(),
                end_time=time.time()
            )
            profiler.entries.append(entry)
        
        for i in range(2):
            entry = ProfileEntry(
                operation_name="op2",
                elapsed_ms=50.0 + i * 5,
                mem_allocated_mb=5.0,
                mem_freed_mb=0.0,
                start_time=time.time(),
                end_time=time.time()
            )
            profiler.entries.append(entry)
        
        summary = profiler.get_operation_summary()
        
        assert 'op1' in summary
        assert 'op2' in summary
        assert summary['op1']['count'] == 3
        assert summary['op2']['count'] == 2
        assert summary['op1']['total_time_ms'] > summary['op2']['total_time_ms']


# =============================================================================
# GPU Manager Integration Tests
# =============================================================================

class TestGPUManagerIntegration:
    """Test profiler integration with GPUManager."""
    
    def test_profiler_property_exists(self):
        """Test profiler property exists on GPUManager."""
        gpu = GPUManager()
        assert hasattr(gpu, 'profiler')
    
    def test_profiler_lazy_loading(self):
        """Test profiler is lazy-loaded."""
        gpu = GPUManager()
        
        # Reset lazy-loaded components
        gpu._profiler = None
        
        # Access profiler
        profiler = gpu.profiler
        
        assert profiler is not None
        assert isinstance(profiler, GPUProfiler)
    
    def test_profiler_singleton_caching(self):
        """Test profiler is cached after first access."""
        gpu = GPUManager()
        
        profiler1 = gpu.profiler
        profiler2 = gpu.profiler
        
        # Should be same instance
        assert profiler1 is profiler2
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_profiler_usage_through_manager(self):
        """Test profiler can be used through GPUManager."""
        gpu = GPUManager()
        
        with gpu.profiler.profile('test_through_manager'):
            arr = cp.random.rand(100, 3)
            _ = cp.sum(arr)
        
        assert len(gpu.profiler.entries) > 0
    
    def test_cleanup_resets_profiler(self):
        """Test GPUManager.cleanup() resets profiler."""
        gpu = GPUManager()
        
        # Add entry
        entry = ProfileEntry(
            operation_name="test",
            elapsed_ms=100.0,
            mem_allocated_mb=10.0,
            mem_freed_mb=0.0,
            start_time=time.time(),
            end_time=time.time()
        )
        gpu.profiler.entries.append(entry)
        
        assert len(gpu.profiler.entries) == 1
        
        # Cleanup
        gpu.cleanup()
        
        # Profiler should be reset
        assert len(gpu.profiler.entries) == 0


# =============================================================================
# Performance Report Tests
# =============================================================================

class TestPerformanceReporting:
    """Test performance reporting functionality."""
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_print_report_basic(self, caplog):
        """Test basic report printing."""
        profiler = GPUProfiler(enabled=True)
        
        with profiler.profile('test_op'):
            _ = cp.random.rand(100, 3)
        
        # Should not raise exception
        profiler.print_report()
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_print_report_detailed(self, caplog):
        """Test detailed report printing."""
        profiler = GPUProfiler(enabled=True)
        
        with profiler.profile('op1'):
            _ = cp.random.rand(100, 3)
        
        with profiler.profile('op2'):
            _ = cp.random.rand(200, 3)
        
        # Should not raise exception
        profiler.print_report(detailed=True)
    
    def test_print_report_empty(self, caplog):
        """Test report printing with no entries."""
        profiler = GPUProfiler(enabled=True)
        
        # Should not raise exception
        profiler.print_report()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_profiler_without_cupy(self):
        """Test profiler behavior when CuPy not available."""
        # This should not raise an error
        profiler = GPUProfiler(enabled=True)
        
        # On systems without GPU, profiler should be disabled
        if not GPU_AVAILABLE:
            assert not profiler.enabled
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_nested_profiling(self):
        """Test nested profiling contexts."""
        profiler = GPUProfiler(enabled=True)
        
        with profiler.profile('outer'):
            _ = cp.random.rand(100, 3)
            
            with profiler.profile('inner'):
                _ = cp.random.rand(50, 3)
        
        # Both should be recorded
        assert len(profiler.entries) == 2
        assert any(e.operation_name == 'outer' for e in profiler.entries)
        assert any(e.operation_name == 'inner' for e in profiler.entries)
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_profiling_with_exception(self):
        """Test profiling when exception occurs."""
        profiler = GPUProfiler(enabled=True)
        
        try:
            with profiler.profile('op_with_exception'):
                _ = cp.random.rand(100, 3)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Entry should still be recorded
        assert len(profiler.entries) == 1


# =============================================================================
# Integration with Real GPU Operations
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestRealGPUOperations:
    """Test profiler with real GPU operations."""
    
    def test_profile_matrix_operations(self):
        """Test profiling matrix operations."""
        gpu = GPUManager()
        gpu.profiler.reset()
        
        size = 1000
        
        with gpu.profiler.profile('matrix_creation'):
            A = cp.random.rand(size, size)
            B = cp.random.rand(size, size)
        
        with gpu.profiler.profile('matrix_multiply'):
            C = cp.dot(A, B)
        
        with gpu.profiler.profile('matrix_sum'):
            result = cp.sum(C)
        
        stats = gpu.profiler.get_stats()
        
        assert stats.num_operations == 3
        assert stats.total_time_ms > 0
        
        # Matrix multiply should typically be slowest
        summary = gpu.profiler.get_operation_summary()
        assert 'matrix_multiply' in summary
    
    def test_profile_memory_transfers(self):
        """Test profiling CPU↔GPU transfers."""
        gpu = GPUManager()
        gpu.profiler.reset()
        
        arr_cpu = np.random.rand(10000, 3).astype(np.float32)
        size_mb = arr_cpu.nbytes / (1024**2)
        
        # Upload
        with gpu.profiler.profile('cpu_to_gpu', transfer='upload', size_mb=size_mb):
            arr_gpu = cp.asarray(arr_cpu)
        
        # Compute
        with gpu.profiler.profile('gpu_compute'):
            result_gpu = cp.sum(arr_gpu, axis=1)
        
        # Download
        with gpu.profiler.profile('gpu_to_cpu', transfer='download', size_mb=size_mb):
            result_cpu = cp.asnumpy(result_gpu)
        
        stats = gpu.profiler.get_stats()
        
        assert stats.upload_count == 1
        assert stats.download_count == 1
        assert stats.upload_mb > 0
        assert stats.download_mb > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
