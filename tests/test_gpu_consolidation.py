"""
Tests for GPU consolidation (Phase 1 - November 2025)

Tests to ensure:
1. No duplicate GPUProfiler implementations
2. Backward compatibility for deprecated imports
3. Core GPU profiler has all expected functionality
4. Legacy modules properly deprecated
"""

import pytest
import warnings
import sys


class TestGPUProfilerConsolidation:
    """Test GPU profiler consolidation."""
    
    def test_no_duplicate_profiler_modules(self):
        """Ensure optimization.gpu_profiler is deprecated stub only."""
        # Import should trigger deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from ign_lidar.optimization import gpu_profiler
            
            # Should have deprecation warning
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
    
    def test_core_profiler_exists(self):
        """Core GPU profiler should be the canonical implementation."""
        from ign_lidar.core import gpu_profiler
        
        assert hasattr(gpu_profiler, 'GPUProfiler')
        assert hasattr(gpu_profiler, 'ProfileEntry')
        assert hasattr(gpu_profiler, 'ProfilingStats')
        assert hasattr(gpu_profiler, 'create_profiler')
    
    def test_backward_compatibility_imports(self):
        """Legacy imports should still work with warnings."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            # Should work but warn
            from ign_lidar.optimization.gpu_profiler import GPUProfiler
            from ign_lidar.optimization.gpu_profiler import GPUOperationMetrics
            from ign_lidar.optimization.gpu_profiler import ProfilerSession
            
            # Should be aliases to core module
            from ign_lidar.core.gpu_profiler import GPUProfiler as CoreGPUProfiler
            from ign_lidar.core.gpu_profiler import ProfileEntry as CoreProfileEntry
            
            assert GPUProfiler is CoreGPUProfiler
            assert GPUOperationMetrics is CoreProfileEntry
    
    def test_core_profiler_has_bottleneck_analysis(self):
        """Core profiler should have bottleneck analysis (migrated from optimization)."""
        from ign_lidar.core.gpu_profiler import create_profiler
        
        profiler = create_profiler(enabled=False)  # Disabled for testing
        
        # Should have get_bottleneck_analysis method
        assert hasattr(profiler, 'get_bottleneck_analysis')
        
        # Should return proper structure
        result = profiler.get_bottleneck_analysis()
        assert 'bottleneck' in result
        assert 'transfer_pct' in result
        assert 'compute_pct' in result
        assert 'recommendation' in result
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not hasattr(sys, 'modules') or 'cupy' not in sys.modules, 
                        reason="GPU not available")
    def test_profiler_cuda_events(self):
        """Core profiler should use CUDA events when available."""
        from ign_lidar.core.gpu_profiler import create_profiler, HAS_CUPY
        
        if not HAS_CUPY:
            pytest.skip("CuPy not available")
        
        profiler = create_profiler(enabled=True, use_cuda_events=True)
        
        # Should be enabled and using CUDA events
        assert profiler.enabled
        assert profiler.use_cuda_events


class TestLegacyModuleRemoval:
    """Test that legacy modules are properly removed."""
    
    def test_faiss_knn_removed(self):
        """faiss_knn.py should be removed (replaced by KNNEngine)."""
        with pytest.raises((ModuleNotFoundError, ImportError)):
            from ign_lidar.features.compute import faiss_knn
    
    def test_knn_engine_available(self):
        """KNNEngine should be available as replacement."""
        from ign_lidar.optimization import KNNEngine, knn_search
        
        assert KNNEngine is not None
        assert callable(knn_search)
    
    def test_gpu_module_renamed(self):
        """gpu.py should be renamed to ground_truth_classifier.py with deprecation stub."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Importing from old location should work but warn
            from ign_lidar.optimization.gpu import GPUGroundTruthClassifier as OldImport
            
            # Should have deprecation warning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("ground_truth_classifier" in str(x.message).lower() for x in deprecation_warnings)
        
        # New location should work without warning
        from ign_lidar.optimization.ground_truth_classifier import GPUGroundTruthClassifier as NewImport
        
        # Should be the same class
        assert OldImport is NewImport


class TestGPUMemoryReorganization:
    """Test GPU memory module reorganization (Phase 2.2)."""
    
    def test_gpu_cache_package_exists(self):
        """New gpu_cache package should exist."""
        from ign_lidar.optimization import gpu_cache
        
        assert hasattr(gpu_cache, 'GPUArrayCache')
        assert hasattr(gpu_cache, 'GPUMemoryPool')
        assert hasattr(gpu_cache, 'TransferOptimizer')
    
    def test_gpu_memory_backward_compatibility(self):
        """Legacy gpu_memory module should still work with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from ign_lidar.optimization.gpu_memory import GPUArrayCache
            from ign_lidar.optimization.gpu_memory import GPUMemoryPool
            
            # Should have deprecation warning
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "gpu_cache" in str(w[0].message)
    
    def test_gpu_cache_submodules(self):
        """gpu_cache should have arrays and transfer submodules."""
        from ign_lidar.optimization.gpu_cache import arrays
        from ign_lidar.optimization.gpu_cache import transfer
        
        # arrays module
        assert hasattr(arrays, 'GPUArrayCache')
        
        # transfer module
        assert hasattr(transfer, 'TransferOptimizer')
        assert hasattr(transfer, 'GPUMemoryPool')
        assert hasattr(transfer, 'estimate_gpu_memory_for_features')
        assert hasattr(transfer, 'optimize_chunk_size_for_vram')


class TestGPUManagerSingleton:
    """Test GPUManager singleton behavior."""
    
    def test_gpu_manager_is_singleton(self):
        """GPUManager should return same instance."""
        from ign_lidar.core.gpu import GPUManager
        
        gpu1 = GPUManager()
        gpu2 = GPUManager()
        
        assert gpu1 is gpu2  # Same instance
    
    def test_gpu_manager_has_profiler_access(self):
        """GPUManager should provide access to profiler (v3.2+)."""
        from ign_lidar.core.gpu import GPUManager
        
        gpu = GPUManager()
        
        # Should have profiler property (lazy-loaded)
        assert hasattr(gpu, 'profiler')


class TestDeprecationWarnings:
    """Test deprecation warnings are properly issued."""
    
    def test_optimization_gpu_profiler_warns(self):
        """Importing optimization.gpu_profiler should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should trigger warning
            import importlib
            importlib.reload(sys.modules.get('ign_lidar.optimization.gpu_profiler', 
                                             importlib.import_module('ign_lidar.optimization.gpu_profiler')))
            
            # Should have at least one deprecation warning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
    
    def test_get_profiler_legacy_warns(self):
        """Legacy get_profiler() should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from ign_lidar.optimization.gpu_profiler import get_profiler
            
            # Calling it should warn
            profiler = get_profiler(enable=False)
            
            # Should have deprecation warning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
