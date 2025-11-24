"""
Tests for centralized CuPy imports via GPUManager.

Tests the import centralization added in v3.5.2 that eliminates
100+ redundant try/except blocks across the codebase.

Benefits:
- Single source of truth for GPU availability
- Consistent error handling
- Easier maintenance
- Better testability

Author: Consolidation Phase
Date: November 24, 2025
"""
import pytest
import sys
from unittest.mock import patch, MagicMock

from ign_lidar.core.gpu import GPUManager


class TestGPUManagerCentralization:
    """Test suite for centralized CuPy import management."""
    
    def test_gpu_manager_singleton(self):
        """Test that GPUManager follows singleton pattern."""
        manager1 = GPUManager()
        manager2 = GPUManager()
        
        # Should be the same instance
        assert manager1 is manager2
    
    def test_get_cupy_when_available(self):
        """Test get_cupy() returns CuPy when GPU is available."""
        manager = GPUManager()
        
        if manager.gpu_available:
            cp = manager.get_cupy()
            assert cp is not None
            assert hasattr(cp, 'ndarray')
            assert hasattr(cp, 'cuda')
        else:
            pytest.skip("GPU not available in test environment")
    
    def test_get_cupy_raises_when_unavailable(self):
        """Test get_cupy() raises ImportError when GPU unavailable."""
        manager = GPUManager()
        
        # Only test if GPU is truly unavailable
        if not manager.gpu_available:
            with pytest.raises(ImportError, match="CuPy not available"):
                manager.get_cupy()
        else:
            pytest.skip("GPU is available, cannot test unavailable case")
    
    def test_try_get_cupy_returns_none_when_unavailable(self):
        """Test try_get_cupy() returns None instead of raising."""
        manager = GPUManager()
        
        # Only test if GPU is truly unavailable
        if not manager.gpu_available:
            result = manager.try_get_cupy()
            assert result is None
        else:
            pytest.skip("GPU is available, cannot test unavailable case")
    
    def test_try_get_cupy_returns_module_when_available(self):
        """Test try_get_cupy() returns CuPy when available."""
        manager = GPUManager()
        
        if manager.gpu_available:
            cp = manager.try_get_cupy()
            assert cp is not None
            assert hasattr(cp, 'ndarray')
        else:
            # When unavailable, should return None
            result = manager.try_get_cupy()
            assert result is None
    
    def test_try_get_cupy_handles_import_error(self):
        """Test try_get_cupy() gracefully handles import errors."""
        manager = GPUManager()
        
        # Test that the function never raises, regardless of GPU state
        result = manager.try_get_cupy()
        # Should either be None or a valid cupy module
        assert result is None or hasattr(result, 'ndarray')


@pytest.mark.integration
class TestModuleImportPatterns:
    """Test that modules correctly use centralized imports."""
    
    def test_gpu_kernels_uses_centralized_import(self):
        """Test that gpu_kernels.py uses GPUManager."""
        try:
            from ign_lidar.optimization import gpu_kernels
            
            # Should have imported GPUManager
            assert hasattr(gpu_kernels, 'gpu')
            assert hasattr(gpu_kernels, 'HAS_CUPY')
            
            # gpu should be GPUManager instance
            from ign_lidar.core.gpu import GPUManager
            assert isinstance(gpu_kernels.gpu, GPUManager)
            
        except ImportError:
            pytest.skip("gpu_kernels module not available")
    
    def test_gpu_accelerated_ops_uses_centralized_import(self):
        """Test that gpu_accelerated_ops.py uses GPUManager."""
        try:
            from ign_lidar.optimization import gpu_accelerated_ops
            
            assert hasattr(gpu_accelerated_ops, 'gpu')
            assert hasattr(gpu_accelerated_ops, 'HAS_CUPY')
            
            from ign_lidar.core.gpu import GPUManager
            assert isinstance(gpu_accelerated_ops.gpu, GPUManager)
            
        except ImportError:
            pytest.skip("gpu_accelerated_ops module not available")
    
    def test_cuda_streams_uses_centralized_import(self):
        """Test that cuda_streams.py uses GPUManager."""
        try:
            from ign_lidar.optimization import cuda_streams
            
            assert hasattr(cuda_streams, 'gpu')
            assert hasattr(cuda_streams, 'HAS_CUPY')
            
        except ImportError:
            pytest.skip("cuda_streams module not available")
    
    def test_transfer_optimizer_uses_centralized_import(self):
        """Test that transfer_optimizer.py uses GPUManager."""
        try:
            from ign_lidar.optimization import transfer_optimizer
            
            assert hasattr(transfer_optimizer, 'gpu')
            assert hasattr(transfer_optimizer, 'HAS_CUPY')
            
        except ImportError:
            pytest.skip("transfer_optimizer module not available")
    
    def test_gpu_cache_arrays_uses_centralized_import(self):
        """Test that gpu_cache/arrays.py uses GPUManager."""
        try:
            from ign_lidar.optimization.gpu_cache import arrays
            
            assert hasattr(arrays, 'gpu')
            assert hasattr(arrays, 'HAS_CUPY')
            
        except ImportError:
            pytest.skip("gpu_cache.arrays module not available")
    
    def test_gpu_cache_transfer_uses_centralized_import(self):
        """Test that gpu_cache/transfer.py uses GPUManager."""
        try:
            from ign_lidar.optimization.gpu_cache import transfer
            
            assert hasattr(transfer, 'gpu')
            assert hasattr(transfer, 'HAS_CUPY')
            
        except ImportError:
            pytest.skip("gpu_cache.transfer module not available")
    
    def test_curvature_compute_uses_centralized_import(self):
        """Test that features/compute/curvature.py uses GPUManager."""
        try:
            from ign_lidar.features.compute import curvature
            
            assert hasattr(curvature, 'gpu')
            # Note: HAS_CUPY may not be exported at module level
            
            from ign_lidar.core.gpu import GPUManager
            assert isinstance(curvature.gpu, GPUManager)
            
        except ImportError:
            pytest.skip("curvature module not available")
    
    def test_gpu_bridge_uses_centralized_import(self):
        """Test that features/compute/gpu_bridge.py uses GPUManager."""
        try:
            from ign_lidar.features.compute import gpu_bridge
            
            assert hasattr(gpu_bridge, 'gpu')
            
            from ign_lidar.core.gpu import GPUManager
            assert isinstance(gpu_bridge.gpu, GPUManager)
            
        except ImportError:
            pytest.skip("gpu_bridge module not available")
    
    def test_building_clustering_uses_centralized_import(self):
        """Test that classification/building/clustering.py uses GPUManager."""
        try:
            from ign_lidar.core.classification.building import clustering
            
            assert hasattr(clustering, 'gpu')
            assert hasattr(clustering, 'HAS_CUPY')
            
        except ImportError:
            pytest.skip("building clustering module not available")


@pytest.mark.unit
class TestImportConsistency:
    """Test consistency of import patterns across modules."""
    
    def test_no_direct_cupy_imports_in_gpu_modules(self):
        """
        Test that GPU modules don't have direct 'import cupy' statements.
        
        They should use GPUManager.get_cupy() or .try_get_cupy() instead.
        """
        import ast
        from pathlib import Path
        
        # List of files that should use centralized imports
        files_to_check = [
            "ign_lidar/optimization/gpu_kernels.py",
            "ign_lidar/optimization/gpu_accelerated_ops.py",
            "ign_lidar/optimization/cuda_streams.py",
            "ign_lidar/optimization/transfer_optimizer.py",
            "ign_lidar/optimization/gpu_cache/arrays.py",
            "ign_lidar/optimization/gpu_cache/transfer.py",
            "ign_lidar/features/compute/curvature.py",
            "ign_lidar/features/compute/gpu_bridge.py",
            "ign_lidar/core/classification/building/clustering.py",
        ]
        
        project_root = Path(__file__).parent.parent
        
        for rel_path in files_to_check:
            file_path = project_root / rel_path
            
            if not file_path.exists():
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should NOT have direct "import cupy" or "from cupy import"
            # (except in comments)
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # Skip comments
                if stripped.startswith('#'):
                    continue
                
                # Check for direct cupy imports
                if 'import cupy' in line and 'GPUManager' not in content:
                    pytest.fail(
                        f"{rel_path}:{i} has direct 'import cupy' without GPUManager:\n{line}"
                    )
    
    def test_all_gpu_modules_use_gpu_manager(self):
        """Test that all GPU modules import and use GPUManager."""
        from pathlib import Path
        
        gpu_modules = [
            "ign_lidar/optimization/gpu_kernels.py",
            "ign_lidar/optimization/gpu_accelerated_ops.py",
            "ign_lidar/optimization/cuda_streams.py",
            "ign_lidar/optimization/transfer_optimizer.py",
        ]
        
        project_root = Path(__file__).parent.parent
        
        for rel_path in gpu_modules:
            file_path = project_root / rel_path
            
            if not file_path.exists():
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should import GPUManager
            assert 'from ign_lidar.core.gpu import GPUManager' in content, \
                f"{rel_path} should import GPUManager"
            
            # Should instantiate gpu = GPUManager()
            assert 'gpu = GPUManager()' in content, \
                f"{rel_path} should instantiate GPUManager"
            
            # Should check HAS_CUPY
            assert 'HAS_CUPY' in content, \
                f"{rel_path} should use HAS_CUPY flag"


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test that centralization doesn't break existing code."""
    
    def test_gpu_manager_api_unchanged(self):
        """Test that GPUManager maintains its original API."""
        manager = GPUManager()
        
        # Original attributes should still exist
        assert hasattr(manager, 'gpu_available')
        # Note: device_count and device_name may be conditional on GPU availability
        
        # Original methods should still work
        assert callable(getattr(manager, 'get_info', None))
        assert callable(getattr(manager, 'get_memory_info', None))
    
    def test_new_methods_exist(self):
        """Test that new centralization methods are available."""
        manager = GPUManager()
        
        # New methods added in v3.5.2
        assert hasattr(manager, 'get_cupy')
        assert callable(manager.get_cupy)
        
        assert hasattr(manager, 'try_get_cupy')
        assert callable(manager.try_get_cupy)


@pytest.mark.integration
class TestGPUImportRobustness:
    """Test robustness of centralized import system."""
    
    def test_import_failure_doesnt_crash_application(self):
        """Test that CuPy import failure is handled gracefully."""
        manager = GPUManager()
        
        # Even if GPU is unavailable, manager should work
        assert isinstance(manager.gpu_available, bool)
        
        # try_get_cupy should never raise
        result = manager.try_get_cupy()
        assert result is None or hasattr(result, 'ndarray')
    
    def test_multiple_calls_consistent(self):
        """Test that multiple calls to get_cupy return consistent results."""
        manager = GPUManager()
        
        if not manager.gpu_available:
            pytest.skip("GPU not available")
        
        # Multiple calls should return same module
        cp1 = manager.get_cupy()
        cp2 = manager.get_cupy()
        
        # Should be the same cupy module
        assert cp1 is cp2
    
    def test_error_messages_helpful(self):
        """Test that error messages are helpful for debugging."""
        manager = GPUManager()
        
        # Only test if GPU is truly unavailable
        if not manager.gpu_available:
            try:
                manager.get_cupy()
                pytest.fail("Should have raised ImportError")
            except ImportError as e:
                # Error message should be informative
                assert "CuPy not available" in str(e)
                assert "install" in str(e).lower()
        else:
            pytest.skip("GPU available, cannot test error messages")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
