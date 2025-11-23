"""
Tests for Audit Fixes (November 23, 2025)

Validates all changes made during codebase audit:
1. Removed gpu_memory_BACKUP.py
2. Renamed PerformanceMonitor classes
3. Added GPU cache metrics
4. Added TransferOptimizer integration
"""

import pytest
from pathlib import Path


def test_backup_file_removed():
    """Test that gpu_memory_BACKUP.py was removed."""
    backup_path = Path("ign_lidar/optimization/gpu_memory_BACKUP.py")
    assert not backup_path.exists(), "BACKUP file should be removed"


def test_processor_performance_monitor_import():
    """Test that ProcessorPerformanceMonitor can be imported."""
    from ign_lidar.core import ProcessorPerformanceMonitor
    assert ProcessorPerformanceMonitor is not None


def test_ground_truth_performance_monitor_import():
    """Test that GroundTruthPerformanceMonitor can be imported."""
    from ign_lidar.optimization import GroundTruthPerformanceMonitor
    assert GroundTruthPerformanceMonitor is not None


def test_backward_compatibility_alias():
    """Test backward compatibility alias for PerformanceMonitor."""
    from ign_lidar.core import PerformanceMonitor, ProcessorPerformanceMonitor
    assert PerformanceMonitor is ProcessorPerformanceMonitor


def test_gpu_cache_metrics():
    """Test that GPUArrayCache has new metrics methods."""
    from ign_lidar.optimization import GPUArrayCache
    
    cache = GPUArrayCache()
    assert hasattr(cache, 'get_hit_rate')
    assert hasattr(cache, 'print_stats')
    
    # Test methods work
    hit_rate = cache.get_hit_rate()
    assert isinstance(hit_rate, (int, float))
    
    # Should not raise
    cache.print_stats()


def test_transfer_optimizer_import():
    """Test that TransferOptimizer can be imported."""
    from ign_lidar.optimization import TransferOptimizer, create_transfer_optimizer
    assert TransferOptimizer is not None
    assert create_transfer_optimizer is not None


def test_no_redundant_prefixes():
    """Test that no files have redundant prefixes."""
    import os
    
    redundant_prefixes = ['unified_', 'enhanced_', 'improved_', 'new_']
    
    for root, dirs, files in os.walk('ign_lidar'):
        for file in files:
            if file.endswith('.py'):
                filename = file.lower()
                for prefix in redundant_prefixes:
                    assert not filename.startswith(prefix), \
                        f"File {file} has redundant prefix '{prefix}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
