"""
Core processing modules for IGN LiDAR HD.

This package contains the main processing logic:
- processor: Main LiDAR processor  
- tile_stitcher: Tile stitching with boundary handling
- memory: Unified memory management (consolidated from memory_manager, memory_utils, modules/memory)
- performance: Unified performance monitoring (consolidated from performance_monitor, performance_monitoring)
- error_handler: Error handling and recovery
- verification: Feature and data verification utilities
"""

from .processor import LiDARProcessor
from .memory import AdaptiveMemoryManager, MemoryConfig
from .performance import PerformanceMonitor, PerformanceSnapshot, PerformanceMetrics
from .error_handler import (
    ProcessingError, 
    GPUMemoryError, 
    GPUNotAvailableError,
    MemoryPressureError,
    FileProcessingError,
    ConfigurationError
)
from .verification import FeatureVerifier, FeatureStats, verify_laz_files

__all__ = [
    'LiDARProcessor',
    'AdaptiveMemoryManager', 
    'MemoryConfig',
    'PerformanceMonitor',
    'PerformanceSnapshot', 
    'ProcessingError',
    'GPUMemoryError',
    'GPUNotAvailableError', 
    'MemoryPressureError',
    'FileProcessingError',
    'ConfigurationError',
    'FeatureVerifier',
    'FeatureStats',
    'verify_laz_files',
]

# Backward compatibility: core.modules moved to core.classification in v3.1.0
# This allows old imports to continue working with a deprecation warning
import sys
import warnings
from types import ModuleType


class _ModulesCompatibilityModule(ModuleType):
    """
    Compatibility shim for core.modules → core.classification rename.
    
    This allows code using the old path to continue working:
        from ign_lidar.core.modules.classification_thresholds import ClassificationThresholds
    
    While showing a deprecation warning guiding users to the new path:
        from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds
    """
    
    def __getattr__(self, name):
        # Handle special module attributes without deprecation warnings
        if name in ('__path__', '__file__', '__package__', '__spec__', '__loader__', '__cached__'):
            # Get the actual classification module to access its attributes
            import importlib
            classification_module = importlib.import_module('ign_lidar.core.classification')
            return getattr(classification_module, name, None)
        
        # For all other attributes, show deprecation warning
        warnings.warn(
            f"Importing from 'ign_lidar.core.modules' is deprecated. "
            f"Use 'ign_lidar.core.classification' instead. "
            f"The 'core.modules' path will be removed in v4.0.0.\n"
            f"  OLD: from ign_lidar.core.modules.{name} import ...\n"
            f"  NEW: from ign_lidar.core.classification.{name} import ...",
            DeprecationWarning,
            stacklevel=2
        )
        # Import the actual module from new location
        import importlib
        module_path = f'ign_lidar.core.classification.{name}'
        try:
            return importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}' from 'ign_lidar.core.modules' (now 'ign_lidar.core.classification'). "
                f"Original error: {e}"
            ) from e


# Register the compatibility module
sys.modules['ign_lidar.core.modules'] = _ModulesCompatibilityModule('ign_lidar.core.modules')
