"""
Core processing modules for IGN LiDAR HD.

This package contains the main processing logic:
- processor: Main LiDAR processor  
- tile_stitcher: Tile stitching with boundary handling
- memory: Memory management (consolidated from memory_manager, memory_utils, modules/memory)
- performance: Performance monitoring (consolidated from performance_monitor, performance_monitoring)
- error_handler: Error handling and recovery
- verification: Feature and data verification utilities
- gpu: Centralized GPU detection and management (v3.4.0+)
- ground_truth_manager: Ground truth data fetching and caching (v3.5.0+)
- tile_io_manager: Tile I/O operations and recovery (v3.5.0+)
- feature_engine: Feature computation wrapper (v3.5.0+)
- classification_engine: Classification operations wrapper (v3.5.0+)
"""

from .processor import LiDARProcessor
from .memory import AdaptiveMemoryManager, MemoryConfig
from .performance import ProcessorPerformanceMonitor, PerformanceMonitor, PerformanceSnapshot, PerformanceMetrics
from .error_handler import (
    ProcessingError, 
    GPUMemoryError, 
    GPUNotAvailableError,
    MemoryPressureError,
    FileProcessingError,
    ConfigurationError
)
from .verification import FeatureVerifier, FeatureStats, verify_laz_files
from .gpu import GPUManager, get_gpu_manager, GPU_AVAILABLE, HAS_CUPY
from .gpu_memory import GPUMemoryManager, get_gpu_memory_manager, cleanup_gpu_memory, check_gpu_memory
from .gpu_unified import UnifiedGPUManager
from .ground_truth_hub import GroundTruthHub, ground_truth
from .ground_truth_manager import GroundTruthManager
from .ground_truth_provider import GroundTruthProvider, get_provider as get_ground_truth_provider
from .tile_io_manager import TileIOManager
from .feature_engine import FeatureEngine
from .classification_engine import ClassificationEngine
from .tile_orchestrator import TileOrchestrator
from .gpu_stream_manager import GPUStreamManager, get_stream_manager
from .performance_manager import PerformanceManager, get_performance_manager
from .config_validator import ConfigValidator, get_config_validator
from .migration_helpers import MigrationHelper, CodeTransformer, create_migration_helper

__all__ = [
    'LiDARProcessor',
    'AdaptiveMemoryManager', 
    'MemoryConfig',
    'ProcessorPerformanceMonitor',
    'PerformanceMonitor',  # Backward compatibility
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
    'GPUManager',
    'get_gpu_manager',
    'GPU_AVAILABLE',
    'HAS_CUPY',
    'GPUMemoryManager',
    'get_gpu_memory_manager',
    'cleanup_gpu_memory',
    'check_gpu_memory',
    'UnifiedGPUManager',
    'GroundTruthHub',
    'ground_truth',
    'GroundTruthManager',
    'GroundTruthProvider',
    'get_ground_truth_provider',
    'TileIOManager',
    'FeatureEngine',
    'ClassificationEngine',
    'TileOrchestrator',
    # Phase 5 (v3.6.0+): Unified managers for GPU, performance, and config
    'GPUStreamManager',
    'get_stream_manager',
    'PerformanceManager',
    'get_performance_manager',
    'ConfigValidator',
    'get_config_validator',
    # Phase 6 (v3.6.0+): Migration helpers
    'MigrationHelper',
    'CodeTransformer',
    'create_migration_helper',
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
