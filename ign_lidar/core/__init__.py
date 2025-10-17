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
