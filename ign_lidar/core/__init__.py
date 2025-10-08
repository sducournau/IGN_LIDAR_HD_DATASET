"""
Core processing modules for IGN LiDAR HD.

This package contains the main processing logic:
- processor: Main LiDAR processor  
- tile_stitcher: Tile stitching with boundary handling
- pipeline_config: Pipeline configuration (legacy, migrating to Hydra)
- memory_manager: Adaptive memory management for processing
- performance_monitor: Real-time performance tracking
- error_handler: Enhanced error handling and recovery
- verification: Feature and data verification utilities
"""

from .processor import LiDARProcessor
from .memory_manager import AdaptiveMemoryManager, MemoryConfig
from .performance_monitor import PerformanceMonitor, PerformanceSnapshot
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
