"""
Classification I/O Module

Provides loading, serialization, and caching for LiDAR classification data.

This module consolidates all I/O-related functionality previously scattered across
multiple files in the classification root directory. It provides a consistent interface
for loading LAZ/LAS files, serializing results, and managing tile data.

Modules:
    - loaders: LiDAR file loading (LAS/LAZ) with error handling and validation
    - serializers: Multi-format export (NPZ, HDF5, PyTorch, LAZ)
    - tiles: Tile-based loading with caching and memory management
    - utils: Shared I/O utilities and helpers

Usage:
    from ign_lidar.core.classification.io import load_laz_file, save_patch_laz
    
    # Load a LAZ file
    data = load_laz_file('tile.laz')
    
    # Save enriched results
    save_patch_laz('output.laz', points, labels)

Author: Classification Enhancement Team
Date: October 23, 2025
Version: 1.0.0 (Task 7: I/O Module Consolidation)
"""

# Import from submodules for convenience
from .loaders import (
    LiDARData,
    LiDARLoadError,
    LiDARCorruptionError,
    load_laz_file,
    validate_lidar_data,
    map_classification,
    get_tile_info,
    estimate_file_size_gb,
    check_file_readable,
)

from .serializers import (
    save_patch_npz,
    save_patch_hdf5,
    save_patch_torch,
    save_patch_laz,
    save_patch_multi_format,
    save_enriched_tile_laz,
    validate_format_support,
)

from .tiles import (
    TileLoader,
    TileDataCache,
    get_global_cache,
    clear_global_cache,
)

__all__ = [
    # Data classes
    'LiDARData',
    
    # Exceptions
    'LiDARLoadError',
    'LiDARCorruptionError',
    
    # Loader functions
    'load_laz_file',
    'validate_lidar_data',
    'map_classification',
    'get_tile_info',
    'estimate_file_size_gb',
    'check_file_readable',
    
    # Serializer functions
    'save_patch_npz',
    'save_patch_hdf5',
    'save_patch_torch',
    'save_patch_laz',
    'save_patch_multi_format',
    'save_enriched_tile_laz',
    'validate_format_support',
    
    # Tile loading classes
    'TileLoader',
    'TileDataCache',
    'get_global_cache',
    'clear_global_cache',
]

__version__ = '1.0.0'
