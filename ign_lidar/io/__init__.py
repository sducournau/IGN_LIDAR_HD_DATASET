"""
Input/Output modules for IGN LiDAR HD.

This package contains data loading and saving:
- laz_reader: Optimized LAZ file reading
- patch_writer: Multi-format patch writing (NPZ, HDF5, PyTorch)
- formatters: Format-specific data formatters
- metadata: Metadata management for datasets
- qgis_converter: QGIS compatibility converter
- wfs_ground_truth: WFS ground truth fetching from IGN BD TOPO®
"""

from .metadata import MetadataManager
from .qgis_converter import simplify_for_qgis

# Import GPU dataframe operations conditionally (requires cudf/cupy)
try:
    from .gpu_dataframe import GPUDataFrameOps
    GPU_DATAFRAME_AVAILABLE = True
except ImportError:
    GPU_DATAFRAME_AVAILABLE = False
    GPUDataFrameOps = None

# Import WFS ground truth conditionally (requires shapely/geopandas)
try:
    from .wfs_ground_truth import (
        IGNWFSConfig,
        IGNGroundTruthFetcher,
        fetch_ground_truth_for_tile,
        generate_patches_with_ground_truth,
    )
    WFS_AVAILABLE = True
except ImportError:
    # shapely/geopandas not available
    WFS_AVAILABLE = False

# Import optimized WFS fetcher conditionally
try:
    from .wfs_optimized import (
        OptimizedWFSConfig,
        OptimizedWFSFetcher,
    )
    WFS_OPTIMIZED_AVAILABLE = True
except ImportError:
    WFS_OPTIMIZED_AVAILABLE = False

# Build __all__ based on available imports
__all__ = [
    'MetadataManager',
    'simplify_for_qgis',
]

if GPU_DATAFRAME_AVAILABLE:
    __all__.append('GPUDataFrameOps')

if WFS_AVAILABLE:
    __all__.extend([
        'IGNWFSConfig',
        'IGNGroundTruthFetcher',
        'fetch_ground_truth_for_tile',
        'generate_patches_with_ground_truth',
    ])

if WFS_OPTIMIZED_AVAILABLE:
    __all__.extend([
        'OptimizedWFSConfig',
        'OptimizedWFSFetcher',
    ])
