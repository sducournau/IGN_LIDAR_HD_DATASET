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

# Import WFS ground truth conditionally (requires shapely/geopandas)
try:
    from .wfs_ground_truth import (
        IGNWFSConfig,
        IGNGroundTruthFetcher,
        fetch_ground_truth_for_tile,
        generate_patches_with_ground_truth,
    )
    __all__ = [
        'MetadataManager',
        'simplify_for_qgis',
        'IGNWFSConfig',
        'IGNGroundTruthFetcher',
        'fetch_ground_truth_for_tile',
        'generate_patches_with_ground_truth',
    ]
except ImportError:
    # shapely/geopandas not available
    __all__ = [
        'MetadataManager',
        'simplify_for_qgis',
    ]
