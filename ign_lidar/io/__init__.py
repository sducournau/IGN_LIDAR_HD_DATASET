"""
Input/Output modules for IGN LiDAR HD.

This package contains data loading and saving:
- laz_reader: Optimized LAZ file reading
- patch_writer: Multi-format patch writing (NPZ, HDF5, PyTorch)
- formatters: Format-specific data formatters
- metadata: Metadata management for datasets
- qgis_converter: QGIS compatibility converter
"""

from .metadata import MetadataManager
from .qgis_converter import simplify_for_qgis

__all__ = [
    'MetadataManager',
    'simplify_for_qgis',
]
