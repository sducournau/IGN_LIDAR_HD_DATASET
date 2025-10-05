"""
IGN LiDAR HD Dataset Processing Library

A Python library for processing IGN LiDAR HD data into machine learning-ready datasets
with building LOD (Level of Detail) classification support.
"""

__version__ = "1.7.5"
__author__ = "imagodata"
__email__ = "simon.ducournau@google.com"

# Core classes
from .processor import LiDARProcessor
from .downloader import IGNLiDARDownloader

# Classification classes
from .classes import LOD2_CLASSES, LOD3_CLASSES

# Feature extraction functions
from .features import compute_normals, compute_curvature, extract_geometric_features

# QGIS compatibility
from .qgis_converter import simplify_for_qgis

# Tile management
from .tile_list import (
    WORKING_TILES,
    get_tiles_by_environment,
    get_tiles_by_priority,
    get_tiles_by_region
)

__all__ = [
    # Core functionality
    "LiDARProcessor",
    "IGNLiDARDownloader",
    
    # Classification
    "LOD2_CLASSES", 
    "LOD3_CLASSES",
    
    # Features
    "compute_normals",
    "compute_curvature", 
    "extract_geometric_features",
    
    # QGIS
    "simplify_for_qgis",
    
    # Tile management
    "WORKING_TILES",
    "get_tiles_by_environment",
    "get_tiles_by_priority",
    "get_tiles_by_region"
]