"""
Preprocessing modules for IGN LiDAR HD.

This package contains data cleaning and augmentation:
- preprocessing: Statistical outlier removal, radius outlier removal, voxel downsampling
- rgb_augmentation: RGB data augmentation from orthophotos
- infrared_augmentation: NIR/IRC data augmentation
- utils: Patch extraction and data augmentation utilities
- tile_analyzer: Tile analysis for optimal processing parameters
"""

from .preprocessing import (
    statistical_outlier_removal,
    radius_outlier_removal,
    voxel_downsample,
    preprocess_point_cloud,
)
from .rgb_augmentation import (
    add_rgb_to_patch,
    augment_tile_with_rgb,
)
from .infrared_augmentation import (
    add_infrared_to_patch,
    augment_tile_with_infrared,
)
from .utils import (
    augment_raw_points,
    extract_patches,
)
from .tile_analyzer import analyze_tile

__all__ = [
    'statistical_outlier_removal',
    'radius_outlier_removal',
    'voxel_downsample',
    'preprocess_point_cloud',
    'add_rgb_to_patch',
    'augment_tile_with_rgb',
    'add_infrared_to_patch',
    'augment_tile_with_infrared',
    'augment_raw_points',
    'extract_patches',
    'analyze_tile',
]
