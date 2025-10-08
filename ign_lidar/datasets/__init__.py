"""
Datasets module for IGN LiDAR HD v2.0

Provides PyTorch-compatible datasets for multi-architecture deep learning
and strategic location management for diverse AI training datasets.
"""

# Always available imports
from .strategic_locations import STRATEGIC_LOCATIONS
from .tile_list import (
    WORKING_TILES,
    TileInfo,
    get_tiles_by_environment,
    get_tiles_by_priority,
    get_tiles_by_region
)

# Optional imports (require PyTorch)
try:
    from .multi_arch_dataset import IGNLiDARMultiArchDataset
    from .augmentation import PatchAugmentation
    
    __all__ = [
        'IGNLiDARMultiArchDataset',
        'PatchAugmentation',
        'STRATEGIC_LOCATIONS',
        'WORKING_TILES',
        'TileInfo',
        'get_tiles_by_environment',
        'get_tiles_by_priority',
        'get_tiles_by_region',
    ]
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        f"⚠️  Dataset classes not available (missing dependencies): {e}\n"
        "   Install PyTorch to use IGNLiDARMultiArchDataset"
    )
    
    __all__ = [
        'STRATEGIC_LOCATIONS',
        'WORKING_TILES',
        'TileInfo',
        'get_tiles_by_environment',
        'get_tiles_by_priority',
        'get_tiles_by_region',
    ]
