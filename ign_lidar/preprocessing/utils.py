"""
Utility functions for patch extraction and data augmentation

DEPRECATED: This module now re-exports functions from core.modules for backward compatibility.

New code should import directly from:
- ign_lidar.core.modules.patch_extractor (for extraction/augmentation)
- ign_lidar.core.modules.serialization (for saving patches)

These re-exports will be removed in v3.0.
"""

from pathlib import Path
from typing import Dict
import numpy as np

# Re-export from modules for backward compatibility
from ..core.modules.patch_extractor import (
    augment_raw_points,
    extract_patches,
    augment_patch
)
from ..core.modules.serialization import save_patch_npz as _save_patch_npz


def save_patch(save_path: Path, patch: Dict[str, np.ndarray], lod_level: str = 'LOD2') -> None:
    """
    DEPRECATED: Use save_patch_npz, save_patch_hdf5, or save_patch_torch instead.
    
    This is a backward-compatible wrapper that calls save_patch_npz.
    
    Args:
        save_path: Path where patch should be saved
        patch: Patch dictionary to save
        lod_level: Level of detail ('LOD2' or 'LOD3')
    """
    _save_patch_npz(save_path, patch, lod_level=lod_level)


# Re-export for backward compatibility
__all__ = [
    'augment_raw_points',
    'extract_patches',
    'augment_patch',
    'save_patch'
]
