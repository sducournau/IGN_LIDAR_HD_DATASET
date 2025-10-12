"""
Serialization module for saving patches and enriched data in various formats.

This module provides functions to save LiDAR patches and enriched point clouds
in multiple output formats: NPZ, HDF5, PyTorch, and LAZ.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    logger.debug("h5py not available, HDF5 format not supported")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.debug("torch not available, PyTorch format not supported")

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    logger.debug("laspy not available, LAZ format not supported")


def save_patch_npz(save_path: Path, data: Dict[str, np.ndarray], 
                   lod_level: Optional[str] = None) -> None:
    """
    Save patch to compressed NPZ file.
    
    Args:
        save_path: Output NPZ file path
        data: Patch data dictionary with numpy arrays
        lod_level: Optional LOD level string to include in saved data
    """
    save_data = data.copy()
    if lod_level is not None:
        save_data['lod_level'] = lod_level
    
    np.savez_compressed(save_path, **save_data)
    logger.debug(f"Saved NPZ patch: {save_path.name}")


def save_patch_hdf5(save_path: Path, data: Dict[str, np.ndarray]) -> None:
    """
    Save patch to HDF5 file with GZIP compression.
    
    Args:
        save_path: Output HDF5 file path
        data: Patch data dictionary
        
    Raises:
        ImportError: If h5py is not installed
    """
    if not H5PY_AVAILABLE:
        raise ImportError(
            "h5py is required for HDF5 format. "
            "Install with: pip install h5py"
        )
    
    with h5py.File(save_path, 'w') as f:
        # Extract metadata (don't modify data - use get instead of pop)
        metadata = data.get('metadata', None)
        
        # Save all numpy arrays as datasets
        for key, value in data.items():
            # Skip metadata and non-array fields
            if key == 'metadata':
                continue
            if isinstance(value, np.ndarray):
                f.create_dataset(
                    key, 
                    data=value, 
                    compression='gzip', 
                    compression_opts=9
                )
            else:
                logger.warning(
                    f"Skipping non-array field '{key}' for HDF5 "
                    f"(type: {type(value).__name__})"
                )
        
        # Save metadata as HDF5 attributes (flattened)
        if metadata is not None:
            for meta_key, meta_value in metadata.items():
                try:
                    # Convert lists to arrays for HDF5 compatibility
                    if isinstance(meta_value, list):
                        meta_value = np.array(meta_value)
                    f.attrs[meta_key] = meta_value
                except (TypeError, ValueError) as e:
                    logger.debug(
                        f"Could not save metadata key '{meta_key}': {e}"
                    )
    
    logger.debug(f"Saved HDF5 patch: {save_path.name}")


def save_patch_torch(save_path: Path, data: Dict[str, np.ndarray]) -> None:
    """
    Save patch to PyTorch .pt file.
    
    Converts numpy arrays to torch tensors before saving.
    
    Args:
        save_path: Output PyTorch file path
        data: Patch data dictionary
        
    Raises:
        ImportError: If torch is not installed
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for torch format. "
            "Install with: pip install torch"
        )
    
    # Convert numpy arrays to torch tensors (skip metadata and non-arrays)
    torch_data = {}
    for k, v in data.items():
        if k == 'metadata':
            # Keep metadata as-is for PyTorch (it supports dicts)
            torch_data[k] = v
        elif isinstance(v, np.ndarray):
            torch_data[k] = torch.from_numpy(v)
        else:
            logger.warning(
                f"Skipping non-array field '{k}' for PyTorch "
                f"(type: {type(v).__name__})"
            )
    
    torch.save(torch_data, save_path)
    logger.debug(f"Saved PyTorch patch: {save_path.name}")


def save_patch_laz(save_path: Path, 
                   arch_data: Dict[str, np.ndarray],
                   original_patch: Dict[str, np.ndarray]) -> None:
    """
    Save a patch as a LAZ point cloud file with all computed features.
    
    This function creates a LAZ file with coordinates, classification, intensity,
    RGB colors (if available), NIR (if available), and adds all geometric and
    radiometric features as extra dimensions.
    
    Args:
        save_path: Output LAZ file path
        arch_data: Formatted patch data (architecture-specific)
        original_patch: Original patch data with metadata and features
        
    Raises:
        ImportError: If laspy is not installed
    """
    if not LASPY_AVAILABLE:
        raise ImportError(
            "laspy is required for LAZ format. "
            "Install with: pip install laspy"
        )
    
    # Extract coordinates from arch_data
    # Most architectures have 'points' or 'coords' key
    if 'points' in arch_data:
        coords = arch_data['points'][:, :3].copy()  # XYZ
    elif 'coords' in arch_data:
        coords = arch_data['coords'][:, :3].copy()
    else:
        logger.warning(
            f"Cannot save LAZ patch: no coordinates found in "
            f"{list(arch_data.keys())}"
        )
        return
    
    # Restore LAMB93 coordinates if metadata available
    coords = _restore_lamb93_coordinates(save_path, coords, original_patch)
    
    # Determine point format based on available data
    has_rgb = 'rgb' in arch_data
    has_nir = 'nir' in original_patch and original_patch['nir'] is not None
    point_format = 8 if (has_rgb and has_nir) else (3 if has_rgb else 6)
    
    # Create LAZ file
    header = laspy.LasHeader(version="1.4", point_format=point_format)
    header.offsets = [
        np.floor(coords[:, 0].min()),
        np.floor(coords[:, 1].min()),
        np.floor(coords[:, 2].min())
    ]
    header.scales = [0.001, 0.001, 0.001]
    
    las = laspy.LasData(header)
    las.x = coords[:, 0]
    las.y = coords[:, 1]
    las.z = coords[:, 2]
    
    # Add standard LAS fields
    _add_standard_las_fields(las, arch_data, original_patch, point_format)
    
    # Add computed features as extra dimensions
    _add_geometric_features(las, original_patch)
    _add_height_features(las, original_patch)
    _add_radiometric_features(las, original_patch, point_format)
    _add_return_number(las, original_patch)
    
    # Write LAZ file
    las.write(str(save_path))
    logger.debug(f"Saved LAZ patch: {save_path.name}")


def _restore_lamb93_coordinates(save_path: Path, 
                                coords: np.ndarray,
                                original_patch: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Restore LAMB93 coordinates from normalized patch coordinates.
    
    Args:
        save_path: Output file path (filename contains tile coordinates)
        coords: Normalized coordinates [N, 3]
        original_patch: Original patch with metadata
        
    Returns:
        Restored coordinates in LAMB93 projection
    """
    try:
        filename = save_path.stem
        parts = filename.split('_')
        
        # Look for tile coordinates in filename (e.g., LHD_FXX_0649_6863_...)
        if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
            tile_x = int(parts[2])
            tile_y = int(parts[3])
            
            # LAMB93 tiles are 1km x 1km
            # tile center at (tile_x * 1000 + 500, tile_y * 1000 + 500)
            tile_center_x = tile_x * 1000 + 500
            tile_center_y = tile_y * 1000 + 500
            
            # If metadata has centroid, restore original coordinates
            if 'metadata' in original_patch and isinstance(original_patch['metadata'], dict):
                metadata = original_patch['metadata']
                if 'centroid' in metadata:
                    centroid = np.array(metadata['centroid'])
                    coords[:, 0] = coords[:, 0] + centroid[0] + tile_center_x
                    coords[:, 1] = coords[:, 1] + centroid[1] + tile_center_y
                    coords[:, 2] = coords[:, 2] + centroid[2]
                else:
                    # No centroid, just apply tile offset
                    coords[:, 0] = coords[:, 0] + tile_center_x
                    coords[:, 1] = coords[:, 1] + tile_center_y
    except Exception as e:
        logger.debug(f"Could not restore LAMB93 coordinates: {e}")
        # Continue with normalized coordinates
    
    return coords


def _add_standard_las_fields(las, arch_data: Dict, original_patch: Dict,
                             point_format: int) -> None:
    """Add standard LAS fields (classification, intensity, RGB, NIR)."""
    # Add classification if available
    if 'labels' in arch_data:
        las.classification = arch_data['labels'].astype(np.uint8)
    elif 'classification' in original_patch:
        las.classification = original_patch['classification'].astype(np.uint8)
    
    # Add intensity if available
    if 'intensity' in original_patch:
        las.intensity = (original_patch['intensity'] * 65535).astype(np.uint16)
    
    # Add RGB if available
    if 'rgb' in arch_data:
        rgb = (arch_data['rgb'] * 65535).astype(np.uint16)
        las.red = rgb[:, 0]
        las.green = rgb[:, 1]
        las.blue = rgb[:, 2]
    
    # Add NIR if available and point format supports it (format 8)
    if point_format == 8 and 'nir' in original_patch and original_patch['nir'] is not None:
        nir = (original_patch['nir'] * 65535).astype(np.uint16)
        las.nir = nir


def _add_geometric_features(las, original_patch: Dict) -> None:
    """Add ALL geometric features as extra dimensions (supports full mode with 40+ features)."""
    # ALL geometric features in consistent order (matches base_formatter.py)
    geometric_features = [
        # Core shape descriptors (6)
        'planarity', 'linearity', 'sphericity', 'anisotropy', 'roughness', 'omnivariance',
        # Curvature features (2)
        'curvature', 'change_curvature',
        # Eigenvalue features (5)
        'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3', 'sum_eigenvalues', 'eigenentropy',
        # Building scores (4)
        'verticality', 'horizontality', 'wall_score', 'roof_score',
        # Density features (5)
        'density', 'local_density', 'num_points_2m', 'neighborhood_extent', 'height_extent_ratio',
        # Architectural features (5)
        'edge_strength', 'corner_likelihood', 'overhang_indicator', 'surface_roughness', 'local_roughness',
    ]
    
    for feat_name in geometric_features:
        if feat_name in original_patch:
            try:
                las.add_extra_dim(laspy.ExtraBytesParams(
                    name=feat_name,
                    type=np.float32,
                    description=f"Geom: {feat_name}"
                ))
                setattr(las, feat_name, original_patch[feat_name].astype(np.float32))
            except Exception as e:
                logger.warning(
                    f"Could not add feature '{feat_name}' to LAZ: {e}"
                )
    
    # Normals (normal_x, normal_y, normal_z)
    if 'normals' in original_patch:
        try:
            normals = original_patch['normals']
            for i, comp in enumerate(['normal_x', 'normal_y', 'normal_z']):
                las.add_extra_dim(laspy.ExtraBytesParams(
                    name=comp,
                    type=np.float32,
                    description=f"Normal {comp[-1]}"
                ))
                setattr(las, comp, normals[:, i].astype(np.float32))
        except Exception as e:
            logger.warning(f"Could not add normals to LAZ: {e}")


def _add_height_features(las, original_patch: Dict) -> None:
    """Add ALL height features as extra dimensions."""
    height_features = [
        # Core height features
        'height', 'height_above_ground', 'vertical_std',
        # Normalized height features
        'z_normalized', 'z_absolute', 'z_from_ground', 'z_from_median',
        # Distance features
        'distance_to_center',
    ]
    
    for feat_name in height_features:
        if feat_name in original_patch:
            try:
                las.add_extra_dim(laspy.ExtraBytesParams(
                    name=feat_name,
                    type=np.float32,
                    description=f"Height feature: {feat_name}"
                ))
                setattr(las, feat_name, original_patch[feat_name].astype(np.float32))
            except Exception as e:
                logger.warning(
                    f"Could not add feature '{feat_name}' to LAZ: {e}"
                )


def _add_radiometric_features(las, original_patch: Dict, 
                              point_format: int) -> None:
    """Add radiometric features (NIR, NDVI) as extra dimensions."""
    # Add NIR as extra dimension if not already included in point format
    if point_format != 8 and 'nir' in original_patch and original_patch['nir'] is not None:
        try:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name='nir',
                type=np.float32,
                description="NIR reflectance (norm 0-1)"
            ))
            las.nir = original_patch['nir'].astype(np.float32)
        except Exception as e:
            logger.warning(f"Could not add NIR to LAZ: {e}")
    
    if 'ndvi' in original_patch and original_patch['ndvi'] is not None:
        try:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name='ndvi',
                type=np.float32,
                description="NDVI (vegetation index)"
            ))
            las.ndvi = original_patch['ndvi'].astype(np.float32)
        except Exception as e:
            logger.warning(f"Could not add NDVI to LAZ: {e}")


def _add_return_number(las, original_patch: Dict) -> None:
    """Add return number if available."""
    if 'return_number' in original_patch:
        try:
            # Return number might already be a standard field
            if not hasattr(las, 'return_number'):
                las.add_extra_dim(laspy.ExtraBytesParams(
                    name='return_number',
                    type=np.uint8,
                    description="Return number"
                ))
                las.return_number = original_patch['return_number'].astype(np.uint8)
        except Exception as e:
            logger.warning(f"Could not add return_number to LAZ: {e}")


def save_patch_multi_format(base_path: Path,
                           data: Dict[str, np.ndarray],
                           formats: List[str],
                           original_patch: Optional[Dict] = None,
                           lod_level: Optional[str] = None) -> int:
    """
    Save patch in multiple formats.
    
    Args:
        base_path: Base path without extension (e.g., "patch_0001")
        data: Patch data dictionary
        formats: List of format strings ('npz', 'hdf5', 'torch', 'laz')
        original_patch: Original patch for LAZ format (required if saving LAZ)
        lod_level: LOD level for NPZ format
        
    Returns:
        Number of files saved
        
    Raises:
        ImportError: If required libraries for requested formats are not installed
    """
    num_saved = 0
    
    for fmt in formats:
        fmt = fmt.strip().lower()
        
        if fmt == 'npz':
            save_path = base_path.with_suffix('.npz')
            save_patch_npz(save_path, data, lod_level)
            num_saved += 1
            
        elif fmt == 'hdf5':
            save_path = base_path.with_suffix('.h5')
            save_patch_hdf5(save_path, data)
            num_saved += 1
            
        elif fmt in ['pytorch', 'torch']:
            save_path = base_path.with_suffix('.pt')
            save_patch_torch(save_path, data)
            num_saved += 1
            
        elif fmt == 'laz':
            if original_patch is None:
                logger.warning(
                    "Cannot save LAZ format: original_patch required"
                )
                continue
            save_path = base_path.with_suffix('.laz')
            save_patch_laz(save_path, data, original_patch)
            num_saved += 1
            
        else:
            logger.warning(f"Unknown format '{fmt}', skipping")
    
    return num_saved


def validate_format_support(formats: List[str]) -> Dict[str, bool]:
    """
    Check which formats are supported based on installed libraries.
    
    Args:
        formats: List of format strings to check
        
    Returns:
        Dictionary mapping format to boolean (supported or not)
    """
    support = {}
    
    for fmt in formats:
        fmt = fmt.strip().lower()
        
        if fmt == 'npz':
            support[fmt] = True  # NumPy is always available
        elif fmt == 'hdf5':
            support[fmt] = H5PY_AVAILABLE
        elif fmt in ['pytorch', 'torch']:
            support[fmt] = TORCH_AVAILABLE
        elif fmt == 'laz':
            support[fmt] = LASPY_AVAILABLE
        else:
            support[fmt] = False
    
    return support
