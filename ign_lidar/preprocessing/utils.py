"""
Utility functions for patch extraction and data augmentation
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np


def augment_raw_points(
    points: np.ndarray,
    intensity: np.ndarray,
    return_number: np.ndarray,
    classification: np.ndarray,
    rgb: np.ndarray = None,
    nir: np.ndarray = None,
    ndvi: np.ndarray = None,
    return_mask: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Apply data augmentation to raw point cloud data BEFORE feature computation.
    
    This ensures geometric features (normals, curvature, planarity, etc.) are
    computed on the augmented geometry, maintaining consistency between
    coordinates and derived features.
    
    Augmentations applied:
    1. Random rotation around Z-axis (0-360°)
    2. Random jitter (Gaussian noise, σ=0.1m)
    3. Random scaling (0.95-1.05)
    4. Random point dropout (5-15%)
    
    Args:
        points: [N, 3] raw point coordinates (X, Y, Z)
        intensity: [N] intensity values
        return_number: [N] return numbers
        classification: [N] ASPRS classification codes
        rgb: [N, 3] RGB values (optional)
        nir: [N] near-infrared values (optional)
        ndvi: [N] NDVI values (optional)
        return_mask: If True, return keep_mask as 8th element
        
    Returns:
        Tuple of (augmented_points, intensity, return_number, classification, rgb, nir, ndvi)
        All arrays are filtered by the same dropout mask
        If return_mask=True, returns (points, intensity, return_number, classification, rgb, nir, ndvi, keep_mask)
    """
    N = len(points)
    points_aug = points.copy()
    
    # 1. Random rotation around Z-axis (preserves vertical structures)
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    points_aug = points_aug @ rotation_matrix.T
    
    # 2. Random jitter (simulates sensor noise)
    jitter = np.random.normal(0, 0.1, (N, 3)).astype(np.float32)
    points_aug += jitter
    
    # 3. Random scaling (simulates distance variations)
    scale = np.random.uniform(0.95, 1.05)
    points_aug *= scale
    
    # 4. Random dropout (simulates occlusion, missing data)
    dropout_ratio = np.random.uniform(0.05, 0.15)
    keep_mask = np.random.random(N) > dropout_ratio
    
    # Apply dropout to all arrays
    points_aug = points_aug[keep_mask]
    intensity_aug = intensity[keep_mask]
    return_number_aug = return_number[keep_mask]
    classification_aug = classification[keep_mask]
    
    # Apply dropout to optional RGB, NIR, NDVI arrays
    rgb_aug = rgb[keep_mask] if rgb is not None else None
    nir_aug = nir[keep_mask] if nir is not None else None
    ndvi_aug = ndvi[keep_mask] if ndvi is not None else None
    
    if return_mask:
        return points_aug, intensity_aug, return_number_aug, classification_aug, rgb_aug, nir_aug, ndvi_aug, keep_mask
    else:
        return points_aug, intensity_aug, return_number_aug, classification_aug, rgb_aug, nir_aug, ndvi_aug


def extract_patches(
    points: np.ndarray,
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    patch_size: float = 150.0,
    overlap: float = 0.1,
    min_points: int = 10000,
    target_num_points: int = None
) -> List[Dict[str, np.ndarray]]:
    """
    Extract patches from point cloud with overlap.
    
    Args:
        points: [N, 3] point coordinates
        features: Dictionary of point features
        labels: [N] classification labels
        patch_size: Patch size in meters
        overlap: Overlap ratio
        min_points: Minimum points per patch
        target_num_points: Target number of points (None = no resampling,
                          otherwise resample to this number)
        
    Returns:
        List of patch dictionaries
    """
    # Compute bounding box
    x_min, y_min = points[:, :2].min(axis=0)
    x_max, y_max = points[:, :2].max(axis=0)
    
    # Compute patch grid with overlap
    stride = patch_size * (1 - overlap)
    x_steps = int(np.ceil((x_max - x_min) / stride))
    y_steps = int(np.ceil((y_max - y_min) / stride))
    
    patches = []
    
    for i in range(x_steps):
        for j in range(y_steps):
            # Define patch bounds
            x_start = x_min + i * stride
            y_start = y_min + j * stride
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            
            # Find points in patch
            mask = (
                (points[:, 0] >= x_start) & (points[:, 0] < x_end) &
                (points[:, 1] >= y_start) & (points[:, 1] < y_end)
            )
            
            if np.sum(mask) < min_points:
                continue
            
            # Extract patch data
            patch_points = points[mask]
            patch_labels = labels[mask]
            
            # Center patch coordinates
            patch_center = np.array([
                (x_start + x_end) / 2,
                (y_start + y_end) / 2,
                0
            ])
            patch_points = patch_points - patch_center
            
            # Build patch dictionary
            patch = {
                'points': patch_points,
                'labels': patch_labels
            }
            
            # Add features
            for feature_name, feature_data in features.items():
                patch[feature_name] = feature_data[mask]
            
            # Resample to target number of points if specified
            if target_num_points is not None:
                num_points_current = len(patch_points)
                
                if num_points_current > target_num_points:
                    # Downsample: random selection
                    indices = np.random.choice(
                        num_points_current,
                        target_num_points,
                        replace=False
                    )
                elif num_points_current < target_num_points:
                    # Upsample: repeat random points
                    indices = np.random.choice(
                        num_points_current,
                        target_num_points,
                        replace=True
                    )
                else:
                    # Exact match, no resampling
                    indices = None
                
                # Apply resampling
                if indices is not None:
                    patch['points'] = patch['points'][indices]
                    patch['labels'] = patch['labels'][indices]
                    for feature_name in list(patch.keys()):
                        if feature_name not in ['points', 'labels']:
                            patch[feature_name] = patch[feature_name][indices]
            
            patches.append(patch)
    
    return patches


def augment_patch(patch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Apply data augmentation to a patch.
    
    Args:
        patch: Input patch dictionary
        
    Returns:
        Augmented patch dictionary
    """
    aug_patch = {}
    
    # Copy points and get dimensions
    points = patch['points'].copy()
    N = len(points)
    
    # 1. Random rotation around Z-axis
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    points = points @ rotation_matrix.T
    
    # 2. Random jitter
    jitter = np.random.normal(0, 0.1, (N, 3))
    points += jitter
    
    # 3. Random scaling
    scale = np.random.uniform(0.95, 1.05)
    points *= scale
    
    # 4. Random dropout
    dropout_ratio = np.random.uniform(0.05, 0.15)
    keep_mask = np.random.random(N) > dropout_ratio
    
    # Apply dropout to all data
    points = points[keep_mask]
    aug_patch['points'] = points
    
    # Apply transformations to other arrays
    for key, value in patch.items():
        if key == 'points':
            continue
        elif key == 'normals':
            # Rotate normals
            normals = value.copy()
            normals = normals @ rotation_matrix.T
            aug_patch['normals'] = normals[keep_mask]
        else:
            # Just apply dropout
            aug_patch[key] = value[keep_mask]
    
    return aug_patch


def save_patch(save_path: Path, patch: Dict[str, np.ndarray], 
               lod_level: str) -> None:
    """
    Save patch to NPZ file.
    
    Args:
        save_path: Output file path
        patch: Patch dictionary
        lod_level: LOD level string
    """
    save_data = {**patch, 'lod_level': lod_level}
    np.savez_compressed(save_path, **save_data)