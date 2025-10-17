"""
Unified Feature API with Mode-Based Selection

This module provides a single import path for all feature functions with 
mode-based selection (CPU/GPU/chunked) via parameters, replacing the need
to import from different modules.

Usage:
    from ign_lidar.features.unified_api import compute_verticality, extract_geometric_features
    
    # Mode-based selection
    verticality = compute_verticality(normals, mode='cpu')
    verticality = compute_verticality(normals, mode='gpu') 
    verticality = compute_verticality(normals, mode='chunked')
    
    # Auto mode selection
    features = extract_geometric_features(points, normals, mode='auto')
"""

import numpy as np
import logging
from typing import Dict, Optional, Union, Tuple
from enum import Enum

# Import core implementations
from .core import (
    compute_verticality as core_compute_verticality,
    extract_geometric_features as core_extract_geometric_features,
    compute_all_features as core_compute_all_features,
    ComputeMode
)

logger = logging.getLogger(__name__)


class FeatureMode(Enum):
    """Feature computation modes."""
    CPU = "cpu"
    GPU = "gpu"
    GPU_CHUNKED = "chunked"  # Alias for gpu_chunked
    BOUNDARY_AWARE = "boundary"
    AUTO = "auto"


def compute_verticality(
    normals: np.ndarray,
    mode: Union[str, FeatureMode] = FeatureMode.CPU
) -> np.ndarray:
    """
    Unified verticality computation with mode selection.
    
    Args:
        normals: [N, 3] surface normal vectors
        mode: Computation mode ('cpu', 'gpu', 'chunked', 'boundary', 'auto')
        
    Returns:
        verticality: [N] verticality values [0, 1]
        
    Example:
        >>> verticality = compute_verticality(normals, mode='gpu')
    """
    if isinstance(mode, str):
        mode = FeatureMode(mode.lower())
    
    # All modes use the same core implementation for consistency
    # The mode parameter is kept for API compatibility and future GPU optimization
    if mode in [FeatureMode.GPU, FeatureMode.GPU_CHUNKED]:
        logger.debug(f"Computing verticality using {mode.value} mode (core implementation)")
    
    return core_compute_verticality(normals)


def extract_geometric_features(
    points: np.ndarray,
    normals: np.ndarray,
    mode: Union[str, FeatureMode] = FeatureMode.CPU,
    k: int = 10,
    radius: Optional[float] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Unified geometric features extraction with mode selection.
    
    Args:
        points: [N, 3] point coordinates
        normals: [N, 3] normal vectors
        mode: Computation mode ('cpu', 'gpu', 'chunked', 'boundary', 'auto')
        k: Number of neighbors
        radius: Search radius in meters
        **kwargs: Additional mode-specific arguments
        
    Returns:
        features: Dictionary of geometric features
        
    Example:
        >>> features = extract_geometric_features(
        ...     points, normals, mode='auto', radius=0.5
        ... )
    """
    if isinstance(mode, str):
        mode = FeatureMode(mode.lower())
    
    # For now, all modes use the core implementation
    # Future enhancement: dispatch to specialized implementations based on mode
    if mode in [FeatureMode.GPU, FeatureMode.GPU_CHUNKED]:
        logger.debug(f"Computing geometric features using {mode.value} mode (core implementation)")
    
    return core_extract_geometric_features(points, normals, k=k, radius=radius)


def compute_all_features(
    points: np.ndarray,
    classification: np.ndarray,
    mode: Union[str, FeatureMode] = FeatureMode.AUTO,
    k_neighbors: int = 10,
    radius: Optional[float] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Unified feature computation with automatic mode selection.
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] ASPRS classification codes
        mode: Computation mode ('cpu', 'gpu', 'chunked', 'boundary', 'auto')
        k_neighbors: Number of neighbors
        radius: Search radius in meters
        **kwargs: Additional arguments
        
    Returns:
        normals: [N, 3] surface normal vectors
        curvature: [N] curvature values
        height: [N] height above ground
        features: Dictionary of geometric features
        
    Example:
        >>> normals, curvature, height, features = compute_all_features(
        ...     points, classification, mode='auto'
        ... )
    """
    if isinstance(mode, str):
        # Map aliases
        mode_map = {
            'chunked': 'gpu_chunked',
            'boundary': 'boundary_aware'
        }
        mode_str = mode_map.get(mode.lower(), mode.lower())
        core_mode = ComputeMode(mode_str)
    else:
        # Convert FeatureMode to ComputeMode
        mode_map = {
            FeatureMode.CPU: ComputeMode.CPU,
            FeatureMode.GPU: ComputeMode.GPU,
            FeatureMode.GPU_CHUNKED: ComputeMode.GPU_CHUNKED,
            FeatureMode.BOUNDARY_AWARE: ComputeMode.BOUNDARY_AWARE,
            FeatureMode.AUTO: ComputeMode.AUTO
        }
        core_mode = mode_map[mode]
    
    return core_compute_all_features(
        points, classification, mode=core_mode, 
        k_neighbors=k_neighbors, radius=radius, **kwargs
    )


def compute_normals(
    points: np.ndarray,
    mode: Union[str, FeatureMode] = FeatureMode.CPU,
    k_neighbors: int = 20,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified normal computation with mode selection.
    
    Args:
        points: [N, 3] point coordinates
        mode: Computation mode ('cpu', 'gpu', 'chunked', 'boundary', 'auto')
        k_neighbors: Number of neighbors
        **kwargs: Additional arguments
        
    Returns:
        normals: [N, 3] surface normal vectors
        eigenvalues: [N, 3] eigenvalues
        
    Example:
        >>> normals, eigenvalues = compute_normals(points, mode='gpu')
    """
    if isinstance(mode, str):
        mode = FeatureMode(mode.lower())
    
    # Import core implementation
    from .core import compute_normals as core_compute_normals
    
    # For now, use core implementation regardless of mode
    # Future enhancement: dispatch based on mode
    return core_compute_normals(points, k_neighbors=k_neighbors, **kwargs)


def compute_curvature(
    normals: np.ndarray,
    points: np.ndarray,
    mode: Union[str, FeatureMode] = FeatureMode.CPU,
    **kwargs
) -> np.ndarray:
    """
    Unified curvature computation with mode selection.
    
    Args:
        normals: [N, 3] surface normal vectors
        points: [N, 3] point coordinates
        mode: Computation mode ('cpu', 'gpu', 'chunked', 'boundary', 'auto')
        **kwargs: Additional arguments
        
    Returns:
        curvature: [N] curvature values
        
    Example:
        >>> curvature = compute_curvature(normals, points, mode='cpu')
    """
    if isinstance(mode, str):
        mode = FeatureMode(mode.lower())
    
    # Import core implementation
    from .core import compute_curvature as core_compute_curvature
    
    # For now, use core implementation regardless of mode
    return core_compute_curvature(normals, **kwargs)


# Legacy compatibility: provide mode-less versions that delegate to core
def compute_verticality_legacy(normals: np.ndarray) -> np.ndarray:
    """Legacy compatibility wrapper."""
    import warnings
    warnings.warn(
        "Direct function call deprecated. Use compute_verticality(normals, mode='cpu') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return core_compute_verticality(normals)


# Export unified API
__all__ = [
    'FeatureMode',
    'compute_verticality',
    'extract_geometric_features', 
    'compute_all_features',
    'compute_normals',
    'compute_curvature',
]