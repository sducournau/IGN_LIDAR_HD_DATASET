"""
Unified feature computation API.

This module provides a single, consolidated API dispatcher that routes
feature computation requests to the appropriate implementation (CPU, GPU, etc.).
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union
from enum import Enum

from .geometric import extract_geometric_features
from .curvature import compute_curvature
from .utils import validate_points

# Import optimized implementations
try:
    from .features import compute_normals, compute_all_features as compute_all_features_optimized
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    # Fallback to standard implementation
    from .normals import compute_normals
    compute_all_features_optimized = None

logger = logging.getLogger(__name__)


class ComputeMode(Enum):
    """Feature computation modes."""
    CPU = "cpu"
    GPU = "gpu" 
    GPU_CHUNKED = "gpu_chunked"
    BOUNDARY_AWARE = "boundary_aware"
    AUTO = "auto"  # Automatic selection based on data size


def compute_all_features(
    points: np.ndarray,
    classification: np.ndarray,
    mode: Union[str, ComputeMode] = ComputeMode.AUTO,
    k_neighbors: int = 10,
    radius: Optional[float] = None,
    gpu_chunk_size: int = 1_000_000,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Unified feature computation API replacing all compute_all_features* variants.
    
    This function provides a single entry point for computing all features,
    automatically selecting the best computation strategy based on the mode
    and data characteristics.
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] ASPRS classification codes
        mode: Computation mode (cpu, gpu, gpu_chunked, boundary_aware, auto)
        k_neighbors: Number of neighbors for feature computation
        radius: Search radius in meters (recommended over k_neighbors)
        gpu_chunk_size: Chunk size for GPU processing
        **kwargs: Additional arguments passed to specific computers
        
    Returns:
        normals: [N, 3] surface normal vectors
        curvature: [N] curvature values
        height: [N] height above ground
        features: Dictionary of geometric features
        
    Raises:
        ValueError: If points array is invalid
        ImportError: If GPU mode requested but GPU libraries unavailable
        
    Example:
        >>> normals, curvature, height, features = compute_all_features(
        ...     points, classification, mode='gpu', k_neighbors=20
        ... )
        >>> print(f"Computed {len(features)} feature types")
    """
    # Validate inputs
    validate_points(points)
    
    if isinstance(mode, str):
        mode = ComputeMode(mode.lower())
    
    n_points = len(points)
    logger.info(f"Computing features for {n_points:,} points using {mode.value} mode")
    
    # Auto mode selection
    if mode == ComputeMode.AUTO:
        mode = _select_optimal_mode(n_points, **kwargs)
        logger.info(f"Auto-selected {mode.value} mode for {n_points:,} points")
    
    # Dispatch to appropriate implementation
    if mode == ComputeMode.CPU:
        return _compute_all_features_cpu(
            points, classification, k_neighbors, radius, **kwargs
        )
    elif mode == ComputeMode.GPU:
        return _compute_all_features_gpu(
            points, classification, k_neighbors, radius, **kwargs
        )
    elif mode == ComputeMode.GPU_CHUNKED:
        return _compute_all_features_gpu_chunked(
            points, classification, k_neighbors, radius, gpu_chunk_size, **kwargs
        )
    elif mode == ComputeMode.BOUNDARY_AWARE:
        return _compute_all_features_boundary_aware(
            points, classification, k_neighbors, radius, **kwargs
        )
    else:
        raise ValueError(f"Unknown computation mode: {mode}")


def _select_optimal_mode(n_points: int, **kwargs) -> ComputeMode:
    """
    Automatically select the optimal computation mode based on data size.
    
    Args:
        n_points: Number of points
        **kwargs: Additional context (gpu_available, use_boundary_aware, etc.)
        
    Returns:
        Optimal computation mode
    """
    # Check GPU availability
    gpu_available = kwargs.get('gpu_available', _check_gpu_available())
    
    # Boundary aware mode takes priority if requested
    if kwargs.get('use_boundary_aware', False):
        return ComputeMode.BOUNDARY_AWARE
    
    # Large datasets with GPU
    if n_points > 5_000_000 and gpu_available:
        return ComputeMode.GPU_CHUNKED
    
    # Medium datasets with GPU  
    if n_points > 500_000 and gpu_available:
        return ComputeMode.GPU
    
    # Small datasets or no GPU
    return ComputeMode.CPU


def _check_gpu_available() -> bool:
    """Check if GPU libraries are available."""
    try:
        import cupy
        return cupy.cuda.is_available()
    except ImportError:
        return False


def _compute_all_features_cpu(
    points: np.ndarray,
    classification: np.ndarray,
    k_neighbors: int,
    radius: Optional[float],
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """CPU implementation using optimized core modules."""
    
    # Use optimized single-pass computation if available
    if OPTIMIZED_AVAILABLE and compute_all_features_optimized is not None:
        try:
            features_dict = compute_all_features_optimized(
                points, k_neighbors=k_neighbors, compute_advanced=True
            )
            
            # Extract components
            normals = features_dict['normals']
            curvature = features_dict['curvature']
            
            # Compute height above ground
            ground_mask = classification == 2
            if np.any(ground_mask):
                ground_height = np.median(points[ground_mask, 2])
                height = points[:, 2] - ground_height
            else:
                height = np.zeros(len(points), dtype=np.float32)
            
            # Remove normals and curvature from features_dict to avoid duplication
            geo_features = {k: v for k, v in features_dict.items() 
                           if k not in ['normals', 'curvature', 'normal_x', 'normal_y', 'normal_z']}
            
            return normals, curvature, height, geo_features
        except Exception as e:
            logger.warning(f"Optimized computation failed ({e}), falling back to standard")
    
    # Fallback to standard implementation
    normals, eigenvalues = compute_normals(points, k_neighbors=k_neighbors)
    
    # Compute curvature from eigenvalues
    curvature = compute_curvature(eigenvalues)
    
    # Compute height above ground
    ground_mask = classification == 2
    if np.any(ground_mask):
        ground_height = np.median(points[ground_mask, 2])
        height = points[:, 2] - ground_height
    else:
        height = np.zeros(len(points), dtype=np.float32)
    
    # Compute geometric features
    geo_features = extract_geometric_features(
        points, normals, k=k_neighbors, radius=radius
    )
    
    return normals, curvature, height, geo_features


def _compute_all_features_gpu(
    points: np.ndarray,
    classification: np.ndarray,
    k_neighbors: int,
    radius: Optional[float],
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """GPU implementation with fallback to CPU."""
    try:
        # Import GPU modules
        from ..features_gpu import GPUFeatureComputer
        
        computer = GPUFeatureComputer(use_gpu=True)
        return computer.compute_all_features(
            points, classification, k=k_neighbors, **kwargs
        )
    except ImportError as e:
        logger.warning(f"GPU not available ({e}), falling back to CPU")
        return _compute_all_features_cpu(
            points, classification, k_neighbors, radius, **kwargs
        )


def _compute_all_features_gpu_chunked(
    points: np.ndarray,
    classification: np.ndarray,
    k_neighbors: int,
    radius: Optional[float],
    chunk_size: int,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """GPU chunked implementation with fallback."""
    try:
        # Import GPU chunked modules
        from ..features_gpu_chunked import GPUChunkedFeatureComputer
        
        computer = GPUChunkedFeatureComputer()
        return computer.compute_all_features_chunked(
            points, classification, k=k_neighbors, radius=radius, **kwargs
        )
    except ImportError as e:
        logger.warning(f"GPU chunked not available ({e}), falling back to CPU")
        return _compute_all_features_cpu(
            points, classification, k_neighbors, radius, **kwargs
        )


def _compute_all_features_boundary_aware(
    points: np.ndarray,
    classification: np.ndarray,
    k_neighbors: int,
    radius: Optional[float],
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Boundary-aware implementation with fallback."""
    try:
        # Import boundary-aware modules
        from ..features_boundary import BoundaryAwareFeatureComputer
        
        computer = BoundaryAwareFeatureComputer(k_neighbors=k_neighbors)
        # Note: BoundaryAwareFeatureComputer has different API
        features = computer.compute_features(points)
        
        # Extract components to match expected return format
        normals = features.get('normals', np.zeros((len(points), 3)))
        curvature = features.get('curvature', np.zeros(len(points)))
        height = features.get('height', np.zeros(len(points)))
        
        # Remove these from geo_features to avoid duplication
        geo_features = {k: v for k, v in features.items() 
                       if k not in ['normals', 'curvature', 'height']}
        
        return normals, curvature, height, geo_features
        
    except ImportError as e:
        logger.warning(f"Boundary-aware not available ({e}), falling back to CPU")
        return _compute_all_features_cpu(
            points, classification, k_neighbors, radius, **kwargs
        )