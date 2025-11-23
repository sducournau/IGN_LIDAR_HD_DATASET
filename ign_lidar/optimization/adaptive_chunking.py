"""
Adaptive Chunking for GPU Processing

Automatically determines optimal chunk sizes based on:
- Available GPU memory
- Point cloud characteristics
- Target memory usage

This module implements recommendations from the November 2025 code audit
to prevent GPU OOM errors and optimize performance.

Author: IGN LiDAR HD Team
Date: November 23, 2025
Version: 3.1.0+
"""

import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def auto_chunk_size(
    points_shape: Tuple[int, int],
    target_memory_usage: float = 0.7,
    safety_factor: float = 0.8,
    min_chunk_size: int = 100_000,
    max_chunk_size: int = 10_000_000,
    feature_count: Optional[int] = None,
    use_gpu: bool = True
) -> int:
    """
    Calculate optimal chunk size based on available GPU memory.
    
    This function prevents GPU OOM errors by automatically sizing chunks
    based on hardware capabilities and data characteristics.
    
    Args:
        points_shape: Shape of points array (N, features)
        target_memory_usage: Target GPU memory usage ratio (0-1, default: 0.7)
            - 0.5: Conservative (safest, slower)
            - 0.7: Balanced (recommended)
            - 0.9: Aggressive (fastest, risk OOM)
        safety_factor: Additional safety margin (0-1, default: 0.8)
        min_chunk_size: Minimum chunk size in points (default: 100k)
        max_chunk_size: Maximum chunk size in points (default: 10M)
        feature_count: Number of features to compute (for better estimation)
        use_gpu: Whether GPU will be used (if False, returns larger chunks)
    
    Returns:
        Optimal chunk size (number of points)
        
    Examples:
        >>> # Basic usage
        >>> chunk_size = auto_chunk_size((5_000_000, 3))
        >>> print(f"Processing {5_000_000} points in chunks of {chunk_size}")
        
        >>> # Conservative for complex features
        >>> chunk_size = auto_chunk_size(
        ...     (10_000_000, 3),
        ...     target_memory_usage=0.5,
        ...     feature_count=38  # LOD3 features
        ... )
        
        >>> # Aggressive for simple features
        >>> chunk_size = auto_chunk_size(
        ...     (10_000_000, 3),
        ...     target_memory_usage=0.9,
        ...     feature_count=12  # LOD2 features
        ... )
    
    Note:
        - For CPU processing (use_gpu=False), returns max_chunk_size
        - Accounts for intermediate arrays (KNN indices, covariance, etc.)
        - Includes safety margin to prevent OOM
    """
    if not use_gpu:
        # CPU mode: no strict memory constraints
        return max_chunk_size
    
    try:
        from ign_lidar.core.gpu import GPUManager
        
        gpu = GPUManager()
        
        if not gpu.gpu_available:
            logger.warning("GPU not available, using max chunk size for CPU")
            return max_chunk_size
        
        # Get available GPU memory
        try:
            available_memory_gb = gpu.memory.get_available_memory() if gpu.memory else None
        except Exception:
            available_memory_gb = None
        
        if not available_memory_gb or available_memory_gb <= 0:
            logger.warning("Could not detect GPU memory, using conservative chunk size")
            return min_chunk_size
        
        # Calculate memory requirements per point
        point_size_bytes = points_shape[1] * 4  # float32
        
        # Estimate memory for intermediate computations
        # - Input points: 1x
        # - KNN indices: 1x (int32, ~30 neighbors)
        # - Covariance matrices: 9x (3x3 float32)
        # - Eigenvalues/vectors: 6x (3 eigenvalues + 9 eigenvectors)
        # - Features: (feature_count or 20) * 1x
        feature_multiplier = feature_count if feature_count else 20
        memory_multiplier = 1 + 1 + 9 + 6 + feature_multiplier
        
        total_memory_per_point = point_size_bytes * memory_multiplier
        
        # Calculate chunk size
        usable_memory_bytes = (
            available_memory_gb * 1024**3 * 
            target_memory_usage * 
            safety_factor
        )
        
        chunk_size = int(usable_memory_bytes / total_memory_per_point)
        
        # Clamp to min/max bounds
        chunk_size = max(min_chunk_size, min(chunk_size, max_chunk_size))
        
        logger.info(
            f"Auto-chunking: {chunk_size:,} points/chunk "
            f"(GPU memory: {available_memory_gb:.1f}GB, "
            f"target: {target_memory_usage*100:.0f}%)"
        )
        
        return chunk_size
        
    except Exception as e:
        logger.warning(
            f"Error calculating auto chunk size: {e}. "
            f"Using conservative default: {min_chunk_size:,}"
        )
        return min_chunk_size


def estimate_gpu_memory_required(
    num_points: int,
    num_features: int = 3,
    feature_count: int = 20,
    k_neighbors: int = 30
) -> float:
    """
    Estimate GPU memory required for processing.
    
    Args:
        num_points: Number of points to process
        num_features: Input features per point (default: 3 for XYZ)
        feature_count: Number of output features to compute
        k_neighbors: Number of neighbors for KNN
    
    Returns:
        Estimated memory in GB
        
    Example:
        >>> required_gb = estimate_gpu_memory_required(5_000_000, feature_count=38)
        >>> print(f"Need ~{required_gb:.1f}GB GPU memory")
    """
    # Memory breakdown (in bytes per point)
    input_memory = num_features * 4  # float32
    knn_indices = k_neighbors * 4  # int32
    covariance = 9 * 4  # 3x3 matrix float32
    eigen = 12 * 4  # 3 eigenvalues + 9 eigenvectors float32
    features = feature_count * 4  # output features float32
    
    total_per_point = input_memory + knn_indices + covariance + eigen + features
    total_bytes = num_points * total_per_point
    
    # Convert to GB with overhead
    total_gb = (total_bytes / 1024**3) * 1.2  # 20% overhead
    
    return total_gb


def get_recommended_strategy(
    num_points: int,
    available_memory_gb: Optional[float] = None
) -> str:
    """
    Recommend processing strategy based on dataset size and GPU memory.
    
    Args:
        num_points: Number of points in dataset
        available_memory_gb: Available GPU memory (auto-detected if None)
    
    Returns:
        Recommended strategy: 'gpu', 'gpu_chunked', or 'cpu'
        
    Example:
        >>> strategy = get_recommended_strategy(15_000_000)
        >>> print(f"Use {strategy} strategy")
        Use gpu_chunked strategy
    """
    try:
        from ign_lidar.core.gpu import GPUManager
        
        gpu = GPUManager()
        
        if not gpu.gpu_available:
            return 'cpu'
        
        if available_memory_gb is None:
            try:
                available_memory_gb = gpu.memory.get_available_memory() if gpu.memory else 8.0
            except Exception:
                available_memory_gb = 8.0  # Default assumption
        
        # Estimate memory required
        required_memory = estimate_gpu_memory_required(num_points)
        
        # Decision logic
        if required_memory < available_memory_gb * 0.7:
            # Fits comfortably in memory
            if num_points < 10_000_000:
                return 'gpu'
            else:
                return 'gpu_chunked'  # Large dataset benefits from chunking
        elif required_memory < available_memory_gb * 2.0:
            # Requires chunking
            return 'gpu_chunked'
        else:
            # Too large even for chunking
            logger.warning(
                f"Dataset requires {required_memory:.1f}GB but only "
                f"{available_memory_gb:.1f}GB available. "
                "Consider using CPU or reducing dataset size."
            )
            return 'cpu'
            
    except Exception as e:
        logger.warning(f"Error determining strategy: {e}. Defaulting to CPU.")
        return 'cpu'


def calculate_optimal_chunk_count(
    num_points: int,
    chunk_size: int
) -> int:
    """
    Calculate optimal number of chunks for processing.
    
    Ensures chunks are as even as possible to avoid
    imbalanced processing.
    
    Args:
        num_points: Total number of points
        chunk_size: Target chunk size
    
    Returns:
        Optimal number of chunks
        
    Example:
        >>> chunks = calculate_optimal_chunk_count(5_234_567, 1_000_000)
        >>> print(f"Process in {chunks} chunks")
        Process in 6 chunks
    """
    if num_points <= chunk_size:
        return 1
    
    # Calculate base number of chunks
    num_chunks = int(np.ceil(num_points / chunk_size))
    
    # Verify last chunk isn't too small
    last_chunk_size = num_points % chunk_size
    if last_chunk_size > 0 and last_chunk_size < chunk_size * 0.3:
        # Last chunk is too small, redistribute
        num_chunks -= 1
    
    return max(1, num_chunks)


__all__ = [
    'auto_chunk_size',
    'estimate_gpu_memory_required',
    'get_recommended_strategy',
    'calculate_optimal_chunk_count',
]
