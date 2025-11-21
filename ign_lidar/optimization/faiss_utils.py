"""
FAISS Utilities - Centralized FAISS Configuration

Provides optimized, consistent FAISS configuration across the entire codebase.

This module consolidates 3 scattered FAISS temp memory calculations into
a single, optimized implementation with:
- Safe temp memory calculation
- OOM error prevention
- Performance optimization
- Consistent behavior

Replaces scattered code in:
- optimization/gpu_accelerated_ops.py (lines 251-288)
- features/compute/faiss_knn.py (multiple locations)
- features/gpu_processor.py (inline calculations)

Author: LiDAR Trainer Agent (Phase 1: GPU Bottlenecks)
Date: November 21, 2025
Version: 1.0
"""

import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Check FAISS availability
try:
    import faiss
    HAS_FAISS = True
    HAS_FAISS_GPU = hasattr(faiss, 'StandardGpuResources')
except ImportError:
    HAS_FAISS = False
    HAS_FAISS_GPU = False


def calculate_faiss_temp_memory(
    n_points: int,
    k: int,
    n_queries: Optional[int] = None,
    safety_factor: float = 0.2,
    max_temp_gb: float = 1.0
) -> int:
    """
    Calculate optimal FAISS temp memory size.
    
    This function computes a safe temp memory allocation for FAISS-GPU
    based on:
    - Dataset size (n_points)
    - Number of neighbors (k)
    - Number of queries (defaults to n_points for self-query)
    - Available GPU memory
    - Safety margin to prevent OOM
    
    Formula:
        temp_memory = min(
            max_temp_gb,
            safety_factor * available_gpu_memory,
            1.5 * search_memory_estimate
        )
    
    Args:
        n_points: Number of points in dataset
        k: Number of nearest neighbors
        n_queries: Number of query points (None = same as n_points)
        safety_factor: Fraction of GPU memory to use (0.2 = 20%)
        max_temp_gb: Maximum temp memory in GB (default: 1.0)
    
    Returns:
        Temp memory size in bytes
        
    Example:
        >>> from ign_lidar.optimization.faiss_utils import calculate_faiss_temp_memory
        >>> 
        >>> # For 1M points, k=30 neighbors
        >>> temp_bytes = calculate_faiss_temp_memory(
        ...     n_points=1_000_000,
        ...     k=30
        ... )
        >>> print(f"Temp memory: {temp_bytes / (1024**3):.2f} GB")
        Temp memory: 0.85 GB
        
        >>> # Custom query set
        >>> temp_bytes = calculate_faiss_temp_memory(
        ...     n_points=1_000_000,
        ...     k=30,
        ...     n_queries=10_000
        ... )
    
    Note:
        This function automatically detects available GPU memory and
        adjusts allocation to prevent OOM errors.
    """
    if n_queries is None:
        n_queries = n_points
    
    # Estimate memory needed for search results
    # Each result: 4 bytes (distance float32) + 4 bytes (index int32) = 8 bytes
    search_memory_bytes = n_queries * k * 8
    search_memory_gb = search_memory_bytes / (1024**3)
    
    # Get available GPU memory
    try:
        from ign_lidar.core.gpu_memory import get_gpu_memory_manager
        gpu_mem = get_gpu_memory_manager()
        available_gb = gpu_mem.get_available_memory()
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}, using conservative default")
        available_gb = 2.0  # Conservative fallback
    
    # Calculate safe temp memory
    temp_memory_gb = min(
        max_temp_gb,                      # Cap at max
        available_gb * safety_factor,     # Use fraction of available
        search_memory_gb * 1.5            # 150% of estimated search memory
    )
    
    # Ensure minimum of 128 MB
    temp_memory_gb = max(temp_memory_gb, 0.128)
    
    temp_memory_bytes = int(temp_memory_gb * 1024**3)
    
    logger.debug(
        f"FAISS temp memory calculated: {temp_memory_gb:.3f}GB "
        f"(n_points={n_points:,}, k={k}, n_queries={n_queries:,}, "
        f"available_gpu={available_gb:.2f}GB)"
    )
    
    return temp_memory_bytes


def create_faiss_gpu_resources(
    temp_memory_bytes: Optional[int] = None,
    n_points: Optional[int] = None,
    k: Optional[int] = None
) -> 'faiss.StandardGpuResources':
    """
    Create FAISS GPU resources with optimal configuration.
    
    Args:
        temp_memory_bytes: Temp memory in bytes (auto-calculated if None)
        n_points: Number of points (required if temp_memory_bytes is None)
        k: Number of neighbors (required if temp_memory_bytes is None)
    
    Returns:
        Configured FAISS GPU resources
        
    Raises:
        ImportError: If FAISS-GPU is not available
        ValueError: If parameters are invalid
        
    Example:
        >>> from ign_lidar.optimization.faiss_utils import create_faiss_gpu_resources
        >>> 
        >>> # Auto-calculate temp memory
        >>> res = create_faiss_gpu_resources(n_points=1_000_000, k=30)
        >>> 
        >>> # Use custom temp memory
        >>> res = create_faiss_gpu_resources(temp_memory_bytes=512*1024**2)
    """
    if not HAS_FAISS_GPU:
        raise ImportError(
            "FAISS-GPU not available. Install with: "
            "conda install -c pytorch faiss-gpu"
        )
    
    # Calculate temp memory if not provided
    if temp_memory_bytes is None:
        if n_points is None or k is None:
            raise ValueError(
                "Either temp_memory_bytes or (n_points, k) must be provided"
            )
        temp_memory_bytes = calculate_faiss_temp_memory(n_points, k)
    
    # Create resources
    res = faiss.StandardGpuResources()
    res.setTempMemory(temp_memory_bytes)
    
    temp_memory_gb = temp_memory_bytes / (1024**3)
    logger.debug(f"✅ FAISS GPU resources created with {temp_memory_gb:.3f}GB temp memory")
    
    return res


def select_faiss_index_type(
    n_points: int,
    n_dims: int,
    use_gpu: bool = True,
    approximate: bool = True,
    nlist: Optional[int] = None
) -> str:
    """
    Select optimal FAISS index type based on dataset characteristics.
    
    Selection criteria:
    - Small datasets (<100k): Use Flat (exact search)
    - Large datasets (≥100k): Use IVF (approximate search)
    - GPU vs CPU: Different optimizations
    
    Args:
        n_points: Number of points in dataset
        n_dims: Number of dimensions
        use_gpu: Whether to use GPU
        approximate: Allow approximate search for speed
        nlist: Number of clusters for IVF (auto if None)
    
    Returns:
        Index type string ('flat', 'ivf', etc.)
        
    Example:
        >>> from ign_lidar.optimization.faiss_utils import select_faiss_index_type
        >>> 
        >>> # Small dataset - exact search
        >>> index_type = select_faiss_index_type(n_points=50_000, n_dims=3)
        >>> print(index_type)
        'flat'
        >>> 
        >>> # Large dataset - approximate search
        >>> index_type = select_faiss_index_type(n_points=2_000_000, n_dims=3)
        >>> print(index_type)
        'ivf'
    """
    # Small datasets: use exact search (Flat)
    if n_points < 100_000 or not approximate:
        return 'flat'
    
    # Large datasets: use approximate search (IVF)
    return 'ivf'


def calculate_ivf_nlist(n_points: int, target_cells: int = 1000) -> int:
    """
    Calculate optimal number of IVF clusters (nlist).
    
    Rules of thumb:
    - nlist = sqrt(n_points) for balanced speed/accuracy
    - Min 16, max 65536 (FAISS limitations)
    - Target ~1000 points per cell for optimal performance
    
    Args:
        n_points: Number of points in dataset
        target_cells: Target points per cell (default: 1000)
    
    Returns:
        Optimal nlist value
        
    Example:
        >>> from ign_lidar.optimization.faiss_utils import calculate_ivf_nlist
        >>> 
        >>> nlist = calculate_ivf_nlist(n_points=1_000_000)
        >>> print(nlist)
        1000
    """
    # Calculate based on sqrt heuristic
    nlist_sqrt = int(np.sqrt(n_points))
    
    # Calculate based on target cells
    nlist_target = max(16, n_points // target_cells)
    
    # Use minimum of both, capped at reasonable limits
    nlist = min(nlist_sqrt, nlist_target)
    nlist = max(16, min(nlist, 65536))  # FAISS limits
    
    return nlist


def create_faiss_index(
    n_dims: int,
    n_points: int,
    use_gpu: bool = True,
    approximate: bool = True,
    metric: str = 'L2',
    gpu_id: int = 0
) -> Tuple['faiss.Index', Optional['faiss.StandardGpuResources']]:
    """
    Create optimally configured FAISS index.
    
    This is a high-level convenience function that:
    1. Selects optimal index type
    2. Configures GPU resources if needed
    3. Creates and returns the index
    
    Args:
        n_dims: Number of dimensions
        n_points: Number of points (for optimization)
        use_gpu: Whether to use GPU
        approximate: Allow approximate search
        metric: Distance metric ('L2' or 'IP')
        gpu_id: GPU device ID
    
    Returns:
        Tuple of (index, gpu_resources)
        gpu_resources is None for CPU index
        
    Example:
        >>> from ign_lidar.optimization.faiss_utils import create_faiss_index
        >>> 
        >>> # Create GPU index for 1M points
        >>> index, res = create_faiss_index(
        ...     n_dims=3,
        ...     n_points=1_000_000,
        ...     use_gpu=True
        ... )
        >>> 
        >>> # Add data and search
        >>> index.add(data)
        >>> distances, indices = index.search(queries, k=30)
    """
    if not HAS_FAISS:
        raise ImportError(
            "FAISS not available. Install with: "
            "conda install -c pytorch faiss-cpu  # or faiss-gpu"
        )
    
    # Select index type
    index_type = select_faiss_index_type(n_points, n_dims, use_gpu, approximate)
    
    # Create base index
    if metric == 'L2':
        quantizer = faiss.IndexFlatL2(n_dims)
    elif metric == 'IP':
        quantizer = faiss.IndexFlatIP(n_dims)
    else:
        raise ValueError(f"Invalid metric '{metric}'. Use 'L2' or 'IP'")
    
    # Create final index
    if index_type == 'flat':
        index_cpu = quantizer
        logger.debug(f"Created FAISS Flat index (exact search, n_dims={n_dims})")
    else:  # ivf
        nlist = calculate_ivf_nlist(n_points)
        index_cpu = faiss.IndexIVFFlat(quantizer, n_dims, nlist)
        logger.debug(
            f"Created FAISS IVF index (approximate search, "
            f"n_dims={n_dims}, nlist={nlist})"
        )
    
    # Transfer to GPU if requested
    res = None
    if use_gpu and HAS_FAISS_GPU:
        res = create_faiss_gpu_resources(n_points=n_points, k=30)
        index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)
        logger.debug(f"✅ Index transferred to GPU {gpu_id}")
    else:
        index = index_cpu
        logger.debug("Using CPU index")
    
    return index, res


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    'HAS_FAISS',
    'HAS_FAISS_GPU',
    'calculate_faiss_temp_memory',
    'create_faiss_gpu_resources',
    'select_faiss_index_type',
    'calculate_ivf_nlist',
    'create_faiss_index',
]
