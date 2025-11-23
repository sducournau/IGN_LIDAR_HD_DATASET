"""
GPU Safety Checks and Memory Pre-validation

This module provides safety utilities to prevent GPU OOM errors
and ensure operations can complete successfully before execution.

Key Features:
- Pre-execution memory availability checks
- Automatic fallback to CPU or chunked processing
- Clear error messages with actionable recommendations
- Integration with adaptive chunking

Author: IGN LiDAR HD Development Team
Date: November 23, 2025 (Phase 3)
Version: 3.8.0
"""

import logging
from typing import Optional, Tuple, Callable, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Recommended processing strategy based on memory analysis."""
    GPU_DIRECT = "gpu_direct"  # Fits in GPU memory, process directly
    GPU_CHUNKED = "gpu_chunked"  # Requires chunking for GPU
    CPU = "cpu"  # GPU insufficient, use CPU
    ERROR = "error"  # Cannot process (data too large even for CPU)


@dataclass
class MemoryCheckResult:
    """
    Result of GPU memory availability check.
    
    Attributes:
        can_proceed: Whether operation can proceed on GPU
        strategy: Recommended processing strategy
        available_gb: Available GPU memory in GB
        required_gb: Required GPU memory in GB
        utilization: Expected GPU memory utilization (0-1)
        chunk_size: Recommended chunk size if chunking needed
        error_message: Error/warning message if applicable
        recommendations: List of actionable recommendations
    """
    can_proceed: bool
    strategy: ProcessingStrategy
    available_gb: float
    required_gb: float
    utilization: float
    chunk_size: Optional[int] = None
    error_message: Optional[str] = None
    recommendations: list = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


def check_gpu_memory_safe(
    points_shape: Tuple[int, ...],
    feature_count: int = 20,
    k_neighbors: int = 30,
    target_utilization: float = 0.7,
    safety_margin: float = 0.8
) -> MemoryCheckResult:
    """
    Check if GPU has sufficient memory for processing.
    
    This function performs a comprehensive check BEFORE starting
    GPU processing to prevent OOM errors and provide clear guidance.
    
    Args:
        points_shape: Shape of point cloud (N, features)
        feature_count: Number of features to compute
        k_neighbors: Number of neighbors for KNN
        target_utilization: Target GPU memory usage (0-1)
        safety_margin: Additional safety factor (0-1)
    
    Returns:
        MemoryCheckResult with recommendations
        
    Example:
        >>> result = check_gpu_memory_safe((5_000_000, 3), feature_count=38)
        >>> if result.can_proceed:
        ...     # Proceed with GPU processing
        ...     process_on_gpu(points)
        >>> elif result.strategy == ProcessingStrategy.GPU_CHUNKED:
        ...     # Use chunked processing
        ...     process_in_chunks(points, chunk_size=result.chunk_size)
        >>> else:
        ...     # Fall back to CPU
        ...     process_on_cpu(points)
    """
    try:
        from ign_lidar.core.gpu import GPUManager
        from ign_lidar.optimization.adaptive_chunking import (
            estimate_gpu_memory_required,
            auto_chunk_size
        )
        
        gpu = GPUManager()
        
        # Check if GPU is available
        if not gpu.gpu_available:
            return MemoryCheckResult(
                can_proceed=False,
                strategy=ProcessingStrategy.CPU,
                available_gb=0.0,
                required_gb=0.0,
                utilization=0.0,
                error_message="GPU not available",
                recommendations=[
                    "Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads",
                    "Install CuPy: pip install cupy-cuda11x (or cuda12x)",
                    "Or use CPU mode: set use_gpu=False in config"
                ]
            )
        
        # Get available memory
        try:
            available_gb = gpu.memory.get_available_memory()
        except Exception:
            available_gb = 8.0  # Default assumption
            logger.warning("Could not detect GPU memory, assuming 8GB")
        
        # Estimate required memory
        num_points = points_shape[0]
        required_gb = estimate_gpu_memory_required(
            num_points=num_points,
            num_features=points_shape[1] if len(points_shape) > 1 else 3,
            feature_count=feature_count,
            k_neighbors=k_neighbors
        )
        
        # Calculate utilization
        utilization = required_gb / available_gb if available_gb > 0 else 1.0
        
        # Determine strategy
        if utilization <= target_utilization * safety_margin:
            # Fits comfortably in GPU memory
            return MemoryCheckResult(
                can_proceed=True,
                strategy=ProcessingStrategy.GPU_DIRECT,
                available_gb=available_gb,
                required_gb=required_gb,
                utilization=utilization,
                recommendations=[
                    f"Processing {num_points:,} points directly on GPU",
                    f"Expected GPU usage: {utilization*100:.1f}%"
                ]
            )
        
        elif utilization <= 2.0:
            # Can fit with chunking
            chunk_size = auto_chunk_size(
                points_shape=points_shape,
                target_memory_usage=target_utilization,
                safety_factor=safety_margin,
                feature_count=feature_count
            )
            
            num_chunks = int(np.ceil(num_points / chunk_size))
            
            return MemoryCheckResult(
                can_proceed=True,
                strategy=ProcessingStrategy.GPU_CHUNKED,
                available_gb=available_gb,
                required_gb=required_gb,
                utilization=utilization,
                chunk_size=chunk_size,
                error_message=(
                    f"Dataset requires {required_gb:.1f}GB but only "
                    f"{available_gb:.1f}GB available. Using chunked processing."
                ),
                recommendations=[
                    f"Processing {num_points:,} points in {num_chunks} chunks",
                    f"Chunk size: {chunk_size:,} points",
                    f"Expected time increase: ~{num_chunks*10:.0f}%"
                ]
            )
        
        else:
            # Too large even for chunking, recommend CPU
            return MemoryCheckResult(
                can_proceed=False,
                strategy=ProcessingStrategy.CPU,
                available_gb=available_gb,
                required_gb=required_gb,
                utilization=utilization,
                error_message=(
                    f"Dataset requires {required_gb:.1f}GB but only "
                    f"{available_gb:.1f}GB available. GPU processing not feasible."
                ),
                recommendations=[
                    "Use CPU processing (set use_gpu=False)",
                    "Reduce dataset size (spatial filtering or downsampling)",
                    "Upgrade to GPU with more memory",
                    "Process tiles separately instead of full dataset"
                ]
            )
    
    except Exception as e:
        logger.error(f"Error checking GPU memory: {e}")
        return MemoryCheckResult(
            can_proceed=False,
            strategy=ProcessingStrategy.ERROR,
            available_gb=0.0,
            required_gb=0.0,
            utilization=0.0,
            error_message=f"Memory check failed: {e}",
            recommendations=["Check GPU configuration", "Use CPU mode as fallback"]
        )


def compute_features_safe(
    points: np.ndarray,
    compute_func: Callable,
    use_gpu: bool = True,
    feature_count: int = 20,
    k_neighbors: int = 30,
    **compute_kwargs
) -> Tuple[np.ndarray, MemoryCheckResult]:
    """
    Safely compute features with automatic GPU memory checking.
    
    This function wraps feature computation with pre-execution memory
    checks and automatic fallback to chunked or CPU processing.
    
    Args:
        points: Point cloud array (N, features)
        compute_func: Function to compute features
            Signature: compute_func(points, use_gpu=True, **kwargs) -> features
        use_gpu: Whether to attempt GPU processing
        feature_count: Number of features to compute
        k_neighbors: Number of neighbors for KNN
        **compute_kwargs: Additional arguments for compute_func
    
    Returns:
        Tuple of (features, memory_check_result)
        
    Example:
        >>> def my_compute_features(points, use_gpu=True, k=30):
        ...     # Your feature computation logic
        ...     return features
        >>> 
        >>> features, check = compute_features_safe(
        ...     points,
        ...     my_compute_features,
        ...     use_gpu=True,
        ...     k=30
        ... )
        >>> 
        >>> if check.strategy == ProcessingStrategy.GPU_CHUNKED:
        ...     logger.info("Used chunked processing")
    """
    # Perform memory check
    check_result = check_gpu_memory_safe(
        points_shape=points.shape,
        feature_count=feature_count,
        k_neighbors=k_neighbors
    )
    
    # Log recommendations
    if check_result.error_message:
        logger.warning(check_result.error_message)
    
    for rec in check_result.recommendations:
        logger.info(f"  → {rec}")
    
    # Execute based on strategy
    if check_result.strategy == ProcessingStrategy.GPU_DIRECT:
        # Direct GPU processing
        logger.info("✓ Sufficient GPU memory - processing directly")
        features = compute_func(points, use_gpu=True, **compute_kwargs)
        
    elif check_result.strategy == ProcessingStrategy.GPU_CHUNKED:
        # Chunked GPU processing
        logger.info(f"⚠ Using chunked GPU processing ({check_result.chunk_size:,} points/chunk)")
        features = _compute_chunked(
            points,
            compute_func,
            chunk_size=check_result.chunk_size,
            use_gpu=True,
            **compute_kwargs
        )
        
    elif check_result.strategy == ProcessingStrategy.CPU:
        # CPU fallback
        logger.info("⚠ Falling back to CPU processing")
        features = compute_func(points, use_gpu=False, **compute_kwargs)
        
    else:
        # Error - cannot proceed
        raise RuntimeError(
            f"Cannot process dataset: {check_result.error_message}\n"
            "Recommendations:\n" +
            "\n".join(f"  - {rec}" for rec in check_result.recommendations)
        )
    
    return features, check_result


def _compute_chunked(
    points: np.ndarray,
    compute_func: Callable,
    chunk_size: int,
    use_gpu: bool = True,
    **compute_kwargs
) -> np.ndarray:
    """
    Process point cloud in chunks.
    
    Args:
        points: Point cloud array
        compute_func: Function to compute features
        chunk_size: Number of points per chunk
        use_gpu: Use GPU for each chunk
        **compute_kwargs: Additional arguments
    
    Returns:
        Combined feature array
    """
    num_points = len(points)
    num_chunks = int(np.ceil(num_points / chunk_size))
    
    results = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_points)
        
        chunk = points[start_idx:end_idx]
        
        logger.info(
            f"Processing chunk {i+1}/{num_chunks} "
            f"({end_idx-start_idx:,} points)"
        )
        
        chunk_features = compute_func(chunk, use_gpu=use_gpu, **compute_kwargs)
        results.append(chunk_features)
    
    # Concatenate results
    return np.vstack(results)


def get_gpu_status_report() -> str:
    """
    Get detailed GPU status report.
    
    Returns:
        Formatted string with GPU information
        
    Example:
        >>> print(get_gpu_status_report())
        GPU Status Report
        =================
        GPU Available: Yes
        GPU Memory: 16.0 GB available
        CuPy Available: Yes
        cuML Available: Yes
        Recommended Strategy: GPU Direct
    """
    try:
        from ign_lidar.core.gpu import GPUManager
        
        gpu = GPUManager()
        
        lines = [
            "GPU Status Report",
            "=" * 50,
            f"GPU Available: {'Yes' if gpu.gpu_available else 'No'}",
        ]
        
        if gpu.gpu_available:
            try:
                available_memory = gpu.memory.get_available_memory()
                total_memory = gpu.memory.get_total_memory() if hasattr(gpu.memory, 'get_total_memory') else None
                
                lines.append(f"Available Memory: {available_memory:.1f} GB")
                if total_memory:
                    lines.append(f"Total Memory: {total_memory:.1f} GB")
                    utilization = (total_memory - available_memory) / total_memory * 100
                    lines.append(f"Current Utilization: {utilization:.1f}%")
            except Exception:
                lines.append("Available Memory: Unknown")
            
            lines.append(f"CuPy Available: {'Yes' if gpu.cupy_available else 'No'}")
            lines.append(f"cuML Available: {'Yes' if gpu.cuml_available else 'No'}")
            
            # Estimate capacity
            try:
                available_memory = gpu.memory.get_available_memory()
                
                # Estimate max points for different scenarios
                max_points_lod2 = int(available_memory * 1024**3 / (3 * 4 * 40))  # ~40x multiplier for LOD2
                max_points_lod3 = int(available_memory * 1024**3 / (3 * 4 * 60))  # ~60x multiplier for LOD3
                
                lines.append("")
                lines.append("Estimated Capacity:")
                lines.append(f"  LOD2 features: ~{max_points_lod2:,} points")
                lines.append(f"  LOD3 features: ~{max_points_lod3:,} points")
            except Exception:
                pass
        else:
            lines.append("")
            lines.append("GPU not available - using CPU mode")
            lines.append("")
            lines.append("To enable GPU:")
            lines.append("  1. Install CUDA Toolkit")
            lines.append("  2. Install CuPy: pip install cupy-cuda11x")
            lines.append("  3. Install cuML: conda install -c rapidsai cuml")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error generating GPU status report: {e}"


__all__ = [
    'check_gpu_memory_safe',
    'compute_features_safe',
    'get_gpu_status_report',
    'ProcessingStrategy',
    'MemoryCheckResult',
]
