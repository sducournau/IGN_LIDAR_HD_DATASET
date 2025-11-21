"""
GPU-Accelerated Normalization Utilities for RGB/NIR Data.

This module provides efficient normalization functions with automatic GPU
acceleration when available, falling back to CPU NumPy operations.

Consolidates all normalization code across the codebase into a single,
tested, and optimized implementation.

Author: IGN LiDAR HD Processing Library
Date: 2025-11-21
"""

import logging
from typing import Union, Optional
import numpy as np

from ..core.gpu import GPUManager

logger = logging.getLogger(__name__)

# GPU availability check (centralized)
_gpu_manager = GPUManager()
GPU_AVAILABLE = _gpu_manager.gpu_available
cp = None

if GPU_AVAILABLE:
    import cupy as cp
    logger.debug("CuPy available - GPU normalization enabled")
else:
    logger.debug("CuPy not available - using CPU normalization")


def normalize_uint8_to_float(
    data: Union[np.ndarray, "cp.ndarray"],
    use_gpu: bool = False,
    inplace: bool = False
) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Normalize uint8 data [0, 255] to float32 [0.0, 1.0].
    
    Supports both CPU (NumPy) and GPU (CuPy) arrays with automatic
    device detection and fallback.
    
    Args:
        data: Input array with uint8 values [0, 255]
        use_gpu: Whether to use GPU acceleration (requires CuPy)
        inplace: Whether to modify the input array (must be float32)
    
    Returns:
        Normalized float32 array [0.0, 1.0]
    
    Raises:
        ValueError: If inplace=True but data is not float32
        RuntimeError: If GPU requested but not available
    
    Example:
        >>> rgb = np.array([0, 127, 255], dtype=np.uint8)
        >>> rgb_norm = normalize_uint8_to_float(rgb)
        >>> rgb_norm
        array([0.0, 0.498, 1.0], dtype=float32)
        
        >>> # GPU acceleration (if available)
        >>> rgb_norm_gpu = normalize_uint8_to_float(rgb, use_gpu=True)
    
    Note:
        - Automatically detects CuPy arrays and processes on GPU
        - Falls back to CPU if GPU fails or unavailable
        - Output is always float32 for consistency
    """
    # Input validation
    if data is None or data.size == 0:
        raise ValueError("Input data cannot be None or empty")
    
    # Check if input is already a GPU array
    is_gpu_array = GPU_AVAILABLE and isinstance(data, cp.ndarray)
    
    # Determine processing mode
    should_use_gpu = (use_gpu or is_gpu_array) and GPU_AVAILABLE
    
    try:
        if should_use_gpu:
            return _normalize_gpu(data, inplace)
        else:
            return _normalize_cpu(data, inplace)
            
    except Exception as e:
        if should_use_gpu:
            logger.warning(
                f"GPU normalization failed: {e}. Falling back to CPU."
            )
            # Convert to NumPy if it was a CuPy array
            if is_gpu_array:
                data = cp.asnumpy(data)
            return _normalize_cpu(data, inplace)
        else:
            raise


def _normalize_cpu(
    data: np.ndarray,
    inplace: bool = False
) -> np.ndarray:
    """
    CPU implementation of uint8 to float32 normalization.
    
    Args:
        data: NumPy array with uint8 values
        inplace: Whether to modify input array
    
    Returns:
        Normalized float32 NumPy array
    """
    if inplace:
        if data.dtype != np.float32:
            raise ValueError(
                f"Inplace normalization requires float32 input, got {data.dtype}"
            )
        data /= 255.0
        return data
    else:
        return data.astype(np.float32) / 255.0


def _normalize_gpu(
    data: Union[np.ndarray, "cp.ndarray"],
    inplace: bool = False
) -> "cp.ndarray":
    """
    GPU implementation of uint8 to float32 normalization using CuPy.
    
    Args:
        data: NumPy or CuPy array with uint8 values
        inplace: Whether to modify input array
    
    Returns:
        Normalized float32 CuPy array
    
    Raises:
        RuntimeError: If GPU operation fails
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for normalization")
    
    # Convert to GPU array if needed
    if isinstance(data, np.ndarray):
        data_gpu = cp.asarray(data)
    else:
        data_gpu = data
    
    if inplace:
        if data_gpu.dtype != cp.float32:
            raise ValueError(
                f"Inplace normalization requires float32 input, got {data_gpu.dtype}"
            )
        data_gpu /= 255.0
        return data_gpu
    else:
        return data_gpu.astype(cp.float32) / 255.0


def denormalize_float_to_uint8(
    data: Union[np.ndarray, "cp.ndarray"],
    clip: bool = True,
    use_gpu: bool = False
) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Denormalize float32 data [0.0, 1.0] back to uint8 [0, 255].
    
    Supports both CPU (NumPy) and GPU (CuPy) arrays with automatic
    device detection.
    
    Args:
        data: Input array with float32 values [0.0, 1.0]
        clip: Whether to clip values to [0, 1] before conversion
        use_gpu: Whether to use GPU acceleration (requires CuPy)
    
    Returns:
        Denormalized uint8 array [0, 255]
    
    Example:
        >>> rgb_norm = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> rgb = denormalize_float_to_uint8(rgb_norm)
        >>> rgb
        array([0, 127, 255], dtype=uint8)
    
    Note:
        - Automatically detects CuPy arrays and processes on GPU
        - Falls back to CPU if GPU fails or unavailable
        - Clipping prevents overflow from out-of-range values
    """
    # Input validation
    if data is None or data.size == 0:
        raise ValueError("Input data cannot be None or empty")
    
    # Check if input is already a GPU array
    is_gpu_array = GPU_AVAILABLE and isinstance(data, cp.ndarray)
    
    # Determine processing mode
    should_use_gpu = (use_gpu or is_gpu_array) and GPU_AVAILABLE
    
    try:
        if should_use_gpu:
            return _denormalize_gpu(data, clip)
        else:
            return _denormalize_cpu(data, clip)
            
    except Exception as e:
        if should_use_gpu:
            logger.warning(
                f"GPU denormalization failed: {e}. Falling back to CPU."
            )
            # Convert to NumPy if it was a CuPy array
            if is_gpu_array:
                data = cp.asnumpy(data)
            return _denormalize_cpu(data, clip)
        else:
            raise


def _denormalize_cpu(
    data: np.ndarray,
    clip: bool = True
) -> np.ndarray:
    """
    CPU implementation of float32 to uint8 denormalization.
    
    Args:
        data: NumPy array with float32 values [0.0, 1.0]
        clip: Whether to clip to valid range
    
    Returns:
        Denormalized uint8 NumPy array
    """
    if clip:
        data = np.clip(data, 0.0, 1.0)
    
    return (data * 255.0).astype(np.uint8)


def _denormalize_gpu(
    data: Union[np.ndarray, "cp.ndarray"],
    clip: bool = True
) -> "cp.ndarray":
    """
    GPU implementation of float32 to uint8 denormalization using CuPy.
    
    Args:
        data: NumPy or CuPy array with float32 values [0.0, 1.0]
        clip: Whether to clip to valid range
    
    Returns:
        Denormalized uint8 CuPy array
    
    Raises:
        RuntimeError: If GPU operation fails
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for denormalization")
    
    # Convert to GPU array if needed
    if isinstance(data, np.ndarray):
        data_gpu = cp.asarray(data)
    else:
        data_gpu = data
    
    if clip:
        data_gpu = cp.clip(data_gpu, 0.0, 1.0)
    
    return (data_gpu * 255.0).astype(cp.uint8)


def normalize_rgb(
    rgb: Union[np.ndarray, "cp.ndarray"],
    use_gpu: bool = False
) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Normalize RGB data from uint8 [0, 255] to float32 [0.0, 1.0].
    
    Convenience wrapper for normalize_uint8_to_float() with RGB-specific
    logging and validation.
    
    Args:
        rgb: RGB array with shape [..., 3] and uint8 values
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Normalized RGB float32 array [0.0, 1.0]
    
    Raises:
        ValueError: If RGB shape is invalid (last dimension must be 3)
    
    Example:
        >>> rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        >>> rgb_norm = normalize_rgb(rgb)
        >>> rgb_norm.shape
        (2, 3)
        >>> rgb_norm.dtype
        dtype('float32')
    """
    if rgb.shape[-1] != 3:
        raise ValueError(
            f"RGB array must have shape [..., 3], got {rgb.shape}"
        )
    
    return normalize_uint8_to_float(rgb, use_gpu=use_gpu)


def normalize_nir(
    nir: Union[np.ndarray, "cp.ndarray"],
    use_gpu: bool = False
) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Normalize NIR data from uint8 [0, 255] to float32 [0.0, 1.0].
    
    Convenience wrapper for normalize_uint8_to_float() with NIR-specific
    logging and validation.
    
    Args:
        nir: NIR array with uint8 values
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Normalized NIR float32 array [0.0, 1.0]
    
    Example:
        >>> nir = np.array([0, 127, 255], dtype=np.uint8)
        >>> nir_norm = normalize_nir(nir)
        >>> nir_norm
        array([0.0, 0.498, 1.0], dtype=float32)
    """
    return normalize_uint8_to_float(nir, use_gpu=use_gpu)


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available for normalization.
    
    Returns:
        bool: True if CuPy is installed and GPU is detected
    
    Example:
        >>> if is_gpu_available():
        ...     rgb_norm = normalize_rgb(rgb, use_gpu=True)
        ... else:
        ...     rgb_norm = normalize_rgb(rgb, use_gpu=False)
    """
    return GPU_AVAILABLE
