"""
Automatic Mode Selection for IGN LIDAR HD Feature Computation

This module provides intelligent mode selection based on:
- Point cloud size
- GPU availability
- Memory constraints
- Feature requirements
- User preferences

Author: Simon Ducournau / GitHub Copilot
Date: October 18, 2025
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.gpu import GPUManager

logger = logging.getLogger(__name__)


class ComputationMode(Enum):
    """Available computation modes for feature extraction."""
    
    CPU = "cpu"
    GPU = "gpu"
    GPU_CHUNKED = "gpu_chunked"
    BOUNDARY = "boundary"


class ModeSelector:
    """
    Intelligent mode selection for point cloud feature computation.
    
    Selects the optimal computation mode based on various factors:
    - Point cloud size
    - Available GPU memory
    - Feature requirements
    - User constraints
    
    Examples:
        >>> selector = ModeSelector()
        >>> mode = selector.select_mode(num_points=1_000_000)
        >>> print(mode)
        ComputationMode.GPU
        
        >>> # Force CPU mode
        >>> mode = selector.select_mode(num_points=1_000_000, force_cpu=True)
        >>> print(mode)
        ComputationMode.CPU
    """
    
    # Thresholds for mode selection (can be tuned based on hardware)
    SMALL_CLOUD_THRESHOLD = 500_000      # < 500K points: CPU or GPU
    MEDIUM_CLOUD_THRESHOLD = 5_000_000   # 500K-5M: GPU preferred
    LARGE_CLOUD_THRESHOLD = 10_000_000   # 5M-10M: GPU or GPU Chunked
    # > 10M: GPU Chunked required
    
    # Memory requirements (approximate, in bytes per point)
    CPU_MEMORY_PER_POINT = 200          # ~200 bytes/point for CPU
    GPU_MEMORY_PER_POINT = 400          # ~400 bytes/point for GPU (more features)
    GPU_CHUNKED_MEMORY_PER_POINT = 100  # ~100 bytes/point (chunked processing)
    
    def __init__(
        self,
        gpu_memory_gb: Optional[float] = None,
        cpu_memory_gb: Optional[float] = None,
        prefer_gpu: bool = True
    ):
        """
        Initialize the mode selector.
        
        Args:
            gpu_memory_gb: Available GPU memory in GB (auto-detected if None)
            cpu_memory_gb: Available CPU memory in GB (auto-detected if None)
            prefer_gpu: Whether to prefer GPU modes when possible
        """
        self.prefer_gpu = prefer_gpu
        self._gpu_manager = GPUManager()
        self.gpu_available = self._gpu_manager.gpu_available
        self.gpu_memory_gb = gpu_memory_gb or self._get_gpu_memory()
        self.cpu_memory_gb = cpu_memory_gb or self._get_cpu_memory()
        
        logger.info(f"ModeSelector initialized:")
        logger.info(f"  GPU available: {self.gpu_available}")
        logger.info(f"  GPU memory: {self.gpu_memory_gb:.2f} GB")
        logger.info(f"  CPU memory: {self.cpu_memory_gb:.2f} GB")
        logger.info(f"  Prefer GPU: {self.prefer_gpu}")
    
    def _check_gpu_availability(self) -> bool:
        """
        Check GPU availability via GPUManager.
        
        DEPRECATED: Access self._gpu_manager.gpu_available directly instead.
        This method is kept for backward compatibility.
        """
        return self._gpu_manager.gpu_available
    
    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        if not self.gpu_available:
            return 0.0
        
        # ✅ Use centralized GPU info (v3.5.3 consolidation)
        from ign_lidar.core.gpu import GPUManager
        gpu = GPUManager()
        
        try:
            mem_info = gpu.get_memory_info()
            return mem_info.get('total_gb', 8.0)
        except Exception as e:
            logger.warning(f"Could not determine GPU memory: {e}")
            return 8.0  # Default assumption: 8 GB
    
    def _get_cpu_memory(self) -> float:
        """Get available CPU memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            logger.warning("psutil not available, using default CPU memory")
            return 16.0  # Default assumption: 16 GB available
    
    def select_mode(
        self,
        num_points: int,
        required_features: Optional[List[str]] = None,
        force_cpu: bool = False,
        force_gpu: bool = False,
        boundary_mode: bool = False,
        user_mode: Optional[ComputationMode] = None
    ) -> ComputationMode:
        """
        Select the optimal computation mode.
        
        Args:
            num_points: Number of points in the cloud
            required_features: List of required features (optional)
            force_cpu: Force CPU mode regardless of other factors
            force_gpu: Force GPU mode (will use chunked if needed)
            boundary_mode: Whether this is boundary computation
            user_mode: User-specified mode (overrides automatic selection)
        
        Returns:
            Selected computation mode
        
        Raises:
            ValueError: If constraints cannot be satisfied
        """
        # User override takes precedence
        if user_mode is not None:
            logger.info(f"Using user-specified mode: {user_mode.value}")
            return user_mode
        
        # Handle force flags
        if force_cpu:
            if not self._can_use_cpu(num_points):
                raise ValueError(
                    f"CPU mode forced but insufficient memory for {num_points} points"
                )
            logger.info("CPU mode forced by user")
            return ComputationMode.CPU
        
        if force_gpu:
            if not self.gpu_available:
                raise ValueError("GPU mode forced but GPU not available")
            # Select GPU or GPU_CHUNKED based on size
            if num_points > self.LARGE_CLOUD_THRESHOLD:
                logger.info("GPU Chunked mode forced (large cloud)")
                return ComputationMode.GPU_CHUNKED
            else:
                logger.info("GPU mode forced by user")
                return ComputationMode.GPU
        
        # Boundary mode has special handling
        if boundary_mode:
            return self._select_boundary_mode(num_points)
        
        # Automatic selection based on size and availability
        return self._automatic_selection(num_points, required_features)
    
    def _can_use_cpu(self, num_points: int) -> bool:
        """Check if CPU has enough memory."""
        required_gb = (num_points * self.CPU_MEMORY_PER_POINT) / (1024**3)
        return required_gb < self.cpu_memory_gb * 0.8  # Use max 80% of available
    
    def _can_use_gpu(self, num_points: int) -> bool:
        """Check if GPU has enough memory."""
        if not self.gpu_available:
            return False
        required_gb = (num_points * self.GPU_MEMORY_PER_POINT) / (1024**3)
        return required_gb < self.gpu_memory_gb * 0.8  # Use max 80% of available
    
    def _select_boundary_mode(self, num_points: int) -> ComputationMode:
        """Select mode for boundary computation."""
        # Boundary mode can use GPU acceleration if available
        if self.gpu_available and num_points > 10_000:
            logger.info(f"Selected BOUNDARY mode with GPU acceleration ({num_points} points)")
        else:
            logger.info(f"Selected BOUNDARY mode with CPU ({num_points} points)")
        return ComputationMode.BOUNDARY
    
    def _automatic_selection(
        self,
        num_points: int,
        required_features: Optional[List[str]] = None
    ) -> ComputationMode:
        """
        Automatic mode selection based on size and capabilities.
        
        Decision tree:
        1. < 500K points: CPU or GPU (prefer GPU if available)
        2. 500K - 5M: GPU if available, else CPU
        3. 5M - 10M: GPU or GPU Chunked
        4. > 10M: GPU Chunked only
        """
        # Very large clouds: GPU Chunked required
        if num_points > self.LARGE_CLOUD_THRESHOLD:
            if self.gpu_available:
                logger.info(
                    f"Selected GPU_CHUNKED mode for large cloud "
                    f"({num_points:,} points)"
                )
                return ComputationMode.GPU_CHUNKED
            else:
                # Fallback to CPU if GPU not available
                if self._can_use_cpu(num_points):
                    logger.warning(
                        f"Large cloud ({num_points:,} points) but GPU unavailable, "
                        f"using CPU (may be slow)"
                    )
                    return ComputationMode.CPU
                else:
                    raise ValueError(
                        f"Cannot process {num_points:,} points: "
                        f"GPU unavailable and insufficient CPU memory"
                    )
        
        # Large clouds: GPU or GPU Chunked
        elif num_points > self.MEDIUM_CLOUD_THRESHOLD:
            if self.gpu_available:
                if self._can_use_gpu(num_points):
                    logger.info(
                        f"Selected GPU mode for large cloud "
                        f"({num_points:,} points)"
                    )
                    return ComputationMode.GPU
                else:
                    logger.info(
                        f"Selected GPU_CHUNKED mode "
                        f"(GPU memory insufficient for {num_points:,} points)"
                    )
                    return ComputationMode.GPU_CHUNKED
            else:
                if self._can_use_cpu(num_points):
                    logger.info(
                        f"Selected CPU mode (GPU unavailable, {num_points:,} points)"
                    )
                    return ComputationMode.CPU
                else:
                    raise ValueError(
                        f"Cannot process {num_points:,} points: "
                        f"GPU unavailable and insufficient CPU memory"
                    )
        
        # Medium clouds: Prefer GPU
        elif num_points > self.SMALL_CLOUD_THRESHOLD:
            if self.gpu_available and self.prefer_gpu:
                logger.info(
                    f"Selected GPU mode for medium cloud "
                    f"({num_points:,} points)"
                )
                return ComputationMode.GPU
            else:
                logger.info(
                    f"Selected CPU mode for medium cloud "
                    f"({num_points:,} points)"
                )
                return ComputationMode.CPU
        
        # Small clouds: CPU or GPU
        else:
            if self.gpu_available and self.prefer_gpu:
                logger.info(
                    f"Selected GPU mode for small cloud "
                    f"({num_points:,} points)"
                )
                return ComputationMode.GPU
            else:
                logger.info(
                    f"Selected CPU mode for small cloud "
                    f"({num_points:,} points)"
                )
                return ComputationMode.CPU
    
    def estimate_memory_usage(
        self,
        num_points: int,
        mode: ComputationMode
    ) -> Tuple[float, float]:
        """
        Estimate memory usage for given mode.
        
        Args:
            num_points: Number of points
            mode: Computation mode
        
        Returns:
            Tuple of (estimated_memory_gb, available_memory_gb)
        """
        if mode == ComputationMode.CPU:
            memory_per_point = self.CPU_MEMORY_PER_POINT
            available = self.cpu_memory_gb
        elif mode in (ComputationMode.GPU, ComputationMode.BOUNDARY):
            memory_per_point = self.GPU_MEMORY_PER_POINT
            available = self.gpu_memory_gb
        else:  # GPU_CHUNKED
            memory_per_point = self.GPU_CHUNKED_MEMORY_PER_POINT
            available = self.gpu_memory_gb
        
        estimated = (num_points * memory_per_point) / (1024**3)
        return estimated, available
    
    def get_recommendations(
        self,
        num_points: int
    ) -> Dict[str, any]:
        """
        Get detailed recommendations for processing a point cloud.
        
        Args:
            num_points: Number of points
        
        Returns:
            Dictionary with recommendations and estimates
        """
        recommended_mode = self.select_mode(num_points)
        estimated_memory, available_memory = self.estimate_memory_usage(
            num_points, recommended_mode
        )
        
        # Estimate processing time (rough estimates based on Phase 2/3 results)
        time_estimates = {
            ComputationMode.CPU: num_points / 200_000,  # ~200K pts/sec
            ComputationMode.GPU: num_points / 1_000_000,  # ~1M pts/sec
            ComputationMode.GPU_CHUNKED: num_points / 5_000_000,  # ~5M pts/sec
            ComputationMode.BOUNDARY: num_points / 500_000,  # ~500K pts/sec
        }
        
        return {
            "recommended_mode": recommended_mode.value,
            "num_points": num_points,
            "estimated_memory_gb": round(estimated_memory, 2),
            "available_memory_gb": round(available_memory, 2),
            "memory_utilization_pct": round(
                (estimated_memory / available_memory) * 100, 1
            ) if available_memory > 0 else 0,
            "estimated_time_seconds": round(
                time_estimates[recommended_mode], 1
            ),
            "gpu_available": self.gpu_available,
            "alternative_modes": self._get_alternative_modes(num_points),
        }
    
    def _get_alternative_modes(
        self,
        num_points: int
    ) -> List[Dict[str, any]]:
        """Get list of alternative modes with their viability."""
        alternatives = []
        
        for mode in ComputationMode:
            if mode == ComputationMode.BOUNDARY:
                continue  # Skip boundary mode (special purpose)
            
            try:
                # Check if mode is viable
                viable = True
                reason = "Available"
                
                if mode == ComputationMode.CPU:
                    if not self._can_use_cpu(num_points):
                        viable = False
                        reason = "Insufficient CPU memory"
                
                elif mode in (ComputationMode.GPU, ComputationMode.GPU_CHUNKED):
                    if not self.gpu_available:
                        viable = False
                        reason = "GPU not available"
                    elif mode == ComputationMode.GPU and not self._can_use_gpu(num_points):
                        viable = False
                        reason = "Insufficient GPU memory (use GPU_CHUNKED)"
                
                est_memory, avail_memory = self.estimate_memory_usage(num_points, mode)
                
                alternatives.append({
                    "mode": mode.value,
                    "viable": viable,
                    "reason": reason,
                    "estimated_memory_gb": round(est_memory, 2),
                    "available_memory_gb": round(avail_memory, 2),
                })
            except Exception:
                continue
        
        return alternatives


def get_mode_selector(
    gpu_memory_gb: Optional[float] = None,
    cpu_memory_gb: Optional[float] = None,
    prefer_gpu: bool = True
) -> ModeSelector:
    """
    Factory function to get a configured ModeSelector instance.
    
    Args:
        gpu_memory_gb: Available GPU memory in GB (auto-detected if None)
        cpu_memory_gb: Available CPU memory in GB (auto-detected if None)
        prefer_gpu: Whether to prefer GPU modes when possible
    
    Returns:
        Configured ModeSelector instance
    
    Example:
        >>> selector = get_mode_selector()
        >>> mode = selector.select_mode(num_points=2_000_000)
    """
    return ModeSelector(
        gpu_memory_gb=gpu_memory_gb,
        cpu_memory_gb=cpu_memory_gb,
        prefer_gpu=prefer_gpu
    )
