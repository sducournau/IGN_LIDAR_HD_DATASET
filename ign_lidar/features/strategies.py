"""
Unified feature computation strategies.

This module consolidates 8+ feature computer implementations into
a clean Strategy pattern with automatic hardware-aware selection.

Strategy Classes:
    - BaseFeatureStrategy: Abstract base class for all strategies
    - CPUStrategy: CPU-based computation (NumPy + scikit-learn)
    - GPUStrategy: GPU-based computation (CuPy + cuML)
    - GPUChunkedStrategy: GPU with chunking for large datasets
    - BoundaryAwareStrategy: Wrapper for boundary-aware processing

Usage:
    # Automatic selection
    strategy = BaseFeatureStrategy.auto_select(n_points=1_000_000)
    features = strategy.compute(points, intensities, rgb, nir)
    
    # Manual selection
    strategy = GPUChunkedStrategy(chunk_size=5_000_000, batch_size=250_000)
    features = strategy.compute(points, intensities, rgb, nir)

Author: IGN LiDAR HD Development Team
Date: October 21, 2025
Version: 3.1.0-dev (Week 2 refactoring)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Literal, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureComputeMode:
    """Feature computation modes."""
    CPU = "cpu"
    GPU = "gpu"
    GPU_CHUNKED = "gpu_chunked"
    AUTO = "auto"


class BaseFeatureStrategy(ABC):
    """
    Abstract base class for all feature computation strategies.
    
    All strategies must implement the compute() method which returns
    a dictionary of computed features.
    
    Attributes:
        k_neighbors (int): Number of neighbors for geometric features
        radius (float): Search radius for neighbor queries
        verbose (bool): Enable verbose logging
    """
    
    def __init__(
        self,
        k_neighbors: int = 20,
        radius: float = 1.0,
        verbose: bool = False
    ):
        """
        Initialize feature strategy.
        
        Args:
            k_neighbors: Number of neighbors for local features
            radius: Search radius for neighbor queries (meters)
            verbose: Enable detailed logging
        """
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.verbose = verbose
        self.logger = logger
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
    
    @abstractmethod
    def compute(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute features using this strategy.
        
        Args:
            points: (N, 3) array of XYZ coordinates
            intensities: (N,) array of intensity values (optional)
            rgb: (N, 3) array of RGB values (optional)
            nir: (N,) array of near-infrared values (optional)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Dictionary mapping feature names to arrays:
            - 'normals': (N, 3) surface normals
            - 'curvature': (N,) curvature values
            - 'planarity': (N,) planarity scores
            - 'linearity': (N,) linearity scores
            - 'rgb_*': RGB-derived features (if rgb provided)
            - 'ndvi': (N,) NDVI values (if nir provided)
        """
        pass
    
    def compute_geometric_features(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute geometric features (normals, curvature, planarity, etc.)
        
        This is a convenience method that calls compute() with geometric-only parameters.
        
        Args:
            points: (N, 3) array of XYZ coordinates
            intensities: (N,) array of intensity values (optional)
            
        Returns:
            Dictionary with geometric feature arrays
        """
        return self.compute(points, intensities, rgb=None, nir=None)
    
    def compute_rgb_features(
        self,
        rgb: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute RGB-based features.
        
        Args:
            rgb: (N, 3) array of RGB values [0-255]
            
        Returns:
            Dictionary with RGB feature arrays:
            - 'rgb_mean': Mean RGB per point
            - 'rgb_std': RGB standard deviation
            - 'rgb_range': RGB range
        """
        # Default implementation - can be overridden
        return {}
    
    def compute_ndvi(
        self,
        nir: np.ndarray,
        red: np.ndarray
    ) -> np.ndarray:
        """
        Compute NDVI (Normalized Difference Vegetation Index).
        
        Args:
            nir: (N,) near-infrared band values
            red: (N,) red band values
            
        Returns:
            (N,) array of NDVI values [-1, 1]
        """
        # Default implementation
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi = np.nan_to_num(ndvi, nan=0.0)
        return ndvi
    
    @staticmethod
    def auto_select(
        n_points: int,
        mode: str = "auto",
        has_gpu: Optional[bool] = None,
        **kwargs
    ) -> 'BaseFeatureStrategy':
        """
        Automatically select optimal strategy based on data size and hardware.
        
        Selection rules:
        - mode='cpu': Force CPU strategy
        - mode='gpu': Force GPU strategy (falls back to CPU if no GPU)
        - mode='gpu_chunked': Force GPU chunked (falls back to CPU if no GPU)
        - mode='auto': Intelligent selection based on data size and hardware
        
        Auto mode rules:
        - < 1M points: CPU (fast enough, no GPU overhead)
        - 1-10M points + GPU: GPU single batch
        - > 10M points + GPU: GPU chunked
        - No GPU: CPU (fallback)
        
        Args:
            n_points: Number of points to process
            mode: Compute mode ('cpu', 'gpu', 'gpu_chunked', 'auto')
            has_gpu: GPU availability (auto-detected if None)
            **kwargs: Additional parameters passed to strategy constructor
            
        Returns:
            Selected feature strategy instance
            
        Examples:
            >>> # Auto-select for 5M points
            >>> strategy = BaseFeatureStrategy.auto_select(n_points=5_000_000)
            >>> # Force CPU
            >>> strategy = BaseFeatureStrategy.auto_select(n_points=5_000_000, mode='cpu')
        """
        # Lazy imports to avoid circular dependencies
        from .strategy_cpu import CPUStrategy
        
        # Detect GPU if not specified
        if has_gpu is None:
            has_gpu = _detect_gpu()
        
        # Handle explicit mode requests
        if mode == "cpu":
            logger.info(f"Using CPU strategy (forced) for {n_points:,} points")
            return CPUStrategy(**kwargs)
        
        elif mode == "gpu":
            if not has_gpu:
                logger.warning("GPU requested but not available, falling back to CPU")
                return CPUStrategy(**kwargs)
            try:
                from .strategy_gpu import GPUStrategy
                logger.info(f"Using GPU strategy (forced) for {n_points:,} points")
                return GPUStrategy(**kwargs)
            except ImportError:
                logger.warning("GPU strategy not available, falling back to CPU")
                return CPUStrategy(**kwargs)
        
        elif mode == "gpu_chunked":
            if not has_gpu:
                logger.warning("GPU chunked requested but GPU not available, falling back to CPU")
                return CPUStrategy(**kwargs)
            try:
                from .strategy_gpu_chunked import GPUChunkedStrategy
                logger.info(f"Using GPU chunked strategy (forced) for {n_points:,} points")
                return GPUChunkedStrategy(**kwargs)
            except ImportError:
                logger.warning("GPU chunked strategy not available, falling back to CPU")
                return CPUStrategy(**kwargs)
        
        elif mode == "auto":
            # Automatic selection based on data size and hardware
            
            # Small datasets: always CPU (GPU overhead not worth it)
            if n_points < 1_000_000:
                logger.info(f"Auto-selected CPU strategy for {n_points:,} points (< 1M)")
                return CPUStrategy(**kwargs)
            
            # No GPU: CPU fallback
            if not has_gpu:
                logger.info(f"Auto-selected CPU strategy for {n_points:,} points (no GPU)")
                return CPUStrategy(**kwargs)
            
            # Medium datasets with GPU: single-batch GPU
            if n_points < 10_000_000:
                try:
                    from .strategy_gpu import GPUStrategy
                    logger.info(f"Auto-selected GPU strategy for {n_points:,} points (1M-10M)")
                    return GPUStrategy(**kwargs)
                except ImportError:
                    logger.warning("GPU strategy not available, falling back to CPU")
                    return CPUStrategy(**kwargs)
            
            # Large datasets with GPU: chunked GPU
            else:
                try:
                    from .strategy_gpu_chunked import GPUChunkedStrategy
                    logger.info(f"Auto-selected GPU chunked strategy for {n_points:,} points (> 10M)")
                    return GPUChunkedStrategy(**kwargs)
                except ImportError:
                    logger.warning("GPU chunked strategy not available, falling back to CPU")
                    return CPUStrategy(**kwargs)
        
        else:
            raise ValueError(f"Unknown compute mode: {mode}. Use 'cpu', 'gpu', 'gpu_chunked', or 'auto'")
    
    @staticmethod
    def get_available_strategies() -> Dict[str, bool]:
        """
        Check which strategies are available.
        
        Returns:
            Dictionary mapping strategy names to availability:
            - 'cpu': Always True
            - 'gpu': True if CuPy available
            - 'gpu_chunked': True if CuPy available
        """
        availability = {
            'cpu': True,  # Always available
            'gpu': False,
            'gpu_chunked': False
        }
        
        # Check GPU availability
        has_gpu = _detect_gpu()
        if has_gpu:
            try:
                from .strategy_gpu import GPUStrategy
                availability['gpu'] = True
            except ImportError:
                pass
            
            try:
                from .strategy_gpu_chunked import GPUChunkedStrategy
                availability['gpu_chunked'] = True
            except ImportError:
                pass
        
        return availability
    
    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.__class__.__name__}(k_neighbors={self.k_neighbors}, radius={self.radius})"


def _detect_gpu() -> bool:
    """
    Detect if GPU is available for computation.
    
    Checks for CuPy availability and tries to create a test array.
    
    Returns:
        True if GPU is available and functional, False otherwise
    """
    try:
        import cupy as cp
        # Try to create a small test array
        test = cp.array([1.0, 2.0, 3.0])
        _ = test.sum()  # Force computation
        return True
    except Exception as e:
        logger.debug(f"GPU not available: {e}")
        return False


def _get_gpu_memory_info() -> Tuple[int, int]:
    """
    Get GPU memory information.
    
    Returns:
        Tuple of (free_memory, total_memory) in bytes
        Returns (0, 0) if GPU not available
    """
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        free = mempool.free_bytes()
        total = mempool.total_bytes()
        return (free, total)
    except:
        return (0, 0)


def estimate_optimal_batch_size(
    n_points: int,
    point_size_bytes: int = 12,  # 3 floats * 4 bytes
    memory_fraction: float = 0.5
) -> int:
    """
    Estimate optimal batch size based on available GPU memory.
    
    Args:
        n_points: Total number of points to process
        point_size_bytes: Memory size per point in bytes
        memory_fraction: Fraction of available memory to use (0-1)
        
    Returns:
        Estimated optimal batch size
    """
    free_memory, _ = _get_gpu_memory_info()
    
    if free_memory == 0:
        # No GPU or can't determine memory
        return min(250_000, n_points)
    
    # Calculate how many points fit in available memory
    usable_memory = free_memory * memory_fraction
    points_per_batch = int(usable_memory / point_size_bytes)
    
    # Clamp to reasonable range
    points_per_batch = max(50_000, min(points_per_batch, 1_000_000))
    
    return points_per_batch


# Export public API
__all__ = [
    'BaseFeatureStrategy',
    'FeatureComputeMode',
    'estimate_optimal_batch_size',
]
