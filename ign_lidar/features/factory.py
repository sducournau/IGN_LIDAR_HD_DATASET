"""
Factory for creating feature computers based on configuration.

Simplifies feature computer selection and initialization, providing a clean
interface for choosing between CPU, GPU, chunked, and boundary-aware processing.

Example:
    >>> from ign_lidar.features.factory import FeatureComputerFactory
    >>> computer = FeatureComputerFactory.create(use_gpu=True, use_chunked=True)
    >>> features = computer.compute_features(points, classification)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray
import logging

logger = logging.getLogger(__name__)

__all__ = ['FeatureComputerFactory', 'BaseFeatureComputer']


class BaseFeatureComputer(ABC):
    """
    Base class for all feature computers.
    
    Provides common interface for different feature computation strategies:
    - CPU-based computation
    - GPU-based computation (CuPy)
    - GPU chunked computation (for large point clouds)
    - Boundary-aware computation (for tile stitching)
    """

    def __init__(self, k_neighbors: int = 20, **kwargs):
        """
        Initialize feature computer.
        
        Args:
            k_neighbors: Number of neighbors for feature computation
            **kwargs: Additional parameters specific to implementation
        """
        self.k_neighbors = k_neighbors
        self.kwargs = kwargs

    @abstractmethod
    def compute_features(
        self,
        points: NDArray[np.float32],
        classification: NDArray[np.uint8],
        **kwargs
    ) -> Dict[str, NDArray]:
        """
        Compute features for point cloud.

        Args:
            points: Point cloud coordinates (N, 3)
            classification: Point classifications (N,)
            **kwargs: Additional parameters

        Returns:
            Dictionary of computed features with keys like:
            - 'normals': (N, 3) surface normals
            - 'curvature': (N,) curvature values
            - 'verticality': (N,) verticality measure
            - etc.
        """
        pass

    def is_available(self) -> bool:
        """Check if this feature computer is available."""
        return True


class CPUFeatureComputer(BaseFeatureComputer):
    """CPU-based feature computation using NumPy."""

    def compute_features(
        self,
        points: NDArray[np.float32],
        classification: NDArray[np.uint8],
        **kwargs
    ) -> Dict[str, NDArray]:
        """Compute features using CPU."""
        from .features import compute_all_features_optimized
        
        # compute_all_features_optimized returns (normals, curvature, height, geo_features)
        normals, curvature, height, geo_features = compute_all_features_optimized(
            points,
            classification,
            k=self.k_neighbors,
            **kwargs
        )
        
        # Return in dict format expected by processor
        result = {
            'normals': normals,
            'curvature': curvature,
            'height_above_ground': height,
        }
        result.update(geo_features)
        
        return result


class GPUFeatureComputer(BaseFeatureComputer):
    """GPU-based feature computation using CuPy."""

    def __init__(self, k_neighbors: int = 20, **kwargs):
        super().__init__(k_neighbors, **kwargs)
        self._check_gpu_availability()

    def _check_gpu_availability(self):
        """Check if GPU is available."""
        try:
            import cupy as cp
            self._gpu_available = True
            logger.info(f"GPU available: {cp.cuda.runtime.getDeviceCount()} device(s)")
        except (ImportError, Exception) as e:
            self._gpu_available = False
            logger.warning(f"GPU not available: {e}")

    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self._gpu_available

    def compute_features(
        self,
        points: NDArray[np.float32],
        classification: NDArray[np.uint8],
        **kwargs
    ) -> Dict[str, NDArray]:
        """Compute features using GPU."""
        if not self.is_available():
            logger.warning("GPU not available, falling back to CPU")
            return CPUFeatureComputer(self.k_neighbors, **self.kwargs).compute_features(
                points, classification, **kwargs
            )
        
        from .features_gpu import GPUFeatureComputer as Impl
        
        # Create GPU computer
        computer = Impl(use_gpu=True, batch_size=250000)
        
        # Compute all features
        normals, curvature, height, geo_features = computer.compute_all_features(
            points=points,
            classification=classification,
            k=self.k_neighbors,
            include_building_features=kwargs.get('include_building_features', False)
        )
        
        # Return in dict format
        result = {
            'normals': normals,
            'curvature': curvature,
            'height_above_ground': height,
        }
        result.update(geo_features)
        
        return result


class GPUChunkedFeatureComputer(BaseFeatureComputer):
    """GPU-based chunked feature computation for large point clouds."""

    def __init__(
        self,
        k_neighbors: int = 20,
        gpu_batch_size: int = 1_000_000,
        **kwargs
    ):
        super().__init__(k_neighbors, **kwargs)
        self.gpu_batch_size = gpu_batch_size
        self._check_gpu_availability()

    def _check_gpu_availability(self):
        """Check if GPU is available."""
        try:
            import cupy as cp
            self._gpu_available = True
            logger.info(f"GPU chunked processing: {cp.cuda.runtime.getDeviceCount()} device(s)")
        except (ImportError, Exception) as e:
            self._gpu_available = False
            logger.warning(f"GPU not available for chunked processing: {e}")

    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self._gpu_available

    def compute_features(
        self,
        points: NDArray[np.float32],
        classification: NDArray[np.uint8],
        **kwargs
    ) -> Dict[str, NDArray]:
        """Compute features using GPU with chunked processing."""
        if not self.is_available():
            logger.warning("GPU not available, falling back to CPU")
            return CPUFeatureComputer(self.k_neighbors, **self.kwargs).compute_features(
                points, classification, **kwargs
            )
        
        from .features_gpu_chunked import GPUChunkedFeatureComputer as Impl
        
        # Create computer with chunk_size instead of gpu_batch_size
        computer = Impl(
            chunk_size=self.gpu_batch_size,
            use_gpu=True,
            show_progress=False,
            auto_optimize=True
        )
        
        # Compute features with k parameter
        normals, curvature, height, geo_features = computer.compute_all_features_chunked(
            points=points,
            classification=classification,
            k=self.k_neighbors,
            radius=kwargs.get('radius', None)
        )
        
        # Return in expected dict format
        result = {
            'normals': normals,
            'curvature': curvature,
            'height_above_ground': height,
        }
        result.update(geo_features)
        
        return result


class BoundaryAwareFeatureComputer(BaseFeatureComputer):
    """
    Boundary-aware feature computation for tile stitching.
    
    Handles features near tile boundaries to ensure consistency
    across tile borders.
    """

    def __init__(
        self,
        k_neighbors: int = 20,
        buffer_size: float = 10.0,
        tile_bounds: Optional[Tuple[float, float, float, float]] = None,
        **kwargs
    ):
        super().__init__(k_neighbors, **kwargs)
        self.buffer_size = buffer_size
        self.tile_bounds = tile_bounds

    def compute_features(
        self,
        points: NDArray[np.float32],
        classification: NDArray[np.uint8],
        **kwargs
    ) -> Dict[str, NDArray]:
        """Compute features with boundary awareness."""
        from .features_boundary import compute_boundary_aware_features
        
        return compute_boundary_aware_features(
            points,
            classification,
            k_neighbors=self.k_neighbors,
            buffer_size=self.buffer_size,
            tile_bounds=self.tile_bounds,
            **kwargs
        )


class FeatureComputerFactory:
    """
    Factory for creating feature computers.
    
    Provides a clean interface for selecting the appropriate feature computation
    strategy based on configuration.
    
    Example:
        >>> factory = FeatureComputerFactory()
        >>> 
        >>> # CPU processing
        >>> computer = factory.create(use_gpu=False)
        >>> 
        >>> # GPU processing
        >>> computer = factory.create(use_gpu=True)
        >>> 
        >>> # GPU chunked for large clouds
        >>> computer = factory.create(use_gpu=True, use_chunked=True)
        >>> 
        >>> # Boundary-aware for stitching
        >>> computer = factory.create(use_boundary_aware=True, buffer_size=10.0)
    """

    @staticmethod
    def create(
        use_gpu: bool = False,
        use_chunked: bool = False,
        use_boundary_aware: bool = False,
        gpu_batch_size: int = 1_000_000,
        k_neighbors: int = 20,
        buffer_size: float = 10.0,
        tile_bounds: Optional[Tuple[float, float, float, float]] = None,
        **kwargs
    ) -> BaseFeatureComputer:
        """
        Create appropriate feature computer based on configuration.

        Args:
            use_gpu: Enable GPU acceleration (requires CuPy)
            use_chunked: Use chunked processing for large point clouds
            use_boundary_aware: Enable boundary-aware processing for tile stitching
            gpu_batch_size: Batch size for GPU chunked processing
            k_neighbors: Number of neighbors for feature computation
            buffer_size: Buffer size for boundary-aware processing
            tile_bounds: Tile boundaries (xmin, ymin, xmax, ymax) for boundary-aware
            **kwargs: Additional parameters for feature computers

        Returns:
            Configured feature computer instance

        Example:
            >>> factory = FeatureComputerFactory()
            >>> computer = factory.create(use_gpu=True, k_neighbors=20)
            >>> features = computer.compute_features(points, classification)
        """
        # Boundary-aware takes precedence (used for tile stitching)
        if use_boundary_aware:
            logger.info(f"Creating boundary-aware feature computer (buffer={buffer_size}m)")
            return BoundaryAwareFeatureComputer(
                k_neighbors=k_neighbors,
                buffer_size=buffer_size,
                tile_bounds=tile_bounds,
                **kwargs
            )

        # GPU processing
        if use_gpu:
            # Check if chunked processing is requested
            if use_chunked:
                logger.info(f"Creating GPU chunked feature computer (batch_size={gpu_batch_size:,})")
                computer = GPUChunkedFeatureComputer(
                    k_neighbors=k_neighbors,
                    gpu_batch_size=gpu_batch_size,
                    **kwargs
                )
                if not computer.is_available():
                    logger.warning("GPU not available, falling back to CPU")
                    return CPUFeatureComputer(k_neighbors=k_neighbors, **kwargs)
                return computer
            else:
                logger.info("Creating GPU feature computer")
                computer = GPUFeatureComputer(k_neighbors=k_neighbors, **kwargs)
                if not computer.is_available():
                    logger.warning("GPU not available, falling back to CPU")
                    return CPUFeatureComputer(k_neighbors=k_neighbors, **kwargs)
                return computer

        # CPU processing (default)
        logger.info("Creating CPU feature computer")
        return CPUFeatureComputer(k_neighbors=k_neighbors, **kwargs)

    @staticmethod
    def list_available() -> Dict[str, bool]:
        """
        List available feature computers and their availability.

        Returns:
            Dictionary mapping computer name to availability status

        Example:
            >>> availability = FeatureComputerFactory.list_available()
            >>> print(availability)
            {'cpu': True, 'gpu': False, 'gpu_chunked': False, 'boundary_aware': True}
        """
        availability = {
            'cpu': True,  # Always available
            'boundary_aware': True,  # Always available
        }

        # Check GPU availability
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            availability['gpu'] = True
            availability['gpu_chunked'] = True
        except (ImportError, Exception):
            availability['gpu'] = False
            availability['gpu_chunked'] = False

        return availability

    @staticmethod
    def get_recommended(
        num_points: int,
        has_gpu: bool = False,
        stitching_enabled: bool = False
    ) -> str:
        """
        Get recommended feature computer based on scenario.

        Args:
            num_points: Number of points in the cloud
            has_gpu: Whether GPU is available
            stitching_enabled: Whether tile stitching is enabled

        Returns:
            Recommended computer type: 'cpu', 'gpu', 'gpu_chunked', 'boundary_aware'

        Example:
            >>> recommendation = FeatureComputerFactory.get_recommended(
            ...     num_points=5_000_000,
            ...     has_gpu=True,
            ...     stitching_enabled=False
            ... )
            >>> print(recommendation)
            'gpu_chunked'
        """
        if stitching_enabled:
            return 'boundary_aware'
        
        if has_gpu:
            if num_points > 2_000_000:
                return 'gpu_chunked'
            return 'gpu'
        
        return 'cpu'
