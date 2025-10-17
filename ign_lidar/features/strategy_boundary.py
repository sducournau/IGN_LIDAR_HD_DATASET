"""
Boundary-aware feature computation strategy (wrapper).

This strategy wraps another strategy to handle tile boundaries by
loading neighboring tiles and computing features in overlapping regions.

Essential for multi-tile processing to avoid edge artifacts.

Author: IGN LiDAR HD Development Team
Date: October 21, 2025
Version: 3.1.0-dev (Week 2 refactoring)
"""

from typing import Dict, Optional
import numpy as np
import logging

from .strategies import BaseFeatureStrategy

logger = logging.getLogger(__name__)


class BoundaryAwareStrategy(BaseFeatureStrategy):
    """
    Boundary-aware feature computation (decorator pattern).
    
    This strategy wraps another strategy (CPU, GPU, or GPU Chunked)
    and adds boundary-aware processing for multi-tile workflows.
    
    Usage:
    - Wraps base strategy (CPU/GPU/GPU Chunked)
    - Loads neighbor tiles
    - Computes features in overlapping regions
    - Stitches results together
    
    This prevents edge artifacts when processing tiled point clouds.
    
    Attributes:
        base_strategy (BaseFeatureStrategy): The underlying strategy to use
        boundary_buffer (float): Buffer zone around tile edges (meters)
    """
    
    def __init__(
        self,
        base_strategy: BaseFeatureStrategy,
        boundary_buffer: float = 10.0,
        verbose: bool = False
    ):
        """
        Initialize boundary-aware strategy.
        
        Args:
            base_strategy: The underlying feature strategy (CPU/GPU/GPU Chunked)
            boundary_buffer: Buffer zone around tile edges in meters
            verbose: Enable detailed logging
        """
        # Initialize with same parameters as base strategy
        super().__init__(
            k_neighbors=base_strategy.k_neighbors,
            radius=base_strategy.radius,
            verbose=verbose
        )
        
        self.base_strategy = base_strategy
        self.boundary_buffer = boundary_buffer
        
        if verbose:
            logger.info(
                f"Initialized boundary-aware strategy: "
                f"base={base_strategy.__class__.__name__}, buffer={boundary_buffer}m"
            )
    
    def compute(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
        tile_bounds: Optional[tuple] = None,
        neighbor_points: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute features with boundary awareness.
        
        Args:
            points: (N, 3) main tile points
            intensities: (N,) intensity values (optional)
            rgb: (N, 3) RGB values (optional)
            nir: (N,) near-infrared values (optional)
            tile_bounds: (xmin, ymin, xmax, ymax) tile boundaries
            neighbor_points: Dict mapping direction to neighbor tile points
                            e.g., {'north': points_n, 'south': points_s, ...}
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with feature arrays (same keys as base strategy)
        """
        n_points = len(points)
        
        if self.verbose:
            logger.info(
                f"Computing boundary-aware features for {n_points:,} points "
                f"(buffer={self.boundary_buffer}m)"
            )
        
        # If no tile bounds or neighbors provided, use base strategy directly
        if tile_bounds is None or neighbor_points is None or len(neighbor_points) == 0:
            if self.verbose:
                logger.info("No boundary info provided, using base strategy directly")
            return self.base_strategy.compute(
                points, intensities, rgb, nir, **kwargs
            )
        
        # Extract tile boundaries
        xmin, ymin, xmax, ymax = tile_bounds
        
        # Identify boundary points (within buffer of tile edge)
        boundary_mask = (
            (points[:, 0] < xmin + self.boundary_buffer) |  # West edge
            (points[:, 0] > xmax - self.boundary_buffer) |  # East edge
            (points[:, 1] < ymin + self.boundary_buffer) |  # South edge
            (points[:, 1] > ymax - self.boundary_buffer)    # North edge
        )
        
        n_boundary = np.sum(boundary_mask)
        n_interior = n_points - n_boundary
        
        if self.verbose:
            logger.info(
                f"Tile classification: {n_interior:,} interior, {n_boundary:,} boundary points "
                f"({100*n_boundary/n_points:.1f}% in buffer zone)"
            )
        
        # If no boundary points, use base strategy directly
        if n_boundary == 0:
            if self.verbose:
                logger.info("No boundary points, using base strategy")
            return self.base_strategy.compute(
                points, intensities, rgb, nir, **kwargs
            )
        
        # Merge main points with neighbor points for boundary region
        all_points = [points]
        for direction, neighbor_pts in neighbor_points.items():
            if neighbor_pts is not None and len(neighbor_pts) > 0:
                all_points.append(neighbor_pts)
                if self.verbose:
                    logger.debug(f"Added {len(neighbor_pts):,} points from {direction} neighbor")
        
        merged_points = np.vstack(all_points)
        n_merged = len(merged_points)
        
        if self.verbose:
            logger.info(
                f"Merged point cloud: {n_merged:,} points "
                f"({n_merged - n_points:,} from neighbors)"
            )
        
        # Compute features on merged point cloud
        # Note: Intensities, RGB, NIR need to be merged too (simplified here)
        all_features = self.base_strategy.compute(
            merged_points,
            intensities=None,  # Simplified: would need to merge these too
            rgb=None,
            nir=None,
            **kwargs
        )
        
        # Extract results for original points only
        result = {}
        for key, value in all_features.items():
            # Keep only features for original points (first n_points)
            result[key] = value[:n_points].astype(np.float32)
        
        if self.verbose:
            logger.info(
                f"Boundary-aware computation complete: {len(result)} feature types"
            )
        
        return result
    
    def compute_features(
        self,
        points: np.ndarray,
        classification: Optional[np.ndarray] = None,
        auto_k: bool = False,
        include_extra: bool = False,
        patch_center: Optional[np.ndarray] = None,
        mode: str = 'lod2',
        radius: Optional[float] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Adapter method for compatibility with FeatureOrchestrator.
        
        This method provides the old interface expected by the orchestrator,
        while internally calling the new compute() method or delegating
        to the base strategy's compute_features() method.
        
        Args:
            points: (N, 3) array of XYZ coordinates
            classification: (N,) array of ASPRS classification codes (optional)
            auto_k: Whether to auto-estimate k
            include_extra: Whether to include extra features
            patch_center: Center point of the patch (for distance features)
            mode: Feature mode ('lod2', 'lod3', etc.)
            radius: Search radius (overrides self.radius if provided)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with feature arrays
        """
        # Check if base_strategy has compute_features, use it if available
        if hasattr(self.base_strategy, 'compute_features'):
            return self.base_strategy.compute_features(
                points=points,
                classification=classification,
                auto_k=auto_k,
                include_extra=include_extra,
                patch_center=patch_center,
                mode=mode,
                radius=radius,
                **kwargs
            )
        else:
            # Fall back to compute() method
            rgb = kwargs.get('rgb', None)
            nir = kwargs.get('nir', None)
            intensities = kwargs.get('intensities', None)
            tile_bounds = kwargs.get('tile_bounds', None)
            neighbor_points = kwargs.get('neighbor_points', None)
            
            result = self.compute(
                points=points,
                intensities=intensities,
                rgb=rgb,
                nir=nir,
                tile_bounds=tile_bounds,
                neighbor_points=neighbor_points,
                **kwargs
            )
            
            # Add distance_to_center if patch_center provided
            if patch_center is not None:
                distances = np.linalg.norm(points - patch_center, axis=1)
                result['distance_to_center'] = distances.astype(np.float32)
            
            return result
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BoundaryAwareStrategy(base={self.base_strategy.__class__.__name__}, "
            f"buffer={self.boundary_buffer}m)"
        )


# Export
__all__ = ['BoundaryAwareStrategy']
