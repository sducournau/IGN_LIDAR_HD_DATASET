"""
Tile Stitching Module for IGN LiDAR HD v2.0

This module implements seamless feature computation at tile boundaries
by creating buffer zones and enabling cross-tile neighborhood queries.

Key Features:
- Load multiple adjacent tiles
- Extract buffer zones from neighbors
- Merge core + buffer point clouds
- Compute features with cross-tile neighborhoods
- Eliminate edge artifacts at boundaries

Author: IGN LiDAR HD Team
Date: October 7, 2025
Sprint: 3 (Tile Stitching)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import laspy
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class TileStitcher:
    """
    Manages multi-tile processing with buffer zones for seamless boundaries.
    
    This class enables seamless feature computation at tile boundaries by:
    1. Loading a core tile and its neighbors
    2. Extracting buffer zones from neighbor tiles
    3. Merging core + buffer points for complete neighborhoods
    4. Computing features with cross-tile support
    
    Example:
        >>> stitcher = TileStitcher(buffer_size=10.0)
        >>> tile_data = stitcher.load_tile_with_neighbors(
        ...     tile_path=Path("tile_A.laz"),
        ...     neighbor_tiles=[Path("tile_B.laz"), Path("tile_C.laz")]
        ... )
        >>> # tile_data contains core + buffer points
    """
    
    def __init__(self, buffer_size: float = 10.0, enable_caching: bool = True):
        """
        Initialize TileStitcher.
        
        Args:
            buffer_size: Buffer zone width in meters (default: 10m)
                        Larger values = more accurate but more memory
            enable_caching: If True, cache loaded tiles to avoid re-reading
        """
        self.buffer_size = buffer_size
        self.enable_caching = enable_caching
        self._tile_cache = {} if enable_caching else None
        
        logger.info(f"TileStitcher initialized (buffer_size={buffer_size}m)")
    
    def load_tile_with_neighbors(
        self,
        tile_path: Path,
        neighbor_tiles: Optional[List[Path]] = None,
        auto_detect_neighbors: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Load a tile and extract buffer zones from its neighbors.
        
        Args:
            tile_path: Path to the main (core) tile LAZ file
            neighbor_tiles: List of paths to neighbor tile LAZ files
                           If None and auto_detect_neighbors=True, will attempt
                           to detect neighbors automatically
            auto_detect_neighbors: If True, automatically detect neighbor tiles
                                  based on filename pattern (e.g., grid coordinates)
        
        Returns:
            Dictionary with:
                - 'core_points': (N, 3) XYZ coordinates of core tile
                - 'core_features': (N, F) Additional features (intensity, etc.)
                - 'buffer_points': (M, 3) XYZ coordinates from buffer zones
                - 'buffer_features': (M, F) Buffer point features
                - 'core_mask': (N+M,) Boolean mask (True = core, False = buffer)
                - 'combined_points': (N+M, 3) Merged XYZ
                - 'combined_features': (N+M, F) Merged features
                - 'core_bounds': (xmin, ymin, xmax, ymax) Core tile bounds
                - 'num_core': N (number of core points)
                - 'num_buffer': M (number of buffer points)
        
        Raises:
            FileNotFoundError: If tile_path or neighbor tiles don't exist
            ValueError: If tile has no points or bounds cannot be determined
        """
        # Check if core tile exists
        if not tile_path.exists():
            raise FileNotFoundError(f"Core tile not found: {tile_path}")
        
        # Load core tile
        logger.info(f"Loading core tile: {tile_path.name}")
        core_data = self._load_tile(tile_path)
        
        if core_data is None or len(core_data['points']) == 0:
            raise ValueError(f"Core tile {tile_path} has no points")
        
        core_points = core_data['points']
        core_features = core_data['features']
        
        # Get core tile bounds
        core_bounds = self._compute_bounds(core_points)
        logger.info(f"Core tile bounds: {core_bounds}")
        
        # Auto-detect neighbors if requested
        if auto_detect_neighbors and neighbor_tiles is None:
            neighbor_tiles = self._detect_neighbor_tiles(tile_path)
            if neighbor_tiles:
                logger.info(f"Auto-detected {len(neighbor_tiles)} neighbor tiles")
        
        # Extract buffer zones from neighbors
        buffer_points_list = []
        buffer_features_list = []
        
        if neighbor_tiles:
            for neighbor_path in neighbor_tiles:
                if not neighbor_path.exists():
                    logger.warning(f"Neighbor tile not found: {neighbor_path}")
                    continue
                
                try:
                    buffer_data = self._extract_buffer_zone(
                        neighbor_path,
                        core_bounds,
                        self.buffer_size
                    )
                    
                    if buffer_data is not None and len(buffer_data['points']) > 0:
                        buffer_points_list.append(buffer_data['points'])
                        buffer_features_list.append(buffer_data['features'])
                        logger.info(
                            f"Extracted {len(buffer_data['points'])} buffer points "
                            f"from {neighbor_path.name}"
                        )
                except Exception as e:
                    logger.error(f"Failed to extract buffer from {neighbor_path}: {e}")
                    continue
        
        # Merge buffer zones
        if buffer_points_list:
            buffer_points = np.vstack(buffer_points_list)
            buffer_features = np.vstack(buffer_features_list)
        else:
            buffer_points = np.empty((0, 3))
            buffer_features = np.empty((0, core_features.shape[1]))
            logger.warning("No buffer points extracted from neighbors")
        
        # Combine core + buffer
        num_core = len(core_points)
        num_buffer = len(buffer_points)
        
        combined_points = np.vstack([core_points, buffer_points])
        combined_features = np.vstack([core_features, buffer_features])
        
        # Create core mask (True for core points, False for buffer)
        core_mask = np.zeros(num_core + num_buffer, dtype=bool)
        core_mask[:num_core] = True
        
        logger.info(
            f"Tile stitching complete: {num_core} core + {num_buffer} buffer "
            f"= {num_core + num_buffer} total points"
        )
        
        return {
            'core_points': core_points,
            'core_features': core_features,
            'buffer_points': buffer_points,
            'buffer_features': buffer_features,
            'core_mask': core_mask,
            'combined_points': combined_points,
            'combined_features': combined_features,
            'core_bounds': core_bounds,
            'num_core': num_core,
            'num_buffer': num_buffer
        }
    
    def detect_boundary_points(
        self,
        points: np.ndarray,
        tile_bounds: Tuple[float, float, float, float],
        threshold: float = None
    ) -> np.ndarray:
        """
        Identify points near tile boundaries.
        
        Args:
            points: (N, 3) XYZ coordinates
            tile_bounds: (xmin, ymin, xmax, ymax) tile boundaries
            threshold: Distance threshold in meters (default: buffer_size)
        
        Returns:
            (N,) Boolean mask: True = near boundary, False = interior
        """
        if threshold is None:
            threshold = self.buffer_size
        
        xmin, ymin, xmax, ymax = tile_bounds
        x, y = points[:, 0], points[:, 1]
        
        # Distance to each boundary
        dist_to_left = x - xmin
        dist_to_right = xmax - x
        dist_to_bottom = y - ymin
        dist_to_top = ymax - y
        
        # Minimum distance to any boundary
        min_dist = np.minimum.reduce([
            dist_to_left, dist_to_right, dist_to_bottom, dist_to_top
        ])
        
        # Points within threshold of any boundary
        near_boundary = min_dist <= threshold
        
        return near_boundary
    
    def build_spatial_index(
        self,
        points: np.ndarray
    ) -> KDTree:
        """
        Build KDTree spatial index for efficient neighborhood queries.
        
        Args:
            points: (N, 3) XYZ coordinates
        
        Returns:
            KDTree for spatial queries
        """
        return KDTree(points)
    
    def get_tile_bounds(self, tile_path: Path) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of a tile.
        
        Args:
            tile_path: Path to the tile LAZ file
        
        Returns:
            (xmin, ymin, xmax, ymax) tile boundaries
        
        Raises:
            FileNotFoundError: If tile doesn't exist
        """
        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")
        
        # Load from cache if available
        if self._tile_cache and tile_path in self._tile_cache:
            cached = self._tile_cache[tile_path]
            return self._compute_bounds(cached['points'])
        
        # Load tile and compute bounds
        tile_data = self._load_tile(tile_path)
        if tile_data is None or len(tile_data['points']) == 0:
            raise ValueError(f"Tile {tile_path} has no points")
        
        return self._compute_bounds(tile_data['points'])
    
    def query_cross_tile_neighbors(
        self,
        query_points: np.ndarray,
        spatial_index: KDTree,
        k_neighbors: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query k-nearest neighbors with cross-tile support.
        
        Args:
            query_points: (N, 3) Points to query
            spatial_index: KDTree built from combined (core + buffer) points
            k_neighbors: Number of neighbors to query
        
        Returns:
            distances: (N, k) Distances to neighbors
            indices: (N, k) Indices of neighbors in combined point cloud
        """
        distances, indices = spatial_index.query(
            query_points,
            k=k_neighbors,
            workers=-1  # Use all CPU cores
        )
        
        return distances, indices
    
    def _load_tile(self, tile_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """
        Load a LAZ/LAS tile and extract points + features.
        
        Args:
            tile_path: Path to LAZ/LAS file
        
        Returns:
            Dictionary with 'points' (N, 3) and 'features' (N, F)
            or None if loading fails
        """
        # Check cache
        if self.enable_caching and str(tile_path) in self._tile_cache:
            logger.debug(f"Using cached tile: {tile_path.name}")
            return self._tile_cache[str(tile_path)]
        
        try:
            las = laspy.read(str(tile_path))
            
            # Extract XYZ
            points = np.vstack([las.x, las.y, las.z]).T
            
            # Extract features
            features_list = []
            
            # Intensity
            if hasattr(las, 'intensity'):
                features_list.append(np.array(las.intensity).reshape(-1, 1))
            
            # Return number
            if hasattr(las, 'return_number'):
                features_list.append(np.array(las.return_number).reshape(-1, 1))
            
            # Classification
            if hasattr(las, 'classification'):
                features_list.append(np.array(las.classification).reshape(-1, 1))
            
            # RGB (if available)
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                features_list.append(np.array(las.red).reshape(-1, 1))
                features_list.append(np.array(las.green).reshape(-1, 1))
                features_list.append(np.array(las.blue).reshape(-1, 1))
            
            features = np.hstack(features_list) if features_list else np.zeros((len(points), 1))
            
            tile_data = {
                'points': points,
                'features': features
            }
            
            # Cache if enabled
            if self.enable_caching:
                self._tile_cache[str(tile_path)] = tile_data
            
            return tile_data
            
        except Exception as e:
            logger.error(f"Failed to load tile {tile_path}: {e}")
            return None
    
    def _extract_buffer_zone(
        self,
        neighbor_path: Path,
        core_bounds: Tuple[float, float, float, float],
        buffer_size: float
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract buffer zone from a neighbor tile.
        
        Only loads points within buffer_size of the core tile boundary.
        
        Args:
            neighbor_path: Path to neighbor tile
            core_bounds: (xmin, ymin, xmax, ymax) of core tile
            buffer_size: Buffer width in meters
        
        Returns:
            Dictionary with 'points' and 'features' in buffer zone,
            or None if no points in buffer
        """
        # Load neighbor tile
        neighbor_data = self._load_tile(neighbor_path)
        
        if neighbor_data is None:
            return None
        
        neighbor_points = neighbor_data['points']
        neighbor_features = neighbor_data['features']
        
        # Expand core bounds by buffer size
        xmin, ymin, xmax, ymax = core_bounds
        buffer_bounds = (
            xmin - buffer_size,
            ymin - buffer_size,
            xmax + buffer_size,
            ymax + buffer_size
        )
        
        # Find points in buffer zone
        # (within expanded bounds but outside core bounds)
        x, y = neighbor_points[:, 0], neighbor_points[:, 1]
        
        in_expanded = (
            (x >= buffer_bounds[0]) & (x <= buffer_bounds[2]) &
            (y >= buffer_bounds[1]) & (y <= buffer_bounds[3])
        )
        
        outside_core = (
            (x < core_bounds[0]) | (x > core_bounds[2]) |
            (y < core_bounds[1]) | (y > core_bounds[3])
        )
        
        in_buffer = in_expanded & outside_core
        
        if not np.any(in_buffer):
            return None
        
        return {
            'points': neighbor_points[in_buffer],
            'features': neighbor_features[in_buffer]
        }
    
    def _compute_bounds(
        self,
        points: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Compute bounding box of points.
        
        Args:
            points: (N, 3) XYZ coordinates
        
        Returns:
            (xmin, ymin, xmax, ymax)
        """
        xmin, ymin = points[:, 0].min(), points[:, 1].min()
        xmax, ymax = points[:, 0].max(), points[:, 1].max()
        return (xmin, ymin, xmax, ymax)
    
    def _detect_neighbor_tiles(
        self,
        tile_path: Path
    ) -> Optional[List[Path]]:
        """
        Auto-detect neighbor tiles based on filename pattern.
        
        Supports IGN LiDAR HD naming convention:
        LIDAR_HD_<XXXX>_<YYYY>.laz where XXXX, YYYY are grid coordinates
        
        Args:
            tile_path: Path to core tile
        
        Returns:
            List of paths to potential neighbor tiles, or None if pattern
            doesn't match
        """
        # Try to parse IGN LiDAR HD grid pattern
        # Example: LIDAR_HD_0450_6250.laz -> x=450, y=6250
        
        filename = tile_path.stem
        parts = filename.split('_')
        
        if len(parts) < 4 or parts[0] != 'LIDAR' or parts[1] != 'HD':
            logger.debug(f"Filename doesn't match IGN pattern: {filename}")
            return None
        
        try:
            x_coord = int(parts[2])
            y_coord = int(parts[3])
        except ValueError:
            logger.debug(f"Could not parse coordinates from: {filename}")
            return None
        
        # IGN tiles are typically 1km x 1km
        # Generate 8 neighbor coordinates (N, S, E, W, NE, NW, SE, SW)
        tile_size = 1000  # 1km in meters
        
        neighbor_coords = [
            (x_coord, y_coord + tile_size),      # North
            (x_coord, y_coord - tile_size),      # South
            (x_coord + tile_size, y_coord),      # East
            (x_coord - tile_size, y_coord),      # West
            (x_coord + tile_size, y_coord + tile_size),  # NE
            (x_coord - tile_size, y_coord + tile_size),  # NW
            (x_coord + tile_size, y_coord - tile_size),  # SE
            (x_coord - tile_size, y_coord - tile_size),  # SW
        ]
        
        # Build neighbor paths
        neighbors = []
        tile_dir = tile_path.parent
        
        for nx, ny in neighbor_coords:
            neighbor_name = f"LIDAR_HD_{nx:04d}_{ny:04d}.laz"
            neighbor_path = tile_dir / neighbor_name
            
            if neighbor_path.exists():
                neighbors.append(neighbor_path)
        
        return neighbors if neighbors else None
    
    def clear_cache(self):
        """Clear the tile cache to free memory."""
        if self._tile_cache:
            self._tile_cache.clear()
            logger.info("Tile cache cleared")
