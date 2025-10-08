"""
Tile Stitching Module for IGN LiDAR HD v2.0

This module implements seamless feature computation at tile boundaries
with flexible capabilities for different use cases.

Key Features:
- Load multiple adjacent tiles with intelligent neighbor detection
- Extract buffer zones from neighbors with multi-scale support
- Merge core + buffer point clouds for seamless processing
- Compute features with cross-tile neighborhoods
- Eliminate edge artifacts with boundary smoothing
- Parallel processing and caching support
- Quality assurance and performance monitoring

Author: IGN LiDAR HD Team
Date: October 8, 2025
Version: 2.1
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import laspy
from scipy.spatial import KDTree, cKDTree
try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_NDIMAGE_AVAILABLE = True
except ImportError:
    SCIPY_NDIMAGE_AVAILABLE = False
    logger.warning("scipy.ndimage not available - boundary smoothing disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")

logger = logging.getLogger(__name__)


class TileStitcher:
    """
    Tile stitching with flexible capabilities.
    
    This class enables seamless feature computation at tile boundaries with:
    1. Loading a core tile and its neighbors (with smart detection)
    2. Extracting buffer zones from neighbor tiles (multi-scale support)
    3. Merging core + buffer points for complete neighborhoods
    4. Computing cross-tile features
    5. Caching and parallel processing capabilities
    6. Boundary smoothing and artifact removal
    
    Examples:
        Basic usage (backward compatible):
        >>> stitcher = TileStitcher(buffer_size=10.0)
        >>> tile_data = stitcher.load_tile_with_neighbors(
        ...     tile_path=Path("tile_A.laz"),
        ...     neighbor_tiles=[Path("tile_B.laz"), Path("tile_C.laz")]
        ... )
        
        Configuration usage:
        >>> config = {'buffer_size': 15.0, 'parallel_loading': True}
        >>> stitcher = TileStitcher(config=config)
        >>> tile_data = stitcher.load_tile_with_smart_neighbors(Path("tile_A.laz"))
    """
    
    def __init__(self, buffer_size: float = 10.0, enable_caching: bool = True, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TileStitcher with configuration.
        
        Args:
            buffer_size: Buffer zone width in meters (default: 10m)
                        Larger values = more accurate but more memory
            enable_caching: If True, cache loaded tiles to avoid re-reading
            config: Configuration dictionary (overrides basic parameters)
        """
        # Backward compatibility: use basic parameters if no config provided
        if config is None:
            self.config = {
                'buffer_size': buffer_size,
                'enable_caching': enable_caching,
                'auto_detect_neighbors': False,
                'parallel_loading': False,
                'boundary_smoothing': False,
                'verbose_logging': False
            }
        else:
            # Configuration with defaults
            self.config = {
                'buffer_size': 15.0,
                'adaptive_buffer': True,
                'min_buffer': 5.0,
                'max_buffer': 25.0,
                'auto_detect_neighbors': True,
                'neighbor_search_radius': 50.0,
                'max_neighbors': 8,
                'use_grid_pattern': True,
                'cache_enabled': True,
                'cache_size': 1000,
                'parallel_loading': True,
                'prefetch_neighbors': True,
                'overlap_threshold': 0.1,
                'boundary_smoothing': True,
                'edge_artifact_removal': True,
                'seamless_blending': True,
                'memory_limit': 8192,
                'chunk_processing': True,
                'gc_frequency': 10,
                'compute_boundary_features': True,
                'boundary_feature_radius': 5.0,
                'cross_tile_neighborhoods': True,
                'preserve_tile_metadata': True,
                'verbose_logging': False
            }
            self.config.update(config)
        
        # Backward compatibility properties
        self.buffer_size = self.config['buffer_size']
        self.enable_caching = self.config.get('cache_enabled', self.config.get('enable_caching', True))
        
        # Initialize components
        self._tile_cache = {} if self.enable_caching else None
        self._neighbor_cache = {}
        self._spatial_index = None
        self._executor = None
        self._stats = {
            'tiles_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_load_time': 0.0,
            'total_stitch_time': 0.0
        }
        
        # Threading lock for thread-safe operations
        self._lock = threading.Lock() if self.config.get('parallel_loading', False) else None
        
        logger.info(f"TileStitcher initialized (buffer_size={self.buffer_size}m, configured={config is not None})")
    
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
        threshold: Optional[float] = None
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
        if self.enable_caching and self._tile_cache is not None and str(tile_path) in self._tile_cache:
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
            if self.enable_caching and self._tile_cache is not None:
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
    
    # ================== SMART METHODS ==================
    
    def load_tile_with_smart_neighbors(
        self,
        tile_path: Path,
        neighbor_paths: Optional[List[Path]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Tile loading with intelligent neighbor detection and processing.
        
        Args:
            tile_path: Path to the main tile
            neighbor_paths: Optional list of neighbor paths (auto-detected if None)
            
        Returns:
            Tile data with smart stitching
        """
        start_time = time.time()
        
        # Load core tile
        core_data = self._load_tile_cached(tile_path)
        if core_data is None:
            raise ValueError(f"Failed to load core tile: {tile_path}")
        
        core_points = core_data['points']
        core_features = core_data['features']
        core_bounds = self._compute_bounds(core_points)
        
        # Detect neighbors if not provided and auto-detection enabled
        if neighbor_paths is None and self.config.get('auto_detect_neighbors', False):
            neighbor_paths = self._smart_neighbor_detection(tile_path, core_bounds)
        
        # Load neighbors in parallel if enabled
        neighbor_data_list = []
        if neighbor_paths and self.config.get('parallel_loading', False):
            neighbor_data_list = self._load_neighbors_parallel(neighbor_paths, core_bounds)
        elif neighbor_paths:
            neighbor_data_list = self._load_neighbors_sequential(neighbor_paths, core_bounds)
        
        # Extract multi-scale buffer zones if boundary features enabled
        if self.config.get('compute_boundary_features', False):
            buffer_zones = self._extract_multi_scale_buffers(neighbor_data_list, core_bounds)
        else:
            # Standard buffer extraction for backward compatibility
            buffer_points_list = []
            buffer_features_list = []
            
            for neighbor_data in neighbor_data_list:
                buffer_data = self._extract_buffer_zone_simple(
                    neighbor_data, core_bounds, self.buffer_size
                )
                if buffer_data is not None and len(buffer_data['points']) > 0:
                    buffer_points_list.append(buffer_data['points'])
                    buffer_features_list.append(buffer_data['features'])
            
            # Merge buffer zones
            if buffer_points_list:
                buffer_points = np.vstack(buffer_points_list)
                buffer_features = np.vstack(buffer_features_list)
            else:
                buffer_points = np.empty((0, 3))
                buffer_features = np.empty((0, core_features.shape[1]))
            
            buffer_zones = {
                'contextual': {
                    'points': buffer_points,
                    'features': buffer_features
                }
            }
        
        # Apply boundary smoothing if enabled
        if self.config.get('boundary_smoothing', False) and SCIPY_NDIMAGE_AVAILABLE:
            core_points, core_features = self._apply_boundary_smoothing(
                core_points, core_features, core_bounds, buffer_zones
            )
        
        # Compute boundary features if enabled
        if self.config.get('compute_boundary_features', False):
            boundary_features = self._compute_boundary_features(
                core_points, core_features, buffer_zones, core_bounds
            )
        else:
            boundary_features = core_features
        
        # Build result dictionary
        result = {
            'core_points': core_points,
            'core_features': boundary_features,
            'buffer_zones': buffer_zones,
            'core_bounds': core_bounds,
            'num_core': len(core_points),
            'stitching_metadata': {
                'neighbors_used': len(neighbor_data_list),
                'buffer_sizes': self._get_adaptive_buffer_sizes(core_points) if self.config.get('adaptive_buffer', False) else {'base': self.buffer_size},
                'processing_time': time.time() - start_time,
                'smart_mode': True
            }
        }
        
        # Add backward compatibility fields
        if 'contextual' in buffer_zones:
            result['buffer_points'] = buffer_zones['contextual']['points']
            result['buffer_features'] = buffer_zones['contextual']['features']
            result['num_buffer'] = len(buffer_zones['contextual']['points'])
            
            # Create combined arrays and core mask
            combined_points = np.vstack([core_points, result['buffer_points']])
            combined_features = np.vstack([boundary_features, result['buffer_features']])
            core_mask = np.zeros(len(combined_points), dtype=bool)
            core_mask[:len(core_points)] = True
            
            result['combined_points'] = combined_points
            result['combined_features'] = combined_features
            result['core_mask'] = core_mask
        
        # Update statistics
        self._stats['tiles_processed'] += 1
        self._stats['total_stitch_time'] += time.time() - start_time
        
        return result
    
    def _smart_neighbor_detection(
        self,
        tile_path: Path,
        core_bounds: Tuple[float, float, float, float]
    ) -> List[Path]:
        """
        Intelligent neighbor detection using multiple strategies.
        """
        neighbors = []
        
        if self.config.get('use_grid_pattern', True):
            # Try IGN grid pattern first
            grid_neighbors = self._detect_grid_neighbors(tile_path)
            if grid_neighbors:
                neighbors.extend(grid_neighbors)
        
        # Fallback to spatial search if no grid neighbors found
        if not neighbors:
            neighbors = self._detect_spatial_neighbors(tile_path, core_bounds)
        
        # Limit number of neighbors
        max_neighbors = self.config.get('max_neighbors', 8)
        if len(neighbors) > max_neighbors:
            # Keep closest neighbors based on distance to tile center
            center_x = (core_bounds[0] + core_bounds[2]) / 2
            center_y = (core_bounds[1] + core_bounds[3]) / 2
            
            distances = []
            for neighbor in neighbors:
                neighbor_bounds = self._get_tile_bounds_fast(neighbor)
                if neighbor_bounds:
                    neighbor_center_x = (neighbor_bounds[0] + neighbor_bounds[2]) / 2
                    neighbor_center_y = (neighbor_bounds[1] + neighbor_bounds[3]) / 2
                    dist = np.sqrt((center_x - neighbor_center_x)**2 + (center_y - neighbor_center_y)**2)
                    distances.append((dist, neighbor))
            
            distances.sort(key=lambda x: x[0])
            neighbors = [neighbor for _, neighbor in distances[:max_neighbors]]
        
        return neighbors
    
    def _detect_grid_neighbors(self, tile_path: Path) -> Optional[List[Path]]:
        """
        Grid neighbor detection with multiple IGN patterns.
        """
        filename = tile_path.stem
        
        # Pattern matching for different IGN formats
        import re
        patterns = [
            # Standard format: HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69
            r'HD_LIDARHD_FXX_(\d+)_(\d+)_PTS_C_LAMB93_IGN69',
            # Alternative format: LIDAR_HD_0450_6250
            r'LIDAR_HD_(\d+)_(\d+)',
            # Simplified format: 0450_6250
            r'(\d+)_(\d+)'
        ]
        
        coords = None
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                coords = (int(match.group(1)), int(match.group(2)))
                break
        
        if not coords:
            return None
        
        x_coord, y_coord = coords
        tile_size = self.config.get('grid_tile_size', 1000)  # Default IGN tile size
        
        # Generate neighbor coordinates (8-connectivity)
        neighbor_coords = [
            (x_coord, y_coord + tile_size),      # North
            (x_coord, y_coord - tile_size),      # South  
            (x_coord + tile_size, y_coord),      # East
            (x_coord - tile_size, y_coord),      # West
            (x_coord + tile_size, y_coord + tile_size),  # NE
            (x_coord - tile_size, y_coord + tile_size),  # NW
            (x_coord + tile_size, y_coord - tile_size),  # SE
            (x_coord - tile_size, y_coord - tile_size)   # SW
        ]
        
        # Find existing neighbor files
        neighbors = []
        tile_dir = tile_path.parent
        
        for nx, ny in neighbor_coords:
            # Try different filename formats
            possible_names = [
                f"HD_LIDARHD_FXX_{nx:04d}_{ny:04d}_PTS_C_LAMB93_IGN69.laz",
                f"LIDAR_HD_{nx:04d}_{ny:04d}.laz",
                f"{nx:04d}_{ny:04d}.laz"
            ]
            
            for name in possible_names:
                neighbor_path = tile_dir / name
                if neighbor_path.exists():
                    neighbors.append(neighbor_path)
                    break
        
        return neighbors if neighbors else None
    
    def _detect_spatial_neighbors(
        self,
        tile_path: Path,
        core_bounds: Tuple[float, float, float, float]
    ) -> List[Path]:
        """
        Detect neighbors using spatial proximity search.
        """
        neighbors = []
        search_radius = self.config.get('neighbor_search_radius', 50.0)
        tile_dir = tile_path.parent
        
        # Get center of core tile
        center_x = (core_bounds[0] + core_bounds[2]) / 2
        center_y = (core_bounds[1] + core_bounds[3]) / 2
        
        # Search for LAZ files in the same directory
        for candidate_path in tile_dir.glob("*.laz"):
            if candidate_path == tile_path:
                continue
                
            candidate_bounds = self._get_tile_bounds_fast(candidate_path)
            if candidate_bounds is None:
                continue
                
            # Check if candidate is within search radius
            candidate_center_x = (candidate_bounds[0] + candidate_bounds[2]) / 2
            candidate_center_y = (candidate_bounds[1] + candidate_bounds[3]) / 2
            
            distance = np.sqrt(
                (center_x - candidate_center_x)**2 + 
                (center_y - candidate_center_y)**2
            )
            
            if distance <= search_radius:
                # Check for actual overlap
                overlap_ratio = self._compute_overlap_ratio(core_bounds, candidate_bounds)
                if overlap_ratio >= self.config.get('overlap_threshold', 0.1):
                    neighbors.append(candidate_path)
        
        return neighbors
    
    def _get_tile_bounds_fast(self, tile_path: Path) -> Optional[Tuple[float, float, float, float]]:
        """
        Fast bounds computation from LAZ header without loading all points.
        """
        try:
            with laspy.open(tile_path) as las_file:
                header = las_file.header
                return (
                    float(header.x_min),
                    float(header.y_min),
                    float(header.x_max), 
                    float(header.y_max)
                )
        except Exception:
            return None
    
    def _compute_overlap_ratio(
        self,
        bounds1: Tuple[float, float, float, float],
        bounds2: Tuple[float, float, float, float]
    ) -> float:
        """
        Compute overlap ratio between two bounding boxes.
        """
        x1_min, y1_min, x1_max, y1_max = bounds1
        x2_min, y2_min, x2_max, y2_max = bounds2
        
        # Compute intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        if x_overlap == 0 or y_overlap == 0:
            return 0.0
        
        intersection_area = x_overlap * y_overlap
        
        # Compute union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _load_neighbors_sequential(
        self,
        neighbor_paths: List[Path],
        core_bounds: Tuple[float, float, float, float]
    ) -> List[Dict]:
        """
        Load neighbor tiles sequentially.
        """
        neighbor_data_list = []
        
        for path in neighbor_paths:
            try:
                data = self._load_tile_cached(path)
                if data is not None:
                    neighbor_data_list.append(data)
            except Exception as e:
                logger.warning(f"Failed to load neighbor {path}: {e}")
                
        return neighbor_data_list
    
    def _load_neighbors_parallel(
        self,
        neighbor_paths: List[Path],
        core_bounds: Tuple[float, float, float, float]
    ) -> List[Dict]:
        """
        Load neighbor tiles in parallel for better performance.
        """
        if not self._executor:
            max_workers = min(4, len(neighbor_paths))
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        futures = []
        for path in neighbor_paths:
            future = self._executor.submit(self._load_tile_cached, path)
            futures.append((future, path))
        
        neighbor_data_list = []
        for future, path in futures:
            try:
                data = future.result(timeout=30)  # 30 second timeout
                if data is not None:
                    neighbor_data_list.append(data)
            except Exception as e:
                logger.warning(f"Failed to load neighbor {path}: {e}")
        
        return neighbor_data_list
    
    def _load_tile_cached(self, tile_path: Path) -> Optional[Dict]:
        """
        Load tile with caching support.
        """
        if not self.enable_caching or self._tile_cache is None:
            return self._load_tile(tile_path)
        
        cache_key = str(tile_path)
        
        # Thread-safe cache access if parallel processing enabled
        if self._lock:
            with self._lock:
                if cache_key in self._tile_cache:
                    self._stats['cache_hits'] += 1
                    return self._tile_cache[cache_key]
        else:
            if cache_key in self._tile_cache:
                self._stats['cache_hits'] += 1
                return self._tile_cache[cache_key]
        
        # Load tile
        start_time = time.time()
        tile_data = self._load_tile(tile_path)
        self._stats['total_load_time'] += time.time() - start_time
        self._stats['cache_misses'] += 1
        
        if tile_data is not None:
            # Thread-safe cache update
            if self._lock:
                with self._lock:
                    # Check cache size and evict if necessary
                    cache_size_limit = self.config.get('cache_size', 1000)
                    if len(self._tile_cache) >= cache_size_limit:
                        # Simple LRU: remove first item
                        oldest_key = next(iter(self._tile_cache))
                        del self._tile_cache[oldest_key]
                    
                    self._tile_cache[cache_key] = tile_data
            else:
                cache_size_limit = self.config.get('cache_size', 1000)
                if len(self._tile_cache) >= cache_size_limit:
                    oldest_key = next(iter(self._tile_cache))
                    del self._tile_cache[oldest_key]
                
                self._tile_cache[cache_key] = tile_data
        
        return tile_data
    
    def _extract_multi_scale_buffers(
        self,
        neighbor_data_list: List[Dict],
        core_bounds: Tuple[float, float, float, float]
    ) -> Dict[str, Dict]:
        """
        Extract buffer zones at multiple scales for different feature types.
        """
        buffer_zones = {
            'geometric': {'points': [], 'features': []},
            'contextual': {'points': [], 'features': []},  
            'semantic': {'points': [], 'features': []}
        }
        
        # Define buffer sizes for different feature types
        geometric_buffer = self.config.get('geometric_buffer', self.buffer_size * 0.5)
        contextual_buffer = self.config.get('contextual_buffer', self.buffer_size)
        semantic_buffer = self.config.get('semantic_buffer', self.buffer_size * 1.5)
        
        buffer_configs = {
            'geometric': geometric_buffer,
            'contextual': contextual_buffer,
            'semantic': semantic_buffer
        }
        
        xmin, ymin, xmax, ymax = core_bounds
        
        for neighbor_data in neighbor_data_list:
            points = neighbor_data['points']
            features = neighbor_data['features']
            
            for buffer_type, buffer_size in buffer_configs.items():
                # Define expanded bounds for this buffer type
                expanded_bounds = (
                    xmin - buffer_size,
                    ymin - buffer_size,
                    xmax + buffer_size,
                    ymax + buffer_size
                )
                
                # Filter points within this buffer zone
                x, y = points[:, 0], points[:, 1]
                mask = (
                    (x >= expanded_bounds[0]) & (x <= expanded_bounds[2]) &
                    (y >= expanded_bounds[1]) & (y <= expanded_bounds[3]) &
                    ~((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax))  # Exclude core
                )
                
                if np.any(mask):
                    buffer_points = points[mask]
                    buffer_features = features[mask] if features is not None else None
                    
                    buffer_zones[buffer_type]['points'].append(buffer_points)
                    if buffer_features is not None:
                        buffer_zones[buffer_type]['features'].append(buffer_features)
        
        # Concatenate buffer zones
        for buffer_type in buffer_zones:
            if buffer_zones[buffer_type]['points']:
                buffer_zones[buffer_type]['points'] = np.vstack(buffer_zones[buffer_type]['points'])
                if buffer_zones[buffer_type]['features']:
                    buffer_zones[buffer_type]['features'] = np.vstack(buffer_zones[buffer_type]['features'])
                else:
                    # Create empty features array with correct shape
                    num_points = len(buffer_zones[buffer_type]['points'])
                    buffer_zones[buffer_type]['features'] = np.zeros((num_points, 1))
            else:
                buffer_zones[buffer_type]['points'] = np.empty((0, 3))
                buffer_zones[buffer_type]['features'] = np.empty((0, 1))
        
        return buffer_zones
    
    def _extract_buffer_zone_simple(
        self,
        neighbor_data: Dict,
        core_bounds: Tuple[float, float, float, float],
        buffer_size: float
    ) -> Optional[Dict]:
        """
        Simple buffer extraction for backward compatibility.
        """
        points = neighbor_data['points']
        features = neighbor_data['features']
        
        if len(points) == 0:
            return None
        
        xmin, ymin, xmax, ymax = core_bounds
        
        # Expanded bounds including buffer
        expanded_bounds = (
            xmin - buffer_size,
            ymin - buffer_size,
            xmax + buffer_size,
            ymax + buffer_size
        )
        
        # Filter points in buffer zone (expanded bounds minus core)
        x, y = points[:, 0], points[:, 1]
        
        # Points within expanded bounds
        in_expanded = (
            (x >= expanded_bounds[0]) & (x <= expanded_bounds[2]) &
            (y >= expanded_bounds[1]) & (y <= expanded_bounds[3])
        )
        
        # Points within core bounds (to exclude)
        in_core = (
            (x >= xmin) & (x <= xmax) &
            (y >= ymin) & (y <= ymax)
        )
        
        # Buffer mask: in expanded but not in core
        buffer_mask = in_expanded & ~in_core
        
        if not np.any(buffer_mask):
            return None
        
        return {
            'points': points[buffer_mask],
            'features': features[buffer_mask] if features is not None else None
        }
    
    def _apply_boundary_smoothing(
        self,
        points: np.ndarray,
        features: np.ndarray,
        bounds: Tuple[float, float, float, float],
        buffer_zones: Dict[str, Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply boundary smoothing to reduce edge artifacts.
        """
        if not SCIPY_NDIMAGE_AVAILABLE:
            return points, features
        
        # Identify boundary points
        boundary_mask = self._identify_boundary_points(points, bounds)
        
        if not np.any(boundary_mask):
            return points, features
        
        # Apply Gaussian smoothing to boundary regions
        smoothed_features = features.copy()
        
        if features is not None and len(features.shape) > 1:
            for i in range(features.shape[1]):
                feature_column = features[:, i]
                if boundary_mask.sum() > 10:  # Sufficient points for smoothing
                    boundary_values = feature_column[boundary_mask]
                    smoothed_values = gaussian_filter1d(
                        boundary_values, 
                        sigma=self.config.get('smoothing_sigma', 1.0)
                    )
                    smoothed_features[boundary_mask, i] = smoothed_values
        
        return points, smoothed_features
    
    def _identify_boundary_points(
        self,
        points: np.ndarray,
        bounds: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Identify points near tile boundaries.
        """
        xmin, ymin, xmax, ymax = bounds
        boundary_threshold = self.config.get('boundary_feature_radius', 5.0)
        
        x, y = points[:, 0], points[:, 1]
        
        # Points near any boundary
        near_left = x <= (xmin + boundary_threshold)
        near_right = x >= (xmax - boundary_threshold)
        near_bottom = y <= (ymin + boundary_threshold)
        near_top = y >= (ymax - boundary_threshold)
        
        boundary_mask = near_left | near_right | near_bottom | near_top
        
        return boundary_mask
    
    def _compute_boundary_features(
        self,
        core_points: np.ndarray,
        core_features: np.ndarray,
        buffer_zones: Dict[str, Dict],
        core_bounds: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Compute boundary features using multi-scale buffer information.
        """
        boundary_features = core_features.copy() if core_features is not None else np.empty((len(core_points), 0))
        
        # Compute boundary-aware features
        boundary_mask = self._identify_boundary_points(core_points, core_bounds)
        
        if np.any(boundary_mask):
            # Add boundary indicator feature
            boundary_feature_col = boundary_mask.astype(float).reshape(-1, 1)
            boundary_features = np.hstack([boundary_features, boundary_feature_col])
            
            # Add cross-tile neighborhood density if buffer zones available
            if 'contextual' in buffer_zones and buffer_zones['contextual']['points'].shape[0] > 0:
                density_feature = self._compute_cross_tile_density(
                    core_points, buffer_zones['contextual']['points']
                )
                boundary_features = np.hstack([boundary_features, density_feature.reshape(-1, 1)])
        
        return boundary_features
    
    def _compute_cross_tile_density(
        self,
        core_points: np.ndarray,
        buffer_points: np.ndarray,
        radius: float = 5.0
    ) -> np.ndarray:
        """
        Compute point density using cross-tile neighborhoods.
        """
        if len(buffer_points) == 0:
            return np.zeros(len(core_points))
        
        # Combine core and buffer points for density computation
        all_points = np.vstack([core_points, buffer_points])
        
        # Build spatial index
        tree = cKDTree(all_points[:, :2])  # Use only X,Y for 2D queries
        
        # Query neighborhoods for core points
        densities = np.zeros(len(core_points))
        
        for i, point in enumerate(core_points[:, :2]):
            neighbors = tree.query_ball_point(point, radius)
            densities[i] = len(neighbors) / (np.pi * radius**2)  # Points per unit area
        
        return densities
    
    def _get_adaptive_buffer_sizes(self, points: np.ndarray) -> Dict[str, float]:
        """
        Get adaptive buffer sizes based on point density.
        """
        if not self.config.get('adaptive_buffer', False):
            base_size = self.config['buffer_size']
            return {
                'geometric': base_size * 0.5,
                'contextual': base_size,
                'semantic': base_size * 1.5
            }
        
        # Compute point density (points per square meter)
        bounds = self._compute_bounds(points)
        area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        density = len(points) / max(area, 1.0) if area > 0 else 0
        
        # Adaptive buffer sizing based on density
        min_buffer = self.config.get('min_buffer', 5.0)
        max_buffer = self.config.get('max_buffer', 25.0)
        base_buffer = self.config['buffer_size']
        
        # Lower density -> larger buffer
        # Higher density -> smaller buffer
        if density > 10:  # High density
            buffer_factor = 0.7
        elif density > 5:  # Medium density
            buffer_factor = 1.0
        else:  # Low density
            buffer_factor = 1.3
        
        adaptive_base = np.clip(base_buffer * buffer_factor, min_buffer, max_buffer)
        
        return {
            'geometric': adaptive_base * 0.5,
            'contextual': adaptive_base,
            'semantic': adaptive_base * 1.5
        }
    
    def get_statistics(self) -> Dict:
        """
        Get processing statistics.
        """
        stats = self._stats.copy()
        
        if self._tile_cache:
            stats['cache_size'] = len(self._tile_cache)
            stats['cache_hit_rate'] = (
                self._stats['cache_hits'] / 
                max(self._stats['cache_hits'] + self._stats['cache_misses'], 1)
            )
        
        if PSUTIL_AVAILABLE:
            stats['memory_usage_mb'] = psutil.Process().memory_info().rss / 1024 / 1024
        
        return stats
    
    def cleanup(self):
        """
        Clean up resources.
        """
        if self._tile_cache:
            self._tile_cache.clear()
        
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        logger.info("TileStitcher cleanup complete")
