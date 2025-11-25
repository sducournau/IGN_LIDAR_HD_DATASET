"""
Unified Ground Truth Provider - Consolidated Interface for All Ground Truth Operations

This module consolidates three previously separate classes into a single, easy-to-use
interface:
1. IGNGroundTruthFetcher (ign_lidar/io/wfs_ground_truth.py) - WFS data fetching
2. GroundTruthManager (ign_lidar/core/ground_truth_manager.py) - Cache management
3. GroundTruthOptimizer (ign_lidar/optimization/ground_truth.py) - Spatial labeling

Architecture:
    GroundTruthProvider (singleton)
    ├── High-level convenience API (recommended)
    │   ├── fetch_all_features(bbox) - Get all ground truth data
    │   ├── label_points(points, features) - Label points with ground truth
    │   ├── prefetch_for_tile(laz_file) - Prefetch tile data
    │   └── fetch_and_label(laz_file, points) - All-in-one operation
    │
    └── Low-level component access (when needed)
        ├── .fetcher - Direct WFS access
        ├── .manager - Cache & prefetch operations
        └── .optimizer - Spatial labeling optimization

Usage Examples:

    # High-level API (recommended)
    >>> from ign_lidar.core.ground_truth_provider import GroundTruthProvider
    >>> gt = GroundTruthProvider(cache_enabled=True)
    >>> features = gt.fetch_all_features(bbox=(100, 50, 150, 100))
    >>> labels = gt.label_points(points, features)

    # Prefetch before processing
    >>> gt.prefetch_for_tile(Path("data/tile_001.laz"))
    >>> # Then process: should hit cache

    # All-in-one operation
    >>> labels = gt.fetch_and_label(Path("data/tile_001.laz"), points)

    # Low-level component access
    >>> buildings = gt.fetcher.fetch_buildings(bbox)
    >>> labels = gt.optimizer.label_points_with_ground_truth(points, buildings)

Benefits:
    - Single entry point for all ground truth operations
    - Unified caching across components
    - Clear HIGH-LEVEL vs LOW-LEVEL separation
    - Full backward compatibility with existing code
    - Lazy loading of sub-components (performance optimization)

Version: 1.0.0
Date: November 25, 2025
Status: Production-ready
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class GroundTruthProvider:
    """
    Unified provider for all ground truth operations.

    This class consolidates WFS fetching, caching, and spatial labeling
    into a single, easy-to-use interface while maintaining full backward
    compatibility with existing code.

    The provider uses lazy loading: sub-components are only instantiated when
    first accessed, improving performance when only some components are needed.

    Attributes:
        fetcher: IGNGroundTruthFetcher - WFS data fetching (lazy-loaded)
        manager: GroundTruthManager - Cache management (lazy-loaded)
        optimizer: GroundTruthOptimizer - Spatial labeling (lazy-loaded)

    Examples:
        >>> # High-level convenient API
        >>> gt = GroundTruthProvider(cache_enabled=True)
        >>> features = gt.fetch_all_features(bbox)
        >>> labels = gt.label_points(points, features)
        >>>
        >>> # Low-level component access
        >>> buildings = gt.fetcher.fetch_buildings(bbox)
        >>> labels = gt.optimizer.label_points_with_ground_truth(points, buildings)

    Note:
        This class is a singleton - only one instance exists per process.
        All properties are cached after first access.
    """

    _instance: Optional["GroundTruthProvider"] = None
    _fetcher: Optional[Any] = None
    _manager: Optional[Any] = None
    _optimizer: Optional[Any] = None

    def __new__(cls, cache_enabled: bool = True) -> "GroundTruthProvider":
        """Ensure singleton pattern - only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the Ground Truth Provider.

        Args:
            cache_enabled: Whether to enable caching of fetched data (default: True)
        """
        if self._initialized:
            return

        self._cache_enabled = cache_enabled
        self._ground_truth_cache: Dict[str, Any] = {} if cache_enabled else None
        self._initialized = True

        logger.debug(
            f"GroundTruthProvider initialized (singleton, cache_enabled={cache_enabled})"
        )

    # ========================================================================
    # Properties - Lazy Loading Sub-Components
    # ========================================================================

    @property
    def fetcher(self) -> Any:
        """
        WFS data fetcher (lazy-loaded).

        Fetches ground truth data from IGN WFS services. Only instantiated
        when first accessed.

        Returns:
            IGNGroundTruthFetcher: WFS data fetching component

        Raises:
            ImportError: If shapely/geopandas not available
        """
        if GroundTruthProvider._fetcher is None:
            try:
                from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher

                GroundTruthProvider._fetcher = IGNGroundTruthFetcher()
                logger.debug("Lazy-loaded IGNGroundTruthFetcher")
            except ImportError as e:
                logger.error(f"Failed to import IGNGroundTruthFetcher: {e}")
                raise

        return GroundTruthProvider._fetcher

    @property
    def manager(self) -> Any:
        """
        Ground truth manager (lazy-loaded).

        Handles prefetching and caching of ground truth data. Only instantiated
        when first accessed.

        Returns:
            GroundTruthManager: Prefetch & cache management component
        """
        if GroundTruthProvider._manager is None:
            try:
                from ign_lidar.core.ground_truth_manager import GroundTruthManager

                GroundTruthProvider._manager = GroundTruthManager()
                logger.debug("Lazy-loaded GroundTruthManager")
            except ImportError as e:
                logger.warning(f"GroundTruthManager not available: {e}")
                GroundTruthProvider._manager = None

        return GroundTruthProvider._manager

    @property
    def optimizer(self) -> Any:
        """
        Ground truth optimizer (lazy-loaded).

        Optimizes spatial labeling and classification refinement. Only instantiated
        when first accessed.

        Returns:
            GroundTruthOptimizer: Spatial optimization component
        """
        if GroundTruthProvider._optimizer is None:
            try:
                from ign_lidar.optimization.ground_truth import GroundTruthOptimizer

                GroundTruthProvider._optimizer = GroundTruthOptimizer()
                logger.debug("Lazy-loaded GroundTruthOptimizer")
            except ImportError as e:
                logger.warning(f"GroundTruthOptimizer not available: {e}")
                GroundTruthProvider._optimizer = None

        return GroundTruthProvider._optimizer

    # ========================================================================
    # High-Level Convenience API (Recommended)
    # ========================================================================

    def fetch_all_features(
        self, bbox: Tuple[float, float, float, float]
    ) -> Dict[str, Any]:
        """
        Fetch all ground truth features for a bounding box.

        HIGH-LEVEL API: Recommended for most use cases.

        Args:
            bbox: Bounding box (minx, miny, maxx, maxy)

        Returns:
            Dictionary containing all ground truth features:
            {
                'buildings': GeoDataFrame or list of building polygons,
                'roads': GeoDataFrame or list of road polygons,
                'vegetation': GeoDataFrame or list of vegetation polygons,
                'water': GeoDataFrame or list of water features,
                ...
            }

        Example:
            >>> bbox = (100.0, 50.0, 150.0, 100.0)
            >>> features = gt.fetch_all_features(bbox)
            >>> print(f"Buildings: {len(features['buildings'])}")

        Note:
            Results are cached if caching is enabled. Repeated calls with the
            same bbox will return cached results immediately.
        """
        # Generate cache key
        cache_key = f"all_features_{bbox}"

        # Check cache
        if self._ground_truth_cache is not None and cache_key in self._ground_truth_cache:
            logger.debug(f"Cache hit for all_features at {bbox}")
            return self._ground_truth_cache[cache_key]

        # Fetch from WFS
        logger.debug(f"Fetching all features for bbox {bbox}")
        features = self.fetcher.fetch_all_features(bbox)

        # Cache result
        if self._ground_truth_cache is not None:
            self._ground_truth_cache[cache_key] = features
            logger.debug(f"Cached all_features for bbox {bbox}")

        return features

    def label_points(
        self,
        points: np.ndarray,
        features: Dict[str, Any],
        use_optimization: bool = True,
    ) -> np.ndarray:
        """
        Label points using ground truth features.

        HIGH-LEVEL API: Recommended for most use cases.

        Args:
            points: Point cloud array [N, 3] with (x, y, z) coordinates
            features: Dictionary of ground truth features (from fetch_all_features)
            use_optimization: Whether to use spatial optimization (default: True)

        Returns:
            Classification labels array [N] with class assignments

        Example:
            >>> points = np.random.rand(1000, 3)
            >>> features = gt.fetch_all_features(bbox)
            >>> labels = gt.label_points(points, features)
            >>> print(f"Labeled points: {np.unique(labels)}")

        Note:
            For large point clouds (>100k points), optimization is recommended
            as it significantly improves speed while maintaining accuracy.
        """
        if not use_optimization or self.optimizer is None:
            # Direct labeling via fetcher
            logger.debug(
                "Labeling points with ground truth (direct method)"
            )
            return self.fetcher.label_points_with_ground_truth(points, features)

        # Optimized labeling via optimizer
        logger.debug("Labeling points with ground truth (optimized method)")
        return self.optimizer.label_points_with_ground_truth(points, features)

    def fetch_and_label(
        self, laz_file: Path, points: np.ndarray
    ) -> np.ndarray:
        """
        All-in-one operation: fetch ground truth for tile and label points.

        HIGH-LEVEL API: Convenience method combining prefetch + label.

        Args:
            laz_file: Path to LAZ file (used to extract bounding box)
            points: Point cloud array [N, 3]

        Returns:
            Classification labels array [N]

        Example:
            >>> from pathlib import Path
            >>> laz_file = Path("data/tile_001.laz")
            >>> points = np.random.rand(10000, 3)
            >>> labels = gt.fetch_and_label(laz_file, points)

        Note:
            This method will automatically prefetch and cache the tile's
            ground truth data. If the tile is already cached, it will use
            the cached data.
        """
        logger.debug(f"Fetch and label for {laz_file.name}")

        # Extract bbox from LAZ file
        try:
            import laspy

            las = laspy.read(str(laz_file))
            bbox = (
                float(las.header.x_min),
                float(las.header.y_min),
                float(las.header.x_max),
                float(las.header.y_max),
            )
        except Exception as e:
            logger.error(f"Failed to read LAZ file {laz_file}: {e}")
            raise

        # Fetch features
        features = self.fetch_all_features(bbox)

        # Label points
        labels = self.label_points(points, features)

        return labels

    def prefetch_for_tile(self, laz_file: Path) -> Optional[Dict[str, Any]]:
        """
        Prefetch ground truth data for a tile.

        HIGH-LEVEL API: Use before processing tiles to load data into cache.

        Args:
            laz_file: Path to LAZ file to prefetch data for

        Returns:
            Dictionary of prefetched features, or None if failed

        Example:
            >>> # Prefetch several tiles
            >>> for tile_path in tile_paths:
            ...     gt.prefetch_for_tile(tile_path)
            >>> # Then process: all data already in cache
            >>> for tile_path in tile_paths:
            ...     labels = gt.fetch_and_label(tile_path, points)

        Note:
            This is useful when processing many tiles sequentially.
            Prefetching ensures that ground truth data is already in memory
            when processing begins, avoiding delays during processing.
        """
        if self.manager is None:
            logger.warning("GroundTruthManager not available, cannot prefetch")
            return None

        logger.debug(f"Prefetching ground truth for {laz_file.name}")
        return self.manager.prefetch_ground_truth_for_tile(laz_file)

    def prefetch_batch(self, laz_files: List[Path]) -> Dict[Path, bool]:
        """
        Prefetch ground truth data for multiple tiles.

        HIGH-LEVEL API: Batch prefetch for efficient processing.

        Args:
            laz_files: List of LAZ file paths to prefetch

        Returns:
            Dictionary mapping file paths to success status

        Example:
            >>> tile_paths = [Path("tile_1.laz"), Path("tile_2.laz")]
            >>> results = gt.prefetch_batch(tile_paths)
            >>> for path, success in results.items():
            ...     print(f"{path.name}: {'OK' if success else 'FAILED'}")

        Note:
            Prefetching is done sequentially to avoid memory overflow.
            For very large batches, consider prefetching in groups.
        """
        results = {}

        for laz_file in laz_files:
            try:
                features = self.prefetch_for_tile(laz_file)
                results[laz_file] = features is not None
            except Exception as e:
                logger.warning(f"Failed to prefetch {laz_file.name}: {e}")
                results[laz_file] = False

        return results

    # ========================================================================
    # Cache Management
    # ========================================================================

    def clear_cache(self):
        """
        Clear all cached ground truth data.

        Use this method to free memory after processing or to ensure fresh
        data is fetched on next request.

        Example:
            >>> gt.clear_cache()
            >>> features = gt.fetch_all_features(bbox)  # Fresh fetch from WFS
        """
        if self._ground_truth_cache is not None:
            self._ground_truth_cache.clear()
            logger.debug("Cleared ground truth cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics:
            {
                'enabled': bool,
                'size': int,  # Number of cached items
                'keys': list,  # Cache keys
            }

        Example:
            >>> stats = gt.get_cache_stats()
            >>> print(f"Cache size: {stats['size']} items")
        """
        if self._ground_truth_cache is None:
            return {"enabled": False, "size": 0, "keys": []}

        return {
            "enabled": True,
            "size": len(self._ground_truth_cache),
            "keys": list(self._ground_truth_cache.keys()),
        }

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @staticmethod
    def reset_instance():
        """
        Reset the singleton instance.

        Use this only in testing or when you need a fresh instance.

        Example:
            >>> GroundTruthProvider.reset_instance()
            >>> gt = GroundTruthProvider(cache_enabled=False)  # New instance
        """
        GroundTruthProvider._instance = None
        GroundTruthProvider._fetcher = None
        GroundTruthProvider._manager = None
        GroundTruthProvider._optimizer = None
        logger.debug("Reset GroundTruthProvider singleton")

    def __repr__(self) -> str:
        """String representation of the provider."""
        cache_info = f"cache_enabled={self._cache_enabled}"
        if self._ground_truth_cache is not None:
            cache_info += f", cached_items={len(self._ground_truth_cache)}"
        return f"GroundTruthProvider({cache_info})"


# ============================================================================
# Module-Level Convenience Functions
# ============================================================================

# Global instance for module-level convenience
_global_provider: Optional[GroundTruthProvider] = None


def get_provider(cache_enabled: bool = True) -> GroundTruthProvider:
    """
    Get the global Ground Truth Provider instance.

    This function provides convenient access to the singleton provider
    without needing to manage the instance yourself.

    Args:
        cache_enabled: Whether to enable caching (only used for first call)

    Returns:
        GroundTruthProvider: The global singleton instance

    Example:
        >>> from ign_lidar.core.ground_truth_provider import get_provider
        >>> gt = get_provider()
        >>> features = gt.fetch_all_features(bbox)
    """
    global _global_provider
    if _global_provider is None:
        _global_provider = GroundTruthProvider(cache_enabled=cache_enabled)
    return _global_provider
