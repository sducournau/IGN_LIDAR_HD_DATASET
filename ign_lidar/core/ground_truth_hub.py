"""
Ground Truth Hub - Unified API for all ground truth operations.

This module provides a composition-based unified interface for ground truth
operations, following the same pattern as GPU Manager v3.1. It consolidates
4 previously separate classes into a single, easy-to-use API.

Architecture:
    GroundTruthHub (singleton)
    ├── .fetcher  → IGNGroundTruthFetcher (lazy-loaded)
    ├── .optimizer → GroundTruthOptimizer (lazy-loaded)
    ├── .manager  → GroundTruthManager (lazy-loaded)
    └── .refiner  → GroundTruthRefiner (lazy-loaded)

Usage:
    High-level convenience API (recommended):
        >>> from ign_lidar.core import ground_truth
        >>> labels = ground_truth.fetch_and_label(tile_path, points)
        >>> ground_truth.prefetch_batch(tile_paths)
    
    Low-level component access (when needed):
        >>> buildings = ground_truth.fetcher.fetch_buildings(bbox)
        >>> labels = ground_truth.optimizer.label_points(points, buildings)
        >>> ground_truth.manager.prefetch_for_tile(tile_path)
        >>> ground_truth.refiner.refine_all(points, labels, features)

Benefits:
    - Single entry point for all ground truth operations
    - Lazy loading of sub-components (performance optimization)
    - Unified caching across components
    - Clear HIGH-LEVEL vs LOW-LEVEL separation
    - Full backward compatibility with existing code

Version: 2.0.0
Date: November 22, 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class GroundTruthHub:
    """
    Unified hub for ground truth operations with lazy-loaded components.
    
    This class follows the composition pattern similar to GPU Manager v3.1,
    providing a single entry point for all ground truth operations while
    maintaining full backward compatibility with existing code.
    
    The hub uses lazy loading: sub-components are only instantiated when
    first accessed, improving performance when only some components are needed.
    
    Attributes:
        fetcher: IGNGroundTruthFetcher - WFS data fetching (lazy-loaded)
        optimizer: GroundTruthOptimizer - Spatial labeling optimization (lazy-loaded)
        manager: GroundTruthManager - Prefetch & cache management (lazy-loaded)
        refiner: GroundTruthRefiner - Classification refinement (lazy-loaded)
    
    Example:
        >>> from ign_lidar.core import ground_truth
        >>> 
        >>> # High-level convenience API
        >>> labels = ground_truth.fetch_and_label(tile_path, points)
        >>> 
        >>> # Low-level component access
        >>> buildings = ground_truth.fetcher.fetch_buildings(bbox)
        >>> labels = ground_truth.optimizer.label_points(points, buildings)
    
    Note:
        This class is a singleton - only one instance exists per process.
        All properties are cached after first access.
    """
    
    _instance: Optional['GroundTruthHub'] = None
    _fetcher: Optional[Any] = None
    _optimizer: Optional[Any] = None
    _manager: Optional[Any] = None
    _refiner: Optional[Any] = None
    
    def __new__(cls) -> 'GroundTruthHub':
        """Ensure singleton pattern - only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.debug("Created GroundTruthHub singleton instance")
        return cls._instance
    
    @property
    def fetcher(self):
        """
        WFS data fetcher (lazy-loaded).
        
        Fetches ground truth data from IGN WFS services. Only instantiated
        when first accessed.
        
        Returns:
            IGNGroundTruthFetcher: WFS data fetching component
        
        Example:
            >>> buildings = ground_truth.fetcher.fetch_buildings(bbox)
            >>> roads = ground_truth.fetcher.fetch_roads_with_polygons(bbox)
        """
        if self._fetcher is None:
            from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
            self._fetcher = IGNGroundTruthFetcher()
            logger.debug("Lazy-loaded IGNGroundTruthFetcher")
        return self._fetcher
    
    @property
    def optimizer(self):
        """
        Spatial labeling optimizer (lazy-loaded).
        
        Optimizes point labeling with spatial indexing and GPU acceleration.
        Only instantiated when first accessed.
        
        Returns:
            GroundTruthOptimizer: Spatial labeling optimization component
        
        Example:
            >>> labels = ground_truth.optimizer.label_points(points, polygons)
            >>> stats = ground_truth.optimizer.get_cache_stats()
        """
        if self._optimizer is None:
            from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
            self._optimizer = GroundTruthOptimizer()
            logger.debug("Lazy-loaded GroundTruthOptimizer")
        return self._optimizer
    
    @property
    def manager(self):
        """
        Prefetch & cache manager (lazy-loaded).
        
        Manages prefetching and caching of ground truth data for tiles.
        Only instantiated when first accessed.
        
        Returns:
            GroundTruthManager: Prefetch & cache management component
        
        Example:
            >>> ground_truth.manager.prefetch_ground_truth_for_tile(tile_path)
            >>> cached = ground_truth.manager.get_cached_ground_truth(tile_id)
        """
        if self._manager is None:
            from ign_lidar.core.ground_truth_manager import GroundTruthManager
            self._manager = GroundTruthManager()
            logger.debug("Lazy-loaded GroundTruthManager")
        return self._manager
    
    @property
    def refiner(self):
        """
        Classification refiner (lazy-loaded).
        
        Refines ground truth classification using feature-based logic.
        Only instantiated when first accessed.
        
        Returns:
            GroundTruthRefiner: Classification refinement component
        
        Example:
            >>> ground_truth.refiner.refine_all(points, labels, features)
            >>> ground_truth.refiner.refine_building_with_expanded_polygons(...)
        """
        if self._refiner is None:
            from ign_lidar.core.classification.ground_truth_refinement import (
                GroundTruthRefiner,
                GroundTruthRefinementConfig
            )
            # Create with default config
            config = GroundTruthRefinementConfig()
            self._refiner = GroundTruthRefiner(config)
            logger.debug("Lazy-loaded GroundTruthRefiner")
        return self._refiner
    
    # ===== Convenience Methods (HIGH-LEVEL API) =====
    
    def fetch_and_label(
        self,
        tile_path: str,
        points: np.ndarray,
        feature_types: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        High-level: Fetch ground truth and label points in one call.
        
        This convenience method combines fetching ground truth data and labeling
        points, handling all the coordination between fetcher, manager, and optimizer.
        
        Args:
            tile_path: Path to LAZ tile file
            points: Point cloud array [N, 3] with XYZ coordinates
            feature_types: List of feature types to fetch (e.g., ['buildings', 'roads'])
                          If None, fetches all available features
            use_cache: Whether to use cached ground truth if available
        
        Returns:
            Tuple of (labels, metadata):
                labels: Array [N] with ground truth class labels
                metadata: Dictionary with statistics and feature info
        
        Example:
            >>> labels, meta = ground_truth.fetch_and_label(
            ...     tile_path="data/tile.laz",
            ...     points=point_cloud[:, :3],
            ...     feature_types=['buildings', 'roads']
            ... )
            >>> print(f"Labeled {meta['n_labeled']} points")
        
        Note:
            This method delegates to:
            1. manager.prefetch_ground_truth_for_tile() - Fetch/cache data
            2. optimizer.label_points() - Label points
        """
        logger.info(f"Fetching and labeling points for tile: {tile_path}")
        
        # Step 1: Prefetch ground truth (uses cache if available)
        ground_truth_data = self.manager.prefetch_ground_truth_for_tile(
            tile_path=tile_path,
            feature_types=feature_types
        )
        
        if not ground_truth_data:
            logger.warning(f"No ground truth data fetched for {tile_path}")
            return np.zeros(len(points), dtype=np.int32), {
                'n_labeled': 0,
                'n_total': len(points),
                'feature_types': feature_types or []
            }
        
        # Step 2: Label points using optimizer
        labels = self.optimizer.label_points(
            points=points,
            geometries=ground_truth_data,
            use_cache=use_cache
        )
        
        # Compute statistics
        n_labeled = np.sum(labels > 0)
        metadata = {
            'n_labeled': int(n_labeled),
            'n_total': len(points),
            'label_rate': float(n_labeled) / len(points) if len(points) > 0 else 0.0,
            'feature_types': list(ground_truth_data.keys()),
            'unique_labels': np.unique(labels).tolist()
        }
        
        logger.info(
            f"Labeled {n_labeled}/{len(points)} points "
            f"({metadata['label_rate']:.1%}) with {len(metadata['unique_labels'])} classes"
        )
        
        return labels, metadata
    
    def prefetch_batch(
        self,
        tile_paths: List[str],
        feature_types: Optional[List[str]] = None,
        num_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Prefetch ground truth for multiple tiles (batch operation).
        
        This convenience method prefetches ground truth data for multiple tiles
        in parallel, improving efficiency for large-scale processing.
        
        Args:
            tile_paths: List of LAZ tile file paths
            feature_types: List of feature types to fetch
            num_workers: Number of parallel workers
        
        Returns:
            Dictionary with prefetch statistics:
                - n_tiles: Number of tiles processed
                - n_success: Number successful
                - n_failed: Number failed
                - cache_hits: Number served from cache
        
        Example:
            >>> stats = ground_truth.prefetch_batch(
            ...     tile_paths=list(Path("data").glob("*.laz")),
            ...     feature_types=['buildings', 'roads'],
            ...     num_workers=8
            ... )
            >>> print(f"Prefetched {stats['n_success']}/{stats['n_tiles']} tiles")
        
        Note:
            Delegates to manager.prefetch_ground_truth_batch()
        """
        logger.info(f"Prefetching ground truth for {len(tile_paths)} tiles")
        
        return self.manager.prefetch_ground_truth_batch(
            tile_paths=tile_paths,
            feature_types=feature_types,
            num_workers=num_workers
        )
    
    def process_tile_complete(
        self,
        tile_path: str,
        points: np.ndarray,
        features: Optional[Dict[str, np.ndarray]] = None,
        refine: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete pipeline: fetch → label → refine (optional).
        
        This is the highest-level convenience method that runs the complete
        ground truth processing pipeline for a single tile.
        
        Args:
            tile_path: Path to LAZ tile file
            points: Point cloud array [N, 3] with XYZ coordinates
            features: Optional precomputed features for refinement
                     Dictionary with feature arrays (e.g., 'normals', 'curvature')
            refine: Whether to apply refinement after labeling
        
        Returns:
            Tuple of (labels, statistics):
                labels: Array [N] with ground truth class labels (refined if requested)
                statistics: Dictionary with processing statistics
        
        Example:
            >>> labels, stats = ground_truth.process_tile_complete(
            ...     tile_path="data/tile.laz",
            ...     points=point_cloud[:, :3],
            ...     features=computed_features,
            ...     refine=True
            ... )
            >>> print(f"Processing: {stats['duration']:.2f}s")
        
        Note:
            Pipeline stages:
            1. Fetch ground truth (manager)
            2. Label points (optimizer)
            3. Refine classification (refiner, optional)
        """
        import time
        
        start_time = time.time()
        logger.info(f"Processing tile complete: {tile_path}")
        
        # Step 1 & 2: Fetch and label
        labels, label_metadata = self.fetch_and_label(tile_path, points)
        
        # Step 3: Refine (optional)
        if refine and features is not None and np.sum(labels > 0) > 0:
            logger.info("Applying classification refinement")
            labels, refine_metadata = self.refiner.refine_all(
                points=points,
                labels=labels,
                features=features
            )
            
            # Merge metadata
            statistics = {
                **label_metadata,
                'refine_metadata': refine_metadata,
                'refined': True
            }
        else:
            statistics = {
                **label_metadata,
                'refined': False,
                'refine_reason': 'disabled' if not refine else 'no_features'
            }
        
        # Add timing
        duration = time.time() - start_time
        statistics['duration'] = duration
        
        logger.info(
            f"Tile processing complete in {duration:.2f}s "
            f"({statistics['n_labeled']} points labeled)"
        )
        
        return labels, statistics
    
    def clear_all_caches(self) -> Dict[str, int]:
        """
        Clear all caches across all components.
        
        This convenience method clears caches from all components that have
        been instantiated, freeing memory.
        
        Returns:
            Dictionary with number of entries cleared per component
        
        Example:
            >>> cleared = ground_truth.clear_all_caches()
            >>> print(f"Cleared {sum(cleared.values())} cache entries")
        """
        cleared = {}
        
        if self._fetcher is not None:
            n_entries = len(self._fetcher._cache)
            self._fetcher._cache.clear()
            cleared['fetcher'] = n_entries
            logger.debug(f"Cleared {n_entries} entries from fetcher cache")
        
        if self._optimizer is not None:
            n_entries = len(self._optimizer._cache)
            self._optimizer.clear_cache()
            cleared['optimizer'] = n_entries
            logger.debug(f"Cleared {n_entries} entries from optimizer cache")
        
        if self._manager is not None:
            n_entries = len(self._manager._ground_truth_cache)
            self._manager.clear_cache()
            cleared['manager'] = n_entries
            logger.debug(f"Cleared {n_entries} entries from manager cache")
        
        total_cleared = sum(cleared.values())
        logger.info(f"Cleared total of {total_cleared} cache entries across all components")
        
        return cleared
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all instantiated components.
        
        Returns:
            Dictionary with statistics from each component
        
        Example:
            >>> stats = ground_truth.get_statistics()
            >>> print(f"Optimizer cache hit rate: {stats['optimizer']['hit_rate']:.1%}")
        """
        statistics = {
            'components_loaded': []
        }
        
        if self._fetcher is not None:
            statistics['components_loaded'].append('fetcher')
            statistics['fetcher'] = {
                'cache_size': len(self._fetcher._cache)
            }
        
        if self._optimizer is not None:
            statistics['components_loaded'].append('optimizer')
            statistics['optimizer'] = self._optimizer.get_cache_stats()
        
        if self._manager is not None:
            statistics['components_loaded'].append('manager')
            statistics['manager'] = {
                'cache_size': len(self._manager._ground_truth_cache)
            }
        
        if self._refiner is not None:
            statistics['components_loaded'].append('refiner')
            statistics['refiner'] = {
                'config': str(self._refiner.config)
            }
        
        return statistics
    
    def __repr__(self) -> str:
        """String representation showing loaded components."""
        loaded = []
        if self._fetcher is not None:
            loaded.append('fetcher')
        if self._optimizer is not None:
            loaded.append('optimizer')
        if self._manager is not None:
            loaded.append('manager')
        if self._refiner is not None:
            loaded.append('refiner')
        
        if loaded:
            components_str = ', '.join(loaded)
            return f"GroundTruthHub(loaded=[{components_str}])"
        else:
            return "GroundTruthHub(no components loaded yet)"


# Create singleton instance for convenient import
ground_truth = GroundTruthHub()

__all__ = ['GroundTruthHub', 'ground_truth']
