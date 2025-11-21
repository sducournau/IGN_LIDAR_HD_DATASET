"""
Feature Engine - Wrapper for FeatureOrchestrator

This module provides a thin wrapper around FeatureOrchestrator to decouple
LiDARProcessor from direct feature computation logic.

The FeatureEngine delegates all feature-related operations to the underlying
FeatureOrchestrator while providing a cleaner API for the processor.

Architecture Pattern:
    LiDARProcessor
        └── FeatureEngine (wrapper/facade)
            └── FeatureOrchestrator (implementation)

Benefits:
- Separation of concerns: processor doesn't need to know FeatureOrchestrator details
- Easier to test: can mock FeatureEngine without touching FeatureOrchestrator
- Cleaner API: processor only sees feature operations it needs
- Future flexibility: can swap implementation without changing processor

Author: Phase 2 Refactoring - Session 3
Date: November 21, 2025
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from ..features.orchestrator import FeatureOrchestrator
from ..features.feature_modes import FeatureMode

logger = logging.getLogger(__name__)

__all__ = ["FeatureEngine"]


class FeatureEngine:
    """
    Wrapper for FeatureOrchestrator providing a clean API for feature operations.
    
    This class acts as a facade to FeatureOrchestrator, exposing only the methods
    needed by LiDARProcessor and hiding internal complexity.
    
    Args:
        config: Configuration dict containing feature settings
        
    Example:
        >>> engine = FeatureEngine(config)
        >>> features = engine.compute_features(tile_data)
        >>> print(f"Computed {len(features)} features")
        
    Note:
        This is a thin wrapper - all actual computation is delegated to
        FeatureOrchestrator. The wrapper only provides API simplification.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the feature engine.
        
        Args:
            config: OmegaConf configuration containing:
                - features: Feature computation settings
                - data_sources: RGB/NIR/GPU configuration
                - processor: General processing settings
                
        Raises:
            InitializationError: If FeatureOrchestrator initialization fails
        """
        self.config = config
        self.orchestrator = FeatureOrchestrator(config)
        logger.debug("✅ FeatureEngine initialized with FeatureOrchestrator V5")
        
    def compute_features(
        self,
        tile_data: Dict[str, Any],
        mode: Optional[FeatureMode] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute features for a tile using the orchestrator.
        
        Args:
            tile_data: Dictionary containing:
                - points: Point cloud array [N, 3+]
                - classification: Classification array [N]
                - laz_file: Path to LAZ file (for RGB/NIR)
                - bbox: Bounding box (optional)
            mode: Feature mode override (MINIMAL, LOD2, LOD3, FULL)
            
        Returns:
            Dictionary mapping feature names to arrays
            
        Raises:
            FeatureComputationError: If feature computation fails
            
        Example:
            >>> features = engine.compute_features(tile_data)
            >>> print(features.keys())
            dict_keys(['normals', 'curvature', 'planarity', ...])
        """
        return self.orchestrator.compute_features(tile_data=tile_data, mode=mode)
        
    def get_feature_list(self, mode: Optional[FeatureMode] = None) -> List[str]:
        """
        Get list of feature names for a given mode.
        
        Args:
            mode: Feature mode (defaults to configured mode)
            
        Returns:
            List of feature names that will be computed
            
        Example:
            >>> features = engine.get_feature_list(FeatureMode.LOD2)
            >>> print(len(features))
            12
        """
        return self.orchestrator.get_feature_list(mode=mode)
        
    def validate_mode(self, mode: FeatureMode) -> bool:
        """
        Validate if a feature mode is supported.
        
        Args:
            mode: FeatureMode to validate
            
        Returns:
            True if mode is valid and supported
            
        Example:
            >>> if engine.validate_mode(FeatureMode.LOD3):
            ...     features = engine.compute_features(tile_data, mode=FeatureMode.LOD3)
        """
        return self.orchestrator.validate_mode(mode)
        
    def filter_features(
        self,
        features: Dict[str, np.ndarray],
        mode: Optional[FeatureMode] = None
    ) -> Dict[str, np.ndarray]:
        """
        Filter features to match a specific mode.
        
        Useful when you've computed more features than needed and want to
        reduce to a specific feature set.
        
        Args:
            features: Dictionary of all computed features
            mode: Target feature mode
            
        Returns:
            Filtered dictionary containing only features for the mode
            
        Example:
            >>> all_features = engine.compute_features(tile_data, mode=FeatureMode.FULL)
            >>> lod2_features = engine.filter_features(all_features, mode=FeatureMode.LOD2)
            >>> print(len(all_features), len(lod2_features))
            38 12
        """
        return self.orchestrator.filter_features(features, mode=mode)
        
    def clear_cache(self):
        """
        Clear feature computation cache.
        
        Useful for freeing memory or forcing recomputation.
        
        Example:
            >>> engine.compute_features(tile_data)  # First computation
            >>> engine.clear_cache()  # Clear cached results
            >>> engine.compute_features(tile_data)  # Recompute
        """
        self.orchestrator.clear_cache()
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance statistics for feature computation.
        
        Returns:
            Dictionary with performance metrics:
                - total_computations: Number of feature computations
                - total_time: Total computation time
                - avg_time: Average time per computation
                - throughput: Points per second
                
        Example:
            >>> summary = engine.get_performance_summary()
            >>> print(f"Throughput: {summary['throughput']:.0f} pts/s")
            Throughput: 125000 pts/s
        """
        return self.orchestrator.get_performance_summary()
        
    def print_performance_insights(self, detailed: bool = False):
        """
        Print human-readable performance insights.
        
        Args:
            detailed: If True, include detailed breakdowns
            
        Example:
            >>> engine.print_performance_insights(detailed=True)
            Feature Computation Performance:
            - Total tiles: 50
            - Avg time: 2.3s
            - Throughput: 125k pts/s
            ...
        """
        self.orchestrator.print_performance_insights(detailed=detailed)
        
    # ==================== Property Accessors ====================
    # These provide access to underlying orchestrator properties
    # without exposing the full orchestrator object
    
    @property
    def use_gpu(self) -> bool:
        """Whether GPU acceleration is enabled."""
        return self.orchestrator.use_gpu
        
    @property
    def has_rgb(self) -> bool:
        """Whether RGB data fetching is available."""
        return self.orchestrator.has_rgb
        
    @property
    def has_infrared(self) -> bool:
        """Whether NIR data fetching is available."""
        return self.orchestrator.has_infrared
        
    @property
    def has_gpu(self) -> bool:
        """Whether GPU is available and configured."""
        return self.orchestrator.has_gpu
        
    @property
    def feature_mode(self) -> FeatureMode:
        """Current feature computation mode."""
        return self.orchestrator.feature_mode
        
    @property
    def rgb_fetcher(self):
        """Access to RGB fetcher (for processor compatibility)."""
        return self.orchestrator.rgb_fetcher
        
    @property
    def infrared_fetcher(self):
        """Access to infrared fetcher (for processor compatibility)."""
        return self.orchestrator.infrared_fetcher
        
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"FeatureEngine("
            f"mode={self.feature_mode.value}, "
            f"gpu={self.use_gpu}, "
            f"rgb={self.has_rgb}, "
            f"nir={self.has_infrared})"
        )
