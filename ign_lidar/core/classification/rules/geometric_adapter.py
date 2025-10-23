"""
Geometric Rules Adapter

This module provides an adapter for the legacy GeometricRulesEngine to work
with the modern rules framework.

The adapter wraps GeometricRulesEngine and provides the standard BaseRule
interface, allowing geometric classification to be used in hierarchical
rule compositions.

Usage:
    from ign_lidar.core.classification.rules.geometric_adapter import GeometricRulesAdapter
    from ign_lidar.core.classification.rules.base import RuleConfig, RuleType, RulePriority
    
    # Create configuration
    config = RuleConfig(
        rule_id="geometric_buildings",
        rule_type=RuleType.GEOMETRIC,
        target_class=6,  # Buildings
        priority=RulePriority.HIGH
    )
    
    # Create adapter with engine parameters
    adapter = GeometricRulesAdapter(
        config=config,
        buffer_distance=2.0
    )
    
    # Use in hierarchical engine
    mask, confidence = adapter.evaluate(points, features, context)

Author: Classification Enhancement Team
Date: October 23, 2025
"""

from typing import Dict, Any, Optional, Tuple, Set
import numpy as np
import logging

from .adapters import MultiClassAdapter
from .base import RuleConfig, RuleType
from ..geometric_rules import GeometricRulesEngine

logger = logging.getLogger(__name__)


class GeometricRulesAdapter(MultiClassAdapter):
    """
    Adapter for GeometricRulesEngine to work with rules framework.
    
    This adapter wraps the legacy GeometricRulesEngine and converts its
    multi-class classification results to the standard (mask, confidence)
    format expected by the rules framework.
    
    The adapter:
    - Wraps GeometricRulesEngine with all its configuration parameters
    - Converts multi-class results to single-class masks
    - Provides confidence scores based on geometric properties
    - Integrates with hierarchical rule execution
    
    Note: GeometricRulesEngine requires ground truth geometries (buildings,
    roads) in the context dictionary.
    
    Attributes:
        engine: GeometricRulesEngine instance
        
    Example:
        >>> config = RuleConfig(
        ...     rule_id="geometric_building_buffer",
        ...     rule_type=RuleType.GEOMETRIC,
        ...     target_class=6  # Buildings
        ... )
        >>> adapter = GeometricRulesAdapter(
        ...     config=config,
        ...     buffer_distance=2.0
        ... )
        >>> # Requires ground_truth_features in context
        >>> context = {'ground_truth_features': {'buildings': gdf_buildings}}
        >>> mask, conf = adapter.evaluate(points, features, context)
    """
    
    def __init__(
        self,
        config: RuleConfig,
        building_buffer_distance: float = 2.0,
        verticality_threshold: float = 0.7,
        road_vegetation_height_threshold: float = 2.0,
        ndvi_vegetation_threshold: float = 0.3,
        use_clustering: bool = True,
        spatial_cluster_eps: float = 0.5
    ):
        """
        Initialize geometric rules adapter.
        
        Args:
            config: Rule configuration (must have rule_type=GEOMETRIC)
            building_buffer_distance: Buffer distance for building zones (meters)
            verticality_threshold: Min verticality for building points
            road_vegetation_height_threshold: Min height difference for road/veg
            ndvi_vegetation_threshold: NDVI threshold for vegetation vs road
            use_clustering: Whether to use clustering optimization
            spatial_cluster_eps: DBSCAN epsilon for clustering (meters)
        """
        # Create GeometricRulesEngine with parameters
        engine = GeometricRulesEngine(
            building_buffer_distance=building_buffer_distance,
            verticality_threshold=verticality_threshold,
            road_vegetation_height_threshold=road_vegetation_height_threshold,
            ndvi_vegetation_threshold=ndvi_vegetation_threshold,
            use_clustering=use_clustering,
            spatial_cluster_eps=spatial_cluster_eps
        )
        
        super().__init__(config, engine)
        
        logger.info(f"Created GeometricRulesAdapter for target class {config.target_class}")
    
    def get_required_features(self) -> Set[str]:
        """Get required features for geometric classification"""
        return set()  # Points (XYZ) are required but not in features dict
    
    def get_optional_features(self) -> Set[str]:
        """Get optional features that improve results"""
        return {'ndvi', 'intensities', 'rgb', 'nir', 'verticality'}
    
    def evaluate(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate geometric rule on points.
        
        This method calls the legacy GeometricRulesEngine and converts its
        multi-class results to a binary mask for the target class with
        confidence scores.
        
        Args:
            points: Point cloud array (N, 3) with XYZ coordinates
            features: Dictionary with optional 'ndvi', 'intensities', 'rgb', 'nir'
            context: Required context with 'ground_truth_features' dict and
                    optionally 'current_labels'
            
        Returns:
            Tuple of:
                - mask: Boolean array (N,) for target class matches
                - confidence: Float array (N,) with confidence [0, 1]
                
        Raises:
            ValueError: If required context missing
        """
        if context is None or 'ground_truth_features' not in context:
            raise ValueError(
                "GeometricRulesAdapter requires 'ground_truth_features' in context. "
                "Context must contain dict of feature_type -> GeoDataFrame "
                "(e.g., {'buildings': gdf_buildings, 'roads': gdf_roads})"
            )
        
        # Get ground truth features
        ground_truth_features = context['ground_truth_features']
        
        # Get current labels from context or create unclassified
        if 'current_labels' in context:
            current_labels = context['current_labels'].copy()
        else:
            current_labels = np.ones(len(points), dtype=np.int32)  # ASPRS unclassified
        
        # Get optional features
        ndvi = features.get('ndvi', None)
        intensities = features.get('intensities', None)
        rgb = features.get('rgb', None)
        nir = features.get('nir', None)
        
        # Call legacy engine
        try:
            updated_labels, stats = self.engine.apply_all_rules(
                points=points,
                labels=current_labels,
                ground_truth_features=ground_truth_features,
                ndvi=ndvi,
                intensities=intensities,
                rgb=rgb,
                nir=nir
            )
        except Exception as e:
            logger.error(f"GeometricRulesEngine evaluation failed: {e}")
            # Return empty mask on error
            n_points = len(points)
            return np.zeros(n_points, dtype=bool), np.zeros(n_points, dtype=np.float32)
        
        # Convert to mask for target class
        # Only consider points that were changed by geometric rules
        changed_mask = (updated_labels != current_labels)
        mask = changed_mask & (updated_labels == self.target_class)
        
        # Compute confidence based on geometric properties
        confidence = self._compute_geometric_confidence(
            mask=mask,
            points=points,
            features=features,
            stats=stats
        )
        
        # Log statistics
        n_matched = np.sum(mask)
        if n_matched > 0:
            logger.debug(
                f"Geometric rule {self.rule_id} matched {n_matched} points "
                f"(mean confidence: {np.mean(confidence[mask]):.3f})"
            )
            logger.debug(f"  Engine stats: {stats}")
        
        return mask, confidence
    
    def _compute_geometric_confidence(
        self,
        mask: np.ndarray,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        stats: Dict[str, int]
    ) -> np.ndarray:
        """
        Compute confidence scores based on geometric properties.
        
        Confidence is higher when geometric evidence is strong.
        
        Args:
            mask: Boolean mask of matched points
            points: Point cloud array
            features: Feature dictionary
            stats: Statistics from geometric engine
            
        Returns:
            Confidence scores [0, 1] for all points
        """
        n_points = len(mask)
        confidence = np.zeros(n_points, dtype=np.float32)
        
        if not np.any(mask):
            return confidence
        
        # Base confidence for geometric rules (conservative)
        base_confidence = 0.75
        
        # Check if verticality feature is available
        if 'verticality' in features:
            verticality = features['verticality'][mask]
            
            if self.target_class == 6:  # Buildings
                # Higher verticality = higher confidence for buildings
                vert_bonus = (verticality - 0.5) * 0.3  # Up to +0.3
                conf_values = np.clip(base_confidence + vert_bonus, 0.6, 0.95)
            else:
                # Other classes
                conf_values = np.full(np.sum(mask), base_confidence, dtype=np.float32)
        else:
            # Default confidence without verticality
            conf_values = np.full(np.sum(mask), base_confidence, dtype=np.float32)
        
        # Adjust confidence based on which rule matched
        # (inferred from target class and available stats)
        if self.target_class == 6 and stats.get('building_buffer_added', 0) > 0:
            # Building buffer zone classifications are slightly less confident
            conf_values *= 0.9
        
        confidence[mask] = conf_values
        
        return confidence


def create_geometric_building_rule(
    config: RuleConfig,
    buffer_distance: float = 2.0,
    min_verticality: float = 0.7
) -> GeometricRulesAdapter:
    """
    Convenience factory for building classification rule.
    
    Args:
        config: Rule configuration (should have target_class=6)
        buffer_distance: Buffer distance around buildings (meters)
        min_verticality: Minimum verticality for building points
        
    Returns:
        Configured GeometricRulesAdapter for buildings
    """
    return GeometricRulesAdapter(
        config=config,
        building_buffer_distance=buffer_distance,
        verticality_threshold=min_verticality
    )


def create_geometric_road_rule(
    config: RuleConfig,
    ndvi_threshold: float = 0.3,
    vertical_separation: float = 2.0
) -> GeometricRulesAdapter:
    """
    Convenience factory for road classification rule.
    
    Args:
        config: Rule configuration (should have target_class=11)
        ndvi_threshold: NDVI threshold for separating vegetation from road
        vertical_separation: Min height difference for road/vegetation
        
    Returns:
        Configured GeometricRulesAdapter for roads
    """
    return GeometricRulesAdapter(
        config=config,
        ndvi_vegetation_threshold=ndvi_threshold,
        road_vegetation_height_threshold=vertical_separation
    )
