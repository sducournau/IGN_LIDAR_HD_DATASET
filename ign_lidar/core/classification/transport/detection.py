"""
Transport Detection Module with Multi-Mode Support

This module provides road and railway detection capabilities for different
classification modes: ASPRS standard, ASPRS extended, and LOD2 training.

Migrated from transport_detection.py to transport/detection.py (Phase 3C).

Modes:
- ASPRS_STANDARD: Simple road (11) and rail (10) detection
- ASPRS_EXTENDED: Detailed road types (motorway, primary, etc.) and rail types
- LOD2: Ground-level transport surfaces for LOD2 training

Author: Transport Detection Enhancement
Date: October 15, 2025
Updated: October 22, 2025 - Migrated to transport module (Phase 3C)
Version: 3.1.0
"""

import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

# Import from transport module base and utils
from .base import (
    TransportMode,
    DetectionConfig,
    TransportStats,
    TransportDetectionResult,
    TransportDetectorBase,
    DetectionStrategy,
)
from .utils import (
    validate_transport_height,
    check_transport_planarity,
    filter_by_roughness,
    filter_by_intensity,
    check_horizontality,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Transport Detector Implementation
# ============================================================================

class TransportDetector(TransportDetectorBase):
    """
    Advanced transport detector supporting multiple detection modes.
    
    Implements different strategies optimized for:
    - ASPRS_STANDARD: Simple road/rail classification
    - ASPRS_EXTENDED: Detailed road/rail type classification
    - LOD2: Ground-level transport surfaces for training
    
    Inherits from TransportDetectorBase and implements abstract methods.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize transport detector.
        
        Args:
            config: Detection configuration (defaults to ASPRS_STANDARD mode)
        """
        if config is None:
            config = DetectionConfig(mode=TransportMode.ASPRS_STANDARD)
        
        super().__init__(config)
        logger.info(f"🚗🚂 Transport Detector initialized in {self.config.mode.value.upper()} mode")
        
    def detect_transport(
        self,
        labels: np.ndarray,
        height: np.ndarray,
        planarity: np.ndarray,
        roughness: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        road_ground_truth_mask: Optional[np.ndarray] = None,
        rail_ground_truth_mask: Optional[np.ndarray] = None,
        road_types: Optional[np.ndarray] = None,
        rail_types: Optional[np.ndarray] = None,
        road_widths: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None
    ) -> TransportDetectionResult:
        """
        Detect roads and railways using mode-appropriate strategies.
        
        Args:
            labels: Current classification labels [N]
            height: Height above ground [N] in meters
            planarity: Planarity values [N], range [0, 1]
            roughness: Surface roughness [N]
            intensity: LiDAR intensity [N], normalized [0, 1]
            normals: Surface normals [N, 3]
            road_ground_truth_mask: Boolean mask [N] for road points
            rail_ground_truth_mask: Boolean mask [N] for rail points
            road_types: Road type classification [N] from BD TOPO
            rail_types: Rail type classification [N] from BD TOPO
            road_widths: Road width values [N] from BD TOPO
            points: Point coordinates [N, 3] (for spatial analysis)
            
        Returns:
            TransportDetectionResult with labels, stats, and confidence
        """
        if self.config.mode == TransportMode.ASPRS_STANDARD:
            refined_labels, stats = self._detect_asprs_standard(
                labels, height, planarity, roughness, intensity, normals,
                road_ground_truth_mask, rail_ground_truth_mask
            )
        elif self.config.mode == TransportMode.ASPRS_EXTENDED:
            refined_labels, stats = self._detect_asprs_extended(
                labels, height, planarity, roughness, intensity, normals,
                road_ground_truth_mask, rail_ground_truth_mask,
                road_types, rail_types, road_widths
            )
        elif self.config.mode == TransportMode.LOD2:
            refined_labels, stats = self._detect_lod2(
                labels, height, planarity, roughness, intensity, normals,
                road_ground_truth_mask, rail_ground_truth_mask
            )
        else:
            raise ValueError(f"Unsupported detection mode: {self.config.mode}")
        
        # Determine primary strategy used
        if road_ground_truth_mask is not None or rail_ground_truth_mask is not None:
            strategy = DetectionStrategy.HYBRID
        elif self.config.use_geometric_detection:
            strategy = DetectionStrategy.GEOMETRIC
        else:
            strategy = DetectionStrategy.GROUND_TRUTH
        
        # Compute confidence scores for detected transport points
        confidence = self._compute_detection_confidence(
            labels=labels,
            refined_labels=refined_labels,
            height=height,
            planarity=planarity,
            roughness=roughness,
            intensity=intensity,
            normals=normals,
            road_ground_truth_mask=road_ground_truth_mask,
            rail_ground_truth_mask=rail_ground_truth_mask
        )
        
        return TransportDetectionResult(
            labels=refined_labels,
            stats=stats,
            confidence=confidence,
            mode=self.config.mode,
            strategy=strategy
        )
    
    def _detect_asprs_standard(
        self,
        labels: np.ndarray,
        height: np.ndarray,
        planarity: np.ndarray,
        roughness: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        road_ground_truth_mask: Optional[np.ndarray],
        rail_ground_truth_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, TransportStats]:
        """
        ASPRS Standard mode: Simple road (11) and rail (10) detection.
        
        Detection strategies:
        1. Ground truth (if available)
        2. Geometric detection (planarity, height, roughness)
        3. Intensity refinement (for roads)
        """
        refined = labels.copy()
        stats = TransportStats()
        
        ASPRS_ROAD = 11
        ASPRS_RAIL = 10
        
        # === ROAD DETECTION ===
        
        # Strategy 1: Ground truth roads
        if road_ground_truth_mask is not None and self.config.use_road_ground_truth:
            if self.config.ground_truth_priority:
                road_points = road_ground_truth_mask & (refined != ASPRS_ROAD)
                refined[road_points] = ASPRS_ROAD
                stats.roads_ground_truth = road_points.sum()
        
        # Strategy 2: Geometric road detection
        if self.config.use_geometric_detection:
            # Use shared validation functions from utils
            valid_height = validate_transport_height(
                height, 
                self.config.road_height_min,
                self.config.road_height_max,
                transport_type="road"
            )
            
            valid_planarity = check_transport_planarity(
                planarity,
                self.config.road_planarity_min,
                transport_type="road"
            )
            
            road_candidates = valid_height & valid_planarity & (refined != ASPRS_ROAD)
            
            # Add roughness constraint
            if roughness is not None:
                valid_roughness = filter_by_roughness(
                    roughness,
                    self.config.road_roughness_max,
                    transport_type="road"
                )
                road_candidates = road_candidates & valid_roughness
            
            # Add horizontality constraint
            if normals is not None:
                valid_horizontal = check_horizontality(normals, horizontality_min=0.9)
                road_candidates = road_candidates & valid_horizontal
            
            # Intensity refinement for asphalt
            if self.config.road_intensity_filter and intensity is not None:
                asphalt_intensity = filter_by_intensity(
                    intensity,
                    self.config.intensity_asphalt_min,
                    self.config.intensity_asphalt_max,
                    material="asphalt"
                )
                road_candidates = road_candidates & asphalt_intensity
            
            refined[road_candidates] = ASPRS_ROAD
            stats.roads_geometric = road_candidates.sum()
        
        # === RAIL DETECTION ===
        
        # Strategy 1: Ground truth rails
        if rail_ground_truth_mask is not None and self.config.use_rail_ground_truth:
            if self.config.ground_truth_priority:
                rail_points = rail_ground_truth_mask & (refined != ASPRS_RAIL)
                refined[rail_points] = ASPRS_RAIL
                stats.rails_ground_truth = rail_points.sum()
        
        # Strategy 2: Geometric rail detection
        if self.config.use_geometric_detection:
            # Use shared validation functions
            valid_height = validate_transport_height(
                height,
                self.config.rail_height_min,
                self.config.rail_height_max,
                transport_type="railway"
            )
            
            valid_planarity = check_transport_planarity(
                planarity,
                self.config.rail_planarity_min,
                transport_type="railway"
            )
            
            rail_candidates = (
                valid_height & 
                valid_planarity & 
                (refined != ASPRS_RAIL) &
                (refined != ASPRS_ROAD)  # Don't overlap with roads
            )
            
            if roughness is not None:
                valid_roughness = filter_by_roughness(
                    roughness,
                    self.config.rail_roughness_max,
                    transport_type="railway"
                )
                rail_candidates = rail_candidates & valid_roughness
            
            refined[rail_candidates] = ASPRS_RAIL
            stats.rails_geometric = rail_candidates.sum()
        
        # Calculate totals
        stats.total_roads = (refined == ASPRS_ROAD).sum()
        stats.total_rails = (refined == ASPRS_RAIL).sum()
        
        return refined, stats
    
    def _detect_asprs_extended(
        self,
        labels: np.ndarray,
        height: np.ndarray,
        planarity: np.ndarray,
        roughness: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        road_ground_truth_mask: Optional[np.ndarray],
        rail_ground_truth_mask: Optional[np.ndarray],
        road_types: Optional[np.ndarray],
        rail_types: Optional[np.ndarray],
        road_widths: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, TransportStats]:
        """
        ASPRS Extended mode: Detailed road and rail type classification.
        
        Road types: Motorway (32), Primary (33), Secondary (34), etc.
        Rail types: Main line (standard 10), Tram (special handling)
        
        Uses BD TOPO attributes to classify road/rail types.
        """
        refined = labels.copy()
        stats = TransportStats()
        
        # Extended ASPRS codes
        ASPRS_ROAD = 11
        ASPRS_RAIL = 10
        ASPRS_MOTORWAY = 32
        ASPRS_PRIMARY = 33
        ASPRS_SECONDARY = 34
        ASPRS_TERTIARY = 35
        ASPRS_RESIDENTIAL = 36
        ASPRS_SERVICE = 37
        
        # === ROAD DETECTION WITH TYPES ===
        
        if road_ground_truth_mask is not None and self.config.use_road_ground_truth:
            road_points = road_ground_truth_mask
            
            # Classify by type if available
            if road_types is not None and self.config.detect_road_types:
                # Use road_types array to assign specific classes
                # Assumes road_types contains ASPRS codes
                refined[road_points] = road_types[road_points]
                
                # Count by type
                stats.motorways = ((refined == ASPRS_MOTORWAY) & road_points).sum()
                stats.primary_roads = ((refined == ASPRS_PRIMARY) & road_points).sum()
                stats.secondary_roads = ((refined == ASPRS_SECONDARY) & road_points).sum()
                stats.residential_roads = ((refined == ASPRS_RESIDENTIAL) & road_points).sum()
                stats.service_roads = ((refined == ASPRS_SERVICE) & road_points).sum()
                stats.other_roads = (
                    road_points.sum() - stats.motorways - stats.primary_roads -
                    stats.secondary_roads - stats.residential_roads - stats.service_roads
                )
            else:
                # No type information, use standard road class
                refined[road_points] = ASPRS_ROAD
            
            stats.roads_ground_truth = road_points.sum()
        
        # === RAIL DETECTION WITH TYPES ===
        
        if rail_ground_truth_mask is not None and self.config.use_rail_ground_truth:
            rail_points = rail_ground_truth_mask
            
            # Classify by type if available
            if rail_types is not None and self.config.detect_rail_types:
                refined[rail_points] = rail_types[rail_points]
                
                # Count types
                stats.main_railways = ((refined == ASPRS_RAIL) & rail_points).sum()
                # Could add tram, metro, etc. if codes defined
            else:
                refined[rail_points] = ASPRS_RAIL
            
            stats.rails_ground_truth = rail_points.sum()
        
        # Calculate totals
        road_classes = [ASPRS_ROAD, ASPRS_MOTORWAY, ASPRS_PRIMARY, ASPRS_SECONDARY,
                       ASPRS_TERTIARY, ASPRS_RESIDENTIAL, ASPRS_SERVICE]
        stats.total_roads = np.isin(refined, road_classes).sum()
        stats.total_rails = (refined == ASPRS_RAIL).sum()
        
        return refined, stats
    
    def _detect_lod2(
        self,
        labels: np.ndarray,
        height: np.ndarray,
        planarity: np.ndarray,
        roughness: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        road_ground_truth_mask: Optional[np.ndarray],
        rail_ground_truth_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, TransportStats]:
        """
        LOD2 mode: Roads and rails as ground-level surfaces.
        
        In LOD2 training, roads and rails are typically part of the ground class.
        This mode helps validate and refine ground classification.
        """
        refined = labels.copy()
        stats = TransportStats()
        
        LOD2_GROUND = 9
        
        # In LOD2, transport surfaces are ground class
        # We validate but don't change the class
        
        if road_ground_truth_mask is not None:
            # Ensure road points are classified as ground
            road_as_ground = road_ground_truth_mask & (refined != LOD2_GROUND)
            refined[road_as_ground] = LOD2_GROUND
            stats.roads_validated = road_ground_truth_mask.sum()
        
        if rail_ground_truth_mask is not None:
            # Ensure rail points are classified as ground
            rail_as_ground = rail_ground_truth_mask & (refined != LOD2_GROUND)
            refined[rail_as_ground] = LOD2_GROUND
            stats.rails_validated = rail_ground_truth_mask.sum()
        
        # Geometric validation of transport surfaces
        if self.config.use_geometric_detection:
            transport_height_max = max(self.config.road_height_max, self.config.rail_height_max)
            transport_planarity_min = min(self.config.road_planarity_min, self.config.rail_planarity_min)
            
            valid_height = validate_transport_height(
                height, -0.5, transport_height_max, transport_type="transport"
            )
            valid_planarity = check_transport_planarity(
                planarity, transport_planarity_min, transport_type="transport"
            )
            
            transport_candidates = valid_height & valid_planarity
            
            if roughness is not None:
                transport_roughness_max = max(self.config.road_roughness_max, self.config.rail_roughness_max)
                valid_roughness = filter_by_roughness(
                    roughness, transport_roughness_max, transport_type="transport"
                )
                transport_candidates = transport_candidates & valid_roughness
            
            # These are likely transport, ensure they're ground class
            transport_as_ground = transport_candidates & (refined != LOD2_GROUND)
            refined[transport_as_ground] = LOD2_GROUND
        
        stats.transport_ground_total = stats.roads_validated + stats.rails_validated
        
        return refined, stats
    
    def _compute_detection_confidence(
        self,
        labels: np.ndarray,
        refined_labels: np.ndarray,
        height: np.ndarray,
        planarity: np.ndarray,
        roughness: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        road_ground_truth_mask: Optional[np.ndarray],
        rail_ground_truth_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Compute confidence scores for detected transport points.
        
        Confidence is based on:
        - Ground truth presence (high confidence = 1.0)
        - Feature quality (planarity, height, roughness, intensity)
        - Multi-feature agreement
        
        Args:
            labels: Original classification labels [N]
            refined_labels: Refined classification labels [N]
            height: Height above ground [N]
            planarity: Planarity values [N]
            roughness: Surface roughness [N]
            intensity: LiDAR intensity [N]
            normals: Surface normals [N, 3]
            road_ground_truth_mask: Boolean mask for ground truth roads
            rail_ground_truth_mask: Boolean mask for ground truth rails
            
        Returns:
            Confidence scores [N] in range [0, 1]
        """
        n_points = len(labels)
        confidence = np.zeros(n_points, dtype=np.float32)
        
        # Identify changed points (newly classified as transport)
        changed_mask = labels != refined_labels
        
        if not changed_mask.any():
            return confidence
        
        # Strategy 1: Ground truth = high confidence (1.0)
        if road_ground_truth_mask is not None:
            gt_road_points = road_ground_truth_mask & changed_mask
            confidence[gt_road_points] = 1.0
        
        if rail_ground_truth_mask is not None:
            gt_rail_points = rail_ground_truth_mask & changed_mask
            confidence[gt_rail_points] = 1.0
        
        # Strategy 2: Geometric detection = feature-based confidence
        geometric_mask = changed_mask & (confidence == 0.0)
        
        if geometric_mask.any():
            # Initialize feature confidences
            planarity_conf = planarity.copy()
            
            # Height confidence (closer to ground = higher confidence)
            max_height = max(self.config.road_height_max, self.config.rail_height_max)
            height_conf = np.clip(1.0 - np.abs(height) / max_height, 0.0, 1.0)
            
            # Roughness confidence (lower roughness = higher confidence)
            roughness_conf = None
            if roughness is not None:
                max_roughness = max(self.config.road_roughness_max, self.config.rail_roughness_max)
                roughness_conf = np.clip(1.0 - roughness / max_roughness, 0.0, 1.0)
            
            # Intensity confidence (for roads, higher intensity often indicates pavement)
            intensity_conf = None
            if intensity is not None:
                # Normalize intensity to [0, 1] if needed
                intensity_norm = intensity.copy()
                if intensity_norm.max() > 1.0:
                    intensity_norm = intensity_norm / intensity_norm.max()
                # Roads typically have moderate-to-high intensity
                intensity_conf = np.clip(intensity_norm, 0.0, 1.0)
            
            # Horizontality confidence (from normals if available)
            horizontality_conf = None
            if normals is not None and normals.shape[1] == 3:
                # Vertical component of normal (should be close to 1.0 for horizontal surfaces)
                vertical_component = np.abs(normals[:, 2])
                horizontality_conf = np.clip(vertical_component, 0.0, 1.0)
            
            # Combine confidences based on available features
            combined_conf = self._combine_feature_confidences(
                planarity_conf, height_conf, roughness_conf, 
                intensity_conf, horizontality_conf
            )
            
            # Assign combined confidence to geometric detection points
            confidence[geometric_mask] = combined_conf[geometric_mask]
        
        return confidence
    
    def _combine_feature_confidences(
        self,
        planarity_conf: np.ndarray,
        height_conf: np.ndarray,
        roughness_conf: Optional[np.ndarray],
        intensity_conf: Optional[np.ndarray],
        horizontality_conf: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Combine multiple feature confidences with adaptive weighting.
        
        Args:
            planarity_conf: Planarity confidence [N]
            height_conf: Height confidence [N]
            roughness_conf: Roughness confidence [N] or None
            intensity_conf: Intensity confidence [N] or None
            horizontality_conf: Horizontality confidence [N] or None
            
        Returns:
            Combined confidence [N] in range [0, 1]
        """
        # Build weighted combination based on available features
        features = [
            (planarity_conf, 0.40),  # Planarity is most important
            (height_conf, 0.30),     # Height is second most important
        ]
        
        if roughness_conf is not None:
            features.append((roughness_conf, 0.20))
        
        if intensity_conf is not None:
            features.append((intensity_conf, 0.10))
        
        if horizontality_conf is not None:
            features.append((horizontality_conf, 0.10))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weight for _, weight in features)
        normalized_features = [(conf, weight / total_weight) for conf, weight in features]
        
        # Weighted combination
        combined = np.zeros_like(planarity_conf, dtype=np.float32)
        for conf, weight in normalized_features:
            combined += weight * conf
        
        return combined


# ============================================================================
# Convenience Functions
# ============================================================================

def detect_transport_multi_mode(
    labels: np.ndarray,
    features: Dict[str, np.ndarray],
    mode: str = 'asprs_standard',
    road_ground_truth_mask: Optional[np.ndarray] = None,
    rail_ground_truth_mask: Optional[np.ndarray] = None,
    road_types: Optional[np.ndarray] = None,
    rail_types: Optional[np.ndarray] = None,
    config: Optional[DetectionConfig] = None
) -> TransportDetectionResult:
    """
    Convenience function for transport detection with automatic mode selection.
    
    Args:
        labels: Current classification labels [N]
        features: Dictionary of computed features
        mode: Detection mode ('asprs_standard', 'asprs_extended', or 'lod2')
        road_ground_truth_mask: Optional ground truth road mask
        rail_ground_truth_mask: Optional ground truth rail mask
        road_types: Optional road type classifications from BD TOPO
        rail_types: Optional rail type classifications from BD TOPO
        config: Optional custom configuration
        
    Returns:
        TransportDetectionResult with labels, stats, and confidence
    """
    # Parse mode
    if isinstance(mode, str):
        mode = mode.lower()
        if mode not in ['asprs_standard', 'asprs_extended', 'lod2']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'asprs_standard', 'asprs_extended', or 'lod2'")
        mode_enum = TransportMode(mode)
    else:
        mode_enum = mode
    
    # Create config if not provided
    if config is None:
        config = DetectionConfig(mode=mode_enum)
    
    # Create detector
    detector = TransportDetector(config=config)
    
    # Extract features
    height = features.get('height')
    planarity = features.get('planarity')
    roughness = features.get('roughness')
    intensity = features.get('intensity')
    normals = features.get('normals')
    road_widths = features.get('road_widths')
    points = features.get('points')
    
    # Validate required features
    if height is None or planarity is None:
        raise ValueError("Height and planarity features are required for transport detection")
    
    # Run detection
    return detector.detect_transport(
        labels=labels,
        height=height,
        planarity=planarity,
        roughness=roughness,
        intensity=intensity,
        normals=normals,
        road_ground_truth_mask=road_ground_truth_mask,
        rail_ground_truth_mask=rail_ground_truth_mask,
        road_types=road_types,
        rail_types=rail_types,
        road_widths=road_widths,
        points=points
    )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'TransportDetector',
    'detect_transport_multi_mode'
]
