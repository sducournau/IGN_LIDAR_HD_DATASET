"""
Transport Detection Module with Multi-Mode Support

This module provides enhanced road and railway detection capabilities for different
classification modes: ASPRS standard, ASPRS extended, and LOD2 training.

Modes:
- ASPRS_STANDARD: Simple road (11) and rail (10) detection
- ASPRS_EXTENDED: Detailed road types (motorway, primary, etc.) and rail types
- LOD2: Ground-level transport surfaces for LOD2 training

Author: Transport Detection Enhancement
Date: October 15, 2025
Updated: October 16, 2025 - Integrated unified thresholds (Issue #8)
"""

import logging
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np

from .classification_thresholds import ClassificationThresholds

logger = logging.getLogger(__name__)


# ============================================================================
# Transport Detection Modes
# ============================================================================

class TransportDetectionMode(str, Enum):
    """Transport detection modes with different strategies and outputs."""
    ASPRS_STANDARD = "asprs_standard"    # Simple road (11) and rail (10)
    ASPRS_EXTENDED = "asprs_extended"    # Detailed road/rail types (32-49)
    LOD2 = "lod2"                        # Ground-level transport (for training)
    

# ============================================================================
# Configuration Classes
# ============================================================================

class TransportDetectionConfig:
    """
    Configuration for transport detection with mode-specific thresholds.
    
    Note: Thresholds now use ClassificationThresholds for consistency across modules.
    See: docs/AUDIT_ACTION_PLAN.md - Issue #8
    """
    
    def __init__(self, mode: TransportDetectionMode = TransportDetectionMode.ASPRS_STANDARD, 
                 strict_mode: bool = False):
        """
        Initialize transport detection configuration.
        
        Args:
            mode: Detection mode (ASPRS_STANDARD, ASPRS_EXTENDED, or LOD2)
            strict_mode: If True, use stricter thresholds (for urban areas)
        """
        self.mode = mode
        self.strict_mode = strict_mode
        
        # === Common Thresholds (from ClassificationThresholds) ===
        # Height thresholds - now using updated values (Issue #1, #4)
        self.road_height_max = ClassificationThresholds.ROAD_HEIGHT_MAX_STRICT if strict_mode else ClassificationThresholds.ROAD_HEIGHT_MAX
        self.road_height_min = ClassificationThresholds.ROAD_HEIGHT_MIN
        self.rail_height_max = ClassificationThresholds.RAIL_HEIGHT_MAX_STRICT if strict_mode else ClassificationThresholds.RAIL_HEIGHT_MAX
        self.rail_height_min = ClassificationThresholds.RAIL_HEIGHT_MIN
        
        # Geometric thresholds
        self.road_planarity_min = ClassificationThresholds.ROAD_PLANARITY_MIN_STRICT if strict_mode else ClassificationThresholds.ROAD_PLANARITY_MIN
        self.rail_planarity_min = ClassificationThresholds.RAIL_PLANARITY_MIN_STRICT if strict_mode else ClassificationThresholds.RAIL_PLANARITY_MIN
        self.road_roughness_max = ClassificationThresholds.ROAD_ROUGHNESS_MAX
        self.rail_roughness_max = ClassificationThresholds.RAIL_ROUGHNESS_MAX
        
        # Intensity thresholds
        self.road_intensity_min = ClassificationThresholds.ROAD_INTENSITY_MIN
        self.road_intensity_max = ClassificationThresholds.ROAD_INTENSITY_MAX
        self.rail_intensity_min = ClassificationThresholds.RAIL_INTENSITY_MIN
        self.rail_intensity_max = ClassificationThresholds.RAIL_INTENSITY_MAX
        
        # === Mode-specific Configuration ===
        if mode == TransportDetectionMode.ASPRS_STANDARD:
            # ASPRS Standard: Simple binary classification
            self.detect_road_types = False
            self.detect_rail_types = False
            self.use_ground_truth = True
            self.ground_truth_priority = True
            self.road_intensity_filter = True
            self.rail_intensity_filter = False
            self.intensity_asphalt_min = 0.2
            self.intensity_asphalt_max = 0.6
            
        elif mode == TransportDetectionMode.ASPRS_EXTENDED:
            # ASPRS Extended: Detailed classification with road/rail types
            self.detect_road_types = True
            self.detect_rail_types = True
            self.use_ground_truth = True
            self.ground_truth_priority = True
            self.road_intensity_filter = True
            self.rail_intensity_filter = True
            self.intensity_asphalt_min = 0.2
            self.intensity_asphalt_max = 0.6
            self.intensity_concrete_min = 0.4
            self.intensity_concrete_max = 0.8
            self.intensity_gravel_min = 0.3
            self.intensity_gravel_max = 0.7
            # Road type detection parameters
            self.motorway_width_min = 10.0
            self.primary_width_min = 7.0
            self.secondary_width_min = 5.0
            self.service_width_max = 4.0
            
        elif mode == TransportDetectionMode.LOD2:
            # LOD2: All transport as ground-level surfaces
            self.detect_road_types = False
            self.detect_rail_types = False
            self.use_ground_truth = True
            self.ground_truth_priority = True
            self.road_intensity_filter = False  # More lenient for training
            self.rail_intensity_filter = False
            # LOD2-specific
            self.classify_as_ground = True      # Roads/rails are ground class
            self.separate_road_rail = False     # Don't separate in LOD2
            
        # === Detection Strategies (flags) ===
        self.use_road_ground_truth = True
        self.use_rail_ground_truth = True
        self.use_geometric_detection = True
        self.use_intensity_refinement = True
        self.use_buffer_tolerance = True
        self.buffer_tolerance = 0.5             # Additional buffer (m)
        

# ============================================================================
# Transport Detection Strategies
# ============================================================================

class TransportDetector:
    """
    Advanced transport detector supporting multiple detection modes.
    
    Implements different strategies optimized for:
    - ASPRS_STANDARD: Simple road/rail classification
    - ASPRS_EXTENDED: Detailed road/rail type classification
    - LOD2: Ground-level transport surfaces for training
    """
    
    def __init__(self, config: Optional[TransportDetectionConfig] = None):
        """
        Initialize transport detector.
        
        Args:
            config: Detection configuration (defaults to ASPRS_STANDARD mode)
        """
        self.config = config or TransportDetectionConfig(mode=TransportDetectionMode.ASPRS_STANDARD)
        logger.info(f"🚗🚂 Transport Detector initialized in {self.config.mode.upper()} mode")
        
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
        road_types: Optional[np.ndarray] = None,  # Road type per point
        rail_types: Optional[np.ndarray] = None,  # Rail type per point
        road_widths: Optional[np.ndarray] = None, # Road width per point
        points: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
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
            Tuple of (refined_labels, detection_stats)
        """
        if self.config.mode == TransportDetectionMode.ASPRS_STANDARD:
            return self._detect_asprs_standard(
                labels, height, planarity, roughness, intensity, normals,
                road_ground_truth_mask, rail_ground_truth_mask
            )
        elif self.config.mode == TransportDetectionMode.ASPRS_EXTENDED:
            return self._detect_asprs_extended(
                labels, height, planarity, roughness, intensity, normals,
                road_ground_truth_mask, rail_ground_truth_mask,
                road_types, rail_types, road_widths
            )
        elif self.config.mode == TransportDetectionMode.LOD2:
            return self._detect_lod2(
                labels, height, planarity, roughness, intensity, normals,
                road_ground_truth_mask, rail_ground_truth_mask
            )
        else:
            raise ValueError(f"Unsupported detection mode: {self.config.mode}")
    
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
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        ASPRS Standard mode: Simple road (11) and rail (10) detection.
        
        Detection strategies:
        1. Ground truth (if available)
        2. Geometric detection (planarity, height, roughness)
        3. Intensity refinement (for roads)
        """
        refined = labels.copy()
        stats = {
            'roads_ground_truth': 0,
            'roads_geometric': 0,
            'rails_ground_truth': 0,
            'rails_geometric': 0,
            'total_roads': 0,
            'total_rails': 0
        }
        
        ASPRS_ROAD = 11
        ASPRS_RAIL = 10
        
        # === ROAD DETECTION ===
        
        # Strategy 1: Ground truth roads
        if road_ground_truth_mask is not None and self.config.use_road_ground_truth:
            if self.config.ground_truth_priority:
                road_points = road_ground_truth_mask & (refined != ASPRS_ROAD)
                refined[road_points] = ASPRS_ROAD
                stats['roads_ground_truth'] = road_points.sum()
        
        # Strategy 2: Geometric road detection
        if self.config.use_geometric_detection:
            # Compute horizontality from normals if available
            horizontality = None
            if normals is not None:
                horizontality = np.abs(normals[:, 2])
            
            road_candidates = (
                (height < self.config.road_height_max) &
                (planarity > self.config.road_planarity_min) &
                (refined != ASPRS_ROAD)  # Not already classified
            )
            
            # Add roughness constraint
            if roughness is not None:
                road_candidates = road_candidates & (roughness < self.config.road_roughness_max)
            
            # Add horizontality constraint
            if horizontality is not None:
                road_candidates = road_candidates & (horizontality > 0.9)
            
            # Intensity refinement for asphalt
            if self.config.road_intensity_filter and intensity is not None:
                asphalt_intensity = (
                    (intensity > self.config.intensity_asphalt_min) &
                    (intensity < self.config.intensity_asphalt_max)
                )
                road_candidates = road_candidates & asphalt_intensity
            
            refined[road_candidates] = ASPRS_ROAD
            stats['roads_geometric'] = road_candidates.sum()
        
        # === RAIL DETECTION ===
        
        # Strategy 1: Ground truth rails
        if rail_ground_truth_mask is not None and self.config.use_rail_ground_truth:
            if self.config.ground_truth_priority:
                rail_points = rail_ground_truth_mask & (refined != ASPRS_RAIL)
                refined[rail_points] = ASPRS_RAIL
                stats['rails_ground_truth'] = rail_points.sum()
        
        # Strategy 2: Geometric rail detection
        if self.config.use_geometric_detection:
            # Rails: similar to roads but can be slightly rougher
            rail_candidates = (
                (height < self.config.rail_height_max) &
                (planarity > self.config.rail_planarity_min) &
                (refined != ASPRS_RAIL) &
                (refined != ASPRS_ROAD)  # Don't overlap with roads
            )
            
            if roughness is not None:
                rail_candidates = rail_candidates & (roughness < self.config.rail_roughness_max)
            
            # Rails are more linear - check if we can add linearity constraint later
            
            refined[rail_candidates] = ASPRS_RAIL
            stats['rails_geometric'] = rail_candidates.sum()
        
        # Calculate totals
        stats['total_roads'] = (refined == ASPRS_ROAD).sum()
        stats['total_rails'] = (refined == ASPRS_RAIL).sum()
        
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
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        ASPRS Extended mode: Detailed road and rail type classification.
        
        Road types: Motorway (32), Primary (33), Secondary (34), etc.
        Rail types: Main line (standard 10), Tram (special handling)
        
        Uses BD TOPO attributes to classify road/rail types.
        """
        refined = labels.copy()
        stats = {
            'roads_ground_truth': 0,
            'rails_ground_truth': 0,
            'motorways': 0,
            'primary_roads': 0,
            'secondary_roads': 0,
            'residential_roads': 0,
            'service_roads': 0,
            'other_roads': 0,
            'main_railways': 0,
            'service_railways': 0,
            'total_roads': 0,
            'total_rails': 0
        }
        
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
                stats['motorways'] = ((refined == ASPRS_MOTORWAY) & road_points).sum()
                stats['primary_roads'] = ((refined == ASPRS_PRIMARY) & road_points).sum()
                stats['secondary_roads'] = ((refined == ASPRS_SECONDARY) & road_points).sum()
                stats['residential_roads'] = ((refined == ASPRS_RESIDENTIAL) & road_points).sum()
                stats['service_roads'] = ((refined == ASPRS_SERVICE) & road_points).sum()
                stats['other_roads'] = (
                    road_points.sum() - stats['motorways'] - stats['primary_roads'] -
                    stats['secondary_roads'] - stats['residential_roads'] - stats['service_roads']
                )
            else:
                # No type information, use standard road class
                refined[road_points] = ASPRS_ROAD
            
            stats['roads_ground_truth'] = road_points.sum()
        
        # === RAIL DETECTION WITH TYPES ===
        
        if rail_ground_truth_mask is not None and self.config.use_rail_ground_truth:
            rail_points = rail_ground_truth_mask
            
            # Classify by type if available
            if rail_types is not None and self.config.detect_rail_types:
                refined[rail_points] = rail_types[rail_points]
                
                # Count types
                stats['main_railways'] = ((refined == ASPRS_RAIL) & rail_points).sum()
                # Could add tram, metro, etc. if codes defined
            else:
                refined[rail_points] = ASPRS_RAIL
            
            stats['rails_ground_truth'] = rail_points.sum()
        
        # Calculate totals
        road_classes = [ASPRS_ROAD, ASPRS_MOTORWAY, ASPRS_PRIMARY, ASPRS_SECONDARY,
                       ASPRS_TERTIARY, ASPRS_RESIDENTIAL, ASPRS_SERVICE]
        stats['total_roads'] = np.isin(refined, road_classes).sum()
        stats['total_rails'] = (refined == ASPRS_RAIL).sum()
        
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
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        LOD2 mode: Roads and rails as ground-level surfaces.
        
        In LOD2 training, roads and rails are typically part of the ground class.
        This mode helps validate and refine ground classification.
        """
        refined = labels.copy()
        stats = {
            'roads_validated': 0,
            'rails_validated': 0,
            'transport_ground_total': 0
        }
        
        LOD2_GROUND = 9
        
        # In LOD2, transport surfaces are ground class
        # We validate but don't change the class
        
        if road_ground_truth_mask is not None:
            # Ensure road points are classified as ground
            road_as_ground = road_ground_truth_mask & (refined != LOD2_GROUND)
            refined[road_as_ground] = LOD2_GROUND
            stats['roads_validated'] = road_ground_truth_mask.sum()
        
        if rail_ground_truth_mask is not None:
            # Ensure rail points are classified as ground
            rail_as_ground = rail_ground_truth_mask & (refined != LOD2_GROUND)
            refined[rail_as_ground] = LOD2_GROUND
            stats['rails_validated'] = rail_ground_truth_mask.sum()
        
        # Geometric validation of transport surfaces
        if self.config.use_geometric_detection:
            transport_candidates = (
                (height < max(self.config.road_height_max, self.config.rail_height_max)) &
                (planarity > min(self.config.road_planarity_min, self.config.rail_planarity_min))
            )
            
            if roughness is not None:
                transport_candidates = transport_candidates & (
                    roughness < max(self.config.road_roughness_max, self.config.rail_roughness_max)
                )
            
            # These are likely transport, ensure they're ground class
            transport_as_ground = transport_candidates & (refined != LOD2_GROUND)
            refined[transport_as_ground] = LOD2_GROUND
        
        stats['transport_ground_total'] = stats['roads_validated'] + stats['rails_validated']
        
        return refined, stats


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
    config: Optional[TransportDetectionConfig] = None
) -> Tuple[np.ndarray, Dict[str, int]]:
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
        Tuple of (refined_labels, detection_stats)
    """
    # Parse mode
    if isinstance(mode, str):
        mode = mode.lower()
        if mode not in ['asprs_standard', 'asprs_extended', 'lod2']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'asprs_standard', 'asprs_extended', or 'lod2'")
        mode_enum = TransportDetectionMode(mode)
    else:
        mode_enum = mode
    
    # Create config if not provided
    if config is None:
        config = TransportDetectionConfig(mode=mode_enum)
    
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
    'TransportDetectionMode',
    'TransportDetectionConfig',
    'TransportDetector',
    'detect_transport_multi_mode'
]
