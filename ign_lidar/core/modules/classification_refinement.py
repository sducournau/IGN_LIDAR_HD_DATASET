"""
Classification Refinement Module

This module refines LOD2/LOD3 classifications using additional data sources:
- Ground truth from WFS (building/vegetation polygons)
- NDVI values for vegetation detection
- Geometric features for better discrimination
- Height information for building vs vegetation

Improves classification accuracy beyond basic ASPRS remapping.
"""

import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

# Import new building detection module
from .building_detection import (
    BuildingDetectionMode,
    BuildingDetectionConfig,
    BuildingDetector,
    detect_buildings_multi_mode
)

# Import new transport detection module
from .transport_detection import (
    TransportDetectionMode,
    TransportDetectionConfig,
    TransportDetector,
    detect_transport_multi_mode
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class RefinementConfig:
    """Configuration for classification refinement."""
    
    # NDVI thresholds for vegetation classification
    NDVI_VEGETATION_MIN = 0.3      # Minimum NDVI for vegetation
    NDVI_HIGH_VEG_MIN = 0.5        # Minimum NDVI for healthy high vegetation
    NDVI_LOW_VEG_MAX = 0.6         # Maximum NDVI for low vegetation
    
    # Height thresholds (meters)
    LOW_VEG_HEIGHT_MAX = 2.0       # Maximum height for low vegetation
    HIGH_VEG_HEIGHT_MIN = 1.5      # Minimum height for high vegetation
    BUILDING_HEIGHT_MIN = 2.5      # Minimum height for buildings
    VEHICLE_HEIGHT_MAX = 3.0       # Maximum height for vehicles
    VEHICLE_HEIGHT_MIN = 0.5       # Minimum height for vehicles
    ROAD_HEIGHT_MAX = 0.5          # Maximum height above ground for road surfaces
    
    # Geometric feature thresholds - General
    PLANARITY_FLAT_MIN = 0.7       # Minimum planarity for flat surfaces (ground, roofs)
    PLANARITY_ROAD_MIN = 0.8       # Minimum planarity for road surfaces (very flat)
    PLANARITY_BUILDING_MIN = 0.5   # Minimum planarity for building walls
    VERTICALITY_WALL_MIN = 0.7     # Minimum verticality for walls
    ROUGHNESS_ROAD_MAX = 0.05      # Maximum roughness for smooth road surfaces
    
    # Building-specific geometric thresholds (ENHANCED)
    HORIZONTALITY_ROOF_MIN = 0.85  # Minimum horizontality for roof surfaces (very horizontal)
    ROOF_PLANARITY_MIN = 0.7       # Minimum planarity for roof surfaces (very flat)
    WALL_SCORE_MIN = 0.35          # Minimum wall score (planarity × verticality)
    ROOF_SCORE_MIN = 0.5           # Minimum roof score (planarity × horizontality)
    ANISOTROPY_BUILDING_MIN = 0.5  # Minimum anisotropy for organized structures
    LINEARITY_EDGE_MIN = 0.4       # Minimum linearity for building edges
    LINEARITY_BUILDING_STRUCT_MIN = 0.3  # Minimum linearity for structural elements
    
    # Geometric feature thresholds - Vegetation specific
    CURVATURE_VEG_MIN = 0.02       # Minimum curvature for vegetation (complex surfaces)
    CURVATURE_VEG_TYPICAL = 0.05   # Typical curvature for vegetation (branches, leaves)
    ROUGHNESS_VEG_MIN = 0.03       # Minimum roughness for vegetation (irregular surfaces)
    ROUGHNESS_VEG_TYPICAL = 0.08   # Typical roughness for dense vegetation
    PLANARITY_VEG_MAX = 0.4        # Maximum planarity for vegetation (non-flat)
    LINEARITY_TREE_MIN = 0.3       # Minimum linearity for tree trunks (vertical structures)
    INTENSITY_VEG_MAX = 0.5        # Maximum intensity for vegetation (lower than buildings)
    
    # Ground truth confidence
    USE_GROUND_TRUTH = True        # Use ground truth when available
    GROUND_TRUTH_PRIORITY = True   # Ground truth overrides other signals
    
    # Refinement flags
    REFINE_VEGETATION = True       # Refine vegetation using NDVI + height
    REFINE_BUILDINGS = True        # Refine buildings using ground truth + geometry
    REFINE_GROUND = True           # Refine ground using planarity + height
    REFINE_ROADS = True            # Refine roads using ground truth + geometry
    REFINE_VEHICLES = True         # Detect vehicles using height + size
    
    # Road-specific parameters
    ROAD_BUFFER_TOLERANCE = 0.3    # Tolerance for matching points to road polygons (meters)
    ROAD_INTENSITY_FILTER = True   # Use intensity to refine road detection
    ROAD_MIN_INTENSITY = 0.2       # Minimum intensity for asphalt roads
    ROAD_MAX_INTENSITY = 0.6       # Maximum intensity for asphalt roads


# ============================================================================
# Refinement Functions
# ============================================================================

def refine_vegetation_classification(
    labels: np.ndarray,
    ndvi: Optional[np.ndarray],
    height: Optional[np.ndarray],
    curvature: Optional[np.ndarray] = None,
    roughness: Optional[np.ndarray] = None,
    planarity: Optional[np.ndarray] = None,
    linearity: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    config: RefinementConfig = None
) -> Tuple[np.ndarray, int]:
    """
    Refine vegetation classification using NDVI and geometric features.
    
    Advanced vegetation classification using multiple signals:
    - NDVI: Primary vegetation indicator (chlorophyll absorption)
    - Height: Distinguish low vegetation (grass, shrubs) from high vegetation (trees)
    - Curvature: High curvature indicates complex surfaces (branches, leaves)
    - Roughness: Vegetation typically has higher surface roughness than buildings
    - Planarity: Low planarity indicates irregular surfaces (vegetation)
    - Linearity: Trees often show vertical linear structures (trunks)
    - Intensity: Vegetation typically has lower intensity than man-made structures
    
    Improves distinction between:
    - Low vegetation (grass, shrubs) vs high vegetation (trees)
    - Vegetation vs non-vegetation (buildings, ground)
    - Healthy vs stressed vegetation
    
    Args:
        labels: Current classification labels [N]
        ndvi: NDVI values [N], range [-1, 1]
        height: Height above ground [N] in meters
        curvature: Surface curvature [N], higher for complex surfaces
        roughness: Surface roughness [N], higher for irregular surfaces
        planarity: Planarity [N], range [0, 1], low for vegetation
        linearity: Linearity [N], range [0, 1], can indicate tree trunks
        intensity: LiDAR intensity [N], range [0, 1]
        config: Refinement configuration
        
    Returns:
        Tuple of (refined_labels, num_changed)
    """
    if config is None:
        config = RefinementConfig()
    
    if not config.REFINE_VEGETATION:
        return labels, 0
    
    refined = labels.copy()
    num_changed = 0
    
    # LOD2 class IDs
    LOD2_VEG_LOW = 10
    LOD2_VEG_HIGH = 11
    LOD2_GROUND = 9
    LOD2_OTHER = 14
    
    # Only process points currently classified as vegetation or uncertain
    veg_mask = np.isin(labels, [LOD2_VEG_LOW, LOD2_VEG_HIGH, LOD2_OTHER, LOD2_GROUND])
    
    # Geometric thresholds for vegetation
    CURVATURE_VEG_MIN = 0.02     # Vegetation has higher curvature (complex surfaces)
    ROUGHNESS_VEG_MIN = 0.03     # Vegetation has higher roughness
    PLANARITY_VEG_MAX = 0.4      # Vegetation has low planarity (irregular)
    INTENSITY_VEG_MAX = 0.5      # Vegetation typically has lower intensity
    
    # Build multi-criteria vegetation confidence score
    veg_confidence = np.zeros(len(labels), dtype=np.float32)
    confidence_components = 0
    
    # 1. NDVI - Primary indicator (weight: 0.4)
    if ndvi is not None:
        # Strong positive NDVI is highly indicative of vegetation
        ndvi_score = np.clip((ndvi - 0.1) / 0.5, 0, 1)  # Normalize 0.1-0.6 → 0-1
        veg_confidence += ndvi_score * 0.4
        confidence_components += 0.4
    
    # 2. Curvature - High curvature indicates complex surfaces like branches (weight: 0.15)
    if curvature is not None:
        # Normalize curvature: higher is more vegetation-like
        curv_score = np.clip(curvature / 0.1, 0, 1)  # 0-0.1 → 0-1
        veg_confidence += curv_score * 0.15
        confidence_components += 0.15
    
    # 3. Roughness - Vegetation has irregular surfaces (weight: 0.15)
    if roughness is not None:
        # Normalize roughness: higher is more vegetation-like
        rough_score = np.clip(roughness / 0.15, 0, 1)  # 0-0.15 → 0-1
        veg_confidence += rough_score * 0.15
        confidence_components += 0.15
    
    # 4. Planarity - Low planarity indicates non-flat surfaces (weight: 0.15)
    if planarity is not None:
        # Invert planarity: low planarity is vegetation-like
        plan_score = 1.0 - np.clip(planarity, 0, 1)
        veg_confidence += plan_score * 0.15
        confidence_components += 0.15
    
    # 5. Intensity - Vegetation typically has lower intensity (weight: 0.15)
    if intensity is not None:
        # Invert intensity: low intensity is vegetation-like
        intensity_score = 1.0 - np.clip(intensity, 0, 1)
        veg_confidence += intensity_score * 0.15
        confidence_components += 0.15
    
    # Normalize confidence by available components
    if confidence_components > 0:
        veg_confidence = veg_confidence / confidence_components
    
    # Apply vegetation detection based on confidence threshold
    high_confidence_veg = (veg_confidence > 0.6) & veg_mask
    medium_confidence_veg = (veg_confidence > 0.4) & (veg_confidence <= 0.6) & veg_mask
    
    if height is not None:
        # Refine with height information
        
        # High vegetation: high confidence + tall
        high_veg_candidates = (
            high_confidence_veg & 
            (height > config.HIGH_VEG_HEIGHT_MIN)
        )
        
        # Low vegetation: high confidence + short
        low_veg_candidates = (
            high_confidence_veg &
            (height <= config.LOW_VEG_HEIGHT_MAX)
        )
        
        # Medium confidence: use NDVI as tie-breaker if available
        if ndvi is not None:
            # Medium confidence + high NDVI → likely vegetation
            medium_high_veg = (
                medium_confidence_veg &
                (ndvi > config.NDVI_HIGH_VEG_MIN) &
                (height > config.HIGH_VEG_HEIGHT_MIN)
            )
            medium_low_veg = (
                medium_confidence_veg &
                (ndvi > config.NDVI_VEGETATION_MIN) &
                (height <= config.LOW_VEG_HEIGHT_MAX)
            )
            
            high_veg_candidates = high_veg_candidates | medium_high_veg
            low_veg_candidates = low_veg_candidates | medium_low_veg
        
        # Apply refinements
        changed_high = high_veg_candidates & (refined != LOD2_VEG_HIGH)
        refined[high_veg_candidates] = LOD2_VEG_HIGH
        num_changed += changed_high.sum()
        
        changed_low = low_veg_candidates & (refined != LOD2_VEG_LOW)
        refined[low_veg_candidates] = LOD2_VEG_LOW
        num_changed += changed_low.sum()
        
        # Reclassify low confidence + low NDVI as ground
        if ndvi is not None:
            non_veg = (
                veg_mask & 
                (veg_confidence < 0.3) & 
                (ndvi < 0.2) & 
                (height < 0.3)
            )
            if non_veg.any():
                changed_ground = non_veg & (refined != LOD2_GROUND)
                refined[non_veg] = LOD2_GROUND
                num_changed += changed_ground.sum()
    
    else:
        # No height available - use confidence scores only
        # High confidence → low vegetation (conservative)
        veg_detected = high_confidence_veg
        changed = veg_detected & (refined != LOD2_VEG_LOW)
        refined[veg_detected] = LOD2_VEG_LOW
        num_changed += changed.sum()
    
    return refined, num_changed


def refine_building_classification(
    labels: np.ndarray,
    height: Optional[np.ndarray],
    planarity: Optional[np.ndarray],
    verticality: Optional[np.ndarray],
    ground_truth_mask: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    linearity: Optional[np.ndarray] = None,
    anisotropy: Optional[np.ndarray] = None,
    wall_score: Optional[np.ndarray] = None,
    roof_score: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    mode: str = 'lod2',
    config: RefinementConfig = None
) -> Tuple[np.ndarray, int]:
    """
    Refine building classification using advanced geometric features and ground truth.
    
    Enhanced building detection using multiple geometric attributes with mode support:
    - ASPRS mode: Simple building detection (class 6)
    - LOD2 mode: Building elements (walls, roofs)
    - LOD3 mode: Detailed architectural elements (windows, doors, etc.)
    
    Features used:
    - Horizontality: For roof detection (high horizontal planarity)
    - Verticality: For wall detection (high vertical planarity)
    - Planarity: For distinguishing flat building surfaces from vegetation
    - Linearity: For detecting building edges and structural elements
    - Anisotropy: For identifying organized structures vs random vegetation
    - Wall/Roof scores: Combined metrics for building element classification
    - Height: For distinguishing buildings from ground-level structures
    
    Improves distinction between:
    - Building walls vs other vertical structures (poles, trees)
    - Building roofs vs ground surfaces
    - Buildings vs vegetation with similar height
    
    Args:
        labels: Current classification labels [N]
        height: Height above ground [N] in meters
        planarity: Planarity values [N], range [0, 1]
        verticality: Verticality values [N], range [0, 1]
        ground_truth_mask: Boolean mask [N] for building points from WFS
        normals: Surface normals [N, 3] for enhanced orientation analysis
        linearity: Linearity values [N], range [0, 1] for edge detection
        anisotropy: Anisotropy values [N], range [0, 1] for structure detection
        wall_score: Pre-computed wall likelihood [N] (planarity × verticality)
        roof_score: Pre-computed roof likelihood [N] (planarity × horizontality)
        curvature: Surface curvature [N] (for LOD3 detail detection)
        intensity: LiDAR intensity [N] (for LOD3 opening detection)
        points: Point coordinates [N, 3] (for LOD3 spatial analysis)
        mode: Detection mode ('asprs', 'lod2', or 'lod3')
        config: Refinement configuration
        
    Returns:
        Tuple of (refined_labels, num_changed)
    """
    if config is None:
        config = RefinementConfig()
    
    if not config.REFINE_BUILDINGS:
        return labels, 0
    
    # Use new building detection module if mode-aware detection is requested
    if mode in ['asprs', 'lod2', 'lod3']:
        # Prepare features dictionary
        features = {
            'height': height,
            'planarity': planarity,
            'verticality': verticality,
            'normals': normals,
            'linearity': linearity,
            'anisotropy': anisotropy,
            'wall_score': wall_score,
            'roof_score': roof_score,
            'curvature': curvature,
            'intensity': intensity,
            'points': points
        }
        
        # Use the new building detection system
        try:
            refined, stats = detect_buildings_multi_mode(
                labels=labels,
                features=features,
                mode=mode,
                ground_truth_mask=ground_truth_mask,
                config=None  # Use default mode-specific config
            )
            
            # Calculate total changes
            num_changed = np.sum(refined != labels)
            
            # Log detection statistics
            logger.debug(f"  Building detection ({mode.upper()} mode): {num_changed} points updated")
            for key, value in stats.items():
                if value > 0:
                    logger.debug(f"    - {key}: {value:,} points")
            
            return refined, num_changed
            
        except Exception as e:
            logger.warning(f"  Mode-aware building detection failed: {e}, falling back to legacy method")
    
    # === LEGACY BUILDING DETECTION (Fallback) ===
    
    refined = labels.copy()
    num_changed = 0
    
    # LOD2 class IDs
    LOD2_WALL = 0
    LOD2_GROUND = 9
    LOD2_OTHER = 14
    
    # Use ground truth if available and prioritized
    if ground_truth_mask is not None and config.USE_GROUND_TRUTH:
        if config.GROUND_TRUTH_PRIORITY:
            # Override with ground truth
            changed = ground_truth_mask & (refined != LOD2_WALL)
            refined[ground_truth_mask] = LOD2_WALL
            num_changed += changed.sum()
            
            logger.debug(f"  Ground truth: {changed.sum()} points assigned to buildings")
    
    # === ENHANCED GEOMETRIC-BASED BUILDING DETECTION ===
    
    # Strategy 1: Wall detection using verticality + planarity
    if height is not None and verticality is not None:
        # Building wall candidates: tall + vertical + planar
        wall_candidates = (
            (height > config.BUILDING_HEIGHT_MIN) &
            (verticality > config.VERTICALITY_WALL_MIN) &
            (labels != LOD2_WALL)  # Not already classified as building
        )
        
        if planarity is not None:
            # Require high planarity (flat walls, not trees)
            wall_candidates = wall_candidates & (planarity > config.PLANARITY_BUILDING_MIN)
        
        # Use wall score if available (more precise)
        if wall_score is not None:
            # Wall score combines planarity × verticality
            wall_candidates = wall_candidates & (wall_score > 0.35)
        
        # Apply wall detection
        changed = wall_candidates & (refined != LOD2_WALL)
        refined[wall_candidates] = LOD2_WALL
        num_changed += changed.sum()
    
    # Strategy 2: Roof detection using horizontality + planarity
    if height is not None and normals is not None:
        # Compute horizontality from normals
        horizontality = np.abs(normals[:, 2]) if normals is not None else None
        
        if horizontality is not None and planarity is not None:
            # Roof candidates: elevated + horizontal + planar
            roof_candidates = (
                (height > config.BUILDING_HEIGHT_MIN) &
                (horizontality > 0.85) &  # Nearly horizontal surfaces
                (planarity > 0.7) &  # Very planar
                (labels != LOD2_WALL)  # Not already classified
            )
            
            # Use roof score if available (more precise)
            if roof_score is not None:
                roof_candidates = roof_candidates & (roof_score > 0.5)
            
            # Apply roof detection
            changed = roof_candidates & (refined != LOD2_WALL)
            refined[roof_candidates] = LOD2_WALL
            num_changed += changed.sum()
    
    # Strategy 3: Use anisotropy to distinguish organized structures
    if anisotropy is not None and height is not None:
        # High anisotropy indicates directional structure (building edges, walls)
        # Low anisotropy indicates isotropic scatter (vegetation)
        structured_candidates = (
            (height > config.BUILDING_HEIGHT_MIN) &
            (anisotropy > 0.5) &  # Organized directional structure
            (labels != LOD2_WALL)
        )
        
        if planarity is not None:
            # Require some planarity to avoid misclassifying edges
            structured_candidates = structured_candidates & (planarity > 0.3)
        
        # Apply structured element detection
        changed = structured_candidates & (refined != LOD2_WALL)
        refined[structured_candidates] = LOD2_WALL
        num_changed += changed.sum()
    
    # Strategy 4: Edge detection for building corners and structural elements
    if linearity is not None and height is not None:
        # High linearity indicates edges (building corners, roof edges)
        edge_candidates = (
            (height > config.BUILDING_HEIGHT_MIN) &
            (linearity > 0.4) &  # Strong linear structure
            (labels != LOD2_WALL)
        )
        
        if verticality is not None:
            # Vertical or horizontal edges (not diagonal vegetation)
            edge_candidates = edge_candidates & (
                (verticality > 0.7) | (verticality < 0.3)  # Vertical or horizontal
            )
        
        # Apply edge detection
        changed = edge_candidates & (refined != LOD2_WALL)
        refined[edge_candidates] = LOD2_WALL
        num_changed += changed.sum()
    
    return refined, num_changed


def refine_ground_classification(
    labels: np.ndarray,
    height: Optional[np.ndarray],
    planarity: Optional[np.ndarray],
    config: RefinementConfig = None
) -> Tuple[np.ndarray, int]:
    """
    Refine ground classification using planarity and height.
    
    Improves ground surface detection using:
    - High planarity (flat surfaces)
    - Low height above ground
    
    Args:
        labels: Current classification labels [N]
        height: Height above ground [N] in meters
        planarity: Planarity values [N], range [0, 1]
        config: Refinement configuration
        
    Returns:
        Tuple of (refined_labels, num_changed)
    """
    if config is None:
        config = RefinementConfig()
    
    if not config.REFINE_GROUND:
        return labels, 0
    
    refined = labels.copy()
    num_changed = 0
    
    # LOD2 class IDs
    LOD2_GROUND = 9
    LOD2_OTHER = 14
    
    if height is not None and planarity is not None:
        # Ground candidates: very flat + very low
        ground_candidates = (
            (height < 0.3) &  # Very close to ground
            (planarity > config.PLANARITY_FLAT_MIN) &  # Very flat
            (labels == LOD2_OTHER)  # Currently unclassified
        )
        
        # Apply refinement
        changed = ground_candidates & (refined != LOD2_GROUND)
        refined[ground_candidates] = LOD2_GROUND
        num_changed += changed.sum()
    
    return refined, num_changed


def refine_road_classification(
    labels: np.ndarray,
    points: np.ndarray,
    height: Optional[np.ndarray],
    planarity: Optional[np.ndarray],
    roughness: Optional[np.ndarray],
    intensity: Optional[np.ndarray],
    ground_truth_road_mask: Optional[np.ndarray] = None,
    ground_truth_rail_mask: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    road_types: Optional[np.ndarray] = None,
    rail_types: Optional[np.ndarray] = None,
    mode: str = 'lod2',
    config: RefinementConfig = None
) -> Tuple[np.ndarray, int]:
    """
    Refine road and rail classification using ground truth and geometric features.
    
    Enhanced transport detection using multiple data sources with mode support:
    - ASPRS_STANDARD mode: Simple road (11) and rail (10) detection
    - ASPRS_EXTENDED mode: Detailed road/rail types
    - LOD2 mode: Roads and rails as ground-level surfaces
    
    Improves transport surface detection using:
    - Ground truth road/rail polygons from WFS (BD TOPO®)
    - High planarity (very flat surfaces)
    - Low roughness (smooth surfaces)
    - Low height above ground
    - Typical surface intensity values
    
    Args:
        labels: Current classification labels [N]
        points: Point coordinates [N, 3] (X, Y, Z)
        height: Height above ground [N] in meters
        planarity: Planarity values [N], range [0, 1]
        roughness: Surface roughness [N]
        intensity: Intensity values [N], normalized [0, 1]
        ground_truth_road_mask: Boolean mask [N] for road points from WFS
        ground_truth_rail_mask: Boolean mask [N] for rail points from WFS
        normals: Surface normals [N, 3]
        road_types: Road type classifications [N] from BD TOPO
        rail_types: Rail type classifications [N] from BD TOPO
        mode: Detection mode ('asprs_standard', 'asprs_extended', or 'lod2')
        config: Refinement configuration
        
    Returns:
        Tuple of (refined_labels, num_changed)
    """
    if config is None:
        config = RefinementConfig()
    
    if not config.REFINE_ROADS:
        return labels, 0
    
    # Use new transport detection module if mode-aware detection is requested
    if mode in ['asprs_standard', 'asprs_extended', 'lod2']:
        # Prepare features dictionary
        features = {
            'height': height,
            'planarity': planarity,
            'roughness': roughness,
            'intensity': intensity,
            'normals': normals,
            'points': points
        }
        
        # Use the new transport detection system
        try:
            refined, stats = detect_transport_multi_mode(
                labels=labels,
                features=features,
                mode=mode,
                road_ground_truth_mask=ground_truth_road_mask,
                rail_ground_truth_mask=ground_truth_rail_mask,
                road_types=road_types,
                rail_types=rail_types,
                config=None  # Use default mode-specific config
            )
            
            # Calculate total changes
            num_changed = np.sum(refined != labels)
            
            # Log detection statistics
            logger.debug(f"  Transport detection ({mode.upper()} mode): {num_changed} points updated")
            for key, value in stats.items():
                if value > 0:
                    logger.debug(f"    - {key}: {value:,} points")
            
            return refined, num_changed
            
        except Exception as e:
            logger.warning(f"  Mode-aware transport detection failed: {e}, falling back to legacy method")
    
    # === LEGACY ROAD DETECTION (Fallback) ===
    
    refined = labels.copy()
    num_changed = 0
    
    # LOD2 class IDs
    LOD2_GROUND = 9
    LOD2_OTHER = 14
    
    # Use ground truth if available and prioritized
    if ground_truth_road_mask is not None and config.USE_GROUND_TRUTH:
        if config.GROUND_TRUTH_PRIORITY:
            # Override with ground truth
            changed = ground_truth_road_mask & (refined != LOD2_GROUND)
            refined[ground_truth_road_mask] = LOD2_GROUND
            num_changed += changed.sum()
            
            logger.debug(f"  Ground truth roads: {changed.sum()} points assigned")
    
    # Use geometric features to refine road detection
    if height is not None and planarity is not None:
        # Road candidates: very flat + very low + smooth
        road_candidates = (
            (height < config.ROAD_HEIGHT_MAX) &  # Very close to ground
            (planarity > config.PLANARITY_ROAD_MIN) &  # Very flat
            (labels == LOD2_GROUND)  # Already classified as ground
        )
        
        # Add roughness constraint if available
        if roughness is not None:
            road_candidates = road_candidates & (roughness < config.ROUGHNESS_ROAD_MAX)
        
        # Add intensity constraint if enabled and available
        if config.ROAD_INTENSITY_FILTER and intensity is not None:
            # Asphalt typically has moderate intensity (not too dark, not too bright)
            intensity_match = (
                (intensity > config.ROAD_MIN_INTENSITY) &
                (intensity < config.ROAD_MAX_INTENSITY)
            )
            road_candidates = road_candidates & intensity_match
        
        # Keep as ground class but mark as refined roads
        # (In LOD2, roads are part of ground class)
        # No actual class change needed, just validation
        n_road_validated = road_candidates.sum()
        if n_road_validated > 0:
            logger.debug(f"  Geometric features: {n_road_validated} road points validated")
    
    return refined, num_changed


def detect_vehicles(
    labels: np.ndarray,
    height: Optional[np.ndarray],
    density: Optional[np.ndarray],
    config: RefinementConfig = None
) -> Tuple[np.ndarray, int]:
    """
    Detect vehicles using height and density characteristics.
    
    Vehicles typically have:
    - Medium height (0.5 - 3.0m)
    - High point density (metal surfaces)
    - Not classified as buildings or vegetation
    
    Args:
        labels: Current classification labels [N]
        height: Height above ground [N] in meters
        density: Point density [N]
        config: Refinement configuration
        
    Returns:
        Tuple of (refined_labels, num_changed)
    """
    if config is None:
        config = RefinementConfig()
    
    if not config.REFINE_VEHICLES:
        return labels, 0
    
    refined = labels.copy()
    num_changed = 0
    
    # LOD2 class IDs
    LOD2_VEHICLE = 13
    LOD2_WALL = 0
    LOD2_VEG_LOW = 10
    LOD2_VEG_HIGH = 11
    LOD2_OTHER = 14
    
    if height is not None:
        # Vehicle candidates: medium height + not building/vegetation
        vehicle_candidates = (
            (height > config.VEHICLE_HEIGHT_MIN) &
            (height < config.VEHICLE_HEIGHT_MAX) &
            ~np.isin(labels, [LOD2_WALL, LOD2_VEG_LOW, LOD2_VEG_HIGH])
        )
        
        # Optionally use density if available
        if density is not None:
            # Vehicles often have higher density than surroundings
            high_density = density > np.median(density[vehicle_candidates]) if vehicle_candidates.any() else False
            vehicle_candidates = vehicle_candidates & high_density
        
        # Apply refinement
        changed = vehicle_candidates & (refined != LOD2_VEHICLE)
        refined[vehicle_candidates] = LOD2_VEHICLE
        num_changed += changed.sum()
    
    return refined, num_changed


# ============================================================================
# Main Refinement Function
# ============================================================================

def refine_classification(
    labels: np.ndarray,
    features: Dict[str, np.ndarray],
    ground_truth_data: Optional[Dict[str, Any]] = None,
    config: RefinementConfig = None,
    lod_level: str = 'LOD2',
    logger_instance: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Main function to refine classification labels using all available data.
    
    Args:
        labels: Initial classification labels [N]
        features: Dictionary of computed features:
            - 'ndvi': NDVI values [N] (optional) - vegetation indicator
            - 'height': Height above ground [N] (optional) - elevation
            - 'curvature': Surface curvature [N] (optional) - surface complexity
            - 'roughness': Surface roughness [N] (optional) - surface irregularity
            - 'planarity': Planarity values [N] (optional) - flatness measure
            - 'linearity': Linearity values [N] (optional) - linear structure detection
            - 'verticality': Verticality values [N] (optional) - vertical structure
            - 'density': Point density [N] (optional) - local point density
            - 'intensity': LiDAR intensity [N] (optional) - return intensity
            - 'points': XYZ coordinates [N, 3] (optional) - for spatial queries
        ground_truth_data: Optional ground truth data:
            - 'building_mask': Boolean mask for building points [N]
            - 'vegetation_mask': Boolean mask for vegetation points [N]
            - 'road_mask': Boolean mask for road points [N]
        config: Refinement configuration
        lod_level: 'LOD2' or 'LOD3'
        logger_instance: Optional logger
        
    Returns:
        Tuple of (refined_labels, refinement_stats)
        refinement_stats contains counts of changed points per category
    """
    if config is None:
        config = RefinementConfig()
    
    log = logger_instance or logger
    
    if lod_level != 'LOD2':
        # Currently only LOD2 refinement is implemented
        log.debug(f"Classification refinement not yet implemented for {lod_level}")
        return labels, {}
    
    refined = labels.copy()
    stats = {}
    
    log.info("  🔧 Refining classification using additional features...")
    
    # Extract features
    ndvi = features.get('ndvi')
    height = features.get('height')
    planarity = features.get('planarity')
    verticality = features.get('verticality')
    density = features.get('density')
    roughness = features.get('roughness')
    curvature = features.get('curvature')
    linearity = features.get('linearity')
    intensity = features.get('intensity')
    points = features.get('points')  # XYZ coordinates needed for road matching
    
    # Extract ground truth masks
    building_mask = None
    road_mask = None
    if ground_truth_data is not None:
        building_mask = ground_truth_data.get('building_mask')
        road_mask = ground_truth_data.get('road_mask')
    
    # 1. Refine vegetation using NDVI + geometric features
    if ndvi is not None or curvature is not None or roughness is not None:
        refined, n_veg = refine_vegetation_classification(
            labels=refined,
            ndvi=ndvi,
            height=height,
            curvature=curvature,
            roughness=roughness,
            planarity=planarity,
            linearity=linearity,
            intensity=intensity,
            config=config
        )
        stats['vegetation_refined'] = n_veg
        if n_veg > 0:
            # Build feature list for logging
            used_features = []
            if ndvi is not None:
                used_features.append('NDVI')
            if curvature is not None:
                used_features.append('curvature')
            if roughness is not None:
                used_features.append('roughness')
            if planarity is not None:
                used_features.append('planarity')
            if height is not None:
                used_features.append('height')
            features_str = ', '.join(used_features) if used_features else 'geometric features'
            log.info(f"    ✓ Vegetation: {n_veg:,} points refined using {features_str}")
    
    # 2. Refine buildings using ground truth + enhanced geometric features
    if height is not None:
        # Extract additional geometric features for enhanced building detection
        normals = features.get('normals')  # [N, 3] surface normals
        linearity_feat = features.get('linearity')  # Edge detection
        anisotropy_feat = features.get('anisotropy')  # Structure detection
        wall_score_feat = features.get('wall_score')  # Wall likelihood
        roof_score_feat = features.get('roof_score')  # Roof likelihood
        curvature_feat = features.get('curvature')  # For LOD3 detail detection
        intensity_feat = features.get('intensity')  # For LOD3 opening detection
        points_feat = features.get('points')  # For LOD3 spatial analysis
        
        # Determine detection mode from lod_level
        if lod_level == 'LOD3':
            detection_mode = 'lod3'
        elif lod_level == 'LOD2':
            detection_mode = 'lod2'
        else:
            detection_mode = 'asprs'  # Default for other levels
        
        refined, n_bldg = refine_building_classification(
            labels=refined,
            height=height,
            planarity=planarity,
            verticality=verticality,
            ground_truth_mask=building_mask,
            normals=normals,
            linearity=linearity_feat,
            anisotropy=anisotropy_feat,
            wall_score=wall_score_feat,
            roof_score=roof_score_feat,
            curvature=curvature_feat,
            intensity=intensity_feat,
            points=points_feat,
            mode=detection_mode,
            config=config
        )
        stats['buildings_refined'] = n_bldg
        if n_bldg > 0:
            # Build feature list for logging
            used_features = ['height', 'planarity', 'verticality']
            if normals is not None:
                used_features.append('normals/horizontality')
            if linearity_feat is not None:
                used_features.append('linearity')
            if anisotropy_feat is not None:
                used_features.append('anisotropy')
            if wall_score_feat is not None or roof_score_feat is not None:
                used_features.append('wall/roof scores')
            if detection_mode == 'lod3':
                used_features.append('LOD3 details')
            features_str = ', '.join(used_features)
            log.info(f"    ✓ Buildings ({detection_mode.upper()}): {n_bldg:,} points refined using {features_str}")
    
    # 3. Refine roads and railways using ground truth + geometry
    if points is not None and height is not None:
        # Extract rail ground truth if available
        rail_mask = None
        if ground_truth_data is not None:
            rail_mask = ground_truth_data.get('rail_mask')
        
        # Determine transport detection mode from lod_level
        if lod_level == 'LOD2':
            transport_mode = 'lod2'
        else:
            transport_mode = 'asprs_standard'  # Default for other levels
        
        # Get road/rail type information if available
        road_types_feat = features.get('road_types')
        rail_types_feat = features.get('rail_types')
        
        refined, n_transport = refine_road_classification(
            labels=refined,
            points=points,
            height=height,
            planarity=planarity,
            roughness=roughness,
            intensity=intensity,
            ground_truth_road_mask=road_mask,
            ground_truth_rail_mask=rail_mask,
            normals=normals,
            road_types=road_types_feat,
            rail_types=rail_types_feat,
            mode=transport_mode,
            config=config
        )
        stats['transport_refined'] = n_transport
        if n_transport > 0:
            log.info(f"    ✓ Transport ({transport_mode.upper()}): {n_transport:,} points refined (roads + rails)")
    
    # 4. Refine ground using planarity + height
    if height is not None and planarity is not None:
        refined, n_gnd = refine_ground_classification(
            refined, height, planarity, config
        )
        stats['ground_refined'] = n_gnd
        if n_gnd > 0:
            log.info(f"    ✓ Ground: {n_gnd:,} points refined")
    
    # 5. Detect vehicles
    if height is not None:
        refined, n_veh = detect_vehicles(
            refined, height, density, config
        )
        stats['vehicles_detected'] = n_veh
        if n_veh > 0:
            log.info(f"    ✓ Vehicles: {n_veh:,} points detected")
    
    # Summary
    total_refined = sum(stats.values())
    if total_refined > 0:
        pct = (total_refined / len(labels)) * 100
        log.info(f"  ✅ Total refined: {total_refined:,} points ({pct:.1f}%)")
    else:
        log.debug("  No classification refinements applied")
    
    return refined, stats


# ============================================================================
# LOD2/LOD3 Building Element Classification
# ============================================================================

def classify_lod2_building_elements(
    points: np.ndarray,
    labels: np.ndarray,
    normals: Optional[np.ndarray],
    planarity: Optional[np.ndarray],
    height: Optional[np.ndarray],
    linearity: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    config: RefinementConfig = None
) -> np.ndarray:
    """
    Classify building points into LOD2 elements: walls, roofs, chimneys, etc.
    
    Uses advanced geometric attributes to distinguish:
    - Walls (vertical planar surfaces)
    - Flat roofs (horizontal planar surfaces)
    - Gable roofs (tilted planar surfaces)
    - Hip roofs (multiple tilted planes)
    - Chimneys (vertical protrusions above roof level)
    - Dormers (vertical elements within roof surface)
    - Balconies (horizontal protrusions from walls)
    - Overhangs (horizontal extensions beyond walls)
    
    Args:
        points: Point coordinates [N, 3] (X, Y, Z)
        labels: Current LOD2 classification labels [N]
        normals: Surface normals [N, 3]
        planarity: Planarity values [N], range [0, 1]
        height: Height above ground [N] in meters
        linearity: Linearity values [N], range [0, 1] for edge detection
        curvature: Surface curvature [N] for detail detection
        config: Refinement configuration
        
    Returns:
        refined_labels: Enhanced LOD2 classification [N]
    """
    if config is None:
        config = RefinementConfig()
    
    refined = labels.copy()
    
    # LOD2 class IDs (from ign_lidar.classes)
    LOD2_WALL = 0
    LOD2_ROOF_FLAT = 1
    LOD2_ROOF_GABLE = 2
    LOD2_ROOF_HIP = 3
    LOD2_CHIMNEY = 4
    LOD2_DORMER = 5
    LOD2_BALCONY = 6
    LOD2_OVERHANG = 7
    
    # Only process points already identified as buildings
    building_mask = (labels == LOD2_WALL)
    if not building_mask.any():
        return refined
    
    if normals is None or planarity is None or height is None:
        logger.warning("Cannot classify LOD2 elements: missing required features (normals, planarity, height)")
        return refined
    
    # Extract building points
    building_idx = np.where(building_mask)[0]
    build_normals = normals[building_mask]
    build_planarity = planarity[building_mask]
    build_height = height[building_mask]
    
    # Compute orientation metrics
    horizontality = np.abs(build_normals[:, 2])  # |nz| - 1 for horizontal, 0 for vertical
    verticality = 1.0 - horizontality
    
    # === WALL vs ROOF CLASSIFICATION ===
    
    # Walls: High verticality + high planarity
    wall_score = verticality * build_planarity
    is_wall = (wall_score > 0.35) & (verticality > 0.7)
    
    # Roofs: High horizontality + high planarity
    roof_score = horizontality * build_planarity
    is_roof = (roof_score > 0.5) & (horizontality > 0.7)
    
    # === ROOF TYPE CLASSIFICATION ===
    
    roof_idx = np.where(is_roof)[0]
    if roof_idx.size > 0:
        roof_heights = build_height[roof_idx]
        roof_horizontality = horizontality[roof_idx]
        
        # Flat roofs: Nearly horizontal (slope < 15°)
        is_flat_roof = roof_horizontality > 0.95  # cos(15°) ≈ 0.966
        
        # Sloped roofs: Moderately tilted (15° - 45°)
        is_sloped_roof = (roof_horizontality < 0.95) & (roof_horizontality > 0.7)
        
        # Distinguish gable vs hip roofs using spatial distribution
        # This would require neighborhood analysis - simplified for now
        # Gable: Two main roof planes
        # Hip: Four or more roof planes
        # For now, classify all sloped roofs as gable (most common)
        
        # Apply roof classifications
        flat_roof_idx = building_idx[roof_idx[is_flat_roof]]
        gable_roof_idx = building_idx[roof_idx[is_sloped_roof]]
        
        refined[flat_roof_idx] = LOD2_ROOF_FLAT
        refined[gable_roof_idx] = LOD2_ROOF_GABLE
    
    # === CHIMNEY DETECTION ===
    # Chimneys: Vertical + elevated above typical roof level
    if roof_idx.size > 0:
        median_roof_height = np.median(build_height[roof_idx])
        
        chimney_candidates = (
            verticality > 0.85 &  # Very vertical
            (build_planarity > 0.6) &  # Planar (brick/concrete)
            (build_height > median_roof_height + 0.5)  # Above roof
        )
        
        if linearity is not None:
            build_linearity = linearity[building_mask]
            # Chimneys often have strong edges
            chimney_candidates = chimney_candidates & (build_linearity > 0.3)
        
        chimney_idx = building_idx[chimney_candidates]
        refined[chimney_idx] = LOD2_CHIMNEY
    
    # === DORMER DETECTION ===
    # Dormers: Vertical elements within roof level
    if roof_idx.size > 0:
        median_roof_height = np.median(build_height[roof_idx])
        
        dormer_candidates = (
            verticality > 0.75 &  # Mostly vertical
            (build_planarity > 0.5) &  # Planar
            (build_height >= median_roof_height - 1.0) &  # Around roof level
            (build_height <= median_roof_height + 0.5) &  # Not above roof
            ~is_wall  # Not classified as main wall
        )
        
        dormer_idx = building_idx[dormer_candidates]
        refined[dormer_idx] = LOD2_DORMER
    
    # === BALCONY DETECTION ===
    # Balconies: Horizontal protrusions at intermediate heights
    if building_idx.size > 0:
        max_height = np.max(build_height)
        
        balcony_candidates = (
            horizontality > 0.85 &  # Mostly horizontal
            (build_planarity > 0.7) &  # Very planar
            (build_height > 2.0) &  # Above ground floor
            (build_height < max_height * 0.8) &  # Below roof
            ~is_roof  # Not the main roof
        )
        
        balcony_idx = building_idx[balcony_candidates]
        refined[balcony_idx] = LOD2_BALCONY
    
    # === OVERHANG DETECTION ===
    # Overhangs: Horizontal extensions at roof edge
    if roof_idx.size > 0:
        median_roof_height = np.median(build_height[roof_idx])
        
        overhang_candidates = (
            horizontality > 0.80 &  # Mostly horizontal
            (build_planarity > 0.6) &  # Planar
            (build_height >= median_roof_height - 0.3) &  # At roof level
            (build_height <= median_roof_height + 0.3) &
            ~is_roof  # Not the main roof
        )
        
        overhang_idx = building_idx[overhang_candidates]
        refined[overhang_idx] = LOD2_OVERHANG
    
    # Ensure walls are properly classified (anything not otherwise classified)
    still_generic_building = (refined == LOD2_WALL) & building_mask & is_wall
    # These remain as walls (already correct)
    
    return refined


def classify_lod3_building_elements(
    points: np.ndarray,
    labels: np.ndarray,
    normals: Optional[np.ndarray],
    planarity: Optional[np.ndarray],
    linearity: Optional[np.ndarray],
    height: Optional[np.ndarray],
    curvature: Optional[np.ndarray] = None,
    anisotropy: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    config: RefinementConfig = None
) -> np.ndarray:
    """
    Classify building points into LOD3 elements with detailed architectural features.
    
    Uses advanced geometric attributes to distinguish:
    - Wall types (plain, with windows, with doors)
    - Detailed roof types (flat, gable, hip, mansard, gambrel)
    - Roof details (chimneys, dormers, skylights, roof edges)
    - Openings (windows, doors, garage doors)
    - Facade elements (balconies, balustrades, overhangs, pillars, cornices)
    - Foundation elements
    
    Args:
        points: Point coordinates [N, 3] (X, Y, Z)
        labels: Current LOD3 classification labels [N]
        normals: Surface normals [N, 3]
        planarity: Planarity values [N], range [0, 1]
        linearity: Linearity values [N], range [0, 1] for edge/opening detection
        height: Height above ground [N] in meters
        curvature: Surface curvature [N] for detail detection
        anisotropy: Anisotropy values [N] for structure detection
        intensity: LiDAR intensity [N] for material detection
        config: Refinement configuration
        
    Returns:
        refined_labels: Enhanced LOD3 classification [N]
    """
    if config is None:
        config = RefinementConfig()
    
    refined = labels.copy()
    
    # LOD3 class IDs (from ign_lidar.classes)
    LOD3_WALL_PLAIN = 0
    LOD3_WALL_WITH_WINDOWS = 1
    LOD3_WALL_WITH_DOOR = 2
    LOD3_ROOF_FLAT = 3
    LOD3_ROOF_GABLE = 4
    LOD3_ROOF_HIP = 5
    LOD3_ROOF_MANSARD = 6
    LOD3_ROOF_GAMBREL = 7
    LOD3_CHIMNEY = 8
    LOD3_DORMER_GABLE = 9
    LOD3_DORMER_SHED = 10
    LOD3_SKYLIGHT = 11
    LOD3_ROOF_EDGE = 12
    LOD3_WINDOW = 13
    LOD3_DOOR = 14
    LOD3_GARAGE_DOOR = 15
    LOD3_BALCONY = 16
    LOD3_BALUSTRADE = 17
    LOD3_OVERHANG = 18
    LOD3_PILLAR = 19
    LOD3_CORNICE = 20
    LOD3_FOUNDATION = 21
    
    # Only process building points
    building_mask = (labels <= LOD3_CORNICE)  # All building elements
    if not building_mask.any():
        return refined
    
    if normals is None or planarity is None or height is None:
        logger.warning("Cannot classify LOD3 elements: missing required features")
        return refined
    
    # Extract building points
    building_idx = np.where(building_mask)[0]
    build_normals = normals[building_mask]
    build_planarity = planarity[building_mask]
    build_height = height[building_mask]
    
    # Compute orientation metrics
    horizontality = np.abs(build_normals[:, 2])
    verticality = 1.0 - horizontality
    
    # === WINDOW DETECTION ===
    # Windows: Openings in walls (low planarity areas in vertical surfaces)
    if linearity is not None:
        build_linearity = linearity[building_mask]
        
        # Windows have strong edge signals (rectangular openings)
        window_candidates = (
            verticality > 0.65 &  # On vertical walls
            (build_planarity < 0.4) &  # Opening (not solid wall)
            (build_linearity > 0.5) &  # Strong rectangular edges
            (build_height > 1.0) &  # Above ground
            (build_height < 15.0)  # Typical window height range
        )
        
        window_idx = building_idx[window_candidates]
        refined[window_idx] = LOD3_WINDOW
    
    # === DOOR DETECTION ===
    # Doors: Openings at ground level
    if linearity is not None:
        build_linearity = linearity[building_mask]
        
        door_candidates = (
            verticality > 0.70 &  # On vertical walls
            (build_planarity < 0.4) &  # Opening
            (build_linearity > 0.5) &  # Strong edges
            (build_height < 3.0)  # Ground level
        )
        
        # Distinguish regular doors vs garage doors by size/intensity
        if intensity is not None:
            build_intensity = intensity[building_mask]
            # Garage doors often have different material (metal - higher intensity)
            garage_candidates = door_candidates & (build_intensity > 0.5)
            regular_door_candidates = door_candidates & (build_intensity <= 0.5)
        else:
            regular_door_candidates = door_candidates
            garage_candidates = np.zeros_like(door_candidates, dtype=bool)
        
        door_idx = building_idx[regular_door_candidates]
        garage_idx = building_idx[garage_candidates]
        refined[door_idx] = LOD3_DOOR
        refined[garage_idx] = LOD3_GARAGE_DOOR
    
    # === WALL CLASSIFICATION (with/without openings) ===
    # Classify walls based on whether they contain openings
    wall_candidates = (verticality > 0.65) & (build_planarity > 0.5)
    wall_idx = building_idx[wall_candidates]
    
    # Check if walls have nearby windows/doors
    has_openings = refined[wall_idx] == LOD3_WINDOW
    has_doors = refined[wall_idx] == LOD3_DOOR
    
    # Plain walls (no openings nearby - simplified, would need spatial analysis)
    plain_wall_idx = wall_idx[(refined[wall_idx] != LOD3_WINDOW) & 
                               (refined[wall_idx] != LOD3_DOOR)]
    refined[plain_wall_idx] = LOD3_WALL_PLAIN
    
    # === ROOF EDGE DETECTION ===
    # Roof edges: High linearity at roof boundaries
    if linearity is not None:
        build_linearity = linearity[building_mask]
        
        # Identify roof height range
        roof_candidates = (horizontality > 0.7) & (build_planarity > 0.6)
        if roof_candidates.any():
            median_roof_height = np.median(build_height[roof_candidates])
            
            edge_candidates = (
                (build_linearity > 0.6) &  # Very strong edges
                (build_height >= median_roof_height - 0.5) &  # At roof level
                (build_height <= median_roof_height + 0.5)
            )
            
            edge_idx = building_idx[edge_candidates]
            refined[edge_idx] = LOD3_ROOF_EDGE
    
    # === PILLAR DETECTION ===
    # Pillars: Vertical linear elements
    if linearity is not None:
        build_linearity = linearity[building_mask]
        
        pillar_candidates = (
            verticality > 0.90 &  # Very vertical
            (build_linearity > 0.7) &  # Very linear (cylindrical or square columns)
            (build_height > 2.0) &  # Significant height
            (build_planarity < 0.5)  # Not planar (cylindrical)
        )
        
        pillar_idx = building_idx[pillar_candidates]
        refined[pillar_idx] = LOD3_PILLAR
    
    # === CORNICE DETECTION ===
    # Cornices: Horizontal decorative elements at top of walls
    if linearity is not None and curvature is not None:
        build_linearity = linearity[building_mask]
        build_curvature = curvature[building_mask]
        
        max_height = np.max(build_height)
        
        cornice_candidates = (
            horizontality > 0.6 &  # Mostly horizontal
            (build_linearity > 0.4) &  # Some linear structure
            (build_curvature > 0.02) &  # Has some detail/curvature
            (build_height > max_height * 0.85)  # Near top of building
        )
        
        cornice_idx = building_idx[cornice_candidates]
        refined[cornice_idx] = LOD3_CORNICE
    
    # === FOUNDATION DETECTION ===
    # Foundation: Elements at/below ground level
    foundation_candidates = (
        verticality > 0.5 &  # Somewhat vertical
        (build_planarity > 0.4) &  # Somewhat planar
        (build_height < 0.5)  # At ground level
    )
    
    foundation_idx = building_idx[foundation_candidates]
    refined[foundation_idx] = LOD3_FOUNDATION
    
    # Additional LOD3 elements (balconies, balustrades, etc.) can be added
    # using similar geometric attribute analysis
    
    return refined


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'RefinementConfig',
    'refine_classification',
    'refine_vegetation_classification',
    'refine_building_classification',
    'refine_ground_classification',
    'refine_road_classification',
    'detect_vehicles',
    'classify_lod2_building_elements',
    'classify_lod3_building_elements',
]
