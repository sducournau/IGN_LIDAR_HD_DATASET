"""
Optimized Classification Thresholds and Rules

This module provides optimized thresholds and decision rules for multi-level
classification. Thresholds are derived from empirical analysis and adjusted
for French topographic context (IGN data).

Features:
- NDVI thresholds optimized for different vegetation types
- Geometric feature thresholds calibrated for French architecture
- Height-based rules for urban and rural contexts
- Context-aware adaptive thresholds

Author: IGN LiDAR HD Dataset Team
Date: October 15, 2025
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Threshold Configurations
# ============================================================================

@dataclass
class NDVIThresholds:
    """Optimized NDVI thresholds for vegetation classification."""
    
    # Primary vegetation detection
    vegetation_min: float = 0.35  # Minimum NDVI for vegetation (was 0.3)
    vegetation_healthy: float = 0.5  # Healthy, dense vegetation
    vegetation_stressed: float = 0.25  # Stressed/sparse vegetation
    
    # Vegetation types
    grass_range: Tuple[float, float] = (0.3, 0.6)  # Grass/lawn NDVI range
    shrubs_range: Tuple[float, float] = (0.4, 0.7)  # Shrubs/bushes
    trees_min: float = 0.5  # Dense tree canopy
    forest_min: float = 0.6  # Dense forest
    
    # Non-vegetation
    building_max: float = 0.15  # Maximum NDVI for buildings (was 0.15)
    road_max: float = 0.2  # Maximum NDVI for paved roads
    bare_soil_max: float = 0.25  # Bare soil/ground
    water_max: float = 0.1  # Water bodies (very low NDVI)
    
    # Seasonal adjustments (multiplicative factors)
    summer_boost: float = 1.1  # NDVI tends higher in summer
    winter_reduction: float = 0.85  # NDVI lower in winter (deciduous trees)
    
    def get_context_adjusted(
        self,
        season: Optional[str] = None,
        urban_context: bool = False
    ) -> 'NDVIThresholds':
        """
        Get context-adjusted thresholds.
        
        Args:
            season: Season name ('summer', 'winter', 'spring', 'autumn')
            urban_context: Whether in urban environment (tends to lower NDVI)
        
        Returns:
            Adjusted NDVIThresholds instance
        """
        adjusted = NDVIThresholds()
        
        # Seasonal adjustment
        if season == 'summer':
            factor = self.summer_boost
        elif season == 'winter':
            factor = self.winter_reduction
        else:
            factor = 1.0
        
        # Urban adjustment (lower NDVI due to stress, sparse vegetation)
        if urban_context:
            factor *= 0.9
        
        # Apply factors
        adjusted.vegetation_min = self.vegetation_min * factor
        adjusted.vegetation_healthy = self.vegetation_healthy * factor
        adjusted.trees_min = self.trees_min * factor
        adjusted.forest_min = self.forest_min * factor
        
        return adjusted


@dataclass
class GeometricThresholds:
    """Optimized geometric feature thresholds."""
    
    # ========================================================================
    # Planarity (flatness of surface)
    # ========================================================================
    planarity_ground_min: float = 0.85  # Ground surfaces (very flat)
    planarity_road_min: float = 0.88  # Roads (extremely flat, well-maintained)
    planarity_parking_min: float = 0.82  # Parking lots (slightly less uniform)
    planarity_roof_flat_min: float = 0.75  # Flat roofs
    planarity_roof_gable_min: float = 0.70  # Gabled roofs (planar sections)
    planarity_wall_min: float = 0.60  # Building walls
    planarity_facade_min: float = 0.55  # Facades (with windows, details)
    planarity_vegetation_max: float = 0.35  # Vegetation (irregular, not flat)
    
    # ========================================================================
    # Verticality / Horizontality (surface orientation)
    # ========================================================================
    horizontality_ground_min: float = 0.90  # Ground (horizontal)
    horizontality_roof_min: float = 0.85  # Flat roofs
    horizontality_road_min: float = 0.92  # Roads (very horizontal)
    
    verticality_wall_min: float = 0.70  # Walls (vertical)
    verticality_facade_min: float = 0.65  # Facades
    verticality_tree_trunk_min: float = 0.60  # Tree trunks
    
    # ========================================================================
    # Curvature (local surface bending)
    # ========================================================================
    curvature_flat_max: float = 0.01  # Flat surfaces (road, ground, roof)
    curvature_vegetation_min: float = 0.03  # Vegetation (curved surfaces)
    curvature_vegetation_typical: float = 0.08  # Dense vegetation
    curvature_chimney_min: float = 0.05  # Cylindrical structures
    curvature_dormer_range: Tuple[float, float] = (0.02, 0.15)  # Curved roof elements
    
    # ========================================================================
    # Roughness (local surface irregularity)
    # ========================================================================
    roughness_smooth_max: float = 0.02  # Very smooth (glass, metal roofs)
    roughness_road_max: float = 0.05  # Paved roads
    roughness_concrete_max: float = 0.08  # Concrete surfaces
    roughness_roof_tile_range: Tuple[float, float] = (0.05, 0.15)  # Tiled roofs
    roughness_vegetation_min: float = 0.08  # Vegetation surfaces
    roughness_vegetation_dense: float = 0.15  # Dense vegetation
    
    # ========================================================================
    # Anisotropy (directionality of point distribution)
    # ========================================================================
    anisotropy_building_min: float = 0.50  # Organized structures
    anisotropy_road_min: float = 0.55  # Linear road structures
    anisotropy_vegetation_max: float = 0.40  # Random vegetation distribution
    
    # ========================================================================
    # Linearity (1D linear structures)
    # ========================================================================
    linearity_edge_min: float = 0.45  # Building edges, roof ridges
    linearity_road_edge_min: float = 0.50  # Road edges
    linearity_tree_trunk_min: float = 0.35  # Tree trunks (vertical lines)
    linearity_power_line_min: float = 0.70  # Power lines (very linear)
    
    # ========================================================================
    # Combined scores (product of features)
    # ========================================================================
    wall_score_min: float = 0.40  # planarity × verticality for walls
    roof_flat_score_min: float = 0.65  # planarity × horizontality for flat roofs
    roof_complex_score_min: float = 0.45  # For complex roof shapes
    ground_score_min: float = 0.75  # planarity × horizontality for ground
    road_score_min: float = 0.80  # planarity × horizontality for roads (very high)
    
    def compute_wall_score(self, planarity: float, verticality: float) -> float:
        """Compute wall likelihood score."""
        return planarity * verticality
    
    def compute_roof_score(self, planarity: float, horizontality: float) -> float:
        """Compute flat roof likelihood score."""
        return planarity * horizontality
    
    def compute_ground_score(self, planarity: float, horizontality: float) -> float:
        """Compute ground likelihood score."""
        return planarity * horizontality


@dataclass
class HeightThresholds:
    """Optimized height-based thresholds (in meters)."""
    
    # ========================================================================
    # Ground and low objects
    # ========================================================================
    ground_max: float = 0.20  # Maximum height for ground classification
    noise_min: float = -0.50  # Minimum height (below ground = noise)
    noise_max: float = 0.10  # Very low points (potential noise)
    
    # ========================================================================
    # Vegetation heights
    # ========================================================================
    grass_max: float = 0.30  # Grass/lawn
    low_vegetation_max: float = 2.0  # Shrubs, low vegetation
    medium_vegetation_range: Tuple[float, float] = (2.0, 5.0)  # Medium veg
    high_vegetation_min: float = 3.0  # Trees (min height)
    forest_canopy_min: float = 8.0  # Forest canopy (tall trees)
    
    # ========================================================================
    # Buildings
    # ========================================================================
    building_min: float = 2.5  # Minimum building height (1-story)
    wall_min: float = 2.0  # Minimum wall height
    single_story_max: float = 4.5  # Single-story building
    multi_story_min: float = 4.5  # Multi-story building
    high_rise_min: float = 15.0  # High-rise building (5+ floors)
    
    # Building components
    foundation_max: float = 1.5  # Foundation/basement
    balcony_range: Tuple[float, float] = (2.5, 15.0)  # Balcony heights
    chimney_min: float = 5.0  # Chimney (above roof)
    dormer_range: Tuple[float, float] = (3.0, 8.0)  # Dormer windows
    
    # ========================================================================
    # Roads and infrastructure
    # ========================================================================
    road_max: float = 0.50  # Roads (above ground)
    bridge_min: float = 3.0  # Bridge deck (above ground/water)
    tunnel_max: float = 0.0  # Tunnel (below or at ground level)
    
    # ========================================================================
    # Vehicles and mobile objects
    # ========================================================================
    vehicle_min: float = 0.80  # Minimum vehicle height
    vehicle_max: float = 3.5  # Maximum car height (trucks can be higher)
    truck_max: float = 4.5  # Maximum truck height
    
    # ========================================================================
    # Utilities
    # ========================================================================
    power_line_min: float = 5.0  # Power lines (above ground)
    transmission_tower_min: float = 15.0  # Transmission towers
    
    def get_building_type_from_height(self, height: float) -> str:
        """Infer building type from height."""
        if height < self.building_min:
            return "not_building"
        elif height < self.single_story_max:
            return "single_story"
        elif height < self.multi_story_min + 5:
            return "two_story"
        elif height < self.high_rise_min:
            return "multi_story"
        else:
            return "high_rise"
    
    def get_vegetation_type_from_height(self, height: float) -> str:
        """Infer vegetation type from height."""
        if height < self.grass_max:
            return "grass"
        elif height < self.low_vegetation_max:
            return "low_vegetation"
        elif height < self.medium_vegetation_range[1]:
            return "medium_vegetation"
        elif height < self.forest_canopy_min:
            return "trees"
        else:
            return "tall_trees"


@dataclass
class IntensityThresholds:
    """Optimized LiDAR intensity thresholds (normalized 0-1)."""
    
    # Material reflectivity
    water_max: float = 0.15  # Water (very low reflectivity)
    asphalt_range: Tuple[float, float] = (0.20, 0.50)  # Asphalt roads
    concrete_range: Tuple[float, float] = (0.40, 0.70)  # Concrete
    vegetation_range: Tuple[float, float] = (0.20, 0.60)  # Vegetation (variable)
    metal_min: float = 0.70  # Metal roofs (high reflectivity)
    glass_range: Tuple[float, float] = (0.10, 0.30)  # Glass (low, variable)
    
    # Building materials
    roof_tile_range: Tuple[float, float] = (0.35, 0.65)  # Clay/concrete tiles
    roof_metal_min: float = 0.70  # Metal roofs
    roof_membrane_range: Tuple[float, float] = (0.25, 0.55)  # Membrane roofs
    
    def classify_material(self, intensity: float) -> str:
        """Classify material type from intensity."""
        if intensity < self.water_max:
            return "water"
        elif self.asphalt_range[0] <= intensity <= self.asphalt_range[1]:
            return "asphalt"
        elif self.concrete_range[0] <= intensity <= self.concrete_range[1]:
            return "concrete"
        elif intensity >= self.metal_min:
            return "metal"
        elif self.vegetation_range[0] <= intensity <= self.vegetation_range[1]:
            return "vegetation"
        else:
            return "unknown"


@dataclass
class ContextThresholds:
    """Context-specific threshold adjustments."""
    
    # Urban vs rural context
    urban_vegetation_ndvi_reduction: float = 0.9  # Urban vegetation has lower NDVI
    rural_building_size_min: float = 25.0  # Min building size in rural (m²)
    urban_building_size_min: float = 15.0  # Min building size in urban (m²)
    
    # Density-based adjustments
    dense_urban_road_width_min: float = 6.0  # Minimum road width in dense urban
    sparse_rural_road_width_min: float = 3.0  # Minimum road width in rural
    
    # Terrain-based adjustments
    mountainous_height_variance_factor: float = 1.5  # More height variation in mountains
    flat_planarity_boost: float = 1.05  # Higher planarity in flat regions
    
    def get_context_type(
        self,
        building_density: float,
        road_density: float
    ) -> str:
        """
        Determine context type from density metrics.
        
        Args:
            building_density: Buildings per km²
            road_density: Road length per km²
        
        Returns:
            Context type: 'dense_urban', 'urban', 'suburban', 'rural'
        """
        if building_density > 100 or road_density > 20:
            return "dense_urban"
        elif building_density > 50 or road_density > 10:
            return "urban"
        elif building_density > 20 or road_density > 5:
            return "suburban"
        else:
            return "rural"


# ============================================================================
# Unified Threshold Configuration
# ============================================================================

@dataclass
class ClassificationThresholds:
    """
    Unified configuration for all classification thresholds.
    
    This class combines all threshold types and provides context-aware
    threshold selection.
    """
    
    ndvi: NDVIThresholds = field(default_factory=NDVIThresholds)
    geometric: GeometricThresholds = field(default_factory=GeometricThresholds)
    height: HeightThresholds = field(default_factory=HeightThresholds)
    intensity: IntensityThresholds = field(default_factory=IntensityThresholds)
    context: ContextThresholds = field(default_factory=ContextThresholds)
    
    def get_adaptive_thresholds(
        self,
        season: Optional[str] = None,
        context_type: str = 'urban',
        terrain_type: str = 'flat'
    ) -> 'ClassificationThresholds':
        """
        Get context-adapted thresholds.
        
        Args:
            season: Season ('summer', 'winter', 'spring', 'autumn')
            context_type: Context ('dense_urban', 'urban', 'suburban', 'rural')
            terrain_type: Terrain ('flat', 'hilly', 'mountainous')
        
        Returns:
            Adapted ClassificationThresholds instance
        """
        adapted = ClassificationThresholds()
        
        # Adapt NDVI thresholds
        urban_context = context_type in ['dense_urban', 'urban']
        adapted.ndvi = self.ndvi.get_context_adjusted(season, urban_context)
        
        # Adapt height thresholds for terrain
        if terrain_type == 'mountainous':
            # More variation in heights on mountainous terrain
            adapted.height.ground_max *= self.context.mountainous_height_variance_factor
        
        # Adapt geometric thresholds for terrain
        if terrain_type == 'flat':
            # Expect higher planarity in flat regions
            adapted.geometric.planarity_ground_min *= self.context.flat_planarity_boost
            adapted.geometric.planarity_road_min *= self.context.flat_planarity_boost
        
        # Adapt building size thresholds by context
        if context_type == 'rural':
            adapted.context.urban_building_size_min = self.context.rural_building_size_min
        else:
            adapted.context.urban_building_size_min = self.context.urban_building_size_min
        
        logger.info(f"Adapted thresholds for context: season={season}, "
                   f"context={context_type}, terrain={terrain_type}")
        
        return adapted
    
    def validate_thresholds(self) -> Tuple[bool, List[str]]:
        """
        Validate that all thresholds are within reasonable ranges.
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Validate NDVI ranges
        if not 0 <= self.ndvi.vegetation_min <= 1:
            warnings.append(f"NDVI vegetation_min out of range: {self.ndvi.vegetation_min}")
        
        if self.ndvi.vegetation_min >= self.ndvi.vegetation_healthy:
            warnings.append("NDVI vegetation_min >= vegetation_healthy")
        
        # Validate geometric ranges
        if not 0 <= self.geometric.planarity_ground_min <= 1:
            warnings.append(f"Planarity ground_min out of range: {self.geometric.planarity_ground_min}")
        
        # Validate height ranges
        if self.height.building_min <= 0:
            warnings.append(f"Building min height invalid: {self.height.building_min}")
        
        if self.height.low_vegetation_max >= self.height.high_vegetation_min:
            warnings.append("Low vegetation max >= high vegetation min")
        
        # Validate intensity ranges
        if not 0 <= self.intensity.water_max <= 1:
            warnings.append(f"Intensity water_max out of range: {self.intensity.water_max}")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """Export thresholds to dictionary format."""
        return {
            'ndvi': {
                'vegetation_min': self.ndvi.vegetation_min,
                'vegetation_healthy': self.ndvi.vegetation_healthy,
                'building_max': self.ndvi.building_max,
                'trees_min': self.ndvi.trees_min,
            },
            'geometric': {
                'planarity_ground_min': self.geometric.planarity_ground_min,
                'planarity_road_min': self.geometric.planarity_road_min,
                'planarity_roof_flat_min': self.geometric.planarity_roof_flat_min,
                'verticality_wall_min': self.geometric.verticality_wall_min,
            },
            'height': {
                'ground_max': self.height.ground_max,
                'low_vegetation_max': self.height.low_vegetation_max,
                'high_vegetation_min': self.height.high_vegetation_min,
                'building_min': self.height.building_min,
            },
            'intensity': {
                'water_max': self.intensity.water_max,
                'asphalt_range': self.intensity.asphalt_range,
                'metal_min': self.intensity.metal_min,
            }
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ClassificationThresholds':
        """Load thresholds from dictionary format."""
        thresholds = cls()
        
        # Load NDVI
        if 'ndvi' in config:
            ndvi_cfg = config['ndvi']
            thresholds.ndvi.vegetation_min = ndvi_cfg.get('vegetation_min', thresholds.ndvi.vegetation_min)
            thresholds.ndvi.vegetation_healthy = ndvi_cfg.get('vegetation_healthy', thresholds.ndvi.vegetation_healthy)
            thresholds.ndvi.building_max = ndvi_cfg.get('building_max', thresholds.ndvi.building_max)
            thresholds.ndvi.trees_min = ndvi_cfg.get('trees_min', thresholds.ndvi.trees_min)
        
        # Load geometric
        if 'geometric' in config:
            geom_cfg = config['geometric']
            thresholds.geometric.planarity_ground_min = geom_cfg.get('planarity_ground_min', thresholds.geometric.planarity_ground_min)
            thresholds.geometric.planarity_road_min = geom_cfg.get('planarity_road_min', thresholds.geometric.planarity_road_min)
        
        # Load height
        if 'height' in config:
            height_cfg = config['height']
            thresholds.height.ground_max = height_cfg.get('ground_max', thresholds.height.ground_max)
            thresholds.height.building_min = height_cfg.get('building_min', thresholds.height.building_min)
        
        return thresholds


# ============================================================================
# Decision Rules
# ============================================================================

class ClassificationRules:
    """
    Decision rules for classification using multiple features.
    
    These rules encode expert knowledge about how different features
    combine to identify specific classes.
    """
    
    def __init__(self, thresholds: Optional[ClassificationThresholds] = None):
        """Initialize with threshold configuration."""
        self.thresholds = thresholds or ClassificationThresholds()
    
    def is_ground(
        self,
        height: float,
        planarity: float,
        horizontality: float
    ) -> Tuple[bool, float]:
        """
        Determine if point is ground.
        
        Returns:
            Tuple of (is_ground, confidence_score)
        """
        # Must be low and very flat
        is_low = height < self.thresholds.height.ground_max
        is_flat = planarity > self.thresholds.geometric.planarity_ground_min
        is_horizontal = horizontality > self.thresholds.geometric.horizontality_ground_min
        
        if is_low and is_flat and is_horizontal:
            # High confidence
            confidence = min(planarity, horizontality) * 0.95
            return True, confidence
        elif is_low and is_flat:
            # Medium confidence
            confidence = planarity * 0.75
            return True, confidence
        else:
            return False, 0.0
    
    def is_vegetation(
        self,
        ndvi: float,
        height: float,
        curvature: Optional[float] = None,
        planarity: Optional[float] = None
    ) -> Tuple[bool, str, float]:
        """
        Determine if point is vegetation and what type.
        
        Returns:
            Tuple of (is_vegetation, vegetation_type, confidence_score)
        """
        # Primary check: NDVI
        has_high_ndvi = ndvi > self.thresholds.ndvi.vegetation_min
        
        if not has_high_ndvi:
            return False, "none", 0.0
        
        # Check geometric features for confirmation
        geometric_support = 0.0
        if curvature is not None:
            if curvature > self.thresholds.geometric.curvature_vegetation_min:
                geometric_support += 0.5
        if planarity is not None:
            if planarity < self.thresholds.geometric.planarity_vegetation_max:
                geometric_support += 0.5
        
        # Determine vegetation type by height
        if height < self.thresholds.height.grass_max:
            veg_type = "grass"
            confidence = (ndvi / 0.6) * 0.8 + geometric_support * 0.2
        elif height < self.thresholds.height.low_vegetation_max:
            veg_type = "low_vegetation"
            confidence = (ndvi / 0.7) * 0.7 + geometric_support * 0.3
        elif height < self.thresholds.height.forest_canopy_min:
            veg_type = "trees"
            confidence = (ndvi / 0.8) * 0.6 + geometric_support * 0.4
        else:
            veg_type = "tall_trees"
            confidence = (ndvi / 0.8) * 0.5 + geometric_support * 0.5
        
        return True, veg_type, min(confidence, 1.0)
    
    def is_building(
        self,
        height: float,
        planarity: float,
        ndvi: float,
        verticality: Optional[float] = None,
        intensity: Optional[float] = None
    ) -> Tuple[bool, str, float]:
        """
        Determine if point is building and what component.
        
        Returns:
            Tuple of (is_building, component_type, confidence_score)
        """
        # Must be elevated and not vegetation
        is_elevated = height > self.thresholds.height.building_min
        not_vegetation = ndvi < self.thresholds.ndvi.building_max
        is_planar = planarity > self.thresholds.geometric.planarity_wall_min
        
        if not (is_elevated and not_vegetation):
            return False, "none", 0.0
        
        # Determine component type
        confidence = 0.0
        
        if verticality is not None and verticality > self.thresholds.geometric.verticality_wall_min:
            # Likely a wall
            component = "wall"
            wall_score = self.thresholds.geometric.compute_wall_score(planarity, verticality)
            confidence = wall_score
        elif planarity > self.thresholds.geometric.planarity_roof_flat_min:
            # Likely a roof
            component = "roof"
            confidence = planarity * 0.9
        else:
            # General building
            component = "building"
            confidence = 0.7
        
        return True, component, min(confidence, 1.0)
    
    def is_road(
        self,
        height: float,
        planarity: float,
        horizontality: float,
        intensity: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Determine if point is road.
        
        Returns:
            Tuple of (is_road, confidence_score)
        """
        # Must be very low, very flat, very horizontal
        is_low = height < self.thresholds.height.road_max
        is_very_flat = planarity > self.thresholds.geometric.planarity_road_min
        is_horizontal = horizontality > self.thresholds.geometric.horizontality_road_min
        
        if not (is_low and is_very_flat and is_horizontal):
            return False, 0.0
        
        # Check intensity for asphalt
        confidence = min(planarity, horizontality)
        if intensity is not None:
            asph_min, asph_max = self.thresholds.intensity.asphalt_range
            if asph_min <= intensity <= asph_max:
                confidence *= 1.1  # Boost for asphalt-like intensity
        
        return True, min(confidence, 1.0)


# ============================================================================
# Threshold Optimization
# ============================================================================

def optimize_thresholds_from_data(
    labels: np.ndarray,
    features: Dict[str, np.ndarray],
    reference_labels: np.ndarray,
    current_thresholds: Optional[ClassificationThresholds] = None
) -> ClassificationThresholds:
    """
    Optimize thresholds based on labeled reference data.
    
    This function analyzes the feature distributions for each class
    in the reference data and suggests optimized thresholds.
    
    Args:
        labels: Current classification labels [N]
        features: Dictionary of features (ndvi, height, planarity, etc.)
        reference_labels: Ground truth labels [N]
        current_thresholds: Current thresholds to optimize (optional)
    
    Returns:
        Optimized ClassificationThresholds
    """
    if current_thresholds is None:
        thresholds = ClassificationThresholds()
    else:
        thresholds = current_thresholds
    
    # Analyze NDVI for vegetation vs non-vegetation
    if 'ndvi' in features:
        ndvi = features['ndvi']
        
        # Find optimal NDVI threshold
        # (This is simplified - in reality, use ROC analysis)
        veg_mask = np.isin(reference_labels, [3, 4, 5])  # ASPRS vegetation classes
        if np.any(veg_mask) and np.any(~veg_mask):
            veg_ndvi = ndvi[veg_mask]
            non_veg_ndvi = ndvi[~veg_mask]
            
            # Use percentiles for robustness
            veg_p10 = np.percentile(veg_ndvi, 10)
            non_veg_p90 = np.percentile(non_veg_ndvi, 90)
            
            # Optimal threshold between these
            optimal_threshold = (veg_p10 + non_veg_p90) / 2
            thresholds.ndvi.vegetation_min = optimal_threshold
            
            logger.info(f"Optimized NDVI vegetation threshold: {optimal_threshold:.3f}")
    
    # Analyze height thresholds
    if 'height' in features:
        height = features['height']
        
        # Optimize ground threshold
        ground_mask = reference_labels == 2  # ASPRS ground
        if np.any(ground_mask):
            ground_heights = height[ground_mask]
            ground_p95 = np.percentile(ground_heights, 95)
            thresholds.height.ground_max = ground_p95
            
            logger.info(f"Optimized ground height threshold: {ground_p95:.2f}m")
    
    return thresholds
