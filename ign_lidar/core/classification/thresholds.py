"""Classification Thresholds

This module consolidates all classification thresholds used across the
IGN LiDAR HD classification system into a single, consistent configuration.

Consolidates and replaces:
- classification_thresholds.py (ClassificationThresholds)
- optimized_thresholds.py (NDVIThresholds, GeometricThresholds)

Author: IGN LiDAR HD Development Team
Date: October 22, 2025
Version: 3.1.0 - Consolidated thresholds
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# NDVI Thresholds
# ============================================================================


@dataclass
class NDVIThresholds:
    """
    NDVI (Normalized Difference Vegetation Index) thresholds.

    Optimized for French topographic context (IGN data).
    Values calibrated from empirical analysis.
    """

    # Primary vegetation detection
    vegetation_min: float = 0.35  # Minimum NDVI for vegetation
    vegetation_healthy: float = 0.5  # Healthy, dense vegetation
    vegetation_stressed: float = 0.25  # Stressed/sparse vegetation

    # Vegetation types
    grass_min: float = 0.3  # Grass/lawn minimum
    grass_max: float = 0.6  # Grass/lawn maximum
    shrubs_min: float = 0.4  # Shrubs/bushes minimum
    shrubs_max: float = 0.7  # Shrubs/bushes maximum
    trees_min: float = 0.5  # Dense tree canopy
    forest_min: float = 0.6  # Dense forest

    # Non-vegetation
    building_max: float = 0.15  # Maximum NDVI for buildings
    road_max: float = 0.2  # Maximum NDVI for paved roads
    bare_soil_max: float = 0.25  # Bare soil/ground
    water_max: float = 0.1  # Water bodies (very low NDVI)

    # High vegetation specific (for advanced classification)
    high_veg_threshold: float = 0.45  # High vegetation minimum

    # Seasonal adjustment factors
    summer_boost: float = 1.1  # NDVI higher in summer
    winter_reduction: float = 0.85  # NDVI lower in winter (deciduous)

    def get_context_adjusted(
        self, season: Optional[str] = None, urban_context: bool = False
    ) -> "NDVIThresholds":
        """
        Get seasonally and contextually adjusted thresholds.

        Args:
            season: Season name ('summer', 'winter', 'spring', 'autumn')
            urban_context: Whether in urban environment

        Returns:
            New NDVIThresholds instance with adjusted values
        """
        # Calculate adjustment factor
        if season == "summer":
            factor = self.summer_boost
        elif season == "winter":
            factor = self.winter_reduction
        else:
            factor = 1.0

        # Urban adjustment (lower NDVI due to stress)
        if urban_context:
            factor *= 0.9

        # Create adjusted thresholds
        adjusted = NDVIThresholds(
            vegetation_min=self.vegetation_min * factor,
            vegetation_healthy=self.vegetation_healthy * factor,
            vegetation_stressed=self.vegetation_stressed * factor,
            grass_min=self.grass_min * factor,
            grass_max=self.grass_max * factor,
            shrubs_min=self.shrubs_min * factor,
            shrubs_max=self.shrubs_max * factor,
            trees_min=self.trees_min * factor,
            forest_min=self.forest_min * factor,
            # Non-vegetation stays the same
            building_max=self.building_max,
            road_max=self.road_max,
            bare_soil_max=self.bare_soil_max,
            water_max=self.water_max,
            high_veg_threshold=self.high_veg_threshold * factor,
            summer_boost=self.summer_boost,
            winter_reduction=self.winter_reduction,
        )

        return adjusted


# ============================================================================
# Geometric Feature Thresholds
# ============================================================================


@dataclass
class GeometricThresholds:
    """
    Geometric feature thresholds for surface classification.

    Thresholds for planarity, verticality/horizontality, curvature,
    roughness, anisotropy, and linearity.
    """

    # Planarity (flatness) - higher = flatter
    planarity_ground_min: float = 0.85  # Ground surfaces (very flat)
    planarity_road_min: float = 0.88  # Roads (extremely flat)
    planarity_road_min_strict: float = 0.8  # Roads strict mode
    planarity_parking_min: float = 0.82  # Parking lots
    planarity_roof_flat_min: float = 0.75  # Flat roofs
    planarity_roof_gable_min: float = 0.70  # Gabled roofs
    planarity_wall_min: float = 0.60  # Building walls
    planarity_facade_min: float = 0.55  # Facades (with details)
    planarity_vegetation_max: float = 0.35  # Vegetation (irregular)
    planarity_building_min_asprs: float = 0.5  # Building min (ASPRS mode)
    planarity_building_min_lod2: float = 0.55  # Building min (LOD2 mode)
    planarity_building_min_lod3: float = 0.60  # Building min (LOD3 mode)

    # Horizontality (upward-facing) - higher = more horizontal
    horizontality_ground_min: float = 0.90  # Ground
    horizontality_roof_min: float = 0.85  # Flat roofs
    horizontality_roof_min_asprs: float = 0.80  # Roof (ASPRS mode)
    horizontality_roof_min_lod2: float = 0.85  # Roof (LOD2 mode)
    horizontality_roof_min_lod3: float = 0.85  # Roof (LOD3 mode)
    horizontality_road_min: float = 0.92  # Roads (very horizontal)

    # Verticality (vertical surfaces) - higher = more vertical
    verticality_wall_min: float = 0.70  # Walls
    verticality_wall_min_asprs: float = 0.65  # Wall (ASPRS mode)
    verticality_wall_min_lod2: float = 0.70  # Wall (LOD2 mode)
    verticality_wall_min_lod3: float = 0.75  # Wall (LOD3 mode)
    verticality_facade_min: float = 0.70  # Facades (improved from 0.65)
    verticality_tree_trunk_min: float = 0.60  # Tree trunks
    verticality_road_max: float = 0.30  # Roads (exclude vertical)

    # Curvature (surface bending) - higher = more curved
    curvature_flat_max: float = 0.01  # Flat surfaces
    curvature_vegetation_min: float = 0.03  # Vegetation
    curvature_vegetation_typical: float = 0.08  # Dense vegetation
    curvature_chimney_min: float = 0.05  # Cylindrical structures
    curvature_dormer_min: float = 0.02  # Curved roof elements min
    curvature_dormer_max: float = 0.15  # Curved roof elements max
    curvature_road_max: float = 0.05  # Roads (exclude complex)

    # Roughness (surface irregularity) - higher = rougher
    roughness_smooth_max: float = 0.02  # Very smooth (glass, metal)
    roughness_road_max: float = 0.05  # Paved roads
    roughness_concrete_max: float = 0.08  # Concrete surfaces
    roughness_roof_tile_min: float = 0.05  # Tiled roofs min
    roughness_roof_tile_max: float = 0.15  # Tiled roofs max
    roughness_vegetation_min: float = 0.08  # Vegetation surfaces
    roughness_vegetation_dense: float = 0.15  # Dense vegetation
    roughness_ground_max: float = 0.05  # Ground

    # Anisotropy (directionality) - higher = more organized/linear
    anisotropy_building_min: float = 0.50  # Organized structures
    anisotropy_road_min: float = 0.55  # Linear road structures
    anisotropy_vegetation_max: float = 0.40  # Random vegetation

    # Linearity (1D linear structures) - higher = more linear
    linearity_edge_min: float = 0.45  # Building edges, roof ridges
    linearity_road_edge_min: float = 0.50  # Road edges
    linearity_tree_trunk_min: float = 0.35  # Tree trunks
    linearity_building_struct_min: float = 0.3  # Structural elements


# ============================================================================
# Height Thresholds
# ============================================================================


@dataclass
class HeightThresholds:
    """
    Height-based classification thresholds (meters above ground).

    Heights computed as Z - DTM elevation.
    """

    # Vegetation height ranges
    low_veg_height_max: float = 2.0  # Maximum for low vegetation
    medium_veg_height_min: float = 0.5  # Minimum for medium vegetation
    high_veg_height_min: float = 1.5  # Minimum for high vegetation

    # Building heights
    building_height_min: float = 2.5  # Minimum building height
    building_height_max: float = 200.0  # Maximum reasonable height

    # Ground/terrain
    ground_height_max: float = 0.5  # Maximum height above ground

    # Vehicles
    vehicle_height_min: float = 1.0  # Minimum vehicle height
    vehicle_height_max: float = 5.0  # Maximum vehicle height

    # Transport (roads, railways)
    road_height_min: float = -0.5  # Roads (depression tolerance)
    road_height_max: float = 0.3  # Roads (ground-level only)
    road_height_max_strict: float = 0.3  # Roads strict mode

    rail_height_min: float = -0.5  # Railways (depression tolerance)
    rail_height_max: float = 2.0  # Railways (elevated tracks)
    rail_height_max_strict: float = 0.8  # Railways strict mode

    # Water
    water_height_max: float = 0.2  # Maximum height variation

    # Bridge
    bridge_height_min: float = 2.0  # Minimum height above ground


# ============================================================================
# Transport-Specific Thresholds
# ============================================================================


@dataclass
class TransportThresholds:
    """
    Thresholds specific to transport infrastructure (roads, railways).
    """

    # Road geometric thresholds
    road_planarity_min: float = 0.7  # Standard mode
    road_planarity_min_strict: float = 0.8  # Urban/strict mode
    road_roughness_max: float = 0.05  # Paved roads
    road_buffer_tolerance: float = 0.5  # Buffer beyond BD TOPO (m)

    # Road exclusion criteria
    road_ndvi_max: float = 0.20  # Exclude vegetation
    road_curvature_max: float = 0.05  # Exclude complex surfaces
    road_verticality_max: float = 0.30  # Exclude walls/poles

    # Road intensity (normalized 0-1)
    road_intensity_min: float = 0.15  # Dark asphalt
    road_intensity_max: float = 0.7  # Concrete

    # Railway geometric thresholds
    rail_planarity_min: float = 0.5  # Standard (ballast)
    rail_planarity_min_strict: float = 0.75  # Strict mode
    rail_roughness_max: float = 0.08  # Higher (ballast)
    rail_buffer_multiplier: float = 1.2  # Wider for ballast

    # Railway intensity (normalized 0-1)
    rail_intensity_min: float = 0.1  # Dark ballast
    rail_intensity_max: float = 0.8  # Rails + ballast mix


# ============================================================================
# Building-Specific Thresholds
# ============================================================================


@dataclass
class BuildingThresholds:
    """
    Thresholds specific to building detection and classification.

    Different values for different detection modes (ASPRS, LOD2, LOD3).
    """

    # Height thresholds (same for all modes)
    height_min: float = 2.5
    height_max: float = 200.0

    # ASPRS mode (lenient)
    wall_verticality_min_asprs: float = 0.65
    wall_planarity_min_asprs: float = 0.5
    roof_horizontality_min_asprs: float = 0.80
    roof_planarity_min_asprs: float = 0.65

    # LOD2 mode (stricter)
    wall_verticality_min_lod2: float = 0.70
    wall_planarity_min_lod2: float = 0.55
    roof_horizontality_min_lod2: float = 0.85
    roof_planarity_min_lod2: float = 0.70

    # LOD3 mode (strictest)
    wall_verticality_min_lod3: float = 0.75
    wall_planarity_min_lod3: float = 0.60
    roof_horizontality_min_lod3: float = 0.85
    roof_planarity_min_lod3: float = 0.75

    # Additional building features
    wall_score_min: float = 0.35  # planarity x verticality
    roof_score_min: float = 0.5  # planarity x horizontality
    anisotropy_building_min: float = 0.5  # Organized structures
    linearity_edge_min: float = 0.4  # Building edges

    def get_for_mode(self, mode: str) -> Dict[str, float]:
        """
        Get thresholds for specific detection mode.

        Args:
            mode: 'asprs', 'lod2', or 'lod3'

        Returns:
            Dictionary of threshold values
        """
        mode = mode.lower()

        if mode == "asprs":
            return {
                "height_min": self.height_min,
                "height_max": self.height_max,
                "wall_verticality_min": self.wall_verticality_min_asprs,
                "wall_planarity_min": self.wall_planarity_min_asprs,
                "roof_horizontality_min": self.roof_horizontality_min_asprs,
                "roof_planarity_min": self.roof_planarity_min_asprs,
            }
        elif mode == "lod2":
            return {
                "height_min": self.height_min,
                "height_max": self.height_max,
                "wall_verticality_min": self.wall_verticality_min_lod2,
                "wall_planarity_min": self.wall_planarity_min_lod2,
                "roof_horizontality_min": self.roof_horizontality_min_lod2,
                "roof_planarity_min": self.roof_planarity_min_lod2,
            }
        elif mode == "lod3":
            return {
                "height_min": self.height_min,
                "height_max": self.height_max,
                "wall_verticality_min": self.wall_verticality_min_lod3,
                "wall_planarity_min": self.wall_planarity_min_lod3,
                "roof_horizontality_min": self.roof_horizontality_min_lod3,
                "roof_planarity_min": self.roof_planarity_min_lod3,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'asprs', 'lod2', or 'lod3'.")


# ============================================================================
# Additional Feature Thresholds
# ============================================================================


@dataclass
class WaterThresholds:
    """Thresholds for water body classification."""

    height_max: float = 0.2  # Maximum height variation
    planarity_min: float = 0.95  # Minimum planarity
    intensity_max: float = 0.1  # Maximum intensity


@dataclass
class BridgeThresholds:
    """Thresholds for bridge detection."""

    height_min: float = 2.0  # Minimum height above ground
    planarity_min: float = 0.7  # Minimum planarity
    width_min: float = 3.0  # Minimum width


@dataclass
class VehicleThresholds:
    """Thresholds for vehicle detection."""

    height_min: float = 1.0  # Minimum height
    height_max: float = 5.0  # Maximum height
    density_min: float = 0.7  # Minimum point density


# ============================================================================
# Threshold Configuration
# ============================================================================


@dataclass
class ThresholdConfig:
    """
    Complete threshold configuration for classification.

    Consolidates all threshold types into a single configuration object.
    Can be customized for different modes and contexts.
    """

    # Threshold groups
    ndvi: NDVIThresholds = field(default_factory=NDVIThresholds)
    geometric: GeometricThresholds = field(default_factory=GeometricThresholds)
    height: HeightThresholds = field(default_factory=HeightThresholds)
    transport: TransportThresholds = field(default_factory=TransportThresholds)
    building: BuildingThresholds = field(default_factory=BuildingThresholds)
    water: WaterThresholds = field(default_factory=WaterThresholds)
    bridge: BridgeThresholds = field(default_factory=BridgeThresholds)
    vehicle: VehicleThresholds = field(default_factory=VehicleThresholds)

    # Context settings
    mode: str = "asprs"  # 'asprs', 'lod2', 'lod3'
    strict: bool = False  # Use strict thresholds
    season: Optional[str] = None  # 'summer', 'winter', etc.
    urban_context: bool = False  # Urban vs rural

    @classmethod
    def for_mode(
        cls,
        mode: str = "asprs",
        strict: bool = False,
        season: Optional[str] = None,
        urban_context: bool = False,
    ) -> "ThresholdConfig":
        """
        Create threshold configuration for specific mode and context.

        Args:
            mode: Classification mode ('asprs', 'lod2', 'lod3')
            strict: Use stricter thresholds
            season: Season for NDVI adjustment
            urban_context: Urban vs rural context

        Returns:
            Configured ThresholdConfig instance
        """
        config = cls(
            mode=mode, strict=strict, season=season, urban_context=urban_context
        )

        # Adjust NDVI thresholds for context
        if season or urban_context:
            config.ndvi = config.ndvi.get_context_adjusted(season, urban_context)

        return config

    def get_all(self) -> Dict[str, Any]:
        """
        Get all thresholds as nested dictionary.

        Returns:
            Complete threshold configuration
        """
        return {
            "ndvi": self.ndvi.__dict__,
            "geometric": self.geometric.__dict__,
            "height": self.height.__dict__,
            "transport": self.transport.__dict__,
            "building": self.building.get_for_mode(self.mode),
            "water": self.water.__dict__,
            "bridge": self.bridge.__dict__,
            "vehicle": self.vehicle.__dict__,
            "mode": self.mode,
            "strict": self.strict,
            "season": self.season,
            "urban_context": self.urban_context,
        }

    def validate(self) -> Dict[str, str]:
        """
        Validate threshold consistency.

        Returns:
            Dictionary of validation warnings (empty if OK)
        """
        warnings = {}

        # Check vegetation height ranges
        if self.height.low_veg_height_max < self.height.high_veg_height_min:
            warnings["vegetation_height_gap"] = (
                f"Gap between low veg max ({self.height.low_veg_height_max}m) "
                f"and high veg min ({self.height.high_veg_height_min}m)"
            )

        # Check building height vs vegetation
        if self.building.height_min < self.height.high_veg_height_min:
            warnings["building_veg_overlap"] = (
                f"Building min ({self.building.height_min}m) overlaps with "
                f"high vegetation ({self.height.high_veg_height_min}m)"
            )

        # Check NDVI consistency
        if self.ndvi.vegetation_min < self.ndvi.building_max:
            warnings["ndvi_overlap"] = (
                f"Vegetation min NDVI ({self.ndvi.vegetation_min}) too close to "
                f"building max ({self.ndvi.building_max})"
            )

        return warnings


# ============================================================================
# Convenience Functions
# ============================================================================


def get_thresholds(
    mode: str = "asprs",
    strict: bool = False,
    season: Optional[str] = None,
    urban_context: bool = False,
) -> ThresholdConfig:
    """
    Get threshold configuration for specific mode and context.

    Args:
        mode: Classification mode ('asprs', 'lod2', 'lod3')
        strict: Use stricter thresholds
        season: Season for NDVI adjustment
        urban_context: Urban vs rural context

    Returns:
        ThresholdConfig instance

    Example:
        >>> thresholds = get_thresholds(mode='lod2', strict=True)
        >>> road_planarity = thresholds.transport.road_planarity_min_strict
    """
    return ThresholdConfig.for_mode(mode, strict, season, urban_context)


def get_default_thresholds() -> ThresholdConfig:
    """
    Get default threshold configuration.

    Returns:
        Default ThresholdConfig (ASPRS mode, non-strict)
    """
    return ThresholdConfig()


def print_threshold_summary(config: Optional[ThresholdConfig] = None):
    """
    Print summary of threshold configuration.

    Args:
        config: ThresholdConfig to print (default if None)
    """
    if config is None:
        config = get_default_thresholds()

    print("=" * 70)
    print(
        f"THRESHOLD CONFIGURATION - Mode: {config.mode.upper()}, Strict: {config.strict}"
    )
    print("=" * 70)

    all_thresholds = config.get_all()

    for category, values in all_thresholds.items():
        if isinstance(values, dict):
            print(f"\n{category.upper()}")
            print("-" * 70)
            for key, value in values.items():
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    warnings = config.validate()
    if warnings:
        for key, msg in warnings.items():
            print(f"  ⚠️  {msg}")
    else:
        print("  ✅ All thresholds are consistent")

    print("=" * 70)


# ============================================================================
# Backward Compatibility
# ============================================================================


class ClassificationThresholds:
    """
    Backward compatibility wrapper for legacy code.

    DEPRECATED: Use ThresholdConfig instead.
    This class provides static access to all thresholds from the old
    classification_thresholds.py module.
    """

    # Create default config
    _config = get_default_thresholds()

    # ===================================================================
    # TRANSPORT THRESHOLDS (Roads & Railways)
    # ===================================================================

    # Road height thresholds (meters)
    ROAD_HEIGHT_MAX = _config.height.road_height_max  # 0.3
    ROAD_HEIGHT_MIN = _config.height.road_height_min  # -0.5
    ROAD_HEIGHT_MAX_STRICT = _config.height.road_height_max_strict  # 0.3

    # Railway height thresholds (meters)
    RAIL_HEIGHT_MAX = _config.height.rail_height_max  # 2.0
    RAIL_HEIGHT_MIN = _config.height.rail_height_min  # -0.5
    RAIL_HEIGHT_MAX_STRICT = _config.height.rail_height_max_strict  # 0.8

    # Road geometric thresholds
    ROAD_PLANARITY_MIN = _config.transport.road_planarity_min  # 0.7
    ROAD_PLANARITY_MIN_STRICT = _config.transport.road_planarity_min_strict  # 0.8
    ROAD_ROUGHNESS_MAX = _config.transport.road_roughness_max  # 0.05
    ROAD_BUFFER_TOLERANCE = _config.transport.road_buffer_tolerance  # 0.5

    # Road vegetation/building exclusion thresholds
    ROAD_NDVI_MAX = _config.transport.road_ndvi_max  # 0.20
    ROAD_CURVATURE_MAX = _config.transport.road_curvature_max  # 0.05
    ROAD_VERTICALITY_MAX = _config.transport.road_verticality_max  # 0.30

    # Railway geometric thresholds
    RAIL_PLANARITY_MIN = _config.transport.rail_planarity_min  # 0.5
    RAIL_PLANARITY_MIN_STRICT = _config.transport.rail_planarity_min_strict  # 0.75
    RAIL_ROUGHNESS_MAX = _config.transport.rail_roughness_max  # 0.08
    RAIL_BUFFER_MULTIPLIER = _config.transport.rail_buffer_multiplier  # 1.2

    # Road intensity thresholds (normalized 0-1)
    ROAD_INTENSITY_MIN = _config.transport.road_intensity_min  # 0.15
    ROAD_INTENSITY_MAX = _config.transport.road_intensity_max  # 0.7

    # Railway intensity thresholds (normalized 0-1)
    RAIL_INTENSITY_MIN = _config.transport.rail_intensity_min  # 0.1
    RAIL_INTENSITY_MAX = _config.transport.rail_intensity_max  # 0.8

    # ===================================================================
    # BUILDING THRESHOLDS
    # ===================================================================

    # Building height thresholds (meters)
    BUILDING_HEIGHT_MIN = _config.building.height_min  # 2.5
    BUILDING_HEIGHT_MAX = _config.building.height_max  # 200.0

    # Building geometric thresholds - ASPRS mode
    BUILDING_WALL_VERTICALITY_MIN_ASPRS = (
        _config.building.wall_verticality_min_asprs
    )  # 0.65
    BUILDING_WALL_PLANARITY_MIN_ASPRS = _config.building.wall_planarity_min_asprs  # 0.5
    BUILDING_ROOF_HORIZONTALITY_MIN_ASPRS = (
        _config.building.roof_horizontality_min_asprs
    )  # 0.80
    BUILDING_ROOF_PLANARITY_MIN_ASPRS = (
        _config.building.roof_planarity_min_asprs
    )  # 0.65

    # Building geometric thresholds - LOD2 mode (stricter)
    BUILDING_WALL_VERTICALITY_MIN_LOD2 = (
        _config.building.wall_verticality_min_lod2
    )  # 0.70
    BUILDING_WALL_PLANARITY_MIN_LOD2 = _config.building.wall_planarity_min_lod2  # 0.55
    BUILDING_ROOF_HORIZONTALITY_MIN_LOD2 = (
        _config.building.roof_horizontality_min_lod2
    )  # 0.85
    BUILDING_ROOF_PLANARITY_MIN_LOD2 = _config.building.roof_planarity_min_lod2  # 0.70

    # Building geometric thresholds - LOD3 mode (strictest)
    BUILDING_WALL_VERTICALITY_MIN_LOD3 = (
        _config.building.wall_verticality_min_lod3
    )  # 0.75
    BUILDING_WALL_PLANARITY_MIN_LOD3 = _config.building.wall_planarity_min_lod3  # 0.60
    BUILDING_ROOF_HORIZONTALITY_MIN_LOD3 = (
        _config.building.roof_horizontality_min_lod3
    )  # 0.85
    BUILDING_ROOF_PLANARITY_MIN_LOD3 = _config.building.roof_planarity_min_lod3  # 0.75

    # ===================================================================
    # VEGETATION THRESHOLDS
    # ===================================================================

    # Height-based vegetation classification (meters)
    LOW_VEG_HEIGHT_MAX = _config.height.low_veg_height_max  # 2.0
    HIGH_VEG_HEIGHT_MIN = _config.height.high_veg_height_min  # 1.5
    MEDIUM_VEG_HEIGHT_MIN = _config.height.medium_veg_height_min  # 0.5

    # NDVI thresholds (normalized -1 to 1)
    NDVI_VEG_THRESHOLD = _config.ndvi.vegetation_min  # 0.35
    NDVI_HIGH_VEG_THRESHOLD = _config.ndvi.high_veg_threshold  # 0.45
    NDVI_BUILDING_THRESHOLD = _config.ndvi.building_max  # 0.15

    # Vegetation geometric thresholds
    VEG_PLANARITY_MAX = _config.geometric.planarity_vegetation_max  # 0.35
    VEG_ROUGHNESS_MIN = _config.geometric.roughness_vegetation_min  # 0.2

    # ===================================================================
    # GROUND THRESHOLDS
    # ===================================================================

    GROUND_HEIGHT_MAX = _config.height.ground_height_max  # 0.5
    GROUND_PLANARITY_MIN = _config.geometric.planarity_ground_min  # 0.85
    GROUND_ROUGHNESS_MAX = _config.geometric.roughness_ground_max  # 0.05
    GROUND_HORIZONTALITY_MIN = _config.geometric.horizontality_ground_min  # 0.9

    # ===================================================================
    # VEHICLE THRESHOLDS
    # ===================================================================

    VEHICLE_HEIGHT_MIN = _config.vehicle.height_min  # 1.0
    VEHICLE_HEIGHT_MAX = _config.vehicle.height_max  # 5.0
    VEHICLE_DENSITY_MIN = _config.vehicle.density_min  # 0.7

    # ===================================================================
    # WATER THRESHOLDS
    # ===================================================================

    WATER_HEIGHT_MAX = _config.water.height_max  # 0.2
    WATER_PLANARITY_MIN = _config.water.planarity_min  # 0.95
    WATER_INTENSITY_MAX = _config.water.intensity_max  # 0.1

    # ===================================================================
    # BRIDGE THRESHOLDS
    # ===================================================================

    BRIDGE_HEIGHT_MIN = _config.bridge.height_min  # 2.0
    BRIDGE_PLANARITY_MIN = _config.bridge.planarity_min  # 0.7
    BRIDGE_WIDTH_MIN = _config.bridge.width_min  # 3.0

    @classmethod
    def get_building_thresholds(cls, mode: str = "asprs"):
        """Backward compatibility method."""
        config = get_thresholds(mode=mode)
        return {
            "height_min": config.building.height_min,
            "height_max": config.building.height_max,
            "wall_verticality_min": getattr(
                config.building, f"wall_verticality_min_{mode}"
            ),
            "wall_planarity_min": getattr(
                config.building, f"wall_planarity_min_{mode}"
            ),
            "roof_horizontality_min": getattr(
                config.building, f"roof_horizontality_min_{mode}"
            ),
            "roof_planarity_min": getattr(
                config.building, f"roof_planarity_min_{mode}"
            ),
        }

    @classmethod
    def get_transport_thresholds(cls, strict_mode: bool = False):
        """Backward compatibility method."""
        config = get_thresholds(strict=strict_mode)
        return {
            "road_height_max": (
                config.height.road_height_max_strict
                if strict_mode
                else config.height.road_height_max
            ),
            "road_height_min": config.height.road_height_min,
            "road_planarity_min": (
                config.transport.road_planarity_min_strict
                if strict_mode
                else config.transport.road_planarity_min
            ),
            "road_roughness_max": config.transport.road_roughness_max,
            "road_intensity_min": config.transport.road_intensity_min,
            "road_intensity_max": config.transport.road_intensity_max,
            "rail_height_max": (
                config.height.rail_height_max_strict
                if strict_mode
                else config.height.rail_height_max
            ),
            "rail_height_min": config.height.rail_height_min,
            "rail_planarity_min": (
                config.transport.rail_planarity_min_strict
                if strict_mode
                else config.transport.rail_planarity_min
            ),
            "rail_roughness_max": config.transport.rail_roughness_max,
            "rail_intensity_min": config.transport.rail_intensity_min,
            "rail_intensity_max": config.transport.rail_intensity_max,
        }

    @classmethod
    def get_all_thresholds(cls):
        """Backward compatibility method."""
        return get_default_thresholds()

    @classmethod
    def validate_thresholds(cls):
        """Backward compatibility method."""
        # No validation warnings in new system
        return {}


__all__ = [
    # Dataclasses
    "NDVIThresholds",
    "GeometricThresholds",
    "HeightThresholds",
    "TransportThresholds",
    "BuildingThresholds",
    "WaterThresholds",
    "BridgeThresholds",
    "VehicleThresholds",
    "ThresholdConfig",
    # Functions
    "get_thresholds",
    "get_default_thresholds",
    "print_threshold_summary",
    # Backward compatibility
    "ClassificationThresholds",
]


if __name__ == "__main__":
    # Print summary when run directly
    print("\n=== DEFAULT CONFIGURATION ===")
    print_threshold_summary()

    print("\n\n=== LOD2 STRICT MODE ===")
    print_threshold_summary(get_thresholds(mode="lod2", strict=True))

    print("\n\n=== URBAN SUMMER CONTEXT ===")
    print_threshold_summary(get_thresholds(urban_context=True, season="summer"))
