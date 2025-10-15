"""
Unified Classification Thresholds

This module provides a central location for all classification thresholds used across
the IGN LiDAR HD classification system. This ensures consistency and makes it easier
to tune parameters.

Created to address Issue #8: Conflicting Height Thresholds
See: docs/AUDIT_ACTION_PLAN.md

Author: IGN LiDAR HD Development Team
Date: October 16, 2025
"""

from typing import Dict, Any


class UnifiedThresholds:
    """
    Unified thresholds for all classification modules.
    
    This class provides a single source of truth for all geometric, height,
    and intensity thresholds used throughout the classification pipeline.
    
    Attributes organized by feature type:
    - Transport (roads, railways)
    - Buildings
    - Vegetation
    - Geometric features
    """
    
    # ===================================================================
    # TRANSPORT THRESHOLDS (Roads & Railways)
    # ===================================================================
    
    # Road height thresholds (meters)
    ROAD_HEIGHT_MAX = 2.0  # Increased from 1.5m to handle elevated sections
    ROAD_HEIGHT_MIN = -0.5  # Increased tolerance from -0.3m for depressions
    ROAD_HEIGHT_MAX_STRICT = 0.5  # For strict mode if needed
    
    # Railway height thresholds (meters)
    RAIL_HEIGHT_MAX = 2.0  # Increased from 1.2m to handle elevated tracks
    RAIL_HEIGHT_MIN = -0.5  # Increased tolerance from -0.2m
    RAIL_HEIGHT_MAX_STRICT = 0.8  # For strict mode if needed
    
    # Road geometric thresholds
    ROAD_PLANARITY_MIN = 0.6  # Minimum planarity for road surfaces
    ROAD_PLANARITY_MIN_STRICT = 0.8  # Strict mode for urban roads
    ROAD_ROUGHNESS_MAX = 0.05  # Maximum roughness for paved roads
    ROAD_BUFFER_TOLERANCE = 0.5  # Additional buffer beyond BD TOPO width (meters)
    
    # Railway geometric thresholds
    RAIL_PLANARITY_MIN = 0.5  # Lower than roads due to ballast
    RAIL_PLANARITY_MIN_STRICT = 0.75  # Strict mode
    RAIL_ROUGHNESS_MAX = 0.08  # Higher than roads due to ballast
    RAIL_BUFFER_MULTIPLIER = 1.2  # Multiplier for railway buffer (wider for ballast)
    
    # Road intensity thresholds (normalized 0-1)
    ROAD_INTENSITY_MIN = 0.15  # Minimum intensity (dark asphalt)
    ROAD_INTENSITY_MAX = 0.7   # Maximum intensity (concrete)
    
    # Railway intensity thresholds (normalized 0-1)
    RAIL_INTENSITY_MIN = 0.1   # Minimum intensity (dark ballast)
    RAIL_INTENSITY_MAX = 0.8   # Maximum intensity (rails + ballast mix)
    
    # ===================================================================
    # BUILDING THRESHOLDS
    # ===================================================================
    
    # Building height thresholds (meters)
    BUILDING_HEIGHT_MIN = 2.5   # Minimum height for building detection
    BUILDING_HEIGHT_MAX = 200.0  # Maximum reasonable building height
    
    # Building geometric thresholds - ASPRS mode
    BUILDING_WALL_VERTICALITY_MIN_ASPRS = 0.65
    BUILDING_WALL_PLANARITY_MIN_ASPRS = 0.5
    BUILDING_ROOF_HORIZONTALITY_MIN_ASPRS = 0.80
    BUILDING_ROOF_PLANARITY_MIN_ASPRS = 0.65
    
    # Building geometric thresholds - LOD2 mode (stricter)
    BUILDING_WALL_VERTICALITY_MIN_LOD2 = 0.70
    BUILDING_WALL_PLANARITY_MIN_LOD2 = 0.55
    BUILDING_ROOF_HORIZONTALITY_MIN_LOD2 = 0.85
    BUILDING_ROOF_PLANARITY_MIN_LOD2 = 0.70
    
    # Building geometric thresholds - LOD3 mode (strictest)
    BUILDING_WALL_VERTICALITY_MIN_LOD3 = 0.75
    BUILDING_WALL_PLANARITY_MIN_LOD3 = 0.60
    BUILDING_ROOF_HORIZONTALITY_MIN_LOD3 = 0.85
    BUILDING_ROOF_PLANARITY_MIN_LOD3 = 0.75
    
    # ===================================================================
    # VEGETATION THRESHOLDS
    # ===================================================================
    
    # Height-based vegetation classification (meters)
    LOW_VEG_HEIGHT_MAX = 2.0      # Maximum height for low vegetation
    HIGH_VEG_HEIGHT_MIN = 1.5      # Minimum height for high vegetation
    MEDIUM_VEG_HEIGHT_MIN = 0.5    # Minimum height for medium vegetation
    
    # NDVI thresholds (normalized -1 to 1)
    NDVI_VEG_THRESHOLD = 0.35      # Minimum NDVI for vegetation
    NDVI_HIGH_VEG_THRESHOLD = 0.45  # Minimum NDVI for high vegetation
    NDVI_BUILDING_THRESHOLD = 0.15  # Maximum NDVI for buildings
    
    # Vegetation geometric thresholds
    VEG_PLANARITY_MAX = 0.3        # Maximum planarity for vegetation
    VEG_ROUGHNESS_MIN = 0.2        # Minimum roughness for vegetation
    
    # ===================================================================
    # GROUND THRESHOLDS
    # ===================================================================
    
    GROUND_HEIGHT_MAX = 0.5         # Maximum height above ground
    GROUND_PLANARITY_MIN = 0.85     # Minimum planarity for ground
    GROUND_ROUGHNESS_MAX = 0.05     # Maximum roughness for ground
    GROUND_HORIZONTALITY_MIN = 0.9  # Minimum horizontality for ground
    
    # ===================================================================
    # VEHICLE THRESHOLDS
    # ===================================================================
    
    VEHICLE_HEIGHT_MIN = 1.0        # Minimum height for vehicle detection
    VEHICLE_HEIGHT_MAX = 5.0        # Maximum height for vehicle detection
    VEHICLE_DENSITY_MIN = 0.7       # Minimum point density for vehicles
    
    # ===================================================================
    # WATER THRESHOLDS
    # ===================================================================
    
    WATER_HEIGHT_MAX = 0.2          # Maximum height variation for water
    WATER_PLANARITY_MIN = 0.95      # Minimum planarity for water
    WATER_INTENSITY_MAX = 0.1       # Maximum intensity for water
    
    # ===================================================================
    # BRIDGE THRESHOLDS
    # ===================================================================
    
    BRIDGE_HEIGHT_MIN = 2.0         # Minimum height above ground
    BRIDGE_PLANARITY_MIN = 0.7      # Minimum planarity for bridge deck
    BRIDGE_WIDTH_MIN = 3.0          # Minimum width for bridge detection
    
    @classmethod
    def get_building_thresholds(cls, mode: str = 'asprs') -> Dict[str, float]:
        """
        Get building detection thresholds for a specific mode.
        
        Args:
            mode: Detection mode ('asprs', 'lod2', or 'lod3')
            
        Returns:
            Dictionary of threshold values for the specified mode
        """
        mode = mode.lower()
        
        if mode == 'asprs':
            return {
                'height_min': cls.BUILDING_HEIGHT_MIN,
                'height_max': cls.BUILDING_HEIGHT_MAX,
                'wall_verticality_min': cls.BUILDING_WALL_VERTICALITY_MIN_ASPRS,
                'wall_planarity_min': cls.BUILDING_WALL_PLANARITY_MIN_ASPRS,
                'roof_horizontality_min': cls.BUILDING_ROOF_HORIZONTALITY_MIN_ASPRS,
                'roof_planarity_min': cls.BUILDING_ROOF_PLANARITY_MIN_ASPRS,
            }
        elif mode == 'lod2':
            return {
                'height_min': cls.BUILDING_HEIGHT_MIN,
                'height_max': cls.BUILDING_HEIGHT_MAX,
                'wall_verticality_min': cls.BUILDING_WALL_VERTICALITY_MIN_LOD2,
                'wall_planarity_min': cls.BUILDING_WALL_PLANARITY_MIN_LOD2,
                'roof_horizontality_min': cls.BUILDING_ROOF_HORIZONTALITY_MIN_LOD2,
                'roof_planarity_min': cls.BUILDING_ROOF_PLANARITY_MIN_LOD2,
            }
        elif mode == 'lod3':
            return {
                'height_min': cls.BUILDING_HEIGHT_MIN,
                'height_max': cls.BUILDING_HEIGHT_MAX,
                'wall_verticality_min': cls.BUILDING_WALL_VERTICALITY_MIN_LOD3,
                'wall_planarity_min': cls.BUILDING_WALL_PLANARITY_MIN_LOD3,
                'roof_horizontality_min': cls.BUILDING_ROOF_HORIZONTALITY_MIN_LOD3,
                'roof_planarity_min': cls.BUILDING_ROOF_PLANARITY_MIN_LOD3,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'asprs', 'lod2', or 'lod3'.")
    
    @classmethod
    def get_transport_thresholds(cls, strict_mode: bool = False) -> Dict[str, float]:
        """
        Get transport detection thresholds.
        
        Args:
            strict_mode: If True, use stricter thresholds for urban areas
            
        Returns:
            Dictionary of threshold values
        """
        return {
            'road_height_max': cls.ROAD_HEIGHT_MAX_STRICT if strict_mode else cls.ROAD_HEIGHT_MAX,
            'road_height_min': cls.ROAD_HEIGHT_MIN,
            'road_planarity_min': cls.ROAD_PLANARITY_MIN_STRICT if strict_mode else cls.ROAD_PLANARITY_MIN,
            'road_roughness_max': cls.ROAD_ROUGHNESS_MAX,
            'road_intensity_min': cls.ROAD_INTENSITY_MIN,
            'road_intensity_max': cls.ROAD_INTENSITY_MAX,
            'rail_height_max': cls.RAIL_HEIGHT_MAX_STRICT if strict_mode else cls.RAIL_HEIGHT_MAX,
            'rail_height_min': cls.RAIL_HEIGHT_MIN,
            'rail_planarity_min': cls.RAIL_PLANARITY_MIN_STRICT if strict_mode else cls.RAIL_PLANARITY_MIN,
            'rail_roughness_max': cls.RAIL_ROUGHNESS_MAX,
            'rail_intensity_min': cls.RAIL_INTENSITY_MIN,
            'rail_intensity_max': cls.RAIL_INTENSITY_MAX,
        }
    
    @classmethod
    def get_all_thresholds(cls) -> Dict[str, Any]:
        """
        Get all thresholds as a dictionary.
        
        Returns:
            Dictionary containing all threshold values organized by category
        """
        return {
            'transport': cls.get_transport_thresholds(),
            'transport_strict': cls.get_transport_thresholds(strict_mode=True),
            'building_asprs': cls.get_building_thresholds('asprs'),
            'building_lod2': cls.get_building_thresholds('lod2'),
            'building_lod3': cls.get_building_thresholds('lod3'),
            'vegetation': {
                'low_veg_height_max': cls.LOW_VEG_HEIGHT_MAX,
                'high_veg_height_min': cls.HIGH_VEG_HEIGHT_MIN,
                'medium_veg_height_min': cls.MEDIUM_VEG_HEIGHT_MIN,
                'ndvi_veg_threshold': cls.NDVI_VEG_THRESHOLD,
                'ndvi_high_veg_threshold': cls.NDVI_HIGH_VEG_THRESHOLD,
                'ndvi_building_threshold': cls.NDVI_BUILDING_THRESHOLD,
            },
            'ground': {
                'height_max': cls.GROUND_HEIGHT_MAX,
                'planarity_min': cls.GROUND_PLANARITY_MIN,
                'roughness_max': cls.GROUND_ROUGHNESS_MAX,
                'horizontality_min': cls.GROUND_HORIZONTALITY_MIN,
            },
            'vehicle': {
                'height_min': cls.VEHICLE_HEIGHT_MIN,
                'height_max': cls.VEHICLE_HEIGHT_MAX,
                'density_min': cls.VEHICLE_DENSITY_MIN,
            },
            'water': {
                'height_max': cls.WATER_HEIGHT_MAX,
                'planarity_min': cls.WATER_PLANARITY_MIN,
                'intensity_max': cls.WATER_INTENSITY_MAX,
            },
            'bridge': {
                'height_min': cls.BRIDGE_HEIGHT_MIN,
                'planarity_min': cls.BRIDGE_PLANARITY_MIN,
                'width_min': cls.BRIDGE_WIDTH_MIN,
            },
        }
    
    @classmethod
    def validate_thresholds(cls) -> Dict[str, str]:
        """
        Validate threshold consistency and return any warnings.
        
        Returns:
            Dictionary of validation messages (empty if all OK)
        """
        warnings = {}
        
        # Check for overlapping vegetation height ranges
        if cls.LOW_VEG_HEIGHT_MAX < cls.HIGH_VEG_HEIGHT_MIN:
            # Gap between ranges
            warnings['vegetation_height_gap'] = (
                f"Gap between low vegetation max ({cls.LOW_VEG_HEIGHT_MAX}m) "
                f"and high vegetation min ({cls.HIGH_VEG_HEIGHT_MIN}m)"
            )
        elif cls.LOW_VEG_HEIGHT_MAX > cls.HIGH_VEG_HEIGHT_MIN:
            # Overlap is intentional (transition zone) - just informational
            warnings['vegetation_height_overlap'] = (
                f"Overlap between low vegetation max ({cls.LOW_VEG_HEIGHT_MAX}m) "
                f"and high vegetation min ({cls.HIGH_VEG_HEIGHT_MIN}m) - "
                "This is intentional for smooth transitions"
            )
        
        # Check building height minimum
        if cls.BUILDING_HEIGHT_MIN < cls.HIGH_VEG_HEIGHT_MIN:
            warnings['building_veg_overlap'] = (
                f"Building height min ({cls.BUILDING_HEIGHT_MIN}m) overlaps with "
                f"high vegetation ({cls.HIGH_VEG_HEIGHT_MIN}m)"
            )
        
        return warnings


# Create a convenience instance for import
thresholds = UnifiedThresholds()


def print_threshold_summary():
    """Print a summary of all thresholds for debugging."""
    print("=" * 70)
    print("UNIFIED CLASSIFICATION THRESHOLDS")
    print("=" * 70)
    
    all_thresholds = UnifiedThresholds.get_all_thresholds()
    
    for category, values in all_thresholds.items():
        print(f"\n{category.upper()}")
        print("-" * 70)
        for key, value in values.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    warnings = UnifiedThresholds.validate_thresholds()
    if warnings:
        for key, msg in warnings.items():
            print(f"  ⚠️  {msg}")
    else:
        print("  ✅ All thresholds are consistent")
    
    print("=" * 70)


if __name__ == "__main__":
    # Print summary when run directly
    print_threshold_summary()
