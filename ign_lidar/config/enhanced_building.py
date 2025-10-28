"""
Enhanced building classification configuration.

This module defines configuration options for LOD3 building classification
using the EnhancedBuildingClassifier (Phase 2.1-2.4).

Usage:
    >>> from ign_lidar.config import Config
    >>> from ign_lidar.config.enhanced_building import EnhancedBuildingConfig
    >>>
    >>> # Create config with enhanced building detection
    >>> config = Config.preset('lod3_buildings')
    >>> config.advanced.classification = {
    ...     'enhanced_building': EnhancedBuildingConfig(
    ...         enable_roof_detection=True,
    ...         enable_chimney_detection=True,
    ...         enable_balcony_detection=True
    ...     ).to_dict()
    ... }
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class EnhancedBuildingConfig:
    """
    Configuration for enhanced LOD3 building classification.

    This configuration enables advanced architectural feature detection
    including roof types, chimneys, and balconies for detailed building
    classification (LOD3).

    Attributes:
        # Feature toggles
        enable_roof_detection: Enable roof type detection (flat, gabled, hipped, complex)
        enable_chimney_detection: Enable chimney and superstructure detection
        enable_balcony_detection: Enable balcony and horizontal protrusion detection

        # Roof detection parameters
        roof_flat_threshold: Maximum angle for flat roof classification (degrees)
        roof_dbscan_eps: DBSCAN epsilon for roof plane segmentation (meters)
        roof_dbscan_min_samples: Minimum samples per roof plane segment

        # Chimney detection parameters
        chimney_min_height_above_roof: Minimum height above roof for chimney (meters)
        chimney_min_points: Minimum points required for chimney cluster
        chimney_dbscan_eps: DBSCAN epsilon for chimney clustering (meters)
        chimney_max_height_above_roof: Maximum height above roof (meters, for filtering)

        # Balcony detection parameters
        balcony_min_distance_from_facade: Minimum protrusion from facade (meters)
        balcony_min_points: Minimum points required for balcony cluster
        balcony_max_depth: Maximum protrusion depth for balconies (meters)
        balcony_dbscan_eps: DBSCAN epsilon for balcony clustering (meters)
        balcony_min_height_above_ground: Minimum height above ground (meters)
        balcony_confidence_threshold: Minimum confidence score for classification

    Building Type Presets:
        Residential (default):
            - Moderate thresholds
            - Detects standard balconies and chimneys
            - roof_flat_threshold=15.0, chimney_min_height=1.0

        Urban High-Density:
            - Stricter thresholds for smaller features
            - roof_flat_threshold=10.0, chimney_min_height=0.5
            - balcony_min_distance=0.3, balcony_min_points=20

        Industrial:
            - Larger features only
            - chimney_min_height=2.0, chimney_min_points=40
            - balcony_detection disabled

        Historic:
            - Sensitive to architectural details
            - roof_flat_threshold=25.0, chimney_min_height=0.8
            - balcony_min_distance=0.3

    Example:
        >>> # Residential building configuration
        >>> config = EnhancedBuildingConfig()
        >>>
        >>> # Urban high-density configuration
        >>> config = EnhancedBuildingConfig(
        ...     roof_flat_threshold=10.0,
        ...     chimney_min_height_above_roof=0.5,
        ...     chimney_min_points=15,
        ...     balcony_min_distance_from_facade=0.3,
        ...     balcony_min_points=20
        ... )
        >>>
        >>> # Industrial configuration (no balconies)
        >>> config = EnhancedBuildingConfig(
        ...     enable_balcony_detection=False,
        ...     chimney_min_height_above_roof=2.0,
        ...     chimney_min_points=40
        ... )

    See Also:
        ign_lidar.core.classification.building.EnhancedBuildingClassifier
        ign_lidar.core.classification.building.EnhancedClassifierConfig
    """

    # ========================================================================
    # Feature Toggles
    # ========================================================================

    enable_roof_detection: bool = True
    """Enable roof type detection (flat, gabled, hipped, complex)"""

    enable_chimney_detection: bool = True
    """Enable chimney and superstructure detection"""

    enable_balcony_detection: bool = True
    """Enable balcony and horizontal protrusion detection"""

    # ========================================================================
    # Roof Detection Parameters
    # ========================================================================

    roof_flat_threshold: float = 15.0
    """
    Maximum angle for flat roof classification (degrees).
    
    Lower values = stricter flat roof definition.
    - 10.0: Very flat roofs only (urban high-density)
    - 15.0: Standard flat roofs (residential default)
    - 20.0: Moderately pitched considered flat (industrial)
    - 25.0: Lenient flat roof definition (historic)
    """

    roof_dbscan_eps: float = 0.3
    """
    DBSCAN epsilon for roof plane segmentation (meters).
    
    Distance threshold for grouping roof points into planes.
    - 0.2: Fine-grained segmentation (complex roofs)
    - 0.3: Standard segmentation (residential default)
    - 0.5: Coarse segmentation (simple roofs)
    """

    roof_dbscan_min_samples: int = 30
    """
    Minimum samples per roof plane segment.
    
    Higher values = more robust but may miss small features.
    - 20: Detect smaller roof planes
    - 30: Standard (residential default)
    - 50: Large planes only (industrial)
    """

    # ========================================================================
    # Chimney Detection Parameters
    # ========================================================================

    chimney_min_height_above_roof: float = 1.0
    """
    Minimum height above roof for chimney detection (meters).
    
    Lower values detect smaller features but increase false positives.
    - 0.5: Detect small chimneys/vents (urban high-density)
    - 1.0: Standard chimneys (residential default)
    - 1.5: Tall chimneys only (reduce false positives)
    - 2.0: Large industrial stacks only
    """

    chimney_min_points: int = 20
    """
    Minimum points required for chimney cluster.
    
    Higher values = more robust detection but may miss small features.
    - 15: Detect smaller features (urban high-density)
    - 20: Standard (residential default)
    - 30: Robust detection
    - 40: Large features only (industrial)
    """

    chimney_dbscan_eps: float = 0.5
    """
    DBSCAN epsilon for chimney clustering (meters).
    
    Distance threshold for grouping chimney points.
    - 0.3: Tight clustering (separate nearby features)
    - 0.5: Standard clustering (residential default)
    - 0.8: Loose clustering (group related structures)
    """

    chimney_max_height_above_roof: float = 10.0
    """
    Maximum height above roof for chimney detection (meters).
    
    Filters out extremely tall structures (likely not chimneys).
    - 5.0: Residential chimneys only
    - 10.0: Standard (default)
    - 20.0: Industrial stacks
    """

    # ========================================================================
    # Balcony Detection Parameters
    # ========================================================================

    balcony_min_distance_from_facade: float = 0.5
    """
    Minimum protrusion from facade for balcony detection (meters).
    
    Lower values detect smaller balconies but increase false positives.
    - 0.3: Small balconies (urban high-density, historic)
    - 0.5: Standard balconies (residential default)
    - 0.8: Large balconies/terraces only
    """

    balcony_min_points: int = 25
    """
    Minimum points required for balcony cluster.
    
    Higher values = more robust but may miss small balconies.
    - 20: Detect smaller balconies (urban high-density)
    - 25: Standard (residential default)
    - 30: Robust detection
    - 40: Large terraces only
    """

    balcony_max_depth: float = 3.0
    """
    Maximum protrusion depth for balconies (meters).
    
    Larger values accommodate terraces and large balconies.
    - 2.0: Small balconies only
    - 3.0: Standard (residential default)
    - 5.0: Large terraces/verandas
    """

    balcony_dbscan_eps: float = 0.5
    """
    DBSCAN epsilon for balcony clustering (meters).
    
    Distance threshold for grouping balcony points.
    - 0.3: Tight clustering (separate nearby balconies)
    - 0.5: Standard clustering (residential default)
    - 0.8: Loose clustering (group related structures)
    """

    balcony_min_height_above_ground: float = 2.0
    """
    Minimum height above ground for balcony detection (meters).
    
    Filters out ground-level patios and terraces.
    - 1.5: Lower balconies (townhouses)
    - 2.0: Standard (residential default)
    - 3.0: Upper-floor balconies only
    """

    balcony_confidence_threshold: float = 0.5
    """
    Minimum confidence score for balcony classification (0.0-1.0).
    
    Higher values reduce false positives but may miss valid balconies.
    - 0.4: Lenient (more detections, historic buildings)
    - 0.5: Standard (residential default)
    - 0.6: Conservative (fewer false positives)
    - 0.7: Strict (high confidence only)
    """

    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation suitable for Config.advanced.classification

        Example:
            >>> config = EnhancedBuildingConfig()
            >>> config_dict = config.to_dict()
            >>> advanced_config.classification = {'enhanced_building': config_dict}
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "EnhancedBuildingConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary with configuration parameters

        Returns:
            EnhancedBuildingConfig instance

        Example:
            >>> data = {'enable_roof_detection': True, 'roof_flat_threshold': 12.0}
            >>> config = EnhancedBuildingConfig.from_dict(data)
        """
        return cls(**data)

    @classmethod
    def preset_residential(cls) -> "EnhancedBuildingConfig":
        """
        Preset for residential buildings (default).

        Returns:
            Configuration optimized for residential buildings
        """
        return cls()  # Use defaults

    @classmethod
    def preset_urban_high_density(cls) -> "EnhancedBuildingConfig":
        """
        Preset for urban high-density areas.

        Optimized for:
        - Smaller architectural features
        - Complex building geometries
        - Many flat roofs

        Returns:
            Configuration optimized for urban high-density areas
        """
        return cls(
            roof_flat_threshold=10.0,
            chimney_min_height_above_roof=0.5,
            chimney_min_points=15,
            balcony_min_distance_from_facade=0.3,
            balcony_min_points=20,
        )

    @classmethod
    def preset_industrial(cls) -> "EnhancedBuildingConfig":
        """
        Preset for industrial buildings.

        Optimized for:
        - Large chimneys and stacks
        - Simple roof geometries
        - No balconies

        Returns:
            Configuration optimized for industrial buildings
        """
        return cls(
            enable_balcony_detection=False,
            roof_flat_threshold=20.0,
            chimney_min_height_above_roof=2.0,
            chimney_min_points=40,
            chimney_max_height_above_roof=20.0,
        )

    @classmethod
    def preset_historic(cls) -> "EnhancedBuildingConfig":
        """
        Preset for historic buildings.

        Optimized for:
        - Complex architectural details
        - Ornate balconies
        - Varied roof types

        Returns:
            Configuration optimized for historic buildings
        """
        return cls(
            roof_flat_threshold=25.0,
            roof_dbscan_min_samples=40,
            chimney_min_height_above_roof=0.8,
            balcony_min_distance_from_facade=0.3,
            balcony_confidence_threshold=0.4,
        )


# Convenience aliases for backward compatibility
RoofDetectionConfig = EnhancedBuildingConfig
ChimneyDetectionConfig = EnhancedBuildingConfig
BalconyDetectionConfig = EnhancedBuildingConfig


__all__ = [
    "EnhancedBuildingConfig",
    "RoofDetectionConfig",
    "ChimneyDetectionConfig",
    "BalconyDetectionConfig",
]
