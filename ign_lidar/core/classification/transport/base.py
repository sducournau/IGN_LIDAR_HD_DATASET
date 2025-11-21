"""
Transport Module Base Classes and Configurations

This module provides the foundational infrastructure for transport classification:
- Abstract base classes for transport detection and enhancement
- Enums for transport modes, types, and strategies
- Configuration dataclasses for detection, buffering, and indexing
- Result types for transport detection outputs

Author: Transport Module Consolidation (Phase 3)
Date: October 22, 2025
Version: 3.1.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import numpy as np

# Import thresholds (v3.1 - consolidated)
from ..thresholds import ClassificationThresholds


# ============================================================================
# Enumerations
# ============================================================================

class TransportMode(str, Enum):
    """
    Transport detection modes with different strategies and outputs.
    
    Modes:
    - ASPRS_STANDARD: Simple road (11) and rail (10) detection
    - ASPRS_EXTENDED: Detailed road types (motorway, primary, etc.) and rail types
    - LOD2: Ground-level transport surfaces for LOD2 training
    """
    ASPRS_STANDARD = "asprs_standard"
    ASPRS_EXTENDED = "asprs_extended"
    LOD2 = "lod2"


class TransportType(str, Enum):
    """Transport infrastructure type."""
    ROAD = "road"
    RAILWAY = "railway"
    UNKNOWN = "unknown"


class DetectionStrategy(str, Enum):
    """Transport detection strategy."""
    GROUND_TRUTH = "ground_truth"      # From BD TOPO® vector data
    GEOMETRIC = "geometric"            # Planarity, height, roughness
    INTENSITY = "intensity"            # LiDAR intensity refinement
    SPATIAL = "spatial"                # Spatial indexing (R-tree)
    HYBRID = "hybrid"                  # Combination of strategies


# ============================================================================
# Configuration Base Class
# ============================================================================

@dataclass
class TransportConfigBase:
    """
    Base configuration class for all transport modules.
    
    Provides common attributes and validation logic.
    """
    
    # Mode and strategy
    mode: TransportMode = TransportMode.ASPRS_STANDARD
    strict_mode: bool = False
    
    # Common flags
    use_ground_truth: bool = True
    ground_truth_priority: bool = True
    use_geometric_detection: bool = True
    use_intensity_refinement: bool = True
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not isinstance(self.mode, TransportMode):
            raise ValueError(f"Invalid mode: {self.mode}")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


# ============================================================================
# Detection Configuration
# ============================================================================

@dataclass
class DetectionConfig(TransportConfigBase):
    """
    Configuration for transport detection with mode-specific thresholds.
    
    Inherits from TransportConfigBase and adds detection-specific parameters.
    Note: Thresholds use ClassificationThresholds for consistency across modules.
    """
    
    # === Height thresholds ===
    road_height_max: float = field(default=None)
    road_height_min: float = field(default=None)
    rail_height_max: float = field(default=None)
    rail_height_min: float = field(default=None)
    
    # === Geometric thresholds ===
    road_planarity_min: float = field(default=None)
    rail_planarity_min: float = field(default=None)
    road_roughness_max: float = field(default=None)
    rail_roughness_max: float = field(default=None)
    
    # === Intensity thresholds ===
    road_intensity_min: float = field(default=None)
    road_intensity_max: float = field(default=None)
    rail_intensity_min: float = field(default=None)
    rail_intensity_max: float = field(default=None)
    
    # === Mode-specific flags ===
    detect_road_types: bool = False
    detect_rail_types: bool = False
    road_intensity_filter: bool = False
    rail_intensity_filter: bool = False
    
    # === Intensity filters for materials ===
    intensity_asphalt_min: float = 0.2
    intensity_asphalt_max: float = 0.6
    intensity_concrete_min: float = 0.4
    intensity_concrete_max: float = 0.8
    intensity_gravel_min: float = 0.3
    intensity_gravel_max: float = 0.7
    
    # === Road type detection (ASPRS Extended) ===
    motorway_width_min: float = 10.0
    primary_width_min: float = 7.0
    secondary_width_min: float = 5.0
    service_width_max: float = 4.0
    
    # === LOD2-specific ===
    classify_as_ground: bool = False
    separate_road_rail: bool = True
    
    # === Detection strategies ===
    use_road_ground_truth: bool = True
    use_rail_ground_truth: bool = True
    use_buffer_tolerance: bool = True
    buffer_tolerance: float = 0.5
    
    def __post_init__(self):
        """Initialize thresholds from ClassificationThresholds if not provided."""
        # Height thresholds
        if self.road_height_max is None:
            self.road_height_max = (ClassificationThresholds.ROAD_HEIGHT_MAX_STRICT 
                                   if self.strict_mode 
                                   else ClassificationThresholds.ROAD_HEIGHT_MAX)
        if self.road_height_min is None:
            self.road_height_min = ClassificationThresholds.ROAD_HEIGHT_MIN
        if self.rail_height_max is None:
            self.rail_height_max = (ClassificationThresholds.RAIL_HEIGHT_MAX_STRICT 
                                   if self.strict_mode 
                                   else ClassificationThresholds.RAIL_HEIGHT_MAX)
        if self.rail_height_min is None:
            self.rail_height_min = ClassificationThresholds.RAIL_HEIGHT_MIN
        
        # Geometric thresholds
        if self.road_planarity_min is None:
            self.road_planarity_min = (ClassificationThresholds.ROAD_PLANARITY_MIN_STRICT 
                                      if self.strict_mode 
                                      else ClassificationThresholds.ROAD_PLANARITY_MIN)
        if self.rail_planarity_min is None:
            self.rail_planarity_min = (ClassificationThresholds.RAIL_PLANARITY_MIN_STRICT 
                                      if self.strict_mode 
                                      else ClassificationThresholds.RAIL_PLANARITY_MIN)
        if self.road_roughness_max is None:
            self.road_roughness_max = ClassificationThresholds.ROAD_ROUGHNESS_MAX
        if self.rail_roughness_max is None:
            self.rail_roughness_max = ClassificationThresholds.RAIL_ROUGHNESS_MAX
        
        # Intensity thresholds
        if self.road_intensity_min is None:
            self.road_intensity_min = ClassificationThresholds.ROAD_INTENSITY_MIN
        if self.road_intensity_max is None:
            self.road_intensity_max = ClassificationThresholds.ROAD_INTENSITY_MAX
        if self.rail_intensity_min is None:
            self.rail_intensity_min = ClassificationThresholds.RAIL_INTENSITY_MIN
        if self.rail_intensity_max is None:
            self.rail_intensity_max = ClassificationThresholds.RAIL_INTENSITY_MAX
        
        # Configure mode-specific settings
        self._configure_mode_settings()
    
    def _configure_mode_settings(self):
        """Configure settings based on detection mode."""
        if self.mode == TransportMode.ASPRS_STANDARD:
            self.detect_road_types = False
            self.detect_rail_types = False
            self.road_intensity_filter = True
            self.rail_intensity_filter = False
            self.classify_as_ground = False
            self.separate_road_rail = True
            
        elif self.mode == TransportMode.ASPRS_EXTENDED:
            self.detect_road_types = True
            self.detect_rail_types = True
            self.road_intensity_filter = True
            self.rail_intensity_filter = True
            self.classify_as_ground = False
            self.separate_road_rail = True
            
        elif self.mode == TransportMode.LOD2:
            self.detect_road_types = False
            self.detect_rail_types = False
            self.road_intensity_filter = False
            self.rail_intensity_filter = False
            self.classify_as_ground = True
            self.separate_road_rail = False


# ============================================================================
# Buffering Configuration
# ============================================================================

@dataclass
class BufferingConfig:
    """Configuration for adaptive buffering strategies."""
    
    # Curvature-aware buffering
    curvature_aware: bool = True
    curvature_factor: float = 0.25
    min_curve_radius: float = 50.0
    
    # Road-type specific tolerances
    type_specific_tolerance: bool = True
    tolerance_motorway: float = 0.6
    tolerance_primary: float = 0.5
    tolerance_secondary: float = 0.4
    tolerance_residential: float = 0.35
    tolerance_service: float = 0.25
    tolerance_railway_main: float = 0.7
    tolerance_railway_tram: float = 0.4
    
    # Intersection enhancement
    intersection_enhancement: bool = True
    intersection_threshold: float = 1.5
    intersection_buffer_multiplier: float = 1.6
    
    # Elevation awareness
    elevation_aware: bool = True
    elevation_tolerance: float = 1.5
    elevation_min: float = -0.3
    elevation_max_road: float = 1.5
    elevation_max_rail: float = 1.2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


# ============================================================================
# Spatial Indexing Configuration
# ============================================================================

@dataclass
class IndexingConfig:
    """Configuration for spatial indexing."""
    
    enabled: bool = True
    index_type: str = "rtree"
    cache_index: bool = True
    cache_dir: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        # Convert Path to string for JSON serialization
        if data.get('cache_dir'):
            data['cache_dir'] = str(data['cache_dir'])
        return data


# ============================================================================
# Quality Metrics Configuration
# ============================================================================

@dataclass
class QualityMetricsConfig:
    """Configuration for quality metrics and validation."""
    
    enabled: bool = True
    save_confidence: bool = True
    detect_overlaps: bool = True
    generate_reports: bool = False
    report_output_dir: Optional[Path] = None
    low_confidence_threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        # Convert Path to string for JSON serialization
        if data.get('report_output_dir'):
            data['report_output_dir'] = str(data['report_output_dir'])
        return data


# ============================================================================
# Result Types
# ============================================================================

@dataclass
class TransportStats:
    """Statistics for transport detection results."""
    
    # Road statistics
    roads_ground_truth: int = 0
    roads_geometric: int = 0
    total_roads: int = 0
    
    # Railway statistics
    rails_ground_truth: int = 0
    rails_geometric: int = 0
    total_rails: int = 0
    
    # Extended statistics (ASPRS Extended mode)
    motorways: int = 0
    primary_roads: int = 0
    secondary_roads: int = 0
    residential_roads: int = 0
    service_roads: int = 0
    other_roads: int = 0
    main_railways: int = 0
    service_railways: int = 0
    
    # LOD2 statistics
    roads_validated: int = 0
    rails_validated: int = 0
    transport_ground_total: int = 0
    
    # Quality metrics
    avg_confidence: float = 0.0
    low_confidence_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return asdict(self)


@dataclass
class TransportDetectionResult:
    """
    Result type for transport detection.
    
    Provides type-safe container for detection outputs with
    labels, confidence scores, and summary statistics.
    """
    
    # Core results
    labels: np.ndarray
    stats: TransportStats
    
    # Optional quality metrics
    confidence: Optional[np.ndarray] = None
    
    # Metadata
    mode: TransportMode = TransportMode.ASPRS_STANDARD
    strategy: DetectionStrategy = DetectionStrategy.HYBRID
    
    def get_summary(self) -> str:
        """Get human-readable summary of detection results."""
        lines = [
            f"Transport Detection Results ({self.mode.value})",
            "=" * 50,
            f"Strategy: {self.strategy.value}",
            f"Total points: {len(self.labels):,}",
            "",
            "Road Detection:",
            f"  Ground truth: {self.stats.roads_ground_truth:,}",
            f"  Geometric:    {self.stats.roads_geometric:,}",
            f"  Total roads:  {self.stats.total_roads:,}",
            "",
            "Railway Detection:",
            f"  Ground truth: {self.stats.rails_ground_truth:,}",
            f"  Geometric:    {self.stats.rails_geometric:,}",
            f"  Total rails:  {self.stats.total_rails:,}",
        ]
        
        if self.confidence is not None:
            lines.extend([
                "",
                "Quality Metrics:",
                f"  Avg confidence: {self.stats.avg_confidence:.2f}",
                f"  Low confidence: {self.stats.low_confidence_ratio*100:.1f}%",
            ])
        
        return "\n".join(lines)


# ============================================================================
# Abstract Base Classes
# ============================================================================

class TransportDetectorBase(ABC):
    """
    Abstract base class for transport detection.
    
    Defines the interface for transport detection implementations
    with mode-specific strategies.
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize transport detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        config.validate()
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def _detect_asprs_standard(self, *args, **kwargs) -> Tuple[np.ndarray, TransportStats]:
        """ASPRS Standard mode: Simple road (11) and rail (10) detection."""
        pass
    
    @abstractmethod
    def _detect_asprs_extended(self, *args, **kwargs) -> Tuple[np.ndarray, TransportStats]:
        """ASPRS Extended mode: Detailed road and rail type classification."""
        pass
    
    @abstractmethod
    def _detect_lod2(self, *args, **kwargs) -> Tuple[np.ndarray, TransportStats]:
        """LOD2 mode: Roads and rails as ground-level surfaces."""
        pass


class TransportBufferBase(ABC):
    """
    Abstract base class for transport buffering.
    
    Defines interface for adaptive buffering strategies.
    """
    
    def __init__(self, config: BufferingConfig):
        """
        Initialize transport buffer.
        
        Args:
            config: Buffering configuration
        """
        self.config = config
    
    @abstractmethod
    def process_roads(self, roads_gdf) -> Any:
        """Process roads with adaptive buffering."""
        pass
    
    @abstractmethod
    def process_railways(self, railways_gdf) -> Any:
        """Process railways with adaptive buffering."""
        pass


class TransportClassifierBase(ABC):
    """
    Abstract base class for spatial transport classification.
    
    Defines interface for fast spatial indexing and classification.
    """
    
    def __init__(self, config: IndexingConfig):
        """
        Initialize spatial classifier.
        
        Args:
            config: Indexing configuration
        """
        self.config = config
    
    @abstractmethod
    def index_roads(self, roads_gdf) -> None:
        """Build spatial index for roads."""
        pass
    
    @abstractmethod
    def index_railways(self, railways_gdf) -> None:
        """Build spatial index for railways."""
        pass
    
    @abstractmethod
    def classify_points_fast(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        asprs_code_road: int = 11,
        asprs_code_rail: int = 10
    ) -> np.ndarray:
        """Fast point classification using spatial index."""
        pass


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Enums
    'TransportMode',
    'TransportType',
    'DetectionStrategy',
    
    # Configuration classes
    'TransportConfigBase',
    'DetectionConfig',
    'BufferingConfig',
    'IndexingConfig',
    'QualityMetricsConfig',
    
    # Result types
    'TransportStats',
    'TransportDetectionResult',
    
    # Abstract base classes
    'TransportDetectorBase',
    'TransportBufferBase',
    'TransportClassifierBase',
]
