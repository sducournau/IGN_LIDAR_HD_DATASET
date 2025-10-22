"""
Building Module Base Classes

This module defines abstract base classes and common interfaces
for building classification modules.

Author: Phase 2 - Building Module Restructuring
Date: October 22, 2025
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations
# ============================================================================

class BuildingMode(str, Enum):
    """Building classification modes."""
    ASPRS = "asprs"        # General building detection (ASPRS class 6)
    LOD2 = "lod2"          # LOD2 building elements (walls, roofs)
    LOD3 = "lod3"          # LOD3 detailed elements (windows, doors, etc.)


class BuildingSource(str, Enum):
    """Building data sources."""
    BD_TOPO = "bd_topo"          # IGN BD TOPO (most accurate)
    CADASTRE = "cadastre"         # Cadastre (parcels)
    OSM = "osm"                   # OpenStreetMap
    FUSED = "fused"               # Multi-source fusion


class ClassificationConfidence(str, Enum):
    """Confidence levels for classification."""
    CERTAIN = "certain"           # >0.9 confidence
    HIGH = "high"                 # 0.7-0.9
    MEDIUM = "medium"             # 0.5-0.7
    LOW = "low"                   # 0.3-0.5
    UNCERTAIN = "uncertain"       # <0.3


# ============================================================================
# Configuration Base Classes
# ============================================================================

@dataclass
class BuildingConfigBase:
    """
    Base configuration class for building classification.
    
    All building classifiers should inherit from this to ensure
    consistent configuration interface.
    """
    
    # Mode selection
    mode: BuildingMode = BuildingMode.ASPRS
    
    # Common height thresholds
    min_height: float = 2.5              # Minimum building height (meters)
    max_height: float = 200.0            # Maximum building height (meters)
    
    # Common geometric thresholds
    min_planarity: float = 0.5           # Minimum planarity for buildings
    
    # Ground truth usage
    use_ground_truth: bool = True        # Use ground truth polygons
    ground_truth_priority: bool = True   # Prioritize ground truth
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if self.min_height < 0:
            logger.error("min_height must be non-negative")
            return False
        
        if self.max_height <= self.min_height:
            logger.error("max_height must be greater than min_height")
            return False
        
        if not (0 <= self.min_planarity <= 1):
            logger.error("min_planarity must be in [0, 1]")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'mode': self.mode.value if isinstance(self.mode, Enum) else self.mode,
            'min_height': self.min_height,
            'max_height': self.max_height,
            'min_planarity': self.min_planarity,
            'use_ground_truth': self.use_ground_truth,
            'ground_truth_priority': self.ground_truth_priority
        }


# ============================================================================
# Result Classes
# ============================================================================

@dataclass
class BuildingClassificationResult:
    """
    Base result class for building classification.
    
    Contains common fields that all building classifiers should return.
    """
    
    # Classification outputs
    classifications: np.ndarray          # Classification codes (N,)
    confidences: np.ndarray              # Confidence scores (N,), range [0, 1]
    
    # Statistics
    n_points: int = 0                    # Total points processed
    n_buildings: int = 0                 # Number of buildings detected
    n_building_points: int = 0           # Points classified as buildings
    
    # Processing info
    processing_time: float = 0.0         # Processing time (seconds)
    mode: BuildingMode = BuildingMode.ASPRS
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary information
        """
        return {
            'n_points': self.n_points,
            'n_buildings': self.n_buildings,
            'n_building_points': self.n_building_points,
            'building_ratio': self.n_building_points / max(self.n_points, 1),
            'avg_confidence': float(np.mean(self.confidences)) if len(self.confidences) > 0 else 0.0,
            'processing_time': self.processing_time,
            'mode': self.mode.value if isinstance(self.mode, Enum) else self.mode
        }


# ============================================================================
# Abstract Base Classes
# ============================================================================

class BuildingClassifierBase(ABC):
    """
    Abstract base class for all building classifiers.
    
    All building classification implementations should inherit from this
    class and implement the required methods.
    """
    
    def __init__(self, config: Optional[BuildingConfigBase] = None):
        """
        Initialize building classifier.
        
        Args:
            config: Configuration object (defaults to BuildingConfigBase)
        """
        self.config = config if config is not None else BuildingConfigBase()
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration")
        
        logger.info(f"Initialized {self.__class__.__name__} in {self.config.mode} mode")
    
    @abstractmethod
    def classify(
        self,
        points: np.ndarray,
        **kwargs
    ) -> BuildingClassificationResult:
        """
        Classify points as building or non-building.
        
        Args:
            points: Point cloud array (N, 3+)
            **kwargs: Additional classifier-specific arguments
            
        Returns:
            BuildingClassificationResult with classifications and metadata
        """
        pass
    
    def preprocess(self, points: np.ndarray) -> np.ndarray:
        """
        Preprocess point cloud before classification.
        
        Default implementation: filter by height range.
        Subclasses can override for custom preprocessing.
        
        Args:
            points: Input point cloud (N, 3+)
            
        Returns:
            Preprocessed point cloud
        """
        if len(points) == 0:
            return points
        
        # Assume column 2 (index 2) is height above ground
        if points.shape[1] > 2:
            heights = points[:, 2]
            mask = (heights >= self.config.min_height) & (heights <= self.config.max_height)
            return points[mask]
        
        return points
    
    def validate_inputs(self, points: np.ndarray) -> bool:
        """
        Validate input point cloud.
        
        Args:
            points: Point cloud to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(points, np.ndarray):
            logger.error("Points must be numpy array")
            return False
        
        if len(points.shape) != 2:
            logger.error(f"Points must be 2D array, got shape {points.shape}")
            return False
        
        if points.shape[1] < 3:
            logger.error(f"Points must have at least 3 columns (XYZ), got {points.shape[1]}")
            return False
        
        return True
    
    def get_config(self) -> BuildingConfigBase:
        """Get current configuration."""
        return self.config
    
    def set_mode(self, mode: BuildingMode):
        """
        Change classification mode.
        
        Args:
            mode: New mode (ASPRS, LOD2, or LOD3)
        """
        self.config.mode = mode
        logger.info(f"Mode changed to {mode}")


class BuildingDetectorBase(BuildingClassifierBase):
    """
    Base class specifically for building detection algorithms.
    
    Extends BuildingClassifierBase with detection-specific methods.
    """
    
    @abstractmethod
    def detect_walls(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Detect wall points.
        
        Args:
            points: Point cloud array (N, 3+)
            **kwargs: Additional arguments
            
        Returns:
            Boolean mask (N,) indicating wall points
        """
        pass
    
    @abstractmethod
    def detect_roofs(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Detect roof points.
        
        Args:
            points: Point cloud array (N, 3+)
            **kwargs: Additional arguments
            
        Returns:
            Boolean mask (N,) indicating roof points
        """
        pass


class BuildingClustererBase(ABC):
    """
    Base class for building point clustering algorithms.
    
    Clusters building points by individual buildings.
    """
    
    @abstractmethod
    def cluster(
        self,
        points: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster building points by building ID.
        
        Args:
            points: Point cloud array (N, 3+)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (cluster_ids, metadata)
            - cluster_ids: Array (N,) with cluster ID for each point (-1 for noise)
            - metadata: Dictionary with clustering statistics
        """
        pass


class BuildingFusionBase(ABC):
    """
    Base class for multi-source building polygon fusion.
    
    Combines building footprints from multiple data sources.
    """
    
    @abstractmethod
    def fuse(
        self,
        sources: Dict[BuildingSource, Any],
        **kwargs
    ) -> Any:
        """
        Fuse building polygons from multiple sources.
        
        Args:
            sources: Dictionary mapping BuildingSource to polygon data
            **kwargs: Additional arguments
            
        Returns:
            Fused polygon dataset
        """
        pass
