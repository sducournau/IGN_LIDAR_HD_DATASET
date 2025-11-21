"""
Classification Engine - Wrapper for Classification Operations

This module provides a thin wrapper around classification components to decouple
LiDARProcessor from direct classification logic.

The ClassificationEngine delegates classification operations to the underlying
Classifier and Reclassifier while providing a cleaner API for the processor.

Architecture Pattern:
    LiDARProcessor
        └── ClassificationEngine (wrapper/facade)
            ├── Classifier (ground truth classification)
            └── Reclassifier (optimized reclassification)

Benefits:
- Separation of concerns: processor doesn't need classification details
- Easier to test: can mock ClassificationEngine
- Cleaner API: processor only sees classification operations it needs
- Future flexibility: can swap implementation without changing processor

Author: Phase 2 Refactoring - Session 4
Date: November 21, 2025
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from ..classification_schema import ASPRS_TO_LOD2, ASPRS_TO_LOD3
from .classification import (
    Classifier,
    ClassifierConfig,
    ClassificationStrategy,
    refine_classification,
)
from .classification.reclassifier import Reclassifier

logger = logging.getLogger(__name__)

__all__ = ["ClassificationEngine"]


class ClassificationEngine:
    """
    Wrapper for classification operations providing a clean API.
    
    This class acts as a facade to Classifier and Reclassifier, exposing only
    the methods needed by LiDARProcessor and hiding internal complexity.
    
    Args:
        config: Configuration dict containing classification settings
        lod_level: Level of detail ('LOD2', 'LOD3', 'ASPRS')
        
    Example:
        >>> engine = ClassificationEngine(config, lod_level='LOD2')
        >>> labels = engine.classify_with_ground_truth(points, features, ground_truth)
        >>> labels = engine.reclassify(points, labels, features)
        
    Note:
        This is a thin wrapper - all actual classification is delegated to
        Classifier and Reclassifier. The wrapper only provides API simplification.
    """
    
    def __init__(self, config: DictConfig, lod_level: str):
        """
        Initialize the classification engine.
        
        Args:
            config: OmegaConf configuration containing:
                - processor: Processing settings
                - features: Feature configuration
                - data_sources: Ground truth data sources
            lod_level: 'LOD2', 'LOD3', or 'ASPRS'
                
        Raises:
            ValueError: If lod_level is invalid
        """
        self.config = config
        self.lod_level = lod_level
        
        # Set class mapping based on LOD level
        if self.lod_level == "ASPRS":
            self.class_mapping = None
            self.default_class = 1  # ASPRS unclassified
        elif self.lod_level == "LOD2":
            self.class_mapping = ASPRS_TO_LOD2
            self.default_class = 14  # LOD2 unclassified
        else:  # LOD3
            self.class_mapping = ASPRS_TO_LOD3
            self.default_class = 29  # LOD3 unclassified
            
        logger.debug(f"✅ ClassificationEngine initialized (LOD={lod_level})")
        
    def create_classifier(
        self,
        strategy: str = 'comprehensive',
        lod_level: Optional[str] = None,
        use_gpu: bool = False
    ) -> Classifier:
        """
        Create a Classifier instance with appropriate configuration.
        
        Args:
            strategy: Classification strategy ('basic', 'adaptive', 'comprehensive')
            lod_level: Override LOD level (defaults to engine's lod_level)
            use_gpu: Enable GPU acceleration
            
        Returns:
            Configured Classifier instance
            
        Example:
            >>> classifier = engine.create_classifier(strategy='comprehensive')
            >>> result = classifier.classify(points, features)
        """
        lod = lod_level or self.lod_level
        
        # Create classifier config
        classifier_config = ClassifierConfig(
            lod_level=lod,
            use_gpu=use_gpu,
            height_threshold=self.config.processor.get("height_threshold", 2.0),
            buffer_distance=self.config.processor.get("buffer_distance", 1.0),
        )
        
        return Classifier(
            strategy=strategy,
            config=classifier_config,
            class_mapping=self.class_mapping
        )
        
    def classify_with_ground_truth(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth: Dict[str, Any],
        strategy: str = 'comprehensive'
    ) -> np.ndarray:
        """
        Classify points using ground truth data.
        
        Args:
            points: Point cloud array [N, 3+]
            features: Dictionary of feature arrays
            ground_truth: Ground truth data (buildings, roads, etc.)
            strategy: Classification strategy
            
        Returns:
            Classification labels array [N]
            
        Example:
            >>> labels = engine.classify_with_ground_truth(
            ...     points, features, ground_truth, strategy='comprehensive'
            ... )
        """
        classifier = self.create_classifier(strategy=strategy)
        
        # Use classifier's ground truth classification method
        labels = classifier._classify_by_ground_truth(
            points=points,
            features=features,
            ground_truth=ground_truth,
            lod_level=self.lod_level
        )
        
        return labels
        
    def create_reclassifier(
        self,
        acceleration_mode: str = 'auto',
        use_gpu: bool = False
    ) -> Reclassifier:
        """
        Create a Reclassifier instance for optimized reclassification.
        
        Args:
            acceleration_mode: 'auto', 'cpu', 'gpu', or 'numba'
            use_gpu: Enable GPU acceleration
            
        Returns:
            Configured Reclassifier instance
            
        Example:
            >>> reclassifier = engine.create_reclassifier(acceleration_mode='auto')
            >>> labels, stats = reclassifier.reclassify(points, labels, features)
        """
        return Reclassifier(
            lod_level=self.lod_level,
            acceleration_mode=acceleration_mode,
            use_gpu=use_gpu,
            config=self.config
        )
        
    def reclassify(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth: Optional[Dict[str, Any]] = None,
        acceleration_mode: str = 'auto'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reclassify points using geometric and spectral rules.
        
        Args:
            points: Point cloud array [N, 3+]
            labels: Current classification labels [N]
            features: Dictionary of feature arrays
            ground_truth: Optional ground truth data
            acceleration_mode: Acceleration strategy
            
        Returns:
            Tuple of (reclassified_labels, statistics)
            
        Example:
            >>> labels, stats = engine.reclassify(points, labels, features)
            >>> print(f"Reclassified {stats['total_reclassified']} points")
        """
        reclassifier = self.create_reclassifier(acceleration_mode=acceleration_mode)
        
        return reclassifier.reclassify(
            points=points,
            labels=labels,
            features=features,
            ground_truth=ground_truth
        )
        
    def reclassify_vegetation_above_surfaces(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth: Dict[str, Any]
    ) -> np.ndarray:
        """
        Reclassify vegetation points above building/road surfaces.
        
        This is a specialized reclassification for handling vegetation that
        appears above artificial surfaces (e.g., trees on roofs, roadside vegetation).
        
        Args:
            points: Point cloud array [N, 3+]
            labels: Current classification labels [N]
            features: Dictionary of feature arrays
            ground_truth: Ground truth data with buildings/roads
            
        Returns:
            Reclassified labels array [N]
            
        Example:
            >>> labels = engine.reclassify_vegetation_above_surfaces(
            ...     points, labels, features, ground_truth
            ... )
        """
        reclassifier = self.create_reclassifier()
        
        return reclassifier.reclassify_vegetation_above_surfaces(
            points=points,
            labels=labels,
            features=features,
            ground_truth=ground_truth
        )
        
    def refine_classification(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Apply classification refinement rules.
        
        Uses geometric and spectral rules to refine classification labels,
        correcting common misclassifications.
        
        Args:
            points: Point cloud array [N, 3+]
            labels: Current classification labels [N]
            features: Dictionary of feature arrays
            
        Returns:
            Refined labels array [N]
            
        Example:
            >>> labels = engine.refine_classification(points, labels, features)
        """
        return refine_classification(
            points=points,
            labels=labels,
            features=features,
            lod_level=self.lod_level
        )
        
    def reclassify_file(
        self,
        laz_file: Path,
        reclassifier: Optional[Reclassifier] = None
    ) -> Dict[str, Any]:
        """
        Reclassify an entire LAZ file in-place.
        
        Args:
            laz_file: Path to LAZ file to reclassify
            reclassifier: Optional pre-configured Reclassifier instance
            
        Returns:
            Dictionary with reclassification statistics
            
        Example:
            >>> stats = engine.reclassify_file(Path('tile.laz'))
            >>> print(f"Processed {stats['total_points']} points")
        """
        if reclassifier is None:
            reclassifier = self.create_reclassifier()
            
        return reclassifier.reclassify_file(laz_file)
        
    # ==================== Property Accessors ====================
    
    @property
    def has_class_mapping(self) -> bool:
        """Whether class mapping is enabled (False for ASPRS mode)."""
        return self.class_mapping is not None
        
    def get_class_name(self, class_code: int) -> str:
        """
        Get human-readable name for a class code.
        
        Args:
            class_code: Numeric class code
            
        Returns:
            Class name string
            
        Example:
            >>> name = engine.get_class_name(2)
            >>> print(name)  # 'Ground'
        """
        if self.lod_level == "ASPRS":
            from ..classification_schema import ASPRS_CLASS_NAMES
            return ASPRS_CLASS_NAMES.get(class_code, f"Class_{class_code}")
        elif self.lod_level == "LOD2":
            from ..classification_schema import LOD2_CLASS_NAMES
            return LOD2_CLASS_NAMES.get(class_code, f"Class_{class_code}")
        else:  # LOD3
            from ..classification_schema import LOD3_CLASS_NAMES
            return LOD3_CLASS_NAMES.get(class_code, f"Class_{class_code}")
            
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ClassificationEngine("
            f"lod={self.lod_level}, "
            f"mapping={'yes' if self.has_class_mapping else 'no'}, "
            f"default_class={self.default_class})"
        )
