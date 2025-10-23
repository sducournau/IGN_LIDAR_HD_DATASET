"""
Adapters for Legacy Rule Engines

This module provides adapter classes that bridge legacy classification engines
(SpectralRulesEngine, GeometricRulesEngine) to the modern rules framework.

The adapter pattern allows:
- Using legacy engines with new HierarchicalRuleEngine
- Applying validation framework to legacy results
- Leveraging confidence scoring with legacy code
- Gradual migration without breaking existing code

Usage:
    from ign_lidar.core.classification.rules.adapters import LegacyEngineAdapter
    
    # Create adapter for legacy engine
    adapter = LegacyEngineAdapter(my_engine, RuleType.GEOMETRIC)
    
    # Use in hierarchical engine
    hierarchical = HierarchicalRuleEngine()
    hierarchical.add_rule(adapter)
    
Author: Classification Enhancement Team
Date: October 23, 2025
"""

from abc import abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import logging

from .base import (
    BaseRule,
    RuleType,
    RulePriority,
    RuleResult,
    RuleConfig
)

logger = logging.getLogger(__name__)


class LegacyEngineAdapter(BaseRule):
    """
    Adapter to use legacy classification engines with the modern rules framework.
    
    This adapter wraps existing classification engines (SpectralRulesEngine,
    GeometricRulesEngine) and provides the standard BaseRule interface, allowing
    them to be used in the hierarchical rule system.
    
    The adapter handles:
    - Converting legacy engine results to standard (mask, confidence) format
    - Mapping legacy classifications to target classes
    - Providing consistent error handling
    - Enabling composition with other rules
    
    Attributes:
        engine: The legacy engine instance to wrap
        
    Example:
        >>> from spectral_rules import SpectralRulesEngine
        >>> config = RuleConfig(
        ...     rule_id="spectral_vegetation",
        ...     rule_type=RuleType.SPECTRAL,
        ...     target_class=3  # Low vegetation
        ... )
        >>> engine = SpectralRulesEngine(nir_threshold=0.4)
        >>> adapter = LegacyEngineAdapter(config, engine)
        >>> mask, confidence = adapter.evaluate(points, features, context)
    """
    
    def __init__(
        self,
        config: RuleConfig,
        engine: Any
    ):
        """
        Initialize legacy engine adapter.
        
        Args:
            config: Rule configuration (see BaseRule)
            engine: Legacy engine instance to wrap
        """
        super().__init__(config)
        self.engine = engine
    
    @abstractmethod
    def evaluate(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate rule on points using legacy engine.
        
        This method must be implemented by subclasses to:
        1. Call appropriate legacy engine method
        2. Convert results to (mask, confidence) format
        3. Handle any engine-specific logic
        
        Args:
            points: Point cloud array (N, 3+) with XYZ coordinates
            features: Dictionary of computed features
            context: Optional execution context
            
        Returns:
            Tuple of:
                - mask: Boolean array (N,) indicating matching points
                - confidence: Float array (N,) with confidence scores [0, 1]
                
        Raises:
            ValueError: If required features missing
            RuntimeError: If engine execution fails
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def get_required_features(self) -> set:
        """Get required features - default to empty set for adapters"""
        return set()
    
    def get_optional_features(self) -> set:
        """Get optional features - default to empty set for adapters"""
        return set()
    
    def _convert_classification_to_mask(
        self,
        classifications: np.ndarray,
        target_class: int,
        num_points: int
    ) -> np.ndarray:
        """
        Convert multi-class classifications to binary mask for target class.
        
        Args:
            classifications: Array of ASPRS class codes
            target_class: Target class to extract
            num_points: Total number of points
            
        Returns:
            Boolean mask indicating points classified as target_class
        """
        if classifications is None or len(classifications) == 0:
            return np.zeros(num_points, dtype=bool)
        
        return classifications == target_class
    
    def _compute_default_confidence(
        self,
        mask: np.ndarray,
        features: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute default confidence scores for matched points.
        
        This provides simple confidence scoring when the legacy engine
        doesn't provide its own confidence values.
        
        Args:
            mask: Boolean mask of matching points
            features: Optional features for confidence computation
            
        Returns:
            Confidence scores [0, 1] for all points
        """
        confidence = np.zeros(len(mask), dtype=np.float32)
        
        # Matched points get default confidence of 0.8
        # (conservative estimate since legacy engines don't provide confidence)
        confidence[mask] = 0.8
        
        return confidence
    
    def __repr__(self) -> str:
        """String representation of adapter"""
        return (
            f"{self.__class__.__name__}("
            f"rule_id={self.rule_id}, "
            f"type={self.rule_type.value}, "
            f"priority={self.priority.value}, "
            f"engine={self.engine.__class__.__name__})"
        )


class MultiClassAdapter(LegacyEngineAdapter):
    """
    Adapter for legacy engines that return multi-class results.
    
    This adapter extends LegacyEngineAdapter to handle engines that
    classify points into multiple classes simultaneously (e.g.,
    GeometricRulesEngine.apply_all_rules()).
    
    The adapter can extract results for a specific target class or
    return results for all classes.
    
    Attributes:
        class_mapping: Optional mapping from engine classes to ASPRS codes
        return_all_classes: If True, return results for all classes
        
    Example:
        >>> from geometric_rules import GeometricRulesEngine
        >>> config = RuleConfig(
        ...     rule_id="geometric_buildings",
        ...     rule_type=RuleType.GEOMETRIC,
        ...     target_class=6  # Buildings
        ... )
        >>> engine = GeometricRulesEngine()
        >>> adapter = MultiClassAdapter(config, engine)
        >>> mask, confidence = adapter.evaluate(points, features, context)
    """
    
    def __init__(
        self,
        config: RuleConfig,
        engine: Any,
        class_mapping: Optional[Dict[int, int]] = None,
        return_all_classes: bool = False
    ):
        """
        Initialize multi-class adapter.
        
        Args:
            config: Rule configuration
            engine: Legacy engine instance
            class_mapping: Optional mapping from engine to ASPRS codes
            return_all_classes: If True, return results for all classes
        """
        super().__init__(config, engine)
        self.class_mapping = class_mapping or {}
        self.return_all_classes = return_all_classes
        
    def _apply_class_mapping(
        self,
        classifications: np.ndarray
    ) -> np.ndarray:
        """
        Apply class mapping to convert engine classes to ASPRS codes.
        
        Args:
            classifications: Array of engine-specific class codes
            
        Returns:
            Array of ASPRS class codes
        """
        if not self.class_mapping:
            return classifications
        
        mapped = classifications.copy()
        for engine_class, asprs_class in self.class_mapping.items():
            mapped[classifications == engine_class] = asprs_class
        
        return mapped
