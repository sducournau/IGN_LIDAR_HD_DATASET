"""
Adaptive Classification Module

This module provides classification rules that automatically adapt to available features.
When features have artifacts or are missing, classification rules adjust to use only
valid features while maintaining accuracy.

Key Features:
1. Automatic feature availability detection
2. Fallback classification rules for missing features
3. Dynamic confidence adjustment based on available features
4. Graceful degradation when features have artifacts

Author: IGN LiDAR HD Classification Team
Date: October 19, 2025
Version: 1.0
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureImportance(Enum):
    """Feature importance levels for classification."""
    CRITICAL = 3    # Classification impossible without this feature
    IMPORTANT = 2   # Accuracy significantly reduced without this feature
    HELPFUL = 1     # Improves accuracy but not essential
    OPTIONAL = 0    # Nice to have but minimal impact


@dataclass
class ClassificationRule:
    """
    Adaptive classification rule that works with varying feature sets.
    """
    name: str
    asprs_class: int
    
    # Feature requirements by importance
    critical_features: Set[str]
    important_features: Set[str]
    helpful_features: Set[str]
    optional_features: Set[str]
    
    # Thresholds (feature_name -> (min, max))
    thresholds: Dict[str, Tuple[Optional[float], Optional[float]]]
    
    # Confidence adjustment
    base_confidence: float = 0.8
    confidence_penalty_per_missing_important: float = 0.1
    confidence_penalty_per_missing_helpful: float = 0.05
    
    def can_classify(self, available_features: Set[str]) -> bool:
        """
        Check if classification is possible with available features.
        
        Args:
            available_features: Set of available feature names
        
        Returns:
            True if all critical features are available
        """
        return self.critical_features.issubset(available_features)
    
    def get_confidence(self, available_features: Set[str]) -> float:
        """
        Calculate confidence based on available features.
        
        Args:
            available_features: Set of available feature names
        
        Returns:
            Confidence score [0, 1]
        """
        if not self.can_classify(available_features):
            return 0.0
        
        confidence = self.base_confidence
        
        # Penalize for missing important features
        missing_important = self.important_features - available_features
        confidence -= len(missing_important) * self.confidence_penalty_per_missing_important
        
        # Penalize for missing helpful features
        missing_helpful = self.helpful_features - available_features
        confidence -= len(missing_helpful) * self.confidence_penalty_per_missing_helpful
        
        return max(0.0, min(1.0, confidence))
    
    def get_usable_features(self, available_features: Set[str]) -> Set[str]:
        """
        Get features that can be used for classification.
        
        Args:
            available_features: Set of available feature names
        
        Returns:
            Set of usable feature names
        """
        all_features = (
            self.critical_features | 
            self.important_features | 
            self.helpful_features | 
            self.optional_features
        )
        return all_features & available_features


class AdaptiveClassifier:
    """
    Classifier that adapts to available features by using fallback rules.
    
    When features have artifacts or are missing, this classifier automatically
    adjusts classification rules to use only valid features.
    """
    
    # ASPRS class constants
    ASPRS_UNCLASSIFIED = 1
    ASPRS_GROUND = 2
    ASPRS_LOW_VEGETATION = 3
    ASPRS_MEDIUM_VEGETATION = 4
    ASPRS_HIGH_VEGETATION = 5
    ASPRS_BUILDING = 6
    ASPRS_WATER = 9
    ASPRS_RAIL = 10
    ASPRS_ROAD = 11
    ASPRS_BRIDGE = 17
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize adaptive classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize classification rules
        self.rules = self._create_classification_rules()
        
        # Feature artifact tracking
        self.artifact_features: Set[str] = set()
        
        logger.info("AdaptiveClassifier initialized with adaptive rules")
    
    def _create_classification_rules(self) -> Dict[str, ClassificationRule]:
        """Create adaptive classification rules for each class."""
        
        rules = {}
        
        # ============================================================
        # BUILDING CLASSIFICATION
        # ============================================================
        rules['building'] = ClassificationRule(
            name='building',
            asprs_class=self.ASPRS_BUILDING,
            
            # Critical: Need at least height
            critical_features={'height'},
            
            # Important: Geometric features for validation
            important_features={'planarity', 'verticality'},
            
            # Helpful: Additional validation
            helpful_features={'curvature', 'normal_z', 'ndvi'},
            
            # Optional: Nice to have
            optional_features={'nir', 'intensity', 'brightness'},
            
            thresholds={
                'height': (1.5, None),        # Min 1.5m
                'planarity': (0.65, None),    # Planar surfaces
                'verticality': (0.6, None),   # Walls
                'curvature': (None, 0.10),    # Low curvature
                'normal_z': (None, 0.85),     # Not too horizontal
                'ndvi': (None, 0.20)          # Not vegetation
            },
            
            base_confidence=0.85
        )
        
        # ============================================================
        # ROAD CLASSIFICATION
        # ============================================================
        rules['road'] = ClassificationRule(
            name='road',
            asprs_class=self.ASPRS_ROAD,
            
            # Critical: Need planarity and height
            critical_features={'planarity', 'height'},
            
            # Important: Horizontal validation
            important_features={'normal_z'},
            
            # Helpful: Additional validation
            helpful_features={'curvature', 'ndvi'},
            
            # Optional
            optional_features={'intensity', 'brightness'},
            
            thresholds={
                'height': (-0.5, 2.0),        # Near ground
                'planarity': (0.85, None),    # Very flat
                'normal_z': (0.90, None),     # Horizontal
                'curvature': (None, 0.05),    # Smooth
                'ndvi': (None, 0.15)          # Not vegetation
            },
            
            base_confidence=0.80
        )
        
        # ============================================================
        # WATER CLASSIFICATION
        # ============================================================
        rules['water'] = ClassificationRule(
            name='water',
            asprs_class=self.ASPRS_WATER,
            
            # Critical: Need planarity
            critical_features={'planarity'},
            
            # Important: Horizontal and near ground
            important_features={'normal_z', 'height'},
            
            # Helpful: Spectral validation
            helpful_features={'ndvi', 'nir'},
            
            # Optional
            optional_features={'curvature', 'intensity'},
            
            thresholds={
                'height': (-0.5, 0.3),        # Very low
                'planarity': (0.90, None),    # Very flat
                'normal_z': (0.95, None),     # Very horizontal
                'curvature': (None, 0.02),    # Very smooth
                'ndvi': (None, 0.10),         # Not vegetation
                'nir': (None, 0.20)           # Low NIR
            },
            
            base_confidence=0.85
        )
        
        # ============================================================
        # VEGETATION CLASSIFICATION
        # ============================================================
        rules['vegetation'] = ClassificationRule(
            name='vegetation',
            asprs_class=self.ASPRS_HIGH_VEGETATION,  # Default to high
            
            # Critical: Need NDVI OR curvature
            critical_features={'ndvi', 'curvature'},  # At least one
            
            # Important: Height for classification
            important_features={'height'},
            
            # Helpful: Additional validation
            helpful_features={'planarity', 'nir'},
            
            # Optional
            optional_features={'normal_z', 'intensity'},
            
            thresholds={
                'ndvi': (0.25, None),         # Vegetation signature
                'curvature': (0.15, None),    # Complex surface
                'planarity': (None, 0.70),    # Not planar
                'nir': (0.25, None),          # High NIR
                'height': (0.0, None)         # Any height
            },
            
            base_confidence=0.75
        )
        
        # ============================================================
        # GROUND CLASSIFICATION
        # ============================================================
        rules['ground'] = ClassificationRule(
            name='ground',
            asprs_class=self.ASPRS_GROUND,
            
            # Critical: Need height
            critical_features={'height'},
            
            # Important: Planarity
            important_features={'planarity'},
            
            # Helpful: Additional validation
            helpful_features={'normal_z', 'curvature', 'ndvi'},
            
            # Optional
            optional_features={'intensity'},
            
            thresholds={
                'height': (None, 0.5),        # Near ground
                'planarity': (0.60, None),    # Relatively flat
                'normal_z': (0.80, None),     # Mostly horizontal
                'curvature': (None, 0.15),    # Not too rough
                'ndvi': (0.10, 0.40)          # Some vegetation OK
            },
            
            base_confidence=0.70
        )
        
        return rules
    
    def set_artifact_features(self, artifact_features: Set[str]):
        """
        Set features that have artifacts and should not be used.
        
        Args:
            artifact_features: Set of feature names with artifacts
        """
        self.artifact_features = artifact_features
        if artifact_features:
            logger.info(f"Adaptive classifier will avoid features with artifacts: {artifact_features}")
    
    def get_available_features(self, features: Dict[str, np.ndarray]) -> Set[str]:
        """
        Get set of available (non-artifact) features.
        
        Args:
            features: Dictionary of feature arrays
        
        Returns:
            Set of available feature names
        """
        available = set()
        
        for name, data in features.items():
            if data is None:
                continue
            
            # Skip if marked as artifact
            if name in self.artifact_features:
                continue
            
            # Check if feature is mostly valid (not all NaN)
            if isinstance(data, np.ndarray):
                valid_ratio = np.sum(np.isfinite(data)) / len(data)
                if valid_ratio > 0.5:  # At least 50% valid
                    available.add(name)
        
        return available
    
    def classify_point(
        self,
        ground_truth_type: str,
        features: Dict[str, float],
        available_features: Optional[Set[str]] = None
    ) -> Tuple[int, float, List[str]]:
        """
        Classify a single point adaptively.
        
        Args:
            ground_truth_type: Ground truth type ('building', 'road', etc.)
            features: Dictionary of feature values for this point
            available_features: Optional set of available features
        
        Returns:
            Tuple of (asprs_class, confidence, reasons)
        """
        if ground_truth_type not in self.rules:
            return self.ASPRS_UNCLASSIFIED, 0.0, [f"Unknown type: {ground_truth_type}"]
        
        rule = self.rules[ground_truth_type]
        
        # Determine available features
        if available_features is None:
            available_features = set(features.keys()) - self.artifact_features
        
        # Check if we can classify
        if not rule.can_classify(available_features):
            missing = rule.critical_features - available_features
            return self.ASPRS_UNCLASSIFIED, 0.0, [
                f"Missing critical features: {missing}"
            ]
        
        # Get usable features
        usable_features = rule.get_usable_features(available_features)
        
        # Check thresholds
        matches = []
        failures = []
        
        for feature_name in usable_features:
            if feature_name not in features:
                continue
            
            if feature_name not in rule.thresholds:
                continue
            
            value = features[feature_name]
            min_val, max_val = rule.thresholds[feature_name]
            
            # Check minimum
            if min_val is not None:
                if value >= min_val:
                    matches.append(f"{feature_name} >= {min_val}")
                else:
                    failures.append(f"{feature_name} < {min_val} (actual: {value:.2f})")
            
            # Check maximum
            if max_val is not None:
                if value <= max_val:
                    matches.append(f"{feature_name} <= {max_val}")
                else:
                    failures.append(f"{feature_name} > {max_val} (actual: {value:.2f})")
        
        # Calculate confidence
        base_confidence = rule.get_confidence(available_features)
        
        # Adjust based on match ratio
        if matches:
            match_ratio = len(matches) / (len(matches) + len(failures))
            confidence = base_confidence * match_ratio
        else:
            confidence = 0.0
        
        # Determine classification
        reasons = []
        if confidence >= 0.5:
            asprs_class = rule.asprs_class
            
            # For vegetation, determine height class
            if ground_truth_type == 'vegetation' and 'height' in features:
                height = features['height']
                if height < 0.5:
                    asprs_class = self.ASPRS_LOW_VEGETATION
                elif height < 2.0:
                    asprs_class = self.ASPRS_MEDIUM_VEGETATION
                else:
                    asprs_class = self.ASPRS_HIGH_VEGETATION
            
            reasons = [
                f"Matched {len(matches)} criteria using {len(usable_features)} features",
                f"Available features: {usable_features}",
                f"Missing features: {(rule.critical_features | rule.important_features) - available_features}"
            ]
        else:
            asprs_class = self.ASPRS_UNCLASSIFIED
            reasons = [
                f"Low confidence ({confidence:.2f})",
                f"Failures: {failures}",
                f"Available features: {usable_features}"
            ]
        
        return asprs_class, confidence, reasons
    
    def classify_batch(
        self,
        labels: np.ndarray,
        ground_truth_types: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classify multiple points adaptively.
        
        Args:
            labels: Current labels [N]
            ground_truth_types: Ground truth types [N]
            features: Dictionary of feature arrays
        
        Returns:
            Tuple of (validated_labels, confidences, valid_mask)
        """
        n_points = len(labels)
        validated_labels = labels.copy()
        confidences = np.zeros(n_points, dtype=np.float32)
        valid_mask = np.zeros(n_points, dtype=bool)
        
        # Get available features
        available_features = self.get_available_features(features)
        
        if not available_features:
            logger.warning("No valid features available for classification")
            return validated_labels, confidences, valid_mask
        
        logger.info(f"Adaptive classification with {len(available_features)} available features: {available_features}")
        
        # Check which rules can be used
        usable_rules = {}
        for type_name, rule in self.rules.items():
            if rule.can_classify(available_features):
                confidence = rule.get_confidence(available_features)
                usable_rules[type_name] = confidence
                logger.info(f"  {type_name}: can classify with confidence {confidence:.2f}")
            else:
                missing = rule.critical_features - available_features
                logger.warning(f"  {type_name}: cannot classify (missing critical: {missing})")
        
        # Classify each point
        for i in range(n_points):
            gt_type = ground_truth_types[i]
            
            if gt_type == 'none' or gt_type == '':
                continue
            
            # Extract features for this point
            point_features = {}
            for name, data in features.items():
                if name in available_features and data is not None:
                    point_features[name] = data[i]
            
            # Classify
            asprs_class, conf, reasons = self.classify_point(
                gt_type, point_features, available_features
            )
            
            validated_labels[i] = asprs_class
            confidences[i] = conf
            valid_mask[i] = (conf >= 0.5)
        
        n_validated = valid_mask.sum()
        n_rejected = (~valid_mask).sum()
        logger.info(f"Adaptive classification: {n_validated} validated, {n_rejected} rejected")
        
        return validated_labels, confidences, valid_mask
    
    def get_feature_importance_report(
        self,
        available_features: Set[str]
    ) -> Dict[str, Dict]:
        """
        Generate report on feature importance and availability.
        
        Args:
            available_features: Set of available feature names
        
        Returns:
            Dictionary with importance analysis per class
        """
        report = {}
        
        for type_name, rule in self.rules.items():
            # Check what's available
            critical_available = rule.critical_features & available_features
            critical_missing = rule.critical_features - available_features
            important_available = rule.important_features & available_features
            important_missing = rule.important_features - available_features
            helpful_available = rule.helpful_features & available_features
            helpful_missing = rule.helpful_features - available_features
            
            report[type_name] = {
                'can_classify': rule.can_classify(available_features),
                'confidence': rule.get_confidence(available_features),
                'critical': {
                    'available': list(critical_available),
                    'missing': list(critical_missing)
                },
                'important': {
                    'available': list(important_available),
                    'missing': list(important_missing)
                },
                'helpful': {
                    'available': list(helpful_available),
                    'missing': list(helpful_missing)
                }
            }
        
        return report
