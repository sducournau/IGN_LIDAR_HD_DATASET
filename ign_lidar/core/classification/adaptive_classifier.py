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
from typing import Dict, Tuple, Optional, List, Set, Any
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


# ============================================================================
# Comprehensive Adaptive Classification for All Feature Types
# ============================================================================

@dataclass
class AdaptiveReclassificationConfig:
    """Configuration for adaptive reclassification with ground truth as guidance."""
    
    # General confidence thresholds
    MIN_CONFIDENCE = 0.5           # Minimum confidence to classify
    HIGH_CONFIDENCE = 0.7          # High confidence classification
    GT_WEIGHT = 0.20               # Weight for ground truth proximity (guidance only)
    GEOMETRY_WEIGHT = 0.80         # Weight for geometric features (primary)
    
    # Fuzzy boundary settings
    BUFFER_DISTANCE = 3.0          # Distance for fuzzy boundary (meters)
    DECAY_RATE = 0.5               # Confidence decay outside GT polygon
    
    # Water parameters
    WATER_HEIGHT_MAX = 0.5
    WATER_PLANARITY_MIN = 0.90
    WATER_CURVATURE_MAX = 0.02
    WATER_NORMAL_Z_MIN = 0.92
    WATER_NDVI_MAX = 0.15
    WATER_ROUGHNESS_MAX = 0.02
    
    # Road parameters
    ROAD_HEIGHT_MAX = 1.5
    ROAD_HEIGHT_MIN = -0.5
    ROAD_PLANARITY_MIN = 0.75
    ROAD_CURVATURE_MAX = 0.05
    ROAD_NORMAL_Z_MIN = 0.85
    ROAD_NDVI_MAX = 0.20
    ROAD_ROUGHNESS_MAX = 0.05
    ROAD_VERTICALITY_MAX = 0.30
    
    # Vegetation parameters
    VEG_NDVI_MIN = 0.25
    VEG_CURVATURE_MIN = 0.02
    VEG_PLANARITY_MAX = 0.50
    VEG_ROUGHNESS_MIN = 0.03
    VEG_LOW_HEIGHT_MAX = 0.5
    VEG_HIGH_HEIGHT_MIN = 2.0
    
    # Bridge parameters
    BRIDGE_HEIGHT_MIN = 3.0
    BRIDGE_HEIGHT_MAX = 50.0


class ComprehensiveAdaptiveClassifier:
    """
    Comprehensive adaptive classifier that treats ground truth as guidance
    for ALL feature types (buildings, vegetation, roads, water, etc.).
    
    Key improvements:
    1. Ground truth is guidance, not absolute truth
    2. Point cloud features drive classification
    3. Fuzzy boundaries with confidence-based voting
    4. Handles all feature types consistently
    """
    
    # ASPRS class codes
    UNCLASSIFIED = 1
    GROUND = 2
    LOW_VEG = 3
    MEDIUM_VEG = 4
    HIGH_VEG = 5
    BUILDING = 6
    WATER = 9
    ROAD = 11
    BRIDGE = 17
    
    def __init__(self, config: Optional[AdaptiveReclassificationConfig] = None):
        """Initialize comprehensive adaptive classifier."""
        self.config = config or AdaptiveReclassificationConfig()
        logger.info("Initialized ComprehensiveAdaptiveClassifier")
        logger.info("  → Ground truth = guidance (not absolute truth)")
        logger.info("  → Point cloud features = primary signal")
    
    def compute_gt_proximity_confidence(
        self,
        points: np.ndarray,
        gt_polygons: Optional[Any],
        buffer_distance: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute soft confidence based on proximity to ground truth polygons.
        Uses fuzzy boundaries with exponential decay.
        """
        if gt_polygons is None:
            return np.zeros(len(points))
        
        buffer_dist = buffer_distance or self.config.BUFFER_DISTANCE
        
        try:
            from shapely.geometry import Point as ShapelyPoint
            from shapely.strtree import STRtree
            import geopandas as gpd
            
            # Handle GeoDataFrame or list of geometries
            if isinstance(gt_polygons, gpd.GeoDataFrame):
                if len(gt_polygons) == 0:
                    return np.zeros(len(points))
                geoms = list(gt_polygons.geometry)
            else:
                geoms = gt_polygons
            
            if not geoms:
                return np.zeros(len(points))
            
            tree = STRtree(geoms)
            confidence = np.zeros(len(points), dtype=np.float32)
            
            for i, pt in enumerate(points):
                shapely_pt = ShapelyPoint(pt[0], pt[1])
                nearest = tree.nearest(shapely_pt)
                
                if nearest is not None:
                    if nearest.contains(shapely_pt):
                        confidence[i] = 1.0
                    else:
                        dist = shapely_pt.distance(nearest)
                        if dist < buffer_dist:
                            # Exponential decay
                            confidence[i] = np.exp(-dist / self.config.DECAY_RATE)
            
            return confidence
            
        except ImportError:
            logger.warning("Shapely not available - skipping GT proximity")
            return np.zeros(len(points))
    
    def refine_water_adaptive(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        gt_water: Optional[Any] = None,
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        ndvi: Optional[np.ndarray] = None,
        roughness: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Adaptive water classification using confidence-based voting.
        Ground truth provides spatial hints, geometry validates.
        """
        stats = {'water_validated': 0, 'water_added': 0, 'water_rejected': 0}
        refined = labels.copy()
        n_points = len(points)
        
        # Build multi-feature confidence score
        water_conf = np.zeros(n_points, dtype=np.float32)
        total_weight = 0.0
        
        # 1. Planarity - Water is extremely flat (weight: 0.30)
        if planarity is not None:
            plan_score = np.clip(
                (planarity - self.config.WATER_PLANARITY_MIN) / 
                (1.0 - self.config.WATER_PLANARITY_MIN), 0, 1
            )
            water_conf += plan_score * 0.30
            total_weight += 0.30
        
        # 2. Height - Near ground (weight: 0.25)
        if height is not None:
            height_score = (np.abs(height) <= self.config.WATER_HEIGHT_MAX).astype(np.float32)
            water_conf += height_score * 0.25
            total_weight += 0.25
        
        # 3. Normals - Horizontal (weight: 0.20)
        if normals is not None:
            normal_score = np.clip(
                (np.abs(normals[:, 2]) - self.config.WATER_NORMAL_Z_MIN) /
                (1.0 - self.config.WATER_NORMAL_Z_MIN), 0, 1
            )
            water_conf += normal_score * 0.20
            total_weight += 0.20
        
        # 4. Curvature - Very low (weight: 0.10)
        if curvature is not None:
            curv_score = 1.0 - np.clip(curvature / self.config.WATER_CURVATURE_MAX, 0, 1)
            water_conf += curv_score * 0.10
            total_weight += 0.10
        
        # 5. NDVI - Low (weight: 0.05)
        if ndvi is not None:
            ndvi_score = 1.0 - np.clip(ndvi / self.config.WATER_NDVI_MAX, 0, 1)
            water_conf += ndvi_score * 0.05
            total_weight += 0.05
        
        # 6. Roughness - Smooth (weight: 0.05)
        if roughness is not None:
            rough_score = 1.0 - np.clip(roughness / self.config.WATER_ROUGHNESS_MAX, 0, 1)
            water_conf += rough_score * 0.05
            total_weight += 0.05
        
        # 7. Ground truth - Guidance only (weight: 0.05)
        if gt_water is not None:
            gt_conf = self.compute_gt_proximity_confidence(points, gt_water)
            water_conf += gt_conf * 0.05
            total_weight += 0.05
        
        # Normalize
        if total_weight > 0:
            water_conf /= total_weight
        
        # Apply high threshold for water
        is_water = water_conf >= self.config.HIGH_CONFIDENCE
        currently_water = labels == self.WATER
        
        refined[currently_water & is_water] = self.WATER
        stats['water_validated'] = np.sum(currently_water & is_water)
        
        refined[~currently_water & is_water] = self.WATER
        stats['water_added'] = np.sum(~currently_water & is_water)
        
        refined[currently_water & ~is_water] = self.GROUND
        stats['water_rejected'] = np.sum(currently_water & ~is_water)
        
        if sum(stats.values()) > 0:
            logger.info(f"  Water: ✓{stats['water_validated']:,} +{stats['water_added']:,} ✗{stats['water_rejected']:,}")
        
        return refined, stats
    
    def refine_roads_adaptive(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        gt_roads: Optional[Any] = None,
        height: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        ndvi: Optional[np.ndarray] = None,
        roughness: Optional[np.ndarray] = None,
        verticality: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Adaptive road classification with tree canopy detection.
        """
        stats = {
            'road_validated': 0, 'road_added': 0, 'road_rejected': 0,
            'tree_canopy': 0, 'bridge': 0
        }
        refined = labels.copy()
        n_points = len(points)
        
        # Build road confidence
        road_conf = np.zeros(n_points, dtype=np.float32)
        total_weight = 0.0
        
        # 1. Planarity (weight: 0.25)
        if planarity is not None:
            plan_score = np.clip(
                (planarity - self.config.ROAD_PLANARITY_MIN) /
                (1.0 - self.config.ROAD_PLANARITY_MIN), 0, 1
            )
            road_conf += plan_score * 0.25
            total_weight += 0.25
        
        # 2. Height - Near ground (weight: 0.20)
        if height is not None:
            height_score = (
                (height >= self.config.ROAD_HEIGHT_MIN) &
                (height <= self.config.ROAD_HEIGHT_MAX)
            ).astype(np.float32)
            road_conf += height_score * 0.20
            total_weight += 0.20
        
        # 3. Normals - Horizontal (weight: 0.15)
        if normals is not None:
            normal_score = np.clip(
                (np.abs(normals[:, 2]) - self.config.ROAD_NORMAL_Z_MIN) /
                (1.0 - self.config.ROAD_NORMAL_Z_MIN), 0, 1
            )
            road_conf += normal_score * 0.15
            total_weight += 0.15
        
        # 4. NDVI - Low (weight: 0.15)
        if ndvi is not None:
            ndvi_score = 1.0 - np.clip(ndvi / self.config.ROAD_NDVI_MAX, 0, 1)
            road_conf += ndvi_score * 0.15
            total_weight += 0.15
        
        # 5. Verticality - Not vertical (weight: 0.10)
        if verticality is not None:
            vert_score = 1.0 - np.clip(verticality / self.config.ROAD_VERTICALITY_MAX, 0, 1)
            road_conf += vert_score * 0.10
            total_weight += 0.10
        
        # 6. Curvature - Smooth (weight: 0.05)
        if curvature is not None:
            curv_score = 1.0 - np.clip(curvature / self.config.ROAD_CURVATURE_MAX, 0, 1)
            road_conf += curv_score * 0.05
            total_weight += 0.05
        
        # 7. Roughness - Smooth (weight: 0.05)
        if roughness is not None:
            rough_score = 1.0 - np.clip(roughness / self.config.ROAD_ROUGHNESS_MAX, 0, 1)
            road_conf += rough_score * 0.05
            total_weight += 0.05
        
        # 8. Ground truth - Guidance (weight: 0.05)
        if gt_roads is not None:
            gt_conf = self.compute_gt_proximity_confidence(points, gt_roads)
            road_conf += gt_conf * 0.05
            total_weight += 0.05
        
        # Normalize
        if total_weight > 0:
            road_conf /= total_weight
        
        # Detect special cases
        is_road = road_conf >= self.config.MIN_CONFIDENCE
        
        # Tree canopy over road: high NDVI + elevated
        if ndvi is not None and height is not None:
            tree_canopy = (
                (ndvi > 0.3) &
                (height > self.config.ROAD_HEIGHT_MAX) &
                (road_conf > 0.3)
            )
            refined[tree_canopy] = self.HIGH_VEG
            stats['tree_canopy'] = np.sum(tree_canopy)
            is_road[tree_canopy] = False
        
        # Bridge: elevated but road-like
        if height is not None:
            bridge_mask = (
                is_road &
                (height >= self.config.BRIDGE_HEIGHT_MIN) &
                (height <= self.config.BRIDGE_HEIGHT_MAX)
            )
            refined[bridge_mask] = self.BRIDGE
            stats['bridge'] = np.sum(bridge_mask)
            is_road[bridge_mask] = False
        
        # Classify roads
        currently_road = labels == self.ROAD
        refined[currently_road & is_road] = self.ROAD
        stats['road_validated'] = np.sum(currently_road & is_road)
        
        refined[~currently_road & is_road] = self.ROAD
        stats['road_added'] = np.sum(~currently_road & is_road)
        
        refined[currently_road & ~is_road] = self.GROUND
        stats['road_rejected'] = np.sum(currently_road & ~is_road)
        
        if sum(stats.values()) > 0:
            logger.info(f"  Roads: ✓{stats['road_validated']:,} +{stats['road_added']:,} ✗{stats['road_rejected']:,}")
            if stats['tree_canopy'] > 0:
                logger.info(f"    🌳 Tree canopy: {stats['tree_canopy']:,}")
            if stats['bridge'] > 0:
                logger.info(f"    🌉 Bridges: {stats['bridge']:,}")
        
        return refined, stats
    
    def refine_vegetation_adaptive(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        gt_vegetation: Optional[Any] = None,
        ndvi: Optional[np.ndarray] = None,
        height: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        planarity: Optional[np.ndarray] = None,
        roughness: Optional[np.ndarray] = None,
        sphericity: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Adaptive vegetation classification using multi-feature confidence.
        """
        stats = {'veg_total': 0, 'low_veg': 0, 'medium_veg': 0, 'high_veg': 0, 'veg_rejected': 0}
        refined = labels.copy()
        n_points = len(points)
        
        # Build vegetation confidence
        veg_conf = np.zeros(n_points, dtype=np.float32)
        total_weight = 0.0
        
        # 1. NDVI - Primary (weight: 0.40)
        if ndvi is not None:
            ndvi_score = np.clip((ndvi - self.config.VEG_NDVI_MIN) / (0.8 - self.config.VEG_NDVI_MIN), 0, 1)
            veg_conf += ndvi_score * 0.40
            total_weight += 0.40
        
        # 2. Curvature - Complex surfaces (weight: 0.20)
        if curvature is not None:
            curv_score = np.clip((curvature - self.config.VEG_CURVATURE_MIN) / 0.10, 0, 1)
            veg_conf += curv_score * 0.20
            total_weight += 0.20
        
        # 3. Planarity - Inverse (weight: 0.15)
        if planarity is not None:
            plan_score = 1.0 - np.clip(planarity / self.config.VEG_PLANARITY_MAX, 0, 1)
            veg_conf += plan_score * 0.15
            total_weight += 0.15
        
        # 4. Roughness (weight: 0.10)
        if roughness is not None:
            rough_score = np.clip((roughness - self.config.VEG_ROUGHNESS_MIN) / 0.10, 0, 1)
            veg_conf += rough_score * 0.10
            total_weight += 0.10
        
        # 5. Sphericity (weight: 0.10)
        if sphericity is not None:
            veg_conf += np.clip(sphericity, 0, 1) * 0.10
            total_weight += 0.10
        
        # 6. Ground truth - Minimal (weight: 0.05)
        if gt_vegetation is not None:
            gt_conf = self.compute_gt_proximity_confidence(points, gt_vegetation)
            veg_conf += gt_conf * 0.05
            total_weight += 0.05
        
        # Normalize
        if total_weight > 0:
            veg_conf /= total_weight
        
        # Classify vegetation
        is_veg = veg_conf >= self.config.MIN_CONFIDENCE
        
        # Classify by height
        if height is not None:
            low_mask = is_veg & (height <= self.config.VEG_LOW_HEIGHT_MAX)
            refined[low_mask] = self.LOW_VEG
            stats['low_veg'] = np.sum(low_mask)
            
            high_mask = is_veg & (height >= self.config.VEG_HIGH_HEIGHT_MIN)
            refined[high_mask] = self.HIGH_VEG
            stats['high_veg'] = np.sum(high_mask)
            
            medium_mask = is_veg & ~low_mask & ~high_mask
            refined[medium_mask] = self.MEDIUM_VEG
            stats['medium_veg'] = np.sum(medium_mask)
        else:
            refined[is_veg] = self.MEDIUM_VEG
            stats['medium_veg'] = np.sum(is_veg)
        
        stats['veg_total'] = np.sum(is_veg)
        
        # Reject low-confidence vegetation
        currently_veg = np.isin(labels, [self.LOW_VEG, self.MEDIUM_VEG, self.HIGH_VEG])
        rejected = currently_veg & ~is_veg
        refined[rejected] = self.UNCLASSIFIED
        stats['veg_rejected'] = np.sum(rejected)
        
        if stats['veg_total'] > 0:
            logger.info(f"  Vegetation: {stats['veg_total']:,} (L:{stats['low_veg']:,} M:{stats['medium_veg']:,} H:{stats['high_veg']:,})")
            if stats['veg_rejected'] > 0:
                logger.info(f"    ✗ Rejected: {stats['veg_rejected']:,}")
        
        return refined, stats
    
    def classify_all_adaptive(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        ground_truth_data: Optional[Dict] = None,
        features: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Comprehensive adaptive classification for all feature types.
        
        Order: Water → Buildings → Roads → Vegetation
        """
        refined = labels.copy()
        all_stats = {}
        
        features = features or {}
        ground_truth_data = ground_truth_data or {}
        
        logger.info("\n=== Adaptive Classification (All Features) ===")
        
        # 1. Water
        refined, water_stats = self.refine_water_adaptive(
            points, refined,
            gt_water=ground_truth_data.get('water'),
            height=features.get('height'),
            planarity=features.get('planarity'),
            curvature=features.get('curvature'),
            normals=features.get('normals'),
            ndvi=features.get('ndvi'),
            roughness=features.get('roughness')
        )
        all_stats.update(water_stats)
        
        # 2. Roads
        refined, road_stats = self.refine_roads_adaptive(
            points, refined,
            gt_roads=ground_truth_data.get('roads'),
            height=features.get('height'),
            planarity=features.get('planarity'),
            curvature=features.get('curvature'),
            normals=features.get('normals'),
            ndvi=features.get('ndvi'),
            roughness=features.get('roughness'),
            verticality=features.get('verticality')
        )
        all_stats.update(road_stats)
        
        # 3. Vegetation
        refined, veg_stats = self.refine_vegetation_adaptive(
            points, refined,
            gt_vegetation=ground_truth_data.get('vegetation'),
            ndvi=features.get('ndvi'),
            height=features.get('height'),
            curvature=features.get('curvature'),
            planarity=features.get('planarity'),
            roughness=features.get('roughness'),
            sphericity=features.get('sphericity')
        )
        all_stats.update(veg_stats)
        
        logger.info("=== Adaptive Classification Complete ===\n")
        
        return refined, all_stats


# Convenience function
def refine_all_classifications_adaptive(
    points: np.ndarray,
    labels: np.ndarray,
    ground_truth_data: Optional[Dict] = None,
    features: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[AdaptiveReclassificationConfig] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply adaptive classification to all feature types.
    
    Treats ground truth as guidance for buildings, vegetation, roads, water, etc.
    Point cloud features are the primary classification signal.
    """
    classifier = ComprehensiveAdaptiveClassifier(config)
    return classifier.classify_all_adaptive(points, labels, ground_truth_data, features)
