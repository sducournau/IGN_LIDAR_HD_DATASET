"""
Feature-based ground truth validation module.

This module provides validation of ground truth classifications using geometric 
and radiometric features. It implements feature signatures for different land 
cover types and provides confidence scoring.

Author: IGN LiDAR HD Classification Team
Date: October 19, 2025
Version: 5.0
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureSignature:
    """Feature signature for a land cover type."""
    
    # Geometric features
    curvature_min: Optional[float] = None
    curvature_max: Optional[float] = None
    planarity_min: Optional[float] = None
    planarity_max: Optional[float] = None
    verticality_min: Optional[float] = None
    verticality_max: Optional[float] = None
    normal_z_min: Optional[float] = None
    normal_z_max: Optional[float] = None
    height_min: Optional[float] = None
    height_max: Optional[float] = None
    
    # Radiometric features
    ndvi_min: Optional[float] = None
    ndvi_max: Optional[float] = None
    nir_min: Optional[float] = None
    nir_max: Optional[float] = None
    intensity_min: Optional[float] = None
    intensity_max: Optional[float] = None
    brightness_min: Optional[float] = None
    brightness_max: Optional[float] = None
    
    def matches(self, features: Dict[str, float], strict: bool = False) -> Tuple[bool, float]:
        """
        Check if features match this signature.
        
        Args:
            features: Dictionary of feature values
            strict: If True, all constraints must be satisfied
                   If False, calculates confidence score
        
        Returns:
            (matches, confidence) where confidence is in [0, 1]
        """
        checks = []
        
        # Geometric feature checks
        if self.curvature_min is not None and 'curvature' in features:
            checks.append(features['curvature'] >= self.curvature_min)
        if self.curvature_max is not None and 'curvature' in features:
            checks.append(features['curvature'] <= self.curvature_max)
        if self.planarity_min is not None and 'planarity' in features:
            checks.append(features['planarity'] >= self.planarity_min)
        if self.planarity_max is not None and 'planarity' in features:
            checks.append(features['planarity'] <= self.planarity_max)
        if self.verticality_min is not None and 'verticality' in features:
            checks.append(features['verticality'] >= self.verticality_min)
        if self.verticality_max is not None and 'verticality' in features:
            checks.append(features['verticality'] <= self.verticality_max)
        if self.normal_z_min is not None and 'normal_z' in features:
            checks.append(features['normal_z'] >= self.normal_z_min)
        if self.normal_z_max is not None and 'normal_z' in features:
            checks.append(features['normal_z'] <= self.normal_z_max)
        if self.height_min is not None and 'height' in features:
            checks.append(features['height'] >= self.height_min)
        if self.height_max is not None and 'height' in features:
            checks.append(features['height'] <= self.height_max)
        
        # Radiometric feature checks
        if self.ndvi_min is not None and 'ndvi' in features:
            checks.append(features['ndvi'] >= self.ndvi_min)
        if self.ndvi_max is not None and 'ndvi' in features:
            checks.append(features['ndvi'] <= self.ndvi_max)
        if self.nir_min is not None and 'nir' in features:
            checks.append(features['nir'] >= self.nir_min)
        if self.nir_max is not None and 'nir' in features:
            checks.append(features['nir'] <= self.nir_max)
        if self.intensity_min is not None and 'intensity' in features:
            checks.append(features['intensity'] >= self.intensity_min)
        if self.intensity_max is not None and 'intensity' in features:
            checks.append(features['intensity'] <= self.intensity_max)
        if self.brightness_min is not None and 'brightness' in features:
            checks.append(features['brightness'] >= self.brightness_min)
        if self.brightness_max is not None and 'brightness' in features:
            checks.append(features['brightness'] <= self.brightness_max)
        
        if not checks:
            return False, 0.0
        
        # Calculate confidence as percentage of checks passed
        confidence = sum(checks) / len(checks)
        matches = all(checks) if strict else confidence >= 0.7
        
        return matches, confidence


class FeatureValidator:
    """Validate classifications using multi-feature signatures."""
    
    # ASPRS class constants (matching asprs_classes.py)
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with feature signatures.
        
        Args:
            config: Configuration dictionary with signature definitions
        """
        self.config = config or {}
        
        # Load feature signatures from config or use defaults
        self.vegetation_signature = self._load_vegetation_signature()
        self.building_signature = self._load_building_signature()
        self.road_signature = self._load_road_signature()
        self.water_signature = self._load_water_signature()
        self.ground_signature = self._load_ground_signature()
        
        # Confidence thresholds
        self.min_confidence = self.config.get('min_validation_confidence', 0.6)
        self.strict_validation = self.config.get('strict_validation', False)
        
        logger.info(f"FeatureValidator initialized with min_confidence={self.min_confidence}")
    
    def _load_vegetation_signature(self) -> FeatureSignature:
        """Load vegetation feature signature from config or defaults."""
        cfg = self.config.get('vegetation_signature', {})
        return FeatureSignature(
            curvature_min=cfg.get('curvature_min', 0.15),
            planarity_max=cfg.get('planarity_max', 0.70),
            ndvi_min=cfg.get('ndvi_min', 0.20),
            nir_min=cfg.get('nir_min', 0.25),
            normal_z_max=cfg.get('normal_z_max', 0.85)  # Vegetation not perfectly horizontal
        )
    
    def _load_building_signature(self) -> FeatureSignature:
        """Load building feature signature from config or defaults."""
        cfg = self.config.get('building_signature', {})
        return FeatureSignature(
            curvature_max=cfg.get('curvature_max', 0.10),
            planarity_min=cfg.get('planarity_min', 0.70),
            ndvi_max=cfg.get('ndvi_max', 0.15),
            verticality_min=cfg.get('verticality_min', 0.60),
            height_min=cfg.get('height_min', 2.0)
        )
    
    def _load_road_signature(self) -> FeatureSignature:
        """Load road feature signature from config or defaults."""
        cfg = self.config.get('road_signature', {})
        return FeatureSignature(
            curvature_max=cfg.get('curvature_max', 0.05),
            planarity_min=cfg.get('planarity_min', 0.85),
            ndvi_max=cfg.get('ndvi_max', 0.15),
            normal_z_min=cfg.get('normal_z_min', 0.90),
            height_max=cfg.get('height_max', 2.0)
        )
    
    def _load_water_signature(self) -> FeatureSignature:
        """Load water feature signature from config or defaults."""
        cfg = self.config.get('water_signature', {})
        return FeatureSignature(
            curvature_max=cfg.get('curvature_max', 0.05),
            planarity_min=cfg.get('planarity_min', 0.90),
            ndvi_max=cfg.get('ndvi_max', 0.10),
            normal_z_min=cfg.get('normal_z_min', 0.95),
            nir_min=cfg.get('nir_min', 0.0),  # Water has low NIR
            nir_max=cfg.get('nir_max', 0.20)
        )
    
    def _load_ground_signature(self) -> FeatureSignature:
        """Load ground feature signature from config or defaults."""
        cfg = self.config.get('ground_signature', {})
        return FeatureSignature(
            curvature_max=cfg.get('curvature_max', 0.15),
            planarity_min=cfg.get('planarity_min', 0.60),
            ndvi_min=cfg.get('ndvi_min', 0.10),
            ndvi_max=cfg.get('ndvi_max', 0.40),
            normal_z_min=cfg.get('normal_z_min', 0.80),
            height_max=cfg.get('height_max', 0.5)
        )
    
    def validate_ground_truth(
        self,
        labels: np.ndarray,
        ground_truth_types: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate ground truth classifications using feature signatures.
        
        This method checks if ground truth labels are consistent with observed
        features. Points that fail validation are marked for reclassification.
        
        Args:
            labels: Current classification labels [N]
            ground_truth_types: Ground truth source types [N] (e.g., 'building', 'road')
            features: Dictionary of feature arrays
        
        Returns:
            Tuple of:
                - validated_labels: Updated labels [N]
                - confidence_scores: Confidence for each point [N]
                - validation_mask: Boolean mask of validated points [N]
        """
        n_points = len(labels)
        validated_labels = labels.copy()
        confidence_scores = np.zeros(n_points, dtype=np.float32)
        validation_mask = np.zeros(n_points, dtype=bool)
        
        # Extract features
        curvature = features.get('curvature')
        planarity = features.get('planarity')
        verticality = features.get('verticality')
        normals = features.get('normals')
        normal_z = normals[:, 2] if normals is not None else None
        height = features.get('height')
        ndvi = features.get('ndvi')
        nir = features.get('nir')
        intensity = features.get('intensity')
        brightness = features.get('brightness')
        
        # Validate each ground truth type
        unique_types = np.unique(ground_truth_types)
        
        for gt_type in unique_types:
            if gt_type == 'none' or gt_type == '':
                continue
            
            mask = ground_truth_types == gt_type
            n_type = mask.sum()
            
            if n_type == 0:
                continue
            
            logger.debug(f"Validating {n_type} points with ground truth type: {gt_type}")
            
            # Build feature dict for this type
            type_features = {}
            if curvature is not None:
                type_features['curvature'] = curvature[mask]
            if planarity is not None:
                type_features['planarity'] = planarity[mask]
            if verticality is not None:
                type_features['verticality'] = verticality[mask]
            if normal_z is not None:
                type_features['normal_z'] = normal_z[mask]
            if height is not None:
                type_features['height'] = height[mask]
            if ndvi is not None:
                type_features['ndvi'] = ndvi[mask]
            if nir is not None:
                type_features['nir'] = nir[mask]
            if intensity is not None:
                type_features['intensity'] = intensity[mask]
            if brightness is not None:
                type_features['brightness'] = brightness[mask]
            
            # Validate based on type
            if gt_type == 'building':
                new_labels, confidences, valid = self._validate_building(
                    labels[mask], type_features
                )
            elif gt_type == 'road':
                new_labels, confidences, valid = self._validate_road(
                    labels[mask], type_features
                )
            elif gt_type == 'water':
                new_labels, confidences, valid = self._validate_water(
                    labels[mask], type_features
                )
            elif gt_type == 'vegetation':
                new_labels, confidences, valid = self._validate_vegetation(
                    labels[mask], type_features
                )
            else:
                # Unknown type, accept as-is with low confidence
                new_labels = labels[mask]
                confidences = np.full(n_type, 0.5, dtype=np.float32)
                valid = np.ones(n_type, dtype=bool)
            
            # Update arrays
            validated_labels[mask] = new_labels
            confidence_scores[mask] = confidences
            validation_mask[mask] = valid
        
        n_validated = validation_mask.sum()
        n_rejected = (~validation_mask).sum()
        logger.info(f"Validation complete: {n_validated} validated, {n_rejected} rejected")
        
        return validated_labels, confidence_scores, validation_mask
    
    def _validate_building(
        self,
        labels: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate building ground truth with features."""
        n_points = len(labels)
        new_labels = labels.copy()
        confidences = np.zeros(n_points, dtype=np.float32)
        valid = np.zeros(n_points, dtype=bool)
        
        for i in range(n_points):
            # Build feature dict for this point
            point_features = {k: v[i] for k, v in features.items()}
            
            # Check if features match building signature
            matches, conf = self.building_signature.matches(point_features, self.strict_validation)
            
            if matches and conf >= self.min_confidence:
                # Features confirm building
                new_labels[i] = self.ASPRS_BUILDING
                confidences[i] = conf
                valid[i] = True
            else:
                # Features don't match - check if it's roof vegetation
                veg_matches, veg_conf = self.vegetation_signature.matches(
                    point_features, self.strict_validation
                )
                
                if veg_matches and veg_conf >= self.min_confidence:
                    # Roof vegetation detected
                    new_labels[i] = self._classify_vegetation_by_height(
                        point_features.get('height', 0.0)
                    )
                    confidences[i] = veg_conf
                    valid[i] = True
                    logger.debug(f"Roof vegetation detected: building → vegetation")
                else:
                    # Keep building label but with low confidence
                    new_labels[i] = self.ASPRS_BUILDING
                    confidences[i] = 0.5
                    valid[i] = False
        
        return new_labels, confidences, valid
    
    def _validate_road(
        self,
        labels: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate road ground truth with features."""
        n_points = len(labels)
        new_labels = labels.copy()
        confidences = np.zeros(n_points, dtype=np.float32)
        valid = np.zeros(n_points, dtype=bool)
        
        for i in range(n_points):
            point_features = {k: v[i] for k, v in features.items()}
            
            # Check road signature
            matches, conf = self.road_signature.matches(point_features, self.strict_validation)
            
            if matches and conf >= self.min_confidence:
                # Features confirm road
                new_labels[i] = self.ASPRS_ROAD
                confidences[i] = conf
                valid[i] = True
            else:
                # Check if it's vegetation (tree canopy over road)
                veg_matches, veg_conf = self.vegetation_signature.matches(
                    point_features, self.strict_validation
                )
                
                if veg_matches and veg_conf >= self.min_confidence:
                    # Tree canopy detected
                    new_labels[i] = self._classify_vegetation_by_height(
                        point_features.get('height', 0.0)
                    )
                    confidences[i] = veg_conf
                    valid[i] = True
                    logger.debug(f"Tree canopy over road detected: road → vegetation")
                else:
                    # Keep road label with low confidence
                    new_labels[i] = self.ASPRS_ROAD
                    confidences[i] = 0.5
                    valid[i] = False
        
        return new_labels, confidences, valid
    
    def _validate_water(
        self,
        labels: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate water ground truth with features."""
        n_points = len(labels)
        new_labels = labels.copy()
        confidences = np.zeros(n_points, dtype=np.float32)
        valid = np.zeros(n_points, dtype=bool)
        
        for i in range(n_points):
            point_features = {k: v[i] for k, v in features.items()}
            
            # Check water signature
            matches, conf = self.water_signature.matches(point_features, self.strict_validation)
            
            if matches and conf >= self.min_confidence:
                new_labels[i] = self.ASPRS_WATER
                confidences[i] = conf
                valid[i] = True
            else:
                # Water false positive - likely ground or vegetation
                new_labels[i] = self.ASPRS_GROUND
                confidences[i] = 0.4
                valid[i] = False
        
        return new_labels, confidences, valid
    
    def _validate_vegetation(
        self,
        labels: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate vegetation ground truth with features."""
        n_points = len(labels)
        new_labels = labels.copy()
        confidences = np.zeros(n_points, dtype=np.float32)
        valid = np.zeros(n_points, dtype=bool)
        
        for i in range(n_points):
            point_features = {k: v[i] for k, v in features.items()}
            
            # Check vegetation signature
            matches, conf = self.vegetation_signature.matches(
                point_features, self.strict_validation
            )
            
            if matches and conf >= self.min_confidence:
                # Classify by height
                new_labels[i] = self._classify_vegetation_by_height(
                    point_features.get('height', 0.0)
                )
                confidences[i] = conf
                valid[i] = True
            else:
                # Not vegetation - likely ground or building
                new_labels[i] = self.ASPRS_GROUND
                confidences[i] = 0.4
                valid[i] = False
        
        return new_labels, confidences, valid
    
    def _classify_vegetation_by_height(self, height: float) -> int:
        """Classify vegetation type based on height."""
        if height < 0.5:
            return self.ASPRS_LOW_VEGETATION
        elif height < 2.0:
            return self.ASPRS_MEDIUM_VEGETATION
        else:
            return self.ASPRS_HIGH_VEGETATION
    
    def check_vegetation_signature(
        self,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check if features match vegetation signature.
        
        Args:
            features: Dictionary of feature arrays
        
        Returns:
            (is_vegetation, confidence) boolean array and confidence scores
        """
        n_points = len(next(iter(features.values())))
        is_vegetation = np.zeros(n_points, dtype=bool)
        confidence = np.zeros(n_points, dtype=np.float32)
        
        for i in range(n_points):
            point_features = {k: v[i] for k, v in features.items()}
            matches, conf = self.vegetation_signature.matches(
                point_features, self.strict_validation
            )
            is_vegetation[i] = matches
            confidence[i] = conf
        
        return is_vegetation, confidence
    
    def check_building_signature(
        self,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Check if features match building signature."""
        n_points = len(next(iter(features.values())))
        is_building = np.zeros(n_points, dtype=bool)
        confidence = np.zeros(n_points, dtype=np.float32)
        
        for i in range(n_points):
            point_features = {k: v[i] for k, v in features.items()}
            matches, conf = self.building_signature.matches(
                point_features, self.strict_validation
            )
            is_building[i] = matches
            confidence[i] = conf
        
        return is_building, confidence
    
    def check_road_signature(
        self,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Check if features match road signature."""
        n_points = len(next(iter(features.values())))
        is_road = np.zeros(n_points, dtype=bool)
        confidence = np.zeros(n_points, dtype=np.float32)
        
        for i in range(n_points):
            point_features = {k: v[i] for k, v in features.items()}
            matches, conf = self.road_signature.matches(
                point_features, self.strict_validation
            )
            is_road[i] = matches
            confidence[i] = conf
        
        return is_road, confidence
    
    def filter_ground_truth_false_positives(
        self,
        labels: np.ndarray,
        ground_truth_mask: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth_types: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter false positives from ground truth classification.
        
        This method removes ground truth labels that are inconsistent with
        observed features, preventing propagation of errors.
        
        Args:
            labels: Classification labels [N]
            ground_truth_mask: Boolean mask of ground truth points [N]
            features: Dictionary of feature arrays
            ground_truth_types: Optional array of ground truth types [N]
        
        Returns:
            (filtered_labels, filtered_mask) with false positives removed
        """
        if ground_truth_types is None:
            ground_truth_types = np.full(len(labels), 'unknown', dtype=object)
        
        # Validate only ground truth points
        gt_indices = np.where(ground_truth_mask)[0]
        
        if len(gt_indices) == 0:
            return labels.copy(), ground_truth_mask.copy()
        
        # Extract features for ground truth points
        gt_features = {k: v[gt_indices] for k, v in features.items()}
        gt_labels = labels[gt_indices]
        gt_types = ground_truth_types[gt_indices]
        
        # Validate
        validated_labels, confidences, valid_mask = self.validate_ground_truth(
            gt_labels, gt_types, gt_features
        )
        
        # Build filtered output
        filtered_labels = labels.copy()
        filtered_mask = ground_truth_mask.copy()
        
        # Update labels for validated points
        filtered_labels[gt_indices] = validated_labels
        
        # Remove invalid points from ground truth mask
        invalid_indices = gt_indices[~valid_mask]
        filtered_mask[invalid_indices] = False
        
        n_removed = len(invalid_indices)
        if n_removed > 0:
            logger.info(f"Filtered {n_removed} false positives from ground truth")
        
        return filtered_labels, filtered_mask
