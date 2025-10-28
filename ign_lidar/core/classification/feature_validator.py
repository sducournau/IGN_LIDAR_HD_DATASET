"""
Feature-based ground truth validation module.

This module provides validation of ground truth classifications using geometric
and radiometric features. It implements feature signatures for different land
cover types and provides confidence scoring.

Author: IGN LiDAR HD Classification Team
Date: October 19, 2025
Version: 5.0
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .constants import ASPRSClass

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

    def matches(
        self, features: Dict[str, float], strict: bool = False
    ) -> Tuple[bool, float]:
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
        if self.curvature_min is not None and "curvature" in features:
            checks.append(features["curvature"] >= self.curvature_min)
        if self.curvature_max is not None and "curvature" in features:
            checks.append(features["curvature"] <= self.curvature_max)
        if self.planarity_min is not None and "planarity" in features:
            checks.append(features["planarity"] >= self.planarity_min)
        if self.planarity_max is not None and "planarity" in features:
            checks.append(features["planarity"] <= self.planarity_max)
        if self.verticality_min is not None and "verticality" in features:
            checks.append(features["verticality"] >= self.verticality_min)
        if self.verticality_max is not None and "verticality" in features:
            checks.append(features["verticality"] <= self.verticality_max)
        if self.normal_z_min is not None and "normal_z" in features:
            checks.append(features["normal_z"] >= self.normal_z_min)
        if self.normal_z_max is not None and "normal_z" in features:
            checks.append(features["normal_z"] <= self.normal_z_max)
        if self.height_min is not None and "height" in features:
            checks.append(features["height"] >= self.height_min)
        if self.height_max is not None and "height" in features:
            checks.append(features["height"] <= self.height_max)

        # Radiometric feature checks
        if self.ndvi_min is not None and "ndvi" in features:
            checks.append(features["ndvi"] >= self.ndvi_min)
        if self.ndvi_max is not None and "ndvi" in features:
            checks.append(features["ndvi"] <= self.ndvi_max)
        if self.nir_min is not None and "nir" in features:
            checks.append(features["nir"] >= self.nir_min)
        if self.nir_max is not None and "nir" in features:
            checks.append(features["nir"] <= self.nir_max)
        if self.intensity_min is not None and "intensity" in features:
            checks.append(features["intensity"] >= self.intensity_min)
        if self.intensity_max is not None and "intensity" in features:
            checks.append(features["intensity"] <= self.intensity_max)
        if self.brightness_min is not None and "brightness" in features:
            checks.append(features["brightness"] >= self.brightness_min)
        if self.brightness_max is not None and "brightness" in features:
            checks.append(features["brightness"] <= self.brightness_max)

        if not checks:
            return False, 0.0

        # Calculate confidence as percentage of checks passed
        confidence = sum(checks) / len(checks)
        matches = all(checks) if strict else confidence >= 0.7

        return matches, confidence


class FeatureValidator:
    """Validate classifications using multi-feature signatures."""

    # ASPRS class constants (matching asprs_classes.py)
    # Use ASPRSClass from constants module

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
        self.min_confidence = self.config.get("min_validation_confidence", 0.6)
        self.strict_validation = self.config.get("strict_validation", False)

        logger.info(
            f"FeatureValidator initialized with min_confidence={self.min_confidence}"
        )

    def _load_vegetation_signature(self) -> FeatureSignature:
        """Load vegetation feature signature from config or defaults."""
        cfg = self.config.get("vegetation_signature", {})
        return FeatureSignature(
            curvature_min=cfg.get("curvature_min", 0.15),
            planarity_max=cfg.get("planarity_max", 0.70),
            ndvi_min=cfg.get("ndvi_min", 0.20),
            nir_min=cfg.get("nir_min", 0.25),
            normal_z_max=cfg.get(
                "normal_z_max", 0.85
            ),  # Vegetation not perfectly horizontal
        )

    def _load_building_signature(self) -> FeatureSignature:
        """Load building feature signature from config or defaults."""
        cfg = self.config.get("building_signature", {})
        return FeatureSignature(
            curvature_max=cfg.get("curvature_max", 0.10),
            planarity_min=cfg.get("planarity_min", 0.70),
            ndvi_max=cfg.get("ndvi_max", 0.15),
            verticality_min=cfg.get("verticality_min", 0.60),
            height_min=cfg.get("height_min", 2.0),
        )

    def _load_road_signature(self) -> FeatureSignature:
        """Load road feature signature from config or defaults."""
        cfg = self.config.get("road_signature", {})
        return FeatureSignature(
            curvature_max=cfg.get("curvature_max", 0.05),
            planarity_min=cfg.get("planarity_min", 0.85),
            ndvi_max=cfg.get("ndvi_max", 0.15),
            normal_z_min=cfg.get("normal_z_min", 0.90),
            height_max=cfg.get("height_max", 2.0),
        )

    def _load_water_signature(self) -> FeatureSignature:
        """Load water feature signature from config or defaults."""
        cfg = self.config.get("water_signature", {})
        return FeatureSignature(
            curvature_max=cfg.get("curvature_max", 0.05),
            planarity_min=cfg.get("planarity_min", 0.90),
            ndvi_max=cfg.get("ndvi_max", 0.10),
            normal_z_min=cfg.get("normal_z_min", 0.95),
            nir_min=cfg.get("nir_min", 0.0),  # Water has low NIR
            nir_max=cfg.get("nir_max", 0.20),
        )

    def _load_ground_signature(self) -> FeatureSignature:
        """Load ground feature signature from config or defaults."""
        cfg = self.config.get("ground_signature", {})
        return FeatureSignature(
            curvature_max=cfg.get("curvature_max", 0.15),
            planarity_min=cfg.get("planarity_min", 0.60),
            ndvi_min=cfg.get("ndvi_min", 0.10),
            ndvi_max=cfg.get("ndvi_max", 0.40),
            normal_z_min=cfg.get("normal_z_min", 0.80),
            height_max=cfg.get("height_max", 0.5),
        )

    def validate_ground_truth(
        self,
        labels: np.ndarray,
        ground_truth_types: np.ndarray,
        features: Dict[str, np.ndarray],
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
        curvature = features.get("curvature")
        planarity = features.get("planarity")
        verticality = features.get("verticality")
        normals = features.get("normals")
        normal_z = normals[:, 2] if normals is not None else None
        height = features.get("height")
        ndvi = features.get("ndvi")
        nir = features.get("nir")
        intensity = features.get("intensity")
        brightness = features.get("brightness")

        # Validate each ground truth type
        unique_types = np.unique(ground_truth_types)

        for gt_type in unique_types:
            if gt_type == "none" or gt_type == "":
                continue

            mask = ground_truth_types == gt_type
            n_type = mask.sum()

            if n_type == 0:
                continue

            logger.debug(
                f"Validating {n_type} points with ground truth type: {gt_type}"
            )

            # Build feature dict for this type
            type_features = {}
            if curvature is not None:
                type_features["curvature"] = curvature[mask]
            if planarity is not None:
                type_features["planarity"] = planarity[mask]
            if verticality is not None:
                type_features["verticality"] = verticality[mask]
            if normal_z is not None:
                type_features["normal_z"] = normal_z[mask]
            if height is not None:
                type_features["height"] = height[mask]
            if ndvi is not None:
                type_features["ndvi"] = ndvi[mask]
            if nir is not None:
                type_features["nir"] = nir[mask]
            if intensity is not None:
                type_features["intensity"] = intensity[mask]
            if brightness is not None:
                type_features["brightness"] = brightness[mask]

            # Validate based on type
            if gt_type == "building":
                new_labels, confidences, valid = self._validate_building(
                    labels[mask], type_features
                )
            elif gt_type == "road":
                new_labels, confidences, valid = self._validate_road(
                    labels[mask], type_features
                )
            elif gt_type == "water":
                new_labels, confidences, valid = self._validate_water(
                    labels[mask], type_features
                )
            elif gt_type == "vegetation":
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
        logger.info(
            f"Validation complete: {n_validated} validated, {n_rejected} rejected"
        )

        return validated_labels, confidence_scores, validation_mask

    def _validate_building(
        self, labels: np.ndarray, features: Dict[str, np.ndarray]
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
            matches, conf = self.building_signature.matches(
                point_features, self.strict_validation
            )

            if matches and conf >= self.min_confidence:
                # Features confirm building
                new_labels[i] = int(ASPRSClass.BUILDING)
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
                        point_features.get("height", 0.0)
                    )
                    confidences[i] = veg_conf
                    valid[i] = True
                    logger.debug(f"Roof vegetation detected: building → vegetation")
                else:
                    # Keep building label but with low confidence
                    new_labels[i] = int(ASPRSClass.BUILDING)
                    confidences[i] = 0.5
                    valid[i] = False

        return new_labels, confidences, valid

    def _validate_road(
        self, labels: np.ndarray, features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate road ground truth with features."""
        n_points = len(labels)
        new_labels = labels.copy()
        confidences = np.zeros(n_points, dtype=np.float32)
        valid = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            point_features = {k: v[i] for k, v in features.items()}

            # Check road signature
            matches, conf = self.road_signature.matches(
                point_features, self.strict_validation
            )

            if matches and conf >= self.min_confidence:
                # Features confirm road
                new_labels[i] = int(ASPRSClass.ROAD_SURFACE)
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
                        point_features.get("height", 0.0)
                    )
                    confidences[i] = veg_conf
                    valid[i] = True
                    logger.debug(f"Tree canopy over road detected: road → vegetation")
                else:
                    # Keep road label with low confidence
                    new_labels[i] = int(ASPRSClass.ROAD_SURFACE)
                    confidences[i] = 0.5
                    valid[i] = False

        return new_labels, confidences, valid

    def _validate_water(
        self, labels: np.ndarray, features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate water ground truth with features."""
        n_points = len(labels)
        new_labels = labels.copy()
        confidences = np.zeros(n_points, dtype=np.float32)
        valid = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            point_features = {k: v[i] for k, v in features.items()}

            # Check water signature
            matches, conf = self.water_signature.matches(
                point_features, self.strict_validation
            )

            if matches and conf >= self.min_confidence:
                new_labels[i] = int(ASPRSClass.WATER)
                confidences[i] = conf
                valid[i] = True
            else:
                # Water false positive - likely ground or vegetation
                new_labels[i] = int(ASPRSClass.GROUND)
                confidences[i] = 0.4
                valid[i] = False

        return new_labels, confidences, valid

    def _validate_vegetation(
        self, labels: np.ndarray, features: Dict[str, np.ndarray]
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
                    point_features.get("height", 0.0)
                )
                confidences[i] = conf
                valid[i] = True
            else:
                # Not vegetation - likely ground or building
                new_labels[i] = int(ASPRSClass.GROUND)
                confidences[i] = 0.4
                valid[i] = False

        return new_labels, confidences, valid

    def _classify_vegetation_by_height(self, height: float) -> int:
        """Classify vegetation type based on height."""
        if height < 0.5:
            return int(ASPRSClass.LOW_VEGETATION)
        elif height < 2.0:
            return int(ASPRSClass.MEDIUM_VEGETATION)
        else:
            return int(ASPRSClass.HIGH_VEGETATION)

    def check_vegetation_signature(
        self, features: Dict[str, np.ndarray]
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
        self, features: Dict[str, np.ndarray]
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
        self, features: Dict[str, np.ndarray]
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
        ground_truth_types: Optional[np.ndarray] = None,
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
            ground_truth_types = np.full(len(labels), "unknown", dtype=object)

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


# ============================================================================
# 🐛 BUGFIX FUNCTIONS - Feature Validation & Sanitization (v3.0.4)
# ============================================================================
# Added: October 26, 2025
# Purpose: Prevent NaN/Inf artifacts in building facade classification
# ============================================================================


def sanitize_feature(
    feature: np.ndarray,
    feature_name: str,
    clip_sigma: float = 5.0,
    fill_nan: float = 0.0,
    fill_inf: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Sanitize a feature array by handling NaN, Inf, and outliers.
    
    This function ensures feature arrays are safe to use in classification
    by:
    1. Replacing NaN values with a safe default
    2. Replacing Inf values with clipped values or safe default
    3. Optionally clipping outliers beyond N standard deviations
    
    Args:
        feature: Feature array to sanitize [N]
        feature_name: Name of feature (for logging)
        clip_sigma: Number of standard deviations for outlier clipping.
                   Set to 0 to disable outlier clipping.
        fill_nan: Value to use for NaN replacement
        fill_inf: Value to use for Inf replacement. If None, clips to
                 mean ± clip_sigma*std
    
    Returns:
        (sanitized_array, stats_dict) where stats_dict contains:
        - 'n_nan': Number of NaN values fixed
        - 'n_inf': Number of Inf values fixed  
        - 'n_outliers': Number of outliers clipped
        - 'n_total_fixed': Total number of values modified
    
    Example:
        >>> features['verticality'], stats = sanitize_feature(
        ...     features['verticality'], 
        ...     'verticality',
        ...     clip_sigma=5.0
        ... )
        >>> if stats['n_total_fixed'] > 0:
        ...     logger.warning(f"Fixed {stats['n_total_fixed']} invalid verticality values")
    """
    sanitized = feature.copy()
    stats = {
        'n_nan': 0,
        'n_inf': 0,
        'n_outliers': 0,
        'n_total_fixed': 0
    }
    
    # 1. Handle NaN
    nan_mask = np.isnan(sanitized)
    stats['n_nan'] = nan_mask.sum()
    if stats['n_nan'] > 0:
        sanitized[nan_mask] = fill_nan
        logger.debug(f"  {feature_name}: Replaced {stats['n_nan']} NaN values with {fill_nan}")
    
    # 2. Handle Inf
    inf_mask = np.isinf(sanitized)
    stats['n_inf'] = inf_mask.sum()
    if stats['n_inf'] > 0:
        if fill_inf is not None:
            sanitized[inf_mask] = fill_inf
            logger.debug(f"  {feature_name}: Replaced {stats['n_inf']} Inf values with {fill_inf}")
        else:
            # Clip to mean ± clip_sigma*std (computed on finite values)
            finite_mask = np.isfinite(feature)
            if finite_mask.any():
                mean_val = feature[finite_mask].mean()
                std_val = feature[finite_mask].std()
                clip_val = mean_val + clip_sigma * std_val
                
                sanitized[inf_mask & (sanitized > 0)] = clip_val
                sanitized[inf_mask & (sanitized < 0)] = mean_val - clip_sigma * std_val
                logger.debug(f"  {feature_name}: Clipped {stats['n_inf']} Inf values to ±{clip_sigma}σ")
    
    # 3. Handle outliers (optional)
    if clip_sigma > 0:
        finite_mask = np.isfinite(sanitized)
        if finite_mask.any():
            mean_val = sanitized[finite_mask].mean()
            std_val = sanitized[finite_mask].std()
            
            if std_val > 0:
                lower_bound = mean_val - clip_sigma * std_val
                upper_bound = mean_val + clip_sigma * std_val
                
                outlier_mask = (sanitized < lower_bound) | (sanitized > upper_bound)
                stats['n_outliers'] = outlier_mask.sum()
                
                if stats['n_outliers'] > 0:
                    sanitized[sanitized < lower_bound] = lower_bound
                    sanitized[sanitized > upper_bound] = upper_bound
                    logger.debug(f"  {feature_name}: Clipped {stats['n_outliers']} outliers to [{lower_bound:.3f}, {upper_bound:.3f}]")
    
    stats['n_total_fixed'] = stats['n_nan'] + stats['n_inf'] + stats['n_outliers']
    
    return sanitized, stats


def validate_features_for_classification(
    features: Dict[str, np.ndarray],
    required_features: list[str],
    point_mask: Optional[np.ndarray] = None,
    clip_sigma: float = 5.0,
    min_valid_ratio: float = 0.5,
) -> Tuple[bool, Dict[str, np.ndarray], list[str]]:
    """
    Validate and sanitize features before building classification.
    
    This function checks that:
    1. All required features are present
    2. Features have no NaN/Inf values (or fixes them)
    3. Features have reasonable value ranges
    4. Sufficient valid points are available
    
    Args:
        features: Dictionary of feature arrays {name: array[N]}
        required_features: List of feature names that must be present
        point_mask: Optional mask to restrict validation to subset of points
        clip_sigma: Number of std devs for outlier clipping (0 = disable)
        min_valid_ratio: Minimum ratio of valid points required (0-1)
    
    Returns:
        (is_valid, sanitized_features, issues) where:
        - is_valid: True if features pass validation
        - sanitized_features: Dict with cleaned feature arrays
        - issues: List of issue descriptions
    
    Example:
        >>> is_valid, clean_features, issues = validate_features_for_classification(
        ...     features={'normals': normals, 'verticality': verticality},
        ...     required_features=['normals', 'verticality'],
        ...     point_mask=building_mask
        ... )
        >>> if not is_valid:
        ...     logger.warning(f"Feature validation failed: {issues}")
        ...     # Use sanitized versions anyway
        ...     normals = clean_features['normals']
    """
    issues = []
    sanitized = {}
    is_valid = True
    
    # Determine number of points
    n_points = None
    for feat_array in features.values():
        if feat_array is not None:
            n_points = len(feat_array)
            break
    
    if n_points is None:
        issues.append("No feature arrays provided")
        return False, {}, issues
    
    # Apply point mask if provided
    if point_mask is not None:
        if len(point_mask) != n_points:
            issues.append(f"Point mask length ({len(point_mask)}) != feature length ({n_points})")
            is_valid = False
            point_mask = None  # Ignore invalid mask
    
    # Check required features present
    for feat_name in required_features:
        if feat_name not in features or features[feat_name] is None:
            issues.append(f"Required feature '{feat_name}' is missing")
            is_valid = False
    
    # Validate and sanitize each feature
    total_fixed = 0
    
    for feat_name, feat_array in features.items():
        if feat_array is None:
            sanitized[feat_name] = None
            continue
        
        # Check shape
        if len(feat_array.shape) == 1:
            # Scalar feature
            expected_shape = (n_points,)
        elif len(feat_array.shape) == 2:
            # Vector feature (e.g., normals [N, 3])
            expected_shape = (n_points, feat_array.shape[1])
        else:
            issues.append(f"Feature '{feat_name}' has unexpected shape {feat_array.shape}")
            is_valid = False
            sanitized[feat_name] = feat_array
            continue
        
        if feat_array.shape != expected_shape:
            issues.append(f"Feature '{feat_name}' shape {feat_array.shape} != expected {expected_shape}")
            is_valid = False
            sanitized[feat_name] = feat_array
            continue
        
        # Apply mask if provided
        feat_to_check = feat_array[point_mask] if point_mask is not None else feat_array
        
        # For vector features, check each component
        if len(feat_array.shape) == 2:
            # Vector feature (e.g., normals)
            sanitized_components = []
            for i in range(feat_array.shape[1]):
                component = feat_array[:, i]
                clean_component, stats = sanitize_feature(
                    component,
                    f"{feat_name}[{i}]",
                    clip_sigma=clip_sigma
                )
                sanitized_components.append(clean_component)
                total_fixed += stats['n_total_fixed']
                
                if stats['n_total_fixed'] > 0:
                    issues.append(
                        f"{feat_name}[{i}]: Fixed {stats['n_total_fixed']} invalid values "
                        f"(NaN={stats['n_nan']}, Inf={stats['n_inf']}, outliers={stats['n_outliers']})"
                    )
            
            sanitized[feat_name] = np.column_stack(sanitized_components)
        
        else:
            # Scalar feature
            clean_feat, stats = sanitize_feature(
                feat_array,
                feat_name,
                clip_sigma=clip_sigma
            )
            sanitized[feat_name] = clean_feat
            total_fixed += stats['n_total_fixed']
            
            if stats['n_total_fixed'] > 0:
                issues.append(
                    f"{feat_name}: Fixed {stats['n_total_fixed']} invalid values "
                    f"(NaN={stats['n_nan']}, Inf={stats['n_inf']}, outliers={stats['n_outliers']})"
                )
        
        # Check valid ratio
        feat_check = sanitized[feat_name][point_mask] if point_mask is not None else sanitized[feat_name]
        
        if len(feat_check.shape) == 2:
            # Vector: check all components finite
            valid_mask = np.all(np.isfinite(feat_check), axis=1)
        else:
            # Scalar
            valid_mask = np.isfinite(feat_check)
        
        valid_ratio = valid_mask.sum() / len(valid_mask) if len(valid_mask) > 0 else 0.0
        
        if valid_ratio < min_valid_ratio:
            issues.append(f"{feat_name}: Only {valid_ratio:.1%} valid values (< {min_valid_ratio:.1%})")
            is_valid = False
    
    # Summary
    if total_fixed > 0:
        logger.debug(f"Feature validation: Fixed {total_fixed} total invalid values across all features")
    
    if not is_valid:
        logger.warning(f"Feature validation failed with {len(issues)} issues")
    
    return is_valid, sanitized, issues


def create_safe_building_mask(
    building_mask: np.ndarray,
    is_ground: Optional[np.ndarray] = None,
    heights: Optional[np.ndarray] = None,
    ground_height_tolerance: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Create a safe building point mask by filtering out ground points.
    
    This ensures building classification only applies to non-ground points,
    preventing misclassification of ground as building walls.
    
    Args:
        building_mask: Initial building points mask [N]
        is_ground: Ground classification feature [N], 1=ground, 0=non-ground
        heights: Height above ground [N] in meters
        ground_height_tolerance: Max height (m) to consider as potential ground
    
    Returns:
        (safe_mask, stats) where:
        - safe_mask: Filtered mask excluding ground points
        - stats: Dictionary with filtering statistics
    
    Example:
        >>> safe_mask, stats = create_safe_building_mask(
        ...     building_mask,
        ...     is_ground=features['is_ground'],
        ...     heights=heights,
        ...     ground_height_tolerance=0.5
        ... )
        >>> logger.debug(f"Filtered {stats['n_ground_removed']} ground points")
    """
    safe_mask = building_mask.copy()
    stats = {
        'n_initial': building_mask.sum(),
        'n_ground_removed': 0,
        'n_low_removed': 0,
        'n_final': 0
    }
    
    # Filter by is_ground feature
    if is_ground is not None:
        ground_in_building = building_mask & (is_ground == 1)
        
        # Additional height check
        if heights is not None:
            # Remove ground points that are also low
            low_ground = ground_in_building & (heights < ground_height_tolerance)
            safe_mask[low_ground] = False
            stats['n_ground_removed'] = low_ground.sum()
        else:
            # Remove all ground points
            safe_mask[ground_in_building] = False
            stats['n_ground_removed'] = ground_in_building.sum()
    
    # Additional height-only filter if no is_ground available
    elif heights is not None:
        very_low = building_mask & (heights < ground_height_tolerance)
        safe_mask[very_low] = False
        stats['n_low_removed'] = very_low.sum()
    
    stats['n_final'] = safe_mask.sum()
    
    if stats['n_ground_removed'] > 0 or stats['n_low_removed'] > 0:
        logger.debug(
            f"Building mask filtering: {stats['n_initial']} → {stats['n_final']} points "
            f"(removed {stats['n_ground_removed']} ground, {stats['n_low_removed']} low)"
        )
    
    return safe_mask, stats
