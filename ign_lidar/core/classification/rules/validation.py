"""
Feature validation utilities for rule-based classification.

This module provides validation functions for checking:
- Feature availability and completeness
- Feature array shapes and dimensions
- Feature value ranges and quality
- Compatibility between features and rules

Usage:
    from ign_lidar.core.classification.rules.validation import (
        FeatureRequirements,
        validate_features,
        validate_feature_quality
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureRequirements:
    """Defines required and optional features for a rule or engine
    
    Attributes:
        required: Set of feature names that must be present
        optional: Set of feature names that improve results if present
        min_quality: Minimum acceptable feature quality score [0, 1]
        allow_nan: Whether NaN values are acceptable
    """
    required: Set[str]
    optional: Set[str] = field(default_factory=set)
    min_quality: float = 0.0
    allow_nan: bool = False
    
    def __post_init__(self):
        """Validate requirements"""
        if not isinstance(self.required, set):
            self.required = set(self.required)
        if not isinstance(self.optional, set):
            self.optional = set(self.optional)
        
        # Ensure no overlap between required and optional
        overlap = self.required & self.optional
        if overlap:
            logger.warning(
                f"Features {overlap} marked as both required and optional, "
                "treating as required"
            )
            self.optional -= overlap


def validate_features(
    features: Dict[str, np.ndarray],
    requirements: FeatureRequirements,
    n_points: Optional[int] = None
) -> None:
    """
    Validate that features meet requirements.
    
    Checks feature availability, array shapes, and value validity (NaN/Inf)
    according to the provided requirements specification.
    
    Args:
        features: Dictionary of feature arrays {name: values}. Each value
                 should be a numpy array with consistent length.
        requirements: Feature requirements specification defining:
                     - required: Features that must be present
                     - optional: Features that improve results if present
                     - min_quality: Minimum fraction of valid values
                     - allow_nan: Whether NaN values are acceptable
        n_points: Expected number of points. If None, inferred from the
                 first feature array length.
    
    Raises:
        ValueError: If required features are missing, arrays have wrong
                   shape, or contain invalid values (NaN/Inf when not allowed)
    
    Example:
        Basic validation with required features:
        
        >>> import numpy as np
        >>> from ign_lidar.core.classification.rules.validation import (
        ...     validate_features, FeatureRequirements
        ... )
        >>> 
        >>> # Define requirements
        >>> requirements = FeatureRequirements(
        ...     required={'height', 'planarity'},
        ...     optional={'ndvi', 'intensity'}
        ... )
        >>> 
        >>> # Valid features - validation passes
        >>> features = {
        ...     'height': np.random.rand(100) * 10,
        ...     'planarity': np.random.rand(100)
        ... }
        >>> validate_features(features, requirements)  # No exception = valid
        >>> 
        >>> # Add optional feature - still valid
        >>> features['ndvi'] = np.random.rand(100) * 2 - 1
        >>> validate_features(features, requirements)
        
        Validation catches missing required features:
        
        >>> # Missing required feature 'planarity'
        >>> bad_features = {'height': np.random.rand(100)}
        >>> try:
        ...     validate_features(bad_features, requirements)
        ... except ValueError as e:
        ...     print(f"Caught error: {e}")
        Caught error: Missing required features: {'planarity'}
        
        Validation catches shape mismatches:
        
        >>> # Mismatched array lengths
        >>> bad_features = {
        ...     'height': np.random.rand(100),
        ...     'planarity': np.random.rand(50)  # Wrong length!
        ... }
        >>> try:
        ...     validate_features(bad_features, requirements)
        ... except ValueError as e:
        ...     print(f"Caught error: {e}")
        Caught error: Feature 'planarity' has 50 values, expected 100
        
        Validation catches NaN values:
        
        >>> # Features with NaN values
        >>> features_with_nan = {
        ...     'height': np.random.rand(100),
        ...     'planarity': np.random.rand(100)
        ... }
        >>> features_with_nan['height'][10:15] = np.nan
        >>> 
        >>> try:
        ...     validate_features(features_with_nan, requirements)
        ... except ValueError as e:
        ...     print(f"Caught error: {e}")
        Caught error: Feature 'height' contains 5 NaN values (5.0%)
        
        Allowing NaN values when needed:
        
        >>> # Create requirements that allow NaN
        >>> flexible_requirements = FeatureRequirements(
        ...     required={'height'},
        ...     allow_nan=True
        ... )
        >>> validate_features(features_with_nan, flexible_requirements)  # OK now
    
    Note:
        - Validation is strict: all required features must be present
        - Optional features are not checked if absent
        - Array shapes must match n_points exactly
        - NaN and Inf values are rejected unless explicitly allowed
        - Extra features (not in required or optional) are ignored
    
    See Also:
        - FeatureRequirements: Define feature requirements
        - check_feature_quality: Check individual feature quality
        - validate_feature_shape: Validate specific array shapes
    """
    # Check for missing required features
    available = set(features.keys())
    missing = requirements.required - available
    
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Infer number of points if not specified
    if n_points is None:
        if not features:
            raise ValueError("Cannot infer n_points from empty features dict")
        n_points = len(next(iter(features.values())))
    
    # Validate each feature
    for name, values in features.items():
        if name not in requirements.required and name not in requirements.optional:
            continue  # Skip validation for extra features
        
        # Check shape
        if len(values) != n_points:
            raise ValueError(
                f"Feature '{name}' has {len(values)} values, expected {n_points}"
            )
        
        # Check for NaN values
        if not requirements.allow_nan and np.any(np.isnan(values)):
            n_nan = np.sum(np.isnan(values))
            raise ValueError(
                f"Feature '{name}' contains {n_nan} NaN values "
                f"({100*n_nan/n_points:.1f}%)"
            )
        
        # Check for infinite values
        if np.any(np.isinf(values)):
            n_inf = np.sum(np.isinf(values))
            raise ValueError(
                f"Feature '{name}' contains {n_inf} infinite values "
                f"({100*n_inf/n_points:.1f}%)"
            )


def validate_feature_shape(
    features: Dict[str, np.ndarray],
    expected_shape: Tuple[int, ...],
    feature_names: Optional[List[str]] = None
) -> None:
    """Validate that features have expected shape
    
    Args:
        features: Dictionary of feature arrays
        expected_shape: Expected array shape (e.g., (N,) or (N, 3))
        feature_names: Specific features to check (all if None)
    
    Raises:
        ValueError: If any feature has incorrect shape
    """
    if feature_names is None:
        feature_names = list(features.keys())
    
    for name in feature_names:
        if name not in features:
            continue
        
        values = features[name]
        if values.shape != expected_shape:
            raise ValueError(
                f"Feature '{name}' has shape {values.shape}, "
                f"expected {expected_shape}"
            )


def check_feature_quality(
    features: Dict[str, np.ndarray],
    feature_name: str,
    min_quality: float = 0.0
) -> float:
    """Check quality of a single feature
    
    Quality is defined as the fraction of non-NaN, non-infinite, valid values.
    
    Args:
        features: Dictionary of feature arrays
        feature_name: Name of feature to check
        min_quality: Minimum acceptable quality [0, 1]
    
    Returns:
        Quality score in [0, 1]
    
    Raises:
        ValueError: If quality is below threshold
    """
    if feature_name not in features:
        raise ValueError(f"Feature '{feature_name}' not found")
    
    values = features[feature_name]
    n_total = len(values)
    
    # Count valid values (not NaN, not infinite)
    valid_mask = np.isfinite(values)
    n_valid = np.sum(valid_mask)
    
    quality = n_valid / n_total if n_total > 0 else 0.0
    
    if quality < min_quality:
        raise ValueError(
            f"Feature '{feature_name}' quality {quality:.2%} is below "
            f"minimum threshold {min_quality:.2%} "
            f"({n_total - n_valid}/{n_total} invalid values)"
        )
    
    return quality


def check_all_feature_quality(
    features: Dict[str, np.ndarray],
    min_quality: float = 0.8
) -> Dict[str, float]:
    """Check quality of all features
    
    Args:
        features: Dictionary of feature arrays
        min_quality: Minimum acceptable quality [0, 1]
    
    Returns:
        Dictionary mapping feature names to quality scores
    
    Raises:
        ValueError: If any feature quality is below threshold
    """
    quality_scores = {}
    
    for name in features.keys():
        quality = check_feature_quality(features, name, min_quality=min_quality)
        quality_scores[name] = quality
    
    return quality_scores


def validate_feature_ranges(
    features: Dict[str, np.ndarray],
    expected_ranges: Dict[str, Tuple[float, float]],
    strict: bool = False
) -> None:
    """Validate that feature values are within expected ranges
    
    Args:
        features: Dictionary of feature arrays
        expected_ranges: Dictionary mapping feature names to (min, max) tuples
        strict: If True, raise error on out-of-range values; if False, only warn
    
    Raises:
        ValueError: If strict=True and values are out of range
    """
    for name, (min_val, max_val) in expected_ranges.items():
        if name not in features:
            continue
        
        values = features[name]
        
        # Check minimum
        below_min = np.sum(values < min_val)
        if below_min > 0:
            msg = (
                f"Feature '{name}' has {below_min} values below minimum "
                f"{min_val} ({100*below_min/len(values):.1f}%)"
            )
            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
        
        # Check maximum
        above_max = np.sum(values > max_val)
        if above_max > 0:
            msg = (
                f"Feature '{name}' has {above_max} values above maximum "
                f"{max_val} ({100*above_max/len(values):.1f}%)"
            )
            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)


def validate_points_array(
    points: np.ndarray,
    min_points: int = 1,
    expected_dims: int = 3
) -> None:
    """Validate point cloud array
    
    Args:
        points: Point coordinates array
        min_points: Minimum number of points required
        expected_dims: Expected number of dimensions (usually 3 for XYZ)
    
    Raises:
        ValueError: If points array is invalid
    """
    if not isinstance(points, np.ndarray):
        raise ValueError(f"Points must be numpy array, got {type(points)}")
    
    if points.ndim != 2:
        raise ValueError(f"Points must be 2D array, got shape {points.shape}")
    
    n_points, n_dims = points.shape
    
    if n_points < min_points:
        raise ValueError(
            f"Need at least {min_points} points, got {n_points}"
        )
    
    if n_dims != expected_dims:
        raise ValueError(
            f"Expected {expected_dims} dimensions, got {n_dims}"
        )
    
    # Check for NaN/inf
    if np.any(np.isnan(points)):
        raise ValueError("Points array contains NaN values")
    
    if np.any(np.isinf(points)):
        raise ValueError("Points array contains infinite values")


def get_feature_statistics(
    features: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """Get statistical summary of all features
    
    Args:
        features: Dictionary of feature arrays
    
    Returns:
        Dictionary mapping feature names to statistics
        {name: {mean, std, min, max, median, quality}}
    """
    stats = {}
    
    for name, values in features.items():
        valid_mask = np.isfinite(values)
        valid_values = values[valid_mask]
        
        if len(valid_values) > 0:
            stats[name] = {
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values)),
                'median': float(np.median(valid_values)),
                'quality': float(np.sum(valid_mask) / len(values)),
                'n_valid': int(np.sum(valid_mask)),
                'n_total': len(values)
            }
        else:
            stats[name] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'quality': 0.0,
                'n_valid': 0,
                'n_total': len(values)
            }
    
    return stats
