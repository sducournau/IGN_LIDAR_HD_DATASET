"""
Feature validation utilities.

Validates and sanitizes computed features to prevent classification errors
caused by NaN, Inf, or out-of-range values.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def validate_features(
    features: Dict[str, np.ndarray], fix_invalid: bool = True, verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Validate and optionally fix invalid feature values.

    Checks for:
    - NaN values
    - Infinite values
    - Out-of-range values (e.g., NDVI outside [-1, 1])

    Args:
        features: Dictionary of feature arrays
        fix_invalid: If True, replace invalid values with safe defaults
        verbose: If True, log validation issues

    Returns:
        Tuple of (validated_features, issue_counts)
        - validated_features: Dictionary with validated/fixed features
        - issue_counts: Dictionary counting issues per feature

    Example:
        >>> features = compute_all_features(points)
        >>> validated, issues = validate_features(features)
        >>> if sum(issues.values()) > 0:
        ...     logger.warning(f"Fixed {sum(issues.values())} invalid values")
    """
    validated = {}
    issue_counts = {}

    for name, values in features.items():
        if values is None:
            continue

        # Count issues
        n_nan = np.isnan(values).sum()
        n_inf = np.isinf(values).sum()
        n_invalid = n_nan + n_inf

        # Range validation for specific features
        n_out_of_range = 0
        if name in ["ndvi", "nir_intensity"]:
            # NDVI and NIR should be in [-1, 1]
            n_out_of_range = ((values < -1.0) | (values > 1.0)).sum()
        elif name in [
            "planarity",
            "linearity",
            "sphericity",
            "verticality",
            "horizontality",
        ]:
            # Geometric features should be in [0, 1]
            n_out_of_range = ((values < 0.0) | (values > 1.0)).sum()
        elif name == "curvature":
            # Curvature should be >= 0
            n_out_of_range = (values < 0.0).sum()

        total_issues = n_invalid + n_out_of_range

        if total_issues > 0:
            issue_counts[name] = total_issues

            if verbose:
                logger.warning(
                    f"Feature '{name}': {n_nan} NaN, {n_inf} Inf, "
                    f"{n_out_of_range} out-of-range values "
                    f"({total_issues}/{len(values)} = "
                    f"{100*total_issues/len(values):.1f}%)"
                )

            if fix_invalid:
                # Create copy to avoid modifying original
                values = values.copy()

                # Fix NaN/Inf values
                if n_invalid > 0:
                    values = _fix_invalid_values(name, values)

                # Fix out-of-range values
                if n_out_of_range > 0:
                    values = _clip_to_valid_range(name, values)

        validated[name] = values

    if verbose and issue_counts:
        total_fixed = sum(issue_counts.values())
        logger.info(
            f"Validated features: fixed {total_fixed} invalid values "
            f"across {len(issue_counts)} features"
        )

    return validated, issue_counts


def _fix_invalid_values(feature_name: str, values: np.ndarray) -> np.ndarray:
    """
    Replace NaN/Inf values with safe defaults.

    Args:
        feature_name: Name of the feature
        values: Feature array

    Returns:
        Fixed array
    """
    # Create mask for invalid values
    invalid_mask = ~np.isfinite(values)

    if not invalid_mask.any():
        return values

    # Determine safe default value based on feature type
    if feature_name in ["ndvi", "nir_intensity"]:
        # Neutral value for spectral features
        default = 0.0
    elif feature_name in ["planarity", "linearity", "sphericity"]:
        # Low shape features = unstructured
        default = 0.0
    elif feature_name in ["verticality", "horizontality"]:
        # Medium values = no strong orientation
        default = 0.5
    elif feature_name == "curvature":
        # Low curvature = flat
        default = 0.0
    elif feature_name == "density":
        # Use median of valid values if available
        valid_values = values[~invalid_mask]
        default = np.median(valid_values) if len(valid_values) > 0 else 10.0
    elif "height" in feature_name:
        # Zero height for ground level
        default = 0.0
    else:
        # Generic default: zero
        default = 0.0

    # Replace invalid values
    values[invalid_mask] = default

    return values


def _clip_to_valid_range(feature_name: str, values: np.ndarray) -> np.ndarray:
    """
    Clip feature values to valid range.

    Args:
        feature_name: Name of the feature
        values: Feature array

    Returns:
        Clipped array
    """
    if feature_name in ["ndvi", "nir_intensity"]:
        # NDVI and NIR: [-1, 1]
        return np.clip(values, -1.0, 1.0)
    elif feature_name in [
        "planarity",
        "linearity",
        "sphericity",
        "verticality",
        "horizontality",
    ]:
        # Geometric features: [0, 1]
        return np.clip(values, 0.0, 1.0)
    elif feature_name == "curvature":
        # Curvature: [0, inf), but clip to reasonable max
        return np.clip(values, 0.0, 1.0)
    else:
        # No clipping needed
        return values


def check_feature_sanity(
    features: Dict[str, np.ndarray], points: Optional[np.ndarray] = None
) -> bool:
    """
    Quick sanity check on computed features.

    Args:
        features: Dictionary of feature arrays
        points: Optional point cloud array for size validation

    Returns:
        True if features pass sanity checks, False otherwise
    """
    if not features:
        logger.error("No features provided")
        return False

    # Check all features have same length
    lengths = {
        name: len(values) for name, values in features.items() if values is not None
    }

    if len(set(lengths.values())) > 1:
        logger.error(f"Features have inconsistent lengths: {lengths}")
        return False

    # Check length matches points if provided
    if points is not None:
        expected_len = len(points)
        actual_len = next(iter(lengths.values()))
        if actual_len != expected_len:
            logger.error(
                f"Feature length {actual_len} doesn't match "
                f"point count {expected_len}"
            )
            return False

    # Check for completely invalid features
    for name, values in features.items():
        if values is None:
            continue

        if np.isnan(values).all():
            logger.error(f"Feature '{name}' is all NaN")
            return False

        if np.isinf(values).all():
            logger.error(f"Feature '{name}' is all Inf")
            return False

    return True
