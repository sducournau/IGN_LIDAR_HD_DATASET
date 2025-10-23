"""
Confidence scoring utilities for rule-based classification.

This module provides functions for calculating and combining confidence scores:
- Various confidence calculation methods (binary, linear, sigmoid, etc.)
- Confidence combination strategies (weighted, max, min, geometric mean)
- Normalization and calibration utilities

Usage:
    from ign_lidar.core.classification.rules.confidence import (
        ConfidenceMethod,
        calculate_confidence,
        combine_confidences
    )
"""

from enum import Enum
from typing import Dict, List, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ConfidenceMethod(str, Enum):
    """Methods for calculating confidence scores"""
    BINARY = "binary"           # 0.0 or 1.0 (hard threshold)
    LINEAR = "linear"           # Linear scaling between min/max
    SIGMOID = "sigmoid"         # Smooth sigmoid curve
    GAUSSIAN = "gaussian"       # Gaussian centered at target
    THRESHOLD = "threshold"     # Step function with soft edges
    EXPONENTIAL = "exponential" # Exponential decay/growth
    COMPOSITE = "composite"     # Weighted combination of scores


class CombinationMethod(str, Enum):
    """Methods for combining multiple confidence scores"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAX = "max"
    MIN = "min"
    PRODUCT = "product"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"


def calculate_confidence(
    scores: np.ndarray,
    method: ConfidenceMethod = ConfidenceMethod.LINEAR,
    params: Optional[Dict] = None
) -> np.ndarray:
    """Calculate confidence scores from raw scores
    
    Args:
        scores: Raw score values [N]
        method: Confidence calculation method
        params: Method-specific parameters
    
    Returns:
        Confidence scores in [0, 1]
    """
    params = params or {}
    
    if method == ConfidenceMethod.BINARY:
        threshold = params.get('threshold', 0.5)
        return (scores >= threshold).astype(float)
    
    elif method == ConfidenceMethod.LINEAR:
        min_val = params.get('min', scores.min())
        max_val = params.get('max', scores.max())
        
        if max_val <= min_val:
            return np.ones_like(scores, dtype=float)
        
        confidence = (scores - min_val) / (max_val - min_val)
        return np.clip(confidence, 0, 1)
    
    elif method == ConfidenceMethod.SIGMOID:
        center = params.get('center', 0.5)
        steepness = params.get('steepness', 10.0)
        
        # Sigmoid: 1 / (1 + exp(-steepness * (x - center)))
        confidence = 1.0 / (1.0 + np.exp(-steepness * (scores - center)))
        return confidence
    
    elif method == ConfidenceMethod.GAUSSIAN:
        center = params.get('center', 0.5)
        sigma = params.get('sigma', 0.2)
        
        # Gaussian: exp(-(x - center)^2 / (2 * sigma^2))
        confidence = np.exp(-((scores - center) ** 2) / (2 * sigma ** 2))
        return confidence
    
    elif method == ConfidenceMethod.THRESHOLD:
        threshold = params.get('threshold', 0.5)
        soft_margin = params.get('soft_margin', 0.1)
        
        # Soft threshold with smooth transition
        confidence = np.zeros_like(scores, dtype=float)
        
        # Below threshold
        below = scores < (threshold - soft_margin)
        confidence[below] = 0.0
        
        # Above threshold
        above = scores > (threshold + soft_margin)
        confidence[above] = 1.0
        
        # In transition zone
        transition = ~(below | above)
        if np.any(transition):
            t_scores = scores[transition]
            t_range = 2 * soft_margin
            confidence[transition] = (t_scores - (threshold - soft_margin)) / t_range
        
        return confidence
    
    elif method == ConfidenceMethod.EXPONENTIAL:
        rate = params.get('rate', 1.0)
        decay = params.get('decay', True)
        
        if decay:
            # Exponential decay: exp(-rate * x)
            confidence = np.exp(-rate * scores)
        else:
            # Exponential growth: 1 - exp(-rate * x)
            confidence = 1.0 - np.exp(-rate * scores)
        
        return np.clip(confidence, 0, 1)
    
    else:
        raise ValueError(f"Unknown confidence method: {method}")


def combine_confidences(
    confidences: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    method: CombinationMethod = CombinationMethod.WEIGHTED_AVERAGE
) -> np.ndarray:
    """Combine multiple confidence scores
    
    Args:
        confidences: Dictionary mapping names to confidence arrays
        weights: Optional weights for each confidence (uniform if None)
        method: Combination method
    
    Returns:
        Combined confidence scores in [0, 1]
    """
    if not confidences:
        raise ValueError("Cannot combine empty confidences dict")
    
    # Get array shape from first confidence
    n_points = len(next(iter(confidences.values())))
    
    # Validate all arrays have same length
    for name, conf in confidences.items():
        if len(conf) != n_points:
            raise ValueError(
                f"Confidence '{name}' has {len(conf)} values, expected {n_points}"
            )
    
    # Default to uniform weights
    if weights is None:
        weights = {name: 1.0 for name in confidences.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Total weight is zero")
    
    weights = {name: w / total_weight for name, w in weights.items()}
    
    # Apply combination method
    if method == CombinationMethod.WEIGHTED_AVERAGE:
        combined = np.zeros(n_points, dtype=float)
        for name, conf in confidences.items():
            weight = weights.get(name, 0.0)
            combined += conf * weight
        return combined
    
    elif method == CombinationMethod.MAX:
        arrays = list(confidences.values())
        return np.maximum.reduce(arrays)
    
    elif method == CombinationMethod.MIN:
        arrays = list(confidences.values())
        return np.minimum.reduce(arrays)
    
    elif method == CombinationMethod.PRODUCT:
        combined = np.ones(n_points, dtype=float)
        for conf in confidences.values():
            combined *= conf
        return combined
    
    elif method == CombinationMethod.GEOMETRIC_MEAN:
        # Weighted geometric mean
        combined = np.ones(n_points, dtype=float)
        for name, conf in confidences.items():
            weight = weights.get(name, 0.0)
            # Avoid issues with zero confidence
            safe_conf = np.maximum(conf, 1e-10)
            combined *= safe_conf ** weight
        return combined
    
    elif method == CombinationMethod.HARMONIC_MEAN:
        # Weighted harmonic mean
        combined = np.zeros(n_points, dtype=float)
        for name, conf in confidences.items():
            weight = weights.get(name, 0.0)
            # Avoid division by zero
            safe_conf = np.maximum(conf, 1e-10)
            combined += weight / safe_conf
        
        # Avoid division by zero in final step
        mask = combined > 0
        result = np.zeros(n_points, dtype=float)
        result[mask] = 1.0 / combined[mask]
        return result
    
    else:
        raise ValueError(f"Unknown combination method: {method}")


def normalize_confidence(
    confidence: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """Normalize confidence scores to [min_val, max_val]
    
    Args:
        confidence: Input confidence scores
        min_val: Minimum output value
        max_val: Maximum output value
    
    Returns:
        Normalized confidence scores
    """
    if max_val <= min_val:
        raise ValueError(f"max_val ({max_val}) must be > min_val ({min_val})")
    
    current_min = confidence.min()
    current_max = confidence.max()
    
    if current_max <= current_min:
        # All values are the same
        return np.full_like(confidence, (min_val + max_val) / 2)
    
    # Scale to [0, 1]
    normalized = (confidence - current_min) / (current_max - current_min)
    
    # Scale to [min_val, max_val]
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized


def calibrate_confidence(
    confidence: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Calibrate confidence scores based on actual accuracy
    
    Args:
        confidence: Predicted confidence scores
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        n_bins: Number of bins for calibration curve
    
    Returns:
        Dictionary with calibration statistics
    """
    # Calculate accuracy in each confidence bin
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_min = bins[i]
        bin_max = bins[i + 1]
        
        # Points in this bin
        in_bin = (confidence >= bin_min) & (confidence < bin_max)
        
        if i == n_bins - 1:  # Last bin includes upper bound
            in_bin = (confidence >= bin_min) & (confidence <= bin_max)
        
        n_in_bin = np.sum(in_bin)
        
        if n_in_bin > 0:
            # Calculate accuracy in this bin
            correct = true_labels[in_bin] == predicted_labels[in_bin]
            accuracy = np.mean(correct)
            mean_confidence = np.mean(confidence[in_bin])
            
            bin_accuracies.append(accuracy)
            bin_confidences.append(mean_confidence)
            bin_counts.append(n_in_bin)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_min + bin_max) / 2)
            bin_counts.append(0)
    
    # Calculate expected calibration error (ECE)
    ece = 0.0
    total_count = len(confidence)
    
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        if count > 0:
            ece += (count / total_count) * abs(acc - conf)
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(true_labels == predicted_labels)
    mean_confidence = np.mean(confidence)
    
    return {
        'expected_calibration_error': ece,
        'overall_accuracy': overall_accuracy,
        'mean_confidence': mean_confidence,
        'confidence_accuracy_gap': abs(mean_confidence - overall_accuracy),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }


def apply_confidence_threshold(
    labels: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """Apply confidence threshold to filter low-confidence predictions
    
    Args:
        labels: Predicted labels
        confidence: Confidence scores
        threshold: Minimum confidence to keep predictions
    
    Returns:
        Filtered labels (0 for below threshold)
    """
    filtered = labels.copy()
    filtered[confidence < threshold] = 0
    return filtered
