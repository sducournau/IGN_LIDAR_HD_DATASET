"""
Classification Normalization Module

Handles normalization and cleanup of classification codes, including
handling of non-standard or unknown classes from IGN data.

This module ensures that all classification codes conform to ASPRS standards
before processing.
"""

import numpy as np
import logging
from typing import Dict, Set, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# Standard ASPRS classes (0-22, plus extended classes 32+)
VALID_ASPRS_CLASSES = set(range(0, 23))  # Standard classes 0-22
VALID_ASPRS_CLASSES.update([32, 33, 34, 35, 36, 37, 38, 39])  # Roads
VALID_ASPRS_CLASSES.update([40, 41, 42, 43, 44])  # Infrastructure & Agriculture
VALID_ASPRS_CLASSES.update(range(50, 63))  # Buildings
VALID_ASPRS_CLASSES.update(range(70, 77))  # Vegetation
VALID_ASPRS_CLASSES.update(range(80, 86))  # Water
VALID_ASPRS_CLASSES.update(range(90, 101))  # Infrastructure
VALID_ASPRS_CLASSES.update(range(110, 115))  # Urban furniture
VALID_ASPRS_CLASSES.update(range(120, 126))  # Terrain
VALID_ASPRS_CLASSES.update(range(130, 136))  # Vehicles

# Known IGN non-standard classes that should be remapped
IGN_CLASS_REMAPPING = {
    67: 1,  # Unknown class 67 -> Unclassified
    # Add more mappings as discovered
}


def normalize_classification(
    classification: np.ndarray,
    strict_mode: bool = False,
    report_unknown: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize classification codes to ASPRS standards.
    
    This function:
    1. Identifies non-standard classification codes
    2. Remaps known IGN-specific codes to ASPRS equivalents
    3. Optionally converts unknown codes to 'Unclassified' (class 1)
    
    Args:
        classification: (N,) array of classification codes
        strict_mode: If True, remap ALL unknown codes to class 1
                    If False, only remap known problematic codes
        report_unknown: If True, log information about unknown classes
        
    Returns:
        normalized_classification: (N,) array with normalized codes
        stats: Dictionary with normalization statistics
    """
    original_classification = classification.copy()
    normalized_classification = classification.copy()
    
    # Find unique classes
    unique_classes = np.unique(classification)
    unknown_classes = [c for c in unique_classes if c not in VALID_ASPRS_CLASSES]
    
    stats = {
        'original_classes': sorted(unique_classes.tolist()),
        'unknown_classes': sorted(unknown_classes),
        'remapped_count': 0,
        'remapping_details': {}
    }
    
    if len(unknown_classes) == 0:
        if report_unknown:
            logger.debug("✓ All classification codes are valid ASPRS standards")
        return normalized_classification, stats
    
    # Report unknown classes
    if report_unknown:
        logger.warning(f"⚠️  Found {len(unknown_classes)} non-standard classification codes: {unknown_classes}")
    
    # Apply remapping
    for unknown_class in unknown_classes:
        # Count points with this class
        mask = classification == unknown_class
        count = np.sum(mask)
        percentage = (count / len(classification)) * 100
        
        # Determine target class
        if unknown_class in IGN_CLASS_REMAPPING:
            target_class = IGN_CLASS_REMAPPING[unknown_class]
            reason = "known IGN class"
        elif strict_mode:
            target_class = 1  # Unclassified
            reason = "strict mode"
        else:
            # In non-strict mode, leave unknown classes as-is
            # (they will be logged but not changed)
            if report_unknown:
                logger.warning(f"   Class {unknown_class}: {count:,} points ({percentage:.2f}%) - NOT REMAPPED (use strict_mode=True)")
            continue
        
        # Apply remapping
        normalized_classification[mask] = target_class
        stats['remapped_count'] += count
        stats['remapping_details'][int(unknown_class)] = {
            'target_class': int(target_class),
            'count': int(count),
            'percentage': float(percentage),
            'reason': reason
        }
        
        if report_unknown:
            logger.info(f"   Class {unknown_class} → {target_class}: {count:,} points ({percentage:.2f}%) [{reason}]")
    
    # Summary
    if stats['remapped_count'] > 0:
        total_percentage = (stats['remapped_count'] / len(classification)) * 100
        logger.info(f"✓ Remapped {stats['remapped_count']:,} points ({total_percentage:.2f}%) from {len(stats['remapping_details'])} classes")
    
    return normalized_classification, stats


def validate_classification(
    classification: np.ndarray,
    allow_extended: bool = True
) -> Dict[str, Any]:
    """
    Validate classification codes and return detailed statistics.
    
    Args:
        classification: (N,) array of classification codes
        allow_extended: If True, allow extended ASPRS classes (32+)
                       If False, only allow standard classes (0-22)
        
    Returns:
        validation_report: Dictionary with validation results
    """
    unique_classes = np.unique(classification)
    
    report = {
        'is_valid': True,
        'total_points': len(classification),
        'num_unique_classes': len(unique_classes),
        'classes': sorted(unique_classes.tolist()),
        'standard_classes': [],
        'extended_classes': [],
        'invalid_classes': [],
        'warnings': []
    }
    
    for cls in unique_classes:
        if cls <= 22:
            report['standard_classes'].append(int(cls))
        elif cls in VALID_ASPRS_CLASSES:
            report['extended_classes'].append(int(cls))
        else:
            report['invalid_classes'].append(int(cls))
            report['is_valid'] = False
    
    # Generate warnings
    if not allow_extended and report['extended_classes']:
        report['warnings'].append(f"Extended classes found but allow_extended=False: {report['extended_classes']}")
        report['is_valid'] = False
    
    if report['invalid_classes']:
        report['warnings'].append(f"Invalid classification codes found: {report['invalid_classes']}")
    
    return report


def get_classification_distribution(
    classification: np.ndarray,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Get detailed distribution of classification codes.
    
    Args:
        classification: (N,) array of classification codes
        class_names: Optional mapping of class codes to names
        
    Returns:
        distribution: Dictionary mapping class code to statistics
    """
    from ign_lidar.classification_schema import ASPRS_CLASS_NAMES
    
    if class_names is None:
        class_names = ASPRS_CLASS_NAMES
    
    unique_classes, counts = np.unique(classification, return_counts=True)
    total_points = len(classification)
    
    distribution = {}
    for cls, count in zip(unique_classes, counts):
        distribution[int(cls)] = {
            'name': class_names.get(cls, f"Unknown ({cls})"),
            'count': int(count),
            'percentage': float(count / total_points * 100),
            'is_standard': cls <= 22,
            'is_valid': cls in VALID_ASPRS_CLASSES
        }
    
    return distribution
