"""
Centralized Classification Priority System

This module defines the canonical priority order for ground truth features
when multiple features overlap at the same spatial location.

Priority Rule: HIGHER number = HIGHER priority (overwrites lower priority)

Usage:
    from ign_lidar.core.classification.priorities import (
        PRIORITY_ORDER,
        get_priority_value,
        get_label_map
    )
"""

from typing import Dict, List, Tuple

# ✅ CANONICAL PRIORITY ORDER (highest to lowest)
# When a point is in multiple polygons, the feature with the HIGHEST priority
# value wins.
#
# Priority values are assigned in descending order from the list:
# - buildings: priority 9 (highest)
# - bridges: priority 8
# - railways: priority 7
# - sports: priority 6
# - parking: priority 5
# - cemeteries: priority 4
# - water: priority 3
# - vegetation: priority 2
# - roads: priority 1 (lowest)
#
# 🔄 V3.0.6 CHANGE: Roads moved to lowest priority (below vegetation)
# This ensures vegetation can overwrite roads (trees on roads remain as
# vegetation) while roads are processed first in sequential iteration.
PRIORITY_ORDER: List[str] = [
    "buildings",  # Priority 9 - Highest (solid structures)
    "bridges",  # Priority 8 - Elevated structures
    "railways",  # Priority 7 - Rail transport
    "sports",  # Priority 6 - Sports facilities
    "parking",  # Priority 5 - Parking areas
    "cemeteries",  # Priority 4 - Cemetery grounds
    "water",  # Priority 3 - Water bodies
    "vegetation",  # Priority 2 - Natural features (can overwrite roads)
    "roads",  # Priority 1 - Lowest (transport infrastructure, processed first)
]

# Simplified label mapping for ground truth labeling
# (used by ground_truth_optimizer.py)
LABEL_MAP: Dict[str, int] = {
    "buildings": 1,
    "roads": 2,
    "water": 3,
    "vegetation": 4,
    "bridges": 5,
    "railways": 6,
    "sports": 7,
    "parking": 8,
    "cemeteries": 9,
}


def get_priority_value(feature_name: str) -> int:
    """
    Get numerical priority value for a feature type.

    Args:
        feature_name: Feature type name (e.g., "buildings", "roads")

    Returns:
        Priority value (higher = higher priority). Returns 0 if feature
        not found.

    Example:
        >>> get_priority_value("buildings")
        9
        >>> get_priority_value("vegetation")
        1
    """
    try:
        # Priority = reverse index in list (last item = highest priority)
        return len(PRIORITY_ORDER) - PRIORITY_ORDER.index(feature_name)
    except ValueError:
        # Feature not in priority list
        return 0


def get_label_map() -> Dict[str, int]:
    """
    Get the simplified label mapping for ground truth features.

    Returns:
        Dictionary mapping feature names to label codes
    """
    return LABEL_MAP.copy()


def get_priority_order_for_iteration() -> List[str]:
    """
    Get priority order for sequential iteration (lowest to highest).

    This is used by reclassifier.py which processes features in sequence
    and allows later features to overwrite earlier ones.

    Returns:
        List of feature names in order from lowest to highest priority
    """
    # Return reversed order so lowest priority is processed first
    return list(reversed(PRIORITY_ORDER))


def validate_priority_consistency() -> bool:
    """
    Validate that priority system is internally consistent.

    Checks:
    - No duplicate features in priority list
    - All features in label map have priorities

    Returns:
        True if consistent, raises AssertionError otherwise
    """
    # Check for duplicates
    assert len(PRIORITY_ORDER) == len(
        set(PRIORITY_ORDER)
    ), "Duplicate features in PRIORITY_ORDER"

    # Check label map consistency
    for feature in LABEL_MAP.keys():
        if feature not in PRIORITY_ORDER:
            raise AssertionError(
                f"Feature '{feature}' in LABEL_MAP but not in PRIORITY_ORDER"
            )

    return True


# Validate on module import
validate_priority_consistency()
