"""
ASPRS Classification Constants - Convenience Wrapper

This module provides a convenient import path and helper functions for the
ASPRS classification codes defined in classification_schema.py.

Instead of:
    from ign_lidar.classification_schema import ASPRSClass
    
You can use:
    from ign_lidar.core.classification.constants import ASPRSClass

This module also provides helper functions for common classification checks.

Note: The canonical ASPRSClass definition is in classification_schema.py.
This module is just a convenience wrapper with added utility functions.
"""

from typing import Dict

# Import the canonical ASPRSClass from classification_schema
from ...classification_schema import ASPRSClass

# Re-export ASPRSClass for convenience
__all__ = [
    "ASPRSClass",
    "get_class_name",
    "is_vegetation",
    "is_ground",
    "is_building",
    "is_water",
    "is_transport",
    "is_noise",
]


# Human-readable names for common classes (extends classification_schema)
_COMMON_CLASS_NAMES: Dict[int, str] = {
    int(ASPRSClass.UNCLASSIFIED): "Unclassified",
    int(ASPRSClass.GROUND): "Ground",
    int(ASPRSClass.LOW_VEGETATION): "Low Vegetation",
    int(ASPRSClass.MEDIUM_VEGETATION): "Medium Vegetation",
    int(ASPRSClass.HIGH_VEGETATION): "High Vegetation",
    int(ASPRSClass.BUILDING): "Building",
    int(ASPRSClass.LOW_POINT): "Low Point (Noise)",
    int(ASPRSClass.WATER): "Water",
    int(ASPRSClass.RAIL): "Rail",
    int(ASPRSClass.ROAD_SURFACE): "Road Surface",
    int(ASPRSClass.BRIDGE_DECK): "Bridge Deck",
    int(ASPRSClass.HIGH_NOISE): "High Noise",
}


def get_class_name(code: int) -> str:
    """
    Get human-readable name for ASPRS classification code.

    Args:
        code: ASPRS classification code

    Returns:
        Human-readable class name

    Example:
        >>> get_class_name(ASPRSClass.BUILDING)
        'Building'
        >>> get_class_name(6)
        'Building'
    """
    # Try to get from our common names first
    if code in _COMMON_CLASS_NAMES:
        return _COMMON_CLASS_NAMES[code]

    # Try to get enum name
    try:
        enum_val = ASPRSClass(code)
        return enum_val.name.replace("_", " ").title()
    except ValueError:
        return f"Unknown ({code})"


def is_vegetation(code: int) -> bool:
    """Check if code represents vegetation."""
    return code in (
        int(ASPRSClass.LOW_VEGETATION),
        int(ASPRSClass.MEDIUM_VEGETATION),
        int(ASPRSClass.HIGH_VEGETATION),
    )


def is_ground(code: int) -> bool:
    """Check if code represents ground."""
    return code == int(ASPRSClass.GROUND)


def is_building(code: int) -> bool:
    """Check if code represents building."""
    return code == int(ASPRSClass.BUILDING) or (
        50 <= code <= 69  # Extended building types
    )


def is_water(code: int) -> bool:
    """Check if code represents water."""
    return code == int(ASPRSClass.WATER)


def is_transport(code: int) -> bool:
    """Check if code represents transportation infrastructure."""
    return code in (
        int(ASPRSClass.ROAD_SURFACE),
        int(ASPRSClass.RAIL),
        int(ASPRSClass.BRIDGE_DECK),
    ) or (
        32 <= code <= 49  # Extended road types
    )


def is_noise(code: int) -> bool:
    """Check if code represents noise/outliers."""
    return code in (int(ASPRSClass.LOW_POINT), int(ASPRSClass.HIGH_NOISE))
