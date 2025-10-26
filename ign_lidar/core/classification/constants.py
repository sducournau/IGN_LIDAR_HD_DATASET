"""
Centralized ASPRS Classification Constants

This module provides canonical ASPRS LAS classification codes used throughout
the library. All classifiers should import from this module instead of defining
their own constants to ensure consistency.

ASPRS Standard Lidar Point Classes:
https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf
"""

from typing import Dict


class ASPRSClass:
    """
    ASPRS LAS classification codes (canonical).

    These codes follow the ASPRS LAS specification for point cloud
    classification. Use these constants throughout the codebase instead
    of hardcoded integers.

    Example:
        >>> from ign_lidar.core.classification.constants import ASPRSClass
        >>> labels[ground_mask] = ASPRSClass.GROUND
        >>> labels[building_mask] = ASPRSClass.BUILDING
    """

    # Standard ASPRS codes (0-18)
    NEVER_CLASSIFIED = 0
    UNCLASSIFIED = 1
    GROUND = 2
    LOW_VEGETATION = 3
    MEDIUM_VEGETATION = 4
    HIGH_VEGETATION = 5
    BUILDING = 6
    LOW_POINT = 7  # Noise
    RESERVED = 8  # Reserved (formerly Model Key-point)
    WATER = 9
    RAIL = 10
    ROAD_SURFACE = 11
    RESERVED_12 = 12  # Reserved
    WIRE_GUARD = 13  # Wire - Guard (Shield)
    WIRE_CONDUCTOR = 14  # Wire - Conductor (Phase)
    TRANSMISSION_TOWER = 15
    WIRE_STRUCTURE_CONNECTOR = 16  # Wire-structure Connector (e.g. Insulator)
    BRIDGE_DECK = 17
    HIGH_NOISE = 18

    # Extended classification codes (user-defined, 19-63 reserved for future)
    # IGN-specific extended codes for LOD2/LOD3 classification
    PARKING = 19  # Parking areas
    SPORTS = 20  # Sports facilities
    CEMETERY = 21  # Cemeteries
    POWER_LINE = 22  # Power lines (combined wire classes)
    AGRICULTURE = 23  # Agricultural areas

    # Aliases for common usage
    ROAD = ROAD_SURFACE  # Alias for clarity
    BRIDGE = BRIDGE_DECK  # Alias for clarity
    RAILWAY = RAIL  # Alias for clarity


# Human-readable names for each class
ASPRS_CLASS_NAMES: Dict[int, str] = {
    ASPRSClass.NEVER_CLASSIFIED: "Never Classified",
    ASPRSClass.UNCLASSIFIED: "Unclassified",
    ASPRSClass.GROUND: "Ground",
    ASPRSClass.LOW_VEGETATION: "Low Vegetation",
    ASPRSClass.MEDIUM_VEGETATION: "Medium Vegetation",
    ASPRSClass.HIGH_VEGETATION: "High Vegetation",
    ASPRSClass.BUILDING: "Building",
    ASPRSClass.LOW_POINT: "Low Point (Noise)",
    ASPRSClass.RESERVED: "Reserved",
    ASPRSClass.WATER: "Water",
    ASPRSClass.RAIL: "Rail",
    ASPRSClass.ROAD_SURFACE: "Road Surface",
    ASPRSClass.RESERVED_12: "Reserved",
    ASPRSClass.WIRE_GUARD: "Wire - Guard",
    ASPRSClass.WIRE_CONDUCTOR: "Wire - Conductor",
    ASPRSClass.TRANSMISSION_TOWER: "Transmission Tower",
    ASPRSClass.WIRE_STRUCTURE_CONNECTOR: "Wire-Structure Connector",
    ASPRSClass.BRIDGE_DECK: "Bridge Deck",
    ASPRSClass.HIGH_NOISE: "High Noise",
    ASPRSClass.PARKING: "Parking",
    ASPRSClass.SPORTS: "Sports",
    ASPRSClass.CEMETERY: "Cemetery",
    ASPRSClass.POWER_LINE: "Power Line",
    ASPRSClass.AGRICULTURE: "Agriculture",
}


# Reverse mapping: name -> code
ASPRS_NAME_TO_CODE: Dict[str, int] = {
    name.lower().replace(" ", "_"): code for code, name in ASPRS_CLASS_NAMES.items()
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
    return ASPRS_CLASS_NAMES.get(code, f"Unknown ({code})")


def get_class_code(name: str) -> int:
    """
    Get ASPRS classification code from name.

    Args:
        name: Class name (case-insensitive, spaces/underscores allowed)

    Returns:
        ASPRS classification code, or 0 if not found

    Example:
        >>> get_class_code("building")
        6
        >>> get_class_code("Road Surface")
        11
    """
    normalized = name.lower().replace(" ", "_")
    return ASPRS_NAME_TO_CODE.get(normalized, 0)


def is_vegetation(code: int) -> bool:
    """Check if code represents vegetation."""
    return code in (
        ASPRSClass.LOW_VEGETATION,
        ASPRSClass.MEDIUM_VEGETATION,
        ASPRSClass.HIGH_VEGETATION,
    )


def is_ground(code: int) -> bool:
    """Check if code represents ground."""
    return code == ASPRSClass.GROUND


def is_building(code: int) -> bool:
    """Check if code represents building."""
    return code == ASPRSClass.BUILDING


def is_water(code: int) -> bool:
    """Check if code represents water."""
    return code == ASPRSClass.WATER


def is_transport(code: int) -> bool:
    """Check if code represents transportation infrastructure."""
    return code in (
        ASPRSClass.ROAD_SURFACE,
        ASPRSClass.RAIL,
        ASPRSClass.BRIDGE_DECK,
    )


def is_noise(code: int) -> bool:
    """Check if code represents noise/outliers."""
    return code in (ASPRSClass.LOW_POINT, ASPRSClass.HIGH_NOISE)


# Export all public symbols
__all__ = [
    "ASPRSClass",
    "ASPRS_CLASS_NAMES",
    "ASPRS_NAME_TO_CODE",
    "get_class_name",
    "get_class_code",
    "is_vegetation",
    "is_ground",
    "is_building",
    "is_water",
    "is_transport",
    "is_noise",
]
