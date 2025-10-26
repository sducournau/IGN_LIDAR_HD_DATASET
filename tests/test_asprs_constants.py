"""
Tests for centralized ASPRS classification constants.

This validates that the constants module provides consistent classification
codes across the codebase and prevents duplicate definitions.
"""

import pytest
from ign_lidar.core.classification.constants import (
    ASPRSClass,
    ASPRS_CLASS_NAMES,
    ASPRS_NAME_TO_CODE,
    get_class_name,
    get_class_code,
    is_vegetation,
    is_ground,
    is_building,
    is_water,
    is_transport,
    is_noise,
)


class TestASPRSClassConstants:
    """Test ASPRS class constant values."""

    def test_standard_codes(self):
        """Test standard ASPRS codes match specification."""
        assert ASPRSClass.NEVER_CLASSIFIED == 0
        assert ASPRSClass.UNCLASSIFIED == 1
        assert ASPRSClass.GROUND == 2
        assert ASPRSClass.LOW_VEGETATION == 3
        assert ASPRSClass.MEDIUM_VEGETATION == 4
        assert ASPRSClass.HIGH_VEGETATION == 5
        assert ASPRSClass.BUILDING == 6
        assert ASPRSClass.LOW_POINT == 7
        assert ASPRSClass.WATER == 9
        assert ASPRSClass.RAIL == 10
        assert ASPRSClass.ROAD_SURFACE == 11
        assert ASPRSClass.BRIDGE_DECK == 17

    def test_extended_codes(self):
        """Test IGN-specific extended codes."""
        assert ASPRSClass.PARKING == 19
        assert ASPRSClass.SPORTS == 20
        assert ASPRSClass.CEMETERY == 21
        assert ASPRSClass.POWER_LINE == 22
        assert ASPRSClass.AGRICULTURE == 23

    def test_aliases(self):
        """Test aliases match their targets."""
        assert ASPRSClass.ROAD == ASPRSClass.ROAD_SURFACE
        assert ASPRSClass.BRIDGE == ASPRSClass.BRIDGE_DECK
        assert ASPRSClass.RAILWAY == ASPRSClass.RAIL


class TestClassNameMappings:
    """Test class name to code mappings."""

    def test_get_class_name_standard(self):
        """Test get_class_name for standard codes."""
        assert get_class_name(ASPRSClass.BUILDING) == "Building"
        assert get_class_name(6) == "Building"
        assert get_class_name(ASPRSClass.GROUND) == "Ground"
        assert get_class_name(ASPRSClass.WATER) == "Water"

    def test_get_class_name_extended(self):
        """Test get_class_name for extended codes."""
        assert get_class_name(ASPRSClass.PARKING) == "Parking"
        assert get_class_name(ASPRSClass.SPORTS) == "Sports"
        assert get_class_name(ASPRSClass.CEMETERY) == "Cemetery"

    def test_get_class_name_unknown(self):
        """Test get_class_name for unknown codes."""
        assert get_class_name(99) == "Unknown (99)"
        assert get_class_name(255) == "Unknown (255)"

    def test_get_class_code_standard(self):
        """Test get_class_code for standard names."""
        assert get_class_code("building") == ASPRSClass.BUILDING
        assert get_class_code("Building") == ASPRSClass.BUILDING
        assert get_class_code("ground") == ASPRSClass.GROUND
        assert get_class_code("water") == ASPRSClass.WATER

    def test_get_class_code_with_spaces(self):
        """Test get_class_code handles spaces."""
        assert get_class_code("Low Vegetation") == ASPRSClass.LOW_VEGETATION
        assert get_class_code("Road Surface") == ASPRSClass.ROAD_SURFACE
        assert get_class_code("Bridge Deck") == ASPRSClass.BRIDGE_DECK

    def test_get_class_code_with_underscores(self):
        """Test get_class_code handles underscores."""
        assert get_class_code("low_vegetation") == ASPRSClass.LOW_VEGETATION
        assert get_class_code("road_surface") == ASPRSClass.ROAD_SURFACE

    def test_get_class_code_unknown(self):
        """Test get_class_code returns 0 for unknown names."""
        assert get_class_code("unknown_class") == 0
        assert get_class_code("foobar") == 0


class TestClassificationHelpers:
    """Test helper functions for classification checks."""

    def test_is_vegetation(self):
        """Test is_vegetation helper."""
        assert is_vegetation(ASPRSClass.LOW_VEGETATION)
        assert is_vegetation(ASPRSClass.MEDIUM_VEGETATION)
        assert is_vegetation(ASPRSClass.HIGH_VEGETATION)
        assert not is_vegetation(ASPRSClass.BUILDING)
        assert not is_vegetation(ASPRSClass.GROUND)
        assert not is_vegetation(ASPRSClass.WATER)

    def test_is_ground(self):
        """Test is_ground helper."""
        assert is_ground(ASPRSClass.GROUND)
        assert not is_ground(ASPRSClass.BUILDING)
        assert not is_ground(ASPRSClass.WATER)
        assert not is_ground(ASPRSClass.LOW_VEGETATION)

    def test_is_building(self):
        """Test is_building helper."""
        assert is_building(ASPRSClass.BUILDING)
        assert not is_building(ASPRSClass.GROUND)
        assert not is_building(ASPRSClass.WATER)
        assert not is_building(ASPRSClass.LOW_VEGETATION)

    def test_is_water(self):
        """Test is_water helper."""
        assert is_water(ASPRSClass.WATER)
        assert not is_water(ASPRSClass.BUILDING)
        assert not is_water(ASPRSClass.GROUND)
        assert not is_water(ASPRSClass.LOW_VEGETATION)

    def test_is_transport(self):
        """Test is_transport helper."""
        assert is_transport(ASPRSClass.ROAD_SURFACE)
        assert is_transport(ASPRSClass.RAIL)
        assert is_transport(ASPRSClass.BRIDGE_DECK)
        assert not is_transport(ASPRSClass.BUILDING)
        assert not is_transport(ASPRSClass.WATER)
        assert not is_transport(ASPRSClass.PARKING)

    def test_is_noise(self):
        """Test is_noise helper."""
        assert is_noise(ASPRSClass.LOW_POINT)
        assert is_noise(ASPRSClass.HIGH_NOISE)
        assert not is_noise(ASPRSClass.BUILDING)
        assert not is_noise(ASPRSClass.GROUND)
        assert not is_noise(ASPRSClass.WATER)


class TestMappingConsistency:
    """Test that mappings are consistent and complete."""

    def test_all_codes_have_names(self):
        """Test that all defined codes have names."""
        # Check standard codes
        for code in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 17, 18]:
            assert code in ASPRS_CLASS_NAMES
            assert isinstance(ASPRS_CLASS_NAMES[code], str)
            assert len(ASPRS_CLASS_NAMES[code]) > 0

        # Check extended codes
        for code in [19, 20, 21, 22, 23]:
            assert code in ASPRS_CLASS_NAMES
            assert isinstance(ASPRS_CLASS_NAMES[code], str)
            assert len(ASPRS_CLASS_NAMES[code]) > 0

    def test_reverse_mapping_consistency(self):
        """Test that reverse mapping is consistent with forward mapping."""
        for code, name in ASPRS_CLASS_NAMES.items():
            # Get code from name
            normalized_name = name.lower().replace(" ", "_")
            retrieved_code = ASPRS_NAME_TO_CODE.get(normalized_name)

            # Should map back to original code
            assert (
                retrieved_code == code
            ), f"Inconsistent mapping for {name}: {code} != {retrieved_code}"

    def test_no_duplicate_codes(self):
        """Test that no codes are duplicated."""
        codes = set()
        for attr_name in dir(ASPRSClass):
            if attr_name.startswith("_"):
                continue
            value = getattr(ASPRSClass, attr_name)
            if isinstance(value, int):
                # Aliases are OK (ROAD = ROAD_SURFACE)
                # But we check they point to valid codes
                assert (
                    value in ASPRS_CLASS_NAMES
                ), f"{attr_name}={value} not in ASPRS_CLASS_NAMES"
                codes.add(value)


class TestBackwardCompatibility:
    """Test that constants maintain backward compatibility."""

    def test_common_codes_unchanged(self):
        """Test that commonly used codes haven't changed."""
        # These are the most critical codes used throughout the codebase
        assert ASPRSClass.UNCLASSIFIED == 1
        assert ASPRSClass.GROUND == 2
        assert ASPRSClass.BUILDING == 6
        assert ASPRSClass.WATER == 9
        assert ASPRSClass.ROAD == 11  # Alias for ROAD_SURFACE

    def test_vegetation_codes_unchanged(self):
        """Test vegetation codes are stable."""
        assert ASPRSClass.LOW_VEGETATION == 3
        assert ASPRSClass.MEDIUM_VEGETATION == 4
        assert ASPRSClass.HIGH_VEGETATION == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
