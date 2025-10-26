"""
Tests for centralized ASPRS classification constants wrapper.

This validates that the constants wrapper provides a convenient import path
and helper functions for the ASPRSClass from classification_schema.py.
"""

import pytest
from ign_lidar.core.classification.constants import (
    ASPRSClass,
    get_class_name,
    is_vegetation,
    is_ground,
    is_building,
    is_water,
    is_transport,
    is_noise,
)


class TestASPRSClassWrapper:
    """Test ASPRS class wrapper provides access to IntEnum."""

    def test_wrapper_exports_enum(self):
        """Test that wrapper exports ASPRSClass from classification_schema."""
        from enum import IntEnum
        assert issubclass(ASPRSClass, IntEnum)

    def test_standard_codes_accessible(self):
        """Test standard ASPRS codes are accessible through wrapper."""
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
        assert ASPRSClass.HIGH_NOISE == 18

    def test_enum_values_work_as_ints(self):
        """Test that enum values work as integers."""
        # Can be used in comparisons
        assert ASPRSClass.BUILDING == 6
        assert ASPRSClass.GROUND < ASPRSClass.BUILDING

        # Can be used in arithmetic
        code = int(ASPRSClass.BUILDING)
        assert code == 6

        # Can be used as array indices
        import numpy as np
        arr = np.zeros(10)
        arr[ASPRSClass.GROUND] = 1.0
        assert arr[2] == 1.0


class TestGetClassName:
    """Test get_class_name helper function."""

    def test_get_class_name_common(self):
        """Test get_class_name for common ASPRS codes."""
        assert get_class_name(ASPRSClass.BUILDING) == "Building"
        assert get_class_name(ASPRSClass.GROUND) == "Ground"
        assert get_class_name(ASPRSClass.LOW_VEGETATION) == "Low Vegetation"
        assert get_class_name(ASPRSClass.WATER) == "Water"
        assert get_class_name(ASPRSClass.ROAD_SURFACE) == "Road Surface"

    def test_get_class_name_with_int(self):
        """Test get_class_name works with integer codes."""
        assert get_class_name(6) == "Building"
        assert get_class_name(2) == "Ground"
        assert get_class_name(9) == "Water"

    def test_get_class_name_unknown(self):
        """Test get_class_name for unknown codes."""
        result = get_class_name(255)
        assert "Unknown" in result or "255" in str(result)

        result = get_class_name(999)
        assert "Unknown" in result or "999" in str(result)


class TestClassificationHelpers:
    """Test helper functions for classification checks."""

    def test_is_vegetation(self):
        """Test is_vegetation helper."""
        # Should identify vegetation classes
        assert is_vegetation(ASPRSClass.LOW_VEGETATION)
        assert is_vegetation(ASPRSClass.MEDIUM_VEGETATION)
        assert is_vegetation(ASPRSClass.HIGH_VEGETATION)

        # Should work with int codes
        assert is_vegetation(3)
        assert is_vegetation(4)
        assert is_vegetation(5)

        # Should reject non-vegetation
        assert not is_vegetation(ASPRSClass.BUILDING)
        assert not is_vegetation(ASPRSClass.GROUND)
        assert not is_vegetation(ASPRSClass.WATER)

    def test_is_ground(self):
        """Test is_ground helper."""
        assert is_ground(ASPRSClass.GROUND)
        assert is_ground(2)

        assert not is_ground(ASPRSClass.BUILDING)
        assert not is_ground(ASPRSClass.WATER)
        assert not is_ground(ASPRSClass.LOW_VEGETATION)

    def test_is_building(self):
        """Test is_building helper including extended building types."""
        assert is_building(ASPRSClass.BUILDING)
        assert is_building(6)

        # Extended building types (50-69)
        assert is_building(50)
        assert is_building(60)
        assert is_building(69)
        assert not is_building(49)  # Just before range
        assert not is_building(70)  # Just after range

        assert not is_building(ASPRSClass.GROUND)
        assert not is_building(ASPRSClass.WATER)

    def test_is_water(self):
        """Test is_water helper."""
        assert is_water(ASPRSClass.WATER)
        assert is_water(9)

        assert not is_water(ASPRSClass.BUILDING)
        assert not is_water(ASPRSClass.GROUND)

    def test_is_transport(self):
        """Test is_transport helper including extended road types."""
        # Standard transport
        assert is_transport(ASPRSClass.ROAD_SURFACE)
        assert is_transport(ASPRSClass.RAIL)
        assert is_transport(ASPRSClass.BRIDGE_DECK)

        # Extended road types (32-49)
        assert is_transport(32)
        assert is_transport(40)
        assert is_transport(49)
        assert not is_transport(31)  # Just before range
        assert not is_transport(50)  # Just after range

        assert not is_transport(ASPRSClass.BUILDING)
        assert not is_transport(ASPRSClass.WATER)

    def test_is_noise(self):
        """Test is_noise helper."""
        assert is_noise(ASPRSClass.LOW_POINT)
        assert is_noise(ASPRSClass.HIGH_NOISE)
        assert is_noise(7)
        assert is_noise(18)

        assert not is_noise(ASPRSClass.BUILDING)
        assert not is_noise(ASPRSClass.WATER)


class TestIntegration:
    """Integration tests for wrapper usage."""

    def test_typical_usage_pattern(self):
        """Test typical usage pattern in classification code."""
        # Simulate classification logic
        import numpy as np
        labels = np.array([1, 2, 6, 9, 3, 4, 5])

        # Count buildings
        building_mask = labels == ASPRSClass.BUILDING
        assert building_mask.sum() == 1

        # Count vegetation
        veg_mask = np.array([is_vegetation(label) for label in labels])
        assert veg_mask.sum() == 3  # LOW, MEDIUM, HIGH

    def test_wrapper_import_convenience(self):
        """Test that wrapper provides convenient import path."""
        # Can import from wrapper
        from ign_lidar.core.classification.constants import ASPRSClass as WrapperClass

        # Can also import from original (should be same)
        from ign_lidar.classification_schema import ASPRSClass as OriginalClass

        # Should be the exact same class
        assert WrapperClass is OriginalClass

    def test_can_use_in_classifiers(self):
        """Test that constants work in typical classifier code."""
        import numpy as np

        # Simulate point cloud with classifications
        classifications = np.array([
            ASPRSClass.GROUND,
            ASPRSClass.BUILDING,
            ASPRSClass.LOW_VEGETATION,
            ASPRSClass.WATER,
        ])

        # Should work with numpy operations
        ground_points = classifications == ASPRSClass.GROUND
        assert ground_points.sum() == 1

        # Should work with helper functions
        veg_points = np.array([is_vegetation(c) for c in classifications])
        assert veg_points.sum() == 1
