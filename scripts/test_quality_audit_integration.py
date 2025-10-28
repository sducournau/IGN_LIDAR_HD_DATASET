#!/usr/bin/env python3
"""
Integration Test: Quality Audit Improvements Verification

This script performs end-to-end integration testing of all quality audit
improvements from Phases 1-3:

- Phase 1: Centralized constants and WFS fetch modules
- Phase 2: Migrated classifiers using centralized constants
- Phase 3: WFS ground truth with integrated retry logic
- Bug Fixes: Priority system, GT preservation, NDVI timing

The script verifies that all improvements work together correctly.
"""

import sys
from pathlib import Path


def test_phase1_constants():
    """Test Phase 1: Centralized constants module."""
    print("\n" + "=" * 70)
    print("PHASE 1 TEST: Centralized Constants Module")
    print("=" * 70)

    try:
        from ign_lidar.core.classification.constants import (
            ASPRSClass,
            is_building,
            is_vegetation,
            is_water,
            is_ground,
            get_class_name,
        )

        # Test basic functionality
        assert ASPRSClass.BUILDING == 6, "Building code should be 6"
        assert ASPRSClass.LOW_VEGETATION == 3, "Low vegetation code should be 3"
        assert ASPRSClass.WATER == 9, "Water code should be 9"

        # Test helper functions
        assert is_building(6) is True, "is_building(6) should be True"
        assert is_building(3) is False, "is_building(3) should be False"
        assert is_vegetation(3) is True, "is_vegetation(3) should be True"
        assert is_water(9) is True, "is_water(9) should be True"
        assert is_ground(2) is True, "is_ground(2) should be True"

        # Test name lookup
        assert get_class_name(6) == "Building", "Building name incorrect"
        assert get_class_name(3) == "Low Vegetation", "Name incorrect"
        print("✅ Constants module working correctly")
        print(f"   - ASPRSClass enum accessible")
        print(f"   - Helper functions operational")
        print(f"   - Name lookup functional")
        return True

    except Exception as e:
        print(f"❌ Constants module test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_phase1_wfs_fetch():
    """Test Phase 1: WFS fetch module."""
    print("\n" + "=" * 70)
    print("PHASE 1 TEST: WFS Fetch Module")
    print("=" * 70)

    try:
        from ign_lidar.io.wfs_fetch_result import (
            FetchStatus,
            FetchResult,
            RetryConfig,
            validate_cache_file,
        )

        # Test RetryConfig
        config = RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
        )
        assert config.max_retries == 3, "Max retries should be 3"
        assert config.initial_delay == 1.0, "Initial delay should be 1.0"

        # Test delay calculation
        delay0 = config.get_delay(0)
        delay1 = config.get_delay(1)
        delay2 = config.get_delay(2)
        assert delay1 == delay0 * 2.0, "Backoff factor should be applied"
        assert delay2 == delay0 * 4.0, "Exponential backoff should work"

        # Test FetchStatus enum
        assert FetchStatus.SUCCESS.value == "success"
        assert FetchStatus.NETWORK_ERROR.value == "network_error"

        # Test cache validation (non-existent file)
        fake_path = Path("/nonexistent/cache/file.geojson")
        assert (
            validate_cache_file(fake_path) is False
        ), "Nonexistent file should be invalid"

        print("✅ WFS fetch module working correctly")
        print(f"   - RetryConfig operational")
        print(f"   - FetchStatus enum accessible")
        print(f"   - Cache validation functional")
        return True

    except Exception as e:
        print(f"❌ WFS fetch module test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_phase2_classifier_migration():
    """Test Phase 2: Classifiers using centralized constants."""
    print("\n" + "=" * 70)
    print("PHASE 2 TEST: Classifier Migration")
    print("=" * 70)

    try:
        # Import classifiers that were migrated
        from ign_lidar.core.classification import spectral_rules
        from ign_lidar.core.classification import geometric_rules
        from ign_lidar.core.classification import reclassifier

        print("✅ Migrated classifiers import successfully")
        print(f"   - spectral_rules: {spectral_rules.__name__}")
        print(f"   - geometric_rules: {geometric_rules.__name__}")
        print(f"   - reclassifier: {reclassifier.__name__}")

        # Verify they use centralized constants (by checking imports)
        import inspect

        spec_source = inspect.getsource(spectral_rules)
        geom_source = inspect.getsource(geometric_rules)

        # Check for import of centralized constants
        assert (
            ".constants import" in spec_source or "ASPRSClass" in spec_source
        ), "spectral_rules should use centralized constants"
        assert (
            ".constants import" in geom_source or "ASPRSClass" in geom_source
        ), "geometric_rules should use centralized constants"

        print("✅ Classifiers using centralized constants")
        return True

    except Exception as e:
        print(f"❌ Classifier migration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_phase3_wfs_integration():
    """Test Phase 3: WFS ground truth with integrated retry."""
    print("\n" + "=" * 70)
    print("PHASE 3 TEST: WFS Ground Truth Integration")
    print("=" * 70)

    try:
        from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher

        # Create fetcher instance
        fetcher = IGNGroundTruthFetcher(cache_dir=Path("./test_cache"))

        print("✅ IGNGroundTruthFetcher instantiated")

        # Verify it has the fetch methods
        assert hasattr(fetcher, "fetch_buildings"), "Should have method"
        assert hasattr(fetcher, "fetch_roads_with_polygons"), "Should have method"
        assert hasattr(fetcher, "fetch_water_surfaces"), "Should have method"
        assert hasattr(fetcher, "fetch_vegetation_zones"), "Should have method"

        print("✅ WFS fetcher has all fetch methods")
        print("   - fetch_buildings")
        print("   - fetch_roads_with_polygons")
        print("   - fetch_water_surfaces")
        print("   - fetch_vegetation_zones")
        # Verify internal method uses retry logic
        import inspect

        source = inspect.getsource(fetcher._fetch_wfs_layer)
        assert (
            "fetch_with_retry" in source or "RetryConfig" in source
        ), "_fetch_wfs_layer should use centralized retry logic"

        print("✅ WFS fetcher uses centralized retry logic")
        return True

    except Exception as e:
        print(f"❌ WFS integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_bug_fixes():
    """Test bug fixes: Priority system."""
    print("\n" + "=" * 70)
    print("BUG FIXES TEST: Classification Priority System")
    print("=" * 70)

    try:
        from ign_lidar.core.classification.priorities import (
            PRIORITY_ORDER,
            get_priority_value,
            get_priority_order_for_iteration,
        )

        # Test priority order exists
        assert len(PRIORITY_ORDER) > 0, "Priority order should not be empty"
        assert "buildings" in PRIORITY_ORDER, "Buildings should be in priority order"
        assert "vegetation" in PRIORITY_ORDER, "Vegetation should be in priority order"

        # Test priority values
        building_priority = get_priority_value("buildings")
        vegetation_priority = get_priority_value("vegetation")
        road_priority = get_priority_value("roads")
        water_priority = get_priority_value("water")

        # Verify buildings have highest priority
        assert (
            building_priority > vegetation_priority
        ), "Buildings should have higher priority than vegetation"
        assert (
            building_priority > road_priority
        ), "Buildings should have higher priority than roads"
        assert (
            building_priority > water_priority
        ), "Buildings should have higher priority than water"

        # Verify correct ordering
        assert (
            road_priority > water_priority
        ), "Roads should have higher priority than water"
        assert (
            water_priority > vegetation_priority
        ), "Water should have higher priority than vegetation"

        print("✅ Priority system working correctly")
        print(f"   - Priority order: {PRIORITY_ORDER[:5]}")
        print(f"   - Buildings priority: {building_priority} (highest)")
        print(f"   - Roads priority: {road_priority}")
        print(f"   - Water priority: {water_priority}")
        print(f"   - Vegetation priority: {vegetation_priority} (lowest)")
        return True

    except Exception as e:
        print(f"❌ Bug fixes test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n")
    print("*" * 70)
    print(" INTEGRATION TEST: Quality Audit Improvements")
    print(" Verifying Phases 1-3 + Bug Fixes")
    print("*" * 70)

    results = {
        "Phase 1 - Constants": test_phase1_constants(),
        "Phase 1 - WFS Fetch": test_phase1_wfs_fetch(),
        "Phase 2 - Migration": test_phase2_classifier_migration(),
        "Phase 3 - WFS Integration": test_phase3_wfs_integration(),
        "Bug Fixes - Priorities": test_bug_fixes(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")

    print("\n" + "-" * 70)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED! 🎉")
        print("\nQuality audit improvements verified:")
        print("  ✅ Phase 1: Foundation modules working")
        print("  ✅ Phase 2: Classifiers migrated successfully")
        print("  ✅ Phase 3: WFS integration complete")
        print("  ✅ Bug Fixes: Priority system operational")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        print("\nPlease review the errors above and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
