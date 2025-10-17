#!/usr/bin/env python3
"""
Quick test script for PresetConfigLoader (Week 3 - Task 4)

Tests the new preset-based configuration system.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.config import PresetConfigLoader, load_config_with_preset


def test_list_presets():
    """Test listing available presets."""
    print("\n" + "=" * 80)
    print("TEST 1: List Available Presets")
    print("=" * 80)
    
    loader = PresetConfigLoader(verbose=False)
    presets = loader.list_presets()
    
    print(f"Found {len(presets)} presets:")
    for preset in presets:
        print(f"  - {preset}")
    
    assert len(presets) == 5, f"Expected 5 presets, found {len(presets)}"
    assert 'minimal' in presets
    assert 'lod2' in presets
    assert 'lod3' in presets
    assert 'asprs' in presets
    assert 'full' in presets
    
    print("✓ PASSED")
    return True


def test_load_base_only():
    """Test loading just base configuration."""
    print("\n" + "=" * 80)
    print("TEST 2: Load Base Configuration Only")
    print("=" * 80)
    
    loader = PresetConfigLoader(verbose=True)
    config = loader.load()
    
    # Check required sections exist
    assert 'processor' in config
    assert 'features' in config
    assert 'data_sources' in config
    assert 'output' in config
    
    # Check some default values
    assert config['processor']['lod_level'] == 'ASPRS'
    assert config['features']['mode'] == 'lod2'
    assert config['output']['format'] == 'laz'
    
    print("✓ PASSED")
    return True


def test_load_preset():
    """Test loading with preset."""
    print("\n" + "=" * 80)
    print("TEST 3: Load with Preset (minimal)")
    print("=" * 80)
    
    loader = PresetConfigLoader(verbose=True)
    config = loader.load(preset="minimal")
    
    # Check preset overrides applied
    assert config['features']['mode'] == 'minimal'
    assert config['features']['k_neighbors'] == 10
    assert config['features']['compute_curvature'] == False
    
    # Check base values still present
    assert 'processor' in config
    assert 'output' in config
    
    print("✓ PASSED")
    return True


def test_load_lod2_preset():
    """Test loading LOD2 preset."""
    print("\n" + "=" * 80)
    print("TEST 4: Load with Preset (lod2)")
    print("=" * 80)
    
    config = load_config_with_preset(preset="lod2", verbose=True)
    
    # Check LOD2 settings
    assert config['processor']['lod_level'] == 'LOD2'
    assert config['features']['mode'] == 'lod2'
    assert config['features']['k_neighbors'] == 20
    assert config['features']['compute_architectural'] == True
    
    print("✓ PASSED")
    return True


def test_overrides():
    """Test CLI overrides."""
    print("\n" + "=" * 80)
    print("TEST 5: CLI Overrides")
    print("=" * 80)
    
    loader = PresetConfigLoader(verbose=True)
    config = loader.load(
        preset="lod2",
        overrides={
            "processor.gpu_batch_size": 2000000,
            "features.k_neighbors": 30
        }
    )
    
    # Check overrides applied
    assert config['processor']['gpu_batch_size'] == 2000000
    assert config['features']['k_neighbors'] == 30
    
    # Check preset values still present
    assert config['processor']['lod_level'] == 'LOD2'
    assert config['features']['mode'] == 'lod2'
    
    print("✓ PASSED")
    return True


def test_preset_info():
    """Test getting preset information."""
    print("\n" + "=" * 80)
    print("TEST 6: Get Preset Info")
    print("=" * 80)
    
    loader = PresetConfigLoader(verbose=False)
    info = loader.get_preset_info("lod2")
    
    print(f"Preset: {info['name']}")
    print(f"Use case: {info['use_case']}")
    print(f"Speed: {info['speed']}")
    
    assert info['name'] == 'lod2'
    assert 'Building modeling' in info['use_case']
    
    print("✓ PASSED")
    return True


def test_print_presets():
    """Test printing all presets."""
    print("\n" + "=" * 80)
    print("TEST 7: Print All Presets")
    print("=" * 80)
    
    loader = PresetConfigLoader(verbose=False)
    loader.print_presets()
    
    print("✓ PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PRESET CONFIG LOADER - TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_list_presets,
        test_load_base_only,
        test_load_preset,
        test_load_lod2_preset,
        test_overrides,
        test_preset_info,
        test_print_presets,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
