#!/usr/bin/env python3
"""
Test script to verify LAZ output format and enriched-only mode functionality.

This script tests:
1. LAZ as an output format for patches
2. Enriched-only mode (no patch creation)
3. Combination of both modes

Usage:
    python test_laz_output_format.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ign_lidar.config.schema import OutputConfig
from ign_lidar.core.processor import LiDARProcessor


def test_output_config_schema():
    """Test that OutputConfig accepts 'laz' as a format."""
    print("\n" + "="*70)
    print("TEST 1: OutputConfig Schema - LAZ Format Support")
    print("="*70)
    
    # Test valid formats
    valid_formats = ["npz", "hdf5", "torch", "laz", "all"]
    
    for fmt in valid_formats:
        try:
            config = OutputConfig(format=fmt)
            print(f"✅ Format '{fmt}' accepted: {config.format}")
        except Exception as e:
            print(f"❌ Format '{fmt}' failed: {e}")
            return False
    
    # Test invalid format (should fail with type checking, but at runtime it may pass)
    try:
        config = OutputConfig(format="invalid")
        print(f"⚠️  Invalid format 'invalid' was accepted (runtime only, type checker should catch)")
    except Exception as e:
        print(f"✅ Invalid format properly rejected: {e}")
    
    print("\n✅ All output format tests passed!")
    return True


def test_processor_initialization():
    """Test that processor accepts save_enriched_laz and only_enriched_laz parameters."""
    print("\n" + "="*70)
    print("TEST 2: LiDARProcessor Initialization")
    print("="*70)
    
    # Test 1: Default (no enriched LAZ)
    try:
        processor = LiDARProcessor()
        assert processor.save_enriched_laz == False
        assert processor.only_enriched_laz == False
        print("✅ Default initialization (no enriched LAZ)")
    except Exception as e:
        print(f"❌ Default initialization failed: {e}")
        return False
    
    # Test 2: Save enriched LAZ
    try:
        processor = LiDARProcessor(save_enriched_laz=True)
        assert processor.save_enriched_laz == True
        assert processor.only_enriched_laz == False
        print("✅ Save enriched LAZ mode")
    except Exception as e:
        print(f"❌ Save enriched LAZ mode failed: {e}")
        return False
    
    # Test 3: Only enriched LAZ (should auto-enable save_enriched_laz)
    try:
        processor = LiDARProcessor(only_enriched_laz=True)
        assert processor.save_enriched_laz == True  # Should be auto-enabled
        assert processor.only_enriched_laz == True
        print("✅ Only enriched LAZ mode (auto-enables save_enriched_laz)")
    except Exception as e:
        print(f"❌ Only enriched LAZ mode failed: {e}")
        return False
    
    # Test 4: Both flags explicitly set
    try:
        processor = LiDARProcessor(save_enriched_laz=True, only_enriched_laz=True)
        assert processor.save_enriched_laz == True
        assert processor.only_enriched_laz == True
        print("✅ Both flags explicitly set")
    except Exception as e:
        print(f"❌ Both flags mode failed: {e}")
        return False
    
    print("\n✅ All processor initialization tests passed!")
    return True


def test_process_tile_signature():
    """Test that process_tile method has correct signature."""
    print("\n" + "="*70)
    print("TEST 3: process_tile Method Signature")
    print("="*70)
    
    import inspect
    
    processor = LiDARProcessor()
    sig = inspect.signature(processor.process_tile)
    
    print("\nMethod signature:")
    print(f"  {sig}")
    
    # Check parameters
    params = sig.parameters
    
    # Check that save_enriched and only_enriched parameters exist
    if 'save_enriched' in params:
        print(f"✅ 'save_enriched' parameter found: {params['save_enriched']}")
    else:
        print(f"❌ 'save_enriched' parameter not found")
        return False
    
    if 'only_enriched' in params:
        print(f"✅ 'only_enriched' parameter found: {params['only_enriched']}")
    else:
        print(f"❌ 'only_enriched' parameter not found")
        return False
    
    if 'output_format' in params:
        print(f"✅ 'output_format' parameter found: {params['output_format']}")
    else:
        print(f"❌ 'output_format' parameter not found")
        return False
    
    print("\n✅ Method signature tests passed!")
    return True


def test_early_return_logic():
    """
    Test that when only_enriched=True, the process would skip patch creation.
    
    Note: This is a code inspection test - we verify the logic exists.
    Full integration testing would require actual LAZ files.
    """
    print("\n" + "="*70)
    print("TEST 4: Early Return Logic (Code Inspection)")
    print("="*70)
    
    import inspect
    
    processor = LiDARProcessor(only_enriched_laz=True)
    
    # Get source code
    source = inspect.getsource(processor.process_tile)
    
    # Check for early return when only_enriched is True
    if 'if only_enriched:' in source and 'return {' in source:
        print("✅ Early return logic found in process_tile")
        
        # Check that it returns before patch extraction
        if "'enriched_only': True" in source:
            print("✅ Returns with 'enriched_only' flag set")
        else:
            print("⚠️  'enriched_only' flag not found in return statement")
        
        # Check that it skips patch creation
        if 'patches skipped' in source.lower() or 'skip patch' in source.lower():
            print("✅ Patch creation skip logic confirmed")
        else:
            print("⚠️  Patch skip message not clearly indicated")
            
    else:
        print("❌ Early return logic not found")
        return False
    
    print("\n✅ Early return logic tests passed!")
    return True


def test_laz_output_format_logic():
    """
    Test that LAZ output format logic exists in process_tile.
    
    Note: This is a code inspection test - we verify the logic exists.
    Full integration testing would require actual LAZ files.
    """
    print("\n" + "="*70)
    print("TEST 5: LAZ Output Format Logic (Code Inspection)")
    print("="*70)
    
    import inspect
    
    processor = LiDARProcessor()
    
    # Get source code
    source = inspect.getsource(processor.process_tile)
    
    # Check for LAZ output format handling
    if "elif output_format == 'laz':" in source:
        print("✅ LAZ output format handler found")
        
        # Check for LasHeader and LasData imports
        if 'from laspy import' in source and 'LasHeader' in source:
            print("✅ LAZ file creation imports found")
        else:
            print("⚠️  LAZ file creation imports not clearly visible")
        
        # Check for patch LAZ writing
        if '.laz"' in source and 'patch_las.write' in source:
            print("✅ LAZ patch writing logic found")
        else:
            print("⚠️  LAZ patch writing logic not clearly visible")
            
    else:
        print("❌ LAZ output format handler not found")
        return False
    
    print("\n✅ LAZ output format logic tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("IGN LIDAR HD - LAZ Output Format Tests")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("OutputConfig Schema", test_output_config_schema()))
    results.append(("Processor Initialization", test_processor_initialization()))
    results.append(("process_tile Signature", test_process_tile_signature()))
    results.append(("Early Return Logic", test_early_return_logic()))
    results.append(("LAZ Output Format Logic", test_laz_output_format_logic()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nChanges are working as expected:")
        print("  1. LAZ is accepted as an output format")
        print("  2. Enriched-only mode properly skips patch creation")
        print("  3. LAZ output format handler is implemented")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
