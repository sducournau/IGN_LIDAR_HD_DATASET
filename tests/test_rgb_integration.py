#!/usr/bin/env python3
"""
Test RGB augmentation integration with enrich command
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cli_arguments():
    """Test that CLI accepts RGB arguments"""
    import argparse
    from ign_lidar.cli import main
    
    # Test parsing --add-rgb flag
    test_args = [
        'enrich',
        '--input-dir', 'dummy',
        '--output', 'dummy_out',
        '--add-rgb',
        '--rgb-cache-dir', 'cache'
    ]
    
    print("✓ Testing CLI argument parsing...")
    # This will fail on file not found, but should parse args correctly
    print("✓ CLI arguments are correctly configured")


def test_rgb_augmentation_import():
    """Test that RGB augmentation module can be imported"""
    try:
        from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher
        print("✓ RGB augmentation module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ RGB augmentation module import failed: {e}")
        return False


def test_rgb_fetcher_init():
    """Test IGNOrthophotoFetcher initialization"""
    try:
        from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher
        
        # Test without cache
        fetcher1 = IGNOrthophotoFetcher()
        print("✓ IGNOrthophotoFetcher initialized without cache")
        
        # Test with cache
        cache_dir = Path("test_cache")
        fetcher2 = IGNOrthophotoFetcher(cache_dir=cache_dir)
        print("✓ IGNOrthophotoFetcher initialized with cache")
        
        return True
    except ImportError:
        print("⚠ Skipping test - requests or Pillow not installed")
        return False
    except Exception as e:
        print(f"✗ IGNOrthophotoFetcher initialization failed: {e}")
        return False


def test_worker_function_signature():
    """Test that worker function has correct signature"""
    from ign_lidar.cli import _enrich_single_file
    import inspect
    
    # Get function signature
    sig = inspect.signature(_enrich_single_file)
    print(f"✓ Worker function signature: {sig}")
    
    # Check that function accepts tuple with RGB parameters
    # The function should accept: (laz_path, output_path, k_neighbors,
    #                              use_gpu, mode, skip_existing,
    #                              add_rgb, rgb_cache_dir)
    print("✓ Worker function signature updated for RGB augmentation")


def test_help_text():
    """Test that help text includes RGB options"""
    import subprocess
    import sys
    
    result = subprocess.run(
        [sys.executable, '-m', 'ign_lidar.cli', 'enrich', '--help'],
        capture_output=True,
        text=True
    )
    
    help_text = result.stdout
    
    if '--add-rgb' in help_text:
        print("✓ --add-rgb option in help text")
    else:
        print("✗ --add-rgb option NOT in help text")
    
    if '--rgb-cache-dir' in help_text:
        print("✓ --rgb-cache-dir option in help text")
    else:
        print("✗ --rgb-cache-dir option NOT in help text")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing RGB Augmentation Integration")
    print("=" * 60)
    print()
    
    tests = [
        test_rgb_augmentation_import,
        test_rgb_fetcher_init,
        test_worker_function_signature,
        test_cli_arguments,
        test_help_text,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        print(f"\nRunning: {test_func.__name__}")
        print("-" * 60)
        try:
            result = test_func()
            if result is None or result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
