#!/usr/bin/env python3
"""
Simple test script to verify the IGN LiDAR HD package installation.
"""

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing package imports...")
    
    try:
        import ign_lidar
        print("✓ Main package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import main package: {e}")
        return False
        
    try:
        from ign_lidar.core.processor import LiDARProcessor
        print("✓ LiDARProcessor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import LiDARProcessor: {e}")
        return False
        
    try:
        from ign_lidar.preprocessing.preprocessing import statistical_outlier_removal
        print("✓ Preprocessing functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import preprocessing: {e}")
        return False
        
    try:
        from ign_lidar.downloader import IGNLiDARDownloader
        print("✓ Downloader imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import downloader: {e}")
        return False
        
    return True

def test_version():
    """Test package version."""
    print("\nTesting package version...")
    try:
        import ign_lidar
        version = ign_lidar.__version__
        print(f"✓ Package version: {version}")
        return True
    except Exception as e:
        print(f"✗ Failed to get package version: {e}")
        return False

def test_sample_data():
    """Test if sample data exists."""
    from pathlib import Path
    print("\nTesting sample data availability...")
    
    sample_dir = Path("data/sample_laz")
    if sample_dir.exists():
        laz_files = list(sample_dir.glob("*.laz"))
        print(f"✓ Found {len(laz_files)} sample LAZ files:")
        for laz_file in laz_files:
            print(f"  - {laz_file.name}")
        return True
    else:
        print("✗ Sample data directory not found")
        return False

def test_cli_functionality():
    """Test CLI entry points."""
    print("\nTesting CLI functionality...")
    
    try:
        from ign_lidar.cli.hydra_main import main
        print("✓ CLI main function accessible")
    except ImportError as e:
        print(f"✗ Failed to import CLI: {e}")
        return False
        
    try:
        from ign_lidar.io.qgis_converter import main as qgis_main
        print("✓ QGIS converter CLI accessible")
        return True
    except ImportError as e:
        print(f"✗ Failed to import QGIS CLI: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("IGN LiDAR HD Package Installation Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    all_tests_passed &= test_imports()
    all_tests_passed &= test_version()
    all_tests_passed &= test_sample_data()
    all_tests_passed &= test_cli_functionality()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - Installation is working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Check the output above")
    print("=" * 60)