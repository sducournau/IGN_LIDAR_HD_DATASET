#!/usr/bin/env python3
"""
Quick validation test for ground truth optimization.

This script validates that:
1. The optimizer can be imported
2. Hardware detection works
3. Different methods can be instantiated
4. Basic labeling works
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Test 1: Checking imports...")
    
    try:
        from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
        logger.info("  ✅ GroundTruthOptimizer imported")
    except ImportError as e:
        logger.error(f"  ❌ Failed to import GroundTruthOptimizer: {e}")
        return False
    
    try:
        from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
        logger.info("  ✅ IGNGroundTruthFetcher imported")
    except ImportError as e:
        logger.error(f"  ❌ Failed to import IGNGroundTruthFetcher: {e}")
        return False
    
    return True


def test_hardware_detection():
    """Test hardware detection."""
    logger.info("\nTest 2: Hardware detection...")
    
    from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
    
    has_gpu = GroundTruthOptimizer._check_gpu()
    has_cuspatial = GroundTruthOptimizer._check_cuspatial()
    
    logger.info(f"  GPU available: {has_gpu}")
    logger.info(f"  cuSpatial available: {has_cuspatial}")
    
    if has_gpu:
        logger.info("  ✅ GPU acceleration available")
    else:
        logger.info("  ℹ️  GPU not available (CPU methods will be used)")
    
    return True


def test_method_selection():
    """Test automatic method selection."""
    logger.info("\nTest 3: Method selection...")
    
    from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
    
    optimizer = GroundTruthOptimizer(verbose=False)
    
    # Test different dataset sizes
    test_cases = [
        (100_000, 100, "small"),
        (1_000_000, 500, "medium"),
        (15_000_000, 1000, "large")
    ]
    
    for n_points, n_polygons, size in test_cases:
        method = optimizer.select_method(n_points, n_polygons)
        logger.info(f"  {size} dataset ({n_points:,} points, {n_polygons} polygons) → {method}")
    
    logger.info("  ✅ Method selection works")
    return True


def test_optimizer_creation():
    """Test creating optimizers with different methods."""
    logger.info("\nTest 4: Optimizer creation...")
    
    from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
    
    methods = ['strtree', 'vectorized']
    
    # Add GPU methods if available
    if GroundTruthOptimizer._check_gpu():
        methods.extend(['gpu', 'gpu_chunked'])
    
    for method in methods:
        try:
            optimizer = GroundTruthOptimizer(
                force_method=method,
                verbose=False
            )
            logger.info(f"  ✅ Created optimizer with method: {method}")
        except Exception as e:
            logger.error(f"  ❌ Failed to create optimizer with {method}: {e}")
            return False
    
    return True


def test_basic_labeling():
    """Test basic labeling functionality."""
    logger.info("\nTest 5: Basic labeling...")
    
    try:
        import numpy as np
        import geopandas as gpd
        from shapely.geometry import box
        from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
    except ImportError as e:
        logger.warning(f"  ⚠️  Skipping labeling test (missing dependency): {e}")
        return True
    
    # Create sample data
    n_points = 1000
    points = np.random.rand(n_points, 3) * 100
    
    # Create sample ground truth
    polygons = []
    for i in range(5):
        x, y = np.random.rand(2) * 80 + 10
        polygons.append(box(x, y, x+10, y+10))
    
    ground_truth_features = {
        'buildings': gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:2154')
    }
    
    # Test labeling
    optimizer = GroundTruthOptimizer(verbose=False)
    
    try:
        labels = optimizer.label_points(
            points=points,
            ground_truth_features=ground_truth_features
        )
        
        n_labeled = np.sum(labels > 0)
        logger.info(f"  Labeled {n_labeled} / {n_points} points")
        logger.info("  ✅ Basic labeling works")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Labeling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wfs_integration():
    """Test integration with WFS ground truth fetcher."""
    logger.info("\nTest 6: WFS integration...")
    
    try:
        from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
        
        fetcher = IGNGroundTruthFetcher(verbose=False)
        
        # Check that the method exists
        if not hasattr(fetcher, 'label_points_with_ground_truth'):
            logger.error("  ❌ Method label_points_with_ground_truth not found")
            return False
        
        logger.info("  ✅ WFS integration looks good")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ WFS integration check failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("="*80)
    logger.info("GROUND TRUTH OPTIMIZATION VALIDATION")
    logger.info("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Hardware Detection", test_hardware_detection),
        ("Method Selection", test_method_selection),
        ("Optimizer Creation", test_optimizer_creation),
        ("Basic Labeling", test_basic_labeling),
        ("WFS Integration", test_wfs_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("\n" + "-"*80)
    logger.info(f"Total: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("\n🎉 All tests passed! Ground truth optimization is ready to use.")
        return 0
    else:
        logger.error(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
