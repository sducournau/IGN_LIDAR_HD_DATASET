#!/usr/bin/env python3
"""
Verification Script for Ground Truth Optimization Integration

This script verifies that the optimization module is properly integrated
and all imports work correctly.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_module_structure():
    """Verify module structure."""
    logger.info("="*80)
    logger.info("CHECKING MODULE STRUCTURE")
    logger.info("="*80)
    
    try:
        import ign_lidar.optimization
        logger.info("✅ ign_lidar.optimization module exists")
    except ImportError as e:
        logger.error(f"❌ Failed to import ign_lidar.optimization: {e}")
        return False
    
    # Check submodules
    submodules = ['auto_select', 'strtree', 'vectorized', 'gpu', 'prefilter']
    for submodule in submodules:
        try:
            module = __import__(f'ign_lidar.optimization.{submodule}', fromlist=[submodule])
            logger.info(f"✅ ign_lidar.optimization.{submodule} exists")
        except ImportError as e:
            logger.error(f"❌ Failed to import ign_lidar.optimization.{submodule}: {e}")
            return False
    
    return True


def check_public_api():
    """Verify public API exports."""
    logger.info("\n" + "="*80)
    logger.info("CHECKING PUBLIC API")
    logger.info("="*80)
    
    try:
        from ign_lidar.optimization import (
            auto_optimize,
            apply_strtree_optimization,
            apply_vectorized_optimization,
            apply_gpu_optimization,
            apply_prefilter_optimization,
        )
        logger.info("✅ All public API functions imported successfully")
        
        # Verify they are callable
        assert callable(auto_optimize), "auto_optimize is not callable"
        assert callable(apply_strtree_optimization), "apply_strtree_optimization is not callable"
        assert callable(apply_vectorized_optimization), "apply_vectorized_optimization is not callable"
        assert callable(apply_gpu_optimization), "apply_gpu_optimization is not callable"
        assert callable(apply_prefilter_optimization), "apply_prefilter_optimization is not callable"
        
        logger.info("✅ All public API functions are callable")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import public API: {e}")
        return False
    except AssertionError as e:
        logger.error(f"❌ API validation failed: {e}")
        return False


def check_dependencies():
    """Check available optimization dependencies."""
    logger.info("\n" + "="*80)
    logger.info("CHECKING DEPENDENCIES")
    logger.info("="*80)
    
    # Check required dependencies
    required = {
        'numpy': 'Required',
        'shapely': 'Required (STRtree)',
        'geopandas': 'Required (Vectorized)',
    }
    
    for package, description in required.items():
        try:
            __import__(package)
            logger.info(f"✅ {package:15} - {description} - INSTALLED")
        except ImportError:
            logger.warning(f"⚠️  {package:15} - {description} - MISSING")
    
    # Check optional dependencies
    optional = {
        'cupy': 'Optional (GPU)',
        'cuspatial': 'Optional (GPU Spatial)',
    }
    
    for package, description in optional.items():
        try:
            __import__(package)
            logger.info(f"✅ {package:15} - {description} - INSTALLED")
        except ImportError:
            logger.info(f"ℹ️  {package:15} - {description} - NOT INSTALLED (optional)")
    
    return True


def check_auto_select():
    """Test automatic optimization selection."""
    logger.info("\n" + "="*80)
    logger.info("TESTING AUTO-SELECT")
    logger.info("="*80)
    
    try:
        from ign_lidar.optimization import auto_optimize
        
        # Test with verbose=False to avoid cluttering output
        level = auto_optimize(verbose=False)
        logger.info(f"✅ auto_optimize() selected: {level}")
        
        # Verify it's a valid level
        valid_levels = ['gpu', 'vectorized', 'strtree', 'prefilter', 'original']
        assert level in valid_levels, f"Invalid level selected: {level}"
        logger.info(f"✅ Selected level is valid: {level}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ auto_optimize() failed: {e}")
        return False


def check_file_locations():
    """Verify files are in correct locations."""
    logger.info("\n" + "="*80)
    logger.info("CHECKING FILE LOCATIONS")
    logger.info("="*80)
    
    import os
    
    # Check optimization module files
    optimization_files = [
        'ign_lidar/optimization/__init__.py',
        'ign_lidar/optimization/README.md',
        'ign_lidar/optimization/auto_select.py',
        'ign_lidar/optimization/strtree.py',
        'ign_lidar/optimization/vectorized.py',
        'ign_lidar/optimization/gpu.py',
        'ign_lidar/optimization/prefilter.py',
    ]
    
    # Get repo root (script is in repo root)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    
    for file_path in optimization_files:
        full_path = os.path.join(repo_root, file_path)
        if os.path.exists(full_path):
            logger.info(f"✅ {file_path}")
        else:
            logger.error(f"❌ {file_path} - NOT FOUND")
            return False
    
    # Check scripts moved
    script_files = [
        'scripts/profile_ground_truth.py',
        'scripts/benchmark_ground_truth.py',
    ]
    
    for file_path in script_files:
        full_path = os.path.join(repo_root, file_path)
        if os.path.exists(full_path):
            logger.info(f"✅ {file_path}")
        else:
            logger.warning(f"⚠️  {file_path} - NOT FOUND")
    
    # Check docs moved
    doc_files = [
        'docs/optimization/GROUND_TRUTH_PERFORMANCE_ANALYSIS.md',
        'docs/optimization/GROUND_TRUTH_QUICK_START.md',
        'docs/optimization/OPTIMIZATION_README.md',
        'docs/optimization/MIGRATION_GUIDE.md',
    ]
    
    for file_path in doc_files:
        full_path = os.path.join(repo_root, file_path)
        if os.path.exists(full_path):
            logger.info(f"✅ {file_path}")
        else:
            logger.warning(f"⚠️  {file_path} - NOT FOUND")
    
    # Check old files removed
    old_files = [
        'optimize_ground_truth.py',
        'optimize_ground_truth_strtree.py',
        'optimize_ground_truth_vectorized.py',
        'optimize_ground_truth_gpu.py',
        'ground_truth_quick_fix.py',
    ]
    
    for file_path in old_files:
        full_path = os.path.join(repo_root, file_path)
        if not os.path.exists(full_path):
            logger.info(f"✅ {file_path} - REMOVED")
        else:
            logger.warning(f"⚠️  {file_path} - STILL EXISTS (should be removed)")
    
    return True


def main():
    """Run all verification checks."""
    logger.info("\n")
    logger.info("="*80)
    logger.info("GROUND TRUTH OPTIMIZATION INTEGRATION VERIFICATION")
    logger.info("="*80)
    logger.info("\n")
    
    checks = [
        ("Module Structure", check_module_structure),
        ("Public API", check_public_api),
        ("Dependencies", check_dependencies),
        ("Auto-Select", check_auto_select),
        ("File Locations", check_file_locations),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            logger.error(f"❌ {check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*80)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status:10} - {check_name}")
        if not result:
            all_passed = False
    
    logger.info("="*80)
    
    if all_passed:
        logger.info("\n✅ ALL CHECKS PASSED - Integration successful!")
        logger.info("\nYou can now use the optimization module:")
        logger.info("  from ign_lidar.optimization import auto_optimize")
        logger.info("  auto_optimize()")
        return 0
    else:
        logger.error("\n❌ SOME CHECKS FAILED - Please review the errors above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
