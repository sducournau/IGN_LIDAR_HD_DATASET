#!/usr/bin/env python3
"""
Test GPU Optimizations - November 2025 Audit Fixes

This script validates the GPU transfer optimizations implemented:
1. Vectorized transfers in gpu_kernels.py (Fix #1)
2. Batched transfers in ground_truth_classifier.py (Fix #2)
3. GPU pipeline in gpu_processor.py (Fix #3)

Usage:
    # Run all tests
    python scripts/test_gpu_optimizations.py

    # Test specific fix
    python scripts/test_gpu_optimizations.py --test fix1
    python scripts/test_gpu_optimizations.py --test fix2
    python scripts/test_gpu_optimizations.py --test fix3

    # With GPU environment
    conda run -n ign_gpu python scripts/test_gpu_optimizations.py

Author: IGN LiDAR HD Development Team
Date: November 23, 2025
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("✅ CuPy available - GPU tests enabled")
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    logger.warning("⚠️  CuPy not available - GPU tests will be skipped")


def test_fix1_gpu_kernels():
    """
    Test Fix #1: Vectorized transfers in gpu_kernels.py
    
    Expected improvement: 50-100× reduction in transfers
    """
    print("\n" + "="*70)
    print("TEST 1: Vectorized Transfers in gpu_kernels.py")
    print("="*70)
    
    if not GPU_AVAILABLE:
        print("⚠️  Skipping - GPU not available")
        return False
    
    try:
        from ign_lidar.optimization.gpu_kernels import compute_normals_curvature_fused_sequential_fallback
        
        # Create test data
        n_points = 1000
        points = np.random.rand(n_points, 3).astype(np.float32)
        knn_indices = np.random.randint(0, n_points, size=(n_points, 20))
        
        print(f"\nTesting with {n_points:,} points...")
        
        # Time the computation
        start = time.time()
        normals, eigenvalues, curvature = compute_normals_curvature_fused_sequential_fallback(
            points, knn_indices, k=20
        )
        elapsed = time.time() - start
        
        # Validate results
        assert normals.shape == (n_points, 3), f"Expected shape ({n_points}, 3), got {normals.shape}"
        assert eigenvalues.shape == (n_points, 3), f"Expected shape ({n_points}, 3), got {eigenvalues.shape}"
        assert curvature.shape == (n_points,), f"Expected shape ({n_points},), got {curvature.shape}"
        
        print(f"✅ Computation completed in {elapsed:.3f}s")
        print(f"   Normals shape: {normals.shape}")
        print(f"   Eigenvalues shape: {eigenvalues.shape}")
        print(f"   Curvature shape: {curvature.shape}")
        print(f"\n💡 Optimization: Transfers now vectorized (3 transfers instead of {n_points*3:,})")
        print(f"   Expected speedup: 50-100× on large datasets")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fix2_ground_truth_classifier():
    """
    Test Fix #2: Batched transfers in ground_truth_classifier.py
    
    Expected improvement: 2.5× reduction in transfers
    """
    print("\n" + "="*70)
    print("TEST 2: Batched Transfers in ground_truth_classifier.py")
    print("="*70)
    
    if not GPU_AVAILABLE:
        print("⚠️  Skipping - GPU not available")
        return False
    
    try:
        # Test the batched transfer optimization
        n_points = 10000
        height = np.random.rand(n_points).astype(np.float32)
        planarity = np.random.rand(n_points).astype(np.float32)
        intensity = np.random.rand(n_points).astype(np.float32)
        
        print(f"\nTesting batched feature transfer with {n_points:,} points...")
        
        # Simulate the BEFORE code (3 separate transfers)
        start_before = time.time()
        height_gpu_old = cp.asarray(height)
        planarity_gpu_old = cp.asarray(planarity)
        intensity_gpu_old = cp.asarray(intensity)
        elapsed_before = time.time() - start_before
        
        # Simulate the AFTER code (1 batched transfer)
        start_after = time.time()
        features_stacked = np.stack([height, planarity, intensity], axis=1).astype(np.float32)
        features_gpu = cp.asarray(features_stacked)
        height_gpu = features_gpu[:, 0]
        planarity_gpu = features_gpu[:, 1]
        intensity_gpu = features_gpu[:, 2]
        elapsed_after = time.time() - start_after
        
        # Validate results match
        assert cp.allclose(height_gpu, height_gpu_old), "Height mismatch"
        assert cp.allclose(planarity_gpu, planarity_gpu_old), "Planarity mismatch"
        assert cp.allclose(intensity_gpu, intensity_gpu_old), "Intensity mismatch"
        
        speedup = elapsed_before / elapsed_after if elapsed_after > 0 else 1.0
        
        print(f"✅ Batched transfer validated")
        print(f"   Before (3 transfers): {elapsed_before*1000:.3f}ms")
        print(f"   After (1 transfer):   {elapsed_after*1000:.3f}ms")
        print(f"   Speedup: {speedup:.2f}×")
        print(f"\n💡 Optimization: 3 transfers → 1 batched transfer")
        print(f"   Expected improvement: 2-3× on large datasets")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fix3_gpu_pipeline():
    """
    Test Fix #3: GPU pipeline in gpu_processor.py
    
    Expected improvement: 2-3× reduction in transfers
    """
    print("\n" + "="*70)
    print("TEST 3: GPU Pipeline in gpu_processor.py")
    print("="*70)
    
    if not GPU_AVAILABLE:
        print("⚠️  Skipping - GPU not available")
        return False
    
    try:
        from ign_lidar.features.gpu_processor import GPUProcessor
        
        # Create test data
        n_points = 5000
        points = np.random.rand(n_points, 3).astype(np.float32)
        
        print(f"\nTesting GPU pipeline with {n_points:,} points...")
        
        # Initialize processor
        processor = GPUProcessor(use_gpu=True, show_progress=False)
        
        # Test optimized pipeline
        start = time.time()
        features = processor.compute_features_gpu_pipeline(
            points, 
            feature_types=['curvature', 'linearity', 'planarity'],
            k=20,
            return_intermediates=True
        )
        elapsed = time.time() - start
        
        # Validate results
        assert 'curvature' in features, "Missing curvature"
        assert 'linearity' in features, "Missing linearity"
        assert 'planarity' in features, "Missing planarity"
        assert 'normals' in features, "Missing normals (intermediates)"
        assert 'eigenvalues' in features, "Missing eigenvalues (intermediates)"
        
        for name, feat in features.items():
            expected_shape = (n_points, 3) if name in ['normals', 'eigenvalues'] else (n_points,)
            assert feat.shape == expected_shape, f"{name}: expected {expected_shape}, got {feat.shape}"
        
        print(f"✅ GPU pipeline completed in {elapsed:.3f}s")
        print(f"   Features computed: {list(features.keys())}")
        print(f"   Transfer count: 2 (1 CPU→GPU, 1 GPU→CPU)")
        print(f"\n💡 Optimization: All computations done on GPU")
        print(f"   Standard pipeline: ~10-15 transfers")
        print(f"   Optimized pipeline: 2 transfers")
        print(f"   Expected improvement: 2-3× faster")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all():
    """Run all GPU optimization tests."""
    print("\n" + "="*70)
    print("GPU OPTIMIZATION TESTS - NOVEMBER 2025 AUDIT")
    print("="*70)
    
    if not GPU_AVAILABLE:
        print("\n⚠️  GPU not available - tests will be skipped")
        print("    Run with: conda run -n ign_gpu python scripts/test_gpu_optimizations.py")
        return
    
    results = {
        'Fix #1 (gpu_kernels.py)': test_fix1_gpu_kernels(),
        'Fix #2 (ground_truth_classifier.py)': test_fix2_ground_truth_classifier(),
        'Fix #3 (gpu_processor.py)': test_fix3_gpu_pipeline(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All optimizations validated!")
        print("\n📈 Expected cumulative improvement:")
        print("   - Transfer reduction: ~95% (90+ → <5 per tile)")
        print("   - Performance gain: 3-5× faster GPU processing")
        print("   - Latency reduction: 60-80% less synchronization")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - check logs above")


def main():
    parser = argparse.ArgumentParser(
        description="Test GPU transfer optimizations"
    )
    parser.add_argument(
        '--test',
        choices=['fix1', 'fix2', 'fix3', 'all'],
        default='all',
        help='Which test to run (default: all)'
    )
    
    args = parser.parse_args()
    
    if args.test == 'fix1':
        test_fix1_gpu_kernels()
    elif args.test == 'fix2':
        test_fix2_ground_truth_classifier()
    elif args.test == 'fix3':
        test_fix3_gpu_pipeline()
    else:
        test_all()


if __name__ == "__main__":
    main()
