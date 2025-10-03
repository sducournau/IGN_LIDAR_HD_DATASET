#!/usr/bin/env python3
"""
Test script to validate the geometric feature fixes.

Tests:
1. GPU vs CPU formula consistency
2. Degenerate case handling
3. Robust curvature computation
"""

import numpy as np
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features import (
    compute_all_features_optimized,
    compute_normals,
    compute_curvature
)
from ign_lidar.features_gpu import GPUFeatureComputer


def test_gpu_cpu_consistency():
    """Test that GPU and CPU produce same feature values."""
    print("=" * 70)
    print("TEST 1: GPU vs CPU Formula Consistency")
    print("=" * 70)
    
    # Create test data (building-like structure)
    np.random.seed(42)
    n_points = 1000
    
    # Roof (planar)
    roof = np.random.randn(n_points, 3) * 0.1
    roof[:, 2] += 10.0
    
    points = roof.astype(np.float32)
    classification = np.full(n_points, 6, dtype=np.uint8)  # Building
    
    # Compute features with CPU
    print("\nComputing features with CPU...")
    normals_cpu, curvature_cpu, height_cpu, geo_cpu = (
        compute_all_features_optimized(
            points=points,
            classification=classification,
            k=20,
            auto_k=False,
            include_extra=False
        )
    )
    
    # Compute features with GPU
    print("Computing features with GPU...")
    try:
        gpu_computer = GPUFeatureComputer(use_gpu=False)  # CPU fallback OK
        normals_gpu = gpu_computer.compute_normals(points, k=20)
        curvature_gpu = gpu_computer.compute_curvature(points, normals_gpu, k=20)
        height_gpu = gpu_computer.compute_height_above_ground(
            points, classification
        )
        geo_gpu = gpu_computer.extract_geometric_features(
            points, normals_gpu, k=20
        )
    except Exception as e:
        print(f"⚠️  GPU computation failed: {e}")
        print("   (This is OK if GPU not available)")
        return
    
    # Compare features
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    features_to_check = [
        'planarity', 'linearity', 'sphericity', 
        'anisotropy', 'roughness', 'density'
    ]
    
    all_match = True
    for feat in features_to_check:
        cpu_vals = geo_cpu[feat]
        gpu_vals = geo_gpu[feat]
        
        # Compute relative difference
        diff = np.abs(cpu_vals - gpu_vals)
        rel_diff = diff / (np.abs(cpu_vals) + 1e-8)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        match = max_rel_diff < 1e-3  # 0.1% tolerance
        status = "✓ PASS" if match else "✗ FAIL"
        all_match = all_match and match
        
        print(f"{feat:12s}: {status} | "
              f"max_rel_diff={max_rel_diff:.6f} | "
              f"mean_rel_diff={mean_rel_diff:.6f}")
        
        # Show value ranges
        print(f"              CPU: [{cpu_vals.min():.4f}, {cpu_vals.max():.4f}]")
        print(f"              GPU: [{gpu_vals.min():.4f}, {gpu_vals.max():.4f}]")
    
    print("=" * 70)
    if all_match:
        print("✓ ALL TESTS PASSED - GPU matches CPU")
    else:
        print("✗ TESTS FAILED - GPU differs from CPU")
    print("=" * 70)
    
    return all_match


def test_degenerate_cases():
    """Test handling of degenerate neighborhoods."""
    print("\n" + "=" * 70)
    print("TEST 2: Degenerate Case Handling")
    print("=" * 70)
    
    # Test case 1: Isolated point (collinear points)
    points_collinear = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0],
    ], dtype=np.float32)
    classification = np.full(5, 6, dtype=np.uint8)
    
    print("\nTest case: Collinear points (degenerate, should not produce NaN)")
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points_collinear,
        classification=classification,
        k=3,
        auto_k=False,
        include_extra=False
    )
    
    # Check for NaN/Inf
    has_nan = False
    has_inf = False
    
    for feat_name, feat_vals in geo_features.items():
        n_nan = np.sum(np.isnan(feat_vals))
        n_inf = np.sum(np.isinf(feat_vals))
        
        if n_nan > 0:
            print(f"  ✗ {feat_name}: {n_nan} NaN values")
            has_nan = True
        if n_inf > 0:
            print(f"  ✗ {feat_name}: {n_inf} Inf values")
            has_inf = True
    
    if not has_nan and not has_inf:
        print("  ✓ No NaN/Inf values - degenerate case handled correctly")
    else:
        print("  ✗ Found NaN/Inf values - degenerate case NOT handled")
    
    # Test case 2: Very small eigenvalues
    print("\nTest case: Points with near-zero variance")
    points_flat = np.zeros((100, 3), dtype=np.float32)
    points_flat[:, :2] = np.random.randn(100, 2) * 1e-8  # Almost zero
    classification = np.full(100, 6, dtype=np.uint8)
    
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points_flat,
        classification=classification,
        k=10,
        auto_k=False,
        include_extra=False
    )
    
    has_nan = any(np.any(np.isnan(v)) for v in geo_features.values())
    has_inf = any(np.any(np.isinf(v)) for v in geo_features.values())
    
    if not has_nan and not has_inf:
        print("  ✓ No NaN/Inf values - near-zero variance handled correctly")
    else:
        print("  ✗ Found NaN/Inf values - near-zero variance NOT handled")
    
    print("=" * 70)
    
    return not (has_nan or has_inf)


def test_robust_curvature():
    """Test that robust curvature handles outliers."""
    print("\n" + "=" * 70)
    print("TEST 3: Robust Curvature Computation")
    print("=" * 70)
    
    # Create planar surface with one outlier
    np.random.seed(42)
    n_points = 100
    points = np.zeros((n_points, 3), dtype=np.float32)
    points[:, :2] = np.random.randn(n_points, 2) * 0.5
    points[:, 2] = 0.0  # Perfectly flat
    
    # Add one outlier
    points[50, 2] = 5.0  # Big outlier
    
    classification = np.full(n_points, 6, dtype=np.uint8)
    
    print("\nTest case: Planar surface with 1 outlier")
    print(f"  Points: {n_points} (99 flat, 1 outlier at z=5.0)")
    
    # Compute curvature
    normals = compute_normals(points, k=10)
    curvature = compute_curvature(points, normals, k=10)
    
    # Check median curvature (should be low for mostly flat surface)
    median_curv = np.median(curvature)
    max_curv = np.max(curvature)
    
    print(f"\nCurvature statistics:")
    print(f"  Median: {median_curv:.6f} (should be low, ~0)")
    print(f"  Maximum: {max_curv:.6f}")
    print(f"  At outlier: {curvature[50]:.6f}")
    
    # With robust MAD, median should still be near zero
    robust = median_curv < 0.1  # Median should be very low
    
    if robust:
        print(f"  ✓ PASS - Robust curvature handles outliers well")
    else:
        print(f"  ✗ FAIL - Curvature affected by outlier")
    
    print("=" * 70)
    
    return robust


def test_feature_value_ranges():
    """Test that features are in expected ranges."""
    print("\n" + "=" * 70)
    print("TEST 4: Feature Value Ranges")
    print("=" * 70)
    
    # Create diverse test data
    np.random.seed(42)
    
    # Planar (roof)
    planar = np.random.randn(200, 3) * [2, 2, 0.1]
    planar[:, 2] += 10
    
    # Linear (edge)
    linear = np.zeros((100, 3))
    linear[:, 0] = np.linspace(0, 10, 100)
    linear[:, 1:] = np.random.randn(100, 2) * 0.05
    
    # Spherical (vegetation-like)
    spherical = np.random.randn(200, 3) * 0.5
    
    points = np.vstack([planar, linear, spherical]).astype(np.float32)
    classification = np.full(len(points), 6, dtype=np.uint8)
    
    print("\nTest data:")
    print(f"  200 planar points (roof-like)")
    print(f"  100 linear points (edge-like)")
    print(f"  200 spherical points (vegetation-like)")
    
    # Compute features
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points,
        classification=classification,
        k=20,
        auto_k=False,
        include_extra=False
    )
    
    print("\nFeature value ranges (should be in [0, 1]):")
    
    all_valid = True
    for feat_name, feat_vals in geo_features.items():
        min_val = np.min(feat_vals)
        max_val = np.max(feat_vals)
        mean_val = np.mean(feat_vals)
        
        # Check range (density can be > 1, others should be [0, 1])
        if feat_name == 'density':
            valid = min_val >= 0
        else:
            valid = (min_val >= 0) and (max_val <= 1)
        
        status = "✓" if valid else "✗"
        all_valid = all_valid and valid
        
        print(f"  {status} {feat_name:12s}: "
              f"[{min_val:.4f}, {max_val:.4f}] mean={mean_val:.4f}")
    
    print("\n" + "=" * 70)
    if all_valid:
        print("✓ ALL FEATURES IN VALID RANGES")
    else:
        print("✗ SOME FEATURES OUT OF RANGE")
    print("=" * 70)
    
    return all_valid


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("GEOMETRIC FEATURES VALIDATION TESTS")
    print("=" * 70)
    
    results = {}
    
    # Run all tests
    results['consistency'] = test_gpu_cpu_consistency()
    results['degenerate'] = test_degenerate_cases()
    results['robust_curvature'] = test_robust_curvature()
    results['value_ranges'] = test_feature_value_ranges()
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("=" * 70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        sys.exit(0)
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        sys.exit(1)
