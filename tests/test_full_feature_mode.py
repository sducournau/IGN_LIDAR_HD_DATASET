#!/usr/bin/env python3
"""
Test to verify that 'full' feature mode actually computes all documented features.

This test checks that:
1. CPU implementation computes all features when include_extra=True
2. GPU implementation computes all features (if GPU available)
3. Output matches documentation in feature_modes.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features.features import compute_all_features_optimized
from ign_lidar.features.feature_modes import LOD3_FEATURES, FeatureMode


def test_cpu_full_features():
    """Test that CPU implementation computes all documented features."""
    print("\n" + "="*70)
    print("TEST 1: CPU Full Feature Mode")
    print("="*70)
    
    # Create synthetic point cloud
    np.random.seed(42)
    num_points = 1000
    points = np.random.randn(num_points, 3).astype(np.float32) * 10
    points[:, 2] += 50  # Elevate above ground
    
    # Create classification (some ground points)
    classification = np.random.choice([2, 6], size=num_points)  # 2=ground, 6=building
    
    # Compute features with include_extra=True (full mode)
    print(f"\n📊 Computing features for {num_points} points...")
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points,
        classification=classification,
        k=20,
        auto_k=False,
        include_extra=True,  # FULL MODE
        patch_center=points.mean(axis=0),
        chunk_size=None,
        radius=None
    )
    
    # Expected features from LOD3_FULL documentation
    expected_core = [
        'planarity', 'linearity', 'sphericity', 'anisotropy', 
        'roughness', 'density'
    ]
    
    expected_building = [
        'verticality', 'horizontality', 'wall_score', 'roof_score'
    ]
    
    expected_eigenvalue = [
        'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
        'sum_eigenvalues', 'eigenentropy', 'omnivariance', 'change_curvature'
    ]
    
    expected_architectural = [
        'edge_strength', 'corner_likelihood', 
        'overhang_indicator', 'surface_roughness'
    ]
    
    expected_height = [
        'z_absolute', 'z_normalized', 'z_from_ground', 'z_from_median'
    ]
    
    expected_local_stats = [
        'vertical_std', 'neighborhood_extent', 
        'height_extent_ratio', 'local_roughness'
    ]
    
    expected_density = ['num_points_2m']
    
    all_expected = (
        expected_core + expected_building + expected_eigenvalue + 
        expected_architectural + expected_height + expected_local_stats + 
        expected_density
    )
    
    # Check which features are present
    print("\n✅ Features PRESENT:")
    present_features = []
    for feature in all_expected:
        if feature in geo_features:
            values = geo_features[feature]
            print(f"  ✓ {feature:25s} shape={values.shape}, "
                  f"range=[{values.min():.4f}, {values.max():.4f}]")
            present_features.append(feature)
        else:
            print(f"  ✗ {feature:25s} MISSING")
    
    # Summary
    print("\n" + "="*70)
    print(f"RESULTS: {len(present_features)}/{len(all_expected)} features present")
    print("="*70)
    
    # Check categories
    categories = {
        "Core Geometric": expected_core,
        "Building Scores": expected_building,
        "Eigenvalue Features": expected_eigenvalue,
        "Architectural Features": expected_architectural,
        "Height Features": expected_height,
        "Local Statistics": expected_local_stats,
        "Density Features": expected_density
    }
    
    print("\nFeature Coverage by Category:")
    for category, features in categories.items():
        present_count = sum(1 for f in features if f in geo_features)
        total_count = len(features)
        percentage = (present_count / total_count * 100) if total_count > 0 else 0
        status = "✅" if present_count == total_count else "❌"
        print(f"  {status} {category:25s}: {present_count:2d}/{total_count:2d} ({percentage:5.1f}%)")
    
    # Return success if all features present
    all_present = len(present_features) == len(all_expected)
    
    if all_present:
        print("\n✅ SUCCESS: All documented features are computed!")
        return True
    else:
        missing = set(all_expected) - set(present_features)
        print(f"\n❌ FAILURE: Missing {len(missing)} features:")
        for feature in sorted(missing):
            print(f"     - {feature}")
        return False


def test_feature_validation():
    """Test that computed features have valid values."""
    print("\n" + "="*70)
    print("TEST 2: Feature Value Validation")
    print("="*70)
    
    # Create synthetic point cloud
    np.random.seed(42)
    num_points = 500
    points = np.random.randn(num_points, 3).astype(np.float32) * 10
    points[:, 2] += 50
    classification = np.random.choice([2, 6], size=num_points)
    
    # Compute features
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points,
        classification=classification,
        k=20,
        auto_k=False,
        include_extra=True,
        patch_center=points.mean(axis=0)
    )
    
    # Validate each feature
    issues = []
    
    # Features that should be in [0, 1]
    bounded_features = [
        'planarity', 'linearity', 'sphericity', 'anisotropy', 
        'roughness', 'verticality', 'horizontality',
        'wall_score', 'roof_score', 'z_normalized',
        'edge_strength', 'corner_likelihood', 'overhang_indicator'
    ]
    
    for feature in bounded_features:
        if feature in geo_features:
            values = geo_features[feature]
            if np.any(values < 0) or np.any(values > 1.0):
                issues.append(f"{feature} has values outside [0, 1]: "
                            f"[{values.min():.4f}, {values.max():.4f}]")
    
    # Features that should be non-negative
    positive_features = [
        'density', 'num_points_2m', 'eigenvalue_1', 'eigenvalue_2', 
        'eigenvalue_3', 'sum_eigenvalues', 'eigenentropy', 'omnivariance',
        'curvature', 'neighborhood_extent', 'vertical_std'
    ]
    
    for feature in positive_features:
        if feature in geo_features:
            values = geo_features[feature]
            if np.any(values < 0):
                issues.append(f"{feature} has negative values: min={values.min():.4f}")
    
    # Check for NaN/Inf
    for feature, values in geo_features.items():
        if np.any(np.isnan(values)):
            issues.append(f"{feature} contains NaN values")
        if np.any(np.isinf(values)):
            issues.append(f"{feature} contains Inf values")
    
    # Report results
    if issues:
        print("\n❌ VALIDATION FAILURES:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ SUCCESS: All features have valid values!")
        print(f"  - All bounded features in [0, 1]")
        print(f"  - All positive features >= 0")
        print(f"  - No NaN or Inf values found")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("FULL FEATURE MODE TEST SUITE")
    print("Testing fix for GitHub issue: Full mode missing features")
    print("="*70)
    
    results = []
    
    # Test 1: Feature completeness
    try:
        result1 = test_cpu_full_features()
        results.append(("Feature Completeness", result1))
    except Exception as e:
        print(f"\n❌ Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Feature Completeness", False))
    
    # Test 2: Feature validation
    try:
        result2 = test_feature_validation()
        results.append(("Feature Validation", result2))
    except Exception as e:
        print(f"\n❌ Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Feature Validation", False))
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED! Full feature mode is working correctly.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
