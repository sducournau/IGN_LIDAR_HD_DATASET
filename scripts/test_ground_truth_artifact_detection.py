#!/usr/bin/env python3
"""
Test Ground Truth Artifact Detection

This script demonstrates how to use the artifact checker to ensure
correct ground truth classification by detecting and filtering artifacts.

Author: IGN LiDAR HD Classification Team
Date: October 19, 2025
"""

import numpy as np
import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.core.modules.ground_truth_artifact_checker import (
    GroundTruthArtifactChecker,
    validate_features_before_classification,
    ArtifactReport
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_features():
    """Create test feature arrays with various artifacts."""
    n_points = 10000
    
    # Clean features
    clean_height = np.random.uniform(0, 50, n_points)
    clean_planarity = np.random.uniform(0, 1, n_points)
    clean_ndvi = np.random.uniform(-0.2, 0.8, n_points)
    
    # Features with artifacts
    
    # 1. Height with NaN values
    height_with_nan = clean_height.copy()
    nan_indices = np.random.choice(n_points, size=100, replace=False)
    height_with_nan[nan_indices] = np.nan
    
    # 2. Planarity with Inf values
    planarity_with_inf = clean_planarity.copy()
    inf_indices = np.random.choice(n_points, size=50, replace=False)
    planarity_with_inf[inf_indices] = np.inf
    
    # 3. NDVI with out-of-range values
    ndvi_out_of_range = clean_ndvi.copy()
    oor_indices = np.random.choice(n_points, size=75, replace=False)
    ndvi_out_of_range[oor_indices] = 2.5  # NDVI should be [-1, 1]
    
    # 4. Curvature with constant values
    curvature_constant = np.full(n_points, 0.123456789)
    
    # 5. Normals with mixed artifacts
    normals_artifacts = np.random.uniform(-1, 1, (n_points, 3))
    # Add some NaN
    normals_artifacts[np.random.choice(n_points, 30), 0] = np.nan
    # Add some Inf
    normals_artifacts[np.random.choice(n_points, 20), 1] = np.inf
    
    return {
        'clean': {
            'height': clean_height,
            'planarity': clean_planarity,
            'ndvi': clean_ndvi
        },
        'artifacts': {
            'height_nan': height_with_nan,
            'planarity_inf': planarity_with_inf,
            'ndvi_oor': ndvi_out_of_range,
            'curvature_const': curvature_constant,
            'normals': normals_artifacts
        }
    }


def test_clean_features():
    """Test 1: Check clean features (should pass)."""
    print("\n" + "="*70)
    print("TEST 1: Clean Features")
    print("="*70)
    
    test_data = create_test_features()
    clean_features = test_data['clean']
    
    is_valid, reports = validate_features_before_classification(
        features=clean_features,
        strict=False,
        log_results=True
    )
    
    if is_valid:
        print("\n✓ TEST 1 PASSED: All features are clean")
        return True
    else:
        print("\n✗ TEST 1 FAILED: Clean features flagged as artifacts")
        return False


def test_features_with_nan():
    """Test 2: Detect NaN values."""
    print("\n" + "="*70)
    print("TEST 2: Features with NaN")
    print("="*70)
    
    test_data = create_test_features()
    
    features = {
        'height': test_data['artifacts']['height_nan']
    }
    
    checker = GroundTruthArtifactChecker()
    report = checker.check_feature('height', features['height'])
    
    print(f"\n{report}")
    
    if report.has_artifacts and report.nan_count == 100:
        print("\n✓ TEST 2 PASSED: NaN values detected correctly")
        return True
    else:
        print("\n✗ TEST 2 FAILED: NaN detection failed")
        return False


def test_features_with_inf():
    """Test 3: Detect Inf values."""
    print("\n" + "="*70)
    print("TEST 3: Features with Inf")
    print("="*70)
    
    test_data = create_test_features()
    
    features = {
        'planarity': test_data['artifacts']['planarity_inf']
    }
    
    checker = GroundTruthArtifactChecker()
    report = checker.check_feature('planarity', features['planarity'])
    
    print(f"\n{report}")
    
    if report.has_artifacts and report.inf_count == 50:
        print("\n✓ TEST 3 PASSED: Inf values detected correctly")
        return True
    else:
        print("\n✗ TEST 3 FAILED: Inf detection failed")
        return False


def test_out_of_range():
    """Test 4: Detect out-of-range values."""
    print("\n" + "="*70)
    print("TEST 4: Out-of-Range Values")
    print("="*70)
    
    test_data = create_test_features()
    
    features = {
        'ndvi': test_data['artifacts']['ndvi_oor']
    }
    
    checker = GroundTruthArtifactChecker()
    report = checker.check_feature('ndvi', features['ndvi'])
    
    print(f"\n{report}")
    
    if report.has_artifacts and report.out_of_range_count > 0:
        print("\n✓ TEST 4 PASSED: Out-of-range values detected")
        return True
    else:
        print("\n✗ TEST 4 FAILED: Out-of-range detection failed")
        return False


def test_constant_values():
    """Test 5: Detect constant values."""
    print("\n" + "="*70)
    print("TEST 5: Constant Values")
    print("="*70)
    
    test_data = create_test_features()
    
    features = {
        'curvature': test_data['artifacts']['curvature_const']
    }
    
    checker = GroundTruthArtifactChecker()
    report = checker.check_feature('curvature', features['curvature'])
    
    print(f"\n{report}")
    
    if report.has_artifacts and report.constant_values:
        print("\n✓ TEST 5 PASSED: Constant values detected")
        return True
    else:
        print("\n✗ TEST 5 FAILED: Constant value detection failed")
        return False


def test_artifact_filtering():
    """Test 6: Filter artifacts from features."""
    print("\n" + "="*70)
    print("TEST 6: Artifact Filtering")
    print("="*70)
    
    test_data = create_test_features()
    
    features = {
        'height': test_data['artifacts']['height_nan'],
        'planarity': test_data['artifacts']['planarity_inf'],
        'ndvi': test_data['artifacts']['ndvi_oor']
    }
    
    checker = GroundTruthArtifactChecker()
    
    # Check before filtering
    reports_before = checker.check_all_features(features)
    artifact_count_before = sum(1 for r in reports_before.values() if r.has_artifacts)
    
    # Filter artifacts
    clean_features, artifact_masks = checker.filter_artifacts(features)
    
    # Check after filtering
    print("\nArtifact masks:")
    for name, mask in artifact_masks.items():
        n_artifacts = np.sum(mask)
        print(f"  {name}: {n_artifacts} points with artifacts")
    
    # Clean features should have NaN where artifacts were
    height_clean_nan = np.sum(np.isnan(clean_features['height']))
    planarity_clean_nan = np.sum(np.isnan(clean_features['planarity']))
    ndvi_clean_nan = np.sum(np.isnan(clean_features['ndvi']))
    
    print(f"\nClean features (NaN where artifacts):")
    print(f"  height: {height_clean_nan} NaN")
    print(f"  planarity: {planarity_clean_nan} NaN")
    print(f"  ndvi: {ndvi_clean_nan} NaN")
    
    if (height_clean_nan == 100 and  # Original NaN
        planarity_clean_nan == 50 and  # Original Inf
        ndvi_clean_nan == 75):  # Out of range
        print("\n✓ TEST 6 PASSED: Artifacts filtered correctly")
        return True
    else:
        print("\n✗ TEST 6 FAILED: Artifact filtering incorrect")
        return False


def test_validation_for_classification():
    """Test 7: Validate features for ground truth classification."""
    print("\n" + "="*70)
    print("TEST 7: Validation for Classification")
    print("="*70)
    
    test_data = create_test_features()
    
    # Test with clean features
    clean_features = test_data['clean']
    
    checker = GroundTruthArtifactChecker(config={
        'max_artifact_ratio': 0.05  # Allow 5% artifacts
    })
    
    is_valid, warnings = checker.validate_for_ground_truth(
        clean_features,
        strict=False
    )
    
    print("\nClean features validation:")
    print(f"  Valid: {is_valid}")
    print(f"  Warnings: {warnings}")
    
    # Test with artifact features (>5% artifacts)
    artifact_features = {
        'height': test_data['artifacts']['height_nan'],  # 1% artifacts
        'planarity': test_data['artifacts']['planarity_inf'],  # 0.5% artifacts
        'ndvi': test_data['artifacts']['ndvi_oor']  # 0.75% artifacts
    }
    
    is_valid_artifacts, warnings_artifacts = checker.validate_for_ground_truth(
        artifact_features,
        strict=False
    )
    
    print("\nArtifact features validation:")
    print(f"  Valid: {is_valid_artifacts}")
    print(f"  Warnings: {len(warnings_artifacts)} warnings")
    for warning in warnings_artifacts:
        print(f"    - {warning}")
    
    if is_valid and not is_valid_artifacts:
        print("\n✓ TEST 7 PASSED: Validation works correctly")
        return True
    else:
        print("\n✗ TEST 7 FAILED: Validation logic incorrect")
        return False


def test_realistic_scenario():
    """Test 8: Realistic ground truth classification scenario."""
    print("\n" + "="*70)
    print("TEST 8: Realistic Classification Scenario")
    print("="*70)
    
    # Simulate real classification scenario
    n_points = 5000
    
    # Ground truth labels
    gt_labels = np.random.choice([6, 11, 9], n_points)  # Building, Road, Water
    
    # Features (mostly clean, some artifacts)
    height = np.random.uniform(0, 30, n_points)
    planarity = np.random.uniform(0, 1, n_points)
    ndvi = np.random.uniform(-0.2, 0.8, n_points)
    curvature = np.random.uniform(0, 0.5, n_points)
    
    # Add realistic artifacts
    # - Boundary points have NaN normals
    artifact_indices = np.random.choice(n_points, size=50, replace=False)
    height[artifact_indices] = np.nan
    
    # - Computation failures
    fail_indices = np.random.choice(n_points, size=25, replace=False)
    curvature[fail_indices] = np.inf
    
    features = {
        'height': height,
        'planarity': planarity,
        'ndvi': ndvi,
        'curvature': curvature
    }
    
    # Validate and filter
    checker = GroundTruthArtifactChecker()
    
    print("\n1. Checking features for artifacts...")
    reports = checker.check_all_features(features)
    
    artifact_count = sum(1 for r in reports.values() if r.has_artifacts)
    print(f"   Features with artifacts: {artifact_count}/4")
    
    print("\n2. Validating for ground truth classification...")
    is_valid, warnings = checker.validate_for_ground_truth(features, strict=False)
    
    if is_valid:
        print("   ✓ Features are suitable for classification")
    else:
        print("   ⚠️ Features have too many artifacts")
        for warning in warnings:
            print(f"      - {warning}")
    
    print("\n3. Filtering artifacts...")
    clean_features, artifact_masks = checker.filter_artifacts(features)
    
    # Count valid points
    valid_points = np.ones(n_points, dtype=bool)
    for mask in artifact_masks.values():
        valid_points &= ~mask
    
    n_valid = np.sum(valid_points)
    print(f"   Valid points: {n_valid}/{n_points} ({n_valid/n_points*100:.1f}%)")
    
    print("\n4. Proceeding with classification on clean features...")
    # In real scenario, would now use clean_features for classification
    
    if n_valid > (n_points * 0.9):  # Should retain >90% of points
        print("\n✓ TEST 8 PASSED: Realistic scenario handled correctly")
        return True
    else:
        print("\n✗ TEST 8 FAILED: Too many points filtered")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("GROUND TRUTH ARTIFACT DETECTION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Clean Features", test_clean_features),
        ("NaN Detection", test_features_with_nan),
        ("Inf Detection", test_features_with_inf),
        ("Out-of-Range Detection", test_out_of_range),
        ("Constant Value Detection", test_constant_values),
        ("Artifact Filtering", test_artifact_filtering),
        ("Classification Validation", test_validation_for_classification),
        ("Realistic Scenario", test_realistic_scenario)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ TEST FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("="*70)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print("="*70)
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Artifact detection is working correctly.")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the output above.")
        return 1


if __name__ == '__main__':
    exit(main())
