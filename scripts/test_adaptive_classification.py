#!/usr/bin/env python3
"""
Test Adaptive Classification with Artifacts

This script demonstrates how the adaptive classifier adjusts rules when features
have artifacts, ensuring robust classification even with incomplete feature sets.

Author: IGN LiDAR HD Classification Team
Date: October 19, 2025
"""

import numpy as np
import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.core.modules.adaptive_classifier import AdaptiveClassifier
from ign_lidar.core.modules.ground_truth_artifact_checker import (
    get_artifact_free_features
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_scenarios():
    """Create test scenarios with varying feature quality."""
    n_points = 1000
    
    # Scenario 1: All features clean
    scenario_1 = {
        'name': 'All Features Clean',
        'ground_truth_types': np.array(['building'] * 300 + ['road'] * 300 + 
                                      ['water'] * 200 + ['vegetation'] * 200),
        'features': {
            'height': np.concatenate([
                np.random.uniform(5, 20, 300),    # Buildings
                np.random.uniform(0, 1.5, 300),   # Roads
                np.random.uniform(-0.3, 0.2, 200),  # Water
                np.random.uniform(0.3, 15, 200)    # Vegetation
            ]),
            'planarity': np.concatenate([
                np.random.uniform(0.70, 0.95, 300),  # Buildings
                np.random.uniform(0.85, 0.98, 300),  # Roads
                np.random.uniform(0.90, 0.99, 200),  # Water
                np.random.uniform(0.20, 0.60, 200)   # Vegetation
            ]),
            'curvature': np.concatenate([
                np.random.uniform(0.01, 0.08, 300),  # Buildings
                np.random.uniform(0.01, 0.04, 300),  # Roads
                np.random.uniform(0.00, 0.02, 200),  # Water
                np.random.uniform(0.20, 0.50, 200)   # Vegetation
            ]),
            'ndvi': np.concatenate([
                np.random.uniform(-0.1, 0.15, 300),  # Buildings
                np.random.uniform(-0.05, 0.12, 300), # Roads
                np.random.uniform(-0.15, 0.08, 200), # Water
                np.random.uniform(0.30, 0.80, 200)   # Vegetation
            ]),
            'normal_z': np.concatenate([
                np.random.uniform(0.7, 0.95, 300),   # Buildings (roofs)
                np.random.uniform(0.92, 0.99, 300),  # Roads
                np.random.uniform(0.96, 0.99, 200),  # Water
                np.random.uniform(0.3, 0.80, 200)    # Vegetation
            ]),
            'verticality': np.concatenate([
                np.random.uniform(0.50, 0.85, 300),  # Buildings (walls)
                np.random.uniform(0.05, 0.20, 300),  # Roads
                np.random.uniform(0.02, 0.10, 200),  # Water
                np.random.uniform(0.15, 0.60, 200)   # Vegetation
            ])
        }
    }
    
    # Scenario 2: Normals/Verticality have artifacts (tile boundaries)
    scenario_2_features = scenario_1['features'].copy()
    # Add artifacts to normals and verticality
    artifact_indices = np.random.choice(n_points, size=150, replace=False)
    scenario_2_features['normal_z'] = scenario_2_features['normal_z'].copy()
    scenario_2_features['verticality'] = scenario_2_features['verticality'].copy()
    scenario_2_features['normal_z'][artifact_indices] = np.nan
    scenario_2_features['verticality'][artifact_indices] = np.nan
    
    scenario_2 = {
        'name': 'Normals Have Artifacts (Tile Boundaries)',
        'ground_truth_types': scenario_1['ground_truth_types'].copy(),
        'features': scenario_2_features
    }
    
    # Scenario 3: Multiple features have artifacts
    scenario_3_features = scenario_1['features'].copy()
    artifact_indices_1 = np.random.choice(n_points, size=120, replace=False)
    artifact_indices_2 = np.random.choice(n_points, size=100, replace=False)
    artifact_indices_3 = np.random.choice(n_points, size=80, replace=False)
    
    scenario_3_features['planarity'] = scenario_3_features['planarity'].copy()
    scenario_3_features['curvature'] = scenario_3_features['curvature'].copy()
    scenario_3_features['normal_z'] = scenario_3_features['normal_z'].copy()
    
    scenario_3_features['planarity'][artifact_indices_1] = np.inf
    scenario_3_features['curvature'][artifact_indices_2] = np.nan
    scenario_3_features['normal_z'][artifact_indices_3] = np.nan
    
    scenario_3 = {
        'name': 'Multiple Features Have Artifacts',
        'ground_truth_types': scenario_1['ground_truth_types'].copy(),
        'features': scenario_3_features
    }
    
    # Scenario 4: Critical feature has artifacts (height)
    scenario_4_features = scenario_1['features'].copy()
    artifact_indices = np.random.choice(n_points, size=200, replace=False)
    scenario_4_features['height'] = scenario_4_features['height'].copy()
    scenario_4_features['height'][artifact_indices] = np.nan
    
    scenario_4 = {
        'name': 'Critical Feature Has Artifacts (Height)',
        'ground_truth_types': scenario_1['ground_truth_types'].copy(),
        'features': scenario_4_features
    }
    
    return [scenario_1, scenario_2, scenario_3, scenario_4]


def test_scenario(scenario):
    """Test a single scenario."""
    print("\n" + "="*70)
    print(f"SCENARIO: {scenario['name']}")
    print("="*70)
    
    features = scenario['features']
    ground_truth_types = scenario['ground_truth_types']
    n_points = len(ground_truth_types)
    
    # Step 1: Detect artifacts
    print("\n1. Detecting artifacts...")
    clean_features, artifact_features = get_artifact_free_features(features)
    
    print(f"   Clean features ({len(clean_features)}): {clean_features}")
    print(f"   Artifact features ({len(artifact_features)}): {artifact_features}")
    
    # Step 2: Setup adaptive classifier
    print("\n2. Setting up adaptive classifier...")
    classifier = AdaptiveClassifier()
    classifier.set_artifact_features(artifact_features)
    
    # Step 3: Get feature importance report
    print("\n3. Feature importance analysis...")
    importance_report = classifier.get_feature_importance_report(clean_features)
    
    for class_name, info in importance_report.items():
        can_classify = "✅ YES" if info['can_classify'] else "❌ NO"
        print(f"\n   {class_name}:")
        print(f"     Can classify: {can_classify}")
        print(f"     Confidence: {info['confidence']:.2f}")
        if info['critical']['missing']:
            print(f"     ⚠️  Missing critical: {info['critical']['missing']}")
        if info['important']['missing']:
            print(f"     ⚠️  Missing important: {info['important']['missing']}")
    
    # Step 4: Classify
    print("\n4. Performing adaptive classification...")
    labels = np.ones(n_points, dtype=np.uint8)  # Dummy initial labels
    
    validated_labels, confidences, valid_mask = classifier.classify_batch(
        labels=labels,
        ground_truth_types=ground_truth_types,
        features=features
    )
    
    # Step 5: Analyze results
    print("\n5. Results:")
    n_classified = valid_mask.sum()
    n_rejected = (~valid_mask).sum()
    
    print(f"   Total points: {n_points}")
    print(f"   Successfully classified: {n_classified} ({n_classified/n_points*100:.1f}%)")
    print(f"   Rejected: {n_rejected} ({n_rejected/n_points*100:.1f}%)")
    
    if n_classified > 0:
        mean_confidence = confidences[valid_mask].mean()
        min_confidence = confidences[valid_mask].min()
        max_confidence = confidences[valid_mask].max()
        
        print(f"   Mean confidence: {mean_confidence:.2f}")
        print(f"   Min confidence: {min_confidence:.2f}")
        print(f"   Max confidence: {max_confidence:.2f}")
        
        # Per-class breakdown
        print("\n   Per-class results:")
        for class_type in ['building', 'road', 'water', 'vegetation']:
            mask = ground_truth_types == class_type
            n_total = mask.sum()
            n_success = (mask & valid_mask).sum()
            
            if n_total > 0:
                success_rate = n_success / n_total * 100
                avg_conf = confidences[mask & valid_mask].mean() if n_success > 0 else 0.0
                
                print(f"     {class_type:12s}: {n_success:4d}/{n_total:4d} "
                      f"({success_rate:5.1f}%) avg_conf={avg_conf:.2f}")
    
    # Determine if test passed
    success_rate = n_classified / n_points
    
    if scenario['name'] == 'All Features Clean':
        # Should classify nearly all points
        passed = success_rate >= 0.95
    elif scenario['name'] == 'Critical Feature Has Artifacts (Height)':
        # Many points will fail (buildings/ground need height)
        passed = success_rate >= 0.20  # At least some should succeed
    else:
        # Should classify most points with degraded confidence
        passed = success_rate >= 0.75
    
    if passed:
        print(f"\n✓ TEST PASSED: {scenario['name']}")
    else:
        print(f"\n✗ TEST FAILED: {scenario['name']}")
        print(f"   Expected success rate >= threshold, got {success_rate:.1%}")
    
    return passed


def main():
    """Run all tests."""
    print("="*70)
    print("ADAPTIVE CLASSIFICATION WITH ARTIFACTS - TEST SUITE")
    print("="*70)
    
    scenarios = create_test_scenarios()
    results = []
    
    for scenario in scenarios:
        try:
            passed = test_scenario(scenario)
            results.append((scenario['name'], passed))
        except Exception as e:
            print(f"\n✗ TEST FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((scenario['name'], False))
    
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
        print("\n🎉 ALL TESTS PASSED! Adaptive classification works correctly.")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the output above.")
        return 1


if __name__ == '__main__':
    exit(main())
