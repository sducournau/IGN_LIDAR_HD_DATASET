#!/usr/bin/env python3
"""
Test geometric features calculation

Validates that linearity, planarity, and sphericity are correctly computed
according to standard formulas (Weinmann et al., Demantké et al.)
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ign_lidar.features import extract_geometric_features


def create_linear_structure():
    """Create a perfect line (high linearity)."""
    # Points along a line with slight noise
    t = np.linspace(0, 10, 100)
    points = np.column_stack([
        t + np.random.normal(0, 0.01, 100),
        np.zeros(100) + np.random.normal(0, 0.01, 100),
        np.zeros(100) + np.random.normal(0, 0.01, 100)
    ])
    return points


def create_planar_structure():
    """Create a perfect plane (high planarity)."""
    # Points randomly distributed on a plane with very slight noise
    # Random positions avoid grid artifacts that can show as linearity
    n_points = 500  # Many points for good statistics
    x = np.random.uniform(0, 10, n_points)
    y = np.random.uniform(0, 10, n_points)
    # Extremely small noise perpendicular to plane (0.0001 = 0.1mm)
    z = np.zeros(n_points) + np.random.normal(0, 0.0001, n_points)
    points = np.column_stack([x, y, z])
    return points


def create_spherical_structure():
    """Create a sphere/scattered points (high sphericity)."""
    # Random points in a sphere
    points = np.random.normal(0, 1, (100, 3))
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    # Add slight noise
    points += np.random.normal(0, 0.05, points.shape)
    return points


def test_geometric_features():
    """Test that geometric features are computed correctly."""
    
    print("\n" + "="*70)
    print("GEOMETRIC FEATURES VALIDATION TEST")
    print("="*70 + "\n")
    
    print("Testing standard formulas (Weinmann et al., Demantké et al.):")
    print("  Linearity  = (λ0-λ1)/Σλ  - should be high for edges/cables")
    print("  Planarity  = (λ1-λ2)/Σλ  - should be high for roofs/walls")
    print("  Sphericity = λ2/Σλ       - should be high for vegetation")
    print("  Property: Linearity + Planarity + Sphericity = 1.0")
    print()
    
    # Test 1: Linear structure
    print("-" * 70)
    print("TEST 1: Linear Structure (edge/cable)")
    print("-" * 70)
    linear_points = create_linear_structure()
    normals = np.zeros_like(linear_points)  # Dummy normals
    features = extract_geometric_features(linear_points, normals, k=10)
    
    lin = np.mean(features['linearity'])
    pla = np.mean(features['planarity'])
    sph = np.mean(features['sphericity'])
    total = lin + pla + sph
    
    print(f"  Linearity:  {lin:.4f}  (should be high, ~0.7-0.9)")
    print(f"  Planarity:  {pla:.4f}  (should be low)")
    print(f"  Sphericity: {sph:.4f}  (should be low)")
    print(f"  Sum:        {total:.4f}  (should be ~1.0)")
    
    assert 0.5 < lin < 1.0, f"Linearity should be high for lines, got {lin}"
    assert total > 0.95 and total < 1.05, f"Sum should be ~1.0, got {total}"
    print("  ✅ PASSED\n")
    
    # Test 2: Planar structure
    print("-" * 70)
    print("TEST 2: Planar Structure (roof/wall)")
    print("-" * 70)
    planar_points = create_planar_structure()
    # Use more neighbors for planar surface to get more isotropic neighborhood
    features = extract_geometric_features(planar_points, normals, k=30)
    
    lin = np.mean(features['linearity'])
    pla = np.mean(features['planarity'])
    sph = np.mean(features['sphericity'])
    total = lin + pla + sph
    
    print(f"  Linearity:  {lin:.4f}  (should be low)")
    print(f"  Planarity:  {pla:.4f}  (should be high, ~0.7-0.9)")
    print(f"  Sphericity: {sph:.4f}  (should be low)")
    print(f"  Sum:        {total:.4f}  (should be ~1.0)")
    
    assert 0.5 < pla < 1.0, f"Planarity should be high for planes, got {pla}"
    assert total > 0.95 and total < 1.05, f"Sum should be ~1.0, got {total}"
    print("  ✅ PASSED\n")
    
    # Test 3: Spherical/scattered structure
    print("-" * 70)
    print("TEST 3: Spherical/Scattered Structure (vegetation)")
    print("-" * 70)
    spherical_points = create_spherical_structure()
    features = extract_geometric_features(spherical_points, normals, k=10)
    
    lin = np.mean(features['linearity'])
    pla = np.mean(features['planarity'])
    sph = np.mean(features['sphericity'])
    total = lin + pla + sph
    
    print(f"  Linearity:  {lin:.4f}  (should be low)")
    print(f"  Planarity:  {pla:.4f}  (should be low)")
    print(f"  Sphericity: {sph:.4f}  (should be high, ~0.5-0.8)")
    print(f"  Sum:        {total:.4f}  (should be ~1.0)")
    
    assert 0.3 < sph < 1.0, f"Sphericity should be high for spheres, got {sph}"
    assert total > 0.95 and total < 1.05, f"Sum should be ~1.0, got {total}"
    print("  ✅ PASSED\n")
    
    # Test 4: Value ranges
    print("-" * 70)
    print("TEST 4: Value Ranges")
    print("-" * 70)
    all_points = np.vstack([linear_points, planar_points, spherical_points])
    features = extract_geometric_features(all_points, 
                                         np.zeros_like(all_points), k=10)
    
    for name, values in features.items():
        vmin, vmax = np.min(values), np.max(values)
        print(f"  {name:12s}: min={vmin:.4f}, max={vmax:.4f}")
        
        # Check that geometric features are in [0,1] range
        if name in ['linearity', 'planarity', 'sphericity', 
                    'anisotropy', 'roughness']:
            assert vmin >= -0.01, f"{name} has negative values: {vmin}"
            assert vmax <= 1.01, f"{name} exceeds 1.0: {vmax}"
    
    print("  ✅ PASSED\n")
    
    # Summary
    print("=" * 70)
    print("SUMMARY: All tests passed! ✅")
    print("=" * 70)
    print("\nGeometric features are correctly computed according to:")
    print("  • Weinmann et al. (2015)")
    print("  • Demantké et al. (2011)")
    print("\nFormulas validated:")
    print("  ✓ Linearity + Planarity + Sphericity = 1.0")
    print("  ✓ All values in range [0, 1]")
    print("  ✓ High linearity for edges/cables")
    print("  ✓ High planarity for roofs/walls")
    print("  ✓ High sphericity for vegetation/scattered points")
    print()


if __name__ == "__main__":
    try:
        test_geometric_features()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
