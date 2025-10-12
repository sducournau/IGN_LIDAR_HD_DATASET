"""
Test Phase 2: Boundary feature completeness
Verify that boundary-aware computation includes all features.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features.features_boundary import BoundaryAwareFeatureComputer


def main():
    """Test boundary feature computation."""
    print("=" * 60)
    print("Phase 2: Boundary Feature Completeness Test")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    core_points = np.random.randn(100, 3).astype(np.float32) * 10
    buffer_points = np.random.randn(50, 3).astype(np.float32) * 10 + 20
    tile_bounds = (-10, -10, 10, 10)
    
    # Create boundary computer
    computer = BoundaryAwareFeatureComputer(
        k_neighbors=15,
        boundary_threshold=2.0,
        compute_normals=True,
        compute_curvature=True,
        compute_planarity=True,
        compute_verticality=True
    )
    
    print(f"\nTest data:")
    print(f"  Core points:   {len(core_points)}")
    print(f"  Buffer points: {len(buffer_points)}")
    
    # Compute features
    print("\nComputing features with boundary awareness...")
    features = computer.compute_features(
        core_points=core_points,
        buffer_points=buffer_points,
        tile_bounds=tile_bounds
    )
    
    # Expected features
    expected_features = {
        'normals', 'eigenvalues', 'curvature',
        'planarity', 'linearity', 'sphericity',
        'anisotropy', 'roughness',  # NEW in Phase 2
        'density',                   # NEW in Phase 2
        'verticality', 'horizontality',  # Updated in Phase 2
        'boundary_mask', 'num_boundary_points'
    }
    
    # Check which features are present
    print("\n" + "-" * 60)
    print("Feature Availability:")
    print("-" * 60)
    
    all_present = True
    for feature_name in sorted(expected_features):
        if feature_name in features:
            if isinstance(features[feature_name], (int, float)):
                print(f"  ✅ {feature_name:20s} = {features[feature_name]}")
            else:
                shape = features[feature_name].shape
                dtype = features[feature_name].dtype
                print(f"  ✅ {feature_name:20s} {shape} {dtype}")
        else:
            print(f"  ❌ {feature_name:20s} MISSING")
            all_present = False
    
    # Check feature ranges
    print("\n" + "-" * 60)
    print("Feature Value Ranges:")
    print("-" * 60)
    
    ranges_valid = True
    for feature_name in ['linearity', 'planarity', 'sphericity', 'anisotropy', 'roughness']:
        if feature_name in features:
            values = features[feature_name]
            min_val = np.min(values)
            max_val = np.max(values)
            has_nan = np.any(np.isnan(values))
            has_inf = np.any(np.isinf(values))
            
            print(f"  {feature_name:15s}: [{min_val:.6f}, {max_val:.6f}]", end="")
            
            if min_val < 0.0 or max_val > 1.0 or has_nan or has_inf:
                print("  ❌")
                ranges_valid = False
            else:
                print("  ✅")
    
    # Check density
    if 'density' in features:
        density = features['density']
        min_d = np.min(density)
        max_d = np.max(density)
        print(f"  {'density':15s}: [{min_d:.2f}, {max_d:.2f}]", end="")
        
        if min_d < 0.0 or max_d > 1000.0 or np.any(np.isnan(density)) or np.any(np.isinf(density)):
            print("  ❌")
            ranges_valid = False
        else:
            print("  ✅")
    
    # Check verticality and horizontality
    for feature_name in ['verticality', 'horizontality']:
        if feature_name in features:
            values = features[feature_name]
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"  {feature_name:15s}: [{min_val:.6f}, {max_val:.6f}]", end="")
            
            if min_val < 0.0 or max_val > 1.0 or np.any(np.isnan(values)) or np.any(np.isinf(values)):
                print("  ❌")
                ranges_valid = False
            else:
                print("  ✅")
    
    # Final result
    print("\n" + "=" * 60)
    if all_present and ranges_valid:
        print("✅ PHASE 2 COMPLETE - All features present and valid!")
    else:
        if not all_present:
            print("❌ MISSING FEATURES")
        if not ranges_valid:
            print("❌ INVALID FEATURE RANGES")
    print("=" * 60)
    
    return 0 if (all_present and ranges_valid) else 1


if __name__ == "__main__":
    sys.exit(main())
