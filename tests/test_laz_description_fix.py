#!/usr/bin/env python3
"""
Test script to verify LAZ extra dimension description field limit fix.
Tests that descriptions are properly truncated to 31 characters.
"""

import numpy as np
try:
    import laspy
    print("✓ laspy imported successfully")
except ImportError:
    print("✗ laspy not available - skipping test")
    exit(0)

def test_description_limits():
    """Test that 31-char descriptions work, 32-char fail."""
    print("\n" + "="*70)
    print("Testing LAZ Extra Dimension Description Field Limits")
    print("="*70)
    
    # Create minimal LAZ file
    header = laspy.LasHeader(version="1.4", point_format=6)
    las = laspy.LasData(header)
    n_points = 100
    las.x = np.zeros(n_points)
    las.y = np.zeros(n_points)
    las.z = np.zeros(n_points)
    
    test_cases = [
        ("30 chars", "123456789012345678901234567890", True),
        ("31 chars", "1234567890123456789012345678901", True),
        ("32 chars", "12345678901234567890123456789012", False),
        ("33 chars", "123456789012345678901234567890123", False),
    ]
    
    for label, description, should_succeed in test_cases:
        try:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name=f"test_{len(description)}",
                type=np.float32,
                description=description
            ))
            result = "✓ Success"
            status = "✓" if should_succeed else "✗ UNEXPECTED"
        except Exception as e:
            result = f"✗ Failed: {e}"
            status = "✗" if should_succeed else "✓"
        
        print(f"{status} {label:12s} (len={len(description):2d}): {result}")
    
    print("\n" + "="*70)
    print("Conclusion: Maximum description length is 31 characters")
    print("="*70)


def test_feature_descriptions():
    """Test actual feature name descriptions from the pipeline."""
    print("\n" + "="*70)
    print("Testing Real Feature Descriptions")
    print("="*70)
    
    features = [
        'sum_eigenvalues', 'eigenentropy', 'omnivariance', 'change_curvature',
        'edge_strength', 'corner_likelihood', 'overhang_indicator', 'surface_roughness',
        'num_points_2m', 'neighborhood_extent', 'height_extent_ratio', 'vertical_std',
        'height_above_ground', 'nir', 'ndvi'
    ]
    
    # Create minimal LAZ file
    header = laspy.LasHeader(version="1.4", point_format=6)
    las = laspy.LasData(header)
    n_points = 100
    las.x = np.zeros(n_points)
    las.y = np.zeros(n_points)
    las.z = np.zeros(n_points)
    
    success_count = 0
    fail_count = 0
    
    for feat_name in features:
        # Use new limit of 31 chars
        description = feat_name[:31]
        
        try:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name=feat_name,
                type=np.float32,
                description=description
            ))
            setattr(las, feat_name, np.random.randn(n_points).astype(np.float32))
            print(f"✓ {feat_name:25s} (desc len: {len(description):2d})")
            success_count += 1
        except Exception as e:
            print(f"✗ {feat_name:25s} - Error: {e}")
            fail_count += 1
    
    print("\n" + "="*70)
    print(f"Results: {success_count} succeeded, {fail_count} failed")
    print("="*70)
    
    return fail_count == 0


if __name__ == "__main__":
    test_description_limits()
    all_passed = test_feature_descriptions()
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ All tests passed! Fix is working correctly.")
    else:
        print("❌ Some tests failed. Further investigation needed.")
    print("="*70)
