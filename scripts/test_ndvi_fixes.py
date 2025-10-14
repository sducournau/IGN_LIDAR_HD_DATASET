#!/usr/bin/env python
"""
Test script to verify NDVI and geometric feature artifact fixes.

This script checks:
1. NDVI is properly computed or excluded
2. No NaN/Inf values in geometric features  
3. All features are in valid ranges
"""

import numpy as np
import sys
from pathlib import Path

def test_npz_file(npz_path: Path):
    """Test a single NPZ file for the fixes."""
    print(f"\n{'='*70}")
    print(f"Testing: {npz_path.name}")
    print(f"{'='*70}")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        return False
    
    print(f"\nKeys in NPZ: {list(data.keys())}")
    
    all_passed = True
    
    # Test 1: NDVI validation
    print(f"\n{'─'*70}")
    print("TEST 1: NDVI Validation")
    print(f"{'─'*70}")
    
    if 'ndvi' in data:
        ndvi = data['ndvi']
        print(f"✓ NDVI key found")
        print(f"  Type: {type(ndvi)}")
        print(f"  Dtype: {ndvi.dtype}")
        print(f"  Shape: {ndvi.shape}")
        
        # Check if it's None or empty
        if ndvi.dtype == object or ndvi.size == 0:
            print(f"  ❌ FAIL: NDVI is empty object or has no data")
            all_passed = False
        elif ndvi is None or (ndvi.size > 0 and ndvi.flatten()[0] is None):
            print(f"  ❌ FAIL: NDVI contains None values")
            all_passed = False
        else:
            # Check for NaN/Inf
            if ndvi.size > 0:
                nan_count = np.isnan(ndvi).sum()
                inf_count = np.isinf(ndvi).sum()
                
                print(f"  Min: {ndvi.min():.6f}")
                print(f"  Max: {ndvi.max():.6f}")
                print(f"  Mean: {ndvi.mean():.6f}")
                print(f"  NaN count: {nan_count} ({100*nan_count/ndvi.size:.2f}%)")
                print(f"  Inf count: {inf_count} ({100*inf_count/ndvi.size:.2f}%)")
                
                if nan_count > 0 or inf_count > 0:
                    print(f"  ❌ FAIL: NDVI contains NaN or Inf values")
                    all_passed = False
                else:
                    print(f"  ✓ PASS: NDVI is clean (no NaN/Inf)")
                
                # Check range
                if ndvi.min() < -1.1 or ndvi.max() > 1.1:
                    print(f"  ⚠️  WARNING: NDVI values outside expected range [-1, 1]")
            else:
                print(f"  ⚠️  WARNING: NDVI array is empty")
    else:
        print(f"ℹ️  NDVI not in patch (likely not computed due to missing RGB/NIR)")
    
    # Test 2: Geometric features validation
    print(f"\n{'─'*70}")
    print("TEST 2: Geometric Features Validation")
    print(f"{'─'*70}")
    
    geometric_features = [
        'planarity', 'linearity', 'sphericity', 'anisotropy',
        'roughness', 'omnivariance', 'curvature', 'change_curvature',
        'verticality', 'horizontality', 'wall_score', 'roof_score',
        'edge_strength', 'corner_likelihood', 'surface_roughness'
    ]
    
    for feat_name in geometric_features:
        if feat_name in data:
            feat = data[feat_name]
            
            nan_count = np.isnan(feat).sum()
            inf_count = np.isinf(feat).sum()
            
            status = "✓ PASS" if (nan_count == 0 and inf_count == 0) else "❌ FAIL"
            
            print(f"\n  {feat_name}:")
            print(f"    Status: {status}")
            print(f"    Shape: {feat.shape}")
            print(f"    Range: [{feat.min():.6f}, {feat.max():.6f}]")
            print(f"    NaN: {nan_count}, Inf: {inf_count}")
            
            if nan_count > 0 or inf_count > 0:
                print(f"    ❌ Contains invalid values!")
                all_passed = False
            
            # Check if all zeros (might indicate issue)
            zero_pct = (feat == 0).sum() / feat.size * 100
            if zero_pct > 90:
                print(f"    ⚠️  WARNING: {zero_pct:.1f}% zeros")
    
    # Test 3: Feature matrix validation
    print(f"\n{'─'*70}")
    print("TEST 3: Feature Matrix Validation")
    print(f"{'─'*70}")
    
    if 'features' in data:
        features = data['features']
        print(f"  Shape: {features.shape}")
        print(f"  Total elements: {features.size}")
        
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        
        print(f"  NaN count: {nan_count} ({100*nan_count/features.size:.4f}%)")
        print(f"  Inf count: {inf_count} ({100*inf_count/features.size:.4f}%)")
        
        if nan_count > 0 or inf_count > 0:
            print(f"  ❌ FAIL: Feature matrix contains NaN or Inf")
            all_passed = False
        else:
            print(f"  ✓ PASS: Feature matrix is clean")
    
    # Summary
    print(f"\n{'='*70}")
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*70}\n")
    
    return all_passed

def main():
    """Test all NPZ files in output directory."""
    import glob
    
    # Find NPZ files
    npz_pattern = "/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/data/test_output/*.npz"
    npz_files = glob.glob(npz_pattern)
    
    if not npz_files:
        print(f"No NPZ files found matching: {npz_pattern}")
        print("\nSearching for any NPZ files in workspace...")
        npz_pattern = "/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/**/*.npz"
        npz_files = glob.glob(npz_pattern, recursive=True)[:5]  # Limit to 5 files
    
    if not npz_files:
        print("No NPZ files found in workspace!")
        return 1
    
    print(f"\nFound {len(npz_files)} NPZ file(s) to test")
    
    all_passed = True
    for npz_path in npz_files:
        passed = test_npz_file(Path(npz_path))
        all_passed = all_passed and passed
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    if all_passed:
        print("✅ All files passed all tests!")
        return 0
    else:
        print("❌ Some files failed tests")
        return 1

if __name__ == "__main__":
    sys.exit(main())
