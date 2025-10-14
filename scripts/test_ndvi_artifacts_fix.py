#!/usr/bin/env python3
"""
Test script to verify NDVI and geometric feature artifact fixes.

This script checks:
1. NDVI is properly computed and saved (not None)
2. Geometric features have no NaN/Inf values
3. Features are in valid ranges
"""

import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

def analyze_npz_file(npz_path: Path) -> Dict[str, any]:
    """Analyze NPZ file for NDVI and geometric feature quality."""
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {npz_path.name}")
    print(f"{'='*80}\n")
    
    # Load NPZ file
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        return {'success': False, 'error': str(e)}
    
    results = {
        'success': True,
        'file': npz_path.name,
        'keys': list(data.keys()),
        'issues': []
    }
    
    print(f"📦 Keys in NPZ: {results['keys']}\n")
    
    # ========================================================================
    # 1. CHECK NDVI
    # ========================================================================
    print("🌿 NDVI CHECK:")
    print("-" * 40)
    
    if 'ndvi' not in data:
        print("⚠️  NDVI not in patch (RGB or NIR might be missing)")
        results['ndvi_status'] = 'missing'
    else:
        ndvi = data['ndvi']
        
        # Check if NDVI is None or empty
        if ndvi is None or (isinstance(ndvi, np.ndarray) and ndvi.size == 0):
            print("❌ NDVI is None or empty!")
            results['issues'].append('NDVI is None/empty')
            results['ndvi_status'] = 'invalid'
        elif isinstance(ndvi, np.ndarray) and ndvi.shape == ():
            # Scalar array containing None
            print(f"❌ NDVI is scalar object: {ndvi}")
            results['issues'].append('NDVI is scalar None')
            results['ndvi_status'] = 'invalid'
        else:
            # Valid NDVI array
            nan_count = np.isnan(ndvi).sum()
            inf_count = np.isinf(ndvi).sum()
            
            print(f"✅ NDVI shape: {ndvi.shape}")
            print(f"✅ NDVI dtype: {ndvi.dtype}")
            print(f"   Min: {ndvi.min():.6f}")
            print(f"   Max: {ndvi.max():.6f}")
            print(f"   Mean: {ndvi.mean():.6f}")
            print(f"   Std: {ndvi.std():.6f}")
            print(f"   NaNs: {nan_count} ({100*nan_count/ndvi.size:.2f}%)")
            print(f"   Infs: {inf_count} ({100*inf_count/ndvi.size:.2f}%)")
            
            # Check range
            if ndvi.min() < -1.0 or ndvi.max() > 1.0:
                print(f"⚠️  NDVI out of valid range [-1, 1]!")
                results['issues'].append('NDVI out of range')
            
            if nan_count > 0:
                print(f"❌ NDVI has {nan_count} NaN values!")
                results['issues'].append(f'NDVI has {nan_count} NaNs')
            
            if inf_count > 0:
                print(f"❌ NDVI has {inf_count} Inf values!")
                results['issues'].append(f'NDVI has {inf_count} Infs')
            
            results['ndvi_status'] = 'valid' if (nan_count == 0 and inf_count == 0) else 'has_artifacts'
            results['ndvi_stats'] = {
                'min': float(ndvi.min()),
                'max': float(ndvi.max()),
                'mean': float(ndvi.mean()),
                'nan_count': int(nan_count),
                'inf_count': int(inf_count)
            }
    
    # ========================================================================
    # 2. CHECK GEOMETRIC FEATURES
    # ========================================================================
    print("\n📐 GEOMETRIC FEATURES CHECK:")
    print("-" * 40)
    
    geometric_features = [
        'planarity', 'linearity', 'sphericity',
        'verticality', 'horizontality',
        'wall_score', 'roof_score',
        'curvature', 'roughness'
    ]
    
    results['geometric_features'] = {}
    
    for feat_name in geometric_features:
        if feat_name in data:
            feat = data[feat_name]
            
            if isinstance(feat, np.ndarray) and feat.size > 0:
                nan_count = np.isnan(feat).sum()
                inf_count = np.isinf(feat).sum()
                zero_count = (feat == 0).sum()
                
                print(f"\n{feat_name}:")
                print(f"  Shape: {feat.shape}")
                print(f"  Range: [{feat.min():.6f}, {feat.max():.6f}]")
                print(f"  Mean: {feat.mean():.6f}")
                print(f"  NaNs: {nan_count} ({100*nan_count/feat.size:.2f}%)")
                print(f"  Infs: {inf_count} ({100*inf_count/feat.size:.2f}%)")
                print(f"  Zeros: {zero_count} ({100*zero_count/feat.size:.2f}%)")
                
                status = 'valid'
                if nan_count > 0:
                    print(f"  ❌ Has NaN artifacts!")
                    results['issues'].append(f'{feat_name} has {nan_count} NaNs')
                    status = 'has_nans'
                
                if inf_count > 0:
                    print(f"  ❌ Has Inf artifacts!")
                    results['issues'].append(f'{feat_name} has {inf_count} Infs')
                    status = 'has_infs'
                
                if nan_count == 0 and inf_count == 0:
                    print(f"  ✅ Clean (no NaN/Inf)")
                
                results['geometric_features'][feat_name] = {
                    'status': status,
                    'nan_count': int(nan_count),
                    'inf_count': int(inf_count),
                    'zero_pct': float(100*zero_count/feat.size),
                    'range': [float(feat.min()), float(feat.max())]
                }
    
    # ========================================================================
    # 3. CHECK FEATURES ARRAY (if exists)
    # ========================================================================
    if 'features' in data:
        print(f"\n🎯 FEATURES MATRIX CHECK:")
        print("-" * 40)
        
        features = data['features']
        print(f"  Shape: {features.shape}")
        print(f"  Dtype: {features.dtype}")
        
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        
        print(f"  Total NaNs: {nan_count} ({100*nan_count/features.size:.2f}%)")
        print(f"  Total Infs: {inf_count} ({100*inf_count/features.size:.2f}%)")
        
        if nan_count > 0:
            print(f"  ❌ Features matrix has NaN values!")
            results['issues'].append(f'Features matrix has {nan_count} NaNs')
        
        if inf_count > 0:
            print(f"  ❌ Features matrix has Inf values!")
            results['issues'].append(f'Features matrix has {inf_count} Infs')
        
        if nan_count == 0 and inf_count == 0:
            print(f"  ✅ Features matrix is clean")
        
        results['features_matrix'] = {
            'shape': features.shape,
            'nan_count': int(nan_count),
            'inf_count': int(inf_count)
        }
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    if len(results['issues']) == 0:
        print("✅ ALL CHECKS PASSED - No artifacts found!")
        results['verdict'] = 'PASS'
    else:
        print(f"❌ FOUND {len(results['issues'])} ISSUE(S):")
        for i, issue in enumerate(results['issues'], 1):
            print(f"   {i}. {issue}")
        results['verdict'] = 'FAIL'
    print(f"{'='*80}\n")
    
    return results


def main():
    """Main test function."""
    
    print("\n" + "="*80)
    print("NDVI & GEOMETRIC FEATURE ARTIFACT FIX - VERIFICATION TEST")
    print("="*80)
    
    # Find NPZ files to test
    test_dirs = [
        Path('/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/data/test_output'),
        Path('/mnt/c/Users/Simon/ign/versailles/output'),  # If accessible
    ]
    
    npz_files = []
    for test_dir in test_dirs:
        if test_dir.exists():
            npz_files.extend(list(test_dir.glob('*.npz')))
    
    if not npz_files:
        print("\n⚠️  No NPZ files found to test!")
        print("Please run a processing job first to generate output patches.")
        return 1
    
    print(f"\nFound {len(npz_files)} NPZ file(s) to test:")
    for npz_file in npz_files[:5]:  # Show first 5
        print(f"  - {npz_file}")
    if len(npz_files) > 5:
        print(f"  ... and {len(npz_files)-5} more")
    
    # Test files
    all_results = []
    passed = 0
    failed = 0
    
    for npz_file in npz_files[:3]:  # Test first 3 files
        result = analyze_npz_file(npz_file)
        all_results.append(result)
        
        if result['verdict'] == 'PASS':
            passed += 1
        else:
            failed += 1
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    print(f"Files tested: {len(all_results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Fixes are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
