#!/usr/bin/env python3
"""
Quick test to generate a patch with the NDVI and geometric feature fixes applied.
"""

import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, '/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET')

from ign_lidar.features.orchestrator import FeatureOrchestrator

def test_feature_computation():
    """Test feature computation with synthetic data."""
    
    print("\n" + "="*80)
    print("TESTING NDVI & GEOMETRIC FEATURE FIXES")
    print("="*80 + "\n")
    
    # Create synthetic point cloud
    np.random.seed(42)
    N = 5000
    
    # Create a building-like structure
    points = np.random.randn(N, 3).astype(np.float32)
    points[:, 0] *= 20  # X: spread 20m
    points[:, 1] *= 20  # Y: spread 20m
    points[:, 2] *= 5   # Z: height 5m
    
    classification = np.random.choice([2, 6], size=N).astype(np.uint8)  # Ground and building
    intensity = np.random.randint(0, 255, size=N).astype(np.uint16)
    return_number = np.ones(N, dtype=np.uint8)
    
    # Create synthetic RGB and NIR
    rgb = np.random.randint(50, 200, size=(N, 3)).astype(np.uint8)
    nir = np.random.randint(100, 255, size=N).astype(np.uint8)
    
    tile_data = {
        'points': points,
        'classification': classification,
        'intensity': intensity,
        'return_number': return_number,
        'input_rgb': rgb,
        'input_nir': nir
    }
    
    # Create config for feature computation
    config = {
        'processor': {
            'compute_ndvi': True,
            'use_rgb': True,
            'use_infrared': True,
            'k_neighbors': 30
        },
        'features': {
            'mode': 'full',
            'use_gpu': False,  # Use CPU for quick test
            'k_neighbors': 30
        }
    }
    
    print("📊 Computing features with orchestrator...")
    print(f"   Points: {N}")
    print(f"   Has RGB: Yes")
    print(f"   Has NIR: Yes")
    print(f"   compute_ndvi: True\n")
    
    # Create orchestrator and compute features
    orchestrator = FeatureOrchestrator(config)
    features = orchestrator.compute_features(tile_data, use_enriched=False)
    
    # ========================================================================
    # CHECK RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80 + "\n")
    
    issues = []
    
    # Check NDVI
    print("🌿 NDVI CHECK:")
    print("-" * 40)
    if 'ndvi' not in features:
        print("❌ NDVI not in features!")
        issues.append("NDVI missing")
    else:
        ndvi = features['ndvi']
        if ndvi is None or ndvi.size == 0:
            print("❌ NDVI is None or empty!")
            issues.append("NDVI is None/empty")
        else:
            nan_count = np.isnan(ndvi).sum()
            inf_count = np.isinf(ndvi).sum()
            
            print(f"✅ NDVI computed successfully")
            print(f"   Shape: {ndvi.shape}")
            print(f"   Range: [{ndvi.min():.4f}, {ndvi.max():.4f}]")
            print(f"   Mean: {ndvi.mean():.4f}")
            print(f"   NaNs: {nan_count}")
            print(f"   Infs: {inf_count}")
            
            if nan_count > 0:
                issues.append(f"NDVI has {nan_count} NaNs")
            if inf_count > 0:
                issues.append(f"NDVI has {inf_count} Infs")
            if ndvi.min() < -1.0 or ndvi.max() > 1.0:
                issues.append("NDVI out of range [-1, 1]")
    
    # Check geometric features
    print("\n📐 GEOMETRIC FEATURES CHECK:")
    print("-" * 40)
    
    geo_features = ['planarity', 'linearity', 'verticality', 'wall_score', 'roof_score', 'curvature']
    
    for feat_name in geo_features:
        if feat_name in features:
            feat = features[feat_name]
            nan_count = np.isnan(feat).sum()
            inf_count = np.isinf(feat).sum()
            
            status = "✅" if (nan_count == 0 and inf_count == 0) else "❌"
            print(f"{status} {feat_name}: shape={feat.shape}, range=[{feat.min():.4f}, {feat.max():.4f}], NaNs={nan_count}, Infs={inf_count}")
            
            if nan_count > 0:
                issues.append(f"{feat_name} has {nan_count} NaNs")
            if inf_count > 0:
                issues.append(f"{feat_name} has {inf_count} Infs")
    
    # Final verdict
    print("\n" + "="*80)
    if len(issues) == 0:
        print("🎉 ALL TESTS PASSED! Fixes are working correctly.")
        print("="*80 + "\n")
        return 0
    else:
        print(f"❌ FOUND {len(issues)} ISSUE(S):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("="*80 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(test_feature_computation())
