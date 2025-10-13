#!/usr/bin/env python3
"""
Check for missing features in full feature mode patches
"""

import numpy as np
import sys

# Expected features in LOD3 FULL mode (35+ features)
EXPECTED_FEATURES_LOD3_FULL = [
    # Core Geometric (9 features)
    'normal_x', 'normal_y', 'normal_z',
    'curvature', 'change_curvature',
    'planarity', 'linearity', 'sphericity', 'roughness',
    
    # Advanced Shape Descriptors (2 features)
    'anisotropy', 'omnivariance',
    
    # Eigenvalue Features (5 features)
    'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
    'sum_eigenvalues', 'eigenentropy',
    
    # Height Features (2 features)
    'height_above_ground', 'vertical_std',
    
    # Building-Specific Scores (3 features)
    'verticality', 'wall_score', 'roof_score',
    
    # Density & Neighborhood (4 features)
    'density', 'num_points_2m', 'neighborhood_extent', 'height_extent_ratio',
    
    # Architectural Features (4 features)
    'edge_strength', 'corner_likelihood', 'overhang_indicator', 'surface_roughness',
    
    # Spectral Features (5 features)
    'red', 'green', 'blue', 'nir', 'ndvi'
]

def check_npz_features(npz_path):
    """Check features in a single NPZ file"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        print(f"\n{'='*80}")
        print(f"Analyzing: {npz_path}")
        print(f"{'='*80}\n")
        
        # Check available arrays
        print("Available arrays in NPZ:")
        for key in sorted(data.files):
            if key in data:
                arr = data[key]
                print(f"  {key:25s} shape: {str(arr.shape):20s} dtype: {arr.dtype}")
        
        # Check features array specifically
        if 'features' in data.files:
            features_array = data['features']
            num_features = features_array.shape[1] if len(features_array.shape) > 1 else 1
            print(f"\n🔍 Features array contains: {num_features} features")
            print(f"   Expected for LOD3 FULL: {len(EXPECTED_FEATURES_LOD3_FULL)} features")
            
            if num_features < len(EXPECTED_FEATURES_LOD3_FULL):
                print(f"\n❌ MISSING FEATURES! Only {num_features} out of {len(EXPECTED_FEATURES_LOD3_FULL)} expected features found!")
                missing_count = len(EXPECTED_FEATURES_LOD3_FULL) - num_features
                print(f"   {missing_count} features are missing")
        
        # Check separate arrays that should be included
        separate_arrays = ['normals', 'rgb', 'curvature', 'verticality', 'ndvi', 'nir']
        present_separate = []
        missing_separate = []
        
        for arr_name in separate_arrays:
            if arr_name in data.files:
                present_separate.append(arr_name)
            else:
                missing_separate.append(arr_name)
        
        print(f"\n📦 Separate feature arrays:")
        print(f"   Present: {', '.join(present_separate)}")
        if missing_separate:
            print(f"   ❌ Missing: {', '.join(missing_separate)}")
        
        # Estimate what might be in the features array
        print(f"\n🔍 Attempting to identify features in the 'features' array:")
        print(f"   With {num_features} features, this likely contains:")
        
        # Common feature combinations
        if num_features == 12:
            print("   - Possible: planarity, linearity, sphericity, roughness (4)")
            print("   - Possible: anisotropy, omnivariance (2)")
            print("   - Possible: height_above_ground, vertical_std (2)")
            print("   - Possible: density, num_points_2m, neighborhood_extent, height_extent_ratio (4)")
            print("\n   ❌ LIKELY MISSING:")
            print("   - change_curvature")
            print("   - eigenvalue_1, eigenvalue_2, eigenvalue_3, sum_eigenvalues, eigenentropy (5)")
            print("   - wall_score, roof_score (2)")
            print("   - edge_strength, corner_likelihood, overhang_indicator, surface_roughness (4)")
            print(f"\n   Total missing: ~12 features")
        
        data.close()
        return num_features
        
    except Exception as e:
        print(f"Error analyzing {npz_path}: {e}")
        return None

def main():
    npz_file = "/mnt/c/Users/Simon/ign/versailles/output/LHD_FXX_0635_6857_PTS_C_LAMB93_IGN69_hybrid_patch_0000.npz"
    
    print("\n" + "="*80)
    print("FEATURE COMPLETENESS CHECK - LOD3 FULL MODE")
    print("="*80)
    
    num_features = check_npz_features(npz_file)
    
    if num_features:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"\n✅ Expected features in LOD3 FULL mode: {len(EXPECTED_FEATURES_LOD3_FULL)}")
        print(f"❌ Actual features in 'features' array: {num_features}")
        print(f"⚠️  Missing features: {len(EXPECTED_FEATURES_LOD3_FULL) - num_features}")
        
        print(f"\n📋 Complete list of expected features:")
        for i, feat in enumerate(EXPECTED_FEATURES_LOD3_FULL, 1):
            print(f"   {i:2d}. {feat}")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   The feature computation needs to be updated to include all {len(EXPECTED_FEATURES_LOD3_FULL)} features.")
        print(f"   Check the feature extraction code in 'ign_lidar/features/' directory.")

if __name__ == "__main__":
    main()
