#!/usr/bin/env python3
"""
Deep dive analysis - check if features are actually in the NPZ file
and trace where they might be getting lost
"""

import numpy as np
import sys

def analyze_npz_detailed(npz_path):
    """Detailed analysis of NPZ file structure"""
    data = np.load(npz_path, allow_pickle=True)
    
    print("\n" + "="*80)
    print("DETAILED NPZ ANALYSIS")
    print("="*80)
    
    print(f"\nFile: {npz_path}")
    print(f"Arrays in file: {len(data.files)}")
    
    # Analyze each array
    for key in sorted(data.files):
        arr = data[key]
        print(f"\n{key}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        
        # For feature array, try to understand its content
        if key == 'features' and len(arr.shape) == 2:
            num_points, num_features = arr.shape
            print(f"  Contains {num_features} features for {num_points} points")
            
            # Sample statistics for each feature dimension
            print(f"\n  Feature statistics (first 5 features):")
            for i in range(min(5, num_features)):
                feat_col = arr[:, i]
                print(f"    Feature {i}: min={feat_col.min():.4f}, max={feat_col.max():.4f}, mean={feat_col.mean():.4f}, std={feat_col.std():.4f}")
            
            print(f"\n  Feature statistics (last 5 features):")
            for i in range(max(0, num_features-5), num_features):
                feat_col = arr[:, i]
                print(f"    Feature {i}: min={feat_col.min():.4f}, max={feat_col.max():.4f}, mean={feat_col.mean():.4f}, std={feat_col.std():.4f}")
        
        # Check for NaN or Inf
        if arr.dtype in [np.float32, np.float64]:
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
            if has_nan or has_inf:
                print(f"  ⚠️  WARNING: Contains NaN={has_nan}, Inf={has_inf}")
    
    # Check metadata
    if 'metadata' in data.files:
        metadata = data['metadata'].item()
        print(f"\nMetadata:")
        if isinstance(metadata, dict):
            for k, v in sorted(metadata.items()):
                if k not in ['file_path', 'tile_name']:  # Skip long values
                    print(f"  {k}: {v}")
    
    data.close()

def main():
    npz_file = "/mnt/c/Users/Simon/ign/versailles/output/LHD_FXX_0635_6857_PTS_C_LAMB93_IGN69_hybrid_patch_0000.npz"
    
    analyze_npz_detailed(npz_file)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("""
The 'features' array contains only 12 features, but FULL mode should have 34+.

Possible causes:
1. Features were computed but not included in the patch dictionary
2. Features were filtered out during patch formatting  
3. Features were not saved to the NPZ file

Next steps:
1. Add logging to trace feature flow:
   - In compute_all_features_chunked() after computing features
   - In format_patch() to see what features are received
   - In save_patch_npz() to see what's being saved

2. Check if compute_eigenvalue_features, compute_architectural_features, 
   compute_density_features are actually being called and returning data

3. Verify that geo_features dict contains all features before filtering
""")

if __name__ == "__main__":
    main()
