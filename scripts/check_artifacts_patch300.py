#!/usr/bin/env python3
"""
Check for dash line artifacts in planarity, roof_score, and linearity features
for patch 0300 in both NPZ and LAZ formats.
"""

import numpy as np
import sys
import os

def analyze_npz_artifacts(npz_path):
    """Analyze NPZ file for artifacts in specific features"""
    print("\n" + "="*80)
    print("ANALYZING NPZ FILE FOR ARTIFACTS")
    print("="*80)
    print(f"\nFile: {npz_path}")
    
    if not os.path.exists(npz_path):
        print(f"ERROR: File not found: {npz_path}")
        return
    
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"\nArrays in file: {data.files}")
    
    # Get features array
    if 'features' not in data.files:
        print("ERROR: 'features' array not found in NPZ file")
        data.close()
        return
    
    features = data['features']
    print(f"\nFeatures shape: {features.shape}")
    
    # Get metadata to identify feature names
    if 'metadata' in data.files:
        metadata = data['metadata'].item()
        if isinstance(metadata, dict) and 'feature_names' in metadata:
            feature_names = metadata['feature_names']
            print(f"\nFeature names: {feature_names}")
            
            # Find indices of interest
            target_features = ['planarity', 'roof_score', 'linearity']
            feature_indices = {}
            
            for target in target_features:
                if target in feature_names:
                    idx = feature_names.index(target)
                    feature_indices[target] = idx
                    print(f"\n{target} found at index {idx}")
                else:
                    print(f"\nWARNING: {target} not found in feature names")
            
            # Analyze each target feature
            print("\n" + "="*80)
            print("ARTIFACT ANALYSIS")
            print("="*80)
            
            for feat_name, feat_idx in feature_indices.items():
                feat_values = features[:, feat_idx]
                
                print(f"\n{feat_name.upper()} (index {feat_idx}):")
                print(f"  Shape: {feat_values.shape}")
                print(f"  Min: {feat_values.min():.6f}")
                print(f"  Max: {feat_values.max():.6f}")
                print(f"  Mean: {feat_values.mean():.6f}")
                print(f"  Std: {feat_values.std():.6f}")
                print(f"  Median: {np.median(feat_values):.6f}")
                
                # Check for NaN/Inf
                n_nan = np.isnan(feat_values).sum()
                n_inf = np.isinf(feat_values).sum()
                print(f"  NaN count: {n_nan}")
                print(f"  Inf count: {n_inf}")
                
                # Check for suspicious patterns (e.g., many zeros or constant values)
                n_zeros = (feat_values == 0).sum()
                n_ones = (feat_values == 1).sum()
                print(f"  Zero count: {n_zeros} ({100*n_zeros/len(feat_values):.2f}%)")
                print(f"  One count: {n_ones} ({100*n_ones/len(feat_values):.2f}%)")
                
                # Check for repeated values (potential artifacts)
                unique_vals, counts = np.unique(feat_values, return_counts=True)
                most_common_idx = np.argsort(counts)[-5:][::-1]  # Top 5 most common
                print(f"  Unique values: {len(unique_vals)}")
                print(f"  Top 5 most common values:")
                for i in most_common_idx:
                    val = unique_vals[i]
                    count = counts[i]
                    pct = 100 * count / len(feat_values)
                    print(f"    Value {val:.6f}: {count} occurrences ({pct:.2f}%)")
                
                # Histogram analysis
                print(f"\n  Value distribution:")
                hist, bin_edges = np.histogram(feat_values, bins=10)
                for i in range(len(hist)):
                    pct = 100 * hist[i] / len(feat_values)
                    print(f"    [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {hist[i]} ({pct:.1f}%)")
        else:
            print("\nWARNING: No feature_names in metadata")
    else:
        print("\nWARNING: No metadata in NPZ file")
    
    data.close()

def analyze_laz_artifacts(laz_path):
    """Analyze LAZ file for artifacts in specific features"""
    print("\n" + "="*80)
    print("ANALYZING LAZ FILE FOR ARTIFACTS")
    print("="*80)
    print(f"\nFile: {laz_path}")
    
    if not os.path.exists(laz_path):
        print(f"ERROR: File not found: {laz_path}")
        return
    
    try:
        import laspy
    except ImportError:
        print("ERROR: laspy not installed. Run: pip install laspy")
        return
    
    las = laspy.read(laz_path)
    print(f"\nPoint count: {len(las.points)}")
    print(f"\nAvailable dimensions: {list(las.point_format.dimension_names)}")
    
    target_features = ['planarity', 'roof_score', 'linearity']
    
    print("\n" + "="*80)
    print("ARTIFACT ANALYSIS")
    print("="*80)
    
    for feat_name in target_features:
        if feat_name in las.point_format.dimension_names:
            feat_values = las[feat_name]
            
            print(f"\n{feat_name.upper()}:")
            print(f"  Shape: {feat_values.shape}")
            print(f"  Min: {feat_values.min():.6f}")
            print(f"  Max: {feat_values.max():.6f}")
            print(f"  Mean: {feat_values.mean():.6f}")
            print(f"  Std: {feat_values.std():.6f}")
            print(f"  Median: {np.median(feat_values):.6f}")
            
            # Check for NaN/Inf
            n_nan = np.isnan(feat_values).sum()
            n_inf = np.isinf(feat_values).sum()
            print(f"  NaN count: {n_nan}")
            print(f"  Inf count: {n_inf}")
            
            # Check for suspicious patterns
            n_zeros = (feat_values == 0).sum()
            n_ones = (feat_values == 1).sum()
            print(f"  Zero count: {n_zeros} ({100*n_zeros/len(feat_values):.2f}%)")
            print(f"  One count: {n_ones} ({100*n_ones/len(feat_values):.2f}%)")
            
            # Check for repeated values
            unique_vals, counts = np.unique(feat_values, return_counts=True)
            most_common_idx = np.argsort(counts)[-5:][::-1]
            print(f"  Unique values: {len(unique_vals)}")
            print(f"  Top 5 most common values:")
            for i in most_common_idx:
                val = unique_vals[i]
                count = counts[i]
                pct = 100 * count / len(feat_values)
                print(f"    Value {val:.6f}: {count} occurrences ({pct:.2f}%)")
            
            # Histogram analysis
            print(f"\n  Value distribution:")
            hist, bin_edges = np.histogram(feat_values, bins=10)
            for i in range(len(hist)):
                pct = 100 * hist[i] / len(feat_values)
                print(f"    [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {hist[i]} ({pct:.1f}%)")
        else:
            print(f"\n{feat_name.upper()}: NOT FOUND in LAZ file")

def main():
    base_path = "/mnt/c/Users/Simon/ign/versailles/output"
    patch_name = "LHD_FXX_0635_6857_PTS_C_LAMB93_IGN69_hybrid_patch_0300"
    
    npz_path = os.path.join(base_path, f"{patch_name}.npz")
    laz_path = os.path.join(base_path, f"{patch_name}.laz")
    
    print("\n" + "="*80)
    print("CHECKING FOR DASH LINE ARTIFACTS IN PATCH 0300")
    print("="*80)
    print("\nTarget features: planarity, roof_score, linearity")
    print("Looking for patterns that might appear as dash lines in visualization")
    
    # Analyze NPZ
    analyze_npz_artifacts(npz_path)
    
    # Analyze LAZ
    analyze_laz_artifacts(laz_path)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Artifacts typically appear as:
1. Repeated constant values (e.g., many 0s or 1s)
2. Bands of specific values creating visual lines
3. NaN or Inf values
4. Extreme outliers
5. Quantization artifacts (values clustered at specific intervals)

Check the histograms and top values above for suspicious patterns.
If you see many points with the same value, this could create "dash lines"
when visualized spatially.
""")

if __name__ == "__main__":
    main()
