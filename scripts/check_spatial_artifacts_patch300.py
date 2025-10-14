#!/usr/bin/env python3
"""
Check for spatial scan line artifacts in patch 0300.
These appear as dash lines or striping patterns in feature visualizations.
"""

import numpy as np
import os

def check_scan_line_artifacts():
    """Check for scan line artifacts in LAZ file"""
    
    laz_path = "/mnt/c/Users/Simon/ign/versailles/output/LHD_FXX_0635_6857_PTS_C_LAMB93_IGN69_hybrid_patch_0300.laz"
    
    print("\n" + "="*80)
    print("CHECKING FOR SCAN LINE ARTIFACTS IN PATCH 0300")
    print("="*80)
    
    try:
        import laspy
    except ImportError:
        print("ERROR: laspy not installed")
        return
    
    las = laspy.read(laz_path)
    
    # Get coordinates
    x = las.X * las.header.scales[0] + las.header.offsets[0]
    y = las.Y * las.header.scales[1] + las.header.offsets[1]
    z = las.Z * las.header.scales[2] + las.header.offsets[2]
    
    # Get features of interest
    target_features = ['planarity', 'roof_score', 'linearity']
    
    print(f"\nPoint cloud bounds:")
    print(f"  X: [{x.min():.2f}, {x.max():.2f}] (range: {x.max()-x.min():.2f}m)")
    print(f"  Y: [{y.min():.2f}, {y.max():.2f}] (range: {y.max()-y.min():.2f}m)")
    print(f"  Z: [{z.min():.2f}, {z.max():.2f}] (range: {z.max()-z.min():.2f}m)")
    print(f"  Total points: {len(x)}")
    
    # Check scan angle for scan line patterns
    if 'scan_angle' in las.point_format.dimension_names:
        scan_angle = las.scan_angle
        print(f"\nScan angle range: [{scan_angle.min()}, {scan_angle.max()}]")
        unique_angles = np.unique(scan_angle)
        print(f"Unique scan angles: {len(unique_angles)}")
    
    # Check gps_time for temporal patterns
    if 'gps_time' in las.point_format.dimension_names:
        gps_time = las.gps_time
        print(f"\nGPS time range: [{gps_time.min():.2f}, {gps_time.max():.2f}]")
        time_sorted_idx = np.argsort(gps_time)
        
        # Check if features vary systematically with time (scan line artifact)
        for feat_name in target_features:
            if feat_name in las.point_format.dimension_names:
                feat = las[feat_name]
                feat_sorted_by_time = feat[time_sorted_idx]
                
                # Check for periodic patterns in feature values along scan lines
                # Split into chunks and check variance
                n_chunks = 50
                chunk_size = len(feat_sorted_by_time) // n_chunks
                chunk_means = []
                chunk_stds = []
                
                for i in range(n_chunks):
                    start = i * chunk_size
                    end = start + chunk_size if i < n_chunks - 1 else len(feat_sorted_by_time)
                    chunk = feat_sorted_by_time[start:end]
                    chunk_means.append(np.mean(chunk))
                    chunk_stds.append(np.std(chunk))
                
                chunk_means = np.array(chunk_means)
                chunk_stds = np.array(chunk_stds)
                
                # Check if chunk means vary significantly (indicates scan line artifacts)
                mean_of_chunks = np.mean(chunk_means)
                std_of_chunk_means = np.std(chunk_means)
                
                print(f"\n{feat_name.upper()} temporal analysis:")
                print(f"  Overall mean: {feat.mean():.4f}")
                print(f"  Mean of chunk means: {mean_of_chunks:.4f}")
                print(f"  Std of chunk means: {std_of_chunk_means:.4f}")
                print(f"  Coefficient of variation: {std_of_chunk_means/mean_of_chunks:.4f}")
                
                # High CV indicates scan line artifacts
                if std_of_chunk_means / mean_of_chunks > 0.15:
                    print(f"  ⚠️  WARNING: High variation in chunk means - possible scan line artifact!")
    
    # Spatial grid analysis - check for striping in X or Y direction
    print("\n" + "="*80)
    print("SPATIAL GRID ANALYSIS")
    print("="*80)
    
    for feat_name in target_features:
        if feat_name not in las.point_format.dimension_names:
            continue
            
        feat = las[feat_name]
        
        print(f"\n{feat_name.upper()}:")
        
        # Create spatial grid and check for striping
        n_bins = 20
        
        # Check X-direction striping
        x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
        x_digitized = np.digitize(x, x_bins)
        
        x_bin_means = []
        for i in range(1, n_bins + 1):
            mask = x_digitized == i
            if mask.sum() > 0:
                x_bin_means.append(feat[mask].mean())
            else:
                x_bin_means.append(np.nan)
        
        x_bin_means = np.array(x_bin_means)
        x_bin_means_valid = x_bin_means[~np.isnan(x_bin_means)]
        
        if len(x_bin_means_valid) > 0:
            x_std = np.std(x_bin_means_valid)
            x_mean = np.mean(x_bin_means_valid)
            x_cv = x_std / x_mean if x_mean > 0 else 0
            
            print(f"  X-direction:")
            print(f"    Mean across bins: {x_mean:.4f}")
            print(f"    Std across bins: {x_std:.4f}")
            print(f"    CV: {x_cv:.4f}")
            
            if x_cv > 0.20:
                print(f"    ⚠️  WARNING: High variation in X-direction - possible striping!")
        
        # Check Y-direction striping
        y_bins = np.linspace(y.min(), y.max(), n_bins + 1)
        y_digitized = np.digitize(y, y_bins)
        
        y_bin_means = []
        for i in range(1, n_bins + 1):
            mask = y_digitized == i
            if mask.sum() > 0:
                y_bin_means.append(feat[mask].mean())
            else:
                y_bin_means.append(np.nan)
        
        y_bin_means = np.array(y_bin_means)
        y_bin_means_valid = y_bin_means[~np.isnan(y_bin_means)]
        
        if len(y_bin_means_valid) > 0:
            y_std = np.std(y_bin_means_valid)
            y_mean = np.mean(y_bin_means_valid)
            y_cv = y_std / y_mean if y_mean > 0 else 0
            
            print(f"  Y-direction:")
            print(f"    Mean across bins: {y_mean:.4f}")
            print(f"    Std across bins: {y_std:.4f}")
            print(f"    CV: {y_cv:.4f}")
            
            if y_cv > 0.20:
                print(f"    ⚠️  WARNING: High variation in Y-direction - possible striping!")
    
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    print("""
Scan line artifacts (dash lines) typically appear when:

1. TEMPORAL ARTIFACTS: Features computed on different scan lines have 
   systematically different values due to:
   - Different neighborhood sizes across scan lines
   - Edge effects at scan line boundaries
   - Varying point densities between scan lines

2. SPATIAL STRIPING: Features show banding patterns in X or Y direction due to:
   - Scan line orientation artifacts
   - Varying search radii at patch boundaries
   - Inconsistent neighbor counting near edges

Look for:
- High CV (>0.15-0.20) in temporal analysis
- High CV (>0.20) in spatial grid analysis
- Systematic patterns in the chunk means

To fix:
- Ensure consistent neighborhood search radii
- Handle scan line boundaries properly
- Use overlap regions between patches
- Apply boundary smoothing or edge trimming
""")

if __name__ == "__main__":
    check_scan_line_artifacts()
