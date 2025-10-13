#!/usr/bin/env python3
"""
Script to analyze feature distributions in NPZ files and detect anomalies
"""

import numpy as np
import glob
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_npz_file(npz_path):
    """Analyze a single NPZ file for anomalies"""
    try:
        data = np.load(npz_path)
        
        results = {
            'file': os.path.basename(npz_path),
            'features': {}
        }
        
        # Check each array in the NPZ
        for key in data.files:
            arr = data[key]
            
            if arr.dtype in [np.float32, np.float64]:
                # Statistical analysis
                has_nan = np.isnan(arr).any()
                has_inf = np.isinf(arr).any()
                
                if not has_nan and not has_inf:
                    mean_val = np.mean(arr)
                    std_val = np.std(arr)
                    min_val = np.min(arr)
                    max_val = np.max(arr)
                    median_val = np.median(arr)
                    
                    # Check for anomalies
                    anomalies = []
                    
                    # Check for all zeros or constant values
                    if std_val == 0:
                        anomalies.append(f"CONSTANT (all values = {mean_val:.6f})")
                    
                    # Check for extreme values
                    if max_val > 1e6 or min_val < -1e6:
                        anomalies.append(f"EXTREME VALUES (min={min_val:.2e}, max={max_val:.2e})")
                    
                    # Check for suspicious distributions (very high std compared to mean)
                    if mean_val != 0 and abs(std_val / mean_val) > 100:
                        anomalies.append(f"HIGH VARIANCE (std/mean = {abs(std_val / mean_val):.1f})")
                    
                    # Check for bimodal or strange distributions
                    if abs(median_val - mean_val) > 0.5 * std_val and std_val > 0.01:
                        anomalies.append(f"SKEWED (median={median_val:.4f} vs mean={mean_val:.4f})")
                    
                    results['features'][key] = {
                        'shape': arr.shape,
                        'dtype': str(arr.dtype),
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'median': median_val,
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'anomalies': anomalies
                    }
                else:
                    results['features'][key] = {
                        'shape': arr.shape,
                        'dtype': str(arr.dtype),
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'anomalies': ['NAN or INF values detected!']
                    }
            else:
                # For non-float arrays (like coordinates)
                results['features'][key] = {
                    'shape': arr.shape,
                    'dtype': str(arr.dtype),
                    'min': np.min(arr) if arr.size > 0 else None,
                    'max': np.max(arr) if arr.size > 0 else None,
                }
        
        data.close()
        return results
        
    except Exception as e:
        return {'file': os.path.basename(npz_path), 'error': str(e)}

def main():
    output_dir = "/mnt/c/Users/Simon/ign/versailles/output"
    
    # Find all NPZ files
    npz_files = sorted(glob.glob(os.path.join(output_dir, "*.npz")))
    
    print(f"\n{'='*80}")
    print(f"FEATURE ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"Found {len(npz_files)} NPZ files")
    print(f"Analyzing first 10 files for detailed inspection...\n")
    
    all_anomalies = {}
    feature_stats = {}
    
    # Analyze a sample of files
    for i, npz_file in enumerate(npz_files[:10]):
        print(f"\n{'='*80}")
        print(f"FILE: {os.path.basename(npz_file)}")
        print(f"{'='*80}")
        
        results = analyze_npz_file(npz_file)
        
        if 'error' in results:
            print(f"ERROR: {results['error']}")
            continue
        
        # Display results
        for feature_name, stats in results['features'].items():
            print(f"\n  Feature: {feature_name}")
            print(f"    Shape: {stats['shape']}")
            print(f"    Dtype: {stats['dtype']}")
            
            if 'mean' in stats:
                print(f"    Mean:   {stats['mean']:.6f}")
                print(f"    Std:    {stats['std']:.6f}")
                print(f"    Min:    {stats['min']:.6f}")
                print(f"    Max:    {stats['max']:.6f}")
                print(f"    Median: {stats['median']:.6f}")
            else:
                if 'min' in stats and stats['min'] is not None:
                    print(f"    Min:    {stats['min']}")
                    print(f"    Max:    {stats['max']}")
            
            if stats.get('has_nan'):
                print(f"    ⚠️  WARNING: Contains NaN values!")
            if stats.get('has_inf'):
                print(f"    ⚠️  WARNING: Contains Inf values!")
            
            # Display anomalies
            if 'anomalies' in stats and stats['anomalies']:
                print(f"    ⚠️  ANOMALIES DETECTED:")
                for anomaly in stats['anomalies']:
                    print(f"       - {anomaly}")
                
                # Track anomalies
                if feature_name not in all_anomalies:
                    all_anomalies[feature_name] = []
                all_anomalies[feature_name].append({
                    'file': results['file'],
                    'anomalies': stats['anomalies']
                })
            
            # Collect statistics across files
            if 'mean' in stats:
                if feature_name not in feature_stats:
                    feature_stats[feature_name] = {
                        'means': [],
                        'stds': [],
                        'mins': [],
                        'maxs': []
                    }
                feature_stats[feature_name]['means'].append(stats['mean'])
                feature_stats[feature_name]['stds'].append(stats['std'])
                feature_stats[feature_name]['mins'].append(stats['min'])
                feature_stats[feature_name]['maxs'].append(stats['max'])
    
    # Summary report
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ANOMALIES")
    print(f"{'='*80}\n")
    
    if all_anomalies:
        for feature_name, anomaly_list in all_anomalies.items():
            print(f"\n🔍 Feature: {feature_name}")
            print(f"   Found anomalies in {len(anomaly_list)} files:")
            for item in anomaly_list:
                print(f"     - {item['file']}")
                for anomaly in item['anomalies']:
                    print(f"         {anomaly}")
    else:
        print("✅ No major anomalies detected in the analyzed files!")
    
    # Feature statistics summary
    print(f"\n\n{'='*80}")
    print("FEATURE STATISTICS ACROSS FILES")
    print(f"{'='*80}\n")
    
    for feature_name, stats in feature_stats.items():
        print(f"\n📊 Feature: {feature_name}")
        means = np.array(stats['means'])
        stds = np.array(stats['stds'])
        mins = np.array(stats['mins'])
        maxs = np.array(stats['maxs'])
        
        print(f"   Mean across files:   {np.mean(means):.6f} ± {np.std(means):.6f}")
        print(f"   Std across files:    {np.mean(stds):.6f} ± {np.std(stds):.6f}")
        print(f"   Min across files:    {np.min(mins):.6f}")
        print(f"   Max across files:    {np.max(maxs):.6f}")
        
        # Check for inconsistencies across files
        if np.std(means) / (np.mean(means) + 1e-10) > 0.5:
            print(f"   ⚠️  WARNING: High variance in mean values across files!")
        
        if np.max(maxs) > 1e6 or np.min(mins) < -1e6:
            print(f"   ⚠️  WARNING: Extreme values detected across files!")

if __name__ == "__main__":
    main()
