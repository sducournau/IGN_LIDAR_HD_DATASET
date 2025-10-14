#!/usr/bin/env python3
"""
Batch analyze all patches in a directory for boundary artifacts.

This script processes multiple LAZ files and generates a report showing
which patches have artifacts and how severe they are.

Usage:
    python scripts/batch_analyze_artifacts.py \
        --input /mnt/c/Users/Simon/ign/versailles/output/*.laz \
        --output artifact_report.csv
"""

import numpy as np
import argparse
import os
from pathlib import Path
import csv
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_single_patch(filepath: str, 
                         n_bins: int = 20) -> Dict:
    """
    Analyze a single LAZ file for spatial artifacts.
    
    Returns:
        Dictionary with artifact metrics
    """
    try:
        import laspy
    except ImportError:
        logger.error("laspy not installed")
        return {}
    
    try:
        las = laspy.read(filepath)
        
        # Get coordinates
        x = las.X * las.header.scales[0] + las.header.offsets[0]
        y = las.Y * las.header.scales[1] + las.header.offsets[1]
        
        target_features = ['planarity', 'roof_score', 'linearity']
        results = {
            'file': os.path.basename(filepath),
            'n_points': len(las.points),
            'x_range': x.max() - x.min(),
            'y_range': y.max() - y.min(),
        }
        
        # Analyze each feature
        for feat_name in target_features:
            if feat_name not in las.point_format.dimension_names:
                results[f'{feat_name}_cv_x'] = np.nan
                results[f'{feat_name}_cv_y'] = np.nan
                continue
            
            feat = las[feat_name]
            
            # X-direction analysis
            x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
            x_digitized = np.digitize(x, x_bins)
            
            x_means = []
            for i in range(1, n_bins + 1):
                mask = x_digitized == i
                if mask.sum() > 0:
                    x_means.append(feat[mask].mean())
            
            x_means = np.array(x_means)
            x_cv = np.std(x_means) / np.mean(x_means) if np.mean(x_means) > 0 else 0
            
            # Y-direction analysis
            y_bins = np.linspace(y.min(), y.max(), n_bins + 1)
            y_digitized = np.digitize(y, y_bins)
            
            y_means = []
            for i in range(1, n_bins + 1):
                mask = y_digitized == i
                if mask.sum() > 0:
                    y_means.append(feat[mask].mean())
            
            y_means = np.array(y_means)
            y_cv = np.std(y_means) / np.mean(y_means) if np.mean(y_means) > 0 else 0
            
            results[f'{feat_name}_cv_x'] = x_cv
            results[f'{feat_name}_cv_y'] = y_cv
            results[f'{feat_name}_mean'] = feat.mean()
        
        # Determine severity
        max_cv = max([
            results.get('planarity_cv_y', 0),
            results.get('roof_score_cv_y', 0),
            results.get('linearity_cv_y', 0)
        ])
        
        if max_cv > 0.30:
            results['severity'] = 'SEVERE'
        elif max_cv > 0.20:
            results['severity'] = 'HIGH'
        elif max_cv > 0.15:
            results['severity'] = 'MODERATE'
        else:
            results['severity'] = 'OK'
        
        results['max_cv'] = max_cv
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return {'file': os.path.basename(filepath), 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Batch analyze patches for boundary artifacts'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input LAZ file pattern (e.g., *.laz)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='artifact_analysis.csv',
        help='Output CSV report'
    )
    parser.add_argument(
        '--n_bins',
        type=int,
        default=20,
        help='Number of spatial bins for analysis'
    )
    
    args = parser.parse_args()
    
    # Find files
    from glob import glob
    files = glob(args.input)
    
    if not files:
        logger.error(f"No files found matching: {args.input}")
        return
    
    logger.info(f"Found {len(files)} files to analyze")
    
    # Analyze all files
    results = []
    for i, filepath in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Analyzing {os.path.basename(filepath)}")
        result = analyze_single_patch(filepath, args.n_bins)
        if result:
            results.append(result)
    
    if not results:
        logger.error("No results to save")
        return
    
    # Save to CSV
    fieldnames = [
        'file', 'severity', 'max_cv', 'n_points', 'x_range', 'y_range',
        'planarity_cv_x', 'planarity_cv_y', 'planarity_mean',
        'roof_score_cv_x', 'roof_score_cv_y', 'roof_score_mean',
        'linearity_cv_x', 'linearity_cv_y', 'linearity_mean'
    ]
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Saved results to {args.output}")
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH ARTIFACT ANALYSIS SUMMARY")
    print("="*80)
    
    severity_counts = {}
    for result in results:
        sev = result.get('severity', 'UNKNOWN')
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    print(f"\nTotal files analyzed: {len(results)}")
    print("\nSeverity Distribution:")
    for sev in ['SEVERE', 'HIGH', 'MODERATE', 'OK', 'UNKNOWN']:
        count = severity_counts.get(sev, 0)
        if count > 0:
            pct = 100 * count / len(results)
            print(f"  {sev:10s}: {count:4d} ({pct:5.1f}%)")
    
    # List worst patches
    sorted_results = sorted(results, key=lambda x: x.get('max_cv', 0), reverse=True)
    
    print("\nTop 10 Patches with Artifacts:")
    print(f"{'Rank':<6} {'File':<50} {'Max CV':<10} {'Severity':<10}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results[:10], 1):
        filename = result.get('file', 'unknown')
        max_cv = result.get('max_cv', 0)
        severity = result.get('severity', 'UNKNOWN')
        print(f"{i:<6} {filename:<50} {max_cv:<10.3f} {severity:<10}")
    
    print("\n" + "="*80)
    print(f"Detailed results saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
