#!/usr/bin/env python3
"""
Quick visual inspection of geometric features for artifact detection.

This script generates histograms and statistics to help identify artifacts
in geometric features (linearity, planarity, sphericity).

Usage:
    python inspect_features.py <enriched_laz> [--save-plots]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import laspy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_features(laz_path: Path) -> dict:
    """Load geometric features from enriched LAZ file."""
    print(f"Loading: {laz_path}")
    las = laspy.read(laz_path)
    
    features = {}
    feature_names = [
        'planarity', 'linearity', 'sphericity',
        'anisotropy', 'roughness', 'density'
    ]
    
    for name in feature_names:
        if hasattr(las, name):
            features[name] = np.array(getattr(las, name))
    
    # Also get classification for context
    features['classification'] = np.array(las.classification)
    
    # Get normals if available
    if hasattr(las, 'normal_z'):
        features['normal_z'] = np.array(las.normal_z)
    
    print(f"✓ Loaded {len(features)} feature arrays")
    return features


def detect_artifacts(values: np.ndarray, name: str) -> dict:
    """Detect potential artifacts in feature distribution."""
    artifacts = {}
    
    # Check for bimodal distribution (potential artifact)
    hist, edges = np.histogram(values, bins=50)
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            if hist[i] > 0.1 * hist.max():  # Significant peak
                peaks.append((edges[i], hist[i]))
    
    if len(peaks) > 2:
        artifacts['multiple_peaks'] = {
            'count': len(peaks),
            'warning': f"Found {len(peaks)} peaks - may indicate artifacts"
        }
    
    # Check for unusual spikes
    std = np.std(hist)
    mean = np.mean(hist)
    spikes = hist > (mean + 3 * std)
    if np.any(spikes):
        artifacts['spikes'] = {
            'count': np.sum(spikes),
            'warning': f"Found {np.sum(spikes)} statistical spikes"
        }
    
    # Check for unnatural gaps
    zero_bins = hist == 0
    if np.sum(zero_bins) > len(hist) * 0.3:  # >30% empty bins
        artifacts['gaps'] = {
            'percentage': 100 * np.sum(zero_bins) / len(hist),
            'warning': f"{100*np.sum(zero_bins)/len(hist):.1f}% empty bins"
        }
    
    return artifacts


def analyze_features(features: dict):
    """Analyze features and detect potential artifacts."""
    print("\n" + "="*70)
    print("FEATURE ANALYSIS")
    print("="*70)
    
    n_points = len(features['classification'])
    print(f"\nTotal points: {n_points:,}")
    
    # Classification breakdown
    print("\nClassification breakdown:")
    unique_classes, counts = np.unique(
        features['classification'],
        return_counts=True
    )
    for cls, count in zip(unique_classes, counts):
        pct = 100 * count / n_points
        print(f"  Class {cls:2d}: {count:8,} ({pct:5.1f}%)")
    
    # Feature statistics
    print("\n" + "-"*70)
    print(f"{'Feature':<15} {'Mean':<10} {'Std':<10} "
          f"{'Min':<10} {'Max':<10} {'Issues'}")
    print("-"*70)
    
    all_artifacts = {}
    feature_names = ['planarity', 'linearity', 'sphericity',
                    'anisotropy', 'roughness', 'density']
    
    for name in feature_names:
        if name in features:
            values = features[name]
            artifacts = detect_artifacts(values, name)
            
            mean = np.mean(values)
            std = np.std(values)
            vmin = np.min(values)
            vmax = np.max(values)
            
            issues = len(artifacts)
            issue_str = f"⚠️ {issues}" if issues > 0 else "✓"
            
            print(f"{name:<15} {mean:>9.4f} {std:>9.4f} "
                  f"{vmin:>9.4f} {vmax:>9.4f} {issue_str:>10}")
            
            if artifacts:
                all_artifacts[name] = artifacts
    
    # Report artifacts
    if all_artifacts:
        print("\n" + "="*70)
        print("POTENTIAL ARTIFACTS DETECTED")
        print("="*70)
        
        for name, artifacts in all_artifacts.items():
            print(f"\n{name.upper()}:")
            for artifact_type, info in artifacts.items():
                print(f"  ⚠️ {artifact_type}: {info['warning']}")
    else:
        print("\n✓ No obvious artifacts detected")
    
    return all_artifacts


def plot_features(features: dict, output_path: Path = None):
    """Create visualization of feature distributions."""
    print("\nGenerating plots...")
    
    feature_names = ['planarity', 'linearity', 'sphericity']
    n_features = len(feature_names)
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main histograms
    for i, name in enumerate(feature_names):
        if name not in features:
            continue
        
        values = features[name]
        
        # Histogram
        ax = fig.add_subplot(gs[i, 0:2])
        ax.hist(values, bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel(name.capitalize())
        ax.set_ylabel('Count')
        ax.set_title(f'{name.capitalize()} Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean = np.mean(values)
        std = np.std(values)
        ax.axvline(mean, color='red', linestyle='--',
                   label=f'Mean: {mean:.3f}')
        ax.axvline(mean + std, color='orange', linestyle=':',
                   label=f'±Std: {std:.3f}')
        ax.axvline(mean - std, color='orange', linestyle=':')
        ax.legend()
        
        # Box plot
        ax2 = fig.add_subplot(gs[i, 2])
        ax2.boxplot(values)
        ax2.set_ylabel(name.capitalize())
        ax2.set_title('Box Plot')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Geometric Feature Analysis', fontsize=16, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect geometric features for artifacts"
    )
    parser.add_argument('input', type=Path, help='Enriched LAZ file')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to file instead of displaying')
    parser.add_argument('--output', type=Path,
                       help='Output plot file (default: <input>_analysis.png)')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        return 1
    
    # Load features
    try:
        features = load_features(args.input)
    except Exception as e:
        print(f"Error loading features: {e}")
        return 1
    
    # Analyze
    artifacts = analyze_features(features)
    
    # Plot
    if args.save_plots or args.output:
        if args.output:
            output_path = args.output
        else:
            output_path = args.input.parent / f"{args.input.stem}_analysis.png"
        plot_features(features, output_path)
    else:
        plot_features(features)
    
    # Summary
    print("\n" + "="*70)
    if artifacts:
        print("⚠️  ARTIFACTS DETECTED - Review plots and consider:")
        print("   1. Using --radius parameter for better feature computation")
        print("   2. Checking if using latest version (v1.1.0+)")
        print("   3. Reviewing RADIUS_PARAMETER_GUIDE.md for tuning")
    else:
        print("✓ FEATURES LOOK GOOD - No obvious artifacts detected")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
