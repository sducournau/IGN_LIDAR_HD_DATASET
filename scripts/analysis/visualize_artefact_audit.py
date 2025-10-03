#!/usr/bin/env python3
"""
Visualize Artefact Audit Results

This script creates visualizations demonstrating that artefact fixes
do not affect other geometric features.

Usage:
    python scripts/analysis/visualize_artefact_audit.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ign_lidar.features import (
    compute_all_features_optimized,
    extract_geometric_features
)


def create_test_geometries():
    """Create synthetic test geometries."""
    np.random.seed(42)
    
    # 1. Planar surface (roof)
    planar = np.random.randn(300, 3) * [2, 2, 0.05]
    planar[:, 2] += 10
    
    # 2. Linear structure (edge)
    linear = np.zeros((200, 3))
    linear[:, 0] = np.linspace(0, 10, 200)
    linear[:, 1:] = np.random.randn(200, 2) * 0.03
    linear[:, 2] += 5
    
    # 3. Spherical (vegetation-like)
    spherical = np.random.randn(300, 3) * 0.5
    spherical[:, 2] += 2
    
    return planar, linear, spherical


def test_radius_vs_knn():
    """Compare radius-based vs k-NN based feature extraction."""
    print("=" * 70)
    print("ARTEFACT AUDIT: RADIUS vs K-NN COMPARISON")
    print("=" * 70)
    
    planar, linear, spherical = create_test_geometries()
    points = np.vstack([planar, linear, spherical]).astype(np.float32)
    classification = np.full(len(points), 6, dtype=np.uint8)
    
    # Compute features with both methods
    print("\n1. Computing with k-NN (k=20)...")
    normals, curvature, height, geo_knn = compute_all_features_optimized(
        points=points,
        classification=classification,
        k=20,
        auto_k=False,
        include_extra=False
    )
    
    print("2. Computing with radius-based (auto)...")
    geo_radius = extract_geometric_features(
        points=points,
        normals=normals,
        k=None,
        radius=None  # Auto-estimate
    )
    
    # Compare features
    print("\n" + "=" * 70)
    print("FEATURE COMPARISON")
    print("=" * 70)
    
    features = ['linearity', 'planarity', 'sphericity', 'anisotropy', 'roughness']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(features):
        vals_knn = geo_knn[feat]
        vals_radius = geo_radius[feat]
        
        ax = axes[idx]
        
        # Scatter plot: k-NN vs radius
        ax.scatter(vals_knn, vals_radius, alpha=0.3, s=1)
        ax.plot([0, 1], [0, 1], 'r--', label='y=x (perfect match)')
        ax.set_xlabel(f'{feat} (k-NN)', fontsize=10)
        ax.set_ylabel(f'{feat} (radius)', fontsize=10)
        ax.set_title(f'{feat.capitalize()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Compute correlation
        corr = np.corrcoef(vals_knn, vals_radius)[0, 1]
        ax.text(0.05, 0.95, f'r={corr:.4f}', 
                transform=ax.transAxes, 
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Print stats
        diff = np.abs(vals_knn - vals_radius)
        print(f"\n{feat}:")
        print(f"  Correlation: {corr:.6f}")
        print(f"  Mean abs diff: {np.mean(diff):.6f}")
        print(f"  Max abs diff: {np.max(diff):.6f}")
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Artefact Audit: k-NN vs Radius-Based Features\n(High correlation = No cross-contamination)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'artefact_audit_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_file}")
    
    return fig


def test_feature_independence():
    """Test independence of features across different geometries."""
    print("\n" + "=" * 70)
    print("ARTEFACT AUDIT: FEATURE INDEPENDENCE")
    print("=" * 70)
    
    planar, linear, spherical = create_test_geometries()
    
    geometries = {
        'Planar (Roof)': planar,
        'Linear (Edge)': linear,
        'Spherical (Vegetation)': spherical
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    for idx, (name, geom) in enumerate(geometries.items()):
        points = geom.astype(np.float32)
        classification = np.full(len(points), 6, dtype=np.uint8)
        
        print(f"\n{name} ({len(points)} points):")
        
        # Compute features
        normals, curvature, height, geo_features = compute_all_features_optimized(
            points=points,
            classification=classification,
            k=20,
            auto_k=False,
            include_extra=False
        )
        
        # Extract feature values
        feature_names = ['linearity', 'planarity', 'sphericity', 
                        'anisotropy', 'roughness', 'density']
        means = [np.mean(geo_features[f]) for f in feature_names]
        stds = [np.std(geo_features[f]) for f in feature_names]
        
        # Print stats
        for fname, mean, std in zip(feature_names, means, stds):
            print(f"  {fname:12s}: {mean:.4f} ± {std:.4f}")
        
        # Plot
        ax = axes[idx]
        x_pos = np.arange(len(feature_names))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('Feature Value', fontsize=11)
        ax.set_title(f'{name} - Feature Profile', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(means) * 1.3])
    
    plt.suptitle('Artefact Audit: Feature Independence Across Geometries\n'
                 '(Different geometries show distinct feature profiles)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'artefact_audit_independence.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved independence plot: {output_file}")
    
    return fig


def test_cross_contamination():
    """Test for cross-contamination between features."""
    print("\n" + "=" * 70)
    print("ARTEFACT AUDIT: CROSS-CONTAMINATION TEST")
    print("=" * 70)
    
    # Create mixed geometry
    planar, linear, spherical = create_test_geometries()
    points = np.vstack([planar, linear, spherical]).astype(np.float32)
    classification = np.full(len(points), 6, dtype=np.uint8)
    
    print(f"\nMixed geometry: {len(points)} points")
    
    # Compute features
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points,
        classification=classification,
        k=20,
        auto_k=False,
        include_extra=False
    )
    
    # Create correlation matrix
    feature_names = ['linearity', 'planarity', 'sphericity', 
                    'anisotropy', 'roughness']
    n_features = len(feature_names)
    
    corr_matrix = np.zeros((n_features, n_features))
    for i, feat_i in enumerate(feature_names):
        for j, feat_j in enumerate(feature_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                vals_i = geo_features[feat_i]
                vals_j = geo_features[feat_j]
                corr_matrix[i, j] = np.corrcoef(vals_i, vals_j)[0, 1]
    
    print("\nFeature Correlation Matrix:")
    print("(Low off-diagonal values = No cross-contamination)")
    print()
    print("         ", "  ".join(f"{f[:4]:>6s}" for f in feature_names))
    for i, feat in enumerate(feature_names):
        row_str = f"{feat[:4]:>8s} "
        row_str += "  ".join(f"{corr_matrix[i, j]:>6.3f}" for j in range(n_features))
        print(row_str)
    
    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels([f.capitalize() for f in feature_names], rotation=45, ha='right')
    ax.set_yticklabels([f.capitalize() for f in feature_names])
    
    # Add correlation values
    for i in range(n_features):
        for j in range(n_features):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                          fontsize=10, fontweight='bold')
    
    ax.set_title('Feature Cross-Correlation Matrix\n'
                 '(Diagonal = 1.0, Off-diagonal should be low)',
                 fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'artefact_audit_correlation.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved correlation plot: {output_file}")
    
    # Check for unexpected correlations
    print("\n" + "=" * 70)
    print("CROSS-CONTAMINATION ANALYSIS")
    print("=" * 70)
    
    threshold = 0.5
    contamination_found = False
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr = corr_matrix[i, j]
            if abs(corr) > threshold:
                print(f"⚠️  High correlation: {feature_names[i]} vs {feature_names[j]}: {corr:.3f}")
                contamination_found = True
    
    if not contamination_found:
        print("✅ NO CROSS-CONTAMINATION DETECTED")
        print("   All off-diagonal correlations < 0.5")
    
    return fig


def main():
    """Run all audit visualizations."""
    print("\n" + "=" * 70)
    print("ARTEFACT AUDIT VISUALIZATION SUITE")
    print("=" * 70)
    print("\nThis script validates that artefact fixes do NOT affect")
    print("other geometric features through comprehensive testing.")
    print()
    
    try:
        # Test 1: Radius vs k-NN
        fig1 = test_radius_vs_knn()
        
        # Test 2: Feature independence
        fig2 = test_feature_independence()
        
        # Test 3: Cross-contamination
        fig3 = test_cross_contamination()
        
        print("\n" + "=" * 70)
        print("✅ ALL AUDIT VISUALIZATIONS COMPLETE")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - artefact_audit_comparison.png")
        print("  - artefact_audit_independence.png")
        print("  - artefact_audit_correlation.png")
        print()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
