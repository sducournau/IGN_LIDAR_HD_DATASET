#!/usr/bin/env python3
"""
Compare classification results between two point clouds (before/after fixes).

Creates side-by-side visualizations and generates a detailed comparison report
showing improvements in building classification.

Usage:
    python scripts/compare_classifications.py <before.laz> <after.laz> [output_dir]

Author: Classification Quality Audit - V2
Date: October 24, 2025
"""
import sys
import os
import numpy as np
from pathlib import Path

try:
    import laspy
except ImportError:
    print("❌ Error: laspy not installed. Run: pip install laspy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("❌ Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def load_and_analyze(las_path: str):
    """Load point cloud and extract statistics."""
    las = laspy.read(las_path)
    points = np.vstack([las.x, las.y, las.z]).T
    labels = las.classification
    
    stats = {
        'total_points': len(points),
        'labels': labels,
        'points': points,
    }
    
    # Count each class
    unique_labels = np.unique(labels)
    class_counts = {}
    for label in unique_labels:
        count = np.sum(labels == label)
        pct = count / len(labels) * 100
        class_counts[int(label)] = {'count': count, 'pct': pct}
    
    stats['class_counts'] = class_counts
    
    # Building-specific stats
    building_mask = labels == 6
    stats['building_count'] = np.sum(building_mask)
    stats['building_pct'] = stats['building_count'] / len(labels) * 100
    
    unclass_mask = labels == 1
    stats['unclass_count'] = np.sum(unclass_mask)
    stats['unclass_pct'] = stats['unclass_count'] / len(labels) * 100
    
    # Check for extra dimensions
    extra_dims = [dim.name for dim in las.point_format.extra_dimensions]
    
    if 'BuildingConfidence' in extra_dims:
        conf = las['BuildingConfidence'][building_mask]
        stats['building_conf_mean'] = np.nanmean(conf) if len(conf) > 0 else 0.0
        stats['building_conf_std'] = np.nanstd(conf) if len(conf) > 0 else 0.0
    else:
        stats['building_conf_mean'] = None
        stats['building_conf_std'] = None
    
    if 'height_above_ground' in extra_dims:
        hag = las['height_above_ground']
        elevated_unclass = np.sum((unclass_mask) & (hag > 2.5))
        stats['elevated_unclass_count'] = elevated_unclass
        stats['elevated_unclass_pct'] = elevated_unclass / len(labels) * 100
    else:
        stats['elevated_unclass_count'] = None
        stats['elevated_unclass_pct'] = None
    
    return stats


def create_comparison_visualization(stats_before, stats_after, output_path):
    """Create side-by-side comparison visualization."""
    
    # Color map for classes
    color_map = {
        0: '#808080',   # Never classified - Gray
        1: '#C0C0C0',   # Unclassified - Light Gray
        2: '#8B4513',   # Ground - Brown
        3: '#90EE90',   # Low Vegetation - Light Green
        4: '#32CD32',   # Medium Vegetation - Green
        5: '#006400',   # High Vegetation - Dark Green
        6: '#FF0000',   # Building - RED
        7: '#000000',   # Low Point (noise) - Black
        9: '#0000FF',   # Water - Blue
        10: '#800080',  # Rail - Purple
        11: '#404040',  # Road Surface - Dark Gray
        17: '#8B008B',  # Bridge Deck - Dark Magenta
        18: '#FF00FF',  # High Noise - Magenta
    }
    
    class_names = {
        0: "Never classified",
        1: "Unclassified",
        2: "Ground",
        3: "Low Veg",
        4: "Med Veg",
        5: "High Veg",
        6: "Building",
        7: "Low Point",
        9: "Water",
        10: "Rail",
        11: "Road",
        17: "Bridge",
        18: "High Noise",
    }
    
    # Sample points for visualization
    max_points = 250000
    
    points_before = stats_before['points']
    labels_before = stats_before['labels']
    if len(points_before) > max_points:
        idx = np.random.choice(len(points_before), max_points, replace=False)
        points_before = points_before[idx]
        labels_before = labels_before[idx]
    
    points_after = stats_after['points']
    labels_after = stats_after['labels']
    if len(points_after) > max_points:
        idx = np.random.choice(len(points_after), max_points, replace=False)
        points_after = points_after[idx]
        labels_after = labels_after[idx]
    
    colors_before = [color_map.get(int(label), '#808080') for label in labels_before]
    colors_after = [color_map.get(int(label), '#808080') for label in labels_after]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle('Classification Comparison: Before vs After', fontsize=16, fontweight='bold')
    
    # Before - Top down view
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(points_before[:, 0], points_before[:, 1], c=colors_before, s=1.0, alpha=0.8)
    ax1.set_title('BEFORE - Top-Down View', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # After - Top down view
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(points_after[:, 0], points_after[:, 1], c=colors_after, s=1.0, alpha=0.8)
    ax2.set_title('AFTER - Top-Down View', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Class distribution comparison
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Get common classes
    all_classes = sorted(set(stats_before['class_counts'].keys()) | set(stats_after['class_counts'].keys()))
    
    class_labels = []
    pcts_before = []
    pcts_after = []
    colors_chart = []
    
    for cls in all_classes:
        if cls in [0, 1, 2, 6, 9, 11]:  # Key classes
            class_labels.append(f"{cls}: {class_names.get(cls, 'Unknown')}")
            pcts_before.append(stats_before['class_counts'].get(cls, {'pct': 0})['pct'])
            pcts_after.append(stats_after['class_counts'].get(cls, {'pct': 0})['pct'])
            colors_chart.append(color_map.get(cls, '#808080'))
    
    x = np.arange(len(class_labels))
    width = 0.35
    
    ax3.barh(x - width/2, pcts_before, width, label='Before', alpha=0.8, color='#999999')
    ax3.barh(x + width/2, pcts_after, width, label='After', alpha=0.8, color=colors_chart)
    ax3.set_yticks(x)
    ax3.set_yticklabels(class_labels, fontsize=9)
    ax3.set_xlabel('Percentage (%)')
    ax3.set_title('Class Distribution Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, axis='x', alpha=0.3)
    
    # Key metrics comparison
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    
    metrics_text = "📊 KEY METRICS COMPARISON\n" + "="*50 + "\n\n"
    
    # Building classification
    build_before = stats_before['building_pct']
    build_after = stats_after['building_pct']
    build_change = build_after - build_before
    build_icon = "✅" if build_change > 0 else "⚠️"
    
    metrics_text += f"🏢 Building Classification:\n"
    metrics_text += f"   Before: {build_before:.2f}%\n"
    metrics_text += f"   After:  {build_after:.2f}%\n"
    metrics_text += f"   Change: {build_icon} {build_change:+.2f}% ({build_change/build_before*100:+.1f}%)\n\n"
    
    # Unclassified rate
    unclass_before = stats_before['unclass_pct']
    unclass_after = stats_after['unclass_pct']
    unclass_change = unclass_after - unclass_before
    unclass_icon = "✅" if unclass_change < 0 else "❌"
    
    metrics_text += f"⚪ Unclassified Points:\n"
    metrics_text += f"   Before: {unclass_before:.2f}%\n"
    metrics_text += f"   After:  {unclass_after:.2f}%\n"
    metrics_text += f"   Change: {unclass_icon} {unclass_change:+.2f}% ({unclass_change/unclass_before*100:+.1f}%)\n\n"
    
    # Elevated unclassified
    if stats_before['elevated_unclass_pct'] is not None and stats_after['elevated_unclass_pct'] is not None:
        elev_before = stats_before['elevated_unclass_pct']
        elev_after = stats_after['elevated_unclass_pct']
        elev_change = elev_after - elev_before
        elev_icon = "✅" if elev_change < 0 else "❌"
        
        metrics_text += f"🔺 Elevated Unclassified (HAG>2.5m):\n"
        metrics_text += f"   Before: {elev_before:.2f}%\n"
        metrics_text += f"   After:  {elev_after:.2f}%\n"
        metrics_text += f"   Change: {elev_icon} {elev_change:+.2f}%\n\n"
    
    # Confidence scores
    if stats_before['building_conf_mean'] is not None and stats_after['building_conf_mean'] is not None:
        conf_before = stats_before['building_conf_mean']
        conf_after = stats_after['building_conf_mean']
        conf_change = conf_after - conf_before
        conf_icon = "✅" if conf_change > 0 else "⚠️"
        
        metrics_text += f"📈 Building Confidence (mean):\n"
        metrics_text += f"   Before: {conf_before:.3f} ± {stats_before['building_conf_std']:.3f}\n"
        metrics_text += f"   After:  {conf_after:.3f} ± {stats_after['building_conf_std']:.3f}\n"
        metrics_text += f"   Change: {conf_icon} {conf_change:+.3f}\n"
    
    ax4.text(0.05, 0.95, metrics_text, 
             transform=ax4.transAxes,
             fontsize=10,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall assessment
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    
    assessment_text = "📋 OVERALL ASSESSMENT\n" + "="*50 + "\n\n"
    
    # Determine status
    if build_after > build_before * 1.5 and unclass_after < unclass_before * 0.6:
        status = "✅ EXCELLENT"
        assessment_text += "✅ Status: EXCELLENT IMPROVEMENT\n\n"
        assessment_text += "Significant improvements across all metrics.\n"
        assessment_text += "Building classification substantially increased.\n"
        assessment_text += "Unclassified rate substantially reduced.\n"
    elif build_after > build_before * 1.2 and unclass_after < unclass_before * 0.8:
        status = "✅ GOOD"
        assessment_text += "✅ Status: GOOD IMPROVEMENT\n\n"
        assessment_text += "Notable improvements in classification.\n"
        assessment_text += "Building coverage increased.\n"
        assessment_text += "Unclassified rate reduced.\n"
    elif build_after > build_before or unclass_after < unclass_before:
        status = "⚠️ MODERATE"
        assessment_text += "⚠️ Status: MODERATE IMPROVEMENT\n\n"
        assessment_text += "Some improvements visible.\n"
        assessment_text += "Consider applying V3 configuration for better results.\n"
    else:
        status = "❌ MINIMAL"
        assessment_text += "❌ Status: MINIMAL IMPROVEMENT\n\n"
        assessment_text += "Limited or no improvement detected.\n"
        assessment_text += "Review configuration parameters.\n"
        assessment_text += "Check ground truth data availability.\n"
    
    assessment_text += "\n"
    assessment_text += f"Building Detection: {build_before:.1f}% → {build_after:.1f}%\n"
    assessment_text += f"Unclassified Rate: {unclass_before:.1f}% → {unclass_after:.1f}%\n"
    
    ax5.text(0.05, 0.95, assessment_text,
             transform=ax5.transAxes,
             fontsize=10,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Recommendations
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    recommendations_text = "💡 RECOMMENDATIONS\n" + "="*50 + "\n\n"
    
    if unclass_after > 20:
        recommendations_text += "🔴 High unclassified rate (>20%)\n"
        recommendations_text += "   → Consider V3 configuration\n"
        recommendations_text += "   → Lower confidence thresholds\n\n"
    elif unclass_after > 15:
        recommendations_text += "🟡 Moderate unclassified rate (15-20%)\n"
        recommendations_text += "   → Try V3 if further reduction needed\n\n"
    else:
        recommendations_text += "✅ Low unclassified rate (<15%)\n"
        recommendations_text += "   → Current configuration is effective\n\n"
    
    if build_after < 10:
        recommendations_text += "🔴 Low building detection (<10%)\n"
        recommendations_text += "   → Check ground truth availability\n"
        recommendations_text += "   → Verify DTM computation\n"
        recommendations_text += "   → Review polygon alignment\n\n"
    elif build_after < 15:
        recommendations_text += "🟡 Moderate building detection (10-15%)\n"
        recommendations_text += "   → Room for improvement\n"
        recommendations_text += "   → Consider V3 configuration\n\n"
    else:
        recommendations_text += "✅ Good building detection (>15%)\n"
        recommendations_text += "   → Classification working well\n\n"
    
    recommendations_text += "📚 Next Steps:\n"
    recommendations_text += "   1. Review visualizations\n"
    recommendations_text += "   2. Check diagnostic reports\n"
    recommendations_text += "   3. Validate specific areas\n"
    recommendations_text += "   4. Process full dataset if satisfied\n"
    
    ax6.text(0.05, 0.95, recommendations_text,
             transform=ax6.transAxes,
             fontsize=10,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison visualization to: {output_path}")
    
    return status


def generate_text_report(stats_before, stats_after, output_path):
    """Generate detailed text comparison report."""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLASSIFICATION COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("POINT CLOUD STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Points (Before): {stats_before['total_points']:,}\n")
        f.write(f"Total Points (After):  {stats_after['total_points']:,}\n")
        f.write("\n")
        
        f.write("BUILDING CLASSIFICATION\n")
        f.write("-"*80 + "\n")
        build_before = stats_before['building_pct']
        build_after = stats_after['building_pct']
        build_change = build_after - build_before
        build_change_pct = (build_change / build_before * 100) if build_before > 0 else 0
        
        f.write(f"Before:  {stats_before['building_count']:,} points ({build_before:.2f}%)\n")
        f.write(f"After:   {stats_after['building_count']:,} points ({build_after:.2f}%)\n")
        f.write(f"Change:  {build_change:+.2f}% ({build_change_pct:+.1f}%)\n")
        f.write(f"Status:  {'✅ Improved' if build_change > 0 else '❌ Declined'}\n")
        f.write("\n")
        
        f.write("UNCLASSIFIED POINTS\n")
        f.write("-"*80 + "\n")
        unclass_before = stats_before['unclass_pct']
        unclass_after = stats_after['unclass_pct']
        unclass_change = unclass_after - unclass_before
        unclass_change_pct = (unclass_change / unclass_before * 100) if unclass_before > 0 else 0
        
        f.write(f"Before:  {stats_before['unclass_count']:,} points ({unclass_before:.2f}%)\n")
        f.write(f"After:   {stats_after['unclass_count']:,} points ({unclass_after:.2f}%)\n")
        f.write(f"Change:  {unclass_change:+.2f}% ({unclass_change_pct:+.1f}%)\n")
        f.write(f"Status:  {'✅ Reduced' if unclass_change < 0 else '❌ Increased'}\n")
        f.write("\n")
        
        if stats_before['elevated_unclass_pct'] is not None:
            f.write("ELEVATED UNCLASSIFIED POINTS (HAG > 2.5m)\n")
            f.write("-"*80 + "\n")
            f.write(f"Before:  {stats_before['elevated_unclass_count']:,} points "
                    f"({stats_before['elevated_unclass_pct']:.2f}%)\n")
            f.write(f"After:   {stats_after['elevated_unclass_count']:,} points "
                    f"({stats_after['elevated_unclass_pct']:.2f}%)\n")
            elev_change = stats_after['elevated_unclass_pct'] - stats_before['elevated_unclass_pct']
            f.write(f"Change:  {elev_change:+.2f}%\n")
            f.write(f"Status:  {'✅ Reduced' if elev_change < 0 else '❌ Increased'}\n")
            f.write("\n")
        
        if stats_before['building_conf_mean'] is not None:
            f.write("BUILDING CONFIDENCE SCORES\n")
            f.write("-"*80 + "\n")
            f.write(f"Before:  {stats_before['building_conf_mean']:.3f} ± "
                    f"{stats_before['building_conf_std']:.3f}\n")
            f.write(f"After:   {stats_after['building_conf_mean']:.3f} ± "
                    f"{stats_after['building_conf_std']:.3f}\n")
            conf_change = stats_after['building_conf_mean'] - stats_before['building_conf_mean']
            f.write(f"Change:  {conf_change:+.3f}\n")
            f.write("\n")
        
        f.write("CLASS DISTRIBUTION (Top Classes)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<20} {'Before (%)':<15} {'After (%)':<15} {'Change':<15}\n")
        f.write("-"*80 + "\n")
        
        class_names = {
            1: "Unclassified",
            2: "Ground",
            6: "Building",
            9: "Water",
            11: "Road",
        }
        
        for cls in [1, 2, 6, 9, 11]:
            name = class_names.get(cls, f"Class {cls}")
            pct_before = stats_before['class_counts'].get(cls, {'pct': 0})['pct']
            pct_after = stats_after['class_counts'].get(cls, {'pct': 0})['pct']
            change = pct_after - pct_before
            f.write(f"{name:<20} {pct_before:>12.2f}   {pct_after:>12.2f}   {change:>+12.2f}\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Saved text report to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/compare_classifications.py <before.laz> <after.laz> [output_dir]")
        print("\nExample:")
        print("  python scripts/compare_classifications.py \\")
        print("    output/original/tile_enriched.laz \\")
        print("    output/v2_fixed/tile_enriched.laz \\")
        print("    comparison_results")
        sys.exit(1)
    
    before_file = sys.argv[1]
    after_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "comparison_results"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("🔍 Loading and analyzing classification results...")
    print(f"   Before: {before_file}")
    print(f"   After:  {after_file}")
    print()
    
    # Load and analyze both files
    print("📊 Analyzing BEFORE...")
    stats_before = load_and_analyze(before_file)
    
    print("📊 Analyzing AFTER...")
    stats_after = load_and_analyze(after_file)
    
    print()
    print("📈 Generating comparison visualization...")
    viz_path = os.path.join(output_dir, "comparison_visualization.png")
    status = create_comparison_visualization(stats_before, stats_after, viz_path)
    
    print("📝 Generating text report...")
    report_path = os.path.join(output_dir, "comparison_report.txt")
    generate_text_report(stats_before, stats_after, report_path)
    
    print()
    print("="*80)
    print(f"✅ Comparison complete! Status: {status}")
    print(f"📁 Results saved to: {output_dir}/")
    print("="*80)
