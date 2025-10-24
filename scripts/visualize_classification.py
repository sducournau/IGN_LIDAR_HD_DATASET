#!/usr/bin/env python3
"""
Visual validation of building classification.

Creates 2D and 3D visualizations of classification results to help
identify classification quality issues.

Usage:
    python scripts/visualize_classification.py <las_file> [output.png]

Author: Classification Quality Audit
Date: October 24, 2025
"""
import sys
import numpy as np

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


def visualize_classification(las_path: str, output_path: str = None, max_points: int = 500000):
    """Create visualization of classification results."""
    
    print(f"📊 Visualizing classification: {las_path}")
    
    # Load point cloud
    try:
        las = laspy.read(las_path)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)
    
    points = np.vstack([las.x, las.y, las.z]).T
    labels = las.classification
    
    print(f"  Total points: {len(points):,}")
    
    # Sample for visualization if too large
    if len(points) > max_points:
        print(f"  Sampling {max_points:,} points for visualization...")
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        labels = labels[idx]
    
    # Define colors (ASPRS standard)
    color_map = {
        0: '#808080',       # Never classified - Gray
        1: '#C0C0C0',       # Unclassified - Light Gray
        2: '#8B4513',       # Ground - Brown
        3: '#90EE90',       # Low Vegetation - Light Green
        4: '#32CD32',       # Medium Vegetation - Green
        5: '#006400',       # High Vegetation - Dark Green
        6: '#FF0000',       # Building - RED (highlight)
        7: '#000000',       # Low Point (noise) - Black
        9: '#0000FF',       # Water - Blue
        10: '#800080',      # Rail - Purple
        11: '#404040',      # Road Surface - Dark Gray
        13: '#FFD700',      # Wire - Guard - Gold
        14: '#FFA500',      # Wire - Conductor - Orange
        15: '#FF6347',      # Transmission Tower - Tomato
        17: '#8B008B',      # Bridge Deck - Dark Magenta
        18: '#FF00FF',      # High Noise - Magenta
    }
    
    # Map labels to colors
    point_colors = [color_map.get(int(label), '#808080') for label in labels]
    
    # Count buildings
    building_count = np.sum(labels == 6)
    building_pct = building_count / len(labels) * 100
    
    print(f"  Building points: {building_count:,} ({building_pct:.2f}%)")
    
    # Create figure
    fig = plt.figure(figsize=(18, 8))
    
    # Add super title
    fig.suptitle(f'Classification Visualization: {las_path.split("/")[-1]}', 
                 fontsize=14, fontweight='bold')
    
    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                c=point_colors, s=0.5, alpha=0.6)
    ax1.set_title('3D View', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.view_init(elev=30, azim=45)
    
    # Top-down view
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 1], 
                c=point_colors, s=1.0, alpha=0.8)
    ax2.set_title('Top-Down View', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Classification distribution
    ax3 = fig.add_subplot(133)
    
    # Count each class
    unique_labels = np.unique(labels)
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
        13: "Wire-Guard",
        14: "Wire-Cond",
        15: "Tower",
        17: "Bridge",
        18: "High Noise",
    }
    
    counts = []
    names = []
    colors = []
    
    for label in sorted(unique_labels):
        count = np.sum(labels == label)
        if count > 0:  # Only show classes that exist
            counts.append(count)
            names.append(f"{int(label)}: {class_names.get(int(label), 'Unknown')}")
            colors.append(color_map.get(int(label), '#808080'))
    
    # Horizontal bar chart
    y_pos = np.arange(len(names))
    ax3.barh(y_pos, counts, color=colors, alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=9)
    ax3.set_xlabel('Point Count')
    ax3.set_title('Classification Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (count, name) in enumerate(zip(counts, names)):
        pct = count / len(labels) * 100
        ax3.text(count, i, f' {count:,} ({pct:.1f}%)', 
                va='center', fontsize=8)
    
    # Highlight building classification
    if building_pct < 1.0:
        status = "🔴 CRITICAL"
    elif building_pct < 5.0:
        status = "⚠️ WARNING"
    else:
        status = "✅ OK"
    
    fig.text(0.5, 0.02, 
             f'Building Classification: {building_count:,} points ({building_pct:.2f}%) - {status}',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization to: {output_path}")
    else:
        print("✅ Displaying visualization...")
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_classification.py <las_file> [output.png]")
        print("\nExamples:")
        print("  python scripts/visualize_classification.py output/tile_enriched.laz")
        print("  python scripts/visualize_classification.py output/tile_enriched.laz results.png")
        sys.exit(1)
    
    las_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualize_classification(las_file, output_file)
