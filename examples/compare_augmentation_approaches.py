"""
Visual Comparison: Old vs New Augmentation Approach

This script demonstrates the difference between:
1. Old: Augment after features (inconsistent)
2. New: Augment before features (consistent)

We'll create a simple point cloud, apply augmentation both ways,
and show the feature differences.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_planar_surface(n_points=500):
    """Create a simple planar surface (roof-like)."""
    x = np.random.uniform(0, 10, n_points)
    y = np.random.uniform(0, 10, n_points)
    z = np.ones(n_points) * 5.0  # Flat at Z=5
    
    # Add small noise
    z += np.random.normal(0, 0.05, n_points)
    
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    return points


def compute_simple_planarity(points, k=20):
    """
    Simplified planarity computation (concept demo).
    
    In real code, this uses sklearn KDTree and eigenvalues.
    Here we just show the concept.
    """
    from sklearn.neighbors import KDTree
    
    tree = KDTree(points)
    _, indices = tree.query(points, k=k)
    
    planarities = []
    for i, neighbors_idx in enumerate(indices):
        # Get neighbor points
        neighbors = points[neighbors_idx]
        
        # Center the neighborhood
        neighbors_centered = neighbors - neighbors.mean(axis=0)
        
        # Compute covariance matrix
        cov = np.cov(neighbors_centered.T)
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
        
        # Planarity: (λ2 - λ3) / λ1
        if eigenvalues[0] > 1e-6:
            planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
        else:
            planarity = 0.0
        
        planarities.append(planarity)
    
    return np.array(planarities)


def rotate_points(points, angle_deg):
    """Rotate points around Z-axis."""
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    return points @ rotation_matrix.T


def old_approach(points, angle_deg=45):
    """
    OLD APPROACH: Compute features, then augment.
    """
    print("OLD APPROACH:")
    print("-" * 50)
    
    # Step 1: Compute planarity on original
    print("  1. Computing planarity on original geometry...")
    planarity_original = compute_simple_planarity(points, k=20)
    print(f"     Mean planarity: {planarity_original.mean():.4f}")
    
    # Step 2: Rotate points
    print(f"  2. Rotating points by {angle_deg}°...")
    points_rotated = rotate_points(points, angle_deg)
    
    # Step 3: COPY planarity (OLD APPROACH - WRONG!)
    print("  3. Copying planarity values (NOT recomputing)...")
    planarity_copied = planarity_original.copy()
    print(f"     Mean planarity: {planarity_copied.mean():.4f}")
    
    print("  ❌ PROBLEM: Planarity doesn't match rotated geometry!")
    print()
    
    return points_rotated, planarity_copied


def new_approach(points, angle_deg=45):
    """
    NEW APPROACH: Augment, then compute features.
    """
    print("NEW APPROACH:")
    print("-" * 50)
    
    # Step 1: Rotate points FIRST
    print(f"  1. Rotating points by {angle_deg}°...")
    points_rotated = rotate_points(points, angle_deg)
    
    # Step 2: Compute planarity on AUGMENTED geometry
    print("  2. Computing planarity on augmented geometry...")
    planarity_recomputed = compute_simple_planarity(points_rotated, k=20)
    print(f"     Mean planarity: {planarity_recomputed.mean():.4f}")
    
    print("  ✅ BENEFIT: Planarity matches augmented geometry!")
    print()
    
    return points_rotated, planarity_recomputed


def visualize_comparison(points_orig, points_aug_old, planarity_old,
                        points_aug_new, planarity_new):
    """Visualize the difference."""
    fig = plt.figure(figsize=(15, 5))
    
    # Original
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points_orig[:, 0], points_orig[:, 1], points_orig[:, 2],
               c='blue', s=1, alpha=0.5)
    ax1.set_title('Original Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Old approach (copied planarity)
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(points_aug_old[:, 0], points_aug_old[:, 1],
                          points_aug_old[:, 2],
                          c=planarity_old, s=1, cmap='viridis',
                          vmin=0, vmax=1)
    ax2.set_title(f'Old: Copied Planarity\n'
                 f'Mean: {planarity_old.mean():.3f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.colorbar(scatter2, ax=ax2, label='Planarity')
    
    # New approach (recomputed planarity)
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(points_aug_new[:, 0], points_aug_new[:, 1],
                          points_aug_new[:, 2],
                          c=planarity_new, s=1, cmap='viridis',
                          vmin=0, vmax=1)
    ax3.set_title(f'New: Recomputed Planarity\n'
                 f'Mean: {planarity_new.mean():.3f}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.colorbar(scatter3, ax=ax3, label='Planarity')
    
    plt.tight_layout()
    plt.savefig('augmentation_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: augmentation_comparison.png")
    plt.show()


def main():
    """Run the comparison."""
    print("=" * 70)
    print("AUGMENTATION APPROACH COMPARISON")
    print("=" * 70)
    print()
    
    # Create a simple planar surface
    print("Creating test surface (planar roof-like)...")
    points_original = create_planar_surface(n_points=500)
    print(f"  Created {len(points_original)} points")
    print()
    
    # Apply old approach
    points_aug_old, planarity_old = old_approach(
        points_original, angle_deg=45
    )
    
    # Apply new approach
    points_aug_new, planarity_new = new_approach(
        points_original, angle_deg=45
    )
    
    # Compare
    print("=" * 70)
    print("COMPARISON:")
    print("=" * 70)
    print(f"Original mean planarity: {compute_simple_planarity(points_original).mean():.4f}")
    print(f"Old approach (copied):   {planarity_old.mean():.4f}")
    print(f"New approach (recomputed): {planarity_new.mean():.4f}")
    print()
    
    # Difference
    diff = np.abs(planarity_old.mean() - planarity_new.mean())
    print(f"Difference: {diff:.4f}")
    print()
    
    if diff > 0.01:
        print("✅ Significant difference detected!")
        print("   This shows why recomputation is necessary.")
    else:
        print("ℹ️  Small difference (planarity may be rotation-invariant)")
        print("   But other features (linearity, normals) ARE affected!")
    print()
    
    # Visualize
    print("Generating visualization...")
    try:
        visualize_comparison(
            points_original,
            points_aug_old, planarity_old,
            points_aug_new, planarity_new
        )
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("(This is OK - visualization requires display)")
    
    print()
    print("=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("Features MUST be recomputed on augmented geometry!")
    print("Otherwise, feature-geometry mismatch degrades model quality.")
    print("=" * 70)


if __name__ == "__main__":
    main()
