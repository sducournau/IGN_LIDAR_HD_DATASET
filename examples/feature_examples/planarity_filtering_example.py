"""
Example: Using Planarity Filtering to Reduce Artifacts

This example demonstrates how to use the planarity filtering functionality
to reduce line/dash artifacts in planarity features.

Author: Simon Ducournau
Date: October 30, 2025
Version: 3.0.6
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import planarity computation and filtering
from ign_lidar.features.compute import (
    compute_planarity,
    smooth_planarity_spatial,
    validate_planarity,
)
from ign_lidar.features.compute.eigenvalues import compute_eigenvalue_features


def example_basic_filtering():
    """Example 1: Basic planarity filtering."""
    print("=" * 60)
    print("Example 1: Basic Planarity Filtering")
    print("=" * 60)

    # Simulate a simple point cloud (100x100 grid)
    np.random.seed(42)
    x, y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(10000)])

    # Add some noise
    points[:, 2] += np.random.normal(0, 0.1, 10000)

    # Simulate planarity with artifacts (NaN at edges)
    planarity = np.random.uniform(0.85, 0.95, 10000)
    # Add some NaN artifacts
    artifact_indices = np.random.choice(10000, 100, replace=False)
    planarity[artifact_indices] = np.nan

    print(f"\nInput:")
    print(f"  Points: {len(points):,}")
    print(f"  NaN values: {np.sum(np.isnan(planarity))}")
    print(
        f"  Planarity range: [{np.nanmin(planarity):.3f}, "
        f"{np.nanmax(planarity):.3f}]"
    )

    # Apply filtering
    smoothed, stats = smooth_planarity_spatial(
        planarity, points, k_neighbors=15, std_threshold=0.3
    )

    print(f"\nOutput:")
    print(f"  NaN fixed: {stats['n_nan_fixed']}")
    print(f"  Artifacts fixed: {stats['n_artifacts_fixed']}")
    print(f"  Unchanged: {stats['n_unchanged']}")
    print(f"  Planarity range: [{smoothed.min():.3f}, {smoothed.max():.3f}]")

    return points, planarity, smoothed


def example_with_validation():
    """Example 2: Filtering with validation."""
    print("\n" + "=" * 60)
    print("Example 2: Filtering with Validation")
    print("=" * 60)

    # Simulate planarity with various issues
    np.random.seed(42)
    n_points = 1000
    points = np.random.rand(n_points, 3) * 10

    planarity = np.random.uniform(0.5, 1.0, n_points)
    # Add problematic values
    planarity[100:110] = np.nan  # NaN
    planarity[200:210] = np.inf  # Inf
    planarity[300:310] = -0.5  # Out of range
    planarity[400:410] = 1.5  # Out of range

    print(f"\nInput issues:")
    print(f"  NaN: {np.sum(np.isnan(planarity))}")
    print(f"  Inf: {np.sum(np.isinf(planarity))}")
    print(f"  Out of range [0,1]: " f"{np.sum((planarity < 0) | (planarity > 1))}")

    # Step 1: Validate
    validated, val_stats = validate_planarity(planarity)

    print(f"\nAfter validation:")
    print(f"  NaN fixed: {val_stats['n_nan']}")
    print(f"  Inf fixed: {val_stats['n_inf']}")
    print(f"  Out of range fixed: {val_stats['n_out_of_range']}")
    print(f"  Range: {val_stats['valid_range']}")

    # Step 2: Smooth
    smoothed, smooth_stats = smooth_planarity_spatial(validated, points, k_neighbors=20)

    print(f"\nAfter smoothing:")
    print(f"  Additional artifacts fixed: {smooth_stats['n_artifacts_fixed']}")

    return points, planarity, validated, smoothed


def example_parameter_tuning():
    """Example 3: Parameter tuning demonstration."""
    print("\n" + "=" * 60)
    print("Example 3: Parameter Tuning")
    print("=" * 60)

    # Create test data with boundary artifacts
    np.random.seed(42)
    n_points = 500

    # Two regions: high planarity and low planarity
    points_high = np.column_stack(
        [
            np.random.uniform(0, 5, n_points // 2),
            np.random.uniform(0, 5, n_points // 2),
            np.random.uniform(0, 1, n_points // 2),
        ]
    )
    points_low = np.column_stack(
        [
            np.random.uniform(5, 10, n_points // 2),
            np.random.uniform(0, 5, n_points // 2),
            np.random.uniform(0, 1, n_points // 2),
        ]
    )
    points = np.vstack([points_high, points_low])

    planarity = np.concatenate(
        [
            np.random.uniform(0.85, 0.95, n_points // 2),  # High
            np.random.uniform(0.1, 0.3, n_points // 2),  # Low (artifacts)
        ]
    )

    print(f"\nTesting different parameters:")

    # Conservative filtering
    smoothed_conservative, stats_cons = smooth_planarity_spatial(
        planarity, points, k_neighbors=10, std_threshold=0.5
    )
    print(f"\n  Conservative (k=10, threshold=0.5):")
    print(f"    Artifacts fixed: {stats_cons['n_artifacts_fixed']}")

    # Balanced filtering
    smoothed_balanced, stats_bal = smooth_planarity_spatial(
        planarity, points, k_neighbors=15, std_threshold=0.3
    )
    print(f"\n  Balanced (k=15, threshold=0.3):")
    print(f"    Artifacts fixed: {stats_bal['n_artifacts_fixed']}")

    # Aggressive filtering
    smoothed_aggressive, stats_agg = smooth_planarity_spatial(
        planarity, points, k_neighbors=25, std_threshold=0.2
    )
    print(f"\n  Aggressive (k=25, threshold=0.2):")
    print(f"    Artifacts fixed: {stats_agg['n_artifacts_fixed']}")

    return (
        points,
        planarity,
        smoothed_conservative,
        smoothed_balanced,
        smoothed_aggressive,
    )


def example_integration_workflow():
    """Example 4: Integration with feature computation workflow."""
    print("\n" + "=" * 60)
    print("Example 4: Integration Workflow")
    print("=" * 60)

    # Simulate a realistic workflow
    np.random.seed(42)
    n_points = 5000
    points = np.random.rand(n_points, 3) * 50

    print(f"\nStep 1: Compute covariances and eigenvalues")
    # In real code, this would use compute_covariances_from_neighbors
    # Here we simulate eigenvalues
    eigenvalues = np.random.rand(n_points, 3)
    eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]  # Descending
    print(f"  Eigenvalues computed for {n_points:,} points")

    print(f"\nStep 2: Compute planarity")
    planarity = compute_planarity(eigenvalues)
    n_invalid_before = np.sum(~np.isfinite(planarity))
    print(f"  Planarity computed")
    print(f"  Invalid values (NaN/Inf): {n_invalid_before}")

    print(f"\nStep 3: Apply filtering")
    smoothed_planarity, stats = smooth_planarity_spatial(
        planarity, points, k_neighbors=15, std_threshold=0.3
    )
    n_invalid_after = np.sum(~np.isfinite(smoothed_planarity))
    print(f"  Filtering applied")
    print(f"  Invalid values remaining: {n_invalid_after}")
    print(f"  Statistics:")
    print(f"    - NaN/Inf fixed: {stats['n_nan_fixed']}")
    print(f"    - Boundary artifacts fixed: {stats['n_artifacts_fixed']}")
    print(f"    - Unchanged points: {stats['n_unchanged']}")

    print(f"\nStep 4: Use in classification")
    # Simulate classification threshold
    threshold = 0.7
    is_planar = smoothed_planarity > threshold
    print(
        f"  Planar points (planarity > {threshold}): "
        f"{np.sum(is_planar):,} ({np.sum(is_planar)/n_points*100:.1f}%)"
    )

    return points, planarity, smoothed_planarity


def visualize_results(points, planarity_original, planarity_filtered):
    """Visualize before/after comparison."""
    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original planarity
    sc1 = axes[0].scatter(
        points[:, 0],
        points[:, 1],
        c=planarity_original,
        cmap="viridis",
        s=1,
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("Original Planarity (with artifacts)", fontsize=14)
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_aspect("equal")
    plt.colorbar(sc1, ax=axes[0], label="Planarity")

    # Filtered planarity
    sc2 = axes[1].scatter(
        points[:, 0],
        points[:, 1],
        c=planarity_filtered,
        cmap="viridis",
        s=1,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Filtered Planarity (artifacts reduced)", fontsize=14)
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].set_aspect("equal")
    plt.colorbar(sc2, ax=axes[1], label="Planarity")

    plt.tight_layout()

    # Save figure
    output_path = Path("planarity_filtering_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PLANARITY FILTERING EXAMPLES")
    print("=" * 60)

    # Example 1: Basic filtering
    points1, orig1, filt1 = example_basic_filtering()

    # Example 2: With validation
    points2, orig2, val2, filt2 = example_with_validation()

    # Example 3: Parameter tuning
    result3 = example_parameter_tuning()

    # Example 4: Integration workflow
    points4, orig4, filt4 = example_integration_workflow()

    # Visualize Example 1
    try:
        visualize_results(points1, orig1, filt1)
    except Exception as e:
        print(f"\nVisualization skipped (matplotlib not available): {e}")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
