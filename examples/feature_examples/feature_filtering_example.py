"""
Feature Filtering Example - Remove Line/Dash Artifacts

This example demonstrates how to use the unified feature filtering module
to remove line/dash artifacts from geometric features (planarity, linearity,
horizontality) that occur at object boundaries due to k-NN crossing surfaces.

**Use Cases:**
1. Remove artifacts before ground truth classification
2. Improve feature quality for machine learning
3. Clean features for visualization
4. Post-process features computed with k-NN neighborhoods

**Features Supported:**
- planarity: (λ2 - λ3) / λ1
- linearity: (λ1 - λ2) / λ1
- horizontality: |dot(normal, vertical)|

Version: 3.1.0
Date: 2025-10-30
"""

import numpy as np
import time
from pathlib import Path

# Import filtering functions
from ign_lidar.features.compute.feature_filter import (
    smooth_feature_spatial,
    validate_feature,
    smooth_planarity_spatial,
    smooth_linearity_spatial,
    smooth_horizontality_spatial,
)

# Import feature computation
from ign_lidar.features.compute import (
    compute_normals,
    compute_eigenvalue_features,
    compute_horizontality,
    compute_linearity,
    compute_planarity,
)


def example_1_basic_usage():
    """
    Example 1: Basic usage - filter a single feature
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Feature Filtering")
    print("=" * 70)

    # Generate synthetic building facade with artifacts
    np.random.seed(42)
    n_points = 10000

    # Wall surface (XZ plane)
    x = np.random.uniform(0, 10, n_points)
    y = np.ones(n_points) * 5.0  # Flat wall
    z = np.random.uniform(0, 10, n_points)

    # Add some outliers (simulate scan line crossing to air)
    n_outliers = int(0.05 * n_points)  # 5% artifacts
    outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
    y[outlier_indices] += np.random.normal(0, 2.0, n_outliers)  # Move off wall

    points = np.column_stack([x, y, z]).astype(np.float32)

    # Compute features (will have artifacts at boundaries)
    print("\n📊 Computing features...")
    start = time.time()
    normals, eigenvalues = compute_normals(points, k_neighbors=20)
    features = compute_eigenvalue_features(eigenvalues)
    planarity = features["planarity"]
    elapsed = time.time() - start
    print(f"  ✓ Features computed in {elapsed:.2f}s")

    # Check for artifacts (before filtering)
    n_artifacts_before = np.sum((planarity < 0.1) | (planarity > 0.9))
    print(f"\n  📉 Before filtering: {n_artifacts_before:,} potential artifacts")

    # Apply spatial filtering
    print("\n🔧 Applying spatial filtering...")
    start = time.time()
    planarity_filtered = smooth_planarity_spatial(
        planarity, points, k_neighbors=15, std_threshold=0.3
    )
    elapsed = time.time() - start
    print(f"  ✓ Filtering completed in {elapsed:.2f}s")

    # Check improvement
    n_artifacts_after = np.sum((planarity_filtered < 0.1) | (planarity_filtered > 0.9))
    improvement = 100 * (n_artifacts_before - n_artifacts_after) / n_artifacts_before
    print(f"  📈 After filtering: {n_artifacts_after:,} artifacts")
    print(f"  ✨ Improvement: {improvement:.1f}% reduction")

    print("\n✓ Example 1 complete")


def example_2_multiple_features():
    """
    Example 2: Filter multiple features simultaneously
    """
    print("\n" + "=" * 70)
    print("Example 2: Multi-Feature Filtering")
    print("=" * 70)

    # Load or generate point cloud
    np.random.seed(123)
    n_points = 5000

    # Complex scene: ground + building + tree
    points_ground = np.column_stack(
        [
            np.random.uniform(0, 20, n_points // 3),
            np.random.uniform(0, 20, n_points // 3),
            np.random.normal(0, 0.1, n_points // 3),
        ]
    )

    points_building = np.column_stack(
        [
            np.random.uniform(5, 15, n_points // 3),
            np.random.uniform(5, 15, n_points // 3),
            np.random.uniform(0, 10, n_points // 3),
        ]
    )

    points_tree = np.column_stack(
        [
            np.random.normal(18, 1, n_points // 3),
            np.random.normal(18, 1, n_points // 3),
            np.random.uniform(0, 8, n_points // 3),
        ]
    )

    points = np.vstack([points_ground, points_building, points_tree]).astype(np.float32)

    print(f"\n📊 Processing {len(points):,} points...")

    # Compute all features
    start = time.time()
    normals, eigenvalues = compute_normals(points, k_neighbors=20)

    planarity = compute_planarity(eigenvalues)
    linearity = compute_linearity(eigenvalues)
    horizontality = compute_horizontality(normals)
    elapsed = time.time() - start

    print(f"  ✓ Features computed in {elapsed:.2f}s")

    # Filter each feature
    print("\n🔧 Filtering features...")
    start = time.time()

    planarity_clean = smooth_planarity_spatial(planarity, points)
    linearity_clean = smooth_linearity_spatial(linearity, points)
    horizontality_clean = smooth_horizontality_spatial(horizontality, points)

    elapsed = time.time() - start
    print(f"  ✓ All features filtered in {elapsed:.2f}s")

    # Statistics
    print("\n📈 Filtering Statistics:")
    for name, original, filtered in [
        ("planarity", planarity, planarity_clean),
        ("linearity", linearity, linearity_clean),
        ("horizontality", horizontality, horizontality_clean),
    ]:
        diff = np.abs(filtered - original)
        n_changed = np.sum(diff > 0.01)
        pct_changed = 100 * n_changed / len(original)
        mean_diff = np.mean(diff[diff > 0.01]) if n_changed > 0 else 0

        print(
            f"  • {name:15s}: {n_changed:6,} points changed "
            f"({pct_changed:4.1f}%), avg Δ={mean_diff:.3f}"
        )

    print("\n✓ Example 2 complete")


def example_3_generic_feature():
    """
    Example 3: Filter custom/generic feature with smooth_feature_spatial
    """
    print("\n" + "=" * 70)
    print("Example 3: Generic Feature Filtering")
    print("=" * 70)

    # Generate custom feature (e.g., a derived metric)
    np.random.seed(456)
    n_points = 3000

    # Create surface
    x = np.random.uniform(0, 10, n_points)
    y = np.random.uniform(0, 10, n_points)
    z = 0.5 * np.sin(x) + 0.5 * np.cos(y)  # Wavy surface
    points = np.column_stack([x, y, z]).astype(np.float32)

    # Compute custom feature (e.g., local complexity)
    normals, eigenvalues = compute_normals(points, k_neighbors=15)
    features = compute_eigenvalue_features(eigenvalues)

    # Custom feature: anisotropy + roughness
    custom_feature = 0.5 * features["anisotropy"] + 0.5 * features["sphericity"]

    # Add artificial artifacts
    artifact_indices = np.random.choice(n_points, n_points // 20, replace=False)
    custom_feature[artifact_indices] = np.random.uniform(
        0.8, 1.0, len(artifact_indices)
    )

    print(f"\n📊 Custom feature computed for {n_points:,} points")
    print(f"  Range: [{custom_feature.min():.3f}, {custom_feature.max():.3f}]")
    print(f"  Mean: {custom_feature.mean():.3f}, Std: {custom_feature.std():.3f}")

    # Filter using generic function
    print("\n🔧 Applying generic spatial filtering...")
    start = time.time()

    custom_filtered = smooth_feature_spatial(
        custom_feature,
        points,
        k_neighbors=15,
        std_threshold=0.25,  # More aggressive for custom feature
        feature_name="custom_complexity",
    )

    elapsed = time.time() - start
    print(f"  ✓ Filtering completed in {elapsed:.2f}s")

    # Validate and clip
    custom_validated = validate_feature(
        custom_filtered,
        feature_name="custom_complexity",
        valid_range=(0.0, 1.0),
        clip_sigma=3.0,
    )

    print(f"\n📈 After filtering:")
    print(f"  Range: [{custom_validated.min():.3f}, {custom_validated.max():.3f}]")
    print(f"  Mean: {custom_validated.mean():.3f}, Std: {custom_validated.std():.3f}")

    print("\n✓ Example 3 complete")


def example_4_parameter_tuning():
    """
    Example 4: Parameter tuning for different artifact severities
    """
    print("\n" + "=" * 70)
    print("Example 4: Parameter Tuning Guide")
    print("=" * 70)

    # Generate test data
    np.random.seed(789)
    n_points = 2000
    points = np.random.rand(n_points, 3).astype(np.float32) * 10

    # Compute feature with known artifacts
    normals, eigenvalues = compute_normals(points, k_neighbors=20)
    planarity = compute_planarity(eigenvalues)

    # Test different parameter combinations
    configs = [
        {
            "name": "Conservative (light filtering)",
            "k": 10,
            "threshold": 0.4,
            "use_case": "Preserve fine details, remove only severe artifacts",
        },
        {
            "name": "Standard (recommended)",
            "k": 15,
            "threshold": 0.3,
            "use_case": "Balance between artifact removal and feature preservation",
        },
        {
            "name": "Aggressive (heavy filtering)",
            "k": 20,
            "threshold": 0.2,
            "use_case": "Maximum artifact suppression, may over-smooth",
        },
    ]

    print("\n🔬 Testing parameter configurations:\n")

    for config in configs:
        filtered = smooth_planarity_spatial(
            planarity,
            points,
            k_neighbors=config["k"],
            std_threshold=config["threshold"],
        )

        diff = np.abs(filtered - planarity)
        n_changed = np.sum(diff > 0.01)
        pct_changed = 100 * n_changed / n_points

        print(f"  {config['name']}:")
        print(f"    k_neighbors={config['k']}, std_threshold={config['threshold']}")
        print(f"    → {n_changed:,} points modified ({pct_changed:.1f}%)")
        print(f"    Use case: {config['use_case']}")
        print()

    print("💡 Recommendation:")
    print("  • Start with k=15, threshold=0.3 (standard)")
    print("  • Increase k if artifacts persist (more spatial context)")
    print("  • Decrease threshold for more aggressive filtering")
    print("  • Validate results visually in CloudCompare or similar")

    print("\n✓ Example 4 complete")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FEATURE FILTERING EXAMPLES")
    print("Remove Line/Dash Artifacts from Geometric Features")
    print("=" * 70)

    # Run all examples
    example_1_basic_usage()
    example_2_multiple_features()
    example_3_generic_feature()
    example_4_parameter_tuning()

    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)
    print("\n📚 Next Steps:")
    print("  1. Integrate filtering into your processing pipeline")
    print("  2. Tune parameters for your specific data")
    print("  3. Validate results visually")
    print("  4. Consider multi-scale feature computation for complex scenes")
    print("\n📖 Documentation: docs/features/feature_filtering.md")
    print("=" * 70 + "\n")
