"""
MINIMAL TEST: Does smooth_feature_spatial actually modify planarity values?

Direct test of the filtering function without complex pipeline.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features.compute.feature_filter import smooth_feature_spatial


def main():
    print("=" * 60)
    print("MINIMAL TEST: smooth_feature_spatial")
    print("=" * 60)

    # Create simple test data
    n = 5000
    points = np.random.rand(n, 3).astype(np.float32) * 100

    # Create planarity with artificial artifacts
    # Half the points have normal values, half have corrupted values
    planarity = np.ones(n, dtype=np.float32) * 0.7

    # Introduce artifacts in bands (like scan lines)
    for i in range(0, n, 100):
        end = min(i + 50, n)
        planarity[i:end] = np.random.uniform(0.2, 0.9, end - i)

    print(f"\n📊 Input:")
    print(f"  Points: {n:,}")
    print(f"  Planarity mean: {np.mean(planarity):.4f}")
    print(f"  Planarity std: {np.std(planarity):.4f}")
    print(f"  Planarity range: [{np.min(planarity):.4f}, {np.max(planarity):.4f}]")

    # Apply filtering
    print(f"\n🧹 Applying spatial filtering...")
    print(f"  k_neighbors=15, std_threshold=0.3")

    planarity_filtered = smooth_feature_spatial(
        feature=planarity,
        points=points,
        k_neighbors=15,
        std_threshold=0.3,
        feature_name="planarity",
    )

    print(f"\n📊 Output:")
    print(f"  Planarity mean: {np.mean(planarity_filtered):.4f}")
    print(f"  Planarity std: {np.std(planarity_filtered):.4f}")
    print(
        f"  Planarity range: [{np.min(planarity_filtered):.4f}, {np.max(planarity_filtered):.4f}]"
    )

    # Compare
    diff = np.abs(planarity_filtered - planarity)
    modified = np.sum(diff > 1e-6)

    print(f"\n📈 Changes:")
    print(f"  Modified points: {modified:,} / {n:,} ({100*modified/n:.1f}%)")
    print(f"  Max difference: {np.max(diff):.6f}")

    if modified > 0:
        mean_change = np.mean(diff[diff > 1e-6])
        print(f"  Mean difference (modified): {mean_change:.6f}")
        print(f"\n✅ FILTERING WORKS! {modified} points were modified")
    else:
        print(f"\n❌ FILTERING FAILED! NO POINTS WERE MODIFIED")
        print(f"   This explains why artifacts remain in the user's data")

    # Show example of modified vs unmodified
    if modified > 0:
        modified_idx = np.where(diff > 1e-6)[0][:5]
        print(f"\n🔍 Examples (first 5 modified points):")
        for idx in modified_idx:
            print(
                f"  Point {idx}: {planarity[idx]:.4f} → {planarity_filtered[idx]:.4f}"
            )


if __name__ == "__main__":
    main()
