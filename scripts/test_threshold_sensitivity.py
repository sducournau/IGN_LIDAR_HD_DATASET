"""Test artifact filtering with threshold=0.2 (new default)."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features.compute.feature_filter import smooth_feature_spatial


def main():
    print("=" * 70)
    print("TEST: Artifact filtering with threshold=0.2 (v3.1.1 default)")
    print("=" * 70)

    # Create test data with artifacts
    n = 10000
    points = np.random.rand(n, 3).astype(np.float32) * 100

    # Base planarity
    planarity = np.ones(n, dtype=np.float32) * 0.7

    # Add scan line artifacts (every 50 points)
    for i in range(0, n, 50):
        end = min(i + 25, n)
        planarity[i:end] = np.random.uniform(0.2, 0.9, end - i)

    print(f"\n📊 Input: {n:,} points")
    print(f"  Planarity: mean={np.mean(planarity):.3f}, std={np.std(planarity):.3f}")

    # Test with threshold=0.2
    print(f"\n🧹 Applying filtering (threshold=0.2)...")
    filtered = smooth_feature_spatial(
        feature=planarity,
        points=points,
        k_neighbors=15,
        std_threshold=0.2,
        feature_name="planarity",
    )

    print(f"  Filtered: mean={np.mean(filtered):.3f}, std={np.std(filtered):.3f}")

    # Compare
    diff = np.abs(filtered - planarity)
    modified = np.sum(diff > 1e-6)

    print(f"\n📈 Results:")
    print(f"  Modified: {modified:,} points ({100*modified/n:.1f}%)")
    print(f"  Max diff: {np.max(diff):.6f}")

    if modified > 0:
        mean_change = np.mean(diff[diff > 1e-6])
        print(f"  Mean diff: {mean_change:.6f}")
        print(f"\n✅ SUCCESS! Filtering works with new threshold")
    else:
        print(f"\n❌ FAILED! Still not filtering")

    # Test different thresholds
    print(f"\n🔬 Threshold sensitivity test:")
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3]:
        filtered_test = smooth_feature_spatial(
            feature=planarity,
            points=points,
            k_neighbors=15,
            std_threshold=thresh,
            feature_name="test",
        )
        diff_test = np.abs(filtered_test - planarity)
        mod_test = np.sum(diff_test > 1e-6)
        pct = 100 * mod_test / n
        print(f"  thresh={thresh:.2f} → {mod_test:,} modified ({pct:.1f}%)")


if __name__ == "__main__":
    main()
