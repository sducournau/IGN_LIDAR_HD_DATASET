"""
Quick test: Planarity artifact filtering effectiveness.

Generates synthetic roof with scan line artifacts, applies filtering, saves LAZ files.

Author: Serena MCP
Date: 2025-10-30
Version: 3.1.0
"""

import numpy as np
import laspy
from pathlib import Path
import logging
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features.feature_computer import FeatureComputer
from ign_lidar.features.compute.feature_filter import smooth_feature_spatial

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_roof_with_artifacts(n=30000):
    """Generate synthetic roof with scan line artifacts."""
    logger.info("🏗️  Generating synthetic roof")

    # Planar roof
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = 20 + 0.1 * x - 0.05 * y + np.random.normal(0, 0.15, n)

    # Add 20 scan line artifacts
    for i in range(20):
        y_line = i * 5
        mask = np.abs(y - y_line) < 0.3
        z[mask] += np.random.uniform(-0.5, 0.5, np.sum(mask))

    return np.column_stack([x, y, z]).astype(np.float32)


def main():
    logger.info("=" * 60)
    logger.info("TESTING PLANARITY ARTIFACT FILTERING")
    logger.info("=" * 60)

    # 1. Generate data
    points = create_roof_with_artifacts(30000)
    logger.info(f"✓ Generated {len(points):,} points")

    # 2. Compute planarity (raw)
    logger.info("\n🔬 Computing RAW planarity...")
    computer = FeatureComputer(use_gpu=False)
    features = computer.compute_all_features(
        points=points,
        k_normals=20,
        k_geometric=20,
        geometric_features=["planarity"],
        mode=None,
    )
    planarity_raw = features["planarity"]

    logger.info(
        f"  Raw: mean={np.mean(planarity_raw):.3f}, " f"std={np.std(planarity_raw):.3f}"
    )

    # 3. Apply filtering
    logger.info("\n🧹 Applying spatial artifact filtering...")
    planarity_filtered = smooth_feature_spatial(
        feature=planarity_raw,
        points=points,
        k_neighbors=15,
        std_threshold=0.3,
        feature_name="planarity",
    )

    logger.info(
        f"  Filtered: mean={np.mean(planarity_filtered):.3f}, "
        f"std={np.std(planarity_filtered):.3f}"
    )

    # 4. Compare
    diff = np.abs(planarity_filtered - planarity_raw)
    modified = np.sum(diff > 1e-6)

    logger.info(f"\n📈 Impact:")
    logger.info(f"  Modified: {modified:,} points ({100*modified/len(points):.1f}%)")
    logger.info(f"  Max diff: {np.max(diff):.6f}")
    logger.info(f"  Mean diff (modified): {np.mean(diff[diff>1e-6]):.6f}")

    # 5. Save LAZ files
    output_dir = Path("test_cache/planarity_filtering")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create LAZ header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])]

    # Save RAW
    las_raw = laspy.LasData(header)
    las_raw.x = points[:, 0]
    las_raw.y = points[:, 1]
    las_raw.z = points[:, 2]
    las_raw.add_extra_dim(
        laspy.ExtraBytesParams(
            name="planarity", type=np.uint16, description="Raw planarity * 65535"
        )
    )
    las_raw.planarity = (planarity_raw * 65535).astype(np.uint16)

    out_raw = output_dir / "synthetic_planarity_RAW.laz"
    las_raw.write(out_raw)
    logger.info(f"\n💾 Saved RAW: {out_raw}")

    # Save FILTERED
    las_filtered = laspy.LasData(header)
    las_filtered.x = points[:, 0]
    las_filtered.y = points[:, 1]
    las_filtered.z = points[:, 2]
    las_filtered.add_extra_dim(
        laspy.ExtraBytesParams(
            name="planarity", type=np.uint16, description="Filtered planarity * 65535"
        )
    )
    las_filtered.planarity = (planarity_filtered * 65535).astype(np.uint16)

    out_filtered = output_dir / "synthetic_planarity_FILTERED.laz"
    las_filtered.write(out_filtered)
    logger.info(f"💾 Saved FILTERED: {out_filtered}")

    # Save DIFF
    las_diff = laspy.LasData(header)
    las_diff.x = points[:, 0]
    las_diff.y = points[:, 1]
    las_diff.z = points[:, 2]
    las_diff.add_extra_dim(
        laspy.ExtraBytesParams(
            name="diff", type=np.uint16, description="Abs difference * 65535"
        )
    )
    las_diff.diff = (diff * 65535).astype(np.uint16)

    out_diff = output_dir / "synthetic_planarity_DIFF.laz"
    las_diff.write(out_diff)
    logger.info(f"💾 Saved DIFF: {out_diff}")

    logger.info("\n✅ DONE! Open in CloudCompare to visualize")
    logger.info("   - RAW: should show parallel line artifacts")
    logger.info("   - FILTERED: artifacts should be smoothed")
    logger.info("   - DIFF: shows where filtering was applied")


if __name__ == "__main__":
    main()
