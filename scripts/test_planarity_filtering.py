"""
Test planarity filtering with visual comparison.

This script:
1. Loads a sample from Versailles tile
2. Computes planarity WITHOUT filtering
3. Computes planarity WITH filtering
4. Saves both to LAZ for visual inspection in CloudCompare
5. Reports statistics

Author: Serena MCP
Date: 2025-10-30
Version: 3.1.0
"""

import numpy as np
import laspy
from pathlib import Path
import logging
import sys

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ign_lidar.features.feature_computer import FeatureComputer
from ign_lidar.features.compute.feature_filter import smooth_feature_spatial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test planarity filtering on real data."""

    # 1. Load Versailles data
    input_laz = Path(
        "/mnt/c/DATA/IGN_LIDAR_HD/VERSAILLES/Semis_2021_0654_6862_LA93_IGN69_subset_1M.laz"
    )

    if not input_laz.exists():
        logger.error(f"Input file not found: {input_laz}")
        return

    logger.info(f"📂 Loading: {input_laz.name}")
    las = laspy.read(input_laz)

    # Load points
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    # Sample for faster testing (100K points)
    sample_size = min(100_000, len(points))
    indices = np.random.choice(len(points), sample_size, replace=False)
    points_sample = points[indices]

    logger.info(f"📊 Sampled {len(points_sample):,} points from {len(points):,}")

    # 2. Compute planarity WITHOUT filtering
    logger.info("🔬 Computing planarity WITHOUT filtering...")

    # Use FeatureComputer to compute geometric features
    computer = FeatureComputer(use_gpu=False)
    features = computer.compute_all_features(
        points=points_sample,
        k_normals=20,
        k_geometric=20,
        geometric_features=["planarity"],
        mode="cpu",
    )
    planarity_raw = features["planarity"]

    logger.info(
        f"  Raw planarity: mean={np.mean(planarity_raw):.3f}, "
        f"std={np.std(planarity_raw):.3f}, "
        f"min={np.min(planarity_raw):.3f}, "
        f"max={np.max(planarity_raw):.3f}"
    )

    # 3. Apply filtering
    logger.info("🧹 Applying spatial artifact filtering...")

    planarity_filtered = smooth_feature_spatial(
        feature=planarity_raw,
        points=points_sample,
        k_neighbors=15,
        std_threshold=0.3,
        feature_name="planarity",
    )

    logger.info(
        f"  Filtered planarity: mean={np.mean(planarity_filtered):.3f}, "
        f"std={np.std(planarity_filtered):.3f}, "
        f"min={np.min(planarity_filtered):.3f}, "
        f"max={np.max(planarity_filtered):.3f}"
    )

    # 4. Compare
    diff = np.abs(planarity_filtered - planarity_raw)
    modified_points = np.sum(diff > 1e-6)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff[diff > 1e-6]) if modified_points > 0 else 0

    logger.info(f"\n📈 Filtering Impact:")
    logger.info(
        f"  Modified points: {modified_points:,} ({100*modified_points/len(points_sample):.1f}%)"
    )
    logger.info(f"  Max difference: {max_diff:.6f}")
    logger.info(f"  Mean difference (modified): {mean_diff:.6f}")

    # 5. Save to LAZ for visual inspection
    output_dir = Path("test_cache/planarity_filtering")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save RAW
    las_raw = laspy.LasData(las.header)
    las_raw.x = points_sample[:, 0]
    las_raw.y = points_sample[:, 1]
    las_raw.z = points_sample[:, 2]

    # Add planarity as scalar field (scale to 0-65535 for visualization)
    planarity_scaled = (planarity_raw * 65535).astype(np.uint16)
    las_raw.add_extra_dim(
        laspy.ExtraBytesParams(
            name="planarity", type=np.uint16, description="Raw planarity [0-1] * 65535"
        )
    )
    las_raw.planarity = planarity_scaled

    output_raw = output_dir / "planarity_RAW.laz"
    las_raw.write(output_raw)
    logger.info(f"💾 Saved RAW: {output_raw}")

    # Save FILTERED
    las_filtered = laspy.LasData(las.header)
    las_filtered.x = points_sample[:, 0]
    las_filtered.y = points_sample[:, 1]
    las_filtered.z = points_sample[:, 2]

    planarity_filtered_scaled = (planarity_filtered * 65535).astype(np.uint16)
    las_filtered.add_extra_dim(
        laspy.ExtraBytesParams(
            name="planarity",
            type=np.uint16,
            description="Filtered planarity [0-1] * 65535",
        )
    )
    las_filtered.planarity = planarity_filtered_scaled

    output_filtered = output_dir / "planarity_FILTERED.laz"
    las_filtered.write(output_filtered)
    logger.info(f"💾 Saved FILTERED: {output_filtered}")

    # Save DIFF
    las_diff = laspy.LasData(las.header)
    las_diff.x = points_sample[:, 0]
    las_diff.y = points_sample[:, 1]
    las_diff.z = points_sample[:, 2]

    diff_scaled = (diff * 65535).astype(np.uint16)
    las_diff.add_extra_dim(
        laspy.ExtraBytesParams(
            name="diff", type=np.uint16, description="Absolute difference * 65535"
        )
    )
    las_diff.diff = diff_scaled

    output_diff = output_dir / "planarity_DIFF.laz"
    las_diff.write(output_diff)
    logger.info(f"💾 Saved DIFF: {output_diff}")

    logger.info(f"\n✅ Done! Open files in CloudCompare to visualize:")
    logger.info(f"   1. {output_raw.name} - Raw planarity (with artifacts)")
    logger.info(f"   2. {output_filtered.name} - Filtered planarity (cleaned)")
    logger.info(
        f"   3. {output_diff.name} - Absolute difference (shows where filtering applied)"
    )


if __name__ == "__main__":
    main()
