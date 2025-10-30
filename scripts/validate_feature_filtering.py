#!/usr/bin/env python3
"""
Quick validation script for feature filtering on Versailles data.

Run with: python scripts/validate_feature_filtering.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import laspy

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features.compute import (
    compute_normals,
    compute_eigenvalue_features,
    compute_horizontality,
)
from ign_lidar.features.compute.feature_filter import (
    smooth_planarity_spatial,
    smooth_linearity_spatial,
    smooth_horizontality_spatial,
)


def load_sample(tile_path, max_points=30000):
    """Load sample points from tile."""
    print(f"📂 Loading: {tile_path.name}")

    # Open file to check size
    with laspy.open(tile_path) as f:
        header = f.header
        total_points = header.point_count
        print(f"   Total points in tile: {total_points:,}")

        # Read only what we need
        if total_points > max_points:
            # Read every Nth point
            step = total_points // max_points
            las = f.read()
            indices = np.arange(0, total_points, step)[:max_points]
            x = las.x[indices]
            y = las.y[indices]
            z = las.z[indices]
        else:
            las = f.read()
            x, y, z = las.x, las.y, las.z

    points = np.vstack([np.array(x), np.array(y), np.array(z)]).T.astype(np.float32)
    print(f"   ✓ Loaded {len(points):,} points")
    return points


def validate_planarity(points):
    """Test planarity filtering."""
    print("\n🔧 Testing PLANARITY filtering:")

    # Compute
    print("   Computing features...")
    t0 = time.time()
    normals, eigenvalues = compute_normals(points, k_neighbors=20)
    features = compute_eigenvalue_features(eigenvalues)
    planarity = features["planarity"]
    t_compute = time.time() - t0

    # Stats before
    invalid_before = np.sum(~np.isfinite(planarity))
    mean_before = np.mean(planarity[np.isfinite(planarity)])
    std_before = np.std(planarity[np.isfinite(planarity)])

    print(
        f"   Before: {invalid_before} invalid, "
        f"mean={mean_before:.3f}, std={std_before:.3f}"
    )

    # Filter
    print("   Filtering...")
    t0 = time.time()
    planarity_clean = smooth_planarity_spatial(planarity, points)
    t_filter = time.time() - t0

    # Stats after
    invalid_after = np.sum(~np.isfinite(planarity_clean))
    mean_after = np.mean(planarity_clean[np.isfinite(planarity_clean)])
    std_after = np.std(planarity_clean[np.isfinite(planarity_clean)])

    print(
        f"   After:  {invalid_after} invalid, "
        f"mean={mean_after:.3f}, std={std_after:.3f}"
    )

    # Validate
    assert not np.any(np.isnan(planarity_clean)), "❌ NaN values remain"
    assert not np.any(np.isinf(planarity_clean)), "❌ Inf values remain"
    assert np.all(
        (planarity_clean >= 0) & (planarity_clean <= 1)
    ), "❌ Values out of range"

    # Report
    if invalid_before > 0:
        reduction = 100 * (invalid_before - invalid_after) / invalid_before
        print(f"   ✅ Reduced invalid by {reduction:.1f}%")
    else:
        print("   ✅ No invalid values (clean data)")

    overhead = 100 * t_filter / t_compute
    print(
        f"   ⏱️  Time: compute={t_compute:.1f}s, "
        f"filter={t_filter:.1f}s (overhead: {overhead:.1f}%)"
    )


def validate_linearity(points):
    """Test linearity filtering."""
    print("\n🔧 Testing LINEARITY filtering:")

    # Compute
    print("   Computing features...")
    normals, eigenvalues = compute_normals(points, k_neighbors=20)
    features = compute_eigenvalue_features(eigenvalues)
    linearity = features["linearity"]

    # Stats before
    invalid_before = np.sum(~np.isfinite(linearity))
    mean_before = np.mean(linearity[np.isfinite(linearity)])
    std_before = np.std(linearity[np.isfinite(linearity)])

    print(
        f"   Before: {invalid_before} invalid, "
        f"mean={mean_before:.3f}, std={std_before:.3f}"
    )

    # Filter
    print("   Filtering...")
    linearity_clean = smooth_linearity_spatial(linearity, points)

    # Stats after
    invalid_after = np.sum(~np.isfinite(linearity_clean))
    mean_after = np.mean(linearity_clean[np.isfinite(linearity_clean)])
    std_after = np.std(linearity_clean[np.isfinite(linearity_clean)])

    print(
        f"   After:  {invalid_after} invalid, "
        f"mean={mean_after:.3f}, std={std_after:.3f}"
    )

    # Validate
    assert not np.any(np.isnan(linearity_clean)), "❌ NaN values remain"
    assert not np.any(np.isinf(linearity_clean)), "❌ Inf values remain"
    assert np.all(
        (linearity_clean >= 0) & (linearity_clean <= 1)
    ), "❌ Values out of range"

    print("   ✅ Linearity filtering validated")


def validate_horizontality(points):
    """Test horizontality filtering."""
    print("\n🔧 Testing HORIZONTALITY filtering:")

    # Compute
    print("   Computing features...")
    normals, eigenvalues = compute_normals(points, k_neighbors=20)
    horizontality = compute_horizontality(normals)

    # Stats before
    invalid_before = np.sum(~np.isfinite(horizontality))
    mean_before = np.mean(horizontality[np.isfinite(horizontality)])
    std_before = np.std(horizontality[np.isfinite(horizontality)])

    print(
        f"   Before: {invalid_before} invalid, "
        f"mean={mean_before:.3f}, std={std_before:.3f}"
    )

    # Filter
    print("   Filtering...")
    horizontality_clean = smooth_horizontality_spatial(horizontality, points)

    # Stats after
    invalid_after = np.sum(~np.isfinite(horizontality_clean))
    mean_after = np.mean(horizontality_clean[np.isfinite(horizontality_clean)])
    std_after = np.std(horizontality_clean[np.isfinite(horizontality_clean)])

    print(
        f"   After:  {invalid_after} invalid, "
        f"mean={mean_after:.3f}, std={std_after:.3f}"
    )

    # Validate
    assert not np.any(np.isnan(horizontality_clean)), "❌ NaN remain"
    assert not np.any(np.isinf(horizontality_clean)), "❌ Inf remain"
    assert np.all(
        (horizontality_clean >= 0) & (horizontality_clean <= 1)
    ), "❌ Values out of range"

    print("   ✅ Horizontality filtering validated")


def main():
    """Main validation."""
    print("=" * 60)
    print("Feature Filtering Validation - Versailles Dataset")
    print("=" * 60)

    # Find tiles
    data_dir = Path("/mnt/d/ign/versailles_tiles")
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 1

    tiles = list(data_dir.glob("*.laz"))
    if not tiles:
        tiles = list(data_dir.glob("*.las"))

    if not tiles:
        print(f"❌ No tiles found in {data_dir}")
        return 1

    print(f"\n📦 Found {len(tiles)} tiles")

    # Test on first tile (use smaller sample for speed)
    tile_path = tiles[0]
    points = load_sample(tile_path, max_points=10000)

    try:
        # Test all three features
        validate_planarity(points)
        validate_linearity(points)
        validate_horizontality(points)

        print("\n" + "=" * 60)
        print("✅ ALL VALIDATIONS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
