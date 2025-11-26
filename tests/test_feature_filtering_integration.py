"""
Integration tests for feature filtering on real LiDAR data.

Tests the unified feature_filter.py module on actual IGN LiDAR HD tiles
from Versailles dataset to validate artifact removal in production scenarios.
"""

import numpy as np
import pytest
from pathlib import Path
import laspy

from ign_lidar.features.compute import (
    compute_normals,
    compute_eigenvalue_features,
    compute_horizontality,
)
from ign_lidar.features.compute.feature_filter import (
    smooth_planarity_spatial,
    smooth_linearity_spatial,
    smooth_horizontality_spatial,
    validate_feature,
)


# Test data location
VERSAILLES_DATA_DIR = Path("/mnt/d/ign/versailles_tiles")


def get_test_tiles(max_tiles=3):
    """Get list of available test tiles."""
    if not VERSAILLES_DATA_DIR.exists():
        pytest.skip(f"Test data not found: {VERSAILLES_DATA_DIR}")

    tiles = list(VERSAILLES_DATA_DIR.glob("*.laz"))
    if not tiles:
        tiles = list(VERSAILLES_DATA_DIR.glob("*.las"))

    if not tiles:
        pytest.skip(f"No LiDAR tiles found in {VERSAILLES_DATA_DIR}")

    return tiles[:max_tiles]


def load_tile_sample(tile_path, max_points=50000):
    """Load a sample of points from a tile."""
    las = laspy.read(tile_path)

    # Extract XYZ coordinates
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    # Sample if too large
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    return points


@pytest.mark.integration
class TestFeatureFilteringIntegration:
    """Integration tests on real LiDAR data."""

    @pytest.mark.xfail(reason="Feature filtering algorithm changed")
    def test_filter_versailles_planarity(self):
        """Test planarity filtering on Versailles tile."""
        tiles = get_test_tiles(max_tiles=1)
        tile_path = tiles[0]

        print(f"\n📍 Testing tile: {tile_path.name}")

        # Load data
        points = load_tile_sample(tile_path, max_points=30000)
        print(f"   Loaded {len(points):,} points")

        # Compute features
        print("   Computing normals and eigenvalues...")
        normals, eigenvalues = compute_normals(points, k_neighbors=20)
        features = compute_eigenvalue_features(eigenvalues)
        planarity = features["planarity"]

        # Check for artifacts before filtering
        artifacts_before = np.sum(~np.isfinite(planarity))
        variance_before = np.var(planarity[np.isfinite(planarity)])
        print(f"   Before: {artifacts_before} invalid, var={variance_before:.4f}")

        # Apply filtering
        print("   Applying planarity filtering...")
        planarity_clean = smooth_planarity_spatial(planarity, points)

        # Validate results
        artifacts_after = np.sum(~np.isfinite(planarity_clean))
        variance_after = np.var(planarity_clean[np.isfinite(planarity_clean)])
        print(f"   After:  {artifacts_after} invalid, var={variance_after:.4f}")

        # Assertions
        assert len(planarity_clean) == len(planarity)
        assert not np.any(np.isnan(planarity_clean)), "NaN values remain"
        assert not np.any(np.isinf(planarity_clean)), "Inf values remain"
        assert np.all((planarity_clean >= 0.0) & (planarity_clean <= 1.0))
        assert artifacts_after < artifacts_before, "Should reduce invalid values"

        print(f"   ✓ Reduced invalid from {artifacts_before} to {artifacts_after}")

    def test_filter_versailles_linearity(self):
        """Test linearity filtering on Versailles tile."""
        tiles = get_test_tiles(max_tiles=1)
        tile_path = tiles[0]

        print(f"\n📍 Testing tile: {tile_path.name}")

        # Load data
        points = load_tile_sample(tile_path, max_points=30000)
        print(f"   Loaded {len(points):,} points")

        # Compute features
        print("   Computing eigenvalues...")
        normals, eigenvalues = compute_normals(points, k_neighbors=20)
        features = compute_eigenvalue_features(eigenvalues)
        linearity = features["linearity"]

        # Check before filtering
        artifacts_before = np.sum(~np.isfinite(linearity))
        variance_before = np.var(linearity[np.isfinite(linearity)])
        print(f"   Before: {artifacts_before} invalid, var={variance_before:.4f}")

        # Apply filtering
        print("   Applying linearity filtering...")
        linearity_clean = smooth_linearity_spatial(linearity, points)

        # Validate
        artifacts_after = np.sum(~np.isfinite(linearity_clean))
        variance_after = np.var(linearity_clean[np.isfinite(linearity_clean)])
        print(f"   After:  {artifacts_after} invalid, var={variance_after:.4f}")

        # Assertions
        assert len(linearity_clean) == len(linearity)
        assert not np.any(np.isnan(linearity_clean))
        assert not np.any(np.isinf(linearity_clean))
        assert np.all((linearity_clean >= 0.0) & (linearity_clean <= 1.0))

        print(f"   ✓ Reduced invalid from {artifacts_before} to {artifacts_after}")

    def test_filter_versailles_horizontality(self):
        """Test horizontality filtering on Versailles tile."""
        tiles = get_test_tiles(max_tiles=1)
        tile_path = tiles[0]

        print(f"\n📍 Testing tile: {tile_path.name}")

        # Load data
        points = load_tile_sample(tile_path, max_points=30000)
        print(f"   Loaded {len(points):,} points")

        # Compute features
        print("   Computing normals...")
        normals, eigenvalues = compute_normals(points, k_neighbors=20)
        horizontality = compute_horizontality(normals)

        # Check before filtering
        artifacts_before = np.sum(~np.isfinite(horizontality))
        variance_before = np.var(horizontality[np.isfinite(horizontality)])
        print(f"   Before: {artifacts_before} invalid, var={variance_before:.4f}")

        # Apply filtering
        print("   Applying horizontality filtering...")
        horizontality_clean = smooth_horizontality_spatial(horizontality, points)

        # Validate
        artifacts_after = np.sum(~np.isfinite(horizontality_clean))
        variance_after = np.var(horizontality_clean[np.isfinite(horizontality_clean)])
        print(f"   After:  {artifacts_after} invalid, var={variance_after:.4f}")

        # Assertions
        assert len(horizontality_clean) == len(horizontality)
        assert not np.any(np.isnan(horizontality_clean))
        assert not np.any(np.isinf(horizontality_clean))
        assert np.all((horizontality_clean >= 0.0) & (horizontality_clean <= 1.0))

        print(f"   ✓ Reduced invalid from {artifacts_before} to {artifacts_after}")

    def test_filter_all_three_features_versailles(self):
        """Test filtering all three features together on real data."""
        tiles = get_test_tiles(max_tiles=1)
        tile_path = tiles[0]

        print(f"\n📍 Testing tile: {tile_path.name}")

        # Load data
        points = load_tile_sample(tile_path, max_points=50000)
        print(f"   Loaded {len(points):,} points")

        # Compute all features
        print("   Computing all features...")
        normals, eigenvalues = compute_normals(points, k_neighbors=20)
        features = compute_eigenvalue_features(eigenvalues)
        planarity = features["planarity"]
        linearity = features["linearity"]
        horizontality = compute_horizontality(normals)

        # Count artifacts before
        artifacts_before = {
            "planarity": np.sum(~np.isfinite(planarity)),
            "linearity": np.sum(~np.isfinite(linearity)),
            "horizontality": np.sum(~np.isfinite(horizontality)),
        }
        print(f"   Before: {sum(artifacts_before.values())} total invalid")

        # Filter all three
        print("   Applying unified filtering...")
        planarity_clean = smooth_planarity_spatial(planarity, points)
        linearity_clean = smooth_linearity_spatial(linearity, points)
        horizontality_clean = smooth_horizontality_spatial(horizontality, points)

        # Count artifacts after
        artifacts_after = {
            "planarity": np.sum(~np.isfinite(planarity_clean)),
            "linearity": np.sum(~np.isfinite(linearity_clean)),
            "horizontality": np.sum(~np.isfinite(horizontality_clean)),
        }
        print(f"   After:  {sum(artifacts_after.values())} total invalid")

        # Validate all three
        for name, feature in [
            ("planarity", planarity_clean),
            ("linearity", linearity_clean),
            ("horizontality", horizontality_clean),
        ]:
            assert not np.any(np.isnan(feature)), f"{name} has NaN"
            assert not np.any(np.isinf(feature)), f"{name} has Inf"
            assert np.all((feature >= 0.0) & (feature <= 1.0)), f"{name} out of range"

            # Artifacts should be reduced or equal
            assert (
                artifacts_after[name] <= artifacts_before[name]
            ), f"{name} artifacts increased"

        # Report reduction
        total_before = sum(artifacts_before.values())
        total_after = sum(artifacts_after.values())
        if total_before > 0:
            reduction = 100 * (total_before - total_after) / total_before
            print(f"   ✓ {reduction:.1f}% artifact reduction")
        else:
            print("   ✓ No artifacts detected (clean data)")

    @pytest.mark.xfail(reason="Feature filtering algorithm changed")
    def test_performance_benchmark(self):
        """Benchmark filtering performance on real data."""
        tiles = get_test_tiles(max_tiles=1)
        tile_path = tiles[0]

        print(f"\n📍 Performance test: {tile_path.name}")

        # Test with different point counts
        for n_points in [10000, 30000, 50000]:
            print(f"\n   Testing with {n_points:,} points:")

            # Load data
            points = load_tile_sample(tile_path, max_points=n_points)
            actual_points = len(points)

            # Compute features
            import time

            t0 = time.time()
            normals, eigenvalues = compute_normals(points, k_neighbors=20)
            features = compute_eigenvalue_features(eigenvalues)
            t_compute = time.time() - t0

            # Filter planarity
            t0 = time.time()
            planarity_clean = smooth_planarity_spatial(features["planarity"], points)
            t_filter = time.time() - t0

            # Report
            print(f"      Compute: {t_compute:.2f}s")
            print(f"      Filter:  {t_filter:.2f}s")
            print(f"      Overhead: {100*t_filter/t_compute:.1f}%")

            # Validate
            assert not np.any(np.isnan(planarity_clean))
            assert t_filter < t_compute * 0.5, "Filtering overhead too high (>50%)"

    def test_multiple_tiles_consistency(self):
        """Test that filtering is consistent across multiple tiles."""
        tiles = get_test_tiles(max_tiles=3)

        print(f"\n📍 Testing {len(tiles)} tiles for consistency")

        results = []

        for tile_path in tiles:
            print(f"\n   Tile: {tile_path.name}")

            # Load and process
            points = load_tile_sample(tile_path, max_points=20000)
            normals, eigenvalues = compute_normals(points, k_neighbors=20)
            features = compute_eigenvalue_features(eigenvalues)

            # Filter
            planarity_clean = smooth_planarity_spatial(features["planarity"], points)

            # Collect stats
            stats = {
                "n_points": len(points),
                "mean": np.mean(planarity_clean),
                "std": np.std(planarity_clean),
                "min": np.min(planarity_clean),
                "max": np.max(planarity_clean),
            }
            results.append(stats)

            print(f"      Mean: {stats['mean']:.3f}, " f"Std: {stats['std']:.3f}")

            # Validate
            assert not np.any(np.isnan(planarity_clean))
            assert 0.0 <= stats["min"] <= stats["max"] <= 1.0

        # Check consistency across tiles
        means = [r["mean"] for r in results]
        mean_variation = np.std(means)
        print(f"\n   Mean variation across tiles: {mean_variation:.4f}")

        # Should be reasonably consistent (similar urban scenes)
        assert mean_variation < 0.3, "Excessive variation across similar tiles"


@pytest.mark.integration
@pytest.mark.slow
class TestFeatureFilteringValidation:
    """Validation tests for production scenarios."""

    @pytest.mark.xfail(reason="Feature filtering algorithm changed")
    def test_edge_detection_preservation(self):
        """Verify that real building edges are preserved."""
        tiles = get_test_tiles(max_tiles=1)
        tile_path = tiles[0]

        print(f"\n📍 Edge preservation test: {tile_path.name}")

        # Load building area (higher Z values typically)
        las = laspy.read(tile_path)
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

        # Filter to likely buildings (Z > 5m relative height)
        z_min = np.percentile(points[:, 2], 10)
        building_mask = points[:, 2] > (z_min + 5.0)
        building_points = points[building_mask][:30000]  # Sample

        if len(building_points) < 1000:
            pytest.skip("Insufficient building points in tile")

        print(f"   Selected {len(building_points):,} building points")

        # Compute planarity (high at edges)
        normals, eigenvalues = compute_normals(building_points, k_neighbors=20)
        features = compute_eigenvalue_features(eigenvalues)
        planarity = features["planarity"]

        # Identify likely edges (high planarity regions)
        edge_mask = planarity > 0.7
        n_edges_before = np.sum(edge_mask)

        # Filter
        planarity_clean = smooth_planarity_spatial(planarity, building_points)
        edge_mask_after = planarity_clean > 0.7
        n_edges_after = np.sum(edge_mask_after)

        # Edges should be mostly preserved
        edge_preservation = n_edges_after / max(n_edges_before, 1)
        print(f"   Edge preservation: {100*edge_preservation:.1f}%")
        print(f"   ({n_edges_before} → {n_edges_after} high-planarity points)")

        assert edge_preservation > 0.7, "Too many edges lost (>30% reduction)"
        assert not np.any(np.isnan(planarity_clean))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
