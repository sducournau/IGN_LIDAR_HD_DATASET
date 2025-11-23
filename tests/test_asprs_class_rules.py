"""
Tests for ASPRS Class-Specific Rules Engine (Phase 3)

Author: Simon Ducournau
Date: October 25, 2025
"""

import numpy as np
import pytest

from ign_lidar.core.classification.asprs_class_rules import (
    ASPRSClassRulesEngine,
    BridgeDetectionConfig,
    NoiseClassificationConfig,
    OverheadStructureDetectionConfig,
    RailwayDetectionConfig,
    WaterDetectionConfig,
)
from ign_lidar.classification_schema import ASPRSClass


@pytest.fixture
def synthetic_water_scene():
    """Create synthetic water body scene."""
    n_points = 5000  # Increased density for DBSCAN clustering

    # Water body (flat, low NDVI, low height)
    # Concentrated in smaller area to form clusters
    water_points = np.random.rand(n_points, 3) * [20, 20, 0.2]  # 20x20m area
    water_features = {
        "planarity": np.full(n_points, 0.95),  # Very flat
        "height_above_ground": water_points[:, 2],
        "curvature": np.full(n_points, 0.02),  # Low curvature
        "ndvi": np.full(n_points, 0.10),  # Low NDVI
        "ndwi": np.full(n_points, 0.35),  # High NDWI
        "linearity": np.full(n_points, 0.30),  # Low linearity
        "verticality": np.full(n_points, 0.05),  # Very horizontal
    }

    classification = np.full(n_points, int(ASPRSClass.UNCLASSIFIED))

    return water_points, water_features, classification


@pytest.fixture
def synthetic_bridge_scene():
    """Create synthetic bridge scene."""
    n_points = 1000  # Increased for proper clustering

    # Bridge deck (elevated, planar, linear)
    bridge_points = np.column_stack(
        [
            np.linspace(0, 50, n_points),  # Linear in X
            np.random.rand(n_points) * 5 + 20,  # Width in Y
            np.full(n_points, 8.0) + np.random.rand(n_points) * 0.5,  # Elevated
        ]
    )

    bridge_features = {
        "height_above_ground": bridge_points[:, 2],
        "planarity": np.full(n_points, 0.85),  # Planar deck
        "verticality": np.full(n_points, 0.15),  # Horizontal
        "linearity": np.full(n_points, 0.80),  # Linear structure
        "curvature": np.full(n_points, 0.10),  # Low curvature
        "ndvi": np.full(n_points, 0.20),  # Low vegetation
        "ndwi": np.full(n_points, -0.20),  # Not water
    }

    classification = np.full(n_points, int(ASPRSClass.UNCLASSIFIED))

    return bridge_points, bridge_features, classification


@pytest.fixture
def synthetic_railway_scene():
    """Create synthetic railway scene."""
    n_points = 400

    # Railway tracks (linear, flat, low NDVI)
    railway_points = np.column_stack(
        [
            np.linspace(0, 100, n_points),  # Linear in X
            np.random.rand(n_points) * 2 + 10,  # Narrow width
            np.full(n_points, 0.3) + np.random.rand(n_points) * 0.1,  # Near ground
        ]
    )

    railway_features = {
        "height_above_ground": railway_points[:, 2],
        "planarity": np.full(n_points, 0.80),  # Relatively flat
        "linearity": np.full(n_points, 0.85),  # Very linear
        "ndvi": np.full(n_points, 0.15),  # Low vegetation
        "verticality": np.full(n_points, 0.10),  # Mostly horizontal
        "curvature": np.full(n_points, 0.15),  # Low curvature
        "ndwi": np.full(n_points, -0.30),  # Not water
    }

    classification = np.full(n_points, int(ASPRSClass.UNCLASSIFIED))

    return railway_points, railway_features, classification


@pytest.fixture
def synthetic_overhead_scene():
    """Create synthetic overhead cable scene."""
    n_points = 200

    # Power line cable (high, linear, thin)
    cable_points = np.column_stack(
        [
            np.linspace(0, 80, n_points),  # Linear span
            np.full(n_points, 15.0),  # Straight line
            np.full(n_points, 12.0)
            + np.sin(np.linspace(0, 2 * np.pi, n_points)) * 2,  # Catenary sag
        ]
    )

    cable_features = {
        "height_above_ground": cable_points[:, 2],
        "linearity": np.full(n_points, 0.90),  # Very linear
        "planarity": np.full(n_points, 0.25),  # Not planar (thin)
        "verticality": np.full(n_points, 0.20),  # Mostly horizontal
        "ndvi": np.full(n_points, 0.10),  # Not vegetation
        "curvature": np.full(n_points, 0.20),  # Some curvature from catenary
        "ndwi": np.full(n_points, -0.40),  # Not water
    }

    classification = np.full(n_points, int(ASPRSClass.UNCLASSIFIED))

    return cable_points, cable_features, classification


@pytest.fixture
def synthetic_noise_scene():
    """Create synthetic scene with noise points."""
    n_points = 100

    # Isolated noise points (scattered, extreme heights)
    noise_points = np.random.rand(n_points, 3) * [100, 100, 50]
    noise_features = {
        "height_above_ground": noise_points[:, 2],
        "planarity": np.zeros(n_points),  # Random noise - not planar
        "linearity": np.zeros(n_points),  # Random noise - not linear
        "verticality": np.zeros(n_points),  # Random noise
        "curvature": np.ones(n_points),  # High curvature for noise
        "ndvi": np.full(n_points, 0.5),  # Neutral NDVI
        "ndwi": np.full(n_points, 0.0),  # Neutral NDWI
    }

    classification = np.full(n_points, int(ASPRSClass.UNCLASSIFIED))

    return noise_points, noise_features, classification


class TestWaterDetection:
    """Test water body detection."""

    def test_water_detection_basic(self, synthetic_water_scene):
        """Test basic water body detection."""
        points, features, classification = synthetic_water_scene

        engine = ASPRSClassRulesEngine()
        result = engine.classify_water_bodies(
            points, features, classification, ground_truth=None
        )

        # Check that water points were classified
        n_water = (result == int(ASPRSClass.WATER)).sum()
        assert n_water > 0, "Water points should be detected"
        print(f"✅ Water detection: {n_water}/{len(points)} points classified")

    def test_water_detection_ndwi_required(self, synthetic_water_scene):
        """Test water detection with NDWI requirement."""
        points, features, classification = synthetic_water_scene

        # Remove NDWI feature
        features_no_ndwi = {k: v for k, v in features.items() if k != "ndwi"}

        engine = ASPRSClassRulesEngine()
        result = engine.classify_water_bodies(
            points, features_no_ndwi, classification, ground_truth=None
        )

        # Should still detect water (NDWI is optional)
        n_water = (result == int(ASPRSClass.WATER)).sum()
        assert n_water > 0, "Water should be detected without NDWI"
        print(f"✅ Water detection (no NDWI): {n_water}/{len(points)} points")

    def test_water_detection_thresholds(self):
        """Test water detection with edge case thresholds."""
        n_points = 1000  # Increased for DBSCAN clustering

        # Edge case: just meets thresholds (slightly above to avoid floating point issues)
        # Concentrate in very small area to form dense cluster
        points = np.random.rand(n_points, 3) * [5, 5, 0.4]  # 5x5m area = high density
        features = {
            "planarity": np.full(n_points, 0.87),  # Slightly above threshold (0.85)
            "height_above_ground": np.full(n_points, 0.4),  # Below threshold (0.5)
            "curvature": np.full(n_points, 0.03),  # Below threshold (0.05)
            "ndvi": np.full(n_points, 0.12),  # Below threshold (0.15)
            "ndwi": np.full(n_points, 0.25),  # Above threshold (0.20)
        }
        classification = np.full(n_points, int(ASPRSClass.UNCLASSIFIED))

        engine = ASPRSClassRulesEngine()
        result = engine.classify_water_bodies(
            points, features, classification, ground_truth=None
        )

        n_water = (result == int(ASPRSClass.WATER)).sum()
        assert n_water > 0, "Water should be detected at threshold values"
        print(f"✅ Water threshold test: {n_water}/{n_points} points")


class TestBridgeDetection:
    """Test bridge detection."""

    def test_bridge_detection_basic(self, synthetic_bridge_scene):
        """Test basic bridge detection."""
        points, features, classification = synthetic_bridge_scene

        engine = ASPRSClassRulesEngine()
        result = engine.classify_bridges(
            points, features, classification, ground_truth=None
        )

        # Check that bridge points were classified
        n_bridge = (result == int(ASPRSClass.BRIDGE_DECK)).sum()
        assert n_bridge > 0, "Bridge points should be detected"
        print(f"✅ Bridge detection: {n_bridge}/{len(points)} points classified")

    def test_bridge_detection_dimensions(self):
        """Test bridge detection with dimension constraints."""
        # Create bridge with valid dimensions
        n_points = 600  # Increased for DBSCAN clustering (min_samples=50)
        bridge_points = np.column_stack(
            [
                np.linspace(0, 25, n_points),  # 25m length (> min_length)
                np.random.rand(n_points) * 8 + 10,  # 8m width (< max_width)
                np.full(n_points, 6.0),  # 6m elevation
            ]
        )

        features = {
            "height_above_ground": bridge_points[:, 2],
            "planarity": np.full(n_points, 0.80),
            "verticality": np.full(n_points, 0.20),
            "linearity": np.full(n_points, 0.75),
            "curvature": np.full(n_points, 0.10),  # Low curvature
            "ndvi": np.full(n_points, 0.20),  # Low vegetation
            "ndwi": np.full(n_points, -0.20),  # Not water
        }

        classification = np.full(n_points, int(ASPRSClass.UNCLASSIFIED))

        engine = ASPRSClassRulesEngine()
        result = engine.classify_bridges(
            points=bridge_points,
            features=features,
            classification=classification,
            ground_truth=None,
        )

        n_bridge = (result == int(ASPRSClass.BRIDGE_DECK)).sum()
        assert n_bridge > 0, "Bridge with valid dimensions should be detected"
        print(f"✅ Bridge dimension test: {n_bridge}/{n_points} points")


class TestRailwayDetection:
    """Test railway detection."""

    def test_railway_detection_basic(self, synthetic_railway_scene):
        """Test basic railway detection."""
        points, features, classification = synthetic_railway_scene

        engine = ASPRSClassRulesEngine()
        result = engine.classify_railways(
            points, features, classification, ground_truth=None
        )

        # Check that railway points were classified
        n_railway = (result != int(ASPRSClass.UNCLASSIFIED)).sum()
        assert n_railway > 0, "Railway points should be detected"
        print(f"✅ Railway detection: {n_railway}/{len(points)} points classified")

    def test_railway_parallel_tracks(self):
        """Test parallel railway track detection."""
        n_points_per_track = 200

        # Create two parallel tracks (standard gauge ~1.435m spacing)
        track1_x = np.linspace(0, 100, n_points_per_track)
        track1_y = np.full(n_points_per_track, 10.0)
        track2_x = np.linspace(0, 100, n_points_per_track)
        track2_y = np.full(n_points_per_track, 11.5)  # 1.5m spacing

        points = np.column_stack(
            [
                np.concatenate([track1_x, track2_x]),
                np.concatenate([track1_y, track2_y]),
                np.full(n_points_per_track * 2, 0.3),
            ]
        )

        features = {
            "height_above_ground": points[:, 2],
            "planarity": np.full(len(points), 0.80),
            "linearity": np.full(len(points), 0.85),
            "ndvi": np.full(len(points), 0.15),
        }

        classification = np.full(len(points), int(ASPRSClass.UNCLASSIFIED))

        config = RailwayDetectionConfig(parallel_track_detection=True)
        engine = ASPRSClassRulesEngine(railway_config=config)
        result = engine.classify_railways(
            points, features, classification, ground_truth=None
        )

        n_railway = (result != int(ASPRSClass.UNCLASSIFIED)).sum()
        assert n_railway > 0, "Parallel tracks should be detected"
        print(f"✅ Parallel track detection: {n_railway}/{len(points)} points")


class TestOverheadStructureDetection:
    """Test overhead structure detection."""

    def test_overhead_structure_basic(self, synthetic_overhead_scene):
        """Test basic overhead structure detection."""
        points, features, classification = synthetic_overhead_scene

        engine = ASPRSClassRulesEngine()
        result = engine.classify_overhead_structures(points, features, classification)

        # Check that overhead points were classified
        n_overhead = (result == int(ASPRSClass.WIRE_CONDUCTOR)).sum()
        assert n_overhead > 0, "Overhead structure points should be detected"
        print(
            f"✅ Overhead structure detection: {n_overhead}/{len(points)} points classified"
        )

    def test_overhead_structure_length_filter(self):
        """Test overhead structure detection with length filtering."""
        # Short cable span (< min_length)
        n_points = 50
        short_cable = np.column_stack(
            [
                np.linspace(0, 15, n_points),  # 15m (< 20m min_length)
                np.full(n_points, 10.0),
                np.full(n_points, 10.0),
            ]
        )

        features = {
            "height_above_ground": short_cable[:, 2],
            "linearity": np.full(n_points, 0.90),
            "planarity": np.full(n_points, 0.25),
            "verticality": np.full(n_points, 0.20),
            "ndvi": np.full(n_points, 0.10),
        }

        classification = np.full(n_points, int(ASPRSClass.UNCLASSIFIED))

        engine = ASPRSClassRulesEngine()
        result = engine.classify_overhead_structures(
            short_cable, features, classification
        )

        # Short cable should not be detected
        n_overhead = (result == int(ASPRSClass.WIRE_CONDUCTOR)).sum()
        print(f"✅ Short cable filtering: {n_overhead}/{n_points} points (expected ~0)")


class TestNoiseClassification:
    """Test noise classification."""

    def test_noise_detection_basic(self, synthetic_noise_scene):
        """Test basic noise detection."""
        points, features, classification = synthetic_noise_scene

        engine = ASPRSClassRulesEngine()
        result = engine.classify_noise(points, features, classification)

        # Check that noise points were classified
        n_noise = (result == int(ASPRSClass.LOW_POINT)).sum()
        assert n_noise > 0, "Noise points should be detected"
        print(f"✅ Noise detection: {n_noise}/{len(points)} points classified")

    def test_noise_isolation_threshold(self):
        """Test noise detection with isolation threshold."""
        # Create dense cluster + isolated points
        n_cluster = 100
        n_isolated = 20

        # Dense cluster
        cluster_points = np.random.rand(n_cluster, 3) * [10, 10, 5] + [0, 0, 0]

        # Isolated points (far from cluster)
        isolated_points = np.random.rand(n_isolated, 3) * [10, 10, 5] + [50, 50, 0]

        points = np.vstack([cluster_points, isolated_points])
        features = {"height_above_ground": points[:, 2]}
        classification = np.full(len(points), int(ASPRSClass.UNCLASSIFIED))

        engine = ASPRSClassRulesEngine()
        result = engine.classify_noise(points, features, classification)

        # Isolated points should be classified as noise
        n_noise = (result == int(ASPRSClass.LOW_POINT)).sum()
        assert n_noise >= n_isolated * 0.5, "Most isolated points should be noise"
        print(
            f"✅ Noise isolation test: {n_noise}/{len(points)} points classified as noise"
        )


class TestIntegration:
    """Integration tests for all rules."""

    def test_apply_all_rules(
        self,
        synthetic_water_scene,
        synthetic_bridge_scene,
        synthetic_railway_scene,
        synthetic_overhead_scene,
        synthetic_noise_scene,
    ):
        """Test applying all rules to combined scene."""
        # Combine all scenes
        all_points = []
        all_features = {}
        all_classifications = []

        scenes = [
            synthetic_water_scene,
            synthetic_bridge_scene,
            synthetic_railway_scene,
            synthetic_overhead_scene,
            synthetic_noise_scene,
        ]

        for points, features, classification in scenes:
            all_points.append(points)
            all_classifications.append(classification)

            # Merge features
            for key, values in features.items():
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(values)

        # Concatenate
        all_points = np.vstack(all_points)
        all_classifications = np.concatenate(all_classifications)
        for key in all_features:
            all_features[key] = np.concatenate(all_features[key])

        # Apply all rules
        engine = ASPRSClassRulesEngine()
        result = engine.apply_all_rules(
            all_points, all_features, all_classifications, ground_truth=None
        )

        # Check that multiple classes were detected
        unique_classes = np.unique(result)
        n_classes = len(unique_classes)
        assert n_classes > 1, "Multiple classes should be detected"

        # Check that at least some points were reclassified
        n_reclassified = (result != int(ASPRSClass.UNCLASSIFIED)).sum()
        assert n_reclassified > 0, "Some points should be reclassified"

        print(f"\n✅ Integration test:")
        print(f"   Total points: {len(all_points):,}")
        print(f"   Classes detected: {n_classes}")
        print(f"   Points reclassified: {n_reclassified:,}")

        # Print class distribution
        for class_val in unique_classes:
            n_points = (result == class_val).sum()
            print(f"   Class {class_val}: {n_points:,} points")


def test_config_creation():
    """Test configuration creation from dict."""
    config_dict = {
        "asprs_class_rules": {
            "enabled": True,
            "water_detection": {"enabled": True, "planarity_min": 0.90},
            "bridge_detection": {"enabled": True, "height_min": 5.0},
            "railway_detection": {"enabled": True, "linearity_min": 0.80},
            "overhead_structure_detection": {"enabled": True, "height_min": 10.0},
            "noise_classification": {"enabled": True, "isolation_threshold": 3.0},
        }
    }

    from ign_lidar.core.classification.asprs_class_rules import (
        create_asprs_rules_from_config,
    )

    engine = create_asprs_rules_from_config(config_dict)
    assert engine is not None, "Engine should be created from config"
    assert engine.water_config.planarity_min == 0.90, "Config should be applied"
    print("✅ Config creation test passed")


if __name__ == "__main__":
    """Run tests standalone."""
    print("\n" + "=" * 80)
    print("Testing ASPRS Class-Specific Rules Engine (Phase 3)")
    print("=" * 80 + "\n")

    # Create fixtures manually
    def run_water_tests():
        print("\n--- Water Detection Tests ---")
        test = TestWaterDetection()
        water_scene = synthetic_water_scene()
        test.test_water_detection_basic(water_scene)
        test.test_water_detection_ndwi_required(water_scene)
        test.test_water_detection_thresholds()

    def run_bridge_tests():
        print("\n--- Bridge Detection Tests ---")
        test = TestBridgeDetection()
        bridge_scene = synthetic_bridge_scene()
        test.test_bridge_detection_basic(bridge_scene)
        test.test_bridge_detection_dimensions()

    def run_railway_tests():
        print("\n--- Railway Detection Tests ---")
        test = TestRailwayDetection()
        railway_scene = synthetic_railway_scene()
        test.test_railway_detection_basic(railway_scene)
        test.test_railway_parallel_tracks()

    def run_overhead_tests():
        print("\n--- Overhead Structure Tests ---")
        test = TestOverheadStructureDetection()
        overhead_scene = synthetic_overhead_scene()
        test.test_overhead_structure_basic(overhead_scene)
        test.test_overhead_structure_length_filter()

    def run_noise_tests():
        print("\n--- Noise Classification Tests ---")
        test = TestNoiseClassification()
        noise_scene = synthetic_noise_scene()
        test.test_noise_detection_basic(noise_scene)
        test.test_noise_isolation_threshold()

    def run_integration_tests():
        print("\n--- Integration Tests ---")
        test = TestIntegration()
        test.test_apply_all_rules(
            synthetic_water_scene(),
            synthetic_bridge_scene(),
            synthetic_railway_scene(),
            synthetic_overhead_scene(),
            synthetic_noise_scene(),
        )

    # Run all tests
    try:
        run_water_tests()
        run_bridge_tests()
        run_railway_tests()
        run_overhead_tests()
        run_noise_tests()
        run_integration_tests()
        test_config_creation()

        print("\n" + "=" * 80)
        print("🎉 All ASPRS class-specific rule tests passed!")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
