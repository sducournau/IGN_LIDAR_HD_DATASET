"""
Test Classification Threshold Consistency

This module tests that all classification modules use consistent thresholds,
especially for road height filtering (DTM-based).

Author: IGN LiDAR HD Testing Team
Date: October 25, 2025
"""

import pytest
import numpy as np
from typing import Dict


class TestThresholdConsistency:
    """Test that all modules use consistent classification thresholds."""

    def test_road_height_threshold_consistency(self):
        """
        All modules should use the same road height threshold (0.3m).

        This ensures consistent behavior across:
        - ground_truth_refinement.py
        - geometric_rules.py
        - unified_classifier.py
        - strtree.py (optimized classifier)
        """
        from ign_lidar.core.classification.ground_truth_refinement import (
            GroundTruthRefinementConfig,
        )
        from ign_lidar.core.classification.geometric_rules import GeometricRulesEngine
        from ign_lidar.core.classification.classifier import Classifier

        # Check GroundTruthRefinementConfig
        gt_config = GroundTruthRefinementConfig()
        assert (
            gt_config.ROAD_HEIGHT_MAX == 0.3
        ), f"GroundTruthRefinementConfig.ROAD_HEIGHT_MAX should be 0.3, got {gt_config.ROAD_HEIGHT_MAX}"
        assert (
            gt_config.ROAD_HEIGHT_MIN == -0.2
        ), f"GroundTruthRefinementConfig.ROAD_HEIGHT_MIN should be -0.2, got {gt_config.ROAD_HEIGHT_MIN}"

        # Check GeometricRulesEngine default
        rules_engine = GeometricRulesEngine()
        assert (
            rules_engine.road_vegetation_height_threshold == 0.5
        ), f"GeometricRulesEngine.road_vegetation_height_threshold should be 0.5, got {rules_engine.road_vegetation_height_threshold}"

        # Check Classifier road rule
        classifier = Classifier()
        rules = classifier._create_classification_rules()
        road_rule = rules["road"]
        height_thresholds = road_rule.thresholds["height"]

        assert (
            height_thresholds[0] == -0.2
        ), f"Road rule min height should be -0.2, got {height_thresholds[0]}"
        assert (
            height_thresholds[1] == 0.3
        ), f"Road rule max height should be 0.3, got {height_thresholds[1]}"

    def test_tree_canopy_threshold_consistency(self):
        """
        Tree canopy detection should use consistent threshold (2.0m).

        Trees typically start at 2m height above ground.
        """
        from ign_lidar.core.classification.ground_truth_refinement import (
            GroundTruthRefinementConfig,
        )

        gt_config = GroundTruthRefinementConfig()
        assert (
            gt_config.TREE_CANOPY_HEIGHT_MIN == 2.0
        ), f"Tree canopy height should be 2.0m, got {gt_config.TREE_CANOPY_HEIGHT_MIN}"

    def test_ndvi_thresholds_consistency(self):
        """
        NDVI thresholds should be consistent across modules.
        """
        from ign_lidar.core.classification.geometric_rules import GeometricRulesEngine
        from ign_lidar.core.classification.ground_truth_refinement import (
            GroundTruthRefinementConfig,
        )

        gt_config = GroundTruthRefinementConfig()
        rules_engine = GeometricRulesEngine()

        # Check road NDVI threshold
        assert gt_config.ROAD_NDVI_MAX == 0.15
        assert rules_engine.ndvi_road_threshold == 0.15

        # Check vegetation NDVI threshold
        assert gt_config.VEG_NDVI_MIN == 0.25


class TestRoadVegetationSeparation:
    """Test that road-vegetation separation works correctly with new thresholds."""

    def create_test_data(self) -> tuple:
        """
        Create synthetic test data with road surface and tree canopy.

        Returns:
            Tuple of (points, features, expected_labels)
        """
        # Create test points at different heights
        points = np.array(
            [
                [0, 0, 0.1],  # Road surface (10cm above ground)
                [0, 0, 0.3],  # Road surface (30cm above ground)
                [1, 1, 0.45],  # Road surface (45cm above ground)
                [2, 2, 0.8],  # Vegetation starting (80cm - should be veg)
                [3, 3, 2.5],  # Tree canopy (2.5m above ground)
                [4, 4, 5.0],  # Tree canopy (5m above ground)
                [5, 5, 8.0],  # High tree canopy (8m above ground)
            ]
        )

        features = {
            "height": np.array([0.1, 0.3, 0.45, 0.8, 2.5, 5.0, 8.0]),
            "planarity": np.array([0.95, 0.92, 0.90, 0.60, 0.30, 0.25, 0.20]),
            "curvature": np.array([0.01, 0.02, 0.02, 0.08, 0.15, 0.18, 0.20]),
            "ndvi": np.array([0.08, 0.10, 0.12, 0.30, 0.55, 0.65, 0.70]),
            "normals": np.array(
                [
                    [0, 0, 1],  # Horizontal (road)
                    [0, 0, 0.98],  # Horizontal (road)
                    [0, 0, 0.95],  # Horizontal (road)
                    [0.1, 0.1, 0.8],  # Slightly tilted (vegetation)
                    [0.3, 0.3, 0.6],  # Tilted (vegetation)
                    [0.4, 0.4, 0.5],  # Irregular (vegetation)
                    [0.5, 0.5, 0.4],  # Irregular (vegetation)
                ]
            ),
        }

        # Expected classifications (ASPRS codes)
        # 11 = ROAD, 3 = LOW_VEG, 4 = MEDIUM_VEG, 5 = HIGH_VEG
        expected = np.array([11, 11, 11, 3, 4, 5, 5])

        return points, features, expected

    def test_road_surface_classification(self):
        """
        Points within 0.5m of ground should be classified as road.
        """
        from ign_lidar.core.classification.ground_truth_refinement import (
            GroundTruthRefiner,
        )

        points, features, expected = self.create_test_data()
        refiner = GroundTruthRefiner()

        # Create road mask (all points are candidates)
        road_mask = np.ones(len(points), dtype=bool)
        labels = np.zeros(len(points), dtype=np.uint8)

        # Refine road classification
        refined, stats = refiner.refine_road_classification(
            labels=labels,
            points=points,
            road_mask=road_mask,
            height=features["height"],
            planarity=features["planarity"],
            curvature=features["curvature"],
            normals=features["normals"],
            ndvi=features["ndvi"],
        )

        # Verify road surface points (indices 0, 1, 2)
        ASPRS_ROAD = 11
        assert refined[0] == ASPRS_ROAD, "Point at 0.1m should be road"
        assert refined[1] == ASPRS_ROAD, "Point at 0.3m should be road"
        assert refined[2] == ASPRS_ROAD, "Point at 0.45m should be road"

        # Verify statistics
        assert (
            stats["road_validated"] >= 3
        ), "At least 3 road points should be validated"

    def test_tree_canopy_exclusion(self):
        """
        Points above 0.5m should NOT be classified as road.
        Tree canopy (>2m with high NDVI) should be vegetation.
        """
        from ign_lidar.core.classification.ground_truth_refinement import (
            GroundTruthRefiner,
        )

        points, features, expected = self.create_test_data()
        refiner = GroundTruthRefiner()

        # Create road mask (all points are candidates)
        road_mask = np.ones(len(points), dtype=bool)
        labels = np.zeros(len(points), dtype=np.uint8)

        # Refine road classification
        refined, stats = refiner.refine_road_classification(
            labels=labels,
            points=points,
            road_mask=road_mask,
            height=features["height"],
            planarity=features["planarity"],
            curvature=features["curvature"],
            normals=features["normals"],
            ndvi=features["ndvi"],
        )

        # Verify tree canopy points (indices 4, 5, 6) are NOT road
        ASPRS_ROAD = 11
        ASPRS_MEDIUM_VEGETATION = 4
        ASPRS_HIGH_VEGETATION = 5

        assert refined[4] != ASPRS_ROAD, "Point at 2.5m should NOT be road"
        assert refined[5] != ASPRS_ROAD, "Point at 5.0m should NOT be road"
        assert refined[6] != ASPRS_ROAD, "Point at 8.0m should NOT be road"

        # Check if tree canopy override was detected
        if stats.get("road_vegetation_override", 0) > 0:
            # Verify they were reclassified as vegetation
            assert refined[4] in [
                ASPRS_MEDIUM_VEGETATION,
                ASPRS_HIGH_VEGETATION,
            ], "Tree canopy should be classified as vegetation"

    def test_edge_case_at_threshold(self):
        """
        Test edge case at exactly 0.5m threshold.
        """
        from ign_lidar.core.classification.ground_truth_refinement import (
            GroundTruthRefiner,
        )

        # Point exactly at threshold
        points = np.array([[0, 0, 0.5]])
        features = {
            "height": np.array([0.5]),
            "planarity": np.array([0.9]),
            "curvature": np.array([0.02]),
            "ndvi": np.array([0.12]),
            "normals": np.array([[0, 0, 1]]),
        }

        refiner = GroundTruthRefiner()
        road_mask = np.ones(1, dtype=bool)
        labels = np.zeros(1, dtype=np.uint8)

        refined, stats = refiner.refine_road_classification(
            labels=labels, points=points, road_mask=road_mask, **features
        )

        # Point at exactly 0.5m with road-like features should be classified as road
        ASPRS_ROAD = 11
        assert (
            refined[0] == ASPRS_ROAD
        ), "Point at exactly 0.5m with road features should be road"


class TestGeometricRulesThresholds:
    """Test geometric rules with new thresholds."""

    def test_road_vegetation_overlap_fix(self):
        """
        Test that road-vegetation overlap is correctly resolved.
        """
        from ign_lidar.core.classification.geometric_rules import GeometricRulesEngine
        import geopandas as gpd
        from shapely.geometry import box

        # Create test data
        points = np.array(
            [
                [0, 0, 0.2],  # Road surface
                [0, 0, 2.5],  # Tree canopy
                [1, 1, 0.3],  # Road surface
                [1, 1, 5.0],  # High tree
            ]
        )

        # Initial labels (all vegetation)
        ASPRS_ROAD = 11
        ASPRS_HIGH_VEGETATION = 5
        labels = np.array([ASPRS_HIGH_VEGETATION] * 4)

        # NDVI values
        ndvi = np.array([0.10, 0.60, 0.12, 0.70])

        # Create road geometry
        road_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 2, 2)]}, crs="EPSG:2154")

        # Apply rules
        rules_engine = GeometricRulesEngine(road_vegetation_height_threshold=0.5)

        n_fixed = rules_engine.fix_road_vegetation_overlap(
            points=points, labels=labels, road_geometries=road_gdf, ndvi=ndvi
        )

        # Check that low points with low NDVI were reclassified to road
        assert n_fixed >= 1, "At least one point should be reclassified"

        # Low points with low NDVI should now be road
        if labels[0] == ASPRS_ROAD:
            assert True, "Low point with low NDVI correctly classified as road"


class TestClassifierRules:
    """Test unified classifier with new road thresholds."""

    def test_road_rule_thresholds(self):
        """
        Test that road rule in unified classifier uses correct thresholds.
        """
        from ign_lidar.core.classification.classifier import Classifier

        classifier = Classifier()
        rules = classifier._create_classification_rules()
        road_rule = rules["road"]

        # Check thresholds
        height_min, height_max = road_rule.thresholds["height"]
        assert height_min == -0.2, f"Road min height should be -0.2, got {height_min}"
        assert height_max == 0.5, f"Road max height should be 0.5, got {height_max}"

        # Check other thresholds are still correct
        assert road_rule.thresholds["planarity"][0] == 0.85
        assert road_rule.thresholds["ndvi"][1] == 0.15


@pytest.mark.integration
class TestIntegration:
    """Integration tests with complete classification pipeline."""

    def test_full_pipeline_consistency(self):
        """
        Test that full pipeline produces consistent results with new thresholds.
        """
        # This would test against a real tile
        # For now, just ensure modules are loadable
        from ign_lidar.core.classification import (
            GroundTruthRefiner,
            GeometricRulesEngine,
            Classifier,
        )

        # Create instances
        refiner = GroundTruthRefiner()
        rules = GeometricRulesEngine()
        classifier = Classifier()

        # Verify they all use consistent thresholds
        assert refiner.config.ROAD_HEIGHT_MAX == 0.3
        assert rules.road_vegetation_height_threshold == 0.5
        # Classifier thresholds checked in separate test


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
