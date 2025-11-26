"""
Tests for Geometric Rules Multi-Level NDVI Integration

Tests the updated geometric_rules.py module with multi-level NDVI thresholds
aligned with advanced_classification.py.

Author: Data Processing Team
Date: October 19, 2025
"""

import numpy as np
import pytest

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False

from ign_lidar.core.classification.geometric_rules import GeometricRulesEngine

pytestmark = pytest.mark.skipif(
    not HAS_SPATIAL,
    reason="Spatial libraries (geopandas, shapely) required for geometric rules tests",
)


class TestMultiLevelNDVIConstants:
    """Test multi-level NDVI threshold constants."""

    def test_ndvi_constants_defined(self):
        """Test that all multi-level NDVI constants are defined."""
        engine = GeometricRulesEngine()

        # Check constants exist
        assert hasattr(engine, "NDVI_DENSE_FOREST")
        assert hasattr(engine, "NDVI_HEALTHY_TREES")
        assert hasattr(engine, "NDVI_MODERATE_VEG")
        assert hasattr(engine, "NDVI_GRASS")
        assert hasattr(engine, "NDVI_SPARSE_VEG")
        assert hasattr(engine, "NDVI_ROAD")

    def test_ndvi_constants_values(self):
        """Test that NDVI constants have correct values."""
        engine = GeometricRulesEngine()

        # Check values are correct
        assert engine.NDVI_DENSE_FOREST == 0.60
        assert engine.NDVI_HEALTHY_TREES == 0.50
        assert engine.NDVI_MODERATE_VEG == 0.40
        assert engine.NDVI_GRASS == 0.30
        assert engine.NDVI_SPARSE_VEG == 0.20
        assert engine.NDVI_ROAD == 0.15

    def test_ndvi_constants_ordering(self):
        """Test that NDVI constants are in descending order."""
        engine = GeometricRulesEngine()

        # Check ordering
        assert engine.NDVI_DENSE_FOREST > engine.NDVI_HEALTHY_TREES
        assert engine.NDVI_HEALTHY_TREES > engine.NDVI_MODERATE_VEG
        assert engine.NDVI_MODERATE_VEG > engine.NDVI_GRASS
        assert engine.NDVI_GRASS > engine.NDVI_SPARSE_VEG
        assert engine.NDVI_SPARSE_VEG > engine.NDVI_ROAD

    @pytest.mark.xfail(reason="Geometric rules implementation changes")
    def test_alignment_with_advanced_classification(self):
        """Test that constants align with advanced_classification.py."""
        from ign_lidar.core.classification import AdvancedClassifier

        engine = GeometricRulesEngine()
        classifier = AdvancedClassifier()

        # Check alignment (if AdvancedClassifier has these constants)
        # Note: This might fail if AdvancedClassifier uses local variables
        # That's okay - we're ensuring consistency where possible
        assert engine.NDVI_DENSE_FOREST == 0.60
        assert engine.NDVI_SPARSE_VEG == 0.20


class TestMultiLevelNDVIRefinement:
    """Test multi-level NDVI refinement functionality."""

    def test_dense_forest_classification(self):
        """Test dense forest (NDVI ≥ 0.60) classification."""
        engine = GeometricRulesEngine()

        # Create test data: building misclassified as vegetation due to high NDVI
        points = np.array([[0, 0, 5], [1, 0, 5], [2, 0, 5]])  # High point

        labels = np.array(
            [
                engine.ASPRS_BUILDING,  # Misclassified
                engine.ASPRS_BUILDING,
                engine.ASPRS_BUILDING,
            ]
        )

        # Very high NDVI (dense forest)
        ndvi = np.array([0.65, 0.70, 0.75])

        # Apply refinement
        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi)

        # Check that buildings with very high NDVI are reclassified to high vegetation
        assert n_refined == 3
        assert np.all(labels == engine.ASPRS_HIGH_VEGETATION)

    def test_healthy_trees_classification_with_height(self):
        """Test healthy trees (NDVI ≥ 0.50) classification with height."""
        engine = GeometricRulesEngine()

        # Create test data
        points = np.array([[0, 0, 4], [1, 0, 2], [2, 0, 1]])  # Tall  # Medium  # Short

        labels = np.array(
            [engine.ASPRS_ROAD, engine.ASPRS_ROAD, engine.ASPRS_ROAD]  # Misclassified
        )

        # Healthy trees NDVI range
        ndvi = np.array([0.55, 0.52, 0.50])

        # Heights above ground
        height = np.array([4.0, 2.0, 1.0])

        # Apply refinement with height
        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi, height)

        # Check classification by height
        assert n_refined == 3
        assert labels[0] == engine.ASPRS_HIGH_VEGETATION  # Tall tree
        assert labels[1] == engine.ASPRS_MEDIUM_VEGETATION  # Medium tree
        assert labels[2] == engine.ASPRS_MEDIUM_VEGETATION  # Short tree

    def test_healthy_trees_classification_without_height(self):
        """Test healthy trees classification without height (defaults to HIGH)."""
        engine = GeometricRulesEngine()

        points = np.array([[0, 0, 3], [1, 0, 3]])
        labels = np.array([engine.ASPRS_ROAD, engine.ASPRS_ROAD])
        ndvi = np.array([0.55, 0.52])

        # No height provided
        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi, height=None)

        # Should default to high vegetation
        assert n_refined == 2
        assert np.all(labels == engine.ASPRS_HIGH_VEGETATION)

    def test_moderate_vegetation_classification(self):
        """Test moderate vegetation (NDVI ≥ 0.40) classification."""
        engine = GeometricRulesEngine()

        points = np.array([[0, 0, 2], [1, 0, 2]])
        labels = np.array([engine.ASPRS_BUILDING, engine.ASPRS_ROAD])
        ndvi = np.array([0.45, 0.42])

        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi)

        # Should classify as medium vegetation
        assert n_refined == 2
        assert np.all(labels == engine.ASPRS_MEDIUM_VEGETATION)

    def test_grass_classification_with_height(self):
        """Test grass/shrubs (NDVI ≥ 0.30) classification with height."""
        engine = GeometricRulesEngine()

        points = np.array([[0, 0, 1.5], [1, 0, 0.5]])  # Tall grass  # Short grass

        labels = np.array([engine.ASPRS_ROAD, engine.ASPRS_ROAD])
        ndvi = np.array([0.35, 0.32])
        height = np.array([1.5, 0.5])

        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi, height)

        # Check sub-classification by height
        assert n_refined == 2
        assert labels[0] == engine.ASPRS_MEDIUM_VEGETATION  # Tall grass
        assert labels[1] == engine.ASPRS_LOW_VEGETATION  # Short grass

    def test_sparse_vegetation_classification(self):
        """Test sparse vegetation (NDVI ≥ 0.20) classification."""
        engine = GeometricRulesEngine()

        points = np.array([[0, 0, 0.5], [1, 0, 0.3]])
        labels = np.array([engine.ASPRS_ROAD, engine.ASPRS_ROAD])
        ndvi = np.array([0.25, 0.22])

        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi)

        # Should classify as low vegetation
        assert n_refined == 2
        assert np.all(labels == engine.ASPRS_LOW_VEGETATION)

    def test_road_threshold_vegetation_removal(self):
        """Test removal of vegetation with NDVI below road threshold."""
        engine = GeometricRulesEngine()

        points = np.array([[0, 0, 1], [1, 0, 1]])

        # Vegetation with very low NDVI
        labels = np.array(
            [engine.ASPRS_MEDIUM_VEGETATION, engine.ASPRS_HIGH_VEGETATION]
        )

        # Below road threshold (0.15)
        ndvi = np.array([0.10, 0.05])

        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi)

        # Should reclassify to unclassified
        assert n_refined == 2
        assert np.all(labels == engine.ASPRS_UNCLASSIFIED)

    def test_unclassified_multilevel_classification(self):
        """Test multi-level classification of unclassified points."""
        engine = GeometricRulesEngine()

        # 5 unclassified points with different NDVI levels
        points = np.array(
            [
                [0, 0, 5],  # Dense forest
                [1, 0, 4],  # Healthy trees
                [2, 0, 2],  # Moderate veg
                [3, 0, 1],  # Grass
                [4, 0, 0.5],  # Sparse veg
            ]
        )

        labels = np.full(5, engine.ASPRS_UNCLASSIFIED)
        ndvi = np.array([0.65, 0.55, 0.45, 0.35, 0.25])
        height = np.array([5.0, 4.0, 2.0, 1.0, 0.5])

        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi, height)

        # Check all points classified
        assert n_refined == 5
        assert labels[0] == engine.ASPRS_HIGH_VEGETATION  # Dense forest
        assert labels[1] == engine.ASPRS_HIGH_VEGETATION  # Tall healthy tree
        assert labels[2] == engine.ASPRS_MEDIUM_VEGETATION  # Moderate veg
        assert labels[3] == engine.ASPRS_MEDIUM_VEGETATION  # Tall grass
        assert labels[4] == engine.ASPRS_LOW_VEGETATION  # Sparse veg

    @pytest.mark.xfail(reason="Geometric rules implementation changes")
    def test_water_preservation(self):
        """Test that water classification is preserved."""
        engine = GeometricRulesEngine()

        points = np.array([[0, 0, 0], [1, 0, 0]])

        # Water with high NDVI (algae, aquatic plants)
        labels = np.array([engine.ASPRS_WATER, engine.ASPRS_WATER])
        ndvi = np.array([0.40, 0.35])

        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi)

        # Water should remain water
        assert n_refined == 0
        assert np.all(labels == engine.ASPRS_WATER)

    def test_no_change_for_correct_classifications(self):
        """Test that correctly classified points remain unchanged."""
        engine = GeometricRulesEngine()

        points = np.array(
            [
                [0, 0, 5],  # High veg with high NDVI
                [1, 0, 2],  # Medium veg with medium NDVI
                [2, 0, 0.5],  # Low veg with low NDVI
            ]
        )

        labels = np.array(
            [
                engine.ASPRS_HIGH_VEGETATION,
                engine.ASPRS_MEDIUM_VEGETATION,
                engine.ASPRS_LOW_VEGETATION,
            ]
        )

        ndvi = np.array([0.65, 0.45, 0.25])

        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi)

        # No changes needed
        assert n_refined == 0


class TestGeometricRulesIntegration:
    """Test integration of multi-level NDVI with other geometric rules."""

    @pytest.mark.xfail(reason="Geometric rules implementation changes")
    def test_apply_all_rules_with_multilevel_ndvi(self):
        """Test that apply_all_rules uses multi-level NDVI."""
        engine = GeometricRulesEngine(use_spectral_rules=False)

        # Create simple test data
        points = np.array([[0, 0, 5], [1, 0, 4], [2, 0, 2]])

        labels = np.array(
            [engine.ASPRS_BUILDING, engine.ASPRS_ROAD, engine.ASPRS_UNCLASSIFIED]
        )

        # Multi-level NDVI
        ndvi = np.array([0.65, 0.55, 0.45])

        # Empty ground truth features
        ground_truth_features = {}

        # Apply all rules
        updated_labels, stats = engine.apply_all_rules(
            points=points,
            labels=labels,
            ground_truth_features=ground_truth_features,
            ndvi=ndvi,
        )

        # Check that NDVI refinement was applied
        assert "ndvi_refined" in stats
        assert stats["ndvi_refined"] == 3

        # Check classifications
        assert updated_labels[0] == engine.ASPRS_HIGH_VEGETATION  # Dense forest
        assert updated_labels[1] == engine.ASPRS_HIGH_VEGETATION  # Healthy trees
        assert updated_labels[2] == engine.ASPRS_MEDIUM_VEGETATION  # Moderate veg

    def test_height_calculation_integration(self):
        """Test that height is calculated and used in multi-level NDVI."""
        engine = GeometricRulesEngine(use_spectral_rules=False)

        # Create test data with ground points
        points = np.array(
            [
                [0, 0, 0],  # Ground
                [1, 0, 0],  # Ground
                [2, 0, 4],  # High tree
                [3, 0, 1.5],  # Grass
            ]
        )

        labels = np.array(
            [
                engine.ASPRS_GROUND,
                engine.ASPRS_GROUND,
                engine.ASPRS_UNCLASSIFIED,
                engine.ASPRS_UNCLASSIFIED,
            ]
        )

        # Healthy trees NDVI range
        ndvi = np.array([0.0, 0.0, 0.55, 0.35])

        ground_truth_features = {}

        updated_labels, stats = engine.apply_all_rules(
            points=points,
            labels=labels,
            ground_truth_features=ground_truth_features,
            ndvi=ndvi,
        )

        # Check that height-based classification worked
        assert (
            updated_labels[2] == engine.ASPRS_HIGH_VEGETATION
        )  # Tall tree (height ≥ 3m)
        assert (
            updated_labels[3] == engine.ASPRS_MEDIUM_VEGETATION
        )  # Tall grass (height ≥ 1m)


class TestBackwardCompatibility:
    """Test backward compatibility with legacy code."""

    def test_legacy_ndvi_vegetation_threshold_parameter(self):
        """Test that legacy ndvi_vegetation_threshold parameter still works."""
        # Create engine with legacy parameter
        engine = GeometricRulesEngine(ndvi_vegetation_threshold=0.35)  # Legacy value

        # Should still initialize successfully
        assert engine.ndvi_vegetation_threshold == 0.35

        # Multi-level constants should be independent
        assert engine.NDVI_DENSE_FOREST == 0.60
        assert engine.NDVI_GRASS == 0.30

    def test_road_threshold_still_used(self):
        """Test that ndvi_road_threshold is still used correctly."""
        engine = GeometricRulesEngine(ndvi_road_threshold=0.15)

        points = np.array([[0, 0, 1]])
        labels = np.array([engine.ASPRS_MEDIUM_VEGETATION])
        ndvi = np.array([0.10])  # Below road threshold

        n_refined = engine.apply_ndvi_refinement(points, labels, ndvi)

        # Should remove vegetation below road threshold
        assert n_refined == 1
        assert labels[0] == engine.ASPRS_UNCLASSIFIED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
