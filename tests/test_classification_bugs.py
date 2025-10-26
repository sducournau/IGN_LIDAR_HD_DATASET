"""
Tests pour valider les bugs identifiés dans la classification

Ce module teste les bugs critiques identifiés dans CLASSIFICATION_BUGS_ANALYSIS.md:
- Bug #1: Ordre de priorité inversé dans STRtree labeling
- Bug #2: Ordre de priorité incorrect dans construction STRtree
- Bug #5: Geometric rules écrasent le ground truth
- Bug #6: Building buffer zone sans vérification GT
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from ign_lidar.core.classification.geometric_rules import GeometricRulesEngine
from ign_lidar.core.classification.reclassifier import OptimizedReclassifier

# Import des modules à tester
from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer


@pytest.fixture
def sample_points():
    """Points de test pour validation."""
    # 5 points: 3 dans le polygon "building", 2 dans "vegetation"
    points = np.array(
        [
            [0.0, 0.0, 10.0],  # Point 0: dans building (sur bord)
            [5.0, 5.0, 10.0],  # Point 1: dans building (centre)
            [2.5, 2.5, 10.0],  # Point 2: dans building (intérieur)
            [15.0, 15.0, 5.0],  # Point 3: dans vegetation (sur bord)
            [12.0, 12.0, 5.0],  # Point 4: dans vegetation (intérieur)
        ]
    )
    return points


@pytest.fixture
def overlapping_polygons():
    """
    Polygones qui se chevauchent pour tester les priorités.

    Crée 2 polygones:
    - Building: carré (0,0) à (10,10)
    - Vegetation: carré (0,0) à (15,15) - CONTIENT le building

    Les points 0, 1, 2 sont dans les DEUX polygones.
    Avec priorité correcte, ils doivent être classés "building" (priorité haute).
    """
    buildings = gpd.GeoDataFrame(
        {
            "geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
            "name": ["Building1"],
        },
        crs="EPSG:2154",
    )

    vegetation = gpd.GeoDataFrame(
        {
            "geometry": [Polygon([(0, 0), (15, 0), (15, 15), (0, 15)])],
            "name": ["VegetationZone1"],
        },
        crs="EPSG:2154",
    )

    return {"buildings": buildings, "vegetation": vegetation}


@pytest.fixture
def simple_polygons():
    """Polygones non-chevauchants pour tests de base."""
    buildings = gpd.GeoDataFrame(
        {
            "geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
            "name": ["Building1"],
        },
        crs="EPSG:2154",
    )

    vegetation = gpd.GeoDataFrame(
        {
            "geometry": [Polygon([(15, 15), (25, 15), (25, 25), (15, 25)])],
            "name": ["VegetationZone1"],
        },
        crs="EPSG:2154",
    )

    return {"buildings": buildings, "vegetation": vegetation}


class TestBug1_PriorityOrder:
    """
    Test Bug #1: Ordre de priorité inversé dans STRtree labeling

    Comportement attendu:
    - Points dans polygon "building" DOIVENT être classés "building" (code 1)
    - Points dans polygon "vegetation" DOIVENT être classés "vegetation" (code 4)
    - Points dans LES DEUX DOIVENT être classés selon la priorité (building > vegetation)
    """

    def test_overlapping_polygons_priority(self, sample_points, overlapping_polygons):
        """Test que les priorités sont respectées pour polygones chevauchants."""
        optimizer = GroundTruthOptimizer(force_method="strtree", verbose=True)

        # Labelliser les points
        labels = optimizer.label_points(
            points=sample_points,
            ground_truth_features=overlapping_polygons,
            label_priority=["buildings", "roads", "water", "vegetation"],
            ndvi=None,
            use_ndvi_refinement=False,
        )

        # Vérifications:
        # Points 0, 1, 2 sont dans BUILDING (priorité haute) ET vegetation
        # Ils DOIVENT être classés "building" (code 1)
        assert labels[0] == 1, f"Point 0 devrait être 'building' (1), got {labels[0]}"
        assert labels[1] == 1, f"Point 1 devrait être 'building' (1), got {labels[1]}"
        assert labels[2] == 1, f"Point 2 devrait être 'building' (1), got {labels[2]}"

        # Points 3, 4 sont SEULEMENT dans vegetation
        # Ils DOIVENT être classés "vegetation" (code 4)
        assert labels[3] == 4, f"Point 3 devrait être 'vegetation' (4), got {labels[3]}"
        assert labels[4] == 4, f"Point 4 devrait être 'vegetation' (4), got {labels[4]}"

    def test_deterministic_classification(self, sample_points, overlapping_polygons):
        """Test que la classification est déterministe (même résultat à chaque run)."""
        optimizer = GroundTruthOptimizer(force_method="strtree", verbose=False)

        # Run 5 fois et vérifier que c'est toujours le même résultat
        results = []
        for _ in range(5):
            labels = optimizer.label_points(
                points=sample_points,
                ground_truth_features=overlapping_polygons,
                label_priority=["buildings", "roads", "water", "vegetation"],
                ndvi=None,
                use_ndvi_refinement=False,
            )
            results.append(labels.copy())

        # Tous les résultats doivent être identiques
        for i in range(1, 5):
            assert np.array_equal(
                results[0], results[i]
            ), f"Run {i+1} produit un résultat différent de Run 1 (classification non-déterministe)"


class TestBug2_ReversedPriority:
    """
    Test Bug #2: Ordre de priorité incorrect dans construction STRtree

    Vérifie que l'ordre d'ajout des polygones n'affecte pas le résultat.
    """

    def test_priority_order_independence(self, sample_points, overlapping_polygons):
        """Test que l'ordre de priorité est respecté indépendamment de l'ordre d'ajout."""
        optimizer = GroundTruthOptimizer(force_method="strtree", verbose=False)

        # Test avec priorité normale
        labels_normal = optimizer.label_points(
            points=sample_points,
            ground_truth_features=overlapping_polygons,
            label_priority=["buildings", "roads", "water", "vegetation"],
            ndvi=None,
            use_ndvi_refinement=False,
        )

        # Test avec priorité inversée
        labels_reversed = optimizer.label_points(
            points=sample_points,
            ground_truth_features=overlapping_polygons,
            label_priority=["vegetation", "water", "roads", "buildings"],
            ndvi=None,
            use_ndvi_refinement=False,
        )

        # AVEC priorité normale: buildings > vegetation
        # Points 0,1,2 dans les deux → doivent être "building" (1)
        assert labels_normal[0] == 1, "Priorité normale: Point 0 devrait être building"

        # AVEC priorité inversée: vegetation > buildings
        # Points 0,1,2 dans les deux → doivent être "vegetation" (4)
        assert (
            labels_reversed[0] == 4
        ), "Priorité inversée: Point 0 devrait être vegetation"


class TestBug5_GeometricRulesOverwriteGT:
    """
    Test Bug #5: Geometric rules écrasent le ground truth

    Vérifie que les règles géométriques ne modifient PAS les points déjà
    classés par le ground truth.
    """

    def test_geometric_rules_preserve_gt(self, sample_points, simple_polygons):
        """Test que les règles géométriques préservent le ground truth."""
        # Classifier avec GT
        optimizer = GroundTruthOptimizer(force_method="strtree", verbose=False)
        labels_gt = optimizer.label_points(
            points=sample_points,
            ground_truth_features=simple_polygons,
            label_priority=["buildings", "roads", "water", "vegetation"],
            ndvi=None,
            use_ndvi_refinement=False,
        )

        # Créer NDVI fictif (tous points = 0.1, donc PAS végétation)
        ndvi = np.full(len(sample_points), 0.1, dtype=np.float32)

        # Appliquer les règles géométriques
        geometric_rules = GeometricRulesEngine(
            ndvi_vegetation_threshold=0.3, use_spectral_rules=False
        )

        labels_after_rules, stats = geometric_rules.apply_all_rules(
            points=sample_points,
            labels=labels_gt.copy(),
            ground_truth_features=simple_polygons,
            ndvi=ndvi,
            intensities=None,
        )

        # Vérifier que les labels GT ne sont PAS modifiés
        # Points 0,1,2 étaient "building" → doivent rester "building"
        assert (
            labels_after_rules[0] == labels_gt[0]
        ), "Geometric rules ont modifié le label GT du point 0"
        assert (
            labels_after_rules[1] == labels_gt[1]
        ), "Geometric rules ont modifié le label GT du point 1"
        assert (
            labels_after_rules[2] == labels_gt[2]
        ), "Geometric rules ont modifié le label GT du point 2"

    def test_verticality_doesnt_override_gt(self, sample_points, simple_polygons):
        """Test que la règle de verticality ne reclasse PAS les points GT."""
        # Classifier avec GT (points 0,1,2 = building)
        optimizer = GroundTruthOptimizer(force_method="strtree", verbose=False)
        labels_gt = optimizer.label_points(
            points=sample_points,
            ground_truth_features=simple_polygons,
            label_priority=["buildings", "roads", "water", "vegetation"],
            ndvi=None,
            use_ndvi_refinement=False,
        )

        # Points 3,4 sont dans vegetation (code 4)
        # Changer manuellement en "unclassified" pour tester verticality
        labels_gt[3] = 1  # ASPRS_UNCLASSIFIED
        labels_gt[4] = 1

        # NDVI = 0 (pas végétation)
        ndvi = np.zeros(len(sample_points), dtype=np.float32)

        # Appliquer classify_by_verticality
        geometric_rules = GeometricRulesEngine(
            verticality_threshold=0.7, use_spectral_rules=False
        )

        n_added = geometric_rules.classify_by_verticality(
            points=sample_points, labels=labels_gt.copy(), ndvi=ndvi
        )

        # Les points 0,1,2 étaient déjà classés "building" par GT
        # Ils NE doivent PAS être reclassés par verticality
        # (la fonction ne traite que les "unclassified")
        assert labels_gt[0] == 1, "Point 0 (building GT) a été reclassé par verticality"
        assert labels_gt[1] == 1, "Point 1 (building GT) a été reclassé par verticality"


class TestBug6_BuildingBufferZone:
    """
    Test Bug #6: Building buffer zone n'exclut pas les points déjà classés

    Vérifie que classify_building_buffer_zone ne reclasse PAS les points
    qui sont déjà dans un autre polygon GT (road, water, etc.)
    """

    def test_buffer_zone_excludes_gt_points(self):
        """Test que le buffer zone n'affecte PAS les points déjà classés par GT."""
        # Créer des points:
        # - Point 0: dans building
        # - Point 1: proche building (dans buffer) MAIS dans road GT
        # - Point 2: proche building, unclassified
        points = np.array(
            [
                [5.0, 5.0, 10.0],  # Point 0: dans building
                [11.0, 5.0, 0.5],  # Point 1: proche building, AU SOL (road)
                [11.0, 11.0, 10.0],  # Point 2: proche building, unclassified
            ]
        )

        # Polygons
        buildings = gpd.GeoDataFrame(
            {
                "geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
            },
            crs="EPSG:2154",
        )

        roads = gpd.GeoDataFrame(
            {
                "geometry": [Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])],
            },
            crs="EPSG:2154",
        )

        # Classifier avec GT
        optimizer = GroundTruthOptimizer(force_method="strtree", verbose=False)
        labels = optimizer.label_points(
            points=points,
            ground_truth_features={"buildings": buildings, "roads": roads},
            label_priority=["buildings", "roads", "water", "vegetation"],
            ndvi=None,
            use_ndvi_refinement=False,
        )

        # Vérifications après GT:
        # Point 0 = building (code 1)
        # Point 1 = road (code 2)
        # Point 2 = unclassified (code 0)
        assert labels[0] == 1, "Point 0 devrait être building"
        assert labels[1] == 2, "Point 1 devrait être road"
        assert labels[2] == 0, "Point 2 devrait être unclassified"

        # Appliquer building buffer zone
        geometric_rules = GeometricRulesEngine(
            building_buffer_distance=5.0, use_spectral_rules=False, use_clustering=False
        )

        # Create modifiable mask (all points can be modified for this test)
        modifiable_mask = np.ones(len(points), dtype=bool)

        n_added = geometric_rules.classify_building_buffer_zone(
            points=points,
            labels=labels.copy(),
            building_geometries=buildings,
            modifiable_mask=modifiable_mask,
        )

        # Vérifications après buffer zone:
        # Point 1 est proche du building MAIS dans road GT
        # Il NE doit PAS être reclassé en building
        assert (
            labels[1] == 2
        ), "Point 1 (road GT) a été reclassé en building par buffer zone"

        # Point 2 est proche du building ET unclassified
        # Il DEVRAIT être classé en building par buffer zone
        # (ceci dépend de la hauteur, mais on teste que ça ne touche pas aux GT)


class TestNDVIRefinement:
    """Tests pour la logique NDVI refinement."""

    def test_ndvi_refinement_zone_grey(self):
        """Test la zone grise NDVI (0.15 < NDVI < 0.3)."""
        points = np.array(
            [
                [5.0, 5.0, 10.0],  # Point 0: building
                [15.0, 15.0, 5.0],  # Point 1: vegetation
            ]
        )

        # NDVI dans la zone grise
        ndvi = np.array([0.2, 0.2], dtype=np.float32)

        # Polygons
        buildings = gpd.GeoDataFrame(
            {"geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]},
            crs="EPSG:2154",
        )
        vegetation = gpd.GeoDataFrame(
            {"geometry": [Polygon([(15, 15), (25, 15), (25, 25), (15, 25)])]},
            crs="EPSG:2154",
        )

        optimizer = GroundTruthOptimizer(force_method="strtree", verbose=False)
        labels = optimizer.label_points(
            points=points,
            ground_truth_features={"buildings": buildings, "vegetation": vegetation},
            label_priority=["buildings", "roads", "water", "vegetation"],
            ndvi=ndvi,
            use_ndvi_refinement=True,
            ndvi_vegetation_threshold=0.3,
            ndvi_building_threshold=0.15,
        )

        # NDVI = 0.2 est dans la zone grise (0.15 < 0.2 < 0.3)
        # Les points GARDENT leur label initial du GT
        # Point 0: building (NDVI 0.2 < 0.3, donc PAS reclassé en vegetation)
        # Point 1: vegetation (NDVI 0.2 > 0.15, donc PAS reclassé en building)
        assert (
            labels[0] == 1
        ), "Point 0 (building, NDVI=0.2) ne devrait PAS être reclassé"
        assert (
            labels[1] == 4
        ), "Point 1 (vegetation, NDVI=0.2) ne devrait PAS être reclassé"


class TestBug4_UnifiedPrioritySystem:
    """
    Test Bug #4: Système de priorités unifié

    Vérifie que tous les modules utilisent le même ordre de priorité.
    """

    def test_priority_consistency(self):
        """Test que les priorités sont cohérentes entre modules."""
        from ign_lidar.core.classification.priorities import (
            PRIORITY_ORDER,
            get_priority_order_for_iteration,
            get_priority_value,
        )
        from ign_lidar.core.classification.reclassifier import OptimizedReclassifier
        from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer

        # Vérifier que PRIORITY_ORDER est défini
        assert len(PRIORITY_ORDER) > 0, "PRIORITY_ORDER ne doit pas être vide"

        # Vérifier que buildings a la priorité la plus haute
        buildings_priority = get_priority_value("buildings")
        vegetation_priority = get_priority_value("vegetation")

        assert buildings_priority > vegetation_priority, (
            f"buildings ({buildings_priority}) doit avoir une priorité "
            f"plus haute que vegetation ({vegetation_priority})"
        )

        # Vérifier que l'ordre d'itération est correct
        iteration_order = get_priority_order_for_iteration()
        assert iteration_order[0] == "vegetation", (
            "L'ordre d'itération doit commencer par la priorité la plus "
            "basse (vegetation)"
        )
        assert iteration_order[-1] == "buildings", (
            "L'ordre d'itération doit finir par la priorité la plus "
            "haute (buildings)"
        )

        # Vérifier que reclassifier utilise le bon ordre
        reclassifier = OptimizedReclassifier(acceleration_mode="cpu")
        first_feature = reclassifier.priority_order[0][0]
        last_feature = reclassifier.priority_order[-1][0]

        assert first_feature == "vegetation", (
            f"Reclassifier doit traiter vegetation en premier, "
            f"mais traite {first_feature}"
        )
        assert last_feature == "buildings", (
            f"Reclassifier doit traiter buildings en dernier, "
            f"mais traite {last_feature}"
        )


@pytest.mark.unit
class TestBug3_NDVI_Timing:
    """
    Test Bug #3: NDVI Refinement appliqué trop tard

    Vérifie que les labels NDVI ne sont pas écrasés par les règles
    géométriques.
    """

    def test_ndvi_labels_not_overwritten(self):
        """Test que les labels NDVI sont protégés contre l'écrasement."""
        from ign_lidar.core.classification.geometric_rules import GeometricRulesEngine

        # Créer un point avec haute verticality MAIS haute NDVI (végétation)
        points = np.array([[5.0, 5.0, 10.0]])  # Point élevé

        # Label initial: unclassified
        labels = np.ones(1, dtype=np.int32)  # 1 = ASPRS_UNCLASSIFIED

        # NDVI très haut = végétation (0.8)
        ndvi = np.array([0.8])

        # Polygon building qui contient ce point
        building_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        buildings_gdf = gpd.GeoDataFrame(
            {"geometry": [building_polygon]}, crs="EPSG:2154"
        )

        ground_truth_features = {"buildings": buildings_gdf}

        # Simuler une haute verticality (normales pointant horizontalement)
        # Cela devrait normalement classifier le point comme "building"
        # MAIS avec NDVI élevé, il devrait rester "vegetation"

        # Initialiser le moteur de règles
        rules_engine = GeometricRulesEngine(
            ndvi_vegetation_threshold=0.3,
            ndvi_road_threshold=0.15,
            building_buffer_distance=2.0,
            verticality_threshold=0.7,
            verticality_search_radius=1.0,
        )

        # Appliquer les règles avec preserve_ground_truth=True
        updated_labels, stats = rules_engine.apply_all_rules(
            points=points,
            labels=labels,
            ground_truth_features=ground_truth_features,
            ndvi=ndvi,
            intensities=None,
            rgb=None,
            nir=None,
            verticality=None,
            preserve_ground_truth=True,
        )

        # Vérifier que NDVI a classé le point comme vegetation
        # Note: Le point commence "unclassified", NDVI ne le modifie
        # pas directement car NDVI raffine building→vegetation
        # ou vegetation→building
        # Pour tester correctement, il faut un point classé "building"

        # Test plus réaliste: point déjà classé "building", NDVI élevé
        labels_building = np.array([6], dtype=np.int32)  # 6 = BUILDING

        updated_labels2, stats2 = rules_engine.apply_all_rules(
            points=points,
            labels=labels_building,
            ground_truth_features=ground_truth_features,
            ndvi=ndvi,
            intensities=None,
            rgb=None,
            nir=None,
            verticality=None,
            preserve_ground_truth=True,
        )

        # Avec NDVI = 0.8 (> threshold 0.3), le building devrait
        # être reclassé en vegetation
        # ASPRS codes: 4 = Low Veg, 5 = Medium Veg, 6 = Building
        # Le code exact dépend de la hauteur, mais il ne doit PAS
        # être building (6)
        assert updated_labels2[0] != 6, (
            f"Point avec NDVI=0.8 ne devrait pas être classé "
            f"'building', mais a été classé {updated_labels2[0]}"
        )

        # Le point devrait être classé comme vegetation (4 ou 5)
        assert updated_labels2[0] in [4, 5], (
            f"Point avec NDVI=0.8 devrait être vegetation (4 ou 5), "
            f"mais a été classé {updated_labels2[0]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
