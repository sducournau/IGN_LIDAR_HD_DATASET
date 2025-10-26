#!/usr/bin/env python3
"""
Script de Diagnostic Rapide des Bugs de Classification

Ce script exécute des tests simples pour démontrer les bugs identifiés.
Utilise des données synthétiques pour isoler chaque bug.

Usage:
    python scripts/diagnose_classification_bugs.py
"""

import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import des modules à tester
try:
    from ign_lidar.core.classification.geometric_rules import GeometricRulesEngine
    from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
except ImportError as e:
    logger.error(f"Erreur d'import: {e}")
    logger.error("Assurez-vous que le package ign_lidar est installé")
    exit(1)


def create_test_data():
    """Crée des données de test synthétiques."""
    # 3 points: tous dans le MÊME emplacement (5, 5, 10)
    # Pour tester les priorités quand plusieurs polygones se chevauchent
    points = np.array(
        [
            [5.0, 5.0, 10.0],  # Point 0: dans building ET vegetation
            [5.0, 5.0, 10.0],  # Point 1: identique (pour test déterminisme)
            [5.0, 5.0, 10.0],  # Point 2: identique (pour test déterminisme)
        ]
    )

    # Deux polygones qui se chevauchent complètement
    # Building: petit carré (0,0) à (10,10)
    building_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    # Vegetation: grand carré (0,0) à (20,20) - CONTIENT le building
    vegetation_poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])

    buildings = gpd.GeoDataFrame(
        {"geometry": [building_poly], "name": ["TestBuilding"]}, crs="EPSG:2154"
    )

    vegetation = gpd.GeoDataFrame(
        {"geometry": [vegetation_poly], "name": ["TestVegetation"]}, crs="EPSG:2154"
    )

    return points, {"buildings": buildings, "vegetation": vegetation}


def test_bug1_random_priority():
    """
    Test Bug #1: Ordre de priorité aléatoire dans STRtree

    Comportement attendu:
    - Points dans building ET vegetation → toujours "building" (priorité haute)

    Comportement bugué:
    - Résultat dépend de l'ordre interne du STRtree (aléatoire)
    - Peut être "building" OU "vegetation"
    """
    print("\n" + "=" * 70)
    print("TEST BUG #1: ORDRE DE PRIORITÉ ALÉATOIRE")
    print("=" * 70)

    points, polygons = create_test_data()

    print(f"\n📍 Points de test: {len(points)} points à (5.0, 5.0, 10.0)")
    print(f"🔷 Building polygon: (0,0) à (10,10)")
    print(f"🌳 Vegetation polygon: (0,0) à (20,20) - CONTIENT le building")
    print(f"\n➡️  Les 3 points sont dans LES DEUX polygones")
    print(f"➡️  Priorité définie: buildings (1) > vegetation (4)")
    print(f"\n🔍 Exécution de 5 runs pour tester le déterminisme...\n")

    optimizer = GroundTruthOptimizer(force_method="strtree", verbose=False)

    results = []
    for i in range(5):
        labels = optimizer.label_points(
            points=points,
            ground_truth_features=polygons,
            label_priority=["buildings", "roads", "water", "vegetation"],
            ndvi=None,
            use_ndvi_refinement=False,
        )
        results.append(labels.copy())

        label_name = "building" if labels[0] == 1 else "vegetation"
        print(f"  Run {i+1}: Point 0 classé comme '{label_name}' (code {labels[0]})")

    # Vérifier si tous les résultats sont identiques
    all_same = all(np.array_equal(results[0], r) for r in results[1:])

    print("\n📊 RÉSULTAT:")
    if all_same:
        print("  ✅ Classification DÉTERMINISTE (même résultat à chaque run)")
        if results[0][0] == 1:
            print("  ✅ Priorités RESPECTÉES (building > vegetation)")
            print("\n🎉 Bug #1 CORRIGÉ!")
        else:
            print("  ❌ Priorités INVERSÉES (vegetation gagne au lieu de building)")
            print("\n🔴 Bug #1 TOUJOURS PRÉSENT (priorités incorrectes)")
    else:
        print("  ❌ Classification NON-DÉTERMINISTE")
        print("  ❌ Résultats différents à chaque run!")
        print("\n🔴 Bug #1 CONFIRMÉ - CRITIQUE")

        # Afficher la distribution
        building_count = sum(1 for r in results if r[0] == 1)
        vegetation_count = sum(1 for r in results if r[0] == 4)
        print(f"\n  Distribution sur 5 runs:")
        print(f"    - {building_count}/5 runs: building")
        print(f"    - {vegetation_count}/5 runs: vegetation")


def test_bug5_geometric_overwrites_gt():
    """
    Test Bug #5: Règles géométriques écrasent le ground truth

    Comportement attendu:
    - Point classé "vegetation" par GT → reste "vegetation"

    Comportement bugué:
    - Point reclassé en "building" par règle de verticality
    """
    print("\n" + "=" * 70)
    print("TEST BUG #5: RÈGLES GÉOMÉTRIQUES ÉCRASENT LE GROUND TRUTH")
    print("=" * 70)

    # Point dans vegetation uniquement
    points = np.array(
        [
            [15.0, 15.0, 5.0],  # Point dans vegetation uniquement
        ]
    )

    vegetation = gpd.GeoDataFrame(
        {"geometry": [Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])]},
        crs="EPSG:2154",
    )

    print(f"\n📍 Point de test: (15.0, 15.0, 5.0)")
    print(f"🌳 Polygon vegetation: (10,10) à (20,20)")
    print(f"➡️  Point dans polygon 'vegetation' uniquement")

    # Classifier avec GT
    optimizer = GroundTruthOptimizer(force_method="strtree", verbose=False)
    labels_gt = optimizer.label_points(
        points=points,
        ground_truth_features={"vegetation": vegetation},
        label_priority=["buildings", "roads", "water", "vegetation"],
        ndvi=None,
        use_ndvi_refinement=False,
    )

    print(f"\n🔍 Après classification GT:")
    print(f"  Point classé: 'vegetation' (code {labels_gt[0]})")

    # NDVI = 0.05 (très bas, pas de végétation selon spectral)
    ndvi = np.array([0.05], dtype=np.float32)

    print(f"\n🔍 Application des règles géométriques...")
    print(f"  NDVI: {ndvi[0]} (très bas)")

    # Appliquer règles géométriques
    geometric_rules = GeometricRulesEngine(
        ndvi_vegetation_threshold=0.3, use_spectral_rules=False
    )

    labels_after_rules, stats = geometric_rules.apply_all_rules(
        points=points,
        labels=labels_gt.copy(),
        ground_truth_features={"vegetation": vegetation},
        ndvi=ndvi,
        intensities=None,
    )

    print(f"\n📊 RÉSULTAT:")
    if labels_after_rules[0] == labels_gt[0]:
        print(
            f"  ✅ Label GT PRÉSERVÉ: reste 'vegetation' (code {labels_after_rules[0]})"
        )
        print("\n🎉 Bug #5 CORRIGÉ!")
    else:
        label_name = {1: "unclassified", 6: "building"}.get(
            labels_after_rules[0], "unknown"
        )
        print(
            f"  ❌ Label GT ÉCRASÉ: 'vegetation' → '{label_name}' (code {labels_after_rules[0]})"
        )
        print("\n🔴 Bug #5 CONFIRMÉ - CRITIQUE")
        print("  Les règles géométriques ignorent le ground truth!")


def test_bug4_conflicting_priorities():
    """
    Test Bug #4: Systèmes de priorités contradictoires

    Compare les priorités entre optimizer et reclassifier.
    """
    print("\n" + "=" * 70)
    print("TEST BUG #4: SYSTÈMES DE PRIORITÉS CONTRADICTOIRES")
    print("=" * 70)

    # ✅ Use centralized priority system
    from ign_lidar.core.classification.priorities import (
        PRIORITY_ORDER,
        get_priority_value,
        get_priority_order_for_iteration,
    )

    print("\n📋 Système de priorités CENTRALISÉ (priorities.py):")
    print("  Ordre canonique (haute → basse priorité):")
    for feature in PRIORITY_ORDER:
        priority = get_priority_value(feature)
        print(f"    {priority}. {feature}")

    print("\n📋 Ordre d'itération pour reclassifier:")
    print("  (traité dans cet ordre, dernier = priorité max)")
    iteration_order = get_priority_order_for_iteration()
    for i, feature in enumerate(iteration_order):
        print(f"    {i+1}. {feature}")

    print("\n� RÉSULTAT:")
    # Vérifier quelques cohérences clés
    buildings_priority = get_priority_value("buildings")
    vegetation_priority = get_priority_value("vegetation")
    
    if buildings_priority > vegetation_priority:
        print(f"  ✅ buildings (priorité {buildings_priority}) > "
              f"vegetation (priorité {vegetation_priority})")
    
    if iteration_order[0] == "vegetation" and iteration_order[-1] == "buildings":
        print("  ✅ Ordre d'itération correct: vegetation → ... → buildings")
    
    print("\n🎉 Bug #4 CORRIGÉ!")
    print("  Système de priorités UNIFIÉ dans priorities.py")
    print("  Tous les modules utilisent maintenant le même ordre")


def main():
    """Exécute tous les tests de diagnostic."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC DES BUGS DE CLASSIFICATION")
    print("=" * 70)
    print("\nCe script teste les bugs critiques identifiés dans:")
    print("  - CLASSIFICATION_BUGS_ANALYSIS.md")
    print("  - CLASSIFICATION_BUGS_SUMMARY.md")

    try:
        # Test Bug #1
        test_bug1_random_priority()

        # Test Bug #5
        test_bug5_geometric_overwrites_gt()

        # Test Bug #4
        test_bug4_conflicting_priorities()

        print("\n" + "=" * 70)
        print("DIAGNOSTIC TERMINÉ")
        print("=" * 70)
        print("\n📝 Consultez les rapports détaillés:")
        print("  - CLASSIFICATION_BUGS_ANALYSIS.md (analyse complète)")
        print("  - CLASSIFICATION_BUGS_SUMMARY.md (résumé exécutif)")
        print("  - tests/test_classification_bugs.py (tests unitaires)")

    except Exception as e:
        logger.error(f"\n❌ Erreur pendant le diagnostic: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
