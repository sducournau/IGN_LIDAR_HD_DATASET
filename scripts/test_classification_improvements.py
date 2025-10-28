#!/usr/bin/env python3
"""
Test des améliorations classification bâtiments/végétation.

Teste:
1. Classification façades bâtiments (points non classifiés)
2. Classification végétation avec NDVI/NIR

Usage:
    python scripts/test_classification_improvements.py
"""

import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_facade_classification_improvements():
    """Test des améliorations de classification des façades."""
    logger.info("=" * 80)
    logger.info("TEST 1: Classification améliorée des façades de bâtiments")
    logger.info("=" * 80)

    from ign_lidar.core.classification.building.facade_processor import (
        FacadeProcessor,
        FacadeSegment,
        FacadeOrientation,
    )
    from shapely.geometry import LineString

    # Créer une façade de test
    facade = FacadeSegment(
        edge_line=LineString([(0, 0), (10, 0)]),
        orientation=FacadeOrientation.NORTH,
        buffer_distance=2.5,
        verticality_threshold=0.55,
    )

    # Créer des points de test avec différentes verticalités
    n_points = 1000
    points = np.random.rand(n_points, 3) * 10
    heights = np.random.rand(n_points) * 5

    # Verticalité: mélange de haute (0.8), moyenne (0.6), et faible (0.4)
    verticality = np.concatenate(
        [
            np.full(300, 0.8),  # 300 points haute verticalité
            np.full(400, 0.6),  # 400 points verticalité moyenne
            np.full(300, 0.4),  # 300 points faible verticalité
        ]
    )
    np.random.shuffle(verticality)

    # Créer le processor
    processor = FacadeProcessor(
        facade=facade, points=points, heights=heights, verticality=verticality
    )

    # Simuler l'identification des candidats (tous les points pour le test)
    processor._candidate_mask = np.ones(n_points, dtype=bool)
    processor.facade.point_indices = np.arange(n_points)
    processor.facade.n_points = n_points

    # Tester la classification
    logger.info(f"Points total: {n_points}")
    logger.info(f"  - Haute verticalité (≥0.7): {np.sum(verticality >= 0.7)}")
    logger.info(
        f"  - Moyenne verticalité (0.5-0.7): {np.sum((verticality >= 0.5) & (verticality < 0.7))}"
    )
    logger.info(f"  - Faible verticalité (<0.5): {np.sum(verticality < 0.5)}")

    processor._classify_wall_points()

    logger.info(f"✅ Points classifiés comme mur: {processor.facade.n_wall_points}")
    logger.info(
        f"✅ Taux de classification: {processor.facade.n_wall_points / n_points * 100:.1f}%"
    )

    # Vérifier que les améliorations fonctionnent
    expected_min_classification = 700  # Au moins 70% avec les améliorations
    if processor.facade.n_wall_points >= expected_min_classification:
        logger.info(
            "✅ TEST RÉUSSI: Classification améliorée détecte plus de points de façade"
        )
    else:
        logger.warning(
            f"⚠️ TEST ÉCHOUÉ: Seulement {processor.facade.n_wall_points} points classifiés (attendu ≥{expected_min_classification})"
        )

    return processor.facade.n_wall_points >= expected_min_classification


def test_vegetation_classification_improvements():
    """Test des améliorations de classification de la végétation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Classification améliorée de la végétation (NDVI/NIR)")
    logger.info("=" * 80)

    from ign_lidar.core.classification.spectral_rules import SpectralRulesEngine
    from ign_lidar.core.classification.constants import ASPRSClass

    # Créer le moteur de règles spectrales
    engine = SpectralRulesEngine()

    # Créer des points de test représentant différents types de végétation
    n_points = 1000

    # Végétation dense: NDVI élevé, NIR élevé
    n_dense = 200
    rgb_dense = np.random.rand(n_dense, 3) * 0.3
    rgb_dense[:, 1] += 0.3  # Plus de vert
    nir_dense = np.random.rand(n_dense) * 0.3 + 0.5  # NIR 0.5-0.8

    # Végétation moyenne: NDVI moyen, NIR moyen
    n_medium = 300
    rgb_medium = np.random.rand(n_medium, 3) * 0.4
    rgb_medium[:, 1] += 0.2
    nir_medium = np.random.rand(n_medium) * 0.2 + 0.35  # NIR 0.35-0.55

    # Végétation sparse/faible: NDVI faible, NIR faible mais positif
    n_sparse = 300
    rgb_sparse = np.random.rand(n_sparse, 3) * 0.5 + 0.3
    rgb_sparse[:, 1] += 0.15
    nir_sparse = np.random.rand(n_sparse) * 0.15 + 0.25  # NIR 0.25-0.40

    # Non-végétation (bâtiments, routes)
    n_other = 200
    rgb_other = np.random.rand(n_other, 3) * 0.4 + 0.3
    nir_other = np.random.rand(n_other) * 0.25 + 0.15  # NIR 0.15-0.40

    # Combiner tous les points
    rgb = np.vstack([rgb_dense, rgb_medium, rgb_sparse, rgb_other])
    nir = np.concatenate([nir_dense, nir_medium, nir_sparse, nir_other])

    # Labels initiaux (tous non classifiés)
    labels = np.full(n_points, int(ASPRSClass.UNCLASSIFIED))

    logger.info(f"Points de test: {n_points}")
    logger.info(f"  - Végétation dense: {n_dense}")
    logger.info(f"  - Végétation moyenne: {n_medium}")
    logger.info(f"  - Végétation sparse: {n_sparse}")
    logger.info(f"  - Non-végétation: {n_other}")

    # Classifier
    new_labels, stats = engine.classify_by_spectral_signature(
        rgb=rgb, nir=nir, current_labels=labels, apply_to_unclassified_only=True
    )

    # Compter les classifications de végétation
    veg_classified = np.sum(
        (new_labels == int(ASPRSClass.MEDIUM_VEGETATION))
        | (new_labels == int(ASPRSClass.LOW_VEGETATION))
    )

    logger.info(f"✅ Total végétation classifiée: {veg_classified}")
    logger.info(f"   - Végétation moyenne: {stats['vegetation_spectral']}")
    logger.info(f"   - Végétation sparse: {stats['vegetation_sparse']}")
    logger.info(
        f"✅ Taux de détection végétation: {veg_classified / (n_dense + n_medium + n_sparse) * 100:.1f}%"
    )

    # Vérifier que les améliorations fonctionnent
    expected_min_veg = 600  # Au moins 75% de la végétation détectée
    if veg_classified >= expected_min_veg:
        logger.info(
            "✅ TEST RÉUSSI: Classification améliorée détecte plus de végétation"
        )
    else:
        logger.warning(
            f"⚠️ TEST ÉCHOUÉ: Seulement {veg_classified} points végétation classifiés (attendu ≥{expected_min_veg})"
        )

    return veg_classified >= expected_min_veg


def main():
    """Exécuter tous les tests."""
    logger.info("🚀 Démarrage des tests d'amélioration de classification")
    logger.info("")

    # Test 1: Façades de bâtiments
    test1_success = test_facade_classification_improvements()

    # Test 2: Végétation
    test2_success = test_vegetation_classification_improvements()

    # Résumé
    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ DES TESTS")
    logger.info("=" * 80)
    logger.info(
        f"Test 1 (Façades bâtiments): {'✅ RÉUSSI' if test1_success else '❌ ÉCHOUÉ'}"
    )
    logger.info(f"Test 2 (Végétation): {'✅ RÉUSSI' if test2_success else '❌ ÉCHOUÉ'}")

    if test1_success and test2_success:
        logger.info("\n🎉 TOUS LES TESTS RÉUSSIS!")
        return 0
    else:
        logger.warning("\n⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
