"""
Test de la Correction: Végétation en Hauteur Classifiée comme Route

Ce script teste la correction du problème où les points en hauteur
(arbres, végétation) au-dessus des routes sont incorrectement classifiés
comme "route" au lieu de "végétation".

Problème:
- Routes (marron) apparaissent en hauteur (arbres)
- Seuls les points au niveau du sol devraient être "route"
- Points élevés avec NDVI élevé = végétation

Solution:
- Filtre de hauteur strict: routes ≤ 1.5m au-dessus du sol
- Points > 1.5m + NDVI > 0.25 → végétation
- Séparation verticale: sol vs hauteur

Auteur: Simon Ducournau
Date: 25 Octobre 2025
"""

import numpy as np
import logging
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_scenario() -> Dict:
    """
    Créer un scénario de test avec:
    - Routes au niveau du sol
    - Arbres au-dessus des routes (hauteur > 2m)
    - NDVI différenciant végétation vs route
    """
    
    # Scénario: Route avec des arbres au-dessus
    n_ground_road = 100  # Points de route au sol
    n_trees_above = 50   # Points d'arbres au-dessus
    
    # 1. Route au niveau du sol (0-0.5m)
    road_points = np.random.rand(n_ground_road, 3)
    road_points[:, 2] = np.random.uniform(0.0, 0.5, n_ground_road)  # Hauteur 0-0.5m
    road_ndvi = np.random.uniform(0.05, 0.15, n_ground_road)  # NDVI faible (asphalte)
    
    # 2. Arbres au-dessus de la route (2-6m de hauteur)
    tree_points = np.random.rand(n_trees_above, 3)
    tree_points[:, 2] = np.random.uniform(2.0, 6.0, n_trees_above)  # Hauteur 2-6m
    tree_ndvi = np.random.uniform(0.4, 0.7, n_trees_above)  # NDVI élevé (végétation)
    
    # Combiner
    points = np.vstack([road_points, tree_points])
    ndvi = np.concatenate([road_ndvi, tree_ndvi])
    height = points[:, 2]
    
    # Labels initiaux: TOUS classifiés comme "route" (PROBLÈME!)
    from ign_lidar.classification_schema import ASPRSClass
    labels = np.full(len(points), ASPRSClass.ROAD_SURFACE, dtype=np.uint8)
    
    # Road mask (tous les points sont dans le polygone route)
    road_mask = np.ones(len(points), dtype=bool)
    
    return {
        'points': points,
        'labels': labels,
        'height': height,
        'ndvi': ndvi,
        'road_mask': road_mask,
        'n_ground_road': n_ground_road,
        'n_trees_above': n_trees_above
    }


def test_elevated_road_fix():
    """Test principal de la correction."""
    
    logger.info("=" * 70)
    logger.info("TEST: Correction Végétation en Hauteur Classifiée comme Route")
    logger.info("=" * 70)
    
    # 1. Créer le scénario de test
    logger.info("\n1️⃣ Création du scénario de test...")
    scenario = create_test_scenario()
    
    points = scenario['points']
    labels = scenario['labels']
    height = scenario['height']
    ndvi = scenario['ndvi']
    road_mask = scenario['road_mask']
    
    logger.info(f"   ✓ {len(points)} points créés")
    logger.info(f"   ✓ {scenario['n_ground_road']} points de route au sol (0-0.5m)")
    logger.info(f"   ✓ {scenario['n_trees_above']} points d'arbres en hauteur (2-6m)")
    
    # 2. État initial (AVANT correction)
    logger.info("\n2️⃣ État AVANT correction:")
    from ign_lidar.classification_schema import ASPRSClass
    
    n_road_before = np.sum(labels == ASPRSClass.ROAD_SURFACE)
    n_veg_before = np.sum(
        (labels == ASPRSClass.LOW_VEGETATION) |
        (labels == ASPRSClass.MEDIUM_VEGETATION) |
        (labels == ASPRSClass.HIGH_VEGETATION)
    )
    
    logger.info(f"   ❌ Routes: {n_road_before} points")
    logger.info(f"   ✅ Végétation: {n_veg_before} points")
    logger.info(f"   ⚠️ PROBLÈME: {scenario['n_trees_above']} arbres classifiés comme route!")
    
    # 3. Appliquer la correction
    logger.info("\n3️⃣ Application de la correction...")
    
    from ign_lidar.core.classification import GroundTruthRefiner
    
    refiner = GroundTruthRefiner()
    
    # Planarity artificielle (routes = plat, arbres = complexe)
    planarity = np.ones(len(points))
    planarity[:scenario['n_ground_road']] = 0.9  # Route = plat
    planarity[scenario['n_ground_road']:] = 0.3  # Arbres = complexe
    
    labels_corrected, stats = refiner.refine_road_classification(
        labels=labels.copy(),
        points=points,
        road_mask=road_mask,
        height=height,
        planarity=planarity,
        ndvi=ndvi
    )
    
    logger.info("   ✓ Correction appliquée")
    
    # 4. État APRÈS correction
    logger.info("\n4️⃣ État APRÈS correction:")
    
    n_road_after = np.sum(labels_corrected == ASPRSClass.ROAD_SURFACE)
    n_high_veg_after = np.sum(labels_corrected == ASPRSClass.HIGH_VEGETATION)
    n_med_veg_after = np.sum(labels_corrected == ASPRSClass.MEDIUM_VEGETATION)
    n_low_veg_after = np.sum(labels_corrected == ASPRSClass.LOW_VEGETATION)
    
    logger.info(f"   ✅ Routes: {n_road_after} points")
    logger.info(f"   ✅ Végétation haute: {n_high_veg_after} points")
    logger.info(f"   ✅ Végétation moyenne: {n_med_veg_after} points")
    logger.info(f"   ✅ Végétation basse: {n_low_veg_after} points")
    
    # 5. Statistiques de la correction
    logger.info("\n5️⃣ Statistiques de correction:")
    
    for key, value in stats.items():
        if value > 0:
            logger.info(f"   • {key}: {value:,} points")
    
    # 6. Vérifications
    logger.info("\n6️⃣ Vérifications:")
    
    success = True
    
    # Vérification 1: Les points au sol restent routes
    ground_points_mask = height <= 1.5
    ground_roads = np.sum((labels_corrected == ASPRSClass.ROAD_SURFACE) & ground_points_mask)
    expected_ground_roads = scenario['n_ground_road']
    
    if ground_roads >= expected_ground_roads * 0.8:  # 80% de tolérance
        logger.info(f"   ✅ Routes au sol préservées: {ground_roads}/{expected_ground_roads}")
    else:
        logger.error(f"   ❌ Routes au sol perdues: {ground_roads}/{expected_ground_roads}")
        success = False
    
    # Vérification 2: Les arbres en hauteur sont végétation
    elevated_mask = height > 2.0
    elevated_veg = np.sum(
        elevated_mask & (
            (labels_corrected == ASPRSClass.LOW_VEGETATION) |
            (labels_corrected == ASPRSClass.MEDIUM_VEGETATION) |
            (labels_corrected == ASPRSClass.HIGH_VEGETATION)
        )
    )
    expected_elevated_veg = scenario['n_trees_above']
    
    if elevated_veg >= expected_elevated_veg * 0.8:  # 80% de tolérance
        logger.info(f"   ✅ Arbres reclassifiés en végétation: {elevated_veg}/{expected_elevated_veg}")
    else:
        logger.error(f"   ❌ Arbres encore routes: {elevated_veg}/{expected_elevated_veg}")
        success = False
    
    # Vérification 3: Aucun point élevé n'est route
    elevated_roads = np.sum((labels_corrected == ASPRSClass.ROAD_SURFACE) & (height > 2.0))
    
    if elevated_roads == 0:
        logger.info(f"   ✅ Aucune route en hauteur (>2m)")
    else:
        logger.error(f"   ❌ ENCORE {elevated_roads} routes en hauteur!")
        success = False
    
    # 7. Résultat final
    logger.info("\n" + "=" * 70)
    if success:
        logger.info("✅ TEST RÉUSSI: La correction fonctionne correctement!")
        logger.info("   • Routes restent au niveau du sol (<1.5m)")
        logger.info("   • Arbres/végétation en hauteur correctement reclassifiés")
        logger.info("   • NDVI utilisé pour confirmation")
    else:
        logger.error("❌ TEST ÉCHOUÉ: Des problèmes subsistent")
    logger.info("=" * 70)
    
    return success


if __name__ == "__main__":
    success = test_elevated_road_fix()
    exit(0 if success else 1)
