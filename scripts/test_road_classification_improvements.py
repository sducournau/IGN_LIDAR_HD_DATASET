"""
Test des Améliorations de Classification des Routes

Ce script teste les nouvelles règles de filtrage pour les routes :
- Exclusion de végétation (NDVI, courbure)
- Exclusion de bâtiments (verticalité, hauteur)
- Protection des classifications existantes

Auteur: Simon Ducournau
Date: 19 Octobre 2025
"""

import numpy as np
import logging
from typing import Tuple

# Import du module de raffinement
from ign_lidar.core.classification.classification_refinement import (
    refine_road_classification,
    RefinementConfig
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_test_data() -> Tuple[np.ndarray, dict]:
    """
    Créer des données de test avec différents scénarios.
    
    Returns:
        Tuple (labels, features) avec:
        - labels: classifications initiales
        - features: dictionnaire de features
    """
    n_points = 1000
    
    # Initialiser toutes les étiquettes comme route (classe 9 = ground/route)
    labels = np.ones(n_points, dtype=np.int32) * 9
    
    # Créer un masque ground truth simulé (tous les points)
    ground_truth_road_mask = np.ones(n_points, dtype=bool)
    
    # Coordonnées XYZ
    points = np.random.rand(n_points, 3) * 100
    
    # === SCÉNARIO 1: Points de Route Valides (0-200) ===
    # Caractéristiques typiques d'une route
    height = np.random.rand(200) * 0.3  # Très proche du sol (0-0.3m)
    planarity = 0.8 + np.random.rand(200) * 0.15  # Très plat (0.8-0.95)
    roughness = np.random.rand(200) * 0.03  # Lisse (0-0.03)
    intensity = 0.3 + np.random.rand(200) * 0.3  # Asphalte typique (0.3-0.6)
    ndvi = np.random.rand(200) * 0.1  # Pas de végétation (0-0.1)
    curvature = np.random.rand(200) * 0.02  # Surface plane (0-0.02)
    verticality = np.random.rand(200) * 0.15  # Horizontal (0-0.15)
    
    # === SCÉNARIO 2: Végétation - Arbres en Bordure (200-400) ===
    # Devrait être EXCLU par filtres NDVI et courbure
    height = np.concatenate([height, 3.0 + np.random.rand(200) * 5.0])  # Élevé (3-8m)
    planarity = np.concatenate([planarity, np.random.rand(200) * 0.3])  # Faible (0-0.3)
    roughness = np.concatenate([roughness, 0.1 + np.random.rand(200) * 0.15])  # Rugueux
    intensity = np.concatenate([intensity, np.random.rand(200) * 0.4])  # Faible intensité
    ndvi = np.concatenate([ndvi, 0.4 + np.random.rand(200) * 0.4])  # NDVI élevé (0.4-0.8) ✓
    curvature = np.concatenate([curvature, 0.08 + np.random.rand(200) * 0.12])  # Forte courbure ✓
    verticality = np.concatenate([verticality, np.random.rand(200) * 0.5])  # Variable
    
    # Marquer certains points comme végétation préclassifiée (protection)
    labels[250:300] = 11  # LOD2_VEG_HIGH
    
    # === SCÉNARIO 3: Bâtiments Adjacents (400-600) ===
    # Devrait être EXCLU par filtres verticalité et hauteur
    height = np.concatenate([height, 2.0 + np.random.rand(200) * 6.0])  # Élevé (2-8m) ✓
    planarity = np.concatenate([planarity, 0.6 + np.random.rand(200) * 0.3])  # Moyennement plat
    roughness = np.concatenate([roughness, np.random.rand(200) * 0.05])  # Lisse (murs)
    intensity = np.concatenate([intensity, 0.4 + np.random.rand(200) * 0.4])  # Bâtiment
    ndvi = np.concatenate([ndvi, np.random.rand(200) * 0.15])  # Pas végétation (0-0.15)
    curvature = np.concatenate([curvature, np.random.rand(200) * 0.03])  # Faible courbure
    verticality = np.concatenate([verticality, 0.6 + np.random.rand(200) * 0.35])  # Vertical ✓
    
    # Marquer certains points comme murs préclassifiés (protection)
    labels[450:500] = 0  # LOD2_WALL
    
    # === SCÉNARIO 4: Zone Mixte - Cas Limites (600-800) ===
    # Cas à la limite des seuils - certains acceptés, d'autres rejetés
    height = np.concatenate([height, 0.5 + np.random.rand(200) * 1.0])  # Limite (0.5-1.5m)
    planarity = np.concatenate([planarity, 0.6 + np.random.rand(200) * 0.15])  # Limite (0.6-0.75)
    roughness = np.concatenate([roughness, 0.04 + np.random.rand(200) * 0.03])  # Limite
    intensity = np.concatenate([intensity, 0.2 + np.random.rand(200) * 0.6])  # Variable
    ndvi = np.concatenate([ndvi, 0.15 + np.random.rand(200) * 0.1])  # Limite (0.15-0.25)
    curvature = np.concatenate([curvature, 0.03 + np.random.rand(200) * 0.04])  # Limite
    verticality = np.concatenate([verticality, 0.2 + np.random.rand(200) * 0.2])  # Limite
    
    # === SCÉNARIO 5: Faux Positifs Potentiels (800-1000) ===
    # Points hors polygones BD TOPO (non marqués dans ground truth)
    ground_truth_road_mask[800:1000] = False
    
    height = np.concatenate([height, np.random.rand(200) * 0.5])
    planarity = np.concatenate([planarity, 0.7 + np.random.rand(200) * 0.2])
    roughness = np.concatenate([roughness, np.random.rand(200) * 0.04])
    intensity = np.concatenate([intensity, 0.3 + np.random.rand(200) * 0.3])
    ndvi = np.concatenate([ndvi, np.random.rand(200) * 0.15])
    curvature = np.concatenate([curvature, np.random.rand(200) * 0.03])
    verticality = np.concatenate([verticality, np.random.rand(200) * 0.2])
    
    # Assembler les features
    features = {
        'points': points,
        'height': height,
        'planarity': planarity,
        'roughness': roughness,
        'intensity': intensity,
        'ndvi': ndvi,
        'curvature': curvature,
        'verticality': verticality,
        'ground_truth_road_mask': ground_truth_road_mask
    }
    
    return labels, features


def test_road_classification_improvements():
    """Test principal des améliorations de classification des routes."""
    
    logger.info("=" * 80)
    logger.info("TEST: Améliorations Classification des Routes")
    logger.info("=" * 80)
    
    # Créer données de test
    labels, features = create_test_data()
    
    logger.info(f"\n📊 Données de test créées:")
    logger.info(f"  - Total points: {len(labels)}")
    logger.info(f"  - Routes valides (0-200): {200} points")
    logger.info(f"  - Végétation (200-400): {200} points (dont 50 préclassifiés)")
    logger.info(f"  - Bâtiments (400-600): {200} points (dont 50 préclassifiés)")
    logger.info(f"  - Cas limites (600-800): {200} points")
    logger.info(f"  - Hors ground truth (800-1000): {200} points")
    
    # Configuration avec nouveaux seuils
    config = RefinementConfig()
    
    logger.info(f"\n⚙️ Seuils de filtrage:")
    logger.info(f"  - ROAD_HEIGHT_MAX: {config.ROAD_HEIGHT_MAX}m")
    logger.info(f"  - ROAD_PLANARITY_MIN: {config.ROAD_PLANARITY_MIN}")
    logger.info(f"  - ROAD_NDVI_MAX: {config.ROAD_NDVI_MAX}")
    logger.info(f"  - ROAD_CURVATURE_MAX: {config.ROAD_CURVATURE_MAX}")
    logger.info(f"  - ROAD_VERTICALITY_MAX: {config.ROAD_VERTICALITY_MAX}")
    
    # Appliquer le raffinement
    logger.info(f"\n🔧 Application du raffinement...")
    
    refined_labels, num_changed = refine_road_classification(
        labels=labels,
        points=features['points'],
        height=features['height'],
        planarity=features['planarity'],
        roughness=features['roughness'],
        intensity=features['intensity'],
        ground_truth_road_mask=features['ground_truth_road_mask'],
        ndvi=features['ndvi'],
        verticality=features['verticality'],
        curvature=features['curvature'],
        mode='lod2',
        config=config
    )
    
    # Analyser les résultats
    logger.info(f"\n📈 Résultats:")
    logger.info(f"  - Points modifiés: {num_changed}")
    
    # Vérifier chaque scénario
    logger.info(f"\n🔍 Analyse par scénario:")
    
    # Scénario 1: Routes valides (devraient être conservées)
    scenario1_kept = np.sum(refined_labels[0:200] == 9)
    logger.info(f"  1️⃣ Routes valides: {scenario1_kept}/200 conservées ({scenario1_kept/2:.1f}%)")
    if scenario1_kept < 180:
        logger.warning(f"     ⚠️  Trop de routes valides rejetées!")
    
    # Scénario 2: Végétation (devrait être exclue ou protégée)
    scenario2_excluded = np.sum(refined_labels[200:400] != 9)
    scenario2_protected = np.sum(refined_labels[250:300] == 11)
    logger.info(f"  2️⃣ Végétation: {scenario2_excluded}/200 exclues ({scenario2_excluded/2:.1f}%)")
    logger.info(f"     - Protégées (préclassifiées): {scenario2_protected}/50")
    if scenario2_excluded < 100:
        logger.warning(f"     ⚠️  Pas assez de végétation filtrée!")
    
    # Scénario 3: Bâtiments (devraient être exclus ou protégés)
    scenario3_excluded = np.sum(refined_labels[400:600] != 9)
    scenario3_protected = np.sum(refined_labels[450:500] == 0)
    logger.info(f"  3️⃣ Bâtiments: {scenario3_excluded}/200 exclus ({scenario3_excluded/2:.1f}%)")
    logger.info(f"     - Protégés (préclassifiés): {scenario3_protected}/50")
    if scenario3_excluded < 100:
        logger.warning(f"     ⚠️  Pas assez de bâtiments filtrés!")
    
    # Scénario 4: Cas limites (résultats variables attendus)
    scenario4_kept = np.sum(refined_labels[600:800] == 9)
    logger.info(f"  4️⃣ Cas limites: {scenario4_kept}/200 conservées ({scenario4_kept/2:.1f}%)")
    logger.info(f"     - Résultats variables attendus (seuils limites)")
    
    # Scénario 5: Hors ground truth (ne devraient pas être affectés)
    scenario5_unchanged = np.sum(refined_labels[800:1000] == labels[800:1000])
    logger.info(f"  5️⃣ Hors ground truth: {scenario5_unchanged}/200 inchangées ({scenario5_unchanged/2:.1f}%)")
    
    # Statistiques globales
    logger.info(f"\n📊 Statistiques globales:")
    total_roads = np.sum(refined_labels == 9)
    total_vegetation = np.sum(refined_labels == 11)
    total_buildings = np.sum(refined_labels == 0)
    total_other = len(labels) - total_roads - total_vegetation - total_buildings
    
    logger.info(f"  - Routes (classe 9): {total_roads} points ({total_roads/10:.1f}%)")
    logger.info(f"  - Végétation (classe 11): {total_vegetation} points ({total_vegetation/10:.1f}%)")
    logger.info(f"  - Bâtiments (classe 0): {total_buildings} points ({total_buildings/10:.1f}%)")
    logger.info(f"  - Autres: {total_other} points ({total_other/10:.1f}%)")
    
    # Conclusion
    logger.info(f"\n✅ RÉSUMÉ:")
    success = True
    
    if scenario1_kept >= 180:
        logger.info(f"  ✅ Routes valides bien conservées")
    else:
        logger.warning(f"  ❌ Trop de routes valides rejetées")
        success = False
    
    if scenario2_excluded >= 100:
        logger.info(f"  ✅ Végétation bien filtrée")
    else:
        logger.warning(f"  ❌ Filtrage végétation insuffisant")
        success = False
    
    if scenario3_excluded >= 100:
        logger.info(f"  ✅ Bâtiments bien filtrés")
    else:
        logger.warning(f"  ❌ Filtrage bâtiments insuffisant")
        success = False
    
    if scenario2_protected == 50 and scenario3_protected == 50:
        logger.info(f"  ✅ Classifications préexistantes protégées")
    else:
        logger.warning(f"  ❌ Protection des classifications insuffisante")
        success = False
    
    if success:
        logger.info(f"\n🎉 Test RÉUSSI - Les améliorations fonctionnent correctement!")
    else:
        logger.warning(f"\n⚠️  Test ÉCHOUÉ - Ajustements nécessaires")
    
    logger.info("=" * 80)
    
    return success


if __name__ == "__main__":
    test_road_classification_improvements()
