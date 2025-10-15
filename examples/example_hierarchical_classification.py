"""
Exemple d'utilisation du système de classification hiérarchique amélioré

Ce script montre comment utiliser le nouveau système de classification multi-niveaux
avec optimisation automatique des seuils, validation et correction d'erreurs.

Fonctionnalités démontrées:
1. Classification hiérarchique (ASPRS -> LOD2 -> LOD3)
2. Utilisation de seuils optimisés et adaptatifs
3. Calcul de métriques de confiance
4. Validation de la qualité
5. Correction automatique des erreurs
6. Génération de rapports de qualité

Auteur: IGN LiDAR HD Dataset Team
Date: 15 octobre 2025
"""

from pathlib import Path
import logging
import numpy as np
import laspy

# Imports du système de classification amélioré
from ign_lidar.core.modules.hierarchical_classifier import (
    classify_hierarchical,
    ClassificationLevel,
    HierarchicalClassifier
)
from ign_lidar.core.modules.optimized_thresholds import (
    ClassificationThresholds,
    ClassificationRules
)
from ign_lidar.core.modules.classification_validation import (
    validate_classification,
    auto_correct_classification,
    ClassificationValidator
)

# Imports pour les features
from ign_lidar.features.geometric import compute_geometric_features
from ign_lidar.preprocessing.rgb_augmentation import IGNOrthophotoFetcher
from ign_lidar.preprocessing.infrared_augmentation import IGNInfraredFetcher
from ign_lidar.core.modules.enrichment import compute_ndvi
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_basic_hierarchical_classification():
    """
    Exemple 1: Classification hiérarchique basique
    
    Démontre la classification d'un fichier LAZ du niveau ASPRS vers LOD2
    avec calcul automatique des scores de confiance.
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 1: Classification Hiérarchique Basique")
    print("=" * 80 + "\n")
    
    # Charger un fichier LAZ
    input_file = Path("data/test_integration/sample.laz")
    
    if not input_file.exists():
        print(f"⚠️  Fichier d'exemple non trouvé: {input_file}")
        print("   Créez un fichier LAZ de test ou modifiez le chemin")
        return
    
    logger.info(f"📂 Chargement: {input_file}")
    las = laspy.read(str(input_file))
    
    # Extraire les labels ASPRS existants
    asprs_labels = np.array(las.classification)
    points = np.vstack([las.x, las.y, las.z]).T
    
    logger.info(f"   {len(points):,} points chargés")
    logger.info(f"   Classes ASPRS uniques: {np.unique(asprs_labels)}")
    
    # Classification hiérarchique vers LOD2
    logger.info("🔄 Classification ASPRS -> LOD2...")
    result = classify_hierarchical(
        asprs_labels=asprs_labels,
        target_level='LOD2',
        use_confidence=True,
        track_hierarchy=True
    )
    
    # Afficher les résultats
    stats = result.get_statistics()
    print("\n📊 Statistiques de classification:")
    print(f"   Total points: {stats['total_points']:,}")
    print(f"   Nombre de classes: {stats['num_classes']}")
    print(f"   Confiance moyenne: {stats.get('avg_confidence', 0):.2%}")
    print(f"   Points à faible confiance: {stats.get('low_confidence_points', 0):,}")
    
    if result.hierarchy_path:
        print("\n🔗 Chemin de classification:")
        for step in result.hierarchy_path:
            print(f"   → {step}")
    
    print("\n📈 Distribution des classes:")
    for class_id, percentage in sorted(stats['class_percentages'].items()):
        count = stats['class_distribution'][class_id]
        print(f"   Classe {class_id:2d}: {count:8,} points ({percentage:5.1f}%)")
    
    # Sauvegarder les résultats
    output_file = input_file.parent / f"{input_file.stem}_lod2.laz"
    las.classification = result.labels
    las.write(str(output_file))
    logger.info(f"💾 Sauvegardé: {output_file}")


def example_advanced_with_features():
    """
    Exemple 2: Classification avancée avec features géométriques et NDVI
    
    Démontre l'utilisation de features additionnelles pour raffiner la
    classification et améliorer la précision.
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 2: Classification Avancée avec Features")
    print("=" * 80 + "\n")
    
    # Charger fichier LAZ
    input_file = Path("data/test_integration/sample.laz")
    
    if not input_file.exists():
        print(f"⚠️  Fichier d'exemple non trouvé: {input_file}")
        return
    
    logger.info(f"📂 Chargement: {input_file}")
    las = laspy.read(str(input_file))
    
    asprs_labels = np.array(las.classification)
    points = np.vstack([las.x, las.y, las.z]).T
    
    # Calculer hauteur au-dessus du sol
    logger.info("📏 Calcul des hauteurs...")
    ground_mask = asprs_labels == 2
    if np.any(ground_mask):
        from scipy.spatial import cKDTree
        ground_points = points[ground_mask]
        tree = cKDTree(ground_points)
        _, nearest_idx = tree.query(points, k=1)
        ground_z = ground_points[nearest_idx, 2]
        height = points[:, 2] - ground_z
    else:
        height = points[:, 2] - points[:, 2].min()
    
    logger.info(f"   Hauteur min: {height.min():.1f}m, max: {height.max():.1f}m")
    
    # Calculer features géométriques
    logger.info("🔧 Calcul des features géométriques...")
    geom_features = compute_geometric_features(
        points=points,
        k_neighbors=20,
        compute_normals=True,
        compute_planarity=True,
        compute_curvature=True
    )
    
    logger.info(f"   ✓ Normales, planéité, courbure calculées")
    
    # Calculer NDVI si RGB et NIR disponibles
    ndvi = None
    if hasattr(las, 'red') and hasattr(las, 'nir'):
        logger.info("🌿 Calcul NDVI...")
        red = np.array(las.red, dtype=float)
        nir = np.array(las.nir, dtype=float)
        ndvi = compute_ndvi(red, nir)
        logger.info(f"   NDVI min: {ndvi.min():.2f}, max: {ndvi.max():.2f}")
    else:
        logger.warning("   ⚠️  RGB/NIR non disponible, NDVI non calculé")
    
    # Préparer le dictionnaire de features
    features = {
        'height': height,
        'normals': geom_features.get('normals'),
        'planarity': geom_features.get('planarity'),
        'curvature': geom_features.get('curvature'),
    }
    
    if ndvi is not None:
        features['ndvi'] = ndvi
    
    # Classification avec features
    logger.info("🔄 Classification avec raffinement par features...")
    result = classify_hierarchical(
        asprs_labels=asprs_labels,
        target_level='LOD2',
        features=features,
        use_confidence=True,
        track_hierarchy=True
    )
    
    # Afficher les résultats
    stats = result.get_statistics()
    print("\n📊 Résultats de classification raffinée:")
    print(f"   Points raffinés: {stats['num_refined']:,}")
    print(f"   Confiance moyenne: {stats.get('avg_confidence', 0):.2%}")
    
    if result.feature_importance:
        print("\n🎯 Importance des features:")
        for feature, importance in sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"   {feature:15s}: {importance:.2%}")
    
    # Sauvegarder
    output_file = input_file.parent / f"{input_file.stem}_lod2_advanced.laz"
    las.classification = result.labels
    
    # Ajouter scores de confiance comme extra dimension
    if result.confidence_scores is not None:
        confidence_scaled = (result.confidence_scores * 255).astype(np.uint8)
        # Note: Nécessite d'ajouter une dimension custom au fichier LAS
        # las.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.uint8))
        # las.confidence = confidence_scaled
    
    las.write(str(output_file))
    logger.info(f"💾 Sauvegardé: {output_file}")


def example_adaptive_thresholds():
    """
    Exemple 3: Seuils adaptatifs selon le contexte
    
    Démontre l'utilisation de seuils adaptés au contexte urbain/rural
    et à la saison.
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 3: Seuils Adaptatifs Selon le Contexte")
    print("=" * 80 + "\n")
    
    # Créer configuration de seuils par défaut
    thresholds = ClassificationThresholds()
    
    print("🔧 Seuils par défaut:")
    print(f"   NDVI végétation min: {thresholds.ndvi.vegetation_min:.2f}")
    print(f"   Hauteur sol max: {thresholds.height.ground_max:.2f}m")
    print(f"   Planéité route min: {thresholds.geometric.planarity_road_min:.2f}")
    
    # Adapter pour contexte urbain + été
    print("\n🏙️  Adaptation pour contexte urbain, saison été:")
    urban_summer = thresholds.get_adaptive_thresholds(
        season='summer',
        context_type='urban',
        terrain_type='flat'
    )
    
    print(f"   NDVI végétation min: {urban_summer.ndvi.vegetation_min:.2f}")
    print(f"   Hauteur sol max: {urban_summer.height.ground_max:.2f}m")
    print(f"   Planéité route min: {urban_summer.geometric.planarity_road_min:.2f}")
    
    # Adapter pour contexte rural + hiver
    print("\n🌲 Adaptation pour contexte rural, saison hiver:")
    rural_winter = thresholds.get_adaptive_thresholds(
        season='winter',
        context_type='rural',
        terrain_type='mountainous'
    )
    
    print(f"   NDVI végétation min: {rural_winter.ndvi.vegetation_min:.2f}")
    print(f"   Hauteur sol max: {rural_winter.height.ground_max:.2f}m")
    print(f"   Planéité route min: {rural_winter.geometric.planarity_road_min:.2f}")
    
    # Valider les seuils
    is_valid, warnings = thresholds.validate_thresholds()
    print(f"\n✅ Seuils valides: {is_valid}")
    if warnings:
        print("⚠️  Avertissements:")
        for warning in warnings:
            print(f"   - {warning}")


def example_validation_and_correction():
    """
    Exemple 4: Validation et correction automatique
    
    Démontre la validation de la qualité de classification et la
    correction automatique des erreurs courantes.
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 4: Validation et Correction Automatique")
    print("=" * 80 + "\n")
    
    # Charger fichier LAZ
    input_file = Path("data/test_integration/sample.laz")
    
    if not input_file.exists():
        print(f"⚠️  Fichier d'exemple non trouvé: {input_file}")
        return
    
    logger.info(f"📂 Chargement: {input_file}")
    las = laspy.read(str(input_file))
    
    asprs_labels = np.array(las.classification)
    points = np.vstack([las.x, las.y, las.z]).T
    
    # Classification avec features
    # (Code simplifié - voir exemple 2 pour version complète)
    height = points[:, 2] - points[:, 2].min()
    
    features = {
        'height': height,
        'points': points
    }
    
    # Classification initiale
    logger.info("🔄 Classification initiale...")
    result = classify_hierarchical(
        asprs_labels=asprs_labels,
        target_level='LOD2',
        features=features,
        use_confidence=True
    )
    
    # Détection d'erreurs
    logger.info("🔍 Détection des erreurs potentielles...")
    validator = ClassificationValidator()
    errors = validator.detect_errors(
        labels=result.labels,
        features=features,
        confidence_scores=result.confidence_scores
    )
    
    print("\n⚠️  Erreurs potentielles détectées:")
    for error_type, error_mask in errors.items():
        count = np.sum(error_mask)
        percentage = count / len(result.labels) * 100
        print(f"   {error_type:20s}: {count:8,} points ({percentage:5.1f}%)")
    
    # Correction automatique
    logger.info("🔧 Correction automatique des erreurs...")
    corrected_labels, correction_counts = auto_correct_classification(
        labels=result.labels,
        features=features,
        confidence_scores=result.confidence_scores,
        confidence_threshold=0.5
    )
    
    print("\n✅ Corrections appliquées:")
    for correction_type, count in correction_counts.items():
        print(f"   {correction_type:15s}: {count:8,} corrections")
    
    # Sauvegarder version corrigée
    output_file = input_file.parent / f"{input_file.stem}_lod2_corrected.laz"
    las.classification = corrected_labels
    las.write(str(output_file))
    logger.info(f"💾 Sauvegardé: {output_file}")


def example_complete_workflow():
    """
    Exemple 5: Workflow complet de classification
    
    Pipeline complet incluant:
    1. Chargement des données
    2. Calcul de toutes les features
    3. Classification hiérarchique
    4. Validation de la qualité
    5. Correction automatique
    6. Génération de rapport
    7. Sauvegarde des résultats
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 5: Workflow Complet de Classification")
    print("=" * 80 + "\n")
    
    # Configuration
    input_file = Path("data/test_integration/sample.laz")
    output_dir = Path("data/test_output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not input_file.exists():
        print(f"⚠️  Fichier d'exemple non trouvé: {input_file}")
        return
    
    # Étape 1: Chargement
    logger.info("=" * 60)
    logger.info("ÉTAPE 1: Chargement des données")
    logger.info("=" * 60)
    
    las = laspy.read(str(input_file))
    asprs_labels = np.array(las.classification)
    points = np.vstack([las.x, las.y, las.z]).T
    
    logger.info(f"✓ {len(points):,} points chargés")
    
    # Étape 2: Calcul des features
    logger.info("\n" + "=" * 60)
    logger.info("ÉTAPE 2: Calcul des features")
    logger.info("=" * 60)
    
    # Hauteur
    ground_mask = asprs_labels == 2
    if np.any(ground_mask):
        from scipy.spatial import cKDTree
        ground_points = points[ground_mask]
        tree = cKDTree(ground_points)
        _, nearest_idx = tree.query(points, k=1)
        height = points[:, 2] - ground_points[nearest_idx, 2]
    else:
        height = points[:, 2] - points[:, 2].min()
    
    logger.info(f"✓ Hauteurs calculées (range: {height.min():.1f} - {height.max():.1f}m)")
    
    # Géométrie
    geom_features = compute_geometric_features(
        points=points,
        k_neighbors=20,
        compute_normals=True,
        compute_planarity=True,
        compute_curvature=True
    )
    logger.info("✓ Features géométriques calculées")
    
    features = {
        'height': height,
        'points': points,
        'normals': geom_features.get('normals'),
        'planarity': geom_features.get('planarity'),
        'curvature': geom_features.get('curvature'),
    }
    
    # Étape 3: Classification
    logger.info("\n" + "=" * 60)
    logger.info("ÉTAPE 3: Classification hiérarchique")
    logger.info("=" * 60)
    
    result = classify_hierarchical(
        asprs_labels=asprs_labels,
        target_level='LOD2',
        features=features,
        use_confidence=True,
        track_hierarchy=True
    )
    
    stats = result.get_statistics()
    logger.info(f"✓ Classification terminée:")
    logger.info(f"  - {stats['num_classes']} classes détectées")
    logger.info(f"  - {stats['num_refined']:,} points raffinés")
    logger.info(f"  - Confiance moyenne: {stats.get('avg_confidence', 0):.2%}")
    
    # Étape 4: Validation
    logger.info("\n" + "=" * 60)
    logger.info("ÉTAPE 4: Validation de la qualité")
    logger.info("=" * 60)
    
    validator = ClassificationValidator()
    errors = validator.detect_errors(
        labels=result.labels,
        features=features,
        confidence_scores=result.confidence_scores
    )
    
    total_errors = sum(np.sum(mask) for mask in errors.values())
    logger.info(f"✓ {total_errors:,} erreurs potentielles détectées")
    
    # Étape 5: Correction
    logger.info("\n" + "=" * 60)
    logger.info("ÉTAPE 5: Correction automatique")
    logger.info("=" * 60)
    
    corrected_labels, correction_counts = auto_correct_classification(
        labels=result.labels,
        features=features,
        confidence_scores=result.confidence_scores
    )
    
    total_corrections = sum(correction_counts.values())
    logger.info(f"✓ {total_corrections:,} corrections appliquées")
    
    # Étape 6: Rapport
    logger.info("\n" + "=" * 60)
    logger.info("ÉTAPE 6: Génération du rapport")
    logger.info("=" * 60)
    
    report_file = output_dir / f"{input_file.stem}_classification_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE CLASSIFICATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Fichier source: {input_file}\n")
        f.write(f"Date: {stats['total_points']:,} points\n\n")
        
        f.write("STATISTIQUES GLOBALES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Nombre de classes: {stats['num_classes']}\n")
        f.write(f"Points raffinés: {stats['num_refined']:,}\n")
        f.write(f"Confiance moyenne: {stats.get('avg_confidence', 0):.2%}\n\n")
        
        f.write("DISTRIBUTION DES CLASSES\n")
        f.write("-" * 80 + "\n")
        for class_id, count in sorted(stats['class_distribution'].items()):
            percentage = stats['class_percentages'][class_id]
            f.write(f"Classe {class_id:2d}: {count:10,} points ({percentage:6.2f}%)\n")
        
        f.write("\n")
        f.write("CORRECTIONS APPLIQUÉES\n")
        f.write("-" * 80 + "\n")
        for corr_type, count in correction_counts.items():
            f.write(f"{corr_type:15s}: {count:10,} corrections\n")
    
    logger.info(f"✓ Rapport sauvegardé: {report_file}")
    
    # Étape 7: Sauvegarde
    logger.info("\n" + "=" * 60)
    logger.info("ÉTAPE 7: Sauvegarde des résultats")
    logger.info("=" * 60)
    
    output_file = output_dir / f"{input_file.stem}_lod2_final.laz"
    las.classification = corrected_labels
    las.write(str(output_file))
    
    logger.info(f"✓ Classification sauvegardée: {output_file}")
    logger.info("\n🎉 Workflow complet terminé avec succès!")


def main():
    """Point d'entrée principal."""
    print("\n" + "=" * 80)
    print("EXEMPLES DE CLASSIFICATION HIÉRARCHIQUE AMÉLIORÉE")
    print("IGN LiDAR HD Dataset - Système Multi-Niveaux")
    print("=" * 80)
    
    examples = {
        '1': ('Classification hiérarchique basique', example_basic_hierarchical_classification),
        '2': ('Classification avancée avec features', example_advanced_with_features),
        '3': ('Seuils adaptatifs contextuels', example_adaptive_thresholds),
        '4': ('Validation et correction automatique', example_validation_and_correction),
        '5': ('Workflow complet', example_complete_workflow),
        'all': ('Tous les exemples', None),
    }
    
    print("\nExemples disponibles:")
    for key, (name, _) in examples.items():
        if key != 'all':
            print(f"  {key}: {name}")
    print(f"  all: Exécuter tous les exemples")
    
    choice = input("\nChoisissez un exemple (1-5 ou 'all', ou 'q' pour quitter): ").strip().lower()
    
    if choice == 'q':
        print("Au revoir!")
        return
    
    if choice == 'all':
        for key in ['1', '2', '3', '4', '5']:
            _, func = examples[key]
            try:
                func()
            except Exception as e:
                logger.error(f"Erreur dans l'exemple {key}: {e}", exc_info=True)
    elif choice in examples and choice != 'all':
        _, func = examples[choice]
        try:
            func()
        except Exception as e:
            logger.error(f"Erreur: {e}", exc_info=True)
    else:
        print(f"Choix invalide: {choice}")
        return
    
    print("\n✅ Terminé!")


if __name__ == "__main__":
    main()
