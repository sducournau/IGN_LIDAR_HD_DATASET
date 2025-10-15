"""
Exemple d'utilisation de la Grammaire 3D pour Classification de Bâtiments

Ce script montre comment utiliser le système de grammaire 3D pour améliorer
la classification des bâtiments et détecter automatiquement leurs sous-éléments
(murs, toits, fenêtres, portes, cheminées, etc.).

La grammaire 3D décompose hiérarchiquement les structures:
  Niveau 0: Détection de bâtiment
  Niveau 1: Composants majeurs (fondation, murs, toit)
  Niveau 2: Raffinement (segments de murs, plans de toit)
  Niveau 3: Éléments détaillés (fenêtres, portes, cheminées, lucarnes)

Auteur: IGN LiDAR HD Dataset Team
Date: 15 octobre 2025
"""

from pathlib import Path
import logging
import numpy as np
import laspy

# Import du système de grammaire 3D
from ign_lidar.core.modules.grammar_3d import (
    classify_with_grammar,
    visualize_derivation_tree,
    BuildingGrammar,
    GrammarParser
)

# Import du système de classification hiérarchique
from ign_lidar.core.modules.hierarchical_classifier import classify_hierarchical

# Import des features
from ign_lidar.features.geometric import compute_geometric_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_grammar_basic():
    """
    Exemple 1: Utilisation basique de la grammaire 3D
    
    Applique les règles de grammaire pour décomposer un bâtiment
    en ses composants principaux.
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 1: Grammaire 3D - Décomposition de Bâtiment")
    print("=" * 80 + "\n")
    
    # Charger fichier LAZ
    input_file = Path("data/test_integration/sample.laz")
    
    if not input_file.exists():
        print(f"⚠️  Fichier non trouvé: {input_file}")
        print("   Veuillez fournir un fichier LAZ de test")
        return
    
    logger.info(f"📂 Chargement: {input_file}")
    las = laspy.read(str(input_file))
    
    points = np.vstack([las.x, las.y, las.z]).T
    asprs_labels = np.array(las.classification)
    
    logger.info(f"   {len(points):,} points chargés")
    
    # Calculer features géométriques
    logger.info("🔧 Calcul des features géométriques...")
    
    # Hauteur
    ground_mask = asprs_labels == 2
    if np.any(ground_mask):
        height = points[:, 2] - points[ground_mask, 2].min()
    else:
        height = points[:, 2] - points[:, 2].min()
    
    # Features géométriques
    geom_features = compute_geometric_features(
        points=points,
        k_neighbors=20,
        compute_normals=True,
        compute_planarity=True,
        compute_curvature=True
    )
    
    features = {
        'height': height,
        'normals': geom_features.get('normals'),
        'planarity': geom_features.get('planarity'),
        'curvature': geom_features.get('curvature'),
    }
    
    logger.info("   ✓ Features calculées")
    
    # Appliquer grammaire 3D
    logger.info("🏗️  Application de la grammaire 3D...")
    
    refined_labels, derivation_tree = classify_with_grammar(
        points=points,
        labels=asprs_labels,
        features=features,
        max_iterations=10,
        min_confidence=0.5
    )
    
    # Afficher résultats
    n_refined = np.sum(refined_labels != asprs_labels)
    logger.info(f"✓ {n_refined:,} points raffinés par la grammaire")
    
    # Visualiser l'arbre de dérivation
    print("\n" + "=" * 80)
    print("ARBRE DE DÉRIVATION:")
    print("=" * 80)
    
    tree_viz = visualize_derivation_tree(derivation_tree)
    print(tree_viz)
    
    # Sauvegarder
    output_file = input_file.parent / f"{input_file.stem}_grammar.laz"
    las.classification = refined_labels
    las.write(str(output_file))
    
    logger.info(f"💾 Résultats sauvegardés: {output_file}")
    
    # Sauvegarder arbre de dérivation
    tree_file = input_file.parent / f"{input_file.stem}_derivation.txt"
    with open(tree_file, 'w') as f:
        f.write(tree_viz)
    logger.info(f"📄 Arbre de dérivation sauvegardé: {tree_file}")


def example_grammar_with_hierarchical():
    """
    Exemple 2: Grammaire 3D + Classification Hiérarchique
    
    Combine la classification hiérarchique (ASPRS -> LOD2) avec
    la grammaire 3D pour obtenir le meilleur des deux approches.
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 2: Grammaire 3D + Classification Hiérarchique")
    print("=" * 80 + "\n")
    
    input_file = Path("data/test_integration/sample.laz")
    
    if not input_file.exists():
        print(f"⚠️  Fichier non trouvé: {input_file}")
        return
    
    logger.info(f"📂 Chargement: {input_file}")
    las = laspy.read(str(input_file))
    
    points = np.vstack([las.x, las.y, las.z]).T
    asprs_labels = np.array(las.classification)
    
    # Calculer features
    logger.info("🔧 Calcul des features...")
    ground_mask = asprs_labels == 2
    if np.any(ground_mask):
        height = points[:, 2] - points[ground_mask, 2].min()
    else:
        height = points[:, 2] - points[:, 2].min()
    
    geom_features = compute_geometric_features(
        points=points,
        k_neighbors=20,
        compute_normals=True,
        compute_planarity=True,
        compute_curvature=True
    )
    
    features = {
        'height': height,
        'normals': geom_features.get('normals'),
        'planarity': geom_features.get('planarity'),
        'curvature': geom_features.get('curvature'),
    }
    
    # Étape 1: Classification hiérarchique ASPRS -> LOD2
    logger.info("\n📊 Étape 1: Classification hiérarchique (ASPRS -> LOD2)...")
    
    hierarchical_result = classify_hierarchical(
        asprs_labels=asprs_labels,
        target_level='LOD2',
        features=features,
        use_confidence=True
    )
    
    lod2_labels = hierarchical_result.labels
    confidence = hierarchical_result.confidence_scores
    
    stats = hierarchical_result.get_statistics()
    logger.info(f"   ✓ Classification LOD2: {stats['num_classes']} classes")
    logger.info(f"   Confiance moyenne: {stats.get('avg_confidence', 0):.2%}")
    
    # Étape 2: Raffiner avec grammaire 3D (spécialement pour bâtiments)
    logger.info("\n🏗️  Étape 2: Raffinement avec grammaire 3D...")
    
    refined_labels, derivation_tree = classify_with_grammar(
        points=points,
        labels=lod2_labels,
        features=features,
        max_iterations=10,
        min_confidence=0.5
    )
    
    # Comparer les résultats
    n_changed_hierarchical = np.sum(lod2_labels != asprs_labels)
    n_changed_grammar = np.sum(refined_labels != lod2_labels)
    n_changed_total = np.sum(refined_labels != asprs_labels)
    
    print("\n" + "=" * 80)
    print("STATISTIQUES DE RAFFINEMENT:")
    print("=" * 80)
    print(f"Points modifiés par classification hiérarchique: {n_changed_hierarchical:,}")
    print(f"Points modifiés par grammaire 3D: {n_changed_grammar:,}")
    print(f"Total points modifiés: {n_changed_total:,}")
    print(f"Pourcentage raffiné: {n_changed_total / len(points) * 100:.1f}%")
    
    # Afficher arbre de dérivation
    if derivation_tree:
        print("\n" + "=" * 80)
        print("ARBRE DE DÉRIVATION (premiers bâtiments):")
        print("=" * 80)
        tree_viz = visualize_derivation_tree(derivation_tree)
        print(tree_viz)
    
    # Sauvegarder résultats
    output_file = input_file.parent / f"{input_file.stem}_hierarchical_grammar.laz"
    las.classification = refined_labels
    
    # Ajouter confiance comme extra dimension (si supporté)
    try:
        # Tenter d'ajouter la confiance
        confidence_scaled = (confidence * 255).astype(np.uint8)
        # Note: Nécessite laspy avec support des extra dimensions
        # las.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.uint8))
        # las.confidence = confidence_scaled
    except:
        pass
    
    las.write(str(output_file))
    logger.info(f"💾 Résultats sauvegardés: {output_file}")


def example_grammar_rules_exploration():
    """
    Exemple 3: Explorer les règles de grammaire disponibles
    
    Affiche toutes les règles de production de la grammaire et
    leurs conditions d'application.
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 3: Exploration des Règles de Grammaire 3D")
    print("=" * 80 + "\n")
    
    # Créer grammaire
    grammar = BuildingGrammar()
    
    print(f"Nombre total de règles: {len(grammar.rules)}")
    print()
    
    # Grouper par niveau
    rules_by_level = {}
    for rule in grammar.rules:
        # Déterminer le niveau
        if rule.priority >= 90:
            level = 0
        elif rule.priority >= 70:
            level = 1
        elif rule.priority >= 60:
            level = 2
        else:
            level = 3
        
        if level not in rules_by_level:
            rules_by_level[level] = []
        rules_by_level[level].append(rule)
    
    # Afficher par niveau
    level_names = {
        0: "Détection de Bâtiment",
        1: "Composants Majeurs",
        2: "Raffinement de Composants",
        3: "Éléments Détaillés"
    }
    
    for level in sorted(rules_by_level.keys()):
        print(f"\n{'=' * 80}")
        print(f"NIVEAU {level}: {level_names[level]}")
        print('=' * 80)
        
        for rule in rules_by_level[level]:
            print(f"\n📋 Règle: {rule.name}")
            print(f"   {rule}")
            print(f"   Priorité: {rule.priority}")
            
            if rule.conditions:
                print(f"   Conditions:")
                for key, value in rule.conditions.items():
                    print(f"      - {key}: {value}")


def example_custom_grammar():
    """
    Exemple 4: Créer une grammaire personnalisée
    
    Montre comment étendre le système de grammaire avec des règles
    personnalisées pour des types de bâtiments spécifiques.
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 4: Grammaire Personnalisée")
    print("=" * 80 + "\n")
    
    from ign_lidar.core.modules.grammar_3d import (
        BuildingGrammar,
        ProductionRule,
        ArchitecturalSymbol
    )
    
    # Créer grammaire de base
    grammar = BuildingGrammar()
    
    print("Grammaire de base créée avec {len(grammar.rules)} règles")
    
    # Ajouter règles personnalisées pour architecture française
    
    # Règle pour toits Mansart (typique de l'architecture française)
    mansard_rule = ProductionRule(
        name="detect_mansard_roof_french",
        left_hand_side=ArchitecturalSymbol.ROOF,
        right_hand_side=[ArchitecturalSymbol.ROOF_MANSARD],
        conditions={
            'has_two_slopes': True,
            'steep_lower_slope': True,
            'lower_slope_angle_range': (60, 80),  # Degrés
            'upper_slope_angle_range': (20, 40),
            'characteristic': 'french_classical'
        },
        priority=66
    )
    
    grammar.rules.append(mansard_rule)
    
    # Règle pour balcons français
    french_balcony_rule = ProductionRule(
        name="detect_french_balcony",
        left_hand_side=ArchitecturalSymbol.WALL_SEGMENT,
        right_hand_side=[
            ArchitecturalSymbol.BALCONY,
            ArchitecturalSymbol.BALUSTRADE
        ],
        conditions={
            'has_railing': True,
            'railing_height_range': (0.9, 1.2),
            'floor_level': True,
            'decorative_elements': True
        },
        priority=56
    )
    
    grammar.rules.append(french_balcony_rule)
    
    # Règle pour lucarnes à fronton (typique parisien)
    dormer_pediment_rule = ProductionRule(
        name="detect_dormer_with_pediment",
        left_hand_side=ArchitecturalSymbol.ROOF,
        right_hand_side=[ArchitecturalSymbol.DORMER],
        conditions={
            'has_pediment': True,
            'triangular_top': True,
            'has_window': True,
            'style': 'parisian'
        },
        priority=56
    )
    
    grammar.rules.append(dormer_pediment_rule)
    
    print(f"\n✅ Grammaire étendue: {len(grammar.rules)} règles")
    print("\nNouvelles règles ajoutées:")
    print(f"  1. {mansard_rule.name}: {mansard_rule}")
    print(f"  2. {french_balcony_rule.name}: {french_balcony_rule}")
    print(f"  3. {dormer_pediment_rule.name}: {dormer_pediment_rule}")
    
    print("\n💡 Cette grammaire personnalisée peut maintenant être utilisée")
    print("   pour classifier des bâtiments avec des caractéristiques")
    print("   architecturales françaises spécifiques!")


def example_grammar_statistics():
    """
    Exemple 5: Statistiques et analyse de la grammaire
    
    Analyse la structure de la grammaire et génère des statistiques.
    """
    print("\n" + "=" * 80)
    print("EXEMPLE 5: Statistiques de la Grammaire 3D")
    print("=" * 80 + "\n")
    
    grammar = BuildingGrammar()
    
    # Statistiques globales
    print("📊 STATISTIQUES GLOBALES:")
    print(f"   Nombre total de règles: {len(grammar.rules)}")
    
    # Compter par symbole de gauche
    lhs_counts = {}
    for rule in grammar.rules:
        symbol = rule.left_hand_side.value
        lhs_counts[symbol] = lhs_counts.get(symbol, 0) + 1
    
    print(f"\n📋 Règles par symbole source (LHS):")
    for symbol, count in sorted(lhs_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {symbol:25s}: {count:2d} règles")
    
    # Compter symboles produits
    rhs_counts = {}
    for rule in grammar.rules:
        for symbol in rule.right_hand_side:
            symbol_name = symbol.value
            rhs_counts[symbol_name] = rhs_counts.get(symbol_name, 0) + 1
    
    print(f"\n🎯 Symboles produits (RHS):")
    for symbol, count in sorted(rhs_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {symbol:25s}: {count:2d} occurrences")
    
    # Profondeur maximale de dérivation
    print(f"\n🌳 Structure hiérarchique:")
    print(f"   Niveaux de dérivation: 4 (0-3)")
    print(f"   Profondeur maximale théorique: illimitée")
    print(f"   Profondeur pratique limitée: 10 itérations")
    
    # Symboles terminaux vs non-terminaux
    terminal_symbols = set()
    non_terminal_symbols = set()
    
    for rule in grammar.rules:
        non_terminal_symbols.add(rule.left_hand_side.value)
        for symbol in rule.right_hand_side:
            # Si un symbole n'apparaît jamais en LHS, c'est un terminal
            is_terminal = not any(
                r.left_hand_side == symbol for r in grammar.rules
            )
            if is_terminal:
                terminal_symbols.add(symbol.value)
    
    print(f"\n🔤 Vocabulaire:")
    print(f"   Symboles non-terminaux: {len(non_terminal_symbols)}")
    print(f"   Symboles terminaux: {len(terminal_symbols)}")
    
    if terminal_symbols:
        print(f"\n   Terminaux: {', '.join(sorted(terminal_symbols)[:5])}...")


def main():
    """Point d'entrée principal."""
    print("\n" + "=" * 80)
    print("EXEMPLES DE GRAMMAIRE 3D POUR CLASSIFICATION DE BÂTIMENTS")
    print("IGN LiDAR HD Dataset - Shape Grammar System")
    print("=" * 80)
    
    examples = {
        '1': ('Grammaire 3D basique', example_grammar_basic),
        '2': ('Grammaire + Classification hiérarchique', example_grammar_with_hierarchical),
        '3': ('Explorer les règles de grammaire', example_grammar_rules_exploration),
        '4': ('Créer une grammaire personnalisée', example_custom_grammar),
        '5': ('Statistiques de grammaire', example_grammar_statistics),
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
