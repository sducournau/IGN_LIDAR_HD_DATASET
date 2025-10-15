# ⚡ Guide de Démarrage Rapide: Classification Multi-Niveau + Grammaire 3D

Guide de démarrage rapide pour utiliser le système complet de classification amélioré avec grammaire 3D.

## 🎯 Cas d'Usage

| Cas d'Usage                                   | Solution                    | Temps         | Niveau        |
| --------------------------------------------- | --------------------------- | ------------- | ------------- |
| Classification simple ASPRS→LOD2              | Classification Hiérarchique | ~1s/100K pts  | Débutant      |
| Classification avancée avec seuils adaptatifs | + Optimized Thresholds      | ~2s/100K pts  | Intermédiaire |
| Validation et correction automatique          | + Validation                | ~3s/100K pts  | Intermédiaire |
| Décomposition de bâtiments en éléments        | + Grammaire 3D              | ~5s/100K pts  | Avancé        |
| Pipeline complet production                   | Tout ci-dessus              | ~10s/100K pts | Expert        |

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/YOUR_ORG/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET

# Installer dépendances
pip install -r requirements.txt

# Optionnel: installer scipy pour cohérence spatiale
pip install scipy>=1.7.0
```

## 📝 Utilisation en 3 Lignes

### Option 1: Classification Hiérarchique Seule

```python
from ign_lidar.core.modules.hierarchical_classifier import classify_hierarchical

result = classify_hierarchical(asprs_labels, target_level='LOD2', features=features)
# Résultat: result.labels, result.confidence_scores
```

### Option 2: Grammaire 3D Seule

```python
from ign_lidar.core.modules.grammar_3d import classify_with_grammar

refined_labels, tree = classify_with_grammar(points, labels, features)
# Résultat: refined_labels, tree (arbre de dérivation)
```

### Option 3: Pipeline Complet (Recommandé)

```python
from ign_lidar.core.modules.hierarchical_classifier import classify_hierarchical
from ign_lidar.core.modules.grammar_3d import classify_with_grammar
from ign_lidar.core.modules.classification_validation import auto_correct_classification

# Étape 1: Classification hiérarchique
result = classify_hierarchical(asprs_labels, 'LOD2', features)

# Étape 2: Raffinement avec grammaire 3D
refined, tree = classify_with_grammar(points, result.labels, features)

# Étape 3: Correction automatique
final_labels, corrections = auto_correct_classification(refined, features, result.confidence_scores)
```

## 🔧 Exemples Complets

### Exemple 1: Traitement Fichier LAS

```python
import laspy
import numpy as np
from ign_lidar.core.modules.hierarchical_classifier import classify_hierarchical
from ign_lidar.core.modules.grammar_3d import classify_with_grammar

# Charger fichier LAS
las = laspy.read("building.las")
points = np.vstack([las.x, las.y, las.z]).T
asprs_labels = las.classification

# Calculer features
height = points[:, 2] - np.min(points[:, 2])
# ... calculer autres features (normals, planarity, etc.)

features = {
    'height': height,
    'normals': normals,
    'planarity': planarity
}

# Pipeline complet
result = classify_hierarchical(asprs_labels, 'LOD2', features)
refined, tree = classify_with_grammar(points, result.labels, features)

# Sauvegarder résultat
las.classification = refined.astype(np.uint8)
las.write("building_refined.las")

print(f"✅ Traitement terminé!")
print(f"   - Classification: {result.num_refined:,} points raffinés")
print(f"   - Grammaire: {np.sum(refined != result.labels):,} points raffinés")
```

### Exemple 2: Batch Processing (Multiples Fichiers)

```python
from pathlib import Path
import laspy
import numpy as np
from tqdm import tqdm
from ign_lidar.core.modules.hierarchical_classifier import classify_hierarchical
from ign_lidar.core.modules.grammar_3d import classify_with_grammar

def process_las_file(input_path, output_path):
    """Traite un fichier LAS avec pipeline complet."""

    # Charger
    las = laspy.read(input_path)
    points = np.vstack([las.x, las.y, las.z]).T
    asprs_labels = las.classification

    # Calculer features (simplifié)
    height = points[:, 2] - np.min(points[:, 2])
    features = {'height': height}

    # Pipeline
    result = classify_hierarchical(asprs_labels, 'LOD2', features)
    refined, tree = classify_with_grammar(points, result.labels, features)

    # Sauvegarder
    las.classification = refined.astype(np.uint8)
    las.write(output_path)

    return {
        'num_points': len(points),
        'num_refined': np.sum(refined != asprs_labels),
        'num_buildings': len(tree)
    }

# Traiter tous les fichiers
input_dir = Path("data/raw")
output_dir = Path("data/refined")
output_dir.mkdir(exist_ok=True, parents=True)

results = []
for las_file in tqdm(list(input_dir.glob("*.las"))):
    output_file = output_dir / las_file.name
    stats = process_las_file(las_file, output_file)
    results.append(stats)
    print(f"  ✓ {las_file.name}: {stats['num_refined']:,} raffinés, {stats['num_buildings']} bâtiments")

# Statistiques globales
total_points = sum(r['num_points'] for r in results)
total_refined = sum(r['num_refined'] for r in results)
print(f"\n📊 Statistiques globales:")
print(f"   - Fichiers traités: {len(results)}")
print(f"   - Points totaux: {total_points:,}")
print(f"   - Points raffinés: {total_refined:,} ({100*total_refined/total_points:.1f}%)")
```

### Exemple 3: Analyse Détaillée d'un Bâtiment

```python
import numpy as np
from ign_lidar.core.modules.grammar_3d import (
    classify_with_grammar,
    visualize_derivation_tree,
    ArchitecturalSymbol
)

# Traiter
refined_labels, tree = classify_with_grammar(points, labels, features)

# Visualiser arbre
tree_viz = visualize_derivation_tree(tree)
print(tree_viz)

# Analyser composants
def analyze_building(root_shape):
    """Analyse détaillée d'un bâtiment."""

    stats = {
        'total_points': len(root_shape.point_indices),
        'components': {}
    }

    def count_recursive(shape):
        symbol_name = shape.symbol.value
        if symbol_name not in stats['components']:
            stats['components'][symbol_name] = {
                'count': 0,
                'total_points': 0,
                'avg_confidence': []
            }

        stats['components'][symbol_name]['count'] += 1
        stats['components'][symbol_name]['total_points'] += len(shape.point_indices)
        stats['components'][symbol_name]['avg_confidence'].append(shape.confidence)

        for child in shape.children:
            count_recursive(child)

    count_recursive(root_shape)

    # Calculer moyennes
    for comp in stats['components'].values():
        comp['avg_confidence'] = np.mean(comp['avg_confidence'])

    return stats

# Analyser chaque bâtiment
for building_name, root_shape in tree.items():
    stats = analyze_building(root_shape)

    print(f"\n🏢 {building_name}")
    print(f"   Points totaux: {stats['total_points']:,}")
    print(f"   Composants détectés:")

    for comp_name, comp_stats in sorted(stats['components'].items()):
        print(f"     - {comp_name}: {comp_stats['count']}x "
              f"({comp_stats['total_points']:,} pts, "
              f"conf={comp_stats['avg_confidence']:.2f})")
```

## 📊 Configuration Recommandée par Contexte

### Contexte Urbain Dense

```python
from ign_lidar.core.modules.hierarchical_classifier import HierarchicalClassifier
from ign_lidar.core.modules.optimized_thresholds import ClassificationRules
from ign_lidar.core.modules.grammar_3d import GrammarParser, BuildingGrammar

# Seuils urbains
rules = ClassificationRules(context='urban')

# Classifier
classifier = HierarchicalClassifier(rules=rules)
result = classifier.classify(asprs_labels, 'LOD2', features, use_confidence=True)

# Grammaire avec détails
parser = GrammarParser(
    grammar=BuildingGrammar(),
    max_iterations=10,       # Plus d'itérations = plus de détails
    min_confidence=0.6       # Confiance plus élevée
)
refined, tree = parser.parse(points, result.labels, features)
```

### Contexte Rural

```python
# Seuils ruraux
rules = ClassificationRules(context='rural')

# Classifier
classifier = HierarchicalClassifier(rules=rules)
result = classifier.classify(asprs_labels, 'LOD2', features, use_confidence=True)

# Grammaire simplifiée
parser = GrammarParser(
    grammar=BuildingGrammar(),
    max_iterations=6,        # Moins d'itérations
    min_confidence=0.4       # Confiance plus basse
)
refined, tree = parser.parse(points, result.labels, features)
```

### Dataset d'Entraînement (Haute Précision)

```python
from ign_lidar.core.modules.classification_validation import validate_classification

# Classifier avec confiance
result = classify_hierarchical(asprs_labels, 'LOD2', features, use_confidence=True)

# Filtrer par confiance élevée
high_confidence_mask = result.confidence_scores > 0.8
filtered_labels = result.labels[high_confidence_mask]
filtered_points = points[high_confidence_mask]

# Valider qualité
metrics = validate_classification(
    predicted=filtered_labels,
    ground_truth=ground_truth[high_confidence_mask],
    features=features
)

print(f"📊 Métriques dataset d'entraînement:")
print(f"   - Accuracy: {metrics.accuracy:.3f}")
print(f"   - Kappa: {metrics.kappa:.3f}")
print(f"   - Points retenus: {np.sum(high_confidence_mask):,}/{len(points):,} "
      f"({100*np.sum(high_confidence_mask)/len(points):.1f}%)")
```

## 🎨 Personnalisation

### Ajouter Règle de Grammaire Personnalisée

```python
from ign_lidar.core.modules.grammar_3d import (
    BuildingGrammar,
    ProductionRule,
    ArchitecturalSymbol
)

# Grammaire de base
grammar = BuildingGrammar()

# Ajouter règle pour toitures parisiennes
parisian_roof_rule = ProductionRule(
    name="detect_parisian_roof",
    left_hand_side=ArchitecturalSymbol.ROOF,
    right_hand_side=[ArchitecturalSymbol.ROOF_MANSARD],
    conditions={
        'has_two_slopes': True,
        'lower_slope_angle': (60, 80),
        'upper_slope_angle': (20, 40),
        'has_dormer_pattern': True,
        'zinc_intensity': (0.3, 0.5)  # Zinc typique
    },
    priority=70
)

grammar.rules.append(parisian_roof_rule)

# Utiliser grammaire personnalisée
parser = GrammarParser(grammar=grammar)
refined, tree = parser.parse(points, labels, features)
```

### Modifier Seuils de Classification

```python
from ign_lidar.core.modules.optimized_thresholds import (
    ClassificationRules,
    NDVIThresholds
)

# Créer seuils personnalisés
rules = ClassificationRules()

# Modifier seuils NDVI pour région méditerranéenne
rules.ndvi.high_vegetation_threshold = 0.45  # Au lieu de 0.5
rules.ndvi.low_vegetation_threshold = 0.25   # Au lieu de 0.3
rules.ndvi.bare_soil_threshold = 0.1         # Au lieu de 0.15

# Adapter à la saison estivale
rules.adapt_to_season('summer')

# Utiliser
from ign_lidar.core.modules.hierarchical_classifier import HierarchicalClassifier
classifier = HierarchicalClassifier(rules=rules)
result = classifier.classify(asprs_labels, 'LOD2', features)
```

## 🐛 Dépannage

### Erreur: "No buildings detected"

**Cause**: Les labels initiaux ne contiennent pas de points classés "Building"

**Solution**:

```python
# Vérifier labels d'entrée
unique_labels = np.unique(labels)
print(f"Labels présents: {unique_labels}")

# Si pas de buildings (classe 6 en ASPRS), classifier d'abord
if 6 not in unique_labels:
    result = classify_hierarchical(labels, 'LOD2', features)
    labels = result.labels
```

### Erreur: "Features missing"

**Cause**: Features géométriques manquantes

**Solution**:

```python
# Features minimales requises
required_features = ['height', 'normals', 'planarity']

# Calculer si manquantes
if 'height' not in features:
    features['height'] = points[:, 2] - np.min(points[:, 2])

if 'normals' not in features:
    # Calculer normales (exemple simplifié)
    from sklearn.decomposition import PCA
    # ... calcul de normales via PCA local

if 'planarity' not in features:
    # Calculer planarite
    # ... via eigenvalues de PCA
```

### Performance Lente

**Solution 1**: Réduire nombre d'itérations de grammaire

```python
parser = GrammarParser(max_iterations=4)  # Au lieu de 10
```

**Solution 2**: Augmenter confiance minimum

```python
parser = GrammarParser(min_confidence=0.7)  # Ignore dérivations peu confiantes
```

**Solution 3**: Désactiver grammaire pour zones non-bâties

```python
# Appliquer grammaire seulement sur bâtiments
building_mask = labels == 6  # Classe building
refined_labels = labels.copy()

if np.any(building_mask):
    building_refined, tree = classify_with_grammar(
        points[building_mask],
        labels[building_mask],
        {k: v[building_mask] for k, v in features.items()}
    )
    refined_labels[building_mask] = building_refined
```

## 📚 Ressources

### Documentation Complète

- **Classification hiérarchique**: `CLASSIFICATION_IMPROVEMENTS.md`
- **Grammaire 3D**: `GRAMMAR_3D_GUIDE.md`
- **Référence API**: `CLASSIFICATION_REFERENCE.md`
- **Guide de test**: `TESTING.md`

### Exemples

- `examples/example_hierarchical_classification.py` - 5 exemples de classification
- `examples/example_grammar_3d.py` - 5 exemples de grammaire 3D

### Exécuter les Exemples

```bash
cd examples

# Classification hiérarchique
python example_hierarchical_classification.py

# Grammaire 3D
python example_grammar_3d.py
```

## 🎯 Prochaines Étapes

### Niveau Débutant

1. ✅ Utiliser classification hiérarchique simple
2. ✅ Comprendre niveaux LOD2/LOD3
3. ✅ Visualiser résultats dans CloudCompare/QGIS

### Niveau Intermédiaire

1. ✅ Utiliser seuils adaptatifs
2. ✅ Valider et corriger automatiquement
3. ✅ Créer pipeline batch processing

### Niveau Avancé

1. ✅ Intégrer grammaire 3D
2. ✅ Personnaliser règles de grammaire
3. ✅ Analyser arbres de dérivation
4. 🔄 Entraîner modèle sur dataset raffiné

### Niveau Expert

1. 🔄 Créer grammaire spécifique à votre région
2. 🔄 Intégrer avec cadastre/BDTopo
3. 🔄 Export vers CityGML/IFC
4. 🔄 Optimiser pour GPU

---

**Besoin d'aide?** Consultez la documentation complète ou ouvrez une issue sur GitHub.

**Version**: 2.1.0  
**Dernière mise à jour**: 15 octobre 2025
