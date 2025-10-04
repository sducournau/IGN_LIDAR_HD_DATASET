---
sidebar_position: 1
title: API Caractéristiques
description: Fonctions principales d'extraction et de traitement des caractéristiques
keywords: [api, caractéristiques, bâtiment, classification, géométrique]
---

# Référence API Caractéristiques

L'API Caractéristiques fournit des outils complets pour extraire des caractéristiques géométriques et sémantiques à partir de nuages de points LiDAR.

## Classes Principales

### FeatureExtractor

Classe principale pour les opérations d'extraction de caractéristiques.

```python
from ign_lidar import FeatureExtractor

extractor = FeatureExtractor(
    building_threshold=0.5,
    min_points_per_building=100,
    use_gpu=True
)
```

#### Méthodes

##### `extract_building_features(points, labels)`

Extrait les caractéristiques géométriques pour la classification de bâtiments.

**Paramètres :**

- `points` (numpy.ndarray) : Données nuage de points (N×3)
- `labels` (numpy.ndarray) : Étiquettes de classification
- `neighborhood_size` (int, optionnel) : Rayon de recherche pour calcul caractéristiques

**Retourne :**

- `dict` : Dictionnaire contenant les caractéristiques extraites

**Exemple :**

```python
features = extractor.extract_building_features(
    points=point_cloud,
    labels=classifications,
    neighborhood_size=1.0
)
```

##### `compute_geometric_features(points)`

Calcule les caractéristiques géométriques de base pour chaque point.

**Paramètres :**

- `points` (numpy.ndarray) : Coordonnées points d'entrée

**Retourne :**

- `numpy.ndarray` : Tableau caractéristiques (N×F où F est nombre caractéristiques)

### BuildingClassifier

Classification avancée pour les composants de bâtiment.

```python
from ign_lidar import BuildingClassifier

classifier = BuildingClassifier(
    model_type="random_forest",
    use_height_features=True,
    enable_planarity=True
)
```

#### Méthodes

##### `classify_components(points, features)`

Classifie les composants de bâtiment (toit, mur, sol).

**Paramètres :**

- `points` (numpy.ndarray) : Coordonnées points
- `features` (dict) : Caractéristiques extraites de FeatureExtractor

**Retourne :**

- `numpy.ndarray` : Étiquettes composants (0=sol, 1=mur, 2=toit)

##### `refine_classification(labels, points)`

Post-traite les résultats de classification pour une meilleure précision.

**Paramètres :**

- `labels` (numpy.ndarray) : Classification initiale
- `points` (numpy.ndarray) : Coordonnées points

**Retourne :**

- `numpy.ndarray` : Étiquettes classification affinées

## Types de Caractéristiques

### Caractéristiques Géométriques

| Caractéristique       | Description                   | Plage   |
| --------------------- | ----------------------------- | ------- |
| `planarity`           | Mesure planarité locale       | [0, 1]  |
| `linearity`           | Indicateur structure linéaire | [0, 1]  |
| `sphericity`          | Compacité structure 3D        | [0, 1]  |
| `height_above_ground` | Hauteur normalisée            | [0, ∞]  |
| `normal_z`            | Composante Z vecteur normal   | [-1, 1] |

### Caractéristiques Architecturales

| Caractéristique      | Description                | Application             |
| -------------------- | -------------------------- | ----------------------- |
| `edge_strength`      | Détection contour bâtiment | Limites mur/toit        |
| `corner_likelihood`  | Probabilité coin           | Coins bâtiment          |
| `surface_roughness`  | Mesure texture             | Classification matériau |
| `overhang_indicator` | Détection porte-à-faux     | Géométries complexes    |

## Configuration

### Paramètres Extraction Caractéristiques

```python
config = {
    "geometric_features": {
        "planarity": True,
        "linearity": True,
        "sphericity": True,
        "normal_vectors": True
    },
    "architectural_features": {
        "edge_detection": True,
        "corner_detection": True,
        "surface_analysis": True
    },
    "computation": {
        "neighborhood_size": 1.0,
        "min_neighbors": 10,
        "max_neighbors": 100
    }
}

extractor = FeatureExtractor(config=config)
```

### Accélération GPU

Activer traitement GPU pour extraction caractéristiques plus rapide :

```python
extractor = FeatureExtractor(
    use_gpu=True,
    gpu_memory_fraction=0.7,
    batch_size=50000
)
```

## Gestion d'Erreurs

```python
try:
    features = extractor.extract_building_features(points, labels)
except InsufficientPointsError:
    print("Pas assez de points pour extraction caractéristiques")
except GPUMemoryError:
    print("Mémoire GPU insuffisante, basculement vers CPU")
    extractor.use_gpu = False
    features = extractor.extract_building_features(points, labels)
```

## Optimisation Performance

### Gestion Mémoire

```python
# Traiter gros jeux données par chunks
def process_large_dataset(large_points):
    chunk_size = 100000
    all_features = []

    for i in range(0, len(large_points), chunk_size):
        chunk = large_points[i:i+chunk_size]
        features = extractor.extract_building_features(chunk)
        all_features.append(features)

    return combine_features(all_features)
```

### Traitement Parallèle

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parallel_feature_extraction(point_chunks):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(extractor.extract_building_features, chunk)
            for chunk in point_chunks
        ]
        results = [future.result() for future in futures]
    return results
```

## Exemples

### Extraction Caractéristiques de Base

```python
import numpy as np
from ign_lidar import FeatureExtractor

# Charger nuage de points
points = np.load('building_points.npy')
labels = np.load('building_labels.npy')

# Initialiser extracteur
extractor = FeatureExtractor()

# Extraire caractéristiques
features = extractor.extract_building_features(points, labels)

# Accéder caractéristiques spécifiques
planarity = features['planarity']
height_features = features['height_above_ground']
```

### Pipeline Classification Avancé

```python
from ign_lidar import FeatureExtractor, BuildingClassifier

# Configurer pipeline traitement
extractor = FeatureExtractor(use_gpu=True)
classifier = BuildingClassifier(model_type="gradient_boosting")

# Traiter nuage points
features = extractor.extract_building_features(points, initial_labels)
refined_labels = classifier.classify_components(points, features)
final_labels = classifier.refine_classification(refined_labels, points)
```

## Documentation Associée

- [API Processeur](./processor.md)
- [API Augmentation RGB](./rgb-augmentation.md)
- [Guide Intégration GPU](../guides/gpu-acceleration.md)
- [Optimisation Performance](../guides/performance.md)
