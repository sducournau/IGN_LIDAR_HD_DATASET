# Classification Avancée - Guide d'Utilisation

## Vue d'ensemble

Le module `advanced_classification.py` fournit un système de classification intelligent qui combine:

1. **Ground Truth IGN BD TOPO®** - Données vectorielles officielles
2. **NDVI** - Détection de végétation par imagerie
3. **Features Géométriques** - Analyse de la géométrie 3D
4. **Buffers Intelligents pour Routes** - Classification précise des routes

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Classification Multi-Sources (Hiérarchique)      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Stage 1: Classification Géométrique                     │
│  ├─ Hauteur + Planéité → Sol/Routes/Bâtiments           │
│  ├─ Normales → Surfaces horizontales/verticales          │
│  └─ Courbure → Surfaces organiques (végétation)          │
│                                                           │
│  Stage 2: Raffinement NDVI                               │
│  ├─ NDVI ≥ 0.35 → Végétation                            │
│  ├─ NDVI ≤ 0.15 → Bâtiments/Routes                      │
│  └─ Corrections des confusions végétation/bâtiment       │
│                                                           │
│  Stage 3: Ground Truth (Priorité Maximale)               │
│  ├─ Bâtiments IGN BD TOPO®                              │
│  ├─ Routes avec buffers intelligents (largeur variable)  │
│  ├─ Surfaces d'eau                                       │
│  └─ Zones de végétation                                  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Utilisation

### 1. Classification Simple

```python
from ign_lidar.core.modules.advanced_classification import AdvancedClassifier
import numpy as np

# Charger les données
points = np.load('points.npy')  # [N, 3] XYZ
height = np.load('height.npy')   # [N] hauteur au-dessus du sol
ndvi = np.load('ndvi.npy')       # [N] valeurs NDVI

# Créer le classificateur
classifier = AdvancedClassifier(
    use_ground_truth=True,
    use_ndvi=True,
    use_geometric=True,
    road_buffer_tolerance=0.5  # 50cm de tolérance autour des routes
)

# Classifier
labels = classifier.classify_points(
    points=points,
    ndvi=ndvi,
    height=height,
    ground_truth_features=ground_truth  # Dict de GeoDataFrames
)
```

### 2. Avec Toutes les Features

```python
from ign_lidar.core.modules.advanced_classification import classify_with_all_features
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher

# Initialiser le fetcher ground truth
fetcher = IGNGroundTruthFetcher(cache_dir="cache/ground_truth")

# Calculer bbox
bbox = (
    points[:, 0].min(),
    points[:, 1].min(),
    points[:, 0].max(),
    points[:, 1].max()
)

# Classification complète
labels = classify_with_all_features(
    points=points,
    ground_truth_fetcher=fetcher,
    bbox=bbox,
    ndvi=ndvi,
    height=height,
    normals=normals,         # [N, 3] vecteurs normales
    planarity=planarity,     # [N] planéité [0-1]
    curvature=curvature,     # [N] courbure
    intensity=intensity,     # [N] intensité LiDAR
    return_number=return_number,  # [N] numéro de retour
    # Paramètres optionnels
    road_buffer_tolerance=0.5,
    ndvi_veg_threshold=0.35,
    ndvi_building_threshold=0.15,
    height_low_veg_threshold=0.5,
    height_medium_veg_threshold=2.0
)
```

### 3. Depuis CLI

```bash
# Mise à jour classification d'un fichier LAZ
ign-lidar-hd update-classification input.laz output.laz \
    --cache-dir cache/ \
    --use-ndvi \
    --fetch-rgb-nir \
    --road-width-fallback 6.0 \
    --update-roads \
    --update-buildings \
    --update-vegetation \
    --update-water

# Avec seuils personnalisés
ign-lidar-hd update-classification input.laz output.laz \
    --ndvi-vegetation-threshold 0.40 \
    --ndvi-building-threshold 0.12 \
    --road-width-fallback 8.0
```

## Features Clés

### 1. Buffers Intelligents pour Routes

Le système utilise l'attribut `largeur` (largeur en mètres) de la BD TOPO® pour créer des buffers adaptés à chaque route:

```python
# Route autoroutière: largeur=12m → buffer=6m de chaque côté
# Route départementale: largeur=8m → buffer=4m de chaque côté
# Chemin rural: largeur=3m → buffer=1.5m de chaque côté
```

**Tolérance additionnelle** (`road_buffer_tolerance`):

- Ajoute un buffer supplémentaire pour capturer les accotements
- Valeur par défaut: 0.5m
- Peut être ajustée selon la précision souhaitée

### 2. Classification Géométrique

#### Sol (Ground)

- Hauteur < 0.2m
- Planéité > 0.85 (très plat)
- Code ASPRS: 2

#### Routes

- Hauteur entre 0.2m et 2.0m
- Planéité > 0.8 (très plat)
- Normales pointant vers le haut (composante Z > 0.9)
- Code ASPRS: 11

#### Bâtiments

- Hauteur ≥ 2.0m
- Planéité > 0.7 (surfaces planes)
- Normales horizontales (toits) ou verticales (murs)
- Code ASPRS: 6

#### Végétation

- Planéité < 0.4 (irrégulier)
- Hauteur > 0.2m
- Classification par hauteur:
  - Basse < 0.5m → Code 3
  - Moyenne 0.5-2.0m → Code 4
  - Haute > 2.0m → Code 5

### 3. Raffinement NDVI

Le NDVI (Normalized Difference Vegetation Index) permet de distinguer végétation et bâtiments:

```
NDVI = (NIR - Red) / (NIR + Red)

Valeurs:
  -1.0 à -0.1  : Eau, surfaces artificielles
   0.0 à  0.2  : Sol nu, routes, bâtiments
   0.2 à  0.4  : Végétation clairsemée
   0.4 à  0.7  : Végétation modérée
   0.7 à  1.0  : Végétation dense
```

**Seuils de décision**:

- `ndvi_veg_threshold=0.35` : Points avec NDVI ≥ 0.35 → Végétation
- `ndvi_building_threshold=0.15` : Points avec NDVI ≤ 0.15 → Non-végétation

**Corrections automatiques**:

- Bâtiment avec NDVI élevé → Possible végétation sur toit (signalé)
- Végétation avec NDVI faible → Reclassé en bâtiment/route

### 4. Hiérarchie de Classification

Les classifications sont appliquées dans l'ordre suivant (du moins prioritaire au plus prioritaire):

1. **Géométrie seule** (confiance: 0.5-0.7)
2. **NDVI** (confiance: 0.8-0.85, écrase géométrie faible confiance)
3. **Ground Truth** (confiance: 1.0, écrase tout)

Cela permet de:

- Utiliser la géométrie comme baseline
- Raffiner avec le NDVI quand disponible
- Garantir la cohérence avec les données officielles IGN

## Codes ASPRS Utilisés

| Code | Classe             | Source Principale        |
| ---- | ------------------ | ------------------------ |
| 1    | Non classifié      | Défaut                   |
| 2    | Sol                | Géométrie + Ground Truth |
| 3    | Végétation Basse   | NDVI + Hauteur           |
| 4    | Végétation Moyenne | NDVI + Hauteur           |
| 5    | Végétation Haute   | NDVI + Hauteur           |
| 6    | Bâtiment           | Ground Truth + Géométrie |
| 9    | Eau                | Ground Truth             |
| 11   | Route              | Ground Truth + Géométrie |

## Optimisations

### Performance

- Utilise des masques NumPy pour traitement vectorisé
- Cache des données ground truth pour éviter requêtes WFS répétées
- Traitement par chunks pour grandes tuiles

### Qualité

- Validation des valeurs NDVI (gestion des NaN/Inf)
- Vérification des géométries Shapely valides
- Logging détaillé des statistiques de classification

## Exemples de Résultats

```
🎯 Classifying 2,450,000 points with advanced method
  Stage 1: Geometric feature classification
    Ground: 120,500 points
    Roads (geometric): 89,200 points
    Buildings (geometric): 156,800 points
    Vegetation (geometric): 1,024,300 points
  Stage 2: NDVI-based vegetation refinement
    Vegetation (NDVI): 1,145,600 points
    Reclassified low-NDVI vegetation: 12,400 points
  Stage 3: Ground truth classification (highest priority)
    Processing buildings: 234 features
    Processing roads: 89 features
      Using intelligent road buffers (tolerance=0.5m)
      Road widths: 3.5m - 14.0m (avg: 7.2m)
      Classified 95,600 road points from 89 roads
      Avg points per road: 1,074
    Processing water: 12 features
📊 Final classification distribution:
  Unclassified        :  245,800 ( 10.0%)
  Ground              :  120,500 (  4.9%)
  Low Vegetation      :  312,400 ( 12.7%)
  Medium Vegetation   :  645,200 ( 26.3%)
  High Vegetation     :  378,100 ( 15.4%)
  Building            :  162,300 (  6.6%)
  Water               :   45,200 (  1.8%)
  Road                :   95,600 (  3.9%)
```

## 🚂 Classification des Voies Ferrées (Railways)

Le module supporte maintenant la classification des voies ferrées depuis IGN BD TOPO®:

### Configuration

```python
from ign_lidar.core.modules.advanced_classification import classify_with_all_features
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.io.bd_foret import BDForetFetcher

# Activer les voies ferrées
labels, forest_attrs = classify_with_all_features(
    points=points,
    ground_truth_fetcher=fetcher,
    bd_foret_fetcher=forest_fetcher,
    bbox=bbox,
    ndvi=ndvi,
    height=height,
    include_railways=True,    # ✅ Activer railways
    include_forest=True       # ✅ Activer BD Forêt®
)
```

### Attributs des Voies Ferrées

Les voies ferrées utilisent le même système de buffering intelligent que les routes:

| Attribut         | Description                 | Défaut  |
| ---------------- | --------------------------- | ------- |
| `nombre_voies`   | Nombre de voies (1, 2, 3+)  | 1       |
| `largeur`        | Largeur d'une voie unique   | 3.5m    |
| `largeur_totale` | `largeur × nombre_voies`    | Calculé |
| `electrifie`     | Ligne électrifiée (OUI/NON) | -       |

### Buffer des Voies Ferrées

```python
# Voie simple: buffer = 3.5m / 2 = 1.75m
# Voie double: buffer = (3.5m × 2) / 2 = 3.5m
# Voie triple: buffer = (3.5m × 3) / 2 = 5.25m
```

### Code ASPRS

Railways sont classifiées avec le code ASPRS standard:

- **Code 10**: Rail (ASPRS_RAIL)

---

## 🌲 BD Forêt® V2 - Classification Précise de la Végétation

Le module BD Forêt® permet d'obtenir des **types forestiers précis** depuis la BD Forêt® V2 de l'IGN:

### Types de Forêts Supportés

| Code TFV | Type                | Description                          |
| -------- | ------------------- | ------------------------------------ |
| `FF1-*`  | **Coniferous**      | Forêt fermée de feuillus (> 75%)     |
| `FF2-*`  | **Mixed**           | Forêt fermée mixte (40-75% feuillus) |
| `FF3-*`  | **Deciduous**       | Forêt fermée de conifères (> 75%)    |
| `FO1-*`  | **Open Coniferous** | Forêt ouverte de conifères           |
| `FO2-*`  | **Open Mixed**      | Forêt ouverte mixte                  |
| `FO3-*`  | **Open Deciduous**  | Forêt ouverte de feuillus            |

### Essences Disponibles

BD Forêt® fournit jusqu'à 3 essences par formation avec leurs taux:

```python
{
    'essence_1': 'Chêne',           # Essence dominante
    'taux_1': 60,                    # 60%
    'essence_2': 'Hêtre',           # Essence secondaire
    'taux_2': 30,                    # 30%
    'essence_3': 'Charme',          # Essence tertiaire
    'taux_3': 10                     # 10%
}
```

### Attributs Forestiers Retournés

La fonction `classify_with_all_features` retourne un dictionnaire `forest_attributes`:

```python
labels, forest_attrs = classify_with_all_features(...)

# Structure de forest_attrs
{
    'forest_type': ['coniferous', 'deciduous', ...],    # [N] Type de forêt
    'primary_species': ['Chêne', 'Sapin', ...],         # [N] Essence dominante
    'species_rate': [60, 75, ...],                      # [N] Taux essence dominante (%)
    'density': ['closed', 'open', ...],                 # [N] Densité (fermée/ouverte)
    'structure': ['mature', 'young', ...],              # [N] Structure
    'estimated_height': [15.0, 20.0, ...]               # [N] Hauteur estimée (m)
}
```

### Exemple d'Utilisation

```python
from ign_lidar.io.bd_foret import BDForetFetcher

# Initialiser le fetcher BD Forêt®
forest_fetcher = BDForetFetcher(cache_dir="cache/bd_foret")

# Classification avec BD Forêt®
labels, forest_attrs = classify_with_all_features(
    points=points,
    ground_truth_fetcher=gt_fetcher,
    bd_foret_fetcher=forest_fetcher,
    bbox=bbox,
    ndvi=ndvi,
    height=height,
    include_forest=True
)

# Analyser les types de forêts
import pandas as pd
forest_df = pd.DataFrame(forest_attrs)

# Filtrer points de végétation avec type forestier
veg_mask = np.isin(labels, [3, 4, 5])  # Low/Medium/High Vegetation
forest_mask = veg_mask & (forest_df['forest_type'] != 'unknown')

print(f"Végétation classifiée: {veg_mask.sum():,} points")
print(f"Avec type forestier: {forest_mask.sum():,} points")

# Distribution des types
type_counts = forest_df[forest_mask]['forest_type'].value_counts()
print("\nTypes forestiers:")
for ftype, count in type_counts.items():
    pct = 100 * count / forest_mask.sum()
    print(f"  {ftype:20s}: {count:8,} ({pct:5.1f}%)")

# Top 5 essences
species_counts = forest_df[forest_mask]['primary_species'].value_counts().head(5)
print("\nEssences dominantes:")
for species, count in species_counts.items():
    print(f"  {species:20s}: {count:8,} points")
```

### Logs de BD Forêt®

```
Refining vegetation classification with BD Forêt® V2...
  Fetching forest polygons...
  Found 142 forest formations
  Labeling 1,335,700 vegetation points...
  Labeled 1,203,450 vegetation points with forest types
    coniferous: 456,200 points
    mixed: 387,100 points
    deciduous: 360,150 points
```

### Estimation de Hauteur par Type

BD Forêt® fournit des estimations de hauteur basées sur le type forestier:

| Type                | Hauteur Estimée |
| ------------------- | --------------- |
| Coniferous (fermée) | 15-25m          |
| Deciduous (fermée)  | 12-20m          |
| Mixed (fermée)      | 10-18m          |
| Open forests        | 5-12m           |
| Young forests       | 5-10m           |

---

## Notes Importantes

1. **Ordre d'importance**: Ground Truth > NDVI > Géométrie > Défaut
2. **Cohérence spatiale**: Les routes et railways utilisent leur largeur réelle depuis BD TOPO®
3. **Robustesse**: Gère les données manquantes gracieusement
4. **Traçabilité**: Logs détaillés de chaque décision de classification
5. **BD Forêt®**: Raffinement optionnel des codes ASPRS 3, 4, 5 (végétation) avec types précis
6. **Railways**: Code ASPRS 10 standard, buffering intelligent basé sur nombre de voies

## Voir Aussi

- `ign_lidar/io/wfs_ground_truth.py` - Récupération données IGN BD TOPO®
- `ign_lidar/io/bd_foret.py` - Récupération données IGN BD Forêt® V2
- `ign_lidar/features/geometric.py` - Calcul features géométriques
- `ign_lidar/preprocessing/rgb_augmentation.py` - Récupération RGB
- `ign_lidar/preprocessing/infrared_augmentation.py` - Récupération NIR
