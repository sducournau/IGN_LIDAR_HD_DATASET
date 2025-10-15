# Intégration du Cadastre (BD PARCELLAIRE)

Ce guide explique comment utiliser le module de cadastre pour grouper les points LiDAR par parcelles cadastrales et enrichir vos données avec les informations parcellaires.

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Source de données](#source-de-données)
- [Installation](#installation)
- [Utilisation rapide](#utilisation-rapide)
- [API détaillée](#api-détaillée)
- [Cas d'usage](#cas-dusage)
- [Formats de sortie](#formats-de-sortie)
- [Performance](#performance)

## Vue d'ensemble

Le module cadastral permet de :

1. **Récupérer les parcelles cadastrales** depuis la BD PARCELLAIRE de l'IGN
2. **Labelliser chaque point** avec l'identifiant de sa parcelle
3. **Grouper les points par parcelle** avec statistiques détaillées
4. **Exporter les résultats** en GeoJSON, CSV ou intégrés au LAZ

### Pourquoi utiliser le cadastre ?

- **Segmentation parcellaire** : Organiser automatiquement les points par parcelle
- **Statistiques foncières** : Analyser les caractéristiques de chaque parcelle (densité, distribution des classes)
- **Gestion territoriale** : Relier les données LiDAR au cadastre pour l'urbanisme
- **Traçabilité** : Identifier précisément la parcelle d'origine de chaque point

## Source de données

### BD PARCELLAIRE (Parcellaire Express)

La BD PARCELLAIRE est la représentation numérique du plan cadastral français maintenue par l'IGN et la DGFiP.

**Caractéristiques :**

- **Couche WFS** : `CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle`
- **Projection** : Lambert 93 (EPSG:2154)
- **Mise à jour** : Trimestrielle
- **Couverture** : France métropolitaine et DROM

**Attributs disponibles :**
| Attribut | Type | Description |
|----------|------|-------------|
| `id` | string | Identifiant unique de parcelle (19 caractères) |
| `numero` | string | Numéro de parcelle dans la section |
| `section` | string | Code section (2 lettres) |
| `commune` | string | Code INSEE commune (5 chiffres) |
| `contenance` | integer | Surface de la parcelle (m²) |
| `geometry` | Polygon | Géométrie de la parcelle |

**Format ID parcelle** : `{commune}{prefixe}{section}{numero}`

- Exemple : `38185000AB0042` = Grenoble (38185), section AB, n°42

### Accès au service WFS

```python
WFS_URL = "https://data.geopf.fr/wfs"
LAYER = "CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle"
```

## Installation

Le module cadastre est inclus dans `ign_lidar.io.cadastre`.

**Dépendances requises :**

```bash
pip install geopandas shapely requests
```

**Dépendances optionnelles pour export :**

```bash
pip install pandas  # Pour export CSV/statistiques
```

## Utilisation rapide

### Exemple simple : Labelliser les points

```python
from pathlib import Path
import numpy as np
import laspy
from ign_lidar.io.cadastre import CadastreFetcher

# Charger le fichier LiDAR
las = laspy.read("data/tile.laz")
points = np.vstack([las.x, las.y, las.z]).T

# Bounding box
bbox = (
    float(las.header.x_min),
    float(las.header.y_min),
    float(las.header.x_max),
    float(las.header.y_max)
)

# Initialiser le fetcher
fetcher = CadastreFetcher(cache_dir=Path("cache/cadastre"))

# Récupérer les parcelles
parcels_gdf = fetcher.fetch_parcels(bbox=bbox, use_cache=True)
print(f"Found {len(parcels_gdf)} parcels")

# Labelliser chaque point avec l'ID parcelle
parcel_labels = fetcher.label_points_with_parcel_id(
    points=points,
    parcels_gdf=parcels_gdf
)

# Afficher distribution
from collections import Counter
label_counts = Counter(parcel_labels)
print(f"Assigned parcels: {len([l for l in parcel_labels if l != 'unassigned'])}")
print(f"Top 5 parcels:")
for parcel_id, count in label_counts.most_common(5):
    if parcel_id != 'unassigned':
        print(f"  {parcel_id}: {count:,} points")
```

### Exemple : Grouper par parcelle avec statistiques

```python
# Grouper les points par parcelle
parcel_groups = fetcher.group_points_by_parcel(
    points=points,
    parcels_gdf=parcels_gdf,
    labels=las.classification  # Optionnel : distribution des classes
)

print(f"Grouped into {len(parcel_groups)} parcels")

# Afficher statistiques d'une parcelle
parcel_id = list(parcel_groups.keys())[0]
stats = parcel_groups[parcel_id]

print(f"\nParcel {parcel_id}:")
print(f"  Points: {stats['n_points']:,}")
print(f"  Density: {stats['point_density']:.1f} pts/m²")
print(f"  Bounds: {stats['bounds']}")

if 'class_distribution' in stats:
    print(f"  Class distribution:")
    for cls, count in stats['class_distribution'].items():
        print(f"    Class {cls}: {count:,} points")
```

### Exemple : Exporter les résultats

```python
# Export GeoJSON avec statistiques
from ign_lidar.io.cadastre import export_parcel_groups_to_geojson

export_parcel_groups_to_geojson(
    parcel_groups=parcel_groups,
    parcels_gdf=parcels_gdf,
    output_path=Path("output/parcel_stats.geojson")
)

# Export CSV des statistiques
stats_df = fetcher.get_parcel_statistics(parcel_groups, parcels_gdf)
stats_df.to_csv("output/parcel_stats.csv", index=False)

# Export LAZ avec labels parcellaires
las.add_extra_dim(laspy.ExtraBytesParams(
    name='parcel_id',
    type='str',
    description='Cadastral parcel ID'
))
las.parcel_id = parcel_labels
las.write("output/tile_with_parcels.laz")
```

## API détaillée

### CadastreFetcher

```python
class CadastreFetcher:
    """
    Fetcher pour récupérer et traiter les données cadastrales.

    Args:
        wfs_url: URL du service WFS (default: IGN Géoplateforme)
        layer_name: Nom de la couche parcellaire (default: PARCELLAIRE_EXPRESS)
        cache_dir: Répertoire de cache (default: None)
        timeout: Timeout requêtes WFS en secondes (default: 60)
    """
```

#### Méthodes principales

##### `fetch_parcels(bbox, use_cache=True)`

Récupère les parcelles cadastrales dans une bounding box.

**Paramètres :**

- `bbox` : Tuple `(xmin, ymin, xmax, ymax)` en Lambert 93
- `use_cache` : Utiliser le cache si disponible

**Retour :** `GeoDataFrame` avec colonnes :

- `id_parcelle` : Identifiant unique
- `numero`, `section`, `commune` : Composants de l'ID
- `contenance` : Surface (m²)
- `geometry` : Géométrie Polygon

**Exemple :**

```python
parcels = fetcher.fetch_parcels(
    bbox=(900000, 6450000, 901000, 6451000),
    use_cache=True
)
```

##### `label_points_with_parcel_id(points, parcels_gdf)`

Assigne à chaque point l'ID de sa parcelle.

**Paramètres :**

- `points` : Array numpy (N, 3) avec X, Y, Z
- `parcels_gdf` : GeoDataFrame des parcelles (résultat de `fetch_parcels`)

**Retour :** Liste de strings (longueur N)

- ID parcelle si point dans parcelle
- `"unassigned"` si point hors parcelle

**Exemple :**

```python
labels = fetcher.label_points_with_parcel_id(points, parcels)

# Statistiques
n_assigned = sum(1 for l in labels if l != 'unassigned')
print(f"Points assigned: {n_assigned}/{len(labels)}")
```

##### `group_points_by_parcel(points, parcels_gdf, labels=None)`

Groupe les points par parcelle avec statistiques complètes.

**Paramètres :**

- `points` : Array numpy (N, 3)
- `parcels_gdf` : GeoDataFrame des parcelles
- `labels` : Array (N,) de classes ASPRS (optionnel)

**Retour :** Dictionnaire `{parcel_id: stats_dict}`

**Structure `stats_dict` :**

```python
{
    'indices': array([...]),           # Indices des points dans cette parcelle
    'n_points': int,                   # Nombre de points
    'point_density': float,            # Densité (pts/m²)
    'bounds': (xmin, ymin, xmax, ymax), # Bounding box des points
    'class_distribution': {            # Si labels fourni
        cls_code: count
    }
}
```

**Exemple :**

```python
groups = fetcher.group_points_by_parcel(
    points=points,
    parcels_gdf=parcels,
    labels=las.classification
)

# Trouver les parcelles les plus denses
sorted_parcels = sorted(
    groups.items(),
    key=lambda x: x[1]['point_density'],
    reverse=True
)

print("Top 5 densest parcels:")
for parcel_id, stats in sorted_parcels[:5]:
    print(f"  {parcel_id}: {stats['point_density']:.1f} pts/m²")
```

##### `get_parcel_statistics(parcel_groups, parcels_gdf)`

Génère un DataFrame avec toutes les statistiques par parcelle.

**Paramètres :**

- `parcel_groups` : Résultat de `group_points_by_parcel`
- `parcels_gdf` : GeoDataFrame des parcelles

**Retour :** `pandas.DataFrame` avec colonnes :

- `parcel_id`, `commune`, `section`, `numero`
- `area_m2` : Surface parcelle (m²)
- `n_points` : Nombre de points
- `point_density` : Densité (pts/m²)
- `bounds_*` : Bounding box
- `class_*` : Nombre de points par classe (si fourni)

**Exemple :**

```python
df = fetcher.get_parcel_statistics(groups, parcels)

# Analyser les parcelles agricoles
agri_parcels = df[df['class_44'] > 0]  # Class 44 = Agriculture
print(f"Agricultural parcels: {len(agri_parcels)}")
print(f"Average density: {agri_parcels['point_density'].mean():.1f} pts/m²")

# Export CSV
df.to_csv("parcel_analysis.csv", index=False)
```

### Fonction utilitaire : `export_parcel_groups_to_geojson`

```python
def export_parcel_groups_to_geojson(
    parcel_groups: dict,
    parcels_gdf: gpd.GeoDataFrame,
    output_path: Path
) -> None:
    """
    Exporte les groupes de parcelles avec statistiques en GeoJSON.

    Args:
        parcel_groups: Résultat de group_points_by_parcel()
        parcels_gdf: GeoDataFrame des parcelles
        output_path: Chemin de sortie .geojson
    """
```

**Exemple :**

```python
export_parcel_groups_to_geojson(
    parcel_groups=groups,
    parcels_gdf=parcels,
    output_path=Path("output/parcels_with_stats.geojson")
)

# Le fichier GeoJSON peut être ouvert dans QGIS pour visualisation
```

## Cas d'usage

### 1. Analyse urbaine : Densité de points par parcelle

```python
# Grouper par parcelle
groups = fetcher.group_points_by_parcel(points, parcels)

# Identifier parcelles sous-échantillonnées
stats_df = fetcher.get_parcel_statistics(groups, parcels)
low_density = stats_df[stats_df['point_density'] < 10]

print(f"Low-density parcels: {len(low_density)}")
print(low_density[['parcel_id', 'area_m2', 'point_density']])

# Recommander nouveau vol LiDAR sur ces zones
```

### 2. Classification par usage : Croiser cadastre et classes ASPRS

```python
# Grouper avec classes
groups = fetcher.group_points_by_parcel(
    points=points,
    parcels_gdf=parcels,
    labels=las.classification
)

# Analyser chaque parcelle
for parcel_id, stats in groups.items():
    if 'class_distribution' not in stats:
        continue

    dist = stats['class_distribution']
    total = sum(dist.values())

    # Identifier type dominant
    if dist.get(6, 0) / total > 0.5:  # Building
        print(f"{parcel_id}: Built parcel ({dist[6]/total*100:.0f}% building)")
    elif dist.get(44, 0) / total > 0.5:  # Agriculture
        print(f"{parcel_id}: Agricultural parcel")
    elif dist.get(5, 0) / total > 0.5:  # High vegetation
        print(f"{parcel_id}: Forested parcel")
```

### 3. Export pour SIG : Intégration QGIS

```python
# Export complet pour visualisation SIG
from ign_lidar.io.cadastre import export_parcel_groups_to_geojson

export_parcel_groups_to_geojson(
    parcel_groups=groups,
    parcels_gdf=parcels,
    output_path=Path("output/parcels_stats.geojson")
)

# Le GeoJSON peut être chargé dans QGIS avec :
# - Géométries des parcelles
# - Statistiques comme attributs (n_points, density, class_*)
# - Possibilité de thématiser par densité ou usage
```

### 4. Traçabilité : Ajouter ID parcelle au LAZ

```python
# Labelliser les points
parcel_labels = fetcher.label_points_with_parcel_id(points, parcels)

# Ajouter au fichier LAZ
las.add_extra_dim(laspy.ExtraBytesParams(
    name='parcel',
    type='str',
    description='Cadastral parcel ID'
))
las.parcel = parcel_labels

# Sauvegarder
las.write("output/tile_with_parcel_ids.laz")

# Les outils de visualisation (CloudCompare, QGIS) pourront
# filtrer/colorier par parcelle
```

## Formats de sortie

### GeoJSON (visualisation SIG)

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[...]]
      },
      "properties": {
        "parcel_id": "38185000AB0042",
        "commune": "38185",
        "section": "AB",
        "numero": "0042",
        "area_m2": 1250,
        "n_points": 15678,
        "point_density": 12.5,
        "class_2": 2345,
        "class_5": 8901,
        "class_6": 4432
      }
    }
  ]
}
```

### CSV (analyse statistique)

| parcel_id      | commune | section | numero | area_m2 | n_points | point_density | class_2 | class_5 | class_6 |
| -------------- | ------- | ------- | ------ | ------- | -------- | ------------- | ------- | ------- | ------- |
| 38185000AB0042 | 38185   | AB      | 0042   | 1250    | 15678    | 12.54         | 2345    | 8901    | 4432    |
| 38185000AB0043 | 38185   | AB      | 0043   | 2100    | 21456    | 10.22         | 3456    | 12000   | 6000    |

### LAZ (extra dimension)

```python
# Point #12345:
#   X, Y, Z: 900123.45, 6450234.56, 212.34
#   Classification: 6 (Building)
#   Parcel: "38185000AB0042"
```

## Performance

### Optimisations implémentées

1. **Index spatial** : Utilisation de `shapely.STRtree` pour tests rapides point-in-polygon
2. **Cache WFS** : Les parcelles sont mises en cache localement
3. **Vectorisation** : Opérations numpy vectorisées quand possible

### Benchmarks

**Configuration test :**

- Tile 1km × 1km
- ~10M points
- ~500 parcelles

**Temps d'exécution :**

| Opération                         | Temps  | Notes               |
| --------------------------------- | ------ | ------------------- |
| `fetch_parcels` (no cache)        | ~2-5s  | Dépend latence WFS  |
| `fetch_parcels` (cached)          | ~0.1s  | Lecture cache local |
| `label_points_with_parcel_id`     | ~5-10s | 10M points          |
| `group_points_by_parcel`          | ~3-8s  | 10M points          |
| `export_parcel_groups_to_geojson` | ~0.5s  | 500 parcelles       |

**Recommandations :**

1. **Activer le cache** : Réutiliser `use_cache=True` entre runs
2. **Traiter par tuiles** : Ne pas charger >50M points en mémoire
3. **Filtrer les parcelles** : Limiter aux parcelles contenant des points

### Limites

- **Taille max** : ~50M points par appel (limite mémoire)
- **Parcelles** : Géométries simples (polygones sans trous complexes)
- **Performance** : Dégradée si >5000 parcelles dans bbox

## Questions fréquentes

### Comment gérer les points hors parcelles ?

Les points hors parcelles reçoivent le label `"unassigned"`. C'est normal pour :

- Points en voirie (domaine public)
- Points à la limite de tuiles
- Zones non cadastrées (rivières, forêts domaniales)

### Puis-je utiliser d'autres couches cadastrales ?

Oui, passez `layer_name` au constructeur :

```python
fetcher = CadastreFetcher(
    layer_name="CADASTRALPARCELS.PARCELLAIRE_EXPRESS:batiment"  # Bâtiments
)
```

### Comment combiner avec RPG (agriculture) ?

Utilisez le système unifié :

```python
from ign_lidar.io.unified_fetcher import create_full_fetcher

fetcher = create_full_fetcher(cache_dir=Path("cache"))
data = fetcher.fetch_all(bbox=bbox)

# data['cadastre'] = Parcelles cadastrales
# data['rpg'] = Parcelles agricoles RPG

# Croiser les deux sources pour identifier parcelles agricoles cadastrées
```

### Le cache expire-t-il ?

Par défaut, le cache n'expire pas (données cadastrales stables). Pour forcer refresh :

```python
parcels = fetcher.fetch_parcels(bbox=bbox, use_cache=False)
```

## Ressources

- **BD PARCELLAIRE** : https://geoservices.ign.fr/parcellaire-express
- **Service WFS** : https://data.geopf.fr/wfs
- **Documentation technique** : https://geoservices.ign.fr/documentation/services/services-geoplateforme/wfs
- **Code source** : `ign_lidar/io/cadastre.py`

## Voir aussi

- [Intégration BD TOPO + RPG](./BD_TOPO_RPG_INTEGRATION.md) : Combiner infrastructure et agriculture
- [BD Forêt®](./BD_FORET_INTEGRATION.md) : Types de forêts
- [Classification Avancée](./ADVANCED_CLASSIFICATION_GUIDE.md) : Système complet multi-sources
