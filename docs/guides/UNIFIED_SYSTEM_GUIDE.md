# Système de Classification Unifié

Guide complet pour utiliser le système de classification unifié qui intègre toutes les sources de données géographiques IGN.

## 📋 Vue d'ensemble

Le système unifié combine **4 sources de données** en un seul pipeline :

1. **BD TOPO® V3** : Infrastructure complète (bâtiments, routes, rails, eau, végétation, ponts, parkings, cimetières, lignes électriques, sports)
2. **BD Forêt® V2** : Types de forêts détaillés (résineux, feuillus, mixtes) avec essences principales
3. **RPG** (Registre Parcellaire Graphique) : Parcelles agricoles et types de cultures (40+ codes)
4. **BD PARCELLAIRE** : Cadastre pour grouper points par parcelles

### Avantages du système unifié

- ✅ **Interface unique** : Un seul point d'entrée pour toutes les sources
- ✅ **Configuration centralisée** : Fichier YAML pour tous les paramètres
- ✅ **Cache intelligent** : Gestion automatique du cache par source
- ✅ **Statistiques complètes** : Enrichissement multi-niveaux des points
- ✅ **Groupement parcellaire** : Organisation automatique par parcelles cadastrales

## 🚀 Démarrage rapide

### Installation

```bash
# Dépendances obligatoires
pip install numpy geopandas shapely laspy requests

# Dépendances optionnelles
pip install pandas pyyaml  # Pour CSV et configuration YAML
```

### Exemple minimal

```python
from pathlib import Path
import numpy as np
import laspy
from ign_lidar.io.unified_fetcher import create_full_fetcher

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

# Créer le fetcher unifié avec toutes les features
fetcher = create_full_fetcher(cache_dir=Path("cache"))

# Récupérer TOUTES les données en un appel
data = fetcher.fetch_all(bbox=bbox, use_cache=True)

print(f"BD TOPO features: {len(data['ground_truth']) if data['ground_truth'] else 0}")
print(f"Forest polygons: {len(data['forest']) if data['forest'] else 0}")
print(f"Agricultural parcels: {len(data['agriculture']) if data['agriculture'] else 0}")
print(f"Cadastral parcels: {len(data['cadastre']) if data['cadastre'] else 0}")
```

### Classification complète

```python
from ign_lidar.core.modules.advanced_classification import classify_with_all_features

# Classification avec toutes les sources
labels, forest_attrs, rpg_attrs = classify_with_all_features(
    points=points,
    ground_truth_fetcher=fetcher.ground_truth_fetcher,
    bd_foret_fetcher=fetcher.forest_fetcher,
    rpg_fetcher=fetcher.rpg_fetcher,
    bbox=bbox,
    include_railways=True,
    include_forest=True,
    include_agriculture=True,
    include_bridges=True,
    include_parking=True,
    include_sports=True
)

# Sauvegarder
las.classification = labels
las.write("output/tile_classified.laz")
```

## 📦 Architecture du système

### Composants principaux

```
ign_lidar/
├── io/
│   ├── wfs_ground_truth.py      # BD TOPO® (10 types de features)
│   ├── bd_foret.py               # BD Forêt® V2
│   ├── rpg.py                    # RPG agriculture
│   ├── cadastre.py               # BD PARCELLAIRE
│   └── unified_fetcher.py        # ✨ Interface unifiée
├── core/modules/
│   └── advanced_classification.py # Classification multi-sources
└── config/
    └── loader.py                 # Chargement configuration YAML
```

### Flux de données

```
1. Configuration YAML
   ↓
2. UnifiedDataFetcher
   ↓
3. Fetchers individuels (parallèle)
   ├── BD TOPO®
   ├── BD Forêt®
   ├── RPG
   └── Cadastre
   ↓
4. Cache local (par source)
   ↓
5. Classification avancée
   ↓
6. Enrichissement attributs
   ↓
7. Groupement parcellaire
   ↓
8. Export (LAZ + GeoJSON + CSV)
```

## 🔧 Configuration

### Configuration par code

```python
from ign_lidar.io.unified_fetcher import UnifiedDataFetcher, DataFetchConfig

# Configuration personnalisée
config = DataFetchConfig(
    # BD TOPO® : Activer seulement certaines features
    include_buildings=True,
    include_roads=True,
    include_railways=True,
    include_water=True,
    include_vegetation=False,  # Désactiver végétation
    include_bridges=True,
    include_parking=False,
    include_cemeteries=False,
    include_power_lines=True,
    include_sports=False,

    # Autres sources
    include_forest=True,           # BD Forêt®
    include_agriculture=True,      # RPG
    rpg_year=2023,                 # Année RPG

    include_cadastre=True,         # BD PARCELLAIRE
    group_by_parcel=True           # Activer groupement
)

fetcher = UnifiedDataFetcher(
    cache_dir=Path("cache"),
    config=config
)
```

### Configuration YAML

Créez un fichier `config.yaml` :

```yaml
data_sources:
  bd_topo:
    features:
      buildings: true
      roads: true
      railways: true
      water: true
      vegetation: true
      bridges: true
      parking: true
      cemeteries: false
      power_lines: true
      sports: true

  bd_foret:
    enabled: true
    layer: "BDFORET_V2:formation_vegetale"

  rpg:
    enabled: true
    year: 2023

  cadastre:
    enabled: true
    group_by_parcel: true

cache:
  directory: "cache"
  ttl:
    bd_topo: 86400 # 1 jour
    bd_foret: 604800 # 7 jours
    rpg: 2592000 # 30 jours
    cadastre: 2592000 # 30 jours

output:
  formats:
    - laz
    - geojson
    - csv
```

Charger la configuration :

```python
from ign_lidar.config.loader import quick_setup

# Setup automatique depuis YAML
config, fetcher, warnings = quick_setup(
    config_path=Path("config.yaml"),
    cache_dir=Path("cache")
)

# Utiliser le fetcher
data = fetcher.fetch_all(bbox=my_bbox)
```

## 📊 Classes ASPRS

### Classes standard (1-17)

| Code | Nom               | Source              |
| ---- | ----------------- | ------------------- |
| 1    | Unclassified      | Par défaut          |
| 2    | Ground            | Géométrique         |
| 3    | Low Vegetation    | NDVI + hauteur < 2m |
| 4    | Medium Vegetation | NDVI + 2m ≤ h < 5m  |
| 5    | High Vegetation   | NDVI + h ≥ 5m       |
| 6    | Building          | BD TOPO®            |
| 9    | Water             | BD TOPO®            |
| 10   | Rail              | BD TOPO®            |
| 11   | Road              | BD TOPO®            |
| 17   | Bridge            | BD TOPO®            |

### Classes étendues (40-44)

| Code | Nom         | Source   | Description                     |
| ---- | ----------- | -------- | ------------------------------- |
| 40   | Parking     | BD TOPO® | Aires de stationnement          |
| 41   | Sports      | BD TOPO® | Terrains de sport               |
| 42   | Cemetery    | BD TOPO® | Cimetières                      |
| 43   | Power Line  | BD TOPO® | Lignes électriques (buffer 10m) |
| 44   | Agriculture | RPG      | Parcelles agricoles             |

### Priorité de classification

Ordre décroissant de priorité :

1. **Bridge** (17) - Infrastructures aériennes
2. **Building** (6) - Structures bâties
3. **Water** (9) - Surfaces en eau
4. **Power Line** (43) - Lignes électriques
5. **Rail** (10) - Voies ferrées
6. **Road** (11) - Routes
7. **Cemetery** (42) - Cimetières
8. **Parking** (40) - Parkings
9. **Sports** (41) - Terrains de sport
10. **Agriculture** (44) - Terres agricoles
11. **Vegetation** (3/4/5) - Végétation (selon hauteur/NDVI)
12. **Ground** (2) - Sol nu
13. **Unclassified** (1) - Non classé

## 🌲 Enrichissement forêts

Lorsque BD Forêt® est activé, les points en forêt reçoivent des attributs supplémentaires.

### Attributs disponibles

```python
forest_attrs = {
    'forest_type': ['coniferous', 'deciduous', 'mixed', 'unknown'],
    'primary_species': ['Pinus sylvestris', 'Quercus robur', ...],
    'secondary_species': [...],
    'estimated_height': [15.2, 18.5, 12.3, ...]  # Estimé par type
}
```

### Types de forêts

| Code         | Type              | Description       | Hauteur estimée |
| ------------ | ----------------- | ----------------- | --------------- |
| `coniferous` | Forêt de résineux | > 75% résineux    | 15-25m          |
| `deciduous`  | Forêt de feuillus | > 75% feuillus    | 10-20m          |
| `mixed`      | Forêt mixte       | Mélange équilibré | 12-22m          |
| `unknown`    | Type inconnu      | Non déterminé     | 15m (défaut)    |

### Export avec attributs forêts

```python
import laspy

# Ajouter attributs forêts au LAZ
if forest_attrs:
    # Type de forêt (encodé)
    forest_type_map = {
        'coniferous': 1,
        'deciduous': 2,
        'mixed': 3,
        'unknown': 0
    }
    las.add_extra_dim(laspy.ExtraBytesParams(
        name='forest_type',
        type=np.uint8
    ))
    las.forest_type = np.array([
        forest_type_map.get(ft, 0)
        for ft in forest_attrs['forest_type']
    ])

    # Hauteur estimée
    las.add_extra_dim(laspy.ExtraBytesParams(
        name='est_height',
        type=np.float32
    ))
    las.est_height = np.array(forest_attrs['estimated_height'])

las.write("output/tile_with_forest.laz")
```

## 🌾 Enrichissement agriculture

Avec RPG activé, les points agricoles reçoivent des informations sur les cultures.

### Attributs disponibles

```python
rpg_attrs = {
    'crop_code': ['BLE', 'MAI', 'COL', ...],
    'crop_category': ['cereals', 'oleaginous', ...],
    'crop_name': ['Wheat', 'Corn', 'Rapeseed', ...],
    'parcel_area': [12450, 8900, ...],  # m²
    'is_organic': [True, False, ...],
    'is_agricultural': [True, True, ...]
}
```

### Catégories de cultures

| Catégorie       | Description  | Codes exemples     |
| --------------- | ------------ | ------------------ |
| `cereals`       | Céréales     | BLE, ORG, MAI, BLO |
| `oleaginous`    | Oléagineux   | COL, TRN, SOR      |
| `protein_crops` | Protéagineux | POI, FEV, LUZ      |
| `fodder`        | Fourrage     | PTR, RGI, MIL      |
| `vegetables`    | Légumes      | TOM, POM, CAR      |
| `fruits`        | Fruits       | POM, CER, VIG      |
| `industrial`    | Industriels  | LIN, CHN, TAB      |
| `grassland`     | Prairies     | PPH, SPH, SPL      |
| `fallow`        | Jachères     | GEL, BRO           |
| `other`         | Autres       | -                  |

### Export CSV avec cultures

```python
import pandas as pd

if rpg_attrs:
    df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'classification': labels,
        'crop_code': rpg_attrs['crop_code'],
        'crop_category': rpg_attrs['crop_category'],
        'crop_name': rpg_attrs['crop_name'],
        'parcel_area_m2': rpg_attrs['parcel_area'],
        'is_organic': rpg_attrs['is_organic']
    })

    # Filtrer uniquement points agricoles
    df_agri = df[df['is_agricultural']]
    df_agri.to_csv("output/agricultural_points.csv", index=False)

    # Statistiques par culture
    crop_stats = df_agri.groupby('crop_category').agg({
        'x': 'count',
        'parcel_area_m2': 'mean'
    }).rename(columns={'x': 'n_points', 'parcel_area_m2': 'avg_area'})

    print(crop_stats)
```

## 🗺️ Groupement parcellaire

Le cadastre permet de grouper les points par parcelles avec statistiques.

### Groupement simple

```python
# Grouper par parcelle
parcel_groups = fetcher.cadastre_fetcher.group_points_by_parcel(
    points=points,
    parcels_gdf=data['cadastre'],
    labels=labels  # Optionnel : pour distribution des classes
)

print(f"Total parcels: {len(parcel_groups)}")

# Accéder aux statistiques d'une parcelle
parcel_id = list(parcel_groups.keys())[0]
stats = parcel_groups[parcel_id]

print(f"\nParcel {parcel_id}:")
print(f"  Points: {stats['n_points']:,}")
print(f"  Density: {stats['point_density']:.1f} pts/m²")
print(f"  Bounds: {stats['bounds']}")
print(f"  Class distribution: {stats['class_distribution']}")
```

### Export statistiques

```python
from ign_lidar.io.cadastre import export_parcel_groups_to_geojson

# Export GeoJSON
export_parcel_groups_to_geojson(
    parcel_groups=parcel_groups,
    parcels_gdf=data['cadastre'],
    output_path=Path("output/parcel_stats.geojson")
)

# Export CSV
stats_df = fetcher.cadastre_fetcher.get_parcel_statistics(
    parcel_groups=parcel_groups,
    parcels_gdf=data['cadastre']
)
stats_df.to_csv("output/parcel_stats.csv", index=False)
```

### Analyse par parcelle

```python
# Identifier parcelles avec types dominants
for parcel_id, stats in parcel_groups.items():
    dist = stats.get('class_distribution', {})
    if not dist:
        continue

    total = sum(dist.values())

    # Trouver classe dominante
    dominant_class = max(dist.items(), key=lambda x: x[1])
    class_code, count = dominant_class
    pct = 100.0 * count / total

    if pct > 60:  # Au moins 60% d'une classe
        class_names = {
            2: 'Ground',
            5: 'Vegetation',
            6: 'Building',
            44: 'Agriculture'
        }
        name = class_names.get(class_code, f'Class {class_code}')
        print(f"{parcel_id}: {name} dominant ({pct:.0f}%)")
```

## 🔄 Pipeline complet

Script complet combinant toutes les fonctionnalités :

```python
from pathlib import Path
import numpy as np
import laspy
import pandas as pd
from ign_lidar.config.loader import quick_setup

# 1. Configuration depuis YAML
config, fetcher, warnings = quick_setup(
    config_path=Path("config.yaml"),
    cache_dir=Path("cache")
)

# 2. Charger LiDAR
las = laspy.read("data/tile.laz")
points = np.vstack([las.x, las.y, las.z]).T
bbox = (
    float(las.header.x_min),
    float(las.header.y_min),
    float(las.header.x_max),
    float(las.header.y_max)
)

# 3. Récupérer toutes les données
data = fetcher.fetch_all(bbox=bbox, use_cache=True)

# 4. Classification
from ign_lidar.core.modules.advanced_classification import classify_with_all_features

labels, forest_attrs, rpg_attrs = classify_with_all_features(
    points=points,
    ground_truth_fetcher=fetcher.ground_truth_fetcher,
    bd_foret_fetcher=fetcher.forest_fetcher,
    rpg_fetcher=fetcher.rpg_fetcher,
    bbox=bbox,
    include_railways=True,
    include_forest=True,
    include_agriculture=True,
    include_bridges=True,
    include_parking=True,
    include_sports=True
)

# 5. Groupement parcellaire
parcel_groups = fetcher.cadastre_fetcher.group_points_by_parcel(
    points=points,
    parcels_gdf=data['cadastre'],
    labels=labels
)

# 6. Export LAZ
las.classification = labels
las.write("output/tile_classified.laz")

# 7. Export attributs CSV
df_data = {
    'x': points[:, 0],
    'y': points[:, 1],
    'z': points[:, 2],
    'classification': labels
}

if forest_attrs:
    df_data.update({
        'forest_type': forest_attrs['forest_type'],
        'primary_species': forest_attrs['primary_species']
    })

if rpg_attrs:
    df_data.update({
        'crop_code': rpg_attrs['crop_code'],
        'crop_category': rpg_attrs['crop_category'],
        'is_organic': rpg_attrs['is_organic']
    })

df = pd.DataFrame(df_data)
df.to_csv("output/points_with_attrs.csv", index=False)

# 8. Export parcelles GeoJSON
from ign_lidar.io.cadastre import export_parcel_groups_to_geojson

export_parcel_groups_to_geojson(
    parcel_groups=parcel_groups,
    parcels_gdf=data['cadastre'],
    output_path=Path("output/parcel_stats.geojson")
)

print("✅ Pipeline complete!")
```

## 📈 Performance

### Optimisations

- **Cache multi-niveaux** : WFS, index spatiaux, résultats
- **Traitement parallèle** : Fetchers indépendants en parallèle
- **Index spatiaux** : STRtree pour tests géométriques rapides
- **Vectorisation** : Opérations numpy vectorisées

### Benchmarks

Tile 1km × 1km, ~10M points :

| Opération               | Temps       | Cache |
| ----------------------- | ----------- | ----- |
| Fetch all data (cold)   | ~10-15s     | ❌    |
| Fetch all data (cached) | ~1-2s       | ✅    |
| Classification          | ~30-45s     | -     |
| Parcel grouping         | ~5-10s      | -     |
| Export LAZ              | ~8-12s      | -     |
| **Total**               | **~60-90s** | Mixte |

### Recommandations

1. **Activer cache** : Réutiliser entre runs
2. **Traiter par tuiles** : < 20M points par tile
3. **Désactiver sources inutiles** : Gain 20-40%
4. **Utiliser SSD** : Cache sur SSD rapide

## 🔍 Voir aussi

- [Intégration Cadastre](./CADASTRE_INTEGRATION.md) : Guide détaillé BD PARCELLAIRE
- [Intégration BD TOPO + RPG](./BD_TOPO_RPG_INTEGRATION.md) : Infrastructure et agriculture
- [BD Forêt®](./BD_FORET_INTEGRATION.md) : Types de forêts détaillés
- [Configuration YAML](../../configs/unified_classification_config.yaml) : Exemple de configuration
