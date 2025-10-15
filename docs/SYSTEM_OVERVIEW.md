# Système de Classification Multi-Sources IGN : Récapitulatif Complet

## 📊 Vue d'ensemble

Ce document récapitule l'ensemble du système de classification multi-sources pour IGN LiDAR HD Dataset, intégrant 4 bases de données géographiques nationales.

---

## 🗃️ Sources de données intégrées

### 1. BD TOPO® V3 (Infrastructure)

**10 types de features** :

| Feature     | ASPRS Code | Description            | Buffer |
| ----------- | ---------- | ---------------------- | ------ |
| Buildings   | 6          | Bâtiments              | -      |
| Roads       | 11         | Routes                 | -      |
| Railways    | 10         | Voies ferrées          | 5m     |
| Water       | 9          | Surfaces en eau        | -      |
| Vegetation  | 5          | Végétation BD TOPO     | -      |
| Bridges     | 17         | Ponts                  | -      |
| Parking     | 40         | Aires de stationnement | -      |
| Cemeteries  | 42         | Cimetières             | -      |
| Power Lines | 43         | Lignes électriques     | 10m    |
| Sports      | 41         | Terrains de sport      | -      |

**Module** : `ign_lidar/io/wfs_ground_truth.py`
**Couche WFS** : `BDTOPO_V3:*`

### 2. BD Forêt® V2 (Types de forêts)

**4 types de forêts** :

| Type       | Description               | Hauteur estimée |
| ---------- | ------------------------- | --------------- |
| Coniferous | Forêt de résineux (> 75%) | 15-25m          |
| Deciduous  | Forêt de feuillus (> 75%) | 10-20m          |
| Mixed      | Forêt mixte               | 12-22m          |
| Unknown    | Type non déterminé        | 15m             |

**Attributs enrichis** :

- Type de forêt
- 3 essences principales (essence_1, essence_2, essence_3)
- Taux de couverture par essence
- Hauteur estimée par type

**Module** : `ign_lidar/io/bd_foret.py` (510 lignes)
**Couche WFS** : `BDFORET_V2:formation_vegetale`

### 3. RPG - Registre Parcellaire Graphique (Agriculture)

**40+ codes de cultures** répartis en **10 catégories** :

| Catégorie     | Nb codes | Exemples                 |
| ------------- | -------- | ------------------------ |
| Cereals       | 5        | BLE, MAI, ORG, BLO, TRI  |
| Oleaginous    | 5        | COL, TRN, SOR, CAR, LIN  |
| Protein crops | 3        | POI, FEV, LUZ            |
| Fodder        | 6        | PTR, RGI, MIL, RGA, RGH  |
| Vegetables    | 8        | TOM, POM, CAR, SAL, etc. |
| Fruits        | 7        | POM, CER, VIG, AGR, etc. |
| Industrial    | 5        | LIN, CHN, TAB, etc.      |
| Grassland     | 8        | PPH, SPH, SPL, etc.      |
| Fallow        | 3        | GEL, BRO                 |
| Other         | -        | Cultures non classifiées |

**Attributs enrichis** :

- Code culture (3 lettres)
- Catégorie culture
- Nom complet de la culture
- Surface de la parcelle
- Agriculture biologique (oui/non)

**Module** : `ign_lidar/io/rpg.py` (420 lignes)
**Couche WFS** : `RPG.{year}:parcelles_graphiques` (2020-2023)
**ASPRS Code** : 44 (Agriculture)

### 4. BD PARCELLAIRE (Cadastre)

**Groupement par parcelles cadastrales** :

| Attribut    | Description                        |
| ----------- | ---------------------------------- |
| id_parcelle | Identifiant unique (19 caractères) |
| numero      | Numéro de parcelle                 |
| section     | Section cadastrale (2 lettres)     |
| commune     | Code INSEE commune                 |
| contenance  | Surface de la parcelle (m²)        |

**Statistiques par parcelle** :

- Nombre de points
- Densité de points (pts/m²)
- Distribution des classes ASPRS
- Bounding box

**Module** : `ign_lidar/io/cadastre.py` (450+ lignes)
**Couche WFS** : `CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle`

---

## 🎯 Codes ASPRS

### Codes standard (1-11, 17)

| Code | Nom               | Description             |
| ---- | ----------------- | ----------------------- |
| 1    | Unclassified      | Points non classés      |
| 2    | Ground            | Sol nu (géométrique)    |
| 3    | Low Vegetation    | Végétation basse < 2m   |
| 4    | Medium Vegetation | Végétation moyenne 2-5m |
| 5    | High Vegetation   | Végétation haute > 5m   |
| 6    | Building          | Bâtiments               |
| 9    | Water             | Surfaces en eau         |
| 10   | Rail              | Voies ferrées           |
| 11   | Road              | Routes et chemins       |
| 17   | Bridge            | Ponts (ASPRS standard)  |

### Codes étendus personnalisés (40-44)

| Code | Nom         | Description            |
| ---- | ----------- | ---------------------- |
| 40   | Parking     | Aires de stationnement |
| 41   | Sports      | Terrains de sport      |
| 42   | Cemetery    | Cimetières             |
| 43   | Power Line  | Lignes électriques     |
| 44   | Agriculture | Parcelles agricoles    |

---

## 🔄 Hiérarchie de classification

**Ordre de priorité décroissante** (10 niveaux) :

1. **Bridge** (17) - Structures aériennes prioritaires
2. **Building** (6) - Bâtiments
3. **Water** (9) - Surfaces en eau
4. **Power Line** (43) - Lignes électriques
5. **Rail** (10) - Voies ferrées
6. **Road** (11) - Routes
7. **Cemetery** (42) - Cimetières
8. **Parking** (40) - Parkings
9. **Sports** (41) - Terrains de sport
10. **Agriculture** (44) - Terres agricoles

Puis : Vegetation (3/4/5) → Ground (2) → Unclassified (1)

---

## 📦 Architecture modulaire

### Structure des fichiers

```
ign_lidar/
├── io/                          # Input/Output et fetchers
│   ├── wfs_ground_truth.py      # BD TOPO® (10 features)
│   ├── bd_foret.py               # BD Forêt® V2 (510 lignes)
│   ├── rpg.py                    # RPG agriculture (420 lignes)
│   ├── cadastre.py               # BD PARCELLAIRE (450+ lignes)
│   └── unified_fetcher.py        # Interface unifiée (400+ lignes)
│
├── core/modules/
│   └── advanced_classification.py # Classification multi-sources
│
└── config/
    └── loader.py                 # Chargement config YAML

configs/
└── unified_classification_config.yaml  # Configuration complète (200+ lignes)

examples/
├── example_unified_classification.py   # Script exemple complet
└── ...

docs/guides/
├── UNIFIED_SYSTEM_GUIDE.md            # Guide système unifié
├── CADASTRE_INTEGRATION.md            # Guide cadastre détaillé
├── BD_TOPO_RPG_INTEGRATION.md         # Guide BD TOPO + RPG
└── BD_FORET_INTEGRATION.md            # Guide BD Forêt
```

### Composants principaux

#### 1. UnifiedDataFetcher (unified_fetcher.py)

**Rôle** : Interface unique pour toutes les sources de données

**Fonctionnalités** :

- Configuration centralisée via `DataFetchConfig`
- Gestion automatique du cache par source
- Méthode `fetch_all()` pour récupérer toutes les données
- Méthode `process_points()` pour pipeline complet
- Logging détaillé des statistiques

**Utilisation** :

```python
from ign_lidar.io.unified_fetcher import create_full_fetcher

fetcher = create_full_fetcher(cache_dir=Path("cache"))
data = fetcher.fetch_all(bbox=bbox, use_cache=True)
```

#### 2. CadastreFetcher (cadastre.py)

**Rôle** : Récupération cadastre et groupement parcellaire

**Méthodes principales** :

- `fetch_parcels()` : Récupérer parcelles cadastrales
- `label_points_with_parcel_id()` : Labelliser points avec ID parcelle
- `group_points_by_parcel()` : Grouper points avec statistiques
- `get_parcel_statistics()` : Générer DataFrame statistiques
- `export_parcel_groups_to_geojson()` : Export GeoJSON

#### 3. RPGFetcher (rpg.py)

**Rôle** : Récupération données agricoles RPG

**Méthodes principales** :

- `fetch_parcels()` : Récupérer parcelles agricoles
- `label_points_with_crops()` : Labelliser points avec types de cultures
- Détection agriculture biologique
- Classification en 10 catégories

#### 4. BDForetFetcher (bd_foret.py)

**Rôle** : Récupération types de forêts

**Méthodes principales** :

- `fetch_forest_polygons()` : Récupérer polygones forêts
- `label_points_with_forest_type()` : Labelliser points avec types
- Extraction essences principales (3 niveaux)
- Estimation hauteur par type de forêt

#### 5. Configuration Loader (config/loader.py)

**Rôle** : Chargement et validation configuration YAML

**Fonctions principales** :

- `load_config_from_yaml()` : Charger fichier YAML
- `validate_config()` : Valider structure et valeurs
- `create_unified_fetcher_from_config()` : Créer fetcher depuis config
- `quick_setup()` : Setup complet en un appel

---

## 🚀 Utilisation

### 1. Configuration rapide (par code)

```python
from ign_lidar.io.unified_fetcher import create_full_fetcher

# Créer fetcher avec toutes les features
fetcher = create_full_fetcher(cache_dir=Path("cache"))

# Récupérer toutes les données
data = fetcher.fetch_all(bbox=bbox, use_cache=True)
```

### 2. Configuration YAML (production)

**Fichier config.yaml** :

```yaml
data_sources:
  bd_topo:
    features:
      buildings: true
      roads: true
      railways: true
      # ... autres features

  bd_foret:
    enabled: true

  rpg:
    enabled: true
    year: 2023

  cadastre:
    enabled: true
    group_by_parcel: true

cache:
  directory: "cache"
  ttl:
    bd_topo: 86400
    bd_foret: 604800
    rpg: 2592000
    cadastre: 2592000
```

**Utilisation** :

```python
from ign_lidar.config.loader import quick_setup

config, fetcher, warnings = quick_setup(
    config_path=Path("config.yaml")
)

data = fetcher.fetch_all(bbox=bbox)
```

### 3. Pipeline complet

```python
from pathlib import Path
import numpy as np
import laspy
from ign_lidar.config.loader import quick_setup
from ign_lidar.core.modules.advanced_classification import classify_with_all_features

# 1. Setup
config, fetcher, warnings = quick_setup(Path("config.yaml"))

# 2. Charger LiDAR
las = laspy.read("data/tile.laz")
points = np.vstack([las.x, las.y, las.z]).T
bbox = (las.header.x_min, las.header.y_min,
        las.header.x_max, las.header.y_max)

# 3. Fetch data
data = fetcher.fetch_all(bbox=bbox, use_cache=True)

# 4. Classification
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

# 6. Export
las.classification = labels
las.write("output/tile_classified.laz")

# Export statistiques parcellaires
from ign_lidar.io.cadastre import export_parcel_groups_to_geojson
export_parcel_groups_to_geojson(
    parcel_groups=parcel_groups,
    parcels_gdf=data['cadastre'],
    output_path=Path("output/parcel_stats.geojson")
)
```

---

## 📤 Formats de sortie

### 1. LAZ/LAS (Point Cloud)

**Contenu** :

- Classification ASPRS (1-11, 17, 40-44)
- Dimensions extra optionnelles :
  - `forest_type` : Type de forêt (uint8)
  - `crop_code` : Code culture (string)
  - `parcel_id` : ID parcelle cadastrale (string)
  - `est_height` : Hauteur estimée (float32)

### 2. GeoJSON (Parcelles)

**Contenu** :

- Géométries des parcelles cadastrales
- Statistiques par parcelle :
  - Nombre de points
  - Densité (pts/m²)
  - Distribution des classes
  - Bounds

**Utilisation** : Visualisation dans QGIS, analyses SIG

### 3. CSV (Attributs détaillés)

**Colonnes** :

- `x`, `y`, `z` : Coordonnées
- `classification` : Code ASPRS
- `forest_type` : Type de forêt
- `primary_species` : Essence principale
- `crop_code` : Code culture
- `crop_category` : Catégorie culture
- `crop_name` : Nom culture
- `is_organic` : Agriculture bio
- `parcel_id` : ID parcelle

**Utilisation** : Analyses statistiques, machine learning

---

## 📈 Performance

### Benchmarks (tile 1km × 1km, 10M points)

| Opération              | Temps (cold) | Temps (cached) |
| ---------------------- | ------------ | -------------- |
| Fetch BD TOPO®         | ~3-5s        | ~0.2s          |
| Fetch BD Forêt®        | ~2-4s        | ~0.2s          |
| Fetch RPG              | ~2-4s        | ~0.2s          |
| Fetch Cadastre         | ~3-6s        | ~0.3s          |
| **Total fetch**        | **~10-19s**  | **~1-2s**      |
| Classification         | ~30-45s      | -              |
| Groupement parcellaire | ~5-10s       | -              |
| Export LAZ             | ~8-12s       | -              |
| **TOTAL PIPELINE**     | **~60-90s**  | **~50-70s**    |

### Optimisations

- ✅ Cache WFS par source avec TTL configurable
- ✅ Index spatiaux (STRtree) pour tests géométriques
- ✅ Vectorisation numpy pour opérations massives
- ✅ Fetchers indépendants (parallélisables)
- ✅ Cache résultats de classification

---

## 📚 Documentation

### Guides utilisateur

1. **[UNIFIED_SYSTEM_GUIDE.md](./UNIFIED_SYSTEM_GUIDE.md)** : Guide complet du système unifié
2. **[CADASTRE_INTEGRATION.md](./CADASTRE_INTEGRATION.md)** : Guide détaillé cadastre
3. **[BD_TOPO_RPG_INTEGRATION.md](./BD_TOPO_RPG_INTEGRATION.md)** : Guide BD TOPO + RPG
4. **[BD_FORET_INTEGRATION.md](./BD_FORET_INTEGRATION.md)** : Guide BD Forêt

### Fichiers de configuration

1. **[unified_classification_config.yaml](../../configs/unified_classification_config.yaml)** : Configuration complète

### Scripts exemples

1. **[example_unified_classification.py](../../examples/example_unified_classification.py)** : Pipeline complet

---

## 🔧 Maintenance et évolution

### Version actuelle

- **BD TOPO®** : V3 (10 features)
- **BD Forêt®** : V2
- **RPG** : 2020-2023
- **BD PARCELLAIRE** : Parcellaire Express (courant)

### Mises à jour prévues

1. **BD TOPO®** : Ajouter features supplémentaires (zones industrielles, aérodromes)
2. **RPG** : Support année 2024+ quand disponible
3. **Cadastre** : Intégration bâtiments cadastraux
4. **Performance** : Parallélisation automatique des fetchers

### Contact et support

- **Issues** : GitHub Issues pour bugs et suggestions
- **Documentation** : docs/guides/ pour guides détaillés
- **Exemples** : examples/ pour scripts d'exemple

---

## ✅ Checklist de déploiement

### Installation

- [ ] Python 3.8+
- [ ] `pip install numpy geopandas shapely laspy requests`
- [ ] `pip install pandas pyyaml` (optionnel)

### Configuration

- [ ] Créer fichier `config.yaml` depuis template
- [ ] Configurer sources de données (BD TOPO, BD Forêt, RPG, Cadastre)
- [ ] Définir répertoire cache
- [ ] Ajuster priorité classification si nécessaire

### Test

- [ ] Exécuter `examples/example_unified_classification.py`
- [ ] Vérifier fichiers de sortie (LAZ + GeoJSON + CSV)
- [ ] Visualiser dans QGIS/CloudCompare
- [ ] Valider statistiques parcellaires

### Production

- [ ] Activer cache pour toutes les sources
- [ ] Configurer logging approprié
- [ ] Définir batch processing si traitement massif
- [ ] Monitorer performance (temps, mémoire)

---

**Date de création** : 15 octobre 2025
**Version** : 2.0
**Auteur** : Data Integration Team
