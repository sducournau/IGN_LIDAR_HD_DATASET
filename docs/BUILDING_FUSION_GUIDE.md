# Guide Complet: Fusion Multi-Sources de Bâtiments

## 🏢 Vue d'ensemble

Ce guide décrit le système avancé de fusion de sources multiples pour la détection et la reconstruction de bâtiments:

- **Comparaison multi-sources** : BD TOPO®, Cadastre, OpenStreetMap
- **Scoring de qualité** : Évaluation automatique de chaque polygone
- **Fusion intelligente** : Sélection du meilleur polygone ou fusion pondérée
- **Adaptation adaptative** : Translation, mise à l'échelle, rotation, buffer
- **Résolution de conflits** : Fusion de bâtiments chevauchants

---

## 📊 Sources de Données

### 1. BD TOPO® (Source Principale)

**Caractéristiques:**

- Source officielle IGN
- Haute précision géométrique
- Mise à jour régulière
- Couverture exhaustive

**Qualité typique:**

- Score moyen: 0.75-0.85
- Couverture: 60-80% des points
- Précision centroïde: ±2m

**Utilisation:**

```yaml
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
```

---

### 2. Cadastre (Source Secondaire)

**Caractéristiques:**

- Parcelles cadastrales (limites légales)
- Géométrie très précise
- Peut différer des bâtiments réels
- Complète BD TOPO où manquant

**Qualité typique:**

- Score moyen: 0.60-0.70
- Couverture: 50-70% des points
- Décalage fréquent (parcelle ≠ bâtiment)

**Utilisation:**

```yaml
data_sources:
  cadastre:
    enabled: true
    use_as_building_proxy: true
```

---

### 3. OpenStreetMap (Source Tertiaire) ⭐ NOUVEAU

**Caractéristiques:**

- Données communautaires
- Qualité variable selon la zone
- Souvent à jour dans les zones urbaines
- Complète les sources officielles

**Qualité typique:**

- Score moyen: 0.50-0.75
- Couverture: 40-70% des points
- Bonne qualité en zones urbaines
- Qualité variable en zones rurales

**Configuration:**

```yaml
data_sources:
  osm:
    enabled: true
    overpass_url: "https://overpass-api.de/api/interpreter"
    timeout: 180

    building_tags:
      - "building=yes"
      - "building=house"
      - "building=residential"
      - "building=apartments"

    # Filtres de qualité
    min_building_area: 10.0 # m²
    max_building_area: 10000.0 # m²

    cache_enabled: true
    cache_ttl_days: 30
```

---

## 🔍 Scoring de Qualité

### Métriques de Qualité

Chaque polygone est évalué selon plusieurs critères:

#### 1. Couverture (Coverage) - 40% du score

```python
coverage_ratio = points_inside_polygon / total_building_points
```

**Critères:**

- Points à l'intérieur du polygone
- Points à proximité (buffer 1m)
- Ratio de couverture (0-1)

**Exemple:**

```
BD TOPO: 850/1000 points = 0.85
Cadastre: 720/1000 points = 0.72
OSM: 780/1000 points = 0.78
→ BD TOPO meilleur score
```

---

#### 2. Ajustement Géométrique (Geometric Fit) - 30% du score

**Décalage centroïde:**

```python
centroid_offset = distance(polygon_centroid, point_cloud_centroid)
penalty = exp(-offset / 2.0)
```

**Ratio de surface:**

```python
area_ratio = polygon_area / point_cloud_area
penalty = 1.0 - abs(1.0 - area_ratio)
```

**Similarité de forme:**

- IoU (Intersection over Union)
- Concordance des contours

**Exemple:**

```
BD TOPO:
  - Décalage: 1.2m → penalty = 0.88
  - Ratio surface: 1.1 → penalty = 0.90
  - IoU: 0.75
  → Score géométrique: (0.88 + 0.90 + 0.75) / 3 = 0.84

Cadastre:
  - Décalage: 3.5m → penalty = 0.55
  - Ratio surface: 0.8 → penalty = 0.80
  - IoU: 0.60
  → Score géométrique: 0.65
```

---

#### 3. Complétude (Completeness) - 30% du score

**Couverture des murs:**

```python
wall_coverage = wall_points_inside / total_wall_points
# Murs détectés par verticality >= 0.7
```

**Couverture des toits:**

```python
roof_coverage = roof_points_inside / total_roof_points
# Toits détectés par verticality < 0.7
```

**Exemple:**

```
BD TOPO:
  - Murs: 320/350 = 0.91
  - Toits: 530/650 = 0.82
  → Score complétude: 0.87

OSM:
  - Murs: 280/350 = 0.80
  - Toits: 500/650 = 0.77
  → Score complétude: 0.78
```

---

### Score Global

```python
quality_score = (
    0.4 * coverage_score +
    0.3 * geometric_score +
    0.3 * completeness_score
)
```

**Interprétation:**

- **> 0.80**: Excellente qualité
- **0.60-0.80**: Bonne qualité
- **0.50-0.60**: Qualité acceptable
- **< 0.50**: Qualité insuffisante (rejeté)

---

## 🔀 Stratégies de Fusion

### 1. Mode "best" (Recommandé)

**Principe:** Sélectionner le polygone avec le meilleur score

**Algorithme:**

```python
# 1. Trier par priorité (BD TOPO > Cadastre > OSM)
# 2. Sélectionner le meilleur score si > seuil qualité
# 3. Basculer vers source inférieure seulement si différence > 0.15
```

**Configuration:**

```yaml
building_fusion:
  fusion_mode: "best"
  source_priority:
    - "bd_topo"
    - "cadastre"
    - "osm"
  min_quality_score: 0.5
  quality_difference_threshold: 0.15
```

**Exemple:**

```
BD TOPO: 0.78
Cadastre: 0.85 (meilleur +0.07)
OSM: 0.62

Résultat: BD TOPO (différence < 0.15, priorité respectée)

Mais si:
BD TOPO: 0.65
Cadastre: 0.85 (meilleur +0.20)

Résultat: Cadastre (différence > 0.15, basculement)
```

---

### 2. Mode "weighted_merge" (Fusion Pondérée)

**Principe:** Fusionner plusieurs polygones selon leurs scores

**Algorithme:**

```python
# 1. Filtrer sources avec score > 0.5
# 2. Union géométrique pondérée par qualité
# 3. Simplification du polygone résultant
```

**Configuration:**

```yaml
building_fusion:
  fusion_mode: "weighted_merge"
  enable_multi_source_fusion: true
```

**Résultat:**

- Polygone plus large (union)
- Capture plus de points
- Peut inclure des zones non-bâtiment

**Usage:** Zones avec données conflictuelles

---

### 3. Mode "consensus" (Conservateur)

**Principe:** Intersection des polygones de bonne qualité

**Algorithme:**

```python
# 1. Filtrer sources avec score > 0.5
# 2. Intersection géométrique
# 3. Si intersection trop petite, fallback vers weighted_merge
```

**Configuration:**

```yaml
building_fusion:
  fusion_mode: "consensus"
```

**Résultat:**

- Polygone plus petit (intersection)
- Très haute confiance
- Peut manquer des extensions

**Usage:** Applications critiques nécessitant haute précision

---

## 🔧 Adaptation Adaptative des Polygones

### 1. Translation (Déplacement)

**Principe:** Déplacer le polygone vers le centroïde des points

**Algorithme:**

```python
# 1. Calculer centroïde du nuage de points
# 2. Calculer centroïde du polygone
# 3. Déplacer si offset > 0.5m et < max_translation
```

**Configuration:**

```yaml
building_fusion:
  enable_translation: true
  max_translation: 5.0 # mètres
```

**Exemple:**

```
Centroïde polygone: (100.0, 200.0)
Centroïde points: (102.5, 201.8)
Offset: 2.9m

→ Translation appliquée: (+2.5m, +1.8m)
```

**Résultat:**

- Meilleure couverture: +10-20%
- Alignement centroïde parfait
- Corrections des décalages GPS/projection

---

### 2. Scaling (Mise à l'Échelle)

**Principe:** Ajuster la taille au nuage de points

**Algorithme:**

```python
# 1. Calculer extent points (largeur × hauteur)
# 2. Calculer extent polygone
# 3. Scale_factor = extent_points / extent_polygon
# 4. Clamper entre 1/max_scale et max_scale
```

**Configuration:**

```yaml
building_fusion:
  enable_scaling: true
  max_scale_factor: 1.5 # 1.5x max expansion/contraction
```

**Exemple:**

```
Polygone: 20m × 15m
Points: 24m × 18m

Scale X: 24/20 = 1.20
Scale Y: 18/15 = 1.20
Scale moyen: 1.20

→ Scaling appliqué: 1.20x (expansion 20%)
```

**Résultat:**

- Capture des murs périphériques
- Meilleure correspondance avec la réalité
- Correction des polygones sous-dimensionnés

---

### 3. Rotation (Alignment)

**Principe:** Aligner avec les axes principaux du nuage de points

**Algorithme:**

```python
# 1. PCA sur les points (axes principaux)
# 2. Calculer angle de rotation
# 3. Appliquer si |angle| > 1° et < max_rotation
```

**Configuration:**

```yaml
building_fusion:
  enable_rotation: false # Désactivé par défaut (coûteux)
  max_rotation: 15.0 # degrés
```

**⚠️ Attention:**

- Très coûteux en CPU (PCA)
- Peu de bénéfice pour bâtiments réguliers
- Utile uniquement pour bâtiments désalignés

**Usage recommandé:** Désactiver sauf cas spécifiques

---

### 4. Buffering Adaptatif

**Principe:** Buffer variable selon la détection de murs

**Algorithme:**

```python
# 1. Détecter les murs (verticality >= 0.7)
# 2. Calculer ratio de murs
# 3. Buffer adaptatif:
buffer = min_buffer + (max_buffer - min_buffer) * wall_ratio
```

**Configuration:**

```yaml
building_fusion:
  enable_buffering: true
  adaptive_buffer_range: [0.3, 1.0] # min/max mètres
```

**Exemple:**

```
Points totaux: 1000
Points murs (verticality >= 0.7): 350
Wall ratio: 0.35

Buffer = 0.3 + (1.0 - 0.3) × 0.35 = 0.55m

→ Buffer appliqué: 0.55m
```

**Résultat:**

- Buffer faible (0.3m) pour toits plats
- Buffer élevé (0.8-1.0m) pour murs nombreux
- Capture optimale des points muraux

---

## 🔗 Résolution de Conflits

### 1. Détection de Chevauchements

**Méthode: IoU (Intersection over Union)**

```python
intersection = polygon1.intersection(polygon2).area
union = polygon1.union(polygon2).area
iou = intersection / union

if iou >= overlap_threshold:  # 0.3 par défaut
    # Conflit détecté
```

**Configuration:**

```yaml
building_fusion:
  overlap_threshold: 0.3 # IoU 30%
```

---

### 2. Fusion de Bâtiments Proches

**Critères:**

- Distance < 2m
- IoU >= 0.3
- Maximum 2 bâtiments à fusionner

**Algorithme:**

```python
# 1. Détecter bâtiments proches
# 2. Union géométrique
# 3. Simplification du polygone
# 4. Cumul des points
```

**Configuration:**

```yaml
building_fusion:
  merge_nearby_buildings: true
  merge_distance_threshold: 2.0 # mètres
```

**Exemple:**

```
Bâtiment A: 850 points, polygone 200m²
Bâtiment B: 320 points, polygone 80m²
Distance: 1.5m
IoU: 0.15 (pas de chevauchement significatif)

→ Fusion appliquée:
  - Polygone: union(A, B) = 275m²
  - Points: 1170 points
  - Source: FUSED
```

---

### 3. Suppression de Doublons

**Critères:**

- IoU > 0.7 (chevauchement important)
- Garder le bâtiment avec plus de points

**Algorithme:**

```python
# 1. Trier par nombre de points
# 2. Supprimer bâtiments fortement chevauchés
```

---

## 📈 Pipeline Complet

### Exemple Intégré

```python
from ign_lidar.core.classification.building_fusion import (
    BuildingFusion, BuildingSource
)

# 1. Charger les données
points, colors = load_point_cloud("tile.laz")
normals = compute_normals(points, k_neighbors=30)
verticality = compute_verticality(normals)

# 2. Charger les sources de bâtiments
building_sources = {
    BuildingSource.BD_TOPO: load_bd_topo(bbox),
    BuildingSource.CADASTRE: load_cadastre(bbox),
    BuildingSource.OSM: load_osm(bbox)
}

# 3. Créer le système de fusion
fusion = BuildingFusion(
    source_priority=[
        BuildingSource.BD_TOPO,
        BuildingSource.CADASTRE,
        BuildingSource.OSM
    ],
    fusion_mode="best",

    # Adaptation
    enable_translation=True,
    enable_scaling=True,
    enable_rotation=False,
    enable_buffering=True,

    max_translation=5.0,
    max_scale_factor=1.5,
    adaptive_buffer_range=(0.3, 1.0),

    # Résolution
    merge_nearby_buildings=True,
    overlap_threshold=0.3
)

# 4. Fusionner les bâtiments
fused_buildings, stats = fusion.fuse_building_sources(
    points=points,
    building_sources=building_sources,
    normals=normals,
    verticality=verticality
)

# 5. Analyser les résultats
print(f"Bâtiments fusionnés: {len(fused_buildings)}")
print(f"Points totaux: {sum(b.n_points for b in fused_buildings):,}")

print("\nSources utilisées:")
for source, count in stats['sources_used'].items():
    print(f"  {source}: {count} bâtiments")

print("\nAdaptations:")
print(f"  Translatés: {stats['adaptations']['translated']}")
print(f"  Mis à l'échelle: {stats['adaptations']['scaled']}")
print(f"  Pivotés: {stats['adaptations']['rotated']}")
print(f"  Bufferisés: {stats['adaptations']['buffered']}")

# 6. Sauvegarder les résultats
save_fused_buildings("output_fused.geojson", fused_buildings)
save_fusion_report("fusion_report.json", stats)
```

---

## 📊 Résultats Attendus

### Distribution des Sources (zone urbaine typique)

```
Entrées:
  BD TOPO: 250 bâtiments
  Cadastre: 380 parcelles
  OSM: 180 bâtiments

Sortie fusionnée: 265 bâtiments

Provenance:
  BD TOPO: 185 (70%)
  Cadastre: 50 (19%)
  OSM: 25 (9%)
  FUSED: 5 (2%)
```

### Scores de Qualité Moyens

| Source       | Score Moyen | Score Min | Score Max |
| ------------ | ----------- | --------- | --------- |
| **BD TOPO**  | 0.78        | 0.55      | 0.92      |
| **Cadastre** | 0.66        | 0.42      | 0.85      |
| **OSM**      | 0.61        | 0.35      | 0.88      |

### Adaptations Appliquées (pour 100 bâtiments)

| Adaptation      | Nombre | Amélioration Moyenne |
| --------------- | ------ | -------------------- |
| **Translation** | 72     | +15% couverture      |
| **Scaling**     | 58     | +10% couverture      |
| **Rotation**    | 12     | +5% couverture       |
| **Buffering**   | 85     | +20% points murs     |

### Performance (RTX 4080, tuile 18.6M points)

| Étape                    | Temps                   |
| ------------------------ | ----------------------- |
| Téléchargement sources   | 2-3 min (première fois) |
| Correspondance polygones | 30 sec                  |
| Scoring qualité          | 1-2 min                 |
| Adaptation polygones     | 1-2 min                 |
| Résolution conflits      | 30 sec                  |
| **Total**                | **8-12 min/tuile**      |

---

## 🎯 Cas d'Usage

### 1. Zones Urbaines Hétérogènes

**Problème:** BD TOPO incomplet, Cadastre décalé, OSM variable

**Solution:**

```yaml
fusion_mode: "best"
enable_multi_source_fusion: true
quality_difference_threshold: 0.10 # Plus permissif
```

**Résultat:** Meilleure couverture avec sources complémentaires

---

### 2. Zones Historiques

**Problème:** BD TOPO peut être obsolète, OSM plus à jour

**Solution:**

```yaml
source_priority:
  - "osm" # Prioriser OSM
  - "bd_topo"
  - "cadastre"
min_quality_score: 0.4 # Plus permissif
```

**Résultat:** Intégration des bâtiments récents d'OSM

---

### 3. Zones Rurales

**Problème:** Peu de données OSM, Cadastre = parcelles agricoles

**Solution:**

```yaml
source_priority:
  - "bd_topo" # Seule source fiable
  - "cadastre"
min_quality_score: 0.6 # Plus strict
```

**Résultat:** Focus sur BD TOPO de haute qualité

---

## 🔧 Configuration Avancée

### Ajustement par Type de Zone

```yaml
# Zone urbaine dense
building_fusion:
  max_translation: 3.0  # Décalages GPS fréquents
  adaptive_buffer_range: [0.3, 0.8]  # Buffer modéré
  merge_nearby_buildings: true

# Zone rurale dispersée
building_fusion:
  max_translation: 5.0  # Plus de flexibilité
  adaptive_buffer_range: [0.5, 1.2]  # Buffer large
  merge_nearby_buildings: false
```

---

## 📚 Références

- **Module principal**: `ign_lidar/core/classification/building_fusion.py`
- **Configuration**: `examples/config_building_fusion.yaml`
- **Documentation API**: `docs/API_BUILDING_FUSION.md`

---

**Version**: 5.1.0  
**Date**: 19 octobre 2025  
**Status**: ✅ Implémentation complète
