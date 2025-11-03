---
sidebar_position: 2
title: Plane Detection
description: Detection of horizontal, vertical and inclined planes for architectural analysis
tags: [plane-detection, roofs, walls, facades, architecture]
---

# 🏗️ Plane Detection Guide

## Overview

Ce guide décrit le système complet de détection de plans pour la classification architecturale:

- **Plans horizontaux** (toits plats, terrasses, dalles)
- **Plans verticaux** (murs, façades, pignons)
- **Plans inclinés** (toits en pente, versants)
- **Éléments architecturaux** (lucarnes, cheminées, balcons)

---

## 📐 Détection de Plans

### 1. Plans Horizontaux (Toits Plats, Terrasses)

**Caractéristiques:**

- Normale presque verticale (nz ≈ ±1)
- Angle < 10° par rapport à l'horizontale
- Haute planarité (surface lisse et plane)
- Cohérence spatiale
- Hauteur typique > 2m (toits)

**Code:**

```python
from ign_lidar.core.classification.plane_detection import PlaneDetector

detector = PlaneDetector(
    horizontal_angle_max=10.0,  # ±10° = horizontal
    horizontal_planarity_min=0.75
)

horizontal_planes = detector.detect_horizontal_planes(
    points=points,
    normals=normals,
    planarity=planarity,
    height=height,
    min_height=2.0  # Toits > 2m
)

# Analyser les résultats
for plane in horizontal_planes:
    print(f"Toit plat détecté:")
    print(f"  Points: {plane.n_points:,}")
    print(f"  Surface: {plane.area:.1f} m²")
    print(f"  Hauteur: {plane.height_mean:.1f}m")
    print(f"  Planarité: {plane.planarity:.3f}")
```

**Applications:**

- Détection de toits-terrasses
- Identification de dalles de béton
- Surfaces de stationnement
- Zones plates pour panneaux solaires

---

### 2. Plans Verticaux (Murs, Façades)

**Caractéristiques:**

- Normale presque horizontale (nz ≈ 0)
- Angle ≥ 75° par rapport à l'horizontale
- Bonne planarité (surface de mur plane)
- Extension verticale (variation en hauteur)

**Code:**

```python
detector = PlaneDetector(
    vertical_angle_min=75.0,  # ≥75° = vertical
    vertical_planarity_min=0.65
)

vertical_planes = detector.detect_vertical_planes(
    points=points,
    normals=normals,
    planarity=planarity,
    height=height,
    min_height=0.5  # Murs > 0.5m
)

# Analyser les murs
for plane in vertical_planes:
    print(f"Mur détecté:")
    print(f"  Points: {plane.n_points:,}")
    print(f"  Surface: {plane.area:.1f} m²")
    print(f"  Hauteur moyenne: {plane.height_mean:.1f}m")
    print(f"  Variation hauteur: ±{plane.height_std:.2f}m")

    # Orientation du mur (direction de la normale)
    nx, ny = plane.normal[0], plane.normal[1]
    azimuth = np.degrees(np.arctan2(ny, nx))
    print(f"  Orientation: {azimuth:.1f}°")
```

**Buffers étendus pour les murs:**

```python
from ign_lidar.core.classification.building import BuildingClusterer

clusterer = BuildingClusterer(
    polygon_buffer=0.5,  # Buffer de base
    wall_buffer=0.3,     # Buffer additionnel pour murs (ASPRS)
    detect_near_vertical_walls=True
)

# Total buffer: 0.5 + 0.3 = 0.8m pour capturer les points jusqu'aux limites
```

**Applications:**

- Détection de façades
- Reconstruction 3D de bâtiments
- Analyse d'orientations (ensoleillement)
- Détection de murs de soutènement

---

### 3. Plans Inclinés (Toits en Pente)

**Caractéristiques:**

- Normale à angle intermédiaire (15° < angle < 70°)
- Bonne planarité (surface de toit lisse)
- Hauteur typique de toit
- Angles courants: 30-45° pour toits en pente

**Code:**

```python
detector = PlaneDetector(
    inclined_angle_min=15.0,   # Angle min
    inclined_angle_max=70.0,   # Angle max
    inclined_planarity_min=0.70
)

inclined_planes = detector.detect_inclined_planes(
    points=points,
    normals=normals,
    planarity=planarity,
    height=height,
    min_height=2.0  # Toits > 2m
)

# Analyser les toits en pente
for plane in inclined_planes:
    print(f"Toit en pente détecté:")
    print(f"  Points: {plane.n_points:,}")
    print(f"  Surface: {plane.area:.1f} m²")
    print(f"  Pente: {plane.orientation_angle:.1f}°")

    # Classification de la pente
    if plane.orientation_angle < 20:
        print(f"  Type: Pente faible")
    elif plane.orientation_angle < 35:
        print(f"  Type: Pente moyenne (30-35°)")
    else:
        print(f"  Type: Pente forte (>35°)")
```

**Applications:**

- Détection de toitures traditionnelles
- Calcul de surfaces de toiture
- Estimation de pente pour panneaux solaires
- Classification de types de toits

---

## 🏠 Classification de Types de Toits

### Détection Automatique

```python
# Détecter tous les plans
planes = detector.detect_all_planes(points, normals, planarity, height)

# Classifier les types de toits
roof_types = detector.classify_roof_types(
    horizontal_planes=planes['horizontal'],
    inclined_planes=planes['inclined']
)

# Analyser par type
for roof_type, roof_planes in roof_types.items():
    if roof_planes:
        print(f"\n{roof_type.upper()} ROOF:")
        print(f"  Nombre de plans: {len(roof_planes)}")

        total_area = sum(p.area for p in roof_planes)
        print(f"  Surface totale: {total_area:.1f} m²")
```

### Types de Toits Détectés

| Type                      | Caractéristiques             | Détection         |
| ------------------------- | ---------------------------- | ----------------- |
| **Toit plat**             | Uniquement plans horizontaux | 0 plans inclinés  |
| **Toit à 2 pans** (gable) | 2 versants inclinés          | 2 plans inclinés  |
| **Toit à 4 pans** (hip)   | 4+ versants inclinés         | ≥4 plans inclinés |
| **Toit complexe**         | Mélange horizontal + incliné | Plans mixtes      |

---

## 🏛️ Détection d'Éléments Architecturaux

### Éléments Détectables

```python
from ign_lidar.core.classification.plane_detection import detect_architectural_elements

elements = detect_architectural_elements(
    points=points,
    normals=normals,
    planarity=planarity,
    height=height,
    planes=planes
)

# Analyser les éléments
for elem_type, elem_list in elements.items():
    print(f"\n{elem_type.upper()}:")
    print(f"  Nombre détecté: {len(elem_list)}")

    if elem_list:
        total_points = sum(len(indices) for indices in elem_list)
        print(f"  Points totaux: {total_points:,}")
```

### 1. Balcons

**Critères:**

- Plans horizontaux de petite taille (&lt;20 m²)
- Hauteur intermédiaire (2-15m)
- Peu de points (&lt;500)
- Projection depuis façade

**Code:**

```python
# Les balcons sont automatiquement détectés
balconies = elements['balconies']

for balcony_indices in balconies:
    balcony_points = points[balcony_indices]
    print(f"Balcon: {len(balcony_indices)} points à {balcony_points[:, 2].mean():.1f}m")
```

### 2. Cheminées

**Critères:**

- Plans verticaux de petite taille (&lt;10 m²)
- Hauteur élevée (>8m)
- Peu de points (&lt;300)
- Sur toiture

**Code:**

```python
chimneys = elements['chimneys']

for chimney_indices in chimneys:
    chimney_points = points[chimney_indices]
    print(f"Cheminée: {len(chimney_indices)} points à {chimney_points[:, 2].mean():.1f}m")
```

### 3. Lucarnes (Dormers)

**Critères:**

- Plans verticaux au-dessus de toits inclinés
- Dépassement > 1m au-dessus du toit
- Structure saillante

**Code:**

```python
dormers = elements['dormers']

for dormer_indices in dormers:
    dormer_points = points[dormer_indices]
    print(f"Lucarne: {len(dormer_indices)} points")
```

### 4. Parapets (Garde-corps)

**Critères:**

- Plans verticaux bas (0.5-2m de haut)
- Juste au-dessus de toits plats
- Peu de points (&lt;200)
- Élément de sécurité

**Code:**

```python
parapets = elements['parapets']

for parapet_indices in parapets:
    parapet_points = points[parapet_indices]
    print(f"Parapet: {len(parapet_indices)} points")
```

---

## 🔧 Pipeline Complet

### Exemple Intégré

```python
from ign_lidar.core.classification.plane_detection import PlaneDetector, detect_architectural_elements
from ign_lidar.features.compute.features import compute_normals

# 1. Charger les données
points, colors = load_point_cloud("building.laz")
height = compute_height_above_ground(points)

# 2. Calculer les features
normals = compute_normals(points, k_neighbors=20)
planarity = compute_planarity(points, k_neighbors=20)

# 3. Créer le détecteur
detector = PlaneDetector(
    horizontal_angle_max=10.0,
    vertical_angle_min=75.0,
    inclined_angle_min=15.0,
    inclined_angle_max=70.0,
    min_points_per_plane=50
)

# 4. Détecter tous les plans
planes = detector.detect_all_planes(points, normals, planarity, height)

# 5. Classifier les toits
roof_types = detector.classify_roof_types(
    planes['horizontal'],
    planes['inclined']
)

# 6. Détecter les éléments architecturaux
elements = detect_architectural_elements(
    points, normals, planarity, height, planes
)

# 7. Créer des labels par type de plan
labels = np.zeros(len(points), dtype=np.uint8)

# Horizontal = 1
for plane in planes['horizontal']:
    labels[plane.point_indices] = 1

# Vertical = 2
for plane in planes['vertical']:
    labels[plane.point_indices] = 2

# Incliné = 3
for plane in planes['inclined']:
    labels[plane.point_indices] = 3

# Balcons = 10, Cheminées = 11, etc.
for balcony_indices in elements['balconies']:
    labels[balcony_indices] = 10

for chimney_indices in elements['chimneys']:
    labels[chimney_indices] = 11

# 8. Sauvegarder
save_classified_las("output.laz", points, labels, colors)
```

---

## 📊 Résultats et Statistiques

### Visualisation des Résultats

```python
import matplotlib.pyplot as plt

# Compter les points par type
n_horizontal = sum(p.n_points for p in planes['horizontal'])
n_vertical = sum(p.n_points for p in planes['vertical'])
n_inclined = sum(p.n_points for p in planes['inclined'])

# Graphique
plt.figure(figsize=(10, 6))
plt.bar(['Horizontal\n(Toits plats)', 'Vertical\n(Murs)', 'Incliné\n(Toits pente)'],
        [n_horizontal, n_vertical, n_inclined])
plt.ylabel('Nombre de points')
plt.title('Distribution des types de plans')
plt.show()

# Surface par type de toit
roof_areas = {
    roof_type: sum(p.area for p in roof_planes)
    for roof_type, roof_planes in roof_types.items()
    if roof_planes
}

for roof_type, area in roof_areas.items():
    print(f"{roof_type}: {area:.1f} m²")
```

### Métriques de Qualité

```python
# Planarité moyenne par type
for plane_type, plane_list in planes.items():
    if plane_list:
        avg_planarity = np.mean([p.planarity for p in plane_list])
        print(f"{plane_type}: planarité moyenne = {avg_planarity:.3f}")

# Distribution des angles
angles = [p.orientation_angle for p in planes['inclined']]
if angles:
    print(f"\nAngles de pente:")
    print(f"  Min: {min(angles):.1f}°")
    print(f"  Max: {max(angles):.1f}°")
    print(f"  Moyenne: {np.mean(angles):.1f}°")
```

---

## 🎯 Cas d'Usage

### 1. Reconstruction 3D LOD2

```python
# Séparer murs et toits pour modélisation
walls = []
roofs = []

for plane in planes['vertical']:
    walls.append(plane)

for plane in planes['horizontal'] + planes['inclined']:
    roofs.append(plane)

print(f"Reconstruction LOD2:")
print(f"  {len(walls)} murs")
print(f"  {len(roofs)} toits")
```

### 2. Analyse Énergétique

```python
# Calculer surface de toit pour panneaux solaires
flat_roof_area = sum(p.area for p in planes['horizontal'] if p.height_mean > 5.0)
inclined_roof_area = sum(p.area for p in planes['inclined'] if 20 < p.orientation_angle < 45)

print(f"Surface exploitable (toits plats): {flat_roof_area:.1f} m²")
print(f"Surface exploitable (toits inclinés 20-45°): {inclined_roof_area:.1f} m²")
print(f"Total potentiel: {flat_roof_area + inclined_roof_area:.1f} m²")
```

### 3. Diagnostic Patrimonial

```python
# Identifier éléments architecturaux remarquables
print("Éléments architecturaux détectés:")
for elem_type in ['chimneys', 'dormers', 'parapets']:
    count = len(elements[elem_type])
    if count > 0:
        print(f"  - {count} {elem_type}")
```

---

## ⚙️ Configuration Avancée

### Ajustement des Seuils

```python
# Pour bâtiments modernes (plans très nets)
detector_modern = PlaneDetector(
    horizontal_planarity_min=0.85,  # Plus strict
    vertical_planarity_min=0.75,
    inclined_planarity_min=0.80
)

# Pour bâtiments anciens (surfaces irrégulières)
detector_historic = PlaneDetector(
    horizontal_planarity_min=0.60,  # Plus permissif
    vertical_planarity_min=0.50,
    inclined_planarity_min=0.55
)

# Pour toits avec végétation
detector_vegetated = PlaneDetector(
    horizontal_planarity_min=0.50,  # Très permissif
    min_points_per_plane=100  # Plus de points requis
)
```

### Segmentation Spatiale

```python
detector = PlaneDetector(
    max_plane_distance=0.15,  # Distance max au plan (15cm)
    use_spatial_coherence=True  # Regroupement spatial
)
```

---

## 📝 Notes Importantes

### Qualité des Normales

La qualité de détection dépend fortement des normales:

```python
# Bonne pratique: k-voisins adapté à la densité
point_density = estimate_point_density(points)

if point_density > 50:  # Dense
    k_neighbors = 30
elif point_density > 20:  # Moyenne
    k_neighbors = 20
else:  # Faible
    k_neighbors = 10

normals = compute_normals(points, k_neighbors=k_neighbors)
```

### Gestion du Bruit

```python
# Filtrer les plans trop petits (bruit)
valid_planes = [
    plane for plane in all_planes
    if plane.n_points >= 50 and plane.area >= 1.0
]
```

### Performance

- **Horizontal**: Très rapide (~0.1s pour 1M points)
- **Vertical**: Rapide (~0.2s pour 1M points)
- **Incliné**: Rapide (~0.2s pour 1M points)
- **Segmentation**: Peut être lente si nombreux plans

---

## 🔗 Références

- Module principal: `ign_lidar/core/classification/plane_detection.py`
- Détection de murs: `ign_lidar/core/classification/building_clustering.py`
- Exemples: `examples/demo_wall_detection.py`
- Documentation: `docs/WALL_DETECTION_GUIDE.md`

---

**Version**: 1.0  
**Date**: 19 octobre 2025  
**Status**: ✅ Implémentation complète
