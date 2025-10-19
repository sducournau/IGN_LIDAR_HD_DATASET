# Filtrage d'objets variables avec RGE ALTI DTM

**Version:** 5.2.1  
**Date:** 19 octobre 2025  
**Objectif:** Utiliser le DTM pour éliminer les objets temporaires/variables de la classification

## 🎯 Principe

Le **Digital Terrain Model (DTM)** RGE ALTI fournit une référence sol stable permettant de calculer précisément la hauteur de chaque point au-dessus du terrain naturel :

```
height_above_ground = point_Z - DTM_elevation_at_XY
```

Cette hauteur permet d'identifier et filtrer les objets variables (voitures, mobilier urbain, véhicules, etc.).

## 📊 Objets variables à filtrer

### 1. Voitures et véhicules 🚗

**Caractéristiques:**

- Hauteur typique: 1.4-1.8m (voitures), 2.5-4.0m (camions/bus)
- Présents sur: routes, parkings, aires de services
- Signature LiDAR: surfaces planes horizontales + parois verticales

**Stratégie de filtrage:**

```python
# Sur routes (classe 11)
vehicle_mask = (classification == 11) & (height_above_ground > 0.8)
classification[vehicle_mask] = 1  # Reclasser en "unassigned" ou créer classe "vehicle"

# Sur parkings (classe 40)
vehicle_mask = (classification == 40) & (height_above_ground > 0.5) & (height_above_ground < 3.0)
classification[vehicle_mask] = 1
```

**Seuils recommandés:**

- Routes: `0.8m < height < 3.0m` → probablement véhicule
- Parkings: `0.5m < height < 4.0m` → probablement véhicule
- Voies ferrées: `1.5m < height < 5.0m` → probablement train/wagon

### 2. Murets et clôtures 🧱

**Caractéristiques:**

- Hauteur: 0.5-2.5m
- Géométrie: structures verticales linéaires
- Verticality: > 0.8

**Stratégie:**

```python
# Détecter les murets (combinaison hauteur + verticalité)
wall_mask = (
    (height_above_ground > 0.5) &
    (height_above_ground < 2.5) &
    (verticality > 0.8) &
    (planarity > 0.7)
)

# Option 1: Créer classe "wall" (61)
classification[wall_mask] = 61

# Option 2: Reclasser selon contexte
# - Si près de bâtiment → partie du bâtiment (6)
# - Si entre parcelles → clôture/limite (classe custom)
```

### 3. Mobilier urbain (bancs, panneaux, poteaux) 🪑

**Caractéristiques:**

- Hauteur: 0.5-4.0m
- Taille limitée: clusters < 2m²
- Présents sur: trottoirs, places, parcs

**Stratégie:**

```python
# Identifier objets isolés de petite taille
from scipy.spatial import cKDTree

# Points élevés sur surfaces artificielles
elevated_mask = (
    (classification.isin([11, 40, 41])) &  # Routes, parking, sports
    (height_above_ground > 0.5) &
    (height_above_ground < 4.0)
)

# Filtrer par taille de cluster
elevated_points = points[elevated_mask]
tree = cKDTree(elevated_points[:, :2])
neighbors = tree.query_ball_point(elevated_points[:, :2], r=1.0)
cluster_sizes = [len(n) for n in neighbors]

# Petits clusters = mobilier urbain
small_cluster_mask = np.array(cluster_sizes) < 50  # < 50 points
classification[elevated_mask][small_cluster_mask] = 64  # "Urban furniture"
```

### 4. Végétation basse variable 🌿

**Caractéristiques:**

- Hauteur: 0.2-2.0m
- NDVI élevé (> 0.3)
- Peut varier saisonnièrement

**Stratégie:**

```python
# Végétation basse vs objets artificiels
low_veg_mask = (
    (height_above_ground > 0.2) &
    (height_above_ground <= 2.0) &
    (ndvi > 0.3) &  # Signature végétale
    (intensity < 180)  # Réflectivité végétation
)
classification[low_veg_mask] = 3  # Low vegetation
```

## 🔧 Implémentation proposée

### Nouvelle classe: `VariableObjectFilter`

```python
# ign_lidar/core/classification/variable_object_filter.py

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


class VariableObjectFilter:
    """
    Filtre les objets temporaires/variables en utilisant la hauteur DTM.

    Élimine:
    - Véhicules sur routes/parkings
    - Mobilier urbain
    - Objets temporaires
    - Murets/clôtures (optionnel)
    """

    def __init__(
        self,
        filter_vehicles: bool = True,
        filter_urban_furniture: bool = True,
        filter_walls: bool = False,
        vehicle_height_range: Tuple[float, float] = (0.8, 4.0),
        furniture_height_range: Tuple[float, float] = (0.5, 4.0),
        furniture_max_cluster_size: int = 50,
        wall_height_range: Tuple[float, float] = (0.5, 2.5),
        wall_min_verticality: float = 0.8
    ):
        self.filter_vehicles = filter_vehicles
        self.filter_urban_furniture = filter_urban_furniture
        self.filter_walls = filter_walls

        self.vehicle_height_min, self.vehicle_height_max = vehicle_height_range
        self.furniture_height_min, self.furniture_height_max = furniture_height_range
        self.furniture_max_cluster = furniture_max_cluster_size
        self.wall_height_min, self.wall_height_max = wall_height_range
        self.wall_min_verticality = wall_min_verticality

    def filter_variable_objects(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        height_above_ground: np.ndarray,
        features: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Applique le filtrage des objets variables.

        Args:
            points: Nuage de points [N, 3]
            classification: Classifications ASPRS [N]
            height_above_ground: Hauteur au-dessus DTM [N]
            features: Features géométriques (verticality, planarity, etc.)

        Returns:
            Classification modifiée [N]
        """
        classification_filtered = classification.copy()
        n_filtered = 0

        # 1. Filtrer véhicules
        if self.filter_vehicles:
            n_veh = self._filter_vehicles(
                classification_filtered,
                height_above_ground
            )
            n_filtered += n_veh
            logger.info(f"  🚗 Filtered {n_veh:,} vehicle points")

        # 2. Filtrer mobilier urbain
        if self.filter_urban_furniture and features is not None:
            n_furn = self._filter_urban_furniture(
                points,
                classification_filtered,
                height_above_ground
            )
            n_filtered += n_furn
            logger.info(f"  🪑 Filtered {n_furn:,} urban furniture points")

        # 3. Filtrer murets/clôtures
        if self.filter_walls and features is not None:
            n_walls = self._filter_walls(
                classification_filtered,
                height_above_ground,
                features
            )
            n_filtered += n_walls
            logger.info(f"  🧱 Filtered {n_walls:,} wall/fence points")

        logger.info(f"  ✅ Total variable objects filtered: {n_filtered:,} points")
        return classification_filtered

    def _filter_vehicles(
        self,
        classification: np.ndarray,
        height: np.ndarray
    ) -> int:
        """Filtre les véhicules sur routes/parkings."""
        # Routes (11), Parkings (40)
        transport_mask = np.isin(classification, [11, 40])

        # Hauteur typique véhicule
        vehicle_height_mask = (
            (height >= self.vehicle_height_min) &
            (height <= self.vehicle_height_max)
        )

        # Combiner
        vehicle_mask = transport_mask & vehicle_height_mask
        n_vehicles = vehicle_mask.sum()

        # Reclasser en "unassigned" (1) ou créer classe véhicule (18?)
        classification[vehicle_mask] = 1  # Unassigned

        return n_vehicles

    def _filter_urban_furniture(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        height: np.ndarray
    ) -> int:
        """Filtre le mobilier urbain (petits objets isolés)."""
        # Surfaces artificielles
        artificial_mask = np.isin(classification, [11, 40, 41, 42])

        # Hauteur mobilier
        furniture_height_mask = (
            (height >= self.furniture_height_min) &
            (height <= self.furniture_height_max)
        )

        candidate_mask = artificial_mask & furniture_height_mask
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            return 0

        # Identifier petits clusters (mobilier)
        candidate_points = points[candidate_indices]
        tree = cKDTree(candidate_points[:, :2])

        # Compter voisins dans rayon 1m
        neighbors_count = tree.query_ball_point(
            candidate_points[:, :2],
            r=1.0,
            return_length=True
        )

        # Petits clusters = mobilier
        small_cluster_mask = neighbors_count < self.furniture_max_cluster
        furniture_indices = candidate_indices[small_cluster_mask]

        classification[furniture_indices] = 1  # Unassigned

        return len(furniture_indices)

    def _filter_walls(
        self,
        classification: np.ndarray,
        height: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> int:
        """Filtre les murets et clôtures."""
        verticality = features.get('verticality')
        planarity = features.get('planarity')

        if verticality is None or planarity is None:
            return 0

        # Critères muret: hauteur + verticalité + planéité
        wall_mask = (
            (height >= self.wall_height_min) &
            (height <= self.wall_height_max) &
            (verticality >= self.wall_min_verticality) &
            (planarity >= 0.7)
        )

        n_walls = wall_mask.sum()

        # Option: créer classe "wall" (61) ou reclasser
        classification[wall_mask] = 61  # Wall class

        return n_walls


def apply_variable_object_filtering(
    points: np.ndarray,
    classification: np.ndarray,
    height_above_ground: np.ndarray,
    config: Dict,
    features: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    Fonction principale pour appliquer le filtrage.

    Usage dans processor.py:
        from ..classification.variable_object_filter import apply_variable_object_filtering

        classification = apply_variable_object_filtering(
            points=points_v,
            classification=labels_v,
            height_above_ground=height_above_ground,
            config=self.config.get('variable_object_filtering', {}),
            features=all_features
        )
    """
    filter_config = config.get('variable_object_filtering', {})

    if not filter_config.get('enabled', False):
        return classification

    logger.info("  🔍 Filtering variable objects using DTM heights...")

    filter = VariableObjectFilter(
        filter_vehicles=filter_config.get('filter_vehicles', True),
        filter_urban_furniture=filter_config.get('filter_urban_furniture', True),
        filter_walls=filter_config.get('filter_walls', False),
        vehicle_height_range=tuple(filter_config.get('vehicle_height_range', [0.8, 4.0])),
        furniture_height_range=tuple(filter_config.get('furniture_height_range', [0.5, 4.0])),
        furniture_max_cluster_size=filter_config.get('furniture_max_cluster_size', 50),
        wall_height_range=tuple(filter_config.get('wall_height_range', [0.5, 2.5])),
        wall_min_verticality=filter_config.get('wall_min_verticality', 0.8)
    )

    return filter.filter_variable_objects(
        points=points,
        classification=classification,
        height_above_ground=height_above_ground,
        features=features
    )
```

### Configuration YAML

```yaml
# Configuration du filtrage d'objets variables
variable_object_filtering:
  enabled: true

  # Filtrer les véhicules
  filter_vehicles: true
  vehicle_height_range: [0.8, 4.0] # min, max en mètres

  # Filtrer le mobilier urbain
  filter_urban_furniture: true
  furniture_height_range: [0.5, 4.0]
  furniture_max_cluster_size: 50 # Nombre de points max

  # Filtrer les murets/clôtures (optionnel)
  filter_walls: false
  wall_height_range: [0.5, 2.5]
  wall_min_verticality: 0.8
```

## 📈 Résultats attendus

### Avant filtrage (routes avec voitures)

```
Classe 11 (Route): 1,245,000 points
├─ Vraie chaussée: ~900,000 points (72%)
├─ Voitures/véhicules: ~250,000 points (20%)
└─ Mobilier urbain: ~95,000 points (8%)
```

### Après filtrage DTM

```
Classe 11 (Route): 900,000 points (propre!)
Classe 1 (Unassigned): +345,000 points filtrés
├─ Véhicules: ~250,000 points
└─ Mobilier: ~95,000 points
```

**Amélioration:** +20-30% de précision sur routes/parkings

## 🔍 Validation

### Test sur tuile Versailles

```bash
ign-lidar-hd process \
  -c examples/config_asprs_bdtopo_cadastre_optimized.yaml \
  variable_object_filtering.enabled=true \
  input_dir="/mnt/d/ign/versailles/" \
  output_dir="/mnt/d/ign/versailles_filtered"
```

### Métriques à surveiller

- Nombre de points filtrés par classe
- Pourcentage de véhicules détectés
- Faux positifs (vraies routes classées comme véhicules)
- Temps de traitement additionnel

## ⚠️ Limitations

1. **Véhicules stationnés longtemps:** Peuvent être dans le DTM original
2. **Seuils de hauteur:** À ajuster selon contexte urbain/rural
3. **Faux positifs:** Bosses/creux de route peuvent être filtrés
4. **Performance:** +10-30 secondes par tuile

## 📚 Références

- RGE ALTI specs: https://geoservices.ign.fr/rgealti
- ASPRS classes: https://www.asprs.org/divisions-committees/lidar
- Configuration DTM: `docs/RGE_ALTI_INTEGRATION.md`
