# Plan de Correction des Bugs de Classification

**Date:** 26 octobre 2025  
**Version:** 1.0  
**Statut:** 🚧 EN ATTENTE D'IMPLÉMENTATION

---

## 📋 Vue d'Ensemble

Ce document décrit les étapes précises pour corriger les bugs de classification identifiés dans `CLASSIFICATION_BUGS_ANALYSIS.md`.

**Objectif:** Rendre la classification déterministe, respecter les priorités GT, et unifier le comportement entre modules.

---

## 🔴 CORRECTION 1: Bug #1 - Priorités STRtree (CRITIQUE)

### Fichier à Modifier

`ign_lidar/io/ground_truth_optimizer.py` - méthode `_label_strtree()` (lignes 360-400)

### Code Actuel (BUGUÉ)

```python
# Ligne 370-380
for i, point_coords in enumerate(batch_points):
    point_geom = Point(point_coords[0], point_coords[1])

    # Find candidate polygon indices
    candidate_indices = tree.query(point_geom)

    # Check actual containment using prepared polygons (faster)
    # Iterate in reverse to give priority to later features (higher priority)
    for candidate_idx in candidate_indices:
        # Use prepared polygon for much faster contains() check
        if prepared_polygons[candidate_idx].contains(point_geom):
            labels[start_idx + i] = polygon_labels[candidate_idx]
            # Don't break - let higher priority features override  ← BUG!
```

### Code Corrigé

```python
# SOLUTION: Vérifier toutes les correspondances et prendre la priorité maximale
for i, point_coords in enumerate(batch_points):
    point_geom = Point(point_coords[0], point_coords[1])

    # Find candidate polygon indices
    candidate_indices = tree.query(point_geom)

    if len(candidate_indices) == 0:
        continue

    # ✅ CORRECTION: Vérifier TOUTES les correspondances et prendre la meilleure priorité
    best_label = 0
    best_priority = -1

    for candidate_idx in candidate_indices:
        if prepared_polygons[candidate_idx].contains(point_geom):
            label = polygon_labels[candidate_idx]

            # Obtenir la priorité de ce label
            # Priorités: buildings=4, roads=3, water=2, vegetation=1
            priority = label_priority_values.get(label, 0)

            if priority > best_priority:
                best_label = label
                best_priority = priority

    # Assigner le label avec la meilleure priorité
    if best_priority > 0:
        labels[start_idx + i] = best_label
```

### Modifications Supplémentaires

Ajouter avant la boucle (ligne ~320):

```python
# ✅ AJOUT: Créer un mapping label → priorité
# Plus le nombre est élevé, plus la priorité est haute
label_priority_values = {}
for idx, feature_type in enumerate(label_priority):
    label_value = label_map.get(feature_type, 0)
    # Priorité = position dans la liste (inversée)
    # buildings (idx=0) → priorité 4 (max)
    # vegetation (idx=3) → priorité 1 (min)
    label_priority_values[label_value] = len(label_priority) - idx
```

### Tests

```bash
# Exécuter les tests de validation
pytest tests/test_classification_bugs.py::TestBug1_PriorityOrder -v

# Exécuter le script de diagnostic
python scripts/diagnose_classification_bugs.py
```

---

## 🔴 CORRECTION 2: Bug #5 - Geometric Rules Écrasent GT (CRITIQUE)

### Fichier à Modifier

`ign_lidar/core/classification/geometric_rules.py` - méthode `apply_all_rules()` (lignes 219-320)

### Solution: Ajouter Flag `preserve_ground_truth`

#### Étape 1: Modifier Signature de la Fonction

```python
# Ligne 219
def apply_all_rules(
    self,
    points: np.ndarray,
    labels: np.ndarray,
    ground_truth_features: Dict[str, gpd.GeoDataFrame],
    ndvi: Optional[np.ndarray] = None,
    intensities: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    nir: Optional[np.ndarray] = None,
    verticality: Optional[np.ndarray] = None,
    preserve_ground_truth: bool = True,  # ✅ AJOUT: Flag pour préserver GT
) -> Tuple[np.ndarray, Dict[str, int]]:
```

#### Étape 2: Créer Masque "Modifiable"

Ajouter après ligne 240:

```python
# ✅ AJOUT: Créer masque des points qui peuvent être modifiés
if preserve_ground_truth:
    # Seuls les points "unclassified" (code 1) peuvent être modifiés
    modifiable_mask = updated_labels == self.ASPRS_UNCLASSIFIED
    logger.info(f"  Préservation GT activée: {np.sum(modifiable_mask):,} points modifiables")
else:
    # Tous les points peuvent être modifiés
    modifiable_mask = np.ones(len(points), dtype=bool)
```

#### Étape 3: Modifier Chaque Règle

**Rule 1: Road-vegetation** (ligne ~245)

```python
# AVANT:
n_fixed = self.fix_road_vegetation_overlap(
    points=points,
    labels=updated_labels,
    road_geometries=ground_truth_features["roads"],
    ndvi=ndvi,
)

# APRÈS:
n_fixed = self.fix_road_vegetation_overlap(
    points=points[modifiable_mask],  # ✅ Seulement points modifiables
    labels=updated_labels,
    road_geometries=ground_truth_features["roads"],
    ndvi=ndvi[modifiable_mask] if ndvi is not None else None,
)
```

**Rule 2: Building buffer** (ligne ~255)

```python
# AVANT:
n_added = self.classify_building_buffer_zone(
    points=points,
    labels=updated_labels,
    building_geometries=ground_truth_features["buildings"],
)

# APRÈS:
n_added = self.classify_building_buffer_zone(
    points=points[modifiable_mask],
    labels=updated_labels,
    building_geometries=ground_truth_features["buildings"],
)
```

**Rule 2b: Verticality** (ligne ~267)

```python
# AVANT:
n_vertical = self.classify_by_verticality(
    points=points,
    labels=updated_labels,
    ndvi=ndvi
)

# APRÈS:
n_vertical = self.classify_by_verticality(
    points=points[modifiable_mask],
    labels=updated_labels,
    ndvi=ndvi[modifiable_mask] if ndvi is not None else None
)
```

#### Étape 4: Mettre à Jour les Appels

Dans `ign_lidar/core/classification/reclassifier.py` (ligne 262):

```python
# AVANT:
refined_labels, rule_stats = self.geometric_rules.apply_all_rules(
    points=points,
    labels=updated_labels,
    ground_truth_features=ground_truth_features,
    ndvi=ndvi,
    intensities=intensities,
)

# APRÈS:
refined_labels, rule_stats = self.geometric_rules.apply_all_rules(
    points=points,
    labels=updated_labels,
    ground_truth_features=ground_truth_features,
    ndvi=ndvi,
    intensities=intensities,
    preserve_ground_truth=True,  # ✅ AJOUT: Préserver le GT
)
```

### Tests

```bash
pytest tests/test_classification_bugs.py::TestBug5_GeometricRulesOverwriteGT -v
```

---

## 🔴 CORRECTION 3: Bug #4 - Unifier Priorités (CRITIQUE)

### Étape 1: Créer Fichier de Configuration Centrale

**Nouveau fichier:** `ign_lidar/config/classification_priorities.py`

```python
"""
Configuration Centrale des Priorités de Classification

Ce module définit l'ordre de priorité unique utilisé par TOUS les modules
de classification (ground_truth_optimizer, reclassifier, etc.).

Plus le rang est élevé, plus la priorité est haute.
"""

from typing import Dict

# Priorités de classification (rang 100 = priorité maximale)
CLASSIFICATION_PRIORITY: Dict[str, int] = {
    'buildings': 100,      # Priorité MAXIMALE (bâtiments toujours prioritaires)
    'bridges': 90,         # Ponts (structures importantes)
    'roads': 80,           # Routes
    'railways': 70,        # Voies ferrées
    'sports': 60,          # Installations sportives
    'parking': 50,         # Parkings
    'cemeteries': 40,      # Cimetières
    'water': 30,           # Surfaces d'eau
    'vegetation': 20,      # Végétation (priorité basse)
    'power_lines': 10,     # Lignes électriques
}

# Mapping inverséé: rang → nom de classe
PRIORITY_TO_CLASS = {v: k for k, v in CLASSIFICATION_PRIORITY.items()}

# Liste ordonnée par priorité décroissante (pour itération)
ORDERED_CLASSES = sorted(
    CLASSIFICATION_PRIORITY.keys(),
    key=lambda x: CLASSIFICATION_PRIORITY[x],
    reverse=True
)

def get_priority(class_name: str) -> int:
    """
    Obtenir la priorité d'une classe.

    Args:
        class_name: Nom de la classe (ex: 'buildings')

    Returns:
        Priorité (100 = max, 10 = min)
    """
    return CLASSIFICATION_PRIORITY.get(class_name, 0)

def compare_priority(class_a: str, class_b: str) -> int:
    """
    Comparer les priorités de deux classes.

    Args:
        class_a: Première classe
        class_b: Deuxième classe

    Returns:
        1 si class_a > class_b
        -1 si class_a < class_b
        0 si égal
    """
    priority_a = get_priority(class_a)
    priority_b = get_priority(class_b)

    if priority_a > priority_b:
        return 1
    elif priority_a < priority_b:
        return -1
    else:
        return 0
```

### Étape 2: Modifier `ground_truth_optimizer.py`

```python
# LIGNE 1: Ajouter import
from ..config.classification_priorities import (
    CLASSIFICATION_PRIORITY,
    ORDERED_CLASSES,
    get_priority
)

# LIGNE 312: Remplacer label_priority par défaut
def _label_strtree(self, ...):
    if label_priority is None:
        # ✅ UTILISER la priorité centralisée
        label_priority = ORDERED_CLASSES  # ['buildings', 'bridges', 'roads', ...]

    # ... reste du code

    # LIGNE 330: Utiliser get_priority() au lieu d'index
    label_priority_values = {}
    for feature_type in label_priority:
        label_value = label_map.get(feature_type, 0)
        label_priority_values[label_value] = get_priority(feature_type)
```

### Étape 3: Modifier `reclassifier.py`

```python
# LIGNE 1: Ajouter import
from ..config.classification_priorities import (
    CLASSIFICATION_PRIORITY,
    ORDERED_CLASSES
)

# LIGNE 190: Remplacer priority_order
def __init__(self, ...):
    # ✅ REMPLACER priority_order par version centralisée
    self.priority_order = [
        (class_name, self._get_asprs_code(class_name))
        for class_name in ORDERED_CLASSES
    ]

def _get_asprs_code(self, class_name: str) -> int:
    """Mapping classe → code ASPRS."""
    mapping = {
        'buildings': self.ASPRS_BUILDING,
        'bridges': self.ASPRS_BRIDGE,
        'roads': self.ASPRS_ROAD,
        'railways': self.ASPRS_RAIL,
        'sports': self.ASPRS_SPORTS,
        'parking': self.ASPRS_PARKING,
        'cemeteries': self.ASPRS_CEMETERY,
        'water': self.ASPRS_WATER,
        'vegetation': self.ASPRS_MEDIUM_VEGETATION,
        'power_lines': self.ASPRS_POWER_LINE,
    }
    return mapping.get(class_name, self.ASPRS_UNCLASSIFIED)
```

### Tests

```bash
pytest tests/test_classification_bugs.py::TestBug4_ConflictingPriorities -v
```

---

## 🟡 CORRECTION 4: Bug #3 - NDVI Timing (MAJEUR)

### Solution: Ajouter Flag de Protection NDVI

**Fichier:** `ign_lidar/core/classification/geometric_rules.py`

#### Modification 1: Ajouter Flag

```python
# Ligne 219
def apply_all_rules(
    self,
    # ... params existants ...
    preserve_ground_truth: bool = True,
    protect_ndvi_labels: bool = True,  # ✅ AJOUT: Protéger labels NDVI
) -> Tuple[np.ndarray, Dict[str, int]]:
```

#### Modification 2: Appliquer NDVI Avant Règles Géométriques

```python
# Ligne 280 - Déplacer NDVI refinement AVANT les autres règles
updated_labels = labels.copy()
stats = {}

# ✅ NOUVEAU: Appliquer NDVI EN PREMIER (avant règles géométriques)
if ndvi is not None:
    height = self.get_height_above_ground(points, labels, search_radius=5.0)
    n_refined = self.apply_ndvi_refinement(
        points=points, labels=updated_labels, ndvi=ndvi, height=height
    )
    stats["ndvi_refined"] = n_refined
    if n_refined > 0:
        logger.info(f"  ✓ NDVI Refinement: {n_refined:,} points (appliqué en premier)")

# Créer masque de protection pour labels NDVI
if protect_ndvi_labels:
    ndvi_modified_mask = (labels != updated_labels)  # Points modifiés par NDVI
    logger.info(f"  {np.sum(ndvi_modified_mask):,} labels protégés (NDVI)")

# Rule 1: Road-vegetation (ne pas modifier labels NDVI)
# ... etc
```

---

## 🟡 CORRECTION 5: Bug #6 - Buffer Zone GT Check (MAJEUR)

**Fichier:** `ign_lidar/core/classification/geometric_rules.py` - méthode `classify_building_buffer_zone()`

### Modification

```python
# Ligne 430
def classify_building_buffer_zone(
    self,
    points: np.ndarray,
    labels: np.ndarray,
    building_geometries: gpd.GeoDataFrame,
    ground_truth_features: Optional[Dict[str, gpd.GeoDataFrame]] = None,  # ✅ AJOUT
) -> int:
    # ... code existant ...

    # Ligne 446 - Après avoir trouvé unclassified_mask
    unclassified_mask = labels == self.ASPRS_UNCLASSIFIED

    # ✅ AJOUT: Exclure points dans d'autres polygones GT
    if ground_truth_features is not None:
        for feature_type, gdf in ground_truth_features.items():
            if feature_type == 'buildings' or gdf is None or len(gdf) == 0:
                continue

            # Vérifier si points unclassified sont dans ces polygones
            for geom in gdf.geometry:
                for idx in np.where(unclassified_mask)[0]:
                    pt = Point(points[idx, 0], points[idx, 1])
                    if geom.contains(pt):
                        # Point dans un autre polygon GT → ne pas classifier
                        unclassified_mask[idx] = False

    if not np.any(unclassified_mask):
        return 0

    # ... reste du code ...
```

---

## 🟡 CORRECTION 6: Bug #8 - NDVI Zone Grise (MAJEUR)

**Fichier:** `ign_lidar/io/ground_truth_optimizer.py` - méthode `_apply_ndvi_refinement()`

### Modification

```python
# Ligne 476
def _apply_ndvi_refinement(self, labels, ndvi, ndvi_vegetation_threshold, ndvi_building_threshold):
    BUILDING = 1
    VEGETATION = 4

    # Refine buildings: high NDVI → vegetation
    building_mask = labels == BUILDING
    high_ndvi_buildings = building_mask & (ndvi >= ndvi_vegetation_threshold)
    n_to_veg = np.sum(high_ndvi_buildings)
    if n_to_veg > 0:
        labels[high_ndvi_buildings] = VEGETATION

    # Refine vegetation: low NDVI → building
    vegetation_mask = labels == VEGETATION
    low_ndvi_vegetation = vegetation_mask & (ndvi <= ndvi_building_threshold)
    n_to_building = np.sum(low_ndvi_vegetation)
    if n_to_building > 0:
        labels[low_ndvi_vegetation] = BUILDING

    # ✅ AJOUT: Gérer la zone grise (0.15 < NDVI < 0.3)
    # Règle: Utiliser la hauteur pour départager
    grey_zone_mask = (ndvi > ndvi_building_threshold) & (ndvi < ndvi_vegetation_threshold)

    if np.any(grey_zone_mask):
        # Points dans zone grise avec label building
        grey_buildings = (labels == BUILDING) & grey_zone_mask

        # Si hauteur disponible, utiliser pour affiner
        # Logique: hauteur basse + NDVI moyen = sol/route (garder building)
        #         hauteur haute + NDVI moyen = arbuste (passer en vegetation)
        # Pour l'instant: conserver le label initial (prudent)

        n_grey = np.sum(grey_zone_mask)
        logger.debug(f"  {n_grey:,} points dans zone grise NDVI (0.15-0.3), labels conservés")

    return labels
```

---

## 📝 Checklist de Validation

### Après Chaque Correction

- [ ] Code modifié et sauvegardé
- [ ] Imports ajustés si nécessaire
- [ ] Tests unitaires passent (`pytest tests/test_classification_bugs.py::TestBugX -v`)
- [ ] Script de diagnostic OK (`python scripts/diagnose_classification_bugs.py`)
- [ ] Pas de régression (run suite complète: `pytest tests/test_classification* -v`)
- [ ] Documentation mise à jour (docstrings)

### Validation Finale

- [ ] Tous les tests passent
- [ ] Script de diagnostic confirme corrections
- [ ] Test sur vraies données (tiles Versailles)
- [ ] Performance acceptable (pas de régression)
- [ ] CHANGELOG.md mis à jour
- [ ] PR créé avec description détaillée

---

## 🚀 Ordre d'Implémentation Recommandé

1. **Jour 1:** Correction Bug #1 (STRtree priorities) - 2-3h
2. **Jour 2:** Correction Bug #5 (Geometric rules) - 3-4h
3. **Jour 3:** Correction Bug #4 (Unifier priorités) - 2-3h
4. **Jour 4:** Corrections Bugs #3, #6, #8 - 3-4h
5. **Jour 5:** Tests intégration + validation données réelles - 4h

**Total estimé:** 3-5 jours de développement

---

## 📊 Métriques de Succès

### Avant Corrections

- ❌ Classification non-déterministe
- ❌ Priorités ignorées
- ❌ GT écrasé par règles
- ❌ Incohérence entre modules

### Après Corrections

- ✅ Classification 100% déterministe
- ✅ Priorités respectées à 100%
- ✅ GT préservé (sauf si désactivé explicitement)
- ✅ Comportement cohérent partout

---

**Questions?** Consulter:

- `CLASSIFICATION_BUGS_ANALYSIS.md` (analyse détaillée)
- `CLASSIFICATION_BUGS_SUMMARY.md` (résumé exécutif)
