# Analyse des Bugs de Classification et Reclassification

**Date:** 26 octobre 2025  
**Analyse:** Classification à partir du Ground Truth (GT) et reclassification  
**Problème rapporté:** Les classifications produisent toujours les mêmes résultats, les bâtiments et autres classes ne respectent pas les règles mises en place.

---

## 🔴 BUGS CRITIQUES IDENTIFIÉS

### Bug #1: Ordre de Priorité Inversé dans STRtree Labeling
**Fichier:** `ign_lidar/io/ground_truth_optimizer.py` (lignes 330-380)  
**Sévérité:** 🔴 CRITIQUE

**Problème:**
```python
# Iterate in reverse to give priority to later features (higher priority)
for candidate_idx in candidate_indices:
    # Use prepared polygon for much faster contains() check
    if prepared_polygons[candidate_idx].contains(point_geom):
        labels[start_idx + i] = polygon_labels[candidate_idx]
        # Don't break - let higher priority features override
```

**Bug:** Le commentaire dit "Don't break - let higher priority features override" mais le code **ne fait PAS de break**, ce qui signifie que **la dernière feature qui contient le point écrase toutes les précédentes**, indépendamment de la priorité définie.

**Comportement actuel:**
- Si un point est dans un polygone "vegetation" (priorité basse) ET un polygone "building" (priorité haute)
- Les deux labels sont successivement assignés
- Le dernier label (celui du dernier polygon dans `candidate_indices`) gagne
- **L'ordre de `candidate_indices` dépend de l'arbre STRtree, PAS de la priorité définie**

**Impact:**
✅ **Ceci explique pourquoi la classification est toujours la même indépendamment des règles!**

**Solution requise:**
```python
# Option 1: Break après la première correspondance (priorité simple)
for candidate_idx in candidate_indices:
    if prepared_polygons[candidate_idx].contains(point_geom):
        labels[start_idx + i] = polygon_labels[candidate_idx]
        break  # ✅ STOP après la première correspondance

# Option 2: Vérifier toutes et prendre la priorité maximale
best_label = 0
best_priority = -1
for candidate_idx in candidate_indices:
    if prepared_polygons[candidate_idx].contains(point_geom):
        label = polygon_labels[candidate_idx]
        priority = self._get_priority(label)
        if priority > best_priority:
            best_label = label
            best_priority = priority
labels[start_idx + i] = best_label
```

---

### Bug #2: Ordre de Priorité Incorrect dans Construction du STRtree
**Fichier:** `ign_lidar/io/ground_truth_optimizer.py` (lignes 325-345)  
**Sévérité:** 🔴 CRITIQUE

**Problème:**
```python
# Add polygons in reverse priority (so higher priority overwrites)
for feature_type in reversed(label_priority):
    if feature_type not in ground_truth_features:
        continue
    # ... ajoute les polygones dans all_polygons
```

**Bug:** Les polygons sont ajoutés dans l'ordre **inversé** de la priorité (`reversed(label_priority)`), mais quand on itère sur `candidate_indices`, on ne tient pas compte de cet ordre!

**label_priority par défaut:**
```python
label_priority = ["buildings", "roads", "water", "vegetation"]
# Priorité: buildings (haute) > roads > water > vegetation (basse)
```

**Ordre d'ajout:**
```python
reversed(label_priority) = ["vegetation", "water", "roads", "buildings"]
# Les buildings sont ajoutés EN DERNIER dans all_polygons
```

**Résultat:** Si on ne break pas (Bug #1), le dernier polygon qui matche gagne, ce qui **devrait** favoriser les buildings (ajoutés en dernier). Mais `candidate_indices` de STRtree **n'est pas dans l'ordre d'ajout**, donc l'ordre est ALÉATOIRE!

**Impact:**
✅ **Classification non-déterministe selon l'ordre interne du STRtree**

---

### Bug #3: NDVI Refinement Appliqué Trop Tard
**Fichier:** `ign_lidar/io/ground_truth_optimizer.py` (ligne 400)  
**Sévérité:** 🟡 MAJEUR

**Problème:**
```python
# Apply NDVI refinement
if ndvi is not None and use_ndvi_refinement:
    labels = self._apply_ndvi_refinement(
        labels, ndvi, ndvi_vegetation_threshold, ndvi_building_threshold
    )
```

**Bug:** Le raffinement NDVI est appliqué **après** toute la classification par polygones. Cela signifie que:
1. Un point est classé "building" par le polygon
2. Le NDVI détecte que c'est de la végétation (NDVI > 0.3)
3. Le point est reclassé en "vegetation"
4. **MAIS** si une règle géométrique est appliquée après (dans `geometric_rules.py`), elle peut RE-reclasser le point en "building"!

**Impact:**
- Les règles NDVI sont écrasées par les règles géométriques
- Végétation sur toits classée comme "building"
- Routes avec arbres classées comme "road" au lieu de "vegetation"

---

### Bug #4: Conflit entre label_priority et priority_order
**Fichier:** `ign_lidar/core/classification/reclassifier.py` (lignes 190-200)  
**Sévérité:** 🟡 MAJEUR

**Problème:**
```python
# Dans reclassifier.py:
self.priority_order = [
    ("vegetation", self.ASPRS_MEDIUM_VEGETATION),
    ("water", self.ASPRS_WATER),
    ("cemeteries", self.ASPRS_CEMETERY),
    ("parking", self.ASPRS_PARKING),
    ("sports", self.ASPRS_SPORTS),
    ("railways", self.ASPRS_RAIL),
    ("roads", self.ASPRS_ROAD),
    ("bridges", self.ASPRS_BRIDGE),
    ("buildings", self.ASPRS_BUILDING),  # Highest priority
]

# Dans ground_truth_optimizer.py:
label_priority = ["buildings", "roads", "water", "vegetation"]
# Priorité: buildings (haute) > roads > water > vegetation (basse)
```

**Bug:** Les deux systèmes ont des priorités **différentes**!
- `reclassifier.py`: buildings > bridges > roads > railways > sports > parking > cemeteries > water > vegetation
- `ground_truth_optimizer.py`: buildings > roads > water > vegetation

**Impact:**
✅ **Comportement incohérent selon qu'on utilise le reclassifier ou le ground truth optimizer**

---

### Bug #5: Geometric Rules Écrasent le Ground Truth
**Fichier:** `ign_lidar/core/classification/geometric_rules.py` (lignes 228-290)  
**Sévérité:** 🔴 CRITIQUE

**Problème:**
```python
def apply_all_rules(self, points, labels, ground_truth_features, ...):
    # Rule 1: Road-vegetation disambiguation
    if "roads" in ground_truth_features and ndvi is not None:
        n_fixed = self.fix_road_vegetation_overlap(...)
        
    # Rule 2: Building buffer zone classification
    if "buildings" in ground_truth_features:
        n_added = self.classify_building_buffer_zone(...)
        
    # Rule 2b: Verticality-based building classification
    n_vertical = self.classify_by_verticality(...)
```

**Bug:** Les règles géométriques **modifient les labels** qui ont été définis par le ground truth, **SANS vérifier** si le point a déjà été classé par le GT!

**Exemple concret:**
1. Point XYZ dans un polygon "vegetation" du BD TOPO → classé "vegetation"
2. Point a une verticality > 0.85 → Rule 2b le reclasse en "building"
3. **Le GT est ignoré!**

**Impact:**
✅ **Les règles géométriques écrasent systématiquement le ground truth, rendant le GT inutile**

---

### Bug #6: Building Buffer Zone N'exclut Pas les Points Déjà Classés
**Fichier:** `ign_lidar/core/classification/geometric_rules.py` (lignes 430-510)  
**Sévérité:** 🟡 MAJEUR

**Problème:**
```python
def classify_building_buffer_zone(self, points, labels, building_geometries):
    # Find unclassified points
    unclassified_mask = labels == self.ASPRS_UNCLASSIFIED
    
    if not np.any(unclassified_mask):
        return 0
    
    unclassified_indices = np.where(unclassified_mask)[0]
```

**Bug partiel:** La fonction ne traite que les points "unclassified", ce qui est bien. MAIS:
- Elle ne vérifie pas si le point est dans un autre polygon GT (road, water, etc.)
- Elle peut classifier comme "building" un point qui est juste à côté d'un bâtiment mais dans une zone "road"

**Impact:**
- Routes proches de bâtiments classées comme "building"
- Zones de parking classées comme "building"

---

### Bug #7: Verticality Classification Sans Vérification GT
**Fichier:** `ign_lidar/core/classification/geometric_rules.py` (lignes 819-930)  
**Sévérité:** 🔴 CRITIQUE

**Problème:**
```python
def classify_by_verticality(self, points, labels, ndvi):
    # Find unclassified points
    unclassified_mask = labels == self.ASPRS_UNCLASSIFIED
    
    if not np.any(unclassified_mask):
        return 0
```

**Bug:** Comme pour le buffer zone, cette fonction ne traite que les "unclassified". MAIS elle est appelée **APRÈS** que les points aient été classés par le GT, donc:
1. Si un point est classé "road" par GT
2. Il n'est plus "unclassified"
3. La règle de verticality ne s'applique pas
4. **C'est bon pour éviter les conflits!**

**MAIS:** Cette fonction est appelée dans `apply_all_rules` qui est elle-même appelée par le `reclassifier`, ce qui signifie:
1. Point classé par GT → "road"
2. Reclassifier appelle `apply_all_rules`
3. `classify_by_verticality` ne fait rien (point pas unclassified)
4. **OK!**

**Mais attendez:** Dans le `reclassifier.py` (ligne 262), on fait:
```python
refined_labels, rule_stats = self.geometric_rules.apply_all_rules(
    points=points,
    labels=updated_labels,  # ← Labels déjà mis à jour par GT
    ground_truth_features=ground_truth_features,
    ndvi=ndvi,
    intensities=intensities,
)
updated_labels = refined_labels  # ← On écrase avec les règles géométriques
```

**Impact:**
- Si `apply_all_rules` modifie `updated_labels` en place, OK
- Si elle retourne un nouveau array, les modifications GT peuvent être perdues

---

## 🟡 BUGS MAJEURS SUPPLÉMENTAIRES

### Bug #8: NDVI Refinement Inversé
**Fichier:** `ign_lidar/io/ground_truth_optimizer.py` (lignes 476-507)  
**Sévérité:** 🟡 MAJEUR

**Problème:**
```python
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
```

**Bug:** Les valeurs NDVI sont correctes, MAIS:
- Un point avec `ndvi = 0.2` (entre 0.15 et 0.3) est classé "building" par GT
- Il passe le test `ndvi >= 0.3`? NON → reste "building"
- Il passe le test `ndvi <= 0.15`? NON → reste "building"
- **OK, pas de bug ici!**

**Mais:** Si initialement classé "vegetation" avec `ndvi = 0.2`:
- Il ne passe PAS `ndvi >= 0.3` → ne reste pas forcément "vegetation"
- Il ne passe PAS `ndvi <= 0.15` → ne devient pas "building"
- **Résultat:** Zone grise entre 0.15 et 0.3 qui n'est pas reclassée!

**Impact:**
- Zones ambiguës (0.15 < NDVI < 0.3) gardent leur label initial du GT
- Peut causer des incohérences

---

### Bug #9: Classification Spectrale Écrase le GT Sans Condition
**Fichier:** `ign_lidar/core/classification/spectral_rules.py` (lignes 188-220)  
**Sévérité:** 🟡 MAJEUR

**Problème:**
```python
def classify_by_spectral_signature(self, rgb, nir, current_labels, ndvi, apply_to_unclassified_only=True):
    # Create mask for points to classify
    if apply_to_unclassified_only:
        mask = current_labels == self.ASPRS_UNCLASSIFIED
    else:
        mask = np.ones(len(current_labels), dtype=bool)
    
    # Rule 3: Moderate NIR + Moderate brightness + Low NDVI = Concrete buildings
    concrete_mask = (
        mask &
        (nir >= self.nir_building_threshold) &
        (nir < self.nir_vegetation_threshold) &
        (brightness >= self.brightness_concrete_min) &
        (brightness <= self.brightness_concrete_max) &
        (ndvi < 0.15)
    )
    labels[concrete_mask] = self.ASPRS_BUILDING
```

**Bug:** Quand `apply_to_unclassified_only=True` (par défaut), c'est OK. MAIS dans `geometric_rules.py` (ligne 295), on appelle:
```python
updated_labels, spectral_stats = self.spectral_rules.classify_by_spectral_signature(
    rgb=rgb,
    nir=nir,
    current_labels=updated_labels,
    ndvi=ndvi,
    apply_to_unclassified_only=True,  # ✅ OK
)
```

**Mais** ensuite (ligne 309):
```python
updated_labels, relaxed_stats = self.spectral_rules.classify_unclassified_relaxed(
    rgb=rgb,
    nir=nir,
    current_labels=updated_labels,
    ndvi=ndvi,
    verticality=verticality,
    heights=height,
)
```

**Et dans `classify_unclassified_relaxed` (ligne 329-350), on a:**
```python
def classify_unclassified_relaxed(self, rgb, nir, current_labels, ndvi, verticality, heights):
    # Masque sur les unclassified
    unclassified_mask = current_labels == self.ASPRS_UNCLASSIFIED
    
    # Règle 2: Bâtiments avec critères géométriques
    if verticality is not None and heights is not None:
        building_vertical_mask = (
            unclassified_mask &
            (verticality > 0.65) &
            (heights > 0.5) &
            (ndvi < 0.25)
        )
        labels[building_vertical_mask] = self.ASPRS_BUILDING
```

**Impact:**
- La classification relaxée ne s'applique QUE aux unclassified
- **OK, pas de conflit avec le GT**

---

## 🔵 BUGS MINEURS

### Bug #10: Absence de Validation des Features
**Fichier:** Multiples fichiers  
**Sévérité:** 🔵 MINEUR

**Problème:** Aucune validation que les features requises existent avant de les utiliser:
```python
# Dans geometric_rules.py
def classify_by_verticality(self, points, labels, ndvi):
    # Pas de vérification que 'verticality' existe dans les features
```

**Impact:**
- Crash si la feature n'existe pas
- Messages d'erreur peu clairs

---

### Bug #11: Hard-coded Thresholds Sans Configuration
**Fichier:** Multiples fichiers  
**Sévérité:** 🔵 MINEUR

**Problème:** De nombreux seuils sont hard-codés:
```python
# geometric_rules.py
NDVI_DENSE_FOREST = 0.60
NDVI_HEALTHY_TREES = 0.50
NDVI_MODERATE_VEG = 0.40
# ...
```

**Impact:**
- Difficile de tuner pour différents environnements
- Pas de cohérence entre modules

---

## 📊 RÉSUMÉ DES IMPACTS

| Bug | Sévérité | Impact sur Classification |
|-----|----------|---------------------------|
| #1: Ordre STRtree | 🔴 CRITIQUE | Classification aléatoire, priorités ignorées |
| #2: Ordre Inversé | 🔴 CRITIQUE | Classification non-déterministe |
| #3: NDVI Timing | 🟡 MAJEUR | NDVI écrasé par règles géométriques |
| #4: Priorités Conflits | 🟡 MAJEUR | Comportement incohérent |
| #5: Geometric Écrase GT | 🔴 CRITIQUE | Ground truth inutile |
| #6: Buffer Zone GT | 🟡 MAJEUR | Routes classées building |
| #7: Verticality GT | 🟡 MAJEUR | Dépend de l'ordre d'exécution |
| #8: NDVI Zone Grise | 🟡 MAJEUR | Incohérences 0.15-0.3 |
| #9: Spectral Timing | 🔵 MINEUR | OK si apply_to_unclassified_only=True |
| #10: Validation Features | 🔵 MINEUR | Crashes potentiels |
| #11: Hard-coded Thresholds | 🔵 MINEUR | Manque de flexibilité |

---

## 🛠️ RECOMMANDATIONS DE CORRECTION

### Priorité 1 (URGENT):
1. **Bug #1 & #2:** Refactoriser `_label_strtree()` pour respecter les priorités
2. **Bug #5:** Ajouter un flag `preserve_ground_truth` dans `apply_all_rules()`
3. **Bug #4:** Unifier les systèmes de priorité

### Priorité 2 (IMPORTANT):
4. **Bug #3:** Appliquer NDVI avant les règles géométriques OU ajouter un flag de protection
5. **Bug #6:** Vérifier les polygones GT dans `classify_building_buffer_zone()`
6. **Bug #8:** Ajouter une règle pour la zone grise 0.15-0.3

### Priorité 3 (AMÉLIORATION):
7. **Bug #10:** Ajouter validation des features
8. **Bug #11:** Externaliser les thresholds dans la config

---

## 🧪 TESTS RECOMMANDÉS

### Test 1: Priorité des Polygones
```python
# Point dans un polygon "vegetation" ET un polygon "building"
# Résultat attendu: "building" (priorité haute)
# Résultat actuel: Aléatoire selon STRtree
```

### Test 2: Préservation du GT
```python
# Point classé "vegetation" par GT, mais verticality > 0.85
# Résultat attendu: "vegetation" (GT préservé)
# Résultat actuel: "building" (GT écrasé)
```

### Test 3: NDVI Zone Grise
```python
# Point avec NDVI = 0.2, classé "vegetation" par GT
# Résultat attendu: "building" ou "vegetation" selon règle claire
# Résultat actuel: Garde "vegetation" (pas de reclassification)
```

---

**Fin du rapport d'analyse**
