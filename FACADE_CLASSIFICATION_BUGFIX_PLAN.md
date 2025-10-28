# 🐛 Plan d'Action - Correction des Bugs de Classification de Façades

**Date:** October 26, 2025  
**Priorité:** HAUTE  
**Objectif:** Corriger les bugs et artefacts dans la classification des façades de bâtiments

---

## 📊 Problèmes Identifiés

### 🔴 Critiques (à corriger immédiatement)

1. **Features corrompues dans classification des toits** (`facade_processor.py:1481-1542`)

   - Features utilisées après modification par classification façades
   - Pas de validation NaN/Inf
   - `building_mask` inclut des points sol

2. **Artefacts de courbure dans détection arêtes** (`facade_processor.py:1360-1383`)

   - `curvature` contient NaN/Inf aux bordures
   - Expansion KD-Tree sans validation spatiale

3. **Incohérence masque de filtrage sol** (`facade_processor.py:1343-1362`)
   - `valid_mask` créé mais pas toujours utilisé
   - `building_height` calculé incohérent

### ⚠️ Modérés (améliorations)

4. **Absence validation features** (général)

   - Pas de vérification NaN/Inf avant utilisation
   - Pas de gestion cas limites (bords, k-voisins insuffisants)

5. **Performance et robustesse**
   - Expansion arêtes peut classifier points hors-bâtiment
   - Manque de logging pour debugging

---

## 🎯 Plan de Correction

### Phase 1: Module de Validation (30-45 min)

**Fichier:** `ign_lidar/core/classification/feature_validator.py` (existe déjà, à améliorer)

**Actions:**

1. ✅ Ajouter fonction `validate_features_for_classification()`
2. ✅ Ajouter fonction `sanitize_feature()`
3. ✅ Ajouter fonction `create_safe_mask()`
4. ✅ Gérer NaN/Inf/outliers
5. ✅ Retourner masque de points valides

**Fonctions à créer:**

```python
def validate_features_for_classification(
    features: Dict[str, np.ndarray],
    required_features: List[str],
    point_mask: Optional[np.ndarray] = None
) -> Tuple[bool, Dict[str, np.ndarray], List[str]]:
    """
    Valider et nettoyer les features avant classification.

    Returns:
        (is_valid, sanitized_features, issues)
    """
    pass

def sanitize_feature(
    feature: np.ndarray,
    feature_name: str,
    clip_sigma: float = 5.0
) -> Tuple[np.ndarray, int]:
    """
    Nettoyer une feature (NaN → 0, Inf → clip, outliers → clip).

    Returns:
        (sanitized_array, n_fixed)
    """
    pass
```

### Phase 2: Corrections dans BuildingFacadeClassifier (60-90 min)

**Fichier:** `ign_lidar/core/classification/building/facade_processor.py`

#### 2.1 Ajouter validation des features (ligne ~1310)

**Avant classification, ajouter:**

```python
# Validate features before use
required_features = ['normals', 'verticality']
if self.enable_edge_detection:
    required_features.append('curvature')
if self.enable_roof_classification:
    required_features.extend(['normals', 'verticality'])

features_dict = {}
if normals is not None:
    features_dict['normals'] = normals
if verticality is not None:
    features_dict['verticality'] = verticality
if curvature is not None:
    features_dict['curvature'] = curvature

# Validate and sanitize
from ign_lidar.core.classification.feature_validator import (
    validate_features_for_classification
)

is_valid, sanitized_features, validation_issues = validate_features_for_classification(
    features=features_dict,
    required_features=required_features,
    point_mask=building_mask
)

if not is_valid:
    logger.warning(f"Building {building_id}: Feature validation issues: {validation_issues}")
    # Use sanitized versions
    normals = sanitized_features.get('normals', normals)
    verticality = sanitized_features.get('verticality', verticality)
    curvature = sanitized_features.get('curvature', curvature)
```

#### 2.2 Utiliser systématiquement valid_mask (lignes 1343-1380)

**Remplacer:**

```python
# Old:
building_height = (
    heights[valid_mask].max()
    if np.any(valid_mask)
    else heights[building_mask].max()
)

# New:
# Always use valid_mask for consistency
if np.any(valid_mask):
    building_height = heights[valid_mask].max()
    building_points_clean = self.points[valid_mask]
    building_heights_clean = heights[valid_mask]
else:
    # Fallback if no valid points after filtering
    logger.warning(f"Building {building_id}: No valid points after ground filtering")
    valid_mask = building_mask.copy()
    building_height = heights[building_mask].max()
    building_points_clean = building_points
    building_heights_clean = heights[building_mask]
```

#### 2.3 Sécuriser détection arêtes (lignes 1360-1383)

**Remplacer:**

```python
# Old:
if self.enable_edge_detection and curvature is not None:
    high_curvature_mask = curvature > self.edge_detection_threshold
    edge_candidates = building_mask & high_curvature_mask

# New:
if self.enable_edge_detection and curvature is not None:
    # Validate curvature first
    curvature_clean = curvature.copy()
    invalid_curvature = ~np.isfinite(curvature)
    if invalid_curvature.any():
        logger.debug(f"Building {building_id}: {invalid_curvature.sum()} invalid curvature values")
        curvature_clean[invalid_curvature] = 0.0  # Safe default

    # Only consider valid building points (not ground)
    valid_edge_candidates_mask = valid_mask & (curvature_clean > self.edge_detection_threshold)

    # Filter by verticality
    if verticality is not None:
        verticality_clean = verticality.copy()
        verticality_clean[~np.isfinite(verticality)] = 0.0
        valid_edge_candidates_mask &= verticality_clean > 0.3

    edge_points_mask[valid_edge_candidates_mask] = True
    stats["edge_points_detected"] = np.sum(valid_edge_candidates_mask)
```

#### 2.4 Sécuriser expansion arêtes avec KD-Tree (lignes 1440-1455)

**Ajouter validation spatiale:**

```python
# Old:
if self.enable_edge_detection and np.any(edge_points_mask):
    # ... existing code ...

    nearby_edges = distances < self.edge_expansion_radius
    nearby_edge_indices = edge_candidates_indices[nearby_edges]

    labels_updated[nearby_edge_indices] = self.building_class

# New:
if self.enable_edge_detection and np.any(edge_points_mask):
    if all_classified_indices:
        # ... KD-Tree construction ...

        nearby_edges = distances < self.edge_expansion_radius
        nearby_edge_indices = edge_candidates_indices[nearby_edges]

        # Additional validation: only expand within building polygon
        if HAS_SHAPELY and polygon is not None:
            nearby_points = points[nearby_edge_indices]
            buffered_polygon = polygon.buffer(self.edge_expansion_radius * 1.5)

            from shapely.vectorized import contains
            inside_polygon = contains(
                buffered_polygon,
                nearby_points[:, 0],
                nearby_points[:, 1]
            )

            # Only keep edges inside building area
            nearby_edge_indices = nearby_edge_indices[inside_polygon]

            if len(nearby_edge_indices) > 0:
                labels_updated[nearby_edge_indices] = self.building_class
                all_classified_indices.update(nearby_edge_indices)
        else:
            # Fallback: use all nearby edges
            labels_updated[nearby_edge_indices] = self.building_class
            all_classified_indices.update(nearby_edge_indices)
```

#### 2.5 Sauvegarder features originales pour classification toits (lignes 1460-1542)

**Avant classification des toits, ajouter:**

```python
# Save original features for roof classification (avoid using modified features)
if self.enable_roof_classification and self.roof_classifier:
    try:
        # Use features on VALID building points only (not ground-filtered)
        roof_building_mask = valid_mask.copy()

        # Validate features are available and clean
        roof_normals = normals[roof_building_mask] if normals is not None else None
        roof_verticality = verticality[roof_building_mask] if verticality is not None else None
        roof_curvature = curvature[roof_building_mask] if curvature is not None else None

        # Validate no NaN/Inf
        if roof_normals is not None and not np.all(np.isfinite(roof_normals)):
            logger.warning(f"Building {building_id}: Invalid normals in roof classification, skipping")
            raise ValueError("Invalid normals")

        if roof_verticality is not None and not np.all(np.isfinite(roof_verticality)):
            logger.warning(f"Building {building_id}: Invalid verticality in roof classification")
            roof_verticality[~np.isfinite(roof_verticality)] = 0.0

        # Prepare features dict with validated data
        roof_features = {
            "normals": roof_normals,
            "verticality": roof_verticality,
            "curvature": roof_curvature,
            "planarity": None,  # Will compute if needed
        }

        # Rest of roof classification...
```

### Phase 3: Tests et Validation (30-45 min)

**Fichier:** `tests/test_facade_classification_bugfixes.py` (nouveau)

**Tests à créer:**

1. ✅ Test validation features avec NaN
2. ✅ Test validation features avec Inf
3. ✅ Test masque filtrage sol cohérent
4. ✅ Test détection arêtes avec curvature invalide
5. ✅ Test expansion arêtes limitée au polygone
6. ✅ Test classification toits avec features propres

### Phase 4: Documentation et Logging (15-30 min)

**Actions:**

1. ✅ Ajouter docstrings aux nouvelles fonctions
2. ✅ Ajouter logging DEBUG pour tracking
3. ✅ Mettre à jour CHANGELOG.md
4. ✅ Documenter dans BUILDING_IMPROVEMENTS_V302.md

---

## 📅 Timeline de Mise en Œuvre

| Phase                           | Durée          | Statut      |
| ------------------------------- | -------------- | ----------- |
| 1. Module validation            | 30-45 min      | 🔄 EN COURS |
| 2. Corrections facade_processor | 60-90 min      | ⏳ À FAIRE  |
| 3. Tests                        | 30-45 min      | ⏳ À FAIRE  |
| 4. Documentation                | 15-30 min      | ⏳ À FAIRE  |
| **TOTAL**                       | **2-3 heures** |             |

---

## ✅ Checklist d'Implémentation

### Phase 1: Module Validation

- [ ] Améliorer `feature_validator.py`
- [ ] Ajouter `validate_features_for_classification()`
- [ ] Ajouter `sanitize_feature()`
- [ ] Ajouter `create_safe_mask()`
- [ ] Tests unitaires pour validation

### Phase 2: Corrections BuildingFacadeClassifier

- [ ] Ajouter validation features en début de `classify_single_building()`
- [ ] Utiliser systématiquement `valid_mask`
- [ ] Sécuriser détection arêtes (validation curvature)
- [ ] Sécuriser expansion arêtes (validation spatiale)
- [ ] Sauvegarder features originales pour toits
- [ ] Ajouter logging DEBUG

### Phase 3: Tests

- [ ] Créer `test_facade_classification_bugfixes.py`
- [ ] Test NaN handling
- [ ] Test Inf handling
- [ ] Test mask consistency
- [ ] Test edge detection robustness
- [ ] Test roof classification with clean features

### Phase 4: Documentation

- [ ] Docstrings complètes
- [ ] Mise à jour CHANGELOG.md
- [ ] Mise à jour BUILDING_IMPROVEMENTS_V302.md
- [ ] Exemples de configuration si nécessaire

---

## 🎯 Critères de Succès

1. ✅ Aucune erreur NaN/Inf pendant classification
2. ✅ Masque de filtrage sol utilisé systématiquement
3. ✅ Détection arêtes robuste aux artefacts
4. ✅ Expansion arêtes limitée au bâtiment
5. ✅ Classification toits avec features propres
6. ✅ Tous les tests passent
7. ✅ Amélioration qualité classification (à mesurer avec audit)

---

## 📊 Validation Post-Implémentation

Après implémentation, exécuter:

```bash
# 1. Tests unitaires
pytest tests/test_facade_classification_bugfixes.py -v

# 2. Tests d'intégration
pytest tests/test_integration_*.py -v -m "not slow"

# 3. Diagnostic sur données réelles
python scripts/diagnose_facade_classification_bugs.py \
    --laz_file /path/to/test_tile.laz \
    --bd_topo_file /path/to/buildings.geojson \
    --output_dir /path/to/diagnostic_after_fix \
    --max_buildings 50

# 4. Comparer avant/après
python scripts/compare_classifications.py \
    --before /path/to/before_fix.laz \
    --after /path/to/after_fix.laz \
    --output_dir /path/to/comparison
```

---

## 🚀 Déploiement

1. Créer branche `bugfix/facade-classification-artifacts`
2. Implémenter toutes les corrections
3. Tests complets
4. PR avec description détaillée
5. Review code
6. Merge vers main
7. Tag version 3.0.4 (bugfix)

---

**Commencer maintenant ?** → Phase 1: Module Validation
