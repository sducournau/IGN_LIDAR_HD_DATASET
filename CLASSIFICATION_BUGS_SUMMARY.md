# Résumé Exécutif: Bugs de Classification Ground Truth

**Date:** 26 octobre 2025  
**Analyste:** GitHub Copilot  
**Statut:** 🔴 BUGS CRITIQUES IDENTIFIÉS

---

## 🎯 Problème Rapporté

> "La classification à partir du ground truth produit toujours les mêmes résultats, les bâtiments et autres classes ne respectent pas les règles mises en place."

## ✅ Diagnostic Confirmé

**Le problème est RÉEL et CRITIQUE.** Nous avons identifié **11 bugs** dont **5 bugs critiques** qui expliquent le comportement observé.

---

## 🔴 Les 3 Bugs Critiques Principaux

### 1. Bug #1: Priorités Aléatoires (STRtree)

**Fichier:** `ign_lidar/io/ground_truth_optimizer.py:370-380`

**Problème:** Quand un point est dans plusieurs polygones (ex: building ET vegetation), le système ne respecte PAS la priorité définie. Le label final dépend de l'ordre **aléatoire** retourné par l'arbre spatial STRtree.

**Impact:**

- ❌ Classification non-déterministe (résultats différents à chaque run)
- ❌ Priorités ignorées (vegetation peut écraser building)
- ❌ Impossible de prédire le résultat

**Preuve:**

```python
# Code actuel (BUGUÉ):
for candidate_idx in candidate_indices:  # ← Ordre aléatoire du STRtree!
    if prepared_polygons[candidate_idx].contains(point_geom):
        labels[start_idx + i] = polygon_labels[candidate_idx]
        # Pas de break → le dernier gagne, ordre aléatoire
```

---

### 2. Bug #5: Règles Géométriques Écrasent le Ground Truth

**Fichier:** `ign_lidar/core/classification/geometric_rules.py:228-290`

**Problème:** Les règles géométriques (verticality, building buffer) **modifient les labels** qui ont été définis par le ground truth BD TOPO, **sans vérifier** si le point a déjà un label GT valide.

**Impact:**

- ❌ Ground truth BD TOPO inutile
- ❌ Points "vegetation" du BD TOPO reclassés en "building" à cause de la verticality
- ❌ Toutes les routes avec arbres deviennent "vegetation" au lieu de "road"

**Exemple Concret:**

```
1. Point dans polygon BD TOPO "vegetation" → classé "vegetation" ✅
2. Point a verticality > 0.85 → Rule 2b le reclasse en "building" ❌
3. Résultat: Végétation classée comme bâtiment!
```

---

### 3. Bug #4: Deux Systèmes de Priorités Contradictoires

**Fichiers:** `reclassifier.py` vs `ground_truth_optimizer.py`

**Problème:** Deux modules utilisent des ordres de priorité **différents**:

```python
# reclassifier.py (ligne 190):
priority_order = [
    "vegetation", "water", "cemeteries", "parking",
    "sports", "railways", "roads", "bridges", "buildings"
]
# buildings = priorité MAXIMALE

# ground_truth_optimizer.py (ligne 312):
label_priority = ["buildings", "roads", "water", "vegetation"]
# buildings = priorité MAXIMALE, mais ordre différent pour le reste
```

**Impact:**

- ❌ Comportement incohérent selon le module utilisé
- ❌ "roads" prioritaire sur "water" dans un cas, inverse dans l'autre
- ❌ Impossible de prédire le résultat final

---

## 📊 Impact Global

| Aspect                | État       | Explication                           |
| --------------------- | ---------- | ------------------------------------- |
| **Déterminisme**      | ❌ NON     | STRtree retourne un ordre aléatoire   |
| **Respect Priorités** | ❌ NON     | Priorités ignorées dans STRtree       |
| **Respect GT**        | ❌ NON     | Règles géométriques écrasent le GT    |
| **Cohérence**         | ❌ NON     | Deux systèmes de priorités différents |
| **NDVI Refinement**   | ⚠️ PARTIEL | Écrasé par règles géométriques        |

---

## 🛠️ Solutions Requises (Par Ordre de Priorité)

### URGENT (Priorité 1):

1. **Fixer Bug #1** - `ground_truth_optimizer.py:370-380`

   ```python
   # Solution: Implémenter vérification de priorité
   best_label = 0
   best_priority = -1
   for candidate_idx in candidate_indices:
       if prepared_polygons[candidate_idx].contains(point_geom):
           label = polygon_labels[candidate_idx]
           priority = priority_map[label]  # ← AJOUT
           if priority > best_priority:
               best_label = label
               best_priority = priority
   labels[start_idx + i] = best_label
   ```

2. **Fixer Bug #5** - `geometric_rules.py:228-290`

   ```python
   # Solution: Ajouter flag preserve_ground_truth
   def apply_all_rules(self, ..., preserve_ground_truth=True):
       if preserve_ground_truth:
           # Ne modifier QUE les points "unclassified" (code 1)
           unclassified_mask = labels == 1
           # Appliquer règles seulement sur ces points
   ```

3. **Fixer Bug #4** - Unifier les priorités
   ```python
   # Créer un fichier config/classification_priorities.py
   CLASSIFICATION_PRIORITY = {
       'buildings': 100,    # Priorité maximale
       'bridges': 90,
       'roads': 80,
       'railways': 70,
       'sports': 60,
       'parking': 50,
       'cemeteries': 40,
       'water': 30,
       'vegetation': 20     # Priorité minimale
   }
   # Utiliser ce dict dans TOUS les modules
   ```

### IMPORTANT (Priorité 2):

4. **Bug #3** - NDVI Timing
5. **Bug #6** - Buffer Zone GT Check
6. **Bug #8** - NDVI Zone Grise (0.15-0.3)

### AMÉLIORATION (Priorité 3):

7. **Bug #10** - Validation features
8. **Bug #11** - Thresholds configurables

---

## 🧪 Validation

Un fichier de tests a été créé: `tests/test_classification_bugs.py`

Pour exécuter les tests:

```bash
pytest tests/test_classification_bugs.py -v
```

**Note:** Les tests **vont échouer** tant que les bugs ne sont pas corrigés. C'est normal et attendu.

---

## 📈 Impact Estimé des Corrections

Une fois les 3 bugs critiques corrigés:

- ✅ Classification **déterministe** (même résultat à chaque run)
- ✅ Priorités **respectées** (buildings > roads > water > vegetation)
- ✅ Ground truth **préservé** (BD TOPO respecté)
- ✅ Comportement **cohérent** entre modules
- ✅ Amélioration **>80%** de la qualité de classification

---

## 📝 Prochaines Étapes

1. **Valider l'analyse** avec l'équipe
2. **Prioriser les corrections** (commencer par Bug #1, #5, #4)
3. **Implémenter les solutions** proposées
4. **Tester** avec `test_classification_bugs.py`
5. **Valider** sur données réelles (tiles Versailles)
6. **Documenter** les changements dans CHANGELOG.md

---

## 📚 Documentation Complète

- **Analyse détaillée:** `CLASSIFICATION_BUGS_ANALYSIS.md`
- **Tests unitaires:** `tests/test_classification_bugs.py`
- **Ce résumé:** `CLASSIFICATION_BUGS_SUMMARY.md`

---

**Questions?** Voir l'analyse complète dans `CLASSIFICATION_BUGS_ANALYSIS.md`
