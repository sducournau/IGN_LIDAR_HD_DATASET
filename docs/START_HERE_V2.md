# 🚀 Classification V2 - Start Here!

**Problème résolu :** Taux élevé de points non classifiés (30-40% blancs)  
**Solution :** Configuration V2 avec 12 paramètres plus agressifs  
**Temps :** 15-20 minutes de test

---

## ⚡ Quick Start (3 Commandes)

```bash
# 1. Traiter avec configuration V2 (déjà prête!)
ign-lidar-hd process \
  -c examples/config_asprs_bdtopo_cadastre_cpu_fixed.yaml \
  input_dir="chemin/vers/tuile" \
  output_dir="output/v2_test"

# 2. Diagnostic
python scripts/diagnose_classification.py output/v2_test/tile_enriched.laz

# 3. Visualisation
python scripts/visualize_classification.py output/v2_test/tile_enriched.laz resultat.png
```

**Objectif :** Zones blanches réduites de 50-75%

---

## 📚 Documentation

### Français 🇫🇷

- **CLASSIFICATION_V2_SUMMARY.md** - Résumé exécutif complet en français

### English 🇬🇧

- **QUICK_START_V2.md** - Quick start guide with V2 fixes
- **CLASSIFICATION_AUDIT_CORRECTION.md** - Corrected problem analysis
- **CLASSIFICATION_AUDIT_INDEX.md** - Complete documentation index

---

## 🎯 Comprendre le Problème

### ✅ Ce Qui Fonctionne

- Bâtiments détectés = **Vert clair** visible ✅
- Eau détectée = **Rose/magenta** visible ✅

### ❌ Le Problème

- Trop de points **BLANCS** (non classifiés) : 30-40%
- Couverture bâtiments incomplète : 60-70%

### 🔧 La Solution V2

- Réduire seuils de confiance : 0.55 → 0.40
- Reclassification plus agressive : 0.75 → 0.50
- 12 paramètres optimisés au total

---

## 📊 Résultat Attendu

| Métrique             | Avant  | Après V2 |
| -------------------- | ------ | -------- |
| Points blancs        | 30-40% | <15%     |
| Couverture bâtiments | 60-70% | 85-95%   |

---

## 🔗 Liens Rapides

- 🇫🇷 [Résumé Complet](CLASSIFICATION_V2_SUMMARY.md)
- 🇬🇧 [Quick Start V2](QUICK_START_V2.md)
- 📋 [Index Complet](CLASSIFICATION_AUDIT_INDEX.md)
- 🔧 [Analyse Corrigée](CLASSIFICATION_AUDIT_CORRECTION.md)

---

**Fichier à utiliser :** `examples/config_asprs_bdtopo_cadastre_cpu_fixed.yaml` ✅
