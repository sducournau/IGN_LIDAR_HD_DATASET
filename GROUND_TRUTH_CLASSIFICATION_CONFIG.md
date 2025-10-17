# 📊 Configuration Reclassification Ground Truth - RTX 4080

## 🎯 Objectif : Tuiles Enrichies + Reclassification BD TOPO

### ✅ **Ajustements de Classification Appliqués :**

#### **1. Reclassification Activée**

```yaml
processor:
  reclassification:
    enabled: true # ✅ ACTIVÉ pour ground truth
    acceleration_mode: cpu # CPU pour stabilité
    chunk_size: 2_000_000 # Chunks moyens pour performance
    use_geometric_rules: true # Règles géométriques activées
  apply_reclassification_inline: true # ✅ Reclassification inline
```

#### **2. Ground Truth Configuration**

```yaml
ground_truth:
  enabled: true # ✅ BD TOPO ground truth
  update_classification: true # ✅ Mise à jour des classes
  apply_reclassification: true # ✅ Application reclassification
  use_ndvi: false # ❌ Pas de NDVI (lent)
  fetch_rgb_nir: false # ❌ Pas de fetch RGB/NIR (lent)
```

#### **3. Sources de Données - BD TOPO Seulement**

```yaml
data_sources:
  # BD TOPO - FEATURES ESSENTIELS
  bd_topo_enabled: true
  bd_topo_buildings: true # ✅ ASPRS Class 6 (Buildings)
  bd_topo_roads: true # ✅ ASPRS Class 11 (Roads)
  bd_topo_water: true # ✅ ASPRS Class 9 (Water)
  bd_topo_vegetation: false # ❌ Désactivé (lent)

  # CADASTRE - COMPLÈTEMENT DÉSACTIVÉ
  cadastre_enabled: false # ❌ Pas de cadastre (très lent)
  cadastre:
    enabled: false # ❌ Double désactivation
    use_cache: false # ❌ Pas de cache cadastre
```

#### **4. Classification Methods**

```yaml
classification:
  enabled: true
  methods:
    ground_truth: true # ✅ PRIMARY: BD TOPO ground truth
    geometric: true # ✅ SECONDARY: Règles géométriques
    ndvi: false # ❌ Pas de NDVI (pas de NIR)
    forest_types: false
    crop_types: false
```

### 🏗️ **Processus de Classification :**

#### **Étape 1 : Ground Truth BD TOPO**

- **Buildings** → ASPRS Class 6 (Bâtiments)
- **Roads** → ASPRS Class 11 (Routes)
- **Water** → ASPRS Class 9 (Eau)

#### **Étape 2 : Règles Géométriques**

- Points non classés par ground truth
- Utilise planarity, height, normals
- Classifications géométriques ASPRS

#### **Étape 3 : Enrichissement Final**

- Sauvegarde LAZ enrichi avec nouvelles classifications
- Format : `*_enriched_reclassified.laz`

### 🚀 **Scripts Disponibles :**

#### **Test Rapide (1 fichier) :**

```bash
./test_ground_truth_reclassification.sh
```

#### **Traitement Complet :**

```bash
./run_ground_truth_reclassification.sh
```

### 📋 **Validation Attendue :**

#### **Logs de Succès :**

```
[INFO] Ground truth processing: ENABLED
[INFO] Classification update: ENABLED
[INFO] NDVI refinement: DISABLED
[INFO] Enabled data sources: BD TOPO (roads, buildings, water)
[INFO] Reclassification: ENABLED (CPU mode)
[INFO] Generated 0 patches (enriched_only mode)
[INFO] ✅ Ground truth applied: X points changed (Y%)
[INFO] ✅ Reclassification completed
```

#### **Fichiers de Sortie :**

```
LHD_FXX_XXXX_YYYY_enriched_reclassified.laz  ✅
(Pas de fichiers *patch*.laz)                 ✅
```

### ⚡ **Performance Attendue :**

- **Temps** : 10-20 minutes/tuile (avec reclassification)
- **Qualité** : Classification précise via BD TOPO
- **Vitesse** : Pas de cadastre = +50% plus rapide
- **GPU** : Utilisation optimale pour features, CPU pour reclassification

### 🔧 **Points de Contrôle :**

1. **Cadastre désactivé** → Pas de messages cadastre dans logs
2. **NDVI désactivé** → Pas de fetch RGB/NIR
3. **Reclassification active** → Messages reclassification dans logs
4. **Ground truth BD TOPO** → Messages "X points changed"
5. **Pas de patches** → Seulement fichiers enriched

### 🚨 **Si Problèmes :**

#### **Reclassification échoue :**

- Désactiver `processor.reclassification.enabled=false`
- Garder seulement `ground_truth.enabled=true`

#### **Cadastre toujours actif :**

- Vérifier logs pour messages "cadastral parcels"
- Forcer `data_sources.cadastre_enabled=false` en ligne de commande

#### **Patches générés :**

- Vérifier `processor.architecture=direct`
- Forcer `processor.generate_patches=false`
