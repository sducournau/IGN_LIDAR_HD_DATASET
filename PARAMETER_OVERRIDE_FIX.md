# 🚨 CORRECTIONS - Paramètres Écrasés par Hydra

## Problèmes Identifiés dans les Logs :

### 1. **GPU Batch Size Écrasé**

- **Configuration** : `features.gpu_batch_size: 16000000` (16M)
- **Logs** : `🚀 GPU mode enabled (batch_size=8,000,000)` ❌
- **Cause** : Defaults Hydra qui écrasent nos paramètres
- **Solution** : Forcer via ligne de commande dans `run_forced_ultra_fast.sh`

### 2. **Cadastre Toujours Activé**

- **Configuration** : `cadastre_enabled: false`
- **Logs** : `✅ Enabled data sources: BD TOPO (...), Cadastre` ❌
- **Cause** : Conflict entre `cadastre_enabled: false` et `cadastre.enabled: true` par défaut
- **Solution** : Double désactivation forcée dans le script

### 3. **NDVI Toujours Activé**

- **Configuration** : `use_ndvi: false`
- **Logs** : `NDVI refinement: ENABLED` ❌
- **Cause** : Paramètre écrasé par les defaults
- **Solution** : Force `ground_truth.use_ndvi=false` en ligne de commande

### 4. **Architecture Potentiellement Écrasée**

- **Configuration** : `architecture: direct`
- **Logs** : Pas de confirmation de l'architecture utilisée
- **Solution** : Double force `processor.architecture=direct` + `processing.architecture=direct`

## ✅ Corrections Appliquées :

### **1. Configuration Modifiée (`config_asprs_rtx4080.yaml`)**

```yaml
# FORCE override des defaults Hydra
defaults:
  - override /processor: null
  - override /features: null
  - override /preprocess: null
  - override /stitching: null
  - override /ground_truth: null
  - _self_

# Double désactivation cadastre
cadastre_enabled: false
cadastre:
  enabled: false
  use_cache: false
  optimization_level: disabled
```

### **2. Script avec Paramètres Forcés (`run_forced_ultra_fast.sh`)**

Force TOUS les paramètres critiques via ligne de commande :

```bash
ign-lidar-hd process \
    --config-file "$CONFIG" \
    # ARCHITECTURE FORCÉE
    processor.architecture=direct \
    processing.architecture=direct \
    processor.generate_patches=false \
    processing.generate_patches=false \

    # GPU BATCH SIZE FORCÉ
    features.gpu_batch_size=16000000 \
    features.use_gpu=true \

    # CADASTRE FORCÉ DISABLED
    data_sources.cadastre_enabled=false \
    data_sources.cadastre.enabled=false \

    # NDVI FORCÉ DISABLED
    ground_truth.use_ndvi=false \
    ground_truth.fetch_rgb_nir=false \
    features.use_nir=false \
    features.compute_ndvi=false \

    # PREPROCESSING FORCÉ DISABLED
    preprocess.enabled=false \
    stitching.enabled=false
```

### **3. Test sur Fichier Unique (`test_single_file.sh`)**

- Teste la configuration sur UN seul fichier
- Valide que les paramètres sont bien pris en compte
- Vérifie qu'aucun patch n'est généré

## 🎯 Validation Attendue :

### **Dans les Logs (Après Correction) :**

```
✅ GPU batch/chunk size: 16,000,000 points (au lieu de 8M)
✅ Processing mode: enriched_only
✅ Enabled data sources: BD TOPO (roads, buildings, water) (PAS de Cadastre)
✅ NDVI refinement: DISABLED (au lieu de ENABLED)
✅ Generated 0 patches (au lieu de 64 patches)
```

## 🚀 Ordre d'Exécution :

1. **Test sur un fichier** : `./test_single_file.sh`
2. **Si test OK** : `./run_forced_ultra_fast.sh`
3. **Monitoring** : `watch -n 1 nvidia-smi`

## ⚡ Performance Attendue :

- **Avant** : 2+ heures/tuile avec génération de patches
- **Après** : 5-10 minutes/tuile, LAZ enrichis uniquement
- **GPU Usage** : 90% VRAM, 16M points/batch
- **Output** : Fichiers `*_enriched.laz` seulement (pas de patches)
