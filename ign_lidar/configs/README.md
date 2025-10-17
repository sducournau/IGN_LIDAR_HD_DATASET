# IGN LiDAR HD - Configuration v4.0 Guide

## 🎯 Vue d'ensemble

Ce dossier contient la **configuration unifiée v4.0** pour IGN LiDAR HD Dataset, remplaçant les systèmes fragmentés des versions précédentes.

### ✨ Nouveautés v4.0

- **Configuration unifiée** : Un seul schéma cohérent pour tous les cas d'usage
- **Presets intelligents** : Configurations prêtes à l'emploi pour différents scenarios
- **Profils hardware** : Optimisations spécifiques par carte graphique
- **Performance GPU** : Résolution de la régression GPU (17% → >80% utilisation)
- **Interface simplifiée** : Réduction de 80% des paramètres CLI nécessaires

---

## 📁 Structure des Configurations

```text
configs/
├── config.yaml              # 🎯 Configuration par défaut
├── presets/                  # 🚀 Presets prêts à l'emploi
│   ├── gpu_optimized.yaml   #     Performance GPU maximale
│   ├── asprs_classification.yaml #  Classification ASPRS standard
│   ├── enrichment_only.yaml #     LAZ enrichis uniquement
│   ├── minimal.yaml         #     Tests rapides/debugging
│   ├── ground_truth_training.yaml # Entraînement avec ground truth
│   ├── architectural_heritage.yaml # Patrimoine architectural
│   ├── building_detection.yaml #   Détection bâtiments optimisée
│   ├── vegetation_analysis.yaml #  Analyse végétation NDVI
│   └── multiscale_analysis.yaml #  Analyse multi-échelle
├── advanced/                 # 🔬 Configurations avancées
│   └── self_supervised_lod2.yaml # Apprentissage auto-supervisé
├── hardware/                 # ⚡ Profils hardware optimisés
│   ├── rtx4080.yaml         #     RTX 4080 (16GB) - Recommandé
│   ├── rtx3080.yaml         #     RTX 3080 (10GB)
│   ├── rtx4090.yaml         #     RTX 4090 (24GB) - Haute performance
│   ├── workstation_cpu.yaml #     CPU haute performance
│   └── cpu_only.yaml        #     Fallback CPU basique
└── README.md                 # 📚 Ce guide
```

## 🚀 Utilisation Rapide

### Commandes Simplifiées

```bash
# Preset GPU optimisé (recommandé RTX 4080)
ign-lidar-hd process --preset gpu_optimized --input /data/tiles --output /data/processed

# Classification ASPRS standard
ign-lidar-hd process --preset asprs_classification --input /data/tiles

# Enrichissement LAZ uniquement (le plus rapide)
ign-lidar-hd process --preset enrichment_only --input /data/tiles

# Test rapide sur un échantillon
ign-lidar-hd process --preset minimal --input /data/test
```

### Avec Profils Hardware

```bash
# Auto-détection + optimisation RTX 4080
ign-lidar-hd process --preset gpu_optimized --hardware rtx4080

# Fallback CPU si pas de GPU
ign-lidar-hd process --preset asprs_classification --hardware cpu_only
```

### Overrides Ciblés (Optionnel)

```bash
# Ajuster uniquement la VRAM si nécessaire
ign-lidar-hd process --preset gpu_optimized \
    processing.gpu.vram_target=0.95

# Activer le cadastre (attention: très lent)
ign-lidar-hd process --preset asprs_classification \
    data_sources.cadastre_enabled=true
```

## 🔧 Structure de Configuration v4.0

### Paramètres Principaux

```yaml
# Métadonnées
config_version: "4.0.0" # Version du schéma
config_name: "default" # Nom de la configuration

# Traitement principal
processing:
  mode: "enriched_only" # enriched_only | patches_only | both
  lod_level: "ASPRS" # ASPRS | LOD2 | LOD3
  use_gpu: true # Activation GPU

  # ⭐ NOUVEAU: GPU centralisé
  gpu:
    features_batch_size: 8_000_000 # Points par batch GPU
    vram_target: 0.85 # % VRAM utilisée
    cuda_streams: 6 # Streams parallèles
    ground_truth_method: "auto" # auto | gpu_chunked | gpu | strtree
    reclassification_mode: "auto" # auto | gpu | cpu

# Features simplifiées
features:
  mode: "asprs_classes" # minimal | asprs_classes | lod2 | lod3 | full
  k_neighbors: 20 # Voisins pour features
  compute_normals: true # Features géométriques de base
  use_rgb: true # Features spectrales

# ⭐ NOUVEAU: Sources de données aplaties
data_sources:
  bd_topo_enabled: true # Activation BD TOPO
  bd_topo_buildings: true # Bâtiments → ASPRS Class 6
  bd_topo_roads: true # Routes → ASPRS Class 11
  bd_topo_water: true # Eau → ASPRS Class 9
  cadastre_enabled: false # Cadastre (lent)
```

### Nouveautés v4.0

1. **GPU Centralisé** : Tous les paramètres GPU dans `processing.gpu`
2. **Data Sources Aplaties** : Paramètres BD TOPO/Cadastre simplifiés
3. **Presets Intelligents** : Configurations prêtes pour chaque usage
4. **Hardware Profiles** : Optimisations par carte graphique
5. **Migration Automatique** : Transition transparente depuis v2.x/v3.0

## 📊 Comparaison des Versions

| Aspect            | v2.x                        | v3.0                     | v4.0 ✨                       |
| ----------------- | --------------------------- | ------------------------ | ----------------------------- |
| **Schémas**       | `processor.*`, `features.*` | `processing.*` mixé      | `processing.*` unifié         |
| **GPU Config**    | Éparpillé dans `features.*` | Partiellement centralisé | Centralisé `processing.gpu.*` |
| **Presets**       | ❌ Aucun                    | ⚠️ Basiques              | ✅ Intelligents avec hardware |
| **CLI Overrides** | 🔴 50+ paramètres           | 🟡 20+ paramètres        | 🟢 <10 paramètres             |
| **Migration**     | ❌ Manuelle                 | ⚠️ Partielle             | ✅ Automatique                |
| **Performance**   | 🔴 CPU fallback fréquent    | 🟡 GPU sous-optimal      | 🟢 GPU optimisé               |

## 🔄 Migration depuis v2.x/v3.0

### Migration Automatique

```bash
# Fichier unique
python scripts/migrate_config_v4.py \
    --input configs/config_old.yaml \
    --output configs_v4/migrated.yaml

# Migration en lot
python scripts/migrate_config_v4.py \
    --batch configs/ \
    --output-dir configs_v4/migrated/

# Dry-run (aperçu)
python scripts/migrate_config_v4.py \
    --input configs/config_old.yaml \
    --dry-run
```

### Correspondances Principales

| v2.x/v3.0                                      | v4.0                                   |
| ---------------------------------------------- | -------------------------------------- |
| `processor.use_gpu`                            | `processing.use_gpu`                   |
| `features.gpu_batch_size`                      | `processing.gpu.features_batch_size`   |
| `features.vram_utilization_target`             | `processing.gpu.vram_target`           |
| `processor.reclassification.acceleration_mode` | `processing.gpu.reclassification_mode` |
| `ground_truth.optimization.force_method`       | `processing.gpu.ground_truth_method`   |

## ⚡ Optimisations de Performance

### GPU Optimisé (RTX 4080)

```yaml
processing:
  gpu:
    features_batch_size: 16_000_000 # 16M points
    vram_target: 0.90 # 90% VRAM
    cuda_streams: 8 # 8 streams
    ground_truth_method: "gpu_chunked" # Force GPU
    reclassification_mode: "gpu" # Force GPU
```

**Résultat attendu** :

- Utilisation GPU : >85% (vs 17% avant)
- Temps par tuile : 30-60s (vs 5-10 minutes)
- Ground truth : 10-100× plus rapide

### CPU Fallback Automatique

```yaml
processing:
  gpu:
    ground_truth_method: "auto" # Auto-fallback si GPU OOM
    reclassification_mode: "auto" # Auto-fallback
```

Le système bascule automatiquement en CPU si :

- GPU non disponible
- Mémoire GPU insuffisante
- Erreurs CUDA

## 🛠️ Développement et Debug

### Configuration de Debug

```bash
# Test rapide avec logs détaillés
ign-lidar-hd process --preset minimal \
    monitoring.log_level=DEBUG \
    monitoring.enable_profiling=true
```

### Monitoring GPU

```bash
# Pendant le processing, dans un autre terminal
./scripts/gpu_monitor.sh 300  # Monitor 5 minutes

# Validation complète
./scripts/validate_gpu_acceleration.sh
```

### Création de Nouveaux Presets

```yaml
# configs_v4/presets/mon_preset.yaml
defaults:
  - ../config # Hérite de la base
  - _self_ # Override local

# Mes modifications spécifiques
processing:
  gpu:
    features_batch_size: 12_000_000 # Ajusté pour ma config

data_sources:
  cadastre_enabled: true # J'active le cadastre
```

## 📚 Ressources Additionnelles

- **Scripts de Validation** : `scripts/validate_gpu_acceleration.sh`
- **Monitoring** : `scripts/gpu_monitor.sh`
- **Migration** : `scripts/migrate_config_v4.py`
- **Audit** : `scripts/audit_configs.py`

## 🆘 Dépannage

### Problèmes Fréquents

**GPU non utilisé (17% utilisation)** :

```yaml
processing:
  gpu:
    reclassification_mode: "gpu" # Au lieu de "cpu"
    ground_truth_method: "gpu_chunked" # Au lieu de "auto"
```

**Erreurs CUDA Out of Memory** :

```yaml
processing:
  gpu:
    features_batch_size: 4_000_000 # Réduire la batch size
    vram_target: 0.70 # Réduire VRAM target
```

**Performance lente** :

```bash
# Utiliser le preset optimisé
ign-lidar-hd process --preset gpu_optimized
```

---

**Version** : 4.0.0  
**Date** : 2025-10-17  
**Maintenance** : Configuration unifiée, performances optimisées
