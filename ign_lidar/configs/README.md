# IGN LiDAR HD - Configuration V5.0 Guide

## 🎯 Vue d'ensemble

Ce dossier contient la **configuration simplifiée V5.0** pour IGN LiDAR HD Dataset, qui harmonise et consolide le système de configuration.

### ✨ Nouveautés V5.0

- **Configuration simplifiée** : Réduction de 60% de la complexité
- **Optimisations intégrées** : Toutes les optimisations sont intégrées dans FeatureOrchestrator V5
- **Structure épurée** : 5 configurations de base au lieu de 14
- **Suppression de la rétrocompatibilité** : Configuration plus claire et maintenable
- **Harmonisation** : Suppression des préfixes "enhanced" et "unified"
- **Performance maintenue** : >80% d'utilisation GPU

---

## 📁 Structure des Configurations V5.0

```text
configs/
├── config.yaml              # 🎯 Configuration par défaut V5
├── config.yaml           # 🎯 Configuration V5 (identique à config.yaml)
├── base/                     # 📦 5 configurations de base V5
│   ├── processor.yaml       #     Paramètres de traitement
│   ├── features.yaml        #     Calcul de features
│   ├── data_sources.yaml    #     Sources de données
│   ├── output.yaml          #     Formats de sortie
│   ├── monitoring.yaml      #     Logging et monitoring
│   └── example_*.yaml       #     Exemples d'utilisation
├── presets/                  # 🚀 Presets prêts à l'emploi V5
│   ├── gpu_optimized.yaml   #     Performance GPU maximale
│   ├── asprs_classification.yaml #  Classification ASPRS standard
│   ├── enrichment_only.yaml #     LAZ enrichis uniquement
│   ├── minimal.yaml         #     Tests rapides/debugging
│   ├── ground_truth_training.yaml # Entraînement avec ground truth
│   ├── architectural_heritage.yaml # Patrimoine architectural
│   ├── building_detection.yaml #   Détection bâtiments optimisée
│   ├── vegetation_analysis.yaml #  Analyse végétation NDVI
│   └── multiscale_analysis.yaml #  Analyse multi-échelle
├── advanced/                 # 🔬 Configurations avancées V5
│   └── self_supervised_lod2.yaml # Apprentissage auto-supervisé
├── hardware/                 # ⚡ Profils hardware optimisés V5
│   ├── rtx4080.yaml         #     RTX 4080 (16GB) - Recommandé
│   ├── rtx3080.yaml         #     RTX 3080 (10GB)
│   ├── rtx4090.yaml         #     RTX 4090 (24GB) - Haute performance
│   ├── workstation_cpu.yaml #     CPU haute performance
│   └── cpu_only.yaml        #     Fallback CPU basique
├── MIGRATION_V5_GUIDE.md     # 📖 Guide de migration V4→V5
└── README.md                 # 📚 Ce guide
```

## 🚀 Utilisation Rapide

### Commandes Simplifiées V5

```bash
# Preset GPU optimisé (recommandé RTX 4080)
ign-lidar-hd process --config-name gpu_optimized --input /data/tiles --output /data/processed

# Classification ASPRS standard
ign-lidar-hd process --config-name asprs_classification --input /data/tiles

# Enrichissement LAZ uniquement (le plus rapide)
ign-lidar-hd process --config-name enrichment_only --input /data/tiles

# Test rapide sur un échantillon
ign-lidar-hd process --config-name minimal --input /data/test
```

### Avec Profils Hardware

```bash
# Auto-détection + optimisation RTX 4080
ign-lidar-hd process --config-name gpu_optimized hardware=rtx4080

# Fallback CPU si pas de GPU
ign-lidar-hd process --config-name asprs_classification hardware=cpu_only
```

### Overrides Ciblés (Optionnel)

```bash
# Ajuster uniquement la VRAM si nécessaire
ign-lidar-hd process --config-name gpu_optimized \
    processor.gpu_memory_target=0.95

# Activer le cadastre (attention: très lent)
ign-lidar-hd process --config-name asprs_classification \
    data_sources.cadastre_enabled=true
```

## 🔧 Structure de Configuration V5.0

### Paramètres Principaux V5

```yaml
# Métadonnées
config_version: "5.0.0" # Version du schéma
config_name: "default" # Nom de la configuration

# Traitement principal (V5 simplifié)
processor:
  mode: "enriched_only" # enriched_only | patches_only | both
  lod_level: "ASPRS" # ASPRS | LOD2 | LOD3
  use_gpu: true # Activation GPU

  # GPU settings (V5 simplifié)
  gpu_batch_size: 8_000_000 # Points par batch GPU
  gpu_memory_target: 0.85 # Utilisation VRAM cible
  num_workers: 1 # GPU works best with single worker

# Features simplifiées V5
features:
  mode: "asprs_classes" # minimal | asprs_classes | lod2 | lod3 | full
  k_neighbors: 20 # Voisins pour features
  compute_normals: true # Features géométriques de base
  use_rgb: true # Features spectrales

# Sources de données V5
data_sources:
  bd_topo_enabled: true # Activation BD TOPO
  bd_topo_buildings: true # Bâtiments → ASPRS Class 6
  bd_topo_roads: true # Routes → ASPRS Class 11
  bd_topo_water: true # Eau → ASPRS Class 9
  cadastre_enabled: false # Cadastre (lent)

# Optimisations V5 (intégrées)
optimizations:
  enable_caching: true
  enable_parallel_processing: true
  enable_auto_tuning: true
  adaptive_parameters: true
```

### Nouveautés V5.0

1. **Simplification** : Réduction de 60% de la complexité de configuration
2. **Optimisations Intégrées** : Toutes les optimisations dans FeatureOrchestrator V5
3. **5 Configs de Base** : processor, features, data_sources, output, monitoring
4. **Harmonisation** : Suppression des préfixes "enhanced" et "unified"
5. **Suppression Rétrocompatibilité** : Configuration plus claire

## 📊 Comparaison des Versions

| Aspect            | v4.0                    | V5.0 ✨                       |
| ----------------- | ----------------------- | ----------------------------- |
| **Base Configs**  | 14 fichiers             | 5 fichiers (60% réduction)    |
| **Paramètres**    | 200+ paramètres         | 80 paramètres (60% réduction) |
| **Orchestrator**  | Base + Enhanced séparés | FeatureOrchestrator V5 unifié |
| **Optimizations** | Configuration séparée   | Intégrées dans le core        |
| **Rétrocompat.**  | Support V2.x/V3.0       | V5 uniquement (clean)         |
| **Performance**   | GPU optimisé (>80%)     | GPU optimisé maintenu         |

## 🔄 Migration depuis V4.0

Voir le guide de migration détaillé : [`MIGRATION_V5_GUIDE.md`](MIGRATION_V5_GUIDE.md)

### Changements Principaux V4 → V5

| V4.0                                 | V5.0                            |
| ------------------------------------ | ------------------------------- |
| `processing.use_gpu`                 | `processor.use_gpu`             |
| `processing.gpu.features_batch_size` | `processor.gpu_batch_size`      |
| `processing.gpu.vram_target`         | `processor.gpu_memory_target`   |
| `processing.gpu.ground_truth_method` | `processor.ground_truth_method` |
| Multiples base configs               | 5 base configs simplifiés       |

## ⚡ Optimisations de Performance V5

### GPU Optimisé (RTX 4080) - V5

```yaml
processor:
  use_gpu: true
  gpu_batch_size: 16_000_000 # 16M points
  gpu_memory_target: 0.90 # 90% VRAM
  num_workers: 1

optimizations:
  enable_caching: true
  cache_max_size_mb: 200
  enable_parallel_processing: true
  enable_auto_tuning: true
  adaptive_parameters: true
```

**Résultat attendu** :

- Utilisation GPU : >85%
- Temps par tuile : 30-60s
- Ground truth : 10-100× plus rapide
- Optimisations automatiques intégrées

### CPU Fallback Automatique

```yaml
processor:
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
processor:
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
processor:
  gpu:
    reclassification_mode: "gpu" # Au lieu de "cpu"
    ground_truth_method: "gpu_chunked" # Au lieu de "auto"
```

**Erreurs CUDA Out of Memory** :

```yaml
processor:
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
