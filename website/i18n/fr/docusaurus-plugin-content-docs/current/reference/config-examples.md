---
sidebar_position: 1
title: Exemples de Configuration
description: Exemples de configuration YAML réutilisables pour des workflows courants
---

# Exemples de Configuration

Cette page contient des extraits de configuration YAML réutilisables qui peuvent être référencés depuis plusieurs guides pour éviter la duplication.

## Modèles de Configuration de Base

### Configuration Minimale

```yaml title="config/minimal.yaml"
# Configuration minimale pour les tests
global:
  num_workers: 2
  verbose: true

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "full"
```

### Configuration de Production

```yaml title="config/production.yaml"
# Configuration prête pour la production
global:
  num_workers: 8
  verbose: false
  log_level: "INFO"

download:
  bbox: "2.3, 48.8, 2.4, 48.9" # Zone de Paris
  output: "data/raw"
  max_tiles: 50
  tile_selection_strategy: "urban"

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "full"
  add_rgb: true
  use_gpu: true
  rgb_cache_dir: "cache/orthophotos"

patch:
  input_dir: "data/enriched"
  output: "data/patches"
  patch_size: 150.0
  num_points: 16384
  lod_level: "LOD2"
  overlap: 0.1
```

### Configuration Optimisée GPU

```yaml title="config/gpu-optimized.yaml"
# Traitement accéléré par GPU
global:
  num_workers: 4
  verbose: true

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "full"
  use_gpu: true
  gpu_batch_size: 10000
  add_rgb: true
  rgb_cache_dir: "cache/orthophotos"

patch:
  input_dir: "data/enriched"
  output: "data/patches"
  use_gpu: true
  patch_size: 150.0
  num_points: 16384
```

## Modèles Spécifiques par Étape

### Téléchargement Uniquement

```yaml title="config/download-only.yaml"
download:
  bbox: "2.3, 48.8, 2.4, 48.9"
  output: "data/raw"
  max_tiles: 10
```

### Enrichissement Uniquement

```yaml title="config/enrich-only.yaml"
enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "full"
  add_rgb: true
```

### Patches Uniquement

```yaml title="config/patch-only.yaml"
patch:
  input_dir: "data/enriched"
  output: "data/patches"
  patch_size: 150.0
  num_points: 16384
```

## Configurations Régionales

### Zone Urbaine Dense

```yaml title="config/urbain-dense.yaml"
# Configuration pour zones urbaines denses
global:
  num_workers: 6
  verbose: true

download:
  tile_selection_strategy: "building_rich"
  max_tiles: 100

enrich:
  mode: "full"
  add_rgb: true
  use_gpu: true

  # Paramètres optimisés pour l'urbain dense
  building_detection:
    min_height: 3.0
    max_footprint: 5000
    roof_complexity: high

  noise_filtering:
    traffic_noise: true
    reflection_noise: true
    intensity_threshold: 0.15

patch:
  lod_level: "LOD3" # Plus de détails pour l'urbain
  num_points: 32768
  overlap: 0.15
```

### Zone Rurale

```yaml title="config/rural.yaml"
# Configuration pour zones rurales
global:
  num_workers: 4
  verbose: true

download:
  tile_selection_strategy: "random"
  max_tiles: 20

enrich:
  mode: "full" # Classification complète pour zones mixtes
  add_rgb: true

  # Paramètres adaptés au rural
  vegetation_analysis:
    forest_detection: true
    agricultural_fields: true

  building_detection:
    residential_focus: true
    farm_buildings: true

patch:
  lod_level: "LOD2"
  patch_size: 200.0 # Patches plus grandes pour zones moins denses
  num_points: 16384
```

## Cas d'Usage Spécialisés

### Développement et Tests

```yaml title="config/dev-test.yaml"
# Configuration pour développement et tests
global:
  num_workers: 2
  verbose: true
  debug: true

download:
  max_tiles: 3 # Limité pour tests rapides

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "full"
  use_gpu: false # CPU pour débuggage

patch:
  input_dir: "data/enriched"
  output: "data/patches"
  patch_size: 100.0 # Patches plus petites pour tests
  num_points: 8192
  max_patches: 10 # Limité pour tests
```

### Traitement par Lots

```yaml title="config/batch-processing.yaml"
# Configuration pour traitement de grands volumes
global:
  num_workers: 12
  verbose: false
  log_level: "WARNING"

download:
  max_tiles: 500
  parallel_downloads: 8

enrich:
  use_gpu: true
  gpu_batch_size: 20000
  parallel_gpu_streams: 2

patch:
  use_gpu: true
  parallel_patch_creation: true
  memory_limit: "16GB"
```
