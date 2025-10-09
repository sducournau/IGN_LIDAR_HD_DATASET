---
sidebar_position: 1
title: "Configuration" Examples
description: Reusable YAML configuration examples for common workflows
---

<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->
<!-- Ce fichier est un modèle qui nécessite une traduction manuelle. -->
<!-- Veuillez traduire le contenu ci-dessous en conservant : -->
<!-- - Le frontmatter (métadonnées en haut) -->
<!-- - Les blocs de code (traduire uniquement les commentaires) -->
<!-- - Les liens et chemins de fichiers -->
<!-- - La structure Markdown -->



# Configuration Examples

This page contains reusable YAML configuration snippets that can be referenced from multiple guides to avoid duplication.

## Basic Configuration Templates

### Minimal Configuration

```yaml title="config/minimal.yaml"
# Minimal configuration for testing
global:
  num_workers: 2
  verbose: true

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "full"
```

### Production Configuration

```yaml title="config/production.yaml"
# Production-ready configuration
global:
  num_workers: 8
  verbose: false
  log_level: "INFO"

download:
  bbox: "2.3, 48.8, 2.4, 48.9" # Paris area
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

### GPU-Optimized Configuration

```yaml title="config/gpu-optimized.yaml"
# GPU-accelerated processing
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

## Stage-Specific Templates

### Download Only

```yaml title="config/download-only.yaml"
download:
  bbox: "2.3, 48.8, 2.4, 48.9"
  output: "data/raw"
  max_tiles: 10
```

### Enrich Only

```yaml title="config/enrich-only.yaml"
enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "full"
  add_rgb: true
```

### Patch Only

```yaml title="config/patch-only.yaml"
patch:
  input_dir: "data/enriched"
  output: "data/patches"
  patch_size: 150.0
  num_points: 16384
```
