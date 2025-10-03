---
sidebar_position: 4
title: Accélération GPU
description: Utiliser l'accélération GPU pour un traitement plus rapide
keywords: [gpu, cuda, performance, accélération, cupy, rapids]
---

Ce guide explique comment utiliser l'accélération GPU avec IGN LiDAR HD Dataset pour un calcul de caractéristiques significativement plus rapide.

## Vue d'Ensemble

L'accélération GPU peut fournir une **accélération de 4-10x** pour le calcul des caractéristiques par rapport au traitement CPU, particulièrement utile pour les grands jeux de données LiDAR.

### Avantages

- ⚡ **4-10x plus rapide** calcul des caractéristiques
- 🔄 **Basculement automatique vers CPU** quand GPU indisponible
- 📦 **Aucune modification de code** requise - ajoutez simplement un flag
- 🎯 **Prêt pour la production** avec gestion complète des erreurs

### Prérequis

- **Matériel:** GPU NVIDIA avec support CUDA
- **Logiciel:** CUDA Toolkit 11.0 ou supérieur
- **Paquets Python:** CuPy (et optionnellement RAPIDS cuML)

## Installation

### Étape 1 : Vérifier la Disponibilité CUDA

D'abord, vérifiez que vous avez un GPU NVIDIA et CUDA installé :

```bash
# Vérifier si vous avez un GPU NVIDIA
nvidia-smi

# Devrait afficher les infos de votre GPU et la version CUDA
```

Si `nvidia-smi` n'est pas trouvé, vous devez d'abord installer les pilotes NVIDIA et le CUDA Toolkit.

### Étape 2 : Installer CUDA Toolkit

Visitez [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) et suivez les instructions pour votre OS.

**Versions recommandées :**

- CUDA 11.8 (plus compatible)
- CUDA 12.x (dernières fonctionnalités)

### Étape 3 : Installer les Dépendances GPU Python

```bash
# Option 1 : Support GPU basique avec CuPy (recommandé pour la plupart des utilisateurs)
pip install ign-lidar-hd[gpu]

# Option 2 : GPU avancé avec RAPIDS cuML (meilleures performances)
pip install ign-lidar-hd[gpu-full]

# Option 3 : RAPIDS via conda (recommandé pour RAPIDS cuML)
conda install -c rapidsai -c conda-forge -c nvidia cuml
pip install ign-lidar-hd[gpu]

# Option 4 : Installation manuelle
# Pour CUDA 11.x
pip install cupy-cuda11x
pip install cuml-cu11  # Optionnel : RAPIDS cuML

# Pour CUDA 12.x
pip install cupy-cuda12x
pip install cuml-cu12  # Optionnel : RAPIDS cuML
```

**Recommandations d'Installation :**

- **CuPy uniquement** (`[gpu]`): Installation la plus facile, accélération 5-6x
- **CuPy + RAPIDS** (`[gpu-full]`): Meilleures performances, jusqu'à 10x d'accélération
- **Conda pour RAPIDS**: Plus fiable pour les dépendances RAPIDS cuML

### Étape 4 : Vérifier l'Installation

```python
from ign_lidar.features_gpu import GPU_AVAILABLE, CUML_AVAILABLE

print(f"GPU (CuPy) disponible: {GPU_AVAILABLE}")
print(f"RAPIDS cuML disponible: {CUML_AVAILABLE}")
```

Sortie attendue :

```text
GPU (CuPy) disponible: True
RAPIDS cuML disponible: True
```

## Utilisation

### Interface en Ligne de Commande

Ajoutez simplement le flag `--use-gpu` à n'importe quelle commande `enrich` :

```bash
# Utilisation basique
ign-lidar-hd enrich \
  --input tiles/ \
  --output enriched/ \
  --use-gpu

# Avec plusieurs workers
ign-lidar-hd enrich \
  --input tiles/ \
  --output enriched/ \
  --use-gpu \
  --num-workers 4

# Mode bâtiment avec GPU
ign-lidar-hd enrich \
  --input raw_tiles/ \
  --output pre_tiles/ \
  --mode building \
  --use-gpu \
  --num-workers 6
```

### API Python

```python
from ign_lidar import LiDARProcessor
from pathlib import Path

# Initialiser le processeur avec support GPU
processor = LiDARProcessor(
    lod_level="LOD2",
    use_gpu=True  # Active l'accélération GPU
)

# Traiter une dalle avec GPU
patches = processor.process_tile(
    Path("data/tile.laz"),
    Path("output/")
)

# Traitement par lots avec GPU
patches = processor.process_directory(
    Path("data/tiles/"),
    Path("output/patches/"),
    num_workers=4  # GPU + traitement parallèle
)
```

## Performances Attendues

### Benchmarks

Tests effectués sur un système avec :

- **GPU:** NVIDIA RTX 3080 (10GB VRAM)
- **CPU:** Intel i9-10900K (10 cores, 20 threads)
- **Dalle test:** 1.2 million de points

| Configuration             | Temps de Traitement | Accélération    |
| ------------------------- | ------------------- | --------------- |
| CPU uniquement (1 worker) | 45.2s               | 1.0x (baseline) |
| CPU (4 workers)           | 18.8s               | 2.4x            |
| GPU (CuPy uniquement)     | 8.1s                | 5.6x            |
| GPU (CuPy + RAPIDS)       | 4.7s                | 9.6x            |

### Facteurs de Performance

**Quand le GPU est plus rapide :**

- 🚀 Grandes dalles (>500K points)
- 🔢 Calculs intensifs de caractéristiques
- 📊 Nombreuses itérations (lots de dalles)

**Quand le CPU peut être compétitif :**

- 📁 Petites dalles (<100K points)
- 💾 Traitement limité par I/O
- ⚡ Surcharge de transfert GPU

## Configuration YAML

```yaml
global:
  use_gpu: true # Active GPU pour toutes les étapes

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "building"
  num_workers: 4 # GPU + traitement parallèle
```

## Dépannage

### GPU Non Détecté

```python
# Vérifier la disponibilité CUDA
import cupy as cp
print(cp.cuda.is_available())  # Devrait être True

# Vérifier la version CUDA
print(cp.cuda.runtime.runtimeGetVersion())
```

**Solutions :**

1. Vérifier que les pilotes NVIDIA sont installés : `nvidia-smi`
2. Réinstaller CuPy pour votre version CUDA
3. Vérifier les variables d'environnement CUDA

### Erreurs de Mémoire GPU

```text
cupy.cuda.memory.OutOfMemoryError: Out of memory
```

**Solutions :**

1. Réduire le nombre de workers : `--num-workers 1`
2. Traiter des dalles plus petites
3. Utiliser un GPU avec plus de VRAM
4. Basculer vers CPU : enlever `--use-gpu`

### Basculement vers CPU

La bibliothèque bascule automatiquement vers CPU si :

- GPU non disponible
- CUDA non installé
- CuPy non installé
- Erreurs de mémoire GPU

```text
⚠️  GPU non disponible, utilisation du CPU
```

## Meilleures Pratiques

### 1. Optimiser l'Utilisation de la Mémoire

```python
# Traiter par lots pour les grands ensembles de données
processor = LiDARProcessor(
    lod_level="LOD2",
    use_gpu=True,
    batch_size=10  # Traiter 10 dalles à la fois
)
```

### 2. Combiner GPU et Traitement Parallèle

```bash
# Utiliser plusieurs workers avec GPU
ign-lidar-hd enrich \
  --input tiles/ \
  --output enriched/ \
  --use-gpu \
  --num-workers 4  # 4 processus GPU en parallèle
```

### 3. Surveiller l'Utilisation GPU

```bash
# Dans un terminal séparé
watch -n 1 nvidia-smi

# Ou utiliser
gpustat -i 1
```

### 4. Profiler les Performances

```python
import time
from ign_lidar import LiDARProcessor

# Comparer CPU vs GPU
for use_gpu in [False, True]:
    processor = LiDARProcessor(lod_level="LOD2", use_gpu=use_gpu)

    start = time.time()
    processor.process_tile("data/tile.laz", "output/")
    elapsed = time.time() - start

    backend = "GPU" if use_gpu else "CPU"
    print(f"{backend}: {elapsed:.2f}s")
```

## Configuration Avancée

### Variables d'Environnement

```bash
# Forcer l'utilisation CPU même avec GPU disponible
export IGN_LIDAR_FORCE_CPU=1

# Définir le device GPU
export CUDA_VISIBLE_DEVICES=0  # Utiliser le GPU 0

# Limiter la mémoire GPU
export CUPY_GPU_MEMORY_LIMIT=8GB
```

### Sélection du Device

```python
from ign_lidar import LiDARProcessor
import cupy as cp

# Sélectionner un GPU spécifique
cp.cuda.Device(0).use()  # Utiliser le GPU 0

processor = LiDARProcessor(
    lod_level="LOD2",
    use_gpu=True
)
```

## Prochaines Étapes

- 📖 Voir [Guide d'Utilisation de Base](basic-usage.md) pour plus d'exemples
- 🔧 Voir [Commandes CLI](cli-commands.md) pour toutes les options
- 🎨 Voir [Augmentation RGB](rgb-augmentation.md) pour l'enrichissement couleur
- 📋 Voir [Configuration Pipeline](pipeline-configuration.md) pour les workflows YAML

## Ressources

- [Documentation CuPy](https://docs.cupy.dev/)
- [Documentation RAPIDS](https://docs.rapids.ai/)
- [Guide d'Installation CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
