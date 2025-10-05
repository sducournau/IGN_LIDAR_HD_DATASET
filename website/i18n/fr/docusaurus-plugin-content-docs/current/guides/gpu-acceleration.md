---
sidebar_position: 4
title: Accélération GPU
description: Utiliser l'accélération GPU pour un traitement LiDAR plus rapide
keywords:
  [gpu, cuda, accélération, performance, optimisation, cuml, rapids, cupy]
---

# Accélération GPU

L'accélération GPU accélère considérablement les workflows de traitement LiDAR, offrant une **accélération de 6-20x** pour les jeux de données à grande échelle et les tâches complexes d'extraction de caractéristiques.

## Vue d'Ensemble

Le processeur IGN LiDAR HD supporte l'accélération GPU avec trois modes de performance :

1. **CPU Uniquement**: Traitement standard (pas de GPU requis)
2. **Mode Hybride (CuPy)**: Tableaux GPU + algorithmes CPU (accélération 6-8x)
3. **Mode GPU Complet (RAPIDS cuML)**: Pipeline GPU complet (accélération 12-20x)

Le mode hybride utilise une **stratégie KDTree per-chunk** intelligente qui évite les goulets d'étranglement de construction d'arbre global, offrant d'excellentes performances même sans RAPIDS cuML.

### Opérations Supportées

- **Extraction de Caractéristiques Géométriques**: Normales de surface, courbure, planarité, verticalité
- **Recherche KNN**: K plus proches voisins accéléré par GPU (avec RAPIDS cuML)
- **Calcul PCA**: Analyse en composantes principales basée sur GPU (avec RAPIDS cuML)
- **Filtrage de Nuages de Points**: Prétraitement parallèle et réduction du bruit
- **Augmentation RGB/NIR**: Intégration d'orthophotos optimisée par GPU

## 🚀 Benchmarks de Performance

### Résultats Réels (17M points, NVIDIA RTX 4080 16GB)

**Performance v1.7.5 (Optimisée)** :

| Mode                      | Temps de Traitement | Accélération | Prérequis                |
| ------------------------- | ------------------- | ------------ | ------------------------ |
| CPU Uniquement            | 60 min → 12 min     | 5x           | Aucun (optimisé!)        |
| Hybride (CuPy + sklearn)  | 7-10 min → 2 min    | 25-30x       | CuPy + CUDA 12.0+        |
| GPU Complet (RAPIDS cuML) | 3-5 min → 1-2 min   | 30-60x       | RAPIDS cuML + CUDA 12.0+ |

:::tip Optimisation v1.7.5
La version v1.7.5 inclut des optimisations majeures de performance qui bénéficient à **tous les modes** (CPU, Hybride, GPU Complet). La stratégie KDTree per-chunk et des chunks plus petits offrent une accélération 5-10x automatiquement !
:::

### Détail des Opérations

| Opération                      | CPU     | GPU Hybride | GPU Complet | Meilleure Accélération |
| ------------------------------ | ------- | ----------- | ----------- | ---------------------- |
| Extraction de Caractéristiques | 45 min  | 8 min       | 3 min       | 15x                    |
| Recherche KNN                  | 30 min  | 15 min      | 2 min       | 15x                    |
| Calcul PCA                     | 10 min  | 8 min       | 1 min       | 10x                    |
| Traitement par Lots            | 120 min | 20 min      | 8 min       | 15x                    |

## 🔧 Prérequis d'Installation

### Configuration Matérielle Requise

- **GPU**: GPU NVIDIA avec Compute Capability 6.0+ (Pascal ou plus récent)
- **Mémoire**: Minimum 4GB VRAM (8GB+ recommandé, 16GB pour grandes tuiles)
- **Pilote**: Pilote NVIDIA compatible CUDA 12.0+
- **Système**: 32GB+ RAM recommandé pour le traitement de grandes tuiles

### Matériel Recommandé

- **Budget**: NVIDIA RTX 3060 12GB
- **Optimal**: NVIDIA RTX 4070/4080 16GB
- **Professionnel**: NVIDIA A6000 48GB

## 📦 Options d'Installation

### Option 1: Mode Hybride (CuPy Uniquement) - Démarrage Rapide

**Idéal pour**: Configuration rapide, tests, ou lorsque RAPIDS cuML n'est pas disponible

```bash
# Installer CuPy pour votre version CUDA
pip install cupy-cuda12x  # Pour CUDA 12.x
# OU
pip install cupy-cuda11x  # Pour CUDA 11.x

# Vérifier la disponibilité GPU
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount(), 'GPU(s) détecté(s)')"
```

**Performance**: Accélération 6-8x (utilise des tableaux GPU avec algorithmes CPU sklearn via optimisation per-chunk)

### Option 2: Mode GPU Complet (RAPIDS cuML) - Performance Maximale

**Idéal pour**: Charges de travail en production, traitement à grande échelle, vitesse maximale

```bash
# Créer un environnement conda (requis pour RAPIDS)
conda create -n ign_gpu python=3.12 -y
conda activate ign_gpu

# Installer RAPIDS cuML (inclut CuPy)
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.10 cupy cuda-version=12.5 -y

# Installer IGN LiDAR HD
pip install ign-lidar-hd

# Vérifier l'installation
python scripts/verify_gpu_setup.py
```

**Performance**: Accélération 12-20x (pipeline GPU complet)

### Option 3: Script d'Installation Automatisé

Pour les systèmes WSL2/Linux, utilisez notre script d'installation automatisé :

```bash
# Télécharger et exécuter le script d'installation
wget https://raw.githubusercontent.com/sducournau/IGN_LIDAR_HD_DATASET/main/install_cuml.sh
chmod +x install_cuml.sh
./install_cuml.sh
```

Le script va :

- Installer Miniconda (si nécessaire)
- Créer l'environnement conda `ign_gpu`
- Installer RAPIDS cuML + toutes les dépendances
- Configurer les chemins CUDA

### Vérification de l'Installation

```bash
# Vérifier la détection GPU
ign-lidar-hd --version

# Tester le traitement GPU (utiliser une petite tuile)
ign-lidar-hd enrich --input test.laz --output test_enriched.laz --use-gpu
```

## 📖 Guide d'Utilisation

### Interface en Ligne de Commande

Le moyen le plus simple d'utiliser l'accélération GPU est via la CLI :

```bash
# Traitement GPU basique
ign-lidar-hd enrich --input-dir data/ --output enriched/ --use-gpu

# Traitement GPU complet avec toutes les options
ign-lidar-hd enrich \
  --input-dir data/ \
  --output enriched/ \
  --use-gpu \
  --auto-params \
  --preprocess \
  --add-rgb \
  --add-infrared \
  --rgb-cache-dir cache/rgb \
  --infrared-cache-dir cache/infrared

# Traiter des tuiles spécifiques
ign-lidar-hd enrich \
  --input tile1.laz tile2.laz \
  --output enriched/ \
  --use-gpu \
  --force  # Retraiter même si les sorties existent
```

### API Python

```python
from ign_lidar import LiDARProcessor

# Initialiser avec support GPU
processor = LiDARProcessor(
    lod_level="LOD2",
    use_gpu=True,
    num_workers=4
)

# Traiter une seule tuile
patches = processor.process_tile(
    "data/tile.laz",
    "output/",
    enable_rgb=True
)

# Traiter un répertoire avec GPU
patches = processor.process_directory(
    "data/",
    "output/",
    num_workers=4
)
```

### Configuration Pipeline (YAML)

```yaml
global:
  num_workers: 4

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  use_gpu: true
  auto_params: true
  preprocess: true
  add_rgb: true
  add_infrared: true
  rgb_cache_dir: "cache/rgb"
  infrared_cache_dir: "cache/infrared"

patch:
  input_dir: "data/enriched"
  output: "data/patches"
  lod_level: "LOD2"
```

Puis exécuter : `ign-lidar-hd pipeline config.yaml`

## 🐛 Dépannage

### Problèmes Courants

#### GPU Non Détecté

**Symptômes**: Message "GPU non disponible, bascule vers CPU"

**Solutions**:

```bash
# 1. Vérifier si le GPU est visible
nvidia-smi

# 2. Vérifier l'installation CUDA
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

# 3. Vérifier la compatibilité de version CUDA
python -c "import cupy; print('Version CUDA CuPy:', cupy.cuda.runtime.runtimeGetVersion())"

# 4. Vérifier LD_LIBRARY_PATH (Linux/WSL2)
echo $LD_LIBRARY_PATH  # Devrait inclure /usr/local/cuda-XX.X/lib64
```

#### Problèmes d'Installation CuPy

**Problème**: CuPy ne trouve pas les bibliothèques CUDA

**Solution WSL2**:

```bash
# Installer CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-13-0

# Ajouter à ~/.zshrc ou ~/.bashrc
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# Recharger et tester
source ~/.zshrc
python -c "import cupy; print('CuPy fonctionne!')"
```

#### Problèmes d'Installation RAPIDS cuML

**Problème**: Erreurs TOS conda lors de l'installation

**Solution**:

```bash
# Accepter les Conditions d'Utilisation conda
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Réessayer l'installation
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.10 -y
```

#### Mémoire CUDA Insuffisante

**Symptômes**: RuntimeError: CUDA out of memory

**Solutions**:

1. **Traiter des tuiles plus petites**: Diviser les gros fichiers en morceaux plus petits
2. **Réduire la taille des chunks**: Le processeur découpe automatiquement les grands nuages de points
3. **Fermer d'autres applications GPU**: Libérer la VRAM
4. **Utiliser un GPU avec plus de mémoire**: 16GB+ recommandé pour les grandes tuiles

```bash
# Surveiller l'utilisation de la mémoire GPU
watch -n 1 nvidia-smi
```

#### Performances Lentes Malgré le GPU

**Causes possibles**:

1. **Utilisation du Mode Hybride au lieu du GPU Complet**: Installer RAPIDS cuML pour une vitesse maximale
2. **Limitation thermique**: Vérifier la température GPU avec `nvidia-smi`
3. **Bande passante PCIe**: S'assurer que le GPU est dans un slot x16
4. **Goulot d'étranglement CPU**: Utiliser `--num-workers` pour paralléliser les E/S

**Vérifier l'utilisation GPU**:

```bash
# Surveiller l'utilisation GPU pendant le traitement
nvidia-smi dmon -s u
```

#### Per-Chunk vs KDTree Global

Le système sélectionne automatiquement la meilleure stratégie:

- **Avec RAPIDS cuML**: Utilise KDTree global sur GPU (le plus rapide, accélération 12-20x)
- **Sans cuML**: Utilise KDTree per-chunk avec CPU sklearn (toujours rapide, accélération 6-8x)

Vous verrez différents messages de log:

```text
# Avec cuML (le plus rapide)
✓ RAPIDS cuML disponible - algorithmes GPU activés
Calcul des normales avec KDTree accéléré GPU (global)

# Sans cuML (toujours rapide)
⚠ RAPIDS cuML non disponible - utilise KDTree per-chunk CPU
Calcul des normales avec KDTree per-chunk (chevauchement 5%)
```

### Basculement Automatique vers CPU

Le système bascule automatiquement vers le traitement CPU si le GPU n'est pas disponible:

- Échec d'import CuPy → mode CPU
- Erreur d'exécution CUDA → mode CPU
- Mémoire GPU insuffisante → mode CPU (avec avertissement)

**Désactiver le GPU** (forcer CPU):

```bash
ign-lidar-hd enrich --input-dir data/ --output enriched/  # Pas de flag --use-gpu
ign-lidar-hd enrich \
  --input tiles/ \
  --output enriched/ \
  --use-gpu \
```

## 📋 Benchmarks Détaillés

### Environnement de Test

- **GPU**: NVIDIA RTX 4080 (16GB VRAM)
- **CPU**: AMD Ryzen 9 / Intel i7 équivalent
- **Système**: WSL2 Ubuntu 24.04, 32GB RAM
- **CUDA**: 13.0
- **Tuile Test**: 17M points (tuile IGN LiDAR HD typique)

### Comparaison des Temps de Traitement

| Configuration             | Temps de Traitement | Accélération | Notes                                 |
| ------------------------- | ------------------- | ------------ | ------------------------------------- |
| CPU Uniquement (sklearn)  | 60 min              | 1x           | Ligne de base                         |
| Hybride (CuPy + sklearn)  | 7-10 min            | 6-8x         | Optimisation KDTree per-chunk         |
| GPU Complet (RAPIDS cuML) | 3-5 min             | 12-20x       | KDTree GPU global + PCA/KNN accélérés |

### Détail de l'Extraction de Caractéristiques

| Opération               | CPU    | GPU Hybride | GPU Complet | Meilleure Accélération |
| ----------------------- | ------ | ----------- | ----------- | ---------------------- |
| Calcul des Normales     | 25 min | 4 min       | 1.5 min     | 16x                    |
| Recherche KNN           | 20 min | 12 min      | 1 min       | 20x                    |
| PCA (valeurs propres)   | 8 min  | 6 min       | 0.5 min     | 16x                    |
| Calcul de Courbure      | 5 min  | 2 min       | 0.5 min     | 10x                    |
| Autres Caractéristiques | 2 min  | 1 min       | 0.5 min     | 4x                     |

### Utilisation Mémoire

| Mode                      | Mémoire GPU | RAM Système | Total |
| ------------------------- | ----------- | ----------- | ----- |
| CPU Uniquement            | 0 GB        | 24 GB       | 24 GB |
| Hybride (CuPy + sklearn)  | 6 GB        | 16 GB       | 22 GB |
| GPU Complet (RAPIDS cuML) | 8 GB        | 12 GB       | 20 GB |

### Traitement par Lots (100 tuiles)

- **CPU Uniquement**: ~100 heures
- **Mode Hybride**: ~12-15 heures (accélération 6-8x)
- **Mode GPU Complet**: ~5-8 heures (accélération 12-20x)

### Validation de la Précision

Les trois modes produisent des **résultats identiques** (vérifié avec corrélation de caractéristiques > 0.9999).

## 🔗 Documentation Connexe

- [Guide de Démarrage Rapide](./quick-start.md)
- [Optimisation des Performances](./performance.md)
- [Dépannage](./troubleshooting.md)
- [Configuration Pipeline](../api/pipeline-config.md)
- [Guide d'Installation](../installation/quick-start.md)

## 💡 Meilleures Pratiques

### 1. Choisir le Bon Mode

- **Développement/Tests**: Mode hybride (configuration facile, bonnes performances)
- **Production**: Mode GPU complet avec RAPIDS cuML (performance maximale)
- **Pas de GPU**: Mode CPU fonctionne bien pour les petits lots

### 2. Optimiser Votre Workflow

```yaml
# Configuration pipeline recommandée pour GPU
global:
  num_workers: 4 # Paralléliser les E/S pendant que le GPU traite

enrich:
  use_gpu: true
  auto_params: true # Laisser le système optimiser les paramètres
  preprocess: true # Nettoyer les données avant l'extraction de caractéristiques
```

### 3. Surveiller les Ressources

```bash
# Surveiller l'utilisation GPU en temps réel
watch -n 1 nvidia-smi

# Surveiller avec des métriques détaillées
nvidia-smi dmon -s pucvmet -d 1
```

### 4. Conseils pour le Traitement par Lots

- **Utiliser --force avec précaution**: Ne retraiter que lorsque nécessaire
- **Activer le cache intelligent**: Utiliser `--rgb-cache-dir` et `--infrared-cache-dir`
- **Paralléliser les E/S**: Utiliser `--num-workers` pour les opérations fichiers concurrentes
- **Traiter stratégiquement**: Commencer par les tuiles urbaines (densité de points élevée) pour tester les paramètres

### 5. Recommandations Matérielles

| Cas d'Usage                       | GPU Minimum   | GPU Recommandé | GPU Optimal      |
| --------------------------------- | ------------- | -------------- | ---------------- |
| Apprentissage/Petits jeux données | GTX 1660 6GB  | RTX 3060 12GB  | RTX 4060 Ti 16GB |
| Production/Lots moyens            | RTX 3060 12GB | RTX 4070 12GB  | RTX 4080 16GB    |
| Traitement grande échelle         | RTX 3080 10GB | RTX 4080 16GB  | A6000 48GB       |

## 🎓 Sujets Avancés

### Stratégie d'Optimisation Per-Chunk

Lorsque RAPIDS cuML n'est pas disponible, le système utilise une stratégie intelligente per-chunk:

1. **Divise le nuage de points** en chunks de ~5M points
2. **Construit un KDTree local** par chunk (rapide avec sklearn)
3. **Utilise un chevauchement de 5%** entre chunks pour gérer les cas limites
4. **Fusionne les résultats** de manière transparente

**Performances**: Cette stratégie évite le goulet d'étranglement de la PCA CPU séquentielle (qui prendrait ~85 minutes), réduisant le temps de traitement à 7-10 minutes. Cela fournit 6-8x d'accélération sans nécessiter l'installation de RAPIDS cuML.

### Gestion de la Mémoire GPU

Le système gère automatiquement la mémoire GPU:

- **Découpage automatique**: Les grands nuages de points sont divisés en chunks de taille GPU
- **Pooling mémoire**: CuPy réutilise la mémoire allouée
- **Garbage collection**: Libère la mémoire entre les tuiles
- **Gestion des erreurs**: Gère gracieusement les erreurs OOM

### Support Multi-GPU

Actuellement, la bibliothèque utilise un seul GPU (device 0). Pour le traitement multi-GPU:

```bash
# Traiter différents répertoires sur différents GPUs
CUDA_VISIBLE_DEVICES=0 ign-lidar-hd enrich --input dir1/ --output out1/ --use-gpu &
CUDA_VISIBLE_DEVICES=1 ign-lidar-hd enrich --input dir2/ --output out2/ --use-gpu &
```

---

_Pour des techniques d'optimisation GPU plus avancées, consultez le [Guide de Performance](./performance.md)._

````

### 3. Surveiller l'Utilisation GPU

```bash
# Dans un terminal séparé
watch -n 1 nvidia-smi

# Ou utiliser
gpustat -i 1
````

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
