---
sidebar_position: 1
title: Installation Rapide
description: Installer IGN LiDAR HD en quelques minutes
keywords: [installation, pip, setup, démarrage rapide]
---

Installez IGN LiDAR HD en quelques commandes et commencez à traiter des données LiDAR immédiatement.

## Prérequis

- **Python 3.8+** (Python 3.9-3.11 recommandé)
- **pip** installé et à jour
- **Windows, Linux ou macOS**

:::tip Vérifier votre version Python

```bash
python --version  # Devrait afficher Python 3.8 ou supérieur
```

:::

## Installation Standard

### Via pip (Recommandé)

```bash
# Installation depuis PyPI
pip install ign-lidar-hd
```

C'est tout ! La bibliothèque est maintenant installée avec toutes les dépendances requises.

### Depuis les Sources

```bash
# Cloner le dépôt
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET

# Installer en mode développement
pip install -e .
```

## Installation GPU (Optionnel)

Pour un traitement jusqu'à 10x plus rapide avec accélération GPU :

### Prérequis GPU

1. **GPU NVIDIA** avec support CUDA
2. **Pilotes NVIDIA** installés
3. **CUDA Toolkit 11.0+** installé

Vérifiez votre configuration GPU :

```bash
nvidia-smi  # Devrait afficher les infos de votre GPU
```

### Installation CuPy (Support GPU de Base)

```bash
# Support GPU basique avec CuPy (accélération 5-6x)
pip install ign-lidar-hd[gpu]
```

OU installez CuPy manuellement selon votre version CUDA :

```bash
# Pour CUDA 11.x
pip install cupy-cuda11x

# Pour CUDA 12.x
pip install cupy-cuda12x
```

### Installation RAPIDS (Performances GPU Avancées)

Pour les meilleures performances (accélération jusqu'à 10x) :

```bash
# Installation complète avec RAPIDS cuML
pip install ign-lidar-hd[gpu-full]
```

OU via conda (plus fiable pour RAPIDS) :

```bash
# Créer un environnement conda
conda create -n lidar python=3.10
conda activate lidar

# Installer RAPIDS
conda install -c rapidsai -c conda-forge -c nvidia cuml

# Installer IGN LiDAR HD
pip install ign-lidar-hd[gpu]
```

## Vérifier l'Installation

### Installation de Base

```python
# Vérifier que la bibliothèque est installée
import ign_lidar
print(f"Version IGN LiDAR HD: {ign_lidar.__version__}")

# Importer les classes principales
from ign_lidar import LiDARProcessor, IGNLiDARDownloader
print("✓ Installation réussie!")
```

### Installation GPU

```python
# Vérifier la disponibilité GPU
from ign_lidar.features_gpu import GPU_AVAILABLE, CUML_AVAILABLE

print(f"GPU (CuPy) disponible: {GPU_AVAILABLE}")
print(f"RAPIDS cuML disponible: {CUML_AVAILABLE}")

if GPU_AVAILABLE:
    print("✓ Accélération GPU activée!")
else:
    print("⚠️  GPU non détecté - utilisation du CPU")
```

### Interface en Ligne de Commande

```bash
# Vérifier que la CLI est disponible
ign-lidar-hd --version

# Afficher l'aide
ign-lidar-hd --help
```

## Environnements Virtuels

### Utiliser venv (Recommandé)

```bash
# Créer un environnement virtuel
python -m venv lidar_env

# Activer l'environnement
# Sur Linux/macOS:
source lidar_env/bin/activate
# Sur Windows:
lidar_env\Scripts\activate

# Installer IGN LiDAR HD
pip install ign-lidar-hd
```

### Utiliser conda

```bash
# Créer un environnement conda
conda create -n lidar python=3.10
conda activate lidar

# Installer IGN LiDAR HD
pip install ign-lidar-hd

# Pour le support GPU avec RAPIDS
conda install -c rapidsai -c conda-forge -c nvidia cuml
```

## Prochaines Étapes

Maintenant que vous avez installé la bibliothèque :

1. 📖 Suivez le [Guide d'Utilisation de Base](../guides/basic-usage.md)
2. 🚀 Essayez les [Commandes CLI](../guides/cli-commands.md)
3. ⚡ Configurez l'[Accélération GPU](../guides/gpu-acceleration.md) (optionnel)
4. 📋 Explorez la [Configuration Pipeline](../features/pipeline-configuration.md)

## Dépannage

### Commande Non Trouvée

Si la commande `ign-lidar-hd` n'est pas trouvée :

```bash
# Vérifier l'installation
pip list | grep ign-lidar

# Vérifier le PATH
which ign-lidar-hd
```

### Erreurs d'Import

Si vous rencontrez des erreurs d'import, vérifiez que vous avez activé le bon environnement virtuel :

```bash
which python  # Devrait pointer vers votre environnement virtuel
```

## Besoin d'Aide ?

- 📚 Lisez la [Documentation Complète](/)
- 🐛 Signalez des problèmes sur [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💬 Consultez les [Exemples](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples)

Si vous obtenez des erreurs d'import :

```bash
# Réinstaller en mode développement
pip install -e .

# Ou vérifier votre chemin Python
python -c "import sys; print('\n'.join(sys.path))"
```

### Dépendances manquantes

Installer tous les paquets requis :

```bash
pip install -r requirements.txt
pip list  # Vérifier l'installation
```
