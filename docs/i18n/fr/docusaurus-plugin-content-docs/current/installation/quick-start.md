---
sidebar_position: 1
title: Guide d'Installation
description: Guide d'installation complet pour la bibliothèque de traitement IGN LiDAR HD
keywords: [installation, pip, setup, gpu, cuda]
---

# Guide d'Installation

Guide d'installation complet pour la bibliothèque de traitement IGN LiDAR HD. Devenez opérationnel en quelques minutes avec nos instructions étape par étape.

## 📋 Prérequis

- **Python 3.8+** (Python 3.9-3.11 recommandé)
- Gestionnaire de packages **pip**
- **Système d'exploitation :** Windows, Linux ou macOS

:::tip Vérifier la Version de Python

```bash
python --version  # Devrait afficher Python 3.8 ou supérieur
```

:::

## 🚀 Installation Standard

### Via PyPI (Recommandé)

```bash
pip install ign-lidar-hd
```

### Vérifier l'Installation

```bash
# Vérifier la version
ign-lidar-hd --version

# Tester la CLI
ign-lidar-hd --help
```

### Options d'Installation

```bash
# Installation standard (CPU uniquement)
pip install ign-lidar-hd

# Avec support pour l'augmentation RGB
pip install ign-lidar-hd[rgb]

# Avec toutes les fonctionnalités (sauf GPU)
pip install ign-lidar-hd[all]
```

## ⚡ Accélération GPU (Optionnel)

**Boost de Performance :** Calcul de caractéristiques 5 à 10x plus rapide

### Prérequis

1. **GPU NVIDIA** avec support CUDA
2. **CUDA Toolkit 11.0+** installé
3. **Mémoire GPU :** 4 Go+ recommandé

Vérifier la configuration GPU :

```bash
nvidia-smi  # Devrait afficher les informations du GPU
```

### Installer le Support GPU

```bash
# Installer d'abord le package de base
pip install ign-lidar-hd

# Puis ajouter CuPy pour votre version CUDA
pip install cupy-cuda11x  # Pour CUDA 11.x
# OU
pip install cupy-cuda12x  # Pour CUDA 12.x
```

### GPU Avancé (RAPIDS cuML)

Pour des performances maximales :

```bash
# Utiliser conda (recommandé pour RAPIDS)
conda create -n ign-lidar python=3.10
conda activate ign-lidar
pip install ign-lidar-hd
conda install -c rapidsai -c conda-forge -c nvidia cuml
```

## 🔧 Installation pour le Développement

### Depuis les Sources

```bash
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET
pip install -e .
```

### Avec les Dépendances de Développement

```bash
pip install -e .[dev,test,docs]
```

## 🐍 Environnements Virtuels

### Utiliser venv (Intégré)

```bash
python -m venv ign-lidar-env
source ign-lidar-env/bin/activate  # Linux/macOS
# ou
ign-lidar-env\Scripts\activate     # Windows
pip install ign-lidar-hd
```

### Utiliser conda

```bash
conda create -n ign-lidar python=3.10
conda activate ign-lidar
pip install ign-lidar-hd
```

## ✅ Vérifier l'Installation

### Vérification de Base

```python
# Tester les imports Python
import ign_lidar
print(f"Version IGN LiDAR HD : {ign_lidar.__version__}")

# Tester les classes principales
from ign_lidar import LiDARProcessor, IGNLiDARDownloader
print("✓ Installation réussie !")
```

### Vérification GPU

```python
# Vérifier la disponibilité du GPU
from ign_lidar.features_gpu import GPU_AVAILABLE, CUML_AVAILABLE

print(f"GPU (CuPy) disponible : {GPU_AVAILABLE}")
print(f"RAPIDS cuML disponible : {CUML_AVAILABLE}")

if GPU_AVAILABLE:
    print("✓ Accélération GPU activée !")
else:
    print("⚠️  GPU non détecté - utilisation du CPU")
```

## 🔧 Dépannage

### Commande Non Trouvée

```bash
# Si la commande ign-lidar-hd n'est pas trouvée, essayez :
python -m ign_lidar.cli --help
```

### Erreurs d'Import

```bash
# Réinstaller en mode développement
pip install -e .

# Vérifier le chemin Python
python -c "import sys; print('\n'.join(sys.path))"
```

### Problèmes GPU

```bash
# Tester la disponibilité de CUDA
python -c "import cupy; print('CUDA fonctionne !')"

# Vérifier la version CUDA
nvcc --version
```

## 🚀 Prochaines Étapes

Maintenant que vous êtes installé :

1. 📖 Suivez le [Guide de Démarrage Rapide](../guides/quick-start)
2. 🖥️ Essayez les [Exemples d'Utilisation de Base](../guides/basic-usage)
3. ⚡ Configurez l'[accélération GPU](../gpu/overview) (si disponible)
4. 📋 Explorez la [Configuration du Pipeline](../features/pipeline-configuration)

## 💡 Besoin d'Aide ?

- 📚 Lisez la [Documentation Complète](/)
- 🐛 Signalez les problèmes sur [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💬 Consultez les [Exemples](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples)
