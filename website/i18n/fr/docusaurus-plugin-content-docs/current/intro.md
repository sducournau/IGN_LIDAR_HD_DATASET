---
slug: /
sidebar_position: 1
title: Bibliothèque de Traitement LiDAR HD de l'IGN
---

# Bibliothèque de Traitement LiDAR HD de l'IGN

**Version 1.7.5** | Python 3.8+ | Licence MIT

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📺 Vidéo de Démonstration

<div align="center">
  <a href="https://www.youtube.com/watch?v=ksBWEhkVqQI" target="_blank">
    <img src="https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/v1.6.3/website/static/img/aerial.png?raw=true" alt="Démonstration du Traitement IGN LiDAR HD" width="800" />
  </a>
  <p><em>Apprenez à traiter les données LiDAR pour les applications d'apprentissage automatique</em></p>
</div>

---

## 🎉 Dernière Version : v1.7.5

### 🚀 OPTIMISATION MASSIVE des Performances - Accélération 100-200x

La dernière version élimine un goulot d'étranglement critique grâce au **calcul vectorisé des caractéristiques** :

**Améliorations Clés :**

- ⚡ **Opérations Vectorisées** : Remplacement des boucles PCA par point par calcul de covariance par batch avec `einsum`
- 💯 **Utilisation GPU à 100%** : GPU pleinement utilisé (était bloqué à 0-5% avant)
- 🎯 **Tous les Modes Optimisés** : CPU, GPU sans cuML, et GPU avec cuML tous optimisés
- ⏱️ **Impact Réel** : 17M points en ~30 secondes (était bloqué à 0% pendant des heures !)
- 🔧 **Correction Stabilité GPU** : Correction des erreurs `CUSOLVER_STATUS_INVALID_VALUE` avec application de la symétrie matricielle et régularisation

:::tip Aucune Configuration Nécessaire

Vos commandes existantes bénéficient automatiquement de l'accélération 100-200x :

```bash
# Même commande, drastiquement plus rapide !
ign-lidar-hd enrich --input-dir data/ --output output/ \
  --auto-params --preprocess --use-gpu
```

:::

**Performance Vérifiée :**

- ✅ CPU : 90k-110k points/sec (test 50k points)
- ✅ GPU : Utilisation 100%, 40% VRAM
- ✅ Pipeline complet : 17M points en 3-4 minutes

📖 [Détails Optimisation](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/VECTORIZED_OPTIMIZATION.md) | [Guide GPU](/gpu/overview)

---

## Mises à Jour Précédentes

### v1.7.4 - Accélération GPU

- 🚀 **Support RAPIDS cuML** : Accélération 12-20x avec GPU complet
- ⚡ **Mode GPU Hybride** : Accélération 6-8x avec CuPy (cuML non requis)
- 🔧 **Trois Niveaux de Performance** : CPU (60 min), Hybride (7-10 min), GPU complet (3-5 min)
- 📚 **Documentation Améliorée** : Guides complets de configuration GPU en anglais et français

### v1.7.3 - Augmentation Infrarouge

- 🌿 **Valeurs NIR** : Proche infrarouge depuis orthophotos IRC IGN
- 📊 **Prêt pour NDVI** : Permet le calcul d'indices de végétation
- 🎨 **Multi-Modal** : Géométrie + RGB + NIR pour ML
- 💾 **Cache Intelligent** : Mise en cache efficace disque/GPU

### v1.7.1 - Analyse Auto-Paramètres

- 🤖 **Analyse Automatique de Dalle** : Détermine les paramètres de traitement optimaux
- 🎯 **Traitement Adaptatif** : Paramètres personnalisés par dalle selon caractéristiques
- ⚡ **Zéro Ajustement Manuel** : Élimine les conjectures pour dalles urbaines/rurales/mixtes

---

## Démarrage Rapide

Bienvenue dans la documentation de la **Bibliothèque de Traitement LiDAR HD IGN** !

Transformez les données LiDAR françaises en jeux de données prêts pour l'apprentissage automatique avec cette boîte à outils Python complète. 🏗️

:::tip Pourquoi utiliser cette bibliothèque ?

- **🎯 Spécialisée pour le LiDAR Français** : Optimisée pour le format LiDAR HD IGN
- **⚡ Prête pour la Production** : Testée en conditions réelles avec plus de 50 dalles
- **🚀 Accélérée par GPU** : Support CUDA optionnel pour un traitement 12-20x plus rapide
- **🌈 Extraction de Caractéristiques Riche** : Plus de 28 caractéristiques géométriques et colorimétriques
- **🌿 Multi-modal** : Support Géométrie + RGB + Infrarouge
- **📦 Prête pour Pipeline** : Configuration YAML, cache intelligent, reprise possible
- **🔧 Flexible** : Outils CLI + API Python

:::

### Installation Rapide

Installer la bibliothèque :

```bash
pip install ign-lidar-hd
```

Traiter votre première dalle :

```bash
ign-lidar-hd enrich \
  --input-dir data/raw_tiles \
  --output data/enriched \
  --auto-params \
  --preprocess \
  --add-rgb \
  --add-infrared
```

Avec accélération GPU :

```bash
# Installer le support GPU (configuration unique)
./install_cuml.sh

# Traiter avec GPU
ign-lidar-hd enrich \
  --input-dir data/raw_tiles \
  --output data/enriched \
  --auto-params \
  --preprocess \
  --use-gpu \
  --add-rgb \
  --add-infrared
```

📖 Continuez vers [Installation](/installation/quick-start) pour des instructions de configuration détaillées.

---

## Fonctionnalités

### Capacités Principales

- **🗺️ Intégration Données IGN** : Téléchargement direct depuis le service WFS IGN
- **🎨 Augmentation RGB** : Ajout de couleurs réelles depuis photos aériennes IGN
- **🌿 Augmentation Infrarouge** : Ajout NIR pour analyse végétation (prêt NDVI)
- **📊 Caractéristiques Riches** : Plus de 28 caractéristiques géométriques (normales, courbure, planarité, etc.)
- **🏠 Classification Bâtiments** : Classification LoD0/LoD1/LoD2/LoD3
- **🚀 Accélération GPU** : Accélération 12-20x avec RAPIDS cuML
- **🔧 Atténuation Artefacts** : Suppression valeurs aberrantes statistique + par rayon
- **🤖 Auto-Paramètres** : Analyse et optimisation automatique des dalles

### Modes de Traitement

| Mode             | Vitesse                          | Prérequis               | Cas d'Usage                      |
| ---------------- | -------------------------------- | ----------------------- | -------------------------------- |
| **CPU**          | Baseline (60 min/dalle)          | Python 3.8+             | Développement, petits jeux       |
| **GPU Hybride**  | 6-8x plus rapide (7-10 min)      | GPU NVIDIA, CuPy        | Bon équilibre                    |
| **GPU Complet**  | 12-20x plus rapide (3-5 min)     | GPU NVIDIA, RAPIDS cuML | Production, gros jeux de données |

### Formats de Sortie

- **LAZ 1.4** : Attributs étendus (28+ caractéristiques) - **Recommandé**
- **LAZ 1.2** : Compatible CloudCompare (RGB + caractéristiques de base)
- **Couches QGIS** : Couches stylisées séparées pour visualisation
- **Statistiques** : Métriques JSON pour suivi qualité

---

## Structure de la Documentation

📚 **Installation**

- [Démarrage Rapide](/installation/quick-start) - Opérationnel en 5 minutes
- [Configuration GPU](/installation/gpu-setup) - Configuration RAPIDS cuML

⚡ **Guides**

- [Accélération GPU](/guides/gpu-acceleration) - Optimisation des performances
- [Utilisation Basique](/guides/basic-usage) - Flux de travail courants
- [Utilisation Avancée](/guides/advanced-usage) - Fonctionnalités pour utilisateurs avancés

🎨 **Fonctionnalités**

- [Augmentation RGB](/features/rgb-augmentation) - Ajout de couleurs réelles
- [Augmentation Infrarouge](/features/infrared-augmentation) - NIR et NDVI
- [Auto Paramètres](/features/auto-params) - Optimisation automatique
- [Classification LoD3](/features/lod3-classification) - Détection de bâtiments

🔧 **Référence API**

- [Commandes CLI](/api/cli) - Interface en ligne de commande
- [API Python](/api/features) - Utilisation programmatique
- [Configuration](/api/configuration) - Pipelines YAML

---

## Performance

Avec l'optimisation vectorielle v1.7.5 :

| Points | CPU  | GPU (cuML) | Accélération |
| ------ | ---- | ---------- | ------------ |
| 1M     | 10s  | <1s        | 15-20x       |
| 5M     | 50s  | 3s         | 100-150x     |
| 17M    | 180s | 30s        | **100-200x** |

Exemple réel (dalle 17M points) :

- Prétraitement : ~2 minutes
- Caractéristiques : ~30 secondes (vectorisé !)
- Augmentation RGB : ~30 secondes
- Augmentation infrarouge : ~30 secondes
- **Total : 3-4 minutes** (était des heures avant l'optimisation !)

---

## Communauté

- 🐛 [Signaler des Problèmes](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 💡 [Demandes de Fonctionnalités](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- �� [Contribuer](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/CONTRIBUTING.md)

---

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/LICENSE) pour plus de détails.

---

## Prochaines Étapes

Prêt à vous lancer ? Commencez avec le [Guide de Démarrage Rapide](/installation/quick-start) pour installer la bibliothèque et traiter votre première dalle !

Pour l'accélération GPU (recommandée pour la production), consultez le [Guide de Configuration GPU](/installation/gpu-setup).
