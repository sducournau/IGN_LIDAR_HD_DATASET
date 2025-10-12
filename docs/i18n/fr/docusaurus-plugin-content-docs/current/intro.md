---
slug: /
sidebar_position: 1
title: Bibliothèque de Traitement LiDAR HD de l'IGN
---

# Bibliothèque de Traitement LiDAR HD de l'IGN

**Version 2.4.4** | Python 3.8+ | Licence MIT

[![PyPI version](https://badge.fury.io/py/ign-lidar-hd.svg)](https://badge.fury.io/py/ign-lidar-hd)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transformez les nuages de points LiDAR HD de l'IGN en jeux de données prêts pour l'apprentissage automatique pour la classification des bâtiments. Avec accélération GPU (6-20x plus rapide), caractéristiques géométriques riches (35-45+ caractéristiques exportées), augmentation RGB/NIR, outils de qualité des données LAZ, et configurations optimisées en mémoire pour toutes les spécifications système.

---

## � Nouveautés

### v2.4.4 (2025-10-12) - Dernière Version

### Outils de Qualité des Données LAZ et Validation

- 🛠️ **Outils Post-Traitement** : Nouveau script `fix_enriched_laz.py` pour la correction automatisée des fichiers LAZ
- 🔍 **Détection Qualité Données** : Identifie les erreurs de calcul NDVI, valeurs propres aberrantes, corruption des caractéristiques dérivées
- 📊 **Rapports Diagnostiques** : Analyse complète avec identification des causes racines et évaluation d'impact
- ✅ **Corrections Automatisées** : Limite les valeurs propres, recalcule les caractéristiques dérivées, valide les résultats
- � **Validation Améliorée** : Vérifications NIR améliorées et gestion des erreurs dans le pipeline d'enrichissement

### Corrections Clés

- 🐛 **Calcul NDVI** : Correction des valeurs = -1.0 quand les données NIR sont manquantes/corrompues
- 🔢 **Valeurs Propres Aberrantes** : Traite les valeurs extrêmes (>10,000) causant l'instabilité de l'entraînement ML
- � **Caractéristiques Dérivées** : Correction de la corruption en cascade dans change_curvature, omnivariance, etc.
- 🏷️ **Champs LAZ Dupliqués** : Correction des avertissements de champs dupliqués lors du traitement de fichiers LAZ pré-enrichis
- ⚡ **Prêt Production** : Validation robuste et gestion des erreurs pour les problèmes de qualité des données du monde réel

### Faits Marquants Récents (v2.3.x)

**Préservation des Données d'Entrée et Amélioration RGB :**

- 🎨 Préserve automatiquement RGB/NIR/NDVI des fichiers LAZ d'entrée
- 🐛 Correction du décalage critique des coordonnées RGB dans les patchs augmentés
- ⚡ Traitement RGB 3x plus rapide (récupération au niveau dalle)
- � Métadonnées de patch ajoutées pour le débogage et la validation

**Optimisation Mémoire :**

- 🧠 Support pour systèmes 8GB-32GB+ avec configurations optimisées
- 📊 Mise à l'échelle automatique des workers selon la pression mémoire
- ⚙️ Mode traitement séquentiel pour empreinte minimale
- Trois profils de configuration pour différentes spécifications système

**Modes de Traitement :**

- Modes clairs : `patches_only`, `both`, `enriched_only`
- Fichiers de configuration YAML avec modèles d'exemples
- Surcharges paramètres CLI avec `--config-file`

📖 [Historique Complet des Versions](CHANGELOG.md)

---

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

| Mode            | Vitesse                      | Prérequis               | Cas d'Usage                      |
| --------------- | ---------------------------- | ----------------------- | -------------------------------- |
| **CPU**         | Baseline (60 min/dalle)      | Python 3.8+             | Développement, petits jeux       |
| **GPU Hybride** | 6-8x plus rapide (7-10 min)  | GPU NVIDIA, CuPy        | Bon équilibre                    |
| **GPU Complet** | 12-20x plus rapide (3-5 min) | GPU NVIDIA, RAPIDS cuML | Production, gros jeux de données |

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
| 1M     | 10s  | &lt;1s     | 15-20x       |
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
