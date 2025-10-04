---
sidebar_position: 2
title: Guide de Démarrage Rapide
---

# Guide de Démarrage Rapide

Commencez avec la Bibliothèque de Traitement LiDAR HD de l'IGN en 5 minutes ! Ce guide vous accompagnera à travers l'installation, l'utilisation de base et votre premier workflow complet.

---

## 📦 Installation

### Installation Standard (CPU Uniquement)

```bash
pip install ign-lidar-hd
```

Cela installe la bibliothèque de base avec toutes les fonctionnalités essentielles pour le traitement basé sur CPU.

### Installation Complète (Toutes les Fonctionnalités)

```bash
pip install ign-lidar-hd[all]
```

Cela inclut :

- 🎨 Support d'augmentation RGB (Pillow, requests)
- 📋 Configuration de pipeline YAML
- 🛠️ Outils de développement

### Installation GPU (Optionnel)

Pour un traitement accéléré par CUDA (5-10x plus rapide) :

```bash
# Installer le package de base
pip install ign-lidar-hd

# Installer CuPy (selon votre version CUDA)
pip install cupy-cuda11x  # Pour CUDA 11.x
# OU
pip install cupy-cuda12x  # Pour CUDA 12.x
```

**Prérequis :**

- GPU NVIDIA avec support CUDA
- CUDA Toolkit 11.0+
- 4Go+ de mémoire GPU recommandée

:::tip Avantages du GPU
L'accélération GPU offre une accélération 5-10x pour :

- Calcul des caractéristiques (normales, courbure)
- Interpolation des couleurs RGB (24x plus rapide)
- Traitement de grandes dalles (>1M points)
  :::

---

## 🚀 Votre Premier Workflow

Traitons des données LiDAR en 3 étapes simples : Télécharger → Enrichir → Créer des Patches

### Étape 1 : Télécharger les Dalles LiDAR

Téléchargez des dalles depuis les serveurs IGN en utilisant des coordonnées géographiques :

```bash
ign-lidar-hd download \
  --bbox 2.3,48.8,2.4,48.9 \
  --output data/brut \
  --max-tiles 5
```

**Ce que cela fait :**

- Interroge le service WFS de l'IGN pour les dalles disponibles
- Télécharge jusqu'à 5 dalles dans la zone spécifiée (région parisienne)
- Sauvegarde les fichiers LAZ dans `data/brut/`
- Ignore les dalles déjà téléchargées

:::info Format de la Boîte Englobante
`--bbox lon_min,lat_min,lon_max,lat_max` (coordonnées WGS84)

Exemples de zones :

- Paris : `2.3,48.8,2.4,48.9`
- Marseille : `5.3,43.2,5.4,43.3`
- Lyon : `4.8,45.7,4.9,45.8`
  :::

### Étape 2 : Enrichir avec des Caractéristiques

Ajoutez des caractéristiques géométriques et des couleurs RGB optionnelles :

```bash
ign-lidar-hd enrich \
  --input-dir data/brut \
  --output data/enrichi \
  --mode building \
  --use-gpu
```

**Ce que cela fait :**

- Calcule les caractéristiques géométriques (normales, courbure, planarité)
- Ajoute des caractéristiques spécifiques aux bâtiments en mode 'building'
- Utilise l'accélération GPU si disponible (repli sur CPU)
- Ignore les dalles déjà enrichies

**Caractéristiques Ajoutées :**

- Normales de surface (vecteurs 3D)
- Courbure (courbure principale)
- Planarité, verticalité, horizontalité
- Densité locale de points
- Labels de classification de bâtiments

:::tip Ajouter des Couleurs RGB
Ajoutez `--add-rgb --rgb-cache-dir cache/` pour enrichir avec les couleurs des orthophotos IGN !
:::

### Étape 3 : Créer des Patches d'Entraînement

Générez des patches prêts pour l'apprentissage automatique :

```bash
ign-lidar-hd patch \
  --input-dir data/enrichi \
  --output data/patches \
  --lod-level LOD2 \
  --num-points 16384 \
  --augment \
  --num-augmentations 3
```

**Ce que cela fait :**

- Crée des patches de 150m × 150m à partir des dalles enrichies
- Échantillonne 16 384 points par patch
- Génère 3 versions augmentées par patch
- Sauvegarde en fichiers NPZ compressés

**Structure de Sortie :**

```text
data/patches/
├── dalle_0501_6320_patch_0.npz
├── dalle_0501_6320_patch_1.npz
├── dalle_0501_6320_patch_2.npz
└── ...
```

Chaque fichier NPZ contient :

- `points` : [N, 3] coordonnées XYZ
- `normals` : [N, 3] normales de surface
- `features` : [N, 27] caractéristiques géométriques
- `labels` : [N] labels de classes de bâtiments

---

## 🎯 Workflow Complet avec YAML

Pour les workflows de production, utilisez des fichiers de configuration YAML pour la reproductibilité :

### Créer une Configuration

```bash
ign-lidar-hd pipeline mon_workflow.yaml --create-example full
```

Cela crée `mon_workflow.yaml` :

```yaml
global:
  num_workers: 4

download:
  bbox: "2.3, 48.8, 2.4, 48.9"
  output: "data/brut"
  max_tiles: 10

enrich:
  input_dir: "data/brut"
  output: "data/enrichi"
  mode: "building"
  use_gpu: true
  add_rgb: true
  rgb_cache_dir: "cache/orthophotos"

patch:
  input_dir: "data/enrichi"
  output: "data/patches"
  lod_level: "LOD2"
  patch_size: 150.0
  num_points: 16384
  augment: true
  num_augmentations: 3
```

### Exécuter le Pipeline

```bash
ign-lidar-hd pipeline mon_workflow.yaml
```

**Avantages :**

- ✅ Workflows reproductibles
- ✅ Compatible avec le contrôle de version
- ✅ Collaboration d'équipe facile
- ✅ Exécuter seulement des étapes spécifiques
- ✅ Documentation de configuration claire

---

## 🐍 API Python

Pour un contrôle programmatique, utilisez l'API Python :

```python
from ign_lidar import LiDARProcessor

# Initialiser le processeur
processor = LiDARProcessor(
    lod_level="LOD2",
    augment=True,
    num_augmentations=3,
    use_gpu=True
)

# Traiter une seule dalle
patches = processor.process_tile(
    input_file="data/brut/dalle.laz",
    output_dir="data/patches"
)

print(f"Généré {len(patches)} patches d'entraînement")

# Ou traiter un répertoire entier
num_patches = processor.process_directory(
    input_dir="data/brut",
    output_dir="data/patches",
    num_workers=4
)

print(f"Total de patches générés : {num_patches}")
```

---

## 🎓 Comprendre les Niveaux LOD

Choisissez le bon Niveau de Détail pour votre tâche :

### LOD2 (15 Classes)

Modèles de bâtiments simplifiés - bon pour la classification générale :

**Classes :**

- Sol, végétation, route, voie ferrée
- Parties de bâtiments : mur, toit, balcon, fenêtre, porte
- Mobilier urbain, lignes électriques, etc.

**Cas d'Usage :**

- Détection et segmentation de bâtiments
- Planification urbaine
- Modélisation 3D de villes (basique)

```python
processor = LiDARProcessor(lod_level="LOD2")
```

### LOD3 (30+ Classes)

Modèles de bâtiments détaillés - pour l'analyse architecturale :

**Classes Supplémentaires :**

- Types de toits détaillés (plat, à pignon, à croupe, etc.)
- Éléments architecturaux (colonnes, corniches, ornements)
- Matériaux de construction
- Styles architecturaux précis

**Cas d'Usage :**

- Documentation du patrimoine architectural
- Reconstruction 3D détaillée
- Évaluation de l'état des bâtiments

```python
processor = LiDARProcessor(lod_level="LOD3")
```

---

## ⚡ Conseils de Performance

### 1. Utiliser l'Accélération GPU

```bash
# 5-10x plus rapide pour le calcul des caractéristiques
ign-lidar-hd enrich --use-gpu --input-dir dalles/ --output enrichi/
```

### 2. Traitement Parallèle

```bash
# Utiliser plusieurs cœurs CPU
ign-lidar-hd enrich --num-workers 8 --input-dir dalles/ --output enrichi/
```

### 3. Reprise Intelligente

Toutes les commandes ignorent automatiquement les fichiers existants :

```bash
# Sûr d'interrompre et de reprendre
ign-lidar-hd enrich --input-dir dalles/ --output enrichi/
# Appuyez sur Ctrl+C à tout moment
# Relancez - continue là où il s'est arrêté
```

### 4. Cache RGB

Lors de l'utilisation de l'augmentation RGB, mettez en cache les orthophotos pour réutilisation :

```bash
ign-lidar-hd enrich \
  --add-rgb \
  --rgb-cache-dir cache/orthophotos \
  --input-dir dalles/ \
  --output enrichi/
```

---

## 🔍 Vérifier Vos Données

### Vérifier les Fichiers Enrichis

```python
import laspy

# Charger le fichier LAZ enrichi
las = laspy.read("data/enrichi/dalle.laz")

# Vérifier les dimensions
print("Dimensions disponibles :", las.point_format.dimension_names)

# Devrait inclure :
# - X, Y, Z (coordonnées)
# - normal_x, normal_y, normal_z
# - curvature
# - planarity, verticality
# - intensity, return_number
# - RGB (si utilisation --add-rgb)
```

### Vérifier les Patches NPZ

```python
import numpy as np

# Charger un patch
data = np.load("data/patches/dalle_patch_0.npz")

# Vérifier le contenu
print("Clés :", list(data.keys()))
print("Forme des points :", data['points'].shape)
print("Forme des labels :", data['labels'].shape)

# Vérifier le nombre de points
assert data['points'].shape[0] == 16384  # num_points par défaut
```

---

## 🐛 Dépannage

### GPU Non Détecté

```bash
# Vérifier la disponibilité CUDA
python -c "import cupy as cp; print('CUDA disponible :', cp.is_available())"
```

Si CUDA n'est pas disponible :

- Assurez-vous que les pilotes GPU NVIDIA sont installés
- Installez la version correcte de CuPy pour votre toolkit CUDA
- La bibliothèque repli automatiquement sur CPU

### Manque de Mémoire

Pour les grandes dalles (>10M points) :

```python
# Réduire la taille des patches ou le nombre de points
processor = LiDARProcessor(
    patch_size=100.0,      # Patches plus petits (défaut : 150.0)
    num_points=8192,       # Moins de points (défaut : 16384)
)
```

### Traitement Lent

1. Activer l'accélération GPU : `--use-gpu`
2. Augmenter les workers : `--num-workers 8`
3. Utiliser le mode 'core' au lieu de 'building' : `--mode core`

---

## 📚 Prochaines Étapes

### En Savoir Plus

- 📖 [Guide des Fonctionnalités](features/overview.md) - Plongée profonde dans toutes les fonctionnalités
- ⚡ [Guide GPU](gpu/overview.md) - Détails de l'accélération GPU
- 🔧 [Guide de Configuration](features/pipeline-configuration.md) - Workflows avancés
- 🎨 [Augmentation RGB](features/rgb-augmentation.md) - Enrichissement couleur

### Exemples

- [Usage de Base](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/basic_usage.py)
- [Configuration Pipeline](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/pipeline_example.py)
- [Traitement GPU](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/processor_gpu_usage.py)
- [Augmentation RGB](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/enrich_with_rgb.py)

### Obtenir de l'Aide

- 🐛 [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues) - Signaler des bugs
- 💬 [Discussions](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions) - Poser des questions
- 📧 Email : simon.ducournau@gmail.com

---

**Prêt à traiter votre premier jeu de données ?** 🚀

```bash
# Télécharger et traiter en une seule commande
ign-lidar-hd download --bbox 2.3,48.8,2.4,48.9 --output brut/ --max-tiles 5
ign-lidar-hd enrich --input-dir brut/ --output enrichi/ --use-gpu
ign-lidar-hd patch --input-dir enrichi/ --output patches/ --augment
```

Bon traitement ! 🎉
