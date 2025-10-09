---
sidebar_position: 2
title: Guide de Démarrage Rapide
description: Démarrez avec IGN LiDAR HD en 5 minutes
keywords: [demarrage-rapide, flux-de-travail, tutoriel, exemples]
---

# Guide de Démarrage Rapide

Démarrez avec la bibliothèque de traitement IGN LiDAR HD en 5 minutes ! Ce guide vous accompagne dans votre premier flux de travail complet, du téléchargement à l'analyse.

:::info Prérequis
Assurez-vous d'avoir installé IGN LiDAR HD. Sinon, consultez d'abord le [Guide d'Installation](../installation/quick-start).
:::

---

## 🚀 Votre Premier Flux de Travail

Traitons les données LiDAR en 3 étapes simples : Télécharger → Enrichir → Créer des Patchs

### Étape 1 : Télécharger les Dalles LiDAR

Téléchargez les dalles depuis les serveurs IGN en utilisant des coordonnées géographiques :

```bash
ign-lidar-hd download \
  --bbox 2.3,48.8,2.4,48.9 \
  --output data/raw \
  --max-tiles 5
```

**Ce que cela fait :**

- Interroge le service WFS de l'IGN pour les dalles disponibles
- Télécharge jusqu'à 5 dalles dans la boîte englobante spécifiée (région parisienne)
- Enregistre les fichiers LAZ dans `data/raw/`
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
  --input-dir data/raw \
  --output data/enriched \
  --mode full \
  --use-gpu
```

**Ce que cela fait :**

- Calcule les caractéristiques géométriques (normales, courbure, planarité)
- Ajoute toutes les caractéristiques supplémentaires en mode 'full'
- Utilise l'accélération GPU si disponible (repli sur CPU)
- **Pas d'augmentation par défaut** (utilisez --augment pour activer)
- Ignore les dalles déjà enrichies

:::info Augmentation de Données (Désactivée par Défaut)
Par défaut, la commande enrich crée **uniquement la dalle originale**. Pour activer l'augmentation, ajoutez `--augment` qui crée **4 versions** de chaque dalle :

- `tile_name.laz` (original)
- `tile_name_aug1.laz` (version augmentée 1)
- `tile_name_aug2.laz` (version augmentée 2)
- `tile_name_aug3.laz` (version augmentée 3)

Chaque version augmentée applique une rotation aléatoire, du bruit, une mise à l'échelle et un dropout avant le calcul des caractéristiques.

Pour activer : ajoutez `--augment`  
Pour changer le nombre : ajoutez `--num-augmentations N`
:::

**Caractéristiques Ajoutées :**

- Normales de surface (vecteurs 3D)
- Courbure (courbure principale)
- Planarité, verticalité, horizontalité
- Densité de points locale
- Étiquettes de classification des bâtiments

:::tip Ajouter des Couleurs RGB
Ajoutez `--add-rgb --rgb-cache-dir cache/` pour enrichir avec les couleurs des orthophotos IGN !
:::
### Étape 3 : Créer des Patchs d'Entraînement

Générez des patchs prêts pour l'apprentissage automatique :

```bash
# Note : L'augmentation se fait pendant la phase ENRICH (désactivée par défaut)
# Utilisez le flag --augment à l'étape enrich pour créer des versions augmentées
ign-lidar-hd patch \
  --input-dir data/enriched \
  --output data/patches \
  --lod-level LOD2 \
  --num-points 16384
```

**Ce que cela fait :**

- Crée des patchs de 150m × 150m à partir des dalles enrichies
- Échantillonne 16 384 points par patch
- Traite à la fois les dalles originales et augmentées (créées lors de l'enrichissement)
- Enregistre sous forme de fichiers NPZ compressés

**Structure de Sortie :**

```text
data/patches/
├── tile_0501_6320_patch_0.npz
├── tile_0501_6320_patch_1.npz
├── tile_0501_6320_patch_2.npz
└── ...
```

Chaque fichier NPZ contient :

- `points` : Coordonnées XYZ [N, 3]
- `normals` : Normales de surface [N, 3]
- `features` : Caractéristiques géométriques [N, 27]
- `labels` : Étiquettes de classe de bâtiment [N]

---

## 🎯 Flux de Travail Complet avec YAML

Pour les flux de travail en production, utilisez des fichiers de configuration YAML pour la reproductibilité :

### Créer une Configuration

```bash
ign-lidar-hd pipeline my_workflow.yaml --create-example full
```

Cela crée un fichier de configuration YAML. Pour des exemples de configuration détaillés, consultez [Exemples de Configuration](../reference/config-examples).

### Exemple Rapide

```yaml
input_dir: "data/enriched"
output: "data/patches"
lod_level: "LOD2"
patch_size: 150.0
num_points: 16384
augment: true
num_augmentations: 3
```

### Exécuter le Pipeline

```bash
ign-lidar-hd pipeline my_workflow.yaml
```

**Avantages :**

- ✅ Flux de travail reproductibles
- ✅ Compatible avec le contrôle de version
- ✅ Collaboration d'équipe facile
- ✅ Exécution de phases spécifiques uniquement
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
    input_file="data/raw/tile.laz",
    output_dir="data/patches"
)

print(f"Généré {len(patches)} patchs d'entraînement")

# Ou traiter un répertoire entier
num_patches = processor.process_directory(
    input_dir="data/raw",
    output_dir="data/patches",
    num_workers=4
)

print(f"Total de patchs générés : {num_patches}")
```

---

## 🎓 Comprendre les Niveaux LOD

Choisissez le bon niveau de détail pour votre tâche :

### LOD2 (15 Classes)

Modèles de bâtiments simplifiés - bon pour la classification générale :

**Classes :**

- Sol, végétation, route, voie ferrée
- Parties de bâtiment : mur, toit, balcon, fenêtre, porte
- Mobilier urbain, lignes électriques, etc.

**Cas d'Usage :**

- Détection et segmentation de bâtiments
- Urbanisme
- Modélisation 3D de ville (basique)

```python
processor = LiDARProcessor(lod_level="LOD2")
```

### LOD3 (30+ Classes)

Modèles de bâtiments détaillés - pour l'analyse architecturale :

**Classes Supplémentaires :**

- Types de toits détaillés (plat, à pignon, à quatre pans, etc.)
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
# Calcul de caractéristiques 5-10x plus rapide
ign-lidar-hd enrich --use-gpu --input-dir tiles/ --output enriched/
```

### 2. Traitement Parallèle

```bash
# Utiliser plusieurs cœurs CPU
ign-lidar-hd enrich --num-workers 8 --input-dir tiles/ --output enriched/
```

### 3. Reprise Intelligente

Toutes les commandes ignorent automatiquement les fichiers existants :

```bash
# Sûr d'interrompre et de reprendre
ign-lidar-hd enrich --input-dir tiles/ --output enriched/
# Appuyez sur Ctrl+C à tout moment
# Réexécutez - continue là où il s'est arrêté
```

### 4. Mise en Cache RGB

Lors de l'utilisation de l'augmentation RGB, mettez en cache les orthophotos pour réutilisation :

```bash
ign-lidar-hd enrich \
  --add-rgb \
  --rgb-cache-dir cache/orthophotos \
  --input-dir tiles/ \
  --output enriched/
```

---

## 🔍 Vérifier Vos Données

### Vérifier les Fichiers Enrichis

```python
import laspy

# Charger le fichier LAZ enrichi
las = laspy.read("data/enriched/tile.laz")

# Vérifier les dimensions
print("Dimensions disponibles :", las.point_format.dimension_names)

# Devrait inclure :
# - X, Y, Z (coordonnées)
# - normal_x, normal_y, normal_z
# - curvature (courbure)
# - planarity, verticality (planarité, verticalité)
# - intensity, return_number
# - RGB (si vous utilisez --add-rgb)
```

### Vérifier les Patchs NPZ

```python
import numpy as np

# Charger un patch
data = np.load("data/patches/tile_patch_0.npz")

# Vérifier le contenu
print("Clés :", list(data.keys()))
print("Forme des points :", data['points'].shape)
print("Forme des étiquettes :", data['labels'].shape)

# Vérifier le nombre de points
assert data['points'].shape[0] == 16384  # num_points par défaut
```

---

## 🐛 Dépannage

### GPU Non Détecté

```bash
# Vérifier la disponibilité de CUDA
python -c "import cupy as cp; print('CUDA disponible :', cp.is_available())"
```

Si CUDA n'est pas disponible :

- Assurez-vous que les pilotes GPU NVIDIA sont installés
- Installez la bonne version de CuPy pour votre boîte à outils CUDA
- La bibliothèque bascule automatiquement sur le CPU

### Mémoire Insuffisante

Pour les dalles volumineuses (>10M points) :

```python
# Réduire la taille des patchs ou le nombre de points
processor = LiDARProcessor(
    patch_size=100.0,      # Patchs plus petits (par défaut : 150.0)
    num_points=8192,       # Moins de points (par défaut : 16384)
)
```

### Traitement Lent

1. Activer l'accélération GPU : `--use-gpu`
2. Augmenter les workers : `--num-workers 8`
3. Utiliser le mode 'core' au lieu de 'full' : `--mode core`

---

## 📚 Prochaines Étapes

### En Savoir Plus

- 📖 [Guide des Caractéristiques](features/overview.md) - Plongée dans toutes les fonctionnalités
- ⚡ [Guide GPU](gpu/overview.md) - Détails sur l'accélération GPU
- 🔧 [Guide de Configuration](features/pipeline-configuration.md) - Flux de travail avancés
- 🎨 [Augmentation RGB](features/rgb-augmentation.md) - Enrichissement couleur

### Exemples

- [Utilisation de Base](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/basic_usage.py)
- [Configuration de Pipeline](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/pipeline_example.py)
- [Traitement GPU](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/processor_gpu_usage.py)
- [Augmentation RGB](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/examples/enrich_with_rgb.py)

### Obtenir de l'Aide

- 🐛 [GitHub Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues) - Signaler des bugs
- 💬 [Discussions](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions) - Poser des questions
- 📧 Email : simon.ducournau@gmail.com

---

**Prêt à traiter votre premier jeu de données ?** 🚀

```bash
# Télécharger et traiter en une seule fois
ign-lidar-hd download --bbox 2.3,48.8,2.4,48.9 --output raw/ --max-tiles 5
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --use-gpu
ign-lidar-hd patch --input-dir enriched/ --output patches/ --augment
```

Bon traitement ! 🎉
