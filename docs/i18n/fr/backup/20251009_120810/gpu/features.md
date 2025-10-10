---
sidebar_position: 2
title: "Calcul de Caractéristiques GPU"
description: "Détails techniques de l'extraction de caractéristiques accélérée par GPU"
keywords: [gpu, features, performance, cupy, benchmarks, api]
---

# Calcul de Caractéristiques GPU

**Disponible dans :** v1.3.0+  
**Accélération :** 5 à 10x plus rapide que le CPU  
**Corrigé dans v1.6.2 :** Les formules GPU correspondent maintenant au CPU (voir les changements incompatibles ci-dessous)

:::warning Changement Incompatible dans v1.6.2
Les formules de caractéristiques GPU ont été corrigées pour correspondre au CPU et à la littérature standard (Weinmann et al., 2015). Si vous avez utilisé l'accélération GPU dans v1.6.1 ou antérieure, les valeurs de caractéristiques ont changé. Vous devrez réentraîner les modèles ou passer au CPU pour la compatibilité avec les anciens modèles.
:::

Ce guide couvre les détails techniques du calcul de caractéristiques accéléré par GPU, incluant quelles caractéristiques sont accélérées, la référence API et les techniques d'optimisation avancées.

## Caractéristiques Accélérées

Les caractéristiques suivantes sont calculées sur GPU lorsque l'accélération GPU est activée :

### Caractéristiques Géométriques de Base

- ✅ **Normales de surface** (nx, ny, nz) - Vecteurs normaux pour chaque point
- ✅ **Valeurs de courbure** - Courbure de surface à chaque point
- ✅ **Hauteur au-dessus du sol** - Valeurs de hauteur normalisées

### Caractéristiques Géométriques Avancées

- ✅ **Planarité** - Mesure du degré de planéité d'une surface (utile pour toits, routes)
- ✅ **Linéarité** - Mesure des structures linéaires (utile pour bords, câbles)
- ✅ **Sphéricité** - Mesure des structures sphériques (utile pour végétation)
- ✅ **Anisotropie** - Mesure de structure directionnelle
- ✅ **Rugosité** - Texture et irrégularité de surface
- ✅ **Densité locale** - Densité de points dans le voisinage local

### Caractéristiques Spécifiques aux Bâtiments

- ✅ **Verticalité** - Mesure d'alignement vertical (murs)
- ✅ **Horizontalité** - Mesure d'alignement horizontal (toits, planchers)
- ✅ **Score de mur** - Probabilité d'être un élément de mur
- ✅ **Score de toit** - Probabilité d'être un élément de toit

### Performance par Type de Caractéristique

| Type de Caractéristique  | Temps CPU | Temps GPU | Accélération |
| ------------------------ | --------- | --------- | ------------ |
| Normales de Surface      | 2.5s      | 0.3s      | 8.3x         |
| Courbure                 | 3.0s      | 0.4s      | 7.5x         |
| Hauteur au-dessus du Sol | 1.5s      | 0.2s      | 7.5x         |
| Caractéristiques Géom.   | 4.0s      | 0.6s      | 6.7x         |
| Caractéristiques Bât.    | 5.0s      | 0.8s      | 6.3x         |
| **Total (1M points)**    | **16s**   | **2.3s**  | **7x**       |

## Changements dans v1.6.2

### Corrections de Formules

Les formules GPU ont été corrigées pour correspondre au CPU et à la littérature standard :

**Avant v1.6.2** (INCORRECT) :

```python
planarity = (λ1 - λ2) / λ0  # Normalisation incorrecte
linearity = (λ0 - λ1) / λ0  # Normalisation incorrecte
sphericity = λ2 / λ0         # Normalisation incorrecte
```

**v1.6.2+** (CORRECT - correspond à [Weinmann et al., 2015](https://www.sciencedirect.com/science/article/pii/S0924271615001842)) :

```python
sum_λ = λ0 + λ1 + λ2
planarity = (λ1 - λ2) / sum_λ   # Formulation standard
linearity = (λ0 - λ1) / sum_λ   # Formulation standard
sphericity = λ2 / sum_λ          # Formulation standard
```

### Nouvelles Fonctionnalités de Robustesse

1. **Filtrage des Cas Dégénérés** : Les points avec voisins insuffisants ou valeurs propres proches de zéro retournent maintenant 0.0 au lieu de NaN/Inf
2. **Courbure Robuste** : Utilise la Déviation Absolue Médiane (MAD) au lieu de std pour la résistance aux valeurs aberrantes
3. **Support Recherche par Rayon** : Recherche de voisinage optionnelle basée sur le rayon (bascule vers CPU)

### Validation

Le GPU produit maintenant des résultats identiques au CPU (validé : différence max < 0.0001%) :

```python
# Exécuter le test de validation
python tests/test_feature_fixes.py
# Attendu : ✓✓✓ TOUS LES TESTS RÉUSSIS ✓✓✓
```

Pour plus de détails, voir :

- [Notes de Version v1.6.2](/docs/release-notes/v1.6.2)
- Fichiers du dépôt : `GEOMETRIC_FEATURES_ANALYSIS.md`, `IMPLEMENTATION_SUMMARY.md`

---

## Référence API

### Classe GPUFeatureComputer

La classe principale pour le calcul de caractéristiques accéléré par GPU.

```python
from ign_lidar.features_gpu import GPUFeatureComputer

# Initialiser le calculateur de caractéristiques GPU
computer = GPUFeatureComputer(
    use_gpu=True,
    batch_size=100000,
    memory_limit=0.8,
    device_id=0
)
```

#### Paramètres du Constructeur

| Paramètre      | Type  | Défaut   | Description                              |
| -------------- | ----- | -------- | ---------------------------------------- |
| `use_gpu`      | bool  | `True`   | Activer l'accélération GPU               |
| `batch_size`   | int   | `100000` | Points traités par lot GPU               |
| `memory_limit` | float | `0.8`    | Limite d'utilisation mémoire GPU (0-1)   |
| `device_id`    | int   | `0`      | ID du périphérique CUDA (pour multi-GPU) |

### Méthodes Principales

#### compute_all_features_with_gpu()

Calculer toutes les caractéristiques pour un nuage de points en utilisant l'accélération GPU.

```python
from ign_lidar.features import compute_all_features_with_gpu
import numpy as np

# Vos données de nuage de points
points = np.random.rand(1000000, 3).astype(np.float32)
classification = np.random.randint(0, 10, 1000000).astype(np.uint8)

# Calculer les caractéristiques
normals, curvature, height, geo_features = compute_all_features_with_gpu(
    points=points,
    classification=classification,
    k=20,
    auto_k=False,
    use_gpu=True,
    batch_size=100000
)
```

**Paramètres :**

| Paramètre        | Type       | Requis | Description                                                 |
| ---------------- | ---------- | ------ | ----------------------------------------------------------- |
| `points`         | np.ndarray | Oui    | Coordonnées des points (N, 3)                               |
| `classification` | np.ndarray | Oui    | Classifications des points (N,)                             |
| `k`              | int        | Non    | Nombre de voisins pour les caractéristiques (défaut : 20)   |
| `auto_k`         | bool       | Non    | Ajuster automatiquement k selon la densité (défaut : False) |
| `use_gpu`        | bool       | Non    | Activer l'accélération GPU (défaut : True)                  |
| `batch_size`     | int        | Non    | Taille de lot pour le traitement GPU (défaut : 100000)      |

**Retourne :**

| Valeur Retournée | Type       | Forme  | Description                           |
| ---------------- | ---------- | ------ | ------------------------------------- |
| `normals`        | np.ndarray | (N, 3) | Vecteurs normaux de surface           |
| `curvature`      | np.ndarray | (N,)   | Valeurs de courbure                   |
| `height`         | np.ndarray | (N,)   | Hauteur au-dessus du sol              |
| `geo_features`   | dict       | -      | Dictionnaire de caractéristiques géom |

#### compute_normals_gpu()

Calculer les normales de surface en utilisant le GPU.

```python
from ign_lidar.features_gpu import compute_normals_gpu

normals = compute_normals_gpu(
    points=points,
    k=20,
    batch_size=100000
)
```

#### compute_curvature_gpu()

Calculer les valeurs de courbure en utilisant le GPU.

```python
from ign_lidar.features_gpu import compute_curvature_gpu

curvature = compute_curvature_gpu(
    points=points,
    normals=normals,
    k=20,
    batch_size=100000
)
```

#### compute_geometric_features_gpu()

Calculer toutes les caractéristiques géométriques en utilisant le GPU.

```python
from ign_lidar.features_gpu import compute_geometric_features_gpu

geo_features = compute_geometric_features_gpu(
    points=points,
    normals=normals,
    k=20,
    batch_size=100000
)

# Accéder aux caractéristiques individuelles
planarity = geo_features['planarity']
linearity = geo_features['linearity']
sphericity = geo_features['sphericity']
```

## Utilisation Avancée

### Optimisation du Traitement par Lots

Pour traiter plusieurs tuiles, réutiliser l'instance de calculateur GPU :

```python
from ign_lidar.features_gpu import GPUFeatureComputer
from pathlib import Path

# Initialiser une fois
computer = GPUFeatureComputer(use_gpu=True, batch_size=100000)

# Traiter plusieurs tuiles
for tile_path in Path("tiles/").glob("*.laz"):
    # Charger la tuile
    points, classification = load_tile(tile_path)

    # Calculer les caractéristiques (GPU reste initialisé)
    normals, curvature, height, geo_features = computer.compute_all(
        points=points,
        classification=classification,
        k=20
    )

    # Sauvegarder les résultats
    save_enriched_tile(tile_path, normals, curvature, height, geo_features)
```

### Gestion de la Mémoire

Contrôler l'utilisation de la mémoire GPU pour les grandes tuiles :

```python
from ign_lidar.features_gpu import GPUFeatureComputer

# Pour les très grandes tuiles (>5M points)
computer = GPUFeatureComputer(
    use_gpu=True,
    batch_size=50000,  # Taille de lot plus petite
    memory_limit=0.6   # Utiliser moins de mémoire GPU
)

# Pour les tuiles petites à moyennes (moins de 1M points)
computer = GPUFeatureComputer(
    use_gpu=True,
    batch_size=200000,  # Taille de lot plus grande
    memory_limit=0.9    # Utiliser plus de mémoire GPU
)
```

### Support Multi-GPU (Expérimental)

:::caution Fonctionnalité Expérimentale
Le support multi-GPU est expérimental dans v1.5.0. Utiliser avec précaution en production.
:::

```python
import os

# Spécifier le périphérique GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Utiliser le premier GPU

# Ou pour un GPU spécifique
computer = GPUFeatureComputer(use_gpu=True, device_id=1)  # Utiliser le second GPU
```

### Calcul de Caractéristiques Personnalisées

Implémenter des caractéristiques personnalisées accélérées par GPU :

```python
import cupy as cp
from ign_lidar.features_gpu import GPUFeatureComputer

class CustomGPUComputer(GPUFeatureComputer):
    def compute_custom_feature(self, points_gpu):
        """Calculer une caractéristique personnalisée sur GPU"""
        # Votre calcul GPU personnalisé utilisant CuPy
        feature = cp.mean(points_gpu, axis=1)
        return cp.asnumpy(feature)

# Utiliser le calculateur personnalisé
computer = CustomGPUComputer(use_gpu=True)
```

## Conseils d'Optimisation des Performances

### 1. Taille de Lot Optimale

Choisir la taille de lot selon la mémoire GPU :

| Mémoire GPU | Taille de Lot Recommandée | Nuage de Points Max |
| ----------- | ------------------------- | ------------------- |
| 4 GB        | 50 000                    | 2M points           |
| 8 GB        | 100 000                   | 5M points           |
| 12 GB       | 150 000                   | 8M points           |
| 16 GB+      | 200 000+                  | 10M+ points         |

### 2. Sélection de K-Neighbors

Les valeurs k plus grandes bénéficient plus du GPU :

```python
# Optimal pour GPU (k >= 20)
features = compute_all_features_with_gpu(points, classification, k=20, use_gpu=True)

# Moins optimal pour GPU (k < 10)
features = compute_all_features_with_gpu(points, classification, k=5, use_gpu=True)
```

### 3. Optimisation des Transferts Mémoire

Minimiser les transferts CPU-GPU en regroupant les opérations :

```python
# ❌ Mauvais : Transferts multiples
normals = compute_normals_gpu(points)
curvature = compute_curvature_gpu(points, normals)  # Transférer normales retour
geo = compute_geometric_features_gpu(points, normals)  # Transférer à nouveau

# ✅ Bon : Lot unique
normals, curvature, height, geo = compute_all_features_with_gpu(points, classification)
```

### 4. Mémoire GPU Persistante

Pour le traitement répété, garder les données sur GPU :

```python
import cupy as cp

# Transférer vers GPU une fois
points_gpu = cp.asarray(points)

# Traiter plusieurs fois sans re-transfert
for k in [10, 20, 30]:
    normals = compute_normals_gpu(points_gpu, k=k)
```

## Benchmark

### Exécuter les Benchmarks

La bibliothèque inclut des outils de benchmark complets :

```bash
# Benchmark synthétique (test rapide)
python scripts/benchmarks/benchmark_gpu.py --synthetic

# Benchmark avec données réelles
python scripts/benchmarks/benchmark_gpu.py path/to/tile.laz

# Benchmark multi-tailles
python scripts/benchmarks/benchmark_gpu.py --multi-size

# Comparer différentes valeurs de k
python scripts/benchmarks/benchmark_gpu.py --test-k
```

### Interprétation des Résultats

Exemple de sortie de benchmark :

```
GPU Benchmark Results
=====================
GPU Model: NVIDIA RTX 3080 (10GB)
CUDA Version: 11.8
CuPy Version: 11.6.0

Point Cloud: 1,000,000 points
K-neighbors: 20

Feature Computation Times:
--------------------------
Normals (CPU):     2.45s
Normals (GPU):     0.31s  → 7.9x speedup

Curvature (CPU):   2.98s
Curvature (GPU):   0.42s  → 7.1x speedup

Geometric (CPU):   3.87s
Geometric (GPU):   0.58s  → 6.7x speedup

Total (CPU):      15.32s
Total (GPU):       2.14s  → 7.2x speedup

Memory Usage:
-------------
GPU Memory Used:   1.2 GB / 10 GB (12%)
Peak Memory:       1.8 GB
CPU Memory Used:   2.4 GB
```

### Facteurs de Performance

Les performances GPU dépendent de :

1. **Taille du nuage de points** : Plus grand = meilleure utilisation GPU
2. **Valeur K-neighbors** : Plus grand = plus de travail parallélisable
3. **Modèle GPU** : Plus récent = traitement plus rapide
4. **Bande passante mémoire** : Plus élevée = transferts plus rapides
5. **Capacité de calcul CUDA** : Plus élevée = plus de fonctionnalités

## Dépannage

### Problèmes de Performance

#### GPU Plus Lent que Prévu

**Symptômes** : Le traitement GPU n'est pas beaucoup plus rapide que le CPU

**Causes Possibles** :

1. Petits nuages de points (&lt;10K points) - La surcharge GPU domine
2. Valeur k faible (&lt;10) - Pas assez de travail parallélisable
3. Goulot d'étranglement de transfert mémoire
4. GPU pas entièrement utilisé

**Solutions** :

```bash
# Vérifier l'utilisation GPU
nvidia-smi -l 1

# Devrait afficher une utilisation GPU élevée pendant le traitement
# Si l'utilisation GPU est faible :
```

```python
# Augmenter la taille de lot
computer = GPUFeatureComputer(batch_size=200000)

# Augmenter la valeur k
features = compute_all_features_with_gpu(points, classification, k=30)

# Utiliser des tuiles plus grandes ou du traitement par lots
```

#### Erreurs de Mémoire Insuffisante

**Symptômes** : Erreurs CUDA de mémoire insuffisante

**Solutions** :

```python
# Réduire la taille de lot
computer = GPUFeatureComputer(batch_size=50000)

# Réduire la limite mémoire
computer = GPUFeatureComputer(memory_limit=0.6)

# Traiter par morceaux plus petits
for chunk in split_point_cloud(points, chunk_size=500000):
    features = compute_all_features_with_gpu(chunk, classification)
```

#### Erreurs d'Importation CuPy

**Symptômes** : ImportError ou avertissements d'incompatibilité de version CUDA

**Solutions** :

```bash
# Vérifier la version CUDA
nvidia-smi | grep "CUDA Version"

# Réinstaller CuPy correspondant
pip uninstall cupy
pip install cupy-cuda11x  # ou cupy-cuda12x
```

### Fuites Mémoire

Si la mémoire GPU continue d'augmenter :

```python
# Forcer le nettoyage de la mémoire GPU
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()

# Ou utiliser un gestionnaire de contexte
from ign_lidar.features_gpu import GPUMemoryManager

with GPUMemoryManager():
    # Mémoire GPU automatiquement libérée après ce bloc
    features = compute_all_features_with_gpu(points, classification)
```

## Limitations

### Limitations Actuelles

1. **Mémoire GPU** : Limitée par la RAM GPU disponible
2. **GPU Unique** : Le support multi-GPU est expérimental
3. **NVIDIA Uniquement** : Nécessite un GPU NVIDIA avec CUDA
4. **Implémentation K-NN** : Utilise force brute pour k < 50, KD-tree pour k >= 50

### Améliorations Futures (Feuille de Route)

- 🔄 **Support Multi-GPU** (v1.6.0) - Distribuer le travail sur plusieurs GPUs
- 🔄 **Précision Mixte** (v1.6.0) - Utiliser FP16 pour un calcul plus rapide
- 🔄 **Support GPU AMD** (v2.0.0) - Support ROCm pour GPUs AMD
- 🔄 **Traitement par Morceaux** (v1.6.0) - Découpage automatique pour très grandes tuiles
- 🔄 **Cache GPU Persistant** (v1.7.0) - Mettre en cache les données prétraitées sur GPU

## Voir Aussi

- **[Vue d'ensemble GPU](overview.md)** - Configuration et installation GPU
- **[Accélération RGB GPU](rgb-augmentation.md)** - Augmentation RGB accélérée par GPU
- **[Architecture](../architecture.md)** - Architecture système
- **[Flux de Travail](../workflows.md)** - Exemples de flux de travail GPU

## Références

- [Documentation CuPy](https://docs.cupy.dev/)
- [Guide de Programmation CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Guide d'Optimisation GPU](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
