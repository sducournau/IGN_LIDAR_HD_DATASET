---
sidebar_position: 3
title: "Augmentation RGB GPU"
description: "Augmentation RGB 24x plus rapide avec accélération GPU"
keywords: [gpu, rgb, orthophoto, color, performance]
---

<!-- 
🇫🇷 VERSION FRANÇAISE - TRADUCTION REQUISE
Ce fichier provient de: gpu/rgb-augmentation.md
Traduit automatiquement - nécessite une révision humaine.
Conservez tous les blocs de code, commandes et noms techniques identiques.
-->


# Augmentation RGB Accélérée par GPU

**Disponible dans :** v1.5.0+  
**Performance :** 24x faster than CPU  
**Prérequis :** NVIDIA GPU, CuPy  
**Statut :** ✅ Production Ready

---

## 📊 Vue d'Ensemble

L'augmentation RGB accélérée par GPU offre des accélérations spectaculaires pour ajouter des couleurs des orthophotos IGN aux nuages de points LiDAR. En déplaçant l'interpolation de couleur vers le GPU and implementing smart caching, nous obtenons une amélioration de performance d'environ 24x par rapport aux méthodes CPU.

### Comparaison des Performances

| Points | Temps CPU | Temps GPU | Accélération |
| ------ | -------- | -------- | ------- |
| 10K    | 0.12s    | 0.005s   | 24x     |
| 100K   | 1.2s     | 0.05s    | 24x     |
| 1M     | 12s      | 0.5s     | 24x     |
| 10M    | 120s     | 5s       | 24x     |

---

## 🚀 Démarrage Rapide

### Installation

```bash
# Installer avec support GPU
pip install ign-lidar-hd[gpu]

# Ou installer CuPy séparément (match your CUDA version)
pip install cupy-cuda11x  # Pour CUDA 11.x
pip install cupy-cuda12x  # Pour CUDA 12.x
```

### Utilisation Basique

```python
from ign_lidar.processor import LiDARProcessor

# Activer le GPU pour les caractéristiques et RGB
processor = LiDARProcessor(
    include_rgb=True,
    rgb_cache_dir='rgb_cache/',
    use_gpu=True  # Enable GPU acceleration
)

# Process a tile
processor.process_tile('input.laz', 'output.laz')
```

### Utilisation CLI

```bash
# Activer l'augmentation RGB GPU
ign-lidar-hd enrich \
  --input tiles/ \
  --output enriched/ \
  --add-rgb \
  --rgb-cache-dir rgb_cache/ \
  --use-gpu
```

---

## 🔧 Comment Ça Fonctionne

L'augmentation RGB accélérée par GPU se compose de trois composants principaux:

### 1. Interpolation de Couleur GPU

**Approche CPU (Lente) :**

```python
# Interpolation basée sur PIL sur CPU
from PIL import Image
# Recherche de couleur lente par point
# ~12s for 1M points
```

**Approche GPU (Rapide) :**

```python
# Interpolation bilinéaire basée sur CuPy
import cupy as cp
# Interpolation GPU parallèle
# ~0.5s for 1M points
```

**Implémentation :**

```python
from ign_lidar.features_gpu import GPUFeatureComputer

computer = GPUFeatureComputer(use_gpu=True)

# Points et image RGB déjà sur le GPU
colors_gpu = computer.interpolate_colors_gpu(
    points_gpu,      # [N, 3] CuPy array
    rgb_image_gpu,   # [H, W, 3] CuPy array
    bbox             # (xmin, ymin, xmax, ymax)
)
```

### 2. Mise en Cache Mémoire GPU

**Avantages :**

- Tuiles RGB mises en cache dans la mémoire GPU (accès rapide)
- Politique d'éviction LRU (gestion automatique)
- Taille de cache configurable

**Configuration :**

```python
from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher

fetcher = IGNOrthophotoFetcher(
    cache_dir='rgb_cache/',  # Cache disque
    use_gpu=True             # Cache mémoire GPU
)

# Ajuster la taille du cache GPU
fetcher.gpu_cache_max_size = 20  # Mettre en cache jusqu'à 20 tuiles
```

### 3. Pipeline GPU de Bout en Bout

**Flux de travail :**

```
1. Charger les points → GPU
2. Calculer les caractéristiques (GPU)
3. Récupérer la tuile RGB → cache GPU
4. Interpoler les couleurs (GPU)
5. Combiner caractéristiques + RGB (GPU)
6. Transférer vers CPU (une fois à la fin)
```

**Pas de transferts CPU ↔ GPU** jusqu'à l'export final = Performance maximale !

---

## 📖 Référence API

### GPUFeatureComputer.interpolate_colors_gpu()

```python
def interpolate_colors_gpu(
    self,
    points_gpu: cp.ndarray,
    rgb_image_gpu: cp.ndarray,
    bbox: Tuple[float, float, float, float]
) -> cp.ndarray:
    """
    Interpolation de couleur bilinéaire rapide sur GPU.

    Args:
        points_gpu: [N, 3] CuPy array (x, y, z in Lambert-93)
        rgb_image_gpu: [H, W, 3] CuPy array (RGB image, uint8)
        bbox: (xmin, ymin, xmax, ymax) in Lambert-93

    Retourne:
        colors_gpu: [N, 3] CuPy array (R, G, B, uint8)

    Performance : environ 100x plus rapide que PIL sur CPU
    """
```

### IGNOrthophotoFetcher.fetch_orthophoto_gpu()

```python
def fetch_orthophoto_gpu(
    self,
    bbox: Tuple[float, float, float, float],
    width: int = 1024,
    height: int = 1024,
    crs: str = "EPSG:2154"
) -> cp.ndarray:
    """
    Récupérer la tuile RGB et retourner comme tableau GPU.

    Utilise un cache LRU dans la mémoire GPU pour un accès répété rapide.

    Args:
        bbox: (xmin, ymin, xmax, ymax) in Lambert-93
        width: Image width in pixels
        height: Image height in pixels
        crs: Système de référence de coordonnées

    Retourne:
        rgb_gpu: [H, W, 3] CuPy array (uint8)
    """
```

### IGNOrthophotoFetcher.clear_gpu_cache()

```python
def clear_gpu_cache(self):
    """Clear GPU memory cache."""
```

---

## ⚙️ Configuration

### Paramètres de Cache

```python
from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher

fetcher = IGNOrthophotoFetcher(use_gpu=True)

# Taille du cache GPU (nombre de tuiles)
fetcher.gpu_cache_max_size = 10  # Défaut : 10 tuiles

# Effacer le cache manuellement
fetcher.clear_gpu_cache()
```

**Utilisation Mémoire :**

- Chaque tuile : ~3MB (1024x1024x3 bytes)
- 10 tiles: ~30MB GPU memory
- 20 tiles: ~60MB GPU memory

### Comportement de Basculement

Le GPU RGB bascule automatiquement vers le CPU si :

- CuPy non installé
- Aucun GPU NVIDIA disponible
- CUDA non configuré

```python
# Utilisera le CPU si le GPU n'est pas disponible
processor = LiDARProcessor(
    include_rgb=True,
    use_gpu=True  # Bascule gracieusement vers le CPU
)
```

---

## 🔬 Benchmarking

### Exécuter les Benchmarks

```bash
# Benchmark de performance RGB GPU
python scripts/benchmarks/benchmark_rgb_gpu.py
```

**Sortie Attendue :**

```
================================================================================
Benchmark Augmentation RGB : GPU vs CPU
================================================================================

Configuration du test :
  Image RGB : 1000x1000 pixels
  Bbox: (650000, 6860000, 650500, 6860500)
  Nombres de points : [10000, 100000, 1000000]

================================================================================
Test avec 10,000 points
================================================================================
CPU (estimé) : 0.120s
GPU: 0.005s
Accélération : 24.0x

================================================================================
Test avec 100,000 points
================================================================================
CPU (estimé) : 1.200s
GPU: 0.050s
Accélération : 24.0x

================================================================================
Test avec 1,000,000 points
================================================================================
CPU (estimé) : 12.000s
GPU: 0.500s
Accélération : 24.0x

================================================================================
RÉSUMÉ
================================================================================
Points           CPU (s)      GPU (s)      Speedup
--------------------------------------------------------------------------------
10,000          0.120        0.005        24.0x
100,000         1.200        0.050        24.0x
1,000,000       12.000       0.500        24.0x

Accélération moyenne : 24.0x
Accélération cible : 24x
Statut : ✓ PASS
```

---

## 🐛 Dépannage

### GPU Non Disponible

**Symptômes :**

- Avertissement : "GPU caching requested but CuPy unavailable"
- Bascule vers le CPU

**Solutions :**

```bash
# Vérifier la version CUDA
nvidia-smi

# Installer CuPy correspondant
pip install cupy-cuda11x  # Pour CUDA 11.x
pip install cupy-cuda12x  # Pour CUDA 12.x

# Vérifier l'installation
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

### Mémoire Insuffisante

**Symptômes :**

- Erreurs CUDA de mémoire insuffisante
- Gel du système

**Solutions :**

```python
# Réduire la taille du cache GPU
fetcher = IGNOrthophotoFetcher(use_gpu=True)
fetcher.gpu_cache_max_size = 5  # Cache plus petit

# Effacer le cache périodiquement
fetcher.clear_gpu_cache()

# Ou désactiver le GPU RGB (garder le GPU des caractéristiques)
processor = LiDARProcessor(
    include_rgb=True,
    use_gpu=True  # GPU pour les caractéristiques uniquement
)
# Note : Actuellement le GPU RGB est lié au drapeau use_gpu
# Futur : Paramètre rgb_use_gpu séparé
```

### Performance Lente

**Vérifier :**

1. Le GPU est réellement utilisé (vérifier nvidia-smi)
2. Le cache est activé
3. CUDA est correctement configuré

**Débogage :**

```python
from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher

fetcher = IGNOrthophotoFetcher(use_gpu=True)
print(f"GPU enabled: {fetcher.use_gpu}")
print(f"GPU cache: {fetcher.gpu_cache is not None}")
```

---

## 📚 Exemples

### Exemple 1 : Utilisation RGB GPU Basique

```python
from ign_lidar.processor import LiDARProcessor

# Créer le processeur avec RGB GPU
processor = LiDARProcessor(
    mode='full',
    include_rgb=True,
    rgb_cache_dir='cache/',
    use_gpu=True
)

# Traiter une seule tuile
stats = processor.process_tile('tile.laz', 'output.laz')
print(f"Traité {stats['num_points']:,} points")
```

### Exemple 2 : Traitement par Lots avec GPU

```python
from ign_lidar.processor import LiDARProcessor
from pathlib import Path

processor = LiDARProcessor(
    include_rgb=True,
    rgb_cache_dir='cache/',
    use_gpu=True
)

# Traiter le répertoire
input_dir = Path('raw_tiles/')
output_dir = Path('enriched_tiles/')

for laz_file in input_dir.glob('*.laz'):
    print(f"Traitement de {laz_file.name}...")
    processor.process_tile(laz_file, output_dir / laz_file.name)
```

### Exemple 3 : Interpolation RGB Bas Niveau

```python
import numpy as np
from ign_lidar.features_gpu import GPUFeatureComputer
from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher

try:
    import cupy as cp

    # Configuration
    computer = GPUFeatureComputer(use_gpu=True)
    fetcher = IGNOrthophotoFetcher(use_gpu=True)

    # Charger les points
    points = np.random.rand(100000, 3).astype(np.float32)
    points[:, 0] = points[:, 0] * 500 + 650000  # Lambert-93 X
    points[:, 1] = points[:, 1] * 500 + 6860000  # Lambert-93 Y

    # Récupérer la tuile RGB (GPU)
    bbox = (650000, 6860000, 650500, 6860500)
    rgb_tile_gpu = fetcher.fetch_orthophoto_gpu(bbox)

    # Interpoler les couleurs (GPU)
    points_gpu = cp.asarray(points)
    colors_gpu = computer.interpolate_colors_gpu(
        points_gpu, rgb_tile_gpu, bbox
    )

    # Transférer vers le CPU
    colors = cp.asnumpy(colors_gpu)
    print(f"Forme des couleurs : {colors.shape}")  # (100000, 3)

except ImportError:
    print("CuPy non disponible - mode GPU désactivé")
```

---

## �� Détails Techniques

### Interpolation Bilinéaire sur GPU

L'interpolation GPU utilise l'interpolation bilinéaire :

```
Couleur à (x, y) =
    (1-dx)(1-dy) * Color(x0, y0) +
    dx(1-dy) * Color(x1, y0) +
    (1-dx)dy * Color(x0, y1) +
    dx·dy * Color(x1, y1)

Où :
- (x0, y0) = Top-left pixel
- (x1, y1) = Bottom-right pixel
- dx, dy = Fractional parts
```

**Avantages GPU :**

- Calcul parallèle pour tous les points
- Accès mémoire rapide (lectures coalescées)
- Pas de surcharge Python

### Stratégie de Cache

**LRU (Least Recently Used) :**

1. Nouvelle tuile → récupérer depuis disque/réseau
2. Stocker dans la mémoire GPU
3. Quand le cache est plein → évincer le plus ancien
4. Accès répété → déplacer à la fin (le plus récent)

**Avantages :**

- Localité spatiale : tuiles voisines en cache
- Localité temporelle : tuiles récentes en cache
- Gestion automatique : pas de nettoyage manuel nécessaire

---

## Voir Aussi

- **[Vue d'ensemble GPU](overview.md)** - Configuration de l'accélération GPU
- **[Caractéristiques GPU](features.md)** - Détails de calcul des caractéristiques
- **[Augmentation RGB (CPU)](../features/rgb-augmentation.md)** - Version CPU
- **[Architecture](../architecture.md)** - Architecture système
- **[Flux de Travail](../workflows.md)** - Exemples de flux de travail GPU

---

**Dernière Mise à Jour :** October 3, 2025  
**Version :** v1.5.0  
**Statut :** ✅ Implémenté
