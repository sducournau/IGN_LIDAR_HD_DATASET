# Guide de Calcul des Normales - API Unifiée

**Date:** 23 Novembre 2025  
**Version:** 3.6.0+

---

## 🎯 Vue d'Ensemble

Le calcul des normales est l'opération la plus critique du pipeline LiDAR. Ce guide explique comment utiliser l'API unifiée pour éviter les duplications et garantir des performances optimales.

## 📊 Hiérarchie d'Implémentation

```
┌─────────────────────────────────────────────────┐
│  FeatureOrchestrator (RECOMMANDÉ)               │
│  - Point d'entrée unifié                        │
│  - Routage automatique CPU/GPU                  │
│  - Gestion de configuration Hydra               │
└──────────────┬──────────────────────────────────┘
               │
               ├─► CPU Strategy
               │   └─► compute.normals.compute_normals()
               │       ├─► knn_search() [Phase 2 unified]
               │       └─► eigh() [GPU-accelerated]
               │
               └─► GPU Strategy
                   └─► strategy_gpu.py
                       └─► cuML batch operations
```

## ✅ API Recommandée

### 1. Point d'Entrée Principal (RECOMMANDÉ)

```python
from ign_lidar.features import FeatureOrchestrator

# Configuration Hydra
orchestrator = FeatureOrchestrator(config)

# Calcul automatique CPU/GPU
features = orchestrator.compute_features(
    points,
    mode='lod2',  # ou 'lod3', 'minimal', 'full'
    use_gpu=True  # Auto-détecte si GPU disponible
)

# Accès aux normales
normals = features['normals']
eigenvalues = features['eigenvalues']
```

**Avantages:**

- ✅ Routage automatique CPU/GPU
- ✅ Configuration centralisée
- ✅ Gestion mémoire intégrée
- ✅ Support multi-échelle
- ✅ Cache GPU automatique

### 2. Calcul Direct (CPU Only)

```python
from ign_lidar.features.compute import compute_normals

# CPU standard (k=20)
normals, eigenvalues = compute_normals(points, k_neighbors=20)

# CPU rapide (k=10, pas d'eigenvalues)
normals, _ = compute_normals(
    points,
    method='fast',
    return_eigenvalues=False
)

# CPU précis (k=50)
normals, eigenvalues = compute_normals(points, method='accurate')

# Avec rayon de recherche
normals, eigenvalues = compute_normals(
    points,
    k_neighbors=20,
    search_radius=2.0
)
```

**Avantages:**

- ✅ API simple et directe
- ✅ Pas de dépendances lourdes
- ✅ Idéal pour prototypage

### 3. GPU Haute Performance

```python
from ign_lidar.features import GPUStrategy

# DEPRECATED: Utiliser FeatureOrchestrator à la place
# Cette API sera supprimée en v4.0.0
```

## ❌ Ce Qu'il Ne Faut PAS Faire

### ❌ N'importez PAS depuis gpu_processor

```python
# ❌ DEPRECATED - Sera supprimé en v4.0.0
from ign_lidar.features.gpu_processor import GPUProcessor
processor = GPUProcessor()
normals = processor.compute_normals(points)
```

**Remplacer par:**

```python
# ✅ Version recommandée
from ign_lidar.features import FeatureOrchestrator
orchestrator = FeatureOrchestrator(config)
features = orchestrator.compute_features(points, use_gpu=True)
normals = features['normals']
```

### ❌ N'utilisez PAS les fonctions deprecated

```python
# ❌ DEPRECATED
from ign_lidar.features.compute.normals import compute_normals_fast
normals = compute_normals_fast(points)

# ❌ DEPRECATED
from ign_lidar.features.compute.normals import compute_normals_accurate
normals, eigs = compute_normals_accurate(points)
```

**Remplacer par:**

```python
# ✅ Version recommandée
from ign_lidar.features.compute import compute_normals

# Fast
normals, _ = compute_normals(points, method='fast', return_eigenvalues=False)

# Accurate
normals, eigs = compute_normals(points, method='accurate')
```

## 🔧 Paramètres Recommandés

### Par Type de Données

| Type de Données                 | k_neighbors | Méthode    | GPU |
| ------------------------------- | ----------- | ---------- | --- |
| **Prototypage rapide**          | 10          | `fast`     | ❌  |
| **Production standard**         | 20          | `standard` | ✅  |
| **Haute précision**             | 50          | `accurate` | ✅  |
| **LOD2 (bâtiments)**            | 30          | `standard` | ✅  |
| **LOD3 (détails architecture)** | 50          | `accurate` | ✅  |

### Par Densité de Points

| Densité (pts/m²) | k_neighbors | Rayon | Recommandation   |
| ---------------- | ----------- | ----- | ---------------- |
| < 5              | 10-15       | 3.0m  | Sparse, k faible |
| 5-10             | 20-30       | 2.0m  | Standard         |
| 10-20            | 30-40       | 1.5m  | Dense            |
| > 20             | 40-50       | 1.0m  | Très dense       |

## ⚡ Optimisations GPU

### Transferts CPU↔GPU Efficaces

```python
# ❌ Inefficace: Transferts dans la boucle
for tile in tiles:
    points_gpu = cp.asarray(tile.points)  # Upload
    normals_gpu = compute_gpu(points_gpu)
    normals = cp.asnumpy(normals_gpu)     # Download
    # Perte de performance: 2N transferts

# ✅ Efficace: Batch processing
all_points_gpu = cp.asarray(all_points)   # Upload 1x
all_normals_gpu = compute_gpu(all_points_gpu)
all_normals = cp.asnumpy(all_normals_gpu) # Download 1x
# Gain: 2 transferts au total
```

### Utilisation du Cache GPU

```python
from ign_lidar.core.gpu import GPUManager

gpu = GPUManager()

# Upload avec cache
points_gpu = gpu.cache.get_or_upload('tile_001', points)

# Calcul
normals = compute_on_gpu(points_gpu)

# Réutilisation automatique
points_gpu = gpu.cache.get_or_upload('tile_001', points)  # Pas de re-upload!
```

## 📈 Benchmarks

### Temps de Calcul (1M points)

| Méthode       | Matériel | k=20  | k=50  | Speedup |
| ------------- | -------- | ----- | ----- | ------- |
| CPU (sklearn) | i7-12700 | 12.3s | 28.7s | 1x      |
| CPU (FAISS)   | i7-12700 | 3.1s  | 7.2s  | 4x      |
| GPU (cuML)    | RTX 3090 | 0.8s  | 1.9s  | 15x     |
| GPU (FAISS)   | RTX 3090 | 0.2s  | 0.5s  | 60x     |

### Recommandations

- **< 100k points:** CPU sklearn (simple, rapide pour petites données)
- **100k-1M points:** CPU FAISS ou GPU cuML
- **> 1M points:** GPU FAISS (meilleure performance)

## 🔍 Validation et Debug

### Vérifier la Qualité des Normales

```python
from ign_lidar.features.utils import validate_normals

# Calculer normales
normals, eigenvalues = compute_normals(points)

# Valider
is_valid, stats = validate_normals(normals, eigenvalues)

print(f"Valid: {is_valid}")
print(f"  Unit length: {stats['unit_length_ratio']:.2%}")
print(f"  Non-zero: {stats['non_zero_ratio']:.2%}")
print(f"  Planarity: {stats['avg_planarity']:.3f}")
```

### Visualiser les Normales

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Visualiser normales
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1, alpha=0.3)

# Normales (sous-échantillon)
sample = slice(0, len(points), 100)
ax.quiver(
    points[sample, 0], points[sample, 1], points[sample, 2],
    normals[sample, 0], normals[sample, 1], normals[sample, 2],
    length=0.5, color='red', alpha=0.8
)

plt.title('Point Cloud avec Normales')
plt.show()
```

## 🐛 Problèmes Courants

### 1. Normales Bruitées

**Symptôme:** Normales erratiques, variations rapides

**Cause:** Bruit dans les données, k trop faible

**Solution:**

```python
# Augmenter k
normals, _ = compute_normals(points, k_neighbors=50)

# Ou prétraiter les données
from ign_lidar.preprocessing import statistical_outlier_removal
points_clean, _ = statistical_outlier_removal(points, k=12, std_multiplier=2.0)
normals, _ = compute_normals(points_clean, k_neighbors=30)
```

### 2. GPU Out of Memory

**Symptôme:** `CUDAMemoryError` ou `RuntimeError: CUDA out of memory`

**Solution:**

```python
# Option 1: Chunked processing
orchestrator = FeatureOrchestrator(config)
orchestrator.config.features.chunk_size = 1_000_000  # 1M points par chunk

# Option 2: Libérer cache
from ign_lidar.core.gpu import GPUManager
gpu = GPUManager()
gpu.memory.free_cache()

# Option 3: Fallback CPU
features = orchestrator.compute_features(points, use_gpu=False)
```

### 3. Normales Inconsistantes aux Frontières

**Symptôme:** Artefacts aux bordures de tuiles

**Solution:**

```python
# Utiliser BoundaryAwareStrategy
from ign_lidar.features import FeatureOrchestrator

config.features.boundary_aware = True
config.features.boundary_buffer = 5.0  # 5m buffer

orchestrator = FeatureOrchestrator(config)
features = orchestrator.compute_features(points)
```

## 📚 Références

- **API Documentation:** `ign_lidar.features.compute.normals`
- **Architecture:** `docs/architecture/features_architecture.md`
- **GPU Guide:** `docs/guides/gpu_acceleration.md`
- **Preprocessing:** `docs/guides/preprocessing.md`

## 🔄 Migration depuis Versions Anciennes

### v2.x → v3.x

```python
# v2.x (OLD)
from ign_lidar.features_gpu import GPUFeatureComputer
computer = GPUFeatureComputer()
normals = computer.compute_normals(points)

# v3.x (NEW)
from ign_lidar.features import FeatureOrchestrator
orchestrator = FeatureOrchestrator(config)
features = orchestrator.compute_features(points, use_gpu=True)
normals = features['normals']
```

### v3.5 → v3.6+

```python
# v3.5 (OLD)
from ign_lidar.features.gpu_processor import GPUProcessor
processor = GPUProcessor()
normals = processor.compute_normals(points)

# v3.6+ (NEW)
from ign_lidar.features import FeatureOrchestrator
orchestrator = FeatureOrchestrator(config)
features = orchestrator.compute_features(points, use_gpu=True)
normals = features['normals']
```

---

**Questions?** Ouvrez une issue sur [GitHub](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
