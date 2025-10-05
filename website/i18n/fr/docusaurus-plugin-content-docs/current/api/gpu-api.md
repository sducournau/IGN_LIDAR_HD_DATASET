---
sidebar_position: 5
title: API GPU
description: Référence API d'accélération GPU pour le traitement LiDAR haute performance
keywords: [gpu, api, cuda, accélération, pytorch, tensor]
---

# Référence API GPU

Documentation complète de l'API pour les composants de traitement LiDAR accélérés par GPU.

## Classes GPU de Base

### GPUProcessor

Processeur principal accéléré par GPU pour les données LiDAR.

```python
from ign_lidar.gpu import GPUProcessor

processor = GPUProcessor(
    device="cuda:0",
    batch_size=10000,
    memory_fraction=0.8
)
```

#### Paramètres

- **device** (`str`) : Identifiant du périphérique GPU (`"cuda:0"`, `"cuda:1"`, etc.)
- **batch_size** (`int`) : Nombre de points traités par lot
- **memory_fraction** (`float`) : Fraction de mémoire GPU à utiliser (0.1-1.0)
- **mixed_precision** (`bool`) : Activer la précision mixte automatique (FP16/FP32)
- **compile_models** (`bool`) : Utiliser la compilation de modèles PyTorch 2.0

#### Méthodes

##### `process_points(points, features=None)`

Traiter les points LiDAR sur GPU.

```python
import torch
from ign_lidar.gpu import GPUProcessor

processor = GPUProcessor(device="cuda:0")

# Tenseur d'entrée (N, 3) pour les coordonnées XYZ
points = torch.tensor([[x1, y1, z1], [x2, y2, z2], ...], device="cuda:0")

# Traiter les points
result = processor.process_points(
    points=points,
    features=["buildings", "vegetation"]
)

# Structure du résultat
{
    'classifications': torch.Tensor,  # (N,) étiquettes de classe
    'features': torch.Tensor,        # (N, F) vecteurs de caractéristiques
    'confidence': torch.Tensor,      # (N,) scores de confiance
    'processing_time': float         # temps de traitement GPU
}
```

##### `extract_buildings_gpu(points, **kwargs)`

Extraction de bâtiments accélérée par GPU.

```python
buildings = processor.extract_buildings_gpu(
    points=points,
    min_points=100,
    height_threshold=2.0,
    planarity_threshold=0.1,
    return_meshes=True
)

# Retourne
{
    'building_points': torch.Tensor,    # (M, 3) points de bâtiments
    'building_ids': torch.Tensor,       # (M,) IDs d'instance de bâtiment
    'meshes': List[torch.Tensor],       # maillages 3D de bâtiments
    'properties': Dict                  # propriétés des bâtiments
}
```

##### `rgb_augmentation_gpu(points, orthophoto, **kwargs)`

Augmentation RGB accélérée par GPU.

```python
# Charger l'orthophoto comme tenseur
orthophoto = torch.from_numpy(orthophoto_array).cuda()

augmented = processor.rgb_augmentation_gpu(
    points=points,
    orthophoto=orthophoto,
    interpolation="bilinear",
    batch_size=50000
)

# Retourne les points avec couleurs RGB
{
    'points': torch.Tensor,          # (N, 3) coordonnées XYZ
    'colors': torch.Tensor,          # (N, 3) couleurs RGB [0-255]
    'interpolation_quality': torch.Tensor  # (N,) scores de qualité
}
```

### GPUMemoryManager

Gérer l'allocation et l'optimisation de la mémoire GPU.

```python
from ign_lidar.gpu import GPUMemoryManager

memory_manager = GPUMemoryManager(device="cuda:0")

# Obtenir les informations mémoire
info = memory_manager.get_memory_info()
print(f"Disponible : {info['available']:.2f} GB")
print(f"Total : {info['total']:.2f} GB")

# Optimiser la taille de lot selon la mémoire disponible
optimal_batch = memory_manager.get_optimal_batch_size(
    point_features=7,  # XYZ + RGB + intensité + classification
    model_memory_mb=500
)
```

#### Méthodes

##### `allocate_tensor_memory(size, dtype=torch.float32)`

Pré-allouer des tenseurs GPU pour une meilleure gestion de la mémoire.

```python
# Pré-allouer la mémoire pour de grands nuages de points
memory_pool = memory_manager.allocate_tensor_memory(
    size=(1000000, 3),  # 1M points, XYZ
    dtype=torch.float32
)

# Utiliser la mémoire pré-allouée
points_tensor = memory_pool[:actual_size]
```

##### `clear_cache()`

Vider le cache mémoire GPU.

```python
# Vider le cache PyTorch
memory_manager.clear_cache()

# Obtenir la mémoire libérée
freed_mb = memory_manager.get_freed_memory()
print(f"Libéré {freed_mb:.1f} MB")
```

### GPUFeatureExtractor

Extraire des caractéristiques géométriques sur GPU.

```python
from ign_lidar.gpu import GPUFeatureExtractor

extractor = GPUFeatureExtractor(
    device="cuda:0",
    neighborhood_size=50,
    feature_types=["eigenvalues", "normals", "curvature"]
)

features = extractor.extract_features(points)
```

#### Caractéristiques Supportées

##### Caractéristiques Basées sur les Valeurs Propres

```python
# Extraire les caractéristiques de valeurs propres
eigenvalue_features = extractor.extract_eigenvalues(
    points=points,
    k_neighbors=20,
    search_radius=1.0
)

# Retourne
{
    'eigenvalues': torch.Tensor,     # (N, 3) λ0, λ1, λ2
    'linearity': torch.Tensor,       # (N,) (λ0 - λ1) / λ0
    'planarity': torch.Tensor,       # (N,) (λ1 - λ2) / λ0
    'sphericity': torch.Tensor,      # (N,) λ2 / λ0
    'anisotropy': torch.Tensor,      # (N,) (λ0 - λ2) / λ0
    'eigenvectors': torch.Tensor     # (N, 3, 3) vecteurs propres
}
```

##### Estimation des Normales

```python
# Calculer les normales des points
normals = extractor.estimate_normals(
    points=points,
    k_neighbors=20,
    orient_normals=True,
    viewpoint=[0, 0, 10]  # Position caméra/capteur
)

# Retourne (N, 3) vecteurs normaux
```

##### Calcul de Courbure

```python
# Calculer la courbure de surface
curvature = extractor.compute_curvature(
    points=points,
    normals=normals,
    method="mean"  # "mean", "gaussian", "principal"
)

# Retourne (N,) valeurs de courbure
```

## Utilitaires GPU

### Opérations sur les Tenseurs

```python
from ign_lidar.gpu.utils import (
    points_to_tensor,
    tensor_to_points,
    batch_process,
    knn_search_gpu
)

# Convertir les points numpy en tenseur GPU
points_np = np.array([[x, y, z], ...])
points_gpu = points_to_tensor(points_np, device="cuda:0")

# Traitement par lots pour grands ensembles de données
results = batch_process(
    data=large_point_cloud,
    process_func=processor.extract_buildings_gpu,
    batch_size=100000,
    device="cuda:0"
)

# K plus proches voisins accéléré par GPU
neighbors, distances = knn_search_gpu(
    query_points=points_gpu,
    reference_points=reference_gpu,
    k=20
)
```

### Surveillance des Performances

```python
from ign_lidar.gpu.profiling import GPUProfiler, benchmark_gpu

# Profiler les opérations GPU
with GPUProfiler() as profiler:
    result = processor.process_points(points)

profiler.print_summary()
# Sortie :
# Utilisation GPU : 85.3%
# Utilisation Mémoire : 6.2/8.0 GB
# Temps de Traitement : 2.35s
# Débit : 425K points/sec

# Benchmark de différentes configurations
benchmark_results = benchmark_gpu(
    point_cloud=test_points,
    batch_sizes=[5000, 10000, 20000],
    devices=["cuda:0", "cuda:1"]
)
```

## Support Multi-GPU

### Traitement DataParallel

```python
from ign_lidar.gpu import MultiGPUProcessor

# Initialiser le processeur multi-GPU
multi_processor = MultiGPUProcessor(
    devices=["cuda:0", "cuda:1"],
    strategy="data_parallel"
)

# Traiter avec plusieurs GPU
results = multi_processor.process_batch(
    point_clouds=batch_of_tiles,
    features=["buildings", "vegetation"]
)
```

### Traitement Distribué

```python
import torch.distributed as dist
from ign_lidar.gpu.distributed import DistributedProcessor

# Initialiser le traitement distribué
dist.init_process_group(backend="nccl")
processor = DistributedProcessor(
    local_rank=0,
    world_size=4
)

# Extraction de caractéristiques distribuée
features = processor.extract_features_distributed(
    points=points,
    feature_types=["geometric", "radiometric"]
)
```

## Opérations GPU Avancées

### Noyaux CUDA Personnalisés

```python
from ign_lidar.gpu.kernels import (
    voxelize_cuda,
    ground_segmentation_cuda,
    noise_removal_cuda
)

# Voxeliser le nuage de points sur GPU
voxel_grid = voxelize_cuda(
    points=points_gpu,
    voxel_size=0.1,
    max_points_per_voxel=100
)

# Segmentation du sol GPU
ground_mask = ground_segmentation_cuda(
    points=points_gpu,
    cloth_resolution=0.5,
    iterations=500
)

# Suppression de bruit GPU
clean_points = noise_removal_cuda(
    points=points_gpu,
    std_ratio=2.0,
    nb_neighbors=20
)
```

### Génération de Maillages

```python
from ign_lidar.gpu.mesh import GPUMeshGenerator

mesh_generator = GPUMeshGenerator(device="cuda:0")

# Générer des maillages de bâtiments
meshes = mesh_generator.generate_building_meshes(
    building_points=building_points,
    method="poisson",
    octree_depth=9
)

# Exporter les maillages
for i, mesh in enumerate(meshes):
    mesh_generator.save_mesh(mesh, f"building_{i}.ply")
```

## Classes de Configuration

### GPUConfig

```python
from ign_lidar.gpu import GPUConfig

config = GPUConfig(
    # Paramètres de périphérique
    device="cuda:0",
    devices=["cuda:0", "cuda:1"],  # Multi-GPU
    fallback_to_cpu=True,

    # Gestion de la mémoire
    memory_fraction=0.8,
    pin_memory=True,
    empty_cache_every=100,

    # Optimisation des performances
    mixed_precision=True,
    compile_models=True,
    benchmark_cudnn=True,

    # Paramètres de traitement
    batch_size="auto",
    num_workers=4,
    prefetch_factor=2,

    # Extraction de caractéristiques
    neighborhood_size=50,
    feature_cache_size=1000000
)

# Utiliser la configuration
processor = GPUProcessor(config=config)
```

### Paramètres d'Optimisation

```python
# Réglage des performances
config.set_optimization_level("aggressive")
# Définit :
# - mixed_precision=True
# - compile_models=True
# - benchmark_cudnn=True
# - memory_fraction=0.9

# Paramètres optimisés pour la mémoire
config.set_optimization_level("memory_efficient")
# Définit :
# - memory_fraction=0.6
# - batch_size=5000
# - gradient_checkpointing=True
```

## Gestion des Erreurs

### Exceptions Spécifiques GPU

```python
from ign_lidar.gpu.exceptions import (
    GPUNotAvailableError,
    CUDAOutOfMemoryError,
    GPUComputeError
)

try:
    result = processor.process_points(points)
except CUDAOutOfMemoryError as e:
    print(f"Mémoire GPU insuffisante : {e}")
    # Réduire la taille du lot et réessayer
    processor.batch_size = processor.batch_size // 2
    result = processor.process_points(points)

except GPUComputeError as e:
    print(f"Échec du calcul GPU : {e}")
    # Repli sur traitement CPU
    cpu_processor = CPUProcessor()
    result = cpu_processor.process_points(points.cpu())
```

### Repli Automatique

```python
from ign_lidar.gpu import AdaptiveProcessor

# Processeur qui gère automatiquement le repli GPU/CPU
processor = AdaptiveProcessor(
    prefer_gpu=True,
    fallback_to_cpu=True,
    retry_on_oom=True
)

# Gère automatiquement les défaillances de périphérique
result = processor.process_points(points)
```

## Exemples d'Intégration

### Avec les Workflows Existants

```python
from ign_lidar import Processor
from ign_lidar.gpu import enable_gpu_acceleration

# Enable GPU acceleration for existing processor
processor = Processor()
enable_gpu_acceleration(
    processor,
    device="cuda:0",
    batch_size=20000
)

# Traiter normalement - l'accélération GPU est transparente
result = processor.process_tile("input.las")
```

### Pipeline GPU Personnalisé

```python
class CustomGPUPipeline:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.feature_extractor = GPUFeatureExtractor(device=device)
        self.classifier = GPUClassifier(device=device)
        self.mesh_generator = GPUMeshGenerator(device=device)

    def process(self, points_tensor):
        # Extraire les caractéristiques sur GPU
        features = self.feature_extractor.extract_features(points_tensor)

        # Classifier les points
        classifications = self.classifier.classify(features)

        # Générer des maillages pour les bâtiments
        building_mask = classifications == BuildingClass.BUILDING
        building_points = points_tensor[building_mask]
        meshes = self.mesh_generator.generate_meshes(building_points)

        return {
            'classifications': classifications,
            'features': features,
            'meshes': meshes
        }

# Utilisation
pipeline = CustomGPUPipeline(device="cuda:0")
result = pipeline.process(points_gpu)
```

## Bonnes Pratiques

### Gestion de la Mémoire

```python
# Utiliser des gestionnaires de contexte pour le nettoyage automatique
from ign_lidar.gpu import gpu_context

with gpu_context(device="cuda:0", memory_fraction=0.8) as ctx:
    processor = GPUProcessor(device=ctx.device)
    result = processor.process_points(points)
# Mémoire GPU automatiquement nettoyée
```

### Optimisation des Performances

```python
# Calcul de la taille de lot optimale
def calculate_optimal_batch_size(gpu_memory_gb, point_features=7):
    bytes_per_point = point_features * 4  # float32
    safety_factor = 0.8
    max_points = int(gpu_memory_gb * 1e9 * safety_factor / bytes_per_point)
    return min(max_points, 100000)  # Limiter à 100k points

batch_size = calculate_optimal_batch_size(8.0)  # GPU 8GB
```

### Récupération d'Erreurs

```python
def robust_gpu_processing(points, max_retries=3):
    for attempt in range(max_retries):
        try:
            return processor.process_points(points)
        except CUDAOutOfMemoryError:
            if attempt < max_retries - 1:
                # Réduire la taille du lot et vider le cache
                processor.batch_size //= 2
                torch.cuda.empty_cache()
                continue
            else:
                # Repli final sur CPU
                return cpu_processor.process_points(points.cpu())
```

## Documentation Connexe

- [Guide de Configuration GPU](../installation/gpu-setup.md)
- [Guide des Performances](../guides/performance.md)
- [Guide d'Accélération GPU](../guides/gpu-acceleration.md)
- [API Processor](./processor.md)
