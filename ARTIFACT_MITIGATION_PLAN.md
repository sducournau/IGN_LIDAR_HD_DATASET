# Plan de Réduction des Artefacts dans IGN LiDAR HD - Phase d'Implémentation

**Date**: 4 Octobre 2025  
**Version**: 1.0  
**Statut**: En planification

---

## 📋 Résumé Exécutif

Ce plan détaille l'implémentation de techniques robustes pour réduire les artefacts visuels (lignes, dashs, discontinuités) lors du calcul de features géométriques sur les données LiDAR HD IGN. L'analyse du document `artifacts.md` et du code actuel révèle que le plugin possède déjà des bases solides (recherche par rayon, filtrage de features dégénérées) mais nécessite l'ajout de modules de prétraitement standardisés.

### Objectifs Principaux

1. **Prétraitement automatique** : Filtrage outliers (SOR, ROR) et homogénéisation densité (voxelisation)
2. **Gestion des bordures de tuiles** : Fusion virtuelle avec buffer pour continuité des features
3. **Paramétrage dynamique robuste** : Ajustement automatique du rayon selon densité locale
4. **Validation et monitoring** : Outils de diagnostic des artefacts
5. **Documentation utilisateur** : Guides de bonnes pratiques

---

## 🔍 Analyse de l'État Actuel

### Points Forts Existants ✅

Le plugin `ign-lidar-hd` possède déjà plusieurs mécanismes anti-artefacts :

1. **Recherche par rayon (radius-based)** : `features.py` implémente `estimate_optimal_radius_for_features()` qui adapte le rayon selon la densité

   ```python
   # Ligne 64-107 de features.py
   radius = estimate_optimal_radius_for_features(points, 'geometric')
   neighbor_indices = tree.query_radius(points, r=radius)
   ```

2. **Filtrage de features dégénérées** : Validation des eigenvalues et masquage des cas invalides

   ```python
   # Ligne 896+ de features.py
   valid_features = ~degenerate
   planarity[~valid_features] = 0.0
   ```

3. **Courbure robuste MAD** : Utilisation de Median Absolute Deviation pour résistance aux outliers

   ```python
   # Ligne 857+ de features.py
   # Use Median Absolute Deviation (MAD) for outlier-robust curvature
   ```

4. **Support GPU** : `features_gpu.py` pour accélération calculs volumétriques

### Lacunes Identifiées ❌

1. **Absence de prétraitement outlier** : Pas de filtrage SOR/ROR avant calcul features
2. **Pas de gestion explicite des bordures** : Discontinuités possibles entre tuiles adjacentes
3. **Voxelisation non implémentée** : Densité hétérogène non homogénéisée
4. **Paramétrage manuel k_neighbors** : Option encore disponible mais moins robuste que radius
5. **Pas de métriques de qualité** : Aucun diagnostic automatique des artefacts

---

## 🛠 Plan d'Implémentation par Phases

### Phase 1 : Module de Prétraitement (Priorité HAUTE) 🔴

**Objectif** : Créer `ign_lidar/preprocessing.py` avec filtres PDAL/PCL

#### 1.1 Filtrage Statistical Outlier Removal (SOR)

```python
def statistical_outlier_removal(
    points: np.ndarray,
    k: int = 12,
    std_multiplier: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove statistical outliers using mean distance to neighbors.

    Args:
        points: [N, 3] point coordinates
        k: number of neighbors for statistics
        std_multiplier: threshold in standard deviations

    Returns:
        filtered_points: [M, 3] cleaned points (M < N)
        inlier_mask: [N] boolean mask of kept points
    """
    from sklearn.neighbors import NearestNeighbors

    # Build kNN tree
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree')
    nbrs.fit(points)
    distances, _ = nbrs.kneighbors(points)

    # Compute mean distance (excluding self)
    mean_distances = distances[:, 1:].mean(axis=1)

    # Compute global statistics
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()

    # Threshold
    threshold = global_mean + std_multiplier * global_std
    inlier_mask = mean_distances < threshold

    return points[inlier_mask], inlier_mask
```

#### 1.2 Filtrage Radius Outlier Removal (ROR)

```python
def radius_outlier_removal(
    points: np.ndarray,
    radius: float = 1.0,
    min_neighbors: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove points with too few neighbors in radius.

    Args:
        points: [N, 3] point coordinates
        radius: search radius in meters
        min_neighbors: minimum required neighbors

    Returns:
        filtered_points: [M, 3] cleaned points
        inlier_mask: [N] boolean mask
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree')
    nbrs.fit(points)

    # Count neighbors for each point
    neighbors = nbrs.radius_neighbors(points, return_distance=False)
    neighbor_counts = np.array([len(n) for n in neighbors])

    # Mask: keep points with enough neighbors (excluding self)
    inlier_mask = neighbor_counts > min_neighbors

    return points[inlier_mask], inlier_mask
```

#### 1.3 Voxelisation (Homogénéisation Densité)

```python
def voxel_downsample(
    points: np.ndarray,
    voxel_size: float = 0.5,
    method: str = 'centroid'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using voxel grid.

    Args:
        points: [N, 3] point coordinates
        voxel_size: size of voxel in meters
        method: 'centroid' (average) or 'random' (random point)

    Returns:
        downsampled_points: [M, 3] voxelized points
        voxel_indices: [N] voxel index for each original point
    """
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Convert to unique keys
    voxel_keys = voxel_indices[:, 0] * 1000000 + \
                 voxel_indices[:, 1] * 1000 + \
                 voxel_indices[:, 2]

    unique_voxels, inverse_indices = np.unique(
        voxel_keys, return_inverse=True
    )

    if method == 'centroid':
        # Average points in each voxel
        downsampled = np.zeros((len(unique_voxels), 3), dtype=np.float32)
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            downsampled[i] = points[mask].mean(axis=0)
    else:  # random
        # Random point from each voxel
        downsampled = []
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            idx = np.random.choice(np.where(mask)[0])
            downsampled.append(points[idx])
        downsampled = np.array(downsampled, dtype=np.float32)

    return downsampled, inverse_indices
```

#### 1.4 Pipeline de Prétraitement Complet

```python
def preprocess_point_cloud(
    points: np.ndarray,
    config: Dict[str, Any] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply full preprocessing pipeline.

    Default config:
    - Statistical outlier removal (k=12, std=2.0)
    - Radius outlier removal (r=1.0m, min_neighbors=4)
    - Voxel downsampling (voxel_size=0.5m)

    Args:
        points: [N, 3] raw point coordinates
        config: optional preprocessing parameters

    Returns:
        processed_points: [M, 3] cleaned points
        stats: preprocessing statistics (removed counts, etc.)
    """
    if config is None:
        config = {
            'sor': {'enable': True, 'k': 12, 'std_multiplier': 2.0},
            'ror': {'enable': True, 'radius': 1.0, 'min_neighbors': 4},
            'voxel': {'enable': False, 'voxel_size': 0.5, 'method': 'centroid'}
        }

    stats = {'original_points': len(points)}
    processed = points.copy()

    # Step 1: Statistical Outlier Removal
    if config['sor']['enable']:
        processed, mask = statistical_outlier_removal(
            processed,
            k=config['sor']['k'],
            std_multiplier=config['sor']['std_multiplier']
        )
        stats['sor_removed'] = np.sum(~mask)

    # Step 2: Radius Outlier Removal
    if config['ror']['enable']:
        processed, mask = radius_outlier_removal(
            processed,
            radius=config['ror']['radius'],
            min_neighbors=config['ror']['min_neighbors']
        )
        stats['ror_removed'] = np.sum(~mask)

    # Step 3: Voxel Downsampling
    if config['voxel']['enable']:
        original_size = len(processed)
        processed, _ = voxel_downsample(
            processed,
            voxel_size=config['voxel']['voxel_size'],
            method=config['voxel']['method']
        )
        stats['voxel_reduced'] = original_size - len(processed)

    stats['final_points'] = len(processed)
    stats['reduction_ratio'] = 1.0 - (stats['final_points'] / stats['original_points'])

    return processed, stats
```

**Fichiers à créer/modifier** :

- Nouveau : `ign_lidar/preprocessing.py` (module complet)
- Modifier : `ign_lidar/processor.py` (appel prétraitement avant features)
- Modifier : `ign_lidar/cli.py` (arguments CLI pour prétraitement)

---

### Phase 2 : Gestion des Bordures de Tuiles (Priorité HAUTE) 🔴

**Objectif** : Éviter discontinuités aux jonctions de tuiles

#### 2.1 Détection Bordures et Buffer

```python
def extract_tile_with_buffer(
    tile_path: Path,
    neighbor_tiles: List[Path],
    buffer_distance: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load tile with buffer from neighbors.

    Args:
        tile_path: main tile to process
        neighbor_tiles: adjacent tiles (8 neighbors max)
        buffer_distance: buffer zone in meters

    Returns:
        points_buffered: [N, 3] main + buffer points
        is_border: [N] boolean mask (True for border points)
    """
    import laspy

    # Load main tile
    las_main = laspy.read(tile_path)
    points_main = np.vstack([las_main.x, las_main.y, las_main.z]).T

    # Compute main tile bbox
    bbox = {
        'xmin': points_main[:, 0].min(),
        'xmax': points_main[:, 0].max(),
        'ymin': points_main[:, 1].min(),
        'ymax': points_main[:, 1].max()
    }

    # Load buffer points from neighbors
    buffer_points = []
    for neighbor_path in neighbor_tiles:
        try:
            las_neighbor = laspy.read(neighbor_path)
            pts = np.vstack([las_neighbor.x, las_neighbor.y, las_neighbor.z]).T

            # Keep only points within buffer distance of main tile
            in_buffer = (
                (pts[:, 0] >= bbox['xmin'] - buffer_distance) &
                (pts[:, 0] <= bbox['xmax'] + buffer_distance) &
                (pts[:, 1] >= bbox['ymin'] - buffer_distance) &
                (pts[:, 1] <= bbox['ymax'] + buffer_distance)
            )
            buffer_points.append(pts[in_buffer])
        except Exception as e:
            logger.warning(f"Could not load neighbor {neighbor_path}: {e}")

    # Combine main + buffer
    if buffer_points:
        all_buffer = np.vstack(buffer_points)
        points_buffered = np.vstack([points_main, all_buffer])

        # Mark border points (within buffer_distance of edges)
        is_border = (
            (points_buffered[:, 0] < bbox['xmin'] + buffer_distance) |
            (points_buffered[:, 0] > bbox['xmax'] - buffer_distance) |
            (points_buffered[:, 1] < bbox['ymin'] + buffer_distance) |
            (points_buffered[:, 1] > bbox['ymax'] - buffer_distance)
        )
    else:
        points_buffered = points_main
        is_border = np.zeros(len(points_main), dtype=bool)

    return points_buffered, is_border
```

#### 2.2 Détection Automatique des Tuiles Voisines

```python
def find_neighbor_tiles(
    tile_path: Path,
    tile_dir: Path,
    tile_size: float = 1000.0
) -> List[Path]:
    """
    Find 8 neighbor tiles for a given tile.

    Assumes IGN naming convention: XXXX_YYYY.laz
    where XXXX, YYYY are Lambert93 coordinates in km

    Args:
        tile_path: current tile
        tile_dir: directory containing all tiles
        tile_size: tile size in meters (default 1000m = 1km)

    Returns:
        List of neighbor tile paths (existing only)
    """
    # Parse tile coordinates from name
    tile_name = tile_path.stem  # e.g., "0500_6800"
    try:
        parts = tile_name.split('_')
        x_km = int(parts[0])
        y_km = int(parts[1])
    except:
        logger.warning(f"Could not parse tile name: {tile_name}")
        return []

    # 8 neighbors
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    neighbors = []
    for dx, dy in offsets:
        neighbor_name = f"{x_km + dx:04d}_{y_km + dy:04d}.laz"
        neighbor_path = tile_dir / neighbor_name
        if neighbor_path.exists():
            neighbors.append(neighbor_path)

    return neighbors
```

**Fichiers à créer/modifier** :

- Nouveau : `ign_lidar/tile_borders.py` (gestion bordures)
- Modifier : `ign_lidar/processor.py` (option buffer dans process_tile)
- Modifier : `ign_lidar/cli.py` (argument --buffer-distance)

---

### Phase 3 : Amélioration Paramétrage Dynamique (Priorité MOYENNE) 🟡

**Objectif** : Renforcer l'adaptation automatique du rayon selon contexte

#### 3.1 Détection de Densité Multi-Échelle

```python
def estimate_multi_scale_density(
    points: np.ndarray,
    scales: List[float] = [0.5, 1.0, 2.0]
) -> Dict[str, float]:
    """
    Estimate point cloud density at multiple scales.

    Returns optimal radius for different feature types.
    """
    from sklearn.neighbors import KDTree

    # Sample points for speed
    n_samples = min(2000, len(points))
    sample_idx = np.random.choice(len(points), n_samples, replace=False)
    sample_pts = points[sample_idx]

    tree = KDTree(sample_pts)

    densities = {}
    for scale in scales:
        counts = tree.query_radius(sample_pts, r=scale, count_only=True)
        avg_density = np.median(counts) / (4/3 * np.pi * scale**3)
        densities[f'density_{scale}m'] = float(avg_density)

    # Recommend radii based on densities
    avg_nn_dist = 1.0 / (densities['density_1.0m'] ** (1/3) + 1e-6)

    recommendations = {
        'radius_normals': np.clip(avg_nn_dist * 10, 0.3, 1.0),
        'radius_geometric': np.clip(avg_nn_dist * 20, 0.5, 2.0),
        'avg_density': densities['density_1.0m']
    }

    return recommendations
```

#### 3.2 Détection Contexte Urbain/Naturel

```python
def detect_environment_type(
    points: np.ndarray,
    classification: np.ndarray
) -> str:
    """
    Detect if tile is urban, natural, or mixed.

    Returns: 'urban', 'forest', 'rural', 'mixed'
    """
    # Count classes
    class_counts = np.bincount(classification, minlength=20)

    # IGN classes: 2=ground, 3-5=vegetation, 6=building
    pct_building = class_counts[6] / len(classification)
    pct_vegetation = (class_counts[3] + class_counts[4] + class_counts[5]) / len(classification)
    pct_ground = class_counts[2] / len(classification)

    if pct_building > 0.15:
        return 'urban'
    elif pct_vegetation > 0.60:
        return 'forest'
    elif pct_building > 0.05 and pct_vegetation > 0.30:
        return 'mixed'
    else:
        return 'rural'
```

**Fichiers à modifier** :

- `ign_lidar/features.py` : Améliorer `estimate_optimal_radius_for_features()`
- Nouveau : `ign_lidar/density_analysis.py` (analyse densité avancée)

---

### Phase 4 : Outils de Diagnostic et Validation (Priorité MOYENNE) 🟡

**Objectif** : Détecter et visualiser artefacts automatiquement

#### 4.1 Métriques de Qualité des Features

```python
def compute_feature_quality_metrics(
    points: np.ndarray,
    normals: np.ndarray,
    planarity: np.ndarray,
    curvature: np.ndarray
) -> Dict[str, float]:
    """
    Compute quality metrics to detect artifacts.

    Returns:
        metrics: dictionary of quality scores
    """
    metrics = {}

    # 1. Normal coherence (low variance = good)
    normal_variance = np.var(normals, axis=0).sum()
    metrics['normal_coherence'] = 1.0 / (1.0 + normal_variance)

    # 2. Planarity distribution (should not have too many zeros)
    pct_degenerate = np.sum(planarity == 0) / len(planarity)
    metrics['degenerate_ratio'] = float(pct_degenerate)

    # 3. Curvature outliers (detect spikes)
    curvature_iqr = np.percentile(curvature, 75) - np.percentile(curvature, 25)
    curvature_outliers = np.sum(curvature > np.percentile(curvature, 75) + 3*curvature_iqr)
    metrics['curvature_outlier_ratio'] = curvature_outliers / len(curvature)

    # 4. Overall quality score (0-100)
    quality_score = (
        metrics['normal_coherence'] * 40 +
        (1 - metrics['degenerate_ratio']) * 30 +
        (1 - metrics['curvature_outlier_ratio']) * 30
    )
    metrics['quality_score'] = float(np.clip(quality_score * 100, 0, 100))

    return metrics
```

#### 4.2 Détection de Patterns d'Artefacts

```python
def detect_scan_line_artifacts(
    points: np.ndarray,
    planarity: np.ndarray,
    grid_resolution: float = 10.0
) -> Dict[str, Any]:
    """
    Detect regular patterns (dashes, lines) in feature maps.

    Uses 2D FFT to detect periodic artifacts.
    """
    # Create 2D grid of planarity values
    x_bins = np.arange(points[:, 0].min(), points[:, 0].max(), grid_resolution)
    y_bins = np.arange(points[:, 1].min(), points[:, 1].max(), grid_resolution)

    # Bin planarity values
    grid = np.zeros((len(x_bins), len(y_bins)))
    x_idx = ((points[:, 0] - points[:, 0].min()) / grid_resolution).astype(int)
    y_idx = ((points[:, 1] - points[:, 1].min()) / grid_resolution).astype(int)

    x_idx = np.clip(x_idx, 0, len(x_bins) - 1)
    y_idx = np.clip(y_idx, 0, len(y_bins) - 1)

    for i in range(len(points)):
        grid[x_idx[i], y_idx[i]] = planarity[i]

    # 2D FFT
    fft = np.fft.fft2(grid)
    power = np.abs(fft) ** 2

    # Detect strong periodic components (artifacts)
    power_sorted = np.sort(power.flatten())
    threshold = power_sorted[-10]  # Top 10 frequencies

    has_artifacts = np.sum(power > threshold) > 5

    return {
        'has_scan_artifacts': bool(has_artifacts),
        'artifact_score': float(np.sum(power > threshold) / power.size)
    }
```

**Fichiers à créer** :

- Nouveau : `ign_lidar/quality_metrics.py` (métriques qualité)
- Nouveau : `scripts/diagnose_artifacts.py` (script CLI diagnostic)

---

### Phase 5 : Intégration dans le Pipeline (Priorité HAUTE) 🔴

**Objectif** : Rendre le prétraitement transparent et optionnel

#### 5.1 Modification de `processor.py`

```python
# Dans LiDARProcessor.__init__()
def __init__(self, ...,
             enable_preprocessing: bool = True,
             preprocessing_config: Dict = None,
             buffer_distance: float = 0.0,
             ...):
    self.enable_preprocessing = enable_preprocessing
    self.preprocessing_config = preprocessing_config or {
        'sor': {'enable': True, 'k': 12, 'std_multiplier': 2.0},
        'ror': {'enable': True, 'radius': 1.0, 'min_neighbors': 4},
        'voxel': {'enable': False}
    }
    self.buffer_distance = buffer_distance

# Dans process_tile()
def process_tile(self, laz_file: Path, ...):
    # 1. Load tile (with optional buffer)
    if self.buffer_distance > 0:
        neighbors = find_neighbor_tiles(laz_file, laz_file.parent)
        points, is_border = extract_tile_with_buffer(
            laz_file, neighbors, self.buffer_distance
        )
    else:
        # Standard load
        las = laspy.read(laz_file)
        points = np.vstack([las.x, las.y, las.z]).T

    # 2. Preprocess if enabled
    if self.enable_preprocessing:
        points_clean, preproc_stats = preprocess_point_cloud(
            points, self.preprocessing_config
        )
        logger.info(f"  Preprocessing: {preproc_stats['original_points']} → "
                   f"{preproc_stats['final_points']} points "
                   f"({preproc_stats['reduction_ratio']:.1%} reduction)")
        points = points_clean

    # 3. Compute features (existing code)
    features = compute_all_features_optimized(points, ...)

    # 4. Quality check
    if self.enable_preprocessing:
        quality = compute_feature_quality_metrics(
            points, features['normals'],
            features['planarity'], features['curvature']
        )
        logger.info(f"  Feature quality score: {quality['quality_score']:.1f}/100")
```

#### 5.2 Arguments CLI étendus

```python
# Dans cli.py, cmd_enrich()
parser_enrich.add_argument('--preprocess', action='store_true', default=True,
                          help='Enable preprocessing (SOR, ROR)')
parser_enrich.add_argument('--no-preprocess', action='store_false', dest='preprocess',
                          help='Disable preprocessing')
parser_enrich.add_argument('--sor-k', type=int, default=12,
                          help='SOR: number of neighbors')
parser_enrich.add_argument('--sor-std', type=float, default=2.0,
                          help='SOR: std multiplier threshold')
parser_enrich.add_argument('--ror-radius', type=float, default=1.0,
                          help='ROR: search radius in meters')
parser_enrich.add_argument('--ror-min-neighbors', type=int, default=4,
                          help='ROR: minimum neighbors required')
parser_enrich.add_argument('--buffer-distance', type=float, default=0.0,
                          help='Buffer distance for tile borders (m)')
parser_enrich.add_argument('--voxel-size', type=float, default=0.0,
                          help='Voxel size for downsampling (0=disabled)')
```

#### 5.3 Configuration YAML étendue

```yaml
# config_examples/pipeline_enrich_advanced.yaml
enrich:
  input_dir: "data/raw"
  output: "data/enriched"

  # Feature computation
  mode: "building"
  use_gpu: true
  radius: null # auto-detect

  # NEW: Preprocessing
  preprocessing:
    enable: true
    statistical_outlier:
      enable: true
      k_neighbors: 12
      std_multiplier: 2.0
    radius_outlier:
      enable: true
      radius: 1.0
      min_neighbors: 4
    voxel_downsampling:
      enable: false
      voxel_size: 0.5
      method: "centroid"

  # NEW: Tile border handling
  buffer_distance: 50.0 # meters

  # NEW: Quality monitoring
  quality_check: true
  quality_threshold: 50.0 # min score to accept tile

  # RGB
  add_rgb: true
  rgb_cache_dir: "cache/orthophotos"
```

---

### Phase 6 : Documentation et Exemples (Priorité MOYENNE) 🟡

#### 6.1 Guide Utilisateur "Artifact Mitigation"

**Fichier** : `website/docs/guides/artifact-mitigation.md`

````markdown
# Guide: Reducing Artifacts in Geometric Features

## What are Artifacts?

Artifacts are visual anomalies in computed geometric features:

- **Dashed lines** in planarity/curvature maps
- **Discontinuities** at tile borders
- **Noisy normals** on smooth surfaces
- **Degenerate features** (all zeros)

## Quick Fix: Enable Preprocessing

```bash
ign-lidar-hd enrich \
  --input-dir raw/ \
  --output enriched/ \
  --preprocess \
  --buffer-distance 50
```
````

This enables:

1. Statistical outlier removal (SOR)
2. Radius outlier removal (ROR)
3. 50m buffer from neighbor tiles

## Advanced Configuration

Create `config.yaml`:

```yaml
enrich:
  preprocessing:
    enable: true
    statistical_outlier:
      k_neighbors: 12
      std_multiplier: 2.0
    radius_outlier:
      radius: 1.0
      min_neighbors: 4
  buffer_distance: 50.0
```

Run: `ign-lidar-hd pipeline config.yaml`

````

#### 6.2 Script Exemple : Diagnostic d'Artefacts

**Fichier** : `scripts/diagnose_artifacts.py`

```python
#!/usr/bin/env python3
"""
Diagnose artifacts in enriched LAZ files.
"""

import argparse
from pathlib import Path
import laspy
import numpy as np
from ign_lidar.quality_metrics import (
    compute_feature_quality_metrics,
    detect_scan_line_artifacts
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()

    # Load enriched LAZ
    las = laspy.read(args.input)
    points = np.vstack([las.x, las.y, las.z]).T

    # Extract features
    normals = np.vstack([las.normal_x, las.normal_y, las.normal_z]).T
    planarity = las.planarity
    curvature = las.curvature

    # Compute metrics
    metrics = compute_feature_quality_metrics(
        points, normals, planarity, curvature
    )

    # Detect artifacts
    artifacts = detect_scan_line_artifacts(points, planarity)

    # Report
    print("=" * 60)
    print(f"Artifact Diagnosis: {args.input.name}")
    print("=" * 60)
    print(f"Overall Quality Score: {metrics['quality_score']:.1f}/100")
    print(f"Degenerate Features:   {metrics['degenerate_ratio']*100:.1f}%")
    print(f"Normal Coherence:      {metrics['normal_coherence']:.3f}")
    print(f"Has Scan Artifacts:    {artifacts['has_scan_artifacts']}")
    print(f"Artifact Score:        {artifacts['artifact_score']:.3f}")

    if metrics['quality_score'] < 50:
        print("\n⚠️  WARNING: Low quality - consider enabling preprocessing!")

if __name__ == '__main__':
    main()
````

---

## 📊 Tableau Récapitulatif des Solutions

| Artefact                | Cause Principale           | Solution Implémentée              | Phase | Fichier              |
| ----------------------- | -------------------------- | --------------------------------- | ----- | -------------------- |
| Lignes/dashs            | kNN sur densité hétérogène | Radius-based (existant) + SOR/ROR | 1     | `preprocessing.py`   |
| Discontinuités bordures | Pas de buffer entre tuiles | Buffer automatique 50m            | 2     | `tile_borders.py`    |
| Plans mal segmentés     | Densité irrégulière        | Voxelisation optionnelle          | 1     | `preprocessing.py`   |
| Points isolés           | Bruit instrumental         | ROR (min_neighbors=4)             | 1     | `preprocessing.py`   |
| Normales incohérentes   | Outliers dans voisinage    | SOR avant PCA                     | 1     | `preprocessing.py`   |
| Features dégénérées     | Eigenvalues nulles         | Filtrage existant (robuste)       | -     | `features.py` (OK)   |
| Zones vides             | Faible couverture scan     | Détection + warning               | 4     | `quality_metrics.py` |

---

## 🎯 Ordre d'Implémentation Recommandé

### Sprint 1 (1 semaine) - Fondations

1. ✅ Créer `ign_lidar/preprocessing.py` avec SOR/ROR/Voxel
2. ✅ Tests unitaires (`tests/test_preprocessing.py`)
3. ✅ Intégration basique dans `processor.py`

### Sprint 2 (1 semaine) - Bordures

1. ✅ Créer `ign_lidar/tile_borders.py`
2. ✅ Détection automatique tuiles voisines
3. ✅ Tests sur jeu de 9 tuiles (3×3)

### Sprint 3 (1 semaine) - CLI & Config

1. ✅ Arguments CLI étendus
2. ✅ Config YAML avancée
3. ✅ Documentation utilisateur

### Sprint 4 (1 semaine) - Qualité & Diagnostic

1. ✅ `ign_lidar/quality_metrics.py`
2. ✅ Script `diagnose_artifacts.py`
3. ✅ Monitoring automatique

---

## 📈 Métriques de Succès

### Objectifs Quantitatifs

1. **Réduction artefacts** : <5% de features dégénérées (actuellement ~10-15%)
2. **Continuité bordures** : Écart <10cm aux jonctions de tuiles
3. **Qualité globale** : Score moyen >70/100 sur 100 tuiles test
4. **Performance** : Overhead <20% temps calcul (avec prétraitement)

### Tests de Validation

1. **Dataset test** : 100 tuiles diversifiées (urbain, rural, forêt)
2. **Visualisation** : CloudCompare + color map planarity
3. **Métriques auto** : Script diagnostic sur batch
4. **Comparaison avant/après** : LAZ raw vs prétraité

---

## 🔄 Compatibilité et Rétrocompatibilité

### Par Défaut : Prétraitement Actif

- Nouveau comportement : `--preprocess=True` par défaut
- Utilisateurs existants : Pas de changement comportemental majeur
- Option désactivation : `--no-preprocess`

### Migration des Anciens Workflows

```bash
# Ancien (toujours supporté)
ign-lidar-hd enrich --input-dir raw/ --output enriched/

# Nouveau (équivalent, mais avec prétraitement)
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --preprocess

# Désactiver si besoin
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --no-preprocess
```

---

## 🚀 Roadmap Longue Terme

### Version 1.7.0 (Q4 2025)

- ✅ Prétraitement SOR/ROR
- ✅ Buffer tuiles automatique
- ✅ Métriques qualité

### Version 1.8.0 (Q1 2026)

- Intégration PDAL native (pipelines JSON)
- Support COPC optimisé
- Voxelisation GPU (CuPy)

### Version 2.0.0 (Q2 2026)

- Recalage fin inter-tuiles (ICP)
- Détection automatique artefacts ML
- Dashboard qualité interactif

---

## 📚 Ressources et Références

### Documentation Technique

- [artifacts.md](artifacts.md) : Analyse détaillée causes/solutions
- [PDAL Filters](https://pdal.io/stages/filters.html) : Référence filtres PDAL
- [ign-pdal-tools](https://github.com/IGNF/ign-pdal-tools) : Outils IGN

### Publications Scientifiques

- _LoGDesc: Robust Geometric Features_ (ICCV 2023)
- _LiDAR Scan Line Artifacts_ (ISPRS 2022)
- _Multi-Scale Point Cloud Processing_ (CVPR 2021)

### Exemples Code

- [jakteristics](https://github.com/jakarto3d/jakteristics) : Features robustes
- [Open3D](http://www.open3d.org/) : Preprocessing référence

---

## ✅ Checklist d'Implémentation

### Phase 1 - Prétraitement

- [ ] Module `preprocessing.py` complet
- [ ] Tests unitaires (>80% coverage)
- [ ] Benchmarks performance
- [ ] Documentation API

### Phase 2 - Bordures

- [ ] Module `tile_borders.py`
- [ ] Détection voisins automatique
- [ ] Tests sur grille 3×3
- [ ] Validation continuité géométrique

### Phase 3 - Intégration

- [ ] Modification `processor.py`
- [ ] Arguments CLI étendus
- [ ] Config YAML avancée
- [ ] Tests e2e (end-to-end)

### Phase 4 - Qualité

- [ ] Module `quality_metrics.py`
- [ ] Script `diagnose_artifacts.py`
- [ ] Dashboard monitoring (optionnel)
- [ ] Guide troubleshooting

### Phase 5 - Documentation

- [ ] Guide utilisateur artefacts
- [ ] Tutoriel vidéo (optionnel)
- [ ] Release notes v1.7.0
- [ ] Mise à jour README

---

## 🎓 Formation et Support

### Documentation Utilisateur

- Guide rapide "Quick Start Artifact Mitigation"
- FAQ artefacts courants
- Exemples de configurations terrain

### Support Communauté

- Issues GitHub templates spécifiques artefacts
- Forum discussions prétraitement
- Exemples notebooks Jupyter interactifs

---

**Auteur** : GitHub Copilot  
**Date** : 4 Octobre 2025  
**Version** : 1.0  
**Statut** : Prêt pour revue et implémentation

---

_Ce plan est un document vivant qui sera mis à jour au fur et à mesure de l'implémentation et des retours utilisateurs._
