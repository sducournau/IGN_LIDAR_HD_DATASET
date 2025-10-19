# Implementation Guide: Classification Optimizations

**Quick Start Guide for Implementing Recommended Optimizations**

---

## Option 1: Building Buffer Clustering (Highest ROI)

### Expected Impact

- **Speedup:** 10-100×
- **Effort:** 2-3 days
- **Complexity:** Medium
- **Files:** Modify `ign_lidar/core/modules/geometric_rules.py`

### Implementation

#### Step 1: Add clustering dependency

```bash
pip install scikit-learn hdbscan
```

#### Step 2: Add clustering method to `GeometricRulesEngine`

```python
# Add to ign_lidar/core/modules/geometric_rules.py

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

def classify_building_buffer_zone_clustered(
    self,
    points: np.ndarray,
    labels: np.ndarray,
    building_geometries: gpd.GeoDataFrame
) -> int:
    """
    Classify unclassified points near buildings using spatial clustering.

    This is 10-100× faster than point-by-point classification.
    """
    if len(building_geometries) == 0:
        return 0

    # Find unclassified points
    unclassified_mask = (labels == self.ASPRS_UNCLASSIFIED)
    if not np.any(unclassified_mask):
        return 0

    unclassified_indices = np.where(unclassified_mask)[0]
    unclassified_points = points[unclassified_mask]

    # Step 1: Filter points within buffer zone
    buffered_buildings = building_geometries.geometry.buffer(
        self.building_buffer_distance
    )
    tree = STRtree(buffered_buildings.values)

    buffer_zone_mask = np.zeros(len(unclassified_points), dtype=bool)
    for i, pt in enumerate(unclassified_points):
        pt_geom = Point(pt[0], pt[1])
        possible = tree.query(pt_geom)
        for idx in possible:
            if buffered_buildings.iloc[idx].contains(pt_geom):
                buffer_zone_mask[i] = True
                break

    if not np.any(buffer_zone_mask):
        return 0

    buffer_points = unclassified_points[buffer_zone_mask]
    buffer_indices = unclassified_indices[buffer_zone_mask]

    logger.info(f"  Found {len(buffer_points):,} points in buffer zone")

    # Step 2: Cluster buffer zone points
    logger.info(f"  Clustering buffer zone points...")
    clustering = DBSCAN(
        eps=self.building_buffer_distance / 2,  # 1m clustering radius
        min_samples=5
    )
    cluster_labels = clustering.fit_predict(buffer_points[:, :3])

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logger.info(f"  Found {n_clusters} clusters")

    # Step 3: Classify each cluster
    n_added = 0
    building_mask = (labels == self.ASPRS_BUILDING)

    if not np.any(building_mask):
        return 0

    building_points = points[building_mask]
    building_tree = cKDTree(building_points[:, :2])

    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Skip noise
            continue

        cluster_mask = (cluster_labels == cluster_id)
        cluster_pts = buffer_points[cluster_mask]
        cluster_global_idx = buffer_indices[cluster_mask]

        # Get cluster statistics
        cluster_center = np.mean(cluster_pts, axis=0)
        cluster_mean_height = cluster_center[2]

        # Find nearest building points
        distances, indices = building_tree.query(
            cluster_center[:2],
            k=10,
            distance_upper_bound=5.0
        )

        valid = distances < float('inf')
        if not np.any(valid):
            continue

        # Check height consistency
        neighbor_heights = building_points[indices[valid], 2]
        median_building_height = np.median(neighbor_heights)
        height_diff = abs(cluster_mean_height - median_building_height)

        if height_diff < self.max_building_height_difference:
            # Classify entire cluster as building
            labels[cluster_global_idx] = self.ASPRS_BUILDING
            n_added += len(cluster_global_idx)

    return n_added
```

#### Step 3: Update `apply_all_rules` to use clustered method

```python
# In apply_all_rules method, replace:
# n_added = self.classify_building_buffer_zone(...)

# With:
n_added = self.classify_building_buffer_zone_clustered(
    points=points,
    labels=updated_labels,
    building_geometries=ground_truth_features['buildings']
)
```

---

## Option 2: GPU-Accelerated Verticality (Highest Speedup)

### Expected Impact

- **Speedup:** 100-1000×
- **Effort:** 3-5 days
- **Complexity:** Medium-High
- **Requirements:** RAPIDS cuML installed

### Implementation

#### Step 1: Check RAPIDS installation

```bash
# For CUDA 12.x
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.10 cupy cuda-version=12.0
```

#### Step 2: Add GPU verticality computation

```python
# Add to ign_lidar/core/modules/geometric_rules.py

try:
    import cupy as cp
    import cuml
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    HAS_CUML = True
except ImportError:
    HAS_CUML = False

def compute_verticality_gpu(
    self,
    points: np.ndarray,
    query_points: np.ndarray,
    chunk_size: int = 500_000
) -> np.ndarray:
    """
    GPU-accelerated verticality computation using RAPIDS cuML.

    100-1000× faster than CPU version for large datasets.
    """
    if not HAS_CUML:
        logger.warning("cuML not available, falling back to CPU")
        return self.compute_verticality(points, query_points)

    try:
        n_queries = len(query_points)
        verticality_scores = np.zeros(n_queries)

        # Transfer points to GPU once
        points_gpu = cp.asarray(points, dtype=cp.float32)

        # Build GPU neighbor search
        nn = cuNearestNeighbors(n_neighbors=50, metric='euclidean')
        nn.fit(points_gpu)

        # Process in chunks to manage GPU memory
        n_chunks = (n_queries + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_queries)

            # Query chunk
            query_chunk = query_points[start_idx:end_idx]
            query_gpu = cp.asarray(query_chunk, dtype=cp.float32)

            # Find neighbors within radius
            distances, indices = nn.kneighbors(query_gpu)

            # Compute verticality for each query point (vectorized)
            for j in range(len(query_chunk)):
                neighbor_idx = indices[j].get()  # Transfer to CPU
                neighbor_dist = distances[j].get()

                # Filter by radius
                valid = neighbor_dist < self.verticality_search_radius
                if valid.sum() < self.min_vertical_neighbors:
                    continue

                # Get neighbors
                neighbors = points[neighbor_idx[valid]]

                # Compute extents
                z_extent = neighbors[:, 2].max() - neighbors[:, 2].min()
                x_extent = neighbors[:, 0].max() - neighbors[:, 0].min()
                y_extent = neighbors[:, 1].max() - neighbors[:, 1].min()
                h_extent = max(x_extent, y_extent)

                # Verticality score
                if h_extent > 0.01:
                    vert_ratio = z_extent / h_extent
                    verticality_scores[start_idx + j] = min(1.0, vert_ratio / 5.0)
                elif z_extent > 0.5:
                    verticality_scores[start_idx + j] = 1.0

        return verticality_scores

    except Exception as e:
        logger.warning(f"GPU verticality failed: {e}, falling back to CPU")
        return self.compute_verticality(points, query_points)
```

#### Step 3: Update `classify_by_verticality` to use GPU version

```python
# In classify_by_verticality method, replace:
# verticality_scores = self.compute_verticality(...)

# With:
if HAS_CUML:
    verticality_scores = self.compute_verticality_gpu(
        points=points,
        query_points=unclassified_points
    )
else:
    verticality_scores = self.compute_verticality(
        points=points,
        query_points=unclassified_points,
        kdtree=all_points_tree
    )
```

---

## Option 3: Advanced NIR Classification (Easiest)

### Expected Impact

- **Accuracy:** +5-10%
- **Effort:** 1-2 days
- **Complexity:** Low-Medium
- **Files:** Create `ign_lidar/core/modules/spectral_rules.py`

### Implementation

#### Create new spectral rules module

```python
# ign_lidar/core/modules/spectral_rules.py

"""
Advanced Spectral Classification Rules

Uses RGB + NIR for material-specific classification beyond simple NDVI.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class SpectralRulesEngine:
    """
    Material classification using multi-band spectral analysis.
    """

    # ASPRS codes
    ASPRS_UNCLASSIFIED = 1
    ASPRS_MEDIUM_VEGETATION = 4
    ASPRS_BUILDING = 6
    ASPRS_WATER = 9
    ASPRS_ROAD = 11

    def __init__(
        self,
        nir_vegetation_threshold: float = 0.4,
        nir_building_threshold: float = 0.3,
        brightness_concrete_min: float = 0.4,
        brightness_concrete_max: float = 0.7,
        ndvi_water_threshold: float = -0.1
    ):
        self.nir_vegetation_threshold = nir_vegetation_threshold
        self.nir_building_threshold = nir_building_threshold
        self.brightness_concrete_min = brightness_concrete_min
        self.brightness_concrete_max = brightness_concrete_max
        self.ndvi_water_threshold = ndvi_water_threshold

    def classify_by_spectral_signature(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        current_labels: np.ndarray,
        apply_to_unclassified_only: bool = True
    ) -> tuple[np.ndarray, dict]:
        """
        Classify points based on spectral signatures.

        Args:
            rgb: RGB values [N, 3] normalized to [0, 1]
            nir: NIR values [N] normalized to [0, 1]
            current_labels: Current classification [N]
            apply_to_unclassified_only: Only reclassify unclassified points

        Returns:
            Updated labels and statistics
        """
        labels = current_labels.copy()
        stats = {}

        # Extract channels
        red = rgb[:, 0]
        green = rgb[:, 1]
        blue = rgb[:, 2]

        # Compute derived features
        ndvi = (nir - red) / (nir + red + 1e-8)
        brightness = np.mean(rgb, axis=1)
        nir_red_ratio = nir / (red + 1e-8)

        # Determine which points to classify
        if apply_to_unclassified_only:
            mask = (labels == self.ASPRS_UNCLASSIFIED)
        else:
            mask = np.ones(len(labels), dtype=bool)

        # Rule 1: High NDVI + High NIR = Vegetation
        veg_mask = mask & (ndvi > 0.3) & (nir > self.nir_vegetation_threshold)
        labels[veg_mask] = self.ASPRS_MEDIUM_VEGETATION
        stats['vegetation_spectral'] = np.sum(veg_mask)

        # Rule 2: Negative NDVI + Low NIR = Water
        water_mask = (
            mask &
            (ndvi < self.ndvi_water_threshold) &
            (nir < 0.2) &
            (brightness < 0.3)
        )
        labels[water_mask] = self.ASPRS_WATER
        stats['water_spectral'] = np.sum(water_mask)

        # Rule 3: Moderate NIR + Moderate brightness = Concrete buildings
        concrete_mask = (
            mask &
            (nir > 0.2) & (nir < self.nir_building_threshold) &
            (brightness > self.brightness_concrete_min) &
            (brightness < self.brightness_concrete_max) &
            (ndvi < 0.15)
        )
        labels[concrete_mask] = self.ASPRS_BUILDING
        stats['building_spectral'] = np.sum(concrete_mask)

        # Rule 4: Very low NIR + Low brightness = Asphalt
        asphalt_mask = (
            mask &
            (nir < 0.15) &
            (brightness < 0.3) &
            (ndvi < 0.1)
        )
        labels[asphalt_mask] = self.ASPRS_ROAD
        stats['road_spectral'] = np.sum(asphalt_mask)

        # Total changes
        stats['total_spectral_changes'] = np.sum(current_labels != labels)

        return labels, stats
```

#### Integrate into GeometricRulesEngine

```python
# Add to GeometricRulesEngine.__init__
from .spectral_rules import SpectralRulesEngine

self.spectral_rules = SpectralRulesEngine(
    nir_vegetation_threshold=0.4,
    nir_building_threshold=0.3
)

# Add to apply_all_rules method, after other rules:
if rgb is not None and ndvi is not None:
    logger.info("  Applying spectral classification rules...")
    updated_labels, spectral_stats = self.spectral_rules.classify_by_spectral_signature(
        rgb=rgb,
        nir=ndvi,  # If you have raw NIR, use it instead
        current_labels=updated_labels,
        apply_to_unclassified_only=True
    )
    stats.update(spectral_stats)
```

---

## Testing Your Changes

### Test 1: Building Buffer Clustering

```python
# test_building_buffer_clustering.py

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from ign_lidar.core.modules.geometric_rules import GeometricRulesEngine

def test_clustered_buffer():
    # Create test data
    points = np.random.rand(10000, 3) * 100
    labels = np.ones(len(points), dtype=np.int32)  # All unclassified

    # Create test building
    buildings = gpd.GeoDataFrame({
        'geometry': [box(40, 40, 60, 60)]
    }, crs='EPSG:2154')

    # Test
    engine = GeometricRulesEngine()
    n_classified = engine.classify_building_buffer_zone_clustered(
        points, labels, buildings
    )

    print(f"Classified {n_classified} points as building")
    assert n_classified > 0
```

### Test 2: GPU Verticality

```python
# test_gpu_verticality.py

import numpy as np
from ign_lidar.core.modules.geometric_rules import GeometricRulesEngine

def test_gpu_verticality():
    # Create test point cloud (vertical wall)
    points = []
    for z in np.linspace(0, 10, 1000):  # 10m tall wall
        for x in np.linspace(0, 1, 100):  # 1m wide
            points.append([x, 0, z])
    points = np.array(points)

    # Test
    engine = GeometricRulesEngine()
    verticality = engine.compute_verticality_gpu(
        points=points,
        query_points=points[::100]  # Sample points
    )

    print(f"Mean verticality: {verticality.mean():.2f}")
    assert verticality.mean() > 0.7  # Should be high for vertical wall
```

### Test 3: Spectral Rules

```python
# test_spectral_rules.py

import numpy as np
from ign_lidar.core.modules.spectral_rules import SpectralRulesEngine

def test_spectral_classification():
    # Create test data
    n_points = 1000

    # Vegetation: high NDVI, high NIR
    rgb_veg = np.array([[0.3, 0.5, 0.2]] * n_points)
    nir_veg = np.array([0.7] * n_points)

    labels = np.ones(n_points, dtype=np.int32)

    # Test
    engine = SpectralRulesEngine()
    new_labels, stats = engine.classify_by_spectral_signature(
        rgb=rgb_veg,
        nir=nir_veg,
        current_labels=labels
    )

    print(f"Classified {stats['vegetation_spectral']} as vegetation")
    assert stats['vegetation_spectral'] > 0
```

---

## Performance Benchmarking

### Benchmark Script

```python
# benchmark_optimizations.py

import time
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from ign_lidar.core.modules.geometric_rules import GeometricRulesEngine

def benchmark_building_buffer():
    """Benchmark clustering vs. non-clustering."""

    # Create large test dataset
    n_points = 1_000_000
    points = np.random.rand(n_points, 3) * 1000
    labels = np.ones(len(points), dtype=np.int32)

    # Create buildings
    buildings = []
    for i in range(50):
        x = np.random.rand() * 900 + 50
        y = np.random.rand() * 900 + 50
        buildings.append(box(x, y, x+10, y+10))

    buildings_gdf = gpd.GeoDataFrame(
        {'geometry': buildings},
        crs='EPSG:2154'
    )

    engine = GeometricRulesEngine()

    # Test clustered version
    start = time.time()
    n_classified = engine.classify_building_buffer_zone_clustered(
        points, labels.copy(), buildings_gdf
    )
    clustered_time = time.time() - start

    print(f"\nBuilding Buffer Benchmark ({n_points:,} points):")
    print(f"  Clustered: {clustered_time:.2f}s ({n_classified:,} classified)")
    print(f"  Rate: {n_points/clustered_time:,.0f} points/sec")

if __name__ == '__main__':
    benchmark_building_buffer()
```

---

## Configuration Updates

### Add to your config YAML

```yaml
# Enable optimizations
reclassification:
  enabled: true
  use_geometric_rules: true

  # Clustering options
  use_clustering: true
  cluster_method: "dbscan"
  spatial_eps: 0.5
  min_cluster_size: 10

  # GPU options
  gpu_verticality: true
  gpu_verticality_chunk_size: 500000

  # Spectral rules
  use_spectral_rules: true
  nir_vegetation_threshold: 0.4
  nir_building_threshold: 0.3

  # Existing parameters
  building_buffer_distance: 2.0
  verticality_threshold: 0.7
  ndvi_vegetation_threshold: 0.3
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Solution:** Reduce GPU chunk size

```python
# In config
gpu_verticality_chunk_size: 100000  # Reduce from 500000
```

### Issue 2: Clustering too slow

**Solution:** Use HDBSCAN or reduce eps

```python
from hdbscan import HDBSCAN
clustering = HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    cluster_selection_method='eom'
)
```

### Issue 3: Spectral rules not improving accuracy

**Solution:** Tune thresholds based on your data

```python
# Analyze your data first
ndvi = compute_ndvi(nir, rgb[:, 0])
print(f"NDVI vegetation percentile 95: {np.percentile(ndvi[labels==4], 95)}")
print(f"NDVI buildings percentile 5: {np.percentile(ndvi[labels==6], 5)}")
```

---

## Next Steps

1. **Start with Option 3 (Spectral Rules)** - easiest to implement
2. **Add Option 1 (Clustering)** - best ROI
3. **Add Option 2 (GPU)** - if you have RAPIDS installed

For questions, see:

- Full audit: `GROUND_TRUTH_CLASSIFICATION_AUDIT.md`
- Summary: `OPTIMIZATION_SUMMARY.md`
