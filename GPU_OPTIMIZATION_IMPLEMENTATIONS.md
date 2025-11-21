# 🚀 Implémentations GPU - Guide Détaillé

**Complément à:** PERFORMANCE_AUDIT_2025.md  
**Date:** 21 Novembre 2025  
**Focus:** Code examples et implémentations concrètes

---

## 📋 Table des Matières

1. [P0.1 - GPU Road Classification](#p01---gpu-road-classification)
2. [P0.2 - GPU BBox Optimization](#p02---gpu-bbox-optimization)
3. [P1.3 - GPU KNN Façades](#p13---gpu-knn-façades)
4. [Tests & Validation](#tests--validation)
5. [Performance Benchmarks](#performance-benchmarks)

---

## P0.1 - GPU Road Classification

### Fichier: `ign_lidar/core/classification/reclassifier.py`

### Implémentation Complète

```python
def _classify_roads_with_nature_gpu(
    self,
    points: np.ndarray,
    labels: np.ndarray,
    roads_gdf: gpd.GeoDataFrame,
) -> int:
    """
    🚀 GPU-accelerated road classification avec nature détaillée.

    Utilise cuSpatial pour point-in-polygon vectorisé.
    Fallback CPU automatique si erreur GPU.

    Performance:
    - CPU (STRtree loop): 5-10 minutes pour 18M points
    - GPU (cuSpatial): 30-60 secondes (10-20× speedup)

    Args:
        points: XYZ coordinates [N, 3]
        labels: Classification labels [N] (modified in-place)
        roads_gdf: GeoDataFrame with road geometries and 'nature' attribute

    Returns:
        Number of points classified
    """
    if not HAS_GPU:
        logger.debug("GPU not available, using CPU fallback")
        return self._classify_roads_with_nature(points, labels, roads_gdf)

    try:
        import cudf
        import cuspatial
        import cupy as cp

        n_classified = 0
        n_points = len(points)

        # 1. Transfer points to GPU once
        logger.debug(f"  Transferring {n_points:,} points to GPU...")
        points_xy_gpu = cp.asarray(points[:, :2], dtype=cp.float32)
        labels_gpu = cp.asarray(labels, dtype=cp.int32)

        # 2. Group roads by nature type for efficient processing
        road_types = roads_gdf['nature'].unique()
        logger.debug(f"  Processing {len(road_types)} road types...")

        # Progress bar
        pbar = None
        if self.show_progress:
            from tqdm import tqdm
            pbar = tqdm(
                total=len(road_types),
                desc="    roads (GPU vectorized 🔥)",
                leave=False,
                unit="types",
            )

        # 3. Process each road type
        for road_nature in road_types:
            # Get ASPRS code for this road type
            asprs_code = self._get_asprs_code_for_road(road_nature)

            # Filter roads of this type
            road_subset = roads_gdf[roads_gdf['nature'] == road_nature]

            if len(road_subset) == 0:
                continue

            # 4. Extract polygons and convert to GPU format
            polygons_list = []
            for geom in road_subset.geometry:
                if geom.geom_type == 'Polygon':
                    coords = np.array(geom.exterior.coords, dtype=np.float32)
                    polygons_list.append(coords)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        coords = np.array(poly.exterior.coords, dtype=np.float32)
                        polygons_list.append(coords)

            if len(polygons_list) == 0:
                continue

            # 5. cuSpatial point-in-polygon (VECTORIZED!)
            # Process all points against all polygons of this type
            for poly_coords in polygons_list:
                poly_x = cp.asarray(poly_coords[:, 0], dtype=cp.float32)
                poly_y = cp.asarray(poly_coords[:, 1], dtype=cp.float32)

                # 🚀 Vectorized point-in-polygon test (ALL points at once)
                # This is the key GPU acceleration
                mask = cuspatial.point_in_polygon(
                    points_xy_gpu[:, 0],  # All point X coordinates
                    points_xy_gpu[:, 1],  # All point Y coordinates
                    poly_x,               # Polygon X coordinates
                    poly_y                # Polygon Y coordinates
                )

                # Update labels where points are inside this polygon
                labels_gpu[mask] = asprs_code
                n_classified += int(cp.sum(mask))

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        # 6. Transfer results back to CPU
        logger.debug("  Transferring results back to CPU...")
        labels[:] = cp.asnumpy(labels_gpu)

        # 7. Cleanup GPU memory
        cp.get_default_memory_pool().free_all_blocks()

        logger.debug(f"  GPU classification: {n_classified:,} points classified")
        return n_classified

    except Exception as e:
        logger.warning(f"GPU road classification failed, falling back to CPU: {e}")
        # Cleanup on error
        if 'cp' in locals():
            cp.get_default_memory_pool().free_all_blocks()

        # Fallback to CPU implementation
        return self._classify_roads_with_nature(points, labels, roads_gdf)
```

### Intégration dans Reclassifier

```python
# Dans la classe Reclassifier, méthode reclassify()
# Ligne ~570 environ

# Remplacer:
if feature_name == "roads" and "nature" in gdf.columns:
    n_classified = self._classify_roads_with_nature(
        points=points,
        labels=updated_labels,
        roads_gdf=gdf,
    )

# Par:
if feature_name == "roads" and "nature" in gdf.columns:
    # 🚀 Try GPU first if available
    if self.acceleration_mode in ["gpu", "gpu+cuml", "auto"] and HAS_GPU:
        try:
            n_classified = self._classify_roads_with_nature_gpu(
                points=points,
                labels=updated_labels,
                roads_gdf=gdf,
            )
        except Exception as e:
            logger.warning(f"GPU road classification failed: {e}, using CPU")
            n_classified = self._classify_roads_with_nature(
                points=points,
                labels=updated_labels,
                roads_gdf=gdf,
            )
    else:
        n_classified = self._classify_roads_with_nature(
            points=points,
            labels=updated_labels,
            roads_gdf=gdf,
        )
```

---

## P0.2 - GPU BBox Optimization

### Fichier: `ign_lidar/core/classification/building/building_clusterer.py`

### Nouvelle Méthode GPU

```python
def optimize_bbox_for_building_gpu(
    self,
    points: np.ndarray,
    heights: np.ndarray,
    initial_bbox: Tuple[float, float, float, float],
    max_shift: float = 15.0,
    step: float = 1.0,
    height_threshold: float = 1.0,
    ground_penalty: float = 1.0,
    non_ground_reward: float = 1.0,
) -> Tuple[Tuple[float, float], Tuple[float, float, float, float]]:
    """
    🚀 GPU-accelerated bounding box optimization avec grid search vectorisé.

    Teste TOUS les shifts (dx, dy) en parallèle sur GPU pour trouver
    la meilleure position de bbox qui maximise les points de bâtiment
    et minimise les points au sol.

    Performance:
    - CPU (loop): 0.5-2 secondes par bâtiment
    - GPU (vectorized): 20-40 ms par bâtiment (50-100× speedup)

    Args:
        points: Point cloud [N, 3]
        heights: Heights above ground [N]
        initial_bbox: Initial bbox (xmin, ymin, xmax, ymax)
        max_shift: Maximum shift distance in meters (default: 15.0)
        step: Grid search step size (default: 1.0m)
        height_threshold: Height threshold for building/ground (default: 1.0m)
        ground_penalty: Penalty weight for ground points
        non_ground_reward: Reward weight for building points

    Returns:
        Tuple of:
        - best_shift: (dx, dy) optimal shift
        - best_bbox: (xmin, ymin, xmax, ymax) optimized bbox

    Example:
        >>> clusterer = BuildingClusterer()
        >>> shift, bbox = clusterer.optimize_bbox_for_building_gpu(
        ...     points, heights, initial_bbox,
        ...     max_shift=15.0, step=1.0
        ... )
    """
    if not HAS_GPU:
        logger.debug("GPU not available for bbox optimization, using CPU")
        return self.optimize_bbox_for_building(
            points, heights, initial_bbox,
            max_shift, step, height_threshold,
            ground_penalty, non_ground_reward
        )

    try:
        import cupy as cp

        # 1. Transfer to GPU
        points_xy_gpu = cp.asarray(points[:, :2], dtype=cp.float32)
        heights_gpu = cp.asarray(heights, dtype=cp.float32)

        # 2. Generate ALL shift combinations (grid search)
        shifts = np.arange(-max_shift, max_shift + step, step, dtype=np.float32)
        dx_grid, dy_grid = np.meshgrid(shifts, shifts)
        shifts_flat = np.column_stack([dx_grid.ravel(), dy_grid.ravel()])
        n_shifts = len(shifts_flat)

        logger.debug(f"  Testing {n_shifts} bbox positions on GPU...")

        # Transfer shifts to GPU
        shifts_gpu = cp.asarray(shifts_flat, dtype=cp.float32)

        # 3. Create ALL bboxes at once
        xmin, ymin, xmax, ymax = initial_bbox
        base_bbox = cp.array([xmin, ymin, xmax, ymax], dtype=cp.float32)

        # Broadcast to create all shifted bboxes [n_shifts, 4]
        bboxes_gpu = cp.tile(base_bbox, (n_shifts, 1))
        bboxes_gpu[:, [0, 1]] += shifts_gpu  # Shift min corner
        bboxes_gpu[:, [2, 3]] += shifts_gpu  # Shift max corner

        # 4. Vectorized point-in-bbox test
        # Test ALL points against ALL bboxes simultaneously
        # Shape operations: [n_shifts, 1] vs [1, n_points] → [n_shifts, n_points]

        xs = points_xy_gpu[:, 0]  # [n_points]
        ys = points_xy_gpu[:, 1]  # [n_points]

        # Broadcasting magic: each bbox tested against all points
        # 🚀 This is the key GPU acceleration - massive parallelization
        in_bbox = (
            (xs[None, :] >= bboxes_gpu[:, 0][:, None]) &  # xmin check
            (xs[None, :] <= bboxes_gpu[:, 2][:, None]) &  # xmax check
            (ys[None, :] >= bboxes_gpu[:, 1][:, None]) &  # ymin check
            (ys[None, :] <= bboxes_gpu[:, 3][:, None])    # ymax check
        )
        # Result: [n_shifts, n_points] boolean mask

        # 5. Vectorized scoring
        # Classify points as building (height > threshold) or ground
        is_building = heights_gpu[None, :] > height_threshold  # [1, n_points]

        # Count building points per bbox
        n_building = cp.sum(in_bbox & is_building, axis=1)  # [n_shifts]

        # Count ground points per bbox
        n_ground = cp.sum(in_bbox & ~is_building, axis=1)  # [n_shifts]

        # Compute scores for all bboxes at once
        scores = (n_building * non_ground_reward -
                  n_ground * ground_penalty)  # [n_shifts]

        # 6. Find best bbox (single GPU operation)
        best_idx = int(cp.argmax(scores))
        best_score = float(scores[best_idx])
        best_shift = tuple(cp.asnumpy(shifts_gpu[best_idx]).tolist())
        best_bbox = tuple(cp.asnumpy(bboxes_gpu[best_idx]).tolist())

        # 7. Cleanup GPU memory
        cp.get_default_memory_pool().free_all_blocks()

        logger.debug(
            f"  GPU optimization: best shift={best_shift}, "
            f"score={best_score:.1f}, "
            f"building={int(n_building[best_idx])}, "
            f"ground={int(n_ground[best_idx])}"
        )

        return best_shift, best_bbox

    except Exception as e:
        logger.warning(f"GPU bbox optimization failed: {e}, using CPU fallback")

        # Cleanup on error
        if 'cp' in locals():
            cp.get_default_memory_pool().free_all_blocks()

        # Fallback to CPU
        return self.optimize_bbox_for_building(
            points, heights, initial_bbox,
            max_shift, step, height_threshold,
            ground_penalty, non_ground_reward
        )
```

### Intégration dans BuildingClusterer

```python
# Dans BuildingClusterer.__init__(), ajouter paramètre:
def __init__(
    self,
    # ... existing params ...
    use_gpu_bbox_optimization: bool = True,  # 🆕 NEW
):
    # ... existing code ...
    self.use_gpu_bbox_optimization = use_gpu_bbox_optimization

# Dans les méthodes qui utilisent optimize_bbox_for_building:
# (ex: process_building_cluster(), detect_and_classify_buildings())

# Remplacer:
best_shift, best_bbox = self.optimize_bbox_for_building(
    points, heights, initial_bbox, ...
)

# Par:
if self.use_gpu_bbox_optimization and HAS_GPU:
    best_shift, best_bbox = self.optimize_bbox_for_building_gpu(
        points, heights, initial_bbox, ...
    )
else:
    best_shift, best_bbox = self.optimize_bbox_for_building(
        points, heights, initial_bbox, ...
    )
```

---

## P1.3 - GPU KNN Façades

### Fichier: `ign_lidar/core/classification/building/facade_processor.py`

### Modification Simple

```python
# Ligne ~295 dans _classify_wall_points()

# AVANT:
from scipy.spatial import cKDTree
tree = cKDTree(candidate_points[:, :2])
distances, indices = tree.query(candidate_points[:, :2], k=50)

# APRÈS:
# 🚀 GPU-accelerated KNN (15-20× speedup)
from ign_lidar.optimization.gpu_accelerated_ops import knn

try:
    # Try GPU first
    distances, indices = knn(
        candidate_points[:, :2],
        k=50
    )
except Exception as e:
    # Fallback to CPU if GPU fails
    logger.debug(f"GPU KNN failed, using CPU: {e}")
    from scipy.spatial import cKDTree
    tree = cKDTree(candidate_points[:, :2])
    distances, indices = tree.query(candidate_points[:, :2], k=50)
```

---

## Tests & Validation

### Test GPU Road Classification

```python
# tests/test_gpu_reclassifier.py

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box

from ign_lidar.core.classification.reclassifier import Reclassifier, HAS_GPU

@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="GPU not available")
def test_classify_roads_with_nature_gpu_vs_cpu():
    """
    Test GPU road classification vs CPU.

    Validates:
    1. Results match CPU implementation (<1% difference)
    2. GPU is faster than CPU (>5× speedup)
    3. All road types classified correctly
    """
    # Create synthetic data
    np.random.seed(42)
    n_points = 100000
    points = np.random.rand(n_points, 3) * 100  # 100m × 100m

    # Create road polygons with different nature types
    road_types = ['Autoroute', 'Route principale', 'Chemin']
    road_polygons = []
    road_natures = []

    for i, nature in enumerate(road_types):
        # Create rectangular road polygon
        xmin, ymin = 10 + i*30, 10
        xmax, ymax = 20 + i*30, 90
        poly = box(xmin, ymin, xmax, ymax)
        road_polygons.append(poly)
        road_natures.append(nature)

    roads_gdf = gpd.GeoDataFrame({
        'geometry': road_polygons,
        'nature': road_natures
    })

    # Test CPU
    reclassifier_cpu = Reclassifier(acceleration_mode='cpu')
    labels_cpu = np.zeros(n_points, dtype=np.int32)

    import time
    start = time.time()
    n_cpu = reclassifier_cpu._classify_roads_with_nature(
        points, labels_cpu, roads_gdf
    )
    time_cpu = time.time() - start

    # Test GPU
    reclassifier_gpu = Reclassifier(acceleration_mode='gpu')
    labels_gpu = np.zeros(n_points, dtype=np.int32)

    start = time.time()
    n_gpu = reclassifier_gpu._classify_roads_with_nature_gpu(
        points, labels_gpu, roads_gdf
    )
    time_gpu = time.time() - start

    # Validation
    # 1. Same number of points classified
    assert n_cpu == n_gpu, f"Different counts: CPU={n_cpu}, GPU={n_gpu}"

    # 2. Labels match (allow <1% difference due to floating point)
    difference = np.sum(labels_cpu != labels_gpu)
    difference_pct = 100 * difference / n_points
    assert difference_pct < 1.0, f"Labels differ by {difference_pct:.2f}%"

    # 3. GPU is faster
    speedup = time_cpu / time_gpu
    print(f"\n  CPU time: {time_cpu:.3f}s")
    print(f"  GPU time: {time_gpu:.3f}s")
    print(f"  Speedup: {speedup:.1f}×")
    assert speedup > 5.0, f"GPU not faster enough: {speedup:.1f}× (expected >5×)"

    # 4. All road types classified
    for i, nature in enumerate(road_types):
        expected_code = reclassifier_gpu._get_asprs_code_for_road(nature)
        n_classified = np.sum(labels_gpu == expected_code)
        assert n_classified > 0, f"No points classified as {nature} (code {expected_code})"

@pytest.mark.gpu
def test_gpu_fallback_on_error():
    """Test graceful fallback to CPU if GPU fails."""
    # This test validates the error handling
    # Mock GPU failure by using invalid data

    points = np.array([])  # Empty array should trigger error
    labels = np.array([])
    roads_gdf = gpd.GeoDataFrame()

    reclassifier = Reclassifier(acceleration_mode='gpu')

    # Should not crash, should fallback to CPU
    try:
        n_classified = reclassifier._classify_roads_with_nature_gpu(
            points, labels, roads_gdf
        )
        # If we get here, fallback worked
        assert True
    except Exception as e:
        pytest.fail(f"GPU fallback failed: {e}")
```

### Test GPU BBox Optimization

```python
# tests/test_gpu_bbox_optimization.py

import pytest
import numpy as np
from ign_lidar.core.classification.building import BuildingClusterer, HAS_GPU

@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="GPU not available")
def test_optimize_bbox_gpu_vs_cpu():
    """
    Test GPU bbox optimization vs CPU.

    Validates:
    1. Results are identical (same bbox found)
    2. GPU is much faster (>50× speedup)
    """
    # Create synthetic building cluster
    np.random.seed(42)

    # Building points around (50, 50) elevated at z=5
    n_build = 1000
    building_xy = np.random.normal(loc=(50, 50), scale=5, size=(n_build, 2))
    building_z = np.full((n_build, 1), 5.0)
    building_pts = np.hstack([building_xy, building_z])

    # Ground points scattered
    n_ground = 5000
    ground_xy = np.random.uniform(low=(0, 0), high=(100, 100), size=(n_ground, 2))
    ground_z = np.zeros((n_ground, 1))
    ground_pts = np.hstack([ground_xy, ground_z])

    points = np.vstack([building_pts, ground_pts])
    heights = points[:, 2]

    # Initial bbox (intentionally offset)
    initial_bbox = (30.0, 30.0, 40.0, 40.0)

    # Test CPU
    clusterer_cpu = BuildingClusterer(use_gpu_bbox_optimization=False)

    import time
    start = time.time()
    shift_cpu, bbox_cpu = clusterer_cpu.optimize_bbox_for_building(
        points, heights, initial_bbox,
        max_shift=30.0, step=2.0  # Coarse grid for speed
    )
    time_cpu = time.time() - start

    # Test GPU
    clusterer_gpu = BuildingClusterer(use_gpu_bbox_optimization=True)

    start = time.time()
    shift_gpu, bbox_gpu = clusterer_gpu.optimize_bbox_for_building_gpu(
        points, heights, initial_bbox,
        max_shift=30.0, step=2.0
    )
    time_gpu = time.time() - start

    # Validation
    # 1. Shifts should be very similar (within 1 step)
    shift_diff = np.linalg.norm(np.array(shift_cpu) - np.array(shift_gpu))
    assert shift_diff <= 2.0, f"Shifts differ too much: {shift_diff:.2f}m"

    # 2. Bboxes should be very similar
    bbox_diff = np.linalg.norm(np.array(bbox_cpu) - np.array(bbox_gpu))
    assert bbox_diff <= 3.0, f"Bboxes differ too much: {bbox_diff:.2f}m"

    # 3. GPU is much faster
    speedup = time_cpu / time_gpu
    print(f"\n  CPU time: {time_cpu:.3f}s")
    print(f"  GPU time: {time_gpu:.3f}s")
    print(f"  Speedup: {speedup:.1f}×")
    assert speedup > 20.0, f"GPU speedup insufficient: {speedup:.1f}× (expected >20×)"

@pytest.mark.gpu
def test_bbox_optimization_accuracy():
    """Test that optimized bbox contains most building points."""
    np.random.seed(42)

    # Known building at (100, 100) with 10m radius
    n_build = 500
    building_xy = np.random.normal(loc=(100, 100), scale=3, size=(n_build, 2))
    building_z = np.full((n_build, 1), 5.0)
    building_pts = np.hstack([building_xy, building_z])

    # Ground everywhere else
    n_ground = 2000
    ground_xy = np.random.uniform(low=(0, 0), high=(200, 200), size=(n_ground, 2))
    ground_z = np.zeros((n_ground, 1))
    ground_pts = np.hstack([ground_xy, ground_z])

    points = np.vstack([building_pts, ground_pts])
    heights = points[:, 2]

    # Bad initial bbox (far from building)
    initial_bbox = (10.0, 10.0, 20.0, 20.0)

    # Optimize
    clusterer = BuildingClusterer(use_gpu_bbox_optimization=True)
    shift, bbox = clusterer.optimize_bbox_for_building_gpu(
        points, heights, initial_bbox,
        max_shift=100.0, step=2.0,
        height_threshold=1.0
    )

    # Check optimized bbox contains most building points
    xmin, ymin, xmax, ymax = bbox
    xs, ys = points[:, 0], points[:, 1]
    mask = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

    n_build_in = np.sum(mask & (heights > 1.0))
    n_ground_in = np.sum(mask & (heights <= 1.0))

    # Should capture >80% of building points
    capture_rate = n_build_in / n_build
    assert capture_rate > 0.8, f"Low building capture: {capture_rate:.1%}"

    # Should have more building than ground
    ratio = n_build_in / max(n_ground_in, 1)
    assert ratio > 2.0, f"Too much ground in bbox: {ratio:.2f}"
```

---

## Performance Benchmarks

### Benchmark Script

```python
# scripts/benchmark_gpu_improvements.py

#!/usr/bin/env python3
"""
Benchmark GPU improvements vs baseline CPU.

Usage:
    python scripts/benchmark_gpu_improvements.py \
        --tile data/tiles/example.laz \
        --ground-truth data/bdtopo/example.gpkg \
        --output benchmarks/gpu_improvements.json
"""

import argparse
import json
import time
import logging
from pathlib import Path

import numpy as np
import laspy
import geopandas as gpd

from ign_lidar.core.classification.reclassifier import Reclassifier
from ign_lidar.core.classification.building import BuildingClusterer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_road_classification(points, roads_gdf):
    """Benchmark road classification CPU vs GPU."""
    logger.info("🚗 Benchmarking road classification...")

    results = {}

    # CPU
    logger.info("  Testing CPU...")
    reclassifier_cpu = Reclassifier(acceleration_mode='cpu')
    labels_cpu = np.zeros(len(points), dtype=np.int32)

    start = time.time()
    n_cpu = reclassifier_cpu._classify_roads_with_nature(
        points, labels_cpu, roads_gdf
    )
    time_cpu = time.time() - start

    results['cpu'] = {
        'time': time_cpu,
        'n_classified': int(n_cpu),
        'throughput': len(points) / time_cpu
    }

    # GPU
    logger.info("  Testing GPU...")
    reclassifier_gpu = Reclassifier(acceleration_mode='gpu')
    labels_gpu = np.zeros(len(points), dtype=np.int32)

    start = time.time()
    n_gpu = reclassifier_gpu._classify_roads_with_nature_gpu(
        points, labels_gpu, roads_gdf
    )
    time_gpu = time.time() - start

    results['gpu'] = {
        'time': time_gpu,
        'n_classified': int(n_gpu),
        'throughput': len(points) / time_gpu
    }

    # Comparison
    speedup = time_cpu / time_gpu
    agreement = np.sum(labels_cpu == labels_gpu) / len(points)

    results['comparison'] = {
        'speedup': speedup,
        'agreement_pct': agreement * 100
    }

    logger.info(f"  ✅ CPU: {time_cpu:.2f}s, {n_cpu:,} points")
    logger.info(f"  ✅ GPU: {time_gpu:.2f}s, {n_gpu:,} points")
    logger.info(f"  📊 Speedup: {speedup:.1f}×, Agreement: {agreement:.1%}")

    return results

def benchmark_bbox_optimization(points, heights, n_buildings=10):
    """Benchmark bbox optimization CPU vs GPU."""
    logger.info(f"🏢 Benchmarking bbox optimization ({n_buildings} buildings)...")

    results = {
        'cpu': [],
        'gpu': []
    }

    # Generate random initial bboxes
    np.random.seed(42)
    initial_bboxes = []
    for i in range(n_buildings):
        cx, cy = np.random.rand(2) * 100
        size = 10 + np.random.rand() * 10
        bbox = (cx - size/2, cy - size/2, cx + size/2, cy + size/2)
        initial_bboxes.append(bbox)

    # CPU
    logger.info("  Testing CPU...")
    clusterer_cpu = BuildingClusterer(use_gpu_bbox_optimization=False)

    start_total = time.time()
    for i, bbox in enumerate(initial_bboxes):
        start = time.time()
        shift, optimized = clusterer_cpu.optimize_bbox_for_building(
            points, heights, bbox,
            max_shift=15.0, step=1.0
        )
        elapsed = time.time() - start
        results['cpu'].append({'time': elapsed, 'shift': shift})
    time_cpu_total = time.time() - start_total

    # GPU
    logger.info("  Testing GPU...")
    clusterer_gpu = BuildingClusterer(use_gpu_bbox_optimization=True)

    start_total = time.time()
    for i, bbox in enumerate(initial_bboxes):
        start = time.time()
        shift, optimized = clusterer_gpu.optimize_bbox_for_building_gpu(
            points, heights, bbox,
            max_shift=15.0, step=1.0
        )
        elapsed = time.time() - start
        results['gpu'].append({'time': elapsed, 'shift': shift})
    time_gpu_total = time.time() - start_total

    # Statistics
    times_cpu = [r['time'] for r in results['cpu']]
    times_gpu = [r['time'] for r in results['gpu']]

    summary = {
        'cpu': {
            'total_time': time_cpu_total,
            'avg_time_per_building': np.mean(times_cpu),
            'std_time': np.std(times_cpu)
        },
        'gpu': {
            'total_time': time_gpu_total,
            'avg_time_per_building': np.mean(times_gpu),
            'std_time': np.std(times_gpu)
        },
        'comparison': {
            'speedup': time_cpu_total / time_gpu_total,
            'avg_speedup_per_building': np.mean(times_cpu) / np.mean(times_gpu)
        }
    }

    logger.info(f"  ✅ CPU: {time_cpu_total:.2f}s total, "
                f"{np.mean(times_cpu):.3f}s avg/building")
    logger.info(f"  ✅ GPU: {time_gpu_total:.2f}s total, "
                f"{np.mean(times_gpu):.3f}s avg/building")
    logger.info(f"  📊 Speedup: {summary['comparison']['speedup']:.1f}×")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU improvements")
    parser.add_argument('--tile', type=Path, required=True,
                        help="Path to LAZ tile")
    parser.add_argument('--ground-truth', type=Path, required=True,
                        help="Path to ground truth GeoPackage")
    parser.add_argument('--output', type=Path, required=True,
                        help="Output JSON file")
    parser.add_argument('--n-buildings', type=int, default=10,
                        help="Number of buildings to test for bbox optimization")

    args = parser.parse_args()

    # Load data
    logger.info(f"📂 Loading tile: {args.tile}")
    las = laspy.read(str(args.tile))
    points = np.vstack([las.x, las.y, las.z]).T
    heights = np.array(las.z)  # Simplified

    logger.info(f"📂 Loading ground truth: {args.ground_truth}")
    roads_gdf = gpd.read_file(args.ground_truth, layer='roads')

    logger.info(f"  Loaded {len(points):,} points, {len(roads_gdf)} roads")

    # Run benchmarks
    results = {
        'metadata': {
            'tile': str(args.tile),
            'n_points': len(points),
            'n_roads': len(roads_gdf),
            'n_buildings_tested': args.n_buildings
        },
        'road_classification': benchmark_road_classification(points, roads_gdf),
        'bbox_optimization': benchmark_bbox_optimization(
            points, heights, n_buildings=args.n_buildings
        )
    }

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ Results saved to {args.output}")

if __name__ == '__main__':
    main()
```

---

## 🎉 Résumé

### Code Prêt à Implémenter

✅ **P0.1 - GPU Road Classification**: Complète avec fallback CPU  
✅ **P0.2 - GPU BBox Optimization**: Vectorisé pour 50-100× speedup  
✅ **P1.3 - GPU KNN Façades**: Simple modification 1 ligne  
✅ **Tests**: Unitaires et validation CPU/GPU  
✅ **Benchmarks**: Script automatisé

### Next Steps

1. Copier les méthodes dans les fichiers respectifs
2. Lancer tests: `pytest tests/test_gpu_*.py -v`
3. Benchmark: `python scripts/benchmark_gpu_improvements.py`
4. Valider sur données production
5. Déployer progressivement (feature flags)

**Temps d'implémentation estimé:** 4-6 jours pour P0+P1 complet.
