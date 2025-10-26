"""
Optimized Ground Truth Computation with Automatic CPU/GPU Selection

This module provides optimized implementations of ground truth labeling
that automatically selects the best method based on available hardware:

1. GPU Chunked (fastest for large datasets with GPU)
2. GPU (fastest for small-medium datasets with GPU)
3. CPU STRtree (10-30× faster than naive, no GPU needed)
4. CPU Vectorized (fallback)

The optimizer automatically detects available hardware and selects
the appropriate implementation.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import centralized priority system
from ign_lidar.core.classification.priorities import (
    PRIORITY_ORDER,
    get_label_map,
    get_priority_value,
)

try:
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Point, Polygon
    from shapely.prepared import prep
    from shapely.strtree import STRtree

    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    logger.warning("Spatial libraries not available")

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import cuspatial

    HAS_CUSPATIAL = True
except ImportError:
    HAS_CUSPATIAL = False


class GroundTruthOptimizer:
    """
    Automatically selects and applies the best ground truth optimization.

    Performance characteristics:
    - GPU Chunked: 100-1000× speedup for datasets > 10M points
    - GPU: 100-500× speedup for datasets < 10M points
    - CPU STRtree: 10-30× speedup, works everywhere
    - CPU Vectorized: 5-10× speedup, GeoPandas fallback
    """

    # Hardware detection cache
    _gpu_available = None
    _cuspatial_available = None

    def __init__(
        self,
        force_method: Optional[str] = None,
        gpu_chunk_size: int = 2_000_000,  # ✅ FIXED: Reduced default from 5M to 2M (prevents OOM Exit 137)
        verbose: bool = True,
    ):
        """
        Initialize optimizer.

        Args:
            force_method: Force specific method ('gpu_chunked', 'gpu', 'strtree', 'vectorized')
            gpu_chunk_size: Number of points per GPU chunk (default: 2M, safer for CPU mode)
            verbose: Enable verbose logging
        """
        self.force_method = force_method
        self.gpu_chunk_size = gpu_chunk_size
        self.verbose = verbose

        # Detect hardware on first use
        if GroundTruthOptimizer._gpu_available is None:
            GroundTruthOptimizer._gpu_available = self._check_gpu()
        if GroundTruthOptimizer._cuspatial_available is None:
            GroundTruthOptimizer._cuspatial_available = self._check_cuspatial()

    @staticmethod
    def _check_gpu() -> bool:
        """Check if GPU is available."""
        if not HAS_CUPY:
            return False
        try:
            _ = cp.array([1.0])
            return True
        except Exception:
            return False

    @staticmethod
    def _check_cuspatial() -> bool:
        """Check if cuSpatial is available."""
        return HAS_CUSPATIAL

    def select_method(self, n_points: int, n_polygons: int) -> str:
        """
        Automatically select the best method based on data size and hardware.

        Args:
            n_points: Number of points to label
            n_polygons: Number of ground truth polygons

        Returns:
            Method name: 'gpu_chunked', 'gpu', 'strtree', or 'vectorized'
        """
        if self.force_method:
            return self.force_method

        # GPU methods (if available)
        if GroundTruthOptimizer._gpu_available:
            # Use chunked for large datasets
            if n_points > 10_000_000:
                return "gpu_chunked"
            else:
                return "gpu"

        # CPU methods
        if HAS_SPATIAL:
            # STRtree is best CPU option
            return "strtree"

        # Fallback
        return "vectorized"

    def label_points(
        self,
        points: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        label_priority: Optional[list] = None,
        ndvi: Optional[np.ndarray] = None,
        use_ndvi_refinement: bool = True,
        ndvi_vegetation_threshold: float = 0.3,
        ndvi_building_threshold: float = 0.15,
    ) -> np.ndarray:
        """
        Label points with ground truth using optimal method.

        Args:
            points: Point cloud [N, 3] with XYZ coordinates
            ground_truth_features: Dictionary of feature type -> GeoDataFrame
            label_priority: Priority order for overlapping features
            ndvi: Optional NDVI values [N] for refinement
            use_ndvi_refinement: Use NDVI to refine labels
            ndvi_vegetation_threshold: NDVI threshold for vegetation (>= threshold)
            ndvi_building_threshold: NDVI threshold for buildings (<= threshold)

        Returns:
            Labels array [N] with feature type indices:
            - 0: unlabeled/ground
            - 1: buildings
            - 2: roads
            - 3: water
            - 4: vegetation
        """
        start_time = time.time()

        # Count polygons
        n_polygons = sum(
            len(gdf) for gdf in ground_truth_features.values() if gdf is not None
        )

        # Select method
        method = self.select_method(len(points), n_polygons)

        if self.verbose:
            logger.info(
                f"Ground truth labeling: {len(points):,} points, {n_polygons:,} polygons"
            )
            logger.info(f"Method: {method}")

        # Apply method
        if method == "gpu_chunked":
            labels = self._label_gpu_chunked(
                points,
                ground_truth_features,
                label_priority,
                ndvi,
                use_ndvi_refinement,
                ndvi_vegetation_threshold,
                ndvi_building_threshold,
            )
        elif method == "gpu":
            labels = self._label_gpu(
                points,
                ground_truth_features,
                label_priority,
                ndvi,
                use_ndvi_refinement,
                ndvi_vegetation_threshold,
                ndvi_building_threshold,
            )
        elif method == "strtree":
            labels = self._label_strtree(
                points,
                ground_truth_features,
                label_priority,
                ndvi,
                use_ndvi_refinement,
                ndvi_vegetation_threshold,
                ndvi_building_threshold,
            )
        else:  # vectorized
            labels = self._label_vectorized(
                points,
                ground_truth_features,
                label_priority,
                ndvi,
                use_ndvi_refinement,
                ndvi_vegetation_threshold,
                ndvi_building_threshold,
            )

        elapsed = time.time() - start_time

        if self.verbose:
            logger.info(f"Ground truth labeling completed in {elapsed:.2f}s")
            # Log distribution
            unique, counts = np.unique(labels, return_counts=True)
            label_names = {
                0: "unlabeled",
                1: "building",
                2: "road",
                3: "water",
                4: "vegetation",
            }
            for label_val, count in zip(unique, counts):
                label_name = label_names.get(label_val, f"unknown_{label_val}")
                pct = 100 * count / len(labels)
                logger.info(f"  {label_name}: {count:,} ({pct:.1f}%)")

        return labels

    def _label_gpu_chunked(
        self,
        points: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        label_priority: Optional[list],
        ndvi: Optional[np.ndarray],
        use_ndvi_refinement: bool,
        ndvi_vegetation_threshold: float,
        ndvi_building_threshold: float,
    ) -> np.ndarray:
        """GPU chunked implementation for large datasets."""
        from ..optimization.gpu import GPUGroundTruthClassifier

        classifier = GPUGroundTruthClassifier(
            gpu_chunk_size=self.gpu_chunk_size,
            use_cuspatial=True,
            ndvi_veg_threshold=ndvi_vegetation_threshold,
            ndvi_building_threshold=ndvi_building_threshold,
            verbose=self.verbose,
        )

        # Initialize labels
        labels = np.zeros(len(points), dtype=np.int32)

        # Use the GPU classifier
        return classifier.classify_with_ground_truth(
            labels, points, ground_truth_features, ndvi=ndvi
        )

    def _label_gpu(
        self,
        points: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        label_priority: Optional[list],
        ndvi: Optional[np.ndarray],
        use_ndvi_refinement: bool,
        ndvi_vegetation_threshold: float,
        ndvi_building_threshold: float,
    ) -> np.ndarray:
        """GPU implementation for small-medium datasets."""
        # Same as chunked but with larger chunk size
        from ..optimization.gpu import GPUGroundTruthClassifier

        classifier = GPUGroundTruthClassifier(
            gpu_chunk_size=len(points),  # Process all at once
            use_cuspatial=True,
            ndvi_veg_threshold=ndvi_vegetation_threshold,
            ndvi_building_threshold=ndvi_building_threshold,
            verbose=self.verbose,
        )

        labels = np.zeros(len(points), dtype=np.int32)

        return classifier.classify_with_ground_truth(
            labels, points, ground_truth_features, ndvi=ndvi
        )

    def _label_strtree(
        self,
        points: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        label_priority: Optional[list],
        ndvi: Optional[np.ndarray],
        use_ndvi_refinement: bool,
        ndvi_vegetation_threshold: float,
        ndvi_building_threshold: float,
    ) -> np.ndarray:
        """CPU STRtree implementation (10-30× faster than naive)."""
        if not HAS_SPATIAL:
            raise ImportError("Shapely and GeoPandas required for STRtree optimization")

        if label_priority is None:
            # ✅ Use centralized priority order
            label_priority = PRIORITY_ORDER

        # ✅ Use centralized label mapping
        label_map = get_label_map()

        # Initialize labels
        labels = np.zeros(len(points), dtype=np.int32)

        # Build spatial index
        if self.verbose:
            logger.info("  Building STRtree spatial index...")

        all_polygons = []
        polygon_labels = []
        polygon_priorities = []  # Store priority for each polygon
        prepared_polygons = []  # Pre-prepared for faster contains()

        # ✅ Use centralized priority function
        # Priority = numerical value from get_priority_value()
        # Higher number = higher priority
        label_priority_values = {}
        for feature_type in label_priority:
            label_value = label_map.get(feature_type, 0)
            priority_value = get_priority_value(feature_type)
            label_priority_values[label_value] = priority_value

        # Add polygons (order doesn't matter, we use explicit priorities)
        for feature_type in label_priority:
            if feature_type not in ground_truth_features:
                continue

            gdf = ground_truth_features[feature_type]
            if gdf is None or len(gdf) == 0:
                continue

            label_value = label_map.get(feature_type, 0)
            priority_value = label_priority_values.get(label_value, 0)

            for idx, row in gdf.iterrows():
                polygon = row["geometry"]
                if isinstance(polygon, (Polygon, MultiPolygon)):
                    all_polygons.append(polygon)
                    polygon_labels.append(label_value)
                    polygon_priorities.append(priority_value)  # ✅ NEW
                    prepared_polygons.append(
                        prep(polygon)
                    )  # Prepare for faster queries

        if len(all_polygons) == 0:
            logger.warning("No valid polygons for labeling")
            return labels

        # Build STRtree
        tree = STRtree(all_polygons)

        if self.verbose:
            logger.info(
                f"  Labeling {len(points):,} points with {len(all_polygons):,} polygons..."
            )

        # ✅ FIXED: Process points in smaller batches to avoid memory issues
        # Use smaller batch size (50K instead of 100K) to reduce memory footprint
        batch_size = 50_000
        n_batches = (len(points) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(points))
            batch_points = points[start_idx:end_idx]

            # ✅ OPTIMIZED: Create Point geometries and query immediately
            # This avoids keeping all points in memory at once
            for i, point_coords in enumerate(batch_points):
                point_geom = Point(point_coords[0], point_coords[1])

                # Find candidate polygon indices
                candidate_indices = tree.query(point_geom)

                # Check if empty (handle numpy array properly)
                if len(candidate_indices) == 0:
                    continue

                # ✅ FIXED: Check all matching polygons and select highest priority
                best_label = 0
                best_priority = -1

                for candidate_idx in candidate_indices:
                    # Use prepared polygon for much faster contains() check
                    # ✅ FIXED: Use covers() instead of contains() to include
                    # boundary points
                    if prepared_polygons[candidate_idx].covers(point_geom):
                        label = polygon_labels[candidate_idx]
                        priority = polygon_priorities[candidate_idx]

                        # Keep the label with highest priority
                        if priority > best_priority:
                            best_label = label
                            best_priority = priority

                # Assign the label with the highest priority
                if best_label > 0:
                    labels[start_idx + i] = best_label

            if self.verbose and n_batches > 1 and (batch_idx + 1) % 10 == 0:
                pct = 100 * (batch_idx + 1) / n_batches
                logger.info(f"    Progress: {pct:.1f}%")

        # Apply NDVI refinement
        if ndvi is not None and use_ndvi_refinement:
            labels = self._apply_ndvi_refinement(
                labels, ndvi, ndvi_vegetation_threshold, ndvi_building_threshold
            )

        return labels

    def _label_vectorized(
        self,
        points: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        label_priority: Optional[list],
        ndvi: Optional[np.ndarray],
        use_ndvi_refinement: bool,
        ndvi_vegetation_threshold: float,
        ndvi_building_threshold: float,
    ) -> np.ndarray:
        """Vectorized CPU implementation using GeoPandas spatial joins."""
        if not HAS_SPATIAL:
            raise ImportError("Shapely and GeoPandas required for vectorized method")

        if label_priority is None:
            label_priority = ["buildings", "roads", "water", "vegetation"]

        label_map = {"buildings": 1, "roads": 2, "water": 3, "vegetation": 4}

        # Initialize labels
        labels = np.zeros(len(points), dtype=np.int32)

        # Create GeoDataFrame of points
        if self.verbose:
            logger.info("  Creating point GeoDataFrame...")

        point_geoms = [Point(p[0], p[1]) for p in points]
        # FIXED: Set CRS to match ground truth features (EPSG:2154 for Lambert 93)
        points_gdf = gpd.GeoDataFrame(
            {"geometry": point_geoms, "index": np.arange(len(points))},
            crs="EPSG:2154",  # Lambert 93 - standard for IGN data
        )

        # Process each feature type in reverse priority
        for feature_type in reversed(label_priority):
            if feature_type not in ground_truth_features:
                continue

            gdf = ground_truth_features[feature_type]
            if gdf is None or len(gdf) == 0:
                continue

            label_value = label_map.get(feature_type, 0)

            if self.verbose:
                logger.info(f"  Processing {feature_type} ({len(gdf)} polygons)...")

            # Spatial join
            joined = gpd.sjoin(points_gdf, gdf, how="inner", predicate="within")

            # Update labels
            if len(joined) > 0:
                point_indices = joined["index"].values
                labels[point_indices] = label_value

        # Apply NDVI refinement
        if ndvi is not None and use_ndvi_refinement:
            labels = self._apply_ndvi_refinement(
                labels, ndvi, ndvi_vegetation_threshold, ndvi_building_threshold
            )

        return labels

    def _apply_ndvi_refinement(
        self,
        labels: np.ndarray,
        ndvi: np.ndarray,
        ndvi_vegetation_threshold: float,
        ndvi_building_threshold: float,
    ) -> np.ndarray:
        """Apply NDVI-based refinement to labels."""
        if self.verbose:
            logger.info("  Applying NDVI refinement...")

        # Label mapping
        BUILDING = 1
        VEGETATION = 4

        # Refine buildings: high NDVI → vegetation
        building_mask = labels == BUILDING
        high_ndvi_buildings = building_mask & (ndvi >= ndvi_vegetation_threshold)
        n_to_veg = np.sum(high_ndvi_buildings)
        if n_to_veg > 0:
            labels[high_ndvi_buildings] = VEGETATION
            if self.verbose:
                logger.info(
                    f"    Reclassified {n_to_veg:,} high-NDVI buildings → vegetation"
                )

        # Refine vegetation: low NDVI → building
        vegetation_mask = labels == VEGETATION
        low_ndvi_vegetation = vegetation_mask & (ndvi <= ndvi_building_threshold)
        n_to_building = np.sum(low_ndvi_vegetation)
        if n_to_building > 0:
            labels[low_ndvi_vegetation] = BUILDING
            if self.verbose:
                logger.info(
                    f"    Reclassified {n_to_building:,} low-NDVI vegetation → building"
                )

        # Label unlabeled high-NDVI points as vegetation
        unlabeled_mask = labels == 0
        high_ndvi_unlabeled = unlabeled_mask & (ndvi >= ndvi_vegetation_threshold)
        n_unlabeled_to_veg = np.sum(high_ndvi_unlabeled)
        if n_unlabeled_to_veg > 0:
            labels[high_ndvi_unlabeled] = VEGETATION
            if self.verbose:
                logger.info(
                    f"    Labeled {n_unlabeled_to_veg:,} high-NDVI unlabeled → vegetation"
                )

        return labels
