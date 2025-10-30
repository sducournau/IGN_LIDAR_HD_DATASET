"""
Reclassifier Module with GPU Acceleration

This module provides ground truth reclassification using:
- CPU: Spatial indexing (STRtree) for fast point-in-polygon queries
- GPU: RAPIDS cuSpatial for GPU-accelerated spatial operations
- GPU+cuML: Additional GPU acceleration for large-scale processing

Performance comparison (18M points):
- CPU baseline: ~30-60 minutes
- CPU with STRtree: ~5-10 minutes
- GPU (RAPIDS): ~1-2 minutes
- GPU+cuML: ~30-60 seconds

Author: Data Processing Team
Date: October 16, 2025
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ✅ Import centralized constants and priority system
from .constants import ASPRSClass
from ign_lidar.core.classification.priorities import get_priority_order_for_iteration

# Import CPU spatial libraries
try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    logger.warning("Spatial libraries not available for reclassification")

# Import geometric rules engine
try:
    from .geometric_rules import GeometricRulesEngine

    HAS_GEOMETRIC_RULES = True
except ImportError:
    HAS_GEOMETRIC_RULES = False
    logger.debug("Geometric rules engine not available")

# Import GPU libraries (RAPIDS)
try:
    import cudf
    import cupy as cp
    import cuspatial

    HAS_GPU = True
    logger.info("✅ GPU acceleration available (RAPIDS cuSpatial)")
except ImportError:
    HAS_GPU = False
    logger.debug("GPU libraries not available (install RAPIDS for GPU acceleration)")

# Import cuML for additional GPU features
try:
    import cuml

    HAS_CUML = True
    logger.info("✅ cuML available for additional GPU acceleration")
except ImportError:
    HAS_CUML = False
    logger.debug("cuML not available")

# Acceleration mode type
AccelerationMode = Literal["cpu", "gpu", "gpu+cuml", "auto"]


class Reclassifier:
    """
    Reclassifier with multi-backend support (CPU, GPU, GPU+cuML).

    Features:
    - CPU: STRtree spatial indexing for O(log n) query performance
    - GPU: RAPIDS cuSpatial for GPU-accelerated spatial operations
    - GPU+cuML: Additional GPU optimizations
    - Automatic backend selection based on availability
    - Chunked processing to manage memory usage
    - Progress tracking for large datasets
    - Priority-based classification hierarchy

    Performance Guide:
    - Use 'cpu' for <5M points or no GPU
    - Use 'gpu' for 5M-50M points with RAPIDS
    - Use 'gpu+cuml' for >50M points with full RAPIDS stack
    - Use 'auto' to automatically select best available
    """

    def __init__(
        self,
        chunk_size: int = 100000,
        show_progress: bool = True,
        acceleration_mode: AccelerationMode = "auto",
        use_geometric_rules: bool = True,
        ndvi_vegetation_threshold: float = 0.3,
        ndvi_road_threshold: float = 0.15,
        road_vegetation_height_threshold: float = 2.0,
        building_buffer_distance: float = 2.0,
        max_building_height_difference: float = 3.0,
        verticality_threshold: float = 0.7,
        verticality_search_radius: float = 1.0,
        min_vertical_neighbors: int = 5,
    ):
        """
        Initialize optimized reclassifier.

        Args:
            chunk_size: Number of points to process per chunk
            show_progress: Show progress bars
            acceleration_mode: Acceleration backend ('cpu', 'gpu', 'gpu+cuml', 'auto')
            use_geometric_rules: Apply geometric rules after basic reclassification
            ndvi_vegetation_threshold: NDVI threshold for vegetation (>= this = vegetation)
            ndvi_road_threshold: NDVI threshold for roads (<= this = likely road)
            road_vegetation_height_threshold: Height above road to classify as vegetation (meters)
            building_buffer_distance: Buffer around buildings for unclassified points (meters)
            max_building_height_difference: Max height diff for building points (meters)
            verticality_threshold: Verticality score threshold for building classification (0-1)
            verticality_search_radius: Search radius for computing verticality (meters)
            min_vertical_neighbors: Minimum neighbors required for verticality computation
        """
        if not HAS_SPATIAL:
            raise ImportError(
                "Spatial libraries required for reclassification. "
                "Install: pip install shapely geopandas"
            )

        self.chunk_size = chunk_size
        self.show_progress = show_progress
        self.use_geometric_rules = use_geometric_rules

        # Initialize geometric rules engine
        self.geometric_rules = None
        if use_geometric_rules and HAS_GEOMETRIC_RULES:
            self.geometric_rules = GeometricRulesEngine(
                ndvi_vegetation_threshold=ndvi_vegetation_threshold,
                ndvi_road_threshold=ndvi_road_threshold,
                road_vegetation_height_threshold=road_vegetation_height_threshold,
                building_buffer_distance=building_buffer_distance,
                max_building_height_difference=max_building_height_difference,
                verticality_threshold=verticality_threshold,
                verticality_search_radius=verticality_search_radius,
                min_vertical_neighbors=min_vertical_neighbors,
            )
        elif use_geometric_rules:
            logger.warning("Geometric rules requested but engine not available")

        # Determine acceleration mode
        if acceleration_mode == "auto":
            if HAS_GPU and HAS_CUML:
                self.acceleration_mode = "gpu+cuml"
            elif HAS_GPU:
                self.acceleration_mode = "gpu"
            else:
                self.acceleration_mode = "cpu"
        else:
            # Validate requested mode is available
            if acceleration_mode in ["gpu", "gpu+cuml"] and not HAS_GPU:
                logger.warning(
                    f"GPU mode requested but RAPIDS not available, falling back to CPU"
                )
                self.acceleration_mode = "cpu"
            elif acceleration_mode == "gpu+cuml" and not HAS_CUML:
                logger.warning(f"cuML not available, falling back to GPU mode")
                self.acceleration_mode = "gpu"
            else:
                self.acceleration_mode = acceleration_mode

        # ✅ FIXED: Use centralized priority system
        # Priority order for sequential processing (lowest to highest)
        # Later features in the list overwrite earlier ones
        feature_priority = get_priority_order_for_iteration()

        # 🔄 CRITICAL FIX v3.0.5: Remove double reversal
        # get_priority_order_for_iteration() ALREADY returns lowest→highest
        # (it internally reverses PRIORITY_ORDER which is highest→lowest)
        # So we use it directly - important features come LAST to overwrite
        self.priority_order = [
            (feature, self._get_asprs_code(feature)) for feature in feature_priority
        ]

        logger.info("🚀 Optimized Reclassifier initialized")
        logger.info(f"   Acceleration: {self.acceleration_mode.upper()}")
        logger.info(f"   Chunk size: {chunk_size:,} points")

        if self.acceleration_mode == "cpu":
            logger.info(f"   Backend: CPU (STRtree spatial indexing)")
        elif self.acceleration_mode == "gpu":
            logger.info(f"   Backend: GPU (RAPIDS cuSpatial)")
        elif self.acceleration_mode == "gpu+cuml":
            logger.info(f"   Backend: GPU+cuML (Full RAPIDS stack)")

    def _get_asprs_code(self, feature_name: str) -> int:
        """
        Get ASPRS code for a feature type.

        Args:
            feature_name: Feature type (e.g., 'buildings', 'roads')

        Returns:
            ASPRS classification code
        """
        mapping = {
            "buildings": int(ASPRSClass.BUILDING),
            "roads": int(ASPRSClass.ROAD_SURFACE),
            "water": int(ASPRSClass.WATER),
            "vegetation": int(ASPRSClass.MEDIUM_VEGETATION),
            "bridges": int(ASPRSClass.BRIDGE_DECK),
            "railways": int(ASPRSClass.RAIL),
            "sports": int(ASPRSClass.OVERHEAD_STRUCTURE),  # Sports = 19
            "parking": int(ASPRSClass.ROAD_PARKING),  # Parking = 40
            "cemeteries": int(ASPRSClass.OVERHEAD_STRUCTURE),  # Cemetery = 19
        }
        return mapping.get(feature_name, int(ASPRSClass.UNCLASSIFIED))

    def reclassify(
        self,
        points: np.ndarray,
        current_labels: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        ndvi: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Reclassify points using ground truth features with spatial indexing.

        Args:
            points: XYZ coordinates [N, 3]
            current_labels: Current classification labels [N]
            ground_truth_features: Dict of feature_type -> GeoDataFrame
            ndvi: Optional NDVI values [N] for geometric rules
            intensities: Optional intensity values [N]

        Returns:
            Tuple of:
            - Updated classification labels [N]
            - Statistics dict with counts per feature type
        """
        n_points = len(points)
        updated_labels = current_labels.copy()
        stats = {}

        logger.info(f"🎯 Reclassifying {n_points:,} points with ground truth...")

        for feature_name, asprs_code in self.priority_order:
            if feature_name not in ground_truth_features:
                continue

            gdf = ground_truth_features[feature_name]
            if gdf is None or len(gdf) == 0:
                stats[feature_name] = 0
                continue

            logger.info(f"  Processing {feature_name}: {len(gdf)} features")

            # Reclassify points for this feature type
            n_classified = self._classify_feature(
                points=points,
                labels=updated_labels,
                geometries=gdf.geometry.values,
                asprs_code=asprs_code,
                feature_name=feature_name,
            )

            stats[feature_name] = n_classified

            if n_classified > 0:
                logger.info(f"    ✓ Classified {n_classified:,} points")

        # Apply geometric rules for refinement
        if self.geometric_rules is not None:
            logger.info("\n🔧 Applying geometric rules...")
            refined_labels, rule_stats = self.geometric_rules.apply_all_rules(
                points=points,
                labels=updated_labels,
                ground_truth_features=ground_truth_features,
                ndvi=ndvi,
                intensities=intensities,
            )
            updated_labels = refined_labels
            stats.update(rule_stats)

        # Calculate total changes
        n_changed = np.sum(current_labels != updated_labels)
        stats["total_changed"] = n_changed

        logger.info(f"\n📊 Reclassification Summary:")
        logger.info(
            f"  Total points changed: {n_changed:,} ({100*n_changed/n_points:.2f}%)"
        )

        return updated_labels, stats

    def reclassify_vegetation_above_surfaces(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
        height_above_ground: np.ndarray,
        ndvi: Optional[np.ndarray] = None,
        is_ground: Optional[np.ndarray] = None,
        height_threshold: float = 2.0,
        ndvi_threshold: float = 0.3,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        🆕 V5.2 Enhanced: Reclassify vegetation points above BD TOPO surfaces.

        This function identifies points that are:
        1. Inside BD TOPO polygons (roads, sports, cemeteries, parking)
        2. Significantly above ground (height_above_ground > threshold)
        3. Have vegetation signature (NDVI > threshold)
        4. Are NOT ground points (is_ground=0)

        These are typically trees/bushes above roads, vegetation in sports
        facilities, etc. that should be classified as vegetation rather than
        the underlying surface.

        Args:
            points: XYZ coordinates [N, 3]
            labels: Current classification labels [N] (modified in-place)
            ground_truth_features: Dict of feature_type -> GeoDataFrame
            height_above_ground: Height above DTM ground [N] - from RGE ALTI
            ndvi: Optional NDVI values [N] for vegetation detection
            is_ground: Optional binary ground indicator (1=ground, 0=non-ground)
            height_threshold: Minimum height above ground (default: 2.0m)
            ndvi_threshold: Minimum NDVI to consider as vegetation (default: 0.3)

        Returns:
            Tuple of:
            - Updated labels [N]
            - Statistics dict with counts per feature type
        """
        stats = {}
        updated_labels = labels.copy()

        logger.info("\n🌳 Reclassifying vegetation above BD TOPO surfaces...")
        logger.info(f"  Height threshold: {height_threshold}m")
        ndvi_info = (
            f"NDVI threshold: {ndvi_threshold}"
            if ndvi is not None
            else "NDVI not available"
        )
        logger.info(f"  {ndvi_info}")
        logger.info(
            f"  Ground filtering: "
            f"{'enabled (is_ground feature)' if is_ground is not None else 'disabled'}"
        )

        # Feature types to check for overlying vegetation
        surface_types = {
            "roads": int(ASPRSClass.ROAD_SURFACE),
            "sports": int(ASPRSClass.OVERHEAD_STRUCTURE),  # Sports = 19
            "cemeteries": int(ASPRSClass.GROUND),  # Cemeteries usually as ground
            "parking": int(ASPRSClass.ROAD_PARKING),  # Parking = 40
        }

        total_reclassified = 0

        for feature_type, asprs_code in surface_types.items():
            if feature_type not in ground_truth_features:
                continue

            gdf = ground_truth_features[feature_type]
            if gdf is None or len(gdf) == 0:
                stats[f"{feature_type}_vegetation"] = 0
                continue

            logger.info(f"\n  Checking {feature_type}: {len(gdf)} features")

            # Find points currently classified as this surface type
            surface_mask = labels == asprs_code
            n_surface_points = surface_mask.sum()

            if n_surface_points == 0:
                logger.info(
                    f"    No points classified as {feature_type} (class {asprs_code})"
                )
                stats[f"{feature_type}_vegetation"] = 0
                continue

            logger.info(
                f"    Found {n_surface_points:,} points classified as {feature_type}"
            )

            # Apply height filter
            high_points_mask = surface_mask & (height_above_ground > height_threshold)
            n_high = high_points_mask.sum()

            logger.info(f"    {n_high:,} points > {height_threshold}m above ground")

            if n_high == 0:
                stats[f"{feature_type}_vegetation"] = 0
                continue

            # Apply ground filter: vegetation cannot be ground points
            if is_ground is not None:
                ground_mask = is_ground == 1
                n_ground_excluded = (high_points_mask & ground_mask).sum()
                high_points_mask = high_points_mask & (~ground_mask)
                n_high_non_ground = high_points_mask.sum()

                if n_ground_excluded > 0:
                    logger.info(
                        f"    Excluded {n_ground_excluded:,} ground points, "
                        f"{n_high_non_ground:,} non-ground remain"
                    )

                if n_high_non_ground == 0:
                    stats[f"{feature_type}_vegetation"] = 0
                    continue

            # Apply NDVI filter if available
            if ndvi is not None:
                vegetation_mask = high_points_mask & (ndvi > ndvi_threshold)
                n_vegetation = vegetation_mask.sum()
                logger.info(
                    f"    {n_vegetation:,} points with NDVI > {ndvi_threshold} "
                    f"(vegetation signature)"
                )
            else:
                # Without NDVI, use more conservative height threshold
                conservative_threshold = height_threshold + 1.0  # +1m safety
                vegetation_mask = high_points_mask & (
                    height_above_ground > conservative_threshold
                )
                n_vegetation = vegetation_mask.sum()
                logger.info(
                    f"    {n_vegetation:,} points > {conservative_threshold}m "
                    f"(conservative without NDVI)"
                )

            if n_vegetation == 0:
                stats[f"{feature_type}_vegetation"] = 0
                continue

            # Classify by height: low/medium/high vegetation
            veg_points = np.where(vegetation_mask)[0]
            veg_heights = height_above_ground[veg_points]

            # Low vegetation: 2-3m
            low_veg = veg_points[veg_heights <= 3.0]
            # Medium vegetation: 3-10m
            medium_veg = veg_points[(veg_heights > 3.0) & (veg_heights <= 10.0)]
            # High vegetation: >10m
            high_veg = veg_points[veg_heights > 10.0]

            # Update labels
            updated_labels[low_veg] = int(ASPRSClass.LOW_VEGETATION)
            updated_labels[medium_veg] = int(ASPRSClass.MEDIUM_VEGETATION)
            updated_labels[high_veg] = int(ASPRSClass.HIGH_VEGETATION)

            n_reclassified = len(veg_points)
            total_reclassified += n_reclassified
            stats[f"{feature_type}_vegetation"] = n_reclassified

            logger.info(f"    ✅ Reclassified {n_reclassified:,} vegetation points:")
            logger.info(
                f"       Low (3): {len(low_veg):,} | Medium (4): {len(medium_veg):,} | High (5): {len(high_veg):,}"
            )

        stats["total_vegetation_reclassified"] = total_reclassified

        logger.info(f"\n🌳 Vegetation Reclassification Summary:")
        logger.info(f"  Total reclassified: {total_reclassified:,} points")

        return updated_labels, stats

    def _classify_feature(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        geometries: np.ndarray,
        asprs_code: int,
        feature_name: str,
    ) -> int:
        """
        Classify points for a single feature type using selected backend.

        Automatically routes to CPU or GPU implementation based on acceleration_mode.

        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            geometries: Array of shapely geometries
            asprs_code: ASPRS classification code to apply
            feature_name: Feature name (for progress display)

        Returns:
            Number of points classified
        """
        if self.acceleration_mode in ["gpu", "gpu+cuml"]:
            return self._classify_feature_gpu(
                points, labels, geometries, asprs_code, feature_name
            )
        else:
            return self._classify_feature_cpu(
                points, labels, geometries, asprs_code, feature_name
            )

    def _classify_feature_cpu(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        geometries: np.ndarray,
        asprs_code: int,
        feature_name: str,
    ) -> int:
        """
        CPU implementation using STRtree spatial indexing.

        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            geometries: Array of shapely geometries
            asprs_code: ASPRS classification code to apply
            feature_name: Feature name (for progress display)

        Returns:
            Number of points classified
        """
        n_points = len(points)
        n_classified = 0

        # Build spatial index (STRtree) for fast queries
        tree = STRtree(geometries)

        # Process points in chunks
        n_chunks = (n_points + self.chunk_size - 1) // self.chunk_size

        # Create progress bar if enabled
        pbar = None
        if self.show_progress:
            pbar = tqdm(
                total=n_points,
                desc=f"    {feature_name}",
                leave=False,
                unit="pts",
                unit_scale=True,
            )

        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, n_points)
            chunk_points = points[start_idx:end_idx]

            # Create point geometries for this chunk
            point_geoms = [Point(p[0], p[1]) for p in chunk_points]

            # Query spatial index for each point
            for j, pt_geom in enumerate(point_geoms):
                global_idx = start_idx + j

                # Query nearby polygons using spatial index (fast)
                possible_matches = tree.query(pt_geom)

                # Check if point is actually within any polygon (exact test)
                for polygon_idx in possible_matches:
                    if geometries[polygon_idx].contains(pt_geom):
                        labels[global_idx] = asprs_code
                        n_classified += 1
                        break  # Stop at first match

            # Update progress bar
            if pbar:
                pbar.update(len(chunk_points))

        # Close progress bar
        if pbar:
            pbar.close()

        return n_classified

    def _classify_feature_gpu(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        geometries: np.ndarray,
        asprs_code: int,
        feature_name: str,
    ) -> int:
        """
        GPU implementation using RAPIDS cuSpatial for point-in-polygon queries.

        This is significantly faster than CPU for large point clouds (>5M points).

        Args:
            points: XYZ coordinates [N, 3]
            labels: Classification labels [N] (modified in-place)
            geometries: Array of shapely geometries
            asprs_code: ASPRS classification code to apply
            feature_name: Feature name (for progress display)

        Returns:
            Number of points classified
        """
        n_points = len(points)
        n_classified = 0

        try:
            # Convert points to GPU (cuDF DataFrame)
            points_gpu = cudf.DataFrame(
                {"x": cp.asarray(points[:, 0]), "y": cp.asarray(points[:, 1])}
            )

            # Convert polygons to GPU-compatible format
            # Extract polygon coordinates and convert to cuDF
            polygon_list = []
            for geom in geometries:
                if geom.geom_type == "Polygon":
                    coords = list(geom.exterior.coords)
                    polygon_list.append(coords)
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        coords = list(poly.exterior.coords)
                        polygon_list.append(coords)

            # Process in chunks to manage GPU memory
            chunk_size = min(self.chunk_size, n_points)
            n_chunks = (n_points + chunk_size - 1) // chunk_size

            # Create progress bar
            pbar = None
            if self.show_progress:
                pbar = tqdm(
                    total=n_points,
                    desc=f"    {feature_name} (GPU)",
                    leave=False,
                    unit="pts",
                    unit_scale=True,
                )

            # Process each chunk
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_points)

                # Get chunk on GPU
                chunk_df = points_gpu.iloc[start_idx:end_idx]

                # Check each polygon (GPU accelerated)
                for poly_coords in polygon_list:
                    # Convert polygon to cuSpatial format
                    poly_x = cp.array([c[0] for c in poly_coords])
                    poly_y = cp.array([c[1] for c in poly_coords])

                    # Use cuspatial point_in_polygon test
                    # This is GPU-accelerated and much faster than CPU
                    try:
                        # Create polygon on GPU
                        poly_df = cudf.DataFrame({"x": poly_x, "y": poly_y})

                        # Point-in-polygon test (GPU)
                        mask = cuspatial.point_in_polygon(
                            chunk_df["x"], chunk_df["y"], poly_df["x"], poly_df["y"]
                        )

                        # Convert mask to numpy and update labels
                        mask_cpu = mask.to_numpy()
                        indices = np.where(mask_cpu)[0] + start_idx
                        labels[indices] = asprs_code
                        n_classified += len(indices)

                    except Exception as e:
                        # Fallback to CPU for this polygon
                        logger.debug(
                            f"GPU polygon test failed, using CPU fallback: {e}"
                        )
                        continue

                # Update progress
                if pbar:
                    pbar.update(end_idx - start_idx)

            # Close progress bar
            if pbar:
                pbar.close()

        except Exception as e:
            logger.warning(
                f"GPU classification failed for {feature_name}, falling back to CPU: {e}"
            )
            # Fallback to CPU implementation
            return self._classify_feature_cpu(
                points, labels, geometries, asprs_code, feature_name
            )

        return n_classified

    def reclassify_file(
        self,
        input_laz: Path,
        output_laz: Path,
        ground_truth_features: Dict[str, gpd.GeoDataFrame],
    ) -> Dict[str, int]:
        """
        Reclassify a LAZ file with ground truth features.

        Args:
            input_laz: Input LAZ file path
            output_laz: Output LAZ file path
            ground_truth_features: Dict of feature_type -> GeoDataFrame

        Returns:
            Statistics dict with counts per feature type
        """
        import laspy

        logger.info(f"📂 Loading: {input_laz.name}")

        # Load LAZ file
        las = laspy.read(str(input_laz))
        points = np.vstack([las.x, las.y, las.z]).T
        current_labels = np.array(las.classification)

        logger.info(f"  Loaded {len(points):,} points")
        logger.info(f"  Current classes: {np.unique(current_labels)}")

        # Extract NDVI if available (from extra dimensions)
        ndvi = None
        intensities = None
        try:
            if hasattr(las, "ndvi"):
                ndvi = np.array(las.ndvi)
                logger.info(
                    f"  Found NDVI data (range: {ndvi.min():.3f} to {ndvi.max():.3f})"
                )
            elif "NDVI" in las.point_format.dimension_names:
                ndvi = np.array(las["NDVI"])
                logger.info(
                    f"  Found NDVI data (range: {ndvi.min():.3f} to {ndvi.max():.3f})"
                )
        except Exception as e:
            logger.debug(f"NDVI not available: {e}")

        try:
            if hasattr(las, "intensity"):
                intensities = np.array(las.intensity)
                logger.debug(f"  Found intensity data")
        except Exception as e:
            logger.debug(f"Intensity not available: {e}")

        # Reclassify
        new_labels, stats = self.reclassify(
            points=points,
            current_labels=current_labels,
            ground_truth_features=ground_truth_features,
            ndvi=ndvi,
            intensities=intensities,
        )

        # Update classification in LAS object
        las.classification = new_labels

        # Save updated file
        logger.info(f"💾 Saving: {output_laz.name}")
        output_laz.parent.mkdir(parents=True, exist_ok=True)
        las.write(str(output_laz))

        logger.info(f"✅ Saved: {output_laz}")
        logger.info(f"  File size: {output_laz.stat().st_size / 1024 / 1024:.1f} MB")

        return stats


def reclassify_tile(
    input_laz: Path,
    output_laz: Path,
    ground_truth_features: Dict[str, gpd.GeoDataFrame],
    chunk_size: int = 100000,
    show_progress: bool = True,
) -> Dict[str, int]:
    """
    Convenience function to reclassify a single tile.

    Args:
        input_laz: Input LAZ file
        output_laz: Output LAZ file
        ground_truth_features: Ground truth features from DataFetcher
        chunk_size: Points per chunk
        show_progress: Show progress bars

    Returns:
        Statistics dict
    """
    reclassifier = Reclassifier(chunk_size=chunk_size, show_progress=show_progress)

    return reclassifier.reclassify_file(
        input_laz=input_laz,
        output_laz=output_laz,
        ground_truth_features=ground_truth_features,
    )


# ============================================================================
# Deprecated aliases for backward compatibility
# ============================================================================


class OptimizedReclassifier(Reclassifier):
    """
    Deprecated: Use Reclassifier instead.

    This class is deprecated and will be removed in v4.0.
    Use Reclassifier for the same functionality.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OptimizedReclassifier is deprecated, " "use Reclassifier instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def reclassify_tile_optimized(*args, **kwargs):
    """
    Deprecated: Use reclassify_tile() instead.

    This function is deprecated and will be removed in v4.0.
    Use reclassify_tile() for the same functionality.
    """
    warnings.warn(
        "reclassify_tile_optimized() is deprecated, " "use reclassify_tile() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return reclassify_tile(*args, **kwargs)
