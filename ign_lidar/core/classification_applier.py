"""
Classification application component for LiDAR processing.

This module handles applying ground truth classification to point clouds,
including BD TOPO data integration and ASPRS classification rules.

Extracted from LiDARProcessor as part of refactoring (v3.4.0).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ClassificationApplier:
    """
    Apply ground truth classification to point clouds.

    Handles:
    - Ground truth data fetching (BD TOPO, BD Forêt, RPG, Cadastre)
    - Optimized ground truth classification (GPU-accelerated)
    - Unified classifier with comprehensive strategy
    - Classification change tracking and reporting
    - ASPRS → LOD mapping

    This class was extracted from LiDARProcessor to separate classification
    concerns from tile processing.
    """

    def __init__(
        self,
        config: DictConfig,
        data_fetcher: Optional[Any] = None,
        class_mapping: Optional[Dict[int, int]] = None,
        default_class: int = 0,
    ):
        """
        Initialize classification applier.

        Args:
            config: Configuration containing classification parameters
            data_fetcher: Data fetcher for ground truth (BD TOPO, etc.)
            class_mapping: ASPRS → LOD class mapping (None = ASPRS mode)
            default_class: Default class for unmapped codes
        """
        self.config = config
        self.data_fetcher = data_fetcher
        self.class_mapping = class_mapping
        self.default_class = default_class

        # Classification configuration
        self.use_optimized_gt = config.processor.get("use_optimized_ground_truth", True)
        self.building_mode = config.processor.get("building_detection_mode", "asprs")
        self.transport_mode = config.processor.get(
            "transport_detection_mode", "asprs_extended"
        )

        # Ground truth configuration
        self.chunk_size = config.ground_truth.get("chunk_size", 2_000_000)
        self.use_cache = (
            config.get("data_sources", {}).get("bd_topo", {}).get("cache_enabled", True)
        )

        # BD TOPO parameters for road/railway filtering
        bd_topo_params = (
            config.get("data_sources", {}).get("bd_topo", {}).get("parameters", {})
        )
        self.road_buffer_tolerance = bd_topo_params.get("road_buffer_tolerance", 0.5)

        # NDVI configuration
        self.use_ndvi = config.features.get("compute_ndvi", False)
        self.ndvi_vegetation_threshold = 0.3
        self.ndvi_building_threshold = 0.15

        logger.debug(
            f"Initialized ClassificationApplier: "
            f"optimized={self.use_optimized_gt}, "
            f"building_mode={self.building_mode}, "
            f"transport_mode={self.transport_mode}"
        )

    def apply_class_mapping(self, classification: np.ndarray) -> np.ndarray:
        """
        Apply ASPRS → LOD class mapping.

        Args:
            classification: ASPRS classification codes

        Returns:
            Mapped classification codes (LOD2/LOD3 or ASPRS)
        """
        if self.class_mapping is not None:
            # LOD2 or LOD3: Apply class mapping
            labels = np.array(
                [self.class_mapping.get(c, self.default_class) for c in classification],
                dtype=np.uint8,
            )
            logger.debug(
                f"Applied class mapping: {len(np.unique(classification))} → "
                f"{len(np.unique(labels))} classes"
            )
        else:
            # ASPRS mode: Use classification codes directly
            labels = classification.astype(np.uint8)
            logger.debug(f"ASPRS mode: using {len(np.unique(labels))} classes")

        return labels

    def apply_ground_truth(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        bbox: tuple,
    ) -> np.ndarray:
        """
        Apply ground truth classification from BD TOPO and other sources.

        Args:
            points: Point cloud [N, 3] (X, Y, Z coordinates)
            labels: Current classification labels [N]
            features: Computed features dictionary
            bbox: Bounding box (xmin, ymin, xmax, ymax)

        Returns:
            Updated classification labels [N]
        """
        if self.data_fetcher is None:
            logger.debug("No data fetcher available, skipping ground truth")
            return labels

        try:
            # Fetch ground truth data
            logger.info(f"  📍 Fetching ground truth for bbox: {bbox}")
            gt_data = self.data_fetcher.fetch_all(bbox=bbox, use_cache=self.use_cache)

            if gt_data is None or "ground_truth" not in gt_data:
                logger.info("  ℹ️  No ground truth data available for this tile")
                return labels

            ground_truth_features = gt_data["ground_truth"]

            # Log available ground truth features
            available_features = [
                k
                for k, v in ground_truth_features.items()
                if v is not None and len(v) > 0
            ]

            if not available_features:
                logger.warning("  ⚠️  No ground truth features found for this tile!")
                return labels

            logger.info(f"  🗺️  Available ground truth: {', '.join(available_features)}")
            for feat_type in available_features:
                logger.info(
                    f"      - {feat_type}: "
                    f"{len(ground_truth_features[feat_type])} features"
                )

            # Apply ground truth using appropriate method
            if self.use_optimized_gt:
                labels = self._apply_optimized_ground_truth(
                    points, labels, features, ground_truth_features
                )
            else:
                labels = self._apply_classifier(
                    points, labels, features, ground_truth_features
                )

            return labels

        except Exception as e:
            logger.error(f"  ❌ Ground truth classification failed: {e}")
            logger.exception(e)
            return labels

    def _apply_optimized_ground_truth(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth_features: Dict[str, Any],
    ) -> np.ndarray:
        """
        Apply ground truth using optimized GPU-accelerated method.

        Args:
            points: Point cloud [N, 3]
            labels: Current labels [N]
            features: Computed features
            ground_truth_features: Ground truth feature geometries

        Returns:
            Updated labels [N]
        """
        from ..optimization.ground_truth import GroundTruthOptimizer

        logger.info(f"  🗺️  Applying ground truth (optimized, GPU-accelerated)")

        # Store original for comparison
        labels_before = labels.copy()

        # Create optimizer with auto-selection
        optimizer = GroundTruthOptimizer(
            force_method="auto",  # Auto-select: GPU chunked, GPU, or CPU STRtree
            gpu_chunk_size=self.chunk_size,
            verbose=True,
        )

        # Apply ground truth classification
        logger.info(f"  🔄 Classifying {len(points):,} points...")
        labels = optimizer.label_points(
            points=points,
            ground_truth_features=ground_truth_features,
            label_priority=None,  # Use default priority
            ndvi=features.get("ndvi"),
            use_ndvi_refinement=self.use_ndvi,
            ndvi_vegetation_threshold=self.ndvi_vegetation_threshold,
            ndvi_building_threshold=self.ndvi_building_threshold,
        )

        # Log results
        self._log_classification_changes(labels_before, labels)

        return labels

    def _apply_classifier(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth_features: Dict[str, Any],
    ) -> np.ndarray:
        """
        Apply ground truth using Classifier.

        Args:
            points: Point cloud [N, 3]
            labels: Current labels [N]
            features: Computed features
            ground_truth_features: Ground truth feature geometries

        Returns:
            Updated labels [N]
        """
        from .classification import ClassificationStrategy, Classifier

        logger.info(f"  🗺️  Applying ground truth (Classifier)")
        logger.info(f"      Building mode: {self.building_mode}")
        logger.info(f"      Transport mode: {self.transport_mode}")

        # Store original for comparison
        labels_before = labels.copy()

        # Create classifier
        classifier = Classifier(
            strategy=ClassificationStrategy.COMPREHENSIVE,
            use_ground_truth=True,
            use_ndvi=self.use_ndvi,
            use_geometric=True,
            building_detection_mode=self.building_mode,
            transport_detection_mode=self.transport_mode,
            road_buffer_tolerance=self.road_buffer_tolerance,
        )

        # Apply classification
        result = classifier.classify(
            points=points,
            features=features,
            ground_truth=ground_truth_features,
            initial_classification=labels,
        )

        # Extract labels from ClassificationResult
        labels = result.labels

        # Log results
        self._log_classification_changes(labels_before, labels)

        return labels

    def _log_classification_changes(
        self, labels_before: np.ndarray, labels_after: np.ndarray
    ):
        """
        Log classification changes and new classes found.

        Args:
            labels_before: Original labels
            labels_after: Updated labels
        """
        # Count changes
        n_changed = np.sum(labels_after != labels_before)
        pct_changed = (n_changed / len(labels_after)) * 100

        logger.info(
            f"  ✅ Ground truth applied: {n_changed:,} points changed "
            f"({pct_changed:.2f}%)"
        )

        # Log new classes
        new_classes = np.unique(labels_after)
        has_roads = 11 in new_classes
        has_rails = 10 in new_classes
        has_parking = 40 in new_classes
        has_sports = 41 in new_classes
        has_cemetery = 42 in new_classes
        has_power = 43 in new_classes

        if has_roads or has_rails:
            logger.info(f"      🛣️  Roads (11): {'✅' if has_roads else '❌'}")
            logger.info(f"      🚂 Railways (10): {'✅' if has_rails else '❌'}")

        if has_parking or has_sports or has_cemetery or has_power:
            logger.info(f"      🅿️  Parking (40): {'✅' if has_parking else '❌'}")
            logger.info(f"      ⚽ Sports (41): {'✅' if has_sports else '❌'}")
            logger.info(f"      🪦 Cemetery (42): {'✅' if has_cemetery else '❌'}")
            logger.info(f"      ⚡ Power Lines (43): {'✅' if has_power else '❌'}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classification applier statistics.

        Returns:
            Dictionary with configuration and stats
        """
        return {
            "has_data_fetcher": self.data_fetcher is not None,
            "use_optimized_gt": self.use_optimized_gt,
            "building_mode": self.building_mode,
            "transport_mode": self.transport_mode,
            "chunk_size": self.chunk_size,
            "use_cache": self.use_cache,
            "use_ndvi": self.use_ndvi,
            "has_class_mapping": self.class_mapping is not None,
        }
