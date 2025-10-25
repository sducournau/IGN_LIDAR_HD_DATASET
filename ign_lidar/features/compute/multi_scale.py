"""
Multi-scale feature computation with variance-weighted aggregation.

Implements intelligent multi-scale feature computation to suppress
artifacts at tile boundaries while maintaining feature quality.

This module provides the core multi-scale computation engine that:
1. Computes features at multiple neighborhood scales
2. Detects artifacts via variance analysis
3. Aggregates results with intelligent weighting
4. Provides adaptive scale selection

Key Components:
    - ScaleConfig: Configuration for a single scale
    - MultiScaleFeatureComputer: Main computation engine
    - Aggregation methods: weighted_average, variance_weighted, adaptive

Example:
    >>> from ign_lidar.features.compute.multi_scale import (
    ...     MultiScaleFeatureComputer, ScaleConfig
    ... )
    >>> scales = [
    ...     ScaleConfig("fine", k_neighbors=30, search_radius=1.0, weight=0.3),
    ...     ScaleConfig("medium", k_neighbors=80, search_radius=2.5, weight=0.5),
    ...     ScaleConfig("coarse", k_neighbors=150, search_radius=5.0, weight=0.2)
    ... ]
    >>> computer = MultiScaleFeatureComputer(
    ...     scales=scales,
    ...     aggregation_method="variance_weighted",
    ...     variance_penalty=2.0
    ... )
    >>> features = computer.compute_features(
    ...     points=point_cloud,
    ...     features_to_compute=["planarity", "linearity", "curvature"]
    ... )

Version: 6.2.0
Author: IGN LiDAR HD Development Team
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


@dataclass
class ScaleConfig:
    """
    Configuration for a single computation scale.

    Attributes:
        name: Human-readable scale name (e.g., "fine", "medium", "coarse")
        k_neighbors: Number of neighbors for this scale
        search_radius: Search radius in meters for this scale
        weight: Base weight for aggregation (before variance adjustment)
    """

    name: str
    k_neighbors: int
    search_radius: float
    weight: float

    def __post_init__(self):
        """Validate scale configuration."""
        if self.k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be > 0, got {self.k_neighbors}")
        if self.search_radius <= 0:
            raise ValueError(f"search_radius must be > 0, got {self.search_radius}")
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")


class MultiScaleFeatureComputer:
    """
    Compute features at multiple scales and intelligently aggregate.

    Uses variance-weighted aggregation to suppress artifacts:
    - Features with high variance get lower weight
    - Features with low variance dominate the result
    - Automatic artifact detection and suppression

    Attributes:
        scales: List of scale configurations
        aggregation_method: Method for combining multi-scale features
        variance_penalty: Penalty factor for high-variance scales
        artifact_detection: Enable artifact detection system
        artifact_variance_threshold: Variance threshold for artifact flag
        artifact_gradient_threshold: Gradient threshold for artifact flag
        adaptive_scale_selection: Enable adaptive scale per point
        reuse_kdtrees: Build KD-tree once and reuse across scales
    """

    def __init__(
        self,
        scales: List[ScaleConfig],
        aggregation_method: str = "variance_weighted",
        variance_penalty: float = 2.0,
        artifact_detection: bool = False,
        artifact_variance_threshold: float = 0.15,
        artifact_gradient_threshold: float = 0.10,
        adaptive_scale_selection: bool = False,
        complexity_threshold: float = 0.5,
        homogeneity_threshold: float = 0.8,
        reuse_kdtrees: bool = True,
        cache_scale_results: bool = True,
    ):
        """
        Initialize multi-scale feature computer.

        Args:
            scales: List of ScaleConfig objects
            aggregation_method: 'weighted_average', 'variance_weighted',
                or 'adaptive'
            variance_penalty: Higher = more penalty for high variance
            artifact_detection: Enable artifact detection
            artifact_variance_threshold: Variance threshold for artifacts
            artifact_gradient_threshold: Gradient threshold for artifacts
            adaptive_scale_selection: Enable per-point scale selection
            complexity_threshold: Threshold for complexity-based adaptation
            homogeneity_threshold: Threshold for homogeneity-based adaptation
            reuse_kdtrees: Reuse KD-tree across scales (faster)
            cache_scale_results: Cache intermediate scale results
        """
        if len(scales) < 2:
            raise ValueError("At least 2 scales required for multi-scale")

        self.scales = scales
        self.aggregation_method = aggregation_method
        self.variance_penalty = variance_penalty
        self.artifact_detection = artifact_detection
        self.artifact_variance_threshold = artifact_variance_threshold
        self.artifact_gradient_threshold = artifact_gradient_threshold
        self.adaptive_scale_selection = adaptive_scale_selection
        self.complexity_threshold = complexity_threshold
        self.homogeneity_threshold = homogeneity_threshold
        self.reuse_kdtrees = reuse_kdtrees
        self.cache_scale_results = cache_scale_results

        # Validate aggregation method
        valid_methods = ["weighted_average", "variance_weighted", "adaptive"]
        if self.aggregation_method not in valid_methods:
            raise ValueError(
                f"aggregation_method must be one of {valid_methods}, "
                f"got {self.aggregation_method}"
            )

        logger.info(
            f"Initialized MultiScaleFeatureComputer with {len(scales)} "
            f"scales and {aggregation_method} aggregation"
        )

    def compute_features(
        self,
        points: np.ndarray,
        features_to_compute: List[str],
        kdtree: Optional[cKDTree] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute features at all scales and aggregate.

        Args:
            points: Point cloud [N, 3] XYZ coordinates
            features_to_compute: List of feature names to compute
            kdtree: Optional pre-built KD-tree (for reuse)

        Returns:
            Dictionary with aggregated feature arrays

        Raises:
            ValueError: If points array is invalid
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points array [N, 3], got {points.shape}")

        n_points = points.shape[0]
        logger.info(
            f"Computing {len(features_to_compute)} features at "
            f"{len(self.scales)} scales for {n_points:,} points"
        )

        # Build KD-tree once if reusing
        if self.reuse_kdtrees and kdtree is None:
            logger.debug("Building KD-tree for all scales")
            kdtree = cKDTree(points)

        # Step 1: Compute features at each scale
        scale_features = []
        scale_variances = []

        for i, scale in enumerate(self.scales):
            logger.debug(
                f"Computing scale {i+1}/{len(self.scales)}: "
                f"{scale.name} (k={scale.k_neighbors}, "
                f"r={scale.search_radius}m)"
            )

            # Compute features for this scale
            features = self._compute_single_scale(
                points=points,
                features=features_to_compute,
                k_neighbors=scale.k_neighbors,
                search_radius=scale.search_radius,
                kdtree=kdtree,
            )
            scale_features.append(features)

            # Compute local variance for variance-based methods
            if self.aggregation_method in ["variance_weighted", "adaptive"]:
                variances = self._compute_local_variance(
                    features=features,
                    points=points,
                    kdtree=kdtree,
                    window_size=min(100, n_points // 2),
                )
                scale_variances.append(variances)

        # Step 2: Detect artifacts (optional)
        artifact_mask = None
        if self.artifact_detection:
            logger.debug("Detecting artifacts across scales")
            artifact_mask = self.detect_artifacts(
                scale_features=scale_features,
                variance_threshold=self.artifact_variance_threshold,
                gradient_threshold=self.artifact_gradient_threshold,
            )
            logger.info(
                f"Detected {artifact_mask.sum():,} / {n_points:,} "
                f"({100*artifact_mask.sum()/n_points:.1f}%) artifact points"
            )

        # Step 3: Aggregate with appropriate method
        if self.aggregation_method == "variance_weighted":
            aggregated = self._variance_weighted_aggregation(
                scale_features=scale_features, scale_variances=scale_variances
            )
        elif self.aggregation_method == "weighted_average":
            aggregated = self._simple_weighted_aggregation(scale_features)
        elif self.aggregation_method == "adaptive":
            aggregated = self._adaptive_aggregation(
                points=points,
                scale_features=scale_features,
                scale_variances=scale_variances,
            )
        else:
            raise ValueError(f"Unknown method: {self.aggregation_method}")

        logger.info("Multi-scale feature computation complete")
        return aggregated

    def _compute_single_scale(
        self,
        points: np.ndarray,
        features: List[str],
        k_neighbors: int,
        search_radius: float,
        kdtree: Optional[cKDTree] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute features at a single scale using existing backend.

        This delegates to the existing feature computation functions
        from ign_lidar.features.compute.geometric.

        Args:
            points: Point cloud [N, 3]
            features: List of feature names
            k_neighbors: Number of neighbors
            search_radius: Search radius in meters
            kdtree: Optional pre-built KD-tree

        Returns:
            Dictionary with feature arrays for this scale
        """
        n_points = len(points)

        # Protect against k_neighbors > n_points
        k_neighbors = min(k_neighbors, n_points - 1)
        if k_neighbors < 3:
            logger.warning(
                f"Too few points ({n_points}) for k_neighbors={k_neighbors}, "
                "returning zero features"
            )
            return {fname: np.zeros(n_points) for fname in features}

        # Import existing feature computation functions and utilities
        from .utils import compute_covariance_matrix
        from .eigenvalues import (
            compute_linearity,
            compute_planarity,
            compute_sphericity,
        )
        from .architectural import (
            compute_verticality,
            compute_horizontality,
        )

        # Build KD-tree if not provided
        if kdtree is None:
            kdtree = cKDTree(points)

        n_points = len(points)

        # Query neighbors using radius search (more artifact-resistant)
        # Then ensure minimum k neighbors
        neighbors_indices = []
        for i in range(n_points):
            # Radius search
            indices = kdtree.query_ball_point(points[i], search_radius)

            # Ensure minimum k neighbors
            if len(indices) < k_neighbors:
                # Fall back to k-NN
                _, knn_idx = kdtree.query(points[i], k=k_neighbors)
                indices = (
                    knn_idx.tolist() if hasattr(knn_idx, "tolist") else list(knn_idx)
                )

            neighbors_indices.append(indices)

        # Compute eigenvalues and normals for all points
        eigenvalues = np.zeros((n_points, 3), dtype=np.float32)
        normals = np.zeros((n_points, 3), dtype=np.float32)

        for i, neighbors_idx in enumerate(neighbors_indices):
            if len(neighbors_idx) < 3:
                eigenvalues[i] = [1.0, 0.0, 0.0]
                normals[i] = [0.0, 0.0, 1.0]
                continue

            neighbors = points[neighbors_idx]
            cov = compute_covariance_matrix(neighbors)

            # Compute eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Sort in descending order
            idx = np.argsort(eigvals)[::-1]
            eigenvalues[i] = eigvals[idx]

            # Normal is eigenvector of smallest eigenvalue
            normals[i] = eigvecs[:, idx[2]]

        # Compute requested features using eigenvalues and normals
        result = {}

        if "normals" in features:
            result["normals"] = normals

        if "linearity" in features:
            result["linearity"] = compute_linearity(eigenvalues)

        if "planarity" in features:
            result["planarity"] = compute_planarity(eigenvalues)

        if "sphericity" in features:
            result["sphericity"] = compute_sphericity(eigenvalues)

        if "verticality" in features:
            result["verticality"] = compute_verticality(normals)

        if "horizontality" in features:
            result["horizontality"] = compute_horizontality(normals)

        # Curvature (change of curvature from eigenvalues)
        if "curvature" in features:
            sum_eig = eigenvalues.sum(axis=1)
            result["curvature"] = (eigenvalues[:, 2] / (sum_eig + 1e-10)).astype(
                np.float32
            )

        return result

    def _compute_local_variance(
        self,
        features: Dict[str, np.ndarray],
        points: Optional[np.ndarray] = None,
        kdtree: Optional[cKDTree] = None,
        window_size: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Compute local variance of features using spatial neighbors.

        High local variance indicates potential artifacts or noise.

        Args:
            features: Dictionary of feature arrays
            points: Point cloud [N, 3] (optional, for spatial variance)
            kdtree: Precomputed KD-tree (optional)
            window_size: Number of neighbors for variance window

        Returns:
            Dictionary with variance arrays for each feature
        """
        variances = {}

        for feature_name, feature_values in features.items():
            if feature_name == "normals":
                # Skip normals (3D, different handling)
                continue

            n = len(feature_values)
            var = np.zeros(n)

            # Use spatial neighbors if available
            if points is not None and points.shape[0] == n:
                tree = kdtree if kdtree is not None else cKDTree(points)

                # Query neighbors for each point
                for i in range(n):
                    k = min(window_size, n)
                    _, neighbor_idxs = tree.query(points[i], k=k)
                    neighborhood_values = feature_values[neighbor_idxs]
                    var[i] = np.var(neighborhood_values)
            else:
                # Fall back to sequential window (less accurate)
                half_window = window_size // 2
                for i in range(n):
                    start = max(0, i - half_window)
                    end = min(n, i + half_window + 1)
                    window = feature_values[start:end]
                    var[i] = np.var(window)

            variances[feature_name] = var

        return variances

    def _variance_weighted_aggregation(
        self,
        scale_features: List[Dict[str, np.ndarray]],
        scale_variances: List[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate features using variance-weighted averaging.

        Formula: weight_i = base_weight / (1 + penalty * variance_i)

        High-variance features get lower weight, suppressing artifacts.

        Args:
            scale_features: List of feature dicts (one per scale)
            scale_variances: List of variance dicts (one per scale)

        Returns:
            Aggregated feature dictionary
        """
        result: Dict[str, np.ndarray] = {}
        feature_names: Set[str] = set()

        # Collect all feature names
        for scale_feat in scale_features:
            feature_names.update(scale_feat.keys())

        # Aggregate each feature
        for feature_name in feature_names:
            if feature_name == "normals":
                # Special handling for 3D normals
                continue

            # Stack feature values and variances from all scales
            values_list = []
            weights_list = []

            for i, (scale_feat, scale_var) in enumerate(
                zip(scale_features, scale_variances)
            ):
                if feature_name not in scale_feat:
                    continue

                values = scale_feat[feature_name]
                base_weight = self.scales[i].weight

                # Compute variance-adjusted weights
                if feature_name in scale_var:
                    variance = scale_var[feature_name]
                    # Down-weight high-variance features
                    penalty_term = self.variance_penalty * variance
                    adj_weight = base_weight / (1.0 + penalty_term)
                else:
                    adj_weight = base_weight

                values_list.append(values)
                weights_list.append(adj_weight)

            # Weighted average across scales
            # Shape: [n_scales, n_points]
            values_stack = np.stack(values_list, axis=0)
            weights_stack = np.stack(weights_list, axis=0)

            # Normalize weights
            weights_sum = weights_stack.sum(axis=0)
            weights_normalized = weights_stack / (weights_sum + 1e-10)

            # Compute weighted average
            weighted_values = values_stack * weights_normalized
            result[feature_name] = weighted_values.sum(axis=0)

        return result

    def _simple_weighted_aggregation(
        self, scale_features: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Simple weighted average using base weights only.

        Args:
            scale_features: List of feature dicts (one per scale)

        Returns:
            Aggregated feature dictionary
        """
        result: Dict[str, np.ndarray] = {}
        feature_names: Set[str] = set()

        for scale_feat in scale_features:
            feature_names.update(scale_feat.keys())

        for feature_name in feature_names:
            if feature_name == "normals":
                continue

            values_list = []
            weights_list = []

            for i, scale_feat in enumerate(scale_features):
                if feature_name in scale_feat:
                    values_list.append(scale_feat[feature_name])
                    weights_list.append(self.scales[i].weight)

            values_stack = np.stack(values_list, axis=0)
            weights = np.array(weights_list).reshape(-1, 1)

            # Normalize weights
            weights_sum = weights.sum()
            weights_normalized = weights / weights_sum

            # Weighted average
            weighted_values = values_stack * weights_normalized
            result[feature_name] = weighted_values.sum(axis=0)

        return result

    def _adaptive_aggregation(
        self,
        points: np.ndarray,
        scale_features: List[Dict[str, np.ndarray]],
        scale_variances: List[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """
        Adaptive aggregation: select best scale per point based on local
        geometry.

        Strategy:
        - Complex geometry (edges, corners): Use finer scales for detail
        - Homogeneous regions (planes, smooth surfaces): Use coarser scales
          for stability
        - Artifact-prone areas: Prefer scales with lower variance

        The method computes a "complexity score" from planarity/linearity
        at the finest scale, then selects the optimal scale per point:
        - High complexity (edges) → fine scale
        - Low complexity (planes) → coarse scale
        - High variance anywhere → down-weight that scale

        Args:
            points: Point cloud [N, 3]
            scale_features: List of feature dicts from each scale
            scale_variances: List of variance dicts for each scale

        Returns:
            Aggregated feature dictionary with adaptive per-point scale
            selection
        """
        n_points = len(points)
        n_scales = len(scale_features)

        if n_scales == 0:
            raise ValueError("No scale features provided")

        # Step 1: Compute local geometry complexity from finest scale
        # Complexity = 1 - planarity (high for edges, low for planes)
        finest_features = scale_features[0]
        if "planarity" in finest_features:
            complexity = 1.0 - finest_features["planarity"]
        elif "linearity" in finest_features:
            # Use linearity as proxy if planarity not available
            complexity = finest_features["linearity"]
        else:
            # Fall back to uniform complexity if no geometric features
            logger.warning(
                "No planarity/linearity in features, " "using uniform complexity"
            )
            complexity = np.ones(n_points) * 0.5

        # Ensure complexity is normalized [0, 1]
        complexity = np.clip(complexity, 0.0, 1.0)

        # Step 2: Compute per-point scale preferences based on complexity
        # High complexity → prefer fine scales (index 0)
        # Low complexity → prefer coarse scales (index n-1)
        scale_preferences = np.zeros((n_points, n_scales), dtype=np.float32)

        for scale_idx in range(n_scales):
            # Normalized scale position: 0 (fine) to 1 (coarse)
            if n_scales > 1:
                scale_position = scale_idx / (n_scales - 1)
            else:
                scale_position = 0.5

            # Preference based on complexity-scale match
            # High complexity matches low scale_position (fine)
            # Low complexity matches high scale_position (coarse)
            match_quality = 1.0 - np.abs(complexity - scale_position)

            # Start with base weight from scale config
            base_weight = self.scales[scale_idx].weight
            scale_preferences[:, scale_idx] = base_weight * match_quality

        # Step 3: Adjust preferences by variance (down-weight high-var)
        result = {}
        all_features = set()
        for feat_dict in scale_features:
            all_features.update(feat_dict.keys())

        for feature_name in all_features:
            # Collect features and variances for this feature
            feature_values = []
            feature_vars = []

            for scale_idx in range(n_scales):
                if feature_name in scale_features[scale_idx]:
                    feat_val = scale_features[scale_idx][feature_name]
                    feature_values.append(feat_val)
                    if feature_name in scale_variances[scale_idx]:
                        var_val = scale_variances[scale_idx][feature_name]
                        feature_vars.append(var_val)
                    else:
                        feature_vars.append(np.zeros(n_points))
                else:
                    # Scale doesn't have this feature, use NaN
                    feature_values.append(np.full(n_points, np.nan))
                    feature_vars.append(np.full(n_points, np.nan))

            feature_array = np.stack(feature_values, axis=1)  # [N, n_scales]
            variance_array = np.stack(feature_vars, axis=1)  # [N, n_scales]

            # Adjust scale preferences by variance
            var_penalty = 1.0 / (1.0 + self.variance_penalty * variance_array)
            adjusted_preferences = scale_preferences * var_penalty

            # Handle NaN features (scales that don't compute this feature)
            valid_mask = ~np.isnan(feature_array)
            adjusted_preferences = np.where(valid_mask, adjusted_preferences, 0.0)

            # Normalize weights to sum to 1.0 per point
            weight_sum = adjusted_preferences.sum(axis=1, keepdims=True)
            # Avoid div by 0
            weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
            normalized_weights = adjusted_preferences / weight_sum

            # Weighted aggregation
            weighted_features = feature_array * normalized_weights
            result[feature_name] = np.nansum(weighted_features, axis=1)

        logger.info(
            f"✓ Adaptive aggregation complete | "
            f"features={len(result)} | "
            f"complexity_range="
            f"[{complexity.min():.2f}, {complexity.max():.2f}]"
        )

        return result

    def detect_artifacts(
        self,
        scale_features: List[Dict[str, np.ndarray]],
        variance_threshold: float = 0.15,
        gradient_threshold: float = 0.10,
    ) -> np.ndarray:
        """
        Detect artifact points by comparing features across scales.

        Artifact detection criteria:
        1. High variance across scales (inconsistent measurements)
        2. High spatial gradient (rapid changes, scan lines)

        Args:
            scale_features: List of feature dicts from different scales
            variance_threshold: Variance threshold for artifact flag
            gradient_threshold: Gradient threshold for artifact flag

        Returns:
            Boolean array [N] where True = artifact detected
        """
        if len(scale_features) < 2:
            raise ValueError("Need at least 2 scales for artifact detection")

        n_points = len(next(iter(scale_features[0].values())))
        artifact_mask = np.zeros(n_points, dtype=bool)

        # For each feature, check variance across scales
        feature_names = set(scale_features[0].keys()) - {"normals"}

        for feature_name in feature_names:
            # Stack feature values from all scales
            values_list = []
            for scale_feat in scale_features:
                if feature_name in scale_feat:
                    values_list.append(scale_feat[feature_name])

            if len(values_list) < 2:
                continue

            # Stack: [n_scales, n_points]
            values_stack = np.stack(values_list, axis=0)

            # Compute variance across scales for each point
            cross_scale_variance = np.var(values_stack, axis=0)

            # Flag points with high cross-scale variance
            artifact_mask |= cross_scale_variance > variance_threshold

        return artifact_mask
