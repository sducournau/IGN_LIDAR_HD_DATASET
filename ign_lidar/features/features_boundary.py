"""
Boundary-Aware Feature Computation Module for IGN LiDAR HD v2.0

This module implements feature computation with cross-tile neighborhood support,
enabling seamless feature quality at tile boundaries.

Key Features:
- Detect boundary points automatically
- Extend neighborhoods across tile boundaries
- Compute geometric features with complete context
- Return features only for core points

Author: IGN LiDAR HD Team
Date: October 7, 2025
Sprint: 3 (Tile Stitching - Phase 3.2)
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.spatial import KDTree

# Import core feature implementations
from ..features.core import (
    compute_curvature as core_compute_curvature,
    compute_linearity,
    compute_planarity,
    compute_sphericity,
    compute_anisotropy,
    compute_verticality as core_compute_verticality,
)

logger = logging.getLogger(__name__)


class BoundaryAwareFeatureComputer:
    """
    Compute geometric features with cross-tile neighborhood support.
    
    This class enables seamless feature computation at tile boundaries by:
    1. Detecting points near tile edges
    2. Using combined (core + buffer) point cloud for neighborhoods
    3. Computing features with complete spatial context
    4. Returning features only for core points
    
    Example:
        >>> computer = BoundaryAwareFeatureComputer(
        ...     k_neighbors=20,
        ...     boundary_threshold=10.0
        ... )
        >>> features = computer.compute_features(
        ...     core_points=core_xyz,
        ...     buffer_points=buffer_xyz,
        ...     tile_bounds=(xmin, ymin, xmax, ymax)
        ... )
    """
    
    def __init__(
        self,
        k_neighbors: int = 20,
        boundary_threshold: float = 10.0,
        compute_normals: bool = True,
        compute_curvature: bool = True,
        compute_planarity: bool = True,
        compute_verticality: bool = True
    ):
        """
        Initialize BoundaryAwareFeatureComputer.
        
        Args:
            k_neighbors: Number of neighbors for feature computation
            boundary_threshold: Distance from boundary to consider a point
                               as "boundary point" (in meters)
            compute_normals: If True, compute normal vectors
            compute_curvature: If True, compute curvature
            compute_planarity: If True, compute planarity/linearity/sphericity
            compute_verticality: If True, compute verticality (wall detection)
        """
        self.k_neighbors = k_neighbors
        self.boundary_threshold = boundary_threshold
        self.compute_normals = compute_normals
        self.compute_curvature = compute_curvature
        self.compute_planarity = compute_planarity
        self.compute_verticality = compute_verticality
        
        logger.info(
            f"BoundaryAwareFeatureComputer initialized "
            f"(k={k_neighbors}, boundary_threshold={boundary_threshold}m)"
        )
    
    def compute_features(
        self,
        core_points: np.ndarray,
        buffer_points: Optional[np.ndarray] = None,
        tile_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute geometric features with boundary awareness.
        
        Args:
            core_points: (N, 3) XYZ coordinates of core tile points
            buffer_points: (M, 3) XYZ coordinates of buffer zone points
                          If None, no boundary awareness (standard computation)
            tile_bounds: (xmin, ymin, xmax, ymax) Core tile boundaries
                        Required if buffer_points provided
        
        Returns:
            Dictionary with computed features:
                - 'normals': (N, 3) Normal vectors
                - 'curvature': (N,) Curvature values [0, 1]
                - 'planarity': (N,) Planarity values [0, 1]
                - 'linearity': (N,) Linearity values [0, 1]
                - 'sphericity': (N,) Sphericity values [0, 1]
                - 'anisotropy': (N,) Anisotropy values [0, 1]
                - 'roughness': (N,) Roughness values [0, 1]
                - 'density': (N,) Density values [0, 1000]
                - 'verticality': (N,) Verticality values [0, 1]
                - 'horizontality': (N,) Horizontality values [0, 1]
                - 'boundary_mask': (N,) Boolean mask (True = near boundary)
                - 'num_boundary_points': int
        
        Raises:
            ValueError: If buffer_points provided but tile_bounds is None
        """
        num_core = len(core_points)
        
        # Combine core + buffer if available
        if buffer_points is not None and len(buffer_points) > 0:
            if tile_bounds is None:
                raise ValueError(
                    "tile_bounds required when buffer_points provided"
                )
            
            combined_points = np.vstack([core_points, buffer_points])
            logger.info(
                f"Computing features with boundary awareness: "
                f"{num_core} core + {len(buffer_points)} buffer = "
                f"{len(combined_points)} total points"
            )
            
            # Detect boundary points
            boundary_mask = self._detect_boundary_points(
                core_points, tile_bounds
            )
            num_boundary = np.sum(boundary_mask)
            
            logger.info(
                f"Detected {num_boundary}/{num_core} points near boundaries "
                f"({100*num_boundary/num_core:.1f}%)"
            )
        else:
            # No buffer - standard computation
            combined_points = core_points
            boundary_mask = np.zeros(num_core, dtype=bool)
            num_boundary = 0
            
            logger.info(
                f"Computing features without boundary awareness "
                f"({num_core} points)"
            )
        
        # Build spatial index on combined points
        logger.debug("Building KDTree spatial index...")
        tree = KDTree(combined_points)
        
        # Query neighbors for core points
        # Cap k at number of available points
        k_actual = min(self.k_neighbors, len(combined_points))
        logger.debug(f"Querying {k_actual} neighbors...")
        distances, indices = tree.query(
            core_points,
            k=k_actual,
            workers=-1  # Use all CPU cores
        )
        
        # Initialize results dictionary
        results = {
            'boundary_mask': boundary_mask,
            'num_boundary_points': num_boundary
        }
        
        # Compute geometric features
        if self.compute_normals:
            logger.debug("Computing normals...")
            normals, eigenvalues = self._compute_normals_and_eigenvalues(
                core_points, combined_points, indices
            )
            results['normals'] = normals
            results['eigenvalues'] = eigenvalues
            
            if self.compute_curvature:
                logger.debug("Computing curvature...")
                results['curvature'] = self._compute_curvature(eigenvalues)
            
            if self.compute_planarity:
                logger.debug("Computing planarity features...")
                planarity_features = self._compute_planarity_features(
                    eigenvalues
                )
                results.update(planarity_features)
            
            # NEW: Compute density
            logger.debug("Computing density...")
            results['density'] = self._compute_density(distances)
            
            if self.compute_verticality:
                logger.debug("Computing verticality and horizontality...")
                verticality, horizontality = self._compute_verticality_and_horizontality(normals)
                results['verticality'] = verticality
                results['horizontality'] = horizontality
        
        # Validate features for artifacts (only if we have boundary points)
        if num_boundary > 0:
            logger.debug("Validating features for artifacts...")
            results = self._validate_features(results, boundary_mask)
        
        logger.info(
            f"Feature computation complete: "
            f"{len(results)-2} feature types computed"
        )
        
        return results
    
    def _detect_boundary_points(
        self,
        points: np.ndarray,
        tile_bounds: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Detect points near tile boundaries.
        
        Args:
            points: (N, 3) XYZ coordinates
            tile_bounds: (xmin, ymin, xmax, ymax)
        
        Returns:
            (N,) Boolean mask: True = near boundary
        """
        xmin, ymin, xmax, ymax = tile_bounds
        x, y = points[:, 0], points[:, 1]
        
        # Distance to each boundary
        dist_to_left = x - xmin
        dist_to_right = xmax - x
        dist_to_bottom = y - ymin
        dist_to_top = ymax - y
        
        # Minimum distance to any boundary
        min_dist = np.minimum.reduce([
            dist_to_left, dist_to_right, dist_to_bottom, dist_to_top
        ])
        
        # Points within threshold
        near_boundary = min_dist <= self.boundary_threshold
        
        return near_boundary
    
    def _compute_normals_and_eigenvalues(
        self,
        query_points: np.ndarray,
        all_points: np.ndarray,
        neighbor_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normal vectors and eigenvalues from local neighborhoods.
        
        ✅ OPTIMIZED: Fully vectorized implementation using einsum.
        This provides 10-100× speedup over the old Python loop approach.
        
        Args:
            query_points: (N, 3) Points to compute normals for
            all_points: (N+M, 3) All points (core + buffer)
            neighbor_indices: (N, K) Neighbor indices in all_points
        
        Returns:
            normals: (N, 3) Normal vectors (unit vectors)
            eigenvalues: (N, 3) Eigenvalues (sorted descending: λ1 ≥ λ2 ≥ λ3)
        """
        num_points = len(query_points)
        k = neighbor_indices.shape[1]
        
        # ✅ VECTORIZED: Gather all neighbors at once [N, k, 3]
        neighbors = all_points[neighbor_indices]
        
        # ✅ VECTORIZED: Center all neighborhoods [N, k, 3]
        centroids = neighbors.mean(axis=1, keepdims=True)  # [N, 1, 3]
        centered = neighbors - centroids
        
        # ✅ VECTORIZED: Compute ALL covariance matrices at once [N, 3, 3]
        cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / k
        
        # Add small regularization for numerical stability
        cov_matrices = cov_matrices + 1e-8 * np.eye(3)
        
        # ✅ VECTORIZED: Batch eigendecomposition [N, 3, 3]
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
        
        # ✅ VECTORIZED: Sort descending [N, 3]
        eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]
        
        # ✅ VECTORIZED: Extract normals (smallest eigenvalue's eigenvector) [N, 3]
        # Since eigh returns ascending order, the first column is the smallest
        normals = eigenvectors[:, :, 0]
        
        # ✅ VECTORIZED: Normalize all normals at once
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        normals = normals / norms
        
        # ✅ VECTORIZED: Orient all normals upward (positive Z component)
        flip_mask = normals[:, 2] < 0
        normals[flip_mask] = -normals[flip_mask]
        
        return normals, eigenvalues
    
    def _compute_curvature(self, eigenvalues: np.ndarray) -> np.ndarray:
        """
        Compute curvature from eigenvalues.
        
        Args:
            eigenvalues: (N, 3) Eigenvalues (λ1, λ2, λ3)
        
        Returns:
            (N,) Curvature values [0, 1]
        
        Note:
            Uses core implementation from ign_lidar.features.core
        """
        return core_compute_curvature(eigenvalues)
    
    def _compute_planarity_features(
        self,
        eigenvalues: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute geometric features from eigenvalues.
        
        Features (all normalized by λ1, largest eigenvalue):
        - Planarity:   (λ2 - λ3) / λ1  → 1 for planes, 0 otherwise
        - Linearity:   (λ1 - λ2) / λ1  → 1 for lines, 0 otherwise
        - Sphericity:  λ3 / λ1         → 1 for spheres, 0 otherwise
        - Anisotropy:  (λ1 - λ3) / λ1  → general directionality
        - Roughness:   λ3 / Σλ         → surface roughness
        
        Args:
            eigenvalues: (N, 3) Eigenvalues (λ1 ≥ λ2 ≥ λ3)
        
        Returns:
            Dictionary with planarity, linearity, sphericity, anisotropy, roughness
        
        Note:
            Uses core implementations from ign_lidar.features.core
        """
        # Use core functions for individual features
        planarity = compute_planarity(eigenvalues)
        linearity = compute_linearity(eigenvalues)
        sphericity = compute_sphericity(eigenvalues)
        anisotropy = compute_anisotropy(eigenvalues)
        
        # Roughness: λ3 / Σλ (same as curvature)
        roughness = core_compute_curvature(eigenvalues)
        
        return {
            'planarity': planarity,
            'linearity': linearity,
            'sphericity': sphericity,
            'anisotropy': anisotropy,
            'roughness': roughness,
        }
    
    def _compute_density(
        self,
        distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute local point density from neighbor distances.
        
        Density = 1 / mean_distance (inverse of average distance to neighbors)
        Capped at 1000 to avoid extreme values in dense clusters.
        
        Args:
            distances: (N, k) Distances to k nearest neighbors
        
        Returns:
            (N,) Density values [0, 1000]
        """
        # Use distances to all neighbors except self (first column)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # Density = inverse of distance (with safety epsilon)
        density = 1.0 / (mean_distances + 1e-8)
        
        # Cap at reasonable maximum to avoid extreme values
        density = np.clip(density, 0.0, 1000.0)
        
        return density.astype(np.float32)
    
    def _compute_verticality_and_horizontality(
        self, 
        normals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute verticality and horizontality from normal vectors.
        
        Args:
            normals: (N, 3) Normal vectors
        
        Returns:
            Tuple of (verticality, horizontality) arrays, both [0, 1]
        
        Note:
            Uses core implementation from ign_lidar.features.core
        """
        verticality = core_compute_verticality(normals)
        # Horizontality is the complement: |nz|
        horizontality = np.abs(normals[:, 2]).astype(np.float32)
        return verticality, horizontality
    
    def _validate_features(
        self,
        features: Dict[str, np.ndarray],
        boundary_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Validate features for artifacts and anomalies.
        
        Common artifacts at tile boundaries:
        - Dash/line patterns (high linearity with low variation)
        - Discontinuous planes (planar features with sharp transitions)
        - Invalid values (NaN, Inf, out of range)
        
        Strategy:
        1. Check for invalid values (NaN, Inf)
        2. Detect artifact patterns in boundary regions
        3. Drop problematic features if artifacts detected
        
        Args:
            features: Dictionary of computed features
            boundary_mask: Boolean mask indicating boundary points
        
        Returns:
            Validated features (problematic ones removed)
        """
        validated = features.copy()
        num_boundary = np.sum(boundary_mask)
        
        if num_boundary == 0:
            return validated  # No boundary points, no validation needed
        
        # Features to validate
        feature_names = ['planarity', 'linearity', 'sphericity', 'verticality']
        features_to_drop = []
        
        for fname in feature_names:
            if fname not in validated:
                continue
            
            feature_values = validated[fname]
            boundary_values = feature_values[boundary_mask]
            
            # 1. Check for invalid values
            has_nan = np.any(np.isnan(boundary_values))
            has_inf = np.any(np.isinf(boundary_values))
            
            if has_nan or has_inf:
                logger.warning(
                    f"  ⚠️  Feature '{fname}' has invalid values "
                    f"(NaN={has_nan}, Inf={has_inf}) - dropping"
                )
                features_to_drop.append(fname)
                continue
            
            # 2. Check for artifact patterns
            # Artifacts typically show:
            # - Very high std (discontinuous/dash patterns)
            # - Very low std (constant values - scan lines)
            # - Values concentrated at extremes (0 or 1)
            
            std_val = np.std(boundary_values)
            mean_val = np.mean(boundary_values)
            
            # Check for dash/line artifacts (high linearity + low variance)
            if fname == 'linearity':
                # High mean linearity (>0.8) with low variance (<0.1) = scan artifact
                if mean_val > 0.8 and std_val < 0.1:
                    logger.warning(
                        f"  ⚠️  Feature '{fname}' shows line artifact pattern "
                        f"(mean={mean_val:.3f}, std={std_val:.3f}) - dropping"
                    )
                    features_to_drop.append(fname)
                    continue
            
            # Check for planar discontinuities
            if fname == 'planarity':
                # Very low variance (<0.05) indicates artificial constant values
                if std_val < 0.05:
                    logger.warning(
                        f"  ⚠️  Feature '{fname}' shows constant pattern "
                        f"(std={std_val:.3f}) - dropping"
                    )
                    features_to_drop.append(fname)
                    continue
                
                # Very high variance (>0.4) indicates discontinuities
                if std_val > 0.4:
                    logger.warning(
                        f"  ⚠️  Feature '{fname}' shows discontinuity pattern "
                        f"(std={std_val:.3f}) - dropping"
                    )
                    features_to_drop.append(fname)
                    continue
            
            # Check for verticality artifacts
            if fname == 'verticality':
                # Abnormal concentration at extremes (bimodal with no middle values)
                high_extreme = np.sum(boundary_values > 0.9) / len(boundary_values)
                low_extreme = np.sum(boundary_values < 0.1) / len(boundary_values)
                
                if (high_extreme + low_extreme) > 0.95:
                    logger.warning(
                        f"  ⚠️  Feature '{fname}' shows bimodal extreme pattern "
                        f"(extreme_ratio={(high_extreme+low_extreme):.2f}) - dropping"
                    )
                    features_to_drop.append(fname)
                    continue
            
            # 3. Check value range validity
            if not np.all((boundary_values >= 0) & (boundary_values <= 1)):
                out_of_range = np.sum(
                    (boundary_values < 0) | (boundary_values > 1)
                )
                logger.warning(
                    f"  ⚠️  Feature '{fname}' has {out_of_range} values "
                    f"out of range [0,1] - dropping"
                )
                features_to_drop.append(fname)
                continue
        
        # Drop problematic features
        if features_to_drop:
            logger.info(
                f"  🔍 Feature validation: dropping {len(features_to_drop)} "
                f"problematic features: {features_to_drop}"
            )
            for fname in features_to_drop:
                del validated[fname]
        else:
            logger.info(
                f"  ✓ Feature validation: all {len(feature_names)} features passed"
            )
        
        return validated
    
    def get_feature_names(self) -> list:
        """
        Get list of computed feature names.
        
        Returns:
            List of feature names that will be computed
        """
        features = []
        
        if self.compute_normals:
            features.extend(['normal_x', 'normal_y', 'normal_z'])
            
            if self.compute_curvature:
                features.append('curvature')
            
            if self.compute_planarity:
                features.extend(['planarity', 'linearity', 'sphericity'])
            
            if self.compute_verticality:
                features.append('verticality')
        
        return features
    
    def get_feature_dimensions(self) -> int:
        """
        Get total number of feature dimensions.
        
        Returns:
            Total feature dimensions (e.g., 8 for all features)
        """
        return len(self.get_feature_names())


def compute_boundary_aware_features(
    core_points: np.ndarray,
    buffer_points: Optional[np.ndarray] = None,
    tile_bounds: Optional[Tuple[float, float, float, float]] = None,
    k_neighbors: int = 20,
    boundary_threshold: float = 10.0
) -> Dict[str, np.ndarray]:
    """
    Convenience function for boundary-aware feature computation.
    
    Args:
        core_points: (N, 3) XYZ coordinates of core tile
        buffer_points: (M, 3) XYZ coordinates of buffer zone (optional)
        tile_bounds: (xmin, ymin, xmax, ymax) Core tile boundaries
        k_neighbors: Number of neighbors
        boundary_threshold: Distance threshold for boundary detection
    
    Returns:
        Dictionary with computed features
    
    Example:
        >>> features = compute_boundary_aware_features(
        ...     core_points=core_xyz,
        ...     buffer_points=buffer_xyz,
        ...     tile_bounds=(0, 0, 1000, 1000),
        ...     k_neighbors=20
        ... )
        >>> normals = features['normals']
        >>> curvature = features['curvature']
    """
    computer = BoundaryAwareFeatureComputer(
        k_neighbors=k_neighbors,
        boundary_threshold=boundary_threshold,
        compute_normals=True,
        compute_curvature=True,
        compute_planarity=True,
        compute_verticality=True
    )
    
    return computer.compute_features(
        core_points=core_points,
        buffer_points=buffer_points,
        tile_bounds=tile_bounds
    )
