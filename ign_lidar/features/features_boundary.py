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
                - 'curvature': (N,) Curvature values
                - 'planarity': (N,) Planarity values
                - 'linearity': (N,) Linearity values
                - 'sphericity': (N,) Sphericity values
                - 'verticality': (N,) Verticality values
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
            
            if self.compute_verticality:
                logger.debug("Computing verticality...")
                results['verticality'] = self._compute_verticality(normals)
        
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
        
        Args:
            query_points: (N, 3) Points to compute normals for
            all_points: (N+M, 3) All points (core + buffer)
            neighbor_indices: (N, K) Neighbor indices in all_points
        
        Returns:
            normals: (N, 3) Normal vectors (unit vectors)
            eigenvalues: (N, 3) Eigenvalues (sorted descending: λ1 ≥ λ2 ≥ λ3)
        """
        num_points = len(query_points)
        normals = np.zeros((num_points, 3))
        eigenvalues = np.zeros((num_points, 3))
        
        for i in range(num_points):
            # Get neighbors
            neighbor_idx = neighbor_indices[i]
            neighbors = all_points[neighbor_idx]
            
            # Center neighborhood
            centroid = neighbors.mean(axis=0)
            centered = neighbors - centroid
            
            # Compute covariance matrix
            cov = (centered.T @ centered) / len(neighbors)
            
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Sort descending
            sort_idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[sort_idx]
            eigvecs = eigvecs[:, sort_idx]
            
            # Normal is eigenvector with smallest eigenvalue
            normal = eigvecs[:, 2]
            
            # Orient normal upward (positive Z component)
            if normal[2] < 0:
                normal = -normal
            
            normals[i] = normal
            eigenvalues[i] = eigvals
        
        return normals, eigenvalues
    
    def _compute_curvature(self, eigenvalues: np.ndarray) -> np.ndarray:
        """
        Compute curvature from eigenvalues.
        
        Curvature = λ3 / (λ1 + λ2 + λ3)
        
        Args:
            eigenvalues: (N, 3) Eigenvalues (λ1, λ2, λ3)
        
        Returns:
            (N,) Curvature values [0, 1]
        """
        lambda_sum = eigenvalues.sum(axis=1)
        # Avoid division by zero
        curvature = np.where(
            lambda_sum > 1e-10,
            eigenvalues[:, 2] / lambda_sum,
            0.0
        )
        return curvature
    
    def _compute_planarity_features(
        self,
        eigenvalues: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute planarity, linearity, and sphericity features.
        
        Planarity: (λ2 - λ3) / λ1  → 1 for planes, 0 otherwise
        Linearity: (λ1 - λ2) / λ1  → 1 for lines, 0 otherwise
        Sphericity: λ3 / λ1        → 1 for spheres, 0 otherwise
        
        Args:
            eigenvalues: (N, 3) Eigenvalues (λ1 ≥ λ2 ≥ λ3)
        
        Returns:
            Dictionary with planarity, linearity, sphericity
        """
        lambda1 = eigenvalues[:, 0]
        lambda2 = eigenvalues[:, 1]
        lambda3 = eigenvalues[:, 2]
        
        # Avoid division by zero
        eps = 1e-10
        
        planarity = np.where(
            lambda1 > eps,
            (lambda2 - lambda3) / lambda1,
            0.0
        )
        
        linearity = np.where(
            lambda1 > eps,
            (lambda1 - lambda2) / lambda1,
            0.0
        )
        
        sphericity = np.where(
            lambda1 > eps,
            lambda3 / lambda1,
            0.0
        )
        
        return {
            'planarity': planarity,
            'linearity': linearity,
            'sphericity': sphericity
        }
    
    def _compute_verticality(self, normals: np.ndarray) -> np.ndarray:
        """
        Compute verticality from normal vectors.
        
        Verticality = 1 - |nz|  where nz is Z component of normal
        → 1 for vertical surfaces (walls), 0 for horizontal
        
        Args:
            normals: (N, 3) Normal vectors
        
        Returns:
            (N,) Verticality values [0, 1]
        """
        verticality = 1.0 - np.abs(normals[:, 2])
        return verticality
    
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
