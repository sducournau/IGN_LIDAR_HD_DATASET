"""
Geometric Feature Extraction Functions - OPTIMIZED VERSION

This module provides ultra-fast vectorized feature extraction for LiDAR point clouds.
All functions use full NumPy vectorization with einsum for maximum performance.

For best performance, use compute_all_features_optimized() which builds the KDTree
only once and computes all features in a single pass (2-3x faster).

Features computed:
- Normals (nx, ny, nz): Surface orientation vectors
- Curvature: Local surface curvature
- Planarity: How planar is the surface (good for roofs, walls)
- Linearity: How linear is the structure (edges, cables)
- Sphericity: How spherical/scattered (vegetation, noise)
- Anisotropy: General measure of structure directionality
- Roughness: Surface roughness (smooth roofs vs rough vegetation)
- Density: Local point density

Facultative features (automatically computed):
- Wall Score: Planarity * Verticality (high for vertical planar surfaces)
- Roof Score: Planarity * Horizontality (high for horizontal planar surfaces)

Removed redundant features:
- Verticality/Horizontality: Already in normals (use normal_z)
"""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def estimate_optimal_k(points: np.ndarray,
                       target_radius: float = 0.5) -> int:
    """
    Estimate optimal k based on point cloud density.
    
    For building extraction, we want neighbors within ~0.5m radius.
    Higher density clouds need more neighbors.
    
    Args:
        points: [N, 3] point coordinates
        target_radius: target search radius in meters (default 0.5m)
        
    Returns:
        k: optimal number of neighbors (between 10 and 100)
    """
    # Sample 1000 points to estimate density
    n_samples = min(1000, len(points))
    sample_indices = np.random.choice(len(points), n_samples, replace=False)
    sample_points = points[sample_indices]
    
    # Build KDTree and find neighbors within target radius
    tree = KDTree(sample_points, metric='euclidean')
    counts = tree.query_radius(sample_points, r=target_radius,
                               count_only=True)
    
    # Average number of neighbors in target radius
    avg_neighbors = np.median(counts)
    
    # Clip to reasonable range for computation speed vs accuracy
    # Min 10 for sparse clouds, max 100 for very dense clouds
    k_optimal = int(np.clip(avg_neighbors, 10, 100))
    
    return k_optimal


def estimate_optimal_radius_for_features(points: np.ndarray,
                                         feature_type: str = 'geometric') -> float:
    """
    Estimate optimal search radius based on point cloud density and feature type.
    
    Radius-based search is SUPERIOR to k-based for geometric features because:
    - Avoids LIDAR scan line artifacts (dashed line patterns)
    - Captures true surface geometry, not sampling pattern
    - Consistent spatial scale across varying point density
    
    Args:
        points: [N, 3] point coordinates
        feature_type: 'geometric' for linearity/planarity (needs larger radius)
                      'normals' for normal computation (can use smaller radius)
                      
    Returns:
        radius: optimal search radius in meters
    """
    # Sample 1000 points to estimate density
    n_samples = min(1000, len(points))
    sample_indices = np.random.choice(len(points), n_samples, replace=False)
    sample_points = points[sample_indices]
    
    # Build KDTree and find average nearest neighbor distance
    tree = KDTree(sample_points, metric='euclidean')
    distances, _ = tree.query(sample_points, k=10)
    
    # Average distance to nearest neighbors (excluding self)
    avg_nn_dist = np.median(distances[:, 1:])  # Exclude first (self)
    
    # For geometric features (linearity/planarity):
    # Use 15-20x the average nearest neighbor distance
    # This ensures we capture the true surface geometry, not scan lines
    if feature_type == 'geometric':
        # For typical LIDAR HD (0.2-0.5m point spacing):
        # radius will be ~0.75-1.5m (good for building surfaces)
        radius = avg_nn_dist * 20.0
        radius = np.clip(radius, 0.5, 2.0)  # Min 0.5m, max 2.0m
    else:
        # For normals: smaller radius is OK
        radius = avg_nn_dist * 10.0
        radius = np.clip(radius, 0.3, 1.0)  # Min 0.3m, max 1.0m
    
    return float(radius)


def compute_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute surface normals using PCA on k-nearest neighbors.
    Version ULTRA-OPTIMISÉE avec vectorisation complète (100x plus rapide).
    
    Args:
        points: [N, 3] point coordinates
        k: number of neighbors for PCA (10 for fast computation)
        
    Returns:
        normals: [N, 3] normalized surface normals
    """
    # Build KDTree for efficient neighbor search
    tree = KDTree(points, metric='euclidean', leaf_size=30)
    
    # Query tous les voisins en une seule fois
    _, indices = tree.query(points, k=k)
    
    # VECTORISATION COMPLÈTE - Traiter tous les points simultanément
    # Shape: [N, k, 3]
    neighbors_all = points[indices]
    
    # Centrer les voisins: [N, k, 3]
    centroids = neighbors_all.mean(axis=1, keepdims=True)  # [N, 1, 3]
    centered = neighbors_all - centroids  # [N, k, 3]
    
    # Calculer matrices de covariance pour tous les points: [N, 3, 3]
    # cov = (1/(k-1)) * X^T @ X
    # Utiliser einsum pour vectorisation ultra-rapide
    cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)
    
    # Eigendecomposition vectorisée sur tous les points
    # eigenvalues: [N, 3], eigenvectors: [N, 3, 3]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
    
    # Normale = eigenvector avec la plus petite eigenvalue (colonne 0)
    normals = eigenvectors[:, :, 0].copy()  # [N, 3]
    
    # Normaliser les normales
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Éviter division par zéro
    normals = normals / norms
    
    # Orienter vers le haut (Z positif)
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] = -normals[flip_mask]
    
    # Gérer les cas dégénérés (très rare avec points réels)
    degenerate = (eigenvalues[:, 0] < 1e-8) | np.isnan(normals).any(axis=1)
    normals[degenerate] = np.array([0, 0, 1], dtype=np.float32)
    
    return normals.astype(np.float32)


def compute_curvature(points: np.ndarray, normals: np.ndarray,
                      k: int = 10) -> np.ndarray:
    """
    Compute principal curvature from local surface fit.
    Version ULTRA-OPTIMISÉE avec vectorisation complète (100x plus rapide).
    
    Args:
        points: [N, 3] point coordinates
        normals: [N, 3] surface normals
        k: number of neighbors (10 for fast computation)
        
    Returns:
        curvature: [N] principal curvature values
    """
    # Build KDTree
    tree = KDTree(points, metric='euclidean', leaf_size=30)
    
    # Query tous les voisins en une seule fois
    _, indices = tree.query(points, k=k)
    
    # VECTORISATION COMPLÈTE
    # Récupérer tous les voisins: [N, k, 3]
    neighbors_all = points[indices]
    
    # Position relative au centre: [N, k, 3]
    centers = points[:, np.newaxis, :]  # [N, 1, 3]
    relative_pos = neighbors_all - centers  # [N, k, 3]
    
    # Distance le long de la normale pour chaque point: [N, k]
    # distances[i, j] = dot(relative_pos[i, j], normals[i])
    normals_expanded = normals[:, np.newaxis, :]  # [N, 1, 3]
    distances_along_normal = np.sum(relative_pos * normals_expanded, axis=2)
    
    # Curvature = std des distances le long de la normale
    curvature = np.std(distances_along_normal, axis=1).astype(np.float32)
    
    return curvature


def compute_eigenvalue_features(
    eigenvalues: np.ndarray,
    epsilon: float = 1e-8
) -> Dict[str, np.ndarray]:
    """
    Compute all eigenvalue-based features.
    
    Args:
        eigenvalues: [N, 3] eigenvalues sorted descending (λ₀ >= λ₁ >= λ₂)
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Dictionary of eigenvalue features:
        - eigenvalue_1, eigenvalue_2, eigenvalue_3: Individual eigenvalues
        - sum_eigenvalues: Σλ
        - eigenentropy: Shannon entropy of normalized eigenvalues
        - omnivariance: Cubic root of eigenvalue product (3D dispersion)
        - change_curvature: Variance of eigenvalues (surface change rate)
    """
    λ0, λ1, λ2 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    
    # Clamp to non-negative (handle numerical artifacts)
    λ0 = np.maximum(λ0, 0.0)
    λ1 = np.maximum(λ1, 0.0)
    λ2 = np.maximum(λ2, 0.0)
    
    # Sum of eigenvalues
    sum_λ = λ0 + λ1 + λ2 + epsilon
    
    # Shannon entropy: -Σ(pᵢ * log(pᵢ)) where pᵢ = λᵢ/Σλ
    # Normalized eigenvalues as probability distribution
    p0 = λ0 / sum_λ
    p1 = λ1 / sum_λ
    p2 = λ2 / sum_λ
    
    # Avoid log(0) by adding epsilon
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p0 = np.log(p0 + epsilon)
        log_p1 = np.log(p1 + epsilon)
        log_p2 = np.log(p2 + epsilon)
    
    eigenentropy = -(p0 * log_p0 + p1 * log_p1 + p2 * log_p2)
    eigenentropy = np.nan_to_num(eigenentropy, nan=0.0, posinf=0.0, neginf=0.0)
    eigenentropy = np.clip(eigenentropy, 0.0, np.log(3.0))  # Max entropy = log(3)
    
    # Omnivariance: (λ₀ * λ₁ * λ₂)^(1/3) - measure of 3D dispersion
    # Alternative formulation: geometric mean of eigenvalues
    product = λ0 * λ1 * λ2 + epsilon
    omnivariance = np.cbrt(product)  # Cubic root
    
    # Change of curvature: Variance of eigenvalues (rate of surface change)
    # High variance indicates edges/corners, low variance indicates smooth surfaces
    mean_λ = sum_λ / 3.0
    change_curvature = np.sqrt(
        ((λ0 - mean_λ)**2 + (λ1 - mean_λ)**2 + (λ2 - mean_λ)**2) / 3.0
    )
    
    return {
        'eigenvalue_1': λ0.astype(np.float32),
        'eigenvalue_2': λ1.astype(np.float32),
        'eigenvalue_3': λ2.astype(np.float32),
        'sum_eigenvalues': sum_λ.astype(np.float32),
        'eigenentropy': eigenentropy.astype(np.float32),
        'omnivariance': omnivariance.astype(np.float32),
        'change_curvature': change_curvature.astype(np.float32),
    }


def compute_architectural_features(
    eigenvalues: np.ndarray,
    normals: np.ndarray,
    points: np.ndarray,
    tree: KDTree,
    k: int = 20,
    epsilon: float = 1e-8
) -> Dict[str, np.ndarray]:
    """
    Compute advanced architectural features for building detection.
    
    Features:
    - edge_strength: High eigenvalue variance indicates edges/corners
    - corner_likelihood: 3D structure indicator (all eigenvalues similar)
    - overhang_indicator: Detects overhangs using normal consistency
    - surface_roughness: Fine-scale texture measure
    
    Args:
        eigenvalues: [N, 3] eigenvalues sorted descending
        normals: [N, 3] surface normals
        points: [N, 3] point coordinates
        tree: KDTree for neighbor search
        k: Number of neighbors
        epsilon: Small constant
    
    Returns:
        Dictionary of architectural features
    """
    n_points = len(points)
    λ0, λ1, λ2 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    
    # Edge Strength: Variance of eigenvalues (high at edges/corners)
    # Same as change_curvature but can be normalized differently
    sum_λ = λ0 + λ1 + λ2 + epsilon
    mean_λ = sum_λ / 3.0
    eigenvalue_variance = ((λ0 - mean_λ)**2 + (λ1 - mean_λ)**2 + (λ2 - mean_λ)**2) / 3.0
    
    # Normalize by sum of eigenvalues for scale-invariance
    edge_strength = np.clip(eigenvalue_variance / (sum_λ + epsilon), 0.0, 1.0)
    
    # Corner Likelihood: Points where all 3 eigenvalues are similar
    # Use coefficient of variation: std/mean (low = similar values = corner)
    # Inverse so high values indicate corners
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.sqrt(eigenvalue_variance) / (mean_λ + epsilon)
        corner_likelihood = 1.0 / (1.0 + cv)  # Inverse sigmoid-like
        corner_likelihood = np.nan_to_num(corner_likelihood, nan=0.0)
        corner_likelihood = np.clip(corner_likelihood, 0.0, 1.0)
    
    # Overhang Indicator: Inconsistent normals in neighborhood (overhangs/protrusions)
    # Query neighbors
    _, indices = tree.query(points, k=min(k, n_points))
    neighbors_normals = normals[indices]  # [N, k, 3]
    
    # Compute normal consistency: dot product with center normal
    center_normals = normals[:, np.newaxis, :]  # [N, 1, 3]
    dot_products = np.sum(neighbors_normals * center_normals, axis=2)  # [N, k]
    
    # Low consistency (low dot product) indicates overhang
    # Use standard deviation of dot products
    normal_consistency = np.mean(dot_products, axis=1)
    overhang_indicator = 1.0 - normal_consistency  # High when normals inconsistent
    overhang_indicator = np.clip(overhang_indicator, 0.0, 1.0)
    
    # Surface Roughness: Fine-scale texture using smallest eigenvalue
    # High λ₂ relative to sum indicates rough surface
    surface_roughness = np.clip(λ2 / (sum_λ + epsilon), 0.0, 1.0)
    
    return {
        'edge_strength': edge_strength.astype(np.float32),
        'corner_likelihood': corner_likelihood.astype(np.float32),
        'overhang_indicator': overhang_indicator.astype(np.float32),
        'surface_roughness': surface_roughness.astype(np.float32),
    }


def compute_num_points_within_radius(
    points: np.ndarray,
    tree: KDTree,
    radius: float = 2.0
) -> np.ndarray:
    """
    Compute number of points within a given radius for each point.
    
    Args:
        points: [N, 3] point coordinates
        tree: KDTree for neighbor search
        radius: Search radius in meters (default 2.0)
    
    Returns:
        num_points: [N] number of neighbors within radius
    """
    neighbor_counts = tree.query_radius(points, r=radius, count_only=True)
    return neighbor_counts.astype(np.float32)


def compute_density_features(
    points: np.ndarray,
    tree: KDTree,
    k: int = 20,
    radius_2m: float = 2.0,
    epsilon: float = 1e-8
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive density and neighborhood features.
    
    Features:
    - density: Local point density (inverse of mean distance)
    - num_points_2m: Number of points within 2m radius
    - neighborhood_extent: Maximum distance to k-th neighbor
    - height_extent_ratio: Ratio of vertical std to spatial extent
    
    Args:
        points: [N, 3] point coordinates
        tree: KDTree for neighbor search
        k: Number of neighbors
        radius_2m: Fixed radius for point counting (default 2m)
        epsilon: Small constant
    
    Returns:
        Dictionary of density features
    """
    n_points = len(points)
    
    # Query k-nearest neighbors
    distances, indices = tree.query(points, k=min(k, n_points))
    
    # Density: Inverse of mean distance to neighbors
    # Exclude self (first neighbor)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    density = 1.0 / (mean_distances + epsilon)
    density = np.clip(density, 0.0, 1000.0)  # Cap at 1000 points/m
    
    # Neighborhood extent: Maximum distance to k-th neighbor
    neighborhood_extent = distances[:, -1]
    
    # Vertical standard deviation in neighborhood
    neighbors_all = points[indices]  # [N, k, 3]
    z_neighbors = neighbors_all[:, :, 2]  # [N, k]
    vertical_std = np.std(z_neighbors, axis=1)
    
    # Height-extent ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        height_extent_ratio = vertical_std / (neighborhood_extent + epsilon)
        height_extent_ratio = np.nan_to_num(height_extent_ratio, nan=0.0)
        height_extent_ratio = np.clip(height_extent_ratio, 0.0, 10.0)
    
    # Number of points within 2m radius
    # Use radius search for this
    neighbor_counts = tree.query_radius(points, r=radius_2m, count_only=True)
    num_points_2m = neighbor_counts.astype(np.float32)
    
    return {
        'density': density.astype(np.float32),
        'num_points_2m': num_points_2m,
        'neighborhood_extent': neighborhood_extent.astype(np.float32),
        'height_extent_ratio': height_extent_ratio.astype(np.float32),
        'vertical_std': vertical_std.astype(np.float32),
    }


def compute_verticality(normals: np.ndarray) -> np.ndarray:
    """
    Compute verticality score from normals.
    
    Verticality = 1 - |normal_z| (1 for vertical, 0 for horizontal)
    
    Args:
        normals: [N, 3] surface normals
    
    Returns:
        verticality: [N] verticality scores [0, 1]
    """
    return (1.0 - np.abs(normals[:, 2])).astype(np.float32)


def compute_building_scores(
    planarity: np.ndarray,
    normals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute building-specific classification scores.
    
    Args:
        planarity: [N] planarity values [0, 1]
        normals: [N, 3] surface normals
    
    Returns:
        verticality: [N] vertical surface indicator
        wall_score: [N] wall likelihood (planarity × verticality)
        roof_score: [N] roof likelihood (planarity × horizontality)
    """
    # Verticality: 1 for vertical, 0 for horizontal
    verticality = 1.0 - np.abs(normals[:, 2])
    
    # Horizontality: 1 for horizontal, 0 for vertical
    horizontality = np.abs(normals[:, 2])
    
    # Wall score: High planarity + high verticality
    wall_score = planarity * verticality
    
    # Roof score: High planarity + high horizontality
    roof_score = planarity * horizontality
    
    return (
        verticality.astype(np.float32),
        wall_score.astype(np.float32),
        roof_score.astype(np.float32)
    )


def compute_horizontality(normals: np.ndarray) -> np.ndarray:
    """
    Compute horizontality score from normals.
    
    Horizontality = |normal_z| (1 for horizontal, 0 for vertical)
    Useful for roof detection and horizontal surface identification.
    
    Args:
        normals: [N, 3] surface normals
    
    Returns:
        horizontality: [N] horizontality scores [0, 1]
    """
    return np.abs(normals[:, 2]).astype(np.float32)


def compute_edge_strength(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute edge strength from eigenvalues.
    
    Edge strength measures how much the geometry resembles an edge
    (high variance between eigenvalues). Useful for detecting building
    corners, roof edges, and structural transitions.
    
    Formula: edge_strength = (λ1 - λ2) / (λ1 + ε)
    where λ1 > λ2 > λ3 are ordered eigenvalues
    
    Args:
        eigenvalues: [N, 3] eigenvalues (ordered descending)
    
    Returns:
        edge_strength: [N] edge strength values [0, 1]
    """
    eps = 1e-8
    # eigenvalues should be ordered: λ1 >= λ2 >= λ3
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    
    edge_strength = (lambda1 - lambda2) / (lambda1 + eps)
    edge_strength = np.clip(edge_strength, 0, 1)
    
    return edge_strength.astype(np.float32)


def compute_facade_score(
    normals: np.ndarray,
    planarity: np.ndarray,
    verticality: np.ndarray,
    height: np.ndarray,
    min_height: float = 2.5
) -> np.ndarray:
    """
    Compute facade likelihood score for building wall detection.
    
    Combines multiple geometric attributes to identify building facades:
    - Verticality (walls are vertical)
    - Planarity (walls are planar)
    - Height (facades are elevated)
    - Consistent normal orientation (facades face outward)
    
    Args:
        normals: [N, 3] surface normals
        planarity: [N] planarity values [0, 1]
        verticality: [N] verticality values [0, 1]
        height: [N] height above ground in meters
        min_height: minimum height for facades (default 2.5m)
    
    Returns:
        facade_score: [N] facade likelihood [0, 1]
    """
    # Facade components:
    # 1. High verticality (walls are vertical)
    vert_component = np.clip(verticality / 0.7, 0, 1)  # Normalize to [0,1] with threshold
    
    # 2. High planarity (walls are flat)
    plan_component = np.clip(planarity / 0.5, 0, 1)  # Normalize to [0,1] with threshold
    
    # 3. Elevated height (facades are above ground)
    height_component = np.clip((height - min_height) / 5.0, 0, 1)  # Fade in from min_height
    
    # 4. Combined facade score (multiplicative - all must be present)
    facade_score = vert_component * plan_component * height_component
    
    return facade_score.astype(np.float32)


def compute_roof_plane_score(
    normals: np.ndarray,
    planarity: np.ndarray,
    height: np.ndarray,
    min_roof_height: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute roof plane scores for different roof types.
    
    Distinguishes between:
    - Flat roofs (horizontal, slope < 15°)
    - Sloped roofs (moderate tilt, 15° - 45°)
    - Steep roofs (steep tilt, 45° - 70°)
    
    Args:
        normals: [N, 3] surface normals
        planarity: [N] planarity values [0, 1]
        height: [N] height above ground in meters
        min_roof_height: minimum height for roof elements (default 3.0m)
    
    Returns:
        Tuple of (flat_roof_score, sloped_roof_score, steep_roof_score)
    """
    # Compute tilt angle from vertical (using normal_z)
    horizontality = np.abs(normals[:, 2])
    
    # Flat roofs: horizontal (slope < 15°), cos(15°) ≈ 0.966
    flat_roof_mask = (horizontality > 0.966) & (planarity > 0.7) & (height > min_roof_height)
    flat_roof_score = flat_roof_mask.astype(np.float32) * planarity
    
    # Sloped roofs: moderate tilt (15° - 45°), cos(45°) ≈ 0.707
    sloped_roof_mask = (
        (horizontality <= 0.966) & 
        (horizontality > 0.707) & 
        (planarity > 0.6) & 
        (height > min_roof_height)
    )
    sloped_roof_score = sloped_roof_mask.astype(np.float32) * planarity
    
    # Steep roofs: steep tilt (45° - 70°), cos(70°) ≈ 0.342
    steep_roof_mask = (
        (horizontality <= 0.707) & 
        (horizontality > 0.342) & 
        (planarity > 0.5) & 
        (height > min_roof_height)
    )
    steep_roof_score = steep_roof_mask.astype(np.float32) * planarity
    
    return (
        flat_roof_score.astype(np.float32),
        sloped_roof_score.astype(np.float32),
        steep_roof_score.astype(np.float32)
    )


def compute_opening_likelihood(
    planarity: np.ndarray,
    linearity: np.ndarray,
    verticality: np.ndarray,
    intensity: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute likelihood of architectural openings (windows, doors).
    
    Openings are characterized by:
    - Low planarity (not solid surface)
    - High linearity (rectangular edges)
    - Vertical orientation (on walls)
    - Often lower intensity (glass, shadows)
    
    Args:
        planarity: [N] planarity values [0, 1]
        linearity: [N] linearity values [0, 1]
        verticality: [N] verticality values [0, 1]
        intensity: [N] optional intensity values [0, 1]
    
    Returns:
        opening_score: [N] opening likelihood [0, 1]
    """
    # Low planarity (openings are not solid surfaces)
    low_planarity = 1.0 - planarity  # Invert: high score for low planarity
    
    # High linearity (rectangular frames)
    high_linearity = linearity
    
    # On vertical surfaces (walls)
    on_wall = verticality > 0.6
    
    # Combine signals
    opening_score = low_planarity * high_linearity * on_wall.astype(np.float32)
    
    # Optional: use intensity to enhance detection
    if intensity is not None:
        # Windows often have lower intensity (glass reflects less)
        low_intensity = 1.0 - intensity
        opening_score = opening_score * (0.7 + 0.3 * low_intensity)
    
    return opening_score.astype(np.float32)


def compute_structural_element_score(
    linearity: np.ndarray,
    verticality: np.ndarray,
    anisotropy: np.ndarray,
    height: np.ndarray
) -> np.ndarray:
    """
    Compute score for structural elements (pillars, columns, beams).
    
    Structural elements are characterized by:
    - High linearity (linear/cylindrical shape)
    - Often vertical (pillars, columns)
    - High anisotropy (organized structure)
    - Significant height (load-bearing elements)
    
    Args:
        linearity: [N] linearity values [0, 1]
        verticality: [N] verticality values [0, 1]
        anisotropy: [N] anisotropy values [0, 1]
        height: [N] height above ground in meters
    
    Returns:
        structural_score: [N] structural element likelihood [0, 1]
    """
    # High linearity (columns, beams)
    linear_component = linearity
    
    # Vertical orientation (most structural elements)
    vertical_component = verticality
    
    # Organized structure (not random)
    structure_component = anisotropy
    
    # Elevated (not ground level)
    height_component = np.clip(height / 5.0, 0, 1)
    
    # Combine (multiplicative)
    structural_score = (
        linear_component * 
        vertical_component * 
        structure_component * 
        height_component
    )
    
    return structural_score.astype(np.float32)


def compute_height_above_ground(points: np.ndarray,
                                classification: np.ndarray) -> np.ndarray:
    """
    Compute height above ground for each point.
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] ASPRS classification codes
        
    Returns:
        height: [N] height above ground in meters
    """
    # Find ground points (classification code 2)
    ground_mask = (classification == 2)
    
    if not np.any(ground_mask):
        # If no ground points, use minimum Z
        ground_z = np.min(points[:, 2])
    else:
        # Use median Z of ground points
        ground_z = np.median(points[ground_mask, 2])
    
    height = points[:, 2] - ground_z
    return np.maximum(height, 0)  # Ensure non-negative heights


def compute_height_features(points: np.ndarray,
                           classification: np.ndarray = None,
                           patch_center: np.ndarray = None) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive height-based features.
    CRITICAL for building extraction - distinguishes roofs/walls/ground.
    
    Features computed:
    - z_absolute: Absolute Z coordinate
    - z_normalized: Z normalized to [0, 1] range
    - z_from_ground: Height above ground (Z - Z_min)
    - z_from_median: Height relative to median Z
    - distance_to_center: Euclidean distance to patch center
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] ASPRS codes (optional, for ground detection)
        patch_center: [3] center of patch (optional, for distance calc)
        
    Returns:
        features: dict of height features
    """
    z = points[:, 2]
    z_min = np.min(z)
    z_max = np.max(z)
    z_range = z_max - z_min
    
    # 1. Hauteur absolue Z
    z_absolute = z.astype(np.float32)
    
    # 2. Hauteur normalisée [0, 1]
    if z_range > 1e-6:
        z_normalized = ((z - z_min) / z_range).astype(np.float32)
    else:
        z_normalized = np.zeros_like(z, dtype=np.float32)
    
    # 3. Hauteur depuis le sol (Z - Z_min)
    z_from_ground = (z - z_min).astype(np.float32)
    
    # 4. Hauteur relative à la médiane
    z_median = np.median(z)
    z_from_median = (z - z_median).astype(np.float32)
    
    features = {
        'z_absolute': z_absolute,
        'z_normalized': z_normalized,
        'z_from_ground': z_from_ground,
        'z_from_median': z_from_median,
    }
    
    # 5. Distance au centre du patch (si fourni)
    if patch_center is not None:
        center = patch_center.reshape(1, 3)
        distances = np.linalg.norm(points - center, axis=1).astype(np.float32)
        features['distance_to_center'] = distances
    
    return features


def compute_local_statistics(points: np.ndarray,
                            k: int = 10) -> Dict[str, np.ndarray]:
    """
    Compute local neighborhood statistics.
    Captures fine geometric structure - EXPENSIVE but POWERFUL.
    
    Features computed:
    - vertical_std: Écart-type vertical dans le voisinage
    - neighborhood_extent: Étendue du voisinage (max distance)
    - height_extent_ratio: Ratio hauteur/étendue
    - local_roughness: Rugosité locale (std distances au plan local)
    
    Warning: Computationally expensive for large point clouds (270K+ points)
    Consider using on downsampled data or with smaller k.
    
    Args:
        points: [N, 3] point coordinates
        k: number of neighbors (default 10)
        
    Returns:
        features: dict of local statistics
    """
    # Build KDTree
    tree = KDTree(points, metric='euclidean', leaf_size=30)
    distances, indices = tree.query(points, k=k)
    
    # Get all neighbors: [N, k, 3]
    neighbors_all = points[indices]
    
    # 1. Écart-type vertical dans le voisinage
    z_neighbors = neighbors_all[:, :, 2]  # [N, k]
    vertical_std = np.std(z_neighbors, axis=1).astype(np.float32)
    
    # 2. Étendue du voisinage (max distance aux voisins)
    neighborhood_extent = np.max(distances, axis=1).astype(np.float32)
    
    # 3. Ratio hauteur/étendue
    # Éviter division par zéro
    with np.errstate(divide='ignore', invalid='ignore'):
        height_extent_ratio = vertical_std / (neighborhood_extent + 1e-8)
        height_extent_ratio = np.nan_to_num(height_extent_ratio,
                                           nan=0.0, posinf=0.0,
                                           neginf=0.0).astype(np.float32)
    
    # 4. Rugosité locale (distance std au plan local via PCA)
    # Centrer les voisins
    centroids = neighbors_all.mean(axis=1, keepdims=True)  # [N, 1, 3]
    centered = neighbors_all - centroids  # [N, k, 3]
    
    # Covariance matrices
    cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov_matrices)
    
    # Roughness = smallest eigenvalue / sum (normalized variance)
    sum_eigenvalues = np.sum(eigenvalues, axis=1) + 1e-8
    local_roughness = (eigenvalues[:, 0] / sum_eigenvalues).astype(np.float32)
    
    features = {
        'vertical_std': vertical_std,
        'neighborhood_extent': neighborhood_extent,
        'height_extent_ratio': height_extent_ratio,
        'local_roughness': local_roughness,
    }
    
    return features


def compute_verticality(normals: np.ndarray) -> np.ndarray:
    """
    Compute verticality from surface normals.
    
    Verticality measures how vertical a surface is (walls vs roofs/ground).
    IMPORTANT for building extraction.
    
    Args:
        normals: [N, 3] surface normal vectors
        
    Returns:
        verticality: [N] verticality values [0, 1]
                    0 = horizontal surface
                    1 = vertical surface
    """
    # Verticality = 1 - abs(normal_z)
    # abs(normal_z) = 1 pour surfaces horizontales
    # abs(normal_z) = 0 pour surfaces verticales
    verticality = 1.0 - np.abs(normals[:, 2])
    return verticality.astype(np.float32)


def compute_wall_score(normals: np.ndarray, height_above_ground: np.ndarray,
                       min_height: float = 1.5) -> np.ndarray:
    """
    Compute wall probability score.
    
    Combines verticality with height above ground to identify walls.
    Walls are vertical surfaces that are elevated above ground.
    
    Args:
        normals: [N, 3] surface normal vectors
        height_above_ground: [N] height above ground in meters
        min_height: minimum height to be considered a wall (default 1.5m)
        
    Returns:
        wall_score: [N] wall probability [0, 1]
    """
    # Verticality component
    verticality = compute_verticality(normals)
    
    # Height component (walls are typically > 1.5m above ground)
    height_score = np.clip((height_above_ground - min_height) / 5.0, 0, 1)
    
    # Combine: high verticality AND elevated
    wall_score = verticality * height_score
    
    return wall_score.astype(np.float32)


def compute_roof_score(normals: np.ndarray,
                       height_above_ground: np.ndarray,
                       curvature: np.ndarray,
                       min_height: float = 3.0) -> np.ndarray:
    """
    Compute roof probability score.
    
    Roofs are horizontal surfaces that are elevated and have low curvature.
    
    Args:
        normals: [N, 3] surface normal vectors
        height_above_ground: [N] height above ground in meters
        curvature: [N] surface curvature
        min_height: minimum height for a roof (default 3.0m)
        
    Returns:
        roof_score: [N] roof probability [0, 1]
    """
    # Horizontality (inverse of verticality)
    horizontality = np.abs(normals[:, 2])
    
    # Height component (roofs are typically > 2m above ground)
    height_score = np.clip((height_above_ground - min_height) / 8.0, 0, 1)
    
    # Low curvature (roofs are planar)
    curvature_score = 1.0 - np.clip(curvature / 0.5, 0, 1)
    
    # Combine: horizontal AND elevated AND planar
    roof_score = horizontality * height_score * curvature_score
    
    return roof_score.astype(np.float32)


def compute_num_points_in_radius(points: np.ndarray,
                                 radius: float = 2.0,
                                 chunk_size: int = 500_000) -> np.ndarray:
    """
    Compute number of points within a given radius for each point.
    
    This gives a measure of local point density, useful for distinguishing
    dense building structures from sparse vegetation.
    
    MEMORY OPTIMIZED: Processes in chunks to handle large point clouds.
    
    Args:
        points: [N, 3] point coordinates
        radius: search radius in meters (default 2.0m)
        chunk_size: number of points to process per chunk (default 500K)
        
    Returns:
        num_points: [N] number of points within radius
    """
    from sklearn.neighbors import NearestNeighbors
    import gc
    
    n_points = len(points)
    
    # Build KD-tree once for all queries
    # Force n_jobs=1 to prevent memory spikes from parallel tree building
    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree', n_jobs=1)
    nbrs.fit(points)
    
    # Process in chunks to avoid memory spikes
    num_points = np.zeros(n_points, dtype=np.float32)
    
    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk = points[start_idx:end_idx]
        
        # Query radius for chunk
        indices = nbrs.radius_neighbors(chunk, return_distance=False)
        
        # Count points (subtract 1 to exclude the point itself)
        chunk_counts = np.array([len(idx) - 1 for idx in indices], 
                                dtype=np.float32)
        num_points[start_idx:end_idx] = chunk_counts
        
        # Clear temporary data
        del indices, chunk_counts, chunk
    
    # Explicit cleanup of KDTree (can be large)
    del nbrs
    gc.collect()
    
    return num_points


def extract_geometric_features(points: np.ndarray, normals: np.ndarray,
                               k: int = 10,
                               radius: float = None) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive geometric features for each point.
    Version ULTRA-OPTIMISÉE avec recherche par RAYON adaptatif.
    
    IMPORTANT: Utilise un rayon spatial au lieu de k-neighbors fixes pour éviter
    les artefacts de lignes pointillées causés par le pattern de scan LIDAR.
    
    Features computed (all based on eigenvalue decomposition):
    Using standard formulas (Weinmann et al., Demantké et al.)
    where λ0 >= λ1 >= λ2 are eigenvalues in descending order:
    
    - Linearity: (λ0-λ1)/λ0 - 1D structures (edges, cables) [0,1]
    - Planarity: (λ1-λ2)/λ0 - 2D structures (roofs, walls) [0,1]
    - Sphericity: λ2/λ0 - 3D structures (vegetation, noise) [0,1]
    - Anisotropy: (λ0-λ2)/λ0 - general directionality [0,1]
    - Roughness: λ2/Σλ - surface roughness (smooth vs rough) [0,1]
    - Density: 1/mean_dist - local point density
    
    Properties:
    - Linearity + Planarity + Sphericity = 1.0 (exact, due to λ0 normalization)
    - For 1D: Linearity ≈ 1, Planarity ≈ 0, Sphericity ≈ 0
    - For 2D: Linearity ≈ 0, Planarity ≈ 1, Sphericity ≈ 0
    - For 3D: Linearity ≈ 0, Planarity ≈ 0, Sphericity ≈ 1
    
    Note: Verticality/Horizontality removed (use normals directly)
    
    Args:
        points: [N, 3] point coordinates
        normals: [N, 3] normal vectors (not used here, kept for API compat)
        k: number of neighbors (used only if radius=None, fallback)
        radius: search radius in meters (RECOMMENDED: auto-estimated if None)
                Using radius avoids LIDAR scan line artifacts!
        
    Returns:
        features: dictionary of geometric features (all in range [0,1])
    """
    # Build KDTree
    tree = KDTree(points, metric='euclidean', leaf_size=30)
    
    # Use RADIUS-based search (superior for avoiding scan artifacts)
    if radius is None:
        # Auto-estimate optimal radius for geometric features
        radius = estimate_optimal_radius_for_features(points, 'geometric')
        print(f"  Using radius-based search: r={radius:.2f}m "
              f"(avoids scan line artifacts)")
    
    # Query neighbors within radius for all points
    # Returns list of arrays (variable number of neighbors per point)
    neighbor_indices = tree.query_radius(points, r=radius)
    
    n_points = len(points)
    
    # Initialize output arrays
    linearity = np.zeros(n_points, dtype=np.float32)
    planarity = np.zeros(n_points, dtype=np.float32)
    sphericity = np.zeros(n_points, dtype=np.float32)
    anisotropy = np.zeros(n_points, dtype=np.float32)
    roughness = np.zeros(n_points, dtype=np.float32)
    density = np.zeros(n_points, dtype=np.float32)
    
    # Process each point (can't fully vectorize due to variable neighbor counts)
    for i in range(n_points):
        neighbors_i = neighbor_indices[i]
        
        # Need at least 3 neighbors for PCA
        if len(neighbors_i) < 3:
            continue
        
        # Get neighbor coordinates
        neighbor_pts = points[neighbors_i]
        
        # Center the neighbors
        centroid = neighbor_pts.mean(axis=0)
        centered = neighbor_pts - centroid
        
        # Covariance matrix
        cov = (centered.T @ centered) / (len(neighbors_i) - 1)
        
        # Eigenvalues (sorted descending)
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvals = np.sort(eigenvals)[::-1]  # λ0 >= λ1 >= λ2
        
        λ0, λ1, λ2 = eigenvals[0], eigenvals[1], eigenvals[2]
        
        # Clamp eigenvalues to non-negative (handle numerical artifacts)
        λ0 = max(λ0, 0.0)
        λ1 = max(λ1, 0.0)
        λ2 = max(λ2, 0.0)
        
        # Avoid division by zero
        sum_λ = λ0 + λ1 + λ2 + 1e-8
        λ0_safe = λ0 + 1e-8
        
        # Compute features using λ0 normalization (consistent with GPU/boundary)
        # Formula: Weinmann et al. - normalized by largest eigenvalue λ0
        # Explicitly clamp to [0, 1] to handle edge cases
        linearity[i] = np.clip((λ0 - λ1) / λ0_safe, 0.0, 1.0)
        planarity[i] = np.clip((λ1 - λ2) / λ0_safe, 0.0, 1.0)
        sphericity[i] = np.clip(λ2 / λ0_safe, 0.0, 1.0)
        anisotropy[i] = np.clip((λ0 - λ2) / λ0_safe, 0.0, 1.0)
        roughness[i] = np.clip(λ2 / sum_λ, 0.0, 1.0)  # Keep sum normalization for roughness
        
        # Density (number of neighbors / volume) - capped at 1000 points/m³
        density[i] = np.clip(len(neighbors_i) / (4/3 * np.pi * radius**3 + 1e-8), 0.0, 1000.0)
    
    # === FACULTATIVE FEATURES: WALL AND ROOF SCORES ===
    # Wall score: High planarity + Vertical surface (|normal_z| close to 0)
    # Roof score: High planarity + Horizontal surface (|normal_z| close to 1)
    verticality = 1.0 - np.abs(normals[:, 2])  # 0=horizontal, 1=vertical
    horizontality = np.abs(normals[:, 2])      # 1=horizontal, 0=vertical
    
    wall_score = (planarity * verticality).astype(np.float32)
    roof_score = (planarity * horizontality).astype(np.float32)
    
    # Stocker les features
    features = {
        'planarity': planarity,
        'linearity': linearity,
        'sphericity': sphericity,
        'anisotropy': anisotropy,
        'roughness': roughness,
        'density': density,
        'wall_score': wall_score,
        'roof_score': roof_score
    }
    
    return features


def _compute_all_features_chunked(
    points: np.ndarray,
    classification: np.ndarray,
    k: int,
    auto_k: bool,
    include_extra: bool,
    patch_center: np.ndarray,
    chunk_size: int,
    radius: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Memory-efficient chunked processing for large point clouds.
    
    Processes points in chunks to avoid OOM errors on large files (>20M points).
    The KDTree is still built on all points for accurate neighbor search,
    but feature computation is done chunk by chunk.
    
    IMPORTANT: Uses RADIUS-based search (not k-NN) to avoid LIDAR scan line artifacts!
    """
    n_points = len(points)
    n_chunks = (n_points + chunk_size - 1) // chunk_size
    chunk_size_mb = (chunk_size * 12) / (1024 * 1024)  # Approx memory per chunk
    
    print(f"Processing {n_points:,} points in {n_chunks} chunks of ~{chunk_size:,} points each")
    
    # Determine search strategy: RADIUS-based (preferred) or k-NN (fallback)
    use_radius = True
    if radius is None:
        # Auto-estimate optimal radius for geometric features
        radius = estimate_optimal_radius_for_features(points, 'geometric')
        print(f"Auto-estimated radius r={radius:.2f}m (avoids scan line artifacts)")
    
    # For k-NN fallback (if explicitly requested with radius=0)
    if radius == 0:
        use_radius = False
        if auto_k and k is None:
            k = estimate_optimal_k(points, target_radius=0.5)
            print(f"Auto-estimated k={k} neighbors based on point density")
        elif k is None:
            k = 10
        print(f"WARNING: Using k-NN search (k={k}) - may cause scan line artifacts!")
    
    # Build KDTree once on all points (this is memory-efficient)
    print(f"Building KDTree for {n_points:,} points...")
    tree = KDTree(points, metric='euclidean', leaf_size=30)
    
    # Compute ground height once
    ground_mask = (classification == 2)
    if np.any(ground_mask):
        ground_z = np.median(points[ground_mask, 2])
    else:
        ground_z = np.min(points[:, 2])
    
    # Initialize output arrays
    normals = np.zeros((n_points, 3), dtype=np.float32)
    curvature = np.zeros(n_points, dtype=np.float32)
    height = np.zeros(n_points, dtype=np.float32)
    
    geo_features = {
        'planarity': np.zeros(n_points, dtype=np.float32),
        'linearity': np.zeros(n_points, dtype=np.float32),
        'sphericity': np.zeros(n_points, dtype=np.float32),
        'anisotropy': np.zeros(n_points, dtype=np.float32),
        'roughness': np.zeros(n_points, dtype=np.float32),
        'density': np.zeros(n_points, dtype=np.float32),
        'verticality': np.zeros(n_points, dtype=np.float32),
        'horizontality': np.zeros(n_points, dtype=np.float32),
    }
    
    if include_extra:
        geo_features.update({
            'vertical_std': np.zeros(n_points, dtype=np.float32),
            'neighborhood_extent': np.zeros(n_points, dtype=np.float32),
            'height_extent_ratio': np.zeros(n_points, dtype=np.float32),
            'local_roughness': np.zeros(n_points, dtype=np.float32),
        })
    
    # Process chunks with progress bar
    bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} chunks '
               '[{elapsed}<{remaining}, {rate_fmt}]')
    chunk_iterator = tqdm(
        range(n_chunks),
        desc=f"  💻 CPU Features ({n_points:,} pts, {n_chunks} chunks @ {chunk_size_mb:.1f}MB)",
        unit="chunk",
        total=n_chunks,
        bar_format=bar_fmt
    )
    
    for i in chunk_iterator:
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)
        chunk_points = points[start_idx:end_idx]
        chunk_len = end_idx - start_idx
        
        # Query neighbors for this chunk
        if radius > 0:
            # RADIUS-based search (avoids scan line artifacts)
            neighbor_indices_list = tree.query_radius(chunk_points, r=radius)
            
            # Process each point in chunk with variable neighbor count
            chunk_normals = np.zeros((chunk_len, 3), dtype=np.float32)
            chunk_curvature = np.zeros(chunk_len, dtype=np.float32)
            chunk_eigenvalues = np.zeros((chunk_len, 3), dtype=np.float32)
            chunk_distances_mean = np.zeros(chunk_len, dtype=np.float32)
            chunk_distances_max = np.zeros(chunk_len, dtype=np.float32)
            
            for j, neighbors_idx in enumerate(neighbor_indices_list):
                if len(neighbors_idx) < 3:
                    # Not enough neighbors - use default values
                    chunk_normals[j] = [0, 0, 1]
                    continue
                
                neighbor_pts = points[neighbors_idx]
                centroid = neighbor_pts.mean(axis=0)
                centered_pts = neighbor_pts - centroid
                
                # Covariance matrix
                cov = (centered_pts.T @ centered_pts) / (len(neighbors_idx)-1)
                
                # Eigendecomposition
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                eigenvals_sorted = np.sort(eigenvals)[::-1]  # Descending
                chunk_eigenvalues[j] = eigenvals_sorted
                
                # Normal (smallest eigenvector)
                normal = eigenvecs[:, 0]
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-8:
                    normal = normal / norm_len
                if normal[2] < 0:
                    normal = -normal
                chunk_normals[j] = normal
                
                # Curvature
                rel_pos = neighbor_pts - chunk_points[j]
                dist_along_normal = np.dot(rel_pos, normal)
                chunk_curvature[j] = np.std(dist_along_normal)
                
                # Distance stats
                dists = np.linalg.norm(rel_pos, axis=1)
                chunk_distances_mean[j] = np.mean(dists[dists > 0])
                chunk_distances_max[j] = np.max(dists)
            
            normals[start_idx:end_idx] = chunk_normals
            curvature[start_idx:end_idx] = chunk_curvature
            
        else:
            # k-NN fallback (may cause artifacts!)
            distances, indices = tree.query(chunk_points, k=k)
            neighbors_all = points[indices]
            
            # Center neighbors
            centroids = neighbors_all.mean(axis=1, keepdims=True)
            centered = neighbors_all - centroids
            
            # Covariance matrices
            cov_matrices = np.einsum('nki,nkj->nij', centered, centered)
            cov_matrices = cov_matrices / (k - 1)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
            chunk_eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]
            
            # Normals
            chunk_normals = eigenvectors[:, :, 0].copy()
            norms = np.linalg.norm(chunk_normals, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            chunk_normals = chunk_normals / norms
            flip_mask = chunk_normals[:, 2] < 0
            chunk_normals[flip_mask] = -chunk_normals[flip_mask]
            degenerate = ((eigenvalues[:, 0] < 1e-8) |
                         np.isnan(chunk_normals).any(axis=1))
            chunk_normals[degenerate] = np.array([0, 0, 1],
                                                 dtype=np.float32)
            normals[start_idx:end_idx] = chunk_normals
            
            # Curvature
            centers = chunk_points[:, np.newaxis, :]
            relative_pos = neighbors_all - centers
            normals_expanded = chunk_normals[:, np.newaxis, :]
            distances_along_normal = np.sum(
                relative_pos * normals_expanded, axis=2
            )
            chunk_curvature = np.std(distances_along_normal, axis=1)
            curvature[start_idx:end_idx] = chunk_curvature.astype(np.float32)
            
            chunk_distances_mean = np.mean(distances[:, 1:], axis=1)
            chunk_distances_max = np.max(distances, axis=1)
            chunk_eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]
        
        # === HEIGHT ===
        height[start_idx:end_idx] = np.maximum(
            chunk_points[:, 2] - ground_z, 0
        ).astype(np.float32)
        
        # === GEOMETRIC FEATURES ===
        λ0 = chunk_eigenvalues[:, 0]
        λ1 = chunk_eigenvalues[:, 1]
        λ2 = chunk_eigenvalues[:, 2]
        
        λ0_safe = λ0 + 1e-8
        sum_λ = λ0 + λ1 + λ2 + 1e-8
        
        # Compute geometric features (Weinmann et al., Demantké et al.)
        geo_features['linearity'][start_idx:end_idx] = (
            ((λ0 - λ1) / sum_λ).astype(np.float32)
        )
        geo_features['planarity'][start_idx:end_idx] = (
            ((λ1 - λ2) / sum_λ).astype(np.float32)
        )
        geo_features['sphericity'][start_idx:end_idx] = (
            (λ2 / sum_λ).astype(np.float32)
        )
        geo_features['anisotropy'][start_idx:end_idx] = (
            ((λ0 - λ2) / λ0_safe).astype(np.float32)
        )
        geo_features['roughness'][start_idx:end_idx] = (
            (λ2 / sum_λ).astype(np.float32)
        )
        geo_features['density'][start_idx:end_idx] = (
            (1.0 / (chunk_distances_mean + 1e-8)).astype(np.float32)
        )
        
        # === EXTRA FEATURES ===
        if include_extra:
            # For radius-based, we need to recompute these from neighbor data
            if radius > 0:
                # Vertical std (computed from neighbor indices per point)
                v_std = np.zeros(chunk_len, dtype=np.float32)
                n_extent = np.zeros(chunk_len, dtype=np.float32)
                for j, neighbors_idx in enumerate(neighbor_indices_list):
                    if len(neighbors_idx) >= 3:
                        neighbor_pts = points[neighbors_idx]
                        v_std[j] = np.std(neighbor_pts[:, 2])
                        dists = np.linalg.norm(
                            neighbor_pts - chunk_points[j], axis=1
                        )
                        n_extent[j] = np.max(dists)
                geo_features['vertical_std'][start_idx:end_idx] = v_std
                geo_features['neighborhood_extent'][start_idx:end_idx] = (
                    n_extent
                )
            else:
                # For k-NN, use vectorized computation
                z_neighbors = neighbors_all[:, :, 2]
                geo_features['vertical_std'][start_idx:end_idx] = (
                    np.std(z_neighbors, axis=1).astype(np.float32)
                )
                geo_features['neighborhood_extent'][start_idx:end_idx] = (
                    chunk_distances_max.astype(np.float32)
                )
            
            with np.errstate(divide='ignore', invalid='ignore'):
                v_std_slice = geo_features['vertical_std'][start_idx:end_idx]
                n_ext_slice = (
                    geo_features['neighborhood_extent'][start_idx:end_idx]
                )
                her = v_std_slice / (n_ext_slice + 1e-8)
                her = np.nan_to_num(
                    her, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)
                geo_features['height_extent_ratio'][start_idx:end_idx] = her
            
            roughness_slice = geo_features['roughness'][start_idx:end_idx]
            geo_features['local_roughness'][start_idx:end_idx] = (
                roughness_slice
            )
        
        # Compute verticality and horizontality from normals (always computed)
        chunk_normals_slice = normals[start_idx:end_idx]
        verticality_chunk = compute_verticality(chunk_normals_slice)
        geo_features['verticality'][start_idx:end_idx] = verticality_chunk
        
        # Horizontality = abs(nz) - how horizontal the surface is
        horizontality_chunk = np.abs(chunk_normals_slice[:, 2]).astype(np.float32)
        geo_features['horizontality'][start_idx:end_idx] = horizontality_chunk
    
    # Log completion statistics
    total_features = len(geo_features) + 3  # +3 for normals, curvature, height
    print(
        f"✓ CPU features computed successfully: "
        f"{total_features} feature types, {n_points:,} points, {n_chunks} chunks processed"
    )
    return normals, curvature, height, geo_features


def compute_all_features_optimized(
    points: np.ndarray,
    classification: np.ndarray,
    k: int = None,
    auto_k: bool = True,
    include_extra: bool = False,
    patch_center: np.ndarray = None,
    chunk_size: int = None,
    radius: float = None,
    mode: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute ALL features in a single pass for maximum speed.
    Builds KDTree once and reuses it for all calculations.
    
    This function is 2-3x faster than calling individual functions.
    
    IMPORTANT: Uses RADIUS-based search by default (avoids scan artifacts)
    
    Features computed:
    - Normals (nx, ny, nz): Surface orientation
    - Curvature: Local surface curvature
    - Height: Height above ground
    - Geometric features: planarity, linearity, sphericity, anisotropy,
                         roughness, density
    
    Extra features (if include_extra=True):
    - Height features: z_absolute, z_normalized, z_from_ground, etc.
    - Local statistics: vertical_std, neighborhood_extent, etc.
    - Verticality: wall detection feature
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] ASPRS classification codes
        k: number of neighbors for KNN (if None, auto-computed)
        auto_k: if True, automatically estimate optimal k based on density
        include_extra: if True, compute expensive extra features
        patch_center: [3] patch center for distance_to_center feature
        chunk_size: if specified, process in chunks to reduce memory
        radius: search radius in meters (None=auto, >0=use radius,
                0=use k-NN)
        mode: Feature computation mode ('minimal', 'lod2', 'lod3', 'full').
              If specified, uses the new feature mode system (overrides include_extra)
        
    Returns:
        normals: [N, 3] surface normals
        curvature: [N] curvature values
        height: [N] height above ground
        geo_features: dict with all geometric features
    """
    # If mode parameter is provided, use the new feature mode system
    if mode is not None:
        return compute_features_by_mode(
            points=points,
            classification=classification,
            mode=mode,
            k=k,
            auto_k=auto_k,
            patch_center=patch_center,
            use_radius=(radius is not None and radius > 0),
            radius=radius
        )
    # If chunk_size specified and data is large, use chunked processing
    if chunk_size is not None and len(points) > chunk_size:
        return _compute_all_features_chunked(
            points, classification, k, auto_k, include_extra,
            patch_center, chunk_size, radius
        )
    
    # Auto-estimate optimal k if requested
    if auto_k and k is None:
        k = estimate_optimal_k(points, target_radius=0.5)
        print(f"Auto-estimated k={k} neighbors based on point density")
    elif k is None:
        k = 10  # Default fallback
    
    # Build KDTree once
    tree = KDTree(points, metric='euclidean', leaf_size=30)
    distances, indices = tree.query(points, k=k)
    
    # Get all neighbors: [N, k, 3]
    neighbors_all = points[indices]
    
    # Center neighbors
    centroids = neighbors_all.mean(axis=1, keepdims=True)
    centered = neighbors_all - centroids
    
    # Covariance matrices: [N, 3, 3]
    cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
    
    # === FEATURE 1: NORMALS ===
    normals = eigenvectors[:, :, 0].copy()
    
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = normals / norms
    
    # Orient upward
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] = -normals[flip_mask]
    
    # Handle degenerate cases
    degenerate = (eigenvalues[:, 0] < 1e-8) | np.isnan(normals).any(axis=1)
    normals[degenerate] = np.array([0, 0, 1], dtype=np.float32)
    normals = normals.astype(np.float32)
    
    # === FEATURE 2: CURVATURE ===
    centers = points[:, np.newaxis, :]
    relative_pos = neighbors_all - centers
    normals_expanded = normals[:, np.newaxis, :]
    distances_along_normal = np.sum(relative_pos * normals_expanded, axis=2)
    
    # Use Median Absolute Deviation (MAD) for outlier-robust curvature
    # MAD = median(|x - median(x)|) * 1.4826 (scaled to match std)
    median_dist = np.median(distances_along_normal, axis=1, keepdims=True)
    mad = np.median(np.abs(distances_along_normal - median_dist), axis=1)
    curvature = (mad * 1.4826).astype(np.float32)
    
    # === FEATURE 3: HEIGHT ABOVE GROUND ===
    ground_mask = (classification == 2)
    
    if np.any(ground_mask):
        ground_z = np.median(points[ground_mask, 2])
    else:
        ground_z = np.min(points[:, 2])
    
    height = np.maximum(points[:, 2] - ground_z, 0).astype(np.float32)
    
    # === FEATURE 4: GEOMETRIC FEATURES ===
    # Sort eigenvalues: λ0 >= λ1 >= λ2
    eigenvalues_sorted = np.sort(eigenvalues, axis=1)[:, ::-1]
    
    # Clamp eigenvalues to non-negative (handle numerical artifacts)
    eigenvalues_sorted = np.maximum(eigenvalues_sorted, 0.0)
    
    λ0 = eigenvalues_sorted[:, 0]
    λ1 = eigenvalues_sorted[:, 1]
    λ2 = eigenvalues_sorted[:, 2]
    
    # Éviter division par zéro
    λ0_safe = λ0 + 1e-8
    sum_λ = λ0 + λ1 + λ2 + 1e-8
    
    # Calculer toutes les features géométriques (formules standards)
    # Updated to use λ0 normalization (consistent with GPU/boundary)
    # Formula: Weinmann et al. - normalized by largest eigenvalue λ0
    # Explicitly clamp to [0, 1] to handle edge cases
    linearity = np.clip((λ0 - λ1) / λ0_safe, 0.0, 1.0).astype(np.float32)
    planarity = np.clip((λ1 - λ2) / λ0_safe, 0.0, 1.0).astype(np.float32)
    sphericity = np.clip(λ2 / λ0_safe, 0.0, 1.0).astype(np.float32)
    anisotropy = np.clip((λ0 - λ2) / λ0_safe, 0.0, 1.0).astype(np.float32)
    roughness = np.clip(λ2 / sum_λ, 0.0, 1.0).astype(np.float32)  # Keep sum normalization for roughness
    
    mean_distances = np.mean(distances[:, 1:], axis=1)
    density = np.clip(1.0 / (mean_distances + 1e-8), 0.0, 1000.0).astype(np.float32)
    
    # === VALIDATE AND FILTER DEGENERATE FEATURES ===
    # Points with insufficient/degenerate eigenvalues produce invalid features
    # Set to zero to distinguish from valid low values and prevent NaN propagation
    valid_features = (
        (eigenvalues_sorted[:, 0] >= 1e-6) &  # Non-degenerate largest eigenvalue
        (eigenvalues_sorted[:, 2] >= 1e-8) &  # Non-zero smallest eigenvalue
        ~np.isnan(linearity) &                 # Check for NaN
        ~np.isinf(linearity)                   # Check for Inf
    )
    
    # Set invalid features to zero (distinguishable from valid low values)
    planarity[~valid_features] = 0.0
    linearity[~valid_features] = 0.0
    sphericity[~valid_features] = 0.0
    anisotropy[~valid_features] = 0.0
    roughness[~valid_features] = 0.0
    # Density can remain (computed from distances, not eigenvalues)
    
    # === FACULTATIVE FEATURES: WALL AND ROOF SCORES ===
    # Wall score: High planarity + Vertical surface (|normal_z| close to 0)
    # Roof score: High planarity + Horizontal surface (|normal_z| close to 1)
    verticality = 1.0 - np.abs(normals[:, 2])  # 0=horizontal, 1=vertical
    horizontality = np.abs(normals[:, 2])      # 1=horizontal, 0=vertical
    
    wall_score = (planarity * verticality).astype(np.float32)
    roof_score = (planarity * horizontality).astype(np.float32)
    
    geo_features = {
        'planarity': planarity,
        'linearity': linearity,
        'sphericity': sphericity,
        'anisotropy': anisotropy,
        'roughness': roughness,
        'density': density,
        'wall_score': wall_score,
        'roof_score': roof_score
    }
    
    # === EXTRA FEATURES (if requested) ===
    if include_extra:
        # Height features (CRITICAL for building extraction)
        height_features = compute_height_features(
            points, classification, patch_center
        )
        geo_features.update(height_features)
        
        # Local statistics (EXPENSIVE but powerful)
        # Already computed vertical_std as part of local stats
        z_neighbors = neighbors_all[:, :, 2]  # [N, k]
        vertical_std = np.std(z_neighbors, axis=1).astype(np.float32)
        
        neighborhood_extent = np.max(distances, axis=1).astype(np.float32)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            height_extent_ratio = vertical_std / (neighborhood_extent + 1e-8)
            height_extent_ratio = np.nan_to_num(
                height_extent_ratio, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
        
        # Roughness already computed as λ2/sum_λ above
        local_roughness = roughness  # Reuse
        
        geo_features.update({
            'vertical_std': vertical_std,
            'neighborhood_extent': neighborhood_extent,
            'height_extent_ratio': height_extent_ratio,
            'local_roughness': local_roughness,
        })
        
        # === EIGENVALUE FEATURES (full feature set) ===
        # Compute all eigenvalue-based features for LOD3_FULL mode
        eigenvalue_features = compute_eigenvalue_features(eigenvalues_sorted)
        geo_features.update(eigenvalue_features)
        
        # === ARCHITECTURAL FEATURES (advanced building detection) ===
        # Compute advanced architectural features for detailed analysis
        architectural_features = compute_architectural_features(
            eigenvalues_sorted, normals, points, tree, k
        )
        geo_features.update(architectural_features)
        
        # === DENSITY FEATURES (neighborhood analysis) ===
        # Compute number of points within 2m radius
        num_points_2m = compute_num_points_within_radius(points, tree, radius=2.0)
        geo_features['num_points_2m'] = num_points_2m
    
    # Verticality and horizontality (IMPORTANT for walls/roofs)
    verticality = compute_verticality(normals)
    geo_features['verticality'] = verticality
    
    # Horizontality = abs(nz) - how horizontal the surface is
    horizontality = np.abs(normals[:, 2]).astype(np.float32)
    geo_features['horizontality'] = horizontality
    
    return normals, curvature, height, geo_features


def compute_all_features_with_gpu(
    points: np.ndarray,
    classification: np.ndarray,
    k: int = None,
    auto_k: bool = True,
    use_gpu: bool = False,
    radius: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute all features with optional GPU acceleration.
    
    This is a wrapper around compute_all_features_optimized() that optionally
    uses GPU acceleration if available and requested.
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] ASPRS classification codes
        k: number of neighbors for KNN (if None, auto-computed)
        auto_k: if True, automatically estimate optimal k based on density
        use_gpu: if True, attempt to use GPU acceleration
        radius: search radius in meters for geometric features (default: None)
                If specified, uses radius-based search to avoid LIDAR artifacts
        
    Returns:
        normals: [N, 3] surface normals
        curvature: [N] curvature values
        height: [N] height above ground
        geo_features: dict with all geometric features
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if use_gpu:
        try:
            from .features_gpu import GPUFeatureComputer, GPU_AVAILABLE
            
            if not GPU_AVAILABLE:
                logger.warning(
                    "GPU requested but CuPy not available. Using CPU."
                )
                return compute_all_features_optimized(
                    points, classification, k, auto_k,
                    include_extra=False, patch_center=None, chunk_size=None
                )
            
            logger.info("Using GPU acceleration for feature computation")
            
            # Auto-estimate k if needed
            if auto_k and k is None:
                k = estimate_optimal_k(points, target_radius=0.5)
                logger.info(
                    f"Auto-estimated k={k} neighbors based on density"
                )
            elif k is None:
                k = 10
            
            # Use GPU computer
            computer = GPUFeatureComputer(use_gpu=True)
            
            # Compute features on GPU
            normals = computer.compute_normals(points, k=k)
            curvature = computer.compute_curvature(points, normals, k=k)
            height = computer.compute_height_above_ground(
                points, classification
            )
            geo_features = computer.extract_geometric_features(
                points, normals, k=k, radius=radius
            )
            
            # Add verticality for building detection (wall/roof scoring)
            verticality = computer.compute_verticality(normals)
            geo_features['verticality'] = verticality
            
            return normals, curvature, height, geo_features
            
        except ImportError as e:
            logger.warning(f"GPU requested but not available: {e}")
            logger.warning("Falling back to CPU processing")
        except Exception as e:
            logger.error(f"GPU processing failed: {e}")
            logger.warning("Falling back to CPU processing")
    
    # CPU fallback - note: radius not used in compute_all_features_optimized
    # because it computes normals/curvature with k-NN, only geometric features
    # can use radius via extract_geometric_features separately
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points, classification, k, auto_k,
        include_extra=False
    )
    
    # Add verticality even in CPU fallback (needed for wall/roof scoring)
    if 'verticality' not in geo_features:
        verticality = compute_verticality(normals)
        geo_features['verticality'] = verticality
    
    return normals, curvature, height, geo_features


def compute_features_by_mode(
    points: np.ndarray,
    classification: np.ndarray,
    mode: str = "lod3",
    k: int = 20,
    auto_k: bool = True,
    patch_center: np.ndarray = None,
    use_radius: bool = True,
    radius: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute features based on processing mode (LOD2, LOD3, etc.).
    
    This is the main entry point for mode-based feature computation.
    It computes the appropriate feature set based on the mode:
    - 'minimal': Fast, essential features only (~8 features)
    - 'lod2': LOD2 simplified features (~11 features)
    - 'lod3': LOD3 complete features (~35 features)
    - 'full': All available features
    - 'custom': User-defined (via configuration)
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] ASPRS classification codes
        mode: Processing mode ('minimal', 'lod2', 'lod3', 'full')
        k: Number of neighbors for feature computation
        auto_k: Auto-estimate k based on point density
        patch_center: [3] patch center for distance features
        use_radius: Use radius-based search (recommended)
        radius: Search radius in meters (auto-estimated if None)
    
    Returns:
        normals: [N, 3] surface normals
        curvature: [N] curvature values  
        height: [N] height above ground
        geo_features: Dictionary of all computed features
        
    Example:
        >>> # LOD3 training with complete features
        >>> normals, curv, height, features = compute_features_by_mode(
        ...     points, classification, mode="lod3", k=30
        ... )
        >>> print(len(features))  # ~35 features
        
        >>> # LOD2 training with essential features
        >>> normals, curv, height, features = compute_features_by_mode(
        ...     points, classification, mode="lod2", k=20
        ... )
        >>> print(len(features))  # ~11 features
    """
    from .feature_modes import get_feature_config
    # Functions now in this module:
    # compute_eigenvalue_features, compute_architectural_features,
    # compute_density_features, compute_building_scores
    
    # Get feature configuration for mode
    # Suppress logging here - it's already logged at the orchestrator level
    feature_config = get_feature_config(
        mode=mode,
        k_neighbors=k,
        use_radius=use_radius,
        radius=radius,
        log_config=False
    )
    
    # Estimate optimal k if requested
    if auto_k and k is None:
        k = estimate_optimal_k(points, target_radius=0.5)
        logger.info(f"Auto-estimated k={k} neighbors")
    elif k is None:
        k = feature_config.k_neighbors
    
    # Build KDTree
    tree = KDTree(points, metric='euclidean', leaf_size=30)
    distances, indices = tree.query(points, k=k)
    
    # Get neighbors
    neighbors_all = points[indices]
    centroids = neighbors_all.mean(axis=1, keepdims=True)
    centered = neighbors_all - centroids
    
    # Covariance matrices and eigendecomposition
    cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
    
    # Sort eigenvalues descending
    eigenvalues_sorted = np.sort(eigenvalues, axis=1)[:, ::-1]
    eigenvalues_sorted = np.maximum(eigenvalues_sorted, 0.0)
    
    λ0 = eigenvalues_sorted[:, 0]
    λ1 = eigenvalues_sorted[:, 1]
    λ2 = eigenvalues_sorted[:, 2]
    λ0_safe = λ0 + 1e-8
    sum_λ = λ0 + λ1 + λ2 + 1e-8
    
    # ============================================================
    # COMPUTE CORE FEATURES (always needed)
    # ============================================================
    
    # Normals
    normals = eigenvectors[:, :, 0].copy()
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = normals / norms
    
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] = -normals[flip_mask]
    
    degenerate = (eigenvalues[:, 0] < 1e-8) | np.isnan(normals).any(axis=1)
    normals[degenerate] = np.array([0, 0, 1], dtype=np.float32)
    normals = normals.astype(np.float32)
    
    # Curvature (using MAD for robustness)
    centers = points[:, np.newaxis, :]
    relative_pos = neighbors_all - centers
    normals_expanded = normals[:, np.newaxis, :]
    distances_along_normal = np.sum(relative_pos * normals_expanded, axis=2)
    
    median_dist = np.median(distances_along_normal, axis=1, keepdims=True)
    mad = np.median(np.abs(distances_along_normal - median_dist), axis=1)
    curvature = (mad * 1.4826).astype(np.float32)
    
    # Height above ground
    ground_mask = (classification == 2)
    if np.any(ground_mask):
        ground_z = np.median(points[ground_mask, 2])
    else:
        ground_z = np.min(points[:, 2])
    height = np.maximum(points[:, 2] - ground_z, 0).astype(np.float32)
    
    # ============================================================
    # COMPUTE MODE-SPECIFIC FEATURES
    # ============================================================
    
    geo_features = {}
    feature_set = feature_config.features
    
    # Basic shape descriptors
    if any(f in feature_set for f in ['planarity', 'linearity', 'sphericity', 
                                       'anisotropy', 'roughness']):
        planarity = np.clip((λ1 - λ2) / λ0_safe, 0.0, 1.0).astype(np.float32)
        linearity = np.clip((λ0 - λ1) / λ0_safe, 0.0, 1.0).astype(np.float32)
        sphericity = np.clip(λ2 / λ0_safe, 0.0, 1.0).astype(np.float32)
        anisotropy = np.clip((λ0 - λ2) / λ0_safe, 0.0, 1.0).astype(np.float32)
        roughness = np.clip(λ2 / sum_λ, 0.0, 1.0).astype(np.float32)
        
        # Validate features
        valid_features = (
            (λ0 >= 1e-6) &
            (λ2 >= 1e-8) &
            ~np.isnan(linearity) &
            ~np.isinf(linearity)
        )
        
        planarity[~valid_features] = 0.0
        linearity[~valid_features] = 0.0
        sphericity[~valid_features] = 0.0
        anisotropy[~valid_features] = 0.0
        roughness[~valid_features] = 0.0
        
        if 'planarity' in feature_set:
            geo_features['planarity'] = planarity
        if 'linearity' in feature_set:
            geo_features['linearity'] = linearity
        if 'sphericity' in feature_set:
            geo_features['sphericity'] = sphericity
        if 'anisotropy' in feature_set:
            geo_features['anisotropy'] = anisotropy
        if 'roughness' in feature_set:
            geo_features['roughness'] = roughness
    
    # Density
    if 'density' in feature_set:
        mean_distances = np.mean(distances[:, 1:], axis=1)
        density = np.clip(1.0 / (mean_distances + 1e-8), 0.0, 1000.0)
        geo_features['density'] = density.astype(np.float32)
    
    # Building scores and verticality
    if any(f in feature_set for f in ['verticality', 'wall_score', 'roof_score']):
        if 'planarity' not in locals():
            planarity = np.clip((λ1 - λ2) / λ0_safe, 0.0, 1.0)
        
        verticality, wall_score, roof_score = compute_building_scores(
            planarity, normals
        )
        
        if 'verticality' in feature_set:
            geo_features['verticality'] = verticality
        if 'wall_score' in feature_set:
            geo_features['wall_score'] = wall_score
        if 'roof_score' in feature_set:
            geo_features['roof_score'] = roof_score
    
    # Eigenvalue features
    if any(f in feature_set for f in [
        'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
        'sum_eigenvalues', 'eigenentropy', 'omnivariance', 'change_curvature'
    ]):
        eigenvalue_feats = compute_eigenvalue_features(eigenvalues_sorted)
        for feat_name, feat_values in eigenvalue_feats.items():
            if feat_name in feature_set:
                geo_features[feat_name] = feat_values
    
    # Architectural features
    if any(f in feature_set for f in [
        'edge_strength', 'corner_likelihood', 'overhang_indicator',
        'surface_roughness'
    ]):
        arch_feats = compute_architectural_features(
            eigenvalues_sorted, normals, points, tree, k
        )
        for feat_name, feat_values in arch_feats.items():
            if feat_name in feature_set:
                geo_features[feat_name] = feat_values
    
    # Density features
    if any(f in feature_set for f in [
        'num_points_2m', 'neighborhood_extent', 'height_extent_ratio',
        'vertical_std', 'density'
    ]):
        density_feats = compute_density_features(points, tree, k)
        for feat_name, feat_values in density_feats.items():
            if feat_name in feature_set:
                geo_features[feat_name] = feat_values
    
    # Height features
    if 'height_above_ground' in feature_set:
        geo_features['height_above_ground'] = height
    
    # Additional height features
    if any(f in feature_set for f in [
        'z_absolute', 'z_normalized', 'z_from_ground', 'z_from_median'
    ]):
        height_feats = compute_height_features(
            points, classification, patch_center
        )
        for feat_name, feat_values in height_feats.items():
            if feat_name in feature_set:
                geo_features[feat_name] = feat_values
    
    # Coordinates (xyz)
    if 'xyz' in feature_set:
        # Store XYZ as separate features (will be expanded later)
        geo_features['xyz'] = points.astype(np.float32)
    
    # Normal components
    if 'normal_x' in feature_set:
        geo_features['normal_x'] = normals[:, 0].astype(np.float32)
    if 'normal_y' in feature_set:
        geo_features['normal_y'] = normals[:, 1].astype(np.float32)
    if 'normal_z' in feature_set:
        geo_features['normal_z'] = normals[:, 2].astype(np.float32)
    
    # Curvature (if requested but not already in geo_features)
    if 'curvature' in feature_set and 'curvature' not in geo_features:
        geo_features['curvature'] = curvature
    
    # Spectral features (RGB, NIR, NDVI)
    # Note: These require color data to be passed in, which is typically done
    # in the higher-level processing pipeline. For now, we'll log a warning
    # if they're requested but not available.
    spectral_features = {'red', 'green', 'blue', 'nir', 'ndvi'}
    requested_spectral = spectral_features & feature_set
    if requested_spectral:
        logger.warning(
            f"⚠️  Spectral features requested ({requested_spectral}) but not available in compute_features_by_mode. "
            f"These should be added from LAZ color data in the processing pipeline."
        )
    
    logger.info(f"✓ Computed {len(geo_features)} features for mode '{mode}'")
    
    return normals, curvature, height, geo_features

