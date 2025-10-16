"""
Canonical implementation of architectural feature computation.

This module provides features for detecting and characterizing
architectural elements (buildings, walls, roofs, etc.).
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_architectural_features(
    points: np.ndarray,
    normals: np.ndarray,
    eigenvalues: np.ndarray,
    epsilon: float = 1e-10
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive architectural features.
    
    These features help identify building elements like walls, roofs,
    facades, and other man-made structures.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3) with XYZ coordinates
    normals : np.ndarray
        Normal vectors of shape (N, 3)
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3), sorted in descending order
    epsilon : float, optional
        Small value to prevent division by zero
        
    Returns
    -------
    features : dict
        Dictionary containing:
        - 'verticality': Degree of vertical alignment
        - 'horizontality': Degree of horizontal alignment
        - 'wall_likelihood': Probability of being a wall
        - 'roof_likelihood': Probability of being a roof
        - 'facade_score': Facade characteristic score
        - 'planarity': Planarity measure (from eigenvalues)
        
    Examples
    --------
    >>> points = np.random.rand(1000, 3)
    >>> normals = np.random.rand(1000, 3)
    >>> eigenvalues = np.random.rand(1000, 3)
    >>> features = compute_architectural_features(points, normals, eigenvalues)
    >>> print(features.keys())
    """
    # Input validation
    if points.shape[0] != normals.shape[0] or points.shape[0] != eigenvalues.shape[0]:
        raise ValueError("points, normals, and eigenvalues must have same number of rows")
    
    features = {}
    
    # Verticality: How vertical is the surface (1 = vertical, 0 = horizontal)
    features['verticality'] = compute_verticality(normals)
    
    # Horizontality: How horizontal is the surface
    features['horizontality'] = compute_horizontality(normals)
    
    # Wall likelihood: Combines verticality and planarity
    planarity = compute_planarity(eigenvalues, epsilon)
    features['wall_likelihood'] = compute_wall_likelihood(normals, planarity)
    
    # Roof likelihood: Combines horizontality and planarity
    features['roof_likelihood'] = compute_roof_likelihood(normals, planarity)
    
    # Facade score: Identifies facade elements
    features['facade_score'] = compute_facade_score(points, normals)
    
    # Include planarity
    features['planarity'] = planarity
    
    return features


def compute_verticality(normals: np.ndarray) -> np.ndarray:
    """
    Compute verticality from normal vectors.
    
    Measures how vertically aligned a surface is (e.g., walls).
    
    Parameters
    ----------
    normals : np.ndarray
        Normal vectors of shape (N, 3)
        
    Returns
    -------
    verticality : np.ndarray
        Verticality values of shape (N,), range [0, 1]
        1 = perfectly vertical, 0 = perfectly horizontal
    """
    # Verticality: 1 - |normal_z|
    # When normal is horizontal (perpendicular to vertical), nz = 0, verticality = 1
    # When normal is vertical (parallel to vertical), nz = ±1, verticality = 0
    verticality = 1.0 - np.abs(normals[:, 2])
    
    return verticality.astype(np.float32)


def compute_horizontality(normals: np.ndarray) -> np.ndarray:
    """
    Compute horizontality from normal vectors.
    
    Measures how horizontally aligned a surface is (e.g., roofs, ground).
    
    Parameters
    ----------
    normals : np.ndarray
        Normal vectors of shape (N, 3)
        
    Returns
    -------
    horizontality : np.ndarray
        Horizontality values of shape (N,), range [0, 1]
        1 = perfectly horizontal, 0 = perfectly vertical
    """
    # Vertical direction
    vertical = np.array([0, 0, 1], dtype=np.float32)
    
    # Angle between normal and vertical
    dot_product = np.abs(np.dot(normals, vertical))
    
    # Horizontality: |dot(normal, vertical)|
    horizontality = dot_product
    
    return horizontality.astype(np.float32)


def compute_planarity(eigenvalues: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute planarity from eigenvalues: (λ2 - λ3) / λ1.
    
    High values indicate planar structures.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3), sorted in descending order
    epsilon : float, optional
        Small value to prevent division by zero
        
    Returns
    -------
    planarity : np.ndarray
        Planarity values of shape (N,)
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]
    
    planarity = (lambda2 - lambda3) / (lambda1 + epsilon)
    
    return planarity.astype(np.float32)


def compute_wall_likelihood(
    normals: np.ndarray,
    planarity: np.ndarray,
    verticality_threshold: float = 0.7,
    planarity_threshold: float = 0.6
) -> np.ndarray:
    """
    Compute likelihood of points belonging to a wall.
    
    Walls are characterized by high verticality and high planarity.
    
    Parameters
    ----------
    normals : np.ndarray
        Normal vectors of shape (N, 3)
    planarity : np.ndarray
        Planarity values of shape (N,)
    verticality_threshold : float, optional
        Minimum verticality for walls (default: 0.7)
    planarity_threshold : float, optional
        Minimum planarity for walls (default: 0.6)
        
    Returns
    -------
    wall_likelihood : np.ndarray
        Wall likelihood scores of shape (N,), range [0, 1]
    """
    verticality = compute_verticality(normals)
    
    # Combine verticality and planarity
    # Use geometric mean to require both to be high
    wall_likelihood = np.sqrt(verticality * planarity)
    
    # Apply soft thresholding
    wall_likelihood = np.clip(wall_likelihood, 0, 1)
    
    return wall_likelihood.astype(np.float32)


def compute_roof_likelihood(
    normals: np.ndarray,
    planarity: np.ndarray,
    horizontality_threshold: float = 0.7,
    planarity_threshold: float = 0.6
) -> np.ndarray:
    """
    Compute likelihood of points belonging to a roof.
    
    Roofs are characterized by high horizontality and high planarity.
    
    Parameters
    ----------
    normals : np.ndarray
        Normal vectors of shape (N, 3)
    planarity : np.ndarray
        Planarity values of shape (N,)
    horizontality_threshold : float, optional
        Minimum horizontality for roofs (default: 0.7)
    planarity_threshold : float, optional
        Minimum planarity for roofs (default: 0.6)
        
    Returns
    -------
    roof_likelihood : np.ndarray
        Roof likelihood scores of shape (N,), range [0, 1]
    """
    horizontality = compute_horizontality(normals)
    
    # Combine horizontality and planarity
    roof_likelihood = np.sqrt(horizontality * planarity)
    
    # Apply soft thresholding
    roof_likelihood = np.clip(roof_likelihood, 0, 1)
    
    return roof_likelihood.astype(np.float32)


def compute_facade_score(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Compute facade characteristic score.
    
    Facades typically have consistent vertical alignment and are
    aligned with building boundaries.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud array of shape (N, 3)
    normals : np.ndarray
        Normal vectors of shape (N, 3)
        
    Returns
    -------
    facade_score : np.ndarray
        Facade scores of shape (N,), higher values indicate facade-like characteristics
    """
    # Verticality component
    verticality = compute_verticality(normals)
    
    # Height above ground (normalized)
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    relative_height = (points[:, 2] - z_min) / (z_max - z_min + 1e-10)
    
    # Facades are typically above ground level
    height_factor = np.clip(relative_height * 2, 0, 1)
    
    # Combine verticality and height
    facade_score = verticality * height_factor
    
    return facade_score.astype(np.float32)


def compute_building_regularity(
    eigenvalues: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Compute regularity score for building structures.
    
    Buildings typically have regular, structured geometry.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3)
    epsilon : float, optional
        Small value to prevent division by zero
        
    Returns
    -------
    regularity : np.ndarray
        Regularity scores of shape (N,)
    """
    planarity = compute_planarity(eigenvalues, epsilon)
    
    # Low sphericity (not volumetric) combined with high planarity
    lambda1 = eigenvalues[:, 0]
    lambda3 = eigenvalues[:, 2]
    sphericity = lambda3 / (lambda1 + epsilon)
    
    regularity = planarity * (1.0 - sphericity)
    
    return regularity.astype(np.float32)


def compute_corner_likelihood(eigenvalues: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute likelihood of being a corner or edge.
    
    Corners have high linearity (edge-like eigenvalue distribution).
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3)
    epsilon : float, optional
        Small value to prevent division by zero
        
    Returns
    -------
    corner_likelihood : np.ndarray
        Corner likelihood scores of shape (N,)
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    
    # Linearity: (λ1 - λ2) / λ1
    linearity = (lambda1 - lambda2) / (lambda1 + epsilon)
    
    return linearity.astype(np.float32)
