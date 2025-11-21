"""
Canonical implementation of eigenvalue-based features.

This module provides eigenvalue feature computation.
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def compute_eigenvalue_features(
    eigenvalues: np.ndarray, epsilon: float = 1e-6, include_all: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive eigenvalue-based geometric features.

    These features describe local geometric properties of the point cloud
    based on the eigenvalues of the local covariance matrix.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of shape (N, 3), sorted descending (λ1 >= λ2 >= λ3)
    epsilon : float, optional
        Small value to prevent division by zero (default: 1e-6)
        Note: 1e-6 chosen for float32 stability (float32 eps ≈ 1.19e-7)
    include_all : bool, optional
        If True, compute all features. If False, compute only basic features.
        (default: True)

    Returns
    -------
    features : dict
        Dictionary containing computed features:

        Basic features:
        - 'linearity': (λ1 - λ2) / λ1 - linear structures
        - 'planarity': (λ2 - λ3) / λ1 - planar structures
        - 'sphericity': λ3 / λ1 - spherical/volumetric structures
        - 'anisotropy': (λ1 - λ3) / λ1 - degree of anisotropy
        - 'eigenentropy': -Σ(λi * log(λi)) - structural complexity
        - 'omnivariance': (λ1 * λ2 * λ3)^(1/3) - local volume
        - 'sum_eigenvalues': λ1 + λ2 + λ3 - total variance

        Advanced features (if include_all=True):
        - 'change_of_curvature': λ3 / (λ1 + λ2 + λ3) - surface variation
        - 'verticality': 1 - |λ3| / λ1 - vertical alignment
        - 'surface_variation': λ3 / (λ1 + λ2 + λ3) - local roughness

    Examples
    --------
    >>> eigenvalues = np.array([[1.0, 0.5, 0.1], [1.0, 0.01, 0.001]])
    >>> features = compute_eigenvalue_features(eigenvalues)
    >>> print(features['linearity'])  # High for second point (linear)
    [0.5 0.99]

    Notes
    -----
    Feature interpretation:
    - Linearity ~ 1: Linear features (poles, wires, tree trunks)
    - Planarity ~ 1: Planar features (walls, roofs, ground)
    - Sphericity ~ 1: Volumetric features (vegetation, complex objects)
    """
    # Input validation
    if not isinstance(eigenvalues, np.ndarray):
        raise ValueError("eigenvalues must be a numpy array")
    if eigenvalues.ndim != 2 or eigenvalues.shape[1] != 3:
        raise ValueError(f"eigenvalues must have shape (N, 3), got {eigenvalues.shape}")

    # Extract eigenvalues
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]

    # Ensure safe denominator to prevent Inf/NaN
    lambda1_safe = np.maximum(lambda1, epsilon)

    # Compute basic features
    features = {}

    # Linearity: (λ1 - λ2) / λ1
    linearity = (lambda1_safe - lambda2) / lambda1_safe
    features["linearity"] = np.clip(linearity, 0.0, 1.0).astype(np.float32)

    # Planarity: (λ2 - λ3) / λ1
    planarity = (lambda2 - lambda3) / lambda1_safe
    features["planarity"] = np.clip(planarity, 0.0, 1.0).astype(np.float32)

    # Sphericity (Scattering): λ3 / λ1
    sphericity = lambda3 / lambda1_safe
    features["sphericity"] = np.clip(sphericity, 0.0, 1.0).astype(np.float32)

    # Anisotropy: (λ1 - λ3) / λ1
    anisotropy = (lambda1_safe - lambda3) / lambda1_safe
    features["anisotropy"] = np.clip(anisotropy, 0.0, 1.0).astype(np.float32)

    # Eigenentropy: -Σ(λi * log(λi))
    # Normalize eigenvalues first
    sum_eigenvalues = lambda1 + lambda2 + lambda3 + epsilon
    l1_norm = lambda1 / sum_eigenvalues
    l2_norm = lambda2 / sum_eigenvalues
    l3_norm = lambda3 / sum_eigenvalues

    eigenentropy = -(
        l1_norm * np.log(l1_norm + epsilon)
        + l2_norm * np.log(l2_norm + epsilon)
        + l3_norm * np.log(l3_norm + epsilon)
    )
    features["eigenentropy"] = eigenentropy

    # Omnivariance: (λ1 * λ2 * λ3)^(1/3)
    features["omnivariance"] = np.cbrt(lambda1 * lambda2 * lambda3 + epsilon)

    # Sum of eigenvalues
    features["sum_eigenvalues"] = lambda1 + lambda2 + lambda3

    # Individual eigenvalues (always include for compatibility)
    features["eigenvalue_1"] = lambda1
    features["eigenvalue_2"] = lambda2
    features["eigenvalue_3"] = lambda3

    if include_all:
        # Change of curvature (surface variation) - alias for compatibility
        change_curvature = lambda3 / (sum_eigenvalues + epsilon)
        features["change_of_curvature"] = change_curvature
        features["change_curvature"] = change_curvature  # Legacy alias

        # Verticality (1 - normalized smallest eigenvalue)
        features["verticality"] = 1.0 - np.abs(lambda3) / (lambda1 + epsilon)

        # Surface variation (same as change of curvature, but different interpretation)
        features["surface_variation"] = lambda3 / (sum_eigenvalues + epsilon)

    # Convert all to float32
    for key in features:
        features[key] = features[key].astype(np.float32)

    return features


def compute_linearity(eigenvalues: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute linearity feature: (λ1 - λ2) / λ1.

    High values indicate linear structures (poles, edges, tree trunks).

    Note: epsilon=1e-6 chosen for float32 numerical stability (float32 epsilon ≈ 1.19e-7).
          This prevents Inf/NaN artifacts in sparse/noisy regions.
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]

    # Ensure safe denominator
    lambda1_safe = np.maximum(lambda1, epsilon)

    # Compute and clamp to valid range [0, 1]
    linearity = (lambda1_safe - lambda2) / lambda1_safe
    linearity = np.clip(linearity, 0.0, 1.0)

    return linearity.astype(np.float32)


def compute_planarity(eigenvalues: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute planarity feature: (λ2 - λ3) / λ1.

    High values indicate planar structures (walls, roofs, ground).

    Note: epsilon=1e-6 chosen for float32 numerical stability (float32 epsilon ≈ 1.19e-7).
          This prevents Inf/NaN artifacts in sparse/noisy regions.
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]

    # Ensure safe denominator
    lambda1_safe = np.maximum(lambda1, epsilon)

    # Compute and clamp to valid range [0, 1]
    planarity = (lambda2 - lambda3) / lambda1_safe
    planarity = np.clip(planarity, 0.0, 1.0)

    return planarity.astype(np.float32)


def compute_sphericity(eigenvalues: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute sphericity feature: λ3 / λ1.

    High values indicate volumetric/spherical structures (vegetation, complex objects).

    Note: epsilon=1e-6 chosen for float32 numerical stability (float32 epsilon ≈ 1.19e-7).
          This prevents Inf/NaN artifacts in sparse/noisy regions.
    """
    lambda1 = eigenvalues[:, 0]
    lambda3 = eigenvalues[:, 2]

    # Ensure safe denominator
    lambda1_safe = np.maximum(lambda1, epsilon)

    # Compute and clamp to valid range [0, 1]
    sphericity = lambda3 / lambda1_safe
    sphericity = np.clip(sphericity, 0.0, 1.0)

    return sphericity.astype(np.float32)


def compute_anisotropy(eigenvalues: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute anisotropy feature: (λ1 - λ3) / λ1.

    Measures the degree of anisotropy in the local neighborhood.

    Note: epsilon=1e-6 chosen for float32 numerical stability (float32 epsilon ≈ 1.19e-7).
          This prevents Inf/NaN artifacts in sparse/noisy regions.
    """
    lambda1 = eigenvalues[:, 0]
    lambda3 = eigenvalues[:, 2]

    # Ensure safe denominator
    lambda1_safe = np.maximum(lambda1, epsilon)

    # Compute and clamp to valid range [0, 1]
    anisotropy = (lambda1_safe - lambda3) / lambda1_safe
    anisotropy = np.clip(anisotropy, 0.0, 1.0)

    return anisotropy.astype(np.float32)


def compute_omnivariance(eigenvalues: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute omnivariance feature: (λ1 * λ2 * λ3)^(1/3).

    Represents the local 3D scatter or density.
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]
    return np.cbrt(lambda1 * lambda2 * lambda3 + epsilon).astype(np.float32)


def compute_eigenentropy(eigenvalues: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute eigenentropy: -Σ(λi * log(λi)).

    Measures structural complexity of the local neighborhood.
    """
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]

    sum_eigenvalues = lambda1 + lambda2 + lambda3 + epsilon
    l1_norm = lambda1 / sum_eigenvalues
    l2_norm = lambda2 / sum_eigenvalues
    l3_norm = lambda3 / sum_eigenvalues

    eigenentropy = -(
        l1_norm * np.log(l1_norm + epsilon)
        + l2_norm * np.log(l2_norm + epsilon)
        + l3_norm * np.log(l3_norm + epsilon)
    )

    return eigenentropy.astype(np.float32)


def compute_verticality(eigenvalues: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute verticality from eigenvalues: 1 - |λ3| / λ1.

    High values indicate vertical structures.

    Note: This is the eigenvalue-based verticality. For normal-based verticality,
    use compute_normal_verticality() from the main features module.
    epsilon=1e-6 chosen for float32 numerical stability (float32 epsilon ≈ 1.19e-7).
    """
    lambda1 = eigenvalues[:, 0]
    lambda3 = eigenvalues[:, 2]

    # Ensure safe denominator
    lambda1_safe = np.maximum(lambda1, epsilon)

    # Compute and clamp to valid range [0, 1]
    verticality = 1.0 - np.abs(lambda3) / lambda1_safe
    verticality = np.clip(verticality, 0.0, 1.0)

    return verticality.astype(np.float32)
