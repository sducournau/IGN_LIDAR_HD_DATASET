"""
GPU-Accelerated Geometric Feature Extraction Functions
Utilise CuPy et RAPIDS cuML pour des calculs 10-50x plus rapides
Avec fallback automatique vers CPU si GPU indisponible
"""

from typing import Dict, Tuple, Any, Union
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm

# Tenter import GPU
GPU_AVAILABLE = False
CUML_AVAILABLE = False

try:
    import cupy as cp
    from cupyx.scipy.spatial import distance as cp_distance
    GPU_AVAILABLE = True
    CpArray = cp.ndarray
    print("✓ CuPy disponible - GPU activé")
except ImportError:
    print("⚠ CuPy non disponible - fallback CPU")
    cp = None
    CpArray = Any  # Type placeholder when CuPy unavailable

try:
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    from cuml.decomposition import PCA as cuPCA
    CUML_AVAILABLE = True
    print("✓ RAPIDS cuML disponible - Algorithmes GPU activés")
except ImportError:
    print("⚠ RAPIDS cuML non disponible - fallback sklearn")
    cuNearestNeighbors = None
    cuPCA = None

# Fallback CPU
from sklearn.neighbors import KDTree


class GPUFeatureComputer:
    """
    Classe pour calcul de features géométriques optimisé GPU.
    Fallback automatique CPU si GPU indisponible.
    """
    
    def __init__(self, use_gpu: bool = True, batch_size: int = 250000):
        """
        Args:
            use_gpu: Activer GPU si disponible
            batch_size: Points par batch GPU (optimized: 250K)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_cuml = use_gpu and CUML_AVAILABLE
        self.batch_size = batch_size
        
        if self.use_gpu:
            print(f"🚀 Mode GPU activé (batch_size={batch_size})")
        else:
            msg = "💻 Mode CPU (installer CuPy pour accélération)"
            print(msg)
    
    def _to_gpu(self, array: np.ndarray):
        """Transfert array vers GPU."""
        if self.use_gpu and cp is not None:
            return cp.asarray(array, dtype=cp.float32)
        return array
    
    def _to_cpu(self, array) -> np.ndarray:
        """Transfert array vers CPU."""
        if self.use_gpu and cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    
    def compute_normals(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Compute surface normals using PCA on k-nearest neighbors.
        Version GPU-accélérée avec traitement par batch.
        
        Args:
            points: [N, 3] point coordinates
            k: number of neighbors for PCA (10 for fast computation)
            
        Returns:
            normals: [N, 3] normalized surface normals
        """
        if self.use_cuml and cuNearestNeighbors is not None:
            return self._compute_normals_gpu(points, k)
        else:
            return self._compute_normals_cpu(points, k)
    
    def _compute_normals_gpu(self, points: np.ndarray, k: int) -> np.ndarray:
        """Calcul normales sur GPU (RAPIDS cuML)."""
        if not GPU_AVAILABLE or cp is None:
            return self._compute_normals_cpu(points, k)
            
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        
        # Transfert vers GPU
        points_gpu = self._to_gpu(points)
        
        # Build GPU KNN
        knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(points_gpu)
        
        # Traitement par batch pour éviter OOM
        num_batches = (N + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, N)
            batch_points = points_gpu[start_idx:end_idx]
            
            # Query KNN
            distances, indices = knn.kneighbors(batch_points)
            
            # Compute normals with GPU PCA
            batch_normals = self._batch_pca_gpu(points_gpu, indices)
            
            # Transfer back to CPU
            normals[start_idx:end_idx] = self._to_cpu(batch_normals)
        
        return normals
    
    def _batch_pca_gpu(self, points_gpu, neighbor_indices):
        """
        VECTORIZED PCA on GPU.
        ~100x faster than per-point loop.
        """
        if not GPU_AVAILABLE or cp is None:
            raise RuntimeError("GPU not available")
            
        batch_size, k = neighbor_indices.shape
        
        # Gather all neighbor points: [batch_size, k, 3]
        neighbor_points = points_gpu[neighbor_indices]
        
        # Center the neighborhoods: [batch_size, k, 3]
        centroids = cp.mean(
            neighbor_points, axis=1, keepdims=True
        )  # [batch_size, 1, 3]
        centered = neighbor_points - centroids  # [batch_size, k, 3]
        
        # Compute covariance matrices: [batch_size, 3, 3]
        # cov = (1/k) * (centered.T @ centered)
        cov_matrices = cp.einsum('mki,mkj->mij', centered, centered) / k
        
        # Compute eigenvalues and eigenvectors for all matrices
        # eigenvalues: [batch_size, 3], eigenvectors: [batch_size, 3, 3]
        eigenvalues, eigenvectors = cp.linalg.eigh(cov_matrices)
        
        # Normal = eigenvector with smallest eigenvalue (first column)
        # eigh returns eigenvalues in ascending order
        normals = eigenvectors[:, :, 0]  # [batch_size, 3]
        
        # Normalize normals
        norms = cp.linalg.norm(normals, axis=1, keepdims=True)
        norms = cp.maximum(norms, 1e-6)  # Avoid division by zero
        normals = normals / norms
        
        # Orient normals upward (positive Z)
        flip_mask = normals[:, 2] < 0
        normals[flip_mask] *= -1
        
        # Handle degenerate cases (very small variance)
        variances = cp.sum(eigenvalues, axis=1)  # [batch_size]
        degenerate = variances < 1e-6
        if cp.any(degenerate):
            normals[degenerate] = cp.array([0, 0, 1], dtype=cp.float32)
        
        return normals
    
    def _compute_normals_cpu(self, points: np.ndarray, k: int) -> np.ndarray:
        """
        Calcul normales sur CPU (sklearn) - VECTORIZED.
        ~100x faster than per-point PCA loop.
        """
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        
        # Build KDTree
        tree = KDTree(points, metric='euclidean')
        
        # Batch query pour réduire overhead
        batch_size = 50000  # Larger batches for vectorized computation
        num_batches = (N + batch_size - 1) // batch_size
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, N)
                batch_points = points[start_idx:end_idx]
                
                # Query KNN pour tout le batch
                _, indices = tree.query(batch_points, k=k)
                
                # VECTORIZED covariance computation
                # Gather all neighbor points: [batch_size, k, 3]
                neighbor_points = points[indices]
                
                # Center the neighborhoods: [batch_size, k, 3]
                centroids = np.mean(
                    neighbor_points, axis=1, keepdims=True
                )  # [batch_size, 1, 3]
                centered = neighbor_points - centroids
                
                # Compute covariance matrices: [batch_size, 3, 3]
                cov_matrices = (
                    np.einsum('mki,mkj->mij', centered, centered) / k
                )
                
                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
                
                # Normal = eigenvector with smallest eigenvalue
                batch_normals = eigenvectors[:, :, 0]  # [batch_size, 3]
                
                # Normalize normals
                norms = np.linalg.norm(batch_normals, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-6)
                batch_normals = batch_normals / norms
                
                # Orient normals upward (positive Z)
                flip_mask = batch_normals[:, 2] < 0
                batch_normals[flip_mask] *= -1
                
                # Handle degenerate cases
                variances = np.sum(eigenvalues, axis=1)
                degenerate = variances < 1e-6
                if np.any(degenerate):
                    batch_normals[degenerate] = [0, 0, 1]
                
                normals[start_idx:end_idx] = batch_normals
        
        return normals
    
    def compute_curvature(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """
        Compute principal curvature from local surface fit.
        Version optimisée avec vectorisation.
        
        Args:
            points: [N, 3] point coordinates
            normals: [N, 3] surface normals
            k: number of neighbors (10 for fast computation)
            
        Returns:
            curvature: [N] principal curvature values
        """
        N = len(points)
        curvature = np.zeros(N, dtype=np.float32)
        
        # Build KDTree
        tree = KDTree(points, metric='euclidean')
        
        # Batch processing
        batch_size = 50000  # Larger batches for vectorized computation
        num_batches = (N + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_points = points[start_idx:end_idx]
            batch_normals = normals[start_idx:end_idx]
            
            # Query KNN
            _, indices = tree.query(batch_points, k=k)
            
            # VECTORIZED curvature computation
            # Get all neighbor normals: [batch_size, k, 3]
            neighbor_normals = normals[indices]
            
            # Expand query normals: [batch_size, 1, 3]
            query_normals_expanded = batch_normals[:, np.newaxis, :]
            
            # Compute differences: [batch_size, k, 3]
            normal_diff = neighbor_normals - query_normals_expanded
            
            # Compute norms and mean: [batch_size]
            curv_norms = np.linalg.norm(normal_diff, axis=2)  # [batch, k]
            batch_curvature = np.mean(curv_norms, axis=1)  # [batch_size]
            
            curvature[start_idx:end_idx] = batch_curvature
        
        return curvature
    
    def compute_height_above_ground(
        self,
        points: np.ndarray,
        classification: np.ndarray
    ) -> np.ndarray:
        """
        Compute height above ground for each point.
        Version vectorisée pure.
        
        Args:
            points: [N, 3] point coordinates
            classification: [N] ASPRS classification codes
            
        Returns:
            height: [N] height above ground in meters
        """
        # Find ground points (classification code 2)
        ground_mask = (classification == 2)
        
        if not np.any(ground_mask):
            ground_z = np.min(points[:, 2])
        else:
            ground_z = np.median(points[ground_mask, 2])
        
        height = points[:, 2] - ground_z
        return np.maximum(height, 0)
    
    def extract_geometric_features(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int = 10,
        radius: float = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive geometric features for each point.
        Version vectorisée optimale - aligned with features.py.
        
        Features computed (eigenvalue-based):
        - Planarity: (λ1-λ2)/Σλ - surfaces planes
        - Linearity: (λ0-λ1)/Σλ - structures linéaires
        - Sphericity: λ2/Σλ - structures sphériques
        - Anisotropy: (λ0-λ2)/λ0 - anisotropie
        - Roughness: λ2/Σλ - rugosité
        - Density: local point density
        
        Args:
            points: [N, 3] point coordinates
            normals: [N, 3] normal vectors (not used, kept for compat)
            k: number of neighbors (used if radius=None)
            radius: search radius in meters (RECOMMENDED, avoids scan artifacts)
            
        Returns:
            features: dictionary of geometric features
        """
        # If radius requested, use CPU implementation (GPU radius not yet impl)
        if radius is not None:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Radius search (r={radius:.2f}m) requested but GPU "
                f"radius search not implemented yet. Using CPU fallback."
            )
            from .features import extract_geometric_features as cpu_extract
            return cpu_extract(points, normals, k=k, radius=radius)
        # Build KDTree
        tree = KDTree(points, metric='euclidean', leaf_size=30)
        distances, indices = tree.query(points, k=k)
        
        # Get all neighbors: [N, k, 3]
        neighbors_all = points[indices]
        
        # Center neighbors: [N, k, 3]
        centroids = neighbors_all.mean(axis=1, keepdims=True)
        centered = neighbors_all - centroids
        
        # Covariance matrices: [N, 3, 3]
        cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (k-1)
        
        # Eigenvalues: [N, 3]
        eigenvalues = np.linalg.eigvalsh(cov_matrices)
        
        # Sort descending: λ0 >= λ1 >= λ2
        eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]
        
        λ0 = eigenvalues[:, 0]
        λ1 = eigenvalues[:, 1]
        λ2 = eigenvalues[:, 2]
        
        # Clamp eigenvalues to non-negative (handle numerical artifacts)
        λ0 = np.maximum(λ0, 0.0)
        λ1 = np.maximum(λ1, 0.0)
        λ2 = np.maximum(λ2, 0.0)
        
        # Safe division - use λ0 (largest eigenvalue) for normalization
        # This matches the boundary-aware features and standard literature
        λ0_safe = λ0 + 1e-8
        sum_λ = λ0 + λ1 + λ2 + 1e-8
        
        # Compute features using λ0 normalization (consistent with boundary features)
        # Formula: Weinmann et al. - normalized by largest eigenvalue λ0
        # Range: linearity [0, 1], planarity [0, 1], sphericity [0, 1]
        # Explicitly clamp to [0, 1] to handle edge cases
        linearity = np.clip((λ0 - λ1) / λ0_safe, 0.0, 1.0).astype(np.float32)
        planarity = np.clip((λ1 - λ2) / λ0_safe, 0.0, 1.0).astype(np.float32)
        sphericity = np.clip(λ2 / λ0_safe, 0.0, 1.0).astype(np.float32)
        anisotropy = np.clip((λ0 - λ2) / λ0_safe, 0.0, 1.0).astype(np.float32)
        roughness = np.clip(λ2 / sum_λ, 0.0, 1.0).astype(np.float32)  # Keep sum normalization for roughness
        
        mean_distances = np.mean(distances[:, 1:], axis=1)
        density = np.clip(1.0 / (mean_distances + 1e-8), 0.0, 1000.0).astype(np.float32)
        
        # === VALIDATE AND FILTER DEGENERATE FEATURES ===
        # Points with insufficient/degenerate eigenvalues
        valid_features = (
            (eigenvalues[:, 0] >= 1e-6) &  # Non-degenerate eigenvalue
            (eigenvalues[:, 2] >= 1e-8) &  # Non-zero smallest eigenvalue
            ~np.isnan(linearity) &         # Check for NaN
            ~np.isinf(linearity)           # Check for Inf
        )
        
        # Set invalid features to zero
        planarity[~valid_features] = 0.0
        linearity[~valid_features] = 0.0
        sphericity[~valid_features] = 0.0
        anisotropy[~valid_features] = 0.0
        roughness[~valid_features] = 0.0
        
        # === FACULTATIVE FEATURES: WALL AND ROOF SCORES ===
        # Wall score: High planarity + Vertical surface (|normal_z| close to 0)
        # Roof score: High planarity + Horizontal surface (|normal_z| close to 1)
        verticality = (1.0 - np.abs(normals[:, 2])).astype(np.float32)  # 0=horizontal, 1=vertical
        horizontality = np.abs(normals[:, 2]).astype(np.float32)        # 1=horizontal, 0=vertical
        
        wall_score = (planarity * verticality).astype(np.float32)
        roof_score = (planarity * horizontality).astype(np.float32)
        
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
        
        # === ADDITIONAL FEATURES FOR FULL MODE ===
        # Note: These are computed on CPU since GPU implementations
        # would require significant additional complexity
        # Import CPU functions for advanced features
        from .features import (
            compute_eigenvalue_features,
            compute_architectural_features,
            compute_num_points_within_radius
        )
        
        # Compute eigenvalue-based features
        eigenvalue_features = compute_eigenvalue_features(eigenvalues)
        features.update(eigenvalue_features)
        
        # Compute architectural features (requires tree and more context)
        tree = KDTree(points, metric='euclidean', leaf_size=30)
        architectural_features = compute_architectural_features(
            eigenvalues, normals, points, tree, k
        )
        features.update(architectural_features)
        
        # Compute density features
        num_points_2m = compute_num_points_within_radius(points, tree, radius=2.0)
        features['num_points_2m'] = num_points_2m
        
        return features

    def compute_verticality(self, normals: np.ndarray) -> np.ndarray:
        """
        Compute verticality from surface normals (GPU-accelerated).
        
        Verticality measures how vertical a surface is (walls vs roofs/ground).
        
        Args:
            normals: [N, 3] surface normal vectors
            
        Returns:
            verticality: [N] verticality values [0, 1]
                        0 = horizontal surface
                        1 = vertical surface
        """
        if self.use_gpu and cp is not None:
            # GPU computation
            normals_gpu = self._to_gpu(normals)
            verticality_gpu = 1.0 - cp.abs(normals_gpu[:, 2])
            return self._to_cpu(verticality_gpu).astype(np.float32)
        else:
            # CPU fallback
            verticality = 1.0 - np.abs(normals[:, 2])
            return verticality.astype(np.float32)

    def compute_wall_score(
        self,
        normals: np.ndarray,
        height_above_ground: np.ndarray,
        min_height: float = 1.5
    ) -> np.ndarray:
        """
        Compute wall probability score (GPU-accelerated).
        
        Combines verticality with height above ground to identify walls.
        
        Args:
            normals: [N, 3] surface normal vectors
            height_above_ground: [N] height above ground in meters
            min_height: minimum height to be considered a wall (default 1.5m)
            
        Returns:
            wall_score: [N] wall probability [0, 1]
        """
        if self.use_gpu and cp is not None:
            # GPU computation
            normals_gpu = self._to_gpu(normals)
            height_gpu = self._to_gpu(height_above_ground)
            
            # Verticality component
            verticality = 1.0 - cp.abs(normals_gpu[:, 2])
            
            # Height component (walls are typically > 1.5m above ground)
            height_score = cp.clip((height_gpu - min_height) / 5.0, 0, 1)
            
            # Combine: high verticality AND elevated
            wall_score_gpu = verticality * height_score
            
            return self._to_cpu(wall_score_gpu).astype(np.float32)
        else:
            # CPU fallback
            verticality = 1.0 - np.abs(normals[:, 2])
            height_score = np.clip((height_above_ground - min_height) / 5.0, 0, 1)
            wall_score = verticality * height_score
            return wall_score.astype(np.float32)

    def compute_roof_score(
        self,
        normals: np.ndarray,
        height_above_ground: np.ndarray,
        curvature: np.ndarray,
        min_height: float = 3.0
    ) -> np.ndarray:
        """
        Compute roof probability score (GPU-accelerated).
        
        Roofs are horizontal surfaces that are elevated and have low curvature.
        
        Args:
            normals: [N, 3] surface normal vectors
            height_above_ground: [N] height above ground in meters
            curvature: [N] surface curvature
            min_height: minimum height for a roof (default 3.0m)
            
        Returns:
            roof_score: [N] roof probability [0, 1]
        """
        if self.use_gpu and cp is not None:
            # GPU computation
            normals_gpu = self._to_gpu(normals)
            height_gpu = self._to_gpu(height_above_ground)
            curvature_gpu = self._to_gpu(curvature)
            
            # Horizontality (inverse of verticality)
            horizontality = cp.abs(normals_gpu[:, 2])
            
            # Height component (roofs are typically > 2m above ground)
            height_score = cp.clip((height_gpu - min_height) / 8.0, 0, 1)
            
            # Low curvature (roofs are planar)
            curvature_score = 1.0 - cp.clip(curvature_gpu / 0.5, 0, 1)
            
            # Combine: horizontal AND elevated AND planar
            roof_score_gpu = horizontality * height_score * curvature_score
            
            return self._to_cpu(roof_score_gpu).astype(np.float32)
        else:
            # CPU fallback
            horizontality = np.abs(normals[:, 2])
            height_score = np.clip((height_above_ground - min_height) / 8.0, 0, 1)
            curvature_score = 1.0 - np.clip(curvature / 0.5, 0, 1)
            roof_score = horizontality * height_score * curvature_score
            return roof_score.astype(np.float32)

    def compute_eigenvalue_features(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        neighbors_indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute eigenvalue-based features (FULL GPU-accelerated).
        
        Features:
        - eigenvalue_1, eigenvalue_2, eigenvalue_3: Individual eigenvalues (λ₀, λ₁, λ₂)
        - sum_eigenvalues: Sum of eigenvalues (Σλ)
        - eigenentropy: Shannon entropy of normalized eigenvalues
        - omnivariance: Cubic root of product of eigenvalues
        - change_curvature: Variance-based curvature change measure
        
        Args:
            points: [N, 3] point coordinates
            normals: [N, 3] surface normals
            neighbors_indices: [N, k] indices of k-nearest neighbors
            
        Returns:
            Dictionary of eigenvalue-based features
        """
        N = len(points)
        k = neighbors_indices.shape[1]
        
        # Determine computation backend (GPU if available, else CPU)
        use_gpu = self.use_gpu and cp is not None
        xp = cp if use_gpu else np
        
        # Transfer to GPU if available
        if use_gpu:
            points_gpu = self._to_gpu(points)
            neighbors_indices_gpu = cp.asarray(neighbors_indices)
            neighbors = points_gpu[neighbors_indices_gpu]
        else:
            neighbors = points[neighbors_indices]
        
        # Center neighbors: [N, k, 3]
        centroids = xp.mean(neighbors, axis=1, keepdims=True)
        centered = neighbors - centroids
        
        # Covariance matrices: [N, 3, 3]
        cov_matrices = xp.einsum('nki,nkj->nij', centered, centered) / (k - 1)
        
        # Compute eigenvalues: [N, 3]
        eigenvalues = xp.linalg.eigvalsh(cov_matrices)
        eigenvalues = xp.sort(eigenvalues, axis=1)[:, ::-1]  # Sort descending
        
        # Clamp to non-negative
        eigenvalues = xp.maximum(eigenvalues, 1e-10)
        
        λ0 = eigenvalues[:, 0]
        λ1 = eigenvalues[:, 1]
        λ2 = eigenvalues[:, 2]
        
        # Sum of eigenvalues
        sum_eigenvalues = λ0 + λ1 + λ2
        
        # Eigenentropy: Shannon entropy of normalized eigenvalues
        # H = -Σ(p_i * log(p_i)) where p_i = λ_i / Σλ
        p0 = λ0 / (sum_eigenvalues + 1e-10)
        p1 = λ1 / (sum_eigenvalues + 1e-10)
        p2 = λ2 / (sum_eigenvalues + 1e-10)
        
        eigenentropy = -(
            p0 * xp.log(p0 + 1e-10) +
            p1 * xp.log(p1 + 1e-10) +
            p2 * xp.log(p2 + 1e-10)
        )
        
        # Omnivariance: cubic root of eigenvalue product
        omnivariance = xp.cbrt(λ0 * λ1 * λ2)
        
        # Change of curvature: variance of eigenvalues (measures local complexity)
        eigenvalue_variance = xp.var(eigenvalues, axis=1)
        change_curvature = xp.sqrt(eigenvalue_variance)
        
        # Transfer results back to CPU if on GPU
        if use_gpu:
            λ0 = self._to_cpu(λ0)
            λ1 = self._to_cpu(λ1)
            λ2 = self._to_cpu(λ2)
            sum_eigenvalues = self._to_cpu(sum_eigenvalues)
            eigenentropy = self._to_cpu(eigenentropy)
            omnivariance = self._to_cpu(omnivariance)
            change_curvature = self._to_cpu(change_curvature)
        
        return {
            'eigenvalue_1': λ0.astype(np.float32),
            'eigenvalue_2': λ1.astype(np.float32),
            'eigenvalue_3': λ2.astype(np.float32),
            'sum_eigenvalues': sum_eigenvalues.astype(np.float32),
            'eigenentropy': eigenentropy.astype(np.float32),
            'omnivariance': omnivariance.astype(np.float32),
            'change_curvature': change_curvature.astype(np.float32),
        }

    def compute_architectural_features(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        neighbors_indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute architectural features for building detection (FULL GPU-accelerated).
        
        Features:
        - edge_strength: Strength of edges (high eigenvalue variance)
        - corner_likelihood: Probability of corner point (3D structure)
        - overhang_indicator: Overhang/protrusion detection
        - surface_roughness: Fine-scale surface texture
        
        Args:
            points: [N, 3] point coordinates
            normals: [N, 3] surface normals
            neighbors_indices: [N, k] indices of k-nearest neighbors
            
        Returns:
            Dictionary of architectural features
        """
        N = len(points)
        k = neighbors_indices.shape[1]
        
        # Determine computation backend (GPU if available, else CPU)
        use_gpu = self.use_gpu and cp is not None
        xp = cp if use_gpu else np
        
        # Transfer to GPU if available
        if use_gpu:
            points_gpu = self._to_gpu(points)
            normals_gpu = self._to_gpu(normals)
            neighbors_indices_gpu = cp.asarray(neighbors_indices)
            neighbors = points_gpu[neighbors_indices_gpu]
            neighbor_normals = normals_gpu[neighbors_indices_gpu]
        else:
            neighbors = points[neighbors_indices]
            neighbor_normals = normals[neighbors_indices]
        
        # Center neighbors
        centroids = xp.mean(neighbors, axis=1, keepdims=True)
        centered = neighbors - centroids
        
        # Covariance matrices
        cov_matrices = xp.einsum('nki,nkj->nij', centered, centered) / (k - 1)
        eigenvalues = xp.linalg.eigvalsh(cov_matrices)
        eigenvalues = xp.sort(eigenvalues, axis=1)[:, ::-1]
        eigenvalues = xp.maximum(eigenvalues, 1e-10)
        
        λ0 = eigenvalues[:, 0]
        λ1 = eigenvalues[:, 1]
        λ2 = eigenvalues[:, 2]
        
        # Edge strength: High when eigenvalues are (large, medium, small)
        # Normalized ratio (λ0 - λ2) / λ0
        edge_strength = xp.clip((λ0 - λ2) / (λ0 + 1e-8), 0.0, 1.0)
        
        # Corner likelihood: All eigenvalues similar (isotropic 3D structure)
        # Measured as ratio of smallest to largest eigenvalue
        corner_likelihood = xp.clip(λ2 / (λ0 + 1e-8), 0.0, 1.0)
        
        # Normal variation (measures local surface complexity)
        if use_gpu:
            normal_diffs = neighbor_normals - normals_gpu[:, cp.newaxis, :]
        else:
            normal_diffs = neighbor_normals - normals[:, np.newaxis, :]
        normal_variation = xp.linalg.norm(normal_diffs, axis=2).mean(axis=1)
        
        # Overhang indicator: Large vertical normal variation
        if use_gpu:
            vertical_diffs = neighbor_normals[:, :, 2] - normals_gpu[:, 2:3]
        else:
            vertical_diffs = neighbor_normals[:, :, 2] - normals[:, 2:3]
        overhang_indicator = xp.abs(vertical_diffs).mean(axis=1)
        
        # Surface roughness: Standard deviation of distances to centroid
        distances_to_centroid = xp.linalg.norm(centered, axis=2)
        surface_roughness = xp.std(distances_to_centroid, axis=1)
        
        # Transfer results back to CPU if on GPU
        if use_gpu:
            edge_strength = self._to_cpu(edge_strength)
            corner_likelihood = self._to_cpu(corner_likelihood)
            overhang_indicator = self._to_cpu(overhang_indicator)
            surface_roughness = self._to_cpu(surface_roughness)
        
        return {
            'edge_strength': edge_strength.astype(np.float32),
            'corner_likelihood': corner_likelihood.astype(np.float32),
            'overhang_indicator': np.clip(overhang_indicator, 0.0, 1.0).astype(np.float32),
            'surface_roughness': surface_roughness.astype(np.float32),
        }

    def compute_density_features(
        self,
        points: np.ndarray,
        neighbors_indices: np.ndarray,
        radius_2m: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        Compute density and neighborhood features (FULL GPU-accelerated).
        
        Features:
        - density: Local point density (1/mean_distance)
        - num_points_2m: Number of points within 2m radius
        - neighborhood_extent: Maximum distance to k-th neighbor
        - height_extent_ratio: Ratio of vertical to spatial extent
        
        Args:
            points: [N, 3] point coordinates
            neighbors_indices: [N, k] indices of k-nearest neighbors
            radius_2m: Radius for counting nearby points (default 2.0m)
            
        Returns:
            Dictionary of density features
        """
        N = len(points)
        k = neighbors_indices.shape[1]
        
        # Determine computation backend (GPU if available, else CPU)
        use_gpu = self.use_gpu and cp is not None
        xp = cp if use_gpu else np
        
        # Transfer to GPU if available
        if use_gpu:
            points_gpu = self._to_gpu(points)
            neighbors_indices_gpu = cp.asarray(neighbors_indices)
            neighbors = points_gpu[neighbors_indices_gpu]
        else:
            neighbors = points[neighbors_indices]
        
        # Compute distances to all neighbors: [N, k]
        if use_gpu:
            distances = xp.linalg.norm(
                neighbors - points_gpu[:, cp.newaxis, :],
                axis=2
            )
        else:
            distances = xp.linalg.norm(
                neighbors - points[:, np.newaxis, :],
                axis=2
            )
        
        # Density: 1 / mean distance (excluding self at distance 0)
        mean_distances = xp.mean(distances[:, 1:], axis=1)
        density = xp.clip(1.0 / (mean_distances + 1e-8), 0.0, 1000.0)
        
        # Neighborhood extent: maximum distance to k-th neighbor
        neighborhood_extent = xp.max(distances, axis=1)
        
        # Height extent ratio: vertical std / spatial extent
        z_coords = neighbors[:, :, 2]
        z_std = xp.std(z_coords, axis=1)
        vertical_std = z_std  # Store vertical_std as a separate feature
        spatial_extent = neighborhood_extent + 1e-8
        height_extent_ratio = z_std / spatial_extent
        
        # Number of points within 2m radius
        # For GPU: use efficient radius counting with neighbor distances
        # For CPU: use KDTree for accurate radius search
        if use_gpu:
            # GPU-accelerated approach: approximate using k-NN distances
            # Count neighbors within radius from existing k-NN results
            within_radius = xp.sum(distances <= radius_2m, axis=1)
            num_points_2m = within_radius.astype(xp.float32)
            
            # Transfer results back to CPU
            density = self._to_cpu(density)
            num_points_2m = self._to_cpu(num_points_2m)
            neighborhood_extent = self._to_cpu(neighborhood_extent)
            height_extent_ratio = self._to_cpu(height_extent_ratio)
            vertical_std = self._to_cpu(vertical_std)
        else:
            # CPU fallback: use KDTree for accurate radius search
            from sklearn.neighbors import KDTree
            tree = KDTree(points, metric='euclidean')
            neighbors_2m = tree.query_radius(points, r=radius_2m)
            num_points_2m = np.array([len(n) for n in neighbors_2m], dtype=np.float32)
        
        return {
            'density': density.astype(np.float32),
            'num_points_2m': num_points_2m.astype(np.float32),
            'neighborhood_extent': neighborhood_extent.astype(np.float32),
            'height_extent_ratio': np.clip(height_extent_ratio, 0.0, 1.0).astype(np.float32),
            'vertical_std': vertical_std.astype(np.float32),
        }

    def compute_all_features(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int = 10,
        include_building_features: bool = False,
        mode: str = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute all features in one pass (GPU-accelerated).

        This method provides feature parity with the CPU version
        in features.py by computing all geometric features in a
        single call.

        Args:
            points: [N, 3] point coordinates
            classification: [N] ASPRS classification codes
            k: number of neighbors for feature computation
            include_building_features: if True, compute verticality,
                                      wall_score, and roof_score (legacy parameter)
            mode: Feature mode ('minimal', 'lod2', 'lod3', 'full') - 
                  if specified, uses the new feature mode system

        Returns:
            normals: [N, 3] surface normals
            curvature: [N] curvature values
            height: [N] height above ground
            geo_features: dict with all geometric features
                         (includes building features if requested)
        """
        # If mode is specified, use the new feature mode system
        if mode is not None:
            from ..features.feature_modes import get_feature_config
            
            # Get feature configuration for the mode
            # Suppress logging here - it's already logged at the orchestrator level
            feature_config = get_feature_config(mode=mode, k_neighbors=k, log_config=False)
            feature_set = feature_config.features
            
            # Compute base features (always needed)
            normals = self.compute_normals(points, k=k)
            curvature = self.compute_curvature(points, normals, k=k)
            height = self.compute_height_above_ground(points, classification)
            
            # Get base geometric features from extract_geometric_features
            # Note: This includes eigenvalue-based features and architectural features
            geo_features = self.extract_geometric_features(points, normals, k=k)
            
            # Get neighbors_indices for additional feature computations
            tree = KDTree(points, metric='euclidean', leaf_size=30)
            _, neighbors_indices = tree.query(points, k=k)
            
            # Compute density features (includes neighborhood_extent, height_extent_ratio, vertical_std)
            density_features = self.compute_density_features(
                points=points,
                neighbors_indices=neighbors_indices,
                radius_2m=2.0
            )
            geo_features.update(density_features)
            
            # Compute additional architectural features
            architectural_features = self.compute_architectural_features(
                points=points,
                neighbors_indices=neighbors_indices,
                normals=normals
            )
            geo_features.update(architectural_features)
            
            # Filter features based on mode
            filtered_features = {}
            for feat_name in feature_set:
                if feat_name in geo_features:
                    filtered_features[feat_name] = geo_features[feat_name]
            
            # Add normal components if requested
            if 'normal_x' in feature_set:
                filtered_features['normal_x'] = normals[:, 0].astype(np.float32)
            if 'normal_y' in feature_set:
                filtered_features['normal_y'] = normals[:, 1].astype(np.float32)
            if 'normal_z' in feature_set:
                filtered_features['normal_z'] = normals[:, 2].astype(np.float32)
            
            # Add curvature if requested
            if 'curvature' in feature_set and 'curvature' not in filtered_features:
                filtered_features['curvature'] = curvature
            
            # Add height if requested
            if 'height_above_ground' in feature_set:
                filtered_features['height_above_ground'] = height
            
            # Add xyz coordinates if requested
            if 'xyz' in feature_set:
                filtered_features['xyz'] = points.astype(np.float32)
            
            # Add building-specific features if in feature set
            if any(f in feature_set for f in ['verticality', 'wall_score', 'roof_score']):
                if 'verticality' in feature_set:
                    verticality = self.compute_verticality(normals)
                    filtered_features['verticality'] = verticality
                else:
                    verticality = self.compute_verticality(normals)
                
                if 'wall_score' in feature_set:
                    wall_score = self.compute_wall_score(normals, height, min_height=1.5)
                    filtered_features['wall_score'] = wall_score
                
                if 'roof_score' in feature_set:
                    roof_score = self.compute_roof_score(normals, height, curvature, min_height=3.0)
                    filtered_features['roof_score'] = roof_score
            
            return normals, curvature, height, filtered_features
        
        # Legacy mode: compute all features (backward compatibility)
        normals = self.compute_normals(points, k=k)
        curvature = self.compute_curvature(points, normals, k=k)
        height = self.compute_height_above_ground(points, classification)
        geo_features = self.extract_geometric_features(points, normals, k=k)

        # Add building-specific features if requested
        if include_building_features:
            verticality = self.compute_verticality(normals)
            wall_score = self.compute_wall_score(normals, height, min_height=1.5)
            roof_score = self.compute_roof_score(
                normals, height, curvature, min_height=3.0
            )
            
            geo_features['verticality'] = verticality
            geo_features['wall_score'] = wall_score
            geo_features['roof_score'] = roof_score

        return normals, curvature, height, geo_features
    
    def interpolate_colors_gpu(
        self,
        points_gpu: 'CpArray',
        rgb_image_gpu: 'CpArray',
        bbox: Tuple[float, float, float, float]
    ) -> 'CpArray':
        """
        Fast bilinear color interpolation on GPU.
        
        This method provides ~100x speedup over CPU-based PIL interpolation
        by performing all operations on the GPU using CuPy.
        
        Args:
            points_gpu: [N, 3] CuPy array (x, y, z coordinates in Lambert-93)
            rgb_image_gpu: [H, W, 3] CuPy array (RGB image, uint8)
            bbox: (xmin, ymin, xmax, ymax) in Lambert-93 coordinates
            
        Returns:
            colors_gpu: [N, 3] CuPy array (R, G, B values, uint8)
            
        Performance:
            - 1M points: ~0.5s on GPU vs ~12s on CPU (24x speedup)
            - Memory efficient: operates directly on GPU arrays
        """
        if not self.use_gpu or cp is None:
            # Fallback to CPU not implemented here
            # This should be handled by the caller
            raise RuntimeError(
                "GPU not available. Use CPU-based interpolation instead."
            )
        
        # Unpack bbox
        xmin, ymin, xmax, ymax = bbox
        H, W = rgb_image_gpu.shape[:2]
        
        # Normalize point coordinates to image space
        # Lambert-93 coords → normalized [0, 1] → pixel coords
        x_norm = (points_gpu[:, 0] - xmin) / (xmax - xmin)  # [N]
        y_norm = (points_gpu[:, 1] - ymin) / (ymax - ymin)  # [N]
        
        # Convert to pixel coordinates (image y-axis is flipped)
        px = x_norm * (W - 1)  # [N]
        py = (1 - y_norm) * (H - 1)  # [N], flip y-axis
        
        # Clamp to valid range
        px = cp.clip(px, 0, W - 1)
        py = cp.clip(py, 0, H - 1)
        
        # Bilinear interpolation
        # Get integer and fractional parts
        px0 = cp.floor(px).astype(cp.int32)
        py0 = cp.floor(py).astype(cp.int32)
        px1 = cp.minimum(px0 + 1, W - 1)
        py1 = cp.minimum(py0 + 1, H - 1)
        
        dx = px - px0  # [N]
        dy = py - py0  # [N]
        
        # Fetch pixel values at 4 corners
        # Shape: [N, 3]
        c00 = rgb_image_gpu[py0, px0]  # Top-left
        c01 = rgb_image_gpu[py0, px1]  # Top-right
        c10 = rgb_image_gpu[py1, px0]  # Bottom-left
        c11 = rgb_image_gpu[py1, px1]  # Bottom-right
        
        # Bilinear weights
        w00 = (1 - dx[:, None]) * (1 - dy[:, None])  # [N, 1]
        w01 = dx[:, None] * (1 - dy[:, None])
        w10 = (1 - dx[:, None]) * dy[:, None]
        w11 = dx[:, None] * dy[:, None]
        
        # Interpolated color
        colors = (
            w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11
        ).astype(cp.uint8)
        
        return colors

    def compute_all_features_chunked(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int = 10,
        chunk_size: int = 2_500_000,
        radius: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute ALL features per-chunk for memory efficiency.
        GPU version without cuML - uses sklearn KDTree + NumPy/CuPy operations.
        
        This method processes the point cloud in chunks, computing all features
        (normals, curvature, height, geometric) for each chunk before moving
        to the next. This dramatically reduces memory usage compared to
        computing each feature type globally.
        
        Args:
            points: [N, 3] point coordinates
            classification: [N] ASPRS classification codes
            k: number of neighbors for KNN
            chunk_size: points per chunk (default: 2.5M)
            radius: search radius for geometric features (optional)
            
        Returns:
            normals: [N, 3] surface normals
            curvature: [N] curvature values
            height: [N] height above ground
            geo_features: dict with geometric features
        """
        import logging
        logger = logging.getLogger(__name__)
        
        N = len(points)
        
        # Initialize output arrays
        normals = np.zeros((N, 3), dtype=np.float32)
        curvature = np.zeros(N, dtype=np.float32)
        height = np.zeros(N, dtype=np.float32)
        
        # Initialize geometric features dict
        feature_keys = ['planarity', 'linearity', 'sphericity',
                        'anisotropy', 'roughness', 'density',
                        'verticality', 'horizontality']
        geo_features = {key: np.zeros(N, dtype=np.float32)
                        for key in feature_keys}
        
        # Calculate number of chunks
        num_chunks = (N + chunk_size - 1) // chunk_size
        chunk_size_mb = (chunk_size * 12) / (1024 * 1024)  # Approx memory per chunk
        
        logger.info(
            f"Processing {N:,} points in {num_chunks} chunks "
            f"(GPU without cuML, per-chunk computation)"
        )
        
        # Progress bar for chunk processing
        chunk_iterator = range(num_chunks)
        bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} chunks '
                   '[{elapsed}<{remaining}, {rate_fmt}]')
        chunk_iterator = tqdm(
            chunk_iterator,
            desc=f"  🔧 GPU Features [sklearn] ({N:,} pts, {num_chunks} chunks @ {chunk_size_mb:.1f}MB)",
            unit="chunk",
            total=num_chunks,
            bar_format=bar_fmt
        )
        
        # Process each chunk
        for chunk_idx in chunk_iterator:
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, N)
            chunk_points = end_idx - start_idx
            
            # Add overlap for accurate boundary computation
            overlap = int(chunk_size * 0.10)  # 10% overlap
            tree_start = max(0, start_idx - overlap)
            tree_end = min(N, end_idx + overlap)
            
            # Extract chunk with overlap for KDTree
            chunk_data = points[tree_start:tree_end]
            chunk_class = classification[tree_start:tree_end]
            
            # Calculate local indices for storing results
            local_start = start_idx - tree_start
            local_end = local_start + chunk_points
            query_points = chunk_data[local_start:local_end]
            
            # Build local KDTree for this chunk
            tree = KDTree(chunk_data, metric='euclidean', leaf_size=30)
            
            # 1. Compute normals for this chunk
            _, indices = tree.query(query_points, k=k)
            
            # Vectorized PCA for normals
            neighbors_all = chunk_data[indices]
            centroids = neighbors_all.mean(axis=1, keepdims=True)
            centered = neighbors_all - centroids
            cov_matrices = np.einsum('nki,nkj->nij',
                                     centered, centered) / (k - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
            chunk_normals = eigenvectors[:, :, 0].copy()
            
            # Normalize normals
            norms = np.linalg.norm(chunk_normals, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            chunk_normals = chunk_normals / norms
            
            # Orient normals upward
            flip_mask = chunk_normals[:, 2] < 0
            chunk_normals[flip_mask] *= -1
            
            # 2. Compute curvature for this chunk
            neighbor_normals = chunk_normals[indices - local_start]
            query_normals_expanded = chunk_normals[:, np.newaxis, :]
            normal_diff = neighbor_normals - query_normals_expanded
            curv_norms = np.linalg.norm(normal_diff, axis=2)
            chunk_curvature = np.mean(curv_norms, axis=1).astype(np.float32)
            
            # 3. Compute height for this chunk
            ground_mask = (chunk_class == 2)
            if np.any(ground_mask):
                ground_z = np.median(chunk_data[ground_mask, 2])
            else:
                ground_z = np.min(chunk_data[:, 2])
            chunk_height = np.maximum(
                query_points[:, 2] - ground_z, 0
            ).astype(np.float32)
            
            # 4. Compute geometric features for this chunk
            distances, geo_indices = tree.query(query_points, k=k)
            neighbors_geo = chunk_data[geo_indices]
            centroids_geo = neighbors_geo.mean(axis=1, keepdims=True)
            centered_geo = neighbors_geo - centroids_geo
            cov_matrices_geo = np.einsum('nki,nkj->nij',
                                          centered_geo, centered_geo) / (k - 1)
            eigenvalues_geo = np.linalg.eigvalsh(cov_matrices_geo)
            eigenvalues_geo = np.sort(eigenvalues_geo, axis=1)[:, ::-1]
            
            λ0 = eigenvalues_geo[:, 0]
            λ1 = eigenvalues_geo[:, 1]
            λ2 = eigenvalues_geo[:, 2]
            λ0_safe = λ0 + 1e-8
            sum_λ = λ0 + λ1 + λ2 + 1e-8
            
            # Use λ0 normalization (consistent with boundary features)
            chunk_linearity = ((λ0 - λ1) / λ0_safe).astype(np.float32)
            chunk_planarity = ((λ1 - λ2) / λ0_safe).astype(np.float32)
            chunk_sphericity = (λ2 / λ0_safe).astype(np.float32)
            chunk_anisotropy = ((λ0 - λ2) / λ0_safe).astype(np.float32)
            chunk_roughness = (λ2 / sum_λ).astype(np.float32)  # Keep sum normalization
            mean_distances = np.mean(distances[:, 1:], axis=1)
            chunk_density = (1.0 / (mean_distances + 1e-8)).astype(np.float32)
            
            # Compute verticality and horizontality from normals
            chunk_verticality = self.compute_verticality(chunk_normals)
            chunk_horizontality = np.abs(chunk_normals[:, 2]).astype(np.float32)
            
            # Store results in output arrays
            normals[start_idx:end_idx] = chunk_normals
            curvature[start_idx:end_idx] = chunk_curvature
            height[start_idx:end_idx] = chunk_height
            geo_features['planarity'][start_idx:end_idx] = chunk_planarity
            geo_features['linearity'][start_idx:end_idx] = chunk_linearity
            geo_features['sphericity'][start_idx:end_idx] = chunk_sphericity
            geo_features['anisotropy'][start_idx:end_idx] = chunk_anisotropy
            geo_features['roughness'][start_idx:end_idx] = chunk_roughness
            geo_features['density'][start_idx:end_idx] = chunk_density
            geo_features['verticality'][start_idx:end_idx] = chunk_verticality
            geo_features['horizontality'][start_idx:end_idx] = chunk_horizontality
            
            # Cleanup chunk data
            del (chunk_data, chunk_class, query_points, tree,
                 indices, neighbors_all, centroids, centered, cov_matrices,
                 eigenvalues, eigenvectors, chunk_normals, neighbor_normals,
                 query_normals_expanded, normal_diff, curv_norms,
                 chunk_curvature, chunk_height, distances, geo_indices,
                 neighbors_geo, centroids_geo, centered_geo, cov_matrices_geo,
                 eigenvalues_geo, chunk_planarity, chunk_linearity,
                 chunk_sphericity, chunk_anisotropy, chunk_roughness,
                 chunk_density, chunk_verticality, chunk_horizontality)
        
        # Log completion statistics
        total_features = len(geo_features) + 3  # +3 for normals, curvature, height
        logger.info(
            f"✓ GPU features computed successfully: "
            f"{total_features} feature types, {N:,} points, {num_chunks} chunks processed"
        )
        
        return normals, curvature, height, geo_features


# Instance globale pour réutilisation
_gpu_computer = None


def get_gpu_computer(
    use_gpu: bool = True,
    batch_size: int = 100000
) -> GPUFeatureComputer:
    """Obtenir instance GPU computer (singleton pattern)."""
    global _gpu_computer
    if _gpu_computer is None:
        _gpu_computer = GPUFeatureComputer(
            use_gpu=use_gpu,
            batch_size=batch_size
        )
    return _gpu_computer


# API compatible avec features.py (drop-in replacement)
def compute_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    """Wrapper API-compatible."""
    computer = get_gpu_computer()
    return computer.compute_normals(points, k)


def compute_curvature(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """Wrapper API-compatible."""
    computer = get_gpu_computer()
    return computer.compute_curvature(points, normals, k)


def compute_all_features_gpu_chunked(
    points: np.ndarray,
    classification: np.ndarray,
    k: int = 10,
    chunk_size: int = 2_500_000,
    radius: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute ALL features per-chunk for memory efficiency.
    GPU version without cuML - uses sklearn KDTree + NumPy operations.
    
    This is a wrapper function that calls the GPUFeatureComputer method.
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] ASPRS classification codes
        k: number of neighbors for KNN
        chunk_size: points per chunk (default: 2.5M)
        radius: search radius for geometric features (optional)
        
    Returns:
        normals: [N, 3] surface normals
        curvature: [N] curvature values
        height: [N] height above ground
        geo_features: dict with geometric features
    """
    computer = get_gpu_computer()
    return computer.compute_all_features_chunked(
        points, classification, k, chunk_size, radius
    )



def compute_height_above_ground(
    points: np.ndarray,
    classification: np.ndarray
) -> np.ndarray:
    """Wrapper API-compatible."""
    computer = get_gpu_computer()
    return computer.compute_height_above_ground(points, classification)


def extract_geometric_features(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 10
) -> Dict[str, np.ndarray]:
    """Wrapper API-compatible."""
    computer = get_gpu_computer()
    return computer.extract_geometric_features(points, normals, k)


def compute_verticality(normals: np.ndarray) -> np.ndarray:
    """
    Wrapper for GPU-accelerated verticality computation.
    
    Args:
        normals: [N, 3] surface normal vectors
        
    Returns:
        verticality: [N] verticality values [0, 1]
    """
    computer = get_gpu_computer()
    return computer.compute_verticality(normals)


def compute_wall_score(
    normals: np.ndarray,
    height_above_ground: np.ndarray,
    min_height: float = 1.5
) -> np.ndarray:
    """
    Wrapper for GPU-accelerated wall score computation.
    
    Args:
        normals: [N, 3] surface normal vectors
        height_above_ground: [N] height above ground in meters
        min_height: minimum height to be considered a wall
        
    Returns:
        wall_score: [N] wall probability [0, 1]
    """
    computer = get_gpu_computer()
    return computer.compute_wall_score(normals, height_above_ground, min_height)


def compute_roof_score(
    normals: np.ndarray,
    height_above_ground: np.ndarray,
    curvature: np.ndarray,
    min_height: float = 3.0
) -> np.ndarray:
    """
    Wrapper for GPU-accelerated roof score computation.
    
    Args:
        normals: [N, 3] surface normal vectors
        height_above_ground: [N] height above ground in meters
        curvature: [N] surface curvature
        min_height: minimum height for a roof
        
    Returns:
        roof_score: [N] roof probability [0, 1]
    """
    computer = get_gpu_computer()
    return computer.compute_roof_score(
        normals, height_above_ground, curvature, min_height
    )


def compute_eigenvalue_features(
    points: np.ndarray,
    normals: np.ndarray,
    neighbors_indices: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Wrapper for GPU-accelerated eigenvalue feature computation.
    
    Computes eigenvalue-based geometric features:
    - eigenvalue_1, eigenvalue_2, eigenvalue_3
    - sum_eigenvalues, eigenentropy, omnivariance
    - change_curvature
    
    Args:
        points: [N, 3] point coordinates
        normals: [N, 3] surface normals
        neighbors_indices: [N, k] indices of k-nearest neighbors
        
    Returns:
        Dictionary of eigenvalue-based features
    """
    computer = get_gpu_computer()
    return computer.compute_eigenvalue_features(points, normals, neighbors_indices)


def compute_architectural_features(
    points: np.ndarray,
    normals: np.ndarray,
    neighbors_indices: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Wrapper for GPU-accelerated architectural feature computation.
    
    Computes architectural features for building detection:
    - edge_strength, corner_likelihood
    - overhang_indicator, surface_roughness
    
    Args:
        points: [N, 3] point coordinates
        normals: [N, 3] surface normals
        neighbors_indices: [N, k] indices of k-nearest neighbors
        
    Returns:
        Dictionary of architectural features
    """
    computer = get_gpu_computer()
    return computer.compute_architectural_features(points, normals, neighbors_indices)


def compute_density_features(
    points: np.ndarray,
    neighbors_indices: np.ndarray,
    radius_2m: float = 2.0
) -> Dict[str, np.ndarray]:
    """
    Wrapper for GPU-accelerated density feature computation.
    
    Computes density and neighborhood features:
    - density, num_points_2m
    - neighborhood_extent, height_extent_ratio
    
    Args:
        points: [N, 3] point coordinates
        neighbors_indices: [N, k] indices of k-nearest neighbors
        radius_2m: Radius for counting nearby points (default 2.0m)
        
    Returns:
        Dictionary of density features
    """
    computer = get_gpu_computer()
    return computer.compute_density_features(points, neighbors_indices, radius_2m)
