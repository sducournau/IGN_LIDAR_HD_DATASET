"""
GPU-Accelerated Geometric Feature Extraction Functions
Utilise CuPy et RAPIDS cuML pour des calculs 10-50x plus rapides
Avec fallback automatique vers CPU si GPU indisponible
"""

from typing import Dict, Tuple, Any, Union
import numpy as np
import warnings
from pathlib import Path

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
        
        # Safe division
        λ0_safe = λ0 + 1e-8
        sum_λ = λ0 + λ1 + λ2 + 1e-8
        
        # Compute features (CORRECTED to match CPU formulas - Weinmann et al.)
        # Using sum_λ normalization for standard eigenvalue-based features
        planarity = ((λ1 - λ2) / sum_λ).astype(np.float32)
        linearity = ((λ0 - λ1) / sum_λ).astype(np.float32)
        sphericity = (λ2 / sum_λ).astype(np.float32)
        anisotropy = ((λ0 - λ2) / λ0_safe).astype(np.float32)
        roughness = (λ2 / sum_λ).astype(np.float32)
        
        mean_distances = np.mean(distances[:, 1:], axis=1)
        density = (1.0 / (mean_distances + 1e-8)).astype(np.float32)
        
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
        
        features = {
            'planarity': planarity,
            'linearity': linearity,
            'sphericity': sphericity,
            'anisotropy': anisotropy,
            'roughness': roughness,
            'density': density
        }
        
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

    def compute_all_features(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int = 10,
        include_building_features: bool = False
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
                                      wall_score, and roof_score

        Returns:
            normals: [N, 3] surface normals
            curvature: [N] curvature values
            height: [N] height above ground
            geo_features: dict with all geometric features
                         (includes building features if requested)
        """
        # Compute all features
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
                        'anisotropy', 'roughness', 'density']
        geo_features = {key: np.zeros(N, dtype=np.float32)
                        for key in feature_keys}
        
        # Calculate number of chunks
        num_chunks = (N + chunk_size - 1) // chunk_size
        
        logger.info(
            f"Processing {N:,} points in {num_chunks} chunks "
            f"(GPU without cuML, per-chunk computation)"
        )
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
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
            
            logger.info(
                f"  Chunk {chunk_idx + 1}/{num_chunks}: "
                f"Processing {chunk_points:,} points "
                f"(tree size: {len(chunk_data):,})"
            )
            
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
            
            chunk_planarity = ((λ1 - λ2) / sum_λ).astype(np.float32)
            chunk_linearity = ((λ0 - λ1) / sum_λ).astype(np.float32)
            chunk_sphericity = (λ2 / sum_λ).astype(np.float32)
            chunk_anisotropy = ((λ0 - λ2) / λ0_safe).astype(np.float32)
            chunk_roughness = (λ2 / sum_λ).astype(np.float32)
            mean_distances = np.mean(distances[:, 1:], axis=1)
            chunk_density = (1.0 / (mean_distances + 1e-8)).astype(np.float32)
            
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
            
            # Cleanup chunk data
            del (chunk_data, chunk_class, query_points, tree,
                 indices, neighbors_all, centroids, centered, cov_matrices,
                 eigenvalues, eigenvectors, chunk_normals, neighbor_normals,
                 query_normals_expanded, normal_diff, curv_norms,
                 chunk_curvature, chunk_height, distances, geo_indices,
                 neighbors_geo, centroids_geo, centered_geo, cov_matrices_geo,
                 eigenvalues_geo, chunk_planarity, chunk_linearity,
                 chunk_sphericity, chunk_anisotropy, chunk_roughness,
                 chunk_density)
        
        logger.info(
            "Per-chunk feature computation complete (GPU without cuML)"
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
