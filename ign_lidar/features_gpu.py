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
from sklearn.decomposition import PCA


class GPUFeatureComputer:
    """
    Classe pour calcul de features géométriques optimisé GPU.
    Fallback automatique CPU si GPU indisponible.
    """
    
    def __init__(self, use_gpu: bool = True, batch_size: int = 100000):
        """
        Args:
            use_gpu: Activer GPU si disponible
            batch_size: Nombre de points à traiter par batch GPU
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
        """PCA par batch sur GPU."""
        if not GPU_AVAILABLE or cp is None:
            raise RuntimeError("GPU not available")
            
        batch_size = len(neighbor_indices)
        normals = cp.zeros((batch_size, 3), dtype=cp.float32)
        
        for i in range(batch_size):
            indices = neighbor_indices[i]
            neighbors = points_gpu[indices]
            
            # Vérifier variance
            variance = cp.var(neighbors, axis=0)
            if cp.sum(variance) < 1e-6:
                normals[i] = cp.array([0, 0, 1], dtype=cp.float32)
                continue
            
            try:
                # PCA GPU
                pca = cuPCA(n_components=3)
                pca.fit(neighbors)
                
                # Normal = dernier composant
                normal = pca.components_[-1]
                
                # Normaliser
                norm = cp.linalg.norm(normal)
                if norm < 1e-6:
                    normal = cp.array([0, 0, 1], dtype=cp.float32)
                else:
                    normal = normal / norm
                
                # Orientation vers le haut
                if normal[2] < 0:
                    normal = -normal
                
                normals[i] = normal
                
            except Exception:
                normals[i] = cp.array([0, 0, 1], dtype=cp.float32)
        
        return normals
    
    def _compute_normals_cpu(self, points: np.ndarray, k: int) -> np.ndarray:
        """Calcul normales sur CPU (sklearn) - version optimisée."""
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        
        # Build KDTree
        tree = KDTree(points, metric='euclidean')
        
        # Batch query pour réduire overhead
        batch_size = 10000
        num_batches = (N + batch_size - 1) // batch_size
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, N)
                batch_points = points[start_idx:end_idx]
                
                # Query KNN pour tout le batch
                _, indices = tree.query(batch_points, k=k)
                
                # PCA pour chaque point du batch
                for i, point_indices in enumerate(indices):
                    neighbors = points[point_indices]
                    
                    # Vérifier variance
                    variance = np.var(neighbors, axis=0)
                    if np.sum(variance) < 1e-6:
                        normals[start_idx + i] = np.array([0, 0, 1], dtype=np.float32)
                        continue
                    
                    try:
                        pca = PCA(n_components=3)
                        pca.fit(neighbors)
                        
                        if np.any(np.isnan(pca.components_)):
                            normals[start_idx + i] = np.array([0, 0, 1], dtype=np.float32)
                            continue
                        
                        normal = pca.components_[-1]
                        norm = np.linalg.norm(normal)
                        
                        if norm < 1e-6:
                            normal = np.array([0, 0, 1], dtype=np.float32)
                        else:
                            normal = normal / norm
                        
                        if normal[2] < 0:
                            normal = -normal
                        
                        normals[start_idx + i] = normal
                        
                    except Exception:
                        normals[start_idx + i] = np.array([0, 0, 1], dtype=np.float32)
        
        return normals
    
    def compute_curvature(self, points: np.ndarray, normals: np.ndarray, 
                         k: int = 10) -> np.ndarray:
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
        batch_size = 10000
        num_batches = (N + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_points = points[start_idx:end_idx]
            
            # Query KNN
            _, indices = tree.query(batch_points, k=k)
            
            # Vectorized curvature computation
            for i, point_indices in enumerate(indices):
                neighbors = points[point_indices]
                center = points[start_idx + i]
                normal = normals[start_idx + i]
                
                relative_pos = neighbors - center
                distances_along_normal = np.dot(relative_pos, normal)
                curvature[start_idx + i] = np.std(distances_along_normal)
        
        return curvature
    
    def compute_height_above_ground(self, points: np.ndarray, 
                                   classification: np.ndarray) -> np.ndarray:
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
    
    def extract_geometric_features(self, points: np.ndarray,
                                  normals: np.ndarray,
                                  k: int = 10) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive geometric features for each point.
        Version vectorisée optimale - aligned with features.py.
        
        Features computed (eigenvalue-based):
        - Planarity: (λ1-λ2)/λ0 - surfaces planes
        - Linearity: (λ0-λ1)/λ0 - structures linéaires
        - Sphericity: λ2/λ0 - structures sphériques
        - Anisotropy: (λ0-λ2)/λ0 - anisotropie
        - Roughness: λ2/(λ0+λ1+λ2) - rugosité
        - Density: local point density
        
        Args:
            points: [N, 3] point coordinates
            normals: [N, 3] normal vectors (not used, kept for compat)
            k: number of neighbors
            
        Returns:
            features: dictionary of geometric features
        """
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
        
        # Compute features
        planarity = ((λ1 - λ2) / λ0_safe).astype(np.float32)
        linearity = ((λ0 - λ1) / λ0_safe).astype(np.float32)
        sphericity = (λ2 / λ0_safe).astype(np.float32)
        anisotropy = ((λ0 - λ2) / λ0_safe).astype(np.float32)
        roughness = (λ2 / sum_λ).astype(np.float32)
        
        mean_distances = np.mean(distances[:, 1:], axis=1)
        density = (1.0 / (mean_distances + 1e-8)).astype(np.float32)
        
        features = {
            'planarity': planarity,
            'linearity': linearity,
            'sphericity': sphericity,
            'anisotropy': anisotropy,
            'roughness': roughness,
            'density': density
        }
        
        return features


# Instance globale pour réutilisation
_gpu_computer = None

def get_gpu_computer(use_gpu: bool = True, batch_size: int = 100000) -> GPUFeatureComputer:
    """Obtenir instance GPU computer (singleton pattern)."""
    global _gpu_computer
    if _gpu_computer is None:
        _gpu_computer = GPUFeatureComputer(use_gpu=use_gpu, batch_size=batch_size)
    return _gpu_computer


# API compatible avec features.py (drop-in replacement)
def compute_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    """Wrapper API-compatible."""
    computer = get_gpu_computer()
    return computer.compute_normals(points, k)


def compute_curvature(points: np.ndarray, normals: np.ndarray, 
                     k: int = 10) -> np.ndarray:
    """Wrapper API-compatible."""
    computer = get_gpu_computer()
    return computer.compute_curvature(points, normals, k)


def compute_height_above_ground(points: np.ndarray, 
                               classification: np.ndarray) -> np.ndarray:
    """Wrapper API-compatible."""
    computer = get_gpu_computer()
    return computer.compute_height_above_ground(points, classification)


def extract_geometric_features(points: np.ndarray, normals: np.ndarray,
                              k: int = 10) -> Dict[str, np.ndarray]:
    """Wrapper API-compatible."""
    computer = get_gpu_computer()
    return computer.extract_geometric_features(points, normals, k)
