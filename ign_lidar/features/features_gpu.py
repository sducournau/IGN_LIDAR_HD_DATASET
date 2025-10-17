"""
GPU-Accelerated Geometric Feature Extraction Functions
Uses CuPy and RAPIDS cuML for 10-50x faster computations
With automatic CPU fallback if GPU unavailable

IMPORTANT: CUSOLVER Batch Size Limits
- CuPy's batched eigenvalue decomposition (cp.linalg.eigh) has internal limits
- Maximum safe batch size: ~500,000 3x3 matrices per call
- For larger batches, automatic sub-batching is applied
- This prevents CUSOLVER_STATUS_INVALID_VALUE errors with large point clouds
"""

from typing import Optional, Tuple, Dict, List, Union, Any
from types import ModuleType
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm

# Try GPU imports
GPU_AVAILABLE = False
CUML_AVAILABLE = False

try:
    import cupy as cp
    from cupyx.scipy.spatial import distance as cp_distance
    GPU_AVAILABLE = True
    CpArray = cp.ndarray
    print("✓ CuPy available - GPU enabled")
except ImportError:
    print("⚠ CuPy not available - CPU fallback")
    cp = None
    CpArray = Any  # Type placeholder when CuPy unavailable

try:
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    from cuml.decomposition import PCA as cuPCA
    CUML_AVAILABLE = True
    print("✓ RAPIDS cuML available - GPU algorithms enabled")
except ImportError:
    print("⚠ RAPIDS cuML not available - sklearn fallback")
    cuNearestNeighbors = None
    cuPCA = None

# CPU fallback
from sklearn.neighbors import KDTree

# Import core feature implementations
from ..features.core import (
    compute_normals as core_compute_normals,
    compute_curvature as core_compute_curvature,
    compute_eigenvalue_features as core_compute_eigenvalue_features,
    compute_density_features as core_compute_density_features,
    compute_verticality as core_compute_verticality,
    extract_geometric_features as core_extract_geometric_features,
)


class GPUFeatureComputer:
    """
    Class for GPU-optimized geometric feature computation.
    Automatic CPU fallback if GPU unavailable.
    """
    
    def __init__(self, use_gpu: bool = True, batch_size: int = 8_000_000):
        """
        Args:
            use_gpu: Enable GPU if available
            batch_size: Points per GPU batch (default: 1M, optimized for reclassification)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_cuml = use_gpu and CUML_AVAILABLE
        
        # Adaptive batch size optimization for reclassification
        if self.use_gpu:
            # Auto-optimize batch size based on available VRAM
            try:
                if cp is not None:
                    _, total_vram = cp.cuda.runtime.memGetInfo()
                    total_vram_gb = total_vram / (1024**3)
                    
                    # More aggressive batch sizes for RTX 4080 Super optimization
                    if total_vram_gb >= 15.0:  # RTX 4080 Super has ~16GB but reports 15.992
                        # RTX 4080 Super and similar - use configured batch size up to 8M
                        self.batch_size = min(batch_size, 8_000_000)  # High-end GPUs
                    elif total_vram_gb >= 12.0:
                        self.batch_size = min(batch_size, 4_000_000)  # Mid-range GPUs  
                    elif total_vram_gb >= 8.0:
                        self.batch_size = min(batch_size, 2_000_000)  # Standard GPUs
                    else:
                        self.batch_size = min(batch_size, 1_000_000)    # Low VRAM
                else:
                    self.batch_size = batch_size
            except Exception:
                self.batch_size = batch_size
        else:
            self.batch_size = batch_size
        
        # Initialize CUDA context safely for multiprocessing
        if self.use_gpu:
            self._initialize_cuda_context()
        
        if self.use_gpu:
            print(f"🚀 GPU mode enabled (batch_size={self.batch_size:,})")
        else:
            print("💻 CPU mode (install CuPy for acceleration)")
    
    def _initialize_cuda_context(self):
        """Initialize CUDA context safely for multiprocessing."""
        if self.use_gpu and cp is not None:
            try:
                # Force CUDA context initialization
                _ = cp.cuda.Device()
                # Test basic operation
                test_array = cp.array([1.0], dtype=cp.float32)
                _ = cp.asnumpy(test_array)
                return True
            except Exception as e:
                print(f"⚠ CUDA initialization failed: {e}")
                print("💻 Falling back to CPU mode")
                self.use_gpu = False
                self.use_cuml = False
                return False
        return False

    def _to_gpu(self, array: np.ndarray):
        """Transfer array to GPU with safe context initialization."""
        if self.use_gpu and cp is not None:
            try:
                return cp.asarray(array, dtype=cp.float32)
            except Exception as e:
                print(f"⚠ GPU transfer failed: {e}")
                print("💻 Falling back to CPU for this operation")
                # Don't disable GPU completely, just this operation
                return array
        return array
    
    def _to_cpu(self, array) -> np.ndarray:
        """Transfer array to CPU."""
        if self.use_gpu and cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    
    def compute_all_features(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int = 10,
        include_building_features: bool = False,
        mode: Optional[str] = None,
        **kwargs
    ):
        """
        Compute all geometric features using GPU acceleration.
        
        Args:
            points: [N, 3] point coordinates
            classification: [N] ASPRS classification codes  
            k: number of neighbors for feature computation
            include_building_features: whether to include building-specific features
            mode: feature computation mode ('minimal', 'lod2', 'lod3', 'full')
            
        Returns:
            tuple: (normals, curvature, height, geo_features)
                - normals: [N, 3] surface normals
                - curvature: [N] curvature values
                - height: [N] height above ground
                - geo_features: dict with geometric features
        """
        # Handle mode-based feature computation
        if mode is not None:
            # Fallback to CPU implementation for mode-based features
            from .features import compute_features_by_mode
            return compute_features_by_mode(
                points=points,
                classification=classification,
                mode=mode,
                k=k,
                auto_k=False,
                patch_center=kwargs.get('patch_center'),
                use_radius=False,
                radius=0.8  # Default radius value
            )
        
        # Compute individual features
        normals = self.compute_normals(points, k=k)
        curvature = self.compute_curvature(points, normals, k=k)
        height = self.compute_height_above_ground(points, classification)
        
        # Compute geometric features
        geo_features = {}
        
        # Basic geometric features that we can compute
        verticality = self.compute_verticality(normals)
        geo_features['verticality'] = verticality
        
        # If building features are requested, add more features
        if include_building_features:
            # Add additional features here if needed
            pass
            
        return normals, curvature, height, geo_features
    
    def compute_normals(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Compute surface normals using PCA on k-nearest neighbors.
        GPU-accelerated version with batch processing.
        
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
        """Compute normals on GPU (RAPIDS cuML)."""
        if not GPU_AVAILABLE or cp is None:
            return self._compute_normals_cpu(points, k)
            
        try:
            # Ensure CUDA context is initialized for this process
            if not hasattr(self, '_cuda_initialized'):
                self._initialize_cuda_context()
                self._cuda_initialized = True
                
            N = len(points)
            normals = np.zeros((N, 3), dtype=np.float32)
            
            # Transfer to GPU
            points_gpu = self._to_gpu(points)
            
            # If GPU transfer failed, fallback to CPU
            if not isinstance(points_gpu, cp.ndarray):
                return self._compute_normals_cpu(points, k)
            
            # Build GPU KNN
            knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(points_gpu)
            
            # Process in batches to avoid OOM
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
            
        except Exception as e:
            print(f"⚠ GPU normal computation failed: {e}")
            print("💻 Falling back to CPU computation")
            return self._compute_normals_cpu(points, k)
    
    def _batch_pca_gpu(self, points_gpu, neighbor_indices):
        """
        VECTORIZED PCA on GPU with sub-batching for CUSOLVER limits.
        CUSOLVER has a maximum batch size limit (~500k matrices).
        """
        if not GPU_AVAILABLE or cp is None:
            raise RuntimeError("GPU not available")
            
        batch_size, k = neighbor_indices.shape
        
        # CUSOLVER batch limit - process in chunks if needed
        # Empirically, ~500k 3x3 matrices is safe for most GPUs
        max_cusolver_batch = 500_000
        
        if batch_size <= max_cusolver_batch:
            # Single batch - original fast path
            return self._batch_pca_gpu_core(points_gpu, neighbor_indices)
        
        # Multiple sub-batches needed
        import logging
        logger = logging.getLogger(__name__)
        num_sub_batches = (batch_size + max_cusolver_batch - 1) // max_cusolver_batch
        logger.info(f"    Splitting {batch_size:,} normals into {num_sub_batches} sub-batches (CUSOLVER limit)")
        
        normals = cp.zeros((batch_size, 3), dtype=cp.float32)
        
        for sub_batch_idx in range(num_sub_batches):
            start_idx = sub_batch_idx * max_cusolver_batch
            end_idx = min((sub_batch_idx + 1) * max_cusolver_batch, batch_size)
            
            sub_indices = neighbor_indices[start_idx:end_idx]
            sub_normals = self._batch_pca_gpu_core(points_gpu, sub_indices)
            normals[start_idx:end_idx] = sub_normals
        
        return normals
    
    def _batch_pca_gpu_core(self, points_gpu, neighbor_indices):
        """
        Core VECTORIZED PCA computation on GPU.
        Assumes batch size is within CUSOLVER limits.
        """
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
        Compute normals on CPU (sklearn) - VECTORIZED.
        ~100x faster than per-point PCA loop.
        """
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        
        # Build KDTree
        tree = KDTree(points, metric='euclidean')
        
        # Batch query to reduce overhead
        batch_size = 50000  # Larger batches for vectorized computation
        num_batches = (N + batch_size - 1) // batch_size
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, N)
                batch_points = points[start_idx:end_idx]
                
                # Query KNN for entire batch
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
        Optimized version with vectorization.
        
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
        Vectorized pure implementation.
        
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
            # GPU computation - optimized version
            normals_gpu = self._to_gpu(normals)
            verticality_gpu = 1.0 - cp.abs(normals_gpu[:, 2])
            return self._to_cpu(verticality_gpu).astype(np.float32)
        else:
            # Use core implementation for CPU fallback
            return core_compute_verticality(normals)

    def compute_reclassification_features_fast(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int = 10,
        mode: str = 'minimal'
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        OPTIMIZED: Fast feature computation specifically for reclassification workflows.
        
        This method computes only the essential features needed for most reclassification
        tasks, with aggressive optimizations for speed:
        - Reduced feature set (normals + planarity + density + height)
        - Optimized batch processing
        - Minimal memory allocations
        - Fast CPU fallback
        
        Args:
            points: [N, 3] point coordinates
            classification: [N] ASPRS classification codes
            k: number of neighbors
            mode: Feature mode ('minimal', 'standard')
            
        Returns:
            normals: [N, 3] surface normals
            features: dict with essential features for reclassification
        """
        N = len(points)
        
        # Determine essential features based on mode
        if mode == 'minimal':
            essential_features = ['planarity', 'height', 'verticality']
        else:  # 'standard'
            essential_features = ['planarity', 'linearity', 'height', 'verticality', 'density']
        
        print(f"🚀 RECLASSIFICATION MODE: Computing {mode} features ({N:,} points)")
        
        # Compute normals first (needed for other features)
        normals = self.compute_normals(points, k=k)
        
        # Initialize feature dictionary
        features = {}
        
        # Height above ground (critical for reclassification)
        features['height'] = self.compute_height_above_ground(points, classification)
        
        # Verticality from normals (fast computation)
        if 'verticality' in essential_features:
            features['verticality'] = self.compute_verticality(normals)
        
        # Geometric features if needed
        if any(feat in essential_features for feat in ['planarity', 'linearity', 'density']):
            # Use fast geometric computation with reduced feature set
            geo_features = self._compute_essential_geometric_features(
                points, normals, k=k, required_features=essential_features
            )
            features.update(geo_features)
        
        # Clean features from NaN/Inf
        for feat_name in features:
            if isinstance(features[feat_name], np.ndarray):
                features[feat_name] = np.nan_to_num(
                    features[feat_name], 
                    nan=0.0, 
                    posinf=1.0, 
                    neginf=0.0
                ).astype(np.float32)
        
        # Clean normals
        normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        print(f"  ✓ Reclassification features computed successfully")
        return normals, features

    def _compute_essential_geometric_features(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int = 10,
        required_features: Optional[list] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute only essential geometric features for reclassification.
        
        This is much faster than full geometric feature computation since it
        only computes what's needed and uses simplified calculations.
        
        Args:
            points: [N, 3] point coordinates
            normals: [N, 3] surface normals
            k: number of neighbors
            required_features: List of required feature names
            
        Returns:
            Dictionary of essential geometric features
        """
        if required_features is None:
            required_features = ['planarity', 'density']
        
        N = len(points)
        features = {}
        
        # Early exit if no eigenvalue-based features needed
        eigenvalue_features = ['planarity', 'linearity', 'sphericity', 'anisotropy']
        need_eigenvalues = any(feat in required_features for feat in eigenvalue_features)
        
        if not need_eigenvalues and 'density' not in required_features:
            return features
        
        # Use vectorized computation with optimized batch size
        if self.use_gpu and cp is not None:
            batch_size = min(self.batch_size, N)
        else:
            batch_size = min(100_000, N)  # Smaller batches for CPU
        
        num_batches = (N + batch_size - 1) // batch_size
        
        # Initialize feature arrays
        for feat in required_features:
            if feat in eigenvalue_features or feat == 'density':
                features[feat] = np.zeros(N, dtype=np.float32)
        
        # Process in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_points = points[start_idx:end_idx]
            
            # Build KDTree for batch
            if self.use_gpu and self.use_cuml and cuNearestNeighbors is not None:
                # GPU path
                points_gpu = cp.asarray(batch_points)
                knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
                knn.fit(points_gpu)
                distances, indices = knn.kneighbors(points_gpu)
                indices = cp.asnumpy(indices)
                distances = cp.asnumpy(distances)
            else:
                # CPU path (fast KDTree)
                tree = KDTree(batch_points, metric='euclidean')
                distances, indices = tree.query(batch_points, k=k)
            
            if need_eigenvalues:
                # Compute eigenvalue features for batch
                batch_eigen_features = self._compute_batch_eigenvalue_features(
                    batch_points, indices, required_features
                )
                for feat, values in batch_eigen_features.items():
                    features[feat][start_idx:end_idx] = values
            
            if 'density' in required_features:
                # Fast density estimation from mean distance
                mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self
                density = np.clip(1.0 / (mean_distances + 1e-8), 0.0, 1000.0)
                features['density'][start_idx:end_idx] = density.astype(np.float32)
        
        return features

    def _compute_batch_eigenvalue_features(
        self,
        points: np.ndarray,
        indices: np.ndarray,
        required_features: list
    ) -> Dict[str, np.ndarray]:
        """
        Compute eigenvalue-based features for a batch using vectorized operations.
        
        Args:
            points: [M, 3] batch of points
            indices: [M, k] neighbor indices
            required_features: List of required feature names
            
        Returns:
            Dictionary of computed features for the batch
        """
        M, k = indices.shape
        use_gpu = self.use_gpu and cp is not None
        xp = cp if use_gpu else np
        
        # Transfer to GPU if available
        if use_gpu:
            points_gpu = cp.asarray(points)
            indices_gpu = cp.asarray(indices)
            neighbors = points_gpu[indices_gpu]
        else:
            neighbors = points[indices]
        
        # Center neighbors
        centroids = xp.mean(neighbors, axis=1, keepdims=True)
        centered = neighbors - centroids
        
        # Compute covariance matrices
        cov_matrices = xp.einsum('mki,mkj->mij', centered, centered) / (k - 1)
        
        # Add regularization for stability
        reg_term = 1e-6 if use_gpu else 1e-8
        if use_gpu:
            eye = cp.eye(3, dtype=cov_matrices.dtype)
        else:
            eye = np.eye(3, dtype=np.float32)
        cov_matrices = cov_matrices + reg_term * eye
        
        # Compute eigenvalues
        try:
            eigenvalues = xp.linalg.eigvalsh(cov_matrices)
            eigenvalues = xp.sort(eigenvalues, axis=1)[:, ::-1]  # Sort descending
            eigenvalues = xp.maximum(eigenvalues, 1e-10)  # Clamp to positive
        except Exception as e:
            print(f"⚠️ Eigenvalue computation failed: {e}, using defaults")
            # Use default values
            batch_features = {}
            for feat in required_features:
                if feat in ['planarity', 'linearity', 'sphericity', 'anisotropy']:
                    batch_features[feat] = np.zeros(M, dtype=np.float32)
            return batch_features
        
        # Extract eigenvalues
        λ0 = eigenvalues[:, 0]
        λ1 = eigenvalues[:, 1]
        λ2 = eigenvalues[:, 2]
        sum_λ = λ0 + λ1 + λ2
        
        # Compute only required features
        batch_features = {}
        
        if 'planarity' in required_features:
            planarity = (λ1 - λ2) / (sum_λ + 1e-8)
            batch_features['planarity'] = self._to_cpu(planarity).astype(np.float32)
        
        if 'linearity' in required_features:
            linearity = (λ0 - λ1) / (sum_λ + 1e-8)
            batch_features['linearity'] = self._to_cpu(linearity).astype(np.float32)
        
        if 'sphericity' in required_features:
            sphericity = λ2 / (sum_λ + 1e-8)
            batch_features['sphericity'] = self._to_cpu(sphericity).astype(np.float32)
        
        if 'anisotropy' in required_features:
            anisotropy = (λ0 - λ2) / (sum_λ + 1e-8)
            batch_features['anisotropy'] = self._to_cpu(anisotropy).astype(np.float32)
        
        return batch_features

    def _to_cpu(self, array) -> np.ndarray:
        """Convert GPU array to CPU if needed."""
        if self.use_gpu and cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array

    def extract_geometric_features(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int = 10,
        radius: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive geometric features for each point.
        Version with fallback to core implementation.
        
        .. deprecated:: 1.8.0
            This class method is deprecated. Use the class for GPU-specific
            operations, but use ign_lidar.features.core.extract_geometric_features() 
            for feature extraction.
        
        Features computed (eigenvalue-based):
        - Planarity: (λ1-λ2)/λ0 - plane surfaces
        - Linearity: (λ0-λ1)/λ0 - linear structures
        - Sphericity: λ2/λ0 - spherical structures
        - Anisotropy: (λ0-λ2)/λ0 - anisotropy
        - Roughness: λ2/Σλ - roughness
        - Density: local point density
        
        Args:
            points: [N, 3] point coordinates
            normals: [N, 3] normal vectors (not used, kept for compat)
            k: number of neighbors (used if radius=None)
            radius: search radius in meters (RECOMMENDED, avoids scan artifacts)
            
        Returns:
            features: dictionary of geometric features
        """
        import warnings
        warnings.warn(
            "GPUFeatureComputer.extract_geometric_features is deprecated. "
            "Use ign_lidar.features.core.extract_geometric_features() directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Use core implementation for consistency
        return core_extract_geometric_features(points, normals, k=k, radius=radius)


# Global instance for reuse
_gpu_computer = None


def get_gpu_computer(
    use_gpu: bool = True,
    batch_size: int = 100000
) -> GPUFeatureComputer:
    """Get GPU computer instance (singleton pattern)."""
    global _gpu_computer
    if _gpu_computer is None:
        _gpu_computer = GPUFeatureComputer(
            use_gpu=use_gpu,
            batch_size=batch_size
        )
    return _gpu_computer


# API compatible with features.py (drop-in replacement)
def compute_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    """API-compatible wrapper."""
    computer = get_gpu_computer()
    return computer.compute_normals(points, k)


def compute_curvature(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """API-compatible wrapper."""
    computer = get_gpu_computer()
    return computer.compute_curvature(points, normals, k)


def compute_height_above_ground(
    points: np.ndarray,
    classification: np.ndarray
) -> np.ndarray:
    """API-compatible wrapper."""
    computer = get_gpu_computer()
    return computer.compute_height_above_ground(points, classification)


def extract_geometric_features(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 10
) -> Dict[str, np.ndarray]:
    """
    Wrapper for geometric features extraction.
    
    .. deprecated:: 1.8.0
        Use ign_lidar.features.core.extract_geometric_features() directly instead.
        This wrapper will be removed in v2.0.0.
    
    Args:
        points: [N, 3] point coordinates
        normals: [N, 3] normal vectors
        k: number of neighbors
        
    Returns:
        features: dictionary of geometric features
    """
    import warnings
    warnings.warn(
        "extract_geometric_features from features_gpu is deprecated. "
        "Use ign_lidar.features.core.extract_geometric_features() directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return core_extract_geometric_features(points, normals, k=k)


def compute_verticality(normals: np.ndarray) -> np.ndarray:
    """
    Wrapper for GPU-accelerated verticality computation.
    
    .. deprecated:: 1.8.0
        Use ign_lidar.features.core.compute_verticality() directly instead.
        This wrapper will be removed in v2.0.0.
    
    Args:
        normals: [N, 3] surface normal vectors
        
    Returns:
        verticality: [N] verticality values [0, 1]
    """
    import warnings
    warnings.warn(
        "compute_verticality from features_gpu is deprecated. "
        "Use ign_lidar.features.core.compute_verticality() directly.",
        DeprecationWarning,
        stacklevel=2
    )
    computer = get_gpu_computer()
    return computer.compute_verticality(normals)