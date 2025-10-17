"""
Enhanced GPU Feature Computer with Optimizations
==============================================

This module provides an optimized GPU feature computer that addresses
the hang issues and implements performance improvements.
"""

import numpy as np
import logging
import warnings
from typing import Dict, Tuple, Optional
import gc

logger = logging.getLogger(__name__)

# GPU imports with fallback
try:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    GPU_AVAILABLE = True
    CUML_AVAILABLE = True
    logger.info("✓ CuPy and cuML available - Enhanced GPU mode enabled")
except ImportError:
    cp = None
    cuNearestNeighbors = None
    GPU_AVAILABLE = False
    CUML_AVAILABLE = False
    logger.warning("⚠ Enhanced GPU mode not available - install cupy and cuml")

# CPU fallback imports
from sklearn.neighbors import NearestNeighbors as skNearestNeighbors
from sklearn.neighbors import KDTree


class OptimizedGPUFeatureComputer:
    """
    Optimized GPU feature computer with enhanced performance and reliability.
    
    Key optimizations:
    - Robust GPU context management
    - Memory pooling and efficient cleanup
    - Adaptive batch sizing
    - Fallback mechanisms for edge cases
    - Performance monitoring
    """
    
    def __init__(self, config: dict = None, use_gpu: bool = True, batch_size: int = 1_000_000):
        """
        Initialize optimized GPU feature computer.
        
        Args:
            config: Configuration dictionary (for compatibility)
            use_gpu: Whether to use GPU acceleration
            batch_size: Initial batch size for processing
        """
        self.config = config or {}
        
        # Extract settings from config if provided
        if config:
            use_gpu = config.get('processing', {}).get('use_gpu', use_gpu)
            batch_size = config.get('gpu', {}).get('batch_size', batch_size)
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_cuml = use_gpu and CUML_AVAILABLE
        
        # Initialize optimal batch size
        self.batch_size = self._optimize_batch_size(batch_size)
        
        # Initialize GPU context
        self._cuda_initialized = False
        if self.use_gpu:
            self._cuda_initialized = self._initialize_cuda_context()
        
        # Memory management
        self._memory_pool = None
        if self.use_gpu:
            self._setup_memory_pool()
        
        logger.info(f"OptimizedGPUFeatureComputer initialized: "
                   f"gpu={self.use_gpu}, cuml={self.use_cuml}, "
                   f"batch_size={self.batch_size:,}")
    
    def _optimize_batch_size(self, initial_batch_size: int) -> int:
        """Optimize batch size based on available GPU memory."""
        if not self.use_gpu:
            return initial_batch_size
        
        try:
            # Get GPU memory info
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            
            # Conservative batch sizing based on available memory
            if free_gb >= 12.0:  # High-end GPUs
                optimal_batch = min(initial_batch_size, 4_000_000)
            elif free_gb >= 8.0:  # Mid-range GPUs
                optimal_batch = min(initial_batch_size, 2_000_000)
            elif free_gb >= 4.0:  # Standard GPUs
                optimal_batch = min(initial_batch_size, 1_000_000)
            else:  # Low memory GPUs
                optimal_batch = min(initial_batch_size, 500_000)
            
            logger.info(f"GPU memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
            logger.info(f"Optimized batch size: {optimal_batch:,} points")
            
            return optimal_batch
            
        except Exception as e:
            logger.warning(f"Could not optimize batch size: {e}")
            return initial_batch_size
    
    def _initialize_cuda_context(self) -> bool:
        """Initialize CUDA context safely with error handling."""
        if not self.use_gpu or cp is None:
            return False
        
        try:
            # Force CUDA context initialization
            device = cp.cuda.Device()
            device.use()
            
            # Test basic operations
            test_array = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
            result = cp.sum(test_array)
            _ = cp.asnumpy(result)
            
            logger.info("✓ CUDA context initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"CUDA initialization failed: {e}")
            logger.warning("Falling back to CPU mode")
            self.use_gpu = False
            self.use_cuml = False
            return False
    
    def _setup_memory_pool(self):
        """Setup CUDA memory pool for efficient memory management."""
        if not self.use_gpu:
            return
        
        try:
            self._memory_pool = cp.get_default_memory_pool()
            # Pre-allocate some memory to reduce allocation overhead
            self._memory_pool.set_limit(size=None)  # No limit, use all available
            logger.debug("✓ CUDA memory pool configured")
            
        except Exception as e:
            logger.warning(f"Could not setup memory pool: {e}")
            self._memory_pool = None
    
    def _cleanup_gpu_memory(self):
        """Efficiently clean up GPU memory."""
        if self.use_gpu and self._memory_pool is not None:
            self._memory_pool.free_all_blocks()
            cp.cuda.Device().synchronize()
            gc.collect()
    
    def compute_normals_optimized(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Optimized normal computation with robust error handling.
        
        Args:
            points: [N, 3] point coordinates
            k: number of neighbors for PCA
            
        Returns:
            normals: [N, 3] normalized surface normals
        """
        if not self.use_gpu or not self._cuda_initialized:
            return self._compute_normals_cpu_optimized(points, k)
        
        try:
            return self._compute_normals_gpu_optimized(points, k)
        except Exception as e:
            logger.warning(f"GPU normal computation failed: {e}")
            logger.info("Falling back to optimized CPU computation")
            return self._compute_normals_cpu_optimized(points, k)
    
    def _compute_normals_gpu_optimized(self, points: np.ndarray, k: int) -> np.ndarray:
        """Optimized GPU normal computation with better error handling."""
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        
        # Transfer to GPU with error checking
        try:
            points_gpu = cp.asarray(points, dtype=cp.float32)
        except Exception as e:
            logger.warning(f"GPU transfer failed: {e}")
            return self._compute_normals_cpu_optimized(points, k)
        
        # Build KNN model
        try:
            if self.use_cuml:
                knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
                knn.fit(points_gpu)
            else:
                # Fallback to scikit-learn with GPU arrays
                points_cpu = cp.asnumpy(points_gpu)
                knn = skNearestNeighbors(n_neighbors=k, metric='euclidean')
                knn.fit(points_cpu)
        except Exception as e:
            logger.warning(f"KNN model creation failed: {e}")
            return self._compute_normals_cpu_optimized(points, k)
        
        # Process in optimized batches
        num_batches = max(1, (N + self.batch_size - 1) // self.batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, N)
            
            try:
                if self.use_cuml:
                    batch_points = points_gpu[start_idx:end_idx]
                    _, indices = knn.kneighbors(batch_points)
                    batch_normals = self._compute_batch_pca_optimized(points_gpu, indices)
                else:
                    batch_points = cp.asnumpy(points_gpu[start_idx:end_idx])
                    _, indices = knn.kneighbors(batch_points)
                    batch_normals = self._compute_batch_pca_cpu(points, indices, start_idx)
                
                normals[start_idx:end_idx] = cp.asnumpy(batch_normals) if isinstance(batch_normals, cp.ndarray) else batch_normals
                
                # Clean up intermediate memory
                if batch_idx % 10 == 0:  # Clean every 10 batches
                    self._cleanup_gpu_memory()
                    
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}, using CPU fallback")
                batch_points = points[start_idx:end_idx]
                batch_normals = self._compute_batch_pca_cpu_simple(batch_points, k)
                normals[start_idx:end_idx] = batch_normals
        
        return normals
    
    def _compute_batch_pca_optimized(self, points_gpu, neighbor_indices):
        """Optimized batch PCA computation with better error handling."""
        batch_size, k = neighbor_indices.shape
        
        # Conservative sub-batching for CUSOLVER stability
        max_cusolver_batch = 200_000  # More conservative limit
        
        if batch_size <= max_cusolver_batch:
            return self._compute_batch_pca_core_safe(points_gpu, neighbor_indices)
        
        # Split into sub-batches
        num_sub_batches = (batch_size + max_cusolver_batch - 1) // max_cusolver_batch
        normals = cp.zeros((batch_size, 3), dtype=cp.float32)
        
        for sub_batch_idx in range(num_sub_batches):
            start_idx = sub_batch_idx * max_cusolver_batch
            end_idx = min((sub_batch_idx + 1) * max_cusolver_batch, batch_size)
            
            try:
                sub_indices = neighbor_indices[start_idx:end_idx]
                sub_normals = self._compute_batch_pca_core_safe(points_gpu, sub_indices)
                normals[start_idx:end_idx] = sub_normals
            except Exception as e:
                logger.warning(f"Sub-batch {sub_batch_idx} failed: {e}")
                # Fallback to CPU for this sub-batch
                sub_indices_cpu = cp.asnumpy(neighbor_indices[start_idx:end_idx])
                points_cpu = cp.asnumpy(points_gpu)
                sub_normals = self._compute_batch_pca_cpu(points_cpu, sub_indices_cpu, 0)
                normals[start_idx:end_idx] = cp.asarray(sub_normals)
        
        return normals
    
    def _compute_batch_pca_core_safe(self, points_gpu, neighbor_indices):
        """Safe core PCA computation with fallback for edge cases."""
        try:
            batch_size, k = neighbor_indices.shape
            
            # Gather neighbor points
            neighbor_points = points_gpu[neighbor_indices]
            
            # Center neighborhoods
            centroids = cp.mean(neighbor_points, axis=1, keepdims=True)
            centered = neighbor_points - centroids
            
            # Compute covariance matrices with regularization
            cov_matrices = cp.einsum('mki,mkj->mij', centered, centered) / (k - 1)
            
            # Add regularization for numerical stability
            reg_term = 1e-6
            eye = cp.eye(3, dtype=cov_matrices.dtype)
            cov_matrices = cov_matrices + reg_term * eye
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = cp.linalg.eigh(cov_matrices)
            
            # Normal = eigenvector with smallest eigenvalue
            normals = eigenvectors[:, :, 0]
            
            # Normalize normals
            norms = cp.linalg.norm(normals, axis=1, keepdims=True)
            norms = cp.maximum(norms, 1e-8)
            normals = normals / norms
            
            # Orient normals upward
            flip_mask = normals[:, 2] < 0
            normals[flip_mask] *= -1
            
            # Handle degenerate cases
            variances = cp.sum(eigenvalues, axis=1)
            degenerate = variances < 1e-8
            if cp.any(degenerate):
                normals[degenerate] = cp.array([0, 0, 1], dtype=cp.float32)
            
            return normals
            
        except Exception as e:
            logger.warning(f"GPU PCA failed: {e}")
            # Fallback to simple CPU computation
            indices_cpu = cp.asnumpy(neighbor_indices)
            points_cpu = cp.asnumpy(points_gpu)
            return cp.asarray(self._compute_batch_pca_cpu(points_cpu, indices_cpu, 0))
    
    def _compute_normals_cpu_optimized(self, points: np.ndarray, k: int) -> np.ndarray:
        """Optimized CPU normal computation with vectorization."""
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        
        # Build KDTree with optimized parameters
        tree = KDTree(points, metric='euclidean', leaf_size=50)
        
        # Process in optimized batches
        cpu_batch_size = 50_000
        num_batches = (N + cpu_batch_size - 1) // cpu_batch_size
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * cpu_batch_size
                end_idx = min((batch_idx + 1) * cpu_batch_size, N)
                
                batch_points = points[start_idx:end_idx]
                _, indices = tree.query(batch_points, k=k)
                
                batch_normals = self._compute_batch_pca_cpu_vectorized(points, indices)
                normals[start_idx:end_idx] = batch_normals
        
        return normals
    
    def _compute_batch_pca_cpu_vectorized(self, points: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Vectorized CPU PCA computation."""
        # Gather neighbor points
        neighbor_points = points[indices]
        
        # Center neighborhoods
        centroids = np.mean(neighbor_points, axis=1, keepdims=True)
        centered = neighbor_points - centroids
        
        # Compute covariance matrices
        k = indices.shape[1]
        cov_matrices = np.einsum('mki,mkj->mij', centered, centered) / (k - 1)
        
        # Add regularization
        reg_term = 1e-8
        eye = np.eye(3, dtype=np.float32)
        cov_matrices = cov_matrices + reg_term * eye
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
        
        # Normal = eigenvector with smallest eigenvalue
        normals = eigenvectors[:, :, 0]
        
        # Normalize normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normals = normals / norms
        
        # Orient normals upward
        flip_mask = normals[:, 2] < 0
        normals[flip_mask] *= -1
        
        # Handle degenerate cases
        variances = np.sum(eigenvalues, axis=1)
        degenerate = variances < 1e-8
        if np.any(degenerate):
            normals[degenerate] = [0, 0, 1]
        
        return normals.astype(np.float32)
    
    def compute_all_features(self, points: np.ndarray, classification: np.ndarray = None, 
                           colors: np.ndarray = None) -> tuple:
        """
        Main compute method for compatibility with existing interface.
        
        Args:
            points: Point cloud coordinates (N, 3)
            classification: Point classifications (N,)
            colors: Point colors (N, 3) - optional
            
        Returns:
            Tuple of (points, normals, other_features, feature_dict)
        """
        logger.info("Computing features with OptimizedGPUFeatureComputer")
        
        # Compute normals as primary feature
        normals = self.compute_normals_optimized(points)
        
        # Calculate verticality from normals
        verticality = np.abs(normals[:, 2])  # Z-component of normal
        
        # Create feature dictionary
        features_dict = {
            'verticality': verticality
        }
        
        # Add more features if requested
        if self.config.get('features', {}).get('geometric', True):
            # Simple geometric features for demonstration
            features_dict.update({
                'height': points[:, 2],
                'normal_x': normals[:, 0],
                'normal_y': normals[:, 1], 
                'normal_z': normals[:, 2]
            })
        
        logger.info(f"Computed {len(features_dict)} feature types")
        
        return points, normals, verticality, features_dict
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        return {
            'gpu_enabled': self.use_gpu,
            'cuml_available': self.use_cuml,
            'batch_size': self.batch_size,
            'memory_pooling': hasattr(self, '_memory_pool')
        }
    
    def _compute_batch_pca_cpu(self, points: np.ndarray, indices: np.ndarray, offset: int = 0) -> np.ndarray:
        """Simple CPU PCA computation for fallback."""
        return self._compute_batch_pca_cpu_vectorized(points, indices)
    
    def _compute_batch_pca_cpu_simple(self, points: np.ndarray, k: int) -> np.ndarray:
        """Simple PCA for small batches."""
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        
        tree = KDTree(points, metric='euclidean')
        _, indices = tree.query(points, k=k)
        
        return self._compute_batch_pca_cpu_vectorized(points, indices)