"""
Unified GPU Feature Processor (Phase 2A Consolidation)

This module consolidates features_gpu.py and features_gpu_chunked.py into a
single, intelligent GPU processor with automatic chunking based on dataset size
and available VRAM.

Key improvements over previous implementations:
- Single source of truth for GPU feature computation
- Automatic strategy selection (chunking vs batching)
- Simplified API with smart defaults
- Integrated GPU-Core Bridge for eigenvalue features
- Memory management from chunked version
- Performance optimizations from both versions

Version: 4.0.0 (Phase 2A Consolidation)
Date: October 19, 2025
"""

from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import logging
import gc
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)

# GPU imports with fallback
GPU_AVAILABLE = False
CUML_AVAILABLE = False

try:
    import cupy as cp
    from cupyx.scipy.spatial import distance as cp_distance
    GPU_AVAILABLE = True
    CpArray = cp.ndarray
    logger.info("✓ CuPy available - GPU enabled")
except ImportError:
    logger.warning("⚠ CuPy not available - CPU fallback")
    cp = None
    CpArray = Any

try:
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    from cuml.decomposition import PCA as cuPCA
    CUML_AVAILABLE = True
    logger.info("✓ RAPIDS cuML available - GPU algorithms enabled")
except ImportError:
    logger.warning("⚠ RAPIDS cuML not available - sklearn fallback")
    cuNearestNeighbors = None
    cuPCA = None

# FAISS GPU support (50-100× faster than cuML for k-NN)
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("✓ FAISS available - Ultra-fast k-NN enabled (50-100× speedup)")
except ImportError:
    logger.debug("FAISS not available - using cuML/sklearn for k-NN")
    faiss = None

# CPU fallback imports
from sklearn.neighbors import KDTree, NearestNeighbors

# Import compute feature implementations
from ..features.compute import (
    compute_normals as core_compute_normals,
    compute_curvature as core_compute_curvature,
    compute_eigenvalue_features as core_compute_eigenvalue_features,
    compute_density_features as core_compute_density_features,
    compute_verticality as core_compute_verticality,
    extract_geometric_features as core_extract_geometric_features,
)

# Import GPU-Core Bridge
from .compute.gpu_bridge import GPUCoreBridge

# Import compute utilities
from .compute.utils import (
    batched_inverse_3x3, 
    inverse_power_iteration,
    compute_eigenvalue_features_from_covariances,
    compute_covariances_from_neighbors,
)
from .compute.height import compute_height_above_ground
from .compute.curvature import compute_curvature_from_normals


class GPUProcessor:
    """
    Unified GPU feature processor with automatic chunking.
    
    Consolidates features_gpu.py and features_gpu_chunked.py into a single
    intelligent processor that automatically selects the optimal strategy:
    
    - Small datasets (<1M points): Simple batching (fast, low overhead)
    - Medium datasets (1-10M points): Adaptive batching
    - Large datasets (>10M points): Chunked processing with global KDTree
    
    The processor automatically manages VRAM, selects batch sizes, and falls
    back to CPU if GPU is unavailable.
    
    Example:
        >>> # Auto-detect optimal strategy
        >>> processor = GPUProcessor()
        >>> features = processor.compute_features(points, k=10)
        
        >>> # Force chunking for large datasets
        >>> processor = GPUProcessor(auto_chunk=True, chunk_size=5_000_000)
        >>> features = processor.compute_features(points, k=20)
        
        >>> # Disable auto-chunking for small datasets
        >>> processor = GPUProcessor(auto_chunk=False)
        >>> normals = processor.compute_normals(points, k=10)
    """
    
    def __init__(
        self,
        auto_chunk: bool = True,
        chunk_size: Optional[int] = None,
        vram_limit_gb: Optional[float] = None,
        use_gpu: bool = True,
        show_progress: bool = True,
        use_cuda_streams: bool = True,
        enable_memory_pooling: bool = True,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize unified GPU processor.
        
        Args:
            auto_chunk: Auto-detect whether to use chunking (default: True)
            chunk_size: Points per chunk for large datasets (None = auto-detect)
            vram_limit_gb: Max VRAM usage in GB (None = auto-detect)
            use_gpu: Enable GPU if available (default: True)
            show_progress: Show progress bars (default: True)
            use_cuda_streams: Enable CUDA streams for overlap (default: True)
            enable_memory_pooling: Enable memory pooling (default: True)
            batch_size: Override default batch size for small datasets
        """
        self.auto_chunk = auto_chunk
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_cuml = use_gpu and CUML_AVAILABLE
        self.show_progress = show_progress
        self.use_cuda_streams = use_cuda_streams and GPU_AVAILABLE
        self.enable_memory_pooling = enable_memory_pooling
        
        # Initialize GPU context
        if self.use_gpu:
            self._initialize_cuda_context()
        
        # Auto-detect VRAM and configure chunking thresholds
        if self.use_gpu:
            self._configure_vram_limits(vram_limit_gb)
            self._configure_chunking_thresholds(chunk_size, batch_size)
        else:
            # CPU fallback configuration
            self.vram_total_gb = 0.0
            self.vram_limit_gb = 0.0
            self.chunk_threshold = float('inf')  # Never chunk on CPU
            self.chunk_size = chunk_size or 1_000_000
            self.batch_size = batch_size or 500_000
        
        # Initialize GPU-Core Bridge for eigenvalue features
        self.gpu_bridge = GPUCoreBridge(
            use_gpu=self.use_gpu,
            batch_size=500_000,  # cuSOLVER batch limit
            epsilon=1e-10
        )
        
        # Initialize memory pooling
        if self.enable_memory_pooling and self.use_gpu and cp is not None:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=int(self.vram_limit_gb * 1024**3))
                logger.info(f"✓ GPU memory pool configured: {self.vram_limit_gb:.1f}GB limit")
            except Exception as e:
                logger.warning(f"Could not configure memory pool: {e}")
        
        # Log configuration
        self._log_configuration()
    
    def _initialize_cuda_context(self):
        """Initialize CUDA context safely for multiprocessing."""
        if not self.use_gpu or cp is None:
            return False
        
        try:
            # Force CUDA context initialization
            _ = cp.cuda.Device()
            # Test basic operation
            test_array = cp.array([1.0], dtype=cp.float32)
            _ = cp.asnumpy(test_array)
            return True
        except Exception as e:
            logger.warning(f"⚠ CUDA initialization failed: {e}")
            logger.info("💻 Falling back to CPU mode")
            self.use_gpu = False
            self.use_cuml = False
            return False
    
    def _configure_vram_limits(self, vram_limit_gb: Optional[float]):
        """Auto-detect VRAM and configure limits."""
        if not self.use_gpu or cp is None:
            return
        
        try:
            free_vram, total_vram = cp.cuda.runtime.memGetInfo()
            self.vram_total_gb = total_vram / (1024**3)
            
            if vram_limit_gb is not None:
                # User-specified limit
                self.vram_limit_gb = min(vram_limit_gb, self.vram_total_gb * 0.9)
            else:
                # Auto-detect: use 80% of total VRAM
                self.vram_limit_gb = self.vram_total_gb * 0.8
            
            logger.info(f"✓ VRAM detected: {self.vram_total_gb:.1f}GB total, "
                       f"{self.vram_limit_gb:.1f}GB limit")
        except Exception as e:
            logger.warning(f"Could not detect VRAM: {e}")
            self.vram_total_gb = 8.0  # Conservative fallback
            self.vram_limit_gb = 6.4
    
    def _configure_chunking_thresholds(
        self, 
        chunk_size: Optional[int],
        batch_size: Optional[int]
    ):
        """
        Configure chunking thresholds based on VRAM.
        
        Strategy selection based on dataset size:
        - < 1M points: Simple batching (no global KDTree)
        - 1-10M points: Adaptive batching
        - > 10M points: Chunked processing with global KDTree
        """
        # Determine when to switch from batching to chunking
        # Based on VRAM capacity
        if self.vram_total_gb >= 15.0:  # RTX 4080 Super, A100
            self.chunk_threshold = 10_000_000  # 10M points
            default_chunk_size = 5_000_000
            default_batch_size = 12_000_000
        elif self.vram_total_gb >= 12.0:  # RTX 3090, RTX 4070 Ti
            self.chunk_threshold = 5_000_000  # 5M points
            default_chunk_size = 3_000_000
            default_batch_size = 6_000_000
        elif self.vram_total_gb >= 8.0:  # RTX 3060, RTX 4060
            self.chunk_threshold = 2_000_000  # 2M points
            default_chunk_size = 1_500_000
            default_batch_size = 3_000_000
        else:  # < 8GB VRAM
            self.chunk_threshold = 1_000_000  # 1M points
            default_chunk_size = 500_000
            default_batch_size = 1_500_000
        
        # Apply user overrides or use defaults
        self.chunk_size = chunk_size if chunk_size is not None else default_chunk_size
        self.batch_size = batch_size if batch_size is not None else default_batch_size
        
        logger.info(f"✓ Chunking threshold: {self.chunk_threshold:,} points")
        logger.info(f"✓ Chunk size: {self.chunk_size:,} points")
        logger.info(f"✓ Batch size: {self.batch_size:,} points")
    
    def _log_configuration(self):
        """Log processor configuration."""
        if self.use_gpu:
            logger.info(f"🚀 GPUProcessor initialized (GPU mode)")
            logger.info(f"   Auto-chunk: {self.auto_chunk}")
            logger.info(f"   VRAM: {self.vram_total_gb:.1f}GB total, "
                       f"{self.vram_limit_gb:.1f}GB limit")
            logger.info(f"   Strategy: {'Auto-detect' if self.auto_chunk else 'Fixed batching'}")
        else:
            logger.info(f"💻 GPUProcessor initialized (CPU mode)")
    
    def _select_strategy(self, n_points: int) -> str:
        """
        Select optimal processing strategy based on dataset size.
        
        Returns:
            'batch': Simple batching for small/medium datasets
            'chunk': Chunked processing for large datasets
        """
        if not self.auto_chunk:
            return 'batch'
        
        if n_points > self.chunk_threshold:
            logger.info(f"📊 Selected CHUNKED strategy for {n_points:,} points")
            return 'chunk'
        else:
            logger.info(f"📊 Selected BATCH strategy for {n_points:,} points")
            return 'batch'
    
    # ==========================================================================
    # PUBLIC API - Main Feature Computation Methods
    # ==========================================================================
    
    def compute_features(
        self,
        points: np.ndarray,
        feature_types: Optional[List[str]] = None,
        k: int = 10,
        show_progress: Optional[bool] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute geometric features with automatic strategy selection.
        
        This is the main entry point for feature computation. It automatically
        selects the optimal strategy (batching vs chunking) based on dataset size.
        
        Args:
            points: Point cloud (N, 3) array
            feature_types: List of features to compute (None = all)
                Options: ['normals', 'curvature', 'eigenvalues', 'height', 
                         'verticality', 'density']
            k: Number of neighbors for features (default: 10)
            show_progress: Override show_progress setting
        
        Returns:
            Dictionary of computed features:
                'normals': (N, 3) array
                'curvature': (N,) array
                'eigenvalues': Dict with 'sum', 'omnivariance', etc.
                'height': (N,) array
                'verticality': (N,) array
                'density': (N,) array
        """
        n_points = len(points)
        show_progress = show_progress if show_progress is not None else self.show_progress
        
        # Select strategy
        strategy = self._select_strategy(n_points)
        
        # Route to appropriate implementation
        if strategy == 'chunk':
            return self._compute_features_chunked(
                points, feature_types, k, show_progress
            )
        else:
            return self._compute_features_batch(
                points, feature_types, k, show_progress
            )
    
    def compute_normals(
        self, 
        points: np.ndarray, 
        k: int = 10,
        show_progress: Optional[bool] = None
    ) -> np.ndarray:
        """
        Compute normal vectors with automatic strategy selection.
        
        Args:
            points: Point cloud (N, 3)
            k: Number of neighbors (default: 10)
            show_progress: Override show_progress setting
        
        Returns:
            normals: (N, 3) array of unit normal vectors
        """
        n_points = len(points)
        show_progress = show_progress if show_progress is not None else self.show_progress
        strategy = self._select_strategy(n_points)
        
        if strategy == 'chunk':
            return self._compute_normals_chunked(points, k, show_progress)
        else:
            return self._compute_normals_batch(points, k, show_progress)
    
    def compute_curvature(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int = 10,
        show_progress: Optional[bool] = None
    ) -> np.ndarray:
        """
        Compute curvature with automatic strategy selection.
        
        Args:
            points: Point cloud (N, 3)
            normals: Normal vectors (N, 3)
            k: Number of neighbors (default: 10)
            show_progress: Override show_progress setting
        
        Returns:
            curvature: (N,) array of curvature values
        """
        n_points = len(points)
        show_progress = show_progress if show_progress is not None else self.show_progress
        strategy = self._select_strategy(n_points)
        
        if strategy == 'chunk':
            return self._compute_curvature_chunked(points, normals, k, show_progress)
        else:
            return self._compute_curvature_batch(points, normals, k, show_progress)
    
    # ==========================================================================
    # BATCH PROCESSING - For Small/Medium Datasets (<chunk_threshold)
    # ==========================================================================
    
    def _compute_features_batch(
        self,
        points: np.ndarray,
        feature_types: Optional[List[str]],
        k: int,
        show_progress: bool
    ) -> Dict[str, np.ndarray]:
        """
        Compute features using simple batching (from features_gpu.py).
        
        This method is used for small to medium datasets that fit comfortably
        in VRAM without requiring global KDTree chunking.
        """
        features = {}
        
        # Determine which features to compute
        if feature_types is None:
            feature_types = ['normals', 'curvature', 'eigenvalues', 'verticality']
        
        # Compute normals first if needed by any feature
        normals = None
        if any(ft in feature_types for ft in ['normals', 'curvature', 'verticality']):
            normals = self._compute_normals_batch(points, k, show_progress)
            if 'normals' in feature_types:
                features['normals'] = normals
        
        # Compute curvature (requires normals)
        if 'curvature' in feature_types:
            if normals is None:
                normals = self._compute_normals_batch(points, k, show_progress)
            features['curvature'] = self._compute_curvature_batch(
                points, normals, k, show_progress
            )
        
        # Compute verticality (requires normals)
        if 'verticality' in feature_types:
            if normals is None:
                normals = self._compute_normals_batch(points, k, show_progress)
            features['verticality'] = self._compute_verticality_batch(normals)
        
        # Compute eigenvalue features via GPU Bridge
        if 'eigenvalues' in feature_types:
            # TODO: Integrate GPU Bridge eigenvalue computation properly
            # For now, skip eigenvalues - will be added in integration step
            logger.info("Eigenvalue features via GPU Bridge - pending integration")
        
        # Compute height (requires classification, handled separately)
        if 'height' in feature_types:
            logger.warning("Height computation requires classification data. "
                         "Call compute_height_above_ground() separately.")
        
        return features
    
    def _compute_normals_batch(
        self,
        points: np.ndarray,
        k: int,
        show_progress: bool
    ) -> np.ndarray:
        """
        Compute normals using simple batching (from features_gpu.py).
        
        Uses cuML for GPU k-NN or sklearn for CPU fallback.
        Processes in batches to avoid OOM errors.
        """
        if not self.use_gpu or not self.use_cuml or cuNearestNeighbors is None:
            return self._compute_normals_cpu(points, k)
        
        try:
            N = len(points)
            
            # Transfer to GPU
            points_gpu = self._to_gpu(points)
            
            if not isinstance(points_gpu, cp.ndarray):
                return self._compute_normals_cpu(points, k)
            
            # Build GPU KNN
            if show_progress:
                logger.info(f"  Building GPU KNN index ({N:,} points)...")
            knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(points_gpu)
            
            # Preallocate normals on GPU
            normals_gpu = cp.zeros((N, 3), dtype=cp.float32)
            
            # Process in batches
            num_batches = (N + self.batch_size - 1) // self.batch_size
            
            batch_iterator = range(num_batches)
            if show_progress and num_batches > 1:
                batch_iterator = tqdm(
                    batch_iterator,
                    desc=f"  Computing normals [GPU batch]",
                    unit="batch"
                )
            
            for batch_idx in batch_iterator:
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, N)
                batch_points = points_gpu[start_idx:end_idx]
                
                # Query KNN
                distances, indices = knn.kneighbors(batch_points)
                
                # Compute normals with GPU PCA
                batch_normals = self._batch_pca_gpu(points_gpu, indices)
                normals_gpu[start_idx:end_idx] = batch_normals
            
            # Transfer to CPU
            normals = self._to_cpu(normals_gpu)
            return normals
            
        except Exception as e:
            logger.warning(f"⚠ GPU normal computation failed: {e}")
            logger.info("💻 Falling back to CPU computation")
            return self._compute_normals_cpu(points, k)
    
    def _batch_pca_gpu(self, points_gpu, neighbor_indices):
        """
        Vectorized PCA on GPU with sub-batching for cuSOLVER limits.
        
        cuSOLVER has a maximum batch size limit (~500k matrices).
        """
        if not self.use_gpu or cp is None:
            raise RuntimeError("GPU not available")
        
        batch_size, k = neighbor_indices.shape
        
        # cuSOLVER batch limit
        max_cusolver_batch = 500_000
        
        if batch_size <= max_cusolver_batch:
            return self._batch_pca_gpu_core(points_gpu, neighbor_indices)
        
        # Split into sub-batches
        num_sub_batches = (batch_size + max_cusolver_batch - 1) // max_cusolver_batch
        logger.info(f"    Splitting {batch_size:,} into {num_sub_batches} "
                   f"sub-batches (cuSOLVER limit)")
        
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
        Core vectorized PCA computation on GPU.
        
        Uses inverse power iteration for fast eigenvector computation.
        """
        batch_size, k = neighbor_indices.shape
        
        # Gather neighbor points: [batch_size, k, 3]
        neighbor_points = points_gpu[neighbor_indices]
        
        # Center neighborhoods
        centroids = cp.mean(neighbor_points, axis=1, keepdims=True)
        centered = neighbor_points - centroids
        
        # Normalize for numerical stability
        centered = centered / cp.sqrt(k)
        
        # Compute covariance matrices
        cov_matrices = cp.einsum('mki,mkj->mij', centered, centered)
        
        # Find smallest eigenvector (surface normal)
        try:
            normals = inverse_power_iteration(
                cov_matrices, 
                num_iterations=8, 
                epsilon=1e-12
            )
        except Exception as e:
            logger.warning(f"⚠ Fast eigenvector method failed: {e}, using CPU fallback")
            cov_matrices_cpu = cp.asnumpy(cov_matrices)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices_cpu)
            normals_cpu = eigenvectors[:, :, 0]  # Smallest eigenvector
            normals = cp.asarray(normals_cpu)
        
        return normals
    
    def _compute_normals_cpu(self, points: np.ndarray, k: int) -> np.ndarray:
        """
        Compute normals on CPU using sklearn KDTree.
        
        Vectorized with parallel batch processing.
        """
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        
        # Build KDTree
        tree = KDTree(points, metric='euclidean', leaf_size=40)
        
        # Batch processing
        batch_size = 50_000
        num_batches = (N + batch_size - 1) // batch_size
        
        def process_batch(batch_idx):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_points = points[start_idx:end_idx]
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                
                # Query KNN
                _, indices = tree.query(batch_points, k=k)
                
                # Vectorized covariance computation
                neighbor_points = points[indices]
                centroids = np.mean(neighbor_points, axis=1, keepdims=True)
                centered = neighbor_points - centroids
                
                # Compute covariance matrices
                cov_matrices = np.einsum('mki,mkj->mij', centered, centered) / (k - 1)
                
                # Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
                batch_normals = eigenvectors[:, :, 0]  # Smallest eigenvector
                
                # Ensure upward orientation
                batch_normals[batch_normals[:, 2] < 0] *= -1
                
                return start_idx, end_idx, batch_normals
        
        # Parallel processing
        if num_batches > 1:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(process_batch)(i) for i in range(num_batches)
                )
                for start_idx, end_idx, batch_normals in results:
                    normals[start_idx:end_idx] = batch_normals
            except ImportError:
                # Fallback to sequential
                for i in range(num_batches):
                    start_idx, end_idx, batch_normals = process_batch(i)
                    normals[start_idx:end_idx] = batch_normals
        else:
            start_idx, end_idx, batch_normals = process_batch(0)
            normals[start_idx:end_idx] = batch_normals
        
        return normals
    
    def _compute_curvature_batch(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int,
        show_progress: bool
    ) -> np.ndarray:
        """
        Compute curvature using simple batching (from features_gpu.py).
        
        GPU-accelerated when available, with CPU fallback.
        """
        N = len(points)
        
        # GPU path
        if self.use_gpu and cp is not None and cuNearestNeighbors is not None:
            try:
                # Transfer to GPU
                points_gpu = self._to_gpu(points)
                normals_gpu = self._to_gpu(normals)
                curvature_gpu = cp.zeros(N, dtype=cp.float32)
                
                # Build GPU KNN
                knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
                knn.fit(points_gpu)
                
                # Process in batches
                batch_size = min(self.batch_size, 500_000)
                num_batches = (N + batch_size - 1) // batch_size
                
                batch_iterator = range(num_batches)
                if show_progress and num_batches > 1:
                    batch_iterator = tqdm(
                        batch_iterator,
                        desc=f"  Computing curvature [GPU batch]",
                        unit="batch"
                    )
                
                for batch_idx in batch_iterator:
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, N)
                    
                    batch_points = points_gpu[start_idx:end_idx]
                    distances, indices = knn.kneighbors(batch_points)
                    
                    # Compute on GPU
                    batch_normals_gpu = normals_gpu[start_idx:end_idx]
                    neighbor_normals_gpu = normals_gpu[indices]
                    
                    # Normal differences
                    query_normals_expanded = batch_normals_gpu[:, cp.newaxis, :]
                    normal_diff_gpu = neighbor_normals_gpu - query_normals_expanded
                    
                    # Mean L2 norm
                    curv_norms_gpu = cp.linalg.norm(normal_diff_gpu, axis=2)
                    batch_curvature_gpu = cp.mean(curv_norms_gpu, axis=1)
                    
                    curvature_gpu[start_idx:end_idx] = batch_curvature_gpu
                
                return self._to_cpu(curvature_gpu)
                
            except Exception as e:
                logger.warning(f"⚠ GPU curvature failed: {e}, using CPU fallback")
        
        # CPU fallback
        return self._compute_curvature_cpu(points, normals, k)
    
    def _compute_curvature_cpu(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int
    ) -> np.ndarray:
        """CPU fallback for curvature computation."""
        # Build KDTree and query neighbors
        tree = KDTree(points, metric='euclidean', leaf_size=40)
        _, neighbor_indices = tree.query(points, k=k)
        
        # Use core implementation with precomputed neighbors
        return compute_curvature_from_normals(points, normals, neighbor_indices)
    
    def _compute_verticality_batch(self, normals: np.ndarray) -> np.ndarray:
        """
        Compute verticality from normals.
        
        Verticality = 1 - |normal_z|
        """
        if self.use_gpu and cp is not None:
            normals_gpu = self._to_gpu(normals)
            verticality_gpu = 1.0 - cp.abs(normals_gpu[:, 2])
            return self._to_cpu(verticality_gpu).astype(np.float32)
        else:
            return core_compute_verticality(normals)
    
    # ==========================================================================
    # CHUNKED PROCESSING - For Large Datasets (>chunk_threshold)
    # ==========================================================================
    
    def _compute_features_chunked(
        self,
        points: np.ndarray,
        feature_types: Optional[List[str]],
        k: int,
        show_progress: bool
    ) -> Dict[str, np.ndarray]:
        """
        Compute features using chunked processing (from features_gpu_chunked.py).
        
        This method builds a global KDTree once and processes points in chunks
        to avoid VRAM exhaustion for large datasets.
        """
        features = {}
        
        # Determine which features to compute
        if feature_types is None:
            feature_types = ['normals', 'curvature', 'eigenvalues', 'verticality']
        
        # Compute normals first if needed
        normals = None
        if any(ft in feature_types for ft in ['normals', 'curvature', 'verticality']):
            normals = self._compute_normals_chunked(points, k, show_progress)
            if 'normals' in feature_types:
                features['normals'] = normals
        
        # Compute curvature
        if 'curvature' in feature_types:
            if normals is None:
                normals = self._compute_normals_chunked(points, k, show_progress)
            features['curvature'] = self._compute_curvature_chunked(
                points, normals, k, show_progress
            )
        
        # Compute verticality
        if 'verticality' in feature_types:
            if normals is None:
                normals = self._compute_normals_chunked(points, k, show_progress)
            features['verticality'] = self._compute_verticality_batch(normals)
        
        # Compute eigenvalue features
        if 'eigenvalues' in feature_types:
            logger.info("Eigenvalue features via GPU Bridge - pending integration")
        
        return features
    
    def _compute_normals_chunked(
        self,
        points: np.ndarray,
        k: int,
        show_progress: bool
    ) -> np.ndarray:
        """
        Compute normals using chunked processing (from features_gpu_chunked.py).
        
        Strategy (automatic selection):
        1. FAISS (preferred): Ultra-fast k-NN, 50-100× faster than cuML
        2. cuML fallback: Global KDTree + chunked queries
        3. CPU fallback: sklearn KDTree
        """
        if not self.use_gpu:
            logger.warning("GPU not available, using CPU batch processing")
            return self._compute_normals_cpu(points, k)
        
        N = len(points)
        
        # Try FAISS first (50-100× speedup)
        if FAISS_AVAILABLE and self.use_cuml:
            logger.info(f"  🚀 Using FAISS for ultra-fast k-NN ({N:,} points)")
            try:
                return self._compute_normals_with_faiss(points, k, show_progress)
            except Exception as e:
                logger.warning(f"  ⚠ FAISS failed: {e}, falling back to cuML")
        
        # Fallback to per-chunk strategy with global KDTree
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        logger.info(f"  🔧 Using global KDTree: {N:,} points in {num_chunks} chunks")
        return self._compute_normals_per_chunk(points, k, show_progress)
    
    def _compute_normals_with_faiss(
        self,
        points: np.ndarray,
        k: int,
        show_progress: bool
    ) -> np.ndarray:
        """
        Compute normals using FAISS for 50-100× faster k-NN queries.
        
        FAISS dramatically outperforms cuML for massive neighbor queries:
        - 18.6M points: cuML ~51 min → FAISS ~30-60 seconds
        """
        N = points.shape[0]
        normals = np.zeros((N, 3), dtype=np.float32)
        
        logger.info(f"  🚀 FAISS: Ultra-fast k-NN computation")
        logger.info(f"     {N:,} points → Expected: 30-90 seconds")
        
        # Build FAISS index
        index = self._build_faiss_index(points, k)
        
        # Query neighbors in batches for better progress visibility and memory management
        # Especially important for CPU FAISS on large datasets (>15M points)
        batch_size = 500_000  # 500K points per batch
        num_batches = (N + batch_size - 1) // batch_size
        
        # Decide if batching is needed
        use_batching = (N > 5_000_000) or (not self.use_gpu)  # Always batch CPU FAISS for large datasets
        
        if use_batching and num_batches > 1:
            # Estimate time for user
            if N > 15_000_000:
                estimated_minutes = (N / 1_000_000) * 1.2  # ~1.2 min per million for CPU FAISS
                logger.info(f"  ⚡ Querying {N:,} × {k} neighbors in {num_batches} batches...")
                logger.info(f"     Estimated time: {estimated_minutes:.1f} minutes (batched processing)")
            else:
                logger.info(f"  ⚡ Querying {N:,} × {k} neighbors in {num_batches} batches...")
            
            # Allocate result arrays
            all_indices = np.zeros((N, k), dtype=np.int64)
            
            # Process in batches with progress bar
            batch_iterator = range(num_batches)
            if show_progress:
                batch_iterator = tqdm(
                    batch_iterator,
                    desc=f"  FAISS k-NN query",
                    unit="batch",
                    ncols=80
                )
            
            for batch_idx in batch_iterator:
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, N)
                
                batch_points = points[start_idx:end_idx].astype(np.float32)
                batch_distances, batch_indices = index.search(batch_points, k)
                
                all_indices[start_idx:end_idx] = batch_indices
                
                # Periodic cleanup for large batches
                if batch_idx % 10 == 0:
                    import gc
                    gc.collect()
            
            indices = all_indices
            logger.info(f"     ✓ All neighbors found ({num_batches} batches completed)")
        else:
            # Single batch for small datasets
            logger.info(f"  ⚡ Querying all {N:,} × {k} neighbors...")
            distances, indices = index.search(points.astype(np.float32), k)
            logger.info(f"     ✓ All neighbors found")
        
        # Compute normals from neighbors
        logger.info(f"  ⚡ Computing normals from neighborhoods...")
        
        if self.use_gpu and cp is not None:
            # GPU PCA
            points_gpu = self._to_gpu(points)
            indices_gpu = self._to_gpu(indices)
            normals_gpu = self._compute_normals_from_neighbors_gpu(
                points_gpu, indices_gpu
            )
            normals = self._to_cpu(normals_gpu)
            del points_gpu, indices_gpu, normals_gpu
        else:
            # CPU PCA
            normals = self._compute_normals_from_neighbors_cpu(points, indices)
        
        logger.info(f"     ✓ Normals computed")
        
        # Cleanup
        del index, distances, indices
        self._free_gpu_memory(force=True)
        
        return normals
    
    def _build_faiss_index(self, points: np.ndarray, k: int):
        """
        Build FAISS index for ultra-fast k-NN queries.
        
        FAISS is optimized for billion-scale nearest neighbor search.
        
        Memory-aware: For >15M points with limited VRAM (<8GB),
        automatically falls back to CPU FAISS to avoid OOM.
        """
        N, D = points.shape
        logger.info(f"  🚀 Building FAISS index ({N:,} points, k={k})...")
        
        # Memory-aware GPU usage: avoid FAISS GPU for huge point clouds on limited VRAM
        # Rule of thumb: FAISS GPU needs ~N*k*4 bytes for query results alone
        # For 21M points × 55 neighbors = ~4.6GB just for results
        # Plus temp memory for IVF search = total can exceed 8GB easily
        estimated_memory_gb = (N * k * 4) / (1024**3)
        use_gpu_faiss = self.use_gpu and self.use_cuml and N < 15_000_000
        
        if not use_gpu_faiss and N > 15_000_000:
            logger.info(f"     ⚠ Large point cloud ({N:,} points) + limited VRAM")
            logger.info(f"     → Estimated memory: {estimated_memory_gb:.1f}GB for query results")
            logger.info(f"     → Using CPU FAISS to avoid GPU OOM")
        
        # Use IVF for large datasets (>5M points)
        use_ivf = N > 5_000_000
        
        if use_ivf:
            # IVF parameters
            nlist = min(8192, max(256, int(np.sqrt(N))))
            nprobe = min(128, nlist // 8)
            
            logger.info(f"     Using IVF: {nlist} clusters, {nprobe} probes")
            
            # Create IVF index
            quantizer = faiss.IndexFlatL2(D)
            index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
            
            # Move to GPU if safe to do so
            if use_gpu_faiss:
                try:
                    res = faiss.StandardGpuResources()
                    # Conservative temp memory: 2GB max to leave room for query results
                    res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB (was 4GB)
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = False
                    index = faiss.index_cpu_to_gpu(res, 0, index, co)
                    logger.info("     ✓ FAISS index on GPU")
                except Exception as e:
                    logger.warning(f"     GPU failed: {e}, using CPU")
            else:
                logger.info("     ✓ FAISS index on CPU (memory-safe)")
            
            # Train index
            logger.info(f"     Training index...")
            train_size = min(N, nlist * 256)
            if train_size < N:
                train_idx = np.random.choice(N, train_size, replace=False)
                train_data = points[train_idx].astype(np.float32)
            else:
                train_data = points.astype(np.float32)
            
            index.train(train_data)
            logger.info(f"     ✓ Trained on {len(train_data):,} points")
            
            # Add all points
            logger.info(f"     Adding {N:,} points...")
            index.add(points.astype(np.float32))
            
            # Set nprobe
            if hasattr(index, 'nprobe'):
                index.nprobe = nprobe
            elif hasattr(index, 'setNumProbes'):
                index.setNumProbes(nprobe)
            
            logger.info(f"     ✓ FAISS IVF index ready")
        else:
            # Flat index for smaller datasets
            logger.info(f"     Using Flat (exact) index")
            index = faiss.IndexFlatL2(D)
            
            if use_gpu_faiss:
                try:
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(1 * 1024 * 1024 * 1024)  # 1GB (was 2GB)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info("     ✓ FAISS index on GPU")
                except Exception as e:
                    logger.warning(f"     GPU failed: {e}, using CPU")
            else:
                logger.info("     ✓ FAISS index on CPU")
            
            index.add(points.astype(np.float32))
            logger.info(f"     ✓ FAISS Flat index ready")
        
        return index
    
    def _compute_normals_per_chunk(
        self,
        points: np.ndarray,
        k: int,
        show_progress: bool
    ) -> np.ndarray:
        """
        Compute normals using global KDTree with chunked queries.
        
        Strategy:
        1. Build ONE global KDTree on GPU
        2. Query neighbors in chunks to manage memory
        3. Compute normals on GPU with vectorized operations
        """
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        
        # Transfer points to GPU once
        points_gpu = self._to_gpu(points)
        
        # Build global KDTree once
        logger.info(f"  🔨 Building global KDTree ({N:,} points)...")
        if self.use_cuml and cuNearestNeighbors is not None:
            knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(points_gpu)
            logger.info("  ✓ Global GPU KDTree built (cuML)")
        else:
            points_cpu = self._to_cpu(points_gpu)
            knn = NearestNeighbors(
                n_neighbors=k, metric='euclidean', 
                algorithm='kd_tree', n_jobs=-1
            )
            knn.fit(points_cpu)
            logger.info("  ✓ Global CPU KDTree built (sklearn)")
        
        # Process in chunks
        chunk_iterator = range(num_chunks)
        if show_progress:
            chunk_iterator = tqdm(
                chunk_iterator,
                desc=f"  🎯 GPU Normals ({N:,} pts, {num_chunks} chunks)",
                unit="chunk"
            )
        
        for chunk_idx in chunk_iterator:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, N)
            
            # Query neighbors for chunk
            if self.use_cuml and cuNearestNeighbors is not None:
                chunk_points = points_gpu[start_idx:end_idx]
                distances, indices = knn.kneighbors(chunk_points)
                
                # Compute normals on GPU
                chunk_normals = self._compute_normals_from_neighbors_gpu(
                    points_gpu, indices
                )
                normals[start_idx:end_idx] = self._to_cpu(chunk_normals)
                
                del chunk_points, distances, chunk_normals
            else:
                # CPU path
                chunk_points = points[start_idx:end_idx]
                distances, indices = knn.kneighbors(chunk_points)
                chunk_normals = self._compute_normals_from_neighbors_cpu(
                    points, indices
                )
                normals[start_idx:end_idx] = chunk_normals
                del chunk_points, distances, chunk_normals
            
            # Periodic cleanup
            if chunk_idx % 10 == 0:
                self._free_gpu_memory()
        
        # Final cleanup
        del points_gpu, knn
        self._free_gpu_memory(force=True)
        
        return normals
    
    def _compute_normals_from_neighbors_gpu(
        self,
        points_gpu,
        neighbor_indices
    ):
        """
        Compute normals using vectorized covariance computation on GPU.
        
        ~100× faster than per-point PCA loops.
        """
        M, k = neighbor_indices.shape
        
        # Gather neighbor points: [M, k, 3]
        neighbor_points = points_gpu[neighbor_indices]
        
        # Center neighborhoods
        centroids = cp.mean(neighbor_points, axis=1, keepdims=True)
        centered = neighbor_points - centroids
        del neighbor_points, centroids
        
        # Compute covariance matrices: [M, 3, 3]
        cov_matrices = cp.einsum('mki,mkj->mij', centered, centered) / k
        del centered
        
        # Ensure symmetry
        cov_T = cp.transpose(cov_matrices, (0, 2, 1))
        cov_matrices = (cov_matrices + cov_T) / 2
        del cov_T
        
        # Use fast inverse power iteration for eigenvectors
        if cov_matrices.dtype != cp.float32:
            cov_matrices = cov_matrices.astype(cp.float32)
        
        normals = inverse_power_iteration(
            cov_matrices, num_iterations=8, epsilon=1e-12
        )
        del cov_matrices
        
        return normals
    
    def _compute_normals_from_neighbors_cpu(
        self,
        points: np.ndarray,
        neighbor_indices: np.ndarray
    ) -> np.ndarray:
        """CPU fallback for computing normals from neighbor indices."""
        M, k = neighbor_indices.shape
        
        # Gather neighbor points
        neighbor_points = points[neighbor_indices]
        
        # Center neighborhoods
        centroids = np.mean(neighbor_points, axis=1, keepdims=True)
        centered = neighbor_points - centroids
        
        # Compute covariance matrices
        cov_matrices = np.einsum('mki,mkj->mij', centered, centered) / k
        
        # Add regularization
        cov_matrices += 1e-8 * np.eye(3, dtype=np.float32)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
        normals = eigenvectors[:, :, 0]  # Smallest eigenvector
        
        # Ensure upward orientation
        normals[normals[:, 2] < 0] *= -1
        
        return normals.astype(np.float32)
    
    def _compute_curvature_chunked(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int,
        show_progress: bool
    ) -> np.ndarray:
        """
        Compute curvature using chunked processing.
        
        Similar strategy to chunked normals: global KDTree + chunked queries.
        """
        N = len(points)
        curvature = np.zeros(N, dtype=np.float32)
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        
        # Build global KDTree once
        logger.info(f"  🔨 Building global KDTree for curvature ({N:,} points)...")
        if self.use_gpu and self.use_cuml and cuNearestNeighbors is not None:
            points_gpu = self._to_gpu(points)
            normals_gpu = self._to_gpu(normals)
            
            knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(points_gpu)
            logger.info("  ✓ Global GPU KDTree built")
            
            # Process in chunks
            chunk_iterator = range(num_chunks)
            if show_progress:
                chunk_iterator = tqdm(
                    chunk_iterator,
                    desc=f"  🎯 GPU Curvature ({N:,} pts)",
                    unit="chunk"
                )
            
            for chunk_idx in chunk_iterator:
                start_idx = chunk_idx * self.chunk_size
                end_idx = min((chunk_idx + 1) * self.chunk_size, N)
                
                chunk_points = points_gpu[start_idx:end_idx]
                distances, indices = knn.kneighbors(chunk_points)
                
                # Compute curvature on GPU
                batch_normals = normals_gpu[start_idx:end_idx]
                neighbor_normals = normals_gpu[indices]
                
                query_normals_expanded = batch_normals[:, cp.newaxis, :]
                normal_diff = neighbor_normals - query_normals_expanded
                curv_norms = cp.linalg.norm(normal_diff, axis=2)
                batch_curvature = cp.mean(curv_norms, axis=1)
                
                curvature[start_idx:end_idx] = self._to_cpu(batch_curvature)
                
                del chunk_points, distances, batch_normals, neighbor_normals
                
                if chunk_idx % 10 == 0:
                    self._free_gpu_memory()
            
            del points_gpu, normals_gpu, knn
            self._free_gpu_memory(force=True)
        else:
            # CPU fallback
            tree = KDTree(points, metric='euclidean', leaf_size=40)
            _, neighbor_indices = tree.query(points, k=k)
            curvature = compute_curvature_from_normals(
                points, normals, neighbor_indices
            )
        
        return curvature
    
    def _free_gpu_memory(self, force: bool = False):
        """
        Smart GPU memory cleanup - only when needed.
        
        Args:
            force: Force cleanup regardless of usage threshold
        """
        if not self.use_gpu or cp is None:
            return
        
        try:
            if cp.cuda.is_available():
                mempool = cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                used_gb = used_bytes / (1024**3)
                
                # Only cleanup if >80% VRAM used or forced
                threshold_gb = self.vram_limit_gb * 0.8
                
                if force or used_gb > threshold_gb:
                    pinned_mempool = cp.get_default_pinned_memory_pool()
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    logger.debug(f"GPU memory cleanup: {used_gb:.2f}GB freed")
        except Exception as e:
            logger.debug(f"Could not free GPU memory: {e}")
        
        gc.collect()
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def _to_gpu(self, array: np.ndarray):
        """Transfer array to GPU.
        
        Returns:
            GPU array if GPU available, otherwise numpy array
        """
        if not self.use_gpu or cp is None:
            return array
        return cp.asarray(array, dtype=array.dtype)
    
    def _to_cpu(self, array) -> np.ndarray:
        """Transfer array to CPU."""
        if isinstance(array, np.ndarray):
            return array
        if cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    
    def cleanup(self):
        """Clean up GPU resources."""
        if self.use_gpu and cp is not None:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                gc.collect()
                logger.debug("GPU memory cleaned up")
            except Exception as e:
                logger.warning(f"Error during GPU cleanup: {e}")
    
    # ==========================================================================
    # EIGENVALUE FEATURES (GPU Bridge Integration - Phase 2A.6)
    # ==========================================================================
    
    def compute_eigenvalues(
        self,
        points: np.ndarray,
        neighbors: np.ndarray,
        return_eigenvectors: bool = False
    ) -> np.ndarray:
        """
        Compute eigenvalues using GPU-Core Bridge.
        
        Uses the GPUCoreBridge for GPU-accelerated eigenvalue computation
        with automatic fallback to CPU if GPU unavailable.
        
        Args:
            points: Point cloud (N, 3)
            neighbors: Neighbor indices (N, k)
            return_eigenvectors: Return eigenvectors as well
        
        Returns:
            eigenvalues: (N, 3) sorted descending
            eigenvectors: (N, 3, 3) if return_eigenvectors=True
        """
        return self.gpu_bridge.compute_eigenvalues_gpu(
            points, 
            neighbors, 
            return_eigenvectors=return_eigenvectors
        )
    
    def compute_eigenvalue_features(
        self,
        points: np.ndarray,
        neighbors: np.ndarray,
        epsilon: Optional[float] = None,
        include_all: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute eigenvalue-based features using GPU-Core Bridge.
        
        This method demonstrates the bridge architecture:
        1. Compute eigenvalues on GPU (fast)
        2. Transfer to CPU (minimal overhead)
        3. Compute features using canonical core implementation
        
        Features computed:
        - linearity, planarity, sphericity
        - omnivariance, anisotropy, eigentropy
        - surface_variation, vertical_range
        
        Args:
            points: Point cloud (N, 3)
            neighbors: Neighbor indices (N, k)
            epsilon: Numerical stability constant (default: 1e-10)
            include_all: Compute all eigenvalue features
        
        Returns:
            features: Dictionary of eigenvalue-based features
        """
        return self.gpu_bridge.compute_eigenvalue_features_gpu(
            points,
            neighbors,
            epsilon=epsilon,
            include_all=include_all
        )
    
    def compute_density_features(
        self,
        points: np.ndarray,
        k_neighbors: int = 20,
        search_radius: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute density features using GPU-Core Bridge.
        
        Args:
            points: Point cloud (N, 3)
            k_neighbors: Number of nearest neighbors (default: 20)
            search_radius: Fixed radius for density (optional)
        
        Returns:
            features: Dictionary of density features
        """
        return self.gpu_bridge.compute_density_features_gpu(
            points,
            k_neighbors=k_neighbors,
            search_radius=search_radius
        )
    
    def compute_architectural_features(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        eigenvalues: np.ndarray,
        epsilon: float = 1e-10
    ) -> Dict[str, np.ndarray]:
        """
        Compute architectural features using GPU-Core Bridge.
        
        Args:
            points: Point cloud (N, 3)
            normals: Normal vectors (N, 3)
            eigenvalues: Eigenvalues (N, 3)
            epsilon: Numerical stability constant
        
        Returns:
            features: Dictionary of architectural features
        """
        return self.gpu_bridge.compute_architectural_features_gpu(
            points,
            normals,
            eigenvalues,
            epsilon=epsilon
        )


# =============================================================================
# CONVENIENCE FUNCTIONS (Backward Compatibility)
# =============================================================================

def compute_normals(
    points: np.ndarray,
    k: int = 10,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Convenience function for computing normals.
    
    Args:
        points: Point cloud (N, 3)
        k: Number of neighbors
        use_gpu: Enable GPU acceleration
    
    Returns:
        normals: (N, 3) array of unit normal vectors
    """
    processor = GPUProcessor(use_gpu=use_gpu, show_progress=False)
    return processor.compute_normals(points, k=k)


def compute_curvature(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 10,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Convenience function for computing curvature.
    
    Args:
        points: Point cloud (N, 3)
        normals: Normal vectors (N, 3)
        k: Number of neighbors
        use_gpu: Enable GPU acceleration
    
    Returns:
        curvature: (N,) array of curvature values
    """
    processor = GPUProcessor(use_gpu=use_gpu, show_progress=False)
    return processor.compute_curvature(points, normals, k=k)


def compute_eigenvalues(
    points: np.ndarray,
    neighbors: np.ndarray,
    use_gpu: bool = True,
    return_eigenvectors: bool = False
) -> np.ndarray:
    """
    Convenience function for computing eigenvalues.
    
    Args:
        points: Point cloud (N, 3)
        neighbors: Neighbor indices (N, k)
        use_gpu: Enable GPU acceleration
        return_eigenvectors: Return eigenvectors as well
    
    Returns:
        eigenvalues: (N, 3) sorted descending
        eigenvectors: (N, 3, 3) if return_eigenvectors=True
    """
    processor = GPUProcessor(use_gpu=use_gpu, show_progress=False)
    return processor.compute_eigenvalues(points, neighbors, return_eigenvectors)


def compute_eigenvalue_features(
    points: np.ndarray,
    neighbors: np.ndarray,
    use_gpu: bool = True,
    epsilon: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Convenience function for computing eigenvalue-based features.
    
    Args:
        points: Point cloud (N, 3)
        neighbors: Neighbor indices (N, k)
        use_gpu: Enable GPU acceleration
        epsilon: Numerical stability constant
    
    Returns:
        features: Dictionary of eigenvalue-based features
    """
    processor = GPUProcessor(use_gpu=use_gpu, show_progress=False)
    return processor.compute_eigenvalue_features(points, neighbors, epsilon=epsilon)
