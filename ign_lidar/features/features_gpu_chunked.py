"""
GPU-Accelerated Feature Computation with Chunked Processing Support
Enables GPU acceleration for large point clouds (>10M points)
and augmented data. Version: 1.7.4
"""

from typing import Dict, Tuple, Optional
import numpy as np
import logging
import gc
from tqdm import tqdm

logger = logging.getLogger(__name__)

# GPU imports with fallback
GPU_AVAILABLE = False
CUML_AVAILABLE = False

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("✓ CuPy available - GPU enabled")
except ImportError:
    logger.warning("⚠ CuPy not available - GPU chunking disabled")
    cp = None

try:
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    from cuml.decomposition import PCA as cuPCA
    CUML_AVAILABLE = True
    logger.info("✓ RAPIDS cuML available - GPU algorithms enabled")
except ImportError:
    logger.warning("⚠ RAPIDS cuML not available - using sklearn fallback")
    cuNearestNeighbors = None
    cuPCA = None

# CPU fallback imports
from sklearn.neighbors import NearestNeighbors

# Import core feature implementations
from ..features.core import (
    compute_eigenvalue_features as core_compute_eigenvalue_features,
    compute_density_features as core_compute_density_features,
)


class GPUChunkedFeatureComputer:
    """
    GPU feature computation with chunked processing for large point clouds.
    
    Key features:
    - Processes large datasets in chunks to avoid VRAM exhaustion
    - Builds global KDTree once, queries per chunk
    - Automatic VRAM management with configurable limits
    - 10-15x speedup over CPU while handling unlimited point counts
    
    Example:
        >>> computer = GPUChunkedFeatureComputer(
        ...     chunk_size=5_000_000,
        ...     vram_limit_gb=8.0
        ... )
        >>> normals = computer.compute_normals_chunked(points, k=10)
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        vram_limit_gb: Optional[float] = None,
        use_gpu: bool = True,
        show_progress: bool = True,
        auto_optimize: bool = True,
        use_cuda_streams: bool = True
    ):
        """
        Initialize GPU chunked feature computer.
        
        INTELLIGENT AUTO-OPTIMIZATION: If chunk_size=None, automatically
        determines optimal size based on available VRAM.
        
        Args:
            chunk_size: Points per chunk (None = auto-optimize)
            vram_limit_gb: Max VRAM usage (None = auto-detect)
            use_gpu: Enable GPU acceleration if available
            show_progress: Show progress bars during processing
            auto_optimize: Enable intelligent parameter optimization
            use_cuda_streams: Enable CUDA streams for overlapped processing
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_cuml = CUML_AVAILABLE
        self.show_progress = show_progress
        self.auto_optimize = auto_optimize
        self.use_cuda_streams = use_cuda_streams and GPU_AVAILABLE
        
        # Initialize CUDA streams manager
        self.stream_manager = None
        if self.use_cuda_streams and self.use_gpu:
            try:
                from ..optimization.cuda_streams import create_stream_manager
                self.stream_manager = create_stream_manager(
                    num_streams=3,  # Upload, compute, download
                    enable_pinned=True
                )
                logger.info("✓ CUDA streams enabled for overlapped processing")
            except Exception as e:
                logger.warning(f"⚠ CUDA streams initialization failed: {e}")
                self.use_cuda_streams = False
        
        # Initialize CUDA context early if GPU is requested
        if self.use_gpu and cp is not None:
            try:
                cp.cuda.Device(0).use()
            except Exception as e:
                logger.warning(
                    f"⚠ GPU initialization failed ({e.__class__.__name__}: "
                    f"{e}), falling back to CPU"
                )
                self.use_gpu = False
                self.use_cuml = False
        
        # INTELLIGENT AUTO-OPTIMIZATION
        if self.use_gpu and auto_optimize:
            from ..core.memory import AdaptiveMemoryManager
            self.memory_manager = AdaptiveMemoryManager()
            
            # Auto-detect VRAM
            if vram_limit_gb is None:
                status = self.memory_manager.get_current_memory_status()
                self.vram_limit_gb = status[2] if len(status) > 2 else 8.0  # Default fallback
            else:
                self.vram_limit_gb = vram_limit_gb
            
            # Auto-optimize chunk size for reclassification workflows
            if chunk_size is None:
                # Detect if we're in reclassification mode by checking for minimal feature requirements
                feature_mode = 'minimal'  # Default for reclassification
                self.chunk_size = (
                    self.memory_manager.calculate_optimal_gpu_chunk_size(
                        num_points=10_000_000,  # Estimate for sizing
                        vram_free_gb=self.vram_limit_gb,
                        feature_mode=feature_mode
                    )
                )
                if self.chunk_size == 0:
                    # Not enough VRAM, fallback to CPU
                    logger.warning("⚠️ Insufficient VRAM for GPU processing, falling back to CPU")
                    self.use_gpu = False
                    self.chunk_size = 2_500_000
            else:
                self.chunk_size = chunk_size
        else:
            # Manual configuration
            self.chunk_size = chunk_size if chunk_size else 2_500_000
            self.vram_limit_gb = vram_limit_gb if vram_limit_gb else 8.0
            self.memory_manager = None
        
        if self.use_gpu:
            # Get available VRAM (CUDA already initialized above)
            if cp is not None:
                try:
                    # Use runtime API instead of deprecated device.mem_info
                    _, total_vram = cp.cuda.runtime.memGetInfo()
                    total_vram = total_vram / (1024**3)
                    if self.use_cuml:
                        logger.info(
                            f"🚀 GPU chunked mode enabled with RAPIDS cuML "
                            f"(chunk_size={self.chunk_size:,}, "
                            f"VRAM limit={self.vram_limit_gb:.1f}GB / "
                            f"{total_vram:.1f}GB total)"
                        )
                    else:
                        logger.info(
                            f"🚀 GPU chunked mode enabled with CuPy + sklearn "
                            f"(chunk_size={self.chunk_size:,}, "
                            f"VRAM limit={self.vram_limit_gb:.1f}GB / "
                            f"{total_vram:.1f}GB total)"
                        )
                        logger.info(
                            "   ℹ️ Install RAPIDS cuML for full GPU acceleration"
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠ Failed to get VRAM info ({e.__class__.__name__}: "
                        f"{e}), using default limits"
                    )
        
        if not self.use_gpu:
            logger.info("💻 CPU mode - GPU not available or disabled")
    
    def _to_gpu(self, array: np.ndarray, stream_idx: Optional[int] = None) -> 'cp.ndarray':
        """
        Transfer array to GPU memory (optionally using CUDA streams).
        
        Args:
            array: NumPy array to transfer
            stream_idx: Optional stream index for async transfer
            
        Returns:
            CuPy array on GPU
        """
        if self.use_gpu and cp is not None:
            # Use stream manager for async transfer if available
            if stream_idx is not None and self.stream_manager and self.use_cuda_streams:
                return self.stream_manager.async_upload(
                    array, 
                    stream_idx=stream_idx,
                    use_pinned=True
                )
            # Standard synchronous transfer
            return cp.asarray(array, dtype=cp.float32)
        return array
    
    def _to_cpu(self, array, stream_idx: Optional[int] = None) -> np.ndarray:
        """
        Transfer array to CPU memory (optionally using CUDA streams).
        
        Args:
            array: CuPy array to transfer
            stream_idx: Optional stream index for async transfer
            
        Returns:
            NumPy array on CPU
        """
        if self.use_gpu and cp is not None and isinstance(array, cp.ndarray):
            # Use stream manager for async transfer if available
            if stream_idx is not None and self.stream_manager and self.use_cuda_streams:
                return self.stream_manager.async_download(
                    array,
                    stream_idx=stream_idx,
                    use_pinned=True
                )
            # Standard synchronous transfer
            return cp.asnumpy(array)
        return np.asarray(array)
    
    def optimize_for_reclassification(
        self,
        num_points: int,
        available_vram_gb: Optional[float] = None
    ):
        """
        Optimize GPU chunked computer settings specifically for reclassification workflows.
        
        This method adjusts internal parameters for maximum performance when processing
        large point clouds for reclassification tasks.
        
        Args:
            num_points: Total number of points to be processed
            available_vram_gb: Available VRAM in GB (auto-detect if None)
        """
        # Auto-detect VRAM if not provided
        if available_vram_gb is None and self.use_gpu and cp is not None:
            try:
                free_vram, total_vram = cp.cuda.runtime.memGetInfo()
                available_vram_gb = free_vram / (1024**3)
            except Exception:
                available_vram_gb = 8.0  # Conservative default
        elif available_vram_gb is None:
            available_vram_gb = 8.0  # CPU fallback
        
        # Update chunk size for optimal reclassification performance
        if self.memory_manager and self.auto_optimize:
            optimal_chunk = self.memory_manager.calculate_optimal_gpu_chunk_size(
                num_points=num_points,
                vram_free_gb=available_vram_gb,
                feature_mode='minimal'  # Reclassification uses minimal features
            )
            
            if optimal_chunk > 0 and optimal_chunk != self.chunk_size:
                logger.info(
                    f"🔧 Optimizing for reclassification: "
                    f"chunk_size {self.chunk_size:,} → {optimal_chunk:,}"
                )
                self.chunk_size = optimal_chunk
                self.vram_limit_gb = available_vram_gb
        
        # Enable aggressive memory optimization for reclassification
        self._enable_reclassification_optimizations()
    
    def _enable_reclassification_optimizations(self):
        """Enable optimizations specific to reclassification workflows."""
        # These optimizations prioritize speed over feature completeness
        self._reclassification_mode = True
        self._reduced_feature_set = True
        self._aggressive_memory_cleanup = True
        
        logger.info("✓ Reclassification optimizations enabled")

    def _free_gpu_memory(self, force: bool = False):
        """
        Smart GPU memory cleanup - only when needed to avoid overhead.
        
        Args:
            force: Force cleanup regardless of usage threshold
        """
        if self.use_gpu and cp is not None:
            try:
                # Check if CUDA is actually available before trying to free memory
                if cp.cuda.is_available():
                    mempool = cp.get_default_memory_pool()
                    used_bytes = mempool.used_bytes()
                    used_gb = used_bytes / (1024**3)
                    
                    # Only cleanup if >80% VRAM used or forced
                    threshold_gb = self.vram_limit_gb * 0.8 if self.vram_limit_gb else 10.0
                    
                    if force or used_gb > threshold_gb:
                        pinned_mempool = cp.get_default_pinned_memory_pool()
                        mempool.free_all_blocks()
                        pinned_mempool.free_all_blocks()
                        logger.debug(f"GPU memory cleanup: {used_gb:.2f}GB freed")
                    else:
                        logger.debug(f"GPU memory OK: {used_gb:.2f}GB < {threshold_gb:.2f}GB threshold")
            except Exception as e:
                # Catch all exceptions including CUDA runtime errors
                logger.debug(f"Could not free GPU memory: {e}")
                pass
            gc.collect()
    
    def compute_normals_chunked(
        self,
        points: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """
        Compute surface normals using GPU with chunked processing.
        
        Strategy (optimized for sklearn fallback):
        1. Process in chunks to avoid building massive global KDTree
        2. Build local KDTree per chunk with overlap for accuracy
        3. Compute PCA on GPU/CPU depending on availability
        4. Much faster than global tree for sklearn fallback
        
        Args:
            points: [N, 3] point coordinates
            k: number of neighbors for PCA
            
        Returns:
            normals: [N, 3] normalized surface normals
        """
        if not self.use_gpu:
            logger.warning("GPU not available, falling back to CPU")
            from .features_gpu import GPUFeatureComputer
            computer = GPUFeatureComputer(use_gpu=False)
            return computer.compute_normals(points, k=k)
        
        N = len(points)
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        
        # ALWAYS use per-chunk strategy (much faster for large datasets)
        # Building a global KDTree for 10M+ points is too slow even on GPU
        # Per-chunk strategy: ~10-20x faster, similar accuracy with overlap
        logger.info(
            f"Computing normals with per-chunk KDTree: "
            f"{N:,} points in {num_chunks} chunks"
        )
        return self._compute_normals_per_chunk(points, k)
        
        # Original global KDTree strategy (DEPRECATED - too slow)
        # Keeping code for reference but never used
        if False:  # Disabled
            logger.info(
                f"Computing normals with global KDTree (cuML): "
                f"{N:,} points in {num_chunks} chunks"
            )
        
        normals = np.zeros((N, 3), dtype=np.float32)
        
        try:
            # Transfer entire point cloud to GPU for KDTree
            points_gpu = self._to_gpu(points)
            
            # Build global KDTree on GPU with cuML
            logger.info("  Building global KDTree on GPU (cuML)...")
            knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(points_gpu)
            
            # Process in chunks with progress bar
            num_chunks = (N + self.chunk_size - 1) // self.chunk_size
            
            chunk_iterator = range(num_chunks)
            if self.show_progress:
                bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} chunks '
                           '[{elapsed}<{remaining}, {rate_fmt}]')
                chunk_iterator = tqdm(
                    chunk_iterator,
                    desc=f"  🎯 GPU Normals [cuML] ({N:,} pts)",
                    unit="chunk",
                    total=num_chunks,
                    bar_format=bar_fmt
                )
            
            for chunk_idx in chunk_iterator:
                start_idx = chunk_idx * self.chunk_size
                end_idx = min((chunk_idx + 1) * self.chunk_size, N)
                
                # Query KNN for chunk
                if self.use_cuml and cuNearestNeighbors is not None:
                    # GPU path: query on GPU
                    chunk_points_gpu = points_gpu[start_idx:end_idx]
                    distances, indices = knn.kneighbors(chunk_points_gpu)
                    
                    # Compute normals for chunk using PCA on GPU
                    chunk_normals_gpu = (
                        self._compute_normals_from_neighbors_gpu(
                            points_gpu, indices
                        )
                    )
                    
                    # OPTIMIZATION: Keep results on GPU, batch transfer later
                    normals[start_idx:end_idx] = (
                        self._to_cpu(chunk_normals_gpu)
                    )
                    
                    # Free GPU memory for chunk
                    del chunk_points_gpu, distances, chunk_normals_gpu
                    # OPTIMIZED: Less frequent cleanup (only when VRAM high)
                    if chunk_idx % 10 == 0:  # Was every 5, now every 10
                        self._free_gpu_memory()  # Smart cleanup (only if >80% VRAM)
                else:
                    # Hybrid path: KNN on CPU, PCA on GPU
                    chunk_points_cpu = self._to_cpu(
                        points_gpu[start_idx:end_idx]
                    )
                    distances, indices = knn.kneighbors(chunk_points_cpu)
                    
                    # Compute normals with GPU PCA on neighbors
                    idx_array = cp.asarray(indices) if cp else indices
                    chunk_normals_gpu = (
                        self._compute_normals_from_neighbors_gpu(
                            points_gpu, idx_array
                        )
                    )
                    
                    # OPTIMIZATION: Keep results on GPU, batch transfer later
                    normals[start_idx:end_idx] = (
                        self._to_cpu(chunk_normals_gpu)
                    )
                    
                    # OPTIMIZED: Less frequent cleanup
                    if chunk_idx % 10 == 0:  # Was every 5, now every 10
                        self._free_gpu_memory()  # Smart cleanup (only if >80% VRAM)
            
            # Final cleanup - force cleanup at end
            del points_gpu, knn
            self._free_gpu_memory(force=True)  # Force final cleanup
            
            logger.info("  ✓ Normals computation complete")
            return normals
            
        except Exception as e:
            logger.error(f"GPU chunked computation failed: {e}")
            logger.warning("Falling back to CPU...")
            self._free_gpu_memory(force=True)  # Force cleanup on error
            
            from .features_gpu import GPUFeatureComputer
            computer = GPUFeatureComputer(use_gpu=False)
            return computer.compute_normals(points, k=k)
    
    def _log_gpu_memory(self, stage: str = ""):
        """Log current GPU memory usage."""
        if self.use_gpu and cp is not None:
            mempool = cp.get_default_memory_pool()
            used = mempool.used_bytes() / (1024**3)  # GB
            total = mempool.total_bytes() / (1024**3)  # GB
            logger.debug(
                f"  GPU Memory {stage}: {used:.2f}GB used, "
                f"{total:.2f}GB allocated"
            )
    
    def _compute_normals_per_chunk(
        self,
        points: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """
        OPTIMIZED: Compute normals using SINGLE GLOBAL KDTree with chunked queries.
        
        Key optimization: Build KDTree ONCE, query in chunks
        - 10-100x faster than building KDTree per chunk
        - Lower memory usage (no overlap needed)
        - Better accuracy (global neighbor search)
        
        Strategy:
        1. Build ONE global KDTree on GPU (fast with cuML)
        2. Query neighbors in chunks to manage memory
        3. Compute normals on GPU with vectorized operations
        
        Args:
            points: [N, 3] point coordinates
            k: number of neighbors
            
        Returns:
            normals: [N, 3] normalized surface normals
        """
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        
        # Transfer points to GPU ONCE
        points_gpu = self._to_gpu(points)
        self._log_gpu_memory("after points transfer")
        
        # BUILD GLOBAL KDTREE ONCE (MASSIVE SPEEDUP!)
        logger.info(f"  🔨 Building global KDTree ({N:,} points)...")
        knn = None
        
        if self.use_cuml and cuNearestNeighbors is not None:
            # GPU KDTree with cuML - extremely fast!
            knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(points_gpu)
            logger.info("  ✓ Global GPU KDTree built (cuML)")
        else:
            # CPU KDTree fallback - still better than per-chunk
            points_cpu = self._to_cpu(points_gpu)
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='kd_tree', n_jobs=-1)
            knn.fit(points_cpu)
            logger.info("  ✓ Global CPU KDTree built (sklearn)")
        
        # Progress bar
        chunk_iterator = range(num_chunks)
        if self.show_progress:
            bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} chunks '
                       '[{elapsed}<{remaining}, {rate_fmt}]')
            chunk_iterator = tqdm(
                chunk_iterator,
                desc=f"  🎯 GPU Normals ({N:,} pts, {num_chunks} chunks)",
                unit="chunk",
                total=num_chunks,
                bar_format=bar_fmt
            )
        
        # QUERY KDTREE IN CHUNKS (reuse tree!)
        # OPTIMIZATION: Batch GPU transfers - accumulate results on GPU, transfer once
        chunk_normals_list = []  # Accumulate on GPU for batched transfer
        
        for chunk_idx in chunk_iterator:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, N)
            
            # Query neighbors for this chunk
            if self.use_cuml and cuNearestNeighbors is not None:
                # GPU query (fast!)
                query_points = points_gpu[start_idx:end_idx]
                distances, indices = knn.kneighbors(query_points)
                # OPTIMIZATION: Keep indices on GPU if possible (avoid GPU->CPU->GPU)
                if not isinstance(indices, cp.ndarray):
                    indices = cp.asarray(indices)
                del query_points, distances
            else:
                # CPU query (still faster than rebuilding tree!)
                query_points = points_cpu[start_idx:end_idx]
                distances, indices = knn.kneighbors(query_points)
                # Convert to GPU for computation
                if self.use_gpu and cp is not None:
                    indices = cp.asarray(indices)
                del query_points, distances
            
            # Compute normals for this chunk (on GPU if available)
            chunk_normals = self._compute_normals_from_neighbors_gpu(
                points_gpu, indices
            )
            
            # OPTIMIZATION: Keep results on GPU, accumulate for batch transfer
            chunk_normals_list.append(chunk_normals)
            
            # Memory cleanup - REDUCED frequency for better batching
            del indices
            if chunk_idx % 20 == 0:  # OPTIMIZED: Was every 10, now every 20
                self._free_gpu_memory()  # Smart cleanup (only if >80% VRAM)
        
        # OPTIMIZATION: Single batched transfer at end (10-100x fewer syncs!)
        logger.info(f"  📦 Batching {len(chunk_normals_list)} chunks for single GPU transfer...")
        if self.use_gpu and cp is not None:
            # Concatenate all results on GPU
            normals_gpu = cp.concatenate(chunk_normals_list)
            # Single transfer to CPU
            normals = self._to_cpu(normals_gpu)
            # Cleanup
            del normals_gpu, chunk_normals_list
        else:
            # CPU fallback: already have numpy arrays
            normals = np.concatenate(chunk_normals_list)
            del chunk_normals_list
        
        # Final cleanup - force cleanup at end
        del knn, points_gpu
        self._free_gpu_memory(force=True)
        
        logger.info("  ✓ Global KDTree normals computation complete (batched transfer)")
        return normals
    
    def _compute_normals_from_neighbors_gpu(
        self,
        points_gpu,
        neighbor_indices
    ):
        """
        Compute normals using VECTORIZED covariance computation.
        
        This is ~100x faster than per-point PCA loops by computing
        all covariance matrices at once using vectorized operations.
        
        Args:
            points_gpu: [N, 3] all points (on GPU if available)
            neighbor_indices: [M, k] neighbor indices for M query points
            
        Returns:
            normals: [M, 3] normals (on GPU if available)
        """
        M, k = neighbor_indices.shape
        
        # Determine computation backend
        use_gpu = cp is not None and isinstance(points_gpu, cp.ndarray)
        xp = cp if use_gpu else np
        
        # Gather all neighbor points: [M, k, 3]
        if use_gpu:
            neighbor_points = points_gpu[neighbor_indices]
        else:
            neighbor_points = points_gpu[neighbor_indices]
        
        # Center the neighborhoods: [M, k, 3]
        centroids = xp.mean(
            neighbor_points, axis=1, keepdims=True
        )  # [M, 1, 3]
        centered = neighbor_points - centroids  # [M, k, 3]
        
        # MEMORY OPTIMIZATION: Free neighbor_points immediately
        del neighbor_points, centroids
        
        # Compute covariance matrices for ALL points at once: [M, 3, 3]
        # cov = (1/k) * (centered.T @ centered)
        # Using einsum for efficient batched matrix multiplication
        cov_matrices = xp.einsum('mki,mkj->mij', centered, centered) / k
        
        # MEMORY OPTIMIZATION: Free centered data after covariance computation
        del centered
        
        # Ensure symmetry (avoid numerical precision issues)
        cov_T = xp.transpose(cov_matrices, (0, 2, 1))
        cov_matrices = (cov_matrices + cov_T) / 2
        del cov_T  # Free transposed copy
        
        # CRITICAL FIX: Use float64 for GPU eigendecomposition
        # CuSOLVER's cusolverDnSsyevd (float32 version) has known stability
        # issues with batched operations, causing CUSOLVER_STATUS_INVALID_VALUE
        # errors. Using float64 (cusolverDnDsyevd) is more stable.
        # We convert back to float32 after eigh to save memory.
        if use_gpu and cov_matrices.dtype == cp.float32:
            cov_matrices = cov_matrices.astype(cp.float64)
        
        # Add MORE AGGRESSIVE regularization for GPU stability
        # CuSOLVER is more sensitive than CPU LAPACK
        reg_term = 1e-6 if use_gpu else 1e-8  # Stronger for GPU
        if use_gpu:
            # Match the dtype of cov_matrices (now float64 for stability)
            eye = cp.eye(3, dtype=cov_matrices.dtype)
        else:
            eye = np.eye(3, dtype=np.float32)
        cov_matrices = cov_matrices + reg_term * eye
        
        # Validate covariance matrices before eigendecomposition
        if use_gpu:
            # Check for NaN/Inf on GPU
            is_valid = cp.all(cp.isfinite(cov_matrices))
            if not is_valid:
                logger.warning(
                    "  ⚠ Invalid covariance matrices detected, sanitizing..."
                )
                # Replace invalid matrices with identity
                invalid_mask = ~cp.all(cp.isfinite(cov_matrices), axis=(1, 2))
                cov_matrices[invalid_mask] = eye
        else:
            # Check for NaN/Inf on CPU
            is_valid = np.all(np.isfinite(cov_matrices))
            if not is_valid:
                logger.warning(
                    "  ⚠ Invalid covariance matrices detected, sanitizing..."
                )
                invalid_mask = ~np.all(np.isfinite(cov_matrices), axis=(1, 2))
                cov_matrices[invalid_mask] = eye
        
        try:
            # Compute eigenvalues/eigenvectors for all covariance matrices
            # eigenvalues: [M, 3], eigenvectors: [M, 3, 3]
            if use_gpu:
                # GPU eigendecomposition with CuSOLVER
                # Process in sub-chunks to avoid CUSOLVER_STATUS_INVALID_VALUE
                # CuSOLVER has limits on batch sizes even with float64
                # INTELLIGENT AUTO-SCALING: Adapt batch size to VRAM
                if self.memory_manager and self.auto_optimize:
                    eigh_chunk_size = (
                        self.memory_manager.calculate_optimal_eigh_batch_size(
                            chunk_size=M,
                            vram_free_gb=self.vram_limit_gb
                        )
                    )
                else:
                    eigh_chunk_size = 250_000  # Default conservative
                
                if M <= eigh_chunk_size:
                    # Small enough to process at once
                    eigenvalues, eigenvectors = cp.linalg.eigh(cov_matrices)
                    # MEMORY OPTIMIZATION: Free cov_matrices immediately
                    del cov_matrices
                else:
                    # Process in sub-chunks
                    dtype = cov_matrices.dtype
                    eigenvalues = cp.zeros((M, 3), dtype=dtype)
                    eigenvectors = cp.zeros((M, 3, 3), dtype=dtype)
                    
                    for i in range(0, M, eigh_chunk_size):
                        end_i = min(i + eigh_chunk_size, M)
                        eigenvalues[i:end_i], eigenvectors[i:end_i] = (
                            cp.linalg.eigh(cov_matrices[i:end_i])
                        )
                    # MEMORY OPTIMIZATION: Free cov_matrices after processing
                    del cov_matrices
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
        except Exception as e:
            # If GPU eigendecomposition fails, fallback to CPU
            error_msg = str(e)
            logger.error(f"  ✗ GPU Eigendecomposition failed: {error_msg}")
            
            if use_gpu and "CUSOLVER" in error_msg.upper():
                # CUSOLVER error - fallback to CPU computation
                logger.warning(
                    "  ⚠ CuSOLVER error detected, falling back to CPU "
                    "eigendecomposition..."
                )
                try:
                    # Transfer to CPU and compute there
                    cov_matrices_cpu = cp.asnumpy(cov_matrices)
                    eigenvalues, eigenvectors = (
                        np.linalg.eigh(cov_matrices_cpu)
                    )
                    # Transfer results back to GPU
                    eigenvalues = cp.asarray(eigenvalues)
                    eigenvectors = cp.asarray(eigenvectors)
                    logger.info("  ✓ CPU fallback successful")
                except Exception as e2:
                    # Even CPU failed - use default normals
                    logger.error(f"  ✗ CPU fallback also failed: {e2}")
                    logger.warning(
                        "  ⚠ Using default normals (vertical orientation)"
                    )
                    if use_gpu:
                        normals = cp.zeros((M, 3), dtype=cp.float32)
                        normals[:, 2] = 1.0
                    else:
                        normals = np.zeros((M, 3), dtype=np.float32)
                        normals[:, 2] = 1.0
                    return normals
            else:
                # Other error - use default normals
                logger.warning(
                    "  ⚠ Using default normals (vertical orientation)"
                )
                if use_gpu:
                    normals = cp.zeros((M, 3), dtype=cp.float32)
                    normals[:, 2] = 1.0
                else:
                    normals = np.zeros((M, 3), dtype=np.float32)
                    normals[:, 2] = 1.0
                return normals
        
        # Normal = eigenvector with smallest eigenvalue
        # (first column after eigh - returns ascending order)
        normals = eigenvectors[:, :, 0]  # [M, 3]
        
        # MEMORY OPTIMIZATION: Free eigenvectors (only need 1st column)
        del eigenvectors
        
        # Convert back to float32 to save memory (float64 only needed for eigh)
        if use_gpu and normals.dtype == cp.float64:
            normals = normals.astype(cp.float32)
            eigenvalues = eigenvalues.astype(cp.float32)
        
        # Normalize normals
        norms = xp.linalg.norm(normals, axis=1, keepdims=True)  # [M, 1]
        norms = xp.maximum(norms, 1e-6)  # Avoid division by zero
        normals = normals / norms
        del norms  # Free immediately
        
        # Orient normals upward (positive Z)
        flip_mask = normals[:, 2] < 0  # [M]
        normals[flip_mask] *= -1
        del flip_mask  # Free mask
        
        # Handle degenerate cases (very small variance)
        variances = xp.sum(eigenvalues, axis=1)  # [M]
        degenerate = variances < 1e-6
        if xp.any(degenerate):
            if use_gpu:
                normals[degenerate] = cp.array([0, 0, 1], dtype=cp.float32)
            else:
                normals[degenerate] = np.array([0, 0, 1], dtype=np.float32)
        del eigenvalues, variances, degenerate  # Free temp arrays
        
        return normals
    
    def compute_curvature_chunked(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """
        Compute curvature using GPU with chunked processing.
        
        Args:
            points: [N, 3] point coordinates
            normals: [N, 3] surface normals
            k: number of neighbors
            
        Returns:
            curvature: [N] curvature values
        """
        if not self.use_gpu:
            from .features_gpu import GPUFeatureComputer
            computer = GPUFeatureComputer(use_gpu=False)
            return computer.compute_curvature(points, normals, k=k)
        
        N = len(points)
        
        # Use per-chunk strategy when cuML not available
        use_per_chunk = not (self.use_cuml and cuNearestNeighbors)
        
        if use_per_chunk:
            logger.info(
                f"Computing curvature with per-chunk KDTree: {N:,} points"
            )
            return self._compute_curvature_per_chunk(points, normals, k)
        
        logger.info(
            f"Computing curvature with global KDTree (cuML): {N:,} points"
        )
        
        curvature = np.zeros(N, dtype=np.float32)
        
        try:
            # Transfer to GPU
            points_gpu = self._to_gpu(points)
            normals_gpu = self._to_gpu(normals)
            
            # Build global KNN on GPU with cuML
            knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(points_gpu)
            
            # Process in chunks with progress bar
            num_chunks = (N + self.chunk_size - 1) // self.chunk_size
            
            chunk_iterator = range(num_chunks)
            if self.show_progress:
                bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} chunks '
                           '[{elapsed}<{remaining}, {rate_fmt}]')
                chunk_iterator = tqdm(
                    chunk_iterator,
                    desc=f"  🎯 GPU Curvature ({N:,} pts)",
                    unit="chunk",
                    total=num_chunks,
                    bar_format=bar_fmt
                )
            
            for chunk_idx in chunk_iterator:
                start_idx = chunk_idx * self.chunk_size
                end_idx = min((chunk_idx + 1) * self.chunk_size, N)
                
                # Query neighbors
                if self.use_cuml and cuNearestNeighbors is not None:
                    chunk_points = points_gpu[start_idx:end_idx]
                    distances, indices = knn.kneighbors(chunk_points)
                else:
                    chunk_points_cpu = self._to_cpu(
                        points_gpu[start_idx:end_idx]
                    )
                    distances, indices = knn.kneighbors(chunk_points_cpu)
                    if self.use_gpu and cp is not None:
                        indices = cp.asarray(indices)
                
                # Compute curvature for chunk (VECTORIZED)
                # Get neighbor normals: [chunk_size, k, 3]
                if self.use_gpu and cp is not None and isinstance(normals_gpu, cp.ndarray):
                    xp = cp
                    # [chunk_size, k, 3]
                    neighbor_normals = normals_gpu[indices]
                    # [chunk_size, 3]
                    query_normals = normals_gpu[start_idx:end_idx]
                else:
                    xp = np
                    neighbor_normals = normals_gpu[indices]
                    query_normals = normals_gpu[start_idx:end_idx]
                
                # Expand query normals: [chunk_size, 1, 3]
                query_normals_expanded = query_normals[:, xp.newaxis, :]
                
                # Compute differences: [chunk_size, k, 3]
                normal_diff = neighbor_normals - query_normals_expanded
                
                # Compute norms and mean: [chunk_size]
                curv_norms = xp.linalg.norm(normal_diff, axis=2)  # [chunk, k]
                chunk_curvature = xp.mean(curv_norms, axis=1)  # [chunk_size]
                
                # Transfer to CPU
                curvature[start_idx:end_idx] = self._to_cpu(chunk_curvature)
                
                if self.use_cuml and cuNearestNeighbors is not None:
                    del chunk_points
                del distances, chunk_curvature
                # OPTIMIZED: Less frequent cleanup
                if chunk_idx % 10 == 0:  # Was every 5, now every 10
                    self._free_gpu_memory()  # Smart cleanup
            
            del points_gpu, normals_gpu, knn
            self._free_gpu_memory(force=True)  # Force final cleanup
            
            logger.info("  ✓ Curvature computation complete")
            return curvature
            
        except Exception as e:
            logger.error(f"GPU curvature failed: {e}")
            self._free_gpu_memory(force=True)  # Force cleanup on error
            self._free_gpu_memory()
            
            from .features_gpu import GPUFeatureComputer
            computer = GPUFeatureComputer(use_gpu=False)
            return computer.compute_curvature(points, normals, k=k)
    
    def _compute_curvature_per_chunk(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """
        Compute curvature using per-chunk KDTree strategy.
        Much faster than global KDTree when using sklearn fallback.
        """
        N = len(points)
        curvature = np.zeros(N, dtype=np.float32)
        
        # Calculate overlap
        overlap_ratio = 0.05
        overlap_size = int(self.chunk_size * overlap_ratio)
        
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        
        # Transfer to GPU
        points_gpu = self._to_gpu(points)
        normals_gpu = self._to_gpu(normals)
        
        # Progress bar
        chunk_iterator = range(num_chunks)
        if self.show_progress:
            bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} chunks '
                       '[{elapsed}<{remaining}, {rate_fmt}]')
            chunk_iterator = tqdm(
                chunk_iterator,
                desc=f"  🎯 GPU Curvature ({N:,} pts)",
                unit="chunk",
                total=num_chunks,
                bar_format=bar_fmt
            )
        
        for chunk_idx in chunk_iterator:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, N)
            
            # Extended range for KDTree
            tree_start = max(0, start_idx - overlap_size)
            tree_end = min(N, end_idx + overlap_size)
            
            # Build local KDTree
            chunk_points_cpu = self._to_cpu(
                points_gpu[tree_start:tree_end]
            )
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(chunk_points_cpu)
            
            # Query
            query_points = chunk_points_cpu[
                (start_idx - tree_start):(end_idx - tree_start)
            ]
            distances, local_indices = knn.kneighbors(query_points)
            global_indices = local_indices + tree_start
            
            # Compute curvature (VECTORIZED)
            # Get neighbor normals for all points at once: [chunk_size, k, 3]
            if self.use_gpu and cp is not None and isinstance(normals_gpu, cp.ndarray):
                neighbor_normals = cp.asnumpy(
                    normals_gpu[global_indices]
                )  # [chunk_size, k, 3]
                query_normals = cp.asnumpy(
                    normals_gpu[start_idx:end_idx]
                )  # [chunk_size, 3]
            else:
                neighbor_normals = normals_gpu[global_indices]
                query_normals = normals_gpu[start_idx:end_idx]
            
            # Expand query normals for broadcasting: [chunk_size, 1, 3]
            query_normals_expanded = query_normals[:, np.newaxis, :]
            
            # Compute differences: [chunk_size, k, 3]
            normal_diff = neighbor_normals - query_normals_expanded
            
            # Compute norms and mean: [chunk_size]
            curv_norms = np.linalg.norm(normal_diff, axis=2)  # [chunk_size, k]
            chunk_curvature = np.mean(curv_norms, axis=1)  # [chunk_size]
            
            curvature[start_idx:end_idx] = chunk_curvature
            
            if chunk_idx % 5 == 0:
                self._free_gpu_memory()
        
        logger.info("  ✓ Per-chunk curvature computation complete")
        return curvature
    
    def compute_eigenvalue_features(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        neighbors_indices: np.ndarray,
        start_idx: int = None,
        end_idx: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute eigenvalue-based features (FULL GPU-accelerated with chunking support).
        
        Features:
        - eigenvalue_1, eigenvalue_2, eigenvalue_3: Individual eigenvalues (λ₀, λ₁, λ₂)
        - sum_eigenvalues: Sum of eigenvalues (Σλ)
        - eigenentropy: Shannon entropy of normalized eigenvalues
        - omnivariance: Cubic root of product of eigenvalues
        - change_curvature: Variance-based curvature change measure
        
        Args:
            points: [N_total, 3] point coordinates (full array for neighbor lookup)
            normals: [N_total, 3] surface normals (full array for neighbor lookup)
            neighbors_indices: [N_chunk, k] indices of k-nearest neighbors
            start_idx: Start index of chunk in full array (optional)
            end_idx: End index of chunk in full array (optional)
            
        Returns:
            Dictionary of eigenvalue-based features for the chunk
        """
        # If start_idx/end_idx provided, we're processing a chunk
        if start_idx is not None and end_idx is not None:
            N = end_idx - start_idx
        else:
            N = len(neighbors_indices)
            start_idx = 0
            end_idx = N
        
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
        neighbors_indices: np.ndarray,
        start_idx: int = None,
        end_idx: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute architectural features for building detection (FULL GPU-accelerated with chunking).
        
        Features:
        - edge_strength: Strength of edges (high eigenvalue variance)
        - corner_likelihood: Probability of corner point (3D structure)
        - overhang_indicator: Overhang/protrusion detection
        - surface_roughness: Fine-scale surface texture
        
        Args:
            points: [N_total, 3] point coordinates (full array for neighbor lookup)
            normals: [N_total, 3] surface normals (full array for neighbor lookup)
            neighbors_indices: [N_chunk, k] indices of k-nearest neighbors
            start_idx: Start index of chunk in full array (optional)
            end_idx: End index of chunk in full array (optional)
            
        Returns:
            Dictionary of architectural features for the chunk
        """
        # If start_idx/end_idx provided, we're processing a chunk
        if start_idx is not None and end_idx is not None:
            N = end_idx - start_idx
            chunk_points = points[start_idx:end_idx]
            chunk_normals = normals[start_idx:end_idx]
        else:
            N = len(neighbors_indices)
            start_idx = 0
            end_idx = N
            chunk_points = points
            chunk_normals = normals
        
        k = neighbors_indices.shape[1]
        
        # Determine computation backend (GPU if available, else CPU)
        use_gpu = self.use_gpu and cp is not None
        xp = cp if use_gpu else np
        
        # Transfer to GPU if available
        if use_gpu:
            points_gpu = self._to_gpu(points)  # Full array for neighbor lookup
            normals_gpu = self._to_gpu(normals)  # Full array for neighbor lookup
            chunk_points_gpu = self._to_gpu(chunk_points)  # Chunk for center point computations
            chunk_normals_gpu = self._to_gpu(chunk_normals)  # Chunk for center normal computations
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
            normal_diffs = neighbor_normals - chunk_normals_gpu[:, cp.newaxis, :]
        else:
            normal_diffs = neighbor_normals - chunk_normals[:, np.newaxis, :]
        normal_variation = xp.linalg.norm(normal_diffs, axis=2).mean(axis=1)
        
        # Overhang indicator: Large vertical normal variation
        if use_gpu:
            vertical_diffs = neighbor_normals[:, :, 2] - chunk_normals_gpu[:, 2:3]
        else:
            vertical_diffs = neighbor_normals[:, :, 2] - chunk_normals[:, 2:3]
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

    def _compute_geometric_features_from_neighbors(
        self,
        points: np.ndarray,
        neighbors_indices: np.ndarray,
        chunk_points: np.ndarray,
        points_gpu=None  # NEW: Optional pre-cached GPU array to avoid re-transfer
    ) -> Dict[str, np.ndarray]:
        """
        Compute geometric features directly from pre-computed neighbor indices.
        
        This avoids rebuilding KDTree (major optimization for chunked processing).
        
        Args:
            points: [N_total, 3] full point cloud for neighbor lookup
            neighbors_indices: [N_chunk, k] indices of k-nearest neighbors  
            chunk_points: [N_chunk, 3] points for this chunk
            points_gpu: Optional pre-cached GPU array (avoids re-transfer, HUGE speedup!)
            
        Returns:
            Dictionary of geometric features for the chunk
        """
        N = len(chunk_points)
        k = neighbors_indices.shape[1]
        
        # OPTIMIZED: Use GPU for fancy indexing if available (10-100x faster than CPU!)
        if self.use_gpu and cp is not None and points_gpu is not None:
            # Keep everything on GPU to avoid slow CPU fancy indexing
            neighbors_indices_gpu = cp.asarray(neighbors_indices)
            neighbors_gpu = points_gpu[neighbors_indices_gpu]  # GPU fancy indexing is FAST!
            neighbors = self._to_cpu(neighbors_gpu)  # Transfer result only
            del neighbors_indices_gpu, neighbors_gpu
        else:
            # CPU fallback (slower)
            neighbors = points[neighbors_indices]  # [N, k, 3]
        
        # Compute eigenvalues from neighborhood covariance
        centroids = np.mean(neighbors, axis=1, keepdims=True)  # [N, 1, 3]
        centered = neighbors - centroids  # [N, k, 3]
        
        # Covariance matrices: [N, 3, 3]
        cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)
        
        # Compute eigenvalues: [N, 3]
        eigenvalues = np.linalg.eigvalsh(cov_matrices)
        eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]  # Sort descending
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Avoid division by zero
        
        λ0 = eigenvalues[:, 0]
        λ1 = eigenvalues[:, 1]
        λ2 = eigenvalues[:, 2]
        λ_sum = λ0 + λ1 + λ2
        
        # Compute geometric features
        features = {
            'linearity': np.clip((λ0 - λ1) / (λ0 + 1e-8), 0.0, 1.0),
            'planarity': np.clip((λ1 - λ2) / (λ0 + 1e-8), 0.0, 1.0),
            'sphericity': np.clip(λ2 / (λ0 + 1e-8), 0.0, 1.0),
            'anisotropy': np.clip((λ0 - λ2) / (λ0 + 1e-8), 0.0, 1.0),
            'roughness': np.clip(λ2 / (λ_sum + 1e-8), 0.0, 1.0),
        }
        
        # Compute density (inverse of mean distance)
        distances = np.linalg.norm(neighbors - chunk_points[:, np.newaxis, :], axis=2)
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self
        features['density'] = np.clip(1.0 / (mean_distances + 1e-8), 0.0, 1000.0)
        
        # Convert to float32
        for key in features:
            features[key] = features[key].astype(np.float32)
        
        return features

    def compute_density_features(
        self,
        points: np.ndarray,
        neighbors_indices: np.ndarray,
        radius_2m: float = 2.0,
        start_idx: int = None,
        end_idx: int = None,
        points_gpu=None  # NEW: Optional pre-cached GPU array
    ) -> Dict[str, np.ndarray]:
        """
        Compute density and neighborhood features (OPTIMIZED: avoids rebuilding KDTree).
        
        Features:
        - density: Local point density (1/mean_distance)
        - num_points_2m: Number of points within 2m radius (approximated from k-NN)
        - neighborhood_extent: Maximum distance to k-th neighbor
        - height_extent_ratio: Ratio of vertical to spatial extent
        
        Args:
            points: [N_total, 3] point coordinates (full array for neighbor lookup)
            neighbors_indices: [N_chunk, k] indices of k-nearest neighbors
            radius_2m: Radius for counting nearby points (default 2.0m)
            start_idx: Start index of chunk in full array (optional)
            end_idx: End index of chunk in full array (optional)
            points_gpu: Pre-cached GPU array (optional, avoids re-transfer)
            
        Returns:
            Dictionary of density features for the chunk
        """
        # If start_idx/end_idx provided, we're processing a chunk
        if start_idx is not None and end_idx is not None:
            N = end_idx - start_idx
            chunk_points = points[start_idx:end_idx]
        else:
            N = len(neighbors_indices)
            start_idx = 0
            end_idx = N
            chunk_points = points
        
        k = neighbors_indices.shape[1]
        
        # Determine computation backend (GPU if available, else CPU)
        use_gpu = self.use_gpu and cp is not None
        xp = cp if use_gpu else np
        
        # Transfer to GPU if available (OPTIMIZED: reuse cached GPU array if provided)
        if use_gpu:
            if points_gpu is None:
                points_gpu = self._to_gpu(points)  # Full array for neighbor lookup
            chunk_points_gpu = self._to_gpu(chunk_points)  # Chunk for center point computations
            neighbors_indices_gpu = cp.asarray(neighbors_indices)
            
            # OPTIMIZED: Batch the neighbor indexing to avoid GPU memory thrashing
            # For large chunks (>1M points), fancy indexing points_gpu[neighbors_indices_gpu]
            # can cause massive slowdowns. Batch it to keep GPU responsive.
            NEIGHBOR_BATCH_SIZE = 500_000  # Process 500K points at a time for neighbor lookup
            if N > NEIGHBOR_BATCH_SIZE:
                num_neighbor_batches = (N + NEIGHBOR_BATCH_SIZE - 1) // NEIGHBOR_BATCH_SIZE
                neighbors_list = []
                
                # DEBUG: Log batching progress
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"    Batching neighbor lookup: {N:,} points → {num_neighbor_batches} batches")
                
                for nb_idx in range(num_neighbor_batches):
                    nb_start = nb_idx * NEIGHBOR_BATCH_SIZE
                    nb_end = min((nb_idx + 1) * NEIGHBOR_BATCH_SIZE, N)
                    batch_indices = neighbors_indices_gpu[nb_start:nb_end]
                    batch_neighbors = points_gpu[batch_indices]
                    neighbors_list.append(batch_neighbors)
                    del batch_indices, batch_neighbors
                
                neighbors = cp.vstack(neighbors_list) if len(neighbors_list) > 1 else neighbors_list[0]
                del neighbors_list
            else:
                # Small chunk, direct indexing is fine
                neighbors = points_gpu[neighbors_indices_gpu]
        else:
            neighbors = points[neighbors_indices]
        
        # Compute distances to all neighbors: [N, k]
        if use_gpu:
            distances = xp.linalg.norm(
                neighbors - chunk_points_gpu[:, cp.newaxis, :],
                axis=2
            )
        else:
            distances = xp.linalg.norm(
                neighbors - chunk_points[:, np.newaxis, :],
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
        
        # Number of points within 2m radius - OPTIMIZED: use k-NN approximation instead of rebuilding KDTree
        # Count neighbors within radius from existing k-NN results (works for both GPU and CPU)
        within_radius = xp.sum(distances <= radius_2m, axis=1)
        num_points_2m = within_radius.astype(xp.float32)
        
        # Transfer results back to CPU if using GPU
        if use_gpu:
            density = self._to_cpu(density)
            num_points_2m = self._to_cpu(num_points_2m)
            neighborhood_extent = self._to_cpu(neighborhood_extent)
            height_extent_ratio = self._to_cpu(height_extent_ratio)
            vertical_std = self._to_cpu(vertical_std)
        
        return {
            'density': density.astype(np.float32),
            'num_points_2m': num_points_2m.astype(np.float32),
            'neighborhood_extent': neighborhood_extent.astype(np.float32),
            'height_extent_ratio': np.clip(height_extent_ratio, 0.0, 1.0).astype(np.float32),
            'vertical_std': vertical_std.astype(np.float32),
        }
    
    def compute_reclassification_features_optimized(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int = 10,
        mode: str = 'minimal'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        OPTIMIZED: Compute minimal features specifically for reclassification workflows.
        
        This method is optimized for speed over completeness, focusing only on features
        that are essential for reclassification tasks:
        - Surface normals (for orientation-based rules)
        - Height above ground (for elevation-based rules)
        - Basic planarity and density (for geometric rules)
        
        Key optimizations:
        1. Reduced feature set for faster computation
        2. Optimized memory usage for large point clouds
        3. Adaptive chunking based on available VRAM
        4. Fallback strategies for robustness
        
        Args:
            points: [N, 3] point coordinates
            classification: [N] ASPRS classification codes
            k: number of neighbors for computations
            mode: Feature mode ('minimal', 'standard', 'full')
            
        Returns:
            normals: [N, 3] surface normals
            curvature: [N] curvature values (basic)
            height: [N] height above ground
            geo_features: dict with minimal geometric features
        """
        N = len(points)
        
        # Determine feature set based on mode
        if mode == 'minimal':
            # Absolute minimum for basic reclassification
            required_features = ['planarity', 'density', 'verticality']
        elif mode == 'standard':
            # Standard reclassification features
            required_features = ['planarity', 'linearity', 'density', 'verticality', 
                               'roughness', 'wall_score', 'roof_score']
        else:  # 'full'
            # All features (same as regular computation)
            return self.compute_all_features_chunked(points, classification, k=k, mode=mode)
        
        # Adaptive chunk size optimization for reclassification
        if self.memory_manager and self.auto_optimize:
            # Recalculate optimal chunk size for this specific point cloud
            optimal_chunk = self.memory_manager.calculate_optimal_gpu_chunk_size(
                num_points=N,
                vram_free_gb=self.vram_limit_gb,
                feature_mode=mode,
                k_neighbors=k
            )
            # Use optimal chunk size if significantly different
            if abs(optimal_chunk - self.chunk_size) > 500_000:
                logger.info(f"🔧 Adapting chunk size: {self.chunk_size:,} → {optimal_chunk:,}")
                self.chunk_size = optimal_chunk
        
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        logger.info(
            f"🚀 RECLASSIFICATION MODE: Computing {mode} features with optimized chunking"
        )
        logger.info(
            f"   {N:,} points → {num_chunks} chunks @ {self.chunk_size:,} pts/chunk"
        )
        
        # Initialize output arrays
        normals = np.zeros((N, 3), dtype=np.float32)
        curvature = np.zeros(N, dtype=np.float32)
        height = np.zeros(N, dtype=np.float32)
        
        # Initialize only required geometric features
        geo_features = {}
        for feat_name in required_features:
            geo_features[feat_name] = np.zeros(N, dtype=np.float32)
        
        # Transfer points to GPU ONCE
        points_gpu = self._to_gpu(points)
        
        # Build global KDTree ONCE (same optimization as full mode)
        logger.info(f"  🔨 Building global KDTree ({N:,} points)...")
        knn = None
        
        try:
            if self.use_cuml and cuNearestNeighbors is not None:
                # GPU KDTree with cuML
                knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
                knn.fit(points_gpu)
                logger.info("     ✓ Global GPU KDTree built (cuML)")
            else:
                # CPU KDTree fallback
                points_cpu = self._to_cpu(points_gpu)
                knn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='kd_tree', n_jobs=-1)
                knn.fit(points_cpu)
                logger.info("     ✓ Global CPU KDTree built (sklearn)")
        except Exception as e:
            logger.error(f"⚠️ KDTree building failed: {e}")
            logger.warning("Falling back to CPU-only processing...")
            self.use_gpu = False
            from .features_gpu import GPUFeatureComputer
            computer = GPUFeatureComputer(use_gpu=False)
            return computer.compute_normals(points, k=k), \
                   computer.compute_curvature(points, normals, k=k), \
                   computer.compute_height_above_ground(points, classification), \
                   {'planarity': np.zeros(N, dtype=np.float32)}
        
        # Process chunks with optimized feature computation
        chunk_iterator = range(num_chunks)
        if self.show_progress:
            bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} chunks '
                       '[{elapsed}<{remaining}, {rate_fmt}]')
            chunk_iterator = tqdm(
                chunk_iterator,
                desc=f"     Computing {mode} features",
                unit="chunk",
                total=num_chunks,
                bar_format=bar_fmt
            )
        
        # Import GPU computer for helper functions
        from .features_gpu import GPUFeatureComputer
        gpu_computer = GPUFeatureComputer(use_gpu=self.use_gpu)
        
        # Cache points_cpu if using CPU KNN
        points_cpu = None if (self.use_cuml and cuNearestNeighbors is not None) else self._to_cpu(points_gpu)
        
        # OPTIMIZATION #1: Persistent GPU arrays - cache normals on GPU to avoid repeated uploads
        normals_gpu_persistent = None
        
        for chunk_idx in chunk_iterator:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, N)
            
            try:
                # Query neighbors for this chunk
                if self.use_cuml and cuNearestNeighbors is not None:
                    # GPU query
                    query_points = points_gpu[start_idx:end_idx]
                    distances, global_indices = knn.kneighbors(query_points)
                    global_indices = self._to_cpu(global_indices)
                    del query_points, distances
                else:
                    # CPU query
                    query_points = points_cpu[start_idx:end_idx]
                    distances, global_indices = knn.kneighbors(query_points)
                    del query_points, distances
                
                # Compute normals (optimized vectorized version)
                if self.use_gpu and cp is not None:
                    global_indices_gpu = cp.asarray(global_indices)
                else:
                    global_indices_gpu = global_indices
                
                chunk_normals = self._compute_normals_from_neighbors_gpu(
                    points_gpu, global_indices_gpu
                )
                normals[start_idx:end_idx] = self._to_cpu(chunk_normals)
                
                # Compute basic curvature if needed
                if 'curvature' in required_features or mode != 'minimal':
                    # OPTIMIZATION #1: Reuse persistent GPU array instead of re-uploading each chunk
                    if self.use_gpu and cp is not None:
                        # Upload normals to GPU once and cache for reuse across chunks
                        if normals_gpu_persistent is None:
                            normals_gpu_persistent = self._to_gpu(normals)
                        else:
                            # Update the cached GPU array with new normals for this chunk
                            normals_gpu_persistent[start_idx:end_idx] = chunk_normals
                        
                        # Fast GPU fancy indexing (reusing cached array)
                        neighbor_normals_gpu = normals_gpu_persistent[global_indices_gpu]
                        neighbor_normals = self._to_cpu(neighbor_normals_gpu)
                        del neighbor_normals_gpu
                        
                        chunk_normals_cpu = self._to_cpu(chunk_normals)
                        normals_expanded = chunk_normals_cpu[:, np.newaxis, :]
                        normal_diff = neighbor_normals - normals_expanded
                    else:
                        # CPU fallback (slower)
                        chunk_normals_cpu = self._to_cpu(chunk_normals)
                        neighbor_normals = normals[global_indices]
                        normals_expanded = chunk_normals_cpu[:, np.newaxis, :]
                        normal_diff = neighbor_normals - normals_expanded
                    
                    curv_norms = np.linalg.norm(normal_diff, axis=2)
                    chunk_curvature = np.mean(curv_norms, axis=1).astype(np.float32)
                    curvature[start_idx:end_idx] = chunk_curvature
                
                # Compute height (essential for reclassification)
                chunk_points_cpu = self._to_cpu(points_gpu[start_idx:end_idx])
                chunk_classification = classification[start_idx:end_idx]
                chunk_height = gpu_computer.compute_height_above_ground(
                    chunk_points_cpu, chunk_classification
                )
                height[start_idx:end_idx] = chunk_height
                
                # Compute only required geometric features
                chunk_normals_cpu = self._to_cpu(chunk_normals)
                
                # Verticality (essential for building/vegetation distinction)
                if 'verticality' in required_features:
                    verticality_chunk = gpu_computer.compute_verticality(chunk_normals_cpu)
                    geo_features['verticality'][start_idx:end_idx] = verticality_chunk
                
                # Compute eigenvalue-based features efficiently
                if any(feat in required_features for feat in ['planarity', 'linearity', 'density']):
                    # Use optimized eigenvalue computation
                    eigenvalue_feats = self._compute_minimal_eigenvalue_features(
                        points_gpu, global_indices_gpu, start_idx, end_idx, required_features
                    )
                    for key, values in eigenvalue_feats.items():
                        if key in geo_features:
                            geo_features[key][start_idx:end_idx] = values
                
                # Compute composite features if needed
                if 'wall_score' in required_features and 'planarity' in geo_features and 'verticality' in geo_features:
                    chunk_planarity = geo_features['planarity'][start_idx:end_idx]
                    chunk_verticality = geo_features['verticality'][start_idx:end_idx]
                    # Clean from NaN/Inf
                    chunk_planarity = np.nan_to_num(chunk_planarity, nan=0.0, posinf=1.0, neginf=0.0)
                    chunk_verticality = np.nan_to_num(chunk_verticality, nan=0.0, posinf=1.0, neginf=0.0)
                    wall_score_chunk = (chunk_planarity * chunk_verticality).astype(np.float32)
                    geo_features['wall_score'][start_idx:end_idx] = wall_score_chunk
                
                if 'roof_score' in required_features and 'planarity' in geo_features:
                    chunk_planarity = geo_features['planarity'][start_idx:end_idx]
                    # Horizontality = abs(normal_z)
                    horizontality_chunk = np.abs(chunk_normals_cpu[:, 2])
                    chunk_planarity = np.nan_to_num(chunk_planarity, nan=0.0, posinf=1.0, neginf=0.0)
                    horizontality_chunk = np.nan_to_num(horizontality_chunk, nan=0.0, posinf=1.0, neginf=0.0)
                    roof_score_chunk = (chunk_planarity * horizontality_chunk).astype(np.float32)
                    geo_features['roof_score'][start_idx:end_idx] = roof_score_chunk
                
                # Cleanup
                del chunk_normals, global_indices_gpu, global_indices
                if chunk_idx % 3 == 0:  # Less frequent cleanup
                    self._free_gpu_memory()
                    
            except Exception as e:
                logger.error(f"⚠️ Error processing chunk {chunk_idx}: {e}")
                # Fill chunk with default values to continue processing
                chunk_size_actual = end_idx - start_idx
                normals[start_idx:end_idx] = np.tile([0, 0, 1], (chunk_size_actual, 1))
                curvature[start_idx:end_idx] = 0.0
                height[start_idx:end_idx] = 0.0
                for feat_name in required_features:
                    geo_features[feat_name][start_idx:end_idx] = 0.0
                continue
        
        # Final cleanup
        del knn, points_gpu
        if points_cpu is not None:
            del points_cpu
        # OPTIMIZATION #1: Cleanup persistent GPU array
        if normals_gpu_persistent is not None:
            del normals_gpu_persistent
        self._free_gpu_memory()
        
        # Clean all features from NaN/Inf
        normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        height = np.nan_to_num(height, nan=0.0).astype(np.float32)
        
        for feat_name in geo_features:
            geo_features[feat_name] = np.nan_to_num(
                geo_features[feat_name], 
                nan=0.0, 
                posinf=1.0, 
                neginf=0.0
            ).astype(np.float32)
        
        logger.info(f"  ✓ Reclassification features computed successfully")
        return normals, curvature, height, geo_features

    def _compute_minimal_eigenvalue_features(
        self,
        points_gpu,
        neighbor_indices,
        start_idx: int,
        end_idx: int,
        required_features: list
    ) -> Dict[str, np.ndarray]:
        """
        Compute only the minimal eigenvalue-based features required for reclassification.
        
        This is much faster than computing all eigenvalue features since it only
        computes what's needed and uses optimized calculations.
        
        Args:
            points_gpu: GPU array of all points
            neighbor_indices: Neighbor indices for current chunk
            start_idx: Start index of chunk
            end_idx: End index of chunk  
            required_features: List of required feature names
            
        Returns:
            Dictionary of computed features
        """
        M, k = neighbor_indices.shape
        use_gpu = cp is not None and isinstance(points_gpu, cp.ndarray)
        xp = cp if use_gpu else np
        
        # Gather neighbor points
        if use_gpu:
            neighbors = points_gpu[neighbor_indices]
        else:
            neighbors = points_gpu[neighbor_indices]
        
        # Center neighbors
        centroids = xp.mean(neighbors, axis=1, keepdims=True)
        centered = neighbors - centroids
        del neighbors, centroids
        
        # Compute covariance matrices
        cov_matrices = xp.einsum('mki,mkj->mij', centered, centered) / k
        del centered
        
        # Only compute eigenvalues if we need them
        need_eigenvalues = any(feat in required_features 
                             for feat in ['planarity', 'linearity', 'sphericity', 'anisotropy'])
        
        features = {}
        
        if need_eigenvalues:
            # OPTIMIZATION #2: GPU eigenvalue computation
            # When use_gpu=True, xp=cp and eigenvalues are computed on GPU
            # This provides 10-15x speedup over CPU for LOD3/Full modes
            
            # Compute eigenvalues (stable version with float64 precision)
            if use_gpu and cov_matrices.dtype == cp.float32:
                cov_matrices = cov_matrices.astype(cp.float64)
            
            # Add regularization
            reg_term = 1e-6 if use_gpu else 1e-8
            if use_gpu:
                eye = cp.eye(3, dtype=cov_matrices.dtype)
            else:
                eye = np.eye(3, dtype=np.float32)
            cov_matrices = cov_matrices + reg_term * eye
            
            try:
                eigenvalues = xp.linalg.eigvalsh(cov_matrices)
                eigenvalues = xp.sort(eigenvalues, axis=1)[:, ::-1]  # Sort descending
                eigenvalues = xp.maximum(eigenvalues, 1e-10)  # Clamp to positive
                
                λ0 = eigenvalues[:, 0]
                λ1 = eigenvalues[:, 1] 
                λ2 = eigenvalues[:, 2]
                sum_λ = λ0 + λ1 + λ2
                
                # Compute only required features
                if 'planarity' in required_features:
                    planarity = (λ1 - λ2) / (sum_λ + 1e-8)
                    features['planarity'] = self._to_cpu(planarity).astype(np.float32)
                
                if 'linearity' in required_features:
                    linearity = (λ0 - λ1) / (sum_λ + 1e-8)
                    features['linearity'] = self._to_cpu(linearity).astype(np.float32)
                
                if 'sphericity' in required_features:
                    sphericity = λ2 / (sum_λ + 1e-8)
                    features['sphericity'] = self._to_cpu(sphericity).astype(np.float32)
                
                if 'anisotropy' in required_features:
                    anisotropy = (λ0 - λ2) / (sum_λ + 1e-8)
                    features['anisotropy'] = self._to_cpu(anisotropy).astype(np.float32)
                    
            except Exception as e:
                logger.warning(f"⚠️ Eigenvalue computation failed: {e}, using defaults")
                # Fill with default values
                chunk_size = end_idx - start_idx
                for feat in ['planarity', 'linearity', 'sphericity', 'anisotropy']:
                    if feat in required_features:
                        features[feat] = np.zeros(chunk_size, dtype=np.float32)
        
        # Density feature (if needed)
        if 'density' in required_features:
            # Simple density estimation from covariance trace
            density_est = 1.0 / (xp.trace(cov_matrices, axis1=1, axis2=2) + 1e-8)
            density_est = xp.clip(density_est, 0.0, 1000.0)
            features['density'] = self._to_cpu(density_est).astype(np.float32)
        
        # Roughness (if needed)
        if 'roughness' in required_features:
            # Simple roughness from covariance determinant
            det = xp.linalg.det(cov_matrices)
            roughness = xp.sqrt(xp.maximum(det, 1e-10))
            features['roughness'] = self._to_cpu(roughness).astype(np.float32)
        
        return features

    def compute_all_features_chunked(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int = 10,
        radius: Optional[float] = None,
        mode: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        HIGHLY OPTIMIZED: Compute ALL features with SINGLE GLOBAL KDTREE.
        
        Key optimizations:
        1. Build KDTree ONCE instead of per-chunk (10-100x speedup!)
        2. Reuse neighbor indices for all feature computations
        3. Minimal GPU↔CPU transfers
        4. Vectorized operations wherever possible
        
        FULL GPU SUPPORT: Advanced features (eigenvalue, architectural, density)
        are computed using GPU acceleration when available.
        
        Args:
            points: [N, 3] point coordinates
            classification: [N] ASPRS classification codes
            k: number of neighbors
            radius: search radius in meters (optional)
            mode: Feature mode ('minimal', 'lod2', 'lod3', 'full') - 
                  if specified, uses the new feature mode system to filter features
            
        Returns:
            normals: [N, 3] surface normals
            curvature: [N] curvature values
            height: [N] height above ground
            geo_features: dict with geometric features
        """
        # Get feature configuration if mode is specified
        feature_set = None
        if mode is not None:
            from ..features.feature_modes import get_feature_config
            # Suppress logging here - it's already logged at the orchestrator level
            feature_config = get_feature_config(mode=mode, k_neighbors=k, log_config=False)
            feature_set = feature_config.features
        
        N = len(points)
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        chunk_size_mb = (self.chunk_size * 12) / (1024 * 1024)
        gpu_status = "GPU-accelerated" if (self.use_gpu and cp is not None) else "CPU"
        logger.info(
            f"🚀 OPTIMIZED: Computing features with GLOBAL KDTree ({gpu_status})"
        )
        logger.info(
            f"   {N:,} points → {num_chunks} chunks @ {chunk_size_mb:.1f}MB each"
        )
        
        # Initialize output arrays
        normals = np.zeros((N, 3), dtype=np.float32)
        curvature = np.zeros(N, dtype=np.float32)
        height = np.zeros(N, dtype=np.float32)
        
        # Initialize geometric features
        geo_features = {
            'anisotropy': np.zeros(N, dtype=np.float32),
            'planarity': np.zeros(N, dtype=np.float32),
            'linearity': np.zeros(N, dtype=np.float32),
            'sphericity': np.zeros(N, dtype=np.float32),
            'roughness': np.zeros(N, dtype=np.float32),
            'density': np.zeros(N, dtype=np.float32),
            'verticality': np.zeros(N, dtype=np.float32),
            'horizontality': np.zeros(N, dtype=np.float32),
            'wall_score': np.zeros(N, dtype=np.float32),
            'roof_score': np.zeros(N, dtype=np.float32)
        }
        
        # Transfer points to GPU ONCE
        points_gpu = self._to_gpu(points)
        
        # ========================================================================
        # PHASE 1: BUILD GLOBAL KDTREE ONCE (MASSIVE SPEEDUP!)
        # ========================================================================
        logger.info(f"  🔨 Phase 1/3: Building global KDTree ({N:,} points)...")
        knn = None
        
        if self.use_cuml and cuNearestNeighbors is not None:
            # GPU KDTree with cuML - extremely fast!
            knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(points_gpu)
            logger.info("     ✓ Global GPU KDTree built (cuML)")
        else:
            # CPU KDTree fallback
            points_cpu = self._to_cpu(points_gpu)
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='kd_tree', n_jobs=-1)
            knn.fit(points_cpu)
            logger.info("     ✓ Global CPU KDTree built (sklearn, parallel)")
        
        # ========================================================================
        # PHASE 2: QUERY NEIGHBORS IN CHUNKS & COMPUTE FEATURES
        # ========================================================================
        logger.info(f"  ⚡ Phase 2/3: Querying neighbors & computing features...")
        
        # Determine which advanced feature groups to compute
        eigenvalue_feature_names = {
            'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3', 
            'sum_eigenvalues', 'eigenentropy', 'omnivariance', 'change_curvature'
        }
        architectural_feature_names = {
            'edge_strength', 'corner_likelihood', 'overhang_indicator', 'surface_roughness'
        }
        # FIXED: Only ADVANCED density features (not basic 'density' which is in geometric features)
        density_feature_names = {
            'density_2d', 'density_vertical', 'local_point_density', 'num_points_2m',
            'neighborhood_extent', 'height_extent_ratio'
        }
        
        compute_eigenvalues = feature_set is None or any(feat in feature_set for feat in eigenvalue_feature_names)
        compute_architectural = feature_set is None or any(feat in feature_set for feat in architectural_feature_names)
        compute_density_advanced = feature_set is None or any(feat in feature_set for feat in density_feature_names)
        
        # Log what's being computed
        if feature_set is not None:
            feature_groups = []
            if compute_eigenvalues:
                feature_groups.append("eigenvalues")
            if compute_architectural:
                feature_groups.append("architectural")
            if compute_density_advanced:
                feature_groups.append("density")
            
            if feature_groups:
                logger.info(f"     Computing advanced features: {', '.join(feature_groups)}")
            else:
                logger.info(f"     ⚡ FAST MODE: Skipping advanced features (not needed for mode '{mode}')")
        
        # Progress bar with GPU/CPU indicator
        backend_label = "GPU" if (self.use_cuml and cuNearestNeighbors is not None) else "CPU"
        chunk_iterator = range(num_chunks)
        if self.show_progress:
            bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} chunks '
                       f'[{{elapsed}}<{{remaining}}, {{rate_fmt}}, {backend_label}]')
            chunk_iterator = tqdm(
                chunk_iterator,
                desc=f"     Computing features",
                unit="chunk",
                total=num_chunks,
                bar_format=bar_fmt
            )
        
        # Import GPU computer for helper functions
        from .features_gpu import GPUFeatureComputer
        gpu_computer = GPUFeatureComputer(use_gpu=self.use_gpu)
        
        # Cache points_cpu if using CPU KNN
        points_cpu = None if (self.use_cuml and cuNearestNeighbors is not None) else self._to_cpu(points_gpu)
        
        # OPTIMIZATION #1: Persistent GPU arrays - cache normals on GPU to avoid repeated uploads
        normals_gpu_persistent = None
        
        # Process chunks
        for chunk_idx in chunk_iterator:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, N)
            
            # Query neighbors for this chunk (REUSE TREE!)
            if self.use_cuml and cuNearestNeighbors is not None:
                # GPU query
                query_points = points_gpu[start_idx:end_idx]
                distances, global_indices_gpu = knn.kneighbors(query_points)
                # Keep indices on GPU to avoid slow transfer!
                del query_points, distances
                
                # Compute normals on GPU (REUSE neighbor indices!)
                chunk_normals = self._compute_normals_from_neighbors_gpu(
                    points_gpu, global_indices_gpu
                )
                normals[start_idx:end_idx] = self._to_cpu(chunk_normals)
                
                # Transfer indices to CPU only when needed
                global_indices = self._to_cpu(global_indices_gpu)
            else:
                # CPU query
                query_points = points_cpu[start_idx:end_idx]
                distances, global_indices = knn.kneighbors(query_points)
                del query_points, distances
                global_indices_gpu = global_indices
                
                # Compute normals for this chunk (REUSE neighbor indices!)
                chunk_normals = self._compute_normals_from_neighbors_gpu(
                    points_gpu if self.use_gpu else points, global_indices_gpu
                )
                normals[start_idx:end_idx] = self._to_cpu(chunk_normals) if self.use_gpu else chunk_normals
            
            # Compute curvature (vectorized, REUSE neighbor indices!)
            # OPTIMIZATION #1: Reuse persistent GPU array instead of re-uploading each chunk
            if self.use_gpu and cp is not None:
                # Upload normals to GPU once and cache for reuse across chunks
                if normals_gpu_persistent is None:
                    normals_gpu_persistent = self._to_gpu(normals)
                else:
                    # Update the cached GPU array with new normals for this chunk
                    normals_gpu_persistent[start_idx:end_idx] = chunk_normals
                
                # Fast GPU fancy indexing (reusing cached array)
                neighbor_normals_gpu = normals_gpu_persistent[global_indices_gpu]
                neighbor_normals = self._to_cpu(neighbor_normals_gpu)
                del neighbor_normals_gpu
                
                chunk_normals_cpu = self._to_cpu(chunk_normals)
                normals_expanded = chunk_normals_cpu[:, np.newaxis, :]
                normal_diff = neighbor_normals - normals_expanded
            else:
                # CPU fallback (slower)
                chunk_normals_cpu = self._to_cpu(chunk_normals)
                neighbor_normals = normals[global_indices]  # CPU fancy indexing
                normals_expanded = chunk_normals_cpu[:, np.newaxis, :]
                normal_diff = neighbor_normals - normals_expanded
            
            curv_norms = np.linalg.norm(normal_diff, axis=2)
            chunk_curvature = np.mean(curv_norms, axis=1).astype(np.float32)
            curvature[start_idx:end_idx] = chunk_curvature
            
            # Compute height
            chunk_points_cpu = self._to_cpu(points_gpu[start_idx:end_idx])
            chunk_classification = classification[start_idx:end_idx]
            chunk_height = gpu_computer.compute_height_above_ground(
                chunk_points_cpu, chunk_classification
            )
            height[start_idx:end_idx] = chunk_height
            
            # OPTIMIZED: Compute geometric features directly from neighbor indices
            # This avoids rebuilding KDTree for each chunk (major bottleneck!)
            # Pass points_gpu to use GPU for fast fancy indexing (10-100x speedup!)
            chunk_geo = self._compute_geometric_features_from_neighbors(
                points, global_indices, chunk_points_cpu, points_gpu=points_gpu
            )
            
            # Store geometric features
            for key in geo_features:
                if key in chunk_geo:
                    geo_features[key][start_idx:end_idx] = chunk_geo[key]
            
            # Compute verticality and horizontality from normals
            chunk_normals_for_vert = normals[start_idx:end_idx]
            verticality_chunk = gpu_computer.compute_verticality(
                chunk_normals_for_vert
            )
            geo_features['verticality'][start_idx:end_idx] = verticality_chunk
            
            # Horizontality = abs(nz) - how horizontal the surface is
            horizontality_chunk = np.abs(chunk_normals_for_vert[:, 2]).astype(np.float32)
            geo_features['horizontality'][start_idx:end_idx] = horizontality_chunk
            
            # === FACULTATIVE FEATURES: WALL AND ROOF SCORES ===
            # Wall score: High planarity + Vertical surface
            # Roof score: High planarity + Horizontal surface
            chunk_planarity = geo_features['planarity'][start_idx:end_idx]
            
            # Clean planarity and verticality/horizontality from NaN/Inf before computing scores
            chunk_planarity = np.nan_to_num(chunk_planarity, nan=0.0, posinf=1.0, neginf=0.0)
            verticality_chunk = np.nan_to_num(verticality_chunk, nan=0.0, posinf=1.0, neginf=0.0)
            horizontality_chunk = np.nan_to_num(horizontality_chunk, nan=0.0, posinf=1.0, neginf=0.0)
            
            wall_score_chunk = (chunk_planarity * verticality_chunk).astype(np.float32)
            roof_score_chunk = (chunk_planarity * horizontality_chunk).astype(np.float32)
            
            # Store cleaned values back
            geo_features['planarity'][start_idx:end_idx] = chunk_planarity
            geo_features['wall_score'][start_idx:end_idx] = wall_score_chunk
            geo_features['roof_score'][start_idx:end_idx] = roof_score_chunk
            
            # === ADVANCED FEATURES FOR FULL MODE (GPU-ACCELERATED) ===
            # Only compute if needed for the selected feature mode (use pre-computed flags)
            
            if compute_eigenvalues:
                # Compute eigenvalue features using GPU-accelerated helper method
                # Pass full arrays for neighbor lookup, but only compute for chunk
                eigenvalue_feats = self.compute_eigenvalue_features(
                    points, normals, global_indices, start_idx, end_idx
                )
                for key, values in eigenvalue_feats.items():
                    if key not in geo_features:
                        geo_features[key] = np.zeros(N, dtype=np.float32)
                    geo_features[key][start_idx:end_idx] = values
            
            if compute_architectural:
                # Compute architectural features using GPU-accelerated helper method
                # Pass full arrays for neighbor lookup, but only compute for chunk
                architectural_feats = self.compute_architectural_features(
                    points, normals, global_indices, start_idx, end_idx
                )
                for key, values in architectural_feats.items():
                    if key not in geo_features:
                        geo_features[key] = np.zeros(N, dtype=np.float32)
                    geo_features[key][start_idx:end_idx] = values
            
            if compute_density_advanced:
                # Compute density features using GPU-accelerated helper method
                # Pass cached points_gpu to avoid redundant transfers
                density_feats = self.compute_density_features(
                    points, global_indices, radius_2m=2.0, 
                    start_idx=start_idx, end_idx=end_idx,
                    points_gpu=points_gpu  # OPTIMIZED: reuse cached GPU array
                )
                for key, values in density_feats.items():
                    if key not in geo_features:
                        geo_features[key] = np.zeros(N, dtype=np.float32)
                    geo_features[key][start_idx:end_idx] = values
            
            # Cleanup (OPTIMIZED: minimal cleanup, only delete what was created)
            del chunk_normals, chunk_normals_cpu, global_indices_gpu, global_indices
            del chunk_geo, chunk_points_cpu, chunk_classification
            del verticality_chunk, horizontality_chunk, wall_score_chunk, roof_score_chunk
            
            # Conditionally delete advanced feature results if they were computed
            if 'eigenvalue_feats' in locals():
                del eigenvalue_feats
            if 'architectural_feats' in locals():
                del architectural_feats
            if 'density_feats' in locals():
                del density_feats
            
            # Less frequent GPU cleanup since we're reusing tree
            if chunk_idx % 3 == 0:
                self._free_gpu_memory()
        
        # ========================================================================
        # PHASE 3: FINAL CLEANUP & VALIDATION
        # ========================================================================
        logger.info(f"  🧹 Phase 3/3: Cleaning up & validating...")
        
        # Cleanup KDTree and GPU memory
        del knn, points_gpu
        if points_cpu is not None:
            del points_cpu
        # OPTIMIZATION #1: Cleanup persistent GPU array
        if normals_gpu_persistent is not None:
            del normals_gpu_persistent
        self._free_gpu_memory()
        
        # === FINAL VALIDATION: Clean all geometric features from NaN/Inf artifacts ===
        # This fixes line/dash artifacts in planarity, linearity, and derived features
        features_to_clean = ['planarity', 'linearity', 'sphericity', 'anisotropy', 
                             'roughness', 'omnivariance', 'curvature', 'change_curvature',
                             'verticality', 'horizontality', 'wall_score', 'roof_score',
                             'edge_strength', 'corner_likelihood', 'surface_roughness']
        
        for feat_name in features_to_clean:
            if feat_name in geo_features:
                geo_features[feat_name] = np.nan_to_num(
                    geo_features[feat_name], 
                    nan=0.0, 
                    posinf=1.0, 
                    neginf=0.0
                ).astype(np.float32)
        
        # Clean normals and curvature
        normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        height = np.nan_to_num(height, nan=0.0).astype(np.float32)
        
        # Filter features based on mode if specified
        if feature_set is not None:
            filtered_features = {}
            
            # Add features that are in the feature set
            for feat_name in geo_features.keys():
                if feat_name in feature_set:
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
            
            geo_features = filtered_features
            logger.info(
                f"✓ Features computed and filtered for mode '{mode}': "
                f"{len(geo_features)} features selected, {N:,} points, {num_chunks} chunks processed"
            )
        else:
            # Log completion statistics (full mode)
            total_features = len(geo_features) + 3  # +3 for normals, curvature, height
            logger.info(
                f"✓ All features computed successfully: "
                f"{total_features} feature types, {N:,} points, {num_chunks} chunks processed"
            )
        
        return normals, curvature, height, geo_features


def compute_all_features_gpu_chunked(
    points: np.ndarray,
    classification: np.ndarray,
    k: int = 10,
    chunk_size: int = 5_000_000,
    vram_limit_gb: float = 8.0,
    radius: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Convenience function for GPU-chunked feature computation.
    
    This function provides a simple interface matching the existing API
    while using GPU acceleration with chunked processing.
    
    Args:
        points: [N, 3] point coordinates
        classification: [N] ASPRS classification codes
        k: number of neighbors
        chunk_size: points per chunk (default: 5M)
        vram_limit_gb: VRAM limit in GB (default: 8.0)
        radius: search radius in meters (optional)
        
    Returns:
        normals, curvature, height, geo_features
        
    Example:
        >>> normals, curv, height, geo = compute_all_features_gpu_chunked(
        ...     points, classification, k=10, chunk_size=5_000_000
        ... )
    """
    computer = GPUChunkedFeatureComputer(
        chunk_size=chunk_size,
        vram_limit_gb=vram_limit_gb,
        use_gpu=True
    )
    
    return computer.compute_all_features_chunked(
        points, classification, k=k, radius=radius
    )


def compute_eigenvalue_features(
    points: np.ndarray,
    normals: np.ndarray,
    neighbors_indices: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Wrapper for GPU-chunked eigenvalue feature computation.
    
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
    computer = GPUChunkedFeatureComputer(use_gpu=True)
    return computer.compute_eigenvalue_features(points, normals, neighbors_indices)


def compute_architectural_features(
    points: np.ndarray,
    normals: np.ndarray,
    neighbors_indices: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Wrapper for GPU-chunked architectural feature computation.
    
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
    computer = GPUChunkedFeatureComputer(use_gpu=True)
    return computer.compute_architectural_features(points, normals, neighbors_indices)


def compute_density_features(
    points: np.ndarray,
    neighbors_indices: np.ndarray,
    radius_2m: float = 2.0
) -> Dict[str, np.ndarray]:
    """
    Wrapper for GPU-chunked density feature computation.
    
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
    computer = GPUChunkedFeatureComputer(use_gpu=True)
    return computer.compute_density_features(points, neighbors_indices, radius_2m)

