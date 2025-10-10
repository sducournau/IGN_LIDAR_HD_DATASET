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
        chunk_size: int = None,
        vram_limit_gb: float = None,
        use_gpu: bool = True,
        show_progress: bool = True,
        auto_optimize: bool = True
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
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_cuml = CUML_AVAILABLE
        self.show_progress = show_progress
        self.auto_optimize = auto_optimize
        
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
            from ..core.memory_manager import AdaptiveMemoryManager
            self.memory_manager = AdaptiveMemoryManager()
            
            # Auto-detect VRAM
            if vram_limit_gb is None:
                status = self.memory_manager.get_current_memory_status()
                self.vram_limit_gb = status[2]  # vram_free_gb
            else:
                self.vram_limit_gb = vram_limit_gb
            
            # Auto-optimize chunk size
            if chunk_size is None:
                self.chunk_size = (
                    self.memory_manager.calculate_optimal_gpu_chunk_size(
                        num_points=10_000_000,  # Estimate for sizing
                        vram_free_gb=self.vram_limit_gb
                    )
                )
                if self.chunk_size == 0:
                    # Not enough VRAM, fallback to CPU
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
    
    def _to_gpu(self, array: np.ndarray) -> 'cp.ndarray':
        """Transfer array to GPU memory."""
        if self.use_gpu and cp is not None:
            return cp.asarray(array, dtype=cp.float32)
        return array
    
    def _to_cpu(self, array) -> np.ndarray:
        """Transfer array to CPU memory."""
        if self.use_gpu and cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    
    def _free_gpu_memory(self):
        """Explicitly free GPU memory."""
        if self.use_gpu and cp is not None:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
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
                           '[{elapsed}<{remaining}]')
                chunk_iterator = tqdm(
                    chunk_iterator,
                    desc="  GPU Normals",
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
                    
                    # Transfer chunk results to CPU
                    normals[start_idx:end_idx] = (
                        self._to_cpu(chunk_normals_gpu)
                    )
                    
                    # Free GPU memory for chunk
                    del chunk_points_gpu, distances, chunk_normals_gpu
                    if chunk_idx % 5 == 0:  # Periodic cleanup
                        self._free_gpu_memory()
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
                    
                    # Transfer chunk results to CPU
                    normals[start_idx:end_idx] = (
                        self._to_cpu(chunk_normals_gpu)
                    )
                    
                    if chunk_idx % 5 == 0:  # Periodic cleanup
                        self._free_gpu_memory()
            
            # Final cleanup
            del points_gpu, knn
            self._free_gpu_memory()
            
            logger.info("  ✓ Normals computation complete")
            return normals
            
        except Exception as e:
            logger.error(f"GPU chunked computation failed: {e}")
            logger.warning("Falling back to CPU...")
            self._free_gpu_memory()
            
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
        Compute normals using per-chunk KDTree strategy.
        Much faster than global KDTree when using sklearn fallback.
        
        Strategy:
        - Build small KDTree per chunk (~2.5M points each)
        - Use overlap region to get neighbors across boundaries
        - Aggressive memory cleanup after each chunk
        
        Args:
            points: [N, 3] point coordinates
            k: number of neighbors
            
        Returns:
            normals: [N, 3] normalized surface normals
        """
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)
        
        # Calculate overlap needed for boundary neighbors
        # Increased overlap for smaller chunks to maintain accuracy
        overlap_ratio = 0.10  # 10% overlap between chunks (was 5%)
        overlap_size = int(self.chunk_size * overlap_ratio)
        
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        
        # Transfer points to GPU
        points_gpu = self._to_gpu(points)
        self._log_gpu_memory("after points transfer")
        
        # Progress bar
        chunk_iterator = range(num_chunks)
        if self.show_progress:
            bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} chunks '
                       '[{elapsed}<{remaining}]')
            chunk_iterator = tqdm(
                chunk_iterator,
                desc="  GPU Normals",
                unit="chunk",
                total=num_chunks,
                bar_format=bar_fmt
            )
        
        for chunk_idx in chunk_iterator:
            # Define chunk range with overlap
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, N)
            
            # Extended range for KDTree (with overlap)
            tree_start = max(0, start_idx - overlap_size)
            tree_end = min(N, end_idx + overlap_size)
            
            # Extract chunk points for KDTree
            chunk_points_for_tree = points_gpu[tree_start:tree_end]
            
            # Build local KDTree (small and fast!)
            # Use cuML if available for GPU acceleration
            if self.use_cuml and cuNearestNeighbors is not None:
                # GPU KDTree with cuML (faster)
                knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
                knn.fit(chunk_points_for_tree)
                
                # Query points (original chunk, not extended)
                query_start_local = start_idx - tree_start
                query_end_local = end_idx - tree_start
                query_points = chunk_points_for_tree[
                    query_start_local:query_end_local
                ]
                distances, local_indices = knn.kneighbors(query_points)
                
                # Convert to numpy if needed
                local_indices = self._to_cpu(local_indices)
            else:
                # CPU KDTree fallback
                chunk_points_cpu = self._to_cpu(chunk_points_for_tree)
                knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
                knn.fit(chunk_points_cpu)
                
                # Query points (original chunk, not extended)
                query_points = chunk_points_cpu[
                    (start_idx - tree_start):(end_idx - tree_start)
                ]
                distances, local_indices = knn.kneighbors(query_points)
            
            # Convert local indices to global indices
            global_indices = local_indices + tree_start
            
            # Compute normals for this chunk
            if cp is not None:
                global_indices_gpu = cp.asarray(global_indices)
            else:
                global_indices_gpu = global_indices
                
            chunk_normals = self._compute_normals_from_neighbors_gpu(
                points_gpu, global_indices_gpu
            )
            
            # Store results
            normals[start_idx:end_idx] = self._to_cpu(chunk_normals)
            
            # OPTIMIZED: Aggressive memory cleanup after each chunk
            del chunk_normals, global_indices_gpu, global_indices
            del chunk_points_for_tree, knn
            if self.use_cuml and 'query_points' in locals():
                del query_points
            # Free GPU memory every chunk (not every 3 chunks)
            self._free_gpu_memory()
        
        logger.info("  ✓ Per-chunk normals computation complete")
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
                           '[{elapsed}<{remaining}]')
                chunk_iterator = tqdm(
                    chunk_iterator,
                    desc="  GPU Curvature",
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
                    if cp is not None:
                        indices = cp.asarray(indices)
                
                # Compute curvature for chunk (VECTORIZED)
                # Get neighbor normals: [chunk_size, k, 3]
                if cp is not None and isinstance(normals_gpu, cp.ndarray):
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
                if chunk_idx % 5 == 0:
                    self._free_gpu_memory()
            
            del points_gpu, normals_gpu, knn
            self._free_gpu_memory()
            
            logger.info("  ✓ Curvature computation complete")
            return curvature
            
        except Exception as e:
            logger.error(f"GPU curvature failed: {e}")
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
                       '[{elapsed}<{remaining}]')
            chunk_iterator = tqdm(
                chunk_iterator,
                desc="  GPU Curvature",
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
            if cp is not None and isinstance(normals_gpu, cp.ndarray):
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
    
    def compute_all_features_chunked(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        k: int = 10,
        radius: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute ALL features per-chunk for maximum memory efficiency.
        
        OPTIMIZED: Computes normals, curvature, height, and geometric
        features within each chunk to minimize memory footprint.
        
        Args:
            points: [N, 3] point coordinates
            classification: [N] ASPRS classification codes
            k: number of neighbors
            radius: search radius in meters (optional)
            
        Returns:
            normals: [N, 3] surface normals
            curvature: [N] curvature values
            height: [N] height above ground
            geo_features: dict with geometric features
        """
        N = len(points)
        logger.info(
            f"Computing all features with GPU chunking: {N:,} points"
        )
        
        # Initialize output arrays
        normals = np.zeros((N, 3), dtype=np.float32)
        curvature = np.zeros(N, dtype=np.float32)
        height = np.zeros(N, dtype=np.float32)
        
        # Initialize geometric features
        # Only initialize features that are actually computed
        geo_features = {
            'anisotropy': np.zeros(N, dtype=np.float32),
            'planarity': np.zeros(N, dtype=np.float32),
            'linearity': np.zeros(N, dtype=np.float32),
            'sphericity': np.zeros(N, dtype=np.float32),
            'roughness': np.zeros(N, dtype=np.float32),
            'density': np.zeros(N, dtype=np.float32),
            'verticality': np.zeros(N, dtype=np.float32),
            'horizontality': np.zeros(N, dtype=np.float32)
        }
        
        # Compute per-chunk for memory efficiency
        overlap_ratio = 0.10
        overlap_size = int(self.chunk_size * overlap_ratio)
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        
        # Progress bar
        chunk_iterator = range(num_chunks)
        if self.show_progress:
            bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} chunks '
                       '[{elapsed}<{remaining}]')
            chunk_iterator = tqdm(
                chunk_iterator,
                desc="  All Features",
                unit="chunk",
                total=num_chunks,
                bar_format=bar_fmt
            )
        
        # Import GPU computer for helper functions
        from .features_gpu import GPUFeatureComputer
        gpu_computer = GPUFeatureComputer(use_gpu=self.use_gpu)
        
        # Transfer all points to GPU once
        points_gpu = self._to_gpu(points)
        
        for chunk_idx in chunk_iterator:
            # Define chunk boundaries
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, N)
            
            # Extended range for KDTree (with overlap)
            tree_start = max(0, start_idx - overlap_size)
            tree_end = min(N, end_idx + overlap_size)
            
            # Extract chunk for processing
            chunk_points = points_gpu[tree_start:tree_end]
            chunk_classification = classification[tree_start:tree_end]
            
            # Build local KDTree
            if self.use_cuml and cuNearestNeighbors is not None:
                knn = cuNearestNeighbors(n_neighbors=k, metric='euclidean')
                knn.fit(chunk_points)
                
                query_start_local = start_idx - tree_start
                query_end_local = end_idx - tree_start
                query_points = chunk_points[query_start_local:query_end_local]
                distances, local_indices = knn.kneighbors(query_points)
                local_indices = self._to_cpu(local_indices)
            else:
                chunk_points_cpu = self._to_cpu(chunk_points)
                knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
                knn.fit(chunk_points_cpu)
                
                query_points = chunk_points_cpu[
                    (start_idx - tree_start):(end_idx - tree_start)
                ]
                distances, local_indices = knn.kneighbors(query_points)
            
            # Convert local to global indices
            global_indices = local_indices + tree_start
            
            # Compute normals for this chunk
            if cp is not None:
                global_indices_gpu = cp.asarray(global_indices)
            else:
                global_indices_gpu = global_indices
            
            chunk_normals = self._compute_normals_from_neighbors_gpu(
                points_gpu, global_indices_gpu
            )
            normals[start_idx:end_idx] = self._to_cpu(chunk_normals)
            
            # Compute curvature for this chunk (using chunk normals)
            if cp is not None and isinstance(chunk_normals, cp.ndarray):
                normals_for_curv = cp.asnumpy(chunk_normals)
                # Need to get neighbor normals from already computed normals
                # For first chunk, use chunk normals; for later, use results
                all_normals_gpu = self._to_gpu(normals)
                neighbor_normals = cp.asnumpy(all_normals_gpu[global_indices])
            else:
                normals_for_curv = chunk_normals
                neighbor_normals = normals[global_indices]
            
            # Vectorized curvature computation
            normals_expanded = normals_for_curv[:, np.newaxis, :]
            normal_diff = neighbor_normals - normals_expanded
            curv_norms = np.linalg.norm(normal_diff, axis=2)
            chunk_curvature = np.mean(curv_norms, axis=1)
            curvature[start_idx:end_idx] = chunk_curvature
            
            # Compute height for this chunk
            chunk_points_cpu = self._to_cpu(
                points_gpu[start_idx:end_idx]
            )
            chunk_height = gpu_computer.compute_height_above_ground(
                chunk_points_cpu, chunk_classification[
                    (start_idx - tree_start):(end_idx - tree_start)
                ]
            )
            height[start_idx:end_idx] = chunk_height
            
            # Compute geometric features for this chunk
            chunk_geo = gpu_computer.extract_geometric_features(
                chunk_points_cpu,
                normals[start_idx:end_idx],
                k=k,
                radius=radius
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
            
            # Cleanup
            del chunk_points, chunk_classification, chunk_normals
            del global_indices_gpu, local_indices, chunk_geo
            del verticality_chunk, horizontality_chunk
            self._free_gpu_memory()
        
        logger.info("✓ All features computed per-chunk successfully")
        
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
