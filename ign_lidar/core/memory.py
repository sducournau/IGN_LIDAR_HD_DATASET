"""
Memory Management Module for LiDAR Processing

This module consolidates memory management functionality from:
- memory_manager.py: Adaptive memory management with real-time monitoring
- memory_utils.py: CLI-oriented memory utilities and worker optimization
- modules/memory.py: Memory cleanup and GPU cache management

Version: 2.0.0 (Consolidated)
"""

import gc
import logging
import numpy as np
import psutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Check if psutil is available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory optimization disabled")


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class MemoryConfig:
    """Memory configuration for processing."""
    available_ram_gb: float
    swap_usage_percent: float
    chunk_size: int
    max_workers: int
    use_gpu: bool
    vram_available_gb: float = 0.0


# ==============================================================================
# Memory Cleanup Functions (from modules/memory.py)
# ==============================================================================

def aggressive_memory_cleanup() -> None:
    """
    Aggressive memory cleanup to prevent OOM.
    
    Clears all caches and forces garbage collection across different
    computation backends (CPU, CUDA, CuPy).
    
    This function:
    1. Forces Python garbage collection
    2. Clears PyTorch CUDA cache if available
    3. Clears CuPy memory pools if available
    4. Performs final garbage collection
    
    Safe to call even if GPU libraries are not installed.
    """
    gc.collect()
    
    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache")
    except (ImportError, RuntimeError):
        pass
    
    # Clear CuPy cache if available
    try:
        import cupy as cp
        # Check if CUDA is actually available before trying to free memory
        if cp.cuda.is_available():
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            logger.debug("Cleared CuPy memory pools")
    except (ImportError, AttributeError, Exception) as e:
        # Catch all exceptions including CUDA runtime errors
        logger.debug(f"Could not clear CuPy memory pools: {e}")
        pass
    
    gc.collect()


def clear_gpu_cache() -> bool:
    """
    Clear GPU memory cache for PyTorch and CuPy.
    
    Returns:
        bool: True if any GPU cache was cleared, False otherwise
    """
    cleared = False
    
    # Clear PyTorch CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            cleared = True
            logger.debug("Cleared PyTorch CUDA cache")
    except (ImportError, RuntimeError):
        pass
    
    # Clear CuPy cache
    try:
        import cupy as cp
        # Check if CUDA is actually available before trying to free memory
        if cp.cuda.is_available():
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            cleared = True
            logger.debug("Cleared CuPy memory pools")
    except (ImportError, AttributeError, Exception) as e:
        # Catch all exceptions including CUDA runtime errors
        logger.debug(f"Could not clear CuPy memory pools: {e}")
        pass
    
    return cleared


def estimate_memory_usage(
    num_points: int, 
    num_features: int = 10,
    include_rgb: bool = False,
    dtype_size: int = 4
) -> float:
    """
    Estimate memory usage for a point cloud in MB.
    
    Args:
        num_points: Number of points in the cloud
        num_features: Number of features per point (default: 10)
        include_rgb: Whether RGB data is included
        dtype_size: Size of data type in bytes (4 for float32, 8 for float64)
        
    Returns:
        float: Estimated memory usage in megabytes
    """
    # Base features (xyz, classification, features)
    base_mem = num_points * (3 + 1 + num_features) * dtype_size
    
    # RGB channels
    rgb_mem = num_points * 3 * dtype_size if include_rgb else 0
    
    # Total in MB
    total_mb = (base_mem + rgb_mem) / (1024 * 1024)
    
    return total_mb


def check_available_memory() -> Optional[float]:
    """
    Check available system memory in GB.
    
    Returns:
        float: Available memory in GB, or None if cannot determine
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        return available_gb
    except ImportError:
        logger.warning("psutil not available, cannot check memory")
        return None


def check_gpu_memory() -> Optional[float]:
    """
    Check available GPU memory in GB.
    
    Returns:
        float: Available GPU memory in GB, or None if no GPU or cannot determine
    """
    # Try PyTorch first
    try:
        import torch
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            return free_mem / (1024 ** 3)
    except (ImportError, RuntimeError):
        pass
    
    # Try CuPy
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        # CuPy doesn't directly give free memory, but we can get used
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()
        free_bytes = total_bytes - used_bytes
        return free_bytes / (1024 ** 3)
    except (ImportError, AttributeError):
        pass
    
    return None


# ==============================================================================
# System Memory Information (from memory_utils.py)
# ==============================================================================

def get_system_memory_info() -> dict:
    """
    Get current system memory information.
    
    Returns:
        dict: Dictionary containing:
            - available_gb: Available RAM in GB
            - total_gb: Total RAM in GB
            - percent_used: Percentage of RAM used
            - swap_percent: Percentage of swap used
            - has_pressure: Boolean indicating if system is under memory pressure
    """
    if not PSUTIL_AVAILABLE:
        return {
            'available_gb': 8.0,  # Conservative default
            'total_gb': 16.0,
            'percent_used': 50.0,
            'swap_percent': 0.0,
            'has_pressure': False
        }
    
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)
    swap_percent = swap.percent
    
    # Consider system under pressure if swap > 50% or RAM > 85%
    has_pressure = swap_percent > 50 or mem.percent > 85
    
    return {
        'available_gb': available_gb,
        'total_gb': total_gb,
        'percent_used': mem.percent,
        'swap_percent': swap_percent,
        'has_pressure': has_pressure
    }


def estimate_memory_per_worker(
    max_file_size_mb: float,
    mode: str = 'core',
    use_gpu: bool = False
) -> float:
    """
    Estimate RAM required per worker based on file size and processing mode.
    
    Args:
        max_file_size_mb: Maximum file size in MB
        mode: Processing mode ('core' or 'full')
        use_gpu: Whether GPU processing is enabled
    
    Returns:
        float: Estimated RAM in GB per worker
    """
    # Base memory for file loading (roughly 2-3x file size)
    base_memory_gb = (max_file_size_mb / 1024) * 2.5
    
    # Additional memory for feature computation
    if mode == 'full':
        # Full mode: all features + KDTree construction
        feature_overhead_gb = 3.0
    else:
        # Core mode: basic geometric features
        feature_overhead_gb = 1.5
    
    # GPU adds memory overhead for data transfer
    if use_gpu:
        gpu_overhead_gb = 0.5
    else:
        gpu_overhead_gb = 0.0
    
    total_gb = base_memory_gb + feature_overhead_gb + gpu_overhead_gb
    
    # Ensure minimum of 2GB per worker
    return max(2.0, total_gb)


def calculate_optimal_workers(
    num_files: int,
    file_sizes_mb: List[float],
    mode: str = 'core',
    use_gpu: bool = False,
    requested_workers: Optional[int] = None,
    min_workers: int = 1,
    max_workers: int = 16
) -> Tuple[int, dict]:
    """
    Calculate optimal number of workers based on system resources and file sizes.
    
    Args:
        num_files: Number of files to process
        file_sizes_mb: List of file sizes in MB
        mode: Processing mode ('core' or 'full')
        use_gpu: Whether GPU processing is enabled
        requested_workers: User-requested worker count (None for auto)
        min_workers: Minimum number of workers
        max_workers: Maximum number of workers
    
    Returns:
        Tuple of (optimal_workers, info_dict) where info_dict contains:
            - memory_per_worker_gb: Estimated memory per worker
            - max_file_size_mb: Maximum file size
            - available_memory_gb: Available system memory
            - recommendation_reason: String explaining the recommendation
    """
    # Get system memory
    mem_info = get_system_memory_info()
    available_gb = mem_info['available_gb']
    has_pressure = mem_info['has_pressure']
    
    # Analyze file sizes
    max_file_size_mb = max(file_sizes_mb) if file_sizes_mb else 100
    avg_file_size_mb = sum(file_sizes_mb) / len(file_sizes_mb) if file_sizes_mb else 100
    
    # Estimate memory per worker
    mem_per_worker_gb = estimate_memory_per_worker(
        max_file_size_mb, mode, use_gpu
    )
    
    # Calculate maximum safe workers based on available memory
    # Reserve 2GB for system
    max_memory_workers = max(1, int((available_gb - 2.0) / mem_per_worker_gb))
    
    # GPU requires single worker (CUDA context limitation)
    if use_gpu:
        optimal_workers = 1
        reason = "GPU mode requires single worker (CUDA context limitation)"
    
    # System under memory pressure
    elif has_pressure:
        optimal_workers = 1
        reason = f"System under memory pressure (swap: {mem_info['swap_percent']:.0f}%)"
    
    # Large files require fewer workers
    elif max_file_size_mb > 500:
        optimal_workers = min(3, max_memory_workers)
        reason = f"Large files detected ({max_file_size_mb:.0f}MB) - limiting concurrency"
    
    elif max_file_size_mb > 300:
        optimal_workers = min(4, max_memory_workers)
        reason = f"Medium-large files ({max_file_size_mb:.0f}MB) - moderate concurrency"
    
    # Use memory-based calculation
    else:
        optimal_workers = max_memory_workers
        reason = f"Based on available memory ({available_gb:.1f}GB)"
    
    # Apply bounds
    optimal_workers = max(min_workers, min(optimal_workers, max_workers))
    
    # If user requested specific count, compare and warn if different
    if requested_workers is not None:
        if requested_workers > optimal_workers:
            reason = f"{reason} (user requested {requested_workers}, recommending {optimal_workers})"
        optimal_workers = min(requested_workers, optimal_workers)
    
    info = {
        'memory_per_worker_gb': mem_per_worker_gb,
        'max_file_size_mb': max_file_size_mb,
        'avg_file_size_mb': avg_file_size_mb,
        'available_memory_gb': available_gb,
        'max_memory_workers': max_memory_workers,
        'recommendation_reason': reason,
        'has_memory_pressure': has_pressure
    }
    
    return optimal_workers, info


def calculate_batch_size(
    num_workers: int,
    max_file_size_mb: float,
    mode: str = 'core'
) -> int:
    """
    Calculate appropriate batch size for parallel processing.
    
    Args:
        num_workers: Number of worker processes
        max_file_size_mb: Maximum file size in MB
        mode: Processing mode ('core' or 'full')
    
    Returns:
        int: Recommended batch size
    """
    if mode == 'full':
        # Full mode is memory intensive
        if max_file_size_mb > 300:
            # Very large files: sequential processing
            return 1
        elif max_file_size_mb > 200:
            # Large files: limited concurrency
            return max(1, num_workers // 2)
        else:
            # Smaller files: use full worker count
            return num_workers
    else:
        # Core mode is less memory intensive
        if max_file_size_mb < 200:
            # Small files: can process multiple batches at once
            return num_workers * 2
        else:
            # Larger files: one batch at a time
            return num_workers


def log_memory_configuration(
    num_files: int,
    num_workers: int,
    worker_info: dict,
    mode: str = 'core'
) -> None:
    """
    Log memory configuration details for transparency.
    
    Args:
        num_files: Number of files to process
        num_workers: Number of workers being used
        worker_info: Info dict from calculate_optimal_workers()
        mode: Processing mode
    """
    logger.info("Memory Configuration:")
    logger.info(f"  Files to process: {num_files}")
    logger.info(f"  Worker processes: {num_workers}")
    logger.info(f"  Processing mode: {mode.upper()}")
    logger.info(
        f"  Max file size: {worker_info['max_file_size_mb']:.0f} MB"
    )
    logger.info(
        f"  Avg file size: {worker_info['avg_file_size_mb']:.0f} MB"
    )
    logger.info(
        f"  Available memory: {worker_info['available_memory_gb']:.1f} GB"
    )
    logger.info(
        f"  Memory per worker: ~{worker_info['memory_per_worker_gb']:.1f} GB"
    )
    
    if worker_info['has_memory_pressure']:
        logger.warning("  ⚠️  System is under memory pressure")
    
    logger.info(f"  Reason: {worker_info['recommendation_reason']}")


def analyze_file_sizes(
    files: List[Path]
) -> Tuple[List[Tuple[Path, int]], dict]:
    """
    Analyze file sizes and provide statistics.
    
    Args:
        files: List of file paths
    
    Returns:
        Tuple of (files_with_sizes, stats) where:
            - files_with_sizes: List of (Path, size_bytes) tuples
            - stats: Dict with size statistics
    """
    files_with_size = [(f, f.stat().st_size) for f in files]
    
    if not files_with_size:
        return [], {
            'max_mb': 0,
            'min_mb': 0,
            'avg_mb': 0,
            'total_mb': 0,
            'count': 0
        }
    
    sizes_mb = [size / (1024**2) for _, size in files_with_size]
    
    stats = {
        'max_mb': max(sizes_mb),
        'min_mb': min(sizes_mb),
        'avg_mb': sum(sizes_mb) / len(sizes_mb),
        'total_mb': sum(sizes_mb),
        'count': len(files_with_size)
    }
    
    return files_with_size, stats


def sort_files_by_size(
    files_with_size: List[Tuple[Path, int]],
    reverse: bool = False
) -> List[Path]:
    """
    Sort files by size.
    
    Args:
        files_with_size: List of (Path, size_bytes) tuples
        reverse: If True, sort largest first; if False, smallest first
    
    Returns:
        List[Path]: Sorted list of file paths
    """
    sorted_files = sorted(files_with_size, key=lambda x: x[1], reverse=reverse)
    return [f for f, _ in sorted_files]


def check_gpu_memory_available() -> Tuple[bool, Optional[float]]:
    """
    Check if GPU is available and get available memory.
    
    Returns:
        Tuple of (is_available, memory_gb):
            - is_available: Whether GPU is available
            - memory_gb: Available GPU memory in GB (None if not available)
    """
    try:
        import cupy as cp
        
        # Get GPU memory info
        mempool = cp.get_default_memory_pool()
        total_bytes = cp.cuda.Device().mem_info[1]
        used_bytes = mempool.used_bytes()
        available_bytes = total_bytes - used_bytes
        
        available_gb = available_bytes / (1024**3)
        
        return True, available_gb
    except Exception:
        return False, None


# ==============================================================================
# Adaptive Memory Manager (from memory_manager.py)
# ==============================================================================

class AdaptiveMemoryManager:
    """
    Adaptive memory manager with real-time monitoring.
    
    Features:
    - Real-time RAM and swap monitoring
    - Dynamic chunk size adjustment based on memory pressure
    - Worker count optimization
    - GPU memory management
    - Predictive memory allocation
    
    Example:
        >>> manager = AdaptiveMemoryManager()
        >>> config = manager.get_optimal_config(
        ...     num_points=17_000_000,
        ...     num_augmentations=2,
        ...     mode='full'
        ... )
        >>> print(f"Chunk size: {config.chunk_size:,}")
    """
    
    # Memory estimates per point (bytes)
    BYTES_PER_POINT = {
        'core': 200,      # Basic features
        'full': 350,      # Full features with geometric
    }
    
    # Augmentation memory multiplier
    AUGMENTATION_MULTIPLIER = 3.5  # Peak memory during augmentation
    
    # INTELLIGENT AUTO-SCALING: Adaptive safety margins
    # Lower margins when more memory available = better utilization
    RAM_SAFETY_MARGIN_HIGH = 0.15   # High RAM (>32GB): use more
    RAM_SAFETY_MARGIN_MED = 0.20    # Medium RAM (16-32GB): balanced
    RAM_SAFETY_MARGIN_LOW = 0.30    # Low RAM (<16GB): conservative
    
    VRAM_SAFETY_MARGIN_HIGH = 0.10  # High VRAM (>12GB): use more
    VRAM_SAFETY_MARGIN_MED = 0.15   # Medium VRAM (8-12GB): balanced
    VRAM_SAFETY_MARGIN_LOW = 0.25   # Low VRAM (<8GB): conservative
    
    # Default safety margin
    RAM_SAFETY_MARGIN = RAM_SAFETY_MARGIN_MED
    
    def __init__(
        self,
        min_chunk_size: int = 1_000_000,
        max_chunk_size: int = 20_000_000,
        enable_gpu: bool = True
    ):
        """
        Initialize adaptive memory manager.
        
        Args:
            min_chunk_size: Minimum chunk size (default: 1M)
            max_chunk_size: Maximum chunk size (default: 20M)
            enable_gpu: Enable GPU memory management
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.enable_gpu = enable_gpu
        
        # Initial system check
        self._log_system_resources()
    
    def _log_system_resources(self):
        """Log available system resources."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        logger.info("=" * 60)
        logger.info("System Resources:")
        logger.info(f"  RAM: {mem.total / (1024**3):.1f} GB total")
        logger.info(
            f"  RAM Available: {mem.available / (1024**3):.1f} GB "
            f"({mem.percent:.1f}% used)"
        )
        logger.info(
            f"  Swap: {swap.total / (1024**3):.1f} GB "
            f"({swap.percent:.1f}% used)"
        )
        
        if self.enable_gpu:
            try:
                import cupy as cp
                # Use runtime API instead of deprecated device.mem_info
                vram_free, vram_total = cp.cuda.runtime.memGetInfo()
                vram_free = vram_free / (1024**3)
                vram_total = vram_total / (1024**3)
                vram_used = vram_total - vram_free
                logger.info(
                    f"  VRAM: {vram_total:.1f} GB total, "
                    f"{vram_free:.1f} GB free "
                    f"({100 * vram_used / vram_total:.1f}% used)"
                )
            except Exception:
                logger.info("  VRAM: Not available")
        
        logger.info("=" * 60)
    
    def get_current_memory_status(self) -> Tuple[float, float, float]:
        """
        Get current memory status.
        
        Returns:
            Tuple of (available_ram_gb, swap_percent, vram_free_gb)
        """
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        available_ram_gb = mem.available / (1024**3)
        swap_percent = swap.percent
        
        vram_free_gb = 0.0
        if self.enable_gpu:
            try:
                import cupy as cp
                # Use runtime API instead of deprecated device.mem_info
                vram_free, _ = cp.cuda.runtime.memGetInfo()
                vram_free_gb = vram_free / (1024**3)
            except Exception:
                pass
        
        return available_ram_gb, swap_percent, vram_free_gb
    
    def estimate_memory_needed(
        self,
        num_points: int,
        mode: str = 'full',
        num_augmentations: int = 0
    ) -> float:
        """
        Estimate memory needed for processing.
        
        Args:
            num_points: Number of points to process
            mode: Processing mode ('core' or 'full')
            num_augmentations: Number of augmented versions
            
        Returns:
            Estimated memory in GB
        """
        bytes_per_point = self.BYTES_PER_POINT.get(mode, 350)
        
        # Base memory for one version
        base_memory_bytes = num_points * bytes_per_point
        
        # Account for augmentations (versions processed sequentially)
        if num_augmentations > 0:
            # Peak memory: original + one augmented version
            total_memory_bytes = (
                base_memory_bytes * self.AUGMENTATION_MULTIPLIER
            )
        else:
            total_memory_bytes = base_memory_bytes
        
        return total_memory_bytes / (1024**3)
    
    def calculate_optimal_chunk_size(
        self,
        num_points: int,
        mode: str = 'full',
        num_augmentations: int = 0,
        target_memory_gb: Optional[float] = None
    ) -> int:
        """
        Calculate optimal chunk size based on available memory.
        
        Strategy:
        1. Get current available RAM
        2. Calculate memory per point
        3. Determine safe chunk size
        4. Apply augmentation penalty
        5. Clamp to min/max bounds
        
        Args:
            num_points: Total number of points
            mode: Processing mode
            num_augmentations: Number of augmented versions
            target_memory_gb: Target memory usage (None = auto)
            
        Returns:
            Optimal chunk size
        """
        available_ram_gb, swap_percent, _ = (
            self.get_current_memory_status()
        )
        
        # If swap is heavily used, be very conservative
        if swap_percent > 50:
            logger.warning(
                f"⚠️  High swap usage ({swap_percent:.1f}%) - "
                f"using minimum chunk size"
            )
            return self.min_chunk_size
        
        # Determine target memory
        if target_memory_gb is None:
            # Use available RAM minus safety margin
            target_memory_gb = (
                available_ram_gb * (1 - self.RAM_SAFETY_MARGIN)
            )
        
        # Calculate memory per point
        bytes_per_point = self.BYTES_PER_POINT.get(mode, 350)
        
        # Apply augmentation penalty
        if num_augmentations > 0:
            bytes_per_point = int(
                bytes_per_point * self.AUGMENTATION_MULTIPLIER
            )
        
        # Calculate chunk size
        chunk_size = int(
            (target_memory_gb * 1024**3) / bytes_per_point
        )
        
        # Clamp to bounds
        chunk_size = max(self.min_chunk_size, chunk_size)
        chunk_size = min(self.max_chunk_size, chunk_size)
        
        # If chunk size >= num_points, don't chunk
        if chunk_size >= num_points:
            logger.info(
                f"✓ No chunking needed: "
                f"{num_points:,} points fit in memory"
            )
            return num_points
        
        logger.info(
            f"✓ Optimal chunk size: {chunk_size:,} points "
            f"(~{chunk_size * bytes_per_point / (1024**3):.1f} GB per chunk)"
        )
        
        return chunk_size
    
    def calculate_optimal_workers(
        self,
        num_files: int,
        file_sizes_mb: list,
        mode: str = 'full'
    ) -> int:
        """
        Calculate optimal number of workers based on memory.
        
        Args:
            num_files: Number of files to process
            file_sizes_mb: List of file sizes in MB
            mode: Processing mode
            
        Returns:
            Optimal worker count
        """
        available_ram_gb, swap_percent, _ = (
            self.get_current_memory_status()
        )
        
        # If swap is heavily used, force single worker
        if swap_percent > 50:
            logger.warning(
                "⚠️  High swap usage - forcing single worker"
            )
            return 1
        
        # Memory per worker estimate (GB)
        gb_per_worker = 5.0 if mode == 'full' else 2.5
        
        # Get total RAM
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        
        # Get adaptive safety margin based on total RAM
        safety_margin = self.get_adaptive_safety_margin(
            total_ram_gb, 'RAM'
        )
        
        # Calculate safe worker count
        max_safe_workers = int(
            available_ram_gb * (1 - safety_margin)
            / gb_per_worker
        )
        
        # Limit by file sizes
        if file_sizes_mb:
            max_file_size_mb = max(file_sizes_mb)
            if max_file_size_mb > 500:
                max_safe_workers = min(max_safe_workers, 3)
            elif max_file_size_mb > 300:
                max_safe_workers = min(max_safe_workers, 4)
        
        # Limit by number of files
        optimal_workers = min(max_safe_workers, num_files)
        
        # Always at least 1
        optimal_workers = max(1, optimal_workers)
        
        logger.info(
            f"✓ Optimal worker count: {optimal_workers} "
            f"(~{gb_per_worker * optimal_workers:.1f} GB total)"
        )
        
        return optimal_workers
    
    def calculate_optimal_gpu_chunk_size(
        self,
        num_points: int,
        vram_free_gb: Optional[float],
        feature_mode: str = 'minimal',
        k_neighbors: int = 10
    ) -> int:
        """
        Calculate optimal GPU chunk size for reclassification processing.
        
        Strategy:
        1. Estimate GPU memory required per point
        2. Calculate safe chunk size based on available VRAM
        3. Apply reclassification-specific optimizations
        4. Ensure minimum viable chunk size
        
        Args:
            num_points: Total number of points to process
            vram_free_gb: Available VRAM in GB (None = auto-detect)
            feature_mode: Feature computation mode ('minimal', 'lod2', 'lod3', 'full')
            k_neighbors: Number of neighbors for computations
            
        Returns:
            Optimal GPU chunk size (0 if insufficient VRAM)
        """
        # Handle None VRAM value
        if vram_free_gb is None:
            try:
                status = self.get_current_memory_status()
                vram_free_gb = status[2] if len(status) > 2 else 8.0
            except Exception:
                vram_free_gb = 8.0  # Conservative default
        
        if vram_free_gb < 1.0:
            logger.warning(
                f"⚠️ Insufficient VRAM ({vram_free_gb:.1f}GB < 1.0GB required)"
            )
            return 0
        
        # GPU memory estimates per point (bytes) for different modes
        # These include neighbor indices, covariance matrices, and intermediate results
        GPU_BYTES_PER_POINT = {
            'minimal': 150,    # Basic features only
            'lod2': 220,      # LOD2 features
            'lod3': 280,      # LOD3 features  
            'full': 350,      # All features including architectural
        }
        
        # K-neighbors memory multiplier (higher k = more memory)
        k_multiplier = max(1.0, k_neighbors / 10.0)
        
        # Base memory per point
        base_bytes = GPU_BYTES_PER_POINT.get(feature_mode, 350)
        bytes_per_point = int(base_bytes * k_multiplier)
        
        # GPU-specific safety margins based on VRAM size
        if vram_free_gb >= 12.0:
            safety_margin = self.VRAM_SAFETY_MARGIN_HIGH
        elif vram_free_gb >= 8.0:
            safety_margin = self.VRAM_SAFETY_MARGIN_MED
        else:
            safety_margin = self.VRAM_SAFETY_MARGIN_LOW
        
        # Calculate safe VRAM usage
        safe_vram_gb = vram_free_gb * (1 - safety_margin)
        
        # Calculate chunk size
        chunk_size = int((safe_vram_gb * 1024**3) / bytes_per_point)
        
        # Apply bounds
        min_chunk = 100_000   # Minimum for efficiency
        max_chunk = 10_000_000  # Maximum to avoid CUSOLVER issues
        
        chunk_size = max(min_chunk, chunk_size)
        chunk_size = min(max_chunk, chunk_size)
        
        # If chunk size >= num_points, no chunking needed
        if chunk_size >= num_points:
            logger.info(
                f"✓ No GPU chunking needed: {num_points:,} points fit in VRAM"
            )
            return num_points
        
        logger.info(
            f"✓ Optimal GPU chunk size: {chunk_size:,} points "
            f"(~{chunk_size * bytes_per_point / (1024**3):.1f}GB per chunk, "
            f"{safe_vram_gb:.1f}GB safe VRAM)"
        )
        
        return chunk_size

    def calculate_optimal_eigh_batch_size(
        self,
        chunk_size: int,
        vram_free_gb: Optional[float]
    ) -> int:
        """
        Calculate optimal batch size for eigenvalue decomposition to avoid CUSOLVER errors.
        
        CuSOLVER has internal limits on batch sizes for eigenvalue computations.
        This function calculates a safe batch size based on available VRAM and
        empirical CUSOLVER limits.
        
        Args:
            chunk_size: Current chunk size being processed
            vram_free_gb: Available VRAM in GB (None = auto-detect)
            
        Returns:
            Safe batch size for cp.linalg.eigh operations
        """
        # Handle None VRAM value
        if vram_free_gb is None:
            try:
                status = self.get_current_memory_status()
                vram_free_gb = status[2] if len(status) > 2 else 8.0
            except Exception:
                vram_free_gb = 8.0  # Conservative default
        
        # Empirical CUSOLVER limits (conservative estimates)
        # These are based on testing and known CUSOLVER behavior
        if vram_free_gb >= 16.0:
            max_eigh_batch = 750_000    # High-end GPUs
        elif vram_free_gb >= 12.0:
            max_eigh_batch = 500_000    # Mid-range GPUs  
        elif vram_free_gb >= 8.0:
            max_eigh_batch = 300_000    # Entry-level GPUs
        else:
            max_eigh_batch = 150_000    # Low VRAM
        
        # Use smaller of chunk_size and max safe batch
        eigh_batch_size = min(chunk_size, max_eigh_batch)
        
        # Ensure minimum efficiency
        eigh_batch_size = max(10_000, eigh_batch_size)
        
        return eigh_batch_size

    def get_adaptive_safety_margin(
        self,
        total_memory_gb: float,
        memory_type: str = 'RAM'
    ) -> float:
        """
        Get adaptive safety margin based on total available memory.
        
        Higher memory systems can use lower safety margins for better utilization.
        
        Args:
            total_memory_gb: Total memory in GB
            memory_type: 'RAM' or 'VRAM'
            
        Returns:
            Safety margin (0.0 to 1.0)
        """
        if memory_type == 'VRAM':
            if total_memory_gb >= 12.0:
                return self.VRAM_SAFETY_MARGIN_HIGH
            elif total_memory_gb >= 8.0:
                return self.VRAM_SAFETY_MARGIN_MED
            else:
                return self.VRAM_SAFETY_MARGIN_LOW
        else:  # RAM
            if total_memory_gb >= 32.0:
                return self.RAM_SAFETY_MARGIN_HIGH
            elif total_memory_gb >= 16.0:
                return self.RAM_SAFETY_MARGIN_MED
            else:
                return self.RAM_SAFETY_MARGIN_LOW

    def get_optimal_config(
        self,
        num_points: int,
        num_augmentations: int = 0,
        mode: str = 'full',
        num_files: int = 1,
        file_sizes_mb: Optional[list] = None
    ) -> MemoryConfig:
        """
        Get complete optimal configuration for processing.
        
        This is the main method to use for configuration.
        
        Args:
            num_points: Number of points to process
            num_augmentations: Number of augmented versions
            mode: Processing mode ('core' or 'full')
            num_files: Number of files (for multi-file processing)
            file_sizes_mb: List of file sizes in MB
            
        Returns:
            MemoryConfig with optimal settings
            
        Example:
            >>> manager = AdaptiveMemoryManager()
            >>> config = manager.get_optimal_config(
            ...     num_points=17_000_000,
            ...     num_augmentations=2,
            ...     mode='full'
            ... )
        """
        logger.info(
            f"\n{'='*60}\n"
            f"Computing Optimal Configuration\n"
            f"{'='*60}"
        )
        logger.info(f"Points: {num_points:,}")
        logger.info(f"Augmentations: {num_augmentations}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Files: {num_files}")
        
        # Get current memory status
        available_ram_gb, swap_percent, vram_free_gb = (
            self.get_current_memory_status()
        )
        
        # Calculate optimal chunk size
        chunk_size = self.calculate_optimal_chunk_size(
            num_points=num_points,
            mode=mode,
            num_augmentations=num_augmentations
        )
        
        # Calculate optimal workers
        if file_sizes_mb is None:
            file_sizes_mb = []
        
        max_workers = self.calculate_optimal_workers(
            num_files=num_files,
            file_sizes_mb=file_sizes_mb,
            mode=mode
        )
        
        # Determine if GPU should be used
        use_gpu = False
        if self.enable_gpu and vram_free_gb > 4.0:
            # GPU available with sufficient VRAM
            if num_augmentations == 0 and chunk_size >= num_points:
                # Can use GPU without chunking
                use_gpu = True
                logger.info("✓ GPU enabled (no chunking needed)")
            elif vram_free_gb > 8.0:
                # Enough VRAM for chunked GPU processing
                use_gpu = True
                logger.info("✓ GPU enabled (chunked mode)")
            else:
                logger.info(
                    "⚠️  GPU disabled (insufficient VRAM for chunking)"
                )
        
        config = MemoryConfig(
            available_ram_gb=available_ram_gb,
            swap_usage_percent=swap_percent,
            chunk_size=chunk_size,
            max_workers=max_workers,
            use_gpu=use_gpu,
            vram_available_gb=vram_free_gb
        )
        
        logger.info(f"\n{'='*60}")
        logger.info("Optimal Configuration:")
        logger.info(f"  Chunk Size: {config.chunk_size:,} points")
        logger.info(f"  Workers: {config.max_workers}")
        logger.info(f"  GPU: {'Enabled' if config.use_gpu else 'Disabled'}")
        logger.info(
            f"  Estimated Memory: "
            f"{self.estimate_memory_needed(num_points, mode, num_augmentations):.1f} GB"
        )
        logger.info(f"{'='*60}\n")
        
        return config
    
    def get_adaptive_safety_margin(
        self,
        total_memory_gb: float,
        memory_type: str = 'ram'
    ) -> float:
        """
        Get adaptive safety margin based on available memory.
        More memory = lower margin = better utilization.
        
        Args:
            total_memory_gb: Total memory available
            memory_type: 'ram' or 'vram'
            
        Returns:
            Safety margin (0.0-1.0)
        """
        if memory_type == 'vram':
            if total_memory_gb >= 12.0:
                return self.VRAM_SAFETY_MARGIN_HIGH
            elif total_memory_gb >= 8.0:
                return self.VRAM_SAFETY_MARGIN_MED
            else:
                return self.VRAM_SAFETY_MARGIN_LOW
        else:  # RAM
            if total_memory_gb >= 32.0:
                return self.RAM_SAFETY_MARGIN_HIGH
            elif total_memory_gb >= 16.0:
                return self.RAM_SAFETY_MARGIN_MED
            else:
                return self.RAM_SAFETY_MARGIN_LOW
    
    def monitor_during_processing(
        self,
        warn_threshold_percent: float = 85.0
    ) -> bool:
        """
        Monitor memory during processing and warn if needed.
        
        Args:
            warn_threshold_percent: Warn if RAM usage exceeds this
            
        Returns:
            True if memory is OK, False if critical
        """
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        if mem.percent > warn_threshold_percent:
            logger.warning(
                f"⚠️  RAM usage high: {mem.percent:.1f}% "
                f"({mem.available / (1024**3):.1f} GB available)"
            )
        
        if swap.percent > 50:
            logger.error(
                f"❌ Critical: Swap usage at {swap.percent:.1f}%"
            )
            return False
        
        return True


def get_adaptive_config(
    num_points: int,
    num_augmentations: int = 0,
    mode: str = 'full',
    **kwargs
) -> MemoryConfig:
    """
    Convenience function to get adaptive configuration.
    
    Args:
        num_points: Number of points
        num_augmentations: Number of augmentations
        mode: Processing mode
        **kwargs: Additional arguments for AdaptiveMemoryManager
        
    Returns:
        MemoryConfig with optimal settings
    """
    manager = AdaptiveMemoryManager(**kwargs)
    return manager.get_optimal_config(
        num_points=num_points,
        num_augmentations=num_augmentations,
        mode=mode
    )


# ==============================================================================
# Public API
# ==============================================================================

__all__ = [
    # Data classes
    'MemoryConfig',
    
    # Memory cleanup
    'aggressive_memory_cleanup',
    'clear_gpu_cache',
    
    # Memory estimation
    'estimate_memory_usage',
    'check_available_memory',
    'check_gpu_memory',
    
    # System information
    'get_system_memory_info',
    'estimate_memory_per_worker',
    
    # Worker optimization
    'calculate_optimal_workers',
    'calculate_batch_size',
    'log_memory_configuration',
    
    # File analysis
    'analyze_file_sizes',
    'sort_files_by_size',
    'check_gpu_memory_available',
    
    # Adaptive manager
    'AdaptiveMemoryManager',
    'get_adaptive_config',
]
