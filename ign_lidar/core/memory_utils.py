"""
Memory management utilities for CLI commands.

Provides functions for:
- Calculating optimal worker counts based on available memory
- Analyzing file sizes and memory requirements
- Managing batch sizes for parallel processing
- Memory-aware configuration adjustments
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory optimization disabled")


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
