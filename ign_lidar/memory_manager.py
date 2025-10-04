"""
Adaptive Memory Management for LiDAR Processing
Real-time memory monitoring and dynamic chunk size adjustment
Version: 1.7.0
"""

from typing import Optional, Tuple
import psutil
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Memory configuration for processing."""
    available_ram_gb: float
    swap_usage_percent: float
    chunk_size: int
    max_workers: int
    use_gpu: bool
    vram_available_gb: float = 0.0


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
    
    # Safety margins
    RAM_SAFETY_MARGIN = 0.20  # Keep 20% RAM free
    VRAM_SAFETY_MARGIN = 0.15  # Keep 15% VRAM free
    
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
                device = cp.cuda.Device()
                mem_info = device.mem_info
                vram_free = mem_info[0] / (1024**3)
                vram_total = mem_info[1] / (1024**3)
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
                vram_free_gb = cp.cuda.Device().mem_info[0] / (1024**3)
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
        
        # Calculate safe worker count
        max_safe_workers = int(
            available_ram_gb * (1 - self.RAM_SAFETY_MARGIN)
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
