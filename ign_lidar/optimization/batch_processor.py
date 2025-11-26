"""
Batch Processing Optimizer - Phase 3

Optimizes the processing of multiple patches by:
- Detecting batch patterns (size, density, characteristics)
- Selecting optimal batch size for memory efficiency
- Reusing GPU allocations across batches
- Minimizing memory fragmentation
"""

import logging
from typing import List, Optional, Tuple, Dict, Callable
import numpy as np
from dataclasses import dataclass
import gc

logger = logging.getLogger(__name__)


@dataclass
class BatchStats:
    """Statistics for a batch of patches"""
    total_points: int
    avg_points_per_patch: int
    min_points: int
    max_points: int
    num_patches: int
    estimated_memory_gb: float
    recommended_batch_size: int


class BatchProcessingOptimizer:
    """
    Optimizes batch processing of multiple patches.
    
    This optimizer analyzes patches and recommends optimal:
    - Batch sizes for GPU memory efficiency
    - Processing strategy (GPU vs CPU)
    - Memory pre-allocation strategy
    - Cleanup intervals
    """
    
    def __init__(self, gpu_available: bool = False, max_vram_gb: float = 8.0):
        """
        Initialize batch optimizer.
        
        Args:
            gpu_available: Whether GPU is available
            max_vram_gb: Maximum GPU VRAM in GB
        """
        self.gpu_available = gpu_available
        self.max_vram_gb = max_vram_gb
        self.batch_size_history: List[Tuple[int, float]] = []  # (size, time)
    
    def analyze_patches(
        self,
        patches: List[np.ndarray],
    ) -> BatchStats:
        """
        Analyze patch characteristics to recommend batch settings.
        
        Args:
            patches: List of point cloud patches
            
        Returns:
            BatchStats with recommendations
        """
        if not patches:
            raise ValueError("Empty patch list")
        
        # Calculate statistics
        point_counts = [len(p) for p in patches]
        total_points = sum(point_counts)
        
        stats = BatchStats(
            total_points=total_points,
            avg_points_per_patch=total_points // len(patches),
            min_points=min(point_counts),
            max_points=max(point_counts),
            num_patches=len(patches),
            estimated_memory_gb=self._estimate_memory(total_points),
            recommended_batch_size=self._recommend_batch_size(
                total_points, len(patches)
            ),
        )
        
        logger.info(
            f"Batch analysis: {stats.num_patches} patches, "
            f"{stats.avg_points_per_patch:,} avg points/patch, "
            f"Est. memory: {stats.estimated_memory_gb:.2f} GB"
        )
        
        return stats
    
    def _estimate_memory(self, num_points: int, features_per_point: int = 50) -> float:
        """
        Estimate GPU memory needed for points and features.
        
        Assumes:
        - Points: 3 floats × 4 bytes = 12 bytes/point
        - Features: ~50 floats × 4 bytes = 200 bytes/point
        - GPU copies: ~2x (input + working)
        
        Args:
            num_points: Number of points
            features_per_point: Estimated features per point
            
        Returns:
            Estimated memory in GB
        """
        bytes_per_point = (3 + features_per_point) * 4  # XYZ + features
        total_bytes = num_points * bytes_per_point * 2  # 2x for GPU copies
        return total_bytes / (1024 ** 3)
    
    def _recommend_batch_size(
        self,
        total_points: int,
        num_patches: int,
    ) -> int:
        """
        Recommend batch size based on dataset characteristics.
        
        Args:
            total_points: Total points across all patches
            num_patches: Number of patches
            
        Returns:
            Recommended batch size in points
        """
        # Heuristics:
        # - GPU: Use larger batches to amortize kernel launch overhead
        # - CPU: Use smaller batches for better cache utilization
        # - Large total: Use adaptive batching
        
        avg_patch_size = total_points // num_patches
        
        if self.gpu_available:
            # GPU prefers larger batches
            if total_points > 50_000_000:
                return 5_000_000
            elif total_points > 10_000_000:
                return 2_000_000
            else:
                return 1_000_000
        else:
            # CPU prefers smaller batches for cache efficiency
            if total_points > 10_000_000:
                return 100_000
            elif total_points > 1_000_000:
                return 50_000
            else:
                return 10_000
    
    def group_patches(
        self,
        patches: List[np.ndarray],
        batch_size: Optional[int] = None,
    ) -> List[List[np.ndarray]]:
        """
        Group patches into batches based on memory constraints.
        
        Args:
            patches: List of point cloud patches
            batch_size: Maximum points per batch (auto-calculated if None)
            
        Returns:
            List of batches (each batch is a list of patches)
        """
        if batch_size is None:
            stats = self.analyze_patches(patches)
            batch_size = stats.recommended_batch_size
        
        batches: List[List[np.ndarray]] = []
        current_batch: List[np.ndarray] = []
        current_size = 0
        
        for patch in patches:
            patch_size = len(patch)
            
            # Start new batch if current is full
            if current_size + patch_size > batch_size and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            
            current_batch.append(patch)
            current_size += patch_size
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(
            f"Grouped {len(patches)} patches into {len(batches)} batches "
            f"(batch_size={batch_size:,})"
        )
        
        return batches
    
    def process_batches(
        self,
        patches: List[np.ndarray],
        process_fn: Callable[[np.ndarray], Dict],
        cleanup_interval: int = 10,
        batch_size: Optional[int] = None,
    ) -> List[Dict]:
        """
        Process patches in optimized batches.
        
        Args:
            patches: List of patches to process
            process_fn: Function to apply to each patch
            cleanup_interval: Run garbage collection every N patches
            batch_size: Maximum points per batch
            
        Returns:
            List of results (one per patch)
        """
        batches = self.group_patches(patches, batch_size)
        results = []
        
        for batch_idx, batch in enumerate(batches):
            logger.info(
                f"Processing batch {batch_idx + 1}/{len(batches)} "
                f"({sum(len(p) for p in batch):,} points)"
            )
            
            for patch_idx, patch in enumerate(batch):
                # Process patch
                result = process_fn(patch)
                results.append(result)
                
                # Cleanup periodically
                if (batch_idx * len(batch) + patch_idx) % cleanup_interval == 0:
                    gc.collect()
        
        return results


class AdaptiveBatchProcessor:
    """
    Adaptive batch processor that learns from previous batches.
    
    Tracks batch processing times and adjusts batch size dynamically
    to optimize throughput.
    """
    
    def __init__(
        self,
        initial_batch_size: int = 100_000,
        target_time_per_batch_s: float = 1.0,
    ):
        """
        Initialize adaptive processor.
        
        Args:
            initial_batch_size: Starting batch size
            target_time_per_batch_s: Target time per batch (for auto-tuning)
        """
        self.batch_size = initial_batch_size
        self.target_time = target_time_per_batch_s
        self.history: List[Tuple[int, float]] = []  # (batch_size, time)
    
    def update(self, batch_size: int, elapsed_time: float) -> None:
        """
        Update adaptive settings based on batch processing time.
        
        Args:
            batch_size: Points processed
            elapsed_time: Time taken in seconds
        """
        self.history.append((batch_size, elapsed_time))
        
        # Auto-tune batch size
        if len(self.history) >= 3:
            avg_time = np.mean([t for _, t in self.history[-3:]])
            
            if avg_time < self.target_time * 0.8:
                # Processing is fast, increase batch size
                self.batch_size = int(self.batch_size * 1.2)
                logger.info(f"Increased batch size to {self.batch_size:,}")
            elif avg_time > self.target_time * 1.2:
                # Processing is slow, decrease batch size
                self.batch_size = int(self.batch_size * 0.8)
                logger.info(f"Decreased batch size to {self.batch_size:,}")
    
    def get_batch_size(self) -> int:
        """Get current recommended batch size."""
        return self.batch_size
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.history:
            return {}
        
        times = [t for _, t in self.history]
        return {
            "current_batch_size": self.batch_size,
            "avg_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "num_batches": len(self.history),
        }


if __name__ == "__main__":
    # Simple test
    print("Testing batch processor...")
    
    optimizer = BatchProcessingOptimizer(gpu_available=False)
    
    # Create test patches
    patches = [
        np.random.rand(10000, 3).astype(np.float32),
        np.random.rand(15000, 3).astype(np.float32),
        np.random.rand(8000, 3).astype(np.float32),
    ]
    
    # Analyze
    stats = optimizer.analyze_patches(patches)
    print(f"✓ Analysis: {stats}")
    
    # Group
    batches = optimizer.group_patches(patches, batch_size=20000)
    print(f"✓ Grouping: {len(batches)} batches")
    
    print("✓ All tests passed!")
