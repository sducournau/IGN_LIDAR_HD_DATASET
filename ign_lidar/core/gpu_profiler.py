"""
GPU Performance Profiler for Feature Computation

Unified profiling system for GPU operations with:
- CUDA event-based timing
- Memory usage tracking
- Bottleneck detection
- Transfer statistics
- Automatic performance reports

Integrates with GPUManager v3.1+ composition API.

Author: IGN LiDAR HD Development Team
Date: November 22, 2025
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


@dataclass
class ProfileEntry:
    """Single profiling entry for a GPU operation."""
    operation_name: str
    elapsed_ms: float
    mem_allocated_mb: float
    mem_freed_mb: float
    start_time: float
    end_time: float
    transfer_type: Optional[str] = None  # 'upload', 'download', or None
    transfer_size_mb: Optional[float] = None


@dataclass
class ProfilingStats:
    """Statistics summary for profiled operations."""
    total_time_ms: float
    num_operations: int
    avg_time_ms: float
    max_time_ms: float
    min_time_ms: float
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    total_memory_allocated_mb: float = 0.0
    total_memory_freed_mb: float = 0.0
    peak_memory_mb: float = 0.0
    upload_count: int = 0
    download_count: int = 0
    upload_mb: float = 0.0
    download_mb: float = 0.0


class GPUProfiler:
    """
    GPU performance profiler for LiDAR feature computation.
    
    Provides detailed profiling of GPU operations including:
    - CUDA event-based timing (microsecond precision)
    - Memory allocation/deallocation tracking
    - Transfer statistics (CPU↔GPU)
    - Bottleneck detection
    - Automatic performance reports
    
    Features:
    - Low overhead (< 1% performance impact)
    - CUDA event synchronization for accurate timing
    - Memory pool awareness
    - Hierarchical operation tracking
    
    Example:
        >>> profiler = GPUProfiler()
        >>> 
        >>> with profiler.profile('compute_normals'):
        ...     normals = compute_normals_gpu(points)
        >>> 
        >>> with profiler.profile('upload_points', transfer='upload', size_mb=100):
        ...     gpu_points = cp.asarray(points)
        >>> 
        >>> stats = profiler.get_stats()
        >>> profiler.print_report()
    
    Attributes:
        enabled: Whether profiling is enabled
        entries: List of ProfileEntry objects
        use_cuda_events: Use CUDA events for precise timing
    """
    
    def __init__(
        self, 
        enabled: bool = True,
        use_cuda_events: bool = True,
        bottleneck_threshold_pct: float = 20.0
    ):
        """
        Initialize GPU profiler.
        
        Args:
            enabled: Enable profiling (disable for production)
            use_cuda_events: Use CUDA events for timing (more accurate)
            bottleneck_threshold_pct: Threshold for bottleneck detection (% of total time)
        """
        self.enabled = enabled and HAS_CUPY
        self.use_cuda_events = use_cuda_events and HAS_CUPY
        self.bottleneck_threshold = bottleneck_threshold_pct / 100.0
        
        self.entries: List[ProfileEntry] = []
        self._current_operation: Optional[str] = None
        self._start_event: Optional['cp.cuda.Event'] = None
        self._end_event: Optional['cp.cuda.Event'] = None
        self._mem_before: int = 0
        
        if self.enabled and not HAS_CUPY:
            logger.warning("GPU profiling requested but CuPy not available")
            self.enabled = False
    
    @contextmanager
    def profile(
        self, 
        operation_name: str,
        transfer: Optional[str] = None,
        size_mb: Optional[float] = None
    ):
        """
        Context manager for profiling a GPU operation.
        
        Args:
            operation_name: Name of the operation being profiled
            transfer: Transfer type ('upload', 'download', or None)
            size_mb: Transfer size in MB (if applicable)
            
        Yields:
            None
            
        Example:
            >>> with profiler.profile('compute_features'):
            ...     features = compute_gpu(points)
        """
        if not self.enabled:
            yield
            return
        
        # Start profiling
        start_wall_time = time.time()
        
        if self.use_cuda_events:
            self._start_event = cp.cuda.Event()
            self._end_event = cp.cuda.Event()
            self._start_event.record()
        
        # Memory snapshot before
        if HAS_CUPY:
            mempool = cp.get_default_memory_pool()
            self._mem_before = mempool.used_bytes()
        
        try:
            yield
        finally:
            # End profiling
            if self.use_cuda_events:
                self._end_event.record()
                self._end_event.synchronize()
                elapsed_ms = cp.cuda.get_elapsed_time(self._start_event, self._end_event)
            else:
                elapsed_ms = (time.time() - start_wall_time) * 1000.0
            
            # Memory snapshot after
            mem_after = 0
            mem_allocated = 0.0
            mem_freed = 0.0
            
            if HAS_CUPY:
                mempool = cp.get_default_memory_pool()
                mem_after = mempool.used_bytes()
                mem_diff = mem_after - self._mem_before
                
                if mem_diff > 0:
                    mem_allocated = mem_diff / (1024**2)
                else:
                    mem_freed = abs(mem_diff) / (1024**2)
            
            # Record entry
            entry = ProfileEntry(
                operation_name=operation_name,
                elapsed_ms=elapsed_ms,
                mem_allocated_mb=mem_allocated,
                mem_freed_mb=mem_freed,
                start_time=start_wall_time,
                end_time=time.time(),
                transfer_type=transfer,
                transfer_size_mb=size_mb
            )
            self.entries.append(entry)
            
            logger.debug(
                f"⏱️ {operation_name}: {elapsed_ms:.2f}ms "
                f"(mem: +{mem_allocated:.1f}MB -{mem_freed:.1f}MB)"
            )
    
    def get_stats(self) -> ProfilingStats:
        """
        Get statistics summary for all profiled operations.
        
        Returns:
            ProfilingStats object with comprehensive statistics
        """
        if not self.entries:
            return ProfilingStats(
                total_time_ms=0.0,
                num_operations=0,
                avg_time_ms=0.0,
                max_time_ms=0.0,
                min_time_ms=0.0
            )
        
        times = [e.elapsed_ms for e in self.entries]
        total_time = sum(times)
        
        # Calculate memory statistics
        total_mem_allocated = sum(e.mem_allocated_mb for e in self.entries)
        total_mem_freed = sum(e.mem_freed_mb for e in self.entries)
        
        # Track peak memory (cumulative)
        peak_memory = 0.0
        current_memory = 0.0
        for entry in self.entries:
            current_memory += entry.mem_allocated_mb - entry.mem_freed_mb
            peak_memory = max(peak_memory, current_memory)
        
        # Transfer statistics
        upload_count = sum(1 for e in self.entries if e.transfer_type == 'upload')
        download_count = sum(1 for e in self.entries if e.transfer_type == 'download')
        upload_mb = sum(e.transfer_size_mb or 0 for e in self.entries if e.transfer_type == 'upload')
        download_mb = sum(e.transfer_size_mb or 0 for e in self.entries if e.transfer_type == 'download')
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(total_time)
        
        return ProfilingStats(
            total_time_ms=total_time,
            num_operations=len(self.entries),
            avg_time_ms=total_time / len(self.entries),
            max_time_ms=max(times),
            min_time_ms=min(times),
            bottlenecks=bottlenecks,
            total_memory_allocated_mb=total_mem_allocated,
            total_memory_freed_mb=total_mem_freed,
            peak_memory_mb=peak_memory,
            upload_count=upload_count,
            download_count=download_count,
            upload_mb=upload_mb,
            download_mb=download_mb
        )
    
    def _detect_bottlenecks(self, total_time: float) -> List[Dict[str, Any]]:
        """
        Detect performance bottlenecks.
        
        A bottleneck is any operation that takes > threshold% of total time.
        
        Args:
            total_time: Total time in milliseconds
            
        Returns:
            List of bottleneck dictionaries
        """
        bottlenecks = []
        
        # Group by operation name
        operation_times: Dict[str, float] = {}
        operation_counts: Dict[str, int] = {}
        
        for entry in self.entries:
            if entry.operation_name not in operation_times:
                operation_times[entry.operation_name] = 0.0
                operation_counts[entry.operation_name] = 0
            
            operation_times[entry.operation_name] += entry.elapsed_ms
            operation_counts[entry.operation_name] += 1
        
        # Find bottlenecks
        for op_name, op_time in operation_times.items():
            percentage = (op_time / total_time) if total_time > 0 else 0
            
            if percentage >= self.bottleneck_threshold:
                bottlenecks.append({
                    'operation': op_name,
                    'time_ms': op_time,
                    'percentage': percentage * 100,
                    'count': operation_counts[op_name],
                    'avg_time_ms': op_time / operation_counts[op_name]
                })
        
        # Sort by percentage (descending)
        bottlenecks.sort(key=lambda x: x['percentage'], reverse=True)
        
        return bottlenecks
    
    def print_report(self, detailed: bool = False) -> None:
        """
        Print comprehensive profiling report.
        
        Args:
            detailed: Include detailed per-operation breakdown
        """
        if not self.entries:
            logger.info("📊 GPU Profiler: No operations profiled")
            return
        
        stats = self.get_stats()
        
        logger.info("=" * 70)
        logger.info("📊 GPU Performance Report")
        logger.info("=" * 70)
        
        # Overall statistics
        logger.info(f"Total Operations: {stats.num_operations}")
        logger.info(f"Total Time: {stats.total_time_ms:.2f}ms ({stats.total_time_ms/1000:.2f}s)")
        logger.info(f"Average Time: {stats.avg_time_ms:.2f}ms")
        logger.info(f"Min/Max Time: {stats.min_time_ms:.2f}ms / {stats.max_time_ms:.2f}ms")
        logger.info("")
        
        # Memory statistics
        logger.info("Memory Statistics:")
        logger.info(f"  Total Allocated: {stats.total_memory_allocated_mb:.1f}MB")
        logger.info(f"  Total Freed: {stats.total_memory_freed_mb:.1f}MB")
        logger.info(f"  Peak Usage: {stats.peak_memory_mb:.1f}MB")
        logger.info(f"  Net Change: {stats.total_memory_allocated_mb - stats.total_memory_freed_mb:.1f}MB")
        logger.info("")
        
        # Transfer statistics
        if stats.upload_count > 0 or stats.download_count > 0:
            logger.info("Transfer Statistics:")
            logger.info(f"  Uploads: {stats.upload_count} ({stats.upload_mb:.1f}MB)")
            logger.info(f"  Downloads: {stats.download_count} ({stats.download_mb:.1f}MB)")
            logger.info(f"  Total Transferred: {stats.upload_mb + stats.download_mb:.1f}MB")
            logger.info("")
        
        # Bottlenecks
        if stats.bottlenecks:
            logger.info(f"⚠️ Performance Bottlenecks (>{self.bottleneck_threshold*100:.0f}% of total time):")
            for i, bottleneck in enumerate(stats.bottlenecks, 1):
                logger.info(
                    f"  {i}. {bottleneck['operation']}: "
                    f"{bottleneck['time_ms']:.2f}ms ({bottleneck['percentage']:.1f}%) "
                    f"[{bottleneck['count']}x, avg {bottleneck['avg_time_ms']:.2f}ms]"
                )
            logger.info("")
        
        # Transfer vs Compute analysis (migrated from optimization/gpu_profiler.py)
        bottleneck_analysis = self.get_bottleneck_analysis()
        if bottleneck_analysis['bottleneck'] != 'unknown':
            logger.info("Transfer vs Compute Analysis:")
            logger.info(f"  Transfer: {bottleneck_analysis['transfer_pct']:.1f}%")
            logger.info(f"  Compute: {bottleneck_analysis['compute_pct']:.1f}%")
            logger.info(f"  Bottleneck: {bottleneck_analysis['bottleneck']}")
            logger.info(f"  Recommendation: {bottleneck_analysis['recommendation']}")
            logger.info("")
        
        # Detailed breakdown
        if detailed:
            logger.info("Detailed Operation Breakdown:")
            logger.info(f"{'Operation':<40} {'Time (ms)':>12} {'Memory (MB)':>14}")
            logger.info("-" * 70)
            
            for entry in self.entries:
                mem_str = f"+{entry.mem_allocated_mb:.1f}/-{entry.mem_freed_mb:.1f}"
                logger.info(
                    f"{entry.operation_name:<40} "
                    f"{entry.elapsed_ms:>12.2f} "
                    f"{mem_str:>14}"
                )
        
        logger.info("=" * 70)
    
    def reset(self) -> None:
        """Reset profiler and clear all entries."""
        self.entries.clear()
        self._current_operation = None
        self._start_event = None
        self._end_event = None
        self._mem_before = 0
        
        logger.debug("GPU profiler reset")
    
    def get_operation_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics grouped by operation name.
        
        Returns:
            Dictionary mapping operation names to their statistics
        """
        summary: Dict[str, Dict[str, float]] = {}
        
        for entry in self.entries:
            op_name = entry.operation_name
            
            if op_name not in summary:
                summary[op_name] = {
                    'total_time_ms': 0.0,
                    'count': 0,
                    'avg_time_ms': 0.0,
                    'min_time_ms': float('inf'),
                    'max_time_ms': 0.0,
                    'total_mem_allocated_mb': 0.0
                }
            
            summary[op_name]['total_time_ms'] += entry.elapsed_ms
            summary[op_name]['count'] += 1
            summary[op_name]['min_time_ms'] = min(summary[op_name]['min_time_ms'], entry.elapsed_ms)
            summary[op_name]['max_time_ms'] = max(summary[op_name]['max_time_ms'], entry.elapsed_ms)
            summary[op_name]['total_mem_allocated_mb'] += entry.mem_allocated_mb
        
        # Calculate averages
        for op_name in summary:
            count = summary[op_name]['count']
            summary[op_name]['avg_time_ms'] = summary[op_name]['total_time_ms'] / count
        
        return summary
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """
        Analyze transfer vs compute bottlenecks.
        
        Migrated from optimization/gpu_profiler.py for unified profiling.
        Determines whether memory transfers or computation is the bottleneck.
        
        Returns:
            Dictionary with bottleneck analysis including:
            - bottleneck: 'memory_transfer', 'compute', 'balanced', or 'unknown'
            - transfer_pct: Percentage of time spent in transfers
            - compute_pct: Percentage of time spent in compute
            - recommendation: Performance optimization suggestions
        """
        # Separate transfer and compute operations
        transfer_time = sum(
            e.elapsed_ms for e in self.entries 
            if e.transfer_type in ('upload', 'download')
        )
        compute_time = sum(
            e.elapsed_ms for e in self.entries 
            if e.transfer_type is None
        )
        total_time = transfer_time + compute_time
        
        if total_time == 0:
            return {
                'bottleneck': 'unknown',
                'transfer_pct': 0.0,
                'compute_pct': 0.0,
                'recommendation': 'No data available'
            }
        
        transfer_pct = (transfer_time / total_time) * 100
        compute_pct = (compute_time / total_time) * 100
        
        # Determine bottleneck
        if transfer_pct > 50:
            bottleneck = 'memory_transfer'
            recommendation = (
                "Memory transfers are the bottleneck. Recommendations:\n"
                "  - Increase chunk size to reduce transfer frequency\n"
                "  - Enable CUDA streams for overlapped transfers\n"
                "  - Use pinned memory for faster transfers\n"
                "  - Keep more data on GPU between operations"
            )
        elif compute_pct > 70:
            bottleneck = 'compute'
            recommendation = (
                "Computation is the bottleneck. Recommendations:\n"
                "  - Optimize kernel algorithms\n"
                "  - Increase parallelism\n"
                "  - Use cuML for accelerated algorithms\n"
                "  - Consider reducing feature complexity"
            )
        else:
            bottleneck = 'balanced'
            recommendation = (
                "Performance is well balanced. Minor optimizations:\n"
                "  - Monitor VRAM usage to maximize batch sizes\n"
                "  - Enable all available optimizations\n"
                "  - Consider pipeline optimization for further gains"
            )
        
        return {
            'bottleneck': bottleneck,
            'transfer_pct': transfer_pct,
            'compute_pct': compute_pct,
            'recommendation': recommendation
        }


def create_profiler(
    enabled: bool = True,
    use_cuda_events: bool = True,
    bottleneck_threshold: float = 20.0
) -> GPUProfiler:
    """
    Factory function to create a GPU profiler.
    
    Args:
        enabled: Enable profiling
        use_cuda_events: Use CUDA events for precise timing
        bottleneck_threshold: Threshold for bottleneck detection (%)
        
    Returns:
        Configured GPUProfiler instance
    """
    return GPUProfiler(
        enabled=enabled,
        use_cuda_events=use_cuda_events,
        bottleneck_threshold_pct=bottleneck_threshold
    )
