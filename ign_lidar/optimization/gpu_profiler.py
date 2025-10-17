"""
GPU Performance Profiling and Monitoring Utilities

This module provides comprehensive profiling tools for GPU operations including:
- Memory transfer tracking
- Compute time measurement
- GPU utilization monitoring
- Bottleneck detection
- Performance recommendations

Author: IGN LiDAR HD Development Team
Date: October 17, 2025
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


@dataclass
class GPUOperationMetrics:
    """Metrics for a single GPU operation."""
    operation_name: str
    start_time: float
    end_time: float = 0.0
    data_size_mb: float = 0.0
    vram_before_mb: float = 0.0
    vram_after_mb: float = 0.0
    transfer_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class ProfilerSession:
    """Complete profiling session with aggregated metrics."""
    session_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    operations: List[GPUOperationMetrics] = field(default_factory=list)
    
    def duration(self) -> float:
        """Total session duration in seconds."""
        if self.end_time == 0.0:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def total_transfer_time(self) -> float:
        """Total data transfer time in milliseconds."""
        return sum(op.transfer_time_ms for op in self.operations)
    
    def total_compute_time(self) -> float:
        """Total compute time in milliseconds."""
        return sum(op.compute_time_ms for op in self.operations)
    
    def total_data_transferred(self) -> float:
        """Total data transferred in MB."""
        return sum(op.data_size_mb for op in self.operations)
    
    def get_bottleneck_analysis(self) -> Dict[str, any]:
        """
        Analyze performance bottlenecks.
        
        Returns:
            Dictionary with bottleneck analysis
        """
        total_transfer = self.total_transfer_time()
        total_compute = self.total_compute_time()
        total_time = total_transfer + total_compute
        
        if total_time == 0:
            return {
                'bottleneck': 'unknown',
                'transfer_pct': 0.0,
                'compute_pct': 0.0,
                'recommendation': 'No data available'
            }
        
        transfer_pct = (total_transfer / total_time) * 100
        compute_pct = (total_compute / total_time) * 100
        
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


class GPUProfiler:
    """
    GPU performance profiler for tracking and analyzing operations.
    
    Usage:
        >>> profiler = GPUProfiler(enable=True)
        >>> 
        >>> with profiler.profile_operation('feature_computation', data_size_mb=100):
        >>>     # GPU operation here
        >>>     features = compute_features_gpu(points)
        >>> 
        >>> profiler.print_summary()
    """
    
    def __init__(self, enable: bool = True, session_name: str = "default"):
        """
        Initialize GPU profiler.
        
        Args:
            enable: Enable profiling (minimal overhead when disabled)
            session_name: Name for this profiling session
        """
        self.enable = enable
        self.sessions: Dict[str, ProfilerSession] = {}
        self.current_session = ProfilerSession(session_name=session_name)
        self.sessions[session_name] = self.current_session
        
        # GPU availability
        self.gpu_available = HAS_CUPY
        
        if enable and not self.gpu_available:
            logger.warning("GPU profiler enabled but CuPy not available")
    
    def start_session(self, session_name: str):
        """Start a new profiling session."""
        self.current_session = ProfilerSession(session_name=session_name)
        self.sessions[session_name] = self.current_session
        
        if self.enable:
            logger.info(f"📊 Profiler session started: {session_name}")
    
    def end_session(self):
        """End current profiling session."""
        if self.current_session:
            self.current_session.end_time = time.time()
            
            if self.enable:
                logger.info(f"✓ Profiler session ended: {self.current_session.session_name}")
    
    def profile_operation(
        self,
        operation_name: str,
        data_size_mb: float = 0.0
    ):
        """
        Context manager for profiling a GPU operation.
        
        Args:
            operation_name: Name of the operation
            data_size_mb: Size of data being processed (MB)
            
        Returns:
            Context manager that profiles the operation
        """
        return _OperationProfiler(self, operation_name, data_size_mb)
    
    def record_operation(self, metrics: GPUOperationMetrics):
        """Record operation metrics."""
        if self.enable:
            self.current_session.operations.append(metrics)
    
    def get_vram_usage(self) -> Tuple[float, float]:
        """
        Get current VRAM usage in MB.
        
        Returns:
            Tuple of (used_mb, total_mb)
        """
        if not self.gpu_available or cp is None:
            return (0.0, 0.0)
        
        try:
            free_vram, total_vram = cp.cuda.runtime.memGetInfo()
            used_vram = total_vram - free_vram
            return (used_vram / (1024**2), total_vram / (1024**2))
        except Exception:
            return (0.0, 0.0)
    
    def print_summary(self, session_name: Optional[str] = None):
        """
        Print performance summary for a session.
        
        Args:
            session_name: Session to summarize (None = current session)
        """
        if not self.enable:
            return
        
        session = self.sessions.get(session_name) if session_name else self.current_session
        if not session:
            logger.warning(f"Session not found: {session_name}")
            return
        
        print("\n" + "=" * 80)
        print(f"GPU Performance Profiler - Session: {session.session_name}")
        print("=" * 80)
        
        print(f"\nSession Duration: {session.duration():.2f}s")
        print(f"Operations Tracked: {len(session.operations)}")
        
        if not session.operations:
            print("No operations recorded")
            print("=" * 80 + "\n")
            return
        
        # Aggregate by operation type
        op_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'total_transfer': 0.0,
            'total_compute': 0.0,
            'total_data': 0.0
        })
        
        for op in session.operations:
            stats = op_stats[op.operation_name]
            stats['count'] += 1
            stats['total_time'] += (op.end_time - op.start_time) * 1000  # ms
            stats['total_transfer'] += op.transfer_time_ms
            stats['total_compute'] += op.compute_time_ms
            stats['total_data'] += op.data_size_mb
        
        # Print operation breakdown
        print("\nOperation Breakdown:")
        print("-" * 80)
        print(f"{'Operation':<30} {'Count':>8} {'Time (ms)':>12} {'Transfer':>12} {'Compute':>12}")
        print("-" * 80)
        
        for op_name, stats in sorted(op_stats.items()):
            print(f"{op_name:<30} {stats['count']:>8} "
                  f"{stats['total_time']:>12.1f} "
                  f"{stats['total_transfer']:>12.1f} "
                  f"{stats['total_compute']:>12.1f}")
        
        # Overall statistics
        print("\nOverall Statistics:")
        print("-" * 80)
        print(f"Total Transfer Time:  {session.total_transfer_time():>12.1f} ms")
        print(f"Total Compute Time:   {session.total_compute_time():>12.1f} ms")
        print(f"Total Data Transfer:  {session.total_data_transferred():>12.1f} MB")
        
        # Bottleneck analysis
        analysis = session.get_bottleneck_analysis()
        print("\nBottleneck Analysis:")
        print("-" * 80)
        print(f"Bottleneck Type: {analysis['bottleneck']}")
        print(f"Transfer Time: {analysis['transfer_pct']:.1f}%")
        print(f"Compute Time:  {analysis['compute_pct']:.1f}%")
        print(f"\nRecommendations:\n{analysis['recommendation']}")
        
        # VRAM usage
        if self.gpu_available:
            used_mb, total_mb = self.get_vram_usage()
            print(f"\nVRAM Usage: {used_mb:.1f}MB / {total_mb:.1f}MB "
                  f"({(used_mb/total_mb*100) if total_mb > 0 else 0:.1f}%)")
        
        print("=" * 80 + "\n")
    
    def get_recommendations(self) -> List[str]:
        """
        Get performance optimization recommendations.
        
        Returns:
            List of recommendation strings
        """
        if not self.enable or not self.current_session.operations:
            return []
        
        recommendations = []
        analysis = self.current_session.get_bottleneck_analysis()
        
        # Add bottleneck-specific recommendations
        if analysis['bottleneck'] == 'memory_transfer':
            if analysis['transfer_pct'] > 60:
                recommendations.append(
                    "CRITICAL: Memory transfers dominate performance (>60%). "
                    "Enable CUDA streams and increase chunk sizes."
                )
            else:
                recommendations.append(
                    "Memory transfers are significant. Consider enabling "
                    "pipeline optimization."
                )
        
        # Check for small operations
        avg_data_size = self.current_session.total_data_transferred() / len(self.current_session.operations)
        if avg_data_size < 10:  # Less than 10MB average
            recommendations.append(
                f"Average operation size is small ({avg_data_size:.1f}MB). "
                "Consider batching operations to reduce overhead."
            )
        
        # Check VRAM utilization
        if self.gpu_available:
            used_mb, total_mb = self.get_vram_usage()
            utilization = used_mb / total_mb if total_mb > 0 else 0
            
            if utilization < 0.5:
                recommendations.append(
                    f"Low VRAM utilization ({utilization*100:.0f}%). "
                    "Can increase chunk sizes for better performance."
                )
            elif utilization > 0.9:
                recommendations.append(
                    f"High VRAM utilization ({utilization*100:.0f}%). "
                    "Risk of OOM errors - consider reducing chunk size."
                )
        
        return recommendations


class _OperationProfiler:
    """Context manager for profiling individual operations."""
    
    def __init__(self, profiler: GPUProfiler, operation_name: str, data_size_mb: float):
        self.profiler = profiler
        self.operation_name = operation_name
        self.data_size_mb = data_size_mb
        self.metrics = None
    
    def __enter__(self):
        if not self.profiler.enable:
            return self
        
        self.metrics = GPUOperationMetrics(
            operation_name=self.operation_name,
            start_time=time.time(),
            data_size_mb=self.data_size_mb
        )
        
        # Record VRAM before
        if self.profiler.gpu_available:
            used_mb, _ = self.profiler.get_vram_usage()
            self.metrics.vram_before_mb = used_mb
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.profiler.enable or self.metrics is None:
            return False
        
        self.metrics.end_time = time.time()
        
        # Record VRAM after
        if self.profiler.gpu_available:
            used_mb, _ = self.profiler.get_vram_usage()
            self.metrics.vram_after_mb = used_mb
        
        # Record success/error
        if exc_type is not None:
            self.metrics.success = False
            self.metrics.error_message = str(exc_val)
        
        # Estimate transfer and compute times (rough approximation)
        total_time_ms = (self.metrics.end_time - self.metrics.start_time) * 1000
        
        # Assume 20% transfer, 80% compute (can be refined with CUDA events)
        self.metrics.transfer_time_ms = total_time_ms * 0.2
        self.metrics.compute_time_ms = total_time_ms * 0.8
        
        # Record metrics
        self.profiler.record_operation(self.metrics)
        
        return False  # Don't suppress exceptions


# Global profiler instance
_global_profiler = None


def get_profiler(enable: bool = True, session_name: str = "default") -> GPUProfiler:
    """
    Get or create global GPU profiler instance.
    
    Args:
        enable: Enable profiling
        session_name: Session name
        
    Returns:
        Global GPUProfiler instance
    """
    global _global_profiler
    
    if _global_profiler is None:
        _global_profiler = GPUProfiler(enable=enable, session_name=session_name)
    
    return _global_profiler
