"""
GPU Transfer Profiler - Track and Optimize CPU↔GPU Transfers

Phase 2 Enhancement: Dedicated profiler for monitoring GPU memory transfers
to identify bottlenecks and excessive synchronization points.

This profiler specifically tracks:
- CPU→GPU transfers (cp.asarray, cp.array)
- GPU→CPU transfers (cp.asnumpy, array.get())
- Transfer sizes and bandwidth
- Transfer hotspots (with stack traces)

Performance Targets (Phase 2):
- Reduce transfers from 90+ to <5 per tile
- Minimize unnecessary synchronization
- Identify transfer bottlenecks

Integration with existing gpu_profiler.py:
- GPUProfiler: Comprehensive GPU metrics (memory, compute, utilization)
- GPUTransferProfiler: Focused transfer tracking (this file)

Author: IGN LiDAR HD Development Team - Phase 2 Refactoring
Date: November 22, 2025
Version: 1.0.0
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


@dataclass
class TransferEvent:
    """Single GPU transfer event."""
    timestamp: float
    direction: str  # 'cpu_to_gpu' or 'gpu_to_cpu'
    bytes: int
    shape: tuple
    dtype: str
    stack_trace: Optional[str] = None


class GPUTransferProfiler:
    """
    Profile GPU memory transfers to identify bottlenecks.
    
    Tracks:
    - Number of transfers (CPU to GPU, GPU to CPU)
    - Transfer sizes
    - Transfer locations (stack traces)
    - Transfer timing
    
    Usage:
        profiler = GPUTransferProfiler()
        with profiler:
            points_gpu = cp.asarray(points)
            features = compute_features_gpu(points_gpu)
            result = cp.asnumpy(features)
        profiler.print_report()
    
    Phase 2 Goals:
        - Identify excessive transfers (>5 per tile is bad)
        - Find transfer hotspots
        - Measure bandwidth utilization
    """
    
    def __init__(self, track_stacks: bool = False):
        """
        Args:
            track_stacks: If True, capture stack traces for each transfer
                         (useful for debugging but adds overhead)
        """
        self.track_stacks = track_stacks
        self.events: List[TransferEvent] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._enabled = False
        
        if not CUPY_AVAILABLE:
            logger.warning("CuPy not available, profiler will not track transfers")
    
    def start(self):
        """Start profiling."""
        self.events.clear()
        self.start_time = time.time()
        self._enabled = True
        logger.info("GPU transfer profiling started")
    
    def stop(self):
        """Stop profiling."""
        self.end_time = time.time()
        self._enabled = False
        duration = self.end_time - self.start_time if self.start_time else 0
        logger.info(f"GPU transfer profiling stopped (duration: {duration:.2f}s)")
    
    def record_cpu_to_gpu(self, array: np.ndarray, gpu_array: Any):
        """Record CPU→GPU transfer."""
        if not self._enabled:
            return
        
        stack = None
        if self.track_stacks:
            import traceback
            stack = ''.join(traceback.format_stack()[:-1])
        
        event = TransferEvent(
            timestamp=time.time(),
            direction='cpu_to_gpu',
            bytes=array.nbytes,
            shape=array.shape,
            dtype=str(array.dtype),
            stack_trace=stack
        )
        self.events.append(event)
    
    def record_gpu_to_cpu(self, gpu_array: Any, array: np.ndarray):
        """Record GPU→CPU transfer."""
        if not self._enabled:
            return
        
        stack = None
        if self.track_stacks:
            import traceback
            stack = ''.join(traceback.format_stack()[:-1])
        
        event = TransferEvent(
            timestamp=time.time(),
            direction='gpu_to_cpu',
            bytes=array.nbytes,
            shape=array.shape,
            dtype=str(array.dtype),
            stack_trace=stack
        )
        self.events.append(event)
    
    @contextmanager
    def profile(self):
        """Context manager for profiling."""
        self.start()
        try:
            yield self
        finally:
            self.stop()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        if not self.events:
            return {
                'total_transfers': 0,
                'cpu_to_gpu': 0,
                'gpu_to_cpu': 0,
                'total_bytes': 0,
                'total_bytes_cpu_to_gpu': 0,
                'total_bytes_gpu_to_cpu': 0,
                'duration_seconds': 0,
                'bandwidth_gbps': 0,
            }
        
        cpu_to_gpu = [e for e in self.events if e.direction == 'cpu_to_gpu']
        gpu_to_cpu = [e for e in self.events if e.direction == 'gpu_to_cpu']
        
        duration = (self.end_time or time.time()) - (self.start_time or 0)
        total_bytes = sum(e.bytes for e in self.events)
        
        return {
            'total_transfers': len(self.events),
            'cpu_to_gpu': len(cpu_to_gpu),
            'gpu_to_cpu': len(gpu_to_cpu),
            'total_bytes': total_bytes,
            'total_bytes_cpu_to_gpu': sum(e.bytes for e in cpu_to_gpu),
            'total_bytes_gpu_to_cpu': sum(e.bytes for e in gpu_to_cpu),
            'duration_seconds': duration,
            'bandwidth_gbps': total_bytes / duration / 1e9 if duration > 0 else 0,
        }
    
    def get_hotspots(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N transfer hotspots by frequency.
        
        Only works if track_stacks=True was set.
        
        Returns:
            List of hotspots with count, bytes, and stack trace
        """
        if not self.track_stacks:
            logger.warning("Stack traces not captured, cannot identify hotspots")
            return []
        
        # Group by stack trace
        hotspots: Dict[str, Dict[str, Any]] = {}
        for event in self.events:
            if event.stack_trace not in hotspots:
                hotspots[event.stack_trace] = {
                    'count': 0,
                    'total_bytes': 0,
                    'stack': event.stack_trace
                }
            hotspots[event.stack_trace]['count'] += 1
            hotspots[event.stack_trace]['total_bytes'] += event.bytes
        
        # Sort by count
        sorted_hotspots = sorted(
            hotspots.values(),
            key=lambda x: x['count'],
            reverse=True
        )
        
        return sorted_hotspots[:top_n]
    
    def print_report(self):
        """Print profiling report."""
        stats = self.get_stats()
        
        print("\n" + "=" * 80)
        print("GPU TRANSFER PROFILING REPORT")
        print("=" * 80)
        
        print(f"\nDuration: {stats['duration_seconds']:.2f}s")
        print(f"\nTotal transfers: {stats['total_transfers']}")
        print(f"  CPU to GPU: {stats['cpu_to_gpu']}")
        print(f"  GPU to CPU: {stats['gpu_to_cpu']}")
        
        print(f"\nTotal data transferred: {stats['total_bytes'] / 1e9:.3f} GB")
        print(f"  CPU to GPU: {stats['total_bytes_cpu_to_gpu'] / 1e9:.3f} GB")
        print(f"  GPU to CPU: {stats['total_bytes_gpu_to_cpu'] / 1e9:.3f} GB")
        
        print(f"\nAverage bandwidth: {stats['bandwidth_gbps']:.2f} GB/s")
        
        # Phase 2 target check
        if stats['total_transfers'] > 5:
            print(f"\n⚠️  WARNING: {stats['total_transfers']} transfers detected (target: <5)")
            print("   Consider using return_gpu=True in KNN operations")
        else:
            print(f"\n✅ Transfer count within target: {stats['total_transfers']} < 5")
        
        if self.track_stacks:
            print("\n" + "-" * 80)
            print("TOP 5 TRANSFER HOTSPOTS")
            print("-" * 80)
            
            hotspots = self.get_hotspots(top_n=5)
            for i, hotspot in enumerate(hotspots, 1):
                print(f"\n{i}. {hotspot['count']} transfers, {hotspot['total_bytes'] / 1e6:.1f} MB")
                print("   Location:")
                # Print last 3 lines of stack trace
                lines = hotspot['stack'].strip().split('\n')
                for line in lines[-3:]:
                    print(f"   {line}")
        
        print("\n" + "=" * 80)
    
    def check_targets(self) -> bool:
        """
        Check if Phase 2 performance targets are met.
        
        Returns:
            True if all targets met, False otherwise
        """
        stats = self.get_stats()
        
        # Phase 2 Target: <5 transfers per operation
        transfers_ok = stats['total_transfers'] < 5
        
        # Reasonable bandwidth (>10 GB/s is good for PCIe 3.0 x16)
        bandwidth_ok = stats['bandwidth_gbps'] > 10 if stats['duration_seconds'] > 0 else True
        
        return transfers_ok and bandwidth_ok


# Global profiler instance (optional convenience)
_global_profiler: Optional[GPUTransferProfiler] = None


def get_global_profiler() -> GPUTransferProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = GPUTransferProfiler()
    return _global_profiler


def enable_automatic_tracking():
    """
    Monkey-patch CuPy to automatically track transfers.
    
    WARNING: This adds overhead. Only use for debugging/profiling.
    
    Usage:
        >>> from ign_lidar.optimization.gpu_transfer_profiler import enable_automatic_tracking
        >>> enable_automatic_tracking()
        >>> # Now all CuPy operations are tracked automatically
    """
    if not CUPY_AVAILABLE:
        logger.warning("CuPy not available, cannot enable automatic tracking")
        return
    
    profiler = get_global_profiler()
    
    # Patch cp.asarray
    original_asarray = cp.asarray
    def tracked_asarray(a, *args, **kwargs):
        result = original_asarray(a, *args, **kwargs)
        if isinstance(a, np.ndarray):
            profiler.record_cpu_to_gpu(a, result)
        return result
    cp.asarray = tracked_asarray
    
    # Patch cp.asnumpy
    original_asnumpy = cp.asnumpy
    def tracked_asnumpy(a, *args, **kwargs):
        result = original_asnumpy(a, *args, **kwargs)
        profiler.record_gpu_to_cpu(a, result)
        return result
    cp.asnumpy = tracked_asnumpy
    
    # Patch array.get()
    original_get = cp.ndarray.get
    def tracked_get(self, *args, **kwargs):
        result = original_get(self, *args, **kwargs)
        profiler.record_gpu_to_cpu(self, result)
        return result
    cp.ndarray.get = tracked_get
    
    logger.info("Automatic GPU transfer tracking enabled")


__all__ = [
    'GPUTransferProfiler',
    'TransferEvent',
    'get_global_profiler',
    'enable_automatic_tracking',
    'CUPY_AVAILABLE',
]
