"""
Real-time Performance Monitoring for IGN LiDAR HD Processing
Tracks GPU/CPU utilization, memory usage, and processing speed.
Version: 1.7.4
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance measurement."""
    timestamp: float
    elapsed_time: float
    points_processed: int
    points_per_second: float
    ram_used_gb: float
    ram_available_gb: float
    swap_used_percent: float
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    cpu_percent: Optional[float] = None
    stage: str = "processing"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    total_points: int = 0
    total_time: float = 0.0
    avg_points_per_second: float = 0.0
    peak_ram_gb: float = 0.0
    peak_vram_gb: float = 0.0
    avg_gpu_utilization: float = 0.0
    avg_cpu_percent: float = 0.0
    snapshots: List[PerformanceSnapshot] = field(default_factory=list)
    stages: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert snapshots to list of dicts
        data['snapshots'] = [s.to_dict() for s in self.snapshots]
        return data


class PerformanceMonitor:
    """
    Real-time performance monitoring for LiDAR processing.
    
    Features:
    - CPU/GPU utilization tracking
    - Memory usage monitoring
    - Processing speed metrics
    - Stage timing
    - Export to JSON
    
    Example:
        >>> monitor = PerformanceMonitor()
        >>> monitor.start_stage('normals')
        >>> # ... processing ...
        >>> monitor.end_stage('normals')
        >>> monitor.record_progress(points_processed=1000000)
        >>> report = monitor.generate_report()
    """
    
    def __init__(
        self,
        enable_gpu_monitoring: bool = True,
        sampling_interval: float = 1.0
    ):
        """
        Initialize performance monitor.
        
        Args:
            enable_gpu_monitoring: Monitor GPU if available
            sampling_interval: How often to sample metrics (seconds)
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring and GPU_AVAILABLE
        self.sampling_interval = sampling_interval
        
        self.start_time = time.time()
        self.last_sample_time = self.start_time
        
        self.metrics = PerformanceMetrics()
        self.current_stage = None
        self.stage_start_times: Dict[str, float] = {}
        
        # Process info
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None
            logger.warning("psutil not available - limited monitoring")
    
    def start_stage(self, stage_name: str):
        """Start timing a processing stage."""
        self.current_stage = stage_name
        self.stage_start_times[stage_name] = time.time()
        logger.info(f"📊 Starting stage: {stage_name}")
    
    def end_stage(self, stage_name: Optional[str] = None):
        """End timing a processing stage."""
        if stage_name is None:
            stage_name = self.current_stage
        
        if stage_name and stage_name in self.stage_start_times:
            elapsed = time.time() - self.stage_start_times[stage_name]
            self.metrics.stages[stage_name] = elapsed
            logger.info(
                f"✓ Completed stage: {stage_name} "
                f"({elapsed:.1f}s)"
            )
            
            if stage_name == self.current_stage:
                self.current_stage = None
    
    def record_progress(
        self,
        points_processed: int,
        force_sample: bool = False
    ):
        """
        Record progress and sample metrics.
        
        Args:
            points_processed: Total points processed so far
            force_sample: Force sampling even if interval not reached
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Check if we should sample
        if not force_sample:
            if current_time - self.last_sample_time < self.sampling_interval:
                return
        
        self.last_sample_time = current_time
        
        # Update total
        self.metrics.total_points = points_processed
        self.metrics.total_time = elapsed
        
        # Calculate rate
        points_per_sec = points_processed / elapsed if elapsed > 0 else 0
        self.metrics.avg_points_per_second = points_per_sec
        
        # Get system metrics
        ram_info = self._get_ram_info()
        gpu_info = self._get_gpu_info() if self.enable_gpu_monitoring else {}
        cpu_percent = self._get_cpu_percent()
        
        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestamp=current_time,
            elapsed_time=elapsed,
            points_processed=points_processed,
            points_per_second=points_per_sec,
            ram_used_gb=ram_info['used_gb'],
            ram_available_gb=ram_info['available_gb'],
            swap_used_percent=ram_info['swap_percent'],
            vram_used_gb=gpu_info.get('used_gb'),
            vram_total_gb=gpu_info.get('total_gb'),
            gpu_utilization=gpu_info.get('utilization'),
            cpu_percent=cpu_percent,
            stage=self.current_stage or "processing"
        )
        
        self.metrics.snapshots.append(snapshot)
        
        # Update peaks
        self.metrics.peak_ram_gb = max(
            self.metrics.peak_ram_gb, ram_info['used_gb']
        )
        if gpu_info.get('used_gb'):
            self.metrics.peak_vram_gb = max(
                self.metrics.peak_vram_gb, gpu_info['used_gb']
            )
    
    def _get_ram_info(self) -> Dict[str, float]:
        """Get RAM usage information."""
        if not PSUTIL_AVAILABLE:
            return {
                'used_gb': 0.0,
                'available_gb': 0.0,
                'swap_percent': 0.0
            }
        
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'used_gb': (mem.total - mem.available) / (1024**3),
                'available_gb': mem.available / (1024**3),
                'swap_percent': swap.percent
            }
        except Exception as e:
            logger.debug(f"Failed to get RAM info: {e}")
            return {
                'used_gb': 0.0,
                'available_gb': 0.0,
                'swap_percent': 0.0
            }
    
    def _get_gpu_info(self) -> Dict[str, float]:
        """Get GPU usage information."""
        if not GPU_AVAILABLE or cp is None:
            return {}
        
        try:
            mempool = cp.get_default_memory_pool()
            device = cp.cuda.Device()
            free_mem, total_mem = device.mem_info
            
            return {
                'used_gb': mempool.used_bytes() / (1024**3),
                'total_gb': total_mem / (1024**3),
                'free_gb': free_mem / (1024**3),
                'utilization': (1.0 - free_mem / total_mem) * 100
            }
        except Exception as e:
            logger.debug(f"Failed to get GPU info: {e}")
            return {}
    
    def _get_cpu_percent(self) -> Optional[float]:
        """Get CPU usage percentage."""
        if not PSUTIL_AVAILABLE or not self.process:
            return None
        
        try:
            return self.process.cpu_percent(interval=0.1)
        except Exception as e:
            logger.debug(f"Failed to get CPU percent: {e}")
            return None
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        elapsed = time.time() - self.start_time
        
        ram_info = self._get_ram_info()
        gpu_info = self._get_gpu_info() if self.enable_gpu_monitoring else {}
        
        stats = {
            'elapsed_time': elapsed,
            'total_points': self.metrics.total_points,
            'points_per_second': self.metrics.avg_points_per_second,
            'ram_used_gb': ram_info['used_gb'],
            'ram_available_gb': ram_info['available_gb'],
            'swap_used_percent': ram_info['swap_percent'],
            'current_stage': self.current_stage
        }
        
        if gpu_info:
            stats.update({
                'vram_used_gb': gpu_info.get('used_gb'),
                'vram_total_gb': gpu_info.get('total_gb'),
                'vram_free_gb': gpu_info.get('free_gb'),
                'gpu_utilization': gpu_info.get('utilization')
            })
        
        return stats
    
    def print_current_stats(self):
        """Print current statistics to console."""
        stats = self.get_current_stats()
        
        print(f"\n{'='*70}")
        print(f"📊 Performance Monitor - {stats['elapsed_time']:.1f}s elapsed")
        print(f"{'='*70}")
        
        if stats['current_stage']:
            print(f"Stage: {stats['current_stage']}")
        
        print(f"Points: {stats['total_points']:,}")
        print(f"Speed: {stats['points_per_second']:,.0f} points/sec")
        
        print(f"\nRAM: {stats['ram_used_gb']:.1f}GB used, "
              f"{stats['ram_available_gb']:.1f}GB available")
        if stats['swap_used_percent'] > 0:
            print(f"Swap: {stats['swap_used_percent']:.0f}%")
        
        if 'vram_used_gb' in stats and stats['vram_used_gb'] is not None:
            print(f"\nVRAM: {stats['vram_used_gb']:.1f}GB / "
                  f"{stats['vram_total_gb']:.1f}GB")
            if stats.get('gpu_utilization'):
                print(f"GPU Utilization: {stats['gpu_utilization']:.0f}%")
        
        print(f"{'='*70}\n")
    
    def generate_report(self) -> PerformanceMetrics:
        """
        Generate final performance report.
        
        Returns:
            PerformanceMetrics with complete statistics
        """
        # Calculate averages
        if self.metrics.snapshots:
            valid_gpu_utils = [
                s.gpu_utilization for s in self.metrics.snapshots
                if s.gpu_utilization is not None
            ]
            if valid_gpu_utils:
                self.metrics.avg_gpu_utilization = (
                    sum(valid_gpu_utils) / len(valid_gpu_utils)
                )
            
            valid_cpu_percents = [
                s.cpu_percent for s in self.metrics.snapshots
                if s.cpu_percent is not None
            ]
            if valid_cpu_percents:
                self.metrics.avg_cpu_percent = (
                    sum(valid_cpu_percents) / len(valid_cpu_percents)
                )
        
        return self.metrics
    
    def export_json(self, output_path: Path):
        """
        Export metrics to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"📊 Performance report saved to: {output_path}")
    
    def export_summary(self, output_path: Path):
        """
        Export summary report to text file.
        
        Args:
            output_path: Path to output text file
        """
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("IGN LiDAR HD Processing - Performance Report\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Points: {report.total_points:,}\n")
            f.write(f"Total Time: {report.total_time:.1f}s\n")
            f.write(f"Average Speed: {report.avg_points_per_second:,.0f} "
                   f"points/sec\n\n")
            
            f.write(f"Peak RAM: {report.peak_ram_gb:.1f}GB\n")
            if report.peak_vram_gb > 0:
                f.write(f"Peak VRAM: {report.peak_vram_gb:.1f}GB\n")
            
            if report.avg_gpu_utilization > 0:
                f.write(f"Avg GPU Utilization: "
                       f"{report.avg_gpu_utilization:.0f}%\n")
            
            if report.avg_cpu_percent > 0:
                f.write(f"Avg CPU Usage: {report.avg_cpu_percent:.0f}%\n")
            
            if report.stages:
                f.write("\nStage Timings:\n")
                for stage, duration in report.stages.items():
                    f.write(f"  {stage}: {duration:.1f}s\n")
            
            f.write("\n" + "="*70 + "\n")
        
        logger.info(f"📊 Summary report saved to: {output_path}")


def create_monitor(
    enable_gpu: bool = True,
    sampling_interval: float = 1.0
) -> PerformanceMonitor:
    """
    Create and return a performance monitor.
    
    Args:
        enable_gpu: Enable GPU monitoring
        sampling_interval: Sampling interval in seconds
    
    Returns:
        PerformanceMonitor instance
    """
    return PerformanceMonitor(
        enable_gpu_monitoring=enable_gpu,
        sampling_interval=sampling_interval
    )
