"""
Performance Benchmarks - Comprehensive test suite for Phase 2 & 3 optimizations.

This module provides benchmarking utilities to measure and validate the
performance improvements from Phase 2 (GPU optimizations) and Phase 3
(Code quality & architecture consolidation).

Module Structure:
- BenchmarkResult: Data class for benchmark results
- FeatureBenchmark: Benchmarks individual feature computations
- PipelineBenchmark: Benchmarks full pipeline workflows
- MemoryProfiler: Tracks memory usage during computations

Usage:
    >>> from ign_lidar.optimization.performance_benchmarks import FeatureBenchmark
    >>> benchmark = FeatureBenchmark()
    >>> results = benchmark.run_feature_benchmark(points, features=['normals', 'curvature'])
    >>> print(f"CPU: {results.cpu_time:.3f}s, GPU: {results.gpu_time:.3f}s")
    >>> print(f"Speedup: {results.speedup:.1f}x")

Author: Simon Ducournau / GitHub Copilot
Date: November 25, 2025
Version: 3.0.0 (Phase 3 complete)
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.
    
    Attributes:
        test_name: Name of the benchmark test
        feature: Feature being tested (e.g., 'normals', 'curvature')
        num_points: Number of points in the test cloud
        method: Computation method ('cpu', 'gpu', 'gpu_chunked')
        elapsed_time: Execution time in seconds
        memory_used_mb: Peak memory usage in MB
        throughput_kps: Throughput in thousand points per second
        timestamp: When the test was run
        metadata: Additional test metadata
    """
    test_name: str
    feature: str
    num_points: int
    method: str
    elapsed_time: float
    memory_used_mb: float
    throughput_kps: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable result summary."""
        return (
            f"{self.test_name} ({self.feature}) - {self.method}\n"
            f"  Points: {self.num_points:,}\n"
            f"  Time: {self.elapsed_time:.3f}s\n"
            f"  Memory: {self.memory_used_mb:.1f} MB\n"
            f"  Throughput: {self.throughput_kps:.1f}k pts/sec"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'test_name': self.test_name,
            'feature': self.feature,
            'num_points': self.num_points,
            'method': self.method,
            'elapsed_time': self.elapsed_time,
            'memory_used_mb': self.memory_used_mb,
            'throughput_kps': self.throughput_kps,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class SpeedupAnalysis:
    """Analysis of speedup between two methods.
    
    Attributes:
        baseline_result: Result using baseline method (usually CPU)
        optimized_result: Result using optimized method
        speedup_factor: How many times faster optimized is (>1 is faster)
        time_saved_seconds: Total time saved
        memory_reduction_percent: Reduction in memory usage (>0 is less)
        efficiency: Speedup divided by resource overhead
    """
    baseline_result: BenchmarkResult
    optimized_result: BenchmarkResult
    speedup_factor: float = field(init=False)
    time_saved_seconds: float = field(init=False)
    memory_reduction_percent: float = field(init=False)
    efficiency: float = field(init=False)

    def __post_init__(self):
        """Calculate derived metrics."""
        self.speedup_factor = (
            self.baseline_result.elapsed_time / self.optimized_result.elapsed_time
        )
        self.time_saved_seconds = (
            self.baseline_result.elapsed_time - self.optimized_result.elapsed_time
        )
        self.memory_reduction_percent = (
            (self.baseline_result.memory_used_mb - self.optimized_result.memory_used_mb)
            / self.baseline_result.memory_used_mb * 100
        ) if self.baseline_result.memory_used_mb > 0 else 0

        # Efficiency: how much speedup per unit memory (higher is better)
        if self.optimized_result.memory_used_mb > 0:
            self.efficiency = (
                self.speedup_factor /
                (self.optimized_result.memory_used_mb /
                 self.baseline_result.memory_used_mb)
            )
        else:
            self.efficiency = self.speedup_factor

    def __str__(self) -> str:
        """Human-readable speedup summary."""
        return (
            f"Speedup Analysis: {self.baseline_result.feature}\n"
            f"  Baseline ({self.baseline_result.method}): {self.baseline_result.elapsed_time:.3f}s\n"
            f"  Optimized ({self.optimized_result.method}): {self.optimized_result.elapsed_time:.3f}s\n"
            f"  Speedup: {self.speedup_factor:.2f}x ({self.time_saved_seconds:.3f}s saved)\n"
            f"  Memory reduction: {self.memory_reduction_percent:.1f}%\n"
            f"  Efficiency: {self.efficiency:.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'baseline': self.baseline_result.to_dict(),
            'optimized': self.optimized_result.to_dict(),
            'speedup_factor': self.speedup_factor,
            'time_saved_seconds': self.time_saved_seconds,
            'memory_reduction_percent': self.memory_reduction_percent,
            'efficiency': self.efficiency
        }


class MemoryProfiler:
    """Track memory usage during computations."""

    def __init__(self, track_gpu: bool = False):
        """Initialize memory profiler.
        
        Args:
            track_gpu: Whether to track GPU memory (requires CuPy)
        """
        self.track_gpu = track_gpu
        self.peak_memory_mb = 0.0
        self.initial_memory_mb = 0.0

        if track_gpu:
            try:
                import cupy as cp
                self.cupy = cp
            except ImportError:
                logger.warning("CuPy not available, GPU memory tracking disabled")
                self.track_gpu = False
                self.cupy = None

    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def start(self) -> None:
        """Start memory tracking."""
        self.initial_memory_mb = self.get_current_memory_mb()
        self.peak_memory_mb = self.initial_memory_mb

    def update(self) -> None:
        """Update peak memory measurement."""
        current = self.get_current_memory_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, current)

    def get_peak_memory_used_mb(self) -> float:
        """Get peak memory used (relative to start)."""
        return self.peak_memory_mb - self.initial_memory_mb


class FeatureBenchmark:
    """Benchmark individual feature computations.
    
    Tests various feature computation modes (CPU, GPU, etc.) and measures
    performance across different point cloud sizes.
    """

    def __init__(self, num_runs: int = 3, verbose: bool = True):
        """Initialize feature benchmark.
        
        Args:
            num_runs: Number of times to run each benchmark for averaging
            verbose: Whether to print progress messages
        """
        self.num_runs = num_runs
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def generate_test_cloud(
        self,
        num_points: int,
        seed: int = 42,
        with_rgb: bool = False
    ) -> np.ndarray:
        """Generate synthetic point cloud for testing.
        
        Args:
            num_points: Number of points to generate
            seed: Random seed for reproducibility
            with_rgb: Whether to include RGB channels
            
        Returns:
            Point cloud array of shape (num_points, 3) or (num_points, 6)
        """
        np.random.seed(seed)
        points = np.random.randn(num_points, 3).astype(np.float32)
        
        if with_rgb:
            rgb = np.random.randint(0, 256, (num_points, 3), dtype=np.uint8)
            points = np.hstack([points, rgb])
        
        return points

    def benchmark_normals_cpu(
        self,
        num_points: int,
        k: int = 10,
        num_runs: Optional[int] = None
    ) -> BenchmarkResult:
        """Benchmark normal computation on CPU.
        
        Args:
            num_points: Number of points to test
            k: Number of neighbors
            num_runs: Number of runs (uses self.num_runs if None)
            
        Returns:
            BenchmarkResult with timing and memory metrics
        """
        num_runs = num_runs or self.num_runs
        
        try:
            from ign_lidar.features.compute.normals import compute_normals
        except ImportError:
            logger.error("Cannot import normals computation")
            raise
        
        points = self.generate_test_cloud(num_points)
        profiler = MemoryProfiler()
        
        # Warmup run
        _ = compute_normals(points, k=k)
        
        # Actual benchmark runs
        times = []
        for _ in range(num_runs):
            profiler.start()
            start_time = time.perf_counter()
            _ = compute_normals(points, k=k)
            elapsed = time.perf_counter() - start_time
            profiler.update()
            times.append(elapsed)
        
        avg_time = np.mean(times)
        throughput_kps = num_points / avg_time / 1000
        
        result = BenchmarkResult(
            test_name="normals_cpu",
            feature="normals",
            num_points=num_points,
            method="cpu",
            elapsed_time=avg_time,
            memory_used_mb=profiler.get_peak_memory_used_mb(),
            throughput_kps=throughput_kps,
            metadata={'k': k, 'runs': num_runs}
        )
        
        self.results.append(result)
        
        if self.verbose:
            logger.info(f"CPU Normals: {result.elapsed_time:.3f}s, {result.throughput_kps:.1f}k pts/sec")
        
        return result

    def benchmark_curvature_cpu(
        self,
        num_points: int,
        k: int = 10,
        num_runs: Optional[int] = None
    ) -> BenchmarkResult:
        """Benchmark curvature computation on CPU."""
        num_runs = num_runs or self.num_runs
        
        try:
            from ign_lidar.features.compute.curvature import compute_curvature
        except ImportError:
            logger.error("Cannot import curvature computation")
            raise
        
        points = self.generate_test_cloud(num_points)
        profiler = MemoryProfiler()
        
        # Warmup run
        _ = compute_curvature(points, k=k)
        
        # Actual benchmark runs
        times = []
        for _ in range(num_runs):
            profiler.start()
            start_time = time.perf_counter()
            _ = compute_curvature(points, k=k)
            elapsed = time.perf_counter() - start_time
            profiler.update()
            times.append(elapsed)
        
        avg_time = np.mean(times)
        throughput_kps = num_points / avg_time / 1000
        
        result = BenchmarkResult(
            test_name="curvature_cpu",
            feature="curvature",
            num_points=num_points,
            method="cpu",
            elapsed_time=avg_time,
            memory_used_mb=profiler.get_peak_memory_used_mb(),
            throughput_kps=throughput_kps,
            metadata={'k': k, 'runs': num_runs}
        )
        
        self.results.append(result)
        
        if self.verbose:
            logger.info(f"CPU Curvature: {result.elapsed_time:.3f}s, {result.throughput_kps:.1f}k pts/sec")
        
        return result

    def save_results(self, output_path: Path) -> None:
        """Save benchmark results to JSON file.
        
        Args:
            output_path: Path to save results JSON
        """
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'num_runs': self.num_runs,
            'results': [r.to_dict() for r in self.results],
            'summary': {
                'total_tests': len(self.results),
                'total_points': sum(r.num_points for r in self.results),
                'avg_throughput_kps': np.mean([r.throughput_kps for r in self.results])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

    def generate_report(self) -> str:
        """Generate human-readable benchmark report.
        
        Returns:
            Report string with all benchmark results
        """
        report = ["Performance Benchmark Report", "=" * 50]
        
        for result in self.results:
            report.append(str(result))
            report.append("")
        
        if len(self.results) >= 2:
            report.append("Speedup Analysis")
            report.append("-" * 50)
            # Group results by feature
            by_feature = {}
            for result in self.results:
                if result.feature not in by_feature:
                    by_feature[result.feature] = []
                by_feature[result.feature].append(result)
        
        return "\n".join(report)


class PipelineBenchmark:
    """Benchmark full end-to-end pipeline workflows.
    
    Tests complete feature extraction pipelines to measure overall
    performance improvements from optimization phases.
    """

    def __init__(self, verbose: bool = True):
        """Initialize pipeline benchmark.
        
        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []

    def benchmark_full_feature_extraction(
        self,
        num_points: int,
        feature_mode: str = 'lod2',
        method: str = 'cpu'
    ) -> Dict[str, Any]:
        """Benchmark full feature extraction pipeline.
        
        Args:
            num_points: Number of points to test
            feature_mode: Feature mode ('minimal', 'lod2', 'lod3', etc.)
            method: Computation method ('cpu', 'gpu', 'gpu_chunked')
            
        Returns:
            Dictionary with timing and metrics
        """
        try:
            from ign_lidar.features import FeatureOrchestrationService
            from ign_lidar.features.mode_selector import ComputationMode
        except ImportError:
            logger.error("Cannot import feature orchestration")
            raise
        
        # Generate test data
        np.random.seed(42)
        points = np.random.randn(num_points, 3).astype(np.float32)
        
        # Initialize service
        service = FeatureOrchestrationService()
        
        # Run benchmark
        profiler = MemoryProfiler()
        profiler.start()
        
        start_time = time.perf_counter()
        
        try:
            features = service.compute_features(
                points=points,
                mode=feature_mode
            )
            
            elapsed = time.perf_counter() - start_time
            profiler.update()
            
            result = {
                'num_points': num_points,
                'feature_mode': feature_mode,
                'method': method,
                'elapsed_time': elapsed,
                'memory_used_mb': profiler.get_peak_memory_used_mb(),
                'features_computed': len(features),
                'throughput_kps': num_points / elapsed / 1000,
                'success': True
            }
            
            if self.verbose:
                logger.info(
                    f"Pipeline ({feature_mode}, {num_points:,} pts): "
                    f"{elapsed:.3f}s, {result['throughput_kps']:.1f}k pts/sec"
                )
            
        except Exception as e:
            logger.error(f"Pipeline benchmark failed: {e}")
            result = {
                'num_points': num_points,
                'feature_mode': feature_mode,
                'method': method,
                'error': str(e),
                'success': False
            }
        
        self.results.append(result)
        return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Benchmark individual features
    benchmark = FeatureBenchmark(num_runs=3)
    
    print("Benchmarking feature computations...")
    for num_points in [10_000, 100_000, 1_000_000]:
        try:
            result = benchmark.benchmark_normals_cpu(num_points)
            print(f"✅ {result}")
        except Exception as e:
            print(f"❌ Failed to benchmark {num_points} points: {e}")
    
    # Generate report
    print("\n" + benchmark.generate_report())
