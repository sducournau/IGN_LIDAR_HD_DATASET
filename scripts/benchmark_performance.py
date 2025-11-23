"""
Automated Performance Benchmarks for CI/CD

Implements recommendations from November 2025 code audit (PRIORITY_ACTIONS.md)
for automated performance regression detection.

Tracks:
- Feature computation speed (CPU/GPU)
- Memory usage
- GPU utilization
- Transfer overhead
- Regression detection vs baseline

Usage:
    # Run all benchmarks
    python scripts/benchmark_performance.py
    
    # Quick benchmarks (for PR checks)
    python scripts/benchmark_performance.py --quick
    
    # Save results for baseline
    python scripts/benchmark_performance.py --save baseline_v3.8.0.json
    
    # Check for regressions
    python scripts/benchmark_performance.py --baseline baseline_v3.8.0.json --threshold 0.05
    
    # CI/CD mode (exit 1 on regression)
    python scripts/benchmark_performance.py --ci --baseline baseline.json

Author: IGN LiDAR HD Team
Date: November 23, 2025
Version: 3.8.1-dev
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU availability check
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("✓ GPU available (CuPy detected)")
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    logger.warning("✗ GPU not available - GPU benchmarks will be skipped")


@dataclass
class BenchmarkMetric:
    """Single benchmark metric."""
    name: str
    value: float
    unit: str
    lower_is_better: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    name: str
    timestamp: str
    version: str
    gpu_available: bool
    metrics: List[BenchmarkMetric]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'version': self.version,
            'gpu_available': self.gpu_available,
            'metrics': [m.to_dict() for m in self.metrics],
            'metadata': self.metadata
        }


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite for CI/CD."""
    
    def __init__(self, quick_mode: bool = False):
        """
        Initialize benchmark suite.
        
        Args:
            quick_mode: If True, run fewer iterations for faster results
        """
        self.quick_mode = quick_mode
        self.iterations = 3 if quick_mode else 10
        self.warmup_iterations = 1 if quick_mode else 2
        self.dataset_sizes = [1_000_000] if quick_mode else [1_000_000, 5_000_000, 10_000_000]
        
        # Import IGN LiDAR modules
        try:
            from ign_lidar.features import FeatureOrchestrator
            from ign_lidar.core.gpu import GPUManager
            self.FeatureOrchestrator = FeatureOrchestrator
            self.GPUManager = GPUManager
            logger.info("✓ IGN LiDAR modules loaded")
        except ImportError as e:
            logger.error(f"Failed to import IGN LiDAR modules: {e}")
            sys.exit(1)
    
    def generate_test_data(self, num_points: int) -> np.ndarray:
        """Generate synthetic point cloud."""
        logger.info(f"Generating test data: {num_points:,} points")
        return np.random.randn(num_points, 3).astype(np.float32)
    
    def benchmark_features_cpu(self, num_points: int) -> BenchmarkMetric:
        """Benchmark CPU feature computation."""
        logger.info(f"Benchmarking CPU features ({num_points:,} points)...")
        
        points = self.generate_test_data(num_points)
        orchestrator = self.FeatureOrchestrator(use_gpu=False)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = orchestrator.compute_features(points, mode='lod2', show_progress=False)
        
        # Benchmark
        times = []
        for i in range(self.iterations):
            start = time.time()
            _ = orchestrator.compute_features(points, mode='lod2', show_progress=False)
            elapsed = time.time() - start
            times.append(elapsed)
            logger.debug(f"  Iteration {i+1}/{self.iterations}: {elapsed:.3f}s")
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = num_points / mean_time / 1_000_000  # M points/s
        
        logger.info(f"  CPU: {mean_time:.3f}±{std_time:.3f}s, {throughput:.2f}M pts/s")
        
        return BenchmarkMetric(
            name=f'cpu_features_{num_points//1_000_000}M',
            value=mean_time,
            unit='seconds',
            lower_is_better=True
        )
    
    def benchmark_features_gpu(self, num_points: int) -> Optional[BenchmarkMetric]:
        """Benchmark GPU feature computation."""
        if not GPU_AVAILABLE:
            return None
        
        logger.info(f"Benchmarking GPU features ({num_points:,} points)...")
        
        points = self.generate_test_data(num_points)
        orchestrator = self.FeatureOrchestrator(use_gpu=True)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = orchestrator.compute_features(points, mode='lod2', show_progress=False)
            cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        times = []
        for i in range(self.iterations):
            start = time.time()
            _ = orchestrator.compute_features(points, mode='lod2', show_progress=False)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
            logger.debug(f"  Iteration {i+1}/{self.iterations}: {elapsed:.3f}s")
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = num_points / mean_time / 1_000_000  # M points/s
        
        logger.info(f"  GPU: {mean_time:.3f}±{std_time:.3f}s, {throughput:.2f}M pts/s")
        
        return BenchmarkMetric(
            name=f'gpu_features_{num_points//1_000_000}M',
            value=mean_time,
            unit='seconds',
            lower_is_better=True
        )
    
    def benchmark_ground_truth_gpu(self) -> Optional[BenchmarkMetric]:
        """Benchmark GPU ground truth processing."""
        if not GPU_AVAILABLE:
            return None
        
        logger.info("Benchmarking GPU ground truth processing...")
        
        try:
            from ign_lidar.optimization.ground_truth import process_ground_truth_gpu_chunked
            
            # Generate test data
            num_points = 1_000_000
            points = self.generate_test_data(num_points)
            
            # Mock ground truth polygons
            from shapely.geometry import Polygon
            polygons = [
                Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
                Polygon([(100, 0), (200, 0), (200, 100), (100, 100)]),
            ]
            
            # Warmup
            for _ in range(self.warmup_iterations):
                _ = process_ground_truth_gpu_chunked(
                    points, polygons, 
                    chunk_size=500_000,
                    show_progress=False
                )
                cp.cuda.Stream.null.synchronize()
            
            # Benchmark
            times = []
            for _ in range(self.iterations):
                start = time.time()
                _ = process_ground_truth_gpu_chunked(
                    points, polygons,
                    chunk_size=500_000,
                    show_progress=False
                )
                cp.cuda.Stream.null.synchronize()
                elapsed = time.time() - start
                times.append(elapsed)
            
            mean_time = np.mean(times)
            logger.info(f"  Ground truth GPU: {mean_time:.3f}s")
            
            return BenchmarkMetric(
                name='ground_truth_gpu_1M',
                value=mean_time,
                unit='seconds',
                lower_is_better=True
            )
            
        except Exception as e:
            logger.warning(f"Ground truth GPU benchmark failed: {e}")
            return None
    
    def benchmark_transfers(self) -> Optional[BenchmarkMetric]:
        """Benchmark GPU transfer overhead."""
        if not GPU_AVAILABLE:
            return None
        
        logger.info("Benchmarking GPU transfers...")
        
        num_points = 5_000_000
        data = np.random.randn(num_points, 3).astype(np.float32)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            gpu_data = cp.asarray(data)
            _ = cp.asnumpy(gpu_data)
        
        # Benchmark upload
        upload_times = []
        for _ in range(self.iterations):
            start = time.time()
            gpu_data = cp.asarray(data)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start
            upload_times.append(elapsed)
        
        # Benchmark download
        gpu_data = cp.asarray(data)
        download_times = []
        for _ in range(self.iterations):
            start = time.time()
            _ = cp.asnumpy(gpu_data)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start
            download_times.append(elapsed)
        
        total_transfer = np.mean(upload_times) + np.mean(download_times)
        logger.info(
            f"  Upload: {np.mean(upload_times)*1000:.1f}ms, "
            f"Download: {np.mean(download_times)*1000:.1f}ms, "
            f"Total: {total_transfer*1000:.1f}ms"
        )
        
        return BenchmarkMetric(
            name='gpu_transfer_overhead_5M',
            value=total_transfer,
            unit='seconds',
            lower_is_better=True
        )
    
    def benchmark_memory_usage(self) -> Optional[BenchmarkMetric]:
        """Benchmark GPU memory usage."""
        if not GPU_AVAILABLE:
            return None
        
        logger.info("Benchmarking GPU memory usage...")
        
        try:
            gpu_manager = self.GPUManager()
            
            if not gpu_manager.memory:
                return None
            
            initial_used = gpu_manager.memory.get_memory_usage_gb()
            
            # Process large dataset
            num_points = 5_000_000
            points = self.generate_test_data(num_points)
            orchestrator = self.FeatureOrchestrator(use_gpu=True)
            _ = orchestrator.compute_features(points, mode='lod2', show_progress=False)
            
            peak_used = gpu_manager.memory.get_peak_memory_gb()
            allocated = peak_used - initial_used
            
            logger.info(f"  Peak memory allocation: {allocated:.2f}GB")
            
            return BenchmarkMetric(
                name='gpu_memory_peak_5M',
                value=allocated,
                unit='GB',
                lower_is_better=True
            )
            
        except Exception as e:
            logger.warning(f"Memory benchmark failed: {e}")
            return None
    
    def run_all_benchmarks(self) -> BenchmarkResult:
        """Run all benchmarks and return results."""
        logger.info("=" * 70)
        logger.info("STARTING PERFORMANCE BENCHMARK SUITE")
        logger.info("=" * 70)
        logger.info(f"Mode: {'QUICK' if self.quick_mode else 'FULL'}")
        logger.info(f"Iterations: {self.iterations}")
        logger.info(f"Dataset sizes: {[f'{s//1_000_000}M' for s in self.dataset_sizes]}")
        logger.info("=" * 70)
        
        metrics = []
        
        # CPU benchmarks
        logger.info("\n📊 CPU BENCHMARKS")
        for num_points in self.dataset_sizes:
            metric = self.benchmark_features_cpu(num_points)
            if metric:
                metrics.append(metric)
        
        # GPU benchmarks
        if GPU_AVAILABLE:
            logger.info("\n🚀 GPU BENCHMARKS")
            
            # Feature computation
            for num_points in self.dataset_sizes:
                metric = self.benchmark_features_gpu(num_points)
                if metric:
                    metrics.append(metric)
            
            # Ground truth
            metric = self.benchmark_ground_truth_gpu()
            if metric:
                metrics.append(metric)
            
            # Transfers
            metric = self.benchmark_transfers()
            if metric:
                metrics.append(metric)
            
            # Memory
            metric = self.benchmark_memory_usage()
            if metric:
                metrics.append(metric)
        
        # Build result
        result = BenchmarkResult(
            name='performance_benchmark',
            timestamp=datetime.now().isoformat(),
            version='3.8.1-dev',
            gpu_available=GPU_AVAILABLE,
            metrics=metrics,
            metadata={
                'quick_mode': self.quick_mode,
                'iterations': self.iterations,
                'dataset_sizes': self.dataset_sizes,
            }
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK SUITE COMPLETE")
        logger.info("=" * 70)
        
        return result


def check_regressions(
    current: BenchmarkResult,
    baseline: BenchmarkResult,
    threshold: float = 0.05
) -> Tuple[bool, List[str]]:
    """
    Check for performance regressions vs baseline.
    
    Args:
        current: Current benchmark results
        baseline: Baseline results to compare against
        threshold: Regression threshold (e.g., 0.05 = 5% slower is regression)
    
    Returns:
        Tuple of (has_regression, regression_messages)
    """
    logger.info("\n🔍 CHECKING FOR REGRESSIONS")
    logger.info(f"Threshold: {threshold*100:.1f}%")
    
    # Build baseline lookup
    baseline_metrics = {m['name']: m['value'] for m in baseline['metrics']}
    
    has_regression = False
    messages = []
    
    for metric in current.metrics:
        if metric.name not in baseline_metrics:
            logger.warning(f"  ⚠️  New metric '{metric.name}' (no baseline)")
            continue
        
        baseline_value = baseline_metrics[metric.name]
        current_value = metric.value
        
        # Calculate percent change
        if baseline_value == 0:
            continue
        
        change = (current_value - baseline_value) / baseline_value
        
        # Check if regression (performance got worse)
        is_regression = False
        if metric.lower_is_better and change > threshold:
            is_regression = True
        elif not metric.lower_is_better and change < -threshold:
            is_regression = True
        
        # Log result
        status = "🔴 REGRESSION" if is_regression else "✅ OK"
        logger.info(
            f"  {status} {metric.name}: "
            f"{baseline_value:.3f} → {current_value:.3f} "
            f"({change*100:+.1f}%)"
        )
        
        if is_regression:
            has_regression = True
            messages.append(
                f"Regression in {metric.name}: "
                f"{change*100:+.1f}% slower (threshold: {threshold*100:.1f}%)"
            )
    
    return has_regression, messages


def save_results(result: BenchmarkResult, filepath: Path):
    """Save benchmark results to JSON file."""
    logger.info(f"\n💾 Saving results to {filepath}")
    
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info("✓ Results saved")


def load_baseline(filepath: Path) -> Optional[Dict]:
    """Load baseline results from JSON file."""
    if not filepath.exists():
        logger.error(f"Baseline file not found: {filepath}")
        return None
    
    logger.info(f"📂 Loading baseline from {filepath}")
    
    with open(filepath, 'r') as f:
        baseline = json.load(f)
    
    logger.info(f"✓ Baseline loaded (version: {baseline.get('version', 'unknown')})")
    return baseline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Automated Performance Benchmarks for CI/CD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python scripts/benchmark_performance.py
  
  # Quick benchmarks (for PR checks)
  python scripts/benchmark_performance.py --quick
  
  # Save baseline
  python scripts/benchmark_performance.py --save baseline_v3.8.0.json
  
  # Check regressions
  python scripts/benchmark_performance.py --baseline baseline_v3.8.0.json
  
  # CI/CD mode (exit 1 on regression)
  python scripts/benchmark_performance.py --ci --baseline baseline.json --threshold 0.05
        """
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (fewer iterations, smaller datasets)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        metavar='FILE',
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        metavar='FILE',
        help='Compare against baseline JSON file'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='Regression threshold (default: 0.05 = 5%%)'
    )
    
    parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode: exit with code 1 if regressions detected'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run benchmarks
    suite = PerformanceBenchmarkSuite(quick_mode=args.quick)
    result = suite.run_all_benchmarks()
    
    # Print summary
    logger.info("\n📊 BENCHMARK SUMMARY")
    for metric in result.metrics:
        logger.info(f"  {metric.name}: {metric.value:.3f} {metric.unit}")
    
    # Save results
    if args.save:
        save_results(result, Path(args.save))
    
    # Check regressions
    if args.baseline:
        baseline = load_baseline(Path(args.baseline))
        if baseline:
            has_regression, messages = check_regressions(
                result, baseline, args.threshold
            )
            
            if has_regression:
                logger.error("\n❌ PERFORMANCE REGRESSIONS DETECTED:")
                for msg in messages:
                    logger.error(f"  • {msg}")
                
                if args.ci:
                    sys.exit(1)
            else:
                logger.info("\n✅ NO REGRESSIONS DETECTED")
    
    logger.info("\n✨ Benchmark complete!")


if __name__ == '__main__':
    main()
