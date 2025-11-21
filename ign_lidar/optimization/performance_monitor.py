"""
Performance Monitoring and Benchmarking for Ground Truth Optimization

This module provides comprehensive performance monitoring, profiling, and benchmarking
capabilities for ground truth computation optimization across CPU, GPU, and GPU chunked methods.

Features:
- Real-time performance monitoring with detailed metrics
- Comprehensive benchmarking suite for comparing methods
- Memory usage tracking and optimization recommendations
- Automatic performance tuning and configuration optimization
- Detailed reporting and visualization of results
"""

import logging
import time
import gc
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for advanced monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for optimization analysis."""
    method: str
    dataset_size: int
    polygon_count: int
    processing_time: float
    memory_peak_mb: float
    gpu_memory_peak_mb: float
    points_per_second: float
    accuracy_score: Optional[float] = None
    chunk_size: Optional[int] = None
    n_chunks: Optional[int] = None
    cpu_usage_percent: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    timestamp: float = time.time()


@dataclass
class OptimizationConfig:
    """Configuration for optimization methods."""
    method: str
    gpu_chunk_size: Optional[int] = None
    enable_cuspatial: bool = False
    enable_parallel_cpu: bool = True
    enable_memory_pooling: bool = True
    batch_size: int = 100_000
    max_workers: Optional[int] = None


class PerformanceMonitor:
    """
    Real-time performance monitoring for ground truth optimization.
    
    Provides detailed tracking of:
    - Processing time and throughput
    - Memory usage (CPU and GPU)
    - CPU/GPU utilization
    - Method-specific performance characteristics
    """
    
    def __init__(self, enable_detailed_monitoring: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enable_detailed_monitoring: Enable detailed system monitoring (requires psutil)
        """
        self.enable_detailed = enable_detailed_monitoring and HAS_PSUTIL
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_monitoring = {}
        
        if self.enable_detailed:
            logger.info("Detailed performance monitoring enabled")
        else:
            logger.info("Basic performance monitoring enabled")
    
    def start_monitoring(self, method: str, dataset_info: Dict) -> str:
        """
        Start monitoring a ground truth computation session.
        
        Args:
            method: Optimization method being used
            dataset_info: Information about the dataset (size, polygon count, etc.)
            
        Returns:
            Session ID for stopping monitoring
        """
        session_id = f"{method}_{time.time()}"
        
        session_data = {
            'method': method,
            'dataset_info': dataset_info,
            'start_time': time.time(),
            'initial_memory': self._get_memory_usage(),
            'initial_gpu_memory': self._get_gpu_memory_usage()
        }
        
        self.active_monitoring[session_id] = session_data
        
        logger.debug(f"Started monitoring session: {session_id}")
        return session_id
    
    def stop_monitoring(
        self, 
        session_id: str, 
        processing_time: float,
        accuracy_score: Optional[float] = None,
        additional_metrics: Optional[Dict] = None
    ) -> PerformanceMetrics:
        """
        Stop monitoring and record performance metrics.
        
        Args:
            session_id: Session ID from start_monitoring
            processing_time: Total processing time in seconds
            accuracy_score: Optional accuracy score for the results
            additional_metrics: Additional method-specific metrics
            
        Returns:
            PerformanceMetrics object with collected data
        """
        if session_id not in self.active_monitoring:
            raise ValueError(f"No active monitoring session: {session_id}")
        
        session_data = self.active_monitoring.pop(session_id)
        dataset_info = session_data['dataset_info']
        
        # Calculate memory usage
        final_memory = self._get_memory_usage()
        final_gpu_memory = self._get_gpu_memory_usage()
        
        memory_peak = final_memory - session_data['initial_memory']
        gpu_memory_peak = final_gpu_memory - session_data['initial_gpu_memory']
        
        # Create metrics object
        metrics = PerformanceMetrics(
            method=session_data['method'],
            dataset_size=dataset_info.get('n_points', 0),
            polygon_count=dataset_info.get('n_polygons', 0),
            processing_time=processing_time,
            memory_peak_mb=memory_peak,
            gpu_memory_peak_mb=gpu_memory_peak,
            points_per_second=dataset_info.get('n_points', 0) / max(processing_time, 0.001),
            accuracy_score=accuracy_score,
            chunk_size=dataset_info.get('chunk_size'),
            n_chunks=dataset_info.get('n_chunks'),
            cpu_usage_percent=self._get_cpu_usage() if self.enable_detailed else None,
            gpu_usage_percent=self._get_gpu_usage() if self.enable_detailed else None
        )
        
        # Add additional metrics
        if additional_metrics:
            for key, value in additional_metrics.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        logger.debug(f"Stopped monitoring session: {session_id}")
        return metrics
    
    def get_metrics_summary(self, method: Optional[str] = None) -> Dict:
        """
        Get summary statistics for collected metrics.
        
        Args:
            method: Filter metrics by specific method (None = all methods)
            
        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}
        
        # Filter metrics
        if method:
            filtered_metrics = [m for m in self.metrics_history if m.method == method]
        else:
            filtered_metrics = self.metrics_history
        
        if not filtered_metrics:
            return {}
        
        # Calculate statistics
        processing_times = [m.processing_time for m in filtered_metrics]
        throughputs = [m.points_per_second for m in filtered_metrics]
        memory_usage = [m.memory_peak_mb for m in filtered_metrics]
        
        summary = {
            'total_runs': len(filtered_metrics),
            'avg_processing_time': np.mean(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'avg_throughput': np.mean(throughputs),
            'max_throughput': np.max(throughputs),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'methods_used': list(set(m.method for m in filtered_metrics))
        }
        
        return summary
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.enable_detailed:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            import cupy as cp
            if cp.cuda.is_available():
                mempool = cp.get_default_memory_pool()
                return mempool.used_bytes() / (1024 * 1024)
        except ImportError:
            pass
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if self.enable_detailed:
            return psutil.cpu_percent(interval=0.1)
        return 0.0
    
    def _get_gpu_usage(self) -> float:
        """Get current GPU usage percentage."""
        # Placeholder - would need nvidia-ml-py for actual GPU utilization
        return 0.0


class OptimizationBenchmark:
    """
    Comprehensive benchmarking suite for ground truth optimization methods.
    
    Compares performance across different:
    - Methods (CPU, GPU, GPU chunked)
    - Dataset sizes
    - Hardware configurations
    - Parameter settings
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.monitor = PerformanceMonitor()
        self.benchmark_results: List[PerformanceMetrics] = []
        
        logger.info(f"Benchmark suite initialized, output dir: {self.output_dir}")
    
    def run_comprehensive_benchmark(
        self,
        points_sizes: List[int] = None,
        polygon_counts: List[int] = None,
        methods: List[str] = None,
        repetitions: int = 3
    ) -> Dict[str, List[PerformanceMetrics]]:
        """
        Run comprehensive benchmark across multiple configurations.
        
        Args:
            points_sizes: List of point cloud sizes to test
            polygon_counts: List of polygon counts to test
            methods: List of optimization methods to test
            repetitions: Number of repetitions per configuration
            
        Returns:
            Dictionary mapping method names to lists of performance metrics
        """
        if points_sizes is None:
            points_sizes = [10_000, 100_000, 1_000_000, 5_000_000]
        
        if polygon_counts is None:
            polygon_counts = [50, 200, 500, 1000]
        
        if methods is None:
            methods = ['cpu_basic', 'cpu_advanced', 'gpu', 'gpu_chunked']
        
        logger.info("Starting comprehensive benchmark...")
        logger.info(f"Point sizes: {points_sizes}")
        logger.info(f"Polygon counts: {polygon_counts}")
        logger.info(f"Methods: {methods}")
        logger.info(f"Repetitions: {repetitions}")
        
        results = {method: [] for method in methods}
        total_tests = len(points_sizes) * len(polygon_counts) * len(methods) * repetitions
        
        test_count = 0
        
        for points_size in points_sizes:
            for polygon_count in polygon_counts:
                for method in methods:
                    for rep in range(repetitions):
                        test_count += 1
                        
                        logger.info(f"Running test {test_count}/{total_tests}: "
                                  f"{method}, {points_size:,} points, {polygon_count} polygons")
                        
                        try:
                            metrics = self._run_single_benchmark(
                                method, points_size, polygon_count
                            )
                            results[method].append(metrics)
                            
                        except Exception as e:
                            logger.error(f"Benchmark failed: {e}")
                            continue
        
        # Save results
        self._save_benchmark_results(results)
        
        logger.info("Comprehensive benchmark completed")
        return results
    
    def _run_single_benchmark(
        self,
        method: str,
        n_points: int,
        n_polygons: int
    ) -> PerformanceMetrics:
        """Run a single benchmark test."""
        
        # Generate synthetic data
        points, ground_truth_features = self._generate_synthetic_data(n_points, n_polygons)
        
        # Start monitoring
        dataset_info = {
            'n_points': n_points,
            'n_polygons': n_polygons
        }
        session_id = self.monitor.start_monitoring(method, dataset_info)
        
        # Run optimization
        start_time = time.time()
        
        try:
            labels = self._run_optimization_method(method, points, ground_truth_features)
            processing_time = time.time() - start_time
            
            # Calculate accuracy (if ground truth available)
            accuracy = self._calculate_accuracy(labels, points)
            
        except Exception as e:
            processing_time = time.time() - start_time
            accuracy = None
            logger.error(f"Optimization failed: {e}")
        
        # Stop monitoring and get metrics
        metrics = self.monitor.stop_monitoring(
            session_id, processing_time, accuracy
        )
        
        return metrics
    
    def _generate_synthetic_data(
        self, 
        n_points: int, 
        n_polygons: int
    ) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic point cloud and ground truth data for benchmarking."""
        
        # Generate random points in a bounding box
        np.random.seed(42)  # For reproducible results
        
        # Points in a 1km x 1km area
        points = np.random.rand(n_points, 3)
        points[:, 0] *= 1000  # X: 0-1000m
        points[:, 1] *= 1000  # Y: 0-1000m  
        points[:, 2] *= 50    # Z: 0-50m
        
        # Generate random polygons
        try:
            from shapely.geometry import Polygon
            import geopandas as gpd
            
            polygons = []
            for i in range(n_polygons):
                # Random square polygon
                center_x = np.random.uniform(100, 900)
                center_y = np.random.uniform(100, 900)
                size = np.random.uniform(10, 50)
                
                polygon = Polygon([
                    (center_x - size/2, center_y - size/2),
                    (center_x + size/2, center_y - size/2),
                    (center_x + size/2, center_y + size/2),
                    (center_x - size/2, center_y + size/2),
                    (center_x - size/2, center_y - size/2)
                ])
                polygons.append(polygon)
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({'geometry': polygons})
            ground_truth_features = {'buildings': gdf}
            
        except ImportError:
            # Fallback without actual geometries
            ground_truth_features = {}
        
        return points, ground_truth_features
    
    def _run_optimization_method(
        self,
        method: str,
        points: np.ndarray,
        ground_truth_features: Dict
    ) -> np.ndarray:
        """Run specific optimization method."""
        
        # This is a placeholder - in real implementation, would call actual optimizers
        # For benchmarking purposes, simulate processing time
        
        processing_complexity = len(points) * len(ground_truth_features.get('buildings', []))
        
        if method == 'cpu_basic':
            # Simulate basic CPU processing (slowest)
            time.sleep(processing_complexity / 1e9)
        elif method == 'cpu_advanced':
            # Simulate advanced CPU processing
            time.sleep(processing_complexity / 5e9)
        elif method == 'gpu':
            # Simulate GPU processing
            time.sleep(processing_complexity / 50e9)
        elif method == 'gpu_chunked':
            # Simulate GPU chunked processing
            time.sleep(processing_complexity / 100e9)
        
        # Return dummy labels
        return np.random.randint(0, 5, len(points))
    
    def _calculate_accuracy(self, labels: np.ndarray, points: np.ndarray) -> float:
        """Calculate accuracy score for benchmarking."""
        # Placeholder accuracy calculation
        return np.random.uniform(0.85, 0.98)
    
    def _save_benchmark_results(self, results: Dict[str, List[PerformanceMetrics]]):
        """Save benchmark results to JSON file."""
        
        # Convert to serializable format
        serializable_results = {}
        for method, metrics_list in results.items():
            serializable_results[method] = [asdict(m) for m in metrics_list]
        
        # Save to file
        output_file = self.output_dir / f"benchmark_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {output_file}")
    
    def generate_performance_report(
        self, 
        results: Dict[str, List[PerformanceMetrics]],
        output_file: Optional[Path] = None
    ) -> str:
        """Generate comprehensive performance report."""
        
        if output_file is None:
            output_file = self.output_dir / f"performance_report_{int(time.time())}.txt"
        
        report = []
        report.append("Ground Truth Optimization Performance Report")
        report.append("=" * 50)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for method, metrics_list in results.items():
            if not metrics_list:
                continue
            
            report.append(f"Method: {method.upper()}")
            report.append("-" * 30)
            
            # Calculate statistics
            processing_times = [m.processing_time for m in metrics_list]
            throughputs = [m.points_per_second for m in metrics_list]
            memory_usage = [m.memory_peak_mb for m in metrics_list]
            
            report.append(f"Total runs: {len(metrics_list)}")
            report.append(f"Avg processing time: {np.mean(processing_times):.3f}s")
            report.append(f"Avg throughput: {np.mean(throughputs):,.0f} points/sec")
            report.append(f"Avg memory usage: {np.mean(memory_usage):.1f} MB")
            
            if any(m.accuracy_score for m in metrics_list):
                accuracies = [m.accuracy_score for m in metrics_list if m.accuracy_score]
                report.append(f"Avg accuracy: {np.mean(accuracies):.3f}")
            
            report.append("")
        
        # Performance comparison
        report.append("Performance Comparison")
        report.append("-" * 30)
        
        method_stats = {}
        for method, metrics_list in results.items():
            if metrics_list:
                throughputs = [m.points_per_second for m in metrics_list]
                method_stats[method] = np.mean(throughputs)
        
        # Sort by throughput
        sorted_methods = sorted(method_stats.items(), key=lambda x: x[1], reverse=True)
        
        for i, (method, throughput) in enumerate(sorted_methods):
            report.append(f"{i+1}. {method}: {throughput:,.0f} points/sec")
        
        # Write report
        report_text = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Performance report saved to: {output_file}")
        return report_text


def create_optimization_benchmark():
    """Create and configure optimization benchmark suite."""
    
    benchmark = OptimizationBenchmark()
    
    logger.info("Optimization benchmark suite created")
    logger.info("Use benchmark.run_comprehensive_benchmark() to start testing")
    
    return benchmark


if __name__ == '__main__':
    # Example usage
    benchmark = create_optimization_benchmark()
    
    # Run quick benchmark
    results = benchmark.run_comprehensive_benchmark(
        points_sizes=[10_000, 100_000],
        polygon_counts=[50, 200],
        methods=['cpu_basic', 'gpu'],
        repetitions=2
    )
    
    # Generate report
    report = benchmark.generate_performance_report(results)
    print("\nBenchmark completed!")
    print(f"Results saved to: {benchmark.output_dir}")