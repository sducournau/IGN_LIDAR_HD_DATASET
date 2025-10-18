#!/usr/bin/env python3
"""
Benchmark Script for Phase 4 Sprint 1 - GPU Array Operations & CUDA Kernels

This script benchmarks the performance improvements from:
1. CuPy-accelerated array operations
2. Custom CUDA kernels for critical paths

Tests:
- Statistical operations (mean, std, percentiles)
- Distance calculations
- Array transformations
- Feature computation with custom kernels

Author: IGN LiDAR HD Development Team
Date: October 18, 2025
"""

import numpy as np
import time
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ign_lidar.optimization.gpu_array_ops import GPUArrayOps, HAS_CUPY
    from ign_lidar.optimization.gpu_kernels import CUDAKernels
except ImportError as e:
    logger.error(f"Failed to import GPU modules: {e}")
    sys.exit(1)


class Phase4Sprint1Benchmark:
    """Benchmark Phase 4 Sprint 1 GPU operations."""
    
    def __init__(self, n_points: int = 1_000_000):
        """
        Initialize benchmark.
        
        Args:
            n_points: Number of points for testing
        """
        self.n_points = n_points
        self.results = {}
        
        # Generate test data
        logger.info(f"Generating test data: {n_points:,} points")
        np.random.seed(42)
        self.points = np.random.randn(n_points, 3).astype(np.float32)
        self.knn_indices = np.random.randint(0, n_points, (n_points, 30), dtype=np.int32)
        
        # Initialize GPU operations
        self.gpu_ops = GPUArrayOps(use_gpu=True)
        self.cuda_kernels = CUDAKernels()
        
        logger.info(f"GPU available: {self.gpu_ops.use_gpu}")
        logger.info(f"CUDA kernels available: {self.cuda_kernels.available}")
    
    def benchmark_statistical_operations(self):
        """Benchmark statistical operations (mean, std, percentiles)."""
        logger.info("\n" + "="*70)
        logger.info("Benchmarking Statistical Operations")
        logger.info("="*70)
        
        results = {}
        
        # Test 1: Mean computation
        logger.info("\n1. Computing mean...")
        
        # CPU (NumPy)
        start = time.time()
        cpu_mean = np.mean(self.points, axis=0)
        cpu_time = time.time() - start
        logger.info(f"   CPU: {cpu_time:.4f}s")
        
        # GPU (CuPy)
        if self.gpu_ops.use_gpu:
            start = time.time()
            gpu_mean = self.gpu_ops.compute_mean(self.points, axis=0)
            gpu_time = time.time() - start
            logger.info(f"   GPU: {gpu_time:.4f}s")
            
            speedup = cpu_time / gpu_time
            error = np.max(np.abs(cpu_mean - gpu_mean))
            logger.info(f"   Speedup: {speedup:.2f}x")
            logger.info(f"   Max error: {error:.2e}")
            
            results['mean'] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'max_error': float(error)
            }
        
        # Test 2: Standard deviation
        logger.info("\n2. Computing standard deviation...")
        
        start = time.time()
        cpu_std = np.std(self.points, axis=0)
        cpu_time = time.time() - start
        logger.info(f"   CPU: {cpu_time:.4f}s")
        
        if self.gpu_ops.use_gpu:
            start = time.time()
            gpu_std = self.gpu_ops.compute_std(self.points, axis=0)
            gpu_time = time.time() - start
            logger.info(f"   GPU: {gpu_time:.4f}s")
            
            speedup = cpu_time / gpu_time
            error = np.max(np.abs(cpu_std - gpu_std))
            logger.info(f"   Speedup: {speedup:.2f}x")
            logger.info(f"   Max error: {error:.2e}")
            
            results['std'] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'max_error': float(error)
            }
        
        # Test 3: Percentiles
        logger.info("\n3. Computing percentiles (25th, 50th, 75th)...")
        
        start = time.time()
        cpu_percentiles = np.percentile(self.points, [25, 50, 75], axis=0)
        cpu_time = time.time() - start
        logger.info(f"   CPU: {cpu_time:.4f}s")
        
        if self.gpu_ops.use_gpu:
            start = time.time()
            gpu_percentiles = self.gpu_ops.compute_percentile(self.points, [25, 50, 75], axis=0)
            gpu_time = time.time() - start
            logger.info(f"   GPU: {gpu_time:.4f}s")
            
            speedup = cpu_time / gpu_time
            error = np.max(np.abs(cpu_percentiles - gpu_percentiles))
            logger.info(f"   Speedup: {speedup:.2f}x")
            logger.info(f"   Max error: {error:.2e}")
            
            results['percentiles'] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'max_error': float(error)
            }
        
        self.results['statistical_operations'] = results
    
    def benchmark_distance_calculations(self):
        """Benchmark distance calculations."""
        logger.info("\n" + "="*70)
        logger.info("Benchmarking Distance Calculations")
        logger.info("="*70)
        
        results = {}
        
        # Reduce size for pairwise distances (too large otherwise)
        small_points = self.points[:10000]
        
        logger.info(f"\nTesting with {len(small_points):,} points...")
        
        # Test: Pairwise distances
        logger.info("\n1. Computing pairwise distances...")
        
        # CPU (scipy)
        start = time.time()
        from scipy.spatial.distance import cdist
        cpu_distances = cdist(small_points, small_points)
        cpu_time = time.time() - start
        logger.info(f"   CPU: {cpu_time:.4f}s")
        
        # GPU (CuPy)
        if self.gpu_ops.use_gpu:
            start = time.time()
            gpu_distances = self.gpu_ops.compute_pairwise_distances(small_points, small_points)
            gpu_time = time.time() - start
            logger.info(f"   GPU: {gpu_time:.4f}s")
            
            speedup = cpu_time / gpu_time
            error = np.max(np.abs(cpu_distances - gpu_distances))
            logger.info(f"   Speedup: {speedup:.2f}x")
            logger.info(f"   Max error: {error:.2e}")
            
            results['pairwise_distances'] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'max_error': float(error),
                'points': len(small_points)
            }
        
        self.results['distance_calculations'] = results
    
    def benchmark_cuda_kernels(self):
        """Benchmark custom CUDA kernels."""
        logger.info("\n" + "="*70)
        logger.info("Benchmarking Custom CUDA Kernels")
        logger.info("="*70)
        
        if not self.cuda_kernels.available:
            logger.warning("CUDA kernels not available, skipping")
            return
        
        results = {}
        k = 30  # Number of neighbors
        
        # Test 1: KNN distance computation
        logger.info(f"\n1. Computing KNN distances (k={k})...")
        
        # CPU reference (simplified)
        start = time.time()
        cpu_distances = np.zeros((len(self.points), k), dtype=np.float32)
        for i in range(min(1000, len(self.points))):  # Only test 1000 points on CPU
            for j in range(k):
                neighbor_idx = self.knn_indices[i, j]
                diff = self.points[i] - self.points[neighbor_idx]
                cpu_distances[i, j] = np.sqrt(np.sum(diff**2))
        cpu_time = time.time() - start
        # Extrapolate to full dataset
        cpu_time_estimated = cpu_time * (len(self.points) / 1000)
        logger.info(f"   CPU (estimated): {cpu_time_estimated:.4f}s")
        
        # GPU (CUDA kernel)
        start = time.time()
        gpu_distances = self.cuda_kernels.compute_knn_distances(
            self.points, self.knn_indices, k
        )
        gpu_time = time.time() - start
        logger.info(f"   GPU: {gpu_time:.4f}s")
        
        speedup = cpu_time_estimated / gpu_time
        # Compare first 1000 points
        error = np.max(np.abs(cpu_distances[:1000] - gpu_distances[:1000]))
        logger.info(f"   Speedup: {speedup:.2f}x")
        logger.info(f"   Max error (1000 points): {error:.2e}")
        
        results['knn_distances'] = {
            'cpu_time_estimated': cpu_time_estimated,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'max_error': float(error)
        }
        
        # Test 2: Covariance computation
        logger.info(f"\n2. Computing covariance matrices...")
        
        start = time.time()
        gpu_covariance, gpu_centroids = self.cuda_kernels.compute_covariance(
            self.points, self.knn_indices, k
        )
        gpu_time = time.time() - start
        logger.info(f"   GPU: {gpu_time:.4f}s")
        logger.info(f"   Computed {len(gpu_covariance):,} covariance matrices")
        
        results['covariance'] = {
            'gpu_time': gpu_time,
            'n_matrices': len(gpu_covariance)
        }
        
        # Test 3: Geometric features
        logger.info(f"\n3. Computing geometric features from eigenvalues...")
        
        # Generate dummy eigenvalues
        eigenvalues = np.random.rand(len(self.points), 3).astype(np.float32)
        eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]  # Sort descending
        
        start = time.time()
        gpu_features = self.cuda_kernels.compute_geometric_features(eigenvalues)
        gpu_time = time.time() - start
        logger.info(f"   GPU: {gpu_time:.4f}s")
        logger.info(f"   Computed {gpu_features.shape[1]} features for {len(gpu_features):,} points")
        
        results['geometric_features'] = {
            'gpu_time': gpu_time,
            'n_features': gpu_features.shape[1],
            'n_points': len(gpu_features)
        }
        
        self.results['cuda_kernels'] = results
    
    def benchmark_transformations(self):
        """Benchmark array transformations."""
        logger.info("\n" + "="*70)
        logger.info("Benchmarking Array Transformations")
        logger.info("="*70)
        
        results = {}
        
        # Test: Normalization
        logger.info("\n1. Z-score normalization...")
        
        # CPU
        start = time.time()
        mean = np.mean(self.points, axis=0, keepdims=True)
        std = np.std(self.points, axis=0, keepdims=True)
        cpu_normalized = (self.points - mean) / (std + 1e-8)
        cpu_time = time.time() - start
        logger.info(f"   CPU: {cpu_time:.4f}s")
        
        # GPU
        if self.gpu_ops.use_gpu:
            start = time.time()
            gpu_normalized = self.gpu_ops.normalize(self.points, axis=0, method='zscore')
            gpu_time = time.time() - start
            logger.info(f"   GPU: {gpu_time:.4f}s")
            
            speedup = cpu_time / gpu_time
            error = np.max(np.abs(cpu_normalized - gpu_normalized))
            logger.info(f"   Speedup: {speedup:.2f}x")
            logger.info(f"   Max error: {error:.2e}")
            
            results['normalization'] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'max_error': float(error)
            }
        
        self.results['transformations'] = results
    
    def run_all_benchmarks(self):
        """Run all benchmarks and save results."""
        logger.info("\n" + "="*70)
        logger.info("Phase 4 Sprint 1 Benchmark Suite")
        logger.info("="*70)
        logger.info(f"Test dataset: {self.n_points:,} points")
        
        if not self.gpu_ops.use_gpu:
            logger.error("GPU not available! Cannot run benchmarks.")
            return
        
        # Run benchmarks
        self.benchmark_statistical_operations()
        self.benchmark_distance_calculations()
        self.benchmark_transformations()
        self.benchmark_cuda_kernels()
        
        # Summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print benchmark summary."""
        logger.info("\n" + "="*70)
        logger.info("SUMMARY - Phase 4 Sprint 1 Performance Gains")
        logger.info("="*70)
        
        all_speedups = []
        
        for category, tests in self.results.items():
            if not tests:
                continue
                
            logger.info(f"\n{category.upper().replace('_', ' ')}:")
            
            for test_name, metrics in tests.items():
                if 'speedup' in metrics:
                    speedup = metrics['speedup']
                    all_speedups.append(speedup)
                    error = metrics.get('max_error', 0)
                    logger.info(f"  • {test_name}: {speedup:.2f}x speedup (error: {error:.2e})")
        
        if all_speedups:
            avg_speedup = np.mean(all_speedups)
            logger.info(f"\n{'='*70}")
            logger.info(f"AVERAGE SPEEDUP: {avg_speedup:.2f}x")
            logger.info(f"TARGET: 5-10x (GPU array ops), 10-20x (CUDA kernels)")
            
            if avg_speedup >= 5:
                logger.info("✅ TARGET ACHIEVED!")
            else:
                logger.warning("⚠️  Below target, needs optimization")
            
            logger.info(f"{'='*70}")
    
    def save_results(self):
        """Save benchmark results to JSON."""
        output_file = Path("benchmark_phase4_sprint1.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n✅ Results saved to: {output_file}")


def main():
    """Run Phase 4 Sprint 1 benchmarks."""
    # Test with different sizes
    sizes = [
        100_000,    # Small
        1_000_000,  # Medium
        # 10_000_000  # Large (uncomment if you have enough GPU memory)
    ]
    
    for n_points in sizes:
        logger.info(f"\n{'#'*70}")
        logger.info(f"Testing with {n_points:,} points")
        logger.info(f"{'#'*70}")
        
        benchmark = Phase4Sprint1Benchmark(n_points=n_points)
        benchmark.run_all_benchmarks()
        
        # Clear GPU memory between runs
        if benchmark.gpu_ops.use_gpu:
            benchmark.gpu_ops.clear_cache()


if __name__ == '__main__':
    main()
