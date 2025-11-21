#!/usr/bin/env python3
"""
Phase 3: Façade Processing Optimization Benchmark

Comprehensive benchmark for GPU KNN, parallel processing, and vectorization
in facade_processor.py. Validates expected 20-30× speedup from:
- Phase 3.1: GPU KNN (15-20× from Phase 1.4)
- Phase 3.2: Parallel facades (4× speedup)
- Phase 3.3: Vectorized calculations (100× for projections)

Usage:
    python scripts/benchmark_facade_optimization.py
    python scripts/benchmark_facade_optimization.py --size 100000
    python scripts/benchmark_facade_optimization.py --detailed

Author: Performance Optimization Team
Date: November 20, 2025
"""

import argparse
import logging
import time
import numpy as np
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_building_points(n_points: int = 50000) -> Dict[str, np.ndarray]:
    """
    Generate realistic building point cloud for benchmarking.
    
    Args:
        n_points: Total number of points to generate
        
    Returns:
        Dictionary with points, normals, verticality, heights
    """
    np.random.seed(42)
    
    # Rectangular building: 20m x 15m x 10m (height)
    logger.info(f"Generating {n_points} building points...")
    
    points = []
    normals = []
    
    # Distribute points across 4 walls
    points_per_wall = n_points // 4
    
    # North wall (y=15)
    x_north = np.random.uniform(0, 20, points_per_wall)
    y_north = np.full(points_per_wall, 15.0) + np.random.normal(0, 0.1, points_per_wall)
    z_north = np.random.uniform(0, 10, points_per_wall)
    points.append(np.column_stack([x_north, y_north, z_north]))
    normals.append(np.tile([0, 1, 0], (points_per_wall, 1)))
    
    # South wall (y=0)
    x_south = np.random.uniform(0, 20, points_per_wall)
    y_south = np.full(points_per_wall, 0.0) + np.random.normal(0, 0.1, points_per_wall)
    z_south = np.random.uniform(0, 10, points_per_wall)
    points.append(np.column_stack([x_south, y_south, z_south]))
    normals.append(np.tile([0, -1, 0], (points_per_wall, 1)))
    
    # East wall (x=20)
    x_east = np.full(points_per_wall, 20.0) + np.random.normal(0, 0.1, points_per_wall)
    y_east = np.random.uniform(0, 15, points_per_wall)
    z_east = np.random.uniform(0, 10, points_per_wall)
    points.append(np.column_stack([x_east, y_east, z_east]))
    normals.append(np.tile([1, 0, 0], (points_per_wall, 1)))
    
    # West wall (x=0)
    remaining = n_points - 3 * points_per_wall
    x_west = np.full(remaining, 0.0) + np.random.normal(0, 0.1, remaining)
    y_west = np.random.uniform(0, 15, remaining)
    z_west = np.random.uniform(0, 10, remaining)
    points.append(np.column_stack([x_west, y_west, z_west]))
    normals.append(np.tile([-1, 0, 0], (remaining, 1)))
    
    all_points = np.vstack(points)
    all_normals = np.vstack(normals).astype(np.float64)
    
    # Add noise to normals
    all_normals = all_normals + np.random.normal(0, 0.05, all_normals.shape)
    norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
    all_normals = all_normals / norms
    
    # Compute verticality (1.0 = vertical wall)
    verticality = 1.0 - np.abs(all_normals[:, 2])
    
    return {
        "points": all_points,
        "normals": all_normals,
        "verticality": verticality,
        "heights": all_points[:, 2],
    }


def benchmark_knn_gpu_vs_cpu(data: Dict, k: int = 30) -> Dict[str, float]:
    """
    Benchmark GPU vs CPU KNN performance.
    
    Phase 3.1: Expected 15-20× speedup from Phase 1.4 results
    """
    from ign_lidar.optimization.gpu_accelerated_ops import knn, set_force_cpu
    
    logger.info("\n" + "="*60)
    logger.info("Phase 3.1: GPU KNN Benchmark")
    logger.info("="*60)
    
    points = data["points"][:, :2]  # XY only
    n_points = len(points)
    
    # GPU benchmark
    set_force_cpu(False)
    start = time.time()
    distances_gpu, indices_gpu = knn(points, k=k)
    time_gpu = time.time() - start
    
    # CPU benchmark
    set_force_cpu(True)
    start = time.time()
    distances_cpu, indices_cpu = knn(points, k=k)
    time_cpu = time.time() - start
    set_force_cpu(False)  # Reset
    
    speedup = time_cpu / time_gpu if time_gpu > 0 else 1.0
    
    logger.info(f"\nKNN Performance ({n_points:,} points, k={k}):")
    logger.info(f"  GPU: {time_gpu:.3f}s ({n_points/time_gpu:,.0f} points/sec)")
    logger.info(f"  CPU: {time_cpu:.3f}s ({n_points/time_cpu:,.0f} points/sec)")
    logger.info(f"  🚀 Speedup: {speedup:.1f}×")
    
    if speedup >= 10:
        logger.info(f"  ✅ EXCELLENT (target: 15-20×)")
    elif speedup >= 5:
        logger.info(f"  ✅ GOOD (above 5×)")
    else:
        logger.warning(f"  ⚠️  Below expected (target: 15-20×)")
    
    return {
        "time_gpu": time_gpu,
        "time_cpu": time_cpu,
        "speedup": speedup,
    }


def benchmark_parallel_facades(data: Dict) -> Dict[str, float]:
    """
    Benchmark parallel vs sequential facade processing.
    
    Phase 3.2: Expected 4× speedup (4 facades in parallel)
    """
    from ign_lidar.core.classification.building.facade_processor import (
        BuildingFacadeClassifier
    )
    from shapely.geometry import Polygon
    
    logger.info("\n" + "="*60)
    logger.info("Phase 3.2: Parallel Facade Processing Benchmark")
    logger.info("="*60)
    
    points = data["points"]
    n_points = len(points)
    labels = np.zeros(n_points, dtype=np.uint8)
    polygon = Polygon([(0, 0), (20, 0), (20, 15), (0, 15)])
    
    # Parallel benchmark
    classifier_parallel = BuildingFacadeClassifier(
        enable_parallel_facades=True,
        max_workers=4
    )
    
    start = time.time()
    labels_parallel, stats_parallel = classifier_parallel.classify_buildings(
        building_id=1,
        points=points,
        labels=labels.copy(),
        polygon=polygon,
        normals=data["normals"],
        verticality=data["verticality"],
    )
    time_parallel = time.time() - start
    
    # Sequential benchmark
    classifier_sequential = BuildingFacadeClassifier(
        enable_parallel_facades=False
    )
    
    start = time.time()
    labels_sequential, stats_sequential = classifier_sequential.classify_buildings(
        building_id=1,
        points=points,
        labels=labels.copy(),
        polygon=polygon,
        normals=data["normals"],
        verticality=data["verticality"],
    )
    time_sequential = time.time() - start
    
    speedup = time_sequential / time_parallel if time_parallel > 0 else 1.0
    
    logger.info(f"\nFacade Processing ({n_points:,} points):")
    logger.info(f"  Parallel (4 workers): {time_parallel:.3f}s")
    logger.info(f"  Sequential:           {time_sequential:.3f}s")
    logger.info(f"  🚀 Speedup: {speedup:.1f}×")
    
    if speedup >= 3.0:
        logger.info(f"  ✅ EXCELLENT (target: 4×)")
    elif speedup >= 2.0:
        logger.info(f"  ✅ GOOD (above 2×)")
    else:
        logger.warning(f"  ⚠️  Below expected (target: 4×)")
    
    # Validate results consistency
    n_classified_parallel = np.sum(labels_parallel != 0)
    n_classified_sequential = np.sum(labels_sequential != 0)
    consistency = 1.0 - abs(n_classified_parallel - n_classified_sequential) / max(
        n_classified_parallel, n_classified_sequential
    )
    
    logger.info(f"\nResults Consistency: {consistency*100:.1f}%")
    logger.info(f"  Parallel classified:   {n_classified_parallel:,} points")
    logger.info(f"  Sequential classified: {n_classified_sequential:,} points")
    
    return {
        "time_parallel": time_parallel,
        "time_sequential": time_sequential,
        "speedup": speedup,
        "consistency": consistency,
    }


def benchmark_vectorized_projection(n_points: int = 100000) -> Dict[str, float]:
    """
    Benchmark vectorized vs loop-based projection calculation.
    
    Phase 3.3: Expected 100× speedup for projection operations
    """
    from shapely.geometry import LineString, Point
    
    logger.info("\n" + "="*60)
    logger.info("Phase 3.3: Vectorized Projection Benchmark")
    logger.info("="*60)
    
    # Generate test points along a line
    np.random.seed(42)
    x = np.linspace(0, 20, n_points)
    y = np.full(n_points, 10.0) + np.random.normal(0, 0.1, n_points)
    points = np.column_stack([x, y])
    
    line = LineString([(0, 10), (20, 10)])
    
    # Vectorized projection
    line_coords = np.array(line.coords)
    p1, p2 = line_coords[0], line_coords[1]
    line_vec = p2 - p1
    line_length = np.linalg.norm(line_vec)
    line_vec_normalized = line_vec / line_length
    
    start = time.time()
    point_vecs = points - p1
    projected_vectorized = np.dot(point_vecs, line_vec_normalized)
    projected_vectorized = np.clip(projected_vectorized, 0, line_length)
    time_vectorized = time.time() - start
    
    # Loop-based projection (sample subset for speed)
    n_sample = min(1000, n_points)
    points_sample = points[:n_sample]
    
    start = time.time()
    projected_loop = np.array([line.project(Point(pt)) for pt in points_sample])
    time_loop_sample = time.time() - start
    
    # Extrapolate to full dataset
    time_loop_extrapolated = time_loop_sample * (n_points / n_sample)
    
    speedup = time_loop_extrapolated / time_vectorized if time_vectorized > 0 else 1.0
    
    logger.info(f"\nProjection Performance ({n_points:,} points):")
    logger.info(f"  Vectorized: {time_vectorized:.4f}s ({n_points/time_vectorized:,.0f} points/sec)")
    logger.info(f"  Loop (extrapolated): {time_loop_extrapolated:.4f}s")
    logger.info(f"  🚀 Speedup: {speedup:.0f}×")
    
    if speedup >= 50:
        logger.info(f"  ✅ EXCELLENT (target: 100×)")
    elif speedup >= 20:
        logger.info(f"  ✅ GOOD (above 20×)")
    else:
        logger.warning(f"  ⚠️  Below expected (target: 100×)")
    
    return {
        "time_vectorized": time_vectorized,
        "time_loop": time_loop_extrapolated,
        "speedup": speedup,
    }


def benchmark_full_pipeline(data: Dict) -> Dict[str, float]:
    """
    Benchmark full optimized pipeline vs baseline.
    
    Expected: 20-30× total speedup from combined optimizations
    """
    from ign_lidar.core.classification.building.facade_processor import (
        BuildingFacadeClassifier
    )
    from shapely.geometry import Polygon
    
    logger.info("\n" + "="*60)
    logger.info("Full Pipeline: All Optimizations Combined")
    logger.info("="*60)
    
    points = data["points"]
    n_points = len(points)
    labels = np.zeros(n_points, dtype=np.uint8)
    polygon = Polygon([(0, 0), (20, 0), (20, 15), (0, 15)])
    
    # Optimized pipeline (all Phase 3 features enabled)
    classifier_optimized = BuildingFacadeClassifier(
        enable_parallel_facades=True,  # Phase 3.2
        max_workers=4,
    )
    
    start = time.time()
    labels_optimized, stats_optimized = classifier_optimized.classify_buildings(
        building_id=1,
        points=points,
        labels=labels.copy(),
        polygon=polygon,
        normals=data["normals"],
        verticality=data["verticality"],
    )
    time_optimized = time.time() - start
    
    logger.info(f"\n🚀 Optimized Pipeline ({n_points:,} points):")
    logger.info(f"  Time: {time_optimized:.3f}s")
    logger.info(f"  Throughput: {n_points/time_optimized:,.0f} points/sec")
    logger.info(f"  Facades processed: {stats_optimized.get('facades_processed', 0)}")
    logger.info(f"  Points classified: {np.sum(labels_optimized != 0):,}")
    
    # Performance targets based on roadmap
    if n_points == 50000:
        baseline_time = 8.0  # Baseline from Phase 2
        target_time = 2.0    # Target from Phase 3
        
        logger.info(f"\nPerformance vs Targets:")
        logger.info(f"  Baseline (Phase 2): {baseline_time:.1f}s")
        logger.info(f"  Target (Phase 3):   {target_time:.1f}s")
        logger.info(f"  Achieved:           {time_optimized:.3f}s")
        
        achieved_speedup = baseline_time / time_optimized
        target_speedup = baseline_time / target_time
        
        logger.info(f"\n  Achieved speedup: {achieved_speedup:.1f}×")
        logger.info(f"  Target speedup:   {target_speedup:.1f}×")
        
        if time_optimized <= target_time:
            logger.info(f"  ✅ TARGET MET!")
        else:
            improvement_needed = time_optimized / target_time
            logger.warning(f"  ⚠️  Need {improvement_needed:.1f}× more improvement")
    
    return {
        "time_optimized": time_optimized,
        "throughput": n_points / time_optimized,
    }


def extrapolate_to_real_tile(results: Dict, tile_points: int = 18_000_000):
    """Extrapolate benchmark results to real IGN LiDAR HD tile."""
    logger.info("\n" + "="*60)
    logger.info("Extrapolation to Real IGN LiDAR HD Tile")
    logger.info("="*60)
    
    # Assume tile has ~100 buildings on average
    n_buildings = 100
    points_per_building = tile_points / n_buildings
    
    logger.info(f"\nTile characteristics:")
    logger.info(f"  Total points: {tile_points:,}")
    logger.info(f"  Buildings: {n_buildings}")
    logger.info(f"  Points per building: {points_per_building:,.0f}")
    
    # Use full pipeline results for extrapolation
    if "time_optimized" in results:
        throughput = results.get("throughput", 10000)
        time_per_tile = tile_points / throughput
        
        logger.info(f"\nExpected processing time:")
        logger.info(f"  Optimized pipeline: {time_per_tile:.1f}s ({time_per_tile/60:.1f} min)")
        
        # Compare with roadmap targets
        baseline_time = 33 * 60  # 33 min baseline
        target_time = 2.5 * 60   # 2.5 min target (Phase 3+)
        
        logger.info(f"\nComparison with roadmap:")
        logger.info(f"  Baseline (pre-optimization): {baseline_time/60:.1f} min")
        logger.info(f"  Ultimate target (all phases): {target_time/60:.1f} min")
        logger.info(f"  Current (Phase 3):            {time_per_tile/60:.1f} min")
        
        improvement_ratio = baseline_time / time_per_tile
        logger.info(f"\n  🚀 Improvement so far: {improvement_ratio:.1f}×")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Phase 3 facade processing optimizations"
    )
    parser.add_argument(
        "--size", type=int, default=50000,
        help="Number of points to generate (default: 50000)"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Run detailed component benchmarks"
    )
    parser.add_argument(
        "--skip-knn", action="store_true",
        help="Skip KNN benchmark (slow on CPU)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Phase 3: Façade Processing Optimization Benchmark")
    logger.info("="*60)
    logger.info(f"Dataset size: {args.size:,} points")
    logger.info(f"Detailed mode: {args.detailed}")
    
    # Generate test data
    data = generate_building_points(n_points=args.size)
    
    results = {}
    
    # Component benchmarks
    if args.detailed:
        if not args.skip_knn:
            results["knn"] = benchmark_knn_gpu_vs_cpu(data)
        
        results["parallel"] = benchmark_parallel_facades(data)
        results["vectorized"] = benchmark_vectorized_projection(n_points=100000)
    
    # Full pipeline benchmark
    results["pipeline"] = benchmark_full_pipeline(data)
    
    # Extrapolate to real tiles
    extrapolate_to_real_tile(results)
    
    logger.info("\n" + "="*60)
    logger.info("Benchmark Complete")
    logger.info("="*60)


if __name__ == "__main__":
    main()
