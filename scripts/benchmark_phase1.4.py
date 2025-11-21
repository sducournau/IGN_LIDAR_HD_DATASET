"""
Phase 1.4 Migration Performance Benchmark

Test performance improvements from GPU KDTree migration in real feature computation.

Usage:
    python scripts/benchmark_phase1.4.py --size medium
    python scripts/benchmark_phase1.4.py --size large --full
    
Author: IGN LiDAR HD Development Team
Date: November 2025
"""

import argparse
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def benchmark_feature_utils():
    """Benchmark features/utils.py compute_local_eigenvalues."""
    from ign_lidar.features.utils import compute_local_eigenvalues
    from ign_lidar.optimization import set_force_cpu
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK: features.utils.compute_local_eigenvalues()")
    logger.info("="*80)
    
    points = np.random.rand(50000, 3).astype(np.float32)
    k = 30
    
    # GPU
    set_force_cpu(False)
    start = time.time()
    eigenvalues_gpu = compute_local_eigenvalues(points, k=k)
    time_gpu = time.time() - start
    
    # CPU
    set_force_cpu(True)
    start = time.time()
    eigenvalues_cpu = compute_local_eigenvalues(points, k=k)
    time_cpu = time.time() - start
    
    speedup = time_cpu / time_gpu
    
    logger.info(f"\n  Points: {len(points):,} | k={k}")
    logger.info(f"  🚀 GPU: {time_gpu:.3f}s")
    logger.info(f"  💻 CPU: {time_cpu:.3f}s")
    logger.info(f"  📊 Speedup: {speedup:.1f}×")
    
    # Reset
    set_force_cpu(False)
    
    return speedup


def benchmark_geometric_features():
    """Benchmark features/compute/geometric.py extract_geometric_features."""
    from ign_lidar.features.compute.geometric import extract_geometric_features
    from ign_lidar.optimization import set_force_cpu
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK: features.compute.geometric.extract_geometric_features()")
    logger.info("="*80)
    
    points = np.random.rand(30000, 3).astype(np.float32)
    k = 30
    
    # GPU
    set_force_cpu(False)
    start = time.time()
    features_gpu = extract_geometric_features(points, k_neighbors=k)
    time_gpu = time.time() - start
    
    # CPU
    set_force_cpu(True)
    start = time.time()
    features_cpu = extract_geometric_features(points, k_neighbors=k)
    time_cpu = time.time() - start
    
    speedup = time_cpu / time_gpu
    
    logger.info(f"\n  Points: {len(points):,} | k={k}")
    logger.info(f"  Features: {len(features_gpu)} computed")
    logger.info(f"  🚀 GPU: {time_gpu:.3f}s")
    logger.info(f"  💻 CPU: {time_cpu:.3f}s")
    logger.info(f"  📊 Speedup: {speedup:.1f}×")
    
    # Reset
    set_force_cpu(False)
    
    return speedup


def benchmark_tile_stitcher():
    """Benchmark core/tile_stitcher.py neighbor detection."""
    from ign_lidar.core.tile_stitcher import TileStitcher
    from ign_lidar.optimization import set_force_cpu, cKDTree
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK: core.tile_stitcher KDTree operations")
    logger.info("="*80)
    
    points = np.random.rand(100000, 3).astype(np.float32)
    query = np.random.rand(10000, 3).astype(np.float32)
    k = 10
    
    # GPU
    set_force_cpu(False)
    start = time.time()
    tree_gpu = cKDTree(points)
    distances_gpu, indices_gpu = tree_gpu.query(query, k=k)
    time_gpu = time.time() - start
    
    # CPU
    set_force_cpu(True)
    start = time.time()
    tree_cpu = cKDTree(points)
    distances_cpu, indices_cpu = tree_cpu.query(query, k=k)
    time_cpu = time.time() - start
    
    speedup = time_cpu / time_gpu
    
    logger.info(f"\n  Tree points: {len(points):,}")
    logger.info(f"  Query points: {len(query):,} | k={k}")
    logger.info(f"  🚀 GPU: {time_gpu:.3f}s")
    logger.info(f"  💻 CPU: {time_cpu:.3f}s")
    logger.info(f"  📊 Speedup: {speedup:.1f}×")
    
    # Reset
    set_force_cpu(False)
    
    return speedup


def benchmark_classification():
    """Benchmark classification KDTree operations."""
    from ign_lidar.core.classification.geometric_rules import apply_geometric_rules
    from ign_lidar.optimization import set_force_cpu
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK: classification.geometric_rules (simplified)")
    logger.info("="*80)
    
    # Simulate classified points
    points = np.random.rand(20000, 3).astype(np.float32)
    classification = np.random.randint(1, 7, len(points), dtype=np.uint8)
    
    # Note: geometric_rules uses KDTree internally
    # We'll measure a representative operation
    from ign_lidar.optimization import cKDTree
    
    # GPU
    set_force_cpu(False)
    start = time.time()
    tree_gpu = cKDTree(points)
    # Simulate classification refinement queries
    for _ in range(10):
        dists, indices = tree_gpu.query(points[:1000], k=20)
    time_gpu = time.time() - start
    
    # CPU
    set_force_cpu(True)
    start = time.time()
    tree_cpu = cKDTree(points)
    for _ in range(10):
        dists, indices = tree_cpu.query(points[:1000], k=20)
    time_cpu = time.time() - start
    
    speedup = time_cpu / time_gpu
    
    logger.info(f"\n  Points: {len(points):,}")
    logger.info(f"  Operations: 10 batches × 1000 queries")
    logger.info(f"  🚀 GPU: {time_gpu:.3f}s")
    logger.info(f"  💻 CPU: {time_cpu:.3f}s")
    logger.info(f"  📊 Speedup: {speedup:.1f}×")
    
    # Reset
    set_force_cpu(False)
    
    return speedup


def main():
    parser = argparse.ArgumentParser(description='Benchmark Phase 1.4 GPU KDTree migration')
    parser.add_argument('--size', choices=['small', 'medium', 'large'], 
                        default='medium', help='Dataset size')
    parser.add_argument('--full', action='store_true', 
                        help='Run all benchmarks (slower)')
    
    args = parser.parse_args()
    
    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "PHASE 1.4 PERFORMANCE BENCHMARK" + " "*26 + "║")
    logger.info("╚" + "="*78 + "╝")
    
    speedups = []
    
    try:
        # Always run these
        speedups.append(("Feature Utils", benchmark_feature_utils()))
        speedups.append(("Tile Stitcher", benchmark_tile_stitcher()))
        
        if args.full:
            speedups.append(("Geometric Features", benchmark_geometric_features()))
            speedups.append(("Classification", benchmark_classification()))
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\n{'Module':<25} {'Speedup':>10}")
        logger.info("-" * 40)
        for name, speedup in speedups:
            logger.info(f"{name:<25} {speedup:>9.1f}×")
        
        avg_speedup = sum(s for _, s in speedups) / len(speedups)
        logger.info("-" * 40)
        logger.info(f"{'Average':<25} {avg_speedup:>9.1f}×")
        
        logger.info(f"\n✅ Phase 1.4 Migration: {len(speedups)} benchmarks completed")
        logger.info(f"\n🎯 Expected pipeline speedup: ~{avg_speedup:.1f}× on KNN-heavy operations")
        logger.info(f"💡 Estimated tile processing time: 33 min → {33/avg_speedup:.0f} min\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
