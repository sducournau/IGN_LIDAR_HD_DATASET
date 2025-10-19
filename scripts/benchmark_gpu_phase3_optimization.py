#!/usr/bin/env python3
"""
Benchmark script for Phase 3.1 GPU optimization.

Tests the smart memory-based batching decision for neighbor queries.
Compares performance across different dataset sizes on RTX 4080 Super.

Author: IGN LiDAR HD Development Team
Date: October 18, 2025
"""

import numpy as np
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Try GPU imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("⚠️  CuPy not available - cannot run GPU benchmarks")

try:
    from ign_lidar.features.gpu_processor import GPUProcessor as GPUChunkedFeatureComputer
    CHUNKED_AVAILABLE = True
except ImportError:
    CHUNKED_AVAILABLE = False
    logger.warning("⚠️  GPU chunked feature computer not available")


def generate_test_data(n_points: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic point cloud for testing."""
    np.random.seed(seed)
    
    # Generate points in a realistic building-like structure
    # Mix of planes, edges, and noise
    points = []
    
    # Ground plane
    n_ground = n_points // 3
    x_ground = np.random.uniform(-50, 50, n_ground)
    y_ground = np.random.uniform(-50, 50, n_ground)
    z_ground = np.random.normal(0, 0.1, n_ground)  # Small noise
    points.append(np.column_stack([x_ground, y_ground, z_ground]))
    
    # Building walls
    n_walls = n_points // 3
    x_walls = np.random.choice([-50, 50], n_walls)  # Two walls
    y_walls = np.random.uniform(-50, 50, n_walls)
    z_walls = np.random.uniform(0, 30, n_walls)
    points.append(np.column_stack([x_walls, y_walls, z_walls]))
    
    # Roof and random
    n_rest = n_points - n_ground - n_walls
    x_rest = np.random.uniform(-50, 50, n_rest)
    y_rest = np.random.uniform(-50, 50, n_rest)
    z_rest = np.random.uniform(25, 35, n_rest)
    points.append(np.column_stack([x_rest, y_rest, z_rest]))
    
    all_points = np.vstack(points).astype(np.float32)
    
    # Shuffle
    np.random.shuffle(all_points)
    
    return all_points


def benchmark_neighbor_query_batching(
    points: np.ndarray,
    k_neighbors: int = 20,
    neighbor_query_batch_size: int = None
):
    """
    Benchmark neighbor query with Phase 3.1 optimization.
    
    Tests smart memory-based batching decision.
    """
    n_points = len(points)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Dataset: {n_points:,} points, k={k_neighbors}")
    if neighbor_query_batch_size:
        logger.info(f"Config: neighbor_query_batch_size={neighbor_query_batch_size:,}")
    logger.info(f"{'='*70}")
    
    if not GPU_AVAILABLE or not CHUNKED_AVAILABLE:
        logger.warning("⚠️  GPU not available, skipping benchmark")
        return None
    
    # Create GPU chunked computer with config
    computer = GPUChunkedFeatureComputer(
        chunk_size=n_points,  # Single chunk
        neighbor_query_batch_size=neighbor_query_batch_size,
        feature_batch_size=2_000_000,
        show_progress=True
    )
    
    # Create dummy classification
    classification = np.zeros(n_points, dtype=np.uint8)
    
    # Warm-up (GPU kernel compilation)
    if n_points > 100_000:
        logger.info("\n🔥 Warming up GPU (kernel compilation)...")
        warmup_points = points[:100_000]
        warmup_class = classification[:100_000]
        try:
            _ = computer.compute_all_features_chunked(
                warmup_points, warmup_class, k=k_neighbors, mode='lod2'
            )
            logger.info("✓ Warm-up complete")
        except Exception as e:
            logger.warning(f"⚠️  Warm-up failed: {e}")
    
    # Actual benchmark
    logger.info("\n⏱️  Running benchmark...")
    
    try:
        start_time = time.time()
        
        normals, curvature, height, geo_features = computer.compute_all_features_chunked(
            points=points,
            classification=classification,
            k=k_neighbors,
            mode='lod2'
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"\n✅ Benchmark complete!")
        logger.info(f"   Total time: {elapsed:.2f}s")
        logger.info(f"   Throughput: {n_points/elapsed:,.0f} points/sec")
        
        # Get GPU memory stats
        if cp is not None:
            free, total = cp.cuda.runtime.memGetInfo()
            logger.info(f"   GPU memory: {(total-free)/(1024**3):.2f}GB used / {total/(1024**3):.2f}GB total")
        
        return {
            'n_points': n_points,
            'k_neighbors': k_neighbors,
            'time': elapsed,
            'throughput': n_points / elapsed,
            'batch_size_config': neighbor_query_batch_size
        }
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run Phase 3.1 optimization benchmarks."""
    
    logger.info("="*70)
    logger.info("Phase 3.1 GPU Optimization Benchmark")
    logger.info("Testing smart memory-based batching for neighbor queries")
    logger.info("="*70)
    
    if not GPU_AVAILABLE or not CHUNKED_AVAILABLE:
        logger.error("❌ GPU not available - cannot run benchmarks")
        return
    
    # Get GPU info
    if cp is not None:
        logger.info("\n📊 GPU Information:")
        _, total_vram = cp.cuda.runtime.memGetInfo()
        logger.info(f"   Total VRAM: {total_vram/(1024**3):.2f}GB")
        logger.info(f"   Device: {cp.cuda.Device().name.decode() if hasattr(cp.cuda.Device().name, 'decode') else cp.cuda.Device().name}")
    
    # Test cases: different dataset sizes
    test_cases = [
        # (n_points, description)
        (1_000_000, "1M points - Small dataset (baseline)"),
        (5_000_000, "5M points - Medium dataset"),
        (10_000_000, "10M points - Large dataset (Week 1 optimization target)"),
        (18_651_688, "18.6M points - Phase 3.1 target (4 batches → 1 batch expected)"),
    ]
    
    results = []
    
    for n_points, description in test_cases:
        logger.info(f"\n\n{'#'*70}")
        logger.info(f"TEST: {description}")
        logger.info(f"{'#'*70}")
        
        # Generate test data
        logger.info(f"Generating {n_points:,} points...")
        points = generate_test_data(n_points)
        
        # Test with default config (5M batch size)
        logger.info("\n📊 Configuration 1: Default (5M batch size)")
        result1 = benchmark_neighbor_query_batching(
            points, k_neighbors=20, neighbor_query_batch_size=5_000_000
        )
        if result1:
            results.append(('default', result1))
        
        # Test with large batch size (30M - should trigger single-pass for Phase 3.1)
        if n_points >= 10_000_000:
            logger.info("\n📊 Configuration 2: Large batch (30M batch size)")
            result2 = benchmark_neighbor_query_batching(
                points, k_neighbors=20, neighbor_query_batch_size=30_000_000
            )
            if result2:
                results.append(('large_batch', result2))
        
        # Cleanup
        del points
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Print summary
    logger.info("\n\n" + "="*70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*70)
    
    if results:
        logger.info("\n{:<15} {:<12} {:<12} {:<15} {:<15}".format(
            "Config", "Points", "K", "Time (s)", "Throughput (pts/s)"
        ))
        logger.info("-"*70)
        
        for config, result in results:
            logger.info("{:<15} {:<12,} {:<12} {:<15.2f} {:<15,.0f}".format(
                config,
                result['n_points'],
                result['k_neighbors'],
                result['time'],
                result['throughput']
            ))
        
        # Calculate improvements
        logger.info("\n📊 Phase 3.1 Optimization Impact:")
        
        # Find 18.6M results
        default_18m = next((r for c, r in results if c == 'default' and r['n_points'] > 18_000_000), None)
        large_18m = next((r for c, r in results if c == 'large_batch' and r['n_points'] > 18_000_000), None)
        
        if default_18m and large_18m:
            speedup = default_18m['time'] / large_18m['time']
            logger.info(f"\n   18.6M points optimization:")
            logger.info(f"   - Default (5M batches): {default_18m['time']:.2f}s")
            logger.info(f"   - Large (30M batch): {large_18m['time']:.2f}s")
            logger.info(f"   - Speedup: {speedup:.2f}× ({(speedup-1)*100:.1f}% faster)")
            
            if speedup > 1.15:
                logger.info(f"   ✅ Phase 3.1 optimization SUCCESSFUL (>15% improvement)")
            elif speedup > 1.05:
                logger.info(f"   ✅ Phase 3.1 optimization effective (5-15% improvement)")
            else:
                logger.info(f"   ⚠️  Optimization impact below expected (check GPU utilization)")
    else:
        logger.warning("⚠️  No results to summarize")
    
    logger.info("\n" + "="*70)
    logger.info("Benchmark complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
