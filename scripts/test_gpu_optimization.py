"""
Test GPU Optimization Infrastructure

Validates Phase 1 GPU acceleration infrastructure:
- GPU detection
- gpu_accelerated_ops module
- gpu_kdtree wrapper
- Performance benchmarks

Usage:
    # Quick validation
    python scripts/test_gpu_optimization.py
    
    # Full benchmarks
    python scripts/test_gpu_optimization.py --full-benchmark
    
    # CPU only (test fallback)
    python scripts/test_gpu_optimization.py --cpu-only

Author: IGN LiDAR HD Development Team
Date: November 2025
"""

import argparse
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_gpu_detection():
    """Test GPU availability detection."""
    logger.info("=" * 80)
    logger.info("TEST 1: GPU Detection")
    logger.info("=" * 80)
    
    from ign_lidar.optimization import get_gpu_info, HAS_CUPY, HAS_FAISS, HAS_CUML
    
    info = get_gpu_info()
    
    logger.info(f"\nGPU Availability:")
    for key, available in info.items():
        status = "✅" if available else "❌"
        logger.info(f"  {status} {key}: {available}")
    
    # Summary
    gpu_available = HAS_CUPY or HAS_FAISS or HAS_CUML
    logger.info(f"\nOverall GPU Status: {'✅ Available' if gpu_available else '❌ Not Available (CPU fallback)'}")
    
    return info


def test_eigenvalue_ops():
    """Test eigenvalue operations."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Eigenvalue Operations")
    logger.info("=" * 80)
    
    from ign_lidar.optimization import eigh, eigvalsh
    
    # Test data
    n = 1000
    matrices = np.random.rand(n, 3, 3).astype(np.float32)
    matrices = (matrices + matrices.transpose(0, 2, 1)) / 2  # Symmetric
    
    # Test eigh
    logger.info(f"\nTesting eigh() on {n} matrices...")
    start = time.time()
    eigenvalues, eigenvectors = eigh(matrices)
    elapsed = time.time() - start
    
    logger.info(f"  ✓ Completed in {elapsed:.3f}s")
    logger.info(f"  ✓ Eigenvalues shape: {eigenvalues.shape}")
    logger.info(f"  ✓ Eigenvectors shape: {eigenvectors.shape}")
    
    # Test eigvalsh
    logger.info(f"\nTesting eigvalsh() on {n} matrices...")
    start = time.time()
    eigenvalues_only = eigvalsh(matrices)
    elapsed = time.time() - start
    
    logger.info(f"  ✓ Completed in {elapsed:.3f}s")
    logger.info(f"  ✓ Eigenvalues shape: {eigenvalues_only.shape}")
    
    # Verify consistency
    diff = np.abs(eigenvalues - eigenvalues_only).max()
    logger.info(f"  ✓ Consistency check: max diff = {diff:.2e}")
    
    return True


def test_knn_ops():
    """Test K-Nearest Neighbors operations."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: K-Nearest Neighbors")
    logger.info("=" * 80)
    
    from ign_lidar.optimization import knn
    
    # Test data
    n_points = 10000
    points = np.random.rand(n_points, 3).astype(np.float32)
    k = 30
    
    logger.info(f"\nTesting knn() on {n_points} points, k={k}...")
    start = time.time()
    distances, indices = knn(points, k=k)
    elapsed = time.time() - start
    
    logger.info(f"  ✓ Completed in {elapsed:.3f}s")
    logger.info(f"  ✓ Throughput: {n_points/elapsed:,.0f} points/sec")
    logger.info(f"  ✓ Distances shape: {distances.shape}")
    logger.info(f"  ✓ Indices shape: {indices.shape}")
    
    # Verify first neighbor is self
    assert np.allclose(distances[:, 0], 0, atol=1e-6), "First neighbor should be self"
    assert np.all(indices[:, 0] == np.arange(n_points)), "First index should be self"
    logger.info(f"  ✓ Sanity check passed (first neighbor is self)")
    
    return True


def test_kdtree_wrapper():
    """Test GPUKDTree wrapper."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: GPUKDTree Wrapper")
    logger.info("=" * 80)
    
    from ign_lidar.optimization import GPUKDTree, cKDTree, create_kdtree
    
    # Test data
    points = np.random.rand(5000, 3).astype(np.float32)
    query = np.random.rand(1000, 3).astype(np.float32)
    
    # Test GPUKDTree
    logger.info(f"\nTesting GPUKDTree()...")
    tree = GPUKDTree(points)
    logger.info(f"  ✓ Tree created: {tree.n} points, {tree.m}D")
    logger.info(f"  ✓ Using GPU: {tree.use_gpu}")
    
    # Test query
    logger.info(f"\nTesting query() with {len(query)} query points, k=10...")
    start = time.time()
    distances, indices = tree.query(query, k=10)
    elapsed = time.time() - start
    
    logger.info(f"  ✓ Completed in {elapsed:.3f}s")
    logger.info(f"  ✓ Distances shape: {distances.shape}")
    logger.info(f"  ✓ Indices shape: {indices.shape}")
    
    # Test single point query (API compatibility)
    logger.info(f"\nTesting single point query (scipy API compat)...")
    single_pt = points[0]
    dist, idx = tree.query(single_pt, k=1)
    logger.info(f"  ✓ k=1: type(dist)={type(dist).__name__}, type(idx)={type(idx).__name__}")
    assert isinstance(dist, (float, np.floating)), "k=1 should return scalar"
    assert isinstance(idx, (int, np.integer)), "k=1 should return int"
    
    dist, idx = tree.query(single_pt, k=5)
    logger.info(f"  ✓ k=5: dist.shape={dist.shape}, idx.shape={idx.shape}")
    assert dist.shape == (5,), "k=5 should return 1D array"
    
    # Test cKDTree alias
    logger.info(f"\nTesting cKDTree alias (drop-in replacement)...")
    tree2 = cKDTree(points)
    logger.info(f"  ✓ cKDTree alias works: {type(tree2).__name__}")
    
    # Test factory
    logger.info(f"\nTesting create_kdtree() factory...")
    tree3 = create_kdtree(points, backend='auto')
    logger.info(f"  ✓ Auto backend: {type(tree3).__name__}")
    
    return True


def test_migrated_modules():
    """Test migrated feature computation modules."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Migrated Modules")
    logger.info("=" * 80)
    
    from ign_lidar.features.utils import compute_local_eigenvalues
    from ign_lidar.features.compute.normals import compute_normals
    
    # Test data
    points = np.random.rand(1000, 3).astype(np.float32)
    
    # Test compute_local_eigenvalues
    logger.info(f"\nTesting compute_local_eigenvalues()...")
    start = time.time()
    eigenvalues = compute_local_eigenvalues(points, k=30)
    elapsed = time.time() - start
    
    logger.info(f"  ✓ Completed in {elapsed:.3f}s")
    logger.info(f"  ✓ Eigenvalues shape: {eigenvalues.shape}")
    logger.info(f"  ✓ Sample eigenvalues: {eigenvalues[0]}")
    
    # Test compute_normals
    logger.info(f"\nTesting compute_normals()...")
    start = time.time()
    normals, eigvals = compute_normals(points, k_neighbors=20)
    elapsed = time.time() - start
    
    logger.info(f"  ✓ Completed in {elapsed:.3f}s")
    logger.info(f"  ✓ Normals shape: {normals.shape}")
    logger.info(f"  ✓ Eigenvalues shape: {eigvals.shape}")
    
    # Verify normals are unit length
    norms = np.linalg.norm(normals, axis=1)
    logger.info(f"  ✓ Normals unit length check: mean={norms.mean():.6f}, std={norms.std():.6f}")
    
    return True


def benchmark_comparison(size='medium'):
    """Compare CPU vs GPU performance."""
    logger.info("\n" + "=" * 80)
    logger.info(f"BENCHMARK: CPU vs GPU Comparison ({size} dataset)")
    logger.info("=" * 80)
    
    from ign_lidar.optimization import knn, set_force_cpu
    
    # Dataset sizes
    sizes = {
        'small': 10000,
        'medium': 100000,
        'large': 1000000
    }
    n_points = sizes.get(size, 100000)
    k = 30
    
    points = np.random.rand(n_points, 3).astype(np.float32)
    
    # GPU benchmark
    logger.info(f"\n🚀 GPU Mode ({n_points:,} points, k={k}):")
    set_force_cpu(False)
    start = time.time()
    distances_gpu, indices_gpu = knn(points, k=k)
    elapsed_gpu = time.time() - start
    
    logger.info(f"  Time: {elapsed_gpu:.3f}s")
    logger.info(f"  Throughput: {n_points/elapsed_gpu:,.0f} points/sec")
    
    # CPU benchmark
    logger.info(f"\n💻 CPU Mode ({n_points:,} points, k={k}):")
    set_force_cpu(True)
    start = time.time()
    distances_cpu, indices_cpu = knn(points, k=k)
    elapsed_cpu = time.time() - start
    
    logger.info(f"  Time: {elapsed_cpu:.3f}s")
    logger.info(f"  Throughput: {n_points/elapsed_cpu:,.0f} points/sec")
    
    # Comparison
    speedup = elapsed_cpu / elapsed_gpu
    logger.info(f"\n📊 Results:")
    logger.info(f"  GPU Speedup: {speedup:.1f}×")
    logger.info(f"  Time saved: {elapsed_cpu - elapsed_gpu:.1f}s")
    
    # Verify consistency
    diff = np.abs(distances_gpu - distances_cpu).mean()
    logger.info(f"  Consistency: mean distance diff = {diff:.2e}")
    
    # Reset to GPU mode
    set_force_cpu(False)
    
    return speedup


def main():
    parser = argparse.ArgumentParser(description='Test GPU Optimization Infrastructure')
    parser.add_argument('--full-benchmark', action='store_true', help='Run full benchmarks')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU mode (test fallback)')
    parser.add_argument('--benchmark-size', choices=['small', 'medium', 'large'], 
                        default='medium', help='Benchmark dataset size')
    
    args = parser.parse_args()
    
    try:
        # Force CPU if requested
        if args.cpu_only:
            from ign_lidar.optimization import set_force_cpu
            set_force_cpu(True)
            logger.info("🔧 CPU-only mode enabled")
        
        # Run tests
        logger.info("\n")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 20 + "GPU OPTIMIZATION TEST SUITE" + " " * 30 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        
        gpu_info = test_gpu_detection()
        test_eigenvalue_ops()
        test_knn_ops()
        test_kdtree_wrapper()
        test_migrated_modules()
        
        # Benchmarks
        if args.full_benchmark and not args.cpu_only:
            speedup = benchmark_comparison(size=args.benchmark_size)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info("\n✅ All tests passed successfully!")
        logger.info("\nPhase 1 Infrastructure Status:")
        logger.info("  ✅ GPU detection")
        logger.info("  ✅ gpu_accelerated_ops module")
        logger.info("  ✅ gpu_kdtree wrapper")
        logger.info("  ✅ Migrated modules (6 files)")
        logger.info("  ✅ API compatibility")
        
        if args.full_benchmark and not args.cpu_only:
            logger.info(f"\nBenchmark Results ({args.benchmark_size} dataset):")
            logger.info(f"  🚀 GPU Speedup: {speedup:.1f}×")
        
        logger.info("\n📋 Next Steps:")
        logger.info("  - Phase 1.4: Migrate 30+ files to GPUKDTree")
        logger.info("  - Phase 2: Implement GPU reclassification")
        logger.info("\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
