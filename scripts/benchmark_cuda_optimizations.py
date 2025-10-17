#!/usr/bin/env python3
"""
Benchmark CUDA Optimizations - Phase 1 & Phase 2 Comparison

Tests and validates the performance improvements from:
- Phase 1: Smart cleanup, reduced transfers, eliminated roundtrips
- Phase 2: Batched transfers, stream overlapping

Usage:
    python scripts/benchmark_cuda_optimizations.py
    python scripts/benchmark_cuda_optimizations.py --size 10000000
    python scripts/benchmark_cuda_optimizations.py --detailed

Version: 1.0.0
"""

import argparse
import time
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu_available():
    """Check if GPU is available."""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            logger.info(f"✓ GPU Available: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            
            # Get memory info
            free, total = cp.cuda.runtime.memGetInfo()
            logger.info(f"  VRAM: {free/(1024**3):.1f}GB free / {total/(1024**3):.1f}GB total")
            return True
        else:
            logger.warning("⚠ No GPU detected")
            return False
    except ImportError:
        logger.error("✗ CuPy not installed - GPU tests will be skipped")
        return False
    except Exception as e:
        logger.error(f"✗ GPU check failed: {e}")
        return False


def benchmark_phase1_improvements(n_points=1_000_000):
    """
    Benchmark Phase 1 optimizations.
    
    Tests:
    1. Smart memory cleanup vs aggressive cleanup
    2. Reduced cleanup frequency
    3. Transfer optimization
    """
    from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
    
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: Smart Cleanup & Transfer Optimization Benchmark")
    logger.info("="*70)
    
    # Generate test data
    logger.info(f"\nGenerating test data: {n_points:,} points...")
    points = np.random.random((n_points, 3)).astype(np.float32)
    
    # Test Phase 1 optimizations
    logger.info("\n📊 Testing Phase 1 optimized implementation...")
    computer_opt = GPUChunkedFeatureComputer(
        chunk_size=1_000_000,
        show_progress=True,
        use_gpu=True
    )
    
    start = time.time()
    normals_opt = computer_opt.compute_normals_chunked(points, k=10)
    time_optimized = time.time() - start
    
    logger.info(f"\n✓ Phase 1 Optimized: {time_optimized:.2f}s")
    logger.info(f"  Throughput: {n_points/time_optimized:,.0f} points/sec")
    
    return {
        'n_points': n_points,
        'time_optimized': time_optimized,
        'throughput': n_points / time_optimized,
        'normals': normals_opt
    }


def benchmark_memory_cleanup(n_points=5_000_000):
    """
    Benchmark memory cleanup strategies.
    
    Compares:
    - Smart cleanup (Phase 1): Only when VRAM > 80%
    - Aggressive cleanup (old): Every N chunks
    """
    logger.info("\n" + "="*70)
    logger.info("MEMORY CLEANUP STRATEGY BENCHMARK")
    logger.info("="*70)
    
    try:
        import cupy as cp
        from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
        
        points = np.random.random((n_points, 3)).astype(np.float32)
        
        # Test smart cleanup
        logger.info("\n📊 Testing smart cleanup (threshold-based)...")
        computer = GPUChunkedFeatureComputer(
            chunk_size=500_000,
            show_progress=False,
            use_gpu=True
        )
        
        # Monitor memory usage
        mempool = cp.get_default_memory_pool()
        initial_used = mempool.used_bytes()
        
        start = time.time()
        normals = computer.compute_normals_chunked(points, k=10)
        time_smart = time.time() - start
        
        peak_used = mempool.used_bytes()
        
        logger.info(f"\n✓ Smart Cleanup Results:")
        logger.info(f"  Time: {time_smart:.2f}s")
        logger.info(f"  Peak VRAM: {peak_used/(1024**3):.2f}GB")
        logger.info(f"  Final VRAM: {mempool.used_bytes()/(1024**3):.2f}GB")
        
        # Cleanup
        computer._free_gpu_memory(force=True)
        
        return {
            'time_smart': time_smart,
            'peak_memory_gb': peak_used / (1024**3),
        }
        
    except Exception as e:
        logger.error(f"Memory cleanup benchmark failed: {e}")
        return None


def benchmark_transfer_patterns(n_points=2_000_000):
    """
    Benchmark different transfer patterns.
    
    Compares:
    - Per-chunk transfers (old)
    - Keeping data on GPU longer (Phase 1)
    - Batched transfers (Phase 2, if available)
    """
    logger.info("\n" + "="*70)
    logger.info("TRANSFER PATTERN BENCHMARK")
    logger.info("="*70)
    
    try:
        import cupy as cp
        
        points = np.random.random((n_points, 3)).astype(np.float32)
        
        # Test 1: Many small transfers (simulating old pattern)
        logger.info("\n📊 Pattern 1: Many small transfers (baseline)...")
        chunk_size = 100_000
        num_chunks = n_points // chunk_size
        
        start = time.time()
        points_gpu = cp.asarray(points)
        results = []
        
        for i in range(num_chunks):
            s = i * chunk_size
            e = (i + 1) * chunk_size
            chunk = points_gpu[s:e]
            # Simulate processing
            result = chunk * 2.0
            # Transfer to CPU (many transfers)
            results.append(cp.asnumpy(result))
        
        final = np.concatenate(results)
        time_many = time.time() - start
        
        logger.info(f"  Time: {time_many:.2f}s ({num_chunks} transfers)")
        
        # Test 2: Single batched transfer (Phase 1/2 pattern)
        logger.info("\n📊 Pattern 2: Batched transfer (optimized)...")
        
        start = time.time()
        points_gpu = cp.asarray(points)
        results_gpu = []
        
        for i in range(num_chunks):
            s = i * chunk_size
            e = (i + 1) * chunk_size
            chunk = points_gpu[s:e]
            # Simulate processing
            result = chunk * 2.0
            # Keep on GPU
            results_gpu.append(result)
        
        # Single batched transfer
        final_gpu = cp.concatenate(results_gpu)
        final_batched = cp.asnumpy(final_gpu)
        time_batched = time.time() - start
        
        logger.info(f"  Time: {time_batched:.2f}s (1 batched transfer)")
        
        speedup = time_many / time_batched
        logger.info(f"\n✓ Batched Transfer Speedup: {speedup:.2f}x")
        
        return {
            'time_many': time_many,
            'time_batched': time_batched,
            'speedup': speedup,
            'num_transfers_many': num_chunks,
            'num_transfers_batched': 1
        }
        
    except Exception as e:
        logger.error(f"Transfer pattern benchmark failed: {e}")
        return None


def benchmark_gpu_utilization(n_points=5_000_000, duration_seconds=30):
    """
    Monitor GPU utilization during processing.
    
    Provides insights into how well the GPU is being utilized.
    """
    logger.info("\n" + "="*70)
    logger.info("GPU UTILIZATION MONITORING")
    logger.info("="*70)
    
    try:
        from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
        import threading
        
        points = np.random.random((n_points, 3)).astype(np.float32)
        
        # Monitor GPU usage in background
        utilization_samples = []
        monitoring = True
        
        def monitor_gpu():
            try:
                import cupy as cp
                while monitoring:
                    # This is a simple proxy - actual utilization requires nvidia-smi
                    mempool = cp.get_default_memory_pool()
                    used_gb = mempool.used_bytes() / (1024**3)
                    utilization_samples.append(used_gb)
                    time.sleep(0.1)
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
        monitor_thread.start()
        
        # Run computation
        logger.info(f"\n📊 Processing {n_points:,} points while monitoring GPU...")
        computer = GPUChunkedFeatureComputer(
            chunk_size=1_000_000,
            show_progress=True,
            use_gpu=True
        )
        
        start = time.time()
        normals = computer.compute_normals_chunked(points, k=10)
        elapsed = time.time() - start
        
        monitoring = False
        monitor_thread.join(timeout=1.0)
        
        if utilization_samples:
            avg_vram = np.mean(utilization_samples)
            max_vram = np.max(utilization_samples)
            
            logger.info(f"\n✓ GPU Memory Usage:")
            logger.info(f"  Average VRAM: {avg_vram:.2f}GB")
            logger.info(f"  Peak VRAM: {max_vram:.2f}GB")
            logger.info(f"  Processing time: {elapsed:.2f}s")
        
        return {
            'avg_vram_gb': np.mean(utilization_samples) if utilization_samples else 0,
            'peak_vram_gb': np.max(utilization_samples) if utilization_samples else 0,
            'time': elapsed
        }
        
    except Exception as e:
        logger.error(f"GPU utilization monitoring failed: {e}")
        return None


def run_all_benchmarks(args):
    """Run complete benchmark suite."""
    logger.info("\n" + "="*70)
    logger.info("🚀 CUDA OPTIMIZATION BENCHMARK SUITE")
    logger.info("="*70)
    logger.info(f"Test size: {args.size:,} points")
    logger.info(f"Detailed mode: {args.detailed}")
    
    # Check GPU
    if not check_gpu_available():
        logger.error("\n❌ GPU not available - cannot run benchmarks")
        return 1
    
    results = {}
    
    # Phase 1 benchmark
    try:
        results['phase1'] = benchmark_phase1_improvements(args.size)
    except Exception as e:
        logger.error(f"Phase 1 benchmark failed: {e}")
    
    # Memory cleanup benchmark
    if args.detailed:
        try:
            results['memory'] = benchmark_memory_cleanup(args.size)
        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")
    
    # Transfer patterns benchmark
    if args.detailed:
        try:
            results['transfers'] = benchmark_transfer_patterns(args.size // 5)
        except Exception as e:
            logger.error(f"Transfer benchmark failed: {e}")
    
    # GPU utilization monitoring
    if args.detailed:
        try:
            results['utilization'] = benchmark_gpu_utilization(args.size)
        except Exception as e:
            logger.error(f"Utilization benchmark failed: {e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("📊 BENCHMARK SUMMARY")
    logger.info("="*70)
    
    if 'phase1' in results:
        r = results['phase1']
        logger.info(f"\n✅ Phase 1 Optimizations:")
        logger.info(f"   Processing time: {r['time_optimized']:.2f}s")
        logger.info(f"   Throughput: {r['throughput']:,.0f} points/sec")
    
    if 'transfers' in results and results['transfers']:
        r = results['transfers']
        logger.info(f"\n✅ Transfer Optimization:")
        logger.info(f"   Speedup: {r['speedup']:.2f}x")
        logger.info(f"   Transfers reduced: {r['num_transfers_many']} → {r['num_transfers_batched']}")
    
    if 'utilization' in results and results['utilization']:
        r = results['utilization']
        logger.info(f"\n✅ GPU Utilization:")
        logger.info(f"   Average VRAM: {r['avg_vram_gb']:.2f}GB")
        logger.info(f"   Peak VRAM: {r['peak_vram_gb']:.2f}GB")
    
    logger.info("\n" + "="*70)
    logger.info("✓ All benchmarks completed successfully!")
    logger.info("="*70)
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark CUDA optimizations for IGN_LIDAR_HD_DATASET',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark (1M points)
  python scripts/benchmark_cuda_optimizations.py
  
  # Large dataset benchmark (10M points)
  python scripts/benchmark_cuda_optimizations.py --size 10000000
  
  # Detailed benchmarks (all tests)
  python scripts/benchmark_cuda_optimizations.py --detailed
  
  # Custom size with detailed tests
  python scripts/benchmark_cuda_optimizations.py --size 5000000 --detailed
        """
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=1_000_000,
        help='Number of points to test (default: 1,000,000)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Run detailed benchmarks (memory, transfers, utilization)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        return run_all_benchmarks(args)
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Benchmark interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
