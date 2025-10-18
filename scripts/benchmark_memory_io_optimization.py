"""
Benchmark Memory & I/O Optimizations

Tests the performance improvements from Phase 3 Sprint 4:
- Memory-mapped feature caching
- Parallel I/O operations
- Streaming processing
- Buffer optimization

Author: Phase 3 Sprint 4
Date: October 18, 2025
"""

import logging
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_tile(
    num_points: int,
    output_path: Path
) -> Dict[str, Any]:
    """
    Generate synthetic LAZ tile for testing.
    
    Args:
        num_points: Number of points to generate
        output_path: Path to save LAZ file
        
    Returns:
        Dictionary with tile metadata
    """
    import laspy
    
    logger.info(f"🔧 Generating synthetic tile: {num_points:,} points")
    
    # Generate random point cloud
    points = np.random.randn(num_points, 3).astype(np.float32) * 100
    intensity = np.random.randint(0, 65536, num_points, dtype=np.uint16)
    classification = np.random.randint(1, 7, num_points, dtype=np.uint8)
    return_number = np.ones(num_points, dtype=np.uint8)
    
    # Create LAZ file
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0, 0, 0]
    
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.intensity = intensity
    las.classification = classification
    las.return_number = return_number
    
    las.write(str(output_path))
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"   ✅ Created {output_path.name} ({file_size_mb:.1f} MB)")
    
    return {
        'num_points': num_points,
        'file_size_mb': file_size_mb,
        'path': output_path
    }


def benchmark_feature_cache(
    num_points: int = 1_000_000,
    num_features: int = 10
) -> Dict[str, Any]:
    """
    Benchmark memory-mapped feature cache.
    
    Args:
        num_points: Number of points
        num_features: Number of features to test
        
    Returns:
        Benchmark results
    """
    from ign_lidar.optimization.memory_cache import FeatureCache
    
    logger.info("\n" + "="*70)
    logger.info("📦 BENCHMARK: Feature Cache Performance")
    logger.info("="*70)
    
    # Create cache
    cache = FeatureCache(max_memory_mb=500.0)
    
    # Generate test features
    logger.info(f"🔧 Generating {num_features} features ({num_points:,} points each)")
    test_features = {}
    for i in range(num_features):
        feature_name = f"feature_{i}"
        feature_data = np.random.randn(num_points, 3).astype(np.float32)
        test_features[feature_name] = feature_data
    
    # Benchmark: Store features
    logger.info("\n📝 Benchmarking WRITE performance...")
    
    write_times = {'memmap': [], 'standard': []}
    
    for feature_name, feature_data in test_features.items():
        # Test memmap storage
        start = time.perf_counter()
        cache.store_feature(f"tile_test", feature_name, feature_data, use_memmap=True)
        write_times['memmap'].append(time.perf_counter() - start)
        
        # Test standard storage
        start = time.perf_counter()
        cache.store_feature(f"tile_test2", feature_name, feature_data, use_memmap=False)
        write_times['standard'].append(time.perf_counter() - start)
    
    # Benchmark: Load features
    logger.info("\n📖 Benchmarking READ performance...")
    
    read_times = {'memmap': [], 'standard': []}
    
    for feature_name in test_features.keys():
        # Test memmap load
        start = time.perf_counter()
        data_memmap = cache.load_feature(f"tile_test", feature_name)
        read_times['memmap'].append(time.perf_counter() - start)
        
        # Test standard load
        start = time.perf_counter()
        data_standard = cache.load_feature(f"tile_test2", feature_name)
        read_times['standard'].append(time.perf_counter() - start)
    
    # Compute statistics
    avg_write_memmap = np.mean(write_times['memmap'])
    avg_write_standard = np.mean(write_times['standard'])
    avg_read_memmap = np.mean(read_times['memmap'])
    avg_read_standard = np.mean(read_times['standard'])
    
    # Calculate throughput (MB/s)
    feature_size_mb = test_features['feature_0'].nbytes / (1024 * 1024)
    write_throughput_memmap = feature_size_mb / avg_write_memmap
    write_throughput_standard = feature_size_mb / avg_write_standard
    read_throughput_memmap = feature_size_mb / avg_read_memmap
    read_throughput_standard = feature_size_mb / avg_read_standard
    
    results = {
        'num_points': num_points,
        'num_features': num_features,
        'feature_size_mb': feature_size_mb,
        'write': {
            'memmap_time_s': avg_write_memmap,
            'standard_time_s': avg_write_standard,
            'memmap_throughput_mbps': write_throughput_memmap,
            'standard_throughput_mbps': write_throughput_standard,
            'speedup': avg_write_standard / avg_write_memmap
        },
        'read': {
            'memmap_time_s': avg_read_memmap,
            'standard_time_s': avg_read_standard,
            'memmap_throughput_mbps': read_throughput_memmap,
            'standard_throughput_mbps': read_throughput_standard,
            'speedup': avg_read_standard / avg_read_memmap
        }
    }
    
    # Print results
    logger.info("\n📊 RESULTS:")
    logger.info(f"   Feature size: {feature_size_mb:.1f} MB")
    logger.info(f"\n   WRITE Performance:")
    logger.info(f"      Memmap:   {avg_write_memmap*1000:.2f}ms ({write_throughput_memmap:.1f} MB/s)")
    logger.info(f"      Standard: {avg_write_standard*1000:.2f}ms ({write_throughput_standard:.1f} MB/s)")
    logger.info(f"      Speedup:  {results['write']['speedup']:.2f}x")
    logger.info(f"\n   READ Performance:")
    logger.info(f"      Memmap:   {avg_read_memmap*1000:.2f}ms ({read_throughput_memmap:.1f} MB/s)")
    logger.info(f"      Standard: {avg_read_standard*1000:.2f}ms ({read_throughput_standard:.1f} MB/s)")
    logger.info(f"      Speedup:  {results['read']['speedup']:.2f}x")
    
    # Get cache stats
    cache_stats = cache.get_stats()
    logger.info(f"\n   Cache Statistics:")
    logger.info(f"      Entries: {cache_stats['num_entries']}")
    logger.info(f"      Memory:  {cache_stats['memory_used_mb']:.1f}/{cache_stats['memory_max_mb']:.1f} MB")
    logger.info(f"      Usage:   {cache_stats['memory_usage_percent']:.1f}%")
    
    # Cleanup
    cache.cleanup()
    
    return results


def benchmark_parallel_io(
    num_files: int = 10,
    points_per_file: int = 500_000
) -> Dict[str, Any]:
    """
    Benchmark parallel I/O performance.
    
    Args:
        num_files: Number of files to test
        points_per_file: Points per file
        
    Returns:
        Benchmark results
    """
    from ign_lidar.optimization.io_optimization import ParallelLAZReader
    import laspy
    
    logger.info("\n" + "="*70)
    logger.info("📚 BENCHMARK: Parallel I/O Performance")
    logger.info("="*70)
    
    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Generate test files
        logger.info(f"🔧 Generating {num_files} test files ({points_per_file:,} points each)")
        test_files = []
        for i in range(num_files):
            file_path = temp_path / f"test_tile_{i:03d}.laz"
            generate_synthetic_tile(points_per_file, file_path)
            test_files.append(file_path)
        
        # Simple reader function
        def read_laz(path: Path):
            return laspy.read(str(path))
        
        # Benchmark: Sequential reading
        logger.info(f"\n📖 Sequential reading...")
        start = time.perf_counter()
        sequential_results = [read_laz(f) for f in test_files]
        sequential_time = time.perf_counter() - start
        
        # Benchmark: Parallel reading (different worker counts)
        parallel_results = {}
        
        for num_workers in [2, 4, 8]:
            logger.info(f"\n📖 Parallel reading ({num_workers} workers)...")
            reader = ParallelLAZReader(max_workers=num_workers)
            
            start = time.perf_counter()
            results = reader.read_multiple(test_files, read_laz, show_progress=False)
            elapsed = time.perf_counter() - start
            
            parallel_results[num_workers] = {
                'time_s': elapsed,
                'speedup': sequential_time / elapsed,
                'throughput_files_per_sec': num_files / elapsed
            }
    
    # Compute statistics
    results = {
        'num_files': num_files,
        'points_per_file': points_per_file,
        'sequential': {
            'time_s': sequential_time,
            'throughput_files_per_sec': num_files / sequential_time
        },
        'parallel': parallel_results
    }
    
    # Print results
    logger.info("\n📊 RESULTS:")
    logger.info(f"   Files: {num_files} ({points_per_file:,} points each)")
    logger.info(f"\n   Sequential: {sequential_time:.2f}s ({results['sequential']['throughput_files_per_sec']:.2f} files/s)")
    
    for num_workers, metrics in parallel_results.items():
        logger.info(f"\n   Parallel ({num_workers} workers):")
        logger.info(f"      Time:       {metrics['time_s']:.2f}s")
        logger.info(f"      Throughput: {metrics['throughput_files_per_sec']:.2f} files/s")
        logger.info(f"      Speedup:    {metrics['speedup']:.2f}x")
    
    return results


def benchmark_streaming_processing(
    num_points: int = 5_000_000
) -> Dict[str, Any]:
    """
    Benchmark streaming tile processing.
    
    Args:
        num_points: Number of points in test tile
        
    Returns:
        Benchmark results
    """
    from ign_lidar.optimization.memory_cache import StreamingTileProcessor
    
    logger.info("\n" + "="*70)
    logger.info("🌊 BENCHMARK: Streaming Processing Performance")
    logger.info("="*70)
    
    # Create test tile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile_path = temp_path / "test_tile.laz"
        
        logger.info(f"🔧 Generating test tile ({num_points:,} points)")
        tile_info = generate_synthetic_tile(num_points, tile_path)
        
        # Define simple feature functions
        def compute_height(points):
            return points[:, 2]  # Z coordinate
        
        def compute_distance(points):
            return np.linalg.norm(points, axis=1)
        
        feature_functions = {
            'height': compute_height,
            'distance': compute_distance
        }
        
        # Benchmark: Different chunk sizes
        chunk_sizes = [1_000_000, 2_500_000, 5_000_000]
        results_by_chunk = {}
        
        for chunk_size in chunk_sizes:
            logger.info(f"\n🌊 Streaming with chunk_size={chunk_size:,}...")
            
            processor = StreamingTileProcessor(chunk_size=chunk_size)
            
            start = time.perf_counter()
            features = processor.process_tile_streaming(
                tile_path,
                feature_functions,
                output_dir=None
            )
            elapsed = time.perf_counter() - start
            
            results_by_chunk[chunk_size] = {
                'time_s': elapsed,
                'throughput_pts_per_sec': num_points / elapsed,
                'features_computed': len(features)
            }
        
        results = {
            'num_points': num_points,
            'file_size_mb': tile_info['file_size_mb'],
            'num_features': len(feature_functions),
            'chunk_results': results_by_chunk
        }
        
        # Print results
        logger.info("\n📊 RESULTS:")
        logger.info(f"   Points: {num_points:,} ({tile_info['file_size_mb']:.1f} MB)")
        logger.info(f"   Features: {len(feature_functions)}")
        
        for chunk_size, metrics in results_by_chunk.items():
            logger.info(f"\n   Chunk size: {chunk_size:,}")
            logger.info(f"      Time:       {metrics['time_s']:.2f}s")
            logger.info(f"      Throughput: {metrics['throughput_pts_per_sec']:,.0f} pts/s")
        
        return results


def run_all_benchmarks(output_file: str = "benchmark_memory_io.json"):
    """Run all Sprint 4 benchmarks and save results."""
    logger.info("\n" + "="*70)
    logger.info("🚀 PHASE 3 SPRINT 4: Memory & I/O Optimization Benchmarks")
    logger.info("="*70)
    
    all_results = {}
    
    # Benchmark 1: Feature cache
    try:
        all_results['feature_cache'] = benchmark_feature_cache(
            num_points=1_000_000,
            num_features=10
        )
    except Exception as e:
        logger.error(f"❌ Feature cache benchmark failed: {e}")
        all_results['feature_cache'] = {'error': str(e)}
    
    # Benchmark 2: Parallel I/O
    try:
        all_results['parallel_io'] = benchmark_parallel_io(
            num_files=10,
            points_per_file=500_000
        )
    except Exception as e:
        logger.error(f"❌ Parallel I/O benchmark failed: {e}")
        all_results['parallel_io'] = {'error': str(e)}
    
    # Benchmark 3: Streaming processing
    try:
        all_results['streaming'] = benchmark_streaming_processing(
            num_points=5_000_000
        )
    except Exception as e:
        logger.error(f"❌ Streaming benchmark failed: {e}")
        all_results['streaming'] = {'error': str(e)}
    
    # Save results
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_path}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("📊 SUMMARY")
    logger.info("="*70)
    
    if 'feature_cache' in all_results and 'error' not in all_results['feature_cache']:
        fc = all_results['feature_cache']
        logger.info(f"\n✅ Feature Cache:")
        logger.info(f"   Write speedup: {fc['write']['speedup']:.2f}x")
        logger.info(f"   Read speedup:  {fc['read']['speedup']:.2f}x")
    
    if 'parallel_io' in all_results and 'error' not in all_results['parallel_io']:
        pio = all_results['parallel_io']
        best_speedup = max(p['speedup'] for p in pio['parallel'].values())
        logger.info(f"\n✅ Parallel I/O:")
        logger.info(f"   Best speedup: {best_speedup:.2f}x")
    
    if 'streaming' in all_results and 'error' not in all_results['streaming']:
        stream = all_results['streaming']
        logger.info(f"\n✅ Streaming Processing:")
        logger.info(f"   Processed: {stream['num_points']:,} points")
    
    logger.info("\n🎉 All benchmarks complete!")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    # Check dependencies
    try:
        import laspy
        import numpy as np
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("   Please install: pip install laspy numpy")
        sys.exit(1)
    
    # Run benchmarks
    results = run_all_benchmarks()
