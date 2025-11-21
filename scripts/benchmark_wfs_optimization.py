#!/usr/bin/env python3
"""
Benchmark WFS Optimization (Phase 5)

Compares performance between:
1. Original IGNGroundTruthFetcher (sequential, no pooling)
2. OptimizedWFSFetcher (pooling, caching, parallel)

Expected improvements:
- Cold fetch: 2-4× faster (HTTP pooling + parallel)
- Warm fetch (disk cache): 60× faster
- Hot fetch (memory cache): 1200× faster
"""

import logging
import time
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher, IGNWFSConfig

try:
    from ign_lidar.io.wfs_optimized import OptimizedWFSFetcher, OptimizedWFSConfig
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False
    print("⚠️  OptimizedWFSFetcher not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Configuration
# ============================================================================

# Test bounding box (small area in Paris for quick testing)
# Coordinates in Lambert 93 (EPSG:2154)
TEST_BBOX: Tuple[float, float, float, float] = (
    648000.0,  # xmin
    6860000.0,  # ymin
    649000.0,  # xmax
    6861000.0   # ymax (1km × 1km)
)

# Test bounding box for larger area (5km × 5km)
LARGE_TEST_BBOX: Tuple[float, float, float, float] = (
    648000.0,
    6860000.0,
    653000.0,
    6865000.0
)


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_original_fetcher(
    bbox: Tuple[float, float, float, float],
    cache_dir: Path,
    use_cache: bool = True
) -> Dict[str, Any]:
    """Benchmark original IGNGroundTruthFetcher."""
    logger.info("=" * 80)
    logger.info("BENCHMARK: Original IGNGroundTruthFetcher")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Initialize fetcher
    fetcher = IGNGroundTruthFetcher(
        cache_dir=cache_dir,
        verbose=False  # Reduce log noise
    )
    
    init_time = time.time() - start_time
    
    # Fetch all features
    fetch_start = time.time()
    results = fetcher.fetch_all_features(
        bbox=bbox,
        use_cache=use_cache,
        include_buildings=True,
        include_roads=True,
        include_railways=True,
        include_water=True,
        include_vegetation=True,
    )
    fetch_time = time.time() - fetch_start
    
    total_time = time.time() - start_time
    
    # Count features
    feature_counts = {}
    for key, gdf in results.items():
        if gdf is not None:
            feature_counts[key] = len(gdf)
        else:
            feature_counts[key] = 0
    
    return {
        'init_time': init_time,
        'fetch_time': fetch_time,
        'total_time': total_time,
        'feature_counts': feature_counts,
        'results': results
    }


def benchmark_optimized_fetcher(
    bbox: Tuple[float, float, float, float],
    cache_dir: Path,
    use_cache: bool = True,
    use_memory_cache: bool = True
) -> Dict[str, Any]:
    """Benchmark OptimizedWFSFetcher."""
    if not HAS_OPTIMIZED:
        logger.warning("OptimizedWFSFetcher not available, skipping")
        return None
    
    logger.info("=" * 80)
    logger.info("BENCHMARK: OptimizedWFSFetcher")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Initialize fetcher
    config = OptimizedWFSConfig(
        max_connections=10,
        max_workers=4,
        enable_disk_cache=use_cache,
        enable_memory_cache=use_memory_cache,
        cache_ttl_days=30,
        max_memory_cache_mb=500
    )
    
    fetcher = OptimizedWFSFetcher(
        cache_dir=cache_dir,
        config=config
    )
    
    init_time = time.time() - start_time
    
    # Fetch all features (using parallel fetching)
    fetch_start = time.time()
    results = fetcher.fetch_ground_truth(
        bbox=bbox,
        layers=['buildings', 'roads', 'railways', 'water', 'vegetation'],
        use_cache=use_cache
    )
    fetch_time = time.time() - fetch_start
    
    total_time = time.time() - start_time
    
    # Count features
    feature_counts = {}
    for key, gdf in results.items():
        if gdf is not None:
            feature_counts[key] = len(gdf)
        else:
            feature_counts[key] = 0
    
    # Get cache statistics
    cache_stats = fetcher.get_cache_stats()
    
    return {
        'init_time': init_time,
        'fetch_time': fetch_time,
        'total_time': total_time,
        'feature_counts': feature_counts,
        'cache_stats': cache_stats,
        'results': results
    }


def compare_results(
    original: Dict[str, Any],
    optimized: Dict[str, Any]
) -> None:
    """Compare benchmark results."""
    if original is None or optimized is None:
        logger.warning("Cannot compare - missing results")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)
    
    # Timing comparison
    logger.info("\n⏱️  Timing Comparison:")
    logger.info(f"  Initialization:")
    logger.info(f"    Original:  {original['init_time']:.3f}s")
    logger.info(f"    Optimized: {optimized['init_time']:.3f}s")
    logger.info(f"    Speedup:   {original['init_time'] / optimized['init_time']:.2f}×")
    
    logger.info(f"\n  Fetch Time:")
    logger.info(f"    Original:  {original['fetch_time']:.3f}s")
    logger.info(f"    Optimized: {optimized['fetch_time']:.3f}s")
    speedup = original['fetch_time'] / optimized['fetch_time']
    logger.info(f"    Speedup:   {speedup:.2f}×")
    
    if speedup >= 2.0:
        logger.info(f"    ✅ Excellent speedup (≥2×)!")
    elif speedup >= 1.5:
        logger.info(f"    ✅ Good speedup (≥1.5×)")
    else:
        logger.info(f"    ⚠️  Modest speedup (<1.5×)")
    
    logger.info(f"\n  Total Time:")
    logger.info(f"    Original:  {original['total_time']:.3f}s")
    logger.info(f"    Optimized: {optimized['total_time']:.3f}s")
    logger.info(f"    Speedup:   {original['total_time'] / optimized['total_time']:.2f}×")
    
    # Feature counts comparison
    logger.info("\n📊 Feature Counts:")
    orig_counts = original['feature_counts']
    opt_counts = optimized['feature_counts']
    
    all_keys = set(orig_counts.keys()) | set(opt_counts.keys())
    for key in sorted(all_keys):
        orig_count = orig_counts.get(key, 0)
        opt_count = opt_counts.get(key, 0)
        match = "✅" if orig_count == opt_count else "❌"
        logger.info(f"  {key:15s}: {orig_count:5d} vs {opt_count:5d} {match}")
    
    # Cache statistics
    if 'cache_stats' in optimized:
        logger.info("\n💾 Cache Statistics (Optimized):")
        stats = optimized['cache_stats']
        logger.info(f"  Disk cache hits:   {stats.get('disk_hits', 0)}")
        logger.info(f"  Memory cache hits: {stats.get('memory_hits', 0)}")
        logger.info(f"  Cache misses:      {stats.get('misses', 0)}")
        if stats.get('total_requests', 0) > 0:
            hit_rate = (stats.get('disk_hits', 0) + stats.get('memory_hits', 0)) / stats['total_requests']
            logger.info(f"  Hit rate:          {hit_rate * 100:.1f}%")


def benchmark_cache_effectiveness(
    bbox: Tuple[float, float, float, float],
    cache_dir: Path
) -> None:
    """Benchmark cache effectiveness (cold vs warm vs hot)."""
    if not HAS_OPTIMIZED:
        logger.warning("OptimizedWFSFetcher not available, skipping cache test")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("CACHE EFFECTIVENESS TEST")
    logger.info("=" * 80)
    
    config = OptimizedWFSConfig(
        max_connections=10,
        max_workers=4,
        enable_disk_cache=True,
        enable_memory_cache=True,
        cache_ttl_days=30,
        max_memory_cache_mb=500
    )
    
    # 1. Cold fetch (no cache)
    logger.info("\n1️⃣  Cold Fetch (no cache):")
    fetcher_cold = OptimizedWFSFetcher(
        cache_dir=cache_dir / "cold",
        config=config
    )
    
    start = time.time()
    result_cold = fetcher_cold.fetch_ground_truth(
        bbox=bbox,
        layers=['buildings', 'roads'],
        use_cache=False
    )
    cold_time = time.time() - start
    logger.info(f"  Time: {cold_time:.3f}s")
    
    # 2. Warm fetch (disk cache)
    logger.info("\n2️⃣  Warm Fetch (disk cache):")
    fetcher_warm = OptimizedWFSFetcher(
        cache_dir=cache_dir / "cold",  # Reuse same cache
        config=config
    )
    
    start = time.time()
    result_warm = fetcher_warm.fetch_ground_truth(
        bbox=bbox,
        layers=['buildings', 'roads'],
        use_cache=True
    )
    warm_time = time.time() - start
    logger.info(f"  Time: {warm_time:.3f}s")
    logger.info(f"  Speedup vs cold: {cold_time / warm_time:.1f}×")
    
    # 3. Hot fetch (memory cache)
    logger.info("\n3️⃣  Hot Fetch (memory cache):")
    start = time.time()
    result_hot = fetcher_warm.fetch_ground_truth(  # Same fetcher instance
        bbox=bbox,
        layers=['buildings', 'roads'],
        use_cache=True
    )
    hot_time = time.time() - start
    logger.info(f"  Time: {hot_time:.3f}s")
    logger.info(f"  Speedup vs cold: {cold_time / hot_time:.1f}×")
    logger.info(f"  Speedup vs warm: {warm_time / hot_time:.1f}×")
    
    # Summary
    logger.info("\n📈 Cache Effectiveness Summary:")
    logger.info(f"  Cold:  {cold_time:.3f}s (baseline)")
    logger.info(f"  Warm:  {warm_time:.3f}s ({cold_time / warm_time:.1f}× faster)")
    logger.info(f"  Hot:   {hot_time:.3f}s ({cold_time / hot_time:.1f}× faster)")


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    """Run all benchmarks."""
    logger.info("=" * 80)
    logger.info("WFS OPTIMIZATION BENCHMARK (Phase 5)")
    logger.info("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        # 1. Small area benchmark (cold fetch)
        logger.info("\n\n🔬 Test 1: Small Area (1km × 1km) - Cold Fetch")
        logger.info("-" * 80)
        
        original_cold = benchmark_original_fetcher(
            TEST_BBOX,
            cache_dir / "original_cold",
            use_cache=False
        )
        
        optimized_cold = benchmark_optimized_fetcher(
            TEST_BBOX,
            cache_dir / "optimized_cold",
            use_cache=False
        )
        
        compare_results(original_cold, optimized_cold)
        
        # 2. Small area benchmark (warm fetch)
        logger.info("\n\n🔬 Test 2: Small Area (1km × 1km) - Warm Fetch (Cached)")
        logger.info("-" * 80)
        
        original_warm = benchmark_original_fetcher(
            TEST_BBOX,
            cache_dir / "original_cold",  # Reuse cache
            use_cache=True
        )
        
        optimized_warm = benchmark_optimized_fetcher(
            TEST_BBOX,
            cache_dir / "optimized_cold",  # Reuse cache
            use_cache=True
        )
        
        compare_results(original_warm, optimized_warm)
        
        # 3. Cache effectiveness test
        logger.info("\n\n🔬 Test 3: Cache Effectiveness")
        logger.info("-" * 80)
        benchmark_cache_effectiveness(TEST_BBOX, cache_dir / "cache_test")
        
        # 4. Larger area benchmark (if time permits)
        logger.info("\n\n🔬 Test 4: Large Area (5km × 5km) - Cold Fetch")
        logger.info("-" * 80)
        logger.info("⚠️  This test may take several minutes...")
        
        try:
            original_large = benchmark_original_fetcher(
                LARGE_TEST_BBOX,
                cache_dir / "original_large",
                use_cache=False
            )
            
            optimized_large = benchmark_optimized_fetcher(
                LARGE_TEST_BBOX,
                cache_dir / "optimized_large",
                use_cache=False
            )
            
            compare_results(original_large, optimized_large)
        except Exception as e:
            logger.error(f"Large area test failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
