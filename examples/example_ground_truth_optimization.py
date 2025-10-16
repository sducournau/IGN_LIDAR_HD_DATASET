#!/usr/bin/env python3
"""
Example: Ground Truth Optimization

This script demonstrates the automatic optimization of ground truth computation
for CPU, GPU, and GPU chunked processing.
"""

import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def example_automatic_optimization():
    """
    Example 1: Automatic optimization (recommended).
    
    The system automatically detects available hardware and selects
    the best method.
    """
    logger.info("="*80)
    logger.info("EXAMPLE 1: Automatic Optimization")
    logger.info("="*80)
    
    from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
    from ign_lidar.io.data_fetcher import DataFetcher
    
    # Create sample data (in real usage, load from LAZ file)
    logger.info("\n1. Creating sample data...")
    n_points = 100_000
    points = np.random.rand(n_points, 3) * 1000  # Random points
    points[:, 0] += 650_000  # Offset to Lambert 93 coordinates
    points[:, 1] += 6_850_000
    ndvi = np.random.rand(n_points) * 2 - 1  # NDVI values [-1, 1]
    
    # Define bounding box
    bbox = (
        points[:, 0].min(),
        points[:, 1].min(),
        points[:, 0].max(),
        points[:, 1].max()
    )
    
    logger.info(f"   Points: {len(points):,}")
    logger.info(f"   Bbox: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
    
    # Fetch ground truth
    logger.info("\n2. Fetching ground truth from IGN WFS...")
    fetcher = DataFetcher(cache_dir="/mnt/d/ign/cache")
    gt_data = fetcher.fetch_all(bbox=bbox, use_cache=True)
    
    if not gt_data or 'ground_truth' not in gt_data:
        logger.warning("   No ground truth data available for this bbox")
        logger.info("   (This is normal if the bbox is outside France)")
        return
    
    ground_truth_features = gt_data['ground_truth']
    
    # Count features
    n_features = sum(len(gdf) for gdf in ground_truth_features.values() if gdf is not None)
    logger.info(f"   Ground truth features: {n_features}")
    for feat_type, gdf in ground_truth_features.items():
        if gdf is not None and len(gdf) > 0:
            logger.info(f"     {feat_type}: {len(gdf)}")
    
    # Apply automatic optimized labeling
    logger.info("\n3. Applying automatic optimized ground truth labeling...")
    logger.info("   (Hardware detection → method selection → labeling)")
    
    gt_fetcher = IGNGroundTruthFetcher()
    labels = gt_fetcher.label_points_with_ground_truth(
        points=points,
        ground_truth_features=ground_truth_features,
        ndvi=ndvi,
        use_ndvi_refinement=True
    )
    
    logger.info("\n✅ Automatic optimization complete!")
    logger.info("   The system automatically selected the best available method")
    
    return labels


def example_manual_method_selection():
    """
    Example 2: Manual method selection.
    
    Force a specific optimization method (useful for benchmarking).
    """
    logger.info("\n\n" + "="*80)
    logger.info("EXAMPLE 2: Manual Method Selection")
    logger.info("="*80)
    
    from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
    
    # Create sample data
    logger.info("\n1. Creating sample data...")
    n_points = 50_000
    points = np.random.rand(n_points, 3) * 1000
    points[:, 0] += 650_000
    points[:, 1] += 6_850_000
    
    # Create mock ground truth features
    logger.info("\n2. Creating mock ground truth...")
    import geopandas as gpd
    from shapely.geometry import box
    
    # Create some sample polygons
    buildings = []
    for i in range(10):
        x = points[i*100, 0]
        y = points[i*100, 1]
        buildings.append(box(x, y, x+50, y+50))
    
    buildings_gdf = gpd.GeoDataFrame({'geometry': buildings}, crs='EPSG:2154')
    
    ground_truth_features = {
        'buildings': buildings_gdf
    }
    
    logger.info(f"   Created {len(buildings)} sample buildings")
    
    # Test different methods
    methods = ['strtree', 'vectorized']  # CPU methods
    
    # Add GPU methods if available
    try:
        import cupy as cp
        _ = cp.array([1.0])
        methods.insert(0, 'gpu')
        methods.insert(0, 'gpu_chunked')
        logger.info("\n✅ GPU detected - testing GPU methods too")
    except Exception:
        logger.info("\n⚠️  No GPU detected - testing CPU methods only")
    
    results = {}
    
    for method in methods:
        logger.info(f"\n3.{methods.index(method)+1}. Testing method: {method}")
        
        optimizer = GroundTruthOptimizer(
            force_method=method,
            gpu_chunk_size=5_000_000,
            verbose=True
        )
        
        import time
        start = time.time()
        
        labels = optimizer.label_points(
            points=points,
            ground_truth_features=ground_truth_features
        )
        
        elapsed = time.time() - start
        n_labeled = np.sum(labels > 0)
        
        results[method] = {
            'time': elapsed,
            'labeled': n_labeled
        }
        
        logger.info(f"   Time: {elapsed:.3f}s")
        logger.info(f"   Labeled: {n_labeled:,} / {len(points):,} points")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*80)
    logger.info(f"\n{'Method':<15} {'Time (s)':<12} {'Speedup':<10} {'Labeled':<12}")
    logger.info("-"*49)
    
    baseline = None
    for method, result in results.items():
        time_str = f"{result['time']:.3f}s"
        labeled_str = f"{result['labeled']:,}"
        
        if baseline is None:
            baseline = result['time']
            speedup_str = "1.0×"
        else:
            speedup = baseline / result['time'] if result['time'] > 0 else float('inf')
            speedup_str = f"{speedup:.1f}×"
        
        logger.info(f"{method:<15} {time_str:<12} {speedup_str:<10} {labeled_str:<12}")
    
    return results


def example_large_dataset_gpu_chunked():
    """
    Example 3: Large dataset with GPU chunked processing.
    
    Demonstrates memory-efficient processing of very large datasets.
    """
    logger.info("\n\n" + "="*80)
    logger.info("EXAMPLE 3: Large Dataset with GPU Chunked Processing")
    logger.info("="*80)
    
    # Check GPU availability
    try:
        import cupy as cp
        _ = cp.array([1.0])
        has_gpu = True
        logger.info("\n✅ GPU detected")
    except Exception:
        has_gpu = False
        logger.warning("\n⚠️  No GPU detected - this example requires a GPU")
        logger.info("   Install CuPy: pip install cupy-cuda11x (or cuda12x)")
        return
    
    from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
    
    # Simulate large dataset
    logger.info("\n1. Simulating large dataset (15M points)...")
    n_points = 15_000_000
    
    # Create in chunks to avoid memory issues
    chunk_size = 5_000_000
    points_list = []
    for i in range(0, n_points, chunk_size):
        chunk = np.random.rand(min(chunk_size, n_points - i), 3) * 1000
        chunk[:, 0] += 650_000
        chunk[:, 1] += 6_850_000
        points_list.append(chunk)
    
    points = np.vstack(points_list)
    logger.info(f"   Created {len(points):,} points")
    
    # Create mock ground truth
    logger.info("\n2. Creating mock ground truth...")
    import geopandas as gpd
    from shapely.geometry import box
    
    buildings = []
    for i in range(100):  # More polygons for realistic scenario
        x = points[i*1000, 0] if i*1000 < len(points) else points[0, 0]
        y = points[i*1000, 1] if i*1000 < len(points) else points[0, 1]
        buildings.append(box(x, y, x+50, y+50))
    
    ground_truth_features = {
        'buildings': gpd.GeoDataFrame({'geometry': buildings}, crs='EPSG:2154')
    }
    
    # Use GPU chunked processing
    logger.info("\n3. Processing with GPU chunked method...")
    logger.info("   Chunk size: 5M points per GPU transfer")
    
    optimizer = GroundTruthOptimizer(
        force_method='gpu_chunked',
        gpu_chunk_size=5_000_000,
        verbose=True
    )
    
    import time
    start = time.time()
    
    labels = optimizer.label_points(
        points=points,
        ground_truth_features=ground_truth_features
    )
    
    elapsed = time.time() - start
    
    logger.info(f"\n✅ Processed {len(points):,} points in {elapsed:.2f}s")
    logger.info(f"   Throughput: {len(points)/elapsed:,.0f} points/second")
    logger.info(f"   Labeled: {np.sum(labels > 0):,} points")
    
    # Estimate time saved
    original_time_estimate = len(points) * 500 * 0.000001  # Rough estimate
    time_saved = original_time_estimate - elapsed
    
    logger.info(f"\n💡 Estimated time saved vs original method:")
    logger.info(f"   Original (estimated): {original_time_estimate/60:.1f} minutes")
    logger.info(f"   GPU Chunked: {elapsed:.1f} seconds")
    logger.info(f"   Time saved: {time_saved/60:.1f} minutes ({time_saved/original_time_estimate*100:.0f}% reduction)")
    
    return labels


def main():
    """Run all examples."""
    logger.info("\n" + "🚀 "*40)
    logger.info("GROUND TRUTH OPTIMIZATION EXAMPLES")
    logger.info("🚀 "*40)
    
    # Example 1: Automatic (recommended for production)
    try:
        example_automatic_optimization()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Manual selection (useful for benchmarking)
    try:
        example_manual_method_selection()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 3: Large dataset with GPU (if available)
    try:
        example_large_dataset_gpu_chunked()
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*80)
    logger.info("✅ Examples completed!")
    logger.info("="*80)
    logger.info("\nKey Takeaways:")
    logger.info("1. Automatic optimization works out-of-the-box (Example 1)")
    logger.info("2. Manual method selection available for control (Example 2)")
    logger.info("3. GPU chunked handles massive datasets efficiently (Example 3)")
    logger.info("\nFor production use, stick with automatic optimization!")


if __name__ == '__main__':
    main()
