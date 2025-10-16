#!/usr/bin/env python3
"""
Benchmark ground truth optimization methods.

This script compares:
1. Original brute-force method
2. Quick fix with pre-filtering
3. STRtree spatial indexing

Usage:
    python benchmark_ground_truth.py /path/to/enriched.laz
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def benchmark_methods(laz_path: Path, num_points_sample: int = None):
    """
    Benchmark different optimization methods.
    
    Args:
        laz_path: Path to enriched LAZ file
        num_points_sample: Sample N points for faster testing (None = use all)
    """
    import laspy
    
    logger.info(f"\n{'='*80}")
    logger.info("GROUND TRUTH OPTIMIZATION BENCHMARK")
    logger.info(f"{'='*80}\n")
    
    # Load data
    logger.info(f"Loading: {laz_path.name}")
    las = laspy.read(str(laz_path))
    
    points = np.vstack([las.x, las.y, las.z]).T
    labels = np.array(las.classification, dtype=np.uint8)
    
    # Sample points if requested
    if num_points_sample and num_points_sample < len(points):
        logger.info(f"Sampling {num_points_sample:,} points from {len(points):,}")
        indices = np.random.choice(len(points), num_points_sample, replace=False)
        points = points[indices]
        labels = labels[indices]
    
    bbox = (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max())
    )
    
    logger.info(f"Points: {len(points):,}")
    logger.info(f"Bbox: {bbox}")
    
    # Get features
    ndvi = getattr(las, 'ndvi', None)
    if ndvi is not None and num_points_sample:
        ndvi = ndvi[indices]
    height = getattr(las, 'height_above_ground', None)
    if height is not None and num_points_sample:
        height = height[indices]
    planarity = getattr(las, 'planarity', None)
    if planarity is not None and num_points_sample:
        planarity = planarity[indices]
    intensity = getattr(las, 'intensity', None)
    if intensity is not None:
        intensity = intensity / 65535.0
        if num_points_sample:
            intensity = intensity[indices]
    
    # Fetch ground truth
    logger.info(f"\n{'- '*40}")
    logger.info("Fetching ground truth...")
    logger.info(f"{'- '*40}\n")
    
    from ign_lidar.io.data_fetcher import DataFetcher, DataFetchConfig
    
    config = DataFetchConfig(
        include_buildings=True,
        include_roads=True,
        include_railways=True,
        include_water=True,
        include_vegetation=False,  # Skip for speed
        include_cemeteries=False,
        include_power_lines=False,
        include_sports=False,
    )
    
    fetcher = DataFetcher(cache_dir="/mnt/d/ign/cache", config=config)
    gt_data = fetcher.fetch_all(bbox=bbox, use_cache=True)
    
    if not gt_data or 'ground_truth' not in gt_data:
        logger.error("Failed to fetch ground truth!")
        return
    
    ground_truth_features = gt_data['ground_truth']
    
    # Count features
    total_features = sum(len(gdf) for gdf in ground_truth_features.values() if gdf is not None)
    logger.info(f"Ground truth features: {total_features}")
    for feat_type, gdf in ground_truth_features.items():
        if gdf is not None and len(gdf) > 0:
            logger.info(f"  {feat_type}: {len(gdf)}")
    
    # Initialize classifier
    from ign_lidar.core.modules.advanced_classification import AdvancedClassifier
    
    classifier = AdvancedClassifier(
        use_ground_truth=True,
        use_ndvi=(ndvi is not None),
        use_geometric=True,
        building_detection_mode='asprs',
        transport_detection_mode='asprs_extended'
    )
    
    results = {}
    
    # Method 1: Original (skip if too many points)
    if len(points) <= 1_000_000:  # Only test on <1M points
        logger.info(f"\n{'='*80}")
        logger.info("METHOD 1: Original (brute-force)")
        logger.info(f"{'='*80}\n")
        
        start = time.time()
        labels_orig = classifier._classify_by_ground_truth(
            labels.copy(), points, ground_truth_features,
            ndvi, height, planarity, intensity
        )
        orig_time = time.time() - start
        
        results['original'] = {
            'time': orig_time,
            'labels': labels_orig,
            'classified': np.sum(labels_orig != labels)
        }
        
        logger.info(f"Time: {orig_time:.2f}s")
        logger.info(f"Classified {results['original']['classified']:,} points")
    else:
        logger.info(f"\n⏭️  Skipping original method (too slow for {len(points):,} points)")
        results['original'] = None
    
    # Method 2: Quick fix with pre-filtering
    logger.info(f"\n{'='*80}")
    logger.info("METHOD 2: Quick fix (pre-filtering)")
    logger.info(f"{'='*80}\n")
    
    try:
        from ground_truth_quick_fix import patch_classifier
        
        # Restore original first
        if hasattr(AdvancedClassifier, '_classify_by_ground_truth_original'):
            AdvancedClassifier._classify_by_ground_truth = \
                AdvancedClassifier._classify_by_ground_truth_original
        
        patch_classifier()
        
        start = time.time()
        labels_quick = classifier._classify_by_ground_truth(
            labels.copy(), points, ground_truth_features,
            ndvi, height, planarity, intensity
        )
        quick_time = time.time() - start
        
        results['quick_fix'] = {
            'time': quick_time,
            'labels': labels_quick,
            'classified': np.sum(labels_quick != labels)
        }
        
        logger.info(f"Time: {quick_time:.2f}s")
        logger.info(f"Classified {results['quick_fix']['classified']:,} points")
        
    except ImportError as e:
        logger.error(f"Failed to load quick fix: {e}")
        results['quick_fix'] = None
    
    # Method 3: STRtree optimization
    logger.info(f"\n{'='*80}")
    logger.info("METHOD 3: STRtree (spatial indexing)")
    logger.info(f"{'='*80}\n")
    
    try:
        from optimize_ground_truth_strtree import patch_advanced_classifier
        
        # Restore original first
        if hasattr(AdvancedClassifier, '_classify_by_ground_truth_original'):
            AdvancedClassifier._classify_by_ground_truth = \
                AdvancedClassifier._classify_by_ground_truth_original
        
        patch_advanced_classifier()
        
        start = time.time()
        labels_strtree = classifier._classify_by_ground_truth(
            labels.copy(), points, ground_truth_features,
            ndvi, height, planarity, intensity
        )
        strtree_time = time.time() - start
        
        results['strtree'] = {
            'time': strtree_time,
            'labels': labels_strtree,
            'classified': np.sum(labels_strtree != labels)
        }
        
        logger.info(f"Time: {strtree_time:.2f}s")
        logger.info(f"Classified {results['strtree']['classified']:,} points")
        
    except ImportError as e:
        logger.error(f"Failed to load STRtree optimization: {e}")
        results['strtree'] = None
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK RESULTS")
    logger.info(f"{'='*80}\n")
    
    logger.info(f"Points: {len(points):,}")
    logger.info(f"Features: {total_features}\n")
    
    logger.info(f"{'Method':<20} {'Time (s)':<12} {'Speedup':<10} {'Classified':<12}")
    logger.info(f"{'-'*54}")
    
    baseline = None
    for method_name in ['original', 'quick_fix', 'strtree']:
        if results.get(method_name):
            result = results[method_name]
            time_str = f"{result['time']:.2f}s"
            
            if baseline is None:
                baseline = result['time']
                speedup_str = "1.0×"
            else:
                speedup = baseline / result['time']
                speedup_str = f"{speedup:.1f}×"
            
            classified_str = f"{result['classified']:,}"
            
            logger.info(f"{method_name:<20} {time_str:<12} {speedup_str:<10} {classified_str:<12}")
    
    logger.info(f"\n{'='*80}")
    
    # Recommendations
    if results.get('strtree'):
        strtree_time = results['strtree']['time']
        if strtree_time < 60:
            logger.info("✅ STRtree optimization is FAST (<1 minute)")
        elif strtree_time < 300:
            logger.info("✅ STRtree optimization is acceptable (<5 minutes)")
        else:
            logger.info("⚠️  Still slow - consider vectorized GeoPandas spatial joins")
    
    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python benchmark_ground_truth.py <enriched.laz> [num_sample_points]")
        print()
        print("Examples:")
        print("  # Test with 100k points (fast)")
        print("  python benchmark_ground_truth.py enriched.laz 100000")
        print()
        print("  # Test with all points")
        print("  python benchmark_ground_truth.py enriched.laz")
        sys.exit(1)
    
    laz_path = Path(sys.argv[1])
    num_sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not laz_path.exists():
        print(f"Error: File not found: {laz_path}")
        sys.exit(1)
    
    results = benchmark_methods(laz_path, num_sample)
    sys.exit(0)
