#!/usr/bin/env python3
"""
Profile ground truth classification performance.

Usage:
    python profile_ground_truth.py /path/to/enriched.laz
"""

import sys
import time
import logging
from pathlib import Path
import laspy
import numpy as np
import cProfile
import pstats
from io import StringIO

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def profile_ground_truth(laz_path: Path):
    """Profile the ground truth classification to find bottlenecks."""
    
    logger.info(f"\n{'='*80}")
    logger.info("PROFILING GROUND TRUTH CLASSIFICATION")
    logger.info(f"{'='*80}\n")
    
    # 1. Load LAZ
    logger.info(f"Loading: {laz_path.name}")
    start = time.time()
    las = laspy.read(str(laz_path))
    load_time = time.time() - start
    
    points = np.vstack([las.x, las.y, las.z]).T
    labels = np.array(las.classification, dtype=np.uint8)
    
    bbox = (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max())
    )
    
    logger.info(f"  Points: {len(points):,}")
    logger.info(f"  Load time: {load_time:.2f}s")
    logger.info(f"  Bbox: {bbox}")
    
    # 2. Fetch ground truth
    logger.info(f"\n{'- '*40}")
    logger.info("Fetching ground truth...")
    logger.info(f"{'- '*40}\n")
    
    from ign_lidar.io.data_fetcher import DataFetcher, DataFetchConfig
    
    config = DataFetchConfig(
        include_buildings=True,
        include_roads=True,
        include_railways=True,
        include_water=True,
        include_vegetation=True,
        include_cemeteries=True,
        include_power_lines=True,
        include_sports=True,
    )
    
    fetcher = DataFetcher(
        cache_dir="/mnt/d/ign/cache",
        config=config
    )
    
    start = time.time()
    gt_data = fetcher.fetch_all(bbox=bbox, use_cache=True)
    fetch_time = time.time() - start
    
    logger.info(f"  Fetch time: {fetch_time:.2f}s")
    
    if not gt_data or 'ground_truth' not in gt_data:
        logger.error("Failed to fetch ground truth!")
        return
    
    ground_truth_features = gt_data['ground_truth']
    
    # Log feature counts
    logger.info(f"\n📊 Ground truth features:")
    total_features = 0
    for feat_type, gdf in ground_truth_features.items():
        if gdf is not None and len(gdf) > 0:
            count = len(gdf)
            total_features += count
            logger.info(f"  {feat_type}: {count} features")
    logger.info(f"  TOTAL: {total_features} features")
    
    # 3. Profile classification
    logger.info(f"\n{'- '*40}")
    logger.info("Profiling classification...")
    logger.info(f"{'- '*40}\n")
    
    from ign_lidar.core.modules.advanced_classification import AdvancedClassifier
    
    # Get features
    ndvi = getattr(las, 'ndvi', None)
    height = getattr(las, 'height_above_ground', None)
    planarity = getattr(las, 'planarity', None)
    intensity = getattr(las, 'intensity', None) / 65535.0 if hasattr(las, 'intensity') else None
    
    classifier = AdvancedClassifier(
        use_ground_truth=True,
        use_ndvi=(ndvi is not None),
        use_geometric=True,
        building_detection_mode='asprs',
        transport_detection_mode='asprs_extended'
    )
    
    # Profile the classification
    profiler = cProfile.Profile()
    
    logger.info("Starting profiled classification...")
    start = time.time()
    
    profiler.enable()
    labels_new = classifier._classify_by_ground_truth(
        labels=labels.copy(),
        points=points,
        ground_truth_features=ground_truth_features,
        ndvi=ndvi,
        height=height,
        planarity=planarity,
        intensity=intensity
    )
    profiler.disable()
    
    classify_time = time.time() - start
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Classification time: {classify_time:.2f}s ({classify_time/60:.2f} minutes)")
    logger.info(f"{'='*80}\n")
    
    # Print profiling stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    logger.info("Top 30 functions by cumulative time:")
    logger.info(s.getvalue())
    
    # Calculate complexity
    total_checks = len(points) * total_features
    logger.info(f"\n{'='*80}")
    logger.info("COMPLEXITY ANALYSIS")
    logger.info(f"{'='*80}")
    logger.info(f"Points: {len(points):,}")
    logger.info(f"Features: {total_features:,}")
    logger.info(f"Theoretical max checks: {total_checks:,} ({total_checks/1e9:.2f} billion)")
    logger.info(f"Time per check: {classify_time/total_checks*1e9:.2f} nanoseconds")
    logger.info(f"{'='*80}\n")
    
    # Performance summary
    logger.info(f"\n{'='*80}")
    logger.info("PERFORMANCE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Load LAZ:          {load_time:8.2f}s")
    logger.info(f"Fetch ground truth: {fetch_time:8.2f}s")
    logger.info(f"Classify points:    {classify_time:8.2f}s  <-- BOTTLENECK")
    logger.info(f"{'- '*40}")
    logger.info(f"TOTAL:             {load_time + fetch_time + classify_time:8.2f}s")
    logger.info(f"{'='*80}\n")
    
    # Recommendations
    if classify_time > 60:
        logger.warning("⚠️  Classification is VERY SLOW (>1 minute)")
        logger.warning("   Recommended optimizations:")
        logger.warning("   1. Implement STRtree spatial indexing")
        logger.warning("   2. Use vectorized GeoPandas spatial joins")
        logger.warning("   3. Pre-filter points by geometric features")
        logger.warning(f"   Expected speedup: 10-100× (target: {classify_time/50:.1f}-{classify_time/10:.1f}s)")
    elif classify_time > 10:
        logger.warning("⚠️  Classification is slow (>10 seconds)")
        logger.warning("   Consider implementing STRtree spatial indexing")
        logger.warning(f"   Expected speedup: 5-20× (target: {classify_time/10:.1f}-{classify_time/5:.1f}s)")
    else:
        logger.info("✅ Classification performance is acceptable")
    
    return {
        'load_time': load_time,
        'fetch_time': fetch_time,
        'classify_time': classify_time,
        'total_time': load_time + fetch_time + classify_time,
        'num_points': len(points),
        'num_features': total_features
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python profile_ground_truth.py <enriched.laz>")
        sys.exit(1)
    
    laz_path = Path(sys.argv[1])
    
    if not laz_path.exists():
        print(f"Error: File not found: {laz_path}")
        sys.exit(1)
    
    results = profile_ground_truth(laz_path)
    
    if results and results['classify_time'] > 60:
        sys.exit(2)  # Slow performance
    else:
        sys.exit(0)
