#!/usr/bin/env python3
"""
Benchmark Script for Classification Optimizations

Compares performance of:
1. Standard building buffer classification vs. clustered
2. With and without spectral rules
3. Overall reclassification pipeline improvements

Usage:
    python scripts/benchmark_classification_optimizations.py [--data-path PATH] [--num-points N]
"""

import argparse
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_data(num_points=1_000_000, num_buildings=50):
    """
    Create synthetic point cloud with buildings for testing.
    
    Args:
        num_points: Number of points to generate
        num_buildings: Number of building polygons to generate
    
    Returns:
        Tuple of (points, labels, buildings_gdf, rgb, nir, ndvi)
    """
    logger.info(f"Creating synthetic dataset with {num_points:,} points and {num_buildings} buildings...")
    
    # Generate random point cloud in 1000x1000m area
    points = np.random.rand(num_points, 3) * 1000
    points[:, 2] = points[:, 2] * 50  # Heights up to 50m
    
    # Most points start as unclassified (1)
    labels = np.ones(num_points, dtype=np.int32)
    
    # Create some building points (about 20%)
    n_buildings = int(num_points * 0.2)
    building_indices = np.random.choice(num_points, n_buildings, replace=False)
    labels[building_indices] = 6  # ASPRS Building
    
    # Create building geometries
    buildings = []
    for i in range(num_buildings):
        x = np.random.rand() * 900 + 50
        y = np.random.rand() * 900 + 50
        width = np.random.rand() * 20 + 10  # 10-30m
        height = np.random.rand() * 20 + 10
        buildings.append(box(x, y, x + width, y + height))
    
    buildings_gdf = gpd.GeoDataFrame(
        {'geometry': buildings},
        crs='EPSG:2154'
    )
    
    # Generate synthetic RGB and NIR data
    rgb = np.random.rand(num_points, 3)
    nir = np.random.rand(num_points)
    
    # Compute NDVI
    red = rgb[:, 0]
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    logger.info(f"✓ Created synthetic data: {num_points:,} points, {num_buildings} buildings")
    
    return points, labels, buildings_gdf, rgb, nir, ndvi


def benchmark_building_buffer(points, labels, buildings_gdf, num_runs=3):
    """
    Benchmark building buffer classification with and without clustering.
    
    Args:
        points: Point cloud XYZ coordinates
        labels: Classification labels
        buildings_gdf: GeoDataFrame with building polygons
        num_runs: Number of runs for averaging
    
    Returns:
        Dict with timing results
    """
    from ign_lidar.core.modules.geometric_rules import GeometricRulesEngine
    
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK: Building Buffer Classification")
    logger.info("="*70)
    
    results = {
        'standard': [],
        'clustered': []
    }
    
    # Benchmark standard method
    logger.info(f"\n1. Standard (point-by-point) method - {num_runs} runs:")
    engine_standard = GeometricRulesEngine(use_clustering=False)
    
    for i in range(num_runs):
        labels_copy = labels.copy()
        start = time.time()
        n_classified = engine_standard.classify_building_buffer_zone(
            points=points,
            labels=labels_copy,
            building_geometries=buildings_gdf
        )
        elapsed = time.time() - start
        results['standard'].append(elapsed)
        logger.info(f"   Run {i+1}: {elapsed:.3f}s ({n_classified:,} points classified)")
    
    # Benchmark clustered method
    logger.info(f"\n2. Clustered method - {num_runs} runs:")
    engine_clustered = GeometricRulesEngine(use_clustering=True)
    
    for i in range(num_runs):
        labels_copy = labels.copy()
        start = time.time()
        n_classified = engine_clustered.classify_building_buffer_zone_clustered(
            points=points,
            labels=labels_copy,
            building_geometries=buildings_gdf
        )
        elapsed = time.time() - start
        results['clustered'].append(elapsed)
        logger.info(f"   Run {i+1}: {elapsed:.3f}s ({n_classified:,} points classified)")
    
    # Calculate statistics
    standard_mean = np.mean(results['standard'])
    clustered_mean = np.mean(results['clustered'])
    speedup = standard_mean / clustered_mean
    
    logger.info("\n" + "-"*70)
    logger.info("RESULTS:")
    logger.info(f"  Standard:  {standard_mean:.3f}s ± {np.std(results['standard']):.3f}s")
    logger.info(f"  Clustered: {clustered_mean:.3f}s ± {np.std(results['clustered']):.3f}s")
    logger.info(f"  Speedup:   {speedup:.2f}×")
    logger.info("-"*70)
    
    return {
        'standard_mean': standard_mean,
        'clustered_mean': clustered_mean,
        'speedup': speedup
    }


def benchmark_spectral_rules(points, labels, rgb, nir, ndvi, num_runs=3):
    """
    Benchmark spectral rules classification.
    
    Args:
        points: Point cloud XYZ coordinates
        labels: Classification labels
        rgb: RGB values
        nir: NIR values
        ndvi: NDVI values
        num_runs: Number of runs for averaging
    
    Returns:
        Dict with timing and accuracy results
    """
    from ign_lidar.core.modules.spectral_rules import SpectralRulesEngine
    
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK: Spectral Rules Classification")
    logger.info("="*70)
    
    results = {
        'times': [],
        'classified_counts': []
    }
    
    engine = SpectralRulesEngine()
    
    logger.info(f"\nRunning spectral classification - {num_runs} runs:")
    
    for i in range(num_runs):
        labels_copy = labels.copy()
        start = time.time()
        new_labels, stats = engine.classify_by_spectral_signature(
            rgb=rgb,
            nir=nir,
            current_labels=labels_copy,
            ndvi=ndvi,
            apply_to_unclassified_only=True
        )
        elapsed = time.time() - start
        results['times'].append(elapsed)
        results['classified_counts'].append(stats['total_reclassified'])
        
        logger.info(f"   Run {i+1}: {elapsed:.3f}s ({stats['total_reclassified']:,} points classified)")
        logger.info(f"      Vegetation: {stats.get('vegetation_spectral', 0):,}")
        logger.info(f"      Water: {stats.get('water_spectral', 0):,}")
        logger.info(f"      Buildings: {stats.get('building_concrete_spectral', 0) + stats.get('building_metal_spectral', 0):,}")
        logger.info(f"      Roads: {stats.get('road_asphalt_spectral', 0):,}")
    
    mean_time = np.mean(results['times'])
    mean_classified = np.mean(results['classified_counts'])
    
    logger.info("\n" + "-"*70)
    logger.info("RESULTS:")
    logger.info(f"  Time: {mean_time:.3f}s ± {np.std(results['times']):.3f}s")
    logger.info(f"  Classified: {mean_classified:.0f} ± {np.std(results['classified_counts']):.0f} points")
    logger.info("-"*70)
    
    return {
        'mean_time': mean_time,
        'mean_classified': mean_classified
    }


def benchmark_full_pipeline(points, labels, buildings_gdf, rgb, nir, ndvi, num_runs=1):
    """
    Benchmark full reclassification pipeline with all optimizations.
    
    Args:
        points: Point cloud XYZ coordinates
        labels: Classification labels
        buildings_gdf: GeoDataFrame with building polygons
        rgb: RGB values
        nir: NIR values
        ndvi: NDVI values
        num_runs: Number of runs
    
    Returns:
        Dict with timing results
    """
    from ign_lidar.core.modules.geometric_rules import GeometricRulesEngine
    
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK: Full Reclassification Pipeline")
    logger.info("="*70)
    
    ground_truth_features = {
        'buildings': buildings_gdf,
        'roads': gpd.GeoDataFrame({'geometry': []}, crs='EPSG:2154'),
        'water': gpd.GeoDataFrame({'geometry': []}, crs='EPSG:2154')
    }
    
    results = {
        'without_optimizations': [],
        'with_optimizations': []
    }
    
    # Without optimizations
    logger.info(f"\n1. Without optimizations (clustering=False, spectral=False) - {num_runs} runs:")
    engine_baseline = GeometricRulesEngine(
        use_clustering=False,
        use_spectral_rules=False
    )
    
    for i in range(num_runs):
        labels_copy = labels.copy()
        start = time.time()
        updated_labels, stats = engine_baseline.apply_all_rules(
            points=points,
            labels=labels_copy,
            ground_truth_features=ground_truth_features,
            ndvi=ndvi
        )
        elapsed = time.time() - start
        results['without_optimizations'].append(elapsed)
        logger.info(f"   Run {i+1}: {elapsed:.3f}s ({stats.get('total_changed', 0):,} points modified)")
    
    # With optimizations
    logger.info(f"\n2. With optimizations (clustering=True, spectral=True) - {num_runs} runs:")
    engine_optimized = GeometricRulesEngine(
        use_clustering=True,
        use_spectral_rules=True
    )
    
    for i in range(num_runs):
        labels_copy = labels.copy()
        start = time.time()
        updated_labels, stats = engine_optimized.apply_all_rules(
            points=points,
            labels=labels_copy,
            ground_truth_features=ground_truth_features,
            ndvi=ndvi,
            rgb=rgb,
            nir=nir
        )
        elapsed = time.time() - start
        results['with_optimizations'].append(elapsed)
        logger.info(f"   Run {i+1}: {elapsed:.3f}s ({stats.get('total_changed', 0):,} points modified)")
    
    baseline_mean = np.mean(results['without_optimizations'])
    optimized_mean = np.mean(results['with_optimizations'])
    speedup = baseline_mean / optimized_mean
    
    logger.info("\n" + "-"*70)
    logger.info("RESULTS:")
    logger.info(f"  Without optimizations: {baseline_mean:.3f}s")
    logger.info(f"  With optimizations:    {optimized_mean:.3f}s")
    logger.info(f"  Speedup:               {speedup:.2f}×")
    logger.info("-"*70)
    
    return {
        'baseline_mean': baseline_mean,
        'optimized_mean': optimized_mean,
        'speedup': speedup
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark classification optimizations"
    )
    parser.add_argument(
        '--num-points',
        type=int,
        default=1_000_000,
        help='Number of points for synthetic data (default: 1,000,000)'
    )
    parser.add_argument(
        '--num-buildings',
        type=int,
        default=50,
        help='Number of buildings (default: 50)'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='Number of runs per benchmark (default: 3)'
    )
    parser.add_argument(
        '--skip-buffer',
        action='store_true',
        help='Skip building buffer benchmark'
    )
    parser.add_argument(
        '--skip-spectral',
        action='store_true',
        help='Skip spectral rules benchmark'
    )
    parser.add_argument(
        '--skip-pipeline',
        action='store_true',
        help='Skip full pipeline benchmark'
    )
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Classification Optimization Benchmark")
    logger.info("="*70)
    logger.info(f"Points: {args.num_points:,}")
    logger.info(f"Buildings: {args.num_buildings}")
    logger.info(f"Runs per benchmark: {args.num_runs}")
    logger.info("="*70)
    
    # Create synthetic data
    points, labels, buildings_gdf, rgb, nir, ndvi = create_synthetic_data(
        num_points=args.num_points,
        num_buildings=args.num_buildings
    )
    
    all_results = {}
    
    # Run benchmarks
    if not args.skip_buffer:
        all_results['building_buffer'] = benchmark_building_buffer(
            points, labels, buildings_gdf, num_runs=args.num_runs
        )
    
    if not args.skip_spectral:
        all_results['spectral'] = benchmark_spectral_rules(
            points, labels, rgb, nir, ndvi, num_runs=args.num_runs
        )
    
    if not args.skip_pipeline:
        all_results['pipeline'] = benchmark_full_pipeline(
            points, labels, buildings_gdf, rgb, nir, ndvi, num_runs=args.num_runs
        )
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    
    if 'building_buffer' in all_results:
        bb = all_results['building_buffer']
        logger.info(f"\n Building Buffer Optimization:")
        logger.info(f"   Speedup: {bb['speedup']:.2f}× faster with clustering")
    
    if 'spectral' in all_results:
        sp = all_results['spectral']
        logger.info(f"\n Spectral Rules:")
        logger.info(f"   Processing time: {sp['mean_time']:.3f}s")
        logger.info(f"   Points classified: {sp['mean_classified']:.0f}")
    
    if 'pipeline' in all_results:
        pl = all_results['pipeline']
        logger.info(f"\n Full Pipeline:")
        logger.info(f"   Overall speedup: {pl['speedup']:.2f}×")
        logger.info(f"   Time saved: {(pl['baseline_mean'] - pl['optimized_mean']):.3f}s")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Benchmark complete!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
