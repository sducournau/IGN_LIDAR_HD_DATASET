"""
Benchmark: Parcel-Based Classification Performance

This script benchmarks the performance improvements of parcel-based
classification compared to traditional point-by-point classification.

Metrics measured:
- Processing time (with/without parcels)
- Memory usage
- Classification accuracy (if ground truth available)
- Point distribution by class

Usage:
    python scripts/benchmark_parcel_classification.py --tile <tile_id>
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.core.classification import AdvancedClassifier
from ign_lidar.io.cadastre import CadastreFetcher
from ign_lidar.io.bd_foret import BDForetFetcher
from ign_lidar.io.rpg import RPGFetcher
from ign_lidar.features.feature_computer import FeatureComputer
from ign_lidar.io.las_io import LasReader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def measure_memory():
    """Measure current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        return 0.0


def compute_classification_stats(labels: np.ndarray) -> Dict[str, Any]:
    """Compute classification statistics."""
    label_names = {
        1: 'Unclassified',
        2: 'Ground',
        3: 'Low Vegetation',
        4: 'Medium Vegetation',
        5: 'High Vegetation',
        6: 'Building',
        9: 'Water',
        11: 'Road'
    }
    
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    stats = {
        'total_points': total,
        'distribution': {},
        'n_classes': len(unique)
    }
    
    for label, count in zip(unique, counts):
        name = label_names.get(label, f'Class {label}')
        pct = 100 * count / total
        stats['distribution'][name] = {
            'count': int(count),
            'percentage': float(pct)
        }
    
    return stats


def benchmark_classification(
    points: np.ndarray,
    features: Dict[str, np.ndarray],
    ground_truth_features: Dict,
    use_parcels: bool = False
) -> Dict[str, Any]:
    """
    Run classification and measure performance.
    
    Args:
        points: Point coordinates [N, 3]
        features: Computed features
        ground_truth_features: Ground truth data
        use_parcels: Whether to use parcel classification
        
    Returns:
        Dictionary with timing, memory, and classification stats
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Benchmarking: {'WITH' if use_parcels else 'WITHOUT'} parcel classification")
    logger.info(f"{'='*80}")
    
    # Initial memory
    mem_start = measure_memory()
    
    # Create classifier
    classifier = AdvancedClassifier(
        use_parcel_classification=use_parcels,
        parcel_classification_config={
            'min_parcel_points': 20,
            'parcel_confidence_threshold': 0.6,
            'refine_points': True
        } if use_parcels else None,
        use_ground_truth=True,
        use_ndvi=True,
        use_geometric=True
    )
    
    # Run classification
    time_start = time.time()
    
    labels = classifier.classify_points(
        points=points,
        ground_truth_features=ground_truth_features,
        ndvi=features.get('ndvi'),
        height=features.get('height'),
        normals=features.get('normals'),
        planarity=features.get('planarity'),
        verticality=features.get('verticality'),
        curvature=features.get('curvature')
    )
    
    time_end = time.time()
    elapsed = time_end - time_start
    
    # Final memory
    mem_end = measure_memory()
    mem_used = mem_end - mem_start
    
    # Compute statistics
    class_stats = compute_classification_stats(labels)
    
    # Points per second
    pps = len(points) / elapsed if elapsed > 0 else 0
    
    results = {
        'use_parcels': use_parcels,
        'n_points': len(points),
        'elapsed_time_sec': elapsed,
        'elapsed_time_min': elapsed / 60,
        'points_per_second': pps,
        'memory_used_mb': mem_used,
        'classification_stats': class_stats,
        'labels': labels
    }
    
    # Log results
    logger.info(f"\n{'='*80}")
    logger.info(f"Results: {'WITH' if use_parcels else 'WITHOUT'} parcel classification")
    logger.info(f"{'='*80}")
    logger.info(f"  Total points:        {len(points):,}")
    logger.info(f"  Processing time:     {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"  Points per second:   {pps:,.0f}")
    logger.info(f"  Memory used:         {mem_used:.1f} MB")
    logger.info(f"  Classified classes:  {class_stats['n_classes']}")
    
    logger.info(f"\n  Classification distribution:")
    for name, info in sorted(class_stats['distribution'].items(), 
                             key=lambda x: x[1]['count'], reverse=True):
        logger.info(f"    {name:20s}: {info['count']:8,} ({info['percentage']:5.1f}%)")
    
    return results


def compare_results(result_no_parcels: Dict, result_with_parcels: Dict):
    """Compare results between traditional and parcel-based classification."""
    logger.info(f"\n{'='*80}")
    logger.info("PERFORMANCE COMPARISON")
    logger.info(f"{'='*80}")
    
    # Time comparison
    time_without = result_no_parcels['elapsed_time_sec']
    time_with = result_with_parcels['elapsed_time_sec']
    speedup = time_without / time_with if time_with > 0 else 0
    time_saved = time_without - time_with
    
    logger.info(f"\n⏱️  Processing Time:")
    logger.info(f"  Without parcels:  {time_without:.2f}s ({time_without/60:.2f} min)")
    logger.info(f"  With parcels:     {time_with:.2f}s ({time_with/60:.2f} min)")
    logger.info(f"  Time saved:       {time_saved:.2f}s ({time_saved/60:.2f} min)")
    logger.info(f"  Speedup:          {speedup:.2f}×")
    
    # Throughput comparison
    pps_without = result_no_parcels['points_per_second']
    pps_with = result_with_parcels['points_per_second']
    
    logger.info(f"\n🚀 Throughput:")
    logger.info(f"  Without parcels:  {pps_without:,.0f} points/sec")
    logger.info(f"  With parcels:     {pps_with:,.0f} points/sec")
    logger.info(f"  Improvement:      {pps_with - pps_without:+,.0f} points/sec")
    
    # Memory comparison
    mem_without = result_no_parcels['memory_used_mb']
    mem_with = result_with_parcels['memory_used_mb']
    mem_diff = mem_with - mem_without
    
    logger.info(f"\n💾 Memory Usage:")
    logger.info(f"  Without parcels:  {mem_without:.1f} MB")
    logger.info(f"  With parcels:     {mem_with:.1f} MB")
    logger.info(f"  Difference:       {mem_diff:+.1f} MB")
    
    # Classification comparison
    logger.info(f"\n📊 Classification Results:")
    
    dist_without = result_no_parcels['classification_stats']['distribution']
    dist_with = result_with_parcels['classification_stats']['distribution']
    
    all_classes = set(dist_without.keys()) | set(dist_with.keys())
    
    logger.info(f"\n  {'Class':<20} {'Without Parcels':>15} {'With Parcels':>15} {'Difference':>15}")
    logger.info(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
    
    for class_name in sorted(all_classes):
        count_without = dist_without.get(class_name, {}).get('count', 0)
        count_with = dist_with.get(class_name, {}).get('count', 0)
        diff = count_with - count_without
        
        logger.info(f"  {class_name:<20} {count_without:>15,} {count_with:>15,} {diff:>+15,}")
    
    # Agreement analysis
    labels_without = result_no_parcels['labels']
    labels_with = result_with_parcels['labels']
    agreement = np.sum(labels_without == labels_with) / len(labels_without)
    
    logger.info(f"\n🎯 Label Agreement:")
    logger.info(f"  Same classification: {agreement*100:.1f}%")
    logger.info(f"  Changed labels:      {(1-agreement)*100:.1f}%")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    
    if speedup >= 1.5:
        logger.info(f"✅ Parcel classification is {speedup:.1f}× FASTER")
    elif speedup >= 1.1:
        logger.info(f"✅ Parcel classification is slightly faster ({speedup:.1f}×)")
    else:
        logger.info(f"⚠️  Parcel classification is slower ({speedup:.1f}×)")
    
    if abs(mem_diff) < 50:
        logger.info(f"✅ Memory usage is similar ({mem_diff:+.1f} MB difference)")
    elif mem_diff > 0:
        logger.info(f"⚠️  Parcel classification uses {mem_diff:.1f} MB more memory")
    else:
        logger.info(f"✅ Parcel classification uses {abs(mem_diff):.1f} MB less memory")
    
    logger.info(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark parcel-based classification performance'
    )
    parser.add_argument(
        '--tile',
        type=str,
        required=True,
        help='LiDAR tile ID (e.g., Classif_0830_6291)'
    )
    parser.add_argument(
        '--bbox',
        nargs=4,
        type=float,
        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
        help='Bounding box for ground truth (Lambert 93)'
    )
    parser.add_argument(
        '--max-points',
        type=int,
        help='Limit number of points for testing (default: use all)'
    )
    parser.add_argument(
        '--skip-traditional',
        action='store_true',
        help='Skip traditional classification (only run parcel-based)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("PARCEL CLASSIFICATION BENCHMARK")
    logger.info("="*80)
    logger.info(f"Tile: {args.tile}")
    
    # ========================================================================
    # Load Point Cloud
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("LOADING POINT CLOUD")
    logger.info("="*80)
    
    las_path = Path(f'./data/{args.tile}.laz')
    if not las_path.exists():
        logger.error(f"LAS file not found: {las_path}")
        return 1
    
    reader = LasReader()
    points, colors, labels_original = reader.read_las_file(str(las_path))
    
    if args.max_points and len(points) > args.max_points:
        logger.info(f"Limiting to {args.max_points:,} points for testing")
        indices = np.random.choice(len(points), args.max_points, replace=False)
        points = points[indices]
        colors = colors[indices] if colors is not None else None
    
    n_points = len(points)
    logger.info(f"✓ Loaded {n_points:,} points")
    
    # ========================================================================
    # Compute Features
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("COMPUTING FEATURES")
    logger.info("="*80)
    
    feature_computer = FeatureComputer(mode='ASPRS_FEATURES')
    features = feature_computer.compute_all_features(
        points=points,
        colors=colors,
        k_neighbors=10
    )
    
    logger.info(f"✓ Computed {len(features)} feature types")
    
    # ========================================================================
    # Fetch Ground Truth
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("FETCHING GROUND TRUTH")
    logger.info("="*80)
    
    # Determine bounding box
    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        buffer = 50.0
        bbox = (
            points[:, 0].min() - buffer,
            points[:, 1].min() - buffer,
            points[:, 0].max() + buffer,
            points[:, 1].max() + buffer
        )
    
    logger.info(f"Bounding box: {bbox}")
    
    ground_truth_features = {}
    
    # Fetch cadastre
    try:
        cadastre_fetcher = CadastreFetcher()
        cadastre = cadastre_fetcher.fetch_parcels(bbox=bbox)
        ground_truth_features['cadastre'] = cadastre
        logger.info(f"✓ Cadastre: {len(cadastre)} parcels")
    except Exception as e:
        logger.error(f"✗ Failed to fetch cadastre: {e}")
        return 1
    
    # Fetch BD Forêt (optional)
    try:
        bd_foret_fetcher = BDForetFetcher()
        bd_foret = bd_foret_fetcher.fetch_forest_polygons(bbox=bbox)
        ground_truth_features['forest'] = bd_foret
        logger.info(f"✓ BD Forêt: {len(bd_foret)} forest parcels")
    except Exception as e:
        logger.warning(f"⚠ BD Forêt not available: {e}")
    
    # Fetch RPG (optional)
    try:
        rpg_fetcher = RPGFetcher()
        rpg = rpg_fetcher.fetch_parcels(bbox=bbox)
        ground_truth_features['rpg'] = rpg
        logger.info(f"✓ RPG: {len(rpg)} agricultural parcels")
    except Exception as e:
        logger.warning(f"⚠ RPG not available: {e}")
    
    # ========================================================================
    # Benchmark: Without Parcel Classification
    # ========================================================================
    if not args.skip_traditional:
        result_no_parcels = benchmark_classification(
            points=points,
            features=features,
            ground_truth_features=ground_truth_features,
            use_parcels=False
        )
    else:
        result_no_parcels = None
    
    # ========================================================================
    # Benchmark: With Parcel Classification
    # ========================================================================
    result_with_parcels = benchmark_classification(
        points=points,
        features=features,
        ground_truth_features=ground_truth_features,
        use_parcels=True
    )
    
    # ========================================================================
    # Compare Results
    # ========================================================================
    if result_no_parcels:
        compare_results(result_no_parcels, result_with_parcels)
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
