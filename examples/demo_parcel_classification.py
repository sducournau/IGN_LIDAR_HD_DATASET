"""
Demo: Parcel-Based Classification

This script demonstrates how to use the new ParcelClassifier for
intelligent, spatially-coherent point cloud classification.

Usage:
    python examples/demo_parcel_classification.py --tile <tile_id>
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.core.modules.parcel_classifier import (
    ParcelClassifier,
    ParcelClassificationConfig
)
from ign_lidar.io.cadastre import CadastreFetcher
from ign_lidar.io.bd_foret import BDForetFetcher
from ign_lidar.io.rpg import RPGFetcher
from ign_lidar.features.feature_computer import FeatureComputer
from ign_lidar.io.las_io import LasReader, LasWriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Demonstrate parcel-based classification'
    )
    parser.add_argument(
        '--tile',
        type=str,
        required=True,
        help='LiDAR tile ID (e.g., Classif_0830_6291)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./output'),
        help='Output directory for classified point cloud'
    )
    parser.add_argument(
        '--bbox',
        nargs=4,
        type=float,
        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
        help='Bounding box for cadastre/ground truth (Lambert 93)'
    )
    parser.add_argument(
        '--min-parcel-points',
        type=int,
        default=20,
        help='Minimum points per parcel'
    )
    parser.add_argument(
        '--no-refinement',
        action='store_true',
        help='Skip point-level refinement within parcels'
    )
    parser.add_argument(
        '--export-stats',
        action='store_true',
        help='Export parcel statistics to CSV'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PARCEL-BASED CLASSIFICATION DEMO")
    logger.info("=" * 80)
    logger.info(f"Tile: {args.tile}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load Point Cloud
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading point cloud")
    logger.info("=" * 80)
    
    las_path = Path(f'./data/{args.tile}.laz')
    if not las_path.exists():
        logger.error(f"LAS file not found: {las_path}")
        return 1
    
    reader = LasReader()
    points, colors, labels_original = reader.read_las_file(str(las_path))
    
    n_points = len(points)
    logger.info(f"✓ Loaded {n_points:,} points")
    logger.info(f"  Point cloud extent:")
    logger.info(f"    X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    logger.info(f"    Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    logger.info(f"    Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # ========================================================================
    # STEP 2: Compute Features
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Computing geometric and radiometric features")
    logger.info("=" * 80)
    
    feature_computer = FeatureComputer(mode='ASPRS_FEATURES')
    features = feature_computer.compute_all_features(
        points=points,
        colors=colors,
        k_neighbors=10
    )
    
    logger.info(f"✓ Computed {len(features)} feature types:")
    for feat_name, feat_array in features.items():
        if feat_array is not None:
            logger.info(f"  - {feat_name}: {feat_array.shape}")
    
    # ========================================================================
    # STEP 3: Fetch Ground Truth Data
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Fetching ground truth data")
    logger.info("=" * 80)
    
    # Determine bounding box
    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        # Use point cloud extent with 50m buffer
        buffer = 50.0
        bbox = (
            points[:, 0].min() - buffer,
            points[:, 1].min() - buffer,
            points[:, 0].max() + buffer,
            points[:, 1].max() + buffer
        )
    
    logger.info(f"  Bounding box: {bbox}")
    
    # Fetch cadastre
    logger.info("\n  Fetching cadastral parcels...")
    cadastre_fetcher = CadastreFetcher()
    try:
        cadastre = cadastre_fetcher.fetch_parcels(bbox=bbox)
        logger.info(f"  ✓ Found {len(cadastre)} cadastral parcels")
    except Exception as e:
        logger.error(f"  ✗ Failed to fetch cadastre: {e}")
        return 1
    
    # Fetch BD Forêt (optional)
    logger.info("\n  Fetching BD Forêt data...")
    bd_foret_fetcher = BDForetFetcher()
    try:
        bd_foret = bd_foret_fetcher.fetch_forest_polygons(bbox=bbox)
        logger.info(f"  ✓ Found {len(bd_foret)} forest parcels")
    except Exception as e:
        logger.warning(f"  ⚠ BD Forêt not available: {e}")
        bd_foret = None
    
    # Fetch RPG (optional)
    logger.info("\n  Fetching RPG data...")
    rpg_fetcher = RPGFetcher()
    try:
        rpg = rpg_fetcher.fetch_parcels(bbox=bbox)
        logger.info(f"  ✓ Found {len(rpg)} agricultural parcels")
    except Exception as e:
        logger.warning(f"  ⚠ RPG not available: {e}")
        rpg = None
    
    # ========================================================================
    # STEP 4: Parcel-Based Classification
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Parcel-based classification")
    logger.info("=" * 80)
    
    # Configure classifier
    config = ParcelClassificationConfig(
        min_parcel_points=args.min_parcel_points,
        refine_points=not args.no_refinement,
        refinement_method='feature_based'
    )
    
    classifier = ParcelClassifier(config=config)
    
    # Run classification
    logger.info("\n  Running parcel classifier...")
    labels = classifier.classify_by_parcels(
        points=points,
        features=features,
        cadastre=cadastre,
        bd_foret=bd_foret,
        rpg=rpg
    )
    
    # Classification statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info("\n  Classification results:")
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
    for label, count in zip(unique_labels, counts):
        pct = 100 * count / n_points
        name = label_names.get(label, f'Class {label}')
        logger.info(f"    {name:20s}: {count:8,} ({pct:5.1f}%)")
    
    # ========================================================================
    # STEP 5: Export Results
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Exporting results")
    logger.info("=" * 80)
    
    # Export classified point cloud
    output_path = args.output_dir / f'{args.tile}_parcel_classified.laz'
    logger.info(f"\n  Writing classified point cloud: {output_path}")
    
    writer = LasWriter()
    writer.write_las_file(
        filepath=str(output_path),
        points=points,
        labels=labels,
        colors=colors
    )
    logger.info(f"  ✓ Saved classified point cloud")
    
    # Export parcel statistics (optional)
    if args.export_stats:
        import pandas as pd
        
        stats_path = args.output_dir / f'{args.tile}_parcel_stats.csv'
        logger.info(f"\n  Exporting parcel statistics: {stats_path}")
        
        stats_list = classifier.export_parcel_statistics()
        if stats_list:
            df = pd.DataFrame(stats_list)
            df.to_csv(stats_path, index=False)
            logger.info(f"  ✓ Saved {len(stats_list)} parcel statistics")
            
            # Summary
            logger.info(f"\n  Parcel type summary:")
            type_counts = df['parcel_type'].value_counts()
            for ptype, count in type_counts.items():
                logger.info(f"    {ptype:15s}: {count:4d} parcels")
        else:
            logger.warning(f"  ⚠ No parcel statistics available")
    
    # ========================================================================
    # DONE
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("✓ CLASSIFICATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutputs:")
    logger.info(f"  - Classified point cloud: {output_path}")
    if args.export_stats:
        logger.info(f"  - Parcel statistics: {stats_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
