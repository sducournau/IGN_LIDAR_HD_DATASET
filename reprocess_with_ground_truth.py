#!/usr/bin/env python3
"""
Reprocess a single enriched tile to apply BD TOPO® ground truth classification.

This script will:
1. Load an existing enriched LAZ file
2. Fetch BD TOPO® ground truth (roads, railways, buildings, etc.)
3. Apply ground truth classification
4. Save updated LAZ file

Usage:
    python reprocess_with_ground_truth.py /path/to/enriched.laz
"""

import sys
import logging
from pathlib import Path
import laspy
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def reprocess_tile_with_ground_truth(laz_path: Path, output_path: Path = None):
    """Reprocess a single tile to apply ground truth classification."""
    
    if output_path is None:
        output_path = laz_path.parent / f"{laz_path.stem}_with_gt.laz"
    
    logger.info(f"\n{'='*80}")
    logger.info("REPROCESSING TILE WITH BD TOPO® GROUND TRUTH")
    logger.info(f"{'='*80}\n")
    
    # 1. Load existing enriched LAZ
    logger.info(f"Loading: {laz_path.name}")
    las = laspy.read(str(laz_path))
    
    # Get points
    points = np.vstack([las.x, las.y, las.z]).T
    labels = np.array(las.classification, dtype=np.uint8)
    
    # Get bbox
    bbox = (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max())
    )
    
    logger.info(f"  Points: {len(points):,}")
    logger.info(f"  Bbox: {bbox}")
    
    # Original classification distribution
    unique_orig, counts_orig = np.unique(labels, return_counts=True)
    logger.info(f"\n📊 Original classification:")
    for cls, count in zip(unique_orig, counts_orig):
        pct = (count / len(labels)) * 100
        logger.info(f"  Class {cls:2d}: {count:10,} ({pct:5.2f}%)")
    
    # 2. Fetch BD TOPO® ground truth
    logger.info(f"\n{'- '*40}")
    logger.info("📍 Fetching BD TOPO® ground truth...")
    logger.info(f"{'- '*40}\n")
    
    from ign_lidar.io.data_fetcher import DataFetcher, DataFetchConfig
    
    config = DataFetchConfig(
        include_buildings=True,
        include_roads=True,
        include_railways=True,
        include_water=True,
        include_vegetation=True,
        include_bridges=True,
        include_parking=True,
        include_cemeteries=True,
        include_power_lines=True,
        include_sports=True,
        road_width_fallback=4.0,
        railway_width_fallback=3.5,
        power_line_buffer=2.0
    )
    
    fetcher = DataFetcher(
        cache_dir="/mnt/d/ign/cache",
        config=config
    )
    
    gt_data = fetcher.fetch_all(bbox=bbox, use_cache=True)
    
    if not gt_data or 'ground_truth' not in gt_data:
        logger.error("❌ Failed to fetch ground truth data!")
        return False
    
    ground_truth_features = gt_data['ground_truth']
    
    # Log what we found
    available_features = [k for k, v in ground_truth_features.items() if v is not None and len(v) > 0]
    logger.info(f"✅ Found {len(available_features)} feature types:")
    for feat_type in available_features:
        logger.info(f"  {feat_type}: {len(ground_truth_features[feat_type])} features")
    
    # 3. Apply ground truth classification
    logger.info(f"\n{'- '*40}")
    logger.info("🔄 Applying ground truth classification...")
    logger.info(f"{'- '*40}\n")
    
    from ign_lidar.core.modules.advanced_classification import AdvancedClassifier
    
    # Get features if available
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
    
    # Apply classification
    logger.info("Classifying points with ground truth...")
    labels_new = classifier._classify_by_ground_truth(
        labels=labels.copy(),
        points=points,
        ground_truth_features=ground_truth_features,
        ndvi=ndvi,
        height=height,
        planarity=planarity,
        intensity=intensity
    )
    
    # 4. Show changes
    logger.info(f"\n📊 Updated classification:")
    unique_new, counts_new = np.unique(labels_new, return_counts=True)
    for cls, count in zip(unique_new, counts_new):
        pct = (count / len(labels_new)) * 100
        # Check if this class existed before
        if cls in unique_orig:
            orig_count = counts_orig[unique_orig == cls][0]
            diff = count - orig_count
            if diff > 0:
                logger.info(f"  Class {cls:2d}: {count:10,} ({pct:5.2f}%) [+{diff:,}]")
            elif diff < 0:
                logger.info(f"  Class {cls:2d}: {count:10,} ({pct:5.2f}%) [{diff:,}]")
            else:
                logger.info(f"  Class {cls:2d}: {count:10,} ({pct:5.2f}%)")
        else:
            logger.info(f"  Class {cls:2d}: {count:10,} ({pct:5.2f}%) [NEW!]")
    
    # Check for roads and railways
    has_roads = 11 in unique_new
    has_rails = 10 in unique_new
    logger.info(f"\n{'='*80}")
    logger.info(f"Class 10 (Rail): {'✅ YES' if has_rails else '❌ NO'}")
    logger.info(f"Class 11 (Road): {'✅ YES' if has_roads else '❌ NO'}")
    logger.info(f"{'='*80}\n")
    
    # 5. Save updated LAZ
    logger.info(f"Saving to: {output_path.name}")
    las.classification = labels_new
    las.write(str(output_path))
    logger.info(f"✅ Done! File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python reprocess_with_ground_truth.py <enriched.laz> [output.laz]")
        sys.exit(1)
    
    laz_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not laz_path.exists():
        print(f"Error: File not found: {laz_path}")
        sys.exit(1)
    
    success = reprocess_tile_with_ground_truth(laz_path, output_path)
    sys.exit(0 if success else 1)
