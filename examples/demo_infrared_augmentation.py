#!/usr/bin/env python3
"""
Demonstration of Infrared Augmentation

This script shows how to:
1. Augment LiDAR point clouds with infrared (NIR) values
2. Compare RGB vs infrared values
3. Calculate NDVI (Normalized Difference Vegetation Index)
"""

from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_infrared_augmentation(
    laz_file: Path,
    output_dir: Path,
    cache_dir: Path = None
):
    """
    Demonstrate infrared augmentation on a LAZ file.
    
    Args:
        laz_file: Path to input LAZ file
        output_dir: Directory for output files
        cache_dir: Optional cache directory
    """
    try:
        import laspy
        from ign_lidar.infrared_augmentation import IGNInfraredFetcher
        from ign_lidar.rgb_augmentation import IGNOrthophotoFetcher
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install with: pip install laspy requests Pillow")
        return
    
    logger.info("=" * 70)
    logger.info("Infrared Augmentation Demo")
    logger.info("=" * 70)
    
    # Load LAZ file
    logger.info(f"Loading {laz_file.name}...")
    las = laspy.read(str(laz_file))
    points = np.vstack([las.x, las.y, las.z]).T
    
    logger.info(f"  Points: {len(points):,}")
    
    # Compute bounding box
    bbox = (
        points[:, 0].min(),
        points[:, 1].min(),
        points[:, 0].max(),
        points[:, 1].max()
    )
    
    logger.info(f"  Bbox: {bbox}")
    
    # Fetch RGB colors
    logger.info("\nFetching RGB colors...")
    rgb_fetcher = IGNOrthophotoFetcher(cache_dir=cache_dir)
    rgb = rgb_fetcher.augment_points_with_rgb(points, bbox=bbox)
    logger.info(f"  ✓ Fetched RGB for {len(points):,} points")
    
    # Fetch infrared values
    logger.info("\nFetching infrared values...")
    nir_fetcher = IGNInfraredFetcher(cache_dir=cache_dir)
    nir = nir_fetcher.augment_points_with_infrared(points, bbox=bbox)
    logger.info(f"  ✓ Fetched NIR for {len(points):,} points")
    
    # Calculate NDVI (Normalized Difference Vegetation Index)
    logger.info("\nCalculating NDVI...")
    red = rgb[:, 0].astype(np.float32)
    nir_float = nir.astype(np.float32)
    
    # NDVI = (NIR - Red) / (NIR + Red)
    # Avoid division by zero
    denominator = nir_float + red
    ndvi = np.zeros_like(red)
    mask = denominator > 0
    ndvi[mask] = (nir_float[mask] - red[mask]) / denominator[mask]
    
    logger.info(f"  NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
    logger.info(f"  NDVI mean: {ndvi.mean():.3f}")
    
    # Classify based on NDVI
    vegetation_mask = ndvi > 0.3  # Typical threshold for vegetation
    building_mask = ndvi < 0.1    # Low NDVI indicates non-vegetation
    
    veg_count = np.sum(vegetation_mask)
    building_count = np.sum(building_mask)
    
    logger.info(f"\nClassification (based on NDVI):")
    logger.info(f"  Vegetation: {veg_count:,} points "
                f"({100*veg_count/len(points):.1f}%)")
    logger.info(f"  Buildings: {building_count:,} points "
                f"({100*building_count/len(points):.1f}%)")
    
    # Save augmented LAZ
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{laz_file.stem}_augmented.laz"
    
    logger.info(f"\nSaving augmented LAZ to {output_path.name}...")
    
    # Add RGB
    if las.header.point_format.id in [2, 3, 5, 6, 7, 8, 10]:
        las.red = rgb[:, 0].astype(np.uint16) * 257
        las.green = rgb[:, 1].astype(np.uint16) * 257
        las.blue = rgb[:, 2].astype(np.uint16) * 257
    else:
        from laspy import ExtraBytesParams
        las.add_extra_dim(ExtraBytesParams(name='red', type=np.uint8))
        las.add_extra_dim(ExtraBytesParams(name='green', type=np.uint8))
        las.add_extra_dim(ExtraBytesParams(name='blue', type=np.uint8))
        las.red = rgb[:, 0]
        las.green = rgb[:, 1]
        las.blue = rgb[:, 2]
    
    # Add infrared
    las.add_extra_dim(laspy.ExtraBytesParams(name='nir', type=np.uint8))
    las.nir = nir
    
    # Add NDVI
    las.add_extra_dim(laspy.ExtraBytesParams(name='ndvi', type=np.float32))
    las.ndvi = ndvi
    
    # Save
    las.write(output_path)
    
    logger.info(f"  ✓ Saved with RGB, NIR, and NDVI")
    logger.info(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Statistics summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary Statistics")
    logger.info("=" * 70)
    logger.info(f"RGB Red:   [{rgb[:, 0].min()}, {rgb[:, 0].max()}], "
                f"mean={rgb[:, 0].mean():.1f}")
    logger.info(f"RGB Green: [{rgb[:, 1].min()}, {rgb[:, 1].max()}], "
                f"mean={rgb[:, 1].mean():.1f}")
    logger.info(f"RGB Blue:  [{rgb[:, 2].min()}, {rgb[:, 2].max()}], "
                f"mean={rgb[:, 2].mean():.1f}")
    logger.info(f"NIR:       [{nir.min()}, {nir.max()}], "
                f"mean={nir.mean():.1f}")
    logger.info(f"NDVI:      [{ndvi.min():.3f}, {ndvi.max():.3f}], "
                f"mean={ndvi.mean():.3f}")
    
    logger.info("\n✅ Demo complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Demonstrate infrared augmentation'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input LAZ file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        help='Cache directory for orthophotos (optional)'
    )
    
    args = parser.parse_args()
    
    demo_infrared_augmentation(
        args.input,
        args.output,
        args.cache_dir
    )
