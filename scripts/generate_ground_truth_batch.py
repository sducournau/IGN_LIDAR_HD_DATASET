#!/usr/bin/env python3
"""
Batch Ground Truth Generation for Enriched Tiles

This script generates ground truth labeled patches from enriched LiDAR tiles
using IGN BD TOPO® WFS service.

Features:
- Automatic bbox detection from LAZ files
- Road buffer creation from 'largeur' field
- NDVI-based building/vegetation refinement
- Batch processing of multiple tiles
- Progress tracking and error handling

Usage:
    python scripts/generate_ground_truth_batch.py
"""

import sys
from pathlib import Path
import logging
import numpy as np
import laspy
from tqdm import tqdm
from typing import Optional, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar import IGNGroundTruthFetcher
from ign_lidar.io.wfs_ground_truth import generate_patches_with_ground_truth

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Input directory (Windows path converted to WSL)
INPUT_DIR = Path("/mnt/c/Users/Simon/ign/enriched_tiles")

# Output directory
OUTPUT_DIR = Path("/mnt/c/Users/Simon/ign/ground_truth_patches")

# Cache directory for WFS data
CACHE_DIR = Path("data/cache/ground_truth")

# Patch generation settings
PATCH_SIZE = 50  # meters
PATCH_OVERLAP = 0  # no overlap
MIN_POINTS_PER_PATCH = 100

# NDVI settings
USE_NDVI_REFINEMENT = True
NDVI_VEGETATION_THRESHOLD = 0.3  # Points with NDVI >= 0.3 are vegetation
NDVI_BUILDING_THRESHOLD = 0.15   # Points with NDVI <= 0.15 are buildings

# Road settings
ROAD_WIDTH_FALLBACK = 6.0  # Fallback width in meters when 'largeur' field is missing

# Processing settings
MAX_TILES = None  # Set to number to limit, None for all


# ============================================================================
# Helper Functions
# ============================================================================

def get_bbox_from_laz(laz_file: Path) -> Tuple[float, float, float, float]:
    """
    Extract bounding box from LAZ file.
    
    Args:
        laz_file: Path to LAZ file
        
    Returns:
        Tuple of (min_x, min_y, max_x, max_y) in Lambert 93
    """
    try:
        las = laspy.read(str(laz_file))
        
        min_x = float(las.header.x_min)
        max_x = float(las.header.x_max)
        min_y = float(las.header.y_min)
        max_y = float(las.header.y_max)
        
        logger.debug(f"Bbox for {laz_file.name}: ({min_x}, {min_y}, {max_x}, {max_y})")
        
        return (min_x, min_y, max_x, max_y)
        
    except Exception as e:
        logger.error(f"Failed to read bbox from {laz_file}: {e}")
        raise


def load_enriched_tile(laz_file: Path) -> dict:
    """
    Load enriched tile with all available features.
    
    Args:
        laz_file: Path to LAZ file
        
    Returns:
        Dictionary with points and features
    """
    try:
        las = laspy.read(str(laz_file))
        
        # Extract coordinates
        points = np.vstack([las.x, las.y, las.z]).T
        
        # Extract features
        features = {}
        
        # Intensity (always available)
        if hasattr(las, 'intensity'):
            features['intensity'] = las.intensity
            logger.debug(f"  Loaded intensity: {len(las.intensity)} points")
        
        # RGB (if available)
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            # Normalize to 0-1 if needed
            max_val = max(las.red.max(), las.green.max(), las.blue.max())
            if max_val > 1.0:
                rgb = np.vstack([las.red / 65535.0, las.green / 65535.0, las.blue / 65535.0]).T
            else:
                rgb = np.vstack([las.red, las.green, las.blue]).T
            features['rgb'] = rgb
            logger.debug(f"  Loaded RGB: shape {rgb.shape}")
        
        # NIR (if available - check for nir or infrared fields)
        nir_field = None
        for field_name in ['nir', 'infrared', 'near_infrared']:
            if hasattr(las, field_name):
                nir_field = field_name
                break
        
        if nir_field:
            nir = getattr(las, nir_field)
            # Normalize to 0-1 if needed
            if nir.max() > 1.0:
                nir = nir / 65535.0
            features['nir'] = nir
            logger.debug(f"  Loaded NIR ({nir_field}): {len(nir)} points")
        
        # Classification (if available)
        if hasattr(las, 'classification'):
            features['classification'] = las.classification
            logger.debug(f"  Loaded classification: {len(las.classification)} points")
        
        logger.info(f"Loaded {laz_file.name}: {len(points)} points, {len(features)} features")
        
        return {
            'points': points,
            'features': features,
            'filename': laz_file.name
        }
        
    except Exception as e:
        logger.error(f"Failed to load {laz_file}: {e}")
        raise


def process_tile(
    laz_file: Path,
    fetcher: IGNGroundTruthFetcher,
    output_dir: Path
) -> dict:
    """
    Process a single tile to generate ground truth patches.
    
    Args:
        laz_file: Path to LAZ file
        fetcher: Ground truth fetcher instance
        output_dir: Output directory for patches
        
    Returns:
        Dictionary with processing statistics
    """
    tile_name = laz_file.stem
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {tile_name}")
    logger.info(f"{'='*80}")
    
    stats = {
        'tile_name': tile_name,
        'success': False,
        'num_patches': 0,
        'num_points': 0,
        'error': None
    }
    
    try:
        # 1. Get bounding box
        logger.info("Step 1/4: Extracting bounding box...")
        bbox = get_bbox_from_laz(laz_file)
        logger.info(f"  Bbox: {bbox}")
        
        # 2. Load tile data
        logger.info("Step 2/4: Loading enriched tile...")
        tile_data = load_enriched_tile(laz_file)
        stats['num_points'] = len(tile_data['points'])
        
        # 3. Fetch ground truth
        logger.info("Step 3/4: Fetching ground truth from IGN BD TOPO®...")
        ground_truth = fetcher.fetch_all_features(
            bbox, 
            use_cache=True,
            road_width_fallback=ROAD_WIDTH_FALLBACK
        )
        
        # Log what was fetched
        for feature_type, gdf in ground_truth.items():
            if gdf is not None and len(gdf) > 0:
                logger.info(f"  {feature_type}: {len(gdf)} features")
                if feature_type == 'roads' and 'width_m' in gdf.columns:
                    logger.info(f"    Road widths: {gdf['width_m'].min():.1f}m - {gdf['width_m'].max():.1f}m")
        
        # 4. Generate patches with ground truth labels
        logger.info("Step 4/4: Generating labeled patches...")
        
        # Check if we have RGB and NIR for NDVI
        has_rgb = 'rgb' in tile_data['features']
        has_nir = 'nir' in tile_data['features']
        can_use_ndvi = has_rgb and has_nir and USE_NDVI_REFINEMENT
        
        if can_use_ndvi:
            logger.info("  NDVI refinement: ENABLED (RGB + NIR available)")
        elif USE_NDVI_REFINEMENT:
            logger.warning("  NDVI refinement: DISABLED (missing RGB or NIR)")
        else:
            logger.info("  NDVI refinement: DISABLED (by configuration)")
        
        patches = generate_patches_with_ground_truth(
            points=tile_data['points'],
            features=tile_data['features'],
            tile_bbox=bbox,
            patch_size=PATCH_SIZE,
            use_ndvi_refinement=can_use_ndvi,
            compute_ndvi_if_missing=True,
            cache_dir=CACHE_DIR
        )
        
        stats['num_patches'] = len(patches)
        logger.info(f"  Generated {len(patches)} patches")
        
        # 5. Save patches
        tile_output_dir = output_dir / tile_name
        tile_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving patches to: {tile_output_dir}")
        
        for i, patch in enumerate(patches):
            patch_file = tile_output_dir / f"patch_{i:04d}.npz"
            
            # Save with all available data
            save_data = {
                'points': patch['points'],
                'labels': patch['labels'],
                'patch_id': patch['patch_id'],
                'tile_name': tile_name,
            }
            
            # Add features if available
            for feature_name, feature_data in patch.get('features', {}).items():
                save_data[feature_name] = feature_data
            
            np.savez_compressed(str(patch_file), **save_data)
        
        logger.info(f"✅ Saved {len(patches)} patches")
        
        # Log label distribution
        all_labels = np.concatenate([p['labels'] for p in patches])
        unique, counts = np.unique(all_labels, return_counts=True)
        logger.info("Label distribution:")
        label_names = {0: 'Ground', 1: 'Building', 2: 'Road', 3: 'Water', 4: 'Vegetation', 5: 'Other'}
        for label, count in zip(unique, counts):
            pct = 100.0 * count / len(all_labels)
            logger.info(f"  Class {label} ({label_names.get(label, 'Unknown')}): {count:,} points ({pct:.1f}%)")
        
        stats['success'] = True
        
    except Exception as e:
        logger.error(f"❌ Failed to process {tile_name}: {e}")
        stats['error'] = str(e)
        import traceback
        logger.debug(traceback.format_exc())
    
    return stats


# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Main processing function."""
    
    logger.info("="*80)
    logger.info("BATCH GROUND TRUTH GENERATION")
    logger.info("="*80)
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info(f"Patch size: {PATCH_SIZE}m")
    logger.info(f"NDVI refinement: {USE_NDVI_REFINEMENT}")
    logger.info("="*80)
    
    # Check input directory exists
    if not INPUT_DIR.exists():
        logger.error(f"Input directory does not exist: {INPUT_DIR}")
        logger.info("Please check the path. If using Windows, the path should be:")
        logger.info("  /mnt/c/Users/Simon/ign/enriched_tiles")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all LAZ files
    laz_files = sorted(INPUT_DIR.glob("*.laz"))
    
    if not laz_files:
        logger.error(f"No LAZ files found in {INPUT_DIR}")
        return
    
    logger.info(f"\nFound {len(laz_files)} LAZ files")
    
    # Limit if requested
    if MAX_TILES is not None:
        laz_files = laz_files[:MAX_TILES]
        logger.info(f"Processing first {len(laz_files)} tiles (MAX_TILES={MAX_TILES})")
    
    # Initialize ground truth fetcher
    logger.info("\nInitializing ground truth fetcher...")
    fetcher = IGNGroundTruthFetcher(cache_dir=CACHE_DIR)
    logger.info("✅ Fetcher initialized")
    
    # Process each tile
    all_stats = []
    
    logger.info(f"\n{'='*80}")
    logger.info("PROCESSING TILES")
    logger.info(f"{'='*80}\n")
    
    for laz_file in tqdm(laz_files, desc="Processing tiles"):
        stats = process_tile(laz_file, fetcher, OUTPUT_DIR)
        all_stats.append(stats)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    
    successful = sum(1 for s in all_stats if s['success'])
    failed = len(all_stats) - successful
    total_patches = sum(s['num_patches'] for s in all_stats if s['success'])
    total_points = sum(s['num_points'] for s in all_stats)
    
    logger.info(f"Total tiles processed: {len(all_stats)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"Total patches generated: {total_patches:,}")
    logger.info(f"Total points processed: {total_points:,}")
    
    if failed > 0:
        logger.info("\nFailed tiles:")
        for stats in all_stats:
            if not stats['success']:
                logger.info(f"  {stats['tile_name']}: {stats['error']}")
    
    logger.info(f"\n✅ Output saved to: {OUTPUT_DIR}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
