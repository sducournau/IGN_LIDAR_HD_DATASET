"""Ground truth command for IGN LiDAR HD CLI."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


@click.command()
@click.argument('tile_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--patch-size', '-s', default=150.0, type=float,
              help='Patch size in meters (default: 150.0)')
@click.option('--cache-dir', '-c', type=click.Path(),
              help='Cache directory for ground truth data')
@click.option('--include-roads/--no-roads', default=True,
              help='Include road polygons from BD TOPO (default: True)')
@click.option('--include-buildings/--no-buildings', default=True,
              help='Include building footprints (default: True)')
@click.option('--include-water/--no-water', default=True,
              help='Include water surfaces (default: True)')
@click.option('--include-vegetation/--no-vegetation', default=True,
              help='Include vegetation zones (default: True)')
@click.option('--save-ground-truth/--no-save-ground-truth', default=True,
              help='Save ground truth vectors to disk (default: True)')
@click.option('--min-points', default=5000, type=int,
              help='Minimum points per patch (default: 5000)')
@click.option('--num-workers', '-j', default=1, type=int,
              help='Number of parallel workers (default: 1)')
@click.option('--use-ndvi/--no-ndvi', default=True,
              help='Use NDVI to refine building/vegetation labels (default: True)')
@click.option('--fetch-rgb-nir/--no-fetch-rgb-nir', default=False,
              help='Fetch RGB and NIR from IGN orthophotos to compute NDVI (default: False)')
@click.option('--road-width-fallback', default=6.0, type=float,
              help='Fallback road width in meters when largeur field is missing (default: 6.0)')
@click.pass_context
def ground_truth_command(
    ctx, tile_file, output_dir, patch_size, cache_dir,
    include_roads, include_buildings, include_water, include_vegetation,
    save_ground_truth, min_points, num_workers, use_ndvi, fetch_rgb_nir,
    road_width_fallback
):
    """
    Generate patches with ground truth labels from IGN BD TOPO®.
    
    Fetches vector data (buildings, roads with width, water, vegetation) from
    IGN's WFS service and uses it to label point clouds for training.
    
    Example:
    
        # Generate patches with all ground truth features
        ign-lidar-hd ground-truth data/tile_0631_6275.laz data/patches_gt
        
        # Only buildings and roads
        ign-lidar-hd ground-truth data/tile.laz data/patches \\
            --no-water --no-vegetation
    """
    try:
        from ...io.wfs_ground_truth import (
            IGNGroundTruthFetcher,
            generate_patches_with_ground_truth
        )
        from ...core.modules.loader import load_laz_file
        from ...core.modules.saver import save_patch_npz
        from ...features.geometric import compute_geometric_features
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install required packages: pip install shapely geopandas")
        raise click.ClickException(
            "Ground truth functionality requires shapely and geopandas"
        )
    
    verbose = ctx.obj.get('verbose', False) if ctx.obj else False
    
    tile_file = Path(tile_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load tile
        logger.info(f"Loading tile: {tile_file}")
        lidar_data = load_laz_file(tile_file)
        points = lidar_data.points
        
        # Compute bounding box
        tile_bbox = (
            points[:, 0].min(),
            points[:, 1].min(),
            points[:, 0].max(),
            points[:, 1].max()
        )
        
        logger.info(f"Tile bbox: {tile_bbox}")
        
        # Fetch ground truth
        logger.info("Fetching ground truth from IGN BD TOPO® WFS...")
        fetcher = IGNGroundTruthFetcher(cache_dir=cache_dir)
        
        ground_truth_features = fetcher.fetch_all_features(
            bbox=tile_bbox,
            include_roads=include_roads,
            include_buildings=include_buildings,
            include_water=include_water,
            include_vegetation=include_vegetation,
            road_width_fallback=road_width_fallback
        )
        
        if not ground_truth_features:
            logger.error("No ground truth features found for tile")
            sys.exit(1)
        
        logger.info(f"Fetched {len(ground_truth_features)} feature types:")
        for feature_type, gdf in ground_truth_features.items():
            logger.info(f"  - {feature_type}: {len(gdf)} features")
        
        # Save ground truth if requested
        if save_ground_truth:
            gt_dir = output_dir / "ground_truth"
            fetcher.save_ground_truth(ground_truth_features, gt_dir, tile_bbox)
        
        # Compute features
        logger.info("Computing geometric features...")
        features = {}
        if hasattr(lidar_data, 'classification'):
            features['classification'] = lidar_data.classification
        if hasattr(lidar_data, 'intensity'):
            features['intensity'] = lidar_data.intensity
        
        # Compute geometric features
        try:
            geom_features = compute_geometric_features(points)
            if geom_features is not None:
                features.update(geom_features)
        except Exception as e:
            logger.warning(f"Failed to compute geometric features: {e}")
        
        # Fetch RGB and NIR if requested for NDVI computation
        ndvi = None
        if use_ndvi and fetch_rgb_nir:
            logger.info("Fetching RGB and NIR from IGN orthophotos for NDVI...")
            try:
                from ...preprocessing.rgb_augmentation import IGNOrthophotoFetcher
                from ...preprocessing.infrared_augmentation import IGNInfraredFetcher
                from ...core.modules.enrichment import compute_ndvi
                
                # Fetch RGB
                rgb_fetcher = IGNOrthophotoFetcher(cache_dir=cache_dir)
                rgb = rgb_fetcher.augment_points_with_rgb(points, resolution=0.2)
                if rgb is not None:
                    features['rgb'] = rgb
                    logger.info(f"  Fetched RGB for {len(rgb)} points")
                
                # Fetch NIR
                nir_fetcher = IGNInfraredFetcher(cache_dir=cache_dir)
                nir = nir_fetcher.augment_points_with_infrared(points, resolution=0.2)
                if nir is not None:
                    features['nir'] = nir
                    logger.info(f"  Fetched NIR for {len(nir)} points")
                
                # Compute NDVI
                if rgb is not None and nir is not None:
                    ndvi = compute_ndvi(rgb, nir)
                    features['ndvi'] = ndvi
                    logger.info(f"  Computed NDVI (range: {ndvi.min():.3f} to {ndvi.max():.3f})")
                
            except ImportError as e:
                logger.warning(f"Cannot fetch RGB/NIR: {e}")
            except Exception as e:
                logger.warning(f"Failed to fetch RGB/NIR: {e}")
        
        # Check if NDVI already exists in features
        elif use_ndvi and 'ndvi' in features:
            ndvi = features['ndvi']
            logger.info("Using existing NDVI from features")
        
        # Label points with ground truth (with optional NDVI refinement)
        logger.info("Labeling points with ground truth...")
        labels = fetcher.label_points_with_ground_truth(
            points=points,
            ground_truth_features=ground_truth_features,
            ndvi=ndvi,
            use_ndvi_refinement=use_ndvi and ndvi is not None
        )
        
        # Extract patches
        logger.info(f"Extracting patches (size={patch_size}m, min_points={min_points})...")
        from ...core.modules.patch_extractor import extract_patches
        
        patches = extract_patches(
            points=points,
            features=features,
            labels=labels,
            patch_size=patch_size,
            min_points=min_points
        )
        
        if not patches:
            logger.error("No valid patches extracted")
            sys.exit(1)
        
        logger.info(f"Extracted {len(patches)} patches")
        
        # Save patches
        logger.info(f"Saving patches to {output_dir}...")
        tile_name = tile_file.stem
        
        for i, patch in enumerate(patches):
            patch_file = output_dir / f"{tile_name}_patch_{i:04d}.npz"
            save_patch_npz(patch, patch_file)
        
        logger.info(f"✅ Successfully generated {len(patches)} patches with ground truth")
        logger.info(f"Output: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error processing tile: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
