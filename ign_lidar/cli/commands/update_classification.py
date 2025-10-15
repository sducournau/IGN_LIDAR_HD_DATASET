"""Update classification command for IGN LiDAR HD CLI."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--cache-dir', '-c', type=click.Path(),
              help='Cache directory for ground truth and orthophoto data')
@click.option('--use-ndvi/--no-ndvi', default=True,
              help='Use NDVI to refine building/vegetation labels (default: True)')
@click.option('--fetch-rgb-nir/--no-fetch-rgb-nir', default=False,
              help='Fetch RGB and NIR from IGN orthophotos to compute NDVI (default: False)')
@click.option('--ndvi-vegetation-threshold', default=0.3, type=float,
              help='NDVI threshold for vegetation (>= value) (default: 0.3)')
@click.option('--ndvi-building-threshold', default=0.15, type=float,
              help='NDVI threshold for buildings (<= value) (default: 0.15)')
@click.option('--update-buildings/--no-update-buildings', default=True,
              help='Update building classification (default: True)')
@click.option('--update-roads/--no-update-roads', default=True,
              help='Update road classification (default: True)')
@click.option('--update-water/--no-update-water', default=True,
              help='Update water classification (default: True)')
@click.option('--update-vegetation/--no-update-vegetation', default=True,
              help='Update vegetation classification (default: True)')
@click.option('--backup/--no-backup', default=True,
              help='Create backup of original classification (default: True)')
@click.option('--road-width-fallback', default=6.0, type=float,
              help='Fallback road width in meters when largeur field is missing (default: 6.0)')
@click.pass_context
def update_classification_command(
    ctx, input_file, output_file, cache_dir, use_ndvi, fetch_rgb_nir,
    ndvi_vegetation_threshold, ndvi_building_threshold,
    update_buildings, update_roads, update_water, update_vegetation,
    backup, road_width_fallback
):
    """
    Update point cloud classification using ground truth and NDVI.
    
    This command updates an existing LAZ file's classification by:
    1. Fetching ground truth vectors from IGN BD TOPO®
    2. Optionally computing NDVI from RGB and NIR
    3. Refining classification with NDVI (buildings vs vegetation)
    4. Writing updated classification to new LAZ file
    
    Examples:
    
        # Update classification with ground truth and NDVI
        ign-lidar-hd update-classification input.laz output.laz --use-ndvi
        
        # Fetch RGB/NIR and compute NDVI automatically
        ign-lidar-hd update-classification input.laz output.laz \\
            --use-ndvi --fetch-rgb-nir --cache-dir cache/
        
        # Only update buildings and roads
        ign-lidar-hd update-classification input.laz output.laz \\
            --no-update-water --no-update-vegetation
        
        # Custom NDVI thresholds for rural areas
        ign-lidar-hd update-classification input.laz output.laz \\
            --use-ndvi --ndvi-vegetation-threshold 0.35 \\
            --ndvi-building-threshold 0.20
    """
    try:
        import laspy
        from ...io.wfs_ground_truth import IGNGroundTruthFetcher
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install: pip install laspy shapely geopandas")
        raise click.ClickException(
            "Classification update requires laspy, shapely and geopandas"
        )
    
    verbose = ctx.obj.get('verbose', False) if ctx.obj else False
    
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load input LAZ file
        logger.info(f"Loading: {input_file}")
        las = laspy.read(str(input_file))
        
        # Extract points
        points = np.vstack([las.x, las.y, las.z]).T
        num_points = len(points)
        logger.info(f"Loaded {num_points:,} points")
        
        # Compute bounding box
        bbox = (
            float(las.header.x_min),
            float(las.header.y_min),
            float(las.header.x_max),
            float(las.header.y_max)
        )
        logger.info(f"Bounding box: {bbox}")
        
        # Create backup of original classification if requested
        if backup and hasattr(las, 'classification'):
            original_classification = las.classification.copy()
            logger.info("Created backup of original classification")
        
        # Fetch ground truth
        logger.info("Fetching ground truth from IGN BD TOPO® WFS...")
        fetcher = IGNGroundTruthFetcher(cache_dir=cache_dir)
        
        ground_truth = fetcher.fetch_all_features(
            bbox=bbox,
            include_buildings=update_buildings,
            include_roads=update_roads,
            include_water=update_water,
            include_vegetation=update_vegetation,
            road_width_fallback=road_width_fallback
        )
        
        if not ground_truth:
            logger.error("No ground truth features found")
            sys.exit(1)
        
        logger.info("Fetched ground truth:")
        for feature_type, gdf in ground_truth.items():
            if gdf is not None and len(gdf) > 0:
                logger.info(f"  - {feature_type}: {len(gdf)} features")
                if feature_type == 'roads' and 'width_m' in gdf.columns:
                    logger.info(f"    Road widths: {gdf['width_m'].min():.1f}m - {gdf['width_m'].max():.1f}m")
        
        # Prepare NDVI if requested
        ndvi = None
        
        if use_ndvi:
            # Try to get NDVI from existing data
            if hasattr(las, 'ndvi'):
                ndvi = las.ndvi
                logger.info(f"Using existing NDVI from LAZ file")
            
            # Check if RGB and NIR are available to compute NDVI
            elif hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                # Check for NIR field
                nir_field = None
                for field_name in ['nir', 'infrared', 'near_infrared']:
                    if hasattr(las, field_name):
                        nir_field = field_name
                        break
                
                if nir_field:
                    logger.info(f"Computing NDVI from RGB and {nir_field}...")
                    try:
                        from ...core.modules.enrichment import compute_ndvi
                        
                        # Normalize RGB
                        max_val = max(las.red.max(), las.green.max(), las.blue.max())
                        if max_val > 1.0:
                            rgb = np.vstack([
                                las.red / 65535.0,
                                las.green / 65535.0,
                                las.blue / 65535.0
                            ]).T
                        else:
                            rgb = np.vstack([las.red, las.green, las.blue]).T
                        
                        # Normalize NIR
                        nir_array = getattr(las, nir_field)
                        if nir_array.max() > 1.0:
                            nir_array = nir_array / 65535.0
                        
                        ndvi = compute_ndvi(rgb, nir_array)
                        logger.info(f"Computed NDVI (range: {ndvi.min():.3f} to {ndvi.max():.3f})")
                        
                    except Exception as e:
                        logger.warning(f"Failed to compute NDVI: {e}")
                else:
                    logger.warning("RGB available but NIR not found in LAZ file")
            
            # Fetch RGB and NIR if requested
            elif fetch_rgb_nir:
                logger.info("Fetching RGB and NIR from IGN orthophotos...")
                try:
                    from ...preprocessing.rgb_augmentation import IGNOrthophotoFetcher
                    from ...preprocessing.infrared_augmentation import IGNInfraredFetcher
                    from ...core.modules.enrichment import compute_ndvi
                    
                    # Fetch RGB
                    rgb_fetcher = IGNOrthophotoFetcher(cache_dir=cache_dir)
                    rgb = rgb_fetcher.augment_points_with_rgb(points, resolution=0.2)
                    if rgb is not None:
                        # Normalize RGB from [0, 255] to [0, 1]
                        rgb = rgb.astype(np.float32) / 255.0
                    
                    # Fetch NIR
                    nir_fetcher = IGNInfraredFetcher(cache_dir=cache_dir)
                    nir = nir_fetcher.augment_points_with_infrared(points, resolution=0.2)
                    if nir is not None:
                        # Normalize NIR from [0, 255] to [0, 1]
                        nir = nir.astype(np.float32) / 255.0
                    
                    # Compute NDVI
                    if rgb is not None and nir is not None:
                        ndvi = compute_ndvi(rgb, nir)
                        logger.info(f"Computed NDVI from fetched data (range: {ndvi.min():.3f} to {ndvi.max():.3f})")
                    else:
                        logger.warning("Failed to fetch RGB or NIR")
                
                except Exception as e:
                    logger.warning(f"Failed to fetch RGB/NIR: {e}")
            else:
                logger.warning("NDVI requested but no RGB/NIR data available")
                logger.info("Use --fetch-rgb-nir to fetch from IGN orthophotos")
        
        # Label points with ground truth (with optional NDVI refinement)
        logger.info("Updating classification with ground truth...")
        if use_ndvi and ndvi is not None:
            logger.info(f"NDVI refinement: ENABLED")
            logger.info(f"  Vegetation threshold: >= {ndvi_vegetation_threshold}")
            logger.info(f"  Building threshold: <= {ndvi_building_threshold}")
        
        new_labels = fetcher.label_points_with_ground_truth(
            points=points,
            ground_truth_features=ground_truth,
            ndvi=ndvi,
            use_ndvi_refinement=use_ndvi and ndvi is not None,
            ndvi_vegetation_threshold=ndvi_vegetation_threshold,
            ndvi_building_threshold=ndvi_building_threshold
        )
        
        # Count changes
        if hasattr(las, 'classification'):
            changes = (original_classification != new_labels).sum()
            change_pct = 100.0 * changes / num_points
            logger.info(f"Classification changes: {changes:,} points ({change_pct:.1f}%)")
            
            # Show label distribution changes
            logger.info("\nLabel distribution:")
            logger.info("  Class | Original  | Updated   | Change")
            logger.info("  ------|-----------|-----------|--------")
            
            label_names = {
                0: 'Ground',
                1: 'Building',
                2: 'Road/Rail',
                3: 'Water',
                4: 'Vegetation',
                5: 'Other'
            }
            
            for label in range(6):
                old_count = (original_classification == label).sum()
                new_count = (new_labels == label).sum()
                diff = new_count - old_count
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                logger.info(
                    f"  {label} {label_names.get(label, 'Unknown'):10s} | "
                    f"{old_count:9,} | {new_count:9,} | {diff_str:>8}"
                )
        else:
            logger.info("No original classification - creating new")
        
        # Update classification in LAS file
        las.classification = new_labels
        
        # Add NDVI to output if computed
        if ndvi is not None and not hasattr(las, 'ndvi'):
            try:
                # Try to add as extra dimension
                las.add_extra_dim(laspy.ExtraDimension(
                    name="ndvi",
                    type=np.float32,
                    description="Normalized Difference Vegetation Index"
                ))
                las.ndvi = ndvi.astype(np.float32)
                logger.info("Added NDVI as extra dimension to output file")
            except Exception as e:
                logger.warning(f"Could not add NDVI to output file: {e}")
        
        # Write output file
        logger.info(f"Writing updated classification to: {output_file}")
        las.write(str(output_file))
        
        logger.info(f"✅ Successfully updated classification")
        logger.info(f"Output: {output_file}")
        
        if backup and hasattr(las, 'classification'):
            logger.info(f"Note: Original classification backed up in memory")
        
    except Exception as e:
        logger.error(f"Error updating classification: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
