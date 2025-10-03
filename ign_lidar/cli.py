#!/usr/bin/env python3
"""
Command-line interface for IGN LiDAR HD processing
Provides subcommands for downloading, enriching, and processing LiDAR data
"""

import argparse
import logging
import sys
from pathlib import Path

from .processor import LiDARProcessor
from .downloader import IGNLiDARDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_process(args):
    """Process LAZ files to create training patches."""
    # Parse bounding box if provided
    bbox = None
    if args.bbox:
        try:
            bbox = [float(x) for x in args.bbox.split(',')]
            if len(bbox) != 4:
                logger.error("Bounding box must have 4 values: xmin,ymin,xmax,ymax")
                return 1
        except ValueError:
            logger.error("Bounding box values must be numeric")
            return 1
    
    # Initialize processor
    processor = LiDARProcessor(
        lod_level=args.lod_level,
        augment=not args.no_augment,
        num_augmentations=args.num_augmentations,
        bbox=bbox,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        num_points=args.num_points,
        k_neighbors=args.k_neighbors,
        include_architectural_style=args.include_architectural_style,
        style_encoding=args.style_encoding
    )
    
    # Process data
    import time
    start_time = time.time()
    
    skip_existing = not args.force if hasattr(args, 'force') else True
    
    if args.input:
        # Process single file
        total_patches = processor.process_tile(
            args.input, args.output, skip_existing=skip_existing
        )
    else:
        # Process directory
        total_patches = processor.process_directory(
            args.input_dir, args.output, args.num_workers,
            skip_existing=skip_existing
        )
    
    elapsed_time = time.time() - start_time
    
    logger.info("✅ Processing complete!")
    logger.info(f"  Total patches: {total_patches:,}")
    logger.info(f"  Processing time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Stats saved to: {args.output}/stats.json")
    logger.info("="*70)
    return 0


def _enrich_single_file(args_tuple):
    """
    Worker function to enrich a single LAZ file (for multiprocessing).
    
    Args:
        args_tuple: (laz_path, output_path, k_neighbors, use_gpu,
                     mode, skip_existing)
    
    Returns:
        str: 'skipped' if file already exists and skip_existing=True,
             'success' if processing succeeded,
             'error' if processing failed
    """
    (laz_path, output_path, k_neighbors,
     use_gpu, mode, skip_existing) = args_tuple
    
    import numpy as np
    import laspy
    import gc
    import logging
    
    # Re-initialize logger for subprocess
    worker_logger = logging.getLogger(__name__)
    
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
    
    # Check if output already exists
    if skip_existing and output_path.exists():
        file_size_mb = output_path.stat().st_size // (1024*1024)
        worker_logger.info(
            f"⏭️  {laz_path.name} already enriched "
            f"({file_size_mb} MB), skipping"
        )
        return 'skipped'
    
    # Import features (GPU if available)
    try:
        if use_gpu:
            from .features_gpu import (
                compute_normals, compute_curvature,
                compute_height_above_ground, extract_geometric_features,
                GPU_AVAILABLE
            )
            if not GPU_AVAILABLE:
                worker_logger.warning(
                    "GPU requested but not available, falling back to CPU"
                )
                from .features import (
                    compute_normals, compute_curvature,
                    compute_height_above_ground, extract_geometric_features
                )
        else:
            from .features import (
                compute_normals, compute_curvature,
                compute_height_above_ground, extract_geometric_features
            )
    except ImportError:
        from .features import (
            compute_normals, compute_curvature,
            compute_height_above_ground, extract_geometric_features
        )
    
    try:
        worker_logger.info(f"Processing {laz_path.name}...")
        
        # Read LAZ
        try:
            with laspy.open(laz_path) as f:
                las = f.read()
        except Exception as read_error:
            worker_logger.error(
                f"  ✗ Cannot read file {laz_path.name}: {read_error}"
            )
            return 'error'
        
        # Detect if this is a COPC file
        is_copc = False
        if hasattr(las.header, 'point_format'):
            if las.header.point_format.id >= 6:
                # Check if this is a COPC file
                if '.copc.' in laz_path.name.lower() or (
                    hasattr(las.header, 'vlrs') and
                    any('copc' in vlr.user_id.lower()
                        for vlr in las.header.vlrs)
                ):
                    is_copc = True
                    worker_logger.info(
                        "  ℹ️ COPC detected - will convert to standard LAZ"
                    )
        
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        classification = np.array(las.classification, dtype=np.uint8)
        
        # Memory check - abort if insufficient memory available
        n_points = len(points)
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                mem_available = mem.available
                available_mb = mem_available / 1024 / 1024
                
                # Estimate required memory based on mode
                # Building mode needs significantly more memory due to:
                # - Main features: ~40 bytes/point
                # - KDTree for main features: ~24 bytes/point
                # - Building features: additional KDTree + features: ~50 bytes/point
                # Total for building mode: ~120-150 bytes/point (conservative)
                # Core mode: ~70 bytes/point
                if mode == 'building':
                    bytes_per_point = 150  # Very conservative for building mode
                else:
                    bytes_per_point = 70   # Conservative for core mode
                
                estimated_needed_mb = (n_points * bytes_per_point) / 1024 / 1024
                
                # Check if we have enough memory (require 60% safety margin)
                # Also warn if swap is being used heavily
                safety_factor = 0.6
                if estimated_needed_mb > available_mb * safety_factor:
                    worker_logger.error(
                        f"  ✗ Insufficient memory for {laz_path.name}"
                    )
                    worker_logger.error(
                        f"    Need ~{estimated_needed_mb:.0f}MB, "
                        f"only {available_mb:.0f}MB available "
                        f"(requires {safety_factor*100:.0f}% safety margin)"
                    )
                    worker_logger.error(
                        "    Reduce --num-workers or process file alone"
                    )
                    return 'error'
                
                # Warn about swap usage
                swap_percent = psutil.swap_memory().percent
                if swap_percent > 50:
                    worker_logger.warning(
                        f"  ⚠️  High swap usage detected ({swap_percent:.0f}%)"
                    )
                    worker_logger.warning(
                        "    System may be under memory pressure"
                    )
                    
            except (AttributeError, NameError):
                # psutil error - continue anyway
                pass
        
        # Determine chunk size based on number of points
        # Memory-efficient processing for large point clouds
        # Conservative chunking to prevent OOM with multiple workers
        if n_points > 40_000_000:
            # Very large (>40M): 5M chunks - aggressive chunking
            chunk_size = 5_000_000
            worker_logger.info("  Using chunked processing (5M per chunk)")
        elif n_points > 20_000_000:
            # Large (20-40M): 10M chunks
            chunk_size = 10_000_000
            worker_logger.info("  Using chunked processing (10M per chunk)")
        elif n_points > 10_000_000:
            # Medium (10-20M): 15M chunks
            chunk_size = 15_000_000
            worker_logger.info("  Using chunked processing (15M per chunk)")
        else:
            # Small (<10M): no chunking - process all at once
            chunk_size = None
        
        # Compute features based on mode
        if mode == 'building':
            worker_logger.info(
                f"  Computing BUILDING features for {len(points):,} points..."
            )
        else:
            worker_logger.info(
                f"  Computing CORE features for {len(points):,} points..."
            )
        
        # Use optimized all-in-one function with chunking support
        try:
            from .features import compute_all_features_optimized
            normals, curvature, height_above_ground, geometric_features = \
                compute_all_features_optimized(
                    points, classification, k=k_neighbors,
                    include_extra=False, chunk_size=chunk_size
                )
        except ImportError:
            # Fallback to individual functions (old code)
            normals = compute_normals(points, k=k_neighbors)
            curvature = compute_curvature(points, normals, k=k_neighbors)
            height_above_ground = compute_height_above_ground(points, classification)
            geometric_features = extract_geometric_features(points, normals, k=k_neighbors)
        
        # Output path is already set (preserves directory structure)
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If COPC, create new header without COPC VLRs to enable extra dims
        if is_copc:
            # Create clean header for standard LAZ (preserving point format)
            from laspy import LasHeader
            new_header = LasHeader(
                version=las.header.version,
                point_format=las.header.point_format.id
            )
            # Copy scale and offset
            new_header.scales = las.header.scales
            new_header.offsets = las.header.offsets
            las_out = laspy.LasData(new_header)
            las_out.x = las.x
            las_out.y = las.y
            las_out.z = las.z
            las_out.classification = las.classification
            # Copy other standard fields if present
            if hasattr(las, 'intensity'):
                las_out.intensity = las.intensity
            if hasattr(las, 'return_number'):
                las_out.return_number = las.return_number
            if hasattr(las, 'number_of_returns'):
                las_out.number_of_returns = las.number_of_returns
        else:
            # Standard LAZ - preserve original header
            las_out = laspy.LasData(las.header)
            las_out.points = las.points
        
        # Add extra dimensions
        for i, dim in enumerate(['normal_x', 'normal_y', 'normal_z']):
            las_out.add_extra_dim(laspy.ExtraBytesParams(name=dim, type=np.float32))
            setattr(las_out, dim, normals[:, i])
        
        las_out.add_extra_dim(laspy.ExtraBytesParams(name='curvature', type=np.float32))
        las_out.curvature = curvature
        
        las_out.add_extra_dim(laspy.ExtraBytesParams(name='height_above_ground', type=np.float32))
        las_out.height_above_ground = height_above_ground
        
        for key, values in geometric_features.items():
            las_out.add_extra_dim(laspy.ExtraBytesParams(name=key, type=np.float32))
            setattr(las_out, key, values)
        
        # Add extra building-specific features if in building mode
        if mode == 'building':
            try:
                # Import building-specific feature functions
                if use_gpu:
                    from .features_gpu import (
                        compute_verticality, compute_wall_score,
                        compute_roof_score, compute_num_points_in_radius
                    )
                else:
                    from .features import (
                        compute_verticality, compute_wall_score,
                        compute_roof_score, compute_num_points_in_radius
                    )
                
                worker_logger.info("  Computing building-specific features...")
                
                # Verticality
                verticality = compute_verticality(normals)
                las_out.add_extra_dim(
                    laspy.ExtraBytesParams(name='verticality', type=np.float32)
                )
                las_out.verticality = verticality
                
                # Wall score
                wall_score = compute_wall_score(normals, height_above_ground)
                las_out.add_extra_dim(
                    laspy.ExtraBytesParams(name='wall_score', type=np.float32)
                )
                las_out.wall_score = wall_score
                
                # Roof score
                roof_score = compute_roof_score(
                    normals, height_above_ground, curvature
                )
                las_out.add_extra_dim(
                    laspy.ExtraBytesParams(name='roof_score', type=np.float32)
                )
                las_out.roof_score = roof_score
                
                # Num points in radius (memory-optimized with chunking)
                # Use aggressive chunking for large point clouds
                # This feature is memory-intensive due to KDTree queries
                if n_points > 5_000_000:
                    # Very large clouds: 500k chunks
                    radius_chunk_size = 500_000
                elif n_points > 3_000_000:
                    # Large clouds: 750k chunks
                    radius_chunk_size = 750_000
                else:
                    # Smaller clouds: 1M chunks
                    radius_chunk_size = 1_000_000
                
                num_points = compute_num_points_in_radius(
                    points, radius=2.0, chunk_size=radius_chunk_size
                )
                las_out.add_extra_dim(
                    laspy.ExtraBytesParams(name='num_points_2m', type=np.float32)
                )
                las_out.num_points_2m = num_points
                
            except Exception as e:
                worker_logger.warning(
                    f"  Could not compute building features: {e}"
                )
        
        # Save with LAZ compression (LASzip format for QGIS compatibility)
        # do_compress=True ensures LAZ compression is applied
        # laspy will automatically use the best available backend (laszip preferred)
        las_out.write(output_path, do_compress=True)
        worker_logger.info(f"  ✓ Saved to {output_path}")
        
        # Explicit cleanup to free memory immediately
        del las, las_out, points, classification, normals
        del curvature, height_above_ground, geometric_features
        gc.collect()
        
        return 'success'
        
    except Exception as e:
        worker_logger.error(f"  ✗ Error processing {laz_path.name}: {e}")
        # Clean up on error
        gc.collect()
        return 'error'


def cmd_enrich(args):
    """Enrich LAZ files with geometric features."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    from .config import PREFER_AUGMENTED_LAZ, AUTO_CONVERT_TO_QGIS
    
    # Setup
    input_path = args.input or args.input_dir
    if not input_path or not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get LAZ files
    if input_path.is_file():
        laz_files = [input_path]
    else:
        laz_files = list(input_path.rglob("*.laz"))
    
    if not laz_files:
        logger.error(f"No LAZ files found in {input_path}")
        return 1
    
    # Get mode (default to 'core')
    mode = getattr(args, 'mode', 'core')
    
    # Check if auto-convert to QGIS is requested
    auto_convert = getattr(args, 'auto_convert_qgis', AUTO_CONVERT_TO_QGIS)
    
    # Log output format preference
    if PREFER_AUGMENTED_LAZ and not auto_convert:
        logger.info("Output format: Augmented LAZ (LAZ 1.4, all features)")
        logger.info("  Use --auto-convert-qgis to also create QGIS versions")
    elif auto_convert:
        logger.info("Output format: Augmented LAZ + QGIS-compatible versions")
    
    logger.info("")
    
    logger.info(f"Found {len(laz_files)} LAZ files to enrich")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info(f"Mode: {mode.upper()}")
    
    # Check system memory status BEFORE starting
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        available_gb = mem.available / (1024**3)
        swap_percent = swap.percent
        
        logger.info(f"System Memory: {available_gb:.1f}GB available")
        
        # If swap is heavily used, reduce workers automatically
        if swap_percent > 50:
            logger.warning(
                f"⚠️  High swap usage detected ({swap_percent:.0f}%)"
            )
            logger.warning(
                "⚠️  System is under memory pressure - reducing workers to 1"
            )
            args.num_workers = 1
        
        # If very low available RAM, force single worker
        # Building mode needs ~4-6GB per worker, core mode ~2-3GB
        min_gb_per_worker = 5.0 if mode == 'building' else 2.5
        max_safe_workers = int(available_gb / min_gb_per_worker)
        
        if max_safe_workers < args.num_workers:
            logger.warning(
                f"⚠️  Limited RAM ({available_gb:.1f}GB available)"
            )
            logger.warning(
                f"⚠️  Reducing workers from {args.num_workers} "
                f"to {max(1, max_safe_workers)}"
            )
            args.num_workers = max(1, max_safe_workers)
            
    except ImportError:
        logger.warning("⚠️  psutil not available - cannot check memory")
    
    # Analyze file sizes and optimize worker count
    laz_files_with_size = [(laz, laz.stat().st_size) for laz in laz_files]
    max_file_size = max(size for _, size in laz_files_with_size)
    
    # Dynamic worker adjustment based on file sizes
    # Rule: Estimate ~1.5-2GB RAM per worker for large files
    if max_file_size > 500_000_000:  # 500MB files
        suggested_workers = min(args.num_workers, 3)
        if args.num_workers > suggested_workers:
            logger.warning(
                f"⚠️  Large files detected "
                f"(max: {max_file_size/1e6:.0f}MB)"
            )
            logger.warning(
                f"⚠️  Reducing workers from {args.num_workers} "
                f"to {suggested_workers} to prevent OOM"
            )
            args.num_workers = suggested_workers
    elif max_file_size > 300_000_000:  # 300MB files
        suggested_workers = min(args.num_workers, 4)
        if args.num_workers > suggested_workers:
            logger.warning(
                f"⚠️  Medium-large files detected "
                f"(max: {max_file_size/1e6:.0f}MB)"
            )
            logger.warning(
                f"⚠️  Reducing workers from {args.num_workers} "
                f"to {suggested_workers} to prevent OOM"
            )
            args.num_workers = suggested_workers
    
    # Sort files by size (process smaller files first)
    # This helps prevent memory spikes from processing large files
    laz_files_with_size.sort(key=lambda x: x[1])
    laz_files_sorted = [laz for laz, _ in laz_files_with_size]
    
    # Copy metadata files (JSON, stats, etc.) and preserve directory structure
    logger.info("Copying metadata files...")
    import shutil
    metadata_count = 0
    
    # Copy root-level metadata files
    for meta_file in input_path.glob("*.json"):
        dest = output_dir / meta_file.name
        shutil.copy2(meta_file, dest)
        metadata_count += 1
        logger.info(f"  Copied: {meta_file.name}")
    
    for meta_file in input_path.glob("*.txt"):
        dest = output_dir / meta_file.name
        shutil.copy2(meta_file, dest)
        metadata_count += 1
        logger.info(f"  Copied: {meta_file.name}")
    
    # For each LAZ file, copy its corresponding metadata
    for laz_file in laz_files_sorted:
        # Check for JSON metadata file
        json_file = laz_file.with_suffix('.json')
        if json_file.exists():
            # Preserve relative path structure
            rel_path = laz_file.relative_to(input_path)
            dest_json = output_dir / rel_path.with_suffix('.json')
            dest_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_file, dest_json)
            metadata_count += 1
    
    if metadata_count > 0:
        logger.info(f"✓ Copied {metadata_count} metadata files")
    
    logger.info("")
    
    # Get skip_existing setting (default: True)
    skip_existing = not getattr(args, 'force', False)
    
    # Prepare arguments for worker function with preserved paths
    worker_args = []
    for laz in laz_files_sorted:
        # Calculate relative path to preserve directory structure
        rel_path = laz.relative_to(input_path)
        output_path = output_dir / rel_path
        worker_args.append(
            (laz, output_path, args.k_neighbors, args.use_gpu,
             mode, skip_existing)
        )
    
    # Process files with better error handling and batching
    if args.num_workers > 1:
        # Process in batches to limit concurrent memory usage
        # For large files, limit concurrent tasks to prevent OOM
        # Very conservative batching for building mode (memory intensive)
        if mode == 'building':
            # Building mode needs SIGNIFICANTLY more memory (extra features + KDTrees)
            # Each worker can use 4-6GB RAM for large files
            if max_file_size > 300_000_000:
                # For very large files (>300MB), process 1 at a time
                batch_size = 1
                logger.warning(
                    "⚠️  Using sequential batching for large files "
                    "to prevent memory issues"
                )
            elif max_file_size > 200_000_000:
                # For large files (>200MB), limit concurrency
                batch_size = max(1, args.num_workers // 2)
            else:
                # Smaller files can use full worker count
                batch_size = args.num_workers
        else:
            # Core mode is less memory intensive
            if max_file_size < 200_000_000:
                batch_size = args.num_workers * 2
            else:
                batch_size = args.num_workers
        
        try:
            enriched_count = 0
            skipped_count = 0
            error_count = 0
            
            with ProcessPoolExecutor(
                max_workers=args.num_workers
            ) as executor:
                # Process in batches
                for i in range(0, len(worker_args), batch_size):
                    batch = worker_args[i:i+batch_size]
                    futures = {
                        executor.submit(_enrich_single_file, arg): arg[0]
                        for arg in batch
                    }
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result == 'success':
                                enriched_count += 1
                            elif result == 'skipped':
                                skipped_count += 1
                            elif result == 'error':
                                error_count += 1
                        except Exception as e:
                            laz_file = futures[future]
                            logger.error(
                                f"Failed to process {laz_file.name}: {e}"
                            )
                            error_count += 1
                    
                    # Force garbage collection between batches
                    import gc
                    gc.collect()
            
            if error_count > 0:
                logger.warning(
                    f"⚠️  {error_count} files failed to process"
                )
                
        except BrokenProcessPool as e:
            logger.error(f"❌ Process pool crashed: {e}")
            logger.error("This usually indicates out-of-memory issues")
            logger.error("Solutions:")
            logger.error(
                f"  1. Reduce --num-workers (current: {args.num_workers})"
            )
            logger.error("  2. Process files individually: --num-workers 1")
            logger.error("  3. Increase system RAM or add swap space")
            return 1
    else:
        # Sequential processing
        enriched_count = 0
        skipped_count = 0
        error_count = 0
        
        for arg in worker_args:
            result = _enrich_single_file(arg)
            if result == 'success':
                enriched_count += 1
            elif result == 'skipped':
                skipped_count += 1
            elif result == 'error':
                error_count += 1
    
    logger.info("")
    logger.info("="*70)
    logger.info("📊 Enrichment Summary:")
    logger.info(f"  Total files: {len(worker_args)}")
    logger.info(f"  ✅ Enriched: {enriched_count}")
    logger.info(f"  ⏭️  Skipped: {skipped_count}")
    if error_count > 0:
        logger.info(f"  ❌ Failed: {error_count}")
    logger.info("="*70)
    
    # Optional: Auto-convert to QGIS format if requested
    if auto_convert:
        logger.info("")
        logger.info("="*70)
        logger.info("🔄 Converting to QGIS-compatible format...")
        logger.info("="*70)
        
        from .qgis_converter import simplify_for_qgis
        
        # Find all enriched LAZ files
        enriched_files = list(output_dir.rglob("*.laz"))
        
        logger.info(f"Found {len(enriched_files)} enriched files to convert")
        
        qgis_success = 0
        for enriched_file in enriched_files:
            try:
                # Create QGIS version with _qgis suffix
                simplify_for_qgis(
                    enriched_file,
                    output_file=None,  # Auto-generate name
                    verbose=False
                )
                qgis_success += 1
            except Exception as e:
                logger.warning(
                    f"  ⚠️  Failed to convert {enriched_file.name}: {e}"
                )
        
        logger.info("")
        logger.info(f"✓ Created {qgis_success} QGIS-compatible versions")
        logger.info("="*70)
    
    return 0


def cmd_download(args):
    """Download LiDAR tiles from IGN."""
    # Parse bounding box
    try:
        bbox = [float(x) for x in args.bbox.split(',')]
        if len(bbox) != 4:
            logger.error("Bounding box must have 4 values: xmin,ymin,xmax,ymax")
            return 1
    except ValueError:
        logger.error("Bounding box values must be numeric")
        return 1
    
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloader = IGNLiDARDownloader(output_dir)
    tiles_data = downloader.fetch_available_tiles(tuple(bbox))
    
    features = tiles_data.get('features', [])
    if args.max_tiles:
        features = features[:args.max_tiles]
    
    logger.info(f"Found {len(features)} tiles to download")
    
    # Extract tile names
    tile_names = []
    for feature in features:
        props = feature.get('properties', {})
        tile_name = props.get('name', '')
        if tile_name:
            tile_names.append(tile_name)
    
    if not tile_names:
        logger.error("No valid tile names found")
        return 1
    
    # Batch download with metadata
    results = downloader.batch_download(
        tile_list=tile_names,
        bbox=tuple(bbox),
        save_metadata=True
    )
    
    success_count = sum(1 for s in results.values() if s)
    
    logger.info("")
    logger.info("="*70)
    logger.info(f"✓ Downloaded {success_count}/{len(tile_names)} tiles")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("="*70)
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IGN LiDAR HD processing toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # PROCESS command
    process_parser = subparsers.add_parser('process', help='Process LAZ files to create training patches')
    process_parser.add_argument('--input', type=Path, help='Input LAZ file')
    process_parser.add_argument('--input-dir', type=Path, help='Input directory of LAZ files')
    process_parser.add_argument('--output', type=Path, required=True, help='Output directory')
    process_parser.add_argument('--lod-level', type=str, choices=['LOD2', 'LOD3'],
                               default='LOD2', help='LOD level (default: LOD2)')
    process_parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    process_parser.add_argument('--num-augmentations', type=int, default=3,
                               help='Number of augmentations per patch (default: 3)')
    process_parser.add_argument('--num-workers', type=int, default=1,
                               help='Number of parallel workers (default: 1)')
    process_parser.add_argument('--bbox', type=str,
                               help='Bounding box filter: xmin,ymin,xmax,ymax (LAMB93 coordinates)')
    process_parser.add_argument('--patch-size', type=float, default=150.0,
                               help='Patch size in meters (default: 150.0)')
    process_parser.add_argument('--patch-overlap', type=float, default=0.1,
                               help='Patch overlap ratio (default: 0.1)')
    process_parser.add_argument('--num-points', type=int, default=16384,
                               choices=[4096, 8192, 16384],
                               help='Target number of points per patch: 4k, 8k, or 16k (default: 16384)')
    process_parser.add_argument('--k-neighbors', type=int, default=None,
                               help='Number of neighbors for feature computation (default: auto)')
    process_parser.add_argument('--include-architectural-style', action='store_true',
                               help='Include architectural style (0-12) as a feature in patches')
    process_parser.add_argument('--style-encoding', type=str, 
                               choices=['constant', 'multihot'],
                               default='constant',
                               help='Style encoding: constant (single ID) or multihot (multi-label with weights)')
    process_parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if patches already exist'
    )
    
    # ENRICH command
    enrich_parser = subparsers.add_parser('enrich', help='Enrich LAZ files with geometric features')
    enrich_parser.add_argument('--input', type=Path, help='Input LAZ file')
    enrich_parser.add_argument('--input-dir', type=Path, help='Input directory of LAZ files')
    enrich_parser.add_argument('--output', type=Path, required=True, help='Output directory')
    enrich_parser.add_argument('--num-workers', type=int, default=1,
                              help='Number of parallel workers (default: 1)')
    enrich_parser.add_argument('--k-neighbors', type=int, default=10,
                              help='Number of neighbors for feature computation (default: 10)')
    enrich_parser.add_argument('--use-gpu', action='store_true',
                              help='Use GPU acceleration if available')
    enrich_parser.add_argument('--mode', type=str, choices=['core', 'building'],
                              default='core',
                              help='Feature mode: core (basic) or building (full) (default: core)')
    enrich_parser.add_argument(
        '--auto-convert-qgis',
        action='store_true',
        help='Also create QGIS versions (LAZ 1.2) after enrichment'
    )
    enrich_parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-enrichment even if enriched file exists'
    )
    
    # DOWNLOAD command
    download_parser = subparsers.add_parser('download', help='Download LiDAR tiles from IGN')
    download_parser.add_argument('--bbox', type=str, required=True,
                                help='Bounding box: lon_min,lat_min,lon_max,lat_max (WGS84)')
    download_parser.add_argument('--output', type=Path, required=True, help='Output directory')
    download_parser.add_argument('--max-tiles', type=int, help='Maximum number of tiles to download')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Validate inputs for process and enrich commands
    if args.command in ['process', 'enrich']:
        if not args.input and not args.input_dir:
            parser.error(f"{args.command}: Either --input or --input-dir must be specified")
    
    # Execute command
    if args.command == 'process':
        return cmd_process(args)
    elif args.command == 'enrich':
        return cmd_enrich(args)
    elif args.command == 'download':
        return cmd_download(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


if __name__ == '__main__':
    main()
