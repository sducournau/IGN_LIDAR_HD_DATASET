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
                     mode, skip_existing, add_rgb, rgb_cache_dir, radius,
                     preprocess, preprocess_config, auto_params, augment,
                     num_augmentations, add_infrared, infrared_cache_dir)
    
    Returns:
        str: 'skipped' if file already exists and skip_existing=True,
             'success' if processing succeeded,
             'error' if processing failed
    """
    (laz_path, output_path, k_neighbors,
     use_gpu, mode, skip_existing, add_rgb, rgb_cache_dir, radius,
     preprocess, preprocess_config, auto_params, augment,
     num_augmentations, add_infrared, infrared_cache_dir) = args_tuple
    
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
    
    # Import feature computation function
    from .features import compute_all_features_with_gpu
    
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
        
        # Store intensity and return_number if available (needed for augmentation)
        intensity = np.array(las.intensity, dtype=np.float32) if hasattr(las, 'intensity') else None
        return_number = np.array(las.return_number, dtype=np.float32) if hasattr(las, 'return_number') else None
        
        # Auto-analyze tile if requested
        if auto_params:
            worker_logger.info("  📊 Analyzing tile for optimal parameters...")
            from .tile_analyzer import analyze_tile, format_analysis_report
            
            try:
                analysis = analyze_tile(laz_path)
                worker_logger.info(format_analysis_report(analysis))
                
                # Override parameters with analyzed values
                if radius is None:
                    radius = analysis['optimal_radius']
                    worker_logger.info(
                        f"  ✓ Using analyzed radius: {radius:.2f}m"
                    )
                
                # Update preprocessing config if enabled
                if preprocess and preprocess_config:
                    preprocess_config['sor']['k'] = analysis['sor_k']
                    preprocess_config['sor']['std_multiplier'] = (
                        analysis['sor_std']
                    )
                    preprocess_config['ror']['radius'] = analysis['ror_radius']
                    preprocess_config['ror']['min_neighbors'] = (
                        analysis['ror_neighbors']
                    )
                    worker_logger.info(
                        f"  ✓ Updated preprocessing: "
                        f"SOR(k={analysis['sor_k']}, "
                        f"std={analysis['sor_std']:.1f}), "
                        f"ROR(r={analysis['ror_radius']:.1f}m, "
                        f"min={analysis['ror_neighbors']})"
                    )
            except Exception as e:
                worker_logger.warning(
                    f"  ⚠️  Tile analysis failed: {e}, "
                    f"using default parameters"
                )
        
        # Apply preprocessing if enabled (artifact mitigation)
        preprocess_mask = None  # Track which points to keep
        if preprocess and preprocess_config:
            worker_logger.info("  🧹 Preprocessing (artifact mitigation)...")
            
            from .preprocessing import (
                statistical_outlier_removal,
                radius_outlier_removal,
                voxel_downsample
            )
            
            original_count = len(points)
            cumulative_mask = np.ones(original_count, dtype=bool)
            
            # Apply SOR
            sor_cfg = preprocess_config.get('sor', {})
            if sor_cfg.get('enable', True):
                _, sor_mask = statistical_outlier_removal(
                    points,
                    k=sor_cfg.get('k', 12),
                    std_multiplier=sor_cfg.get('std_multiplier', 2.0)
                )
                cumulative_mask &= sor_mask
            
            # Apply ROR
            ror_cfg = preprocess_config.get('ror', {})
            if ror_cfg.get('enable', True):
                _, ror_mask = radius_outlier_removal(
                    points,
                    radius=ror_cfg.get('radius', 1.0),
                    min_neighbors=ror_cfg.get('min_neighbors', 4)
                )
                cumulative_mask &= ror_mask
            
            # Store the mask for later use when creating output
            preprocess_mask = cumulative_mask.copy()
            
            # Filter points and classification
            points = points[cumulative_mask]
            classification = classification[cumulative_mask]
            
            # Apply voxel downsampling if enabled
            voxel_cfg = preprocess_config.get('voxel', {})
            if voxel_cfg.get('enable', False):
                points, voxel_indices = voxel_downsample(
                    points,
                    voxel_size=voxel_cfg.get('voxel_size', 0.5),
                    method=voxel_cfg.get('method', 'centroid')
                )
                classification = classification[voxel_indices]
                # Update mask to reflect voxel downsampling
                indices_after_sor_ror = np.where(cumulative_mask)[0]
                final_indices = indices_after_sor_ror[voxel_indices]
                preprocess_mask = np.zeros(original_count, dtype=bool)
                preprocess_mask[final_indices] = True
            
            final_count = len(points)
            reduction = 1 - final_count / original_count
            worker_logger.info(
                f"  ✓ Preprocessing: {final_count:,}/{original_count:,} "
                f"({reduction:.1%} reduction)"
            )
            
            # Update intensity and return_number with preprocessing mask
            if intensity is not None:
                intensity = intensity[cumulative_mask]
                if voxel_cfg.get('enable', False):
                    intensity = intensity[voxel_indices]
            if return_number is not None:
                return_number = return_number[cumulative_mask]
                if voxel_cfg.get('enable', False):
                    return_number = return_number[voxel_indices]
        
        # Apply data augmentation if requested
        # This creates augmented versions BEFORE feature computation
        # to ensure features are computed on augmented geometry
        versions_to_process = []
        if augment and num_augmentations > 0:
            worker_logger.info(
                f"  🔄 Generating {num_augmentations} augmented versions..."
            )
            from .utils import augment_raw_points
            
            # Version 0: Original (no augmentation)
            versions_to_process.append({
                'suffix': '',
                'points': points,
                'classification': classification,
                'intensity': intensity,
                'return_number': return_number
            })
            
            # Versions 1 to N: Augmented
            for aug_idx in range(num_augmentations):
                # Create fallback arrays if intensity/return_number not available
                intensity_for_aug = (
                    intensity if intensity is not None
                    else np.ones(len(points), dtype=np.float32)
                )
                return_number_for_aug = (
                    return_number if return_number is not None
                    else np.ones(len(points), dtype=np.float32)
                )
                
                # Apply augmentation
                (aug_points, aug_intensity,
                 aug_return_number, aug_classification) = augment_raw_points(
                    points, intensity_for_aug,
                    return_number_for_aug, classification
                )
                
                versions_to_process.append({
                    'suffix': f'_aug{aug_idx + 1}',
                    'points': aug_points,
                    'classification': aug_classification,
                    'intensity': (
                        aug_intensity if intensity is not None else None
                    ),
                    'return_number': (
                        aug_return_number if return_number is not None else None
                    )
                })
            
            worker_logger.info(
                f"  ✓ Created {len(versions_to_process)} versions "
                f"(1 original + {num_augmentations} augmented)"
            )
        else:
            # No augmentation - process only original
            versions_to_process.append({
                'suffix': '',
                'points': points,
                'classification': classification,
                'intensity': intensity,
                'return_number': return_number
            })
        
        # Process each version (original + augmented)
        for version_idx, version_data in enumerate(versions_to_process):
            points_ver = version_data['points']
            classification_ver = version_data['classification']
            intensity_ver = version_data['intensity']
            return_number_ver = version_data['return_number']
            suffix = version_data['suffix']
            
            # Create versioned output path
            if suffix:
                output_path_ver = output_path.parent / (
                    output_path.stem + suffix + output_path.suffix
                )
            else:
                output_path_ver = output_path
            
            # Skip if already exists
            if skip_existing and output_path_ver.exists():
                worker_logger.info(
                    f"  ⏭️  Version {version_idx + 1} already exists, "
                    f"skipping"
                )
                continue
            
            if suffix:
                worker_logger.info(
                    f"  Processing version "
                    f"{version_idx + 1}/{len(versions_to_process)} "
                    f"{suffix}..."
                )
        
            # Memory check - abort if insufficient memory available
            n_points = len(points_ver)
            if PSUTIL_AVAILABLE:
                try:
                    mem = psutil.virtual_memory()
                    mem_available = mem.available
                    available_mb = mem_available / 1024 / 1024
                    
                    # Estimate required memory based on mode
                    # Building mode needs significantly more memory due to:
                    # - Main features: ~40 bytes/point
                    # - KDTree for main features: ~24 bytes/point
                    # - Building features: additional KDTree + features
                    # Total for full mode: ~120-150 bytes/point (conservative)
                    # Core mode: ~70 bytes/point
                    if mode == 'full':
                        bytes_per_point = 150
                    else:
                        bytes_per_point = 70
                    
                    estimated_needed_mb = (
                        (n_points * bytes_per_point) / 1024 / 1024
                    )
                    
                    # Check if we have enough memory (60% safety margin)
                    safety_factor = 0.6
                    if estimated_needed_mb > available_mb * safety_factor:
                        worker_logger.error(
                            f"  ✗ Insufficient memory for {laz_path.name}"
                        )
                        worker_logger.error(
                            f"    Need ~{estimated_needed_mb:.0f}MB, "
                            f"only {available_mb:.0f}MB available "
                            f"(requires {safety_factor*100:.0f}% "
                            f"safety margin)"
                        )
                        worker_logger.error(
                            "    Reduce --num-workers or process alone"
                        )
                        return 'error'
                    
                    # Warn about swap usage
                    swap_percent = psutil.swap_memory().percent
                    if swap_percent > 50:
                        worker_logger.warning(
                            f"  ⚠️  High swap usage ({swap_percent:.0f}%)"
                        )
                        worker_logger.warning(
                            "    System may be under memory pressure"
                        )
                        
                except (AttributeError, NameError):
                    # psutil error - continue anyway
                    pass
            
            # INTELLIGENT AUTO-SCALING: Use memory manager if available
            use_intelligent_chunking = False
            try:
                from .memory_manager import AdaptiveMemoryManager
                mem_mgr = AdaptiveMemoryManager(
                    enable_gpu=use_gpu
                )
                
                # Calculate optimal chunk size based on available resources
                chunk_size = mem_mgr.calculate_optimal_chunk_size(
                    num_points=n_points,
                    mode=mode,
                    num_augmentations=num_augmentations if augment else 0
                )
                
                use_intelligent_chunking = True
                if version_idx == 0:
                    worker_logger.info(
                        f"  🎯 Intelligent chunking: "
                        f"{chunk_size:,} points per chunk"
                    )
            except (ImportError, Exception):
                # Fallback to fixed logic
                use_intelligent_chunking = False
            
            # Fallback: Fixed chunk sizes
            if not use_intelligent_chunking:
                if augment and num_augmentations > 0:
                    # With augmentation: smaller chunks (multiple versions)
                    # OPTIMIZED: Reduced for better GPU stability
                    if n_points > 20_000_000:
                        chunk_size = 2_000_000  # 2M chunks (was 3M)
                        if version_idx == 0:
                            worker_logger.info(
                                "  Chunked (2M per chunk, augmented)"
                            )
                    elif n_points > 10_000_000:
                        chunk_size = 2_500_000  # 2.5M chunks (was 5M)
                        if version_idx == 0:
                            worker_logger.info(
                                "  Chunked (2.5M per chunk, augmented)"
                            )
                    else:
                        chunk_size = 3_000_000  # 3M chunks (was 8M)
                        if version_idx == 0:
                            worker_logger.info(
                                "  Chunked (3M per chunk, augmented)"
                            )
                else:
                    # Without augmentation: optimized chunking
                    # Smaller chunks = faster KDTree building
                    # OPTIMIZED: Reduced chunk sizes for better GPU
                    if n_points > 40_000_000:
                        # Very large (>40M): 2M chunks
                        chunk_size = 2_000_000
                        if version_idx == 0:
                            worker_logger.info(
                                "  Chunked processing (2M per chunk)"
                            )
                    elif n_points > 20_000_000:
                        # Large (20-40M): 2.5M chunks
                        chunk_size = 2_500_000
                        if version_idx == 0:
                            worker_logger.info(
                                "  Chunked processing (2.5M per chunk)"
                            )
                    elif n_points > 10_000_000:
                        # Medium (10-20M): 2.5M chunks
                        chunk_size = 2_500_000
                        if version_idx == 0:
                            worker_logger.info(
                                "  Chunked processing (2.5M per chunk)"
                            )
                    else:
                        # Small (<10M): 3M chunks
                        chunk_size = (
                            3_000_000 if n_points > 5_000_000 else None
                        )
            
            # Compute features based on mode
            if mode == 'full':
                if version_idx == 0 or suffix:
                    worker_logger.info(
                        f"  Computing FULL features for "
                        f"{len(points_ver):,} points..."
                    )
            else:
                if version_idx == 0 or suffix:
                    worker_logger.info(
                        f"  Computing CORE features for "
                        f"{len(points_ver):,} points..."
                    )
            
            # Compute features with optional GPU acceleration
            # v1.7.0: GPU now supports chunked processing!
            if chunk_size is None:
                # Use GPU-enabled function (no chunking needed)
                (normals, curvature,
                 height_above_ground, geometric_features) = \
                    compute_all_features_with_gpu(
                        points_ver, classification_ver,
                        k=k_neighbors,
                        auto_k=False,
                        use_gpu=use_gpu,
                        radius=radius
                    )
            else:
                # Chunked processing - GPU or CPU
                if use_gpu:
                    # v1.7.0: Use new GPU chunked implementation
                    if version_idx == 0:
                        worker_logger.info(
                            "  Using GPU acceleration with chunking"
                        )
                    try:
                        from .features_gpu_chunked import (
                            compute_all_features_gpu_chunked
                        )
                        (normals, curvature,
                         height_above_ground, geometric_features) = \
                            compute_all_features_gpu_chunked(
                                points_ver, classification_ver,
                                k=k_neighbors,
                                chunk_size=chunk_size,
                                radius=radius
                            )
                    except Exception as e:
                        worker_logger.warning(
                            f"  GPU chunking failed ({e}), using CPU"
                        )
                        from .features import compute_all_features_optimized
                        (normals, curvature,
                         height_above_ground, geometric_features) = \
                            compute_all_features_optimized(
                                points_ver, classification_ver,
                                k=k_neighbors,
                                include_extra=False,
                                chunk_size=chunk_size,
                                radius=radius
                            )
                else:
                    # CPU chunked processing
                    from .features import compute_all_features_optimized
                    (normals, curvature,
                     height_above_ground, geometric_features) = \
                        compute_all_features_optimized(
                            points_ver, classification_ver,
                            k=k_neighbors,
                            include_extra=False,
                            chunk_size=chunk_size,
                            radius=radius
                        )
            
            # Ensure output directory exists
            output_path_ver.parent.mkdir(parents=True, exist_ok=True)
        
            # Create output LAZ with versioned geometry
            from laspy import LasHeader
            
            # Determine target format (RGB support if needed)
            target_format = las.header.point_format.id
            if add_rgb and target_format == 6:
                target_format = 7
            
            # Create new LAS data with appropriate header
            new_header = LasHeader(
                version=las.header.version,
                point_format=target_format
            )
            new_header.scales = las.header.scales
            new_header.offsets = las.header.offsets
            las_out = laspy.LasData(new_header)
            
            # Set coordinates and classification from versioned data
            las_out.x = points_ver[:, 0]
            las_out.y = points_ver[:, 1]
            las_out.z = points_ver[:, 2]
            las_out.classification = classification_ver.astype(np.uint8)
            
            # Copy other standard fields if available
            # Ensure correct data types for LAZ encoding
            if intensity_ver is not None:
                las_out.intensity = intensity_ver.astype(np.uint16)
            if return_number_ver is not None:
                las_out.return_number = return_number_ver.astype(np.uint8)
            
            # Add computed features as extra dimensions
            for i, dim in enumerate(['normal_x', 'normal_y', 'normal_z']):
                las_out.add_extra_dim(
                    laspy.ExtraBytesParams(name=dim, type=np.float32)
                )
                setattr(las_out, dim, normals[:, i])
            
            las_out.add_extra_dim(
                laspy.ExtraBytesParams(name='curvature', type=np.float32)
            )
            las_out.curvature = curvature
            
            las_out.add_extra_dim(
                laspy.ExtraBytesParams(
                    name='height_above_ground', type=np.float32
                )
            )
            las_out.height_above_ground = height_above_ground
            
            for key, values in geometric_features.items():
                las_out.add_extra_dim(
                    laspy.ExtraBytesParams(name=key, type=np.float32)
                )
                setattr(las_out, key, values)
            
            # Add extra building-specific features if in full mode
            if mode == 'full':
                try:
                    # Import building-specific feature functions
                    from .features import (
                        compute_verticality, compute_wall_score,
                        compute_roof_score, compute_num_points_in_radius
                    )
                    
                    if version_idx == 0 or suffix:
                        worker_logger.info(
                            "  Computing additional features..."
                        )
                    
                    # Verticality
                    verticality = compute_verticality(normals)
                    las_out.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name='verticality', type=np.float32
                        )
                    )
                    las_out.verticality = verticality
                    
                    # Wall score
                    wall_score = compute_wall_score(
                        normals, height_above_ground
                    )
                    las_out.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name='wall_score', type=np.float32
                        )
                    )
                    las_out.wall_score = wall_score
                    
                    # Roof score
                    roof_score = compute_roof_score(
                        normals, height_above_ground, curvature
                    )
                    las_out.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name='roof_score', type=np.float32
                        )
                    )
                    las_out.roof_score = roof_score
                    
                    # Num points in radius (memory-optimized)
                    if n_points > 5_000_000:
                        radius_chunk_size = 500_000
                    elif n_points > 3_000_000:
                        radius_chunk_size = 750_000
                    else:
                        radius_chunk_size = 1_000_000
                    
                    num_points_rad = compute_num_points_in_radius(
                        points_ver, radius=2.0,
                        chunk_size=radius_chunk_size
                    )
                    las_out.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name='num_points_2m', type=np.float32
                        )
                    )
                    las_out.num_points_2m = num_points_rad
                    
                except Exception as e:
                    worker_logger.warning(
                        f"  Could not compute building features: {e}"
                    )
            
            # Add RGB augmentation if requested
            if add_rgb:
                try:
                    from .rgb_augmentation import IGNOrthophotoFetcher
                    
                    if version_idx == 0 or suffix:
                        worker_logger.info(
                            "  Fetching RGB from IGN orthophotos..."
                        )
                    
                    # Compute bbox from versioned points
                    bbox = (
                        points_ver[:, 0].min(),
                        points_ver[:, 1].min(),
                        points_ver[:, 0].max(),
                        points_ver[:, 1].max()
                    )
                    
                    # Create fetcher with cache if specified
                    fetcher = IGNOrthophotoFetcher(cache_dir=rgb_cache_dir)
                    
                    # Fetch RGB colors
                    rgb = fetcher.augment_points_with_rgb(
                        points_ver, bbox=bbox
                    )
                    
                    # Add RGB to LAZ
                    if las_out.header.point_format.id in [
                        2, 3, 5, 6, 7, 8, 10
                    ]:
                        # Native RGB support - scale to 16-bit
                        las_out.red = rgb[:, 0].astype(np.uint16) * 257
                        las_out.green = rgb[:, 1].astype(np.uint16) * 257
                        las_out.blue = rgb[:, 2].astype(np.uint16) * 257
                    else:
                        # Add as extra dimensions
                        las_out.add_extra_dim(
                            laspy.ExtraBytesParams(
                                name='red', type=np.uint8
                            )
                        )
                        las_out.add_extra_dim(
                            laspy.ExtraBytesParams(
                                name='green', type=np.uint8
                            )
                        )
                        las_out.add_extra_dim(
                            laspy.ExtraBytesParams(
                                name='blue', type=np.uint8
                            )
                        )
                        las_out.red = rgb[:, 0]
                        las_out.green = rgb[:, 1]
                        las_out.blue = rgb[:, 2]
                    
                    if version_idx == 0 or suffix:
                        worker_logger.info(
                            f"  ✓ Added RGB to {len(points_ver):,} points"
                        )
                    
                except ImportError as ie:
                    worker_logger.warning(
                        f"  ⚠️  RGB requires 'requests' and 'Pillow': {ie}"
                    )
                except Exception as e:
                    worker_logger.warning(
                        f"  ⚠️  Could not add RGB colors: {e}"
                    )
            
            # Add Infrared augmentation if requested
            if add_infrared:
                try:
                    from .infrared_augmentation import IGNInfraredFetcher
                    
                    if version_idx == 0 or suffix:
                        worker_logger.info(
                            "  Fetching infrared from IGN orthophotos..."
                        )
                    
                    # Compute bbox from versioned points
                    bbox = (
                        points_ver[:, 0].min(),
                        points_ver[:, 1].min(),
                        points_ver[:, 0].max(),
                        points_ver[:, 1].max()
                    )
                    
                    # Create fetcher with cache if specified
                    nir_fetcher = IGNInfraredFetcher(
                        cache_dir=infrared_cache_dir
                    )
                    
                    # Fetch infrared values
                    nir = nir_fetcher.augment_points_with_infrared(
                        points_ver, bbox=bbox
                    )
                    
                    # Add infrared to LAZ as extra dimension
                    las_out.add_extra_dim(
                        laspy.ExtraBytesParams(name='nir', type=np.uint8)
                    )
                    las_out.nir = nir
                    
                    if version_idx == 0 or suffix:
                        worker_logger.info(
                            f"  ✓ Added infrared to {len(points_ver):,} "
                            f"points"
                        )
                    
                except ImportError as ie:
                    worker_logger.warning(
                        f"  ⚠️  Infrared requires 'requests' and "
                        f"'Pillow': {ie}"
                    )
                except Exception as e:
                    worker_logger.warning(
                        f"  ⚠️  Could not add infrared values: {e}"
                    )
            
            # Save with LAZ compression
            las_out.write(output_path_ver, do_compress=True)
            if suffix:
                worker_logger.info(
                    f"  ✓ Saved{suffix} to {output_path_ver.name}"
                )
            
            # Cleanup this version's data
            del las_out, normals, curvature
            del height_above_ground, geometric_features
            del points_ver, classification_ver
            
            # Force garbage collection to free memory for next version
            gc.collect()
        
        # All versions processed successfully
        worker_logger.info(f"  ✓ Completed {laz_path.name}")
        if augment and num_augmentations > 0:
            worker_logger.info(
                f"  Created {len(versions_to_process)} total files"
            )
        
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
    
    # Filter by specific files if provided
    if hasattr(args, 'files') and args.files:
        from pathlib import Path
        # Convert file patterns to Path objects relative to input_path
        file_patterns = [Path(f) for f in args.files]
        filtered_files = []
        
        for pattern in file_patterns:
            # Check if pattern is relative or absolute
            if pattern.is_absolute():
                # Absolute path - use directly if it exists
                if pattern.exists() and pattern.suffix == '.laz':
                    filtered_files.append(pattern)
            else:
                # Relative path - resolve against input_path
                full_path = input_path / pattern
                if full_path.exists() and full_path.suffix == '.laz':
                    filtered_files.append(full_path)
                else:
                    # Try matching against collected laz_files
                    for laz in laz_files:
                        if laz.name == pattern.name or \
                           laz.relative_to(input_path) == pattern:
                            filtered_files.append(laz)
                            break
        
        if filtered_files:
            laz_files = filtered_files
            logger.info(f"Filtering to {len(laz_files)} specified file(s)")
        else:
            logger.error(f"None of the specified files found in {input_path}")
            logger.error(f"Specified: {args.files}")
            return 1
    
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
    
    # v1.7.0: Use adaptive memory manager for intelligent configuration
    try:
        from .memory_manager import AdaptiveMemoryManager
        memory_manager = AdaptiveMemoryManager(
            enable_gpu=getattr(args, 'use_gpu', False)
        )
        
        # Get file sizes
        laz_files_with_size = [(laz, laz.stat().st_size) for laz in laz_files]
        file_sizes_mb = [size / (1024**2) for _, size in laz_files_with_size]
        
        # Use adaptive memory manager for worker optimization
        optimal_workers = memory_manager.calculate_optimal_workers(
            num_files=len(laz_files),
            file_sizes_mb=file_sizes_mb,
            mode=mode
        )
        
        if optimal_workers < args.num_workers:
            logger.warning(
                f"⚠️  Adaptive memory manager recommends "
                f"{optimal_workers} workers (requested: {args.num_workers})"
            )
            args.num_workers = optimal_workers
        
    except ImportError:
        # Fallback to old method if new module not available
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
        # Full mode needs ~4-6GB per worker, core mode ~2-3GB
        min_gb_per_worker = 5.0 if mode == 'full' else 2.5
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
    
    # Only copy root-level metadata files if input_path is a directory
    if input_path.is_dir():
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
            if input_path.is_file():
                # Single file input: copy metadata to output directory root
                dest_json = output_dir / json_file.name
            else:
                # Directory input: preserve relative path structure
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
    
    # Get RGB augmentation settings
    add_rgb = getattr(args, 'add_rgb', False)
    rgb_cache_dir = getattr(args, 'rgb_cache_dir', None)
    
    # Log RGB settings
    if add_rgb:
        logger.info("RGB augmentation: ENABLED")
        if rgb_cache_dir:
            logger.info(f"RGB cache directory: {rgb_cache_dir}")
            rgb_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info("RGB cache: Disabled (orthophotos will be fetched each time)")
    
    # Get Infrared augmentation settings
    add_infrared = getattr(args, 'add_infrared', False)
    infrared_cache_dir = getattr(args, 'infrared_cache_dir', None)
    
    # Log Infrared settings
    if add_infrared:
        logger.info("Infrared augmentation: ENABLED")
        if infrared_cache_dir:
            logger.info(f"Infrared cache directory: {infrared_cache_dir}")
            infrared_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info("Infrared cache: Disabled (orthophotos will be fetched each time)")
    
    # Get preprocessing settings
    preprocess = (getattr(args, 'preprocess', False) and
                  not getattr(args, 'no_preprocess', False))
    preprocess_config = None
    if preprocess:
        logger.info("Preprocessing: ENABLED (artifact mitigation)")
        preprocess_config = {
            'sor': {
                'enable': True,
                'k': getattr(args, 'sor_k', 12),
                'std_multiplier': getattr(args, 'sor_std', 2.0)
            },
            'ror': {
                'enable': True,
                'radius': getattr(args, 'ror_radius', 1.0),
                'min_neighbors': getattr(args, 'ror_neighbors', 4)
            },
            'voxel': {
                'enable': args.voxel_size is not None,
                'voxel_size': getattr(args, 'voxel_size', 0.5),
                'method': 'centroid'
            }
        }
        logger.info(f"  SOR: k={preprocess_config['sor']['k']}, std={preprocess_config['sor']['std_multiplier']}")
        logger.info(f"  ROR: radius={preprocess_config['ror']['radius']}m, min_neighbors={preprocess_config['ror']['min_neighbors']}")
        if preprocess_config['voxel']['enable']:
            logger.info(f"  Voxel: size={preprocess_config['voxel']['voxel_size']}m")
    
    # Get auto-params setting
    auto_params = getattr(args, 'auto_params', False)
    if auto_params:
        logger.info("Auto-parameter analysis: ENABLED")
        logger.info("  Each tile will be analyzed for optimal parameters")
    
    # Get augmentation settings
    augment = getattr(args, 'augment', True)
    num_augmentations = getattr(args, 'num_augmentations', 3)
    if augment:
        logger.info(f"Data augmentation: ENABLED ({num_augmentations} versions)")
        logger.info("  Augmented versions will be created before feature computation")
    
    # Prepare arguments for worker function with preserved paths
    worker_args = []
    radius = getattr(args, 'radius', None)  # Get radius if available
    for laz in laz_files_sorted:
        # Calculate relative path to preserve directory structure
        rel_path = laz.relative_to(input_path)
        output_path = output_dir / rel_path
        worker_args.append(
            (laz, output_path, args.k_neighbors, args.use_gpu,
             mode, skip_existing, add_rgb, rgb_cache_dir, radius,
             preprocess, preprocess_config, auto_params, augment,
             num_augmentations, add_infrared, infrared_cache_dir)
        )
    
    # Process files with better error handling and batching
    if args.num_workers > 1:
        # Process in batches to limit concurrent memory usage
        # For large files, limit concurrent tasks to prevent OOM
        # Very conservative batching for full mode (memory intensive)
        if mode == 'full':
            # Full mode needs SIGNIFICANTLY more memory (extra features + KDTrees)
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


def cmd_pipeline(args):
    """Execute full pipeline from YAML configuration."""
    from .pipeline_config import PipelineConfig
    import time
    
    # Load configuration
    try:
        config = PipelineConfig(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    logger.info("="*70)
    logger.info("🚀 Starting Pipeline Execution")
    logger.info("="*70)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Stages: {config}")
    logger.info("")
    
    start_time = time.time()
    global_config = config.get_global_config()
    
    # Stage 1: Download
    if config.has_download:
        logger.info("📥 Stage 1: Download")
        logger.info("-" * 70)
        
        download_cfg = config.get_download_config()
        
        # Parse bounding box
        try:
            bbox = [float(x) for x in download_cfg['bbox'].split(',')]
            if len(bbox) != 4:
                logger.error("Bounding box must have 4 values")
                return 1
        except Exception as e:
            logger.error(f"Invalid bbox in config: {e}")
            return 1
        
        output_dir = Path(download_cfg['output'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_workers = download_cfg.get('num_workers', 
                                       global_config.get('num_workers', 3))
        max_tiles = download_cfg.get('max_tiles', None)
        
        downloader = IGNLiDARDownloader(output_dir, max_concurrent=num_workers)
        tiles_data = downloader.fetch_available_tiles(tuple(bbox))
        
        features = tiles_data.get('features', [])
        if max_tiles:
            features = features[:max_tiles]
        
        logger.info(f"Found {len(features)} tiles to download")
        
        tile_names = []
        for feature in features:
            props = feature.get('properties', {})
            tile_name = props.get('name', '')
            if tile_name:
                tile_names.append(tile_name)
        
        if not tile_names:
            logger.error("No valid tile names found")
            return 1
        
        results = downloader.batch_download(
            tile_list=tile_names,
            bbox=tuple(bbox),
            save_metadata=True,
            num_workers=num_workers
        )
        
        success_count = sum(1 for s in results.values() if s)
        logger.info(f"✓ Downloaded {success_count}/{len(tile_names)} tiles")
        logger.info("")
    
    # Stage 2: Enrich
    if config.has_enrich:
        logger.info("⚙️  Stage 2: Enrich")
        logger.info("-" * 70)
        
        enrich_cfg = config.get_enrich_config()
        
        # Create args object for cmd_enrich
        class EnrichArgs:
            pass
        
        eargs = EnrichArgs()
        eargs.input = None
        eargs.input_dir = Path(enrich_cfg['input_dir'])
        eargs.output = Path(enrich_cfg['output'])
        eargs.num_workers = enrich_cfg.get('num_workers',
                                           global_config.get('num_workers', 1))
        eargs.k_neighbors = enrich_cfg.get('k_neighbors', 10)
        eargs.radius = enrich_cfg.get('radius', None)
        eargs.use_gpu = enrich_cfg.get('use_gpu', False)
        eargs.mode = enrich_cfg.get('mode', 'core')
        eargs.auto_convert_qgis = enrich_cfg.get('auto_convert_qgis', False)
        eargs.force = enrich_cfg.get('force', False)
        eargs.add_rgb = enrich_cfg.get('add_rgb', False)
        eargs.rgb_cache_dir = Path(enrich_cfg['rgb_cache_dir']) \
            if 'rgb_cache_dir' in enrich_cfg else None
        eargs.add_infrared = enrich_cfg.get('add_infrared', False)
        eargs.infrared_cache_dir = Path(enrich_cfg['infrared_cache_dir']) \
            if 'infrared_cache_dir' in enrich_cfg else None
        
        result = cmd_enrich(eargs)
        if result != 0:
            logger.error("Enrich stage failed")
            return result
        logger.info("")
    
    # Stage 3: Patch
    if config.has_patch:
        logger.info("📦 Stage 3: Patch")
        logger.info("-" * 70)
        
        patch_cfg = config.get_patch_config()
        
        # Create args object for cmd_process
        class PatchArgs:
            pass
        
        pargs = PatchArgs()
        pargs.input = None
        pargs.input_dir = Path(patch_cfg['input_dir'])
        pargs.output = Path(patch_cfg['output'])
        pargs.lod_level = patch_cfg.get('lod_level', 'LOD2')
        pargs.num_workers = patch_cfg.get('num_workers',
                                          global_config.get('num_workers', 1))
        pargs.bbox = patch_cfg.get('bbox', None)
        pargs.patch_size = patch_cfg.get('patch_size', 150.0)
        pargs.patch_overlap = patch_cfg.get('patch_overlap', 0.1)
        pargs.num_points = patch_cfg.get('num_points', 16384)
        pargs.k_neighbors = patch_cfg.get('k_neighbors', None)
        pargs.include_architectural_style = patch_cfg.get(
            'include_architectural_style', False)
        pargs.style_encoding = patch_cfg.get('style_encoding', 'constant')
        pargs.force = patch_cfg.get('force', False)
        
        result = cmd_process(pargs)
        if result != 0:
            logger.error("Patch stage failed")
            return result
        logger.info("")
    
    # Summary
    elapsed_time = time.time() - start_time
    
    logger.info("="*70)
    logger.info("✅ Pipeline Complete!")
    logger.info("="*70)
    logger.info(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    
    if config.has_download:
        download_cfg = config.get_download_config()
        logger.info(f"  Downloaded: {download_cfg['output']}")
    if config.has_enrich:
        enrich_cfg = config.get_enrich_config()
        logger.info(f"  Enriched: {enrich_cfg['output']}")
    if config.has_patch:
        patch_cfg = config.get_patch_config()
        logger.info(f"  Patches: {patch_cfg['output']}")
    
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
    
    # Get num_workers (default 3 if not specified)
    num_workers = getattr(args, 'num_workers', 3)
    
    downloader = IGNLiDARDownloader(output_dir, max_concurrent=num_workers)
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
        save_metadata=True,
        num_workers=num_workers
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
    
    # PATCH command (renamed from process)
    patch_parser = subparsers.add_parser('patch', help='Create training patches from LAZ files')
    patch_parser.add_argument('--input', type=Path, help='Input LAZ file')
    patch_parser.add_argument('--input-dir', type=Path, help='Input directory of LAZ files')
    patch_parser.add_argument('--output', type=Path, required=True, help='Output directory')
    patch_parser.add_argument('--lod-level', type=str, choices=['LOD2', 'LOD3'],
                               default='LOD2', help='LOD level (default: LOD2)')
    patch_parser.add_argument('--num-workers', type=int, default=1,
                               help='Number of parallel workers (default: 1)')
    patch_parser.add_argument('--bbox', type=str,
                               help='Bounding box filter: xmin,ymin,xmax,ymax (LAMB93 coordinates)')
    patch_parser.add_argument('--patch-size', type=float, default=150.0,
                               help='Patch size in meters (default: 150.0)')
    patch_parser.add_argument('--patch-overlap', type=float, default=0.1,
                               help='Patch overlap ratio (default: 0.1)')
    patch_parser.add_argument('--num-points', type=int, default=16384,
                               choices=[4096, 8192, 16384],
                               help='Target number of points per patch: 4k, 8k, or 16k (default: 16384)')
    patch_parser.add_argument('--k-neighbors', type=int, default=None,
                               help='Number of neighbors for feature computation (default: auto)')
    patch_parser.add_argument('--include-architectural-style', action='store_true',
                               help='Include architectural style (0-12) as a feature in patches')
    patch_parser.add_argument('--style-encoding', type=str, 
                               choices=['constant', 'multihot'],
                               default='constant',
                               help='Style encoding: constant (single ID) or multihot (multi-label with weights)')
    patch_parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if patches already exist'
    )
    
    # Keep PROCESS as alias for backwards compatibility
    process_parser = subparsers.add_parser('process', help='Alias for "patch" command (deprecated)')
    process_parser.add_argument('--input', type=Path, help='Input LAZ file')
    process_parser.add_argument('--input-dir', type=Path, help='Input directory of LAZ files')
    process_parser.add_argument('--output', type=Path, required=True, help='Output directory')
    process_parser.add_argument('--lod-level', type=str, choices=['LOD2', 'LOD3'],
                               default='LOD2', help='LOD level (default: LOD2)')
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
    enrich_parser.add_argument('files', nargs='*', help='Optional: specific file(s) to process (relative to input-dir)')
    enrich_parser.add_argument('--num-workers', type=int, default=1,
                              help='Number of parallel workers (default: 1)')
    enrich_parser.add_argument('--k-neighbors', type=int, default=30,
                              help='Number of neighbors for feature computation (default: 30)')
    enrich_parser.add_argument('--radius', type=float, default=None,
                              help='Search radius in meters for geometric features (default: auto-estimate). '
                                   'Radius-based search eliminates LIDAR scan line artifacts. '
                                   'Typical values: 0.5-2.0m. Larger radius = smoother features.')
    # TODO: GPU integration - currently non-functional, needs connection to features_gpu.py
    # See GPU_ANALYSIS.md for implementation details
    enrich_parser.add_argument('--use-gpu', action='store_true',
                              help='[Non-functional in v1.2.0] Use GPU acceleration if available')
    enrich_parser.add_argument('--mode', type=str, choices=['core', 'full'],
                              default='core',
                              help='Feature mode: core (basic) or full (all features) (default: core)')
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
    enrich_parser.add_argument(
        '--add-rgb',
        action='store_true',
        help='Augment with RGB colors from IGN orthophotos (requires requests and Pillow)'
    )
    enrich_parser.add_argument(
        '--rgb-cache-dir',
        type=Path,
        help='Directory to cache downloaded orthophotos (optional)'
    )
    enrich_parser.add_argument(
        '--add-infrared',
        action='store_true',
        help='Augment with infrared (NIR) values from IGN orthophotos (requires requests and Pillow)'
    )
    enrich_parser.add_argument(
        '--infrared-cache-dir',
        type=Path,
        help='Directory to cache downloaded infrared orthophotos (optional)'
    )
    enrich_parser.add_argument(
        '--auto-params',
        action='store_true',
        help='Automatically analyze each tile and determine optimal parameters'
    )
    enrich_parser.add_argument(
        '--preprocess',
        action='store_true',
        default=False,
        help='Apply preprocessing to reduce artifacts (default: False)'
    )
    enrich_parser.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Explicitly disable preprocessing (overrides --preprocess)'
    )
    enrich_parser.add_argument(
        '--sor-k',
        type=int,
        default=15,
        help='Statistical Outlier Removal: number of neighbors (default: 15)'
    )
    enrich_parser.add_argument(
        '--sor-std',
        type=float,
        default=2.5,
        help='Statistical Outlier Removal: std multiplier (default: 2.5)'
    )
    enrich_parser.add_argument(
        '--ror-radius',
        type=float,
        default=0.8,
        help='Radius Outlier Removal: search radius in meters (default: 0.8)'
    )
    enrich_parser.add_argument(
        '--ror-neighbors',
        type=int,
        default=5,
        help='Radius Outlier Removal: min neighbors (default: 5)'
    )
    enrich_parser.add_argument(
        '--voxel-size',
        type=float,
        default=None,
        help='Voxel downsampling size in meters (optional, e.g., 0.5)'
    )
    enrich_parser.add_argument(
        '--augment',
        action='store_true',
        dest='augment',
        default=False,
        help='Enable geometric data augmentation (default: disabled)'
    )
    enrich_parser.add_argument(
        '--no-augment',
        action='store_false',
        dest='augment',
        help='Disable geometric data augmentation'
    )
    enrich_parser.add_argument(
        '--num-augmentations',
        type=int,
        default=3,
        help='Number of augmented versions per tile (default: 3)'
    )
    
    # PIPELINE command - Execute full workflow from YAML
    pipeline_parser = subparsers.add_parser(
        'pipeline',
        help='Execute full pipeline from YAML configuration'
    )
    pipeline_parser.add_argument(
        'config',
        type=Path,
        help='Path to YAML configuration file'
    )
    pipeline_parser.add_argument(
        '--create-example',
        type=str,
        choices=['full', 'enrich', 'patch'],
        help='Create example configuration file and exit'
    )
    
    # DOWNLOAD command
    download_parser = subparsers.add_parser('download', help='Download LiDAR tiles from IGN')
    download_parser.add_argument('--bbox', type=str, required=True,
                                help='Bounding box: lon_min,lat_min,lon_max,lat_max (WGS84)')
    download_parser.add_argument('--output', type=Path, required=True, help='Output directory')
    download_parser.add_argument('--max-tiles', type=int, help='Maximum number of tiles to download')
    download_parser.add_argument('--num-workers', type=int, default=3,
                                help='Number of parallel downloads (default: 3)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Validate inputs for patch, process and enrich commands
    if args.command in ['patch', 'process', 'enrich']:
        if not args.input and not args.input_dir:
            parser.error(f"{args.command}: Either --input or --input-dir must be specified")
    
    # Handle pipeline --create-example
    if args.command == 'pipeline' and hasattr(args, 'create_example') and args.create_example:
        from .pipeline_config import create_example_config
        output_path = args.config if args.config else Path(f'pipeline_{args.create_example}.yaml')
        create_example_config(output_path, args.create_example)
        logger.info(f"✓ Created example configuration: {output_path}")
        logger.info(f"  Edit the file and run: ign-lidar-hd pipeline {output_path}")
        return 0
    
    # Execute command
    if args.command == 'patch':
        return cmd_process(args)  # Uses same implementation
    elif args.command == 'process':
        # Show deprecation warning
        logger.warning("⚠️  'process' command is deprecated, use 'patch' instead")
        return cmd_process(args)
    elif args.command == 'enrich':
        return cmd_enrich(args)
    elif args.command == 'download':
        return cmd_download(args)
    elif args.command == 'pipeline':
        return cmd_pipeline(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


if __name__ == '__main__':
    main()
