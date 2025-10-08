"""
Main LiDAR Processing Class
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import multiprocessing as mp
from functools import partial
import time

import numpy as np
import laspy
from tqdm import tqdm

from ..features.features import (
    compute_all_features_optimized,
    compute_all_features_with_gpu
)
from ..classes import ASPRS_TO_LOD2, ASPRS_TO_LOD3
from ..preprocessing.utils import (
    extract_patches,
    augment_raw_points,
    save_patch
)
from ..io.metadata import MetadataManager
from ..features.architectural_styles import (
    get_architectural_style_id,
    encode_style_as_feature,
    encode_multi_style_feature,
    infer_multi_styles_from_characteristics
)

# Configure logging
logger = logging.getLogger(__name__)


class LiDARProcessor:
    """
    Main class for processing IGN LiDAR HD data into ML-ready datasets.
    """
    
    def __init__(self, lod_level: str = 'LOD2', augment: bool = False,
                 num_augmentations: int = 3, bbox=None,
                 patch_size: float = 150.0,
                 patch_overlap: float = 0.1,
                 num_points: int = 16384,
                 include_extra_features: bool = False,
                 k_neighbors: int = None,
                 include_architectural_style: bool = False,
                 style_encoding: str = 'constant',
                 include_rgb: bool = False,
                 rgb_cache_dir: Path = None,
                 use_gpu: bool = False,
                 preprocess: bool = False,
                 preprocess_config: dict = None,
                 use_stitching: bool = False,
                 buffer_size: float = 10.0):
        """
        Initialize processor.
        
        Args:
            lod_level: 'LOD2' or 'LOD3'
            augment: Enable data augmentation
            num_augmentations: Number of augmentations per patch
            bbox: Bounding box (xmin, ymin, xmax, ymax) for filtering
            patch_size: Patch size in meters
            patch_overlap: Overlap ratio between patches
            num_points: Target number of points per patch (4096, 8192, 16384)
            include_extra_features: If True, compute extra features
                                   (height, local stats, verticality)
                                   for building extraction (+40% time)
            k_neighbors: Number of neighbors for feature computation (None = auto)
            include_architectural_style: If True, include architectural style
                                         as a feature (requires tile metadata)
            style_encoding: Style encoding method:
                           - 'constant': Single dominant style [N]
                           - 'multihot': Multi-label with weights [N, 13]
            include_rgb: If True, add RGB from IGN orthophotos
            rgb_cache_dir: Directory to cache orthophoto tiles
            use_gpu: If True, use GPU acceleration for feature computation
                    (requires CuPy and RAPIDS cuML)
            preprocess: If True, apply preprocessing to reduce artifacts
                       (SOR, ROR, optional voxel downsampling)
            preprocess_config: Custom preprocessing configuration dict
                              If None, uses sensible defaults
            use_stitching: If True, enable tile stitching for boundary-aware
                          feature computation (Sprint 3)
            buffer_size: Buffer zone size for tile stitching (in meters)
        """
        self.lod_level = lod_level
        self.augment = augment
        self.num_augmentations = num_augmentations
        self.bbox = bbox
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.num_points = num_points
        self.include_extra_features = include_extra_features
        self.k_neighbors = k_neighbors
        self.include_architectural_style = include_architectural_style
        self.style_encoding = style_encoding
        self.include_rgb = include_rgb
        self.rgb_cache_dir = rgb_cache_dir
        self.use_gpu = use_gpu
        self.preprocess = preprocess
        self.use_stitching = use_stitching
        self.buffer_size = buffer_size
        self.preprocess_config = preprocess_config
        
        # Initialize RGB fetcher if needed
        self.rgb_fetcher = None
        if include_rgb:
            try:
                from .rgb_augmentation import IGNOrthophotoFetcher
                self.rgb_fetcher = IGNOrthophotoFetcher(cache_dir=rgb_cache_dir)
                logger.info("RGB augmentation enabled (IGN orthophotos)")
            except ImportError as e:
                logger.error(
                    f"RGB augmentation requires additional packages: {e}"
                )
                logger.error("Install with: pip install requests Pillow")
                self.include_rgb = False

        # Validate GPU availability if requested
        if use_gpu:
            try:
                from .features_gpu import GPU_AVAILABLE
                if not GPU_AVAILABLE:
                    logger.warning(
                        "GPU requested but CuPy not available. "
                        "Using CPU."
                    )
                    self.use_gpu = False
                else:
                    logger.info("GPU acceleration enabled")
            except ImportError:
                logger.warning("GPU module not available. Using CPU.")
                self.use_gpu = False
        
        # Set class mapping
        if lod_level == 'LOD2':
            self.class_mapping = ASPRS_TO_LOD2
            self.default_class = 14
        else:
            self.class_mapping = ASPRS_TO_LOD3
            self.default_class = 29
            
        logger.info(f"Initialized LiDARProcessor with {lod_level}")
    
    def process_tile(self, laz_file: Path, output_dir: Path,
                     tile_idx: int = 0, total_tiles: int = 0,
                     skip_existing: bool = True) -> int:
        """
        Process a single LAZ tile.
        
        Args:
            laz_file: Path to LAZ file
            output_dir: Output directory
            tile_idx: Current tile index (for progress display)
            total_tiles: Total number of tiles (for progress display)
            skip_existing: Skip processing if patches already exist
            
        Returns:
            Number of patches created (0 if skipped)
        """
        progress_prefix = f"[{tile_idx}/{total_tiles}]" if total_tiles > 0 else ""
        
        # Check if patches from this tile already exist
        if skip_existing:
            tile_stem = laz_file.stem
            pattern = f"{tile_stem}_patch_*.npz"
            existing_patches = list(output_dir.glob(pattern))
            
            if existing_patches:
                num_existing = len(existing_patches)
                logger.info(
                    f"{progress_prefix} ⏭️  {laz_file.name}: "
                    f"{num_existing} patches exist, skipping"
                )
                return 0
        
        logger.info(f"{progress_prefix} Processing: {laz_file.name}")
        
        tile_start = time.time()
        
        # Load tile metadata if architectural style is requested
        architectural_style_id = 0  # Default: unknown
        multi_styles = None  # For multi-label encoding
        
        if self.include_architectural_style:
            metadata_mgr = MetadataManager(laz_file.parent)
            tile_metadata = metadata_mgr.load_tile_metadata(laz_file)
            
            if tile_metadata:
                # Check for new multi-label styles
                if "architectural_styles" in tile_metadata:
                    multi_styles = tile_metadata["architectural_styles"]
                    style_names = [s.get("style_name", "?") 
                                 for s in multi_styles]
                    logger.info(f"  🏛️  Multi-style: {', '.join(style_names)}")
                else:
                    # Fall back to single style (legacy)
                    characteristics = tile_metadata.get("characteristics", [])
                    category = tile_metadata.get("location", {}).get("category")
                    architectural_style_id = get_architectural_style_id(
                        characteristics=characteristics,
                        category=category
                    )
                    loc_name = tile_metadata.get("location", {}).get("name", "?")
                    logger.info(f"  🏛️  Style: {architectural_style_id} ({loc_name})")
            else:
                logger.debug(f"  No metadata for {laz_file.name}, style=0")
        
        # 1. Load LAZ file
        try:
            las = laspy.read(str(laz_file))
        except Exception as e:
            logger.error(f"  ✗ Failed to read {laz_file}: {e}")
            return 0
        
        # Extract basic data
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        intensity = np.array(las.intensity, dtype=np.float32) / 65535.0
        return_number = np.array(las.return_number, dtype=np.float32)
        classification = np.array(las.classification, dtype=np.uint8)
        
        logger.info(f"  📊 Loaded {len(points):,} points | "
                   f"Classes: {len(np.unique(classification))}")
        
        # Apply bounding box filter if specified
        if self.bbox is not None:
            xmin, ymin, xmax, ymax = self.bbox
            mask = (
                (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
            )
            points = points[mask]
            intensity = intensity[mask]
            return_number = return_number[mask]
            classification = classification[mask]
            logger.info(f"  After bbox filter: {len(points):,} points")
        
        # Store original data for potential augmentation
        original_data = {
            'points': points,
            'intensity': intensity,
            'return_number': return_number,
            'classification': classification
        }
        
        # Determine number of versions to process (original + augmentations)
        num_versions = 1 + (self.num_augmentations if self.augment else 0)
        all_patches_collected = []
        
        for version_idx in range(num_versions):
            # Apply augmentation to raw data if not the first version
            if version_idx == 0:
                # Original version - no augmentation
                points_v = original_data['points']
                intensity_v = original_data['intensity']
                return_number_v = original_data['return_number']
                classification_v = original_data['classification']
                version_label = "original"
            else:
                # Augmented version - apply transformations BEFORE features
                (points_v, intensity_v,
                 return_number_v, classification_v) = augment_raw_points(
                    original_data['points'],
                    original_data['intensity'],
                    original_data['return_number'],
                    original_data['classification']
                )
                version_label = f"aug_{version_idx-1}"
                logger.info(
                    f"  🔄 Augmented v{version_idx}/{num_versions-1} "
                    f"({len(points_v):,} points after dropout)"
                )
        
            # 1b. Apply preprocessing if enabled (before feature computation)
            if self.preprocess:
                logger.info("  🧹 Preprocessing (artifact mitigation)...")
                preprocess_start = time.time()
                
                from ..preprocessing.preprocessing import (
                    statistical_outlier_removal,
                    radius_outlier_removal,
                    voxel_downsample
                )
                
                # Get config or use defaults
                cfg = self.preprocess_config or {}
                sor_cfg = cfg.get('sor', {'enable': True})
                ror_cfg = cfg.get('ror', {'enable': True})
                voxel_cfg = cfg.get('voxel', {'enable': False})
                
                # Track cumulative mask
                cumulative_mask = np.ones(len(points_v), dtype=bool)
                
                # Apply SOR if enabled
                if sor_cfg.get('enable', True):
                    _, sor_mask = statistical_outlier_removal(
                        points_v,
                        k=sor_cfg.get('k', 12),
                        std_multiplier=sor_cfg.get('std_multiplier', 2.0)
                    )
                    cumulative_mask &= sor_mask
                
                # Apply ROR if enabled
                if ror_cfg.get('enable', True):
                    _, ror_mask = radius_outlier_removal(
                        points_v,
                        radius=ror_cfg.get('radius', 1.0),
                        min_neighbors=ror_cfg.get('min_neighbors', 4)
                    )
                    cumulative_mask &= ror_mask
                
                # Filter all arrays
                original_count = len(points_v)
                points_v = points_v[cumulative_mask]
                intensity_v = intensity_v[cumulative_mask]
                return_number_v = return_number_v[cumulative_mask]
                classification_v = classification_v[cumulative_mask]
                
                # Apply voxel downsampling if enabled
                if voxel_cfg.get('enable', False):
                    points_v, voxel_indices = voxel_downsample(
                        points_v,
                        voxel_size=voxel_cfg.get('voxel_size', 0.5),
                        method=voxel_cfg.get('method', 'centroid')
                    )
                    # Filter other arrays by voxel indices
                    intensity_v = intensity_v[voxel_indices]
                    return_number_v = return_number_v[voxel_indices]
                    classification_v = classification_v[voxel_indices]
                
                final_count = len(points_v)
                reduction = 1 - final_count / original_count
                preprocess_time = time.time() - preprocess_start
                
                logger.info(
                    f"  ✓ Preprocessing: {final_count:,}/{original_count:,} "
                    f"({reduction:.1%} reduction, {preprocess_time:.2f}s)"
                )
            
            # 2. Compute geometric features (optimized, single pass)
            feature_mode = ("FULL" if self.include_extra_features
                           else "CORE")
            k_display = self.k_neighbors if self.k_neighbors else "auto"
            logger.info(
                f"  🔧 Computing features | k={k_display} | "
                f"mode={feature_mode}"
            )
            
            feature_start = time.time()
            
            # Compute patch center for distance_to_center feature
            patch_center = (np.mean(points_v, axis=0)
                           if self.include_extra_features else None)
            
            # Use manual k if specified, otherwise auto-estimate
            use_auto_k = self.k_neighbors is None
            k_value = (self.k_neighbors
                      if self.k_neighbors is not None else None)
            
            # Choose GPU or CPU based on configuration
            if self.use_gpu:
                normals, curvature, height, geo_features = (
                    compute_all_features_with_gpu(
                        points=points_v,
                        classification=classification_v,
                        k=k_value,
                        auto_k=use_auto_k,
                        use_gpu=True
                    )
                )
            else:
                normals, curvature, height, geo_features = (
                    compute_all_features_optimized(
                        points=points_v,
                        classification=classification_v,
                        k=k_value,
                        auto_k=use_auto_k,
                        include_extra=self.include_extra_features,
                        patch_center=patch_center
                    )
                )
            
            feature_time = time.time() - feature_start
            logger.info(f"  ⏱️  Features computed in {feature_time:.1f}s")
            
            # 3. Remap labels for this version
            labels_v = np.array([
                self.class_mapping.get(c, self.default_class)
                for c in classification_v
            ], dtype=np.uint8)
            
            # 4. Combine features
            all_features_v = {
                'normals': normals,
                'curvature': curvature,
                'intensity': intensity_v,
                'return_number': return_number_v,
                'height': height,
                **geo_features
            }
            
            # Add architectural style if requested
            if self.include_architectural_style:
                if multi_styles and self.style_encoding == 'multihot':
                    # Multi-label encoding with weights
                    style_ids = [s["style_id"] for s in multi_styles]
                    weights = [s.get("weight", 1.0) for s in multi_styles]
                    architectural_style = encode_multi_style_feature(
                        style_ids=style_ids,
                        weights=weights,
                        num_points=len(points_v),
                        encoding="multihot"
                    )
                else:
                    # Single style (constant or legacy)
                    architectural_style = encode_style_as_feature(
                        style_id=architectural_style_id,
                        num_points=len(points_v),
                        encoding="constant"
                    )
                all_features_v['architectural_style'] = architectural_style
            
            # 5. Extract patches from this version
            logger.info(
                f"  📦 Extracting patches (size={self.patch_size}m, "
                f"target_points={self.num_points})..."
            )
            patches_v = extract_patches(
                points_v, all_features_v, labels_v,
                patch_size=self.patch_size,
                overlap=self.patch_overlap,
                min_points=10000,
                target_num_points=self.num_points
            )
            
            # 5b. Add RGB if requested
            if self.include_rgb and self.rgb_fetcher:
                logger.info(
                    "  🎨 Augmenting patches with RGB "
                    "from IGN orthophotos..."
                )
                # Get tile bounding box for orthophoto fetch
                tile_bbox = (
                    points_v[:, 0].min(),
                    points_v[:, 1].min(),
                    points_v[:, 0].max(),
                    points_v[:, 1].max()
                )
                
                for patch in patches_v:
                    try:
                        # Get absolute coordinates for this patch
                        patch_points_abs = patch['points'].copy()
                        patch_center_xy = np.array([
                            (tile_bbox[0] + tile_bbox[2]) / 2,
                            (tile_bbox[1] + tile_bbox[3]) / 2,
                            0
                        ])
                        patch_points_abs[:, :2] += patch_center_xy[:2]
                        
                        # Fetch RGB
                        rgb = self.rgb_fetcher.augment_points_with_rgb(
                            patch_points_abs,
                            bbox=tile_bbox
                        )
                        patch['rgb'] = rgb.astype(np.float32) / 255.0
                    except Exception as e:
                        logger.warning(
                            f"  ⚠️  RGB augmentation failed: {e}"
                        )
                        # Add default gray color
                        patch['rgb'] = np.full(
                            (len(patch['points']), 3),
                            0.5,
                            dtype=np.float32
                        )
            
            # Store patches with version suffix
            for patch in patches_v:
                patch['_version'] = version_label
                all_patches_collected.append(patch)
        
        # 6. Save all collected patches
        output_dir.mkdir(parents=True, exist_ok=True)
        num_saved = 0
        
        rgb_suffix = " + RGB" if self.include_rgb else ""
        total_patches = len(all_patches_collected)
        logger.info(
            f"  💾 Saving {total_patches} patches{rgb_suffix} "
            f"({num_versions} versions)"
        )
        
        # Group patches by base patch index for naming
        patches_by_base = {}
        for patch in all_patches_collected:
            version = patch.pop('_version', 'original')
            # Find base patch index (all patches with same spatial origin)
            base_key = tuple(patch['points'][:3, :].flatten())
            if base_key not in patches_by_base:
                patches_by_base[base_key] = []
            patches_by_base[base_key].append((version, patch))
        
        # Save patches with proper naming
        for base_idx, (_, patch_group) in enumerate(
            patches_by_base.items()
        ):
            for version_label, patch in patch_group:
                if version_label == 'original':
                    patch_name = f"{laz_file.stem}_patch_{base_idx:04d}.npz"
                else:
                    patch_name = (
                        f"{laz_file.stem}_patch_{base_idx:04d}_"
                        f"{version_label}.npz"
                    )
                save_path = output_dir / patch_name
                save_patch(save_path, patch, self.lod_level)
                num_saved += 1
        
        tile_time = time.time() - tile_start
        pts_processed = (len(original_data['points']) * num_versions)
        logger.info(
            f"  ✅ Completed: {num_saved} patches in {tile_time:.1f}s "
            f"({pts_processed/tile_time:.0f} pts/s)"
        )
        return num_saved
    
    def process_directory(self, input_dir: Path, output_dir: Path,
                          num_workers: int = 1, save_metadata: bool = True,
                          skip_existing: bool = True) -> int:
        """
        Process directory of LAZ files.
        
        Args:
            input_dir: Directory containing LAZ files
            output_dir: Output directory
            num_workers: Number of parallel workers
            save_metadata: Whether to save stats.json
            skip_existing: Skip tiles that already have patches in output
            
        Returns:
            Total number of patches created
        """
        start_time = time.time()
        
        # Check system memory and adjust workers if needed
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
                    "⚠️  Memory pressure detected - reducing workers to 1"
                )
                num_workers = 1
            
            # Processing needs ~2-3GB per worker
            min_gb_per_worker = 2.5
            max_safe_workers = int(available_gb / min_gb_per_worker)
            
            if num_workers > max_safe_workers:
                logger.warning(
                    f"⚠️  Limited RAM ({available_gb:.1f}GB available)"
                )
                logger.warning(
                    f"⚠️  Reducing workers from {num_workers} "
                    f"to {max(1, max_safe_workers)}"
                )
                num_workers = max(1, max_safe_workers)
                
        except ImportError:
            logger.debug("psutil not available - skipping memory checks")
        
        # Find LAZ files (recursively)
        laz_files = (list(input_dir.rglob("*.laz")) +
                     list(input_dir.rglob("*.LAZ")))
        
        if not laz_files:
            logger.error(f"No LAZ files found in {input_dir}")
            return 0
        
        total_tiles = len(laz_files)
        logger.info(f"Found {total_tiles} LAZ files")
        k_display = self.k_neighbors or 'auto'
        logger.info(
            f"Configuration: LOD={self.lod_level} | k={k_display} | "
            f"patch_size={self.patch_size}m | augment={self.augment}"
        )
        logger.info("")
        
        # Initialize metadata manager
        metadata_mgr = MetadataManager(output_dir) if save_metadata else None
        
        # Copy directory structure from source
        if metadata_mgr:
            logger.info("Copying directory structure from source...")
            metadata_mgr.copy_directory_structure(input_dir)
        
        # Process files
        tiles_processed = 0
        tiles_skipped = 0
        
        if num_workers > 1:
            logger.info(f"🚀 Processing with {num_workers} parallel workers")
            logger.info("="*70)
            
            # For parallel processing, we can't easily pass tile index
            process_func = partial(
                self.process_tile,
                output_dir=output_dir,
                total_tiles=total_tiles,
                skip_existing=skip_existing
            )
            
            with mp.Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_func, laz_files),
                    total=total_tiles,
                    desc="Processing tiles",
                    unit="tile"
                ))
            
            total_patches = sum(results)
            tiles_skipped = sum(1 for r in results if r == 0)
            tiles_processed = total_tiles - tiles_skipped
        else:
            logger.info("🔄 Processing sequentially")
            logger.info("="*70)
            
            total_patches = 0
            for idx, laz_file in enumerate(laz_files, 1):
                num_patches = self.process_tile(
                    laz_file, output_dir,
                    tile_idx=idx, total_tiles=total_tiles,
                    skip_existing=skip_existing
                )
                total_patches += num_patches
                
                if num_patches == 0:
                    tiles_skipped += 1
                else:
                    tiles_processed += 1
        
        logger.info("")
        logger.info("="*70)
        logger.info("📊 Processing Summary:")
        logger.info(f"  Total tiles: {total_tiles}")
        logger.info(f"  ✅ Processed: {tiles_processed}")
        logger.info(f"  ⏭️  Skipped: {tiles_skipped}")
        logger.info(f"  📦 Total patches created: {total_patches}")
        logger.info("="*70)
        
        # Save metadata
        if metadata_mgr:
            processing_time = time.time() - start_time
            stats = metadata_mgr.create_processing_stats(
                input_dir=input_dir,
                num_tiles=len(laz_files),
                num_patches=total_patches,
                lod_level=self.lod_level,
                k_neighbors=self.k_neighbors,
                patch_size=self.patch_size,
                augmentation=self.augment,
                num_augmentations=self.num_augmentations
            )
            stats["processing_time_seconds"] = round(processing_time, 2)
            metadata_mgr.save_stats(stats)
        
        return total_patches

    def process_tile_unified(
        self,
        laz_file: Path,
        output_dir: Path,
        architecture: str = 'pointnet++',
        save_enriched: bool = False,
        output_format: str = 'npz',
        build_spatial_index: bool = False,
        tile_idx: int = 0,
        total_tiles: int = 0,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single LAZ tile with unified pipeline.
        
        This method implements the v2.0 unified pipeline that processes
        RAW LiDAR → Features → Architecture-formatted patches in a single
        pass, eliminating intermediate LAZ files and reducing I/O by 50%.
        
        Args:
            laz_file: Path to input LAZ file
            output_dir: Output directory for patches
            architecture: Target DL architecture ('pointnet++', 'octree', 
                         'transformer', 'sparse_conv', 'multi')
            save_enriched: If True, save intermediate enriched LAZ file
                          (default: False for faster processing)
            output_format: Output format - 'npz', 'hdf5', 'pytorch', 'multi'
            build_spatial_index: If True, build octree/KNN spatial index
            tile_idx: Current tile index (for progress display)
            total_tiles: Total number of tiles (for progress display)
            skip_existing: Skip processing if patches already exist
            
        Returns:
            Dict with processing stats:
            {
                'num_patches': int,
                'processing_time': float,
                'points_processed': int,
                'skipped': bool
            }
        """
        from .formatters import MultiArchitectureFormatter
        
        progress_prefix = f"[{tile_idx}/{total_tiles}]" if total_tiles > 0 else ""
        
        # Check if patches already exist
        if skip_existing:
            tile_stem = laz_file.stem
            pattern = f"{tile_stem}_*_patch_*.{output_format}"
            existing_patches = list(output_dir.glob(pattern))
            
            if existing_patches:
                num_existing = len(existing_patches)
                logger.info(
                    f"{progress_prefix} ⏭️  {laz_file.name}: "
                    f"{num_existing} patches exist, skipping"
                )
                return {
                    'num_patches': 0,
                    'processing_time': 0.0,
                    'points_processed': 0,
                    'skipped': True
                }
        
        logger.info(f"{progress_prefix} 🚀 Unified processing: {laz_file.name}")
        tile_start = time.time()
        
        # ===== STEP 1: Load RAW LiDAR =====
        logger.info(f"  📂 Loading RAW LiDAR...")
        try:
            las = laspy.read(str(laz_file))
        except Exception as e:
            logger.error(f"  ✗ Failed to read {laz_file}: {e}")
            return {
                'num_patches': 0,
                'processing_time': 0.0,
                'points_processed': 0,
                'skipped': False,
                'error': str(e)
            }
        
        # Extract basic data
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        intensity = np.array(las.intensity, dtype=np.float32) / 65535.0
        return_number = np.array(las.return_number, dtype=np.float32)
        classification = np.array(las.classification, dtype=np.uint8)
        
        # Try to load NIR if available
        nir = None
        if hasattr(las, 'nir') or hasattr(las, 'near_infrared'):
            try:
                nir_raw = (las.nir if hasattr(las, 'nir') 
                          else las.near_infrared)
                nir = np.array(nir_raw, dtype=np.float32) / 65535.0
                logger.info(f"  ✓ NIR channel detected")
            except:
                pass
        
        logger.info(
            f"  📊 Loaded {len(points):,} points | "
            f"Classes: {len(np.unique(classification))}"
        )
        
        # Apply bounding box filter if specified
        if self.bbox is not None:
            xmin, ymin, xmax, ymax = self.bbox
            mask = (
                (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
            )
            points = points[mask]
            intensity = intensity[mask]
            return_number = return_number[mask]
            classification = classification[mask]
            if nir is not None:
                nir = nir[mask]
            logger.info(f"  After bbox filter: {len(points):,} points")
        
        # ===== STEP 2: Preprocessing (if enabled) =====
        if self.preprocess:
            logger.info("  🧹 Preprocessing...")
            from ..preprocessing.preprocessing import (
                statistical_outlier_removal,
                radius_outlier_removal
            )
            
            cfg = self.preprocess_config or {}
            sor_cfg = cfg.get('sor', {'enable': True})
            ror_cfg = cfg.get('ror', {'enable': True})
            
            cumulative_mask = np.ones(len(points), dtype=bool)
            
            if sor_cfg.get('enable', True):
                _, sor_mask = statistical_outlier_removal(
                    points,
                    k=sor_cfg.get('k', 12),
                    std_multiplier=sor_cfg.get('std_multiplier', 2.0)
                )
                cumulative_mask &= sor_mask
            
            if ror_cfg.get('enable', True):
                _, ror_mask = radius_outlier_removal(
                    points,
                    radius=ror_cfg.get('radius', 1.0),
                    min_neighbors=ror_cfg.get('min_neighbors', 4)
                )
                cumulative_mask &= ror_mask
            
            original_count = len(points)
            points = points[cumulative_mask]
            intensity = intensity[cumulative_mask]
            return_number = return_number[cumulative_mask]
            classification = classification[cumulative_mask]
            if nir is not None:
                nir = nir[cumulative_mask]
            
            reduction = 1 - len(points) / original_count
            logger.info(
                f"  ✓ Preprocessing: {len(points):,}/{original_count:,} "
                f"({reduction:.1%} reduction)"
            )
        
        # ===== STEP 3: Compute Features (in-memory) =====
        logger.info("  🔧 Computing features...")
        feature_start = time.time()
        
        # Determine if we should use tile stitching for boundary-aware features
        use_boundary_aware = self.use_stitching and laz_file.parent.exists()
        
        if use_boundary_aware:
            logger.info("  🔗 Enabling tile stitching for boundary-aware features...")
            try:
                from .tile_stitcher import TileStitcher
                from .features_boundary import BoundaryAwareFeatureComputer
                
                # Initialize stitcher
                stitcher = TileStitcher(buffer_size=self.buffer_size)
                
                # Load tile with neighbors
                tile_data = stitcher.load_tile_with_neighbors(
                    tile_path=laz_file,
                    auto_detect_neighbors=True
                )
                
                core_points = tile_data['core_points']
                buffer_points = tile_data['buffer_points']
                
                # Get tile bounds from stitcher
                tile_bounds = stitcher.get_tile_bounds(laz_file)
                
                # Compute boundary-aware features
                computer = BoundaryAwareFeatureComputer(
                    k_neighbors=self.k_neighbors or 20,
                    boundary_threshold=self.buffer_size,
                    compute_normals=True,
                    compute_curvature=True,
                    compute_planarity=self.include_extra_features,
                    compute_verticality=self.include_extra_features
                )
                
                features = computer.compute_features(
                    core_points=core_points,
                    buffer_points=buffer_points if len(buffer_points) > 0 else None,
                    tile_bounds=tile_bounds
                )
                
                # Extract feature arrays
                normals = features['normals']
                curvature = features['curvature']
                
                # Build geo_features array
                if self.include_extra_features:
                    geo_features = np.column_stack([
                        features['planarity'],
                        features['linearity'],
                        features['sphericity'],
                        features['verticality']
                    ])
                else:
                    geo_features = None
                
                # Height feature (relative to local minimum)
                height = points[:, 2] - points[:, 2].min()
                
                num_boundary = features['num_boundary_points']
                logger.info(
                    f"  ✓ Boundary-aware features computed "
                    f"({num_boundary} boundary points affected)"
                )
                
            except Exception as e:
                logger.warning(
                    f"  ⚠️  Tile stitching failed, falling back to standard: {e}"
                )
                use_boundary_aware = False
        
        # Standard feature computation (no stitching)
        if not use_boundary_aware:
            # Choose GPU or CPU based on configuration
            if self.use_gpu:
                from .features_gpu import compute_all_features_with_gpu
                normals, curvature, height, geo_features = (
                    compute_all_features_with_gpu(
                        points=points,
                        classification=classification,
                        k=self.k_neighbors,
                        auto_k=(self.k_neighbors is None),
                        use_gpu=True
                    )
                )
            else:
                normals, curvature, height, geo_features = (
                    compute_all_features_optimized(
                        points=points,
                        classification=classification,
                        k=self.k_neighbors,
                        auto_k=(self.k_neighbors is None),
                        include_extra=self.include_extra_features,
                        patch_center=np.mean(points, axis=0) if self.include_extra_features else None
                    )
                )
        
        feature_time = time.time() - feature_start
        logger.info(f"  ⏱️  Features: {feature_time:.1f}s")
        
        # ===== STEP 4: Add RGB (if requested) =====
        rgb = None
        if self.include_rgb and self.rgb_fetcher:
            logger.info("  🎨 Fetching RGB from IGN orthophotos...")
            try:
                tile_bbox = (
                    points[:, 0].min(),
                    points[:, 1].min(),
                    points[:, 0].max(),
                    points[:, 1].max()
                )
                rgb = self.rgb_fetcher.augment_points_with_rgb(
                    points, bbox=tile_bbox
                )
                rgb = rgb.astype(np.float32) / 255.0
                logger.info(f"  ✓ RGB added")
            except Exception as e:
                logger.warning(f"  ⚠️  RGB fetch failed: {e}")
                rgb = np.full((len(points), 3), 0.5, dtype=np.float32)
        
        # ===== STEP 5: Compute NDVI (if NIR available) =====
        ndvi = None
        if nir is not None and rgb is not None:
            # NDVI = (NIR - Red) / (NIR + Red)
            red = rgb[:, 0]
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir - red) / (nir + red + 1e-8)
                ndvi = np.clip(ndvi, -1, 1)
            logger.info(f"  ✓ NDVI computed")
        
        # ===== STEP 6: Save Enriched LAZ (optional) =====
        if save_enriched:
            logger.info("  💾 Saving enriched LAZ...")
            enriched_path = output_dir / "enriched" / f"{laz_file.stem}_enriched.laz"
            enriched_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create new LAS with features
            new_las = laspy.LasData(las.header)
            new_las.x = las.x
            new_las.y = las.y
            new_las.z = las.z
            new_las.intensity = las.intensity
            new_las.return_number = las.return_number
            new_las.classification = las.classification
            
            # Add extra dimensions for features
            try:
                new_las.add_extra_dim(laspy.ExtraBytesParams(
                    name="normal_x", type=np.float32
                ))
                new_las.add_extra_dim(laspy.ExtraBytesParams(
                    name="normal_y", type=np.float32
                ))
                new_las.add_extra_dim(laspy.ExtraBytesParams(
                    name="normal_z", type=np.float32
                ))
                new_las.add_extra_dim(laspy.ExtraBytesParams(
                    name="curvature", type=np.float32
                ))
                
                new_las.normal_x = normals[:, 0]
                new_las.normal_y = normals[:, 1]
                new_las.normal_z = normals[:, 2]
                new_las.curvature = curvature
                
                new_las.write(enriched_path)
                logger.info(f"  ✓ Enriched LAZ saved: {enriched_path.name}")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to save enriched LAZ: {e}")
        
        # ===== STEP 7: Remap Labels =====
        labels = np.array([
            self.class_mapping.get(c, self.default_class)
            for c in classification
        ], dtype=np.uint8)
        
        # ===== STEP 8: Extract Patches =====
        logger.info(
            f"  📦 Extracting patches "
            f"(size={self.patch_size}m, points={self.num_points})..."
        )
        
        # Build feature dictionary
        all_features = {
            'normals': normals,
            'curvature': curvature,
            'intensity': intensity,
            'return_number': return_number,
            'height': height
        }
        
        # Add geometric features if available
        if geo_features is not None:
            all_features.update(geo_features)
        
        # Add optional features
        if rgb is not None:
            all_features['rgb'] = rgb
        if nir is not None:
            all_features['nir'] = nir
        if ndvi is not None:
            all_features['ndvi'] = ndvi
        
        # Extract patches
        patches = extract_patches(
            points, all_features, labels,
            patch_size=self.patch_size,
            overlap=self.patch_overlap,
            min_points=10000,
            target_num_points=self.num_points
        )
        
        logger.info(f"  ✓ Extracted {len(patches)} patches")
        
        # ===== STEP 9: Format for Target Architecture =====
        logger.info(f"  🏗️  Formatting for architecture: {architecture}")
        
        # Determine which architectures to format
        target_archs = (
            ['pointnet++', 'octree', 'transformer', 'sparse_conv']
            if architecture == 'multi'
            else [architecture]
        )
        
        # Initialize formatter
        formatter = MultiArchitectureFormatter(
            target_archs=target_archs,
            num_points=self.num_points,
            use_rgb=(rgb is not None),
            use_infrared=(nir is not None),
            use_geometric=True,
            use_radiometric=False,
            use_contextual=False
        )
        
        # ===== STEP 10: Save Formatted Patches =====
        output_dir.mkdir(parents=True, exist_ok=True)
        num_saved = 0
        
        for patch_idx, patch in enumerate(patches):
            formatted = formatter.format_patch(patch)
            
            # Save based on output format
            for arch in target_archs:
                arch_data = formatted[arch]
                
                if output_format == 'npz':
                    filename = f"{laz_file.stem}_{arch}_patch_{patch_idx:04d}.npz"
                    save_path = output_dir / filename
                    np.savez_compressed(save_path, **arch_data)
                    
                elif output_format == 'pytorch':
                    import torch
                    filename = f"{laz_file.stem}_{arch}_patch_{patch_idx:04d}.pt"
                    save_path = output_dir / filename
                    tensors = {}
                    for k, v in arch_data.items():
                        if isinstance(v, np.ndarray):
                            tensors[k] = torch.from_numpy(v)
                        elif isinstance(v, (int, float)):
                            tensors[k] = torch.tensor(v)
                        elif isinstance(v, dict):
                            # Handle nested dicts (like octree structure)
                            tensors[k] = v
                    torch.save(tensors, save_path)
                    
                elif output_format == 'hdf5':
                    import h5py
                    filename = f"{laz_file.stem}_{arch}_patch_{patch_idx:04d}.h5"
                    save_path = output_dir / filename
                    with h5py.File(save_path, 'w') as f:
                        for k, v in arch_data.items():
                            f.create_dataset(k, data=v, compression='gzip')
                
                num_saved += 1
        
        tile_time = time.time() - tile_start
        logger.info(
            f"  ✅ Unified processing complete: {num_saved} patches in "
            f"{tile_time:.1f}s ({len(points)/tile_time:.0f} pts/s)"
        )
        
        return {
            'num_patches': num_saved,
            'processing_time': tile_time,
            'points_processed': len(points),
            'skipped': False
        }
