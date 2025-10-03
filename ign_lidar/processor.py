"""
Main LiDAR Processing Class
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import multiprocessing as mp
from functools import partial
import time

import numpy as np
import laspy
from tqdm import tqdm

from .features import compute_all_features_optimized
from .classes import ASPRS_TO_LOD2, ASPRS_TO_LOD3
from .utils import extract_patches, augment_patch, save_patch
from .metadata import MetadataManager
from .architectural_styles import (
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
    
    def __init__(self, lod_level: str = 'LOD2', augment: bool = True,
                 num_augmentations: int = 3, bbox=None,
                 patch_size: float = 150.0,
                 patch_overlap: float = 0.1,
                 num_points: int = 16384,
                 include_extra_features: bool = False,
                 k_neighbors: int = None,
                 include_architectural_style: bool = False,
                 style_encoding: str = 'constant'):
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
        
        # 2. Compute geometric features (optimized, single pass)
        feature_mode = "BUILDING" if self.include_extra_features else "CORE"
        k_display = self.k_neighbors if self.k_neighbors else "auto"
        logger.info(f"  🔧 Computing features | k={k_display} | mode={feature_mode}")
        
        feature_start = time.time()
        
        # Compute patch center for distance_to_center feature
        patch_center = np.mean(points, axis=0) if self.include_extra_features else None
        
        # Use manual k if specified, otherwise auto-estimate
        use_auto_k = self.k_neighbors is None
        k_value = self.k_neighbors if self.k_neighbors is not None else None
        
        normals, curvature, height, geo_features = compute_all_features_optimized(
            points=points,
            classification=classification,
            k=k_value,
            auto_k=use_auto_k,
            include_extra=self.include_extra_features,
            patch_center=patch_center
        )
        
        feature_time = time.time() - feature_start
        logger.info(f"  ⏱️  Features computed in {feature_time:.1f}s")
        
        # 3. Remap labels
        labels = np.array([
            self.class_mapping.get(c, self.default_class)
            for c in classification
        ], dtype=np.uint8)
        
        # 4. Combine features
        all_features = {
            'normals': normals,
            'curvature': curvature,
            'intensity': intensity,
            'return_number': return_number,
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
                    num_points=len(points),
                    encoding="multihot"
                )
            else:
                # Single style (constant or legacy)
                architectural_style = encode_style_as_feature(
                    style_id=architectural_style_id,
                    num_points=len(points),
                    encoding="constant"
                )
            all_features['architectural_style'] = architectural_style
        
        # 5. Extract and save patches
        logger.info(f"  📦 Extracting patches (size={self.patch_size}m, "
                   f"target_points={self.num_points})...")
        patches = extract_patches(
            points, all_features, labels,
            patch_size=self.patch_size,
            overlap=self.patch_overlap,
            min_points=10000,
            target_num_points=self.num_points
        )
        
        # 6. Save patches
        output_dir.mkdir(parents=True, exist_ok=True)
        num_saved = 0
        num_original = len(patches)
        
        logger.info(f"  💾 Saving {num_original} patches" + 
                   (f" + {num_original * self.num_augmentations} augmented" if self.augment else ""))
        
        for j, patch in enumerate(patches):
            # Save original patch
            patch_name = f"{laz_file.stem}_patch_{j:04d}.npz"
            save_path = output_dir / patch_name
            save_patch(save_path, patch, self.lod_level)
            num_saved += 1
            
            # Save augmented versions if enabled
            if self.augment:
                for aug_idx in range(self.num_augmentations):
                    aug_patch = augment_patch(patch)
                    aug_name = f"{laz_file.stem}_patch_{j:04d}_aug_{aug_idx}.npz"
                    aug_path = output_dir / aug_name
                    save_patch(aug_path, aug_patch, self.lod_level)
                    num_saved += 1
        
        tile_time = time.time() - tile_start
        logger.info(f"  ✅ Completed: {num_saved} patches in {tile_time:.1f}s "
                   f"({len(points)/tile_time:.0f} pts/s)")
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
