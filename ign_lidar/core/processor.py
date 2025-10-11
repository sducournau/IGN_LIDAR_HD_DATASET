"""
Main LiDAR Processing Class
"""

import logging
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Any, Literal
import multiprocessing as mp
from functools import partial
import time
import gc

import numpy as np
import laspy
from tqdm import tqdm
import h5py

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
from .skip_checker import PatchSkipChecker

# Configure logging
logger = logging.getLogger(__name__)

# Processing mode type definition
ProcessingMode = Literal["patches_only", "both", "enriched_only"]


def aggressive_memory_cleanup():
    """
    Aggressive memory cleanup to prevent OOM.
    Clears all caches and forces garbage collection.
    """
    gc.collect()
    
    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass
    
    # Clear CuPy cache if available
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    except (ImportError, AttributeError):
        pass
    
    gc.collect()


class LiDARProcessor:
    """
    Main class for processing IGN LiDAR HD data into ML-ready datasets.
    """
    
    def __init__(self, lod_level: str = 'LOD2', 
                 processing_mode: ProcessingMode = "patches_only",
                 augment: bool = False,
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
                 include_infrared: bool = False,
                 compute_ndvi: bool = False,
                 use_gpu: bool = False,
                 use_gpu_chunked: bool = True,
                 gpu_batch_size: int = 1_000_000,
                 preprocess: bool = False,
                 preprocess_config: dict = None,
                 use_stitching: bool = False,
                 buffer_size: float = 10.0,
                 stitching_config: dict = None,
                 save_enriched_laz: bool = None,
                 only_enriched_laz: bool = None,
                 architecture: str = 'pointnet++',
                 output_format: str = 'npz'):
        """
        Initialize processor.
        
        Args:
            lod_level: 'LOD2' or 'LOD3'
            processing_mode: Processing mode - 'patches_only' (default), 'both', or 'enriched_only'
                           - 'patches_only': Create ML patches only (default, fastest for training)
                           - 'both': Create both patches and enriched LAZ files  
                           - 'enriched_only': Only create enriched LAZ (fastest for GIS)
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
            include_infrared: If True, add NIR (near-infrared) channel from LAZ files
            compute_ndvi: If True, compute NDVI from RGB and NIR (requires both)
            use_gpu: If True, use GPU acceleration for feature computation
                    (requires CuPy and RAPIDS cuML)
            use_gpu_chunked: If True, use chunked GPU processing for large tiles
                            (>5M points, requires CuPy and RAPIDS cuML)
            gpu_batch_size: Batch size for GPU processing (default: 1M points)
            preprocess: If True, apply preprocessing to reduce artifacts
                       (SOR, ROR, optional voxel downsampling)
            preprocess_config: Custom preprocessing configuration dict
                              If None, uses sensible defaults
            use_stitching: If True, enable tile stitching for boundary-aware
                          feature computation (Sprint 3)
            buffer_size: Buffer zone size for tile stitching (in meters)
            save_enriched_laz: DEPRECATED - use processing_mode instead
            only_enriched_laz: DEPRECATED - use processing_mode instead
            architecture: Target DL architecture ('pointnet++', 'octree', 'transformer', 
                         'sparse_conv', 'hybrid', 'multi')
            output_format: Output format - 'npz', 'hdf5', 'pytorch'/'torch', 'laz'
                          Supports multi-format: 'hdf5,laz' to save in both formats
                          (Note: PyTorch format requires torch to be installed)
        """
        # Handle backward compatibility for deprecated flags
        if save_enriched_laz is not None or only_enriched_laz is not None:
            logger.warning(
                "⚠️  'save_enriched_laz' and 'only_enriched_laz' are deprecated. "
                "Please use 'processing_mode' instead:\n"
                "  - processing_mode='patches_only' (default)\n"
                "  - processing_mode='both' (patches + enriched LAZ)\n"
                "  - processing_mode='enriched_only' (LAZ only)"
            )
            # Convert old flags to new mode
            if only_enriched_laz:
                processing_mode = "enriched_only"
                logger.info("→ Using processing_mode='enriched_only'")
            elif save_enriched_laz:
                processing_mode = "both"
                logger.info("→ Using processing_mode='both'")
            else:
                processing_mode = "patches_only"
                logger.info("→ Using processing_mode='patches_only'")
        
        # Store processing mode
        self.processing_mode = processing_mode
        
        # Derive save/only flags from processing mode for backward compatibility
        self.save_enriched_laz = processing_mode in ["both", "enriched_only"]
        self.only_enriched_laz = processing_mode == "enriched_only"
        
        logger.info(f"✨ Processing mode: {self.processing_mode}")
        
        # Store other parameters
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
        self.include_infrared = include_infrared
        self.compute_ndvi = compute_ndvi
        self.use_gpu = use_gpu
        self.use_gpu_chunked = use_gpu_chunked
        self.gpu_batch_size = gpu_batch_size
        self.preprocess = preprocess
        self.use_stitching = use_stitching
        self.buffer_size = buffer_size
        self.preprocess_config = preprocess_config
        # Note: save_enriched_laz and only_enriched_laz set above from processing_mode
        self.architecture = architecture
        self.output_format = output_format
        
        # Validate output format (supports comma-separated multi-format)
        SUPPORTED_FORMATS = ['npz', 'hdf5', 'pytorch', 'torch', 'laz']
        formats_list = [fmt.strip() for fmt in output_format.split(',')]
        
        for fmt in formats_list:
            if fmt not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported output format: '{fmt}'. "
                    f"Supported formats: {', '.join(SUPPORTED_FORMATS)}\n"
                    f"For multiple formats, use comma-separated list: 'hdf5,laz'"
                )
        
        # Check PyTorch availability if torch format requested
        if any(fmt in ['pytorch', 'torch'] for fmt in formats_list) and not TORCH_AVAILABLE:
            raise ImportError(
                f"PyTorch format requested but torch is not installed. "
                f"Install with: pip install torch"
            )
        
        # Enhanced stitching configuration
        if stitching_config is None:
            self.stitching_config = {
                'enabled': use_stitching,
                'buffer_size': buffer_size,
                'auto_detect_neighbors': True,
                'auto_download_neighbors': False,  # Download missing adjacent tiles if needed
                'cache_enabled': True
            }
        else:
            self.stitching_config = stitching_config.copy()
            # Override enable flag and buffer size
            self.stitching_config['enabled'] = use_stitching
            if 'buffer_size' not in self.stitching_config:
                self.stitching_config['buffer_size'] = buffer_size
            # Default auto_download_neighbors to False if not specified
            if 'auto_download_neighbors' not in self.stitching_config:
                self.stitching_config['auto_download_neighbors'] = False
        
        # Initialize advanced stitcher if needed
        self.stitcher = None
        if use_stitching and self.stitching_config.get('use_stitcher', False):
            try:
                from .tile_stitcher import TileStitcher
                self.stitcher = TileStitcher(config=self.stitching_config)
                logger.info("Advanced tile stitcher initialized")
            except ImportError as e:
                logger.warning(f"Advanced stitcher unavailable: {e}")
                self.stitcher = None
        
        # Initialize RGB fetcher if needed
        self.rgb_fetcher = None
        if include_rgb:
            try:
                from ..preprocessing.rgb_augmentation import IGNOrthophotoFetcher
                # Use provided cache dir or default to user temp folder
                if rgb_cache_dir is None:
                    rgb_cache_dir = Path(tempfile.gettempdir()) / "ign_lidar_cache" / "orthophotos"
                    rgb_cache_dir.mkdir(parents=True, exist_ok=True)
                self.rgb_fetcher = IGNOrthophotoFetcher(cache_dir=rgb_cache_dir)
                logger.info(f"RGB augmentation enabled (IGN orthophotos, cache: {rgb_cache_dir})")
            except ImportError as e:
                logger.error(
                    f"RGB augmentation requires additional packages: {e}"
                )
                logger.error("Install with: pip install requests Pillow")
                self.include_rgb = False
        
        # Initialize Infrared fetcher if needed
        self.infrared_fetcher = None
        if include_infrared:
            try:
                from ..preprocessing.infrared_augmentation import IGNInfraredFetcher
                # Use same cache dir as RGB or default to user temp folder
                if self.rgb_cache_dir is None:
                    infrared_cache_dir = Path(tempfile.gettempdir()) / "ign_lidar_cache" / "infrared"
                else:
                    infrared_cache_dir = Path(self.rgb_cache_dir).parent / "infrared"
                infrared_cache_dir.mkdir(parents=True, exist_ok=True)
                self.infrared_fetcher = IGNInfraredFetcher(cache_dir=infrared_cache_dir)
                logger.info(f"Infrared augmentation enabled (IGN IRC orthophotos, cache: {infrared_cache_dir})")
            except ImportError as e:
                logger.error(
                    f"Infrared augmentation requires additional packages: {e}"
                )
                logger.error("Install with: pip install requests Pillow")
                self.include_infrared = False

        # Validate GPU availability if requested
        if use_gpu:
            try:
                from ..features.features_gpu import GPU_AVAILABLE
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
        
        # Initialize intelligent skip checker
        self.skip_checker = PatchSkipChecker(
            output_format=output_format,
            architecture=architecture,
            num_augmentations=num_augmentations,
            augment=augment,
            validate_content=True,  # Enable content validation
            min_file_size=1024,  # 1KB minimum
            only_enriched_laz=only_enriched_laz,  # Check for enriched LAZ if enabled
        )
            
        logger.info(f"Initialized LiDARProcessor with {lod_level}")
    
    def _save_patch_as_laz(self, save_path: Path, arch_data: Dict[str, np.ndarray], 
                           original_patch: Dict[str, np.ndarray]) -> None:
        """
        Save a patch as a LAZ point cloud file with all computed features.
        
        Args:
            save_path: Output LAZ file path
            arch_data: Formatted patch data (architecture-specific)
            original_patch: Original patch data with metadata
        """
        # Extract coordinates from arch_data
        # Most architectures have 'points' or 'coords' key
        if 'points' in arch_data:
            coords = arch_data['points'][:, :3].copy()  # XYZ
        elif 'coords' in arch_data:
            coords = arch_data['coords'][:, :3].copy()
        else:
            logger.warning(f"Cannot save LAZ patch: no coordinates found in {list(arch_data.keys())}")
            return
        
        # Restore LAMB93 coordinates if metadata available
        # Extract tile coordinates from filename if possible
        try:
            filename = save_path.stem
            parts = filename.split('_')
            # Look for tile coordinates in filename (e.g., LHD_FXX_0649_6863_...)
            if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
                tile_x = int(parts[2])
                tile_y = int(parts[3])
                # LAMB93 tiles are 1km x 1km, tile center at (tile_x * 1000 + 500, tile_y * 1000 + 500)
                tile_center_x = tile_x * 1000 + 500
                tile_center_y = tile_y * 1000 + 500
                
                # If metadata has centroid, restore original coordinates
                if 'metadata' in original_patch and isinstance(original_patch['metadata'], dict):
                    metadata = original_patch['metadata']
                    if 'centroid' in metadata:
                        centroid = np.array(metadata['centroid'])
                        coords[:, 0] = coords[:, 0] + centroid[0] + tile_center_x
                        coords[:, 1] = coords[:, 1] + centroid[1] + tile_center_y
                        coords[:, 2] = coords[:, 2] + centroid[2]
                    else:
                        # No centroid, just apply tile offset
                        coords[:, 0] = coords[:, 0] + tile_center_x
                        coords[:, 1] = coords[:, 1] + tile_center_y
        except Exception as e:
            logger.debug(f"Could not restore LAMB93 coordinates: {e}")
            # Continue with normalized coordinates
        
        # Determine point format based on available data
        # Format 3: supports RGB (LAS 1.4) - most compatible
        # Format 8: supports RGB + NIR (LAS 1.4)
        has_rgb = 'rgb' in arch_data
        has_nir = 'nir' in original_patch and original_patch['nir'] is not None
        point_format = 8 if (has_rgb and has_nir) else (3 if has_rgb else 6)
        
        # Create LAZ file
        header = laspy.LasHeader(version="1.4", point_format=point_format)
        header.offsets = [np.floor(coords[:, 0].min()), np.floor(coords[:, 1].min()), np.floor(coords[:, 2].min())]
        header.scales = [0.001, 0.001, 0.001]
        
        las = laspy.LasData(header)
        las.x = coords[:, 0]
        las.y = coords[:, 1]
        las.z = coords[:, 2]
        
        # Add classification if available
        if 'labels' in arch_data:
            las.classification = arch_data['labels'].astype(np.uint8)
        elif 'classification' in original_patch:
            las.classification = original_patch['classification'].astype(np.uint8)
        
        # Add intensity if available
        if 'intensity' in original_patch:
            las.intensity = (original_patch['intensity'] * 65535).astype(np.uint16)
        
        # Add RGB if available
        if 'rgb' in arch_data:
            rgb = (arch_data['rgb'] * 65535).astype(np.uint16)
            las.red = rgb[:, 0]
            las.green = rgb[:, 1]
            las.blue = rgb[:, 2]
        
        # Add NIR if available and point format supports it (format 8)
        if point_format == 8 and 'nir' in original_patch and original_patch['nir'] is not None:
            nir = (original_patch['nir'] * 65535).astype(np.uint16)
            las.nir = nir
        
        # ===== Add computed features as extra dimensions =====
        # Geometric features (planarity, linearity, sphericity, etc.)
        geometric_features = [
            'planarity', 'linearity', 'sphericity', 'anisotropy', 
            'roughness', 'density', 'curvature', 'verticality'
        ]
        
        for feat_name in geometric_features:
            if feat_name in original_patch:
                try:
                    # Add as float32 extra dimension
                    las.add_extra_dim(laspy.ExtraBytesParams(
                        name=feat_name,
                        type=np.float32,
                        description=f"Geom: {feat_name}"
                    ))
                    setattr(las, feat_name, original_patch[feat_name].astype(np.float32))
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not add feature '{feat_name}' to LAZ: {e}")
        
        # Normals (normal_x, normal_y, normal_z for consistency)
        if 'normals' in original_patch:
            try:
                normals = original_patch['normals']
                for i, comp in enumerate(['normal_x', 'normal_y', 'normal_z']):
                    las.add_extra_dim(laspy.ExtraBytesParams(
                        name=comp,
                        type=np.float32,
                        description=f"Normal {comp[-1]}"
                    ))
                    setattr(las, comp, normals[:, i].astype(np.float32))
            except Exception as e:
                logger.warning(f"  ⚠️  Could not add normals to LAZ: {e}")
        
        # Height features
        height_features = ['height', 'z_normalized', 'z_from_ground', 'z_from_median']
        for feat_name in height_features:
            if feat_name in original_patch:
                try:
                    las.add_extra_dim(laspy.ExtraBytesParams(
                        name=feat_name,
                        type=np.float32,
                        description=f"Height feature: {feat_name}"
                    ))
                    setattr(las, feat_name, original_patch[feat_name].astype(np.float32))
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not add feature '{feat_name}' to LAZ: {e}")
        
        # Radiometric features (NIR, NDVI, etc.)
        # Add NIR as extra dimension if not already included in point format
        if point_format != 8 and 'nir' in original_patch and original_patch['nir'] is not None:
            try:
                las.add_extra_dim(laspy.ExtraBytesParams(
                    name='nir',
                    type=np.float32,
                    description="NIR reflectance (norm 0-1)"
                ))
                las.nir = original_patch['nir'].astype(np.float32)
            except Exception as e:
                logger.warning(f"  ⚠️  Could not add NIR to LAZ: {e}")
        
        if 'ndvi' in original_patch and original_patch['ndvi'] is not None:
            try:
                las.add_extra_dim(laspy.ExtraBytesParams(
                    name='ndvi',
                    type=np.float32,
                    description="NDVI (vegetation index)"
                ))
                las.ndvi = original_patch['ndvi'].astype(np.float32)
            except Exception as e:
                logger.warning(f"  ⚠️  Could not add NDVI to LAZ: {e}")
        
        # Return number (if not already in standard fields)
        if 'return_number' in original_patch:
            try:
                # Return number might already be a standard field
                if not hasattr(las, 'return_number'):
                    las.add_extra_dim(laspy.ExtraBytesParams(
                        name='return_number',
                        type=np.uint8,
                        description="Return number"
                    ))
                    las.return_number = original_patch['return_number'].astype(np.uint8)
            except Exception as e:
                logger.warning(f"  ⚠️  Could not add return_number to LAZ: {e}")
        
        # Write LAZ file
        las.write(str(save_path))
    
    def _redownload_tile(self, laz_file: Path) -> bool:
        """
        Attempt to re-download a corrupted tile from IGN WFS.
        
        Args:
            laz_file: Path to the corrupted LAZ file
            
        Returns:
            True if re-download succeeded, False otherwise
        """
        try:
            from ..downloader import IGNLiDARDownloader
            import shutil
            
            # Get filename
            filename = laz_file.name
            
            logger.info(f"  🌐 Re-downloading {filename} from IGN WFS...")
            
            # Backup corrupted file
            backup_path = laz_file.with_suffix('.laz.corrupted')
            if laz_file.exists():
                shutil.move(str(laz_file), str(backup_path))
                logger.debug(f"  Backed up corrupted file to {backup_path.name}")
            
            # Initialize downloader with output directory
            downloader = IGNLiDARDownloader(output_dir=laz_file.parent)
            
            # Download tile (force re-download, don't skip)
            success, was_skipped = downloader.download_tile(
                filename=filename,
                force=True,
                skip_existing=False
            )
            
            if success and laz_file.exists():
                # Verify the download
                try:
                    import laspy
                    test_las = laspy.read(str(laz_file))
                    if len(test_las.points) > 0:
                        logger.info(
                            f"  ✓ Re-downloaded tile verified "
                            f"({len(test_las.points):,} points)"
                        )
                        # Remove backup if successful
                        if backup_path.exists():
                            backup_path.unlink()
                        return True
                    else:
                        logger.error(f"  ✗ Re-downloaded tile has no points")
                        # Restore backup
                        if backup_path.exists():
                            if laz_file.exists():
                                laz_file.unlink()
                            shutil.move(str(backup_path), str(laz_file))
                        return False
                except Exception as verify_error:
                    logger.error(
                        f"  ✗ Re-downloaded tile is also corrupted: {verify_error}"
                    )
                    # Restore backup
                    if backup_path.exists():
                        if laz_file.exists():
                            laz_file.unlink()
                        shutil.move(str(backup_path), str(laz_file))
                    return False
            else:
                logger.error(f"  ✗ Download failed or file not created")
                # Restore backup
                if backup_path.exists():
                    if not laz_file.exists():
                        shutil.move(str(backup_path), str(laz_file))
                return False
                
        except ImportError as ie:
            logger.warning(
                f"  ⚠️  IGNLidarDownloader not available for auto-recovery: {ie}"
            )
            return False
        except Exception as e:
            logger.error(f"  ✗ Re-download failed: {e}")
            return False
    
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
        
        # 1. Load LAZ file (with auto-recovery for corrupted files)
        max_retries = 2
        las = None
        for attempt in range(max_retries):
            try:
                las = laspy.read(str(laz_file))
                break  # Success
            except Exception as e:
                error_msg = str(e)
                is_corruption_error = (
                    'failed to fill whole buffer' in error_msg.lower() or
                    'ioerror' in error_msg.lower() or
                    'unexpected end of file' in error_msg.lower() or
                    'invalid' in error_msg.lower()
                )
                
                if is_corruption_error and attempt < max_retries - 1:
                    logger.warning(f"  ⚠️  Corrupted LAZ file detected: {error_msg}")
                    logger.info(
                        f"  🔄 Attempting to re-download tile "
                        f"(attempt {attempt + 2}/{max_retries})..."
                    )
                    
                    if self._redownload_tile(laz_file):
                        logger.info(f"  ✓ Tile re-downloaded successfully")
                        continue  # Retry loading
                    else:
                        logger.error(f"  ✗ Failed to re-download tile")
                        return 0
                else:
                    logger.error(f"  ✗ Failed to read {laz_file}: {e}")
                    return 0
        
        if las is None:
            logger.error(f"  ✗ Failed to load LAZ file after retries")
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
        
        # Store original data - used for ALL versions
        original_data = {
            'points': points,
            'intensity': intensity,
            'return_number': return_number,
            'classification': classification
        }
        
        # Process original data ONCE to extract patch locations
        points_v = original_data['points']
        intensity_v = original_data['intensity']
        return_number_v = original_data['return_number']
        classification_v = original_data['classification']
        
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
        
        # 2. Compute geometric features (optimized, single pass) on ORIGINAL data
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
                  if self.k_neighbors is not None else 20)  # Default value
        
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
        
        # 3. Remap labels
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
        
        # 5. Extract patches from ORIGINAL data ONCE (not per augmentation!)
        logger.info(
            f"  📦 Extracting patches (size={self.patch_size}m, "
            f"target_points={self.num_points})..."
        )
        base_patches = extract_patches(
            points_v, all_features_v, labels_v,
            patch_size=self.patch_size,
            overlap=self.patch_overlap,
            min_points=10000,
            target_num_points=self.num_points
        )
        
        logger.info(f"  ✓ Extracted {len(base_patches)} base patches")
        
        # 5b. Add RGB if requested (to base patches only)
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
            
            for patch in base_patches:
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
        
        # 6. Create augmented versions of each patch
        num_versions = 1 + (self.num_augmentations if self.augment else 0)
        all_patches_collected = []
        
        for patch_idx, base_patch in enumerate(base_patches):
            # Version 0: Original patch
            original_patch = base_patch.copy()
            original_patch['_version'] = 'original'
            original_patch['_patch_idx'] = patch_idx
            all_patches_collected.append(original_patch)
            
            # Augmented versions: Apply augmentation to EACH PATCH individually
            if self.augment:
                for aug_idx in range(self.num_augmentations):
                    # Extract raw data from patch
                    patch_points = base_patch['points'].copy()
                    patch_intensity = base_patch['intensity'].copy()
                    patch_return_num = base_patch['return_number'].copy()
                    patch_labels = base_patch['labels'].copy()
                    patch_rgb = base_patch.get('rgb', None)
                    
                    if patch_rgb is not None:
                        patch_rgb = patch_rgb.copy()
                    
                    # Apply augmentation to THIS patch - get mask to align labels
                    result = augment_raw_points(
                        patch_points,
                        patch_intensity,
                        patch_return_num,
                        np.zeros(len(patch_points), dtype=np.uint8),  # dummy classification
                        rgb=patch_rgb,
                        nir=None,
                        ndvi=None,
                        return_mask=True
                    )
                    
                    # Unpack results including keep_mask
                    (aug_points, aug_intensity, aug_return, _, 
                     aug_rgb, _, _, keep_mask) = result
                    
                    # Apply same mask to labels to keep alignment
                    aug_labels = patch_labels[keep_mask]
                    
                    # Create augmented patch
                    aug_patch = {
                        'points': aug_points,
                        'intensity': aug_intensity,
                        'return_number': aug_return,
                        'labels': aug_labels,
                        '_version': f'aug_{aug_idx}',
                        '_patch_idx': patch_idx
                    }
                    
                    # Copy RGB if present
                    if aug_rgb is not None:
                        aug_patch['rgb'] = aug_rgb
                    
                    # Copy other features (normals, curvature, etc. need recomputation)
                    # For now, we skip recomputing features for augmented patches
                    # TODO: Optionally recompute geometric features on augmented geometry
                    
                    all_patches_collected.append(aug_patch)
        
        logger.info(
            f"  🔄 Created {len(all_patches_collected)} total patches "
            f"({len(base_patches)} base × {num_versions} versions)"
        )
        
        # 7. Save all collected patches
        output_dir.mkdir(parents=True, exist_ok=True)
        num_saved = 0
        
        rgb_suffix = " + RGB" if self.include_rgb else ""
        total_patches = len(all_patches_collected)
        logger.info(
            f"  💾 Saving {total_patches} patches{rgb_suffix} "
            f"({num_versions} versions)"
        )
        
        # Save patches with proper naming
        # Each patch has _version and _patch_idx metadata
        for patch in all_patches_collected:
            version = patch.pop('_version', 'original')
            base_idx = patch.pop('_patch_idx', 0)
            
            if version == 'original':
                patch_name = f"{laz_file.stem}_patch_{base_idx:04d}.npz"
            else:
                patch_name = (
                    f"{laz_file.stem}_patch_{base_idx:04d}_"
                    f"{version}.npz"
                )
            save_path = output_dir / patch_name
            save_patch(save_path, patch, self.lod_level)
            num_saved += 1
        
        tile_time = time.time() - tile_start
        pts_processed = len(original_data['points'])
        logger.info(
            f"  ✅ Completed: {num_saved} patches in {tile_time:.1f}s "
            f"(from {pts_processed:,} original points)"
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
                architecture=self.architecture,
                save_enriched=self.save_enriched_laz,
                only_enriched=self.only_enriched_laz,
                output_format=self.output_format,
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
            
            # Handle both dict and int return types for backwards compatibility
            num_patches_list = []
            for r in results:
                if isinstance(r, dict):
                    num_patches_list.append(r.get('num_patches', 0))
                else:
                    num_patches_list.append(r)
            
            total_patches = sum(num_patches_list)
            tiles_skipped = sum(1 for n in num_patches_list if n == 0)
            tiles_processed = total_tiles - tiles_skipped
        else:
            logger.info("🔄 Processing sequentially")
            logger.info("="*70)
            
            total_patches = 0
            for idx, laz_file in enumerate(laz_files, 1):
                result = self.process_tile(
                    laz_file, output_dir,
                    architecture=self.architecture,
                    save_enriched=self.save_enriched_laz,
                    only_enriched=self.only_enriched_laz,
                    output_format=self.output_format,
                    tile_idx=idx, total_tiles=total_tiles,
                    skip_existing=skip_existing
                )
                # Handle both dict and int return types for backwards compatibility
                if isinstance(result, dict):
                    num_patches = result.get('num_patches', 0)
                else:
                    num_patches = result
                    
                total_patches += num_patches
                
                if num_patches == 0:
                    tiles_skipped += 1
                else:
                    tiles_processed += 1
                
                # ✅ OPTIMIZATION: Explicit garbage collection every 5 tiles
                if idx % 5 == 0:
                    gc.collect()
                    try:
                        import psutil
                        mem = psutil.virtual_memory()
                        logger.debug(f"  Memory: {mem.available/(1024**3):.1f}GB available")
                    except ImportError:
                        pass
        
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

    def process_tile(
        self,
        laz_file: Path,
        output_dir: Path,
        architecture: str = 'pointnet++',
        save_enriched: Optional[bool] = None,
        only_enriched: Optional[bool] = None,
        output_format: str = 'npz',
        build_spatial_index: bool = False,
        tile_idx: int = 0,
        total_tiles: int = 0,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single LAZ tile with processing pipeline.
        
        This method implements the v2.0 unified pipeline that processes
        RAW LiDAR → Features → Architecture-formatted patches in a single
        pass, eliminating intermediate LAZ files and reducing I/O by 50%.
        
        Args:
            laz_file: Path to input LAZ file
            output_dir: Output directory for patches
            architecture: Target DL architecture ('pointnet++', 'octree', 
                         'transformer', 'sparse_conv', 'multi')
            save_enriched: If True, save intermediate enriched LAZ file
                          (default: None, uses self.save_enriched_laz)
            only_enriched: If True, only save enriched LAZ and skip patch creation
                          (default: None, uses self.only_enriched_laz)
            output_format: Output format - 'npz', 'hdf5', 'pytorch'/'torch', 'laz'
                          Supports multi-format: 'hdf5,laz' to save in both formats
                          (Note: PyTorch format requires torch to be installed)
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
                'skipped': bool,
                'enriched_only': bool (if only_enriched=True)
            }
        """
        from ..io.formatters.multi_arch_formatter import MultiArchitectureFormatter
        
        # Use instance variables if not explicitly provided
        if save_enriched is None:
            save_enriched = self.save_enriched_laz
        if only_enriched is None:
            only_enriched = self.only_enriched_laz
        
        progress_prefix = f"[{tile_idx}/{total_tiles}]" if total_tiles > 0 else ""
        
        # ===== INTELLIGENT SKIP: Check if patches already exist =====
        # This skip checker validates:
        # - Patches exist for this tile
        # - Patches are not corrupted (file size, content validation)
        # - Expected number of patches present (if augmentation enabled)
        # If valid patches exist, skip ALL processing including:
        # - LAZ loading, preprocessing, feature computation, patch extraction
        if skip_existing:
            should_skip, skip_info = self.skip_checker.should_skip_tile(
                laz_file,
                output_dir,
                expected_patches=None  # We don't know expected count yet
            )
            
            if should_skip:
                skip_msg = self.skip_checker.format_skip_message(laz_file, skip_info)
                logger.info(f"{progress_prefix} {skip_msg}")
                
                # Return early - NO preprocessing, NO feature computation, NO patch extraction
                return {
                    'num_patches': 0,
                    'processing_time': 0.0,
                    'points_processed': 0,
                    'skipped': True,
                    'skip_reason': skip_info.get('reason', 'unknown'),
                    'skip_info': skip_info
                }
            else:
                # Log reason for processing (e.g., corrupted patches, incomplete, etc.)
                skip_msg = self.skip_checker.format_skip_message(laz_file, skip_info)
                logger.info(f"{progress_prefix} {skip_msg}")
        
        logger.info(f"{progress_prefix} 🚀 Unified processing: {laz_file.name}")
        tile_start = time.time()
        
        # ===== STEP 1: Load RAW LiDAR (with memory-efficient chunking) =====
        logger.info(f"  📂 Loading RAW LiDAR...")
        
        # Check file size first to determine if chunking is needed
        file_size_mb = laz_file.stat().st_size / (1024 * 1024)
        use_chunked_loading = file_size_mb > 500  # Use chunking for files > 500MB
        
        if use_chunked_loading:
            logger.info(f"  ⚠️  Large file detected ({file_size_mb:.1f}MB), using chunked loading...")
        
        # Initialize variables (will be set in loading branches)
        points = None
        intensity = None
        return_number = None
        classification = None
        nir = None
        nir_from_laz = False
        las = None
        
        # Try to load LAZ file, with auto-recovery for corrupted files
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if use_chunked_loading:
                    # Use memory-mapped reading for large files
                    with laspy.open(str(laz_file)) as laz_reader:
                        # Read header to get point count
                        header = laz_reader.header
                        total_points = header.point_count
                        logger.info(f"  📊 File contains {total_points:,} points - loading in chunks...")
                        
                        # Define chunk size based on available memory (10M points per chunk)
                        chunk_size = 10_000_000
                        
                        # Pre-allocate arrays
                        all_points = []
                        all_intensity = []
                        all_return_number = []
                        all_classification = []
                        all_nir = [] if self.include_infrared else None
                        
                        # Read in chunks
                        for i, points_chunk in enumerate(laz_reader.chunk_iterator(chunk_size)):
                            chunk_num = i + 1
                            logger.info(f"    📦 Loading chunk {chunk_num}/{(total_points + chunk_size - 1) // chunk_size}...")
                            
                            chunk_xyz = np.vstack([points_chunk.x, points_chunk.y, points_chunk.z]).T.astype(np.float32)
                            all_points.append(chunk_xyz)
                            all_intensity.append(np.array(points_chunk.intensity, dtype=np.float32) / 65535.0)
                            all_return_number.append(np.array(points_chunk.return_number, dtype=np.float32))
                            all_classification.append(np.array(points_chunk.classification, dtype=np.uint8))
                            
                            # Try to load NIR if available
                            if self.include_infrared and all_nir is not None:
                                if hasattr(points_chunk, 'nir') or hasattr(points_chunk, 'near_infrared'):
                                    try:
                                        nir_raw = (points_chunk.nir if hasattr(points_chunk, 'nir') 
                                                  else points_chunk.near_infrared)
                                        all_nir.append(np.array(nir_raw, dtype=np.float32) / 65535.0)
                                    except:
                                        all_nir = None  # Disable NIR if any chunk fails
                            
                            # Clean up chunk memory
                            del points_chunk, chunk_xyz
                            gc.collect()
                        
                        # Concatenate all chunks
                        logger.info(f"  🔗 Concatenating {len(all_points)} chunks...")
                        points = np.vstack(all_points)
                        intensity = np.concatenate(all_intensity)
                        return_number = np.concatenate(all_return_number)
                        classification = np.concatenate(all_classification)
                        nir = np.concatenate(all_nir) if all_nir is not None and len(all_nir) > 0 else None
                        nir_from_laz = nir is not None
                        
                        if nir_from_laz:
                            logger.info(f"  ✓ NIR channel detected in LAZ file")
                        
                        # Clean up chunk lists
                        del all_points, all_intensity, all_return_number, all_classification, all_nir
                        gc.collect()
                        
                        # Create a minimal las object for compatibility
                        las = type('obj', (object,), {
                            'x': points[:, 0], 
                            'y': points[:, 1], 
                            'z': points[:, 2],
                            'intensity': (intensity * 65535).astype(np.uint16),
                            'return_number': return_number.astype(np.uint8),
                            'classification': classification
                        })()
                        if nir is not None:
                            las.nir = (nir * 65535).astype(np.uint16)
                else:
                    # Standard loading for smaller files
                    las = laspy.read(str(laz_file))
                    
                    # Extract basic data
                    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
                    intensity = np.array(las.intensity, dtype=np.float32) / 65535.0
                    return_number = np.array(las.return_number, dtype=np.float32)
                    classification = np.array(las.classification, dtype=np.uint8)
                    
                    # Try to load NIR if available and requested
                    nir = None
                    nir_from_laz = False
                    if self.include_infrared:
                        if hasattr(las, 'nir') or hasattr(las, 'near_infrared'):
                            try:
                                nir_raw = (las.nir if hasattr(las, 'nir') 
                                          else las.near_infrared)
                                nir = np.array(nir_raw, dtype=np.float32) / 65535.0
                                nir_from_laz = True
                                logger.info(f"  ✓ NIR channel detected in LAZ file")
                            except Exception as e:
                                logger.warning(f"  ⚠️  NIR channel in LAZ but failed to load: {e}")
                
                break  # Success
            except Exception as e:
                error_msg = str(e)
                is_corruption_error = (
                    'failed to fill whole buffer' in error_msg.lower() or
                    'ioerror' in error_msg.lower() or
                    'unexpected end of file' in error_msg.lower() or
                    'invalid' in error_msg.lower()
                )
                
                if is_corruption_error and attempt < max_retries - 1:
                    logger.warning(
                        f"  ⚠️  Corrupted LAZ file detected: {error_msg}"
                    )
                    logger.info(
                        f"  🔄 Attempting to re-download tile "
                        f"(attempt {attempt + 2}/{max_retries})..."
                    )
                    
                    # Try to re-download the tile
                    if self._redownload_tile(laz_file):
                        logger.info(f"  ✓ Tile re-downloaded successfully")
                        continue  # Retry loading
                    else:
                        logger.error(f"  ✗ Failed to re-download tile")
                        return {
                            'num_patches': 0,
                            'processing_time': 0.0,
                            'points_processed': 0,
                            'skipped': False,
                            'error': f'Corrupted file, re-download failed: {error_msg}'
                        }
                else:
                    logger.error(f"  ✗ Failed to read {laz_file}: {e}")
                    return {
                        'num_patches': 0,
                        'processing_time': 0.0,
                        'points_processed': 0,
                        'skipped': False,
                        'error': str(e)
                    }
        
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
        
        # ===== STEP 2: Preprocessing (if enabled) - BEFORE augmentation =====
        # Apply preprocessing to the base data before augmentation so that:
        # 1. All augmented versions inherit the preprocessing
        # 2. Stitching can use the preprocessed data directly
        preprocessing_mask = None
        if self.preprocess:
            logger.info("  🧹 Preprocessing...")
            from ..preprocessing.preprocessing import (
                statistical_outlier_removal,
                radius_outlier_removal,
                voxel_downsample
            )
            
            cfg = self.preprocess_config or {}
            original_count = len(points)
            
            # STEP 1: Apply voxel downsampling FIRST if enabled (GPU accelerated!)
            # This drastically reduces point count before expensive operations
            if cfg.get('voxel_enabled', False):
                voxel_size = cfg.get('voxel_size', 0.25)
                before_voxel = len(points)
                
                # GPU-accelerated voxel downsampling
                use_gpu_voxel = self.use_gpu and self.use_gpu_chunked
                if use_gpu_voxel:
                    logger.info(f"  📦 GPU voxel downsampling (size={voxel_size}m)...")
                else:
                    logger.info(f"  📦 CPU voxel downsampling (size={voxel_size}m)...")
                
                # Voxel downsample returns: (downsampled_points, keep_indices)
                points_voxel, keep_indices = voxel_downsample(
                    points, 
                    voxel_size=voxel_size,
                    method='centroid',
                    use_gpu=use_gpu_voxel
                )
                
                # Apply keep_indices to other arrays
                points = points_voxel
                intensity = intensity[keep_indices]
                return_number = return_number[keep_indices]
                classification = classification[keep_indices]
                if nir is not None:
                    nir = nir[keep_indices]
            
            # STEP 2: Now apply outlier removal on the reduced point set
            # Note: Outlier removal uses CPU (kd-tree). GPU doesn't help much here.
            cumulative_mask = np.ones(len(points), dtype=bool)
            
            # SOR - Statistical Outlier Removal
            if cfg.get('enabled', True):
                logger.info(f"  🧹 Statistical outlier removal (k={cfg.get('sor_k', 12)})...")
                _, sor_mask = statistical_outlier_removal(
                    points,
                    k=cfg.get('sor_k', 12),
                    std_multiplier=cfg.get('sor_std', 2.0)
                )
                cumulative_mask &= sor_mask
                logger.info(f"  ✓ SOR: kept {np.sum(sor_mask):,}/{len(points):,} points")
            
            # ROR - Radius Outlier Removal  
            if cfg.get('enabled', True):
                logger.info(f"  🧹 Radius outlier removal (r={cfg.get('ror_radius', 1.0)}m)...")
                _, ror_mask = radius_outlier_removal(
                    points,
                    radius=cfg.get('ror_radius', 1.0),
                    min_neighbors=cfg.get('ror_neighbors', 4)
                )
                cumulative_mask &= ror_mask
                logger.info(f"  ✓ ROR: kept {np.sum(ror_mask):,}/{len(points):,} points")
            
            before_outlier_removal = len(points)
            preprocessing_mask = cumulative_mask.copy()
            
            points = points[cumulative_mask]
            intensity = intensity[cumulative_mask]
            return_number = return_number[cumulative_mask]
            classification = classification[cumulative_mask]
            if nir is not None:
                nir = nir[cumulative_mask]
            
            outlier_reduction = 1 - len(points) / before_outlier_removal
            logger.info(
                f"  ✓ Outlier removal: {len(points):,}/{before_outlier_removal:,} "
                f"({outlier_reduction:.1%} reduction)"
            )
            
            total_reduction = 1 - len(points) / original_count
            logger.info(
                f"  ✓ Preprocessing complete: {len(points):,}/{original_count:,} points "
                f"({total_reduction:.1%} total reduction)"
            )
        
        # ===== STEP 3: Fetch RGB and NIR ONCE (before augmentation loop) =====
        # RGB and NIR are fetched from orthophotos based on point coordinates.
        # Since augmentation only changes geometry slightly (rotations, jitter),
        # we fetch RGB/NIR once and reuse for all augmented versions.
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
        
        # Fetch NIR (if requested and not already from LAZ)
        if self.include_infrared and not nir_from_laz and self.infrared_fetcher:
            logger.info("  📡 Fetching NIR from infrared orthophotos...")
            try:
                tile_bbox = (
                    points[:, 0].min(),
                    points[:, 1].min(),
                    points[:, 0].max(),
                    points[:, 1].max()
                )
                nir = self.infrared_fetcher.augment_points_with_infrared(
                    points, bbox=tile_bbox
                )
                if nir is not None:
                    nir = nir.astype(np.float32) / 255.0  # Normalize to [0, 1]
                    logger.info(f"  ✓ NIR fetched from orthophotos")
                else:
                    logger.warning(f"  ⚠️  NIR fetch returned None")
            except Exception as e:
                logger.warning(f"  ⚠️  NIR fetch failed: {e}")
        
        # Compute NDVI (if requested and both NIR & RGB available)
        ndvi = None
        if self.compute_ndvi and nir is not None and rgb is not None:
            # NDVI = (NIR - Red) / (NIR + Red)
            red = rgb[:, 0]
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir - red) / (nir + red + 1e-8)
                ndvi = np.clip(ndvi, -1, 1)
            logger.info(f"  ✓ NDVI computed")
        elif self.compute_ndvi:
            if nir is None:
                logger.warning(f"  ⚠️  NDVI requested but NIR not available")
            if rgb is None:
                logger.warning(f"  ⚠️  NDVI requested but RGB not available")
        
        # Store preprocessed data for potential augmentation
        original_data = {
            'points': points.copy(),
            'intensity': intensity.copy(),
            'return_number': return_number.copy(),
            'classification': classification.copy(),
            'nir': nir.copy() if nir is not None else None,
            'rgb': rgb.copy() if rgb is not None else None,
            'ndvi': ndvi.copy() if ndvi is not None else None
        }
        
        # Memory cleanup after loading and preprocessing
        del las
        if preprocessing_mask is not None:
            del preprocessing_mask
        aggressive_memory_cleanup()
        
        # Log memory status if psutil available
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"  💾 Memory: {mem.available/(1024**3):.1f}GB available ({mem.percent:.1f}% used)")
        except ImportError:
            pass
        
        logger.debug(f"  🧹 Memory cleaned after loading/preprocessing")
        
        # ===== ENRICHED LAZ ONLY MODE: Save enriched LAZ and exit early =====
        # If only_enriched_laz is enabled, save the enriched LAZ file with all computed features
        # and then skip ALL patch processing
        if only_enriched:
            logger.info("  🎯 Only enriched LAZ mode - computing features and saving enriched file...")
            
            # Extract data from original_data dict for feature computation
            points = original_data['points']
            intensity = original_data['intensity']
            return_number = original_data['return_number']
            classification = original_data['classification']
            rgb = original_data['rgb']
            nir = original_data['nir']
            ndvi = original_data['ndvi']
            
            # ===== STEP 4: Compute geometric features =====
            feature_start = time.time()
            logger.info("  🔧 Computing geometric features...")
            
            # Determine if we should use boundary-aware stitching
            use_boundary_aware = False
            if self.use_stitching and self.stitcher is not None:
                # Check if neighbors exist for boundary-aware processing
                neighbors_exist = self.stitcher.check_neighbors_exist(laz_file)
                if neighbors_exist:
                    logger.info("  🔗 Using tile stitching for boundary features...")
                    use_boundary_aware = True
                    
                    try:
                        # Load adjacent tiles and compute boundary-aware features
                        features = self.stitcher.compute_boundary_aware_features(
                            laz_file=laz_file,
                            k=self.k_neighbors if self.k_neighbors else 20
                        )
                        
                        # Extract feature components
                        normals = features['normals']
                        curvature = features['curvature']
                        
                        # Extract geometric features
                        if 'geometric_features' in features:
                            geo_dict = features['geometric_features']
                            # Convert dict to array or keep as dict based on what we have
                            if isinstance(geo_dict, dict):
                                geo_features = geo_dict
                            else:
                                geo_features = {
                                    'planarity': geo_dict[:, 0],
                                    'linearity': geo_dict[:, 1],
                                    'sphericity': geo_dict[:, 2],
                                    'verticality': geo_dict[:, 3] if geo_dict.shape[1] > 3 else np.abs(normals[:, 2])
                                }
                        else:
                            geo_features = None
                        
                        # Height feature (relative to local minimum)
                        height = points[:, 2] - points[:, 2].min()
                        
                        num_boundary = features.get('num_boundary_points', 0)
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
                    # Check if we should use chunked GPU processing for large tiles
                    num_points = len(points)
                    use_chunked = (
                        self.use_gpu_chunked and 
                        num_points > 500_000  # Use GPU chunked for >500K points
                    )
                    
                    if use_chunked:
                        # Use chunked GPU processing for large tiles
                        try:
                            from ..features.features_gpu_chunked import (
                                GPUChunkedFeatureComputer,
                                GPU_AVAILABLE,
                                CUML_AVAILABLE
                            )
                            
                            if GPU_AVAILABLE and CUML_AVAILABLE:
                                logger.info(
                                    f"🚀 Using GPU chunked processing "
                                    f"({num_points:,} points, "
                                    f"batch_size={self.gpu_batch_size:,})"
                                )
                                
                                # Initialize chunked computer
                                computer = GPUChunkedFeatureComputer(
                                    chunk_size=self.gpu_batch_size,
                                    use_gpu=True,
                                    show_progress=False
                                )
                                
                                # Compute all features with chunked processing
                                k = self.k_neighbors if self.k_neighbors else 20
                                normals, curvature, height, geo_features = (
                                    computer.compute_all_features_chunked(
                                        points=points,
                                        classification=classification,
                                        k=k
                                    )
                                )
                                
                                # Add verticality if not present
                                if isinstance(geo_features, dict) and 'verticality' not in geo_features:
                                    verticality = np.abs(normals[:, 2])
                                    geo_features['verticality'] = verticality
                            else:
                                logger.warning(
                                    "GPU chunked requested but not available. Using standard GPU."
                                )
                                use_chunked = False
                        except Exception as e:
                            logger.warning(
                                f"GPU chunked processing failed: {e}. Falling back to standard GPU."
                            )
                            use_chunked = False
                    
                    if not use_chunked:
                        # Use standard GPU processing
                        from ..features.features import compute_all_features_with_gpu
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
                    # CPU processing
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
            logger.info(f"  ⏱️  Features computed: {feature_time:.1f}s")
            
            # ===== STEP 5: Save Enriched LAZ =====
            logger.info("  💾 Saving enriched LAZ file...")
            enriched_path = output_dir / f"{laz_file.stem}_enriched.laz"
            
            # Create new LAS with features
            original_format_id = las.header.point_format.id
            
            # If we have RGB, use a format that supports it
            if rgb is not None:
                # Map to RGB-compatible formats
                if original_format_id in [0, 1]:
                    target_format = 2  # Basic + RGB
                elif original_format_id in [6]:
                    target_format = 7  # LAS 1.4 + RGB
                elif original_format_id in [2, 3, 5, 7, 8, 10]:
                    target_format = original_format_id  # Already supports RGB
                else:
                    target_format = 7  # Default to LAS 1.4 with RGB
            else:
                target_format = original_format_id
            
            # Create a fresh header with the appropriate point format
            new_header = laspy.LasHeader(version=las.header.version, point_format=target_format)
            new_header.offsets = las.header.offsets
            new_header.scales = las.header.scales
            
            # Create new LAS with the correct size
            new_las = laspy.LasData(new_header)
            new_las.x = points[:, 0]
            new_las.y = points[:, 1]
            new_las.z = points[:, 2]
            new_las.intensity = (intensity * 65535.0).astype(np.uint16)
            new_las.return_number = return_number.astype(np.uint8)
            new_las.classification = classification
            
            # Add RGB if available
            if rgb is not None:
                new_las.red = (rgb[:, 0] * 65535.0).astype(np.uint16)
                new_las.green = (rgb[:, 1] * 65535.0).astype(np.uint16)
                new_las.blue = (rgb[:, 2] * 65535.0).astype(np.uint16)
            
            # Add extra dimensions for features
            try:
                expected_size = len(points)
                
                # Core geometric features (always computed)
                new_las.add_extra_dim(laspy.ExtraBytesParams(name="normal_x", type=np.float32))
                new_las.add_extra_dim(laspy.ExtraBytesParams(name="normal_y", type=np.float32))
                new_las.add_extra_dim(laspy.ExtraBytesParams(name="normal_z", type=np.float32))
                new_las.add_extra_dim(laspy.ExtraBytesParams(name="curvature", type=np.float32))
                new_las.add_extra_dim(laspy.ExtraBytesParams(name="height", type=np.float32))
                
                new_las.normal_x = normals[:, 0].astype(np.float32)
                new_las.normal_y = normals[:, 1].astype(np.float32)
                new_las.normal_z = normals[:, 2].astype(np.float32)
                new_las.curvature = curvature.astype(np.float32)
                new_las.height = height.astype(np.float32)
                
                # Add geometric features if computed
                if geo_features is not None:
                    if isinstance(geo_features, dict):
                        for feature_name, feature_values in geo_features.items():
                            if len(feature_values) == expected_size:
                                new_las.add_extra_dim(laspy.ExtraBytesParams(
                                    name=feature_name, type=np.float32
                                ))
                                setattr(new_las, feature_name, feature_values.astype(np.float32))
                    else:
                        feature_names = ['planarity', 'linearity', 'sphericity', 'verticality']
                        for i, feature_name in enumerate(feature_names[:geo_features.shape[1]]):
                            if len(geo_features[:, i]) == expected_size:
                                new_las.add_extra_dim(laspy.ExtraBytesParams(
                                    name=feature_name, type=np.float32
                                ))
                                setattr(new_las, feature_name, geo_features[:, i].astype(np.float32))
                
                # Add NIR if available
                if nir is not None and len(nir) == expected_size:
                    new_las.add_extra_dim(laspy.ExtraBytesParams(name="nir", type=np.float32))
                    new_las.nir = nir.astype(np.float32)
                
                # Add NDVI if computed
                if ndvi is not None and len(ndvi) == expected_size:
                    new_las.add_extra_dim(laspy.ExtraBytesParams(name="ndvi", type=np.float32))
                    new_las.ndvi = ndvi.astype(np.float32)
                
                # Write the enriched LAZ file
                new_las.write(enriched_path)
                logger.info(f"  ✅ Enriched LAZ saved: {enriched_path.name}")
                
                # Verify the file
                if enriched_path.exists():
                    file_size_mb = enriched_path.stat().st_size / (1024 * 1024)
                    verify_las = laspy.read(str(enriched_path))
                    extra_dims = verify_las.point_format.extra_dimension_names
                    logger.info(
                        f"  📊 Enriched LAZ: {len(verify_las.points):,} points, "
                        f"{len(extra_dims)} extra dimensions, {file_size_mb:.1f} MB"
                    )
                    logger.info(f"     Extra dimensions: {extra_dims}")
                else:
                    logger.error(f"  ❌ Failed to write enriched LAZ file!")
                
            except Exception as e:
                logger.error(f"  ❌ Error saving enriched LAZ: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Clean up and return
            tile_time = time.time() - tile_start
            num_points_processed = len(points)
            original_data.clear()
            aggressive_memory_cleanup()
            
            logger.info(
                f"  ✅ Enrichment complete (only_enriched_laz mode - NO patches created): "
                f"{tile_time:.1f}s ({num_points_processed/tile_time:.0f} pts/s)"
            )
            
            return {
                'num_patches': 0,
                'processing_time': tile_time,
                'points_processed': num_points_processed,
                'skipped': False,
                'enriched_only': True,
                'enriched_laz_path': str(enriched_path)
            }
        
        # Determine number of versions to process (original + augmentations)
        num_versions = 1 + (self.num_augmentations if self.augment else 0)
        all_patches_collected = []
        num_saved = 0  # Track total patches saved across all versions
        
        # Prepare formatter and output directory BEFORE the loop
        # to enable incremental saving
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which architectures to format and initialize appropriate formatter
        if architecture == 'hybrid':
            # Hybrid mode: use HybridFormatter (single comprehensive format)
            from ..io.formatters import HybridFormatter
            formatter = HybridFormatter(
                num_points=self.num_points,
                use_rgb=(rgb is not None),
                use_infrared=(nir is not None),
                use_geometric=True,
                use_radiometric=True,
                use_contextual=True
            )
            target_archs = ['hybrid']  # Will save as single hybrid format
        elif architecture == 'multi':
            # Multi mode: format for all architectures separately
            from ..io.formatters import MultiArchitectureFormatter
            target_archs = ['pointnet++', 'octree', 'transformer', 'sparse_conv']
            formatter = MultiArchitectureFormatter(
                target_archs=target_archs,
                num_points=self.num_points,
                use_rgb=(rgb is not None),
                use_infrared=(nir is not None),
                use_geometric=True,
                use_radiometric=True,
                use_contextual=True
            )
        else:
            # Single architecture mode
            from ..io.formatters import MultiArchitectureFormatter
            target_archs = [architecture]
            formatter = MultiArchitectureFormatter(
                target_archs=target_archs,
                num_points=self.num_points,
                use_rgb=(rgb is not None),
                use_infrared=(nir is not None),
                use_geometric=True,
                use_radiometric=True,
                use_contextual=True
            )
        
        # ===== AUGMENTATION LOOP: Process original + augmented versions =====
        for version_idx in range(num_versions):
            # Apply augmentation to raw data if not the first version
            if version_idx == 0:
                # Original version - no augmentation
                points = original_data['points'].copy()
                intensity = original_data['intensity'].copy()
                return_number = original_data['return_number'].copy()
                classification = original_data['classification'].copy()
                nir = original_data['nir'].copy() if original_data['nir'] is not None else None
                rgb = original_data['rgb'].copy() if original_data['rgb'] is not None else None
                ndvi = original_data['ndvi'].copy() if original_data['ndvi'] is not None else None
                version_label = "original"
            else:
                # Augmented version - apply transformations BEFORE features
                # Note: RGB, NIR, and NDVI must also be filtered by dropout mask
                (points, intensity,
                 return_number, classification, rgb, nir, ndvi) = augment_raw_points(
                    original_data['points'],
                    original_data['intensity'],
                    original_data['return_number'],
                    original_data['classification'],
                    rgb=original_data['rgb'],
                    nir=original_data['nir'],
                    ndvi=original_data['ndvi']
                )
                version_label = f"aug_{version_idx-1}"
                logger.info(
                    f"  🔄 Augmented v{version_idx}/{num_versions-1} "
                    f"({len(points):,} points after dropout) - using standard features"
                )
            
            # ===== STEP 4: Compute Features (in-memory) =====
            if version_idx == 0:
                logger.info("  🔧 Computing features...")
            feature_start = time.time()
            
            # Determine if we should use tile stitching for boundary-aware features
            # NOTE: Stitching is ONLY available for the original (non-augmented) version
            # because augmentation changes point counts (dropout), making it impossible
            # to align with the reloaded tile data
            use_boundary_aware = (
                self.use_stitching 
                and laz_file.parent.exists() 
                and version_idx == 0  # Only for original version
            )
            
            if use_boundary_aware:
                logger.info("  🔗 Enabling tile stitching for boundary-aware features...")
                try:
                    from .tile_stitcher import TileStitcher
                    from ..features.features_boundary import BoundaryAwareFeatureComputer
                    
                    # Initialize stitcher
                    stitcher = TileStitcher(buffer_size=self.buffer_size)
                    
                    # ✅ NEW APPROACH: Pass preprocessed core points directly to avoid reloading
                    # This eliminates coordinate mismatch issues caused by:
                    # 1. Different loading order between initial load and stitcher reload
                    # 2. Floating point precision differences
                    # 3. Need for expensive KD-tree alignment
                    
                    # Load neighbors ONLY (not the core tile)
                    tile_data = stitcher.load_tile_with_neighbors(
                        tile_path=laz_file,
                        auto_detect_neighbors=True,
                        use_provided_core_points=True,  # Tell stitcher we'll provide core points
                        core_points=points  # Pass preprocessed points directly
                    )
                    
                    # Core points are now exactly the preprocessed points (no alignment needed!)
                    core_points = tile_data['core_points']  # Should be identical to 'points'
                    buffer_points = tile_data['buffer_points']
                    
                    # Get tile bounds from stitcher (computed from provided core points)
                    tile_bounds = stitcher.get_tile_bounds(laz_file)
                    
                    # Verify that core_points match preprocessed points exactly
                    if not np.array_equal(core_points, points):
                        logger.error(
                            f"  ❌ Core points mismatch after stitcher! "
                            f"This should never happen. "
                            f"core: {len(core_points)}, preproc: {len(points)}"
                        )
                        use_boundary_aware = False
                    
                    if use_boundary_aware:
                        # At this point:
                        # - core_points are exactly the preprocessed points (passed directly)
                        # - core_points[i] == points[i] (guaranteed by np.array_equal check above)
                        # - buffer_points contains the buffer zone from neighbors
                        
                        num_core_points = len(core_points)
                        num_buffer_points = len(buffer_points) if buffer_points is not None else 0
                        
                        logger.info(
                            f"  🔧 Using preprocessed core points: {len(core_points):,} points "
                            f"+ {num_buffer_points:,} buffer points"
                        )
                        
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
                        
                        # Features are now already aligned with preprocessed points
                        # (because we filtered core_points before computing features)
                        
                        # Extract feature arrays
                        normals = features['normals']
                        curvature = features['curvature']
                        
                        # Verify feature sizes match preprocessed data
                        if len(normals) != len(points):
                            logger.error(
                                f"❌ Feature size mismatch! "
                                f"normals={len(normals)}, points={len(points)}, "
                                f"core_points={len(core_points)}"
                            )
                            raise ValueError(
                                f"Feature computation failed: size mismatch "
                                f"(normals: {len(normals)}, points: {len(points)})"
                            )
                        
                        # Build geo_features dictionary (not array!)
                        # Only include features that are present (validation may drop some)
                        if self.include_extra_features:
                            geo_features = {}
                            for feat_name in ['planarity', 'linearity', 'sphericity', 'verticality']:
                                if feat_name in features:
                                    geo_features[feat_name] = features[feat_name]
                            
                            # If all features were dropped, set to None
                            if not geo_features:
                                geo_features = None
                                logger.warning(
                                    "  ⚠️  All geometric features dropped due to artifacts"
                                )
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
                    # Check if we should use chunked GPU processing for large tiles
                    num_points = len(points)
                    use_chunked = (
                        self.use_gpu_chunked and 
                        num_points > 500_000  # Use GPU chunked for >500K points (aggressive for speed!)
                    )
                    
                    if use_chunked:
                        # Use chunked GPU processing for large tiles
                        try:
                            from ..features.features_gpu_chunked import (
                                GPUChunkedFeatureComputer,
                                GPU_AVAILABLE,
                                CUML_AVAILABLE
                            )
                            
                            if GPU_AVAILABLE and CUML_AVAILABLE:
                                logger.info(
                                    f"🚀 Using GPU chunked processing "
                                    f"({num_points:,} points, "
                                    f"batch_size={self.gpu_batch_size:,})"
                                )
                                
                                # Initialize chunked computer
                                computer = GPUChunkedFeatureComputer(
                                    chunk_size=self.gpu_batch_size,
                                    use_gpu=True,
                                    show_progress=False
                                )
                                
                                # Compute all features with chunked processing
                                k = self.k_neighbors if self.k_neighbors else 20
                                normals, curvature, height, geo_features = (
                                    computer.compute_all_features_chunked(
                                        points=points,
                                        classification=classification,
                                        k=k
                                    )
                                )
                                
                                # Add verticality if not present
                                if 'verticality' not in geo_features:
                                    verticality = np.abs(normals[:, 2])
                                    geo_features['verticality'] = verticality
                            else:
                                # Fall back to standard GPU if chunked not available
                                logger.warning(
                                    "GPU chunked requested but not available. "
                                    "Using standard GPU."
                                )
                                use_chunked = False
                        except Exception as e:
                            logger.warning(
                                f"GPU chunked processing failed: {e}. "
                                "Falling back to standard GPU."
                            )
                            use_chunked = False
                    
                    if not use_chunked:
                        # Use standard GPU processing
                        from ..features.features import compute_all_features_with_gpu
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
                    # CPU processing
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
            if version_idx == 0:
                logger.info(f"  ⏱️  Features: {feature_time:.1f}s")
            else:
                logger.info(f"  ⏱️  Features for {version_label}: {feature_time:.1f}s")
            
            # Memory cleanup after feature computation
            if use_boundary_aware and 'computer' in locals():
                del computer
            gc.collect()
            
            # Note: RGB, NIR, and NDVI were already fetched before the augmentation loop
            # and are preserved in the version-specific variables above
            
            # ===== STEP 5: Save Enriched LAZ (optional, only for original version) =====
            if save_enriched and version_idx == 0:
                logger.info("  💾 Saving enriched LAZ...")
                enriched_path = output_dir / "enriched" / f"{laz_file.stem}_enriched.laz"
                enriched_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create new LAS with features (using FILTERED points only)
                # After preprocessing, points/intensity/etc are filtered arrays
                # so we need to create a new LasData from scratch with the correct size
                # Determine appropriate point format based on whether RGB is needed
                original_format_id = las.header.point_format.id
                
                # If we have RGB, use a format that supports it
                if rgb is not None:
                    # Map to RGB-compatible formats
                    # Format 2,3,5 (LAS 1.2-1.3) or 7,8,10 (LAS 1.4) support RGB
                    if original_format_id in [0, 1]:
                        target_format = 2  # Basic + RGB
                    elif original_format_id in [6]:
                        target_format = 7  # LAS 1.4 + RGB
                    elif original_format_id in [2, 3, 5, 7, 8, 10]:
                        target_format = original_format_id  # Already supports RGB
                    else:
                        target_format = 7  # Default to LAS 1.4 with RGB
                else:
                    target_format = original_format_id
                
                # Create a fresh header with the appropriate point format
                new_header = laspy.LasHeader(version=las.header.version, point_format=target_format)
                new_header.offsets = las.header.offsets
                new_header.scales = las.header.scales
                
                # Create new LAS with the correct size
                new_las = laspy.LasData(new_header)
                new_las.x = points[:, 0]
                new_las.y = points[:, 1]
                new_las.z = points[:, 2]
                new_las.intensity = (intensity * 65535.0).astype(np.uint16)
                new_las.return_number = return_number.astype(np.uint8)
                new_las.classification = classification
                
                # Add RGB FIRST if computed (before extra dimensions)
                if rgb is not None:
                    # Now the point format should support RGB
                    new_las.red = (rgb[:, 0] * 65535.0).astype(np.uint16)
                    new_las.green = (rgb[:, 1] * 65535.0).astype(np.uint16)
                    new_las.blue = (rgb[:, 2] * 65535.0).astype(np.uint16)
                
                # Add extra dimensions for features
                try:
                    # Validate data sizes before adding features
                    expected_size = len(points)
                    if len(normals) != expected_size:
                        raise ValueError(f"Normals size mismatch: {len(normals)} != {expected_size}")
                    if len(curvature) != expected_size:
                        raise ValueError(f"Curvature size mismatch: {len(curvature)} != {expected_size}")
                    if len(height) != expected_size:
                        raise ValueError(f"Height size mismatch: {len(height)} != {expected_size}")
                    
                    # Core geometric features (always computed)
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
                    new_las.add_extra_dim(laspy.ExtraBytesParams(
                        name="height", type=np.float32
                    ))
                    
                    # Assign feature values with explicit type conversion
                    new_las.normal_x = normals[:, 0].astype(np.float32)
                    new_las.normal_y = normals[:, 1].astype(np.float32)
                    new_las.normal_z = normals[:, 2].astype(np.float32)
                    new_las.curvature = curvature.astype(np.float32)
                    new_las.height = height.astype(np.float32)
                    
                    logger.debug(f"  ✓ Added core features: normals, curvature, height")
                    
                    # Add geometric features if computed
                    if geo_features is not None:
                        if isinstance(geo_features, dict):
                            # Dictionary format (from GPU/boundary-aware processing)
                            for feature_name, feature_values in geo_features.items():
                                # Validate size
                                if len(feature_values) != expected_size:
                                    logger.warning(
                                        f"  ⚠️  Skipping feature {feature_name}: "
                                        f"size mismatch ({len(feature_values)} != {expected_size})"
                                    )
                                    continue
                                
                                new_las.add_extra_dim(laspy.ExtraBytesParams(
                                    name=feature_name, type=np.float32
                                ))
                                setattr(new_las, feature_name, feature_values.astype(np.float32))
                            logger.debug(f"  ✓ Added {len(geo_features)} geometric features (dict)")
                        else:
                            # Array format (from CPU processing)
                            feature_names = ['planarity', 'linearity', 'sphericity', 'verticality']
                            for i, feature_name in enumerate(feature_names[:geo_features.shape[1]]):
                                # Validate size
                                if len(geo_features[:, i]) != expected_size:
                                    logger.warning(
                                        f"  ⚠️  Skipping feature {feature_name}: size mismatch"
                                    )
                                    continue
                                
                                new_las.add_extra_dim(laspy.ExtraBytesParams(
                                    name=feature_name, type=np.float32
                                ))
                                setattr(new_las, feature_name, geo_features[:, i].astype(np.float32))
                            logger.debug(f"  ✓ Added {geo_features.shape[1]} geometric features (array)")
                    else:
                        logger.debug(f"  ⚠️  No geometric features (geo_features is None)")
                    
                    # Add NIR if available
                    if nir is not None:
                        if len(nir) != expected_size:
                            logger.warning(f"  ⚠️  Skipping NIR: size mismatch ({len(nir)} != {expected_size})")
                        else:
                            new_las.add_extra_dim(laspy.ExtraBytesParams(
                                name="nir", type=np.float32
                            ))
                            new_las.nir = nir.astype(np.float32)
                            logger.debug(f"  ✓ Added NIR")
                    
                    # Add NDVI if computed
                    if ndvi is not None:
                        if len(ndvi) != expected_size:
                            logger.warning(f"  ⚠️  Skipping NDVI: size mismatch ({len(ndvi)} != {expected_size})")
                        else:
                            new_las.add_extra_dim(laspy.ExtraBytesParams(
                                name="ndvi", type=np.float32
                            ))
                            new_las.ndvi = ndvi.astype(np.float32)
                            logger.debug(f"  ✓ Added NDVI")
                    
                    # Verify extra dimensions were added
                    extra_dims_added = new_las.point_format.extra_dimension_names
                    logger.debug(f"  📊 Total extra dimensions added: {len(extra_dims_added)}")
                    logger.debug(f"     Extra dimensions: {extra_dims_added}")
                    
                    # Write the LAZ file
                    logger.debug(f"  💾 Writing enriched LAZ to: {enriched_path}")
                    new_las.write(enriched_path)
                    
                    # Verify file was written and has features
                    if enriched_path.exists():
                        # Re-read to verify features were saved
                        verify_las = laspy.read(str(enriched_path))
                        verify_extra_dims = verify_las.point_format.extra_dimension_names
                        
                        if len(verify_extra_dims) == 0:
                            logger.error(
                                f"  ❌ CRITICAL: Enriched LAZ was written but contains NO extra dimensions!"
                            )
                            logger.error(f"     This is a BUG - features were added but not persisted")
                        else:
                            logger.debug(f"  ✓ Verified: {len(verify_extra_dims)} extra dimensions in written file")
                    
                    # Log what was saved
                    features_saved = ['normals', 'curvature', 'height']
                    if geo_features is not None:
                        if isinstance(geo_features, dict):
                            features_saved.extend(geo_features.keys())
                        else:
                            features_saved.extend(['planarity', 'linearity', 'sphericity', 'verticality'])
                    if rgb is not None:
                        features_saved.append('RGB')
                    if nir is not None:
                        features_saved.append('NIR')
                    if ndvi is not None:
                        features_saved.append('NDVI')
                    
                    logger.info(
                        f"  ✓ Enriched LAZ saved: {enriched_path.name} "
                        f"({len(features_saved)} feature groups: {', '.join(features_saved)})"
                    )
                except Exception as e:
                    logger.error(f"  ❌ Failed to save enriched LAZ: {e}")
                    logger.error(f"     Error type: {type(e).__name__}")
                    logger.error(f"     File: {enriched_path}")
                    # Log more details for debugging
                    import traceback
                    logger.error(f"     Traceback:\n{traceback.format_exc()}")
            
            # ===== STEP 6: Remap Labels =====
            labels = np.array([
                self.class_mapping.get(c, self.default_class)
                for c in classification
            ], dtype=np.uint8)
            
            # ===== STEP 7: Extract Patches =====
            # Note: If only_enriched_laz=True, we already returned early above
            # This code only runs when creating patches
            # Log extraction start for ALL versions (not just original)
            if version_idx == 0:
                logger.info(
                    f"  📦 Extracting patches "
                    f"(size={self.patch_size}m, points={self.num_points})..."
                )
            else:
                logger.info(
                    f"  📦 Extracting patches from {version_label} "
                    f"({len(points):,} points)..."
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
            
            # Tag patches with version label AND spatial index
            # The spatial index is determined from the patch order (consistent across augmentations)
            for patch_idx, patch in enumerate(patches):
                patch['_version'] = version_label
                patch['_spatial_idx'] = patch_idx  # Track spatial location
            
            # Always log extraction result for debugging
            if version_idx == 0:
                logger.info(f"  ✓ Extracted {len(patches)} patches from original")
            else:
                logger.info(f"  ✓ Extracted {len(patches)} patches from {version_label}")
            
            # ===== INCREMENTAL SAVE: Save patches immediately after extraction =====
            # This reduces memory pressure by not accumulating all patches in memory
            num_patches_this_version = len(patches)
            logger.info(f"  💾 Saving {num_patches_this_version} patches from {version_label}...")
            
            for patch in patches:
                version = patch.pop('_version', 'original')
                spatial_idx = patch.pop('_spatial_idx', 0)
                
                # Format patch
                formatted = formatter.format_patch(patch)
                
                # Handle hybrid vs multi-arch vs single-arch formatter
                if architecture == 'hybrid':
                    # Hybrid mode: HybridFormatter returns a single comprehensive dict
                    # Save as one file with all data (points, features, labels, rgb, nir, normals, etc.)
                    patches_to_save = [('hybrid', formatted)]
                elif architecture == 'multi':
                    # Multi mode: MultiArchitectureFormatter returns dict with architecture keys
                    # Save separate files for each architecture
                    patches_to_save = [(arch, formatted[arch]) for arch in target_archs]
                else:
                    # Single architecture mode: MultiArchitectureFormatter returns dict with architecture keys
                    # Extract the specific architecture data
                    patches_to_save = [(architecture, formatted[architecture])]
                
                # Save based on output format (supports multi-format)
                # Parse output_format: can be single format or comma-separated list
                formats_to_save = [fmt.strip() for fmt in output_format.split(',')]
                
                for arch, arch_data in patches_to_save:
                    # Determine base filename (without extension)
                    if version == 'original':
                        base_filename = f"{laz_file.stem}_{arch}_patch_{spatial_idx:04d}"
                    else:
                        base_filename = f"{laz_file.stem}_{arch}_patch_{spatial_idx:04d}_{version}"
                    
                    # Save in each requested format
                    for fmt in formats_to_save:
                        if fmt == 'npz':
                            save_path = output_dir / f"{base_filename}.npz"
                            np.savez_compressed(save_path, **arch_data)
                            num_saved += 1
                        elif fmt == 'hdf5':
                            save_path = output_dir / f"{base_filename}.h5"
                            with h5py.File(save_path, 'w') as f:
                                # Extract metadata (don't modify arch_data - use get instead of pop)
                                metadata = arch_data.get('metadata', None)
                                
                                # Save all numpy arrays as datasets
                                for key, value in arch_data.items():
                                    # Skip metadata and non-array fields
                                    if key == 'metadata':
                                        continue
                                    if isinstance(value, np.ndarray):
                                        f.create_dataset(key, data=value, compression='gzip', compression_opts=9)
                                    else:
                                        logger.warning(f"  ⚠️  Skipping non-array field '{key}' for HDF5 (type: {type(value).__name__})")
                                
                                # Save metadata as HDF5 attributes (flattened)
                                if metadata is not None:
                                    for meta_key, meta_value in metadata.items():
                                        try:
                                            # Convert lists to arrays for HDF5 compatibility
                                            if isinstance(meta_value, list):
                                                meta_value = np.array(meta_value)
                                            f.attrs[meta_key] = meta_value
                                        except (TypeError, ValueError) as e:
                                            logger.debug(f"  Could not save metadata key '{meta_key}': {e}")
                            num_saved += 1
                        elif fmt in ['pytorch', 'torch']:
                            save_path = output_dir / f"{base_filename}.pt"
                            # Convert numpy arrays to torch tensors (skip metadata and non-arrays)
                            torch_data = {}
                            for k, v in arch_data.items():
                                if k == 'metadata':
                                    # Keep metadata as-is for PyTorch (it supports dicts)
                                    torch_data[k] = v
                                elif isinstance(v, np.ndarray):
                                    torch_data[k] = torch.from_numpy(v)
                                else:
                                    logger.warning(f"  ⚠️  Skipping non-array field '{k}' for PyTorch (type: {type(v).__name__})")
                            torch.save(torch_data, save_path)
                            num_saved += 1
                        elif fmt == 'laz':
                            # Save patch as LAZ point cloud
                            save_path = output_dir / f"{base_filename}.laz"
                            self._save_patch_as_laz(save_path, arch_data, patch)
                            num_saved += 1
            
            # Clear patches list to free memory
            patches.clear()
            del patches
            
            # Aggressive cleanup after saving each version
            if version_idx > 0:  # For augmented versions, clean up aggressively
                gc.collect()
            
            logger.info(f"  ✓ Saved {num_patches_this_version} patches from {version_label}, freed memory")
        
        # END OF AUGMENTATION LOOP
        
        logger.info(f"  ✅ Total patches saved: {num_saved} ({num_versions} versions)")
        
        # ===== STEP 8: Patches Already Saved Incrementally ===== 
        # (Patches were saved immediately after extraction in the augmentation loop)
        # This eliminates memory buildup from accumulating all patches
        
        # Calculate final statistics
        tile_time = time.time() - tile_start
        pts_processed = len(original_data['points']) * num_versions
        logger.info(
            f"  ✅ Unified processing complete: {num_saved} patches in "
            f"{tile_time:.1f}s ({pts_processed/tile_time:.0f} pts/s)"
        )
        
        # ===== AGGRESSIVE MEMORY CLEANUP =====
        # Force garbage collection and clear large data structures
        original_data.clear()
        if 'points' in locals():
            del points
        if 'intensity' in locals():
            del intensity
        if 'return_number' in locals():
            del return_number
        if 'classification' in locals():
            del classification
        if 'rgb' in locals():
            del rgb
        if 'nir' in locals():
            del nir
        if 'ndvi' in locals():
            del ndvi
        if 'normals' in locals():
            del normals
        if 'curvature' in locals():
            del curvature
        if 'height' in locals():
            del height
        if 'geo_features' in locals():
            del geo_features
        if 'features' in locals():
            del features
        if 'all_features' in locals():
            del all_features
        if 'stitcher' in locals():
            del stitcher
        if 'tile_data' in locals():
            del tile_data
        if 'core_points' in locals():
            del core_points
        if 'buffer_points' in locals():
            del buffer_points
        
        # Use our aggressive cleanup function
        aggressive_memory_cleanup()
        logger.debug(f"  🧹 Aggressive memory cleanup completed")
        
        return {
            'num_patches': num_saved,
            'processing_time': tile_time,
            'points_processed': pts_processed,
            'skipped': False
        }
