"""
Intelligent skip detection for patch processing.

This module provides smart skip logic that validates:
- Expected number of patches exist
- Patches are not corrupted
- Patches contain required data
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PatchSkipChecker:
    """
    Intelligent skip checker for patch processing.
    
    Validates that existing patches are complete and valid before skipping.
    """
    
    def __init__(
        self,
        output_format: str = 'npz',
        architecture: str = 'pointnet++',
        num_augmentations: int = 0,
        augment: bool = False,
        validate_content: bool = True,
        min_file_size: int = 1024,  # 1KB minimum
        only_enriched_laz: bool = False,
    ):
        """
        Initialize skip checker.
        
        Args:
            output_format: Expected output format ('npz', 'hdf5', 'torch', 'laz')
            architecture: Target architecture ('pointnet++', 'hybrid', etc.)
            num_augmentations: Number of augmentation versions
            augment: Whether augmentation is enabled
            validate_content: Whether to validate patch content
            min_file_size: Minimum file size in bytes for validity check
            only_enriched_laz: If True, check for enriched LAZ instead of patches
        """
        self.output_format = output_format
        self.architecture = architecture
        self.num_augmentations = num_augmentations if augment else 0
        self.augment = augment
        self.validate_content = validate_content
        self.min_file_size = min_file_size
        self.only_enriched_laz = only_enriched_laz
        
        # Parse multiple formats if comma-separated
        self.formats = [fmt.strip() for fmt in output_format.split(',')]
    
    def should_skip_tile(
        self,
        tile_path: Path,
        output_dir: Path,
        expected_patches: Optional[int] = None,
        save_enriched: bool = False,
        include_rgb: bool = False,
        include_infrared: bool = False,
        compute_ndvi: bool = False,
        include_extra_features: bool = False,
        include_classification: bool = False,
        include_forest: bool = False,
        include_agriculture: bool = False,
        include_cadastre: bool = False,
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if a tile should be skipped based on existing patches or enriched LAZ.
        
        Args:
            tile_path: Path to input tile
            output_dir: Directory where patches are stored
            expected_patches: Expected number of base patches (if known)
            save_enriched: Whether enriched LAZ saving is enabled
            include_rgb: Whether RGB enrichment is enabled
            include_infrared: Whether infrared enrichment is enabled
            compute_ndvi: Whether NDVI computation is enabled
            include_extra_features: Whether extra geometric features are enabled
            include_classification: Whether ASPRS classification is enabled
            include_forest: Whether BD Forêt forest types are enabled
            include_agriculture: Whether RPG agriculture data is enabled
            include_cadastre: Whether cadastral parcel data is enabled
        
        Returns:
            Tuple of (should_skip, info_dict)
            - should_skip: Whether to skip processing this tile
            - info_dict: Dictionary with skip information
        """
        tile_stem = tile_path.stem
        
        # Special case: If only_enriched_laz mode, check for enriched LAZ file
        if self.only_enriched_laz:
            # In enriched_only mode, files can be saved in multiple locations/names:
            # 1. Directly in output_dir with _enriched suffix (current behavior)
            # 2. In output_dir/enriched/ subdirectory with _enriched suffix (legacy)
            # 3. In output_dir/enriched/ subdirectory without _enriched suffix (very old)
            # 4. Directly in output_dir without _enriched suffix (edge case)
            possible_paths = [
                output_dir / f"{tile_stem}_enriched.laz",  # Current
                output_dir / "enriched" / f"{tile_stem}_enriched.laz",  # Legacy with suffix
                output_dir / "enriched" / f"{tile_stem}.laz",  # Legacy without suffix
                output_dir / f"{tile_stem}.laz",  # Direct without suffix
            ]
            
            # Find first existing file with valid size
            enriched_path = None
            for path in possible_paths:
                if path.exists() and path.stat().st_size > self.min_file_size:
                    enriched_path = path
                    break
            
            if enriched_path is None:
                # File doesn't exist in any location - process it
                return False, {
                    'reason': 'no_enriched_laz',
                    'checked_paths': [str(p) for p in possible_paths],
                }
            
            # Validate both existence and content
            if enriched_path.exists() and enriched_path.stat().st_size > self.min_file_size:
                # File exists - now validate it has required features
                is_valid, validation_info = self._validate_enriched_laz(
                    enriched_path,
                    include_rgb=include_rgb,
                    include_infrared=include_infrared,
                    compute_ndvi=compute_ndvi,
                    include_extra_features=include_extra_features,
                    include_classification=include_classification,
                    include_forest=include_forest,
                    include_agriculture=include_agriculture,
                    include_cadastre=include_cadastre
                )
                
                if is_valid:
                    # File exists and has all required features - skip processing
                    return True, {
                        'reason': 'enriched_laz_complete',
                        'enriched_path': str(enriched_path),
                        'file_size_mb': enriched_path.stat().st_size / (1024 * 1024),
                        'features_validated': validation_info.get('features_present', []),
                    }
                else:
                    # File exists but is missing features - reprocess it
                    missing_features = validation_info.get('missing_features', [])
                    return False, {
                        'reason': 'enriched_laz_incomplete',
                        'enriched_path': str(enriched_path),
                        'missing_features': missing_features,
                        'error': validation_info.get('error', 'Unknown validation error'),
                    }
            else:
                # File doesn't exist or is too small - process it
                return False, {
                    'reason': 'no_enriched_laz',
                    'enriched_path': str(enriched_path),
                }
        
        # For "both" mode: check if BOTH enriched LAZ AND patches exist
        if save_enriched and not self.only_enriched_laz:
            # In "both" mode, enriched LAZ is saved in "enriched" subdirectory
            enriched_path = output_dir / "enriched" / f"{tile_stem}_enriched.laz"
            
            # Check enriched LAZ
            enriched_exists = False
            validation_info = {}
            if enriched_path.exists() and enriched_path.stat().st_size > self.min_file_size:
                is_valid, validation_info = self._validate_enriched_laz(
                    enriched_path,
                    include_rgb=include_rgb,
                    include_infrared=include_infrared,
                    compute_ndvi=compute_ndvi,
                    include_extra_features=include_extra_features,
                    include_classification=include_classification,
                    include_forest=include_forest,
                    include_agriculture=include_agriculture,
                    include_cadastre=include_cadastre
                )
                enriched_exists = is_valid
            
            # Check patches
            existing_patches = self._find_tile_patches(tile_stem, output_dir)
            if existing_patches:
                validation_results = self._validate_patches(existing_patches)
                patches_exist = validation_results['num_valid'] > 0 and validation_results['num_corrupted'] == 0
            else:
                patches_exist = False
            
            # Skip only if BOTH exist and are valid
            if enriched_exists and patches_exist:
                return True, {
                    'reason': 'both_enriched_and_patches_exist',
                    'enriched_path': str(enriched_path),
                    'enriched_file_size_mb': enriched_path.stat().st_size / (1024 * 1024),
                    'num_patches': len(existing_patches),
                    'features_validated': validation_info.get('features_present', []),
                }
            elif enriched_exists and not patches_exist:
                return False, {
                    'reason': 'enriched_exists_but_no_patches',
                    'enriched_path': str(enriched_path),
                }
            elif patches_exist and not enriched_exists:
                return False, {
                    'reason': 'patches_exist_but_no_enriched',
                    'num_patches': len(existing_patches),
                }
            else:
                return False, {
                    'reason': 'neither_enriched_nor_patches_exist',
                }
        
        # Calculate expected number of patches
        num_versions = 1 + self.num_augmentations
        
        # Find existing patches for this tile
        existing_patches = self._find_tile_patches(tile_stem, output_dir)
        
        if not existing_patches:
            return False, {
                'reason': 'no_patches_found',
                'existing_count': 0,
                'expected_count': None,
                'valid_count': 0,
            }
        
        # Validate existing patches
        validation_results = self._validate_patches(existing_patches)
        
        num_existing = len(existing_patches)
        num_valid = validation_results['num_valid']
        num_corrupted = validation_results['num_corrupted']
        
        # If we know expected count, check completeness
        if expected_patches is not None:
            expected_total = expected_patches * num_versions
            
            # Skip only if we have all expected valid patches
            if num_valid >= expected_total:
                return True, {
                    'reason': 'complete_and_valid',
                    'existing_count': num_existing,
                    'expected_count': expected_total,
                    'valid_count': num_valid,
                    'corrupted_count': num_corrupted,
                    'validation_passed': True,
                }
            else:
                return False, {
                    'reason': 'incomplete_patches',
                    'existing_count': num_existing,
                    'expected_count': expected_total,
                    'valid_count': num_valid,
                    'corrupted_count': num_corrupted,
                    'missing_count': expected_total - num_valid,
                }
        
        # If we don't know expected count, use heuristics
        # Skip if we have valid patches and no corrupted ones
        if num_valid > 0 and num_corrupted == 0:
            return True, {
                'reason': 'patches_exist_and_valid',
                'existing_count': num_existing,
                'expected_count': None,
                'valid_count': num_valid,
                'corrupted_count': num_corrupted,
                'validation_passed': True,
            }
        elif num_corrupted > 0:
            return False, {
                'reason': 'corrupted_patches_detected',
                'existing_count': num_existing,
                'expected_count': None,
                'valid_count': num_valid,
                'corrupted_count': num_corrupted,
            }
        else:
            return False, {
                'reason': 'no_valid_patches',
                'existing_count': num_existing,
                'expected_count': None,
                'valid_count': num_valid,
                'corrupted_count': num_corrupted,
            }
    
    def _find_tile_patches(self, tile_stem: str, output_dir: Path) -> List[Path]:
        """
        Find all patches for a given tile.
        
        Args:
            tile_stem: Tile name without extension
            output_dir: Directory to search
        
        Returns:
            List of patch file paths
        """
        patches = []
        
        for fmt in self.formats:
            # Map format aliases
            if fmt in ['pytorch', 'torch']:
                ext = 'pt'
            elif fmt == 'hdf5':
                ext = 'h5'
            else:
                ext = fmt
            
            # Find patches for all architectures (hybrid, multi-arch, single)
            # Pattern: {tile_stem}_{arch}_patch_{idx:04d}[_{version}].{ext}
            pattern = f"{tile_stem}_*_patch_*.{ext}"
            found = list(output_dir.glob(pattern))
            patches.extend(found)
        
        return patches
    
    def _validate_patches(self, patch_files: List[Path]) -> Dict[str, any]:
        """
        Validate a list of patch files.
        
        Args:
            patch_files: List of patch file paths
        
        Returns:
            Dictionary with validation results
        """
        num_valid = 0
        num_corrupted = 0
        corrupted_files = []
        validation_errors = []
        
        for patch_file in patch_files:
            is_valid, error = self._validate_single_patch(patch_file)
            
            if is_valid:
                num_valid += 1
            else:
                num_corrupted += 1
                corrupted_files.append(patch_file.name)
                if error:
                    validation_errors.append(error)
        
        return {
            'num_valid': num_valid,
            'num_corrupted': num_corrupted,
            'corrupted_files': corrupted_files,
            'validation_errors': validation_errors,
        }
    
    def _validate_single_patch(self, patch_file: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate a single patch file.
        
        Args:
            patch_file: Path to patch file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file exists
        if not patch_file.exists():
            return False, f"File not found: {patch_file}"
        
        # Check file size
        file_size = patch_file.stat().st_size
        if file_size < self.min_file_size:
            return False, f"File too small ({file_size} bytes): {patch_file.name}"
        
        # If content validation is disabled, consider it valid
        if not self.validate_content:
            return True, None
        
        # Validate content based on format
        try:
            ext = patch_file.suffix.lower()
            
            if ext == '.npz':
                return self._validate_npz(patch_file)
            elif ext == '.h5':
                return self._validate_hdf5(patch_file)
            elif ext == '.pt':
                return self._validate_torch(patch_file)
            elif ext == '.laz':
                return self._validate_laz(patch_file)
            else:
                # Unknown format, assume valid
                return True, None
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_npz(self, patch_file: Path) -> Tuple[bool, Optional[str]]:
        """Validate NPZ file."""
        try:
            data = np.load(patch_file, allow_pickle=True)
            
            # Check for required keys (architecture-agnostic)
            # Accept either 'coords' (PointNet++) or 'points' (Hybrid)
            has_coords = 'coords' in data.files or 'points' in data.files
            has_labels = 'labels' in data.files
            
            if not has_coords:
                return False, "Missing coordinates (neither 'coords' nor 'points' found)"
            
            if not has_labels:
                return False, "Missing 'labels' key"
            
            # Check data shapes
            coords = data['coords'] if 'coords' in data.files else data['points']
            labels = data['labels']
            
            if coords.shape[0] == 0:
                return False, "Empty coords array"
            
            if coords.shape[0] != labels.shape[0]:
                return False, f"Shape mismatch: coords={coords.shape[0]}, labels={labels.shape[0]}"
            
            return True, None
            
        except Exception as e:
            return False, f"NPZ load error: {str(e)}"
    
    def _validate_hdf5(self, patch_file: Path) -> Tuple[bool, Optional[str]]:
        """Validate HDF5 file."""
        try:
            import h5py
            
            with h5py.File(patch_file, 'r') as f:
                # Check for required datasets
                required_keys = ['coords', 'labels']
                
                missing_keys = [key for key in required_keys if key not in f.keys()]
                if missing_keys:
                    return False, f"Missing datasets: {missing_keys}"
                
                # Check data shapes
                coords = f['coords'][:]
                labels = f['labels'][:]
                
                if coords.shape[0] == 0:
                    return False, "Empty coords array"
                
                if coords.shape[0] != labels.shape[0]:
                    return False, f"Shape mismatch: coords={coords.shape[0]}, labels={labels.shape[0]}"
            
            return True, None
            
        except ImportError:
            # h5py not available, skip validation
            return True, None
        except Exception as e:
            return False, f"HDF5 load error: {str(e)}"
    
    def _validate_torch(self, patch_file: Path) -> Tuple[bool, Optional[str]]:
        """Validate PyTorch file."""
        try:
            import torch
            
            data = torch.load(patch_file, weights_only=False)
            
            # Check for required keys
            required_keys = ['coords', 'labels']
            
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                return False, f"Missing keys: {missing_keys}"
            
            # Check data shapes
            coords = data['coords']
            labels = data['labels']
            
            if coords.shape[0] == 0:
                return False, "Empty coords tensor"
            
            if coords.shape[0] != labels.shape[0]:
                return False, f"Shape mismatch: coords={coords.shape[0]}, labels={labels.shape[0]}"
            
            return True, None
            
        except ImportError:
            # torch not available, skip validation
            return True, None
        except Exception as e:
            return False, f"PyTorch load error: {str(e)}"
    
    def _validate_laz(self, patch_file: Path) -> Tuple[bool, Optional[str]]:
        """Validate LAZ file."""
        try:
            import laspy
            
            las = laspy.read(str(patch_file))
            
            if len(las.points) == 0:
                return False, "Empty point cloud"
            
            # Check for X, Y, Z coordinates
            if not hasattr(las, 'x') or not hasattr(las, 'y') or not hasattr(las, 'z'):
                return False, "Missing coordinate attributes"
            
            return True, None
            
        except ImportError:
            # laspy not available, skip validation
            return True, None
        except Exception as e:
            return False, f"LAZ load error: {str(e)}"
    
    def _validate_enriched_laz(
        self,
        enriched_path: Path,
        include_rgb: bool = False,
        include_infrared: bool = False,
        compute_ndvi: bool = False,
        include_extra_features: bool = False,
        include_classification: bool = False,
        include_forest: bool = False,
        include_agriculture: bool = False,
        include_cadastre: bool = False,
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Validate enriched LAZ file has expected features.
        
        Args:
            enriched_path: Path to enriched LAZ file
            include_rgb: Whether RGB should be present
            include_infrared: Whether NIR should be present
            compute_ndvi: Whether NDVI should be present
            include_extra_features: Whether extra geometric features should be present
            include_classification: Whether ASPRS classification should be present
            include_forest: Whether BD Forêt forest type attributes should be present
            include_agriculture: Whether RPG agriculture attributes should be present
            include_cadastre: Whether cadastral parcel ID should be present
        
        Returns:
            Tuple of (is_valid, info_dict)
        """
        try:
            import laspy
            
            # Load the LAZ file
            las = laspy.read(str(enriched_path))
            
            # Check basic validity
            if len(las.points) == 0:
                return False, {'error': 'Empty point cloud'}
            
            # Get extra dimensions present in file (convert generator to list!)
            if hasattr(las.point_format, 'extra_dimension_names'):
                extra_dims = list(las.point_format.extra_dimension_names)
            else:
                extra_dims = []
            
            # Core features that should always be present
            core_features = ['normal_x', 'normal_y', 'normal_z', 'curvature', 'height']
            missing_core = [f for f in core_features if f not in extra_dims]
            
            if missing_core:
                return False, {
                    'error': f'Missing core features: {missing_core}',
                    'missing_features': missing_core,
                    'features_present': extra_dims,
                }
            
            # Check optional features based on configuration
            missing_optional = []
            
            if include_rgb:
                # RGB might be in standard RGB fields or extra dimensions
                has_rgb = (hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'))
                if not has_rgb:
                    missing_optional.append('RGB')
            
            if include_infrared:
                has_nir = 'nir' in extra_dims or hasattr(las, 'nir')
                if not has_nir:
                    missing_optional.append('NIR')
            
            if compute_ndvi:
                has_ndvi = 'ndvi' in extra_dims
                if not has_ndvi:
                    missing_optional.append('NDVI')
            
            if include_extra_features:
                # Check for geometric features
                geo_features = ['planarity', 'linearity', 'sphericity', 'verticality']
                missing_geo = [f for f in geo_features if f not in extra_dims]
                if missing_geo:
                    missing_optional.extend(missing_geo)
            
            # Check for classification (ASPRS codes)
            if include_classification:
                # Classification should be in the standard 'classification' field
                has_classification = hasattr(las, 'classification') and len(las.classification) > 0
                if not has_classification:
                    missing_optional.append('classification')
            
            # Check for forest attributes
            if include_forest:
                forest_attrs = ['forest_type', 'primary_species']
                missing_forest = [f for f in forest_attrs if f not in extra_dims]
                if missing_forest:
                    missing_optional.extend(missing_forest)
            
            # Check for agriculture attributes
            if include_agriculture:
                agri_attrs = ['crop_code', 'crop_category']
                missing_agri = [f for f in agri_attrs if f not in extra_dims]
                if missing_agri:
                    missing_optional.extend(missing_agri)
            
            # Check for cadastre parcel ID
            if include_cadastre:
                has_parcel = 'parcel_id' in extra_dims
                if not has_parcel:
                    missing_optional.append('parcel_id')
            
            # If critical optional features are missing, consider invalid
            if missing_optional:
                logger.debug(f"Enriched LAZ missing optional features: {missing_optional}")
                return False, {
                    'error': f'Missing optional features: {missing_optional}',
                    'missing_features': missing_optional,
                    'features_present': extra_dims,
                }
            
            # All checks passed
            return True, {
                'features_present': core_features + extra_dims,
                'point_count': len(las.points),
            }
            
        except ImportError:
            # laspy not available, assume valid
            logger.warning("laspy not available, cannot validate enriched LAZ features")
            return True, {'error': 'laspy_not_available'}
        except Exception as e:
            return False, {'error': f'Validation error: {str(e)}'}
    
    def format_skip_message(self, tile_path: Path, skip_info: Dict[str, any]) -> str:
        """
        Format a user-friendly skip message.
        
        Args:
            tile_path: Path to the tile
            skip_info: Skip information dictionary
        
        Returns:
            Formatted message string
        """
        reason = skip_info.get('reason', 'unknown')
        valid_count = skip_info.get('valid_count', 0)
        existing_count = skip_info.get('existing_count', 0)
        expected_count = skip_info.get('expected_count', None)
        corrupted_count = skip_info.get('corrupted_count', 0)
        
        # New enriched LAZ skip reasons
        if reason == 'enriched_laz_complete':
            file_size_mb = skip_info.get('file_size_mb', 0)
            features = skip_info.get('features_validated', [])
            return f"⏭️  {tile_path.name}: Valid enriched LAZ exists ({file_size_mb:.1f} MB, {len(features)} features), skipping"
        
        elif reason == 'enriched_laz_incomplete':
            missing = skip_info.get('missing_features', [])
            error = skip_info.get('error', 'unknown')
            return f"🔄 {tile_path.name}: Enriched LAZ incomplete (missing: {missing}), reprocessing"
        
        elif reason == 'enriched_laz_exists':
            file_size_mb = skip_info.get('file_size_mb', 0)
            return f"⏭️  {tile_path.name}: Enriched LAZ exists ({file_size_mb:.1f} MB), skipping"
        
        elif reason == 'enriched_laz_exists_valid':
            file_size_mb = skip_info.get('file_size_mb', 0)
            features = skip_info.get('features_validated', [])
            return f"⏭️  {tile_path.name}: Valid enriched LAZ exists ({file_size_mb:.1f} MB, {len(features)} features), skipping"
        
        elif reason == 'enriched_laz_invalid':
            error = skip_info.get('validation_error', 'unknown')
            missing = skip_info.get('missing_features', [])
            return f"🔄 {tile_path.name}: Enriched LAZ invalid ({error}, missing: {missing}), reprocessing"
        
        elif reason == 'both_enriched_and_patches_exist':
            file_size_mb = skip_info.get('enriched_file_size_mb', 0)
            num_patches = skip_info.get('num_patches', 0)
            return f"⏭️  {tile_path.name}: Both enriched LAZ ({file_size_mb:.1f} MB) and {num_patches} patches exist, skipping"
        
        elif reason == 'enriched_exists_but_no_patches':
            return f"🔄 {tile_path.name}: Enriched LAZ exists but patches missing, processing patches only"
        
        elif reason == 'patches_exist_but_no_enriched':
            num_patches = skip_info.get('num_patches', 0)
            return f"🔄 {tile_path.name}: {num_patches} patches exist but enriched LAZ missing, processing enriched LAZ only"
        
        elif reason == 'neither_enriched_nor_patches_exist':
            return f"🔄 {tile_path.name}: No outputs found, full processing"
        
        elif reason == 'no_enriched_laz':
            return f"🔄 {tile_path.name}: No enriched LAZ found, processing"
        
        elif reason == 'complete_and_valid':
            if expected_count:
                return f"⏭️  {tile_path.name}: {valid_count}/{expected_count} patches valid, skipping"
            else:
                return f"⏭️  {tile_path.name}: {valid_count} patches exist and valid, skipping"
        
        elif reason == 'patches_exist_and_valid':
            return f"⏭️  {tile_path.name}: {valid_count} patches exist, skipping"
        
        elif reason == 'incomplete_patches':
            missing = skip_info.get('missing_count', 0)
            return f"🔄 {tile_path.name}: Only {valid_count}/{expected_count} patches valid (missing {missing}), reprocessing"
        
        elif reason == 'corrupted_patches_detected':
            return f"🔄 {tile_path.name}: {corrupted_count} corrupted patches detected, reprocessing"
        
        elif reason == 'no_valid_patches':
            return f"🔄 {tile_path.name}: No valid patches found, processing"
        
        elif reason == 'no_patches_found':
            return f"🔄 {tile_path.name}: No patches found, processing"
        
        else:
            return f"🔄 {tile_path.name}: Processing ({reason})"
