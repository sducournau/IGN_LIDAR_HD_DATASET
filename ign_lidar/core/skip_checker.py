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
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if a tile should be skipped based on existing patches or enriched LAZ.
        
        Args:
            tile_path: Path to input tile
            output_dir: Directory where patches are stored
            expected_patches: Expected number of base patches (if known)
        
        Returns:
            Tuple of (should_skip, info_dict)
            - should_skip: Whether to skip processing this tile
            - info_dict: Dictionary with skip information
        """
        tile_stem = tile_path.stem
        
        # Special case: If only_enriched_laz mode, check for enriched LAZ file
        if self.only_enriched_laz:
            enriched_dir = output_dir / "enriched"
            enriched_path = enriched_dir / f"{tile_stem}_enriched.laz"
            
            if enriched_path.exists() and enriched_path.stat().st_size > self.min_file_size:
                return True, {
                    'reason': 'enriched_laz_exists',
                    'enriched_path': str(enriched_path),
                    'file_size_mb': enriched_path.stat().st_size / (1024 * 1024),
                }
            else:
                return False, {
                    'reason': 'no_enriched_laz',
                    'enriched_path': str(enriched_path),
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
        
        if reason == 'enriched_laz_exists':
            file_size_mb = skip_info.get('file_size_mb', 0)
            return f"⏭️  {tile_path.name}: Enriched LAZ exists ({file_size_mb:.1f} MB), skipping"
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
