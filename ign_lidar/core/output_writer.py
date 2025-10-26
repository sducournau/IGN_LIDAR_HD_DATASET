"""
Output generation component for LiDAR processing.

This module handles writing processed data in multiple formats,
including LAZ tiles, NumPy arrays, HDF5, PyTorch tensors, and metadata.

Extracted from LiDARProcessor as part of refactoring (v3.4.0).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class OutputWriter:
    """
    Write processed LiDAR data in multiple formats.

    Handles:
    - Multi-format patch saving (NPZ, HDF5, LAZ, PyTorch)
    - Enriched LAZ tile generation
    - Processing metadata management
    - Format preference handling
    - Dataset manager integration
    - Error handling for I/O operations

    This class was extracted from LiDARProcessor to separate output
    concerns from tile processing.
    """

    def __init__(
        self,
        config: DictConfig,
        dataset_manager: Optional[Any] = None,
    ):
        """
        Initialize output writer.

        Args:
            config: Configuration containing output parameters
            dataset_manager: Optional dataset manager for path organization
        """
        self.config = config
        self.dataset_manager = dataset_manager

        # Output configuration
        self.output_format = config.processor.output_format
        self.processing_mode = config.processor.get("processing_mode", "patches_only")
        self.architecture = config.processor.architecture
        self.lod_level = config.processor.lod_level
        self.patch_size = config.processor.patch_size

        # Mode flags
        self.should_save_enriched_laz = self.processing_mode in [
            "both",
            "enriched_only",
        ]
        self.only_enriched_laz = self.processing_mode == "enriched_only"
        self.should_save_patches = self.processing_mode in ["patches_only", "both"]

        # Parse output formats
        self.formats_list = [fmt.strip() for fmt in self.output_format.split(",")]

        logger.debug(
            f"Initialized OutputWriter: "
            f"formats={self.formats_list}, "
            f"mode={self.processing_mode}, "
            f"save_enriched={self.should_save_enriched_laz}"
        )

    def save_patches(
        self,
        patches: List[Dict[str, np.ndarray]],
        laz_file: Path,
        output_dir: Path,
        tile_split: Optional[str] = None,
    ) -> int:
        """
        Save patches in configured format(s).

        Args:
            patches: List of patch dictionaries with metadata
            laz_file: Original LAZ file (for naming)
            output_dir: Output directory
            tile_split: Optional split name (train/val/test)

        Returns:
            Number of patches saved
        """
        from .classification.io import (
            save_patch_hdf5,
            save_patch_laz,
            save_patch_multi_format,
            save_patch_npz,
            save_patch_torch,
        )
        from .classification.patch_extractor import format_patch_for_architecture

        output_dir.mkdir(parents=True, exist_ok=True)
        num_saved = 0

        logger.info(f"  💾 Saving {len(patches)} patches...")

        for patch in patches:
            # Extract metadata (ensure proper types)
            version = str(patch.pop("_version", "original"))
            base_idx = int(patch.pop("_patch_idx", 0))

            # Determine output path
            base_path = self._get_patch_path(
                laz_file, base_idx, version, output_dir, tile_split
            )

            # Save in requested format(s)
            if len(self.formats_list) > 1:
                # Multi-format output
                num_saved += self._save_patch_multi_format(patch, base_path)
            else:
                # Single format output
                self._save_patch_single_format(patch, base_path, self.formats_list[0])
                num_saved += 1

            # Record in dataset manager
            if self.dataset_manager is not None:
                self.dataset_manager.record_patch_saved(
                    tile_name=laz_file.stem,
                    split=tile_split,
                    patch_size=int(self.patch_size),
                )

        logger.info(f"  ✅ Saved {num_saved} patches")
        return num_saved

    def save_enriched_laz(
        self,
        laz_file: Path,
        output_dir: Path,
        points: np.ndarray,
        classification: np.ndarray,
        features: Dict[str, np.ndarray],
        original_data: Dict[str, Any],
    ) -> bool:
        """
        Save enriched LAZ tile with all computed features.

        Args:
            laz_file: Original LAZ file (for naming)
            output_dir: Output directory
            points: Point cloud [N, 3]
            classification: Classification labels [N]
            features: Computed features dictionary
            original_data: Original data dict with intensity, returns, etc.

        Returns:
            True if saved successfully, False otherwise
        """
        from .classification.io import save_enriched_tile_laz

        if not self.should_save_enriched_laz:
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{laz_file.stem}_enriched.laz"

        try:
            logger.info(f"  💾 Saving enriched LAZ tile...")

            # Determine RGB/NIR to save (prefer computed over input)
            save_rgb = (
                features.get("rgb")
                if features.get("rgb") is not None
                else original_data.get("input_rgb")
            )
            save_nir = (
                features.get("nir")
                if features.get("nir") is not None
                else original_data.get("input_nir")
            )

            # Prepare features (exclude RGB/NIR and points)
            features_to_save = {
                k: v
                for k, v in features.items()
                if k not in ["rgb", "nir", "input_rgb", "input_nir", "points"]
            }

            # Log features being saved
            logger.debug(f"  Features to save: {sorted(features_to_save.keys())}")
            for k, v in features_to_save.items():
                if hasattr(v, "shape"):
                    logger.debug(
                        f"      - {k}: shape={v.shape}, "
                        f"dtype={v.dtype}, ndim={v.ndim}"
                    )

            # Save enriched tile
            save_enriched_tile_laz(
                save_path=output_path,
                points=points,
                classification=classification,
                intensity=original_data["intensity"],
                return_number=original_data["return_number"],
                features=features_to_save,
                original_las=original_data.get("las"),
                header=original_data.get("header"),
                input_rgb=save_rgb,
                input_nir=save_nir,
            )

            logger.info(f"  ✅ Enriched LAZ saved: {output_path.name}")
            return True

        except Exception as e:
            logger.error(f"  ✗ Failed to save enriched LAZ: {e}")
            logger.exception(e)
            return False

    def save_metadata(
        self,
        laz_file: Path,
        output_dir: Path,
        processing_time: float,
        num_points: int,
        num_patches_saved: int,
    ) -> bool:
        """
        Save processing metadata for intelligent skip detection.

        Args:
            laz_file: Original LAZ file
            output_dir: Output directory
            processing_time: Processing time in seconds
            num_points: Number of points processed
            num_patches_saved: Number of patches saved

        Returns:
            True if metadata saved successfully, False otherwise
        """
        from ..io.metadata import ProcessingMetadata

        try:
            metadata_mgr = ProcessingMetadata(output_dir)
            output_files = {}

            # Record enriched LAZ if saved
            if self.should_save_enriched_laz:
                enriched_path = output_dir / f"{laz_file.stem}_enriched.laz"
                if enriched_path.exists():
                    output_files["enriched_laz"] = {
                        "path": str(enriched_path),
                        "size_bytes": enriched_path.stat().st_size,
                    }

            # Record patch information
            if num_patches_saved > 0:
                output_files["patches"] = {
                    "count": num_patches_saved,
                    "format": self.output_format,
                }

            # Save metadata
            metadata_mgr.save_metadata(
                tile_name=laz_file.stem,
                config=self.config,
                processing_time=processing_time,
                num_points=num_points,
                output_files=output_files,
            )

            logger.debug(f"  ✅ Metadata saved for {laz_file.stem}")
            return True

        except Exception as e:
            logger.warning(f"  ⚠️  Failed to save metadata: {e}")
            return False

    def _get_patch_path(
        self,
        laz_file: Path,
        patch_idx: int,
        version: str,
        output_dir: Path,
        tile_split: Optional[str] = None,
    ) -> Path:
        """
        Get output path for a patch.

        Args:
            laz_file: Original LAZ file
            patch_idx: Patch index
            version: Patch version (original/augmented)
            output_dir: Output directory
            tile_split: Optional split name

        Returns:
            Base path for patch (without extension)
        """
        # Use dataset manager if available
        if self.dataset_manager is not None:
            # Get extension from format
            ext = self.formats_list[0] if len(self.formats_list) == 1 else "npz"
            if ext in ["pt", "pth", "pytorch", "torch"]:
                ext = "pt"
            elif ext == "hdf5":
                ext = "h5"

            base_path = self.dataset_manager.get_patch_path(
                tile_name=laz_file.stem,
                patch_idx=patch_idx,
                architecture=self.architecture,
                version=version,
                split=tile_split,
                extension=ext,
            ).with_suffix(
                ""
            )  # Remove extension
        else:
            # Traditional naming
            if version == "original":
                patch_name = (
                    f"{laz_file.stem}_{self.architecture}_patch_{patch_idx:04d}"
                )
            else:
                patch_name = f"{laz_file.stem}_{self.architecture}_patch_{patch_idx:04d}_{version}"
            base_path = output_dir / patch_name

        return base_path

    def _save_patch_multi_format(
        self, patch: Dict[str, np.ndarray], base_path: Path
    ) -> int:
        """
        Save patch in multiple formats.

        Args:
            patch: Patch dictionary
            base_path: Base path (without extension)

        Returns:
            Number of formats saved
        """
        from .classification.io import save_patch_multi_format
        from .classification.patch_extractor import format_patch_for_architecture

        # Format for architecture
        arch_formatted = format_patch_for_architecture(
            patch,
            self.architecture,
            num_points=None,  # Keep original num_points
        )

        # Save in all formats
        num_saved = save_patch_multi_format(
            base_path,
            arch_formatted,
            self.formats_list,
            original_patch=patch,
            lod_level=self.lod_level,
        )

        return num_saved

    def _save_patch_single_format(
        self, patch: Dict[str, np.ndarray], base_path: Path, fmt: str
    ):
        """
        Save patch in a single format.

        Args:
            patch: Patch dictionary
            base_path: Base path (without extension)
            fmt: Format name (npz, hdf5, pt, laz)
        """
        from .classification.io import (
            save_patch_hdf5,
            save_patch_laz,
            save_patch_npz,
            save_patch_torch,
        )
        from .classification.patch_extractor import format_patch_for_architecture

        if fmt == "npz":
            save_path = base_path.with_suffix(".npz")
            save_patch_npz(save_path, patch, lod_level=self.lod_level)
        elif fmt == "hdf5":
            save_path = base_path.with_suffix(".h5")
            save_patch_hdf5(save_path, patch)
        elif fmt in ["pt", "pth", "pytorch", "torch"]:
            save_path = base_path.with_suffix(".pt")
            save_patch_torch(save_path, patch)
        elif fmt == "laz":
            save_path = base_path.with_suffix(".laz")
            arch_formatted = format_patch_for_architecture(
                patch,
                self.architecture,
                num_points=None,  # Keep original num_points
            )
            save_patch_laz(save_path, arch_formatted, patch)
        else:
            # Fallback to NPZ
            logger.warning(f"Unknown format '{fmt}', using NPZ")
            save_path = base_path.with_suffix(".npz")
            save_patch_npz(save_path, patch, lod_level=self.lod_level)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get output writer statistics.

        Returns:
            Dictionary with configuration and stats
        """
        return {
            "output_format": self.output_format,
            "formats_list": self.formats_list,
            "processing_mode": self.processing_mode,
            "save_enriched_laz": self.should_save_enriched_laz,
            "only_enriched_laz": self.only_enriched_laz,
            "save_patches": self.should_save_patches,
            "architecture": self.architecture,
            "lod_level": self.lod_level,
            "has_dataset_manager": self.dataset_manager is not None,
        }
