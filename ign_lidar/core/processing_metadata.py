"""
Metadata tracking for processed tiles to enable intelligent skip detection.

This module creates and validates metadata files alongside processed outputs to:
- Track processing configuration (features, enrichments, filters)
- Store checksums for data sources (BD TOPO, BD Forêt, RPG, Cadastre)
- Record processing timestamps
- Detect configuration changes that require reprocessing
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ProcessingMetadata:
    """
    Manages metadata for processed tiles to enable intelligent skip detection.

    Metadata includes:
    - Processing configuration hash
    - Data source versions/checksums
    - Processing timestamp
    - Feature configuration
    - Enrichment settings
    """

    METADATA_VERSION = "1.0"

    def __init__(self, output_dir: Path):
        """
        Initialize metadata manager.

        Args:
            output_dir: Directory where processed outputs are stored
        """
        self.output_dir = Path(output_dir)
        self.metadata_dir = self.output_dir / ".processing_metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def get_metadata_path(self, tile_name: str) -> Path:
        """Get path to metadata file for a tile."""
        return self.metadata_dir / f"{tile_name}.json"

    def compute_config_hash(self, config: DictConfig) -> str:
        """
        Compute hash of processing configuration.

        Only includes parameters that affect output:
        - Feature computation settings
        - Enrichment sources and parameters
        - Preprocessing filters
        - LOD level and detection modes

        Args:
            config: Configuration object

        Returns:
            SHA256 hash of relevant configuration
        """
        # Extract only parameters that affect processing output
        relevant_config = {
            "processor": {
                "lod_level": config.processor.lod_level,
                "building_detection_mode": config.processor.get(
                    "building_detection_mode", "asprs"
                ),
                "transport_detection_mode": config.processor.get(
                    "transport_detection_mode", "asprs_standard"
                ),
            },
            "features": {
                "mode": config.features.mode,
                "k_neighbors": config.features.k_neighbors,
                "search_radius": config.features.search_radius,
                "compute_normals": config.features.compute_normals,
                "compute_planarity": config.features.compute_planarity,
                "compute_curvature": config.features.compute_curvature,
                "compute_architectural_features": config.features.get(
                    "compute_architectural_features", False
                ),
                "use_rgb": config.features.use_rgb,
                "use_infrared": config.features.use_infrared,
                "compute_ndvi": config.features.compute_ndvi,
                "include_extra": config.features.include_extra,
            },
            "preprocess": {
                "sor_enabled": config.preprocess.sor_enabled,
                "sor_k": config.preprocess.get("sor_k", 10),
                "sor_std": config.preprocess.get("sor_std", 2.0),
                "ror_enabled": config.preprocess.ror_enabled,
                "ror_radius": config.preprocess.get("ror_radius", 1.0),
                "ror_neighbors": config.preprocess.get("ror_neighbors", 5),
            },
            "data_sources": {
                "bd_topo": {
                    "enabled": config.data_sources.bd_topo.enabled,
                    "features": OmegaConf.to_container(
                        config.data_sources.bd_topo.features
                    ),
                    "parameters": OmegaConf.to_container(
                        config.data_sources.bd_topo.get("parameters", {})
                    ),
                },
                "bd_foret": {
                    "enabled": config.data_sources.bd_foret.enabled,
                },
                "rpg": {
                    "enabled": config.data_sources.rpg.enabled,
                    "year": config.data_sources.rpg.get("year", 2024),
                },
                "cadastre": {
                    "enabled": config.data_sources.cadastre.enabled,
                },
            },
            "ground_truth": {
                "enabled": config.ground_truth.enabled,
                "preclassify": config.ground_truth.get("preclassify", False),
                "post_processing": OmegaConf.to_container(
                    config.ground_truth.get("post_processing", {})
                ),
            },
        }

        # Convert to JSON and compute hash
        config_json = json.dumps(relevant_config, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()

    def save_metadata(
        self,
        tile_name: str,
        config: DictConfig,
        processing_time: float,
        num_points: int,
        output_files: Dict[str, Any],
    ) -> None:
        """
        Save processing metadata for a tile.

        Args:
            tile_name: Name of the tile (without extension)
            config: Processing configuration
            processing_time: Time taken to process (seconds)
            num_points: Number of points in processed output
            output_files: Dictionary of output files created
        """
        metadata = {
            "version": self.METADATA_VERSION,
            "tile_name": tile_name,
            "timestamp": datetime.utcnow().isoformat(),
            "config_hash": self.compute_config_hash(config),
            "processing_time_seconds": processing_time,
            "num_points": num_points,
            "output_files": output_files,
            "data_source_versions": {
                "bd_topo": "V3",
                "bd_foret": "V2",
                "rpg": config.data_sources.rpg.get("year", 2024),
            },
        }

        metadata_path = self.get_metadata_path(tile_name)

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata for {tile_name}")
        except Exception as e:
            logger.warning(f"Failed to save metadata for {tile_name}: {e}")

    def load_metadata(self, tile_name: str) -> Optional[Dict[str, Any]]:
        """
        Load processing metadata for a tile.

        Args:
            tile_name: Name of the tile (without extension)

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self.get_metadata_path(tile_name)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {tile_name}: {e}")
            return None

    def should_reprocess(
        self,
        tile_name: str,
        config: DictConfig,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a tile should be reprocessed based on configuration changes.

        Args:
            tile_name: Name of the tile (without extension)
            config: Current processing configuration

        Returns:
            Tuple of (should_reprocess, reason)
        """
        # Load existing metadata
        metadata = self.load_metadata(tile_name)

        if metadata is None:
            return True, "no_metadata_found"

        # Check metadata version
        if metadata.get("version") != self.METADATA_VERSION:
            return True, "metadata_version_mismatch"

        # Compute current config hash
        current_hash = self.compute_config_hash(config)
        stored_hash = metadata.get("config_hash")

        if current_hash != stored_hash:
            return True, "config_changed"

        # Check if output files still exist
        output_files = metadata.get("output_files", {})
        for file_type, file_info in output_files.items():
            if isinstance(file_info, dict):
                file_path = Path(file_info.get("path", ""))
            else:
                file_path = Path(file_info)

            if not file_path.exists():
                return True, f"output_file_missing_{file_type}"

        # All checks passed - no reprocessing needed
        return False, None

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics for all processed tiles.

        Returns:
            Dictionary with processing statistics
        """
        metadata_files = list(self.metadata_dir.glob("*.json"))

        stats = {
            "total_tiles": len(metadata_files),
            "total_processing_time": 0.0,
            "total_points": 0,
            "oldest_processing": None,
            "newest_processing": None,
            "config_hashes": set(),
        }

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                stats["total_processing_time"] += metadata.get(
                    "processing_time_seconds", 0
                )
                stats["total_points"] += metadata.get("num_points", 0)
                stats["config_hashes"].add(metadata.get("config_hash", ""))

                timestamp = metadata.get("timestamp")
                if timestamp:
                    if (
                        stats["oldest_processing"] is None
                        or timestamp < stats["oldest_processing"]
                    ):
                        stats["oldest_processing"] = timestamp
                    if (
                        stats["newest_processing"] is None
                        or timestamp > stats["newest_processing"]
                    ):
                        stats["newest_processing"] = timestamp

            except Exception as e:
                logger.warning(f"Failed to read metadata from {metadata_file}: {e}")

        stats["config_hashes"] = list(stats["config_hashes"])
        stats["unique_configs"] = len(stats["config_hashes"])

        return stats

    def cleanup_orphaned_metadata(self, output_dir: Path) -> int:
        """
        Remove metadata files for tiles that no longer have outputs.

        Args:
            output_dir: Directory containing processed outputs

        Returns:
            Number of metadata files removed
        """
        removed_count = 0

        for metadata_file in self.metadata_dir.glob("*.json"):
            tile_name = metadata_file.stem

            # Check if any output files exist for this tile
            has_output = False
            patterns = [
                f"{tile_name}_enriched.laz",
                f"{tile_name}_patch_*.npz",
                f"{tile_name}_patch_*.h5",
            ]

            for pattern in patterns:
                if list(output_dir.glob(pattern)):
                    has_output = True
                    break

            if not has_output:
                try:
                    metadata_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed orphaned metadata: {metadata_file.name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove orphaned metadata {metadata_file}: {e}"
                    )

        return removed_count
