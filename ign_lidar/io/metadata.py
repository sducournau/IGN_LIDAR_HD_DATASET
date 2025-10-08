"""
Metadata management for IGN LiDAR HD processing.
Creates and manages stats.json files for tracking dataset metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manages metadata and stats.json files for datasets."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize metadata manager.
        
        Args:
            base_dir: Base directory for the dataset
        """
        self.base_dir = Path(base_dir)
        self.stats_file = self.base_dir / "stats.json"
    
    def create_download_stats(self, 
                             tiles_info: List[Dict[str, Any]],
                             bbox: Optional[tuple] = None,
                             download_results: Optional[Dict[str, bool]] = None) -> Dict:
        """
        Create stats.json for downloaded tiles.
        
        Args:
            tiles_info: List of tile information dictionaries
            bbox: Optional bounding box used for download
            download_results: Dict mapping filename to download success
            
        Returns:
            Stats dictionary
        """
        stats = {
            "created_at": datetime.now().isoformat(),
            "type": "raw_tiles",
            "tiles": {
                "total": len(tiles_info),
                "successful": sum(1 for r in (download_results or {}).values() if r) if download_results else len(tiles_info),
                "failed": sum(1 for r in (download_results or {}).values() if not r) if download_results else 0
            },
            "bbox": bbox,
            "tiles_list": []
        }
        
        # Add detailed tile information
        for tile_info in tiles_info:
            tile_entry = {
                "filename": tile_info.get("filename", ""),
                "success": download_results.get(tile_info.get("filename", ""), True) if download_results else True
            }
            
            # Add optional fields
            for key in ["tile_x", "tile_y", "center_x", "center_y", "distance_km", "bbox"]:
                if key in tile_info:
                    tile_entry[key] = tile_info[key]
            
            stats["tiles_list"].append(tile_entry)
        
        return stats
    
    def create_processing_stats(self,
                               input_dir: Path,
                               num_tiles: int,
                               num_patches: int,
                               lod_level: str = "LOD2",
                               k_neighbors: Optional[int] = None,
                               patch_size: float = 150.0,
                               augmentation: bool = True,
                               num_augmentations: int = 3) -> Dict:
        """
        Create stats.json for processed/preprocessed data.
        
        Args:
            input_dir: Input directory with raw tiles
            num_tiles: Number of tiles processed
            num_patches: Total number of patches created
            lod_level: LOD level used
            k_neighbors: Number of neighbors for features
            patch_size: Patch size in meters
            augmentation: Whether augmentation was used
            num_augmentations: Number of augmentations per patch
            
        Returns:
            Stats dictionary
        """
        stats = {
            "created_at": datetime.now().isoformat(),
            "type": "processed_dataset",
            "processing": {
                "lod_level": lod_level,
                "k_neighbors": k_neighbors if k_neighbors else "auto",
                "patch_size": patch_size,
                "augmentation_enabled": augmentation,
                "num_augmentations": num_augmentations if augmentation else 0
            },
            "tiles": {
                "total_processed": num_tiles,
                "source_directory": str(input_dir)
            },
            "patches": {
                "total": num_patches,
                "original": num_patches // (num_augmentations + 1) if augmentation else num_patches,
                "augmented": num_patches - (num_patches // (num_augmentations + 1)) if augmentation else 0
            }
        }
        
        # Copy source stats if available
        source_stats_file = input_dir / "stats.json"
        if source_stats_file.exists():
            try:
                with open(source_stats_file, 'r') as f:
                    source_stats = json.load(f)
                stats["source_metadata"] = source_stats
                logger.info(f"Copied source metadata from {source_stats_file}")
            except Exception as e:
                logger.warning(f"Could not read source stats.json: {e}")
        
        return stats
    
    def save_stats(self, stats: Dict) -> None:
        """
        Save stats dictionary to stats.json.
        
        Args:
            stats: Stats dictionary to save
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✓ Saved metadata to {self.stats_file}")
    
    def load_stats(self) -> Optional[Dict]:
        """
        Load stats from stats.json.
        
        Returns:
            Stats dictionary or None if file doesn't exist
        """
        if not self.stats_file.exists():
            return None
        
        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load stats.json: {e}")
            return None
    
    def copy_directory_structure(self, source_dir: Path) -> None:
        """
        Copy directory structure from source to base_dir.
        
        Args:
            source_dir: Source directory to copy structure from
        """
        source_dir = Path(source_dir)
        
        if not source_dir.exists():
            logger.warning(f"Source directory does not exist: {source_dir}")
            return
        
        # Find all subdirectories
        for subdir in source_dir.rglob("*"):
            if subdir.is_dir():
                relative_path = subdir.relative_to(source_dir)
                target_dir = self.base_dir / relative_path
                target_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {relative_path}")
        
        logger.info(f"✓ Copied directory structure from {source_dir}")
    
    def copy_stats_from_source(self, source_dir: Path) -> bool:
        """
        Copy stats.json from source directory.
        
        Args:
            source_dir: Source directory containing stats.json
            
        Returns:
            True if successful
        """
        source_stats = source_dir / "stats.json"
        
        if not source_stats.exists():
            logger.warning(f"No stats.json found in {source_dir}")
            return False
        
        try:
            shutil.copy2(source_stats, self.stats_file)
            logger.info(f"✓ Copied stats.json from {source_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy stats.json: {e}")
            return False
    
    def update_stats(self, updates: Dict) -> None:
        """
        Update existing stats.json with new information.
        
        Args:
            updates: Dictionary of updates to apply
        """
        stats = self.load_stats() or {}
        stats.update(updates)
        stats["updated_at"] = datetime.now().isoformat()
        self.save_stats(stats)
    
    def add_processing_info(self, 
                           num_patches: int,
                           processing_time: Optional[float] = None) -> None:
        """
        Add processing information to existing stats.
        
        Args:
            num_patches: Number of patches created
            processing_time: Processing time in seconds
        """
        stats = self.load_stats() or {}
        
        if "patches" not in stats:
            stats["patches"] = {}
        
        stats["patches"]["total"] = num_patches
        
        if processing_time:
            stats["processing_time_seconds"] = round(processing_time, 2)
        
        self.save_stats(stats)
    
    def create_tile_metadata(self,
                            filename: str,
                            location_name: Optional[str] = None,
                            category: Optional[str] = None,
                            characteristics: Optional[List[str]] = None,
                            description: Optional[str] = None,
                            architectural_style: Optional[str] = None,
                            architectural_styles: Optional[List[Dict]] = None,
                            bbox: Optional[tuple] = None,
                            additional_info: Optional[Dict] = None) -> Dict:
        """
        Create metadata for an individual tile.
        
        Args:
            filename: Tile filename
            location_name: Name of the location
            category: Category (e.g., urban_dense, heritage_palace)
            characteristics: List of characteristics
            description: Human-readable description
            architectural_style: Architectural style (legacy, single)
            architectural_styles: List of styles with weights (new multi-label)
            bbox: Bounding box
            additional_info: Any additional information
            
        Returns:
            Tile metadata dictionary
        """
        metadata = {
            "filename": filename,
            "downloaded_at": datetime.now().isoformat(),
            "location": {
                "name": location_name or "Unknown",
                "category": category or "general",
            }
        }
        
        if characteristics:
            metadata["characteristics"] = characteristics
        
        if description:
            metadata["description"] = description
        
        # New multi-label style support
        if architectural_styles:
            metadata["architectural_styles"] = architectural_styles
            # Set dominant style for backward compatibility
            if architectural_styles:
                dominant = max(architectural_styles, 
                             key=lambda x: x.get("weight", 0))
                metadata["dominant_style_id"] = dominant.get("style_id", 0)
        elif architectural_style:
            # Legacy single style support
            metadata["architectural_style"] = architectural_style
        
        if bbox:
            metadata["bbox"] = bbox
        
        if additional_info:
            metadata["additional_info"] = additional_info
        
        return metadata
    
    def save_tile_metadata(self, tile_metadata: Dict, subdirectory: Optional[str] = None) -> None:
        """
        Save individual tile metadata to a JSON file.
        
        Args:
            tile_metadata: Tile metadata dictionary
            subdirectory: Optional subdirectory for the metadata file
        """
        filename = tile_metadata.get("filename", "unknown")
        base_name = Path(filename).stem
        
        # Determine output directory
        if subdirectory:
            output_dir = self.base_dir / subdirectory
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.base_dir
        
        # Save metadata file next to the tile
        metadata_file = output_dir / f"{base_name}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(tile_metadata, f, indent=2)
        
        logger.debug(f"Saved tile metadata: {metadata_file.name}")
    
    def load_tile_metadata(self, tile_path: Path) -> Optional[Dict]:
        """
        Load metadata for a specific tile.
        
        Args:
            tile_path: Path to the tile file
            
        Returns:
            Tile metadata or None if not found
        """
        metadata_file = tile_path.parent / f"{tile_path.stem}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load tile metadata: {e}")
            return None
