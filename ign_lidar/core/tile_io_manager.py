"""
Tile I/O Management Module

This module handles all file I/O operations for LiDAR tiles including
downloading, validation, and recovery of corrupted tiles.

Extracted from LiDARProcessor to improve separation of concerns.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import laspy

logger = logging.getLogger(__name__)


class TileIOManager:
    """
    Manages I/O operations for LiDAR tiles.
    
    Responsibilities:
    - Load LAZ files with validation
    - Re-download corrupted tiles
    - Backup and recovery operations
    - File verification
    
    This class extracts I/O operations from LiDARProcessor to improve
    modularity and testability.
    """
    
    def __init__(self, input_dir: Optional[Path] = None):
        """
        Initialize tile I/O manager.
        
        Args:
            input_dir: Default input directory for tiles
        """
        self.input_dir = Path(input_dir) if input_dir else None
        logger.debug(f"TileIOManager initialized with input_dir: {self.input_dir}")
    
    def load_tile(self, laz_file: Path) -> Optional[laspy.LasData]:
        """
        Load a LAZ tile with validation.
        
        Args:
            laz_file: Path to LAZ file
            
        Returns:
            LasData object, or None if loading failed
        """
        try:
            if not laz_file.exists():
                logger.error(f"Tile file not found: {laz_file}")
                return None
            
            las = laspy.read(str(laz_file))
            
            if len(las.points) == 0:
                logger.warning(f"Tile has no points: {laz_file.name}")
                return None
            
            logger.debug(f"Loaded tile {laz_file.name}: {len(las.points):,} points")
            return las
            
        except Exception as e:
            logger.error(f"Failed to load tile {laz_file.name}: {e}")
            return None
    
    def verify_tile(self, laz_file: Path) -> Tuple[bool, Optional[str]]:
        """
        Verify that a LAZ tile is valid and readable.
        
        Args:
            laz_file: Path to LAZ file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not laz_file.exists():
                return False, "File does not exist"
            
            las = laspy.read(str(laz_file))
            
            if len(las.points) == 0:
                return False, "Tile has no points"
            
            # Basic sanity checks
            if las.header.x_min >= las.header.x_max:
                return False, "Invalid X bounds"
            if las.header.y_min >= las.header.y_max:
                return False, "Invalid Y bounds"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def redownload_tile(self, laz_file: Path) -> bool:
        """
        Attempt to re-download a corrupted tile from IGN WFS.
        
        Args:
            laz_file: Path to the corrupted LAZ file
            
        Returns:
            True if re-download succeeded, False otherwise
        """
        try:
            from ..downloader import IGNLiDARDownloader
            
            # Get filename
            filename = laz_file.name
            
            logger.info(f"  🌐 Re-downloading {filename} from IGN WFS...")
            
            # Backup corrupted file
            backup_path = laz_file.with_suffix(".laz.corrupted")
            if laz_file.exists():
                shutil.move(str(laz_file), str(backup_path))
                logger.debug(f"  Backed up corrupted file to {backup_path.name}")
            
            # Initialize downloader with output directory
            downloader = IGNLiDARDownloader(output_dir=laz_file.parent)
            
            # Download tile (force re-download, don't skip)
            success, was_skipped = downloader.download_tile(
                filename=filename, force=True, skip_existing=False
            )
            
            if success and laz_file.exists():
                # Verify the download
                is_valid, error = self.verify_tile(laz_file)
                
                if is_valid:
                    logger.info(
                        f"  ✓ Re-downloaded tile verified "
                        f"({self._get_point_count(laz_file):,} points)"
                    )
                    # Remove backup if successful
                    if backup_path.exists():
                        backup_path.unlink()
                    return True
                else:
                    logger.error(f"  ✗ Re-downloaded tile is invalid: {error}")
                    # Restore backup
                    self._restore_backup(laz_file, backup_path)
                    return False
            else:
                logger.error(f"  ✗ Download failed or file not created")
                # Restore backup
                self._restore_backup(laz_file, backup_path)
                return False
                
        except ImportError as ie:
            logger.warning(
                f"  ⚠️  IGNLidarDownloader not available for auto-recovery: {ie}"
            )
            return False
        except Exception as e:
            logger.error(f"  ✗ Re-download failed: {e}")
            return False
    
    def _get_point_count(self, laz_file: Path) -> int:
        """Get point count from LAZ file."""
        try:
            las = laspy.read(str(laz_file))
            return len(las.points)
        except:
            return 0
    
    def _restore_backup(self, original: Path, backup: Path):
        """Restore backup file."""
        if backup.exists():
            if original.exists():
                original.unlink()
            shutil.move(str(backup), str(original))
            logger.debug(f"Restored backup {backup.name}")
    
    def create_backup(self, laz_file: Path, suffix: str = ".backup") -> Optional[Path]:
        """
        Create a backup of a LAZ file.
        
        Args:
            laz_file: Path to LAZ file to backup
            suffix: Suffix for backup file
            
        Returns:
            Path to backup file, or None if failed
        """
        try:
            if not laz_file.exists():
                logger.error(f"Cannot backup non-existent file: {laz_file}")
                return None
            
            backup_path = laz_file.with_suffix(laz_file.suffix + suffix)
            shutil.copy2(str(laz_file), str(backup_path))
            
            logger.debug(f"Created backup: {backup_path.name}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create backup of {laz_file.name}: {e}")
            return None
    
    def cleanup_backups(self, directory: Path, pattern: str = "*.backup"):
        """
        Clean up backup files in a directory.
        
        Args:
            directory: Directory to clean
            pattern: Glob pattern for backup files
        """
        try:
            backup_files = list(directory.glob(pattern))
            
            for backup_file in backup_files:
                backup_file.unlink()
                logger.debug(f"Removed backup: {backup_file.name}")
            
            if backup_files:
                logger.info(f"Cleaned up {len(backup_files)} backup files")
                
        except Exception as e:
            logger.error(f"Failed to cleanup backups in {directory}: {e}")
