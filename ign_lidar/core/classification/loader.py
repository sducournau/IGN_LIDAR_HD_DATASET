"""
Loader module for reading and validating LiDAR data.

This module provides functions to load LAZ/LAS files with proper error handling,
data validation, and typed data structures.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Check laspy availability
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    logger.warning("laspy not available, LAZ/LAS loading not supported")


@dataclass
class LiDARData:
    """
    Typed container for LiDAR point cloud data.
    
    Attributes:
        points: [N, 3] XYZ coordinates (float32)
        intensity: [N] Intensity values normalized to [0, 1] (float32)
        return_number: [N] Return numbers (float32)
        classification: [N] ASPRS classification codes (uint8)
        rgb: Optional [N, 3] RGB colors normalized to [0, 1] (float32)
        nir: Optional [N] Near-infrared values normalized to [0, 1] (float32)
        num_points: Number of points in the cloud
        num_classes: Number of unique classification codes
        bounds: Tuple of (xmin, ymin, zmin, xmax, ymax, zmax)
        file_path: Source file path
    """
    points: np.ndarray
    intensity: np.ndarray
    return_number: np.ndarray
    classification: np.ndarray
    rgb: Optional[np.ndarray] = None
    nir: Optional[np.ndarray] = None
    num_points: int = 0
    num_classes: int = 0
    bounds: Optional[Tuple[float, float, float, float, float, float]] = None
    file_path: Optional[Path] = None
    
    def __post_init__(self):
        """Compute derived attributes after initialization."""
        if self.num_points == 0:
            self.num_points = len(self.points)
        if self.num_classes == 0:
            self.num_classes = len(np.unique(self.classification))
        if self.bounds is None:
            self.bounds = (
                float(self.points[:, 0].min()),
                float(self.points[:, 1].min()),
                float(self.points[:, 2].min()),
                float(self.points[:, 0].max()),
                float(self.points[:, 1].max()),
                float(self.points[:, 2].max())
            )
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Convert to dictionary format for compatibility with existing code.
        
        Returns:
            Dictionary with point cloud data
        """
        data = {
            'points': self.points,
            'intensity': self.intensity,
            'return_number': self.return_number,
            'classification': self.classification
        }
        if self.rgb is not None:
            data['rgb'] = self.rgb
        if self.nir is not None:
            data['nir'] = self.nir
        return data
    
    def filter_by_bbox(self, bbox: Tuple[float, float, float, float]) -> 'LiDARData':
        """
        Filter points by bounding box.
        
        Args:
            bbox: (xmin, ymin, xmax, ymax)
            
        Returns:
            New LiDARData with filtered points
        """
        xmin, ymin, xmax, ymax = bbox
        mask = (
            (self.points[:, 0] >= xmin) & (self.points[:, 0] <= xmax) &
            (self.points[:, 1] >= ymin) & (self.points[:, 1] <= ymax)
        )
        
        return LiDARData(
            points=self.points[mask],
            intensity=self.intensity[mask],
            return_number=self.return_number[mask],
            classification=self.classification[mask],
            rgb=self.rgb[mask] if self.rgb is not None else None,
            nir=self.nir[mask] if self.nir is not None else None,
            file_path=self.file_path
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LiDARData(points={self.num_points:,}, "
            f"classes={self.num_classes}, "
            f"rgb={'Yes' if self.rgb is not None else 'No'}, "
            f"nir={'Yes' if self.nir is not None else 'No'})"
        )


class LiDARLoadError(Exception):
    """Exception raised when LiDAR file loading fails."""
    pass


class LiDARCorruptionError(LiDARLoadError):
    """Exception raised when LiDAR file is corrupted."""
    pass


def load_laz_file(
    file_path: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_retries: int = 2,
    auto_redownload: bool = True,
    redownload_callback: Optional[callable] = None
) -> LiDARData:
    """
    Load a LAZ/LAS file with error handling and retry logic.
    
    Args:
        file_path: Path to LAZ/LAS file
        bbox: Optional bounding box filter (xmin, ymin, xmax, ymax)
        max_retries: Maximum number of retry attempts
        auto_redownload: If True, attempt to re-download corrupted files
        redownload_callback: Optional function to call for re-downloading
                            Should accept file_path and return bool (success)
                            
    Returns:
        LiDARData object with loaded point cloud
        
    Raises:
        LiDARLoadError: If file cannot be loaded
        LiDARCorruptionError: If file is corrupted and cannot be recovered
        ImportError: If laspy is not installed
    """
    if not LASPY_AVAILABLE:
        raise ImportError(
            "laspy is required for loading LAZ/LAS files. "
            "Install with: pip install laspy[lazrs]"
        )
    
    if not file_path.exists():
        raise LiDARLoadError(f"File not found: {file_path}")
    
    las = None
    last_error = None
    
    for attempt in range(max_retries):
        try:
            las = laspy.read(str(file_path))
            break  # Success
            
        except Exception as e:
            error_msg = str(e)
            last_error = e
            
            # Check if it's a corruption error
            is_corruption_error = any([
                'failed to fill whole buffer' in error_msg.lower(),
                'ioerror' in error_msg.lower(),
                'unexpected end of file' in error_msg.lower(),
                'invalid' in error_msg.lower()
            ])
            
            if is_corruption_error and attempt < max_retries - 1:
                logger.warning(
                    f"Corrupted LAZ file detected: {error_msg}"
                )
                
                if auto_redownload and redownload_callback is not None:
                    logger.info(
                        f"Attempting to re-download tile "
                        f"(attempt {attempt + 2}/{max_retries})..."
                    )
                    
                    if redownload_callback(file_path):
                        logger.info("Tile re-downloaded successfully")
                        continue  # Retry loading
                    else:
                        logger.error("Failed to re-download tile")
                        raise LiDARCorruptionError(
                            f"File corrupted and re-download failed: {file_path}"
                        ) from e
                else:
                    raise LiDARCorruptionError(
                        f"File corrupted: {file_path}"
                    ) from e
            else:
                raise LiDARLoadError(
                    f"Failed to read {file_path}: {e}"
                ) from e
    
    if las is None:
        raise LiDARLoadError(
            f"Failed to load LAZ file after {max_retries} retries: {file_path}"
        ) from last_error
    
    # Extract basic data with proper normalization
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    intensity = np.array(las.intensity, dtype=np.float32) / 65535.0
    return_number = np.array(las.return_number, dtype=np.float32)
    classification = np.array(las.classification, dtype=np.uint8)
    
    # Extract RGB if available
    rgb = None
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        try:
            rgb = np.vstack([
                np.array(las.red, dtype=np.float32),
                np.array(las.green, dtype=np.float32),
                np.array(las.blue, dtype=np.float32)
            ]).T / 65535.0
        except Exception as e:
            logger.debug(f"Could not extract RGB: {e}")
    
    # Extract NIR if available
    nir = None
    if hasattr(las, 'nir'):
        try:
            nir = np.array(las.nir, dtype=np.float32) / 65535.0
        except Exception as e:
            logger.debug(f"Could not extract NIR: {e}")
    
    # Create LiDARData object
    data = LiDARData(
        points=points,
        intensity=intensity,
        return_number=return_number,
        classification=classification,
        rgb=rgb,
        nir=nir,
        file_path=file_path
    )
    
    logger.debug(
        f"Loaded {data.num_points:,} points from {file_path.name} | "
        f"Classes: {data.num_classes}"
    )
    
    # Apply bounding box filter if specified
    if bbox is not None:
        data = data.filter_by_bbox(bbox)
        logger.debug(f"After bbox filter: {data.num_points:,} points")
    
    return data


def validate_lidar_data(data: LiDARData, min_points: int = 100) -> bool:
    """
    Validate LiDAR data for common issues.
    
    Args:
        data: LiDARData object to validate
        min_points: Minimum required number of points
        
    Returns:
        True if data is valid
        
    Raises:
        ValueError: If validation fails
    """
    # Check minimum points
    if data.num_points < min_points:
        raise ValueError(
            f"Insufficient points: {data.num_points} < {min_points}"
        )
    
    # Check array shapes match
    if not (
        len(data.intensity) == data.num_points and
        len(data.return_number) == data.num_points and
        len(data.classification) == data.num_points
    ):
        raise ValueError("Array shape mismatch")
    
    # Check for NaN or Inf in coordinates
    if np.any(~np.isfinite(data.points)):
        raise ValueError("Invalid coordinates (NaN or Inf detected)")
    
    # Check intensity range [0, 1]
    if np.any(data.intensity < 0) or np.any(data.intensity > 1):
        raise ValueError(f"Intensity out of range [0, 1]")
    
    # Check RGB range if present
    if data.rgb is not None:
        if np.any(data.rgb < 0) or np.any(data.rgb > 1):
            raise ValueError(f"RGB out of range [0, 1]")
    
    # Check NIR range if present
    if data.nir is not None:
        if np.any(data.nir < 0) or np.any(data.nir > 1):
            raise ValueError(f"NIR out of range [0, 1]")
    
    return True


def map_classification(
    classification: np.ndarray,
    mapping: Dict[int, int],
    default_class: int = 0
) -> np.ndarray:
    """
    Map ASPRS classification codes to LOD classes.
    
    Args:
        classification: Array of ASPRS classification codes
        mapping: Dictionary mapping ASPRS codes to LOD codes
        default_class: Default class for unmapped codes
        
    Returns:
        Array of mapped classification codes
    """
    mapped = np.full_like(classification, default_class, dtype=np.uint8)
    
    for asprs_code, lod_code in mapping.items():
        mask = classification == asprs_code
        mapped[mask] = lod_code
    
    return mapped


def get_tile_info(file_path: Path) -> Dict[str, any]:
    """
    Extract tile information from filename.
    
    Expected format: LHD_FXX_0649_6863_*.laz
    
    Args:
        file_path: Path to LAZ file
        
    Returns:
        Dictionary with tile information
    """
    filename = file_path.stem
    parts = filename.split('_')
    
    info = {
        'filename': filename,
        'tile_x': None,
        'tile_y': None,
        'tile_center_x': None,
        'tile_center_y': None
    }
    
    # Try to extract tile coordinates
    if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
        tile_x = int(parts[2])
        tile_y = int(parts[3])
        
        # LAMB93 tiles are 1km x 1km
        # Tile center at (tile_x * 1000 + 500, tile_y * 1000 + 500)
        info['tile_x'] = tile_x
        info['tile_y'] = tile_y
        info['tile_center_x'] = tile_x * 1000 + 500
        info['tile_center_y'] = tile_y * 1000 + 500
    
    return info


def estimate_file_size_gb(file_path: Path) -> float:
    """
    Get file size in gigabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in GB
    """
    if file_path.exists():
        size_bytes = file_path.stat().st_size
        return size_bytes / (1024 ** 3)
    return 0.0


def check_file_readable(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Quick check if LAZ file is readable.
    
    Args:
        file_path: Path to LAZ file
        
    Returns:
        Tuple of (is_readable, error_message)
    """
    if not LASPY_AVAILABLE:
        return False, "laspy not installed"
    
    if not file_path.exists():
        return False, "File not found"
    
    try:
        # Try to read just the header (fast check)
        las = laspy.read(str(file_path))
        num_points = len(las.points)
        if num_points == 0:
            return False, "File contains no points"
        return True, None
    except Exception as e:
        return False, str(e)
