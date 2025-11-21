"""
Dataset Manager for ML Training Dataset Creation

This module handles automatic train/validation/test splitting during patch processing,
supporting multiple patch sizes (50m, 100m, 150m) for multi-scale training.

Features:
- Automatic train/val/test split with configurable ratios
- Deterministic splitting based on seed for reproducibility
- Support for multiple patch sizes in a single dataset
- Metadata tracking for dataset statistics
- Compatible with existing IGN LiDAR HD pipeline
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass, field
import json
import hashlib
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

SplitType = Literal["train", "val", "test"]


@dataclass
class DatasetConfig:
    """Configuration for dataset creation.
    
    Attributes:
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_seed: Random seed for reproducible splitting (default: 42)
        split_by_tile: Split by tile (True) or by patch (False) to avoid data leakage
        create_split_dirs: Create separate train/val/test directories
        patch_sizes: List of patch sizes to support (e.g., [50, 100, 150])
        balance_across_sizes: Balance samples across different patch sizes
    """
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    split_by_tile: bool = True
    create_split_dirs: bool = True
    patch_sizes: List[int] = field(default_factory=lambda: [50, 100, 150])
    balance_across_sizes: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.3f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )
        
        if self.train_ratio < 0 or self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("Split ratios must be non-negative")


class DatasetManager:
    """
    Manages ML dataset creation with automatic train/val/test splitting.
    
    This class handles:
    - Deterministic tile/patch assignment to train/val/test splits
    - Directory structure creation for split datasets
    - Metadata tracking and statistics generation
    - Support for multiple patch sizes (multi-scale training)
    
    Examples:
        >>> # Single patch size dataset
        >>> config = DatasetConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        >>> manager = DatasetManager(output_dir="data/patches", config=config)
        >>> 
        >>> # Determine split for a tile
        >>> split = manager.get_tile_split("LHD_FXX_0649_6863")
        >>> save_path = manager.get_patch_path(
        ...     tile_name="LHD_FXX_0649_6863",
        ...     patch_idx=0,
        ...     split=split
        ... )
        
        >>> # Multi-scale dataset
        >>> config = DatasetConfig(
        ...     patch_sizes=[50, 100, 150],
        ...     balance_across_sizes=True
        ... )
        >>> manager = DatasetManager(output_dir="data/patches_multiscale", config=config)
    """
    
    def __init__(
        self,
        output_dir: Path,
        config: Optional[DatasetConfig] = None,
        patch_size: int = 150,
    ):
        """
        Initialize dataset manager.
        
        Args:
            output_dir: Base output directory for patches
            config: Dataset configuration (if None, uses defaults)
            patch_size: Current patch size in meters (for multi-scale support)
        """
        self.output_dir = Path(output_dir)
        self.config = config or DatasetConfig()
        self.patch_size = patch_size
        
        # Initialize random state for reproducible splits
        self.rng = np.random.RandomState(self.config.random_seed)
        
        # Create directory structure
        if self.config.create_split_dirs:
            for split in ["train", "val", "test"]:
                split_dir = self.output_dir / split
                split_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created split directory: {split_dir}")
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking for statistics
        self.tile_splits: Dict[str, SplitType] = {}
        self.patch_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.tiles_processed: List[str] = []
        
        # Load existing split assignments if available
        self._load_split_assignments()
        
        logger.info(f"📊 Dataset Manager initialized")
        logger.info(f"   Output: {self.output_dir}")
        logger.info(f"   Split ratios: train={self.config.train_ratio:.1%}, "
                   f"val={self.config.val_ratio:.1%}, test={self.config.test_ratio:.1%}")
        logger.info(f"   Patch size: {patch_size}m")
        logger.info(f"   Split by tile: {self.config.split_by_tile}")
        logger.info(f"   Random seed: {self.config.random_seed}")
    
    def get_tile_split(self, tile_name: str) -> SplitType:
        """
        Determine which split a tile belongs to (train/val/test).
        
        Uses deterministic hashing of tile name to ensure:
        - Same tile always goes to same split (reproducibility)
        - Tiles are distributed according to configured ratios
        - No randomness once seed is set
        
        Args:
            tile_name: Tile identifier (e.g., "LHD_FXX_0649_6863")
            
        Returns:
            Split assignment ("train", "val", or "test")
        """
        # Check if already assigned
        if tile_name in self.tile_splits:
            return self.tile_splits[tile_name]
        
        # Use hash of tile name + seed for deterministic assignment
        hash_input = f"{tile_name}_{self.config.random_seed}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Convert hash to 0-1 range
        hash_ratio = (hash_value % 1000000) / 1000000.0
        
        # Assign to split based on ratios
        if hash_ratio < self.config.train_ratio:
            split = "train"
        elif hash_ratio < (self.config.train_ratio + self.config.val_ratio):
            split = "val"
        else:
            split = "test"
        
        # Cache assignment
        self.tile_splits[tile_name] = split
        
        logger.debug(f"Tile {tile_name} assigned to {split} split")
        return split
    
    def get_patch_path(
        self,
        tile_name: str,
        patch_idx: int,
        architecture: str = "hybrid",
        version: str = "original",
        split: Optional[SplitType] = None,
        extension: str = "npz"
    ) -> Path:
        """
        Get the full path for saving a patch.
        
        Args:
            tile_name: Tile identifier
            patch_idx: Patch index within tile
            architecture: Target architecture (e.g., 'hybrid', 'pointnet++')
            version: Patch version ('original' or augmentation id)
            split: Split assignment (auto-determined if None)
            extension: File extension (default: 'npz')
            
        Returns:
            Full path where patch should be saved
        """
        # Determine split if not provided
        if split is None:
            split = self.get_tile_split(tile_name)
        
        # Build filename
        if version == "original":
            filename = f"{tile_name}_{architecture}_patch_{patch_idx:04d}"
        else:
            filename = f"{tile_name}_{architecture}_patch_{patch_idx:04d}_{version}"
        
        # Add patch size suffix for multi-scale datasets
        if len(self.config.patch_sizes) > 1:
            filename = f"{filename}_scale{self.patch_size}m"
        
        filename = f"{filename}.{extension}"
        
        # Build path with or without split directories
        if self.config.create_split_dirs:
            return self.output_dir / split / filename
        else:
            # Add split prefix to filename if not using directories
            return self.output_dir / f"{split}_{filename}"
    
    def record_patch_saved(
        self,
        tile_name: str,
        split: SplitType,
        patch_size: Optional[int] = None
    ):
        """
        Record that a patch has been saved for statistics tracking.
        
        Args:
            tile_name: Tile identifier
            split: Split the patch belongs to
            patch_size: Patch size in meters (uses self.patch_size if None)
        """
        if patch_size is None:
            patch_size = self.patch_size
        
        # Track patch count by split and size
        size_key = f"{patch_size}m"
        self.patch_counts[split][size_key] += 1
        
        # Track tiles processed
        if tile_name not in self.tiles_processed:
            self.tiles_processed.append(tile_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing:
            - Total patches per split
            - Patches per split per patch size
            - Number of tiles processed
            - Split ratios achieved
        """
        total_patches = sum(
            sum(counts.values()) 
            for counts in self.patch_counts.values()
        )
        
        stats = {
            "total_patches": total_patches,
            "total_tiles": len(self.tiles_processed),
            "patch_size_current": f"{self.patch_size}m",
            "patch_sizes_supported": [f"{s}m" for s in self.config.patch_sizes],
            "splits": {},
            "config": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
                "random_seed": self.config.random_seed,
                "split_by_tile": self.config.split_by_tile,
            }
        }
        
        # Calculate statistics per split
        for split in ["train", "val", "test"]:
            split_total = sum(self.patch_counts[split].values())
            split_ratio = split_total / total_patches if total_patches > 0 else 0
            
            stats["splits"][split] = {
                "count": split_total,
                "ratio": split_ratio,
                "by_patch_size": dict(self.patch_counts[split])
            }
        
        return stats
    
    def save_metadata(self, additional_info: Optional[Dict[str, Any]] = None):
        """
        Save dataset metadata to JSON file.
        
        Args:
            additional_info: Optional additional information to include
        """
        stats = self.get_statistics()
        
        if additional_info:
            stats["additional_info"] = additional_info
        
        # Save to output directory
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"💾 Dataset metadata saved: {metadata_path}")
        
        # Print summary
        total = stats["total_patches"]
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"   Total patches: {total:,}")
        logger.info(f"   Total tiles: {stats['total_tiles']:,}")
        for split in ["train", "val", "test"]:
            count = stats["splits"][split]["count"]
            ratio = stats["splits"][split]["ratio"]
            logger.info(f"   {split.capitalize():5s}: {count:,} ({ratio:.1%})")
    
    def _load_split_assignments(self):
        """Load existing split assignments from metadata if available."""
        metadata_path = self.output_dir / "dataset_metadata.json"
        
        if not metadata_path.exists():
            return
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Reconstruct tile splits if possible
            # This ensures consistency when resuming dataset creation
            logger.debug(f"Loaded existing metadata from {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Could not load existing metadata: {e}")
    
    def _save_split_assignments(self):
        """Save current split assignments for reproducibility."""
        assignments_path = self.output_dir / "split_assignments.json"
        
        with open(assignments_path, 'w') as f:
            json.dump(self.tile_splits, f, indent=2)
        
        logger.debug(f"Split assignments saved: {assignments_path}")


class MultiScaleDatasetManager:
    """
    Manager for multi-scale datasets combining patches of different sizes.
    
    Coordinates multiple DatasetManager instances to create a cohesive
    multi-scale training dataset.
    
    Examples:
        >>> # Create multi-scale dataset
        >>> manager = MultiScaleDatasetManager(
        ...     base_output_dir="data/patches_multiscale",
        ...     patch_sizes=[50, 100, 150],
        ...     config=DatasetConfig(train_ratio=0.7)
        ... )
        >>> 
        >>> # Get manager for specific patch size
        >>> manager_50m = manager.get_manager_for_size(50)
    """
    
    def __init__(
        self,
        base_output_dir: Path,
        patch_sizes: List[int],
        config: Optional[DatasetConfig] = None,
        balance_samples: bool = False,
    ):
        """
        Initialize multi-scale dataset manager.
        
        Args:
            base_output_dir: Base directory for all patch sizes
            patch_sizes: List of patch sizes to support (e.g., [50, 100, 150])
            config: Dataset configuration (shared across all sizes)
            balance_samples: Balance number of samples across patch sizes
        """
        self.base_output_dir = Path(base_output_dir)
        self.patch_sizes = sorted(patch_sizes)
        self.config = config or DatasetConfig(patch_sizes=patch_sizes)
        self.balance_samples = balance_samples
        
        # Create managers for each patch size
        self.managers: Dict[int, DatasetManager] = {}
        
        for size in patch_sizes:
            manager = DatasetManager(
                output_dir=self.base_output_dir,
                config=self.config,
                patch_size=size
            )
            self.managers[size] = manager
        
        logger.info(f"🔗 Multi-scale dataset manager initialized")
        logger.info(f"   Patch sizes: {', '.join(f'{s}m' for s in patch_sizes)}")
        logger.info(f"   Output: {self.base_output_dir}")
    
    def get_manager_for_size(self, patch_size: int) -> DatasetManager:
        """
        Get dataset manager for specific patch size.
        
        Args:
            patch_size: Patch size in meters
            
        Returns:
            DatasetManager for that size
        """
        if patch_size not in self.managers:
            raise ValueError(
                f"No manager for patch size {patch_size}m. "
                f"Available sizes: {self.patch_sizes}"
            )
        return self.managers[patch_size]
    
    def get_combined_statistics(self) -> Dict[str, Any]:
        """
        Get combined statistics across all patch sizes.
        
        Returns:
            Dictionary with aggregated statistics
        """
        combined = {
            "patch_sizes": [f"{s}m" for s in self.patch_sizes],
            "total_patches": 0,
            "by_size": {},
            "by_split": defaultdict(lambda: {"count": 0, "by_size": {}})
        }
        
        for size, manager in self.managers.items():
            stats = manager.get_statistics()
            size_key = f"{size}m"
            
            combined["by_size"][size_key] = stats["total_patches"]
            combined["total_patches"] += stats["total_patches"]
            
            for split in ["train", "val", "test"]:
                split_stats = stats["splits"][split]
                combined["by_split"][split]["count"] += split_stats["count"]
                combined["by_split"][split]["by_size"][size_key] = split_stats["count"]
        
        return combined
    
    def save_combined_metadata(self):
        """Save combined metadata for multi-scale dataset."""
        stats = self.get_combined_statistics()
        
        metadata_path = self.base_output_dir / "multiscale_dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"💾 Multi-scale metadata saved: {metadata_path}")
        
        # Print summary
        logger.info(f"📊 Multi-Scale Dataset Statistics:")
        logger.info(f"   Total patches: {stats['total_patches']:,}")
        for size_key, count in stats['by_size'].items():
            logger.info(f"   {size_key} patches: {count:,}")
        
        for split in ["train", "val", "test"]:
            split_stats = stats["by_split"][split]
            logger.info(f"   {split.capitalize():5s}: {split_stats['count']:,}")
