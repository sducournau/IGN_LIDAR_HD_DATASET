"""
PyTorch Dataset for IGN LiDAR HD multi-architecture support.

Provides flexible data loading for:
- PointNet++
- Octree-CNN
- Point Transformer
- Sparse Convolutions
- Hybrid models
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

# Try to import formatters
try:
    from ..io.formatters import MultiArchitectureFormatter
except ImportError:
    MultiArchitectureFormatter = None

logger = logging.getLogger(__name__)


class IGNLiDARMultiArchDataset(Dataset):
    """
    PyTorch Dataset for IGN LiDAR HD patches with multi-architecture support.
    
    Features:
    - Lazy loading (load on __getitem__)
    - Multiple architecture formats
    - Optional GPU cache
    - On-the-fly augmentation
    - Configuration presets
    - Multi-modal features (RGB, NIR, NDVI, Geometric, etc.)
    
    Examples:
        >>> # Buildings classification (lightweight)
        >>> dataset = IGNLiDARMultiArchDataset(
        ...     'data/patches',
        ...     architecture='pointnet++',
        ...     preset='buildings'
        ... )
        
        >>> # Vegetation segmentation (NDVI-focused)
        >>> dataset = IGNLiDARMultiArchDataset(
        ...     'data/patches',
        ...     architecture='transformer',
        ...     preset='vegetation',
        ...     use_infrared=True,
        ...     compute_ndvi=True
        ... )
        
        >>> # SOTA semantic segmentation (full features)
        >>> dataset = IGNLiDARMultiArchDataset(
        ...     'data/patches',
        ...     architecture='transformer',
        ...     preset='semantic_sota',
        ...     use_rgb=True,
        ...     use_infrared=True,
        ...     use_geometric=True,
        ...     use_radiometric=True,
        ...     use_contextual=True
        ... )
    """
    
    # Configuration presets
    PRESETS = {
        'buildings': {
            'description': 'LOD2/LOD3 building classification (lightweight)',
            'use_rgb': False,
            'use_infrared': False,
            'use_geometric': True,
            'use_radiometric': False,
            'use_contextual': False,
            'num_points': 8192,
            'features': ['xyz', 'normals', 'planarity', 'verticality', 'height'],
            'expected_accuracy': 0.85,
        },
        'vegetation': {
            'description': 'Vegetation segmentation (NDVI-focused)',
            'use_rgb': True,
            'use_infrared': True,
            'use_geometric': True,
            'use_radiometric': True,
            'use_contextual': False,
            'num_points': 16384,
            'features': ['xyz', 'rgb', 'nir', 'ndvi', 'sphericity', 'return_number'],
            'expected_accuracy': 0.90,
        },
        'semantic_sota': {
            'description': 'SOTA semantic segmentation (full features)',
            'use_rgb': True,
            'use_infrared': True,
            'use_geometric': True,
            'use_radiometric': True,
            'use_contextual': True,
            'num_points': 16384,
            'features': 'all',
            'expected_accuracy': 0.94,
        },
        'fast': {
            'description': 'Fast inference (minimal features)',
            'use_rgb': False,
            'use_infrared': False,
            'use_geometric': True,
            'use_radiometric': False,
            'use_contextual': False,
            'num_points': 4096,
            'features': ['xyz', 'normals', 'planarity', 'verticality'],
            'expected_accuracy': 0.82,
        },
    }
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        architecture: str = 'pointnet++',
        preset: Optional[str] = None,
        # Architecture parameters
        num_points: int = 16384,
        octree_depth: int = 6,
        knn_k: int = 32,
        voxel_size: float = 0.1,
        # Feature flags
        use_rgb: bool = True,
        use_infrared: bool = False,
        use_geometric: bool = True,
        use_radiometric: bool = False,
        use_contextual: bool = False,
        # Normalization
        normalize: bool = True,
        normalize_rgb: bool = True,
        standardize_features: bool = True,
        # Augmentation
        augment: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None,
        # Performance
        cache_in_memory: bool = False,
        use_gpu_cache: bool = False,
        # Data split
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize IGN LiDAR HD dataset.
        
        Args:
            data_dir: Path to patches directory
            architecture: Target architecture ('pointnet++', 'octree', 'transformer', 'sparse_conv', 'hybrid')
            preset: Configuration preset ('buildings', 'vegetation', 'semantic_sota', 'fast')
            num_points: Number of points per patch (for sampling/padding)
            octree_depth: Octree depth (for octree architecture)
            knn_k: K neighbors (for transformer architecture)
            voxel_size: Voxel size in meters (for sparse_conv architecture)
            use_rgb: Use RGB features
            use_infrared: Use infrared (NIR) features
            use_geometric: Use geometric features (normals, curvature, etc.)
            use_radiometric: Use radiometric features (intensity, return_number, etc.)
            use_contextual: Use contextual features (local density, height stats, etc.)
            normalize: Normalize XYZ coordinates
            normalize_rgb: Normalize RGB to [0, 1]
            standardize_features: Standardize features (zero mean, unit variance)
            augment: Apply data augmentation
            augmentation_config: Custom augmentation configuration
            cache_in_memory: Cache all patches in RAM
            use_gpu_cache: Cache on GPU (requires CUDA)
            split: Data split ('train', 'val', 'test')
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.architecture = architecture
        self.split = split
        
        # Apply preset if specified
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError(
                    f"Unknown preset '{preset}'. "
                    f"Available: {list(self.PRESETS.keys())}"
                )
            logger.info(f"📋 Using preset: '{preset}' - {self.PRESETS[preset]['description']}")
            preset_config = self.PRESETS[preset]
            
            # Override with preset values
            use_rgb = preset_config.get('use_rgb', use_rgb)
            use_infrared = preset_config.get('use_infrared', use_infrared)
            use_geometric = preset_config.get('use_geometric', use_geometric)
            use_radiometric = preset_config.get('use_radiometric', use_radiometric)
            use_contextual = preset_config.get('use_contextual', use_contextual)
            num_points = preset_config.get('num_points', num_points)
            
            logger.info(
                f"   Expected accuracy: ~{preset_config['expected_accuracy']*100:.0f}%"
            )
        
        # Validate architecture
        valid_archs = ['pointnet++', 'octree', 'transformer', 'sparse_conv', 'hybrid']
        if architecture not in valid_archs:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Valid: {valid_archs}"
            )
        
        # Store configuration
        self.num_points = num_points
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        self.use_gpu_cache = use_gpu_cache
        
        # Initialize formatter
        if MultiArchitectureFormatter is None:
            raise ImportError(
                "MultiArchitectureFormatter not available. "
                "Please ensure ign_lidar.formatters is properly installed."
            )
        
        self.formatter = MultiArchitectureFormatter(
            target_archs=[architecture],
            num_points=num_points,
            octree_depth=octree_depth,
            knn_k=knn_k,
            voxel_size=voxel_size,
            normalize=normalize,
            standardize_features=standardize_features,
            use_rgb=use_rgb,
            use_infrared=use_infrared,
            use_geometric=use_geometric,
            use_radiometric=use_radiometric,
            use_contextual=use_contextual,
        )
        
        # Load patch list
        self.patch_files = self._load_patch_list()
        
        # Split data
        self.patch_files = self._split_data(
            self.patch_files,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_seed=random_seed,
        )
        
        logger.info(
            f"📊 Dataset initialized: {len(self.patch_files)} patches "
            f"({split} split, architecture={architecture})"
        )
        
        # Initialize cache
        self.cache: Dict[int, Any] = {}
        if cache_in_memory:
            logger.info("💾 Pre-loading all patches into memory...")
            self._preload_cache()
        
        # Initialize augmentation
        self.augmentation = None
        if augment:
            try:
                from .augmentation import PatchAugmentation
                self.augmentation = PatchAugmentation(
                    config=augmentation_config or {}
                )
                logger.info("🔄 Data augmentation enabled")
            except ImportError:
                logger.warning("⚠️  Augmentation requested but module not available")
    
    def _load_patch_list(self) -> List[Path]:
        """
        Load list of patch files from data directory.
        
        Returns:
            List of patch file paths
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Look for .npz or .npy files
        patch_files = list(self.data_dir.glob("**/*.npz"))
        if not patch_files:
            patch_files = list(self.data_dir.glob("**/*.npy"))
        
        if not patch_files:
            raise FileNotFoundError(
                f"No patch files found in {self.data_dir}. "
                f"Expected .npz or .npy files."
            )
        
        # Sort for reproducibility
        patch_files = sorted(patch_files)
        
        return patch_files
    
    def _split_data(
        self,
        files: List[Path],
        split: str,
        train_ratio: float,
        val_ratio: float,
        random_seed: int,
    ) -> List[Path]:
        """
        Split data into train/val/test sets.
        
        Args:
            files: List of all patch files
            split: Desired split ('train', 'val', 'test')
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            random_seed: Random seed
            
        Returns:
            List of files for the specified split
        """
        # Set random seed for reproducibility
        rng = np.random.RandomState(random_seed)
        
        # Shuffle files
        indices = np.arange(len(files))
        rng.shuffle(indices)
        
        # Calculate split points
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split indices
        if split == 'train':
            split_indices = indices[:n_train]
        elif split == 'val':
            split_indices = indices[n_train:n_train + n_val]
        elif split == 'test':
            split_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        return [files[i] for i in split_indices]
    
    def _preload_cache(self):
        """Pre-load all patches into memory cache."""
        for idx in range(len(self)):
            self.cache[idx] = self._load_patch(idx)
        logger.info(f"✅ Cached {len(self.cache)} patches in memory")
    
    def _load_patch(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load a single patch from disk.
        
        Args:
            idx: Patch index
            
        Returns:
            Dictionary with patch data
        """
        patch_file = self.patch_files[idx]
        
        try:
            # Load .npz file
            data = np.load(patch_file, allow_pickle=True)
            
            # Convert to dictionary
            patch = {key: data[key] for key in data.files}
            
            return patch
            
        except Exception as e:
            logger.error(f"❌ Failed to load patch {patch_file}: {e}")
            raise
    
    def __len__(self) -> int:
        """Return number of patches in dataset."""
        return len(self.patch_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single patch by index.
        
        Args:
            idx: Patch index
            
        Returns:
            Dictionary with formatted patch data (as PyTorch tensors)
        """
        # Check cache first
        if idx in self.cache:
            patch = self.cache[idx]
        else:
            patch = self._load_patch(idx)
            if self.cache_in_memory:
                self.cache[idx] = patch
        
        # Apply augmentation if enabled
        if self.augment and self.augmentation is not None:
            patch = self.augmentation(patch)
        
        # Format for target architecture(s)
        formatted = self.formatter.format_patch(patch)
        
        # Extract specific architecture format (or all if hybrid)
        if self.architecture in formatted:
            arch_data = formatted[self.architecture]
        else:
            # For hybrid or if formatter returns all architectures
            arch_data = formatted
        
        # Convert to PyTorch tensors
        output = {}
        for key, value in arch_data.items():
            if isinstance(value, np.ndarray):
                # Move to GPU if cache is on GPU
                tensor = torch.from_numpy(value).float()
                if self.use_gpu_cache and torch.cuda.is_available():
                    tensor = tensor.cuda()
                output[key] = tensor
            elif isinstance(value, dict):
                # Nested dictionary (e.g., for hybrid models)
                output[key] = {
                    k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                output[key] = value
        
        return output
    
    def get_dataloader(
        self,
        batch_size: int = 16,
        shuffle: bool = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ) -> "torch.utils.data.DataLoader":
        """
        Create a PyTorch DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Shuffle data (default: True for train, False for val/test)
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            **kwargs: Additional DataLoader arguments
            
        Returns:
            PyTorch DataLoader
        """
        if shuffle is None:
            shuffle = (self.split == 'train')
        
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_patches': len(self),
            'split': self.split,
            'architecture': self.architecture,
            'num_points': self.num_points,
            'augmentation': self.augment,
            'cached': self.cache_in_memory,
            'gpu_cache': self.use_gpu_cache,
        }
        
        # Sample first patch for feature info
        if len(self) > 0:
            sample = self[0]
            if 'features' in sample:
                stats['num_features'] = sample['features'].shape[-1]
            if 'points' in sample:
                stats['points_shape'] = tuple(sample['points'].shape)
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of dataset."""
        stats = self.get_stats()
        return (
            f"IGNLiDARMultiArchDataset(\n"
            f"  num_patches={stats['num_patches']},\n"
            f"  split='{stats['split']}',\n"
            f"  architecture='{stats['architecture']}',\n"
            f"  num_points={stats['num_points']},\n"
            f"  augmentation={stats['augmentation']},\n"
            f"  cached={stats['cached']}\n"
            f")"
        )
