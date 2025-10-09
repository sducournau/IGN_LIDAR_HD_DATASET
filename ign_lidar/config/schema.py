"""
Structured configuration schema for IGN LiDAR HD.

Uses dataclasses with OmegaConf for type-safe, validated configuration.
Compatible with Hydra for hierarchical composition and CLI overrides.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union
from omegaconf import MISSING


@dataclass
class ProcessorConfig:
    """
    Configuration for the main LiDAR processor.
    
    Attributes:
        lod_level: Level of Detail classification target ('LOD2' or 'LOD3')
        use_gpu: Enable GPU acceleration (requires CuPy)
        num_workers: Number of parallel workers for processing
        patch_size: Size of patches in meters
        patch_overlap: Overlap ratio between patches (0.0-1.0)
        num_points: Target number of points per patch
        augment: Enable data augmentation
        num_augmentations: Number of augmentations per patch
        batch_size: Batch size for processing ('auto' or int)
        prefetch_factor: Number of batches to prefetch
        pin_memory: Pin memory for faster GPU transfer
    """
    lod_level: Literal["LOD2", "LOD3"] = "LOD2"
    use_gpu: bool = False
    num_workers: int = 4
    
    # Patch configuration
    patch_size: float = 150.0
    patch_overlap: float = 0.1
    num_points: int = 16384
    
    # Augmentation
    augment: bool = False
    num_augmentations: int = 3
    
    # Performance tuning
    batch_size: Union[str, int] = "auto"
    prefetch_factor: int = 2
    pin_memory: bool = False


@dataclass
class FeaturesConfig:
    """
    Configuration for feature computation.
    
    Attributes:
        mode: Feature computation mode ('minimal', 'full', 'custom')
        k_neighbors: Number of neighbors for geometric features
        include_extra: Include extra features (height stats, verticality, etc.)
        use_rgb: Include RGB from IGN orthophotos
        use_infrared: Include near-infrared from IRC
        compute_ndvi: Compute NDVI vegetation index
        sampling_method: Point sampling strategy ('random', 'fps', 'grid')
        normalize_xyz: Normalize XYZ coordinates to [-1, 1]
        normalize_features: Standardize features (z-score)
    """
    mode: Literal["minimal", "full", "custom"] = "full"
    k_neighbors: int = 20
    
    # Feature flags
    include_extra: bool = False
    use_rgb: bool = False
    use_infrared: bool = False
    compute_ndvi: bool = False
    
    # PointNet++ specific optimizations
    sampling_method: Literal["random", "fps", "grid"] = "random"
    normalize_xyz: bool = False
    normalize_features: bool = False
    
    # GPU configuration
    gpu_batch_size: int = 1_000_000
    use_gpu_chunked: bool = True


@dataclass
class PreprocessConfig:
    """
    Configuration for preprocessing (outlier removal, etc.).
    
    Attributes:
        enabled: Enable preprocessing
        sor_k: Number of neighbors for Statistical Outlier Removal
        sor_std: Standard deviation threshold for SOR
        ror_radius: Search radius for Radius Outlier Removal (meters)
        ror_neighbors: Minimum neighbors for ROR
        voxel_enabled: Enable voxel downsampling
        voxel_size: Voxel size for downsampling (meters)
    """
    enabled: bool = False
    
    # Statistical Outlier Removal
    sor_k: int = 12
    sor_std: float = 2.0
    
    # Radius Outlier Removal
    ror_radius: float = 1.0
    ror_neighbors: int = 4
    
    # Voxel downsampling
    voxel_enabled: bool = False
    voxel_size: float = 0.1


@dataclass
class StitchingConfig:
    """
    Configuration for tile stitching (boundary-aware processing).
    
    Attributes:
        enabled: Enable tile stitching
        buffer_size: Buffer zone size in meters
        auto_detect_neighbors: Automatically detect neighbor tiles
        cache_enabled: Cache loaded tiles to avoid re-reading
    """
    enabled: bool = False
    buffer_size: float = 10.0
    auto_detect_neighbors: bool = True
    cache_enabled: bool = True


@dataclass
class OutputConfig:
    """
    Configuration for output formats and saving.
    
    Attributes:
        format: Output format ('npz', 'hdf5', 'torch', 'laz', 'all')
        save_enriched_laz: Save enriched LAZ files with features
        only_enriched_laz: If True, only save enriched LAZ files (skip patch creation)
        save_stats: Save processing statistics
        save_metadata: Save patch metadata
        compression: Compression level (0-9, None for no compression)
    """
    format: Literal["npz", "hdf5", "torch", "laz", "all"] = "npz"
    save_enriched_laz: bool = False
    only_enriched_laz: bool = False
    save_stats: bool = True
    save_metadata: bool = True
    compression: Optional[int] = None


@dataclass
class BBoxConfig:
    """
    Optional bounding box configuration for spatial filtering.
    
    Attributes:
        xmin: Minimum X coordinate
        ymin: Minimum Y coordinate
        xmax: Maximum X coordinate
        ymax: Maximum Y coordinate
    """
    xmin: Optional[float] = None
    ymin: Optional[float] = None
    xmax: Optional[float] = None
    ymax: Optional[float] = None
    
    def to_tuple(self) -> Optional[tuple]:
        """Convert to tuple format (xmin, ymin, xmax, ymax)."""
        if all(v is not None for v in [self.xmin, self.ymin, self.xmax, self.ymax]):
            return (self.xmin, self.ymin, self.xmax, self.ymax)
        return None


@dataclass
class IGNLiDARConfig:
    """
    Root configuration for IGN LiDAR HD processing.
    
    This is the main configuration class that composes all sub-configurations.
    Used with Hydra for hierarchical configuration management.
    
    Example:
        >>> cfg = IGNLiDARConfig(
        ...     input_dir="data/raw",
        ...     output_dir="data/patches",
        ...     processor=ProcessorConfig(use_gpu=True)
        ... )
    """
    # Sub-configurations (populated from YAML files)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    bbox: BBoxConfig = field(default_factory=BBoxConfig)
    
    # I/O paths (required - must be provided by user)
    input_dir: str = MISSING
    output_dir: str = MISSING
    
    # Global settings
    num_workers: int = 4
    verbose: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Hydra configuration composition
    # This defines which config groups to load by default
    defaults: List[str] = field(default_factory=lambda: [
        {"processor": "default"},
        {"features": "full"},
        {"preprocess": "default"},
        {"stitching": "disabled"},
        {"output": "default"},
        "_self_"
    ])
    
    def validate(self) -> None:
        """
        Validate configuration consistency.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate patch configuration
        if self.processor.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        
        if not 0 <= self.processor.patch_overlap < 1:
            raise ValueError("patch_overlap must be in [0, 1)")
        
        if self.processor.num_points <= 0:
            raise ValueError("num_points must be > 0")
        
        # Validate feature configuration
        if self.features.k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")
        
        # Validate preprocessing
        if self.preprocess.enabled:
            if self.preprocess.sor_k <= 0:
                raise ValueError("sor_k must be > 0")
            if self.preprocess.ror_radius <= 0:
                raise ValueError("ror_radius must be > 0")
        
        # Validate stitching
        if self.stitching.enabled:
            if self.stitching.buffer_size <= 0:
                raise ValueError("buffer_size must be > 0")
            
            # Buffer size should be >= 2 * radius used for features
            # Estimate radius from k_neighbors (rough heuristic)
            estimated_radius = self.features.k_neighbors * 0.5  # meters
            if self.stitching.buffer_size < 2 * estimated_radius:
                import logging
                logging.warning(
                    f"buffer_size ({self.stitching.buffer_size}m) should be "
                    f">= 2 * feature_radius (~{2*estimated_radius:.1f}m) "
                    f"for optimal boundary quality"
                )
