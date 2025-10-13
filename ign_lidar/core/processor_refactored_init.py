"""
Refactored LiDARProcessor __init__ - Draft Implementation

This is a draft showing how the new __init__ should look.
Will be integrated into processor.py after validation.
"""

from typing import Union, Dict
from omegaconf import DictConfig, OmegaConf
import logging

from ..classes import ASPRS_TO_LOD2, ASPRS_TO_LOD3
from .skip_checker import PatchSkipChecker
from .modules.feature_manager import FeatureManager
from .modules.config_validator import ConfigValidator

logger = logging.getLogger(__name__)


class LiDARProcessorRefactored:
    """
    Refactored LiDAR Processor with simplified initialization.
    
    Accepts either a Hydra config object or individual parameters (backward compat).
    """
    
    def __init__(self, config: Union[DictConfig, Dict] = None, **kwargs):
        """
        Initialize processor with config object or individual parameters.
        
        Modern usage (config object):
            cfg = OmegaConf.load("config.yaml")
            processor = LiDARProcessor(config=cfg)
        
        Legacy usage (individual params - backward compatible):
            processor = LiDARProcessor(
                lod_level='LOD2',
                use_gpu=True,
                patch_size=150.0,
                ...
            )
        
        Args:
            config: Hydra config object or dict (preferred)
            **kwargs: Individual parameters for backward compatibility
        """
        # Handle both config object and individual parameters
        if config is None:
            # Legacy mode: build config from kwargs
            config = self._build_config_from_kwargs(kwargs)
        elif not isinstance(config, (DictConfig, dict)):
            raise TypeError(
                f"config must be DictConfig, dict, or None. Got: {type(config)}"
            )
        
        # Ensure config is OmegaConf for consistency
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        
        # Store config
        self.config = config
        
        # Extract commonly used values for convenience
        self.lod_level = config.processor.lod_level
        self.processing_mode = config.output.processing_mode
        self.use_gpu = config.processor.use_gpu
        self.architecture = config.processor.get('architecture', 'pointnet++')
        self.output_format = config.output.format
        
        # Derive save/only flags from processing mode
        self.save_enriched_laz = self.processing_mode in ["both", "enriched_only"]
        self.only_enriched_laz = self.processing_mode == "enriched_only"
        
        logger.info(f"✨ Processing mode: {self.processing_mode}")
        
        # Validate configuration
        self._validate_config()
        
        # Initialize feature manager (handles RGB, NIR, GPU)
        self.feature_manager = FeatureManager(config)
        
        # Initialize stitcher if needed
        self.stitcher = ConfigValidator.init_stitcher(
            config.stitching if hasattr(config, 'stitching') else {}
        )
        
        # Set class mapping based on LOD level
        if self.lod_level == 'LOD2':
            self.class_mapping = ASPRS_TO_LOD2
            self.default_class = 14
        else:
            self.class_mapping = ASPRS_TO_LOD3
            self.default_class = 29
        
        # Initialize intelligent skip checker
        self.skip_checker = PatchSkipChecker(
            output_format=self.output_format,
            architecture=self.architecture,
            num_augmentations=config.processor.get('num_augmentations', 3),
            augment=config.processor.get('augment', False),
            validate_content=True,
            min_file_size=1024,
            only_enriched_laz=self.only_enriched_laz,
        )
        
        logger.info(f"Initialized LiDARProcessor with {self.lod_level}")
    
    def _validate_config(self):
        """Validate configuration using ConfigValidator."""
        # Validate output format
        formats_list = ConfigValidator.validate_output_format(self.output_format)
        
        # Check PyTorch availability
        ConfigValidator.check_pytorch_availability(formats_list)
        
        # Validate processing mode
        ConfigValidator.validate_processing_mode(self.processing_mode)
    
    def _build_config_from_kwargs(self, kwargs: Dict) -> DictConfig:
        """
        Build Hydra config from legacy kwargs for backward compatibility.
        
        This allows existing code using individual parameters to continue working.
        """
        # Extract parameters with defaults
        lod_level = kwargs.get('lod_level', 'LOD2')
        processing_mode = kwargs.get('processing_mode', 'patches_only')
        use_gpu = kwargs.get('use_gpu', False)
        
        # Build structured config
        config = OmegaConf.create({
            'processor': {
                'lod_level': lod_level,
                'use_gpu': use_gpu,
                'use_gpu_chunked': kwargs.get('use_gpu_chunked', True),
                'gpu_batch_size': kwargs.get('gpu_batch_size', 1_000_000),
                'patch_size': kwargs.get('patch_size', 150.0),
                'patch_overlap': kwargs.get('patch_overlap', 0.1),
                'num_points': kwargs.get('num_points', 16384),
                'augment': kwargs.get('augment', False),
                'num_augmentations': kwargs.get('num_augmentations', 3),
                'num_workers': kwargs.get('num_workers', 4),
                'architecture': kwargs.get('architecture', 'pointnet++'),
            },
            'features': {
                'mode': kwargs.get('feature_mode', 'full'),
                'k_neighbors': kwargs.get('k_neighbors', 20),
                'include_extra': kwargs.get('include_extra_features', False),
                'use_rgb': kwargs.get('include_rgb', False),
                'rgb_cache_dir': kwargs.get('rgb_cache_dir', None),
                'use_infrared': kwargs.get('include_infrared', False),
                'compute_ndvi': kwargs.get('compute_ndvi', False),
                'include_architectural_style': kwargs.get('include_architectural_style', False),
                'style_encoding': kwargs.get('style_encoding', 'constant'),
            },
            'preprocess': {
                'enabled': kwargs.get('preprocess', False),
                'config': kwargs.get('preprocess_config', None),
            },
            'stitching': {
                'enabled': kwargs.get('use_stitching', False),
                'buffer_size': kwargs.get('buffer_size', 10.0),
                'config': kwargs.get('stitching_config', None),
            },
            'output': {
                'format': kwargs.get('output_format', 'npz'),
                'processing_mode': processing_mode,
            },
            'bbox': kwargs.get('bbox', None),
        })
        
        return config
    
    # Backward compatibility properties
    @property
    def rgb_fetcher(self):
        """Backward compatibility: access RGB fetcher."""
        return self.feature_manager.rgb_fetcher
    
    @property
    def infrared_fetcher(self):
        """Backward compatibility: access infrared fetcher."""
        return self.feature_manager.infrared_fetcher
    
    @property
    def patch_size(self):
        """Backward compatibility: access patch size."""
        return self.config.processor.patch_size
    
    @property
    def num_points(self):
        """Backward compatibility: access num_points."""
        return self.config.processor.num_points
    
    # ... more properties as needed for backward compatibility


# Example usage
if __name__ == "__main__":
    # Modern usage with config
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'processor': {
            'lod_level': 'LOD2',
            'use_gpu': False,
            'patch_size': 150.0,
            'num_points': 16384,
            'architecture': 'pointnet++',
        },
        'features': {
            'mode': 'full',
            'use_rgb': False,
            'use_infrared': False,
        },
        'output': {
            'format': 'npz',
            'processing_mode': 'patches_only',
        }
    })
    
    processor = LiDARProcessorRefactored(config=config)
    print(f"Processor initialized: {processor.lod_level}, GPU: {processor.use_gpu}")
    
    # Legacy usage (backward compatible)
    processor_legacy = LiDARProcessorRefactored(
        lod_level='LOD3',
        use_gpu=True,
        patch_size=100.0
    )
    print(f"Legacy processor: {processor_legacy.lod_level}, GPU: {processor_legacy.use_gpu}")
