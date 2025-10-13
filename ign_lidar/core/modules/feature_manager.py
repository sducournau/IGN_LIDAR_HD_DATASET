"""
Feature Manager Module

Manages feature computation resources including RGB/NIR fetchers and GPU validation.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FeatureManager:
    """
    Manages feature computation resources (RGB, NIR, GPU).
    
    Centralizes initialization of:
    - RGB orthophoto fetcher
    - Infrared (NIR) fetcher  
    - GPU availability validation
    """
    
    def __init__(self, config):
        """
        Initialize feature manager from config.
        
        Args:
            config: Configuration object with features section
        """
        self.config = config
        self.rgb_fetcher = None
        self.infrared_fetcher = None
        self.use_gpu = False
        
        # Initialize components
        if hasattr(config, 'features') and config.features.get('use_rgb', False):
            self.rgb_fetcher = self._init_rgb_fetcher()
        
        if hasattr(config, 'features') and config.features.get('use_infrared', False):
            self.infrared_fetcher = self._init_infrared_fetcher()
        
        if hasattr(config, 'processor') and config.processor.get('use_gpu', False):
            self.use_gpu = self._validate_gpu()
    
    def _init_rgb_fetcher(self):
        """Initialize RGB orthophoto fetcher."""
        try:
            from ...preprocessing.rgb_augmentation import IGNOrthophotoFetcher
            
            # Determine cache directory
            rgb_cache_dir = getattr(self.config.features, 'rgb_cache_dir', None)
            if rgb_cache_dir is None:
                rgb_cache_dir = Path(tempfile.gettempdir()) / "ign_lidar_cache" / "orthophotos"
                rgb_cache_dir.mkdir(parents=True, exist_ok=True)
            else:
                rgb_cache_dir = Path(rgb_cache_dir)
            
            fetcher = IGNOrthophotoFetcher(cache_dir=rgb_cache_dir)
            logger.info(
                f"RGB enabled (will use from input LAZ if present, "
                f"otherwise fetch from IGN orthophotos)"
            )
            return fetcher
            
        except ImportError as e:
            logger.error(f"RGB augmentation requires additional packages: {e}")
            logger.error("Install with: pip install requests Pillow")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize RGB fetcher: {e}")
            return None
    
    def _init_infrared_fetcher(self):
        """Initialize infrared (NIR) fetcher."""
        try:
            from ...preprocessing.infrared_augmentation import IGNInfraredFetcher
            
            # Determine cache directory
            rgb_cache_dir = getattr(self.config.features, 'rgb_cache_dir', None)
            if rgb_cache_dir is None:
                infrared_cache_dir = Path(tempfile.gettempdir()) / "ign_lidar_cache" / "infrared"
            else:
                infrared_cache_dir = Path(rgb_cache_dir).parent / "infrared"
            
            infrared_cache_dir.mkdir(parents=True, exist_ok=True)
            
            fetcher = IGNInfraredFetcher(cache_dir=infrared_cache_dir)
            logger.info(
                f"NIR enabled (will use from input LAZ if present, "
                f"otherwise fetch from IGN IRC)"
            )
            return fetcher
            
        except ImportError as e:
            logger.error(f"Infrared augmentation requires additional packages: {e}")
            logger.error("Install with: pip install requests Pillow")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize infrared fetcher: {e}")
            return None
    
    def _validate_gpu(self) -> bool:
        """
        Validate GPU availability.
        
        Returns:
            bool: True if GPU is available and working
        """
        try:
            from ...features.features_gpu import GPU_AVAILABLE
            
            if not GPU_AVAILABLE:
                logger.warning(
                    "GPU requested but CuPy not available. Using CPU."
                )
                return False
            
            logger.info("GPU acceleration enabled")
            return True
            
        except ImportError:
            logger.warning("GPU module not available. Using CPU.")
            return False
        except Exception as e:
            logger.error(f"GPU validation failed: {e}")
            return False
    
    @property
    def has_rgb(self) -> bool:
        """Check if RGB fetcher is available."""
        return self.rgb_fetcher is not None
    
    @property
    def has_infrared(self) -> bool:
        """Check if infrared fetcher is available."""
        return self.infrared_fetcher is not None
    
    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self.use_gpu
