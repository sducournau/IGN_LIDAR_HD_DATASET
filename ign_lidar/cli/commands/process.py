"""Process command for IGN LiDAR HD CLI."""

import logging
import time
from pathlib import Path
from typing import Optional

import click
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ...core.processor import LiDARProcessor

logger = logging.getLogger(__name__)


def get_config_dir() -> str:
    """Get absolute path to configs directory."""
    package_dir = Path(__file__).parent.parent.parent
    config_dir = package_dir / "configs"
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found at: {config_dir}")
    
    return str(config_dir.absolute())


def setup_logging(cfg: DictConfig) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, cfg.get('log_level', 'INFO').upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(message)s'
    )


def print_config_summary(cfg: DictConfig) -> None:
    """Print a summary of the configuration."""
    logger.info("="*70)
    logger.info("IGN LiDAR HD v2.0 - Configuration Summary")
    logger.info("="*70)
    logger.info(f"Input:  {cfg.input_dir}")
    logger.info(f"Output: {cfg.output_dir}")
    logger.info(f"LOD Level: {cfg.processor.lod_level}")
    logger.info(f"GPU: {cfg.processor.use_gpu}")
    logger.info(f"Workers: {cfg.processor.num_workers}")
    logger.info(f"Patch size: {cfg.processor.patch_size}m")
    logger.info(f"Points per patch: {cfg.processor.num_points}")
    logger.info(f"Features mode: {cfg.features.mode}")
    logger.info(f"Preprocessing: {cfg.preprocess.enabled}")
    logger.info(f"Tile stitching: {cfg.stitching.enabled}")
    logger.info(f"Output format: {cfg.output.format}")
    logger.info("="*70)


def load_hydra_config(overrides: Optional[list] = None) -> DictConfig:
    """Load Hydra configuration with overrides."""
    config_dir = get_config_dir()
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
        return cfg


def create_default_config():
    """Create a default configuration when config files are not available."""
    return OmegaConf.create({
        "processor": {
            "lod_level": "LOD2",
            "use_gpu": False,
            "num_workers": 4,
            "patch_size": 150.0,
            "patch_overlap": 0.1,
            "num_points": 16384,
            "augment": False,
            "num_augmentations": 3,
            "batch_size": "auto",
            "prefetch_factor": 2,
            "pin_memory": False
        },
        "features": {
            "mode": "full",
            "k_neighbors": 20,
            "include_extra": True,
            "use_rgb": True,
            "use_infrared": False,
            "compute_ndvi": False,
            "sampling_method": "random",
            "normalize_xyz": False,
            "normalize_features": False,
            "gpu_batch_size": 1000000,
            "use_gpu_chunked": True
        },
        "preprocess": {
            "enabled": True,
            "sor_k": 12,
            "sor_std": 2.0,
            "ror_radius": 1.0,
            "ror_neighbors": 4,
            "voxel_enabled": False,
            "voxel_size": 0.1
        },
        "stitching": {
            "enabled": False,
            "buffer_size": 10.0,
            "auto_detect_neighbors": False,
            "cache_enabled": False
        },
        "output": {
            "format": "npz",
            "save_enriched_laz": False,
            "save_stats": True,
            "save_metadata": True,
            "compression": None
        },
        "input_dir": "???",
        "output_dir": "???",
        "num_workers": 4,
        "verbose": True,
        "log_level": "INFO",
        "bbox": {
            "xmin": None,
            "ymin": None,
            "xmax": None,
            "ymax": None
        }
    })


def process_lidar(cfg: DictConfig) -> None:
    """Process LiDAR tiles to create training patches."""
    # Setup logging
    setup_logging(cfg)
    
    # Print configuration summary
    if cfg.get('verbose', True):
        print_config_summary(cfg)
        logger.info("Full configuration:")
        logger.info(OmegaConf.to_yaml(cfg))
    
    # Convert paths
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    
    # Validate paths
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration to output directory
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Extract preprocessing config
    preprocess_config = None
    if cfg.preprocess.enabled:
        preprocess_config = OmegaConf.to_container(cfg.preprocess, resolve=True)
    
    # Initialize processor
    logger.info("Initializing LiDAR processor...")
    processor = LiDARProcessor(
        lod_level=cfg.processor.lod_level,
        augment=cfg.processor.augment,
        num_augmentations=cfg.processor.num_augmentations,
        bbox=cfg.bbox.to_tuple() if hasattr(cfg.bbox, 'to_tuple') else None,
        patch_size=cfg.processor.patch_size,
        patch_overlap=cfg.processor.patch_overlap,
        num_points=cfg.processor.num_points,
        include_extra_features=cfg.features.include_extra,
        k_neighbors=cfg.features.k_neighbors,
        include_rgb=cfg.features.use_rgb,
        use_gpu=cfg.processor.use_gpu,
        preprocess=cfg.preprocess.enabled,
        preprocess_config=preprocess_config,
        use_stitching=cfg.stitching.enabled,
        buffer_size=cfg.stitching.buffer_size,
    )
    
    # Process
    logger.info("Starting processing...")
    start_time = time.time()
    
    try:
        total_patches = processor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            num_workers=cfg.processor.num_workers,
            skip_existing=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Summary
        logger.info("="*70)
        logger.info("✅ Processing complete!")
        logger.info(f"  Total patches: {total_patches:,}")
        logger.info(f"  Processing time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Configuration: {config_save_path}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


@click.command()
@click.argument('overrides', nargs=-1)
def process_command(overrides):
    """
    Process LiDAR tiles to create training patches.
    
    OVERRIDES: Hydra configuration overrides in key=value format
    
    Examples:
        ign-lidar-hd process input_dir=data/raw output_dir=data/patches
        ign-lidar-hd process experiment=buildings_lod2 input_dir=data/raw output_dir=data/patches  
        ign-lidar-hd process processor.use_gpu=true input_dir=data/raw output_dir=data/patches
    """
    try:
        # Try to use Hydra configuration
        cfg = load_hydra_config(list(overrides))
        process_lidar(cfg)
    except Exception as e:
        # Fallback to direct parameters
        if not overrides:
            logger.error("No configuration provided. Use input_dir=path output_dir=path")
            raise click.ClickException("Configuration required")
        
        # Parse overrides into dict
        config_dict = {}
        for override in overrides:
            if '=' in override:
                key, value = override.split('=', 1)
                # Handle nested keys
                keys = key.split('.')
                current = config_dict
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
        
        # Create config from parsed overrides
        cfg = create_default_config()
        if 'input_dir' in config_dict:
            cfg.input_dir = config_dict['input_dir']
        if 'output_dir' in config_dict:
            cfg.output_dir = config_dict['output_dir']
        
        # Apply nested overrides
        for key, value in config_dict.items():
            if '.' not in key and key not in ['input_dir', 'output_dir']:
                continue
        
        process_lidar(cfg)