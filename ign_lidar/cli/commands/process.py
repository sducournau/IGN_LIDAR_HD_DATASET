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
    
    # Get processing mode
    processing_mode = OmegaConf.select(cfg, "output.processing_mode", default="patches_only")
    logger.info(f"Processing mode: {processing_mode}")
    
    if processing_mode != "enriched_only":
        logger.info(f"Patch size: {cfg.processor.patch_size}m")
        logger.info(f"Points per patch: {cfg.processor.num_points}")
        logger.info(f"Output format: {cfg.output.format}")
    
    logger.info(f"Features mode: {cfg.features.mode}")
    logger.info(f"Preprocessing: {cfg.preprocess.enabled}")
    logger.info(f"Tile stitching: {cfg.stitching.enabled}")
    
    if cfg.stitching.enabled and hasattr(cfg.stitching, 'auto_download_neighbors'):
        logger.info(f"Auto-download neighbors: {cfg.stitching.auto_download_neighbors}")
    
    logger.info("="*70)


def load_hydra_config(overrides: Optional[list] = None) -> DictConfig:
    """Load Hydra configuration with overrides."""
    config_dir = get_config_dir()
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
        
        # Handle output shorthand immediately after loading
        # This ensures cfg.output is always a dict, never a string
        if hasattr(cfg, 'output') and isinstance(cfg.output, str):
            output_mode = cfg.output
            cfg.output = OmegaConf.create({
                "format": "npz",
                "processing_mode": output_mode,
                "save_stats": True,
                "save_metadata": output_mode != 'enriched_only',
                "compression": None
            })
        
        return cfg


def load_config_from_file(
    config_file: str,
    overrides: Optional[list] = None
) -> DictConfig:
    """
    Load Hydra configuration from a custom YAML file.
    
    This allows users to provide their own config files instead of
    using only the built-in presets.
    
    Args:
        config_file: Path to custom YAML config file (absolute or relative)
        overrides: List of CLI overrides to apply on top
        
    Returns:
        Composed Hydra configuration
        
    Example:
        >>> cfg = load_config_from_file(
        ...     'my_config.yaml',
        ...     ['processor.use_gpu=true']
        ... )
    """
    import yaml
    
    config_path = Path(config_file)
    
    # Resolve to absolute path
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading custom config from: {config_path}")
    
    # Load the custom config
    with open(config_path, 'r') as f:
        custom_config = yaml.safe_load(f)
    
    # Get the package config directory for defaults
    package_config_dir = get_config_dir()
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with package config directory
    with initialize_config_dir(config_dir=package_config_dir, version_base=None):
        # Start with base config
        cfg = compose(config_name="config", overrides=[])
        
        # Set struct mode to False to allow new keys
        OmegaConf.set_struct(cfg, False)
        
        # Merge custom config on top
        custom_omega = OmegaConf.create(custom_config)
        cfg = OmegaConf.merge(cfg, custom_omega)
        
        # Apply CLI overrides (highest priority)
        if overrides:
            override_omega = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_omega)
        
        # Re-enable struct mode
        OmegaConf.set_struct(cfg, True)
        
        # Handle output shorthand
        if hasattr(cfg, 'output') and isinstance(cfg.output, str):
            output_mode = cfg.output
            cfg.output = OmegaConf.create({
                "format": "npz",
                "processing_mode": output_mode,
                "save_stats": True,
                "save_metadata": output_mode != 'enriched_only',
                "compression": None
            })
        
        logger.info("✅ Custom config loaded successfully")
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
            "processing_mode": "patches_only",
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
    # CRITICAL: Handle 'output' shorthand parameter FIRST
    # This must happen before any code tries to access cfg.output properties
    # Maps: output=enriched_only -> output.processing_mode='enriched_only'
    #       output=both -> output.processing_mode='both'
    #       output=patches -> output.processing_mode='patches_only' (default)
    if hasattr(cfg, 'output') and isinstance(cfg.output, str):
        output_mode = cfg.output
        # Replace string with proper OutputConfig
        cfg.output = OmegaConf.create({
            "format": "npz",
            "processing_mode": output_mode,
            "save_stats": True,
            "save_metadata": output_mode != 'enriched_only',  # No metadata for enriched_only mode
            "compression": None
        })
    
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
    
    # Extract stitching config
    stitching_config = None
    if cfg.stitching.enabled:
        stitching_config = OmegaConf.to_container(cfg.stitching, resolve=True)
    
    # Initialize processor
    logger.info("Initializing LiDAR processor...")
    
    # Get processing mode
    processing_mode = OmegaConf.select(cfg, "output.processing_mode", default="patches_only")
    
    processor = LiDARProcessor(
        lod_level=cfg.processor.lod_level,
        processing_mode=processing_mode,
        augment=cfg.processor.augment,
        num_augmentations=cfg.processor.num_augmentations,
        bbox=cfg.bbox.to_tuple() if hasattr(cfg.bbox, 'to_tuple') else None,
        patch_size=cfg.processor.patch_size,
        patch_overlap=cfg.processor.patch_overlap,
        num_points=cfg.processor.num_points,
        include_extra_features=cfg.features.include_extra,
        k_neighbors=cfg.features.k_neighbors,
        include_rgb=cfg.features.use_rgb,
        include_infrared=cfg.features.use_infrared,
        compute_ndvi=cfg.features.compute_ndvi,
        use_gpu=cfg.processor.use_gpu,
        preprocess=cfg.preprocess.enabled,
        preprocess_config=preprocess_config,
        use_stitching=cfg.stitching.enabled,
        buffer_size=cfg.stitching.buffer_size,
        stitching_config=stitching_config,
        architecture=OmegaConf.select(cfg, "processor.architecture", default="pointnet++"),
        output_format=OmegaConf.select(cfg, "output.format", default="npz"),
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
        if OmegaConf.select(cfg, "output.only_enriched_laz", default=False):
            logger.info(f"  Mode: Enriched LAZ only (no patches)")
            logger.info(f"  Enriched LAZ files: {output_dir / 'enriched'}")
        else:
            logger.info(f"  Total patches: {total_patches:,}")
        logger.info(f"  Processing time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Configuration: {config_save_path}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


@click.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True),
    help='Path to custom YAML config file (optional)'
)
@click.option(
    '--show-config',
    is_flag=True,
    help='Show the composed configuration and exit (no processing)'
)
@click.argument('overrides', nargs=-1)
def process_command(config_file, show_config, overrides):
    """
    Process LiDAR tiles to create training patches.
    
    OVERRIDES: Hydra configuration overrides in key=value format
    
    Examples:
        # Use built-in config with overrides
        ign-lidar-hd process input_dir=data/raw output_dir=data/patches
        
        # Use experiment preset
        ign-lidar-hd process experiment=buildings_lod2 input_dir=data/raw output_dir=data/patches
        
        # Load custom config file
        ign-lidar-hd process --config-file my_config.yaml
        
        # Custom config + overrides (overrides have highest priority)
        ign-lidar-hd process -c my_config.yaml processor.use_gpu=true
        
        # Preview config without processing
        ign-lidar-hd process -c my_config.yaml --show-config
    """
    # Load configuration
    if config_file:
        cfg = load_config_from_file(config_file, list(overrides))
    else:
        cfg = load_hydra_config(list(overrides))
    
    # Show config and exit if requested
    if show_config:
        logger.info("="*70)
        logger.info("Configuration Preview")
        logger.info("="*70)
        print(OmegaConf.to_yaml(cfg))
        logger.info("="*70)
        logger.info("(Configuration preview only - no processing performed)")
        return
    
    # Process with the loaded config
    try:
        process_lidar(cfg)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise click.ClickException(str(e))