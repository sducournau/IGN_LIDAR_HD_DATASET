"""Process command for IGN LiDAR HD CLI."""

import logging
import time
from pathlib import Path
from typing import Optional

import click
from omegaconf import DictConfig, OmegaConf

from ...core.processor import LiDARProcessor
from ..hydra_runner import HydraRunner

logger = logging.getLogger(__name__)


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
    # Initialize HydraRunner
    runner = HydraRunner()
    
    # Load configuration using HydraRunner
    try:
        cfg = runner.load_config(
            config_name="config",
            overrides=list(overrides),
            config_file=config_file
        )
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise click.ClickException(str(e))
    
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