"""
Hydra-based CLI for IGN LiDAR HD v2.0.

Modern command-line interface using Hydra for configuration management.
Provides composable, type-safe configuration with powerful override system.

Usage:
    # Basic processing
    python -m ign_lidar.cli.hydra_main input_dir=data/raw output_dir=data/patches
    
    # GPU processing
    python -m ign_lidar.cli.hydra_main processor=gpu input_dir=data/raw output_dir=data/patches
    
    # Experiment preset
    python -m ign_lidar.cli.hydra_main experiment=pointnet_training \\
        input_dir=data/raw output_dir=data/patches
    
    # Override specific parameters
    python -m ign_lidar.cli.hydra_main processor.num_points=32768 \\
        features.k_neighbors=30 input_dir=data/raw output_dir=data/patches
    
    # Multi-run (parameter sweep)
    python -m ign_lidar.cli.hydra_main -m processor.num_points=4096,8192,16384 \\
        input_dir=data/raw output_dir=data/patches
"""

import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from ..core.processor import LiDARProcessor
from ..config.schema import IGNLiDARConfig

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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





def process_lidar(cfg: DictConfig) -> None:
    """
    Process LiDAR tiles to create training patches.
    
    This is the main entry point for Hydra-based processing.
    Configuration is composed from YAML files and command-line overrides.
    
    Args:
        cfg: Hydra configuration (automatically loaded and composed)
    """
    # Setup logging
    setup_logging(cfg)
    
    # Validate configuration
    try:
        # Convert to structured config for validation
        structured_cfg = OmegaConf.to_object(cfg)
        if hasattr(structured_cfg, 'validate'):
            structured_cfg.validate()
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    
    # Print configuration summary
    if cfg.verbose:
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
    
    # Setup cache directory for RGB/Infrared orthophotos in user temp folder
    # This ensures cross-platform compatibility and automatic cleanup
    temp_dir = Path(tempfile.gettempdir())
    rgb_cache_dir = temp_dir / "ign_lidar_cache" / "orthophotos"
    rgb_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"RGB/Infrared cache directory: {rgb_cache_dir}")
    
    # Initialize processor
    logger.info("Initializing LiDAR processor...")
    processor = LiDARProcessor(
        lod_level=cfg.processor.lod_level,
        processing_mode=cfg.output.processing_mode,
        augment=cfg.processor.augment,
        num_augmentations=cfg.processor.num_augmentations,
        bbox=cfg.bbox.to_tuple() if hasattr(cfg.bbox, 'to_tuple') else None,
        patch_size=cfg.processor.patch_size,
        patch_overlap=cfg.processor.patch_overlap,
        num_points=cfg.processor.num_points,
        include_extra_features=cfg.features.include_extra,
        k_neighbors=cfg.features.k_neighbors,
        include_rgb=cfg.features.use_rgb,
        rgb_cache_dir=rgb_cache_dir,  # Add cache directory
        include_infrared=cfg.features.use_infrared,
        compute_ndvi=cfg.features.compute_ndvi,
        use_gpu=cfg.processor.use_gpu,
        use_gpu_chunked=cfg.features.use_gpu_chunked,
        gpu_batch_size=cfg.features.gpu_batch_size,
        preprocess=cfg.preprocess.enabled,
        preprocess_config=preprocess_config,
        use_stitching=cfg.stitching.enabled,
        buffer_size=cfg.stitching.buffer_size,
        architecture=cfg.processor.architecture,
        output_format=cfg.output.format,
    )
    
    # Process
    logger.info("Starting processing...")
    start_time = time.time()
    
    try:
        total_patches = processor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            num_workers=cfg.processor.num_workers,
            skip_existing=cfg.output.skip_existing
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


def verify(cfg: DictConfig) -> None:
    """
    Verify dataset quality and features.
    
    Args:
        cfg: Hydra configuration
    """
    setup_logging(cfg)
    
    from ..core.verification import FeatureVerifier
    
    output_dir = Path(cfg.output_dir)
    
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return
    
    logger.info("Verifying dataset...")
    verifier = FeatureVerifier(output_dir)
    
    # Run verification
    results = verifier.verify_all()
    
    # Print results
    logger.info("="*70)
    logger.info("Verification Results")
    logger.info("="*70)
    logger.info(f"Total patches verified: {results['num_patches']}")
    logger.info(f"Valid patches: {results['valid_patches']}")
    logger.info(f"Invalid patches: {results['invalid_patches']}")
    
    if results['errors']:
        logger.warning("Errors found:")
        for error in results['errors']:
            logger.warning(f"  - {error}")
    else:
        logger.info("✅ All patches are valid!")
    
    logger.info("="*70)


def info(cfg: DictConfig) -> None:
    """
    Display configuration information and presets.
    
    Args:
        cfg: Hydra configuration
    """
    setup_logging(cfg)
    
    # List experiment YAML files instead of Python presets
    from pathlib import Path
    
    logger.info("="*70)
    logger.info("IGN LiDAR HD v2.0 - Configuration Info")
    logger.info("="*70)
    
    logger.info("\nAvailable Experiment Presets:")
    logger.info("-" * 70)
    
    config_dir = Path(__file__).parent.parent / "configs" / "experiment"
    if config_dir.exists():
        yaml_files = sorted(config_dir.glob("*.yaml"))
        for yaml_file in yaml_files:
            preset_name = yaml_file.stem
            # Try to read description from YAML comment
            try:
                with open(yaml_file, 'r') as f:
                    for line in f:
                        if line.strip().startswith('# Optimized') or line.strip().startswith('# State-of'):
                            description = line.strip()[2:].strip()
                            break
                        elif line.strip() and not line.strip().startswith('#'):
                            description = "Experiment configuration"
                            break
                    else:
                        description = "Experiment configuration"
            except Exception:
                description = "Experiment configuration"
            
            logger.info(f"  {preset_name:30s} - {description}")
    else:
        logger.warning("Preset directory not found")
    
    logger.info("\n" + "="*70)
    logger.info("Current Configuration:")
    logger.info("="*70)
    logger.info(OmegaConf.to_yaml(cfg))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    process_lidar(cfg)


if __name__ == "__main__":
    main()
