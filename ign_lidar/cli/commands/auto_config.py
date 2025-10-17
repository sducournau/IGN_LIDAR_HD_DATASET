"""Auto-configuration command for IGN LiDAR HD CLI."""

import logging
from pathlib import Path
from typing import Optional

import click
from omegaconf import OmegaConf

from ...core.auto_configuration import generate_auto_config, save_auto_config, print_auto_config_summary

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('output_dir', type=click.Path(file_okay=False, path_type=Path))
@click.option(
    '--output-config', '-o',
    type=click.Path(path_type=Path),
    help='Path to save generated configuration (default: output_dir/auto_config.yaml)'
)
@click.option(
    '--base-config', '-b',
    type=click.Path(exists=True, path_type=Path),
    help='Base configuration file to start from'
)
@click.option(
    '--force-gpu/--force-cpu',
    default=None,
    help='Force GPU or CPU processing'
)
@click.option(
    '--processing-mode',
    type=click.Choice(['patches_only', 'both', 'enriched_only']),
    help='Override processing mode'
)
@click.option(
    '--enable-rgb/--disable-rgb',
    default=None,
    help='Enable or disable RGB features'
)
@click.option(
    '--num-workers',
    type=int,
    help='Override number of worker processes'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show recommendations without saving configuration'
)
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def auto_config_command(input_dir: Path,
                       output_dir: Path,
                       output_config: Optional[Path],
                       base_config: Optional[Path],
                       force_gpu: Optional[bool],
                       processing_mode: Optional[str],
                       enable_rgb: Optional[bool],
                       num_workers: Optional[int],
                       dry_run: bool,
                       verbose: bool):
    """
    Generate optimized configuration based on system capabilities and data analysis.
    
    INPUT_DIR: Directory containing LiDAR files (.laz/.las)
    OUTPUT_DIR: Directory where processed data will be saved
    
    This command analyzes your system capabilities, input data characteristics,
    and generates an optimized configuration for maximum performance.
    
    Examples:
        ign-lidar-hd auto-config data/raw data/output
        ign-lidar-hd auto-config data/raw data/output --force-gpu --enable-rgb
        ign-lidar-hd auto-config data/raw data/output --base-config my_config.yaml
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')
    
    logger.info("🧠 IGN LiDAR HD Auto-Configuration")
    logger.info("="*50)
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        raise click.ClickException(f"Input directory not found: {input_dir}")
    
    # Check for LiDAR files
    laz_files = list(input_dir.glob("**/*.laz"))
    las_files = list(input_dir.glob("**/*.las"))
    
    if not laz_files and not las_files:
        logger.warning("No LiDAR files (.laz/.las) found in input directory")
        logger.warning("Auto-configuration will use default assumptions")
    else:
        logger.info(f"Found {len(laz_files + las_files)} LiDAR files for analysis")
    
    # Load base configuration if provided
    base_config_dict = None
    if base_config:
        try:
            base_config_dict = OmegaConf.to_object(OmegaConf.load(base_config))
            logger.info(f"Loaded base configuration from: {base_config}")
        except Exception as e:
            logger.error(f"Failed to load base configuration: {e}")
            raise click.ClickException(f"Invalid base configuration file: {base_config}")
    
    # Build user preferences
    user_preferences = {}
    
    if force_gpu is not None:
        user_preferences['force_gpu'] = force_gpu
        user_preferences['force_cpu'] = not force_gpu
        logger.info(f"User preference: {'GPU' if force_gpu else 'CPU'} processing")
    
    if processing_mode:
        user_preferences['processing_mode'] = processing_mode
        logger.info(f"User preference: {processing_mode} processing mode")
    
    if enable_rgb is not None:
        user_preferences['enable_rgb'] = enable_rgb
        logger.info(f"User preference: RGB features {'enabled' if enable_rgb else 'disabled'}")
    
    if num_workers:
        user_preferences['num_workers'] = num_workers
        logger.info(f"User preference: {num_workers} worker processes")
    
    # Generate auto-configuration
    logger.info("\n🔍 Analyzing system and data...")
    
    try:
        optimized_config, recommendation = generate_auto_config(
            input_dir=input_dir,
            output_dir=output_dir,
            base_config=base_config_dict,
            user_preferences=user_preferences if user_preferences else None
        )
        
        # Print recommendations
        print_auto_config_summary(recommendation)
        
        if dry_run:
            logger.info("\n🚫 Dry run mode - configuration not saved")
            logger.info("Use without --dry-run to save the configuration")
            return
        
        # Determine output path
        if output_config is None:
            output_config = output_dir / "auto_config.yaml"
        
        # Create output directory if needed
        output_config.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        save_auto_config(optimized_config, output_config, recommendation)
        
        # Print next steps
        print("\n" + "="*70)
        print("✅ AUTO-CONFIGURATION COMPLETE")
        print("="*70)
        print(f"📄 Configuration saved to: {output_config}")
        print(f"🎯 Confidence level: {recommendation.confidence_score:.1%}")
        print(f"⚡ Expected performance: {recommendation.estimated_performance}")
        
        print("\n🚀 Next steps:")
        print(f"   ign-lidar-hd process --config-file {output_config}")
        
        if recommendation.warnings:
            print("\n⚠️  Please review these warnings before processing:")
            for warning in recommendation.warnings:
                print(f"   • {warning}")
                
        if recommendation.alternative_configs:
            print("\n💡 Consider these alternatives for specific use cases:")
            for alt in recommendation.alternative_configs:
                print(f"   • {alt['name']}: {alt['description']}")
        
        print("="*70)
        
    except Exception as e:
        logger.error(f"Auto-configuration failed: {e}")
        if verbose:
            logger.exception("Detailed error:")
        raise click.ClickException(f"Auto-configuration failed: {e}")


# Export the command
auto_config = auto_config_command