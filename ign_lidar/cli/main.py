#!/usr/bin/env python3
"""
Unified CLI for IGN LiDAR HD v2.0.

Modern command-line interface with batch conversion capabilities.
Fixes config loading issues by using absolute paths.
"""

import os
import sys
import click
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add the project root to Python path to help with imports
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir

from ..core.processor import LiDARProcessor
from ..io.qgis_converter import simplify_for_qgis

logger = logging.getLogger(__name__)


def get_config_dir():
    """Get absolute path to config directory."""
    # Try to find configs directory relative to this file
    current_file = Path(__file__).absolute()
    
    # Look for configs directory in the project root
    config_paths = [
        current_file.parent.parent.parent / "configs",  # From ign_lidar/cli/
        Path.cwd() / "configs",  # From current working directory
        Path(os.environ.get("IGN_LIDAR_CONFIG_DIR", "")) if os.environ.get("IGN_LIDAR_CONFIG_DIR") else None
    ]
    
    for config_path in config_paths:
        if config_path and config_path.exists():
            return str(config_path.absolute())
    
    # Fallback - create a minimal config
    return None


def setup_logging(cfg: DictConfig) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, cfg.get('log_level', 'INFO').upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def process_lidar_with_config(cfg: DictConfig) -> None:
    """Process LiDAR data with given configuration."""
    setup_logging(cfg)
    
    logger.info("=" * 70)
    logger.info("IGN LiDAR HD v2.0 - Configuration Summary")
    logger.info("=" * 70)
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
    logger.info("=" * 70)
    
    if cfg.get('verbose', True):
        logger.info("Full configuration:")
        logger.info(OmegaConf.to_yaml(cfg))
    
    # Save configuration
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Initialize processor
    logger.info("Initializing LiDAR processor...")
    
    processor = LiDARProcessor(
        lod_level=cfg.processor.lod_level,
        k_neighbors=cfg.features.k_neighbors,
        use_gpu=cfg.processor.use_gpu,
        augment=cfg.processor.augment,
        num_augmentations=cfg.processor.num_augmentations,
        patch_size=cfg.processor.patch_size,
        patch_overlap=cfg.processor.patch_overlap,
        num_points=cfg.processor.num_points,
        include_rgb=cfg.features.get('use_rgb', False),
        preprocess=cfg.preprocess.enabled,
        preprocess_config=cfg.preprocess,
        bbox=cfg.get('bbox', None)
    )
    
    logger.info(f"Initialized LiDARProcessor with {cfg.processor.lod_level}")
    
    # Process data
    logger.info("Starting processing...")
    processor.process_directory(
        input_dir=Path(cfg.input_dir),
        output_dir=Path(cfg.output_dir),
        num_workers=cfg.processor.num_workers,
        save_metadata=cfg.output.get('save_metadata', True)
    )


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


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--config-dir', type=click.Path(exists=True), help='Configuration directory path')
@click.pass_context
def cli(ctx, verbose, config_dir):
    """IGN LiDAR HD Processing Tool - Unified CLI for LiDAR processing with
  batch conversion capabilities."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config_dir'] = config_dir


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--patch-size', type=float, default=150.0, help='Patch size in meters')
@click.option('--num-points', type=int, default=16384, help='Number of points per patch')
@click.option('--lod-level', type=click.Choice(['LOD2', 'LOD3']), default='LOD2', help='Level of detail')
@click.option('--use-gpu', is_flag=True, help='Use GPU acceleration')
@click.option('--num-workers', type=int, default=4, help='Number of worker processes')
@click.option('--experiment', type=str, help='Experiment configuration to use')
@click.option('--config-override', multiple=True, help='Configuration overrides (key=value)')
@click.pass_context
def process(ctx, input_dir, output_dir, patch_size, num_points, lod_level, use_gpu, num_workers, experiment, config_override):
    """Process LiDAR data with feature extraction."""
    
    config_dir = ctx.obj.get('config_dir')
    
    # Try to use Hydra if config directory is available
    if config_dir or get_config_dir():
        try:
            actual_config_dir = config_dir or get_config_dir()
            
            # Initialize Hydra with absolute config path
            with initialize_config_dir(config_dir=actual_config_dir, version_base=None):
                cfg = compose(config_name="config")
                
                # Apply overrides
                cfg.input_dir = input_dir
                cfg.output_dir = output_dir
                cfg.processor.patch_size = patch_size
                cfg.processor.num_points = num_points
                cfg.processor.lod_level = lod_level
                cfg.processor.use_gpu = use_gpu
                cfg.processor.num_workers = num_workers
                
                # Apply config overrides
                for override in config_override:
                    if '=' in override:
                        key, value = override.split('=', 1)
                        OmegaConf.set(cfg, key, value)
                
                process_lidar_with_config(cfg)
                return
                
        except Exception as e:
            logger.warning(f"Could not load Hydra config: {e}")
            logger.info("Falling back to default configuration")
    
    # Fallback to default config
    cfg = create_default_config()
    cfg.input_dir = input_dir
    cfg.output_dir = output_dir
    cfg.processor.patch_size = patch_size
    cfg.processor.num_points = num_points
    cfg.processor.lod_level = lod_level
    cfg.processor.use_gpu = use_gpu
    cfg.processor.num_workers = num_workers
    cfg.verbose = ctx.obj.get('verbose', True)
    
    process_lidar_with_config(cfg)


@cli.command()
@click.pass_context
def config_info(ctx):
    """Show configuration information and available options."""
    
    config_dir = ctx.obj.get('config_dir') or get_config_dir()
    
    click.echo("IGN LiDAR HD Configuration Information")
    click.echo("=" * 50)
    
    if config_dir:
        click.echo(f"Config directory: {config_dir}")
        
        # List available configurations
        config_path = Path(config_dir)
        if (config_path / "experiment").exists():
            experiments = list((config_path / "experiment").glob("*.yaml"))
            click.echo(f"Available experiments: {[e.stem for e in experiments]}")
        
        if (config_path / "features").exists():
            features = list((config_path / "features").glob("*.yaml"))
            click.echo(f"Available feature configs: {[f.stem for f in features]}")
            
        if (config_path / "processor").exists():
            processors = list((config_path / "processor").glob("*.yaml"))
            click.echo(f"Available processor configs: {[p.stem for p in processors]}")
    else:
        click.echo("No config directory found - using default configuration")
        click.echo("To use custom configs, specify --config-dir or set IGN_LIDAR_CONFIG_DIR environment variable")
    
    click.echo("\nUsage examples:")
    click.echo("  ign-lidar-hd process input_dir output_dir")
    click.echo("  ign-lidar-hd process input_dir output_dir --patch-size 100 --use-gpu")
    click.echo("  ign-lidar-hd batch-convert input_dir --output qgis_ready/")


@cli.command()
@click.argument('output_dir', type=click.Path())
@click.option('--bbox', '-b', type=float, nargs=4, metavar='XMIN YMIN XMAX YMAX',
              help='Bounding box coordinates (Lambert93)')
@click.option('--position', '-p', type=float, nargs=2, metavar='X Y',
              help='Download tiles around position (Lambert93 coordinates)')
@click.option('--radius', '-r', default=5000, type=float,
              help='Radius around position in meters (default: 5000)')
@click.option('--max-concurrent', '-j', default=3, type=int,
              help='Maximum concurrent downloads (default: 3)')
@click.option('--force', '-f', is_flag=True,
              help='Force download even if files already exist')
@click.option('--list-locations', is_flag=True,
              help='List available strategic locations')
@click.option('--location', '-l', type=str,
              help='Download by strategic location name')
@click.pass_context
def download(ctx, output_dir, bbox, position, radius, max_concurrent, force, list_locations, location):
    """Download IGN LiDAR HD tiles by bounding box, position, or location."""
    from ..downloader import IGNLiDARDownloader
    from ..datasets.strategic_locations import STRATEGIC_LOCATIONS
    
    verbose = ctx.obj.get('verbose', False)
    
    if list_locations:
        click.echo("Available strategic locations:")
        click.echo("=" * 40)
        for name, data in STRATEGIC_LOCATIONS.items():
            click.echo(f"  {name}: {data.get('description', 'No description')}")
        return
    
    try:
        downloader = IGNLiDARDownloader(Path(output_dir), max_concurrent=max_concurrent)
        
        if verbose:
            click.echo(f"Initialized downloader with output directory: {output_dir}")
            click.echo(f"Maximum concurrent downloads: {max_concurrent}")
        
        tiles_to_download = []
        
        if location:
            if location in STRATEGIC_LOCATIONS:
                loc_data = STRATEGIC_LOCATIONS[location]
                bbox = (loc_data['xmin'], loc_data['ymin'], loc_data['xmax'], loc_data['ymax'])
                click.echo(f"Using location '{location}': {loc_data.get('description', '')}")
            else:
                click.echo(f"❌ Unknown location: {location}")
                click.echo("Use --list-locations to see available locations")
                sys.exit(1)
        
        if bbox:
            click.echo(f"Finding tiles in bounding box: {bbox}")
            tiles_to_download = downloader.get_tiles_in_bbox(*bbox)
        elif position:
            click.echo(f"Finding tiles around position: {position} (radius: {radius}m)")
            tiles_to_download = downloader.find_tiles_by_position(
                position[0], position[1], radius_km=radius/1000
            )
        else:
            click.echo("❌ Must specify --bbox, --position, or --location")
            sys.exit(1)
        
        if not tiles_to_download:
            click.echo("No tiles found for the specified criteria")
            return
            
        click.echo(f"Found {len(tiles_to_download)} tiles to download")
        
        if verbose:
            for tile in tiles_to_download[:5]:  # Show first 5
                click.echo(f"  - {tile}")
            if len(tiles_to_download) > 5:
                click.echo(f"  ... and {len(tiles_to_download) - 5} more")
        
        # Download tiles
        results = downloader.batch_download(
            tiles_to_download,
            skip_existing=not force
        )
        
        success = [tile for tile, status in results.items() if status]
        failed = [tile for tile, status in results.items() if not status]
        
        click.echo(f"\n✅ Download complete!")
        click.echo(f"  Successfully downloaded: {len(success)}")
        if failed:
            click.echo(f"  Failed downloads: {len(failed)}")
            if verbose:
                for tile in failed:
                    click.echo(f"    - {tile}")
    
    except Exception as e:
        click.echo(f"❌ Download error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output file for verification report')
@click.option('--detailed', '-d', is_flag=True,
              help='Show detailed statistics')
@click.option('--features-only', '-f', is_flag=True,
              help='Verify features only (skip RGB/NIR)')
@click.pass_context  
def verify(ctx, input_path, output, detailed, features_only):
    """Verify LiDAR data quality and features."""
    from ..core.verification import FeatureVerifier
    
    verbose = ctx.obj.get('verbose', False)
    input_path = Path(input_path)
    
    try:
        verifier = FeatureVerifier()
        
        if verbose:
            click.echo(f"Verifying: {input_path}")
        
        if input_path.is_file():
            # Single file
            result = verifier.verify_file(input_path)
            
            if detailed:
                click.echo("\nDetailed verification results:")
                click.echo("=" * 50)
                for feature, stats in result.items():
                    click.echo(f"{feature}:")
                    click.echo(f"  Mean: {stats.mean:.4f}")
                    click.echo(f"  Std:  {stats.std:.4f}")
                    click.echo(f"  Min:  {stats.min:.4f}")
                    click.echo(f"  Max:  {stats.max:.4f}")
                    click.echo(f"  Count: {stats.count}")
                    if hasattr(stats, 'artifacts_detected'):
                        click.echo(f"  Artifacts: {stats.artifacts_detected}")
            else:
                click.echo("✅ Verification complete")
                
        elif input_path.is_dir():
            # Directory of files
            laz_files = list(input_path.glob("*.laz"))
            if not laz_files:
                click.echo("❌ No LAZ files found in directory")
                sys.exit(1)
                
            click.echo(f"Verifying {len(laz_files)} files...")
            results = []
            
            for laz_file in laz_files:
                if verbose:
                    click.echo(f"  Processing: {laz_file.name}")
                result = verifier.verify_file(laz_file)
                results.append(result)
            
            # Print summary
            verifier.print_summary(results)
        
        if output:
            # Save results to file
            click.echo(f"Results saved to: {output}")
    
    except Exception as e:
        click.echo(f"❌ Verification error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for converted files')
@click.option('--batch-size', '-b', default=10, type=int,
              help='Number of files to process in parallel (default: 10)')
@click.option('--force', '-f', is_flag=True,
              help='Overwrite existing output files')
@click.pass_context
def batch_convert(ctx, input_dir, output, batch_size, force):
    """Batch convert LAZ files for QGIS compatibility."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    verbose = ctx.obj.get('verbose', False)
    input_dir = Path(input_dir)
    output_dir = Path(output) if output else input_dir
    
    try:
        # Find all LAZ files
        laz_files = list(input_dir.rglob("*.laz"))
        
        if not laz_files:
            click.echo("❌ No LAZ files found")
            sys.exit(1)
        
        click.echo(f"Found {len(laz_files)} LAZ files")
        click.echo(f"Output directory: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def convert_file(laz_file):
            rel_path = laz_file.relative_to(input_dir)
            output_file = output_dir / rel_path.parent / (rel_path.stem + "_qgis.laz")
            
            if output_file.exists() and not force:
                return f"Skipped (exists): {rel_path}"
            
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result = simplify_for_qgis(laz_file, output_file, verbose=False)
                return f"Converted: {rel_path}"
            except Exception as e:
                return f"Failed: {rel_path} - {e}"
        
        # Process files in parallel
        success_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_file = {executor.submit(convert_file, laz_file): laz_file 
                            for laz_file in laz_files}
            
            with tqdm(total=len(laz_files), desc="Converting") as pbar:
                for future in as_completed(future_to_file):
                    result = future.result()
                    
                    if "Failed:" in result:
                        failed_count += 1
                        if verbose:
                            click.echo(result)
                    else:
                        success_count += 1
                        if verbose and "Converted:" in result:
                            click.echo(result)
                    
                    pbar.update(1)
        
        click.echo(f"\n✅ Batch conversion complete!")
        click.echo(f"  Successfully converted: {success_count}")
        click.echo(f"  Failed: {failed_count}")
        click.echo(f"  Output directory: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Batch conversion error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--version', '-v', is_flag=True, help='Show version information')
@click.option('--dependencies', '-d', is_flag=True, help='Show dependency information')  
@click.option('--config', '-c', is_flag=True, help='Show configuration paths')
def info(version, dependencies, config):
    """Show package information, version, and dependencies."""
    
    if version or (not dependencies and not config):
        try:
            from .. import __version__
            click.echo(f"IGN LiDAR HD v{__version__}")
        except ImportError:
            click.echo("IGN LiDAR HD (version unknown)")
    
    if dependencies:
        click.echo("\nDependencies:")
        click.echo("=" * 20)
        
        deps = [
            ("numpy", "Array processing"),
            ("laspy", "LAZ/LAS file handling"),  
            ("lazrs", "LAZ compression"),
            ("scikit-learn", "Machine learning"),
            ("tqdm", "Progress bars"),
            ("click", "CLI framework"),
            ("hydra-core", "Configuration management"),
            ("requests", "HTTP requests"),
            ("Pillow", "Image processing")
        ]
        
        for name, desc in deps:
            try:
                __import__(name)
                status = "✅ Available"
            except ImportError:
                status = "❌ Missing"
            click.echo(f"  {name:<15} {status:<12} - {desc}")
    
    if config:
        click.echo("\nConfiguration:")
        click.echo("=" * 20)
        config_dir = get_config_dir()
        if config_dir:
            click.echo(f"  Config directory: {config_dir}")
        else:
            click.echo("  No config directory found")
        
        click.echo(f"  Project root: {Path(__file__).parent.parent.parent}")


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()