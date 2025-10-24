"""
Validate configuration files command.

This command validates configuration files against the schema,
catching errors before processing starts.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import click

from ign_lidar.config.validator import validate_config_file, ConfigSchemaValidator

logger = logging.getLogger(__name__)


@click.command('validate-config')
@click.argument('config_file', type=click.Path(exists=True))
@click.option(
    '--strict/--no-strict',
    default=False,
    help='Exit with error on validation warnings (default: warnings only)'
)
@click.option(
    '--allow-partial',
    is_flag=True,
    help='Allow partial configs (for Hydra composition)'
)
@click.option(
    '--show-suggestions',
    is_flag=True,
    help='Show suggestions for fixing errors'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Verbose output'
)
def validate_config_command(
    config_file: str,
    strict: bool,
    allow_partial: bool,
    show_suggestions: bool,
    verbose: bool
):
    """
    Validate a configuration file.
    
    This command checks if a configuration file is valid according to
    the schema, catching errors before processing starts.
    
    Examples:
    
        # Validate a config file
        ign-lidar-hd validate-config my_config.yaml
        
        # Strict validation (fail on warnings)
        ign-lidar-hd validate-config --strict my_config.yaml
        
        # Allow partial configs (for Hydra composition)
        ign-lidar-hd validate-config --allow-partial presets/my_preset.yaml
        
        # Show suggestions for fixing errors
        ign-lidar-hd validate-config --show-suggestions my_config.yaml
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(message)s'
    )
    
    config_path = Path(config_file)
    
    click.echo(f"Validating: {config_path}")
    click.echo("-" * 80)
    
    try:
        # Validate the config
        is_valid = validate_config_file(
            config_path,
            strict=False,  # Don't raise, we'll handle it
            allow_partial=allow_partial
        )
        
        if is_valid:
            click.secho("✓ Configuration is valid!", fg='green', bold=True)
            sys.exit(0)
        else:
            click.secho("✗ Configuration has errors", fg='red', bold=True)
            
            # Show suggestions if requested
            if show_suggestions:
                click.echo("\n" + "=" * 80)
                click.echo("Common fixes:")
                click.echo("-" * 80)
                click.echo("""
1. Missing 'preprocess' section:
   Add to your config:
   preprocess:
     enabled: false
     remove_duplicates: true

2. Missing 'stitching' section:
   Add to your config:
   stitching:
     enabled: false
     buffer_size: 10.0

3. Missing required sections:
   Use base_complete as a starting point:
   defaults:
     - /base_complete
   
4. Invalid enum values:
   Check valid values in the error messages above
   
5. For custom configs:
   Inherit from base_complete to get all required sections:
   defaults:
     - /base_complete
     - /profiles/gpu_rtx4080
""")
            
            sys.exit(1 if strict else 0)
            
    except FileNotFoundError as e:
        click.secho(f"✗ File not found: {e}", fg='red')
        sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Validation error: {e}", fg='red')
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command('list-profiles')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def list_profiles_command(verbose: bool):
    """
    List available hardware profiles.
    
    Shows all available hardware profiles with their characteristics.
    
    Example:
        ign-lidar-hd list-profiles
        ign-lidar-hd list-profiles -v
    """
    from ign_lidar import configs
    
    configs_dir = Path(configs.__file__).parent / 'profiles'
    
    if not configs_dir.exists():
        click.echo("No profiles directory found")
        return
    
    profiles = sorted(configs_dir.glob('*.yaml'))
    
    if not profiles:
        click.echo("No profiles found")
        return
    
    click.echo("\nAvailable Hardware Profiles:")
    click.echo("=" * 80)
    
    for profile_path in profiles:
        profile_name = profile_path.stem
        click.echo(f"\n• {profile_name}")
        
        if verbose:
            # Try to read description from config
            try:
                import yaml
                with open(profile_path) as f:
                    config = yaml.safe_load(f)
                    if config and 'config_description' in config:
                        click.echo(f"  Description: {config['config_description']}")
                    
                    # Show key settings
                    if 'processor' in config:
                        proc = config['processor']
                        if 'use_gpu' in proc:
                            click.echo(f"  GPU: {'Yes' if proc['use_gpu'] else 'No'}")
                        if 'gpu_batch_size' in proc:
                            click.echo(f"  Batch size: {proc['gpu_batch_size']:,} points")
                        if 'num_workers' in proc:
                            click.echo(f"  Workers: {proc['num_workers']}")
            except Exception as e:
                if verbose:
                    click.echo(f"  (Could not read config: {e})")
        
        click.echo(f"  Usage: --config-name profiles/{profile_name}")
    
    click.echo("\n" + "=" * 80)
    click.echo("\nExample:")
    click.echo("  ign-lidar-hd process --config-name profiles/gpu_rtx4080 \\")
    click.echo("    input_dir=/data output_dir=/output\n")


@click.command('list-presets')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def list_presets_command(verbose: bool):
    """
    List available task presets.
    
    Shows all available task presets with their characteristics.
    
    Example:
        ign-lidar-hd list-presets
        ign-lidar-hd list-presets -v
    """
    from ign_lidar import configs
    
    configs_dir = Path(configs.__file__).parent / 'presets'
    
    if not configs_dir.exists():
        click.echo("No presets directory found")
        return
    
    presets = sorted(configs_dir.glob('*.yaml'))
    
    if not presets:
        click.echo("No presets found")
        return
    
    click.echo("\nAvailable Task Presets:")
    click.echo("=" * 80)
    
    for preset_path in presets:
        preset_name = preset_path.stem
        click.echo(f"\n• {preset_name}")
        
        if verbose:
            # Try to read description from config
            try:
                import yaml
                with open(preset_path) as f:
                    config = yaml.safe_load(f)
                    if config and 'config_description' in config:
                        click.echo(f"  Description: {config['config_description']}")
                    
                    # Show key settings
                    if 'processor' in config:
                        proc = config['processor']
                        if 'lod_level' in proc:
                            click.echo(f"  LOD Level: {proc['lod_level']}")
                    
                    if 'features' in config:
                        feat = config['features']
                        if 'mode' in feat:
                            click.echo(f"  Feature mode: {feat['mode']}")
                        if 'k_neighbors' in feat:
                            click.echo(f"  K-neighbors: {feat['k_neighbors']}")
            except Exception as e:
                if verbose:
                    click.echo(f"  (Could not read config: {e})")
        
        click.echo(f"  Usage: --config-name presets/{preset_name}")
    
    click.echo("\n" + "=" * 80)
    click.echo("\nExample:")
    click.echo("  ign-lidar-hd process --config-name presets/asprs_classification_gpu \\")
    click.echo("    input_dir=/data output_dir=/output\n")


@click.command('show-config')
@click.argument('config_name')
@click.option('--resolve', is_flag=True, help='Resolve defaults and show merged config')
def show_config_command(config_name: str, resolve: bool):
    """
    Show the contents of a configuration.
    
    Display the raw or merged configuration file.
    
    Examples:
    
        # Show a profile
        ign-lidar-hd show-config profiles/gpu_rtx4080
        
        # Show a preset
        ign-lidar-hd show-config presets/asprs_classification_gpu
        
        # Show merged config (with defaults resolved)
        ign-lidar-hd show-config --resolve presets/asprs_classification_gpu
    """
    from ign_lidar import configs
    
    # Try to find the config file
    configs_dir = Path(configs.__file__).parent
    
    # Try different locations
    possible_paths = [
        configs_dir / f"{config_name}.yaml",
        configs_dir / config_name,
        Path(config_name)
    ]
    
    config_path = None
    for path in possible_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path is None:
        click.secho(f"✗ Config not found: {config_name}", fg='red')
        click.echo("\nTried locations:")
        for path in possible_paths:
            click.echo(f"  - {path}")
        sys.exit(1)
    
    click.echo(f"Config: {config_path}")
    click.echo("=" * 80)
    
    if resolve:
        # Show merged config with defaults resolved
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(config_path)
            
            # Try to resolve defaults if they exist
            if 'defaults' in cfg:
                click.echo("Note: Defaults found but not fully resolved")
                click.echo("Use --cfg job with the process command for full resolution")
                click.echo("-" * 80)
            
            click.echo(OmegaConf.to_yaml(cfg))
        except Exception as e:
            click.secho(f"✗ Error loading config: {e}", fg='red')
            sys.exit(1)
    else:
        # Show raw config
        try:
            with open(config_path) as f:
                click.echo(f.read())
        except Exception as e:
            click.secho(f"✗ Error reading config: {e}", fg='red')
            sys.exit(1)


# Export all commands
__all__ = [
    'validate_config_command',
    'list_profiles_command',
    'list_presets_command',
    'show_config_command'
]
