"""Info command for IGN LiDAR HD CLI."""

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def get_config_dir() -> str:
    """Get absolute path to configs directory."""
    package_dir = Path(__file__).parent.parent.parent
    config_dir = package_dir / "configs"
    return str(config_dir.absolute()) if config_dir.exists() else None


@click.command()
@click.option('--version', '-v', is_flag=True, help='Show version information')
@click.option('--dependencies', '-d', is_flag=True, help='Show dependency information')  
@click.option('--config', '-c', is_flag=True, help='Show configuration paths')
@click.option('--presets', '-p', is_flag=True, help='Show available presets')
@click.pass_context
def info_command(ctx, version, dependencies, config, presets):
    """Show package information, version, dependencies, and configuration."""
    
    # If no specific option is selected, show all
    if not any([version, dependencies, config, presets]):
        version = dependencies = config = presets = True
    
    if version:
        click.echo("IGN LiDAR HD Dataset Processing Library")
        click.echo("=" * 50)
        try:
            import ign_lidar
            click.echo(f"Version: {ign_lidar.__version__}")
        except (ImportError, AttributeError):
            click.echo("Version: 2.0.0 (development)")
        click.echo("Description: Advanced LiDAR processing for building LOD classification")
        click.echo()
    
    if dependencies:
        click.echo("Dependencies:")
        click.echo("=" * 20)
        
        deps = [
            ("numpy", "Array processing", True),
            ("laspy", "LAZ/LAS file handling", True),  
            ("lazrs", "LAZ compression", True),
            ("scikit-learn", "Machine learning", True),
            ("tqdm", "Progress bars", True),
            ("click", "CLI framework", True),
            ("hydra-core", "Configuration management", True),
            ("omegaconf", "Configuration validation", True),
            ("requests", "HTTP requests", False),
            ("Pillow", "Image processing", False),
            ("cupy", "GPU acceleration", False),
            ("cuml", "GPU ML acceleration", False),
            ("pandas", "Data analysis", False)
        ]
        
        for name, desc, required in deps:
            try:
                __import__(name)
                status = "✅"
            except ImportError:
                status = "❌" if required else "⚠️ "
            
            req_text = " (required)" if required else " (optional)"
            click.echo(f"  {status} {name:15s} - {desc}{req_text}")
        click.echo()
    
    if config:
        click.echo("Configuration:")
        click.echo("=" * 20)
        config_dir = get_config_dir()
        
        if config_dir:
            click.echo(f"Config directory: {config_dir}")
            
            # List available configurations
            config_path = Path(config_dir)
            
            # Check for experiment configs
            exp_dir = config_path / "experiment"
            if exp_dir.exists():
                exp_files = list(exp_dir.glob("*.yaml"))
                if exp_files:
                    click.echo(f"Experiment presets: {len(exp_files)} available")
            
            # Check for other config types
            for subdir in ["features", "processor", "preprocess"]:
                sub_path = config_path / subdir
                if sub_path.exists():
                    configs = list(sub_path.glob("*.yaml"))
                    if configs:
                        click.echo(f"{subdir.capitalize()} configs: {len(configs)} available")
        else:
            click.echo("Config directory: Not found")
            click.echo("Using default configuration")
        
        click.echo()
    
    if presets:
        click.echo("Available Presets:")
        click.echo("=" * 20)
        
        try:
            from ...config.defaults import list_presets
            presets_dict = list_presets()
            
            if presets_dict:
                for name, description in presets_dict.items():
                    click.echo(f"  {name:20s} - {description}")
            else:
                click.echo("  No presets found")
        except ImportError:
            click.echo("  Preset information not available")
        
        click.echo()
        click.echo("Usage examples:")
        click.echo("  ign-lidar-hd process input_dir=data/raw output_dir=data/patches")
        click.echo("  ign-lidar-hd process experiment=buildings_lod2 input_dir=data/raw")
        click.echo("  ign-lidar-hd download --position 650000 6860000 --radius 5000 data/")
        click.echo("  ign-lidar-hd verify data/patches/")
        click.echo("  ign-lidar-hd batch-convert data/patches/ --format qgis")