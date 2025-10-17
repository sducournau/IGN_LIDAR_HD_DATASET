"""Presets command for IGN LiDAR HD CLI - Week 3."""

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed preset information')
@click.pass_context
def presets_command(ctx, detailed):
    """List available configuration presets.
    
    Presets are pre-configured settings for common use cases:
    
    \b
    - minimal:  Fast preview, testing, development
    - lod2:     Building modeling, facade detection  
    - lod3:     Detailed architectural analysis
    - asprs:    ASPRS LAS 1.4 classification
    - full:     Maximum detail, research
    
    Examples:
    
    \b
        # List all presets
        ign-lidar-hd presets
        
        # Show detailed information
        ign-lidar-hd presets --detailed
        
        # Use a preset
        ign-lidar-hd process --preset lod2 input/ output/
    """
    
    try:
        from ...config import PresetConfigLoader
        
        loader = PresetConfigLoader(verbose=False)
        presets = loader.list_presets()
        
        if not presets:
            click.echo("⚠️  No presets found.")
            click.echo("   Check installation: pip install -e .")
            return
        
        click.echo("\n" + "="*80)
        click.echo("Available Configuration Presets")
        click.echo("="*80 + "\n")
        
        if not detailed:
            # Simple list
            click.echo(f"Found {len(presets)} presets:\n")
            
            for preset in presets:
                try:
                    info = loader.get_preset_info(preset)
                    speed_emoji = info.get('speed', '').split()[0] if info.get('speed') else '  '
                    use_case = info.get('use_case', 'No description')
                    click.echo(f"  {speed_emoji} {preset:10s} - {use_case}")
                except Exception:
                    click.echo(f"     {preset:10s} - (info unavailable)")
            
            click.echo("\n" + "-"*80)
            click.echo("Usage: ign-lidar-hd process --preset <name> input/ output/")
            click.echo("       ign-lidar-hd presets --detailed (for more info)")
            click.echo("="*80 + "\n")
        
        else:
            # Detailed information
            for preset in presets:
                try:
                    info = loader.get_preset_info(preset)
                    
                    click.echo(f"📋 {preset.upper()}")
                    click.echo("-" * 40)
                    click.echo(f"Use case: {info.get('use_case', 'Unknown')}")
                    click.echo(f"Speed:    {info.get('speed', 'Unknown')}")
                    
                    # Try to load and show key settings
                    try:
                        config = loader.load(preset=preset)
                        click.echo(f"\nKey settings:")
                        click.echo(f"  - LOD level: {config.get('processor', {}).get('lod_level', 'N/A')}")
                        click.echo(f"  - Feature mode: {config.get('features', {}).get('mode', 'N/A')}")
                        click.echo(f"  - K neighbors: {config.get('features', {}).get('k_neighbors', 'N/A')}")
                        click.echo(f"  - GPU batch size: {config.get('processor', {}).get('gpu_batch_size', 'N/A'):,}")
                    except Exception as e:
                        logger.debug(f"Could not load config for {preset}: {e}")
                    
                    click.echo()
                
                except Exception as e:
                    click.echo(f"⚠️  {preset}: Could not load info")
                    logger.debug(f"Error loading preset info: {e}")
                    click.echo()
            
            click.echo("="*80)
            click.echo("Usage: ign-lidar-hd process --preset <name> input/ output/")
            click.echo("="*80 + "\n")
    
    except ImportError as e:
        click.echo(f"❌ Error: Preset system not available")
        click.echo(f"   {e}")
        click.echo("   Install with: pip install -e .")
        ctx.exit(1)
    except Exception as e:
        click.echo(f"❌ Error loading presets: {e}")
        logger.exception("Preset loading error")
        ctx.exit(1)
