#!/usr/bin/env python3
"""
CLI for IGN LiDAR HD v2.0.

Modern command-line interface combining Hydra configuration management
with Click commands for improved user experience.

Commands:
    process       - Process LiDAR tiles to create training patches
    verify        - Verify dataset quality and features
    info          - Display configuration and preset information
    batch-convert - Convert patches to QGIS-compatible format
    download      - Download IGN LiDAR HD tiles

Usage:
    ign-lidar-hd process input_dir=data/raw output_dir=data/patches
    ign-lidar-hd process experiment=buildings_lod2 input_dir=data/raw output_dir=data/patches
    ign-lidar-hd verify output_dir=data/patches
    ign-lidar-hd info
    ign-lidar-hd batch-convert input_dir=data/patches --output data/qgis --format qgis
    ign-lidar-hd download --position 650000 6860000 --radius 5000 data/
"""

import logging
import sys
from pathlib import Path

import click

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, verbose):
    """IGN LiDAR HD v2.0 - LiDAR Processing Library"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    setup_logging(verbose)

    if ctx.invoked_subcommand is None:
        # If no command specified, show help
        click.echo(ctx.get_help())


# Import and register commands
try:
    from .commands import (
        auto_config,
        batch_convert_command,
        download_command,
        ground_truth_command,
        info_command,
        list_presets_command,
        list_profiles_command,
        migrate_config,
        presets_command,
        process_command,
        show_config_command,
        update_classification_command,
        validate_config_command,
        verify_command,
    )

    # Register commands with the CLI group
    cli.add_command(process_command, name="process")
    cli.add_command(download_command, name="download")
    cli.add_command(verify_command, name="verify")
    cli.add_command(batch_convert_command, name="batch-convert")
    cli.add_command(info_command, name="info")
    cli.add_command(ground_truth_command, name="ground-truth")
    cli.add_command(update_classification_command, name="update-classification")
    cli.add_command(auto_config, name="auto-config")
    cli.add_command(presets_command, name="presets")
    cli.add_command(migrate_config, name="migrate-config")
    cli.add_command(validate_config_command, name="validate-config")
    cli.add_command(list_profiles_command, name="list-profiles")
    cli.add_command(list_presets_command, name="list-presets")
    cli.add_command(show_config_command, name="show-config")

except ImportError as e:
    logger.warning(
        f"Some commands may not be available due to missing dependencies: {e}"
    )

    # Fallback - add minimal commands
    @cli.command()
    @click.argument("overrides", nargs=-1)
    def process(overrides):
        """Process LiDAR tiles (fallback implementation)."""
        click.echo("Processing functionality requires full installation.")
        click.echo("Please install with: pip install -e .")
        sys.exit(1)

    @cli.command()
    def info():
        """Show basic package information."""
        click.echo("IGN LiDAR HD v2.0 - LiDAR Processing Library")
        click.echo(
            "For full functionality, install dependencies with: pip install -e ."
        )


def main():
    """Main entry point for console scripts."""
    try:
        cli()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if logger.getEffectiveLevel() <= logging.DEBUG:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
