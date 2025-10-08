"""Verify command for IGN LiDAR HD CLI."""

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output file for verification report')
@click.option('--detailed', '-d', is_flag=True,
              help='Show detailed statistics')
@click.option('--features-only', '-f', is_flag=True,
              help='Verify features only (skip RGB/NIR)')
@click.pass_context  
def verify_command(ctx, input_path, output, detailed, features_only):
    """Verify LiDAR data quality and features."""
    try:
        from ...core.verification import FeatureVerifier
    except ImportError as e:
        logger.error(f"Missing dependencies for verification: {e}")
        # Basic verification without advanced features
        return basic_verify(input_path, detailed)
    
    verbose = ctx.obj.get('verbose', False) if ctx.obj else False
    input_path = Path(input_path)
    
    try:
        verifier = FeatureVerifier()
        
        if verbose:
            logger.info(f"Verifying: {input_path}")
        
        if input_path.is_file():
            # Verify single file
            results = verifier.verify_file(input_path)
            
            click.echo("=" * 70)
            click.echo("Verification Results")
            click.echo("=" * 70)
            click.echo(f"File: {input_path.name}")
            click.echo(f"Valid: {'✅' if results['valid'] else '❌'}")
            
            if detailed and results.get('stats'):
                stats = results['stats']
                click.echo(f"Points: {stats.get('num_points', 'N/A'):,}")
                click.echo(f"Features: {stats.get('num_features', 'N/A')}")
                click.echo(f"Size: {input_path.stat().st_size / 1024 / 1024:.1f} MB")
                
        elif input_path.is_dir():
            # Verify directory
            results = verifier.verify_directory(input_path)
            
            click.echo("=" * 70)
            click.echo("Verification Results")
            click.echo("=" * 70)
            click.echo(f"Directory: {input_path}")
            click.echo(f"Total files: {results['total_files']}")
            click.echo(f"Valid files: {results['valid_files']}")
            click.echo(f"Invalid files: {results['invalid_files']}")
            
            if results['errors']:
                click.echo("\nErrors found:")
                for error in results['errors'][:10]:  # Show first 10
                    click.echo(f"  - {error}")
                if len(results['errors']) > 10:
                    click.echo(f"  ... and {len(results['errors']) - 10} more")
        
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                import json
                json.dump(results, f, indent=2)
            click.echo(f"Report saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"❌ Verification error: {e}", err=True)
        sys.exit(1)


def basic_verify(input_path: Path, detailed: bool = False) -> None:
    """Basic verification without advanced features."""
    click.echo("=" * 70)
    click.echo("Basic Verification Results")
    click.echo("=" * 70)
    
    if input_path.is_file():
        click.echo(f"File: {input_path.name}")
        click.echo(f"Exists: ✅")
        click.echo(f"Size: {input_path.stat().st_size / 1024 / 1024:.1f} MB")
        click.echo(f"Extension: {input_path.suffix}")
        
    elif input_path.is_dir():
        # Count files
        laz_files = list(input_path.rglob("*.laz"))
        las_files = list(input_path.rglob("*.las"))
        npz_files = list(input_path.rglob("*.npz"))
        
        click.echo(f"Directory: {input_path}")
        click.echo(f"LAZ files: {len(laz_files)}")
        click.echo(f"LAS files: {len(las_files)}")
        click.echo(f"NPZ files: {len(npz_files)}")
        click.echo(f"Total point cloud files: {len(laz_files) + len(las_files)}")
        
        if detailed:
            total_size = sum(
                f.stat().st_size 
                for f in (laz_files + las_files + npz_files)
            )
            click.echo(f"Total size: {total_size / 1024 / 1024 / 1024:.1f} GB")
            
        if len(laz_files) + len(las_files) + len(npz_files) == 0:
            click.echo("⚠️  No point cloud files found!")
        else:
            click.echo("✅ Files found!")
    
    click.echo("=" * 70)