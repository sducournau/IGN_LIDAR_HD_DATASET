"""Batch convert command for IGN LiDAR HD CLI."""

import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from tqdm import tqdm

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for converted files')
@click.option('--batch-size', '-b', default=10, type=int,
              help='Number of files to process in parallel (default: 10)')
@click.option('--force', '-f', is_flag=True,
              help='Overwrite existing output files')
@click.option('--max-points', '-m', default=100000, type=int,
              help='Maximum points per output file for simplification')
@click.option('--format', '-fmt', default='las', 
              type=click.Choice(['las', 'csv', 'qgis']), 
              help='Output format (default: las)')
@click.pass_context
def batch_convert_command(ctx, input_dir, output, batch_size, force, max_points, format):
    """Batch convert LAZ files for QGIS compatibility or other formats."""
    try:
        from ...io.qgis_converter import simplify_for_qgis
    except ImportError as e:
        logger.error(f"Missing dependencies for conversion: {e}")
        raise click.ClickException("Conversion functionality requires additional dependencies")
    
    verbose = ctx.obj.get('verbose', False) if ctx.obj else False
    input_dir = Path(input_dir)
    output_dir = Path(output) if output else input_dir / f"{format}_converted"
    
    try:
        # Find all LAZ/LAS files
        laz_files = list(input_dir.rglob("*.laz"))
        las_files = list(input_dir.rglob("*.las"))
        all_files = laz_files + las_files
        
        if not all_files:
            click.echo(f"No LAZ/LAS files found in {input_dir}")
            return
        
        click.echo(f"Found {len(all_files)} point cloud files")
        click.echo(f"Output directory: {output_dir}")
        click.echo(f"Format: {format}")
        click.echo(f"Max points per file: {max_points:,}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def convert_file(input_file):
            """Convert a single file."""
            try:
                # Determine output extension based on format
                if format == 'qgis' or format == 'las':
                    ext = '.las'
                elif format == 'csv':
                    ext = '.csv'
                else:
                    ext = '.las'
                
                output_file = output_dir / f"{input_file.stem}_converted{ext}"
                
                if output_file.exists() and not force:
                    return input_file.name, "skipped (exists)"
                
                if format == 'qgis' or format == 'las':
                    # Use QGIS converter for simplification
                    simplify_for_qgis(
                        input_file, 
                        output_file, 
                        max_points=max_points,
                        verbose=False
                    )
                elif format == 'csv':
                    # Convert to CSV
                    convert_to_csv(input_file, output_file, max_points)
                
                return input_file.name, "success"
                
            except Exception as e:
                logger.error(f"Error converting {input_file}: {e}")
                return input_file.name, f"error: {e}"
        
        # Process files in parallel
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all tasks
            futures = {executor.submit(convert_file, f): f for f in all_files}
            
            # Process results with progress bar
            with tqdm(total=len(all_files), desc="Converting") as pbar:
                for future in as_completed(futures):
                    filename, status = future.result()
                    
                    if status == "success":
                        success_count += 1
                    elif status.startswith("skipped"):
                        skipped_count += 1
                    else:
                        failed_count += 1
                        if verbose:
                            click.echo(f"❌ {filename}: {status}")
                    
                    pbar.update(1)
        
        click.echo(f"\n✅ Batch conversion complete!")
        click.echo(f"  Successfully converted: {success_count}")
        click.echo(f"  Skipped (existing): {skipped_count}")
        click.echo(f"  Failed: {failed_count}")
        click.echo(f"  Output directory: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Batch conversion error: {e}", err=True)
        sys.exit(1)


def convert_to_csv(input_file: Path, output_file: Path, max_points: int = 100000) -> None:
    """Convert LAS/LAZ to CSV format."""
    import laspy
    import numpy as np
    
    with laspy.open(input_file) as las_file:
        las = las_file.read()
        
        # Sample points if too many
        if len(las.points) > max_points:
            indices = np.random.choice(len(las.points), max_points, replace=False)
            x = las.x[indices]
            y = las.y[indices] 
            z = las.z[indices]
            
            # Get additional attributes if available
            attrs = {}
            if hasattr(las, 'classification'):
                attrs['classification'] = las.classification[indices]
            if hasattr(las, 'intensity'):
                attrs['intensity'] = las.intensity[indices]
            if hasattr(las, 'red'):
                attrs['red'] = las.red[indices]
                attrs['green'] = las.green[indices]
                attrs['blue'] = las.blue[indices]
        else:
            x, y, z = las.x, las.y, las.z
            attrs = {}
            if hasattr(las, 'classification'):
                attrs['classification'] = las.classification
            if hasattr(las, 'intensity'):
                attrs['intensity'] = las.intensity
            if hasattr(las, 'red'):
                attrs['red'] = las.red
                attrs['green'] = las.green
                attrs['blue'] = las.blue
    
    # Write to CSV
    import pandas as pd
    
    data = {'x': x, 'y': y, 'z': z}
    data.update(attrs)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)