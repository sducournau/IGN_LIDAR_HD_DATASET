#!/usr/bin/env python3
"""
Download LOD3 tiles from strategic locations.

This script downloads LiDAR tiles from locations known to contain
complex architectural features suitable for LOD3 modeling.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import click

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ign_lidar.downloader import IGNLiDARDownloader
from ign_lidar.datasets.strategic_locations import STRATEGIC_LOCATIONS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_lod3_locations() -> Dict[str, Dict[str, Any]]:
    """
    Get strategic locations suitable for LOD3 modeling.
    
    Returns:
        Dictionary of locations with LOD3 characteristics
    """
    lod3_locations = {}
    
    for name, data in STRATEGIC_LOCATIONS.items():
        characteristics = data.get('characteristics', [])
        category = data.get('category', '')
        
        # Filter for LOD3-suitable locations
        if ('lod3' in characteristics or 
            'heritage' in category or
            'chateau' in category or
            'cathedrale' in name or
            'palais' in name or
            'toitures_complexes' in characteristics or
            'geometrie_complexe' in characteristics):
            lod3_locations[name] = data
    
    return lod3_locations


@click.command()
@click.option('--output-dir', '-o', 
              type=click.Path(),
              default='/mnt/c/Users/Simon/ign/unified_dataset/lod3/tiles',
              help='Output directory for downloaded tiles')
@click.option('--max-tiles', '-n',
              type=int,
              default=60,
              help='Maximum number of tiles to download')
@click.option('--max-concurrent', '-j',
              type=int,
              default=3,
              help='Maximum concurrent downloads')
@click.option('--skip-existing',
              is_flag=True,
              default=True,
              help='Skip tiles that already exist')
@click.option('--list-locations',
              is_flag=True,
              help='List available LOD3 locations without downloading')
@click.option('--location', '-l',
              multiple=True,
              help='Download specific locations (can be used multiple times)')
@click.option('--dry-run',
              is_flag=True,
              help='Show what would be downloaded without actually downloading')
def main(output_dir, max_tiles, max_concurrent, skip_existing, list_locations, location, dry_run):
    """Download LOD3 tiles from strategic architectural locations."""
    
    # Get LOD3 locations
    lod3_locations = get_lod3_locations()
    
    if list_locations:
        click.echo("=" * 80)
        click.echo("Available LOD3 Locations")
        click.echo("=" * 80)
        for name, data in lod3_locations.items():
            category = data.get('category', 'unknown')
            characteristics = ', '.join(data.get('characteristics', []))
            target_tiles = data.get('target_tiles', 'unknown')
            click.echo(f"\n{name}:")
            click.echo(f"  Category: {category}")
            click.echo(f"  Characteristics: {characteristics}")
            click.echo(f"  Target tiles: {target_tiles}")
            bbox = data.get('bbox', (0, 0, 0, 0))
            click.echo(f"  BBox: {bbox}")
        click.echo(f"\nTotal LOD3 locations: {len(lod3_locations)}")
        return
    
    # Filter locations if specific ones requested
    if location:
        filtered_locations = {
            name: data for name, data in lod3_locations.items()
            if name in location
        }
        if not filtered_locations:
            click.echo(f"❌ No matching locations found for: {', '.join(location)}", err=True)
            sys.exit(1)
        lod3_locations = filtered_locations
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    click.echo("=" * 80)
    click.echo("LOD3 Tile Download")
    click.echo("=" * 80)
    click.echo(f"Output directory: {output_path}")
    click.echo(f"Max tiles: {max_tiles}")
    click.echo(f"Locations to process: {len(lod3_locations)}")
    click.echo(f"Skip existing: {skip_existing}")
    click.echo(f"Dry run: {dry_run}")
    click.echo()
    
    # Initialize downloader
    downloader = IGNLiDARDownloader(output_path, max_concurrent=max_concurrent)
    
    # Collect tiles from all locations
    all_tiles = []
    tiles_per_location = {}
    
    for name, data in lod3_locations.items():
        bbox = data.get('bbox')
        if not bbox:
            logger.warning(f"No bbox for location: {name}")
            continue
        
        # bbox is in WGS84 (lon, lat), need to convert to Lambert93
        lon_min, lat_min, lon_max, lat_max = bbox
        
        try:
            logger.info(f"Fetching tiles for {name}...")
            
            # Use pyproj to convert WGS84 to Lambert93
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
                xmin, ymin = transformer.transform(lon_min, lat_min)
                xmax, ymax = transformer.transform(lon_max, lat_max)
            except ImportError:
                # Fallback: approximate conversion
                # Lambert93 origin: lon=3°, lat=46.5°
                # Rough approximation (not accurate but better than nothing)
                xmin = (lon_min - 3) * 111000 * 0.7 + 700000
                ymin = (lat_min - 46.5) * 111000 + 6600000
                xmax = (lon_max - 3) * 111000 * 0.7 + 700000
                ymax = (lat_max - 46.5) * 111000 + 6600000
                logger.warning("Using approximate coordinate conversion (install pyproj for accuracy)")
            
            # Get tile coordinates in the bbox
            tile_coords = downloader.get_tiles_in_bbox(xmin, ymin, xmax, ymax)
            
            if tile_coords:
                # Convert to tile filenames
                tiles = [downloader.tile_coords_to_filename(tx, ty) for tx, ty in tile_coords]
                tiles_per_location[name] = tiles
                all_tiles.extend(tiles)
                click.echo(f"  ✓ {name}: {len(tiles)} tiles")
            else:
                click.echo(f"  ⚠ {name}: No tiles found")
        
        except Exception as e:
            logger.error(f"Error fetching tiles for {name}: {e}")
            click.echo(f"  ✗ {name}: Error - {e}")
    
    # Remove duplicates while preserving order
    unique_tiles = []
    seen = set()
    for tile in all_tiles:
        if tile not in seen:
            unique_tiles.append(tile)
            seen.add(tile)
    
    click.echo()
    click.echo(f"Found {len(unique_tiles)} unique tiles across {len(tiles_per_location)} locations")
    
    # Limit to max_tiles
    if len(unique_tiles) > max_tiles:
        click.echo(f"Limiting to {max_tiles} tiles (removing {len(unique_tiles) - max_tiles})")
        unique_tiles = unique_tiles[:max_tiles]
    
    if dry_run:
        click.echo("\n🔍 Dry run - would download these tiles:")
        for i, tile in enumerate(unique_tiles[:20], 1):
            click.echo(f"  {i}. {tile}")
        if len(unique_tiles) > 20:
            click.echo(f"  ... and {len(unique_tiles) - 20} more")
        return
    
    if not unique_tiles:
        click.echo("❌ No tiles to download")
        return
    
    # Download tiles
    click.echo(f"\n📥 Downloading {len(unique_tiles)} tiles...")
    click.echo()
    
    results = downloader.batch_download(
        unique_tiles,
        skip_existing=skip_existing
    )
    
    # Report results
    success = [tile for tile, status in results.items() if status]
    failed = [tile for tile, status in results.items() if not status]
    
    click.echo()
    click.echo("=" * 80)
    click.echo("Download Summary")
    click.echo("=" * 80)
    click.echo(f"✅ Successfully downloaded: {len(success)}")
    click.echo(f"❌ Failed: {len(failed)}")
    
    if failed:
        click.echo("\nFailed tiles:")
        for tile in failed[:10]:
            click.echo(f"  - {tile}")
        if len(failed) > 10:
            click.echo(f"  ... and {len(failed) - 10} more")
    
    click.echo(f"\n✓ LOD3 tiles saved to: {output_path}")


if __name__ == '__main__':
    main()
