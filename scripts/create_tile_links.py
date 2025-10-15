#!/usr/bin/env python3
"""
Create Tile Links - Multi-Scale Training Pipeline

This script creates symbolic links or copies selected tiles from the unified
dataset to the appropriate output directories for processing.

It reads the tile selection lists and creates organized directory structures
for each classification level.

Options:
- Symbolic links (default, saves space)
- Hard links (more compatible)
- File copies (most compatible, uses more space)
"""

import argparse
import logging
import sys
import shutil
from pathlib import Path
from typing import List, Dict
import os


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_tile_list(list_file: Path) -> List[str]:
    """
    Read tile names from a list file.
    
    Args:
        list_file: Path to the tile list file
        
    Returns:
        List of tile names
    """
    if not list_file.exists():
        logger.warning(f"Tile list not found: {list_file}")
        return []
    
    with open(list_file, 'r') as f:
        tiles = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Read {len(tiles)} tile names from {list_file.name}")
    return tiles


def create_link(source: Path, target: Path, link_type: str = 'symlink') -> bool:
    """
    Create a link from source to target.
    
    Args:
        source: Source file path
        target: Target link path
        link_type: Type of link ('symlink', 'hardlink', 'copy')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure target directory exists
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing target if it exists
        if target.exists() or target.is_symlink():
            target.unlink()
        
        if link_type == 'symlink':
            # Create symbolic link
            target.symlink_to(source)
            
        elif link_type == 'hardlink':
            # Create hard link
            os.link(source, target)
            
        elif link_type == 'copy':
            # Copy file
            shutil.copy2(source, target)
            
        else:
            logger.error(f"Unknown link type: {link_type}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating link {target}: {e}")
        return False


def process_level(
    source_dir: Path,
    target_dir: Path,
    tile_names: List[str],
    link_type: str,
    level_name: str
) -> Dict[str, int]:
    """
    Process tiles for a classification level.
    
    Args:
        source_dir: Source directory containing LAZ files
        target_dir: Target directory for links
        tile_names: List of tile names to link
        link_type: Type of link to create
        level_name: Name of the classification level
        
    Returns:
        Dictionary with statistics
    """
    logger.info(f"\nProcessing {level_name.upper()} tiles...")
    logger.info(f"  Source: {source_dir}")
    logger.info(f"  Target: {target_dir}")
    
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return {'requested': len(tile_names), 'created': 0, 'failed': 0, 'skipped': 0}
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'requested': len(tile_names),
        'created': 0,
        'failed': 0,
        'skipped': 0
    }
    
    for tile_name in tile_names:
        source_file = source_dir / tile_name
        target_file = target_dir / tile_name
        
        if not source_file.exists():
            logger.warning(f"Source file not found: {source_file.name}")
            stats['failed'] += 1
            continue
        
        if target_file.exists() or target_file.is_symlink():
            logger.debug(f"Target already exists: {target_file.name}")
            stats['skipped'] += 1
            continue
        
        if create_link(source_file, target_file, link_type):
            stats['created'] += 1
        else:
            stats['failed'] += 1
    
    logger.info(f"{level_name.upper()} Results:")
    logger.info(f"  Requested: {stats['requested']}")
    logger.info(f"  Created:   {stats['created']}")
    logger.info(f"  Skipped:   {stats['skipped']}")
    logger.info(f"  Failed:    {stats['failed']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Create links for selected tiles"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source unified_dataset directory"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target selected_tiles directory"
    )
    parser.add_argument(
        "--lists",
        type=str,
        required=False,
        help="Pattern for tile list files (e.g., 'selected_tiles/*.txt')"
    )
    parser.add_argument(
        "--link-type",
        type=str,
        choices=['symlink', 'hardlink', 'copy'],
        default='symlink',
        help="Type of link to create (default: symlink)"
    )
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    target_path = Path(args.target)
    
    # Validate source directory
    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_path}")
        sys.exit(1)
    
    logger.info(f"Creating tile links...")
    logger.info(f"  Link type: {args.link_type}")
    
    # Process each classification level
    levels = ['asprs', 'lod2', 'lod3']
    total_stats = {
        'requested': 0,
        'created': 0,
        'failed': 0,
        'skipped': 0
    }
    
    for level in levels:
        # Find tile list file
        list_file = target_path / f"{level}_selected_tiles.txt"
        
        if not list_file.exists():
            logger.warning(f"Tile list not found: {list_file}")
            logger.info(f"Skipping {level.upper()}")
            continue
        
        # Read tile list
        tile_names = read_tile_list(list_file)
        
        if not tile_names:
            logger.warning(f"No tiles in list for {level.upper()}")
            continue
        
        # Create links
        source_dir = source_path / level
        target_dir = target_path / level
        
        stats = process_level(source_dir, target_dir, tile_names, args.link_type, level)
        
        # Accumulate statistics
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tiles requested: {total_stats['requested']}")
    logger.info(f"Total links created:   {total_stats['created']}")
    logger.info(f"Total skipped:         {total_stats['skipped']}")
    logger.info(f"Total failed:          {total_stats['failed']}")
    
    if total_stats['failed'] > 0:
        logger.warning(f"\n{total_stats['failed']} tiles failed to link!")
        return 1
    
    logger.info("\nTile linking complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
