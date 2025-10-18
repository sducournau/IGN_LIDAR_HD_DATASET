#!/usr/bin/env python3
"""
Merge multiple LAZ files into a single LAZ file.

This script reads all LAZ files from a specified directory and merges them
into a single output LAZ file, preserving point attributes and metadata.
"""

import argparse
import logging
from pathlib import Path
from typing import List
import numpy as np
import laspy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_laz_files(directory: Path) -> List[Path]:
    """
    Find all LAZ files in the specified directory.
    
    Args:
        directory: Path to directory containing LAZ files
        
    Returns:
        List of Path objects for LAZ files
    """
    laz_files = sorted(directory.glob("*.laz"))
    if not laz_files:
        laz_files = sorted(directory.glob("*.LAZ"))
    
    logger.info(f"Found {len(laz_files)} LAZ files in {directory}")
    return laz_files


def merge_laz_files(input_files: List[Path], output_file: Path, chunk_size: int = 10000000):
    """
    Merge multiple LAZ files into a single LAZ file.
    
    Args:
        input_files: List of input LAZ file paths
        output_file: Output LAZ file path
        chunk_size: Number of points to process at once (default: 10M)
    """
    if not input_files:
        raise ValueError("No input files provided")
    
    logger.info(f"Merging {len(input_files)} files into {output_file}")
    
    # Read the first file to get header information
    logger.info(f"Reading header from {input_files[0].name}")
    with laspy.open(input_files[0]) as first_file:
        # Create header based on first file
        header = laspy.LasHeader(
            point_format=first_file.header.point_format,
            version=first_file.header.version
        )
        
        # Copy scales and offsets
        header.scales = first_file.header.scales
        header.offsets = first_file.header.offsets
        
        # Calculate total number of points
        total_points = 0
        for laz_file in input_files:
            with laspy.open(laz_file) as f:
                total_points += f.header.point_count
        
        logger.info(f"Total points to merge: {total_points:,}")
        
        # Create output file
        logger.info(f"Creating output file: {output_file}")
        with laspy.open(output_file, mode='w', header=header) as out_file:
            points_written = 0
            
            # Process each input file
            for i, laz_file in enumerate(input_files, 1):
                logger.info(f"Processing file {i}/{len(input_files)}: {laz_file.name}")
                
                with laspy.open(laz_file) as in_file:
                    file_point_count = in_file.header.point_count
                    logger.info(f"  Points in file: {file_point_count:,}")
                    
                    # Read and write in chunks
                    for points in in_file.chunk_iterator(chunk_size):
                        out_file.write_points(points)
                        points_written += len(points.x)
                        
                        if points_written % (chunk_size * 10) == 0:
                            logger.info(f"  Progress: {points_written:,}/{total_points:,} points ({100*points_written/total_points:.1f}%)")
            
            logger.info(f"Successfully wrote {points_written:,} points to {output_file}")
    
    # Verify output file
    with laspy.open(output_file) as verify_file:
        actual_points = verify_file.header.point_count
        logger.info(f"Verification: Output file contains {actual_points:,} points")
        
        if actual_points != total_points:
            logger.warning(f"Point count mismatch! Expected {total_points:,}, got {actual_points:,}")
        else:
            logger.info("✓ Point count verified successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple LAZ files into a single LAZ file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all LAZ files from a directory
  python merge_laz_files.py -i /path/to/laz/files -o merged.laz
  
  # Merge with custom chunk size (for memory management)
  python merge_laz_files.py -i /path/to/laz/files -o merged.laz --chunk-size 5000000
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=Path,
        required=True,
        help='Directory containing LAZ files to merge'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output LAZ file path'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000000,
        help='Number of points to process at once (default: 10,000,000)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    if not args.input_dir.is_dir():
        logger.error(f"Input path is not a directory: {args.input_dir}")
        return 1
    
    # Find LAZ files
    laz_files = find_laz_files(args.input_dir)
    
    if not laz_files:
        logger.error(f"No LAZ files found in {args.input_dir}")
        return 1
    
    # Create output directory if it doesn't exist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge files
    try:
        merge_laz_files(laz_files, args.output, args.chunk_size)
        logger.info("✓ Merge completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error during merge: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
