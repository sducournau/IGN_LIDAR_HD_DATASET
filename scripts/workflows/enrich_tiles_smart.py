#!/usr/bin/env python3
"""
Script optimisé pour enrichir les tuiles LAZ avec gestion intelligente de la mémoire.
Groupe les tuiles par taille et traite les petites en parallèle, les grandes en séquentiel.
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Dict
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_laz_file_sizes(input_dir: Path) -> List[tuple]:
    """Get all LAZ files with their sizes in MB."""
    laz_files = []
    for laz in sorted(input_dir.rglob("*.laz")):
        size_mb = laz.stat().st_size / (1024 * 1024)
        laz_files.append((laz, size_mb))
    return laz_files


def categorize_files(laz_files: List[tuple], threshold_mb: float = 100) -> Dict:
    """Categorize files as small, medium, or large."""
    small = []  # < threshold
    medium = []  # threshold to 2*threshold
    large = []  # > 2*threshold
    
    for laz, size in laz_files:
        if size < threshold_mb:
            small.append(laz)
        elif size < threshold_mb * 2:
            medium.append(laz)
        else:
            large.append(laz)
    
    return {
        'small': small,
        'medium': medium,
        'large': large
    }


def process_files(
    laz_files: List[Path],
    output_dir: Path,
    k_neighbors: int,
    num_workers: int,
    mode: str,
    category: str
) -> bool:
    """Process a list of LAZ files."""
    if not laz_files:
        return True
    
    logger.info("")
    logger.info("="*70)
    logger.info(f"Processing {len(laz_files)} {category} files...")
    logger.info(f"Workers: {num_workers}")
    logger.info("="*70)
    
    # Create temporary directory
    temp_input = Path("/tmp/enrich_smart_batch")
    temp_input.mkdir(exist_ok=True)
    
    # Create symlinks
    for laz in laz_files:
        symlink = temp_input / laz.name
        if symlink.exists():
            symlink.unlink()
        symlink.symlink_to(laz)
    
    # Run enrich command
    cmd = [
        sys.executable, "-m", "ign_lidar.cli", "enrich",
        "--input-dir", str(temp_input),
        "--output", str(output_dir),
        "--k-neighbors", str(k_neighbors),
        "--num-workers", str(num_workers),
        "--mode", mode
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        success = result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Processing failed: {e}")
        success = False
    finally:
        # Cleanup symlinks
        for symlink in temp_input.glob("*.laz"):
            symlink.unlink()
    
    if success:
        logger.info(f"✓ {category.capitalize()} files completed")
    else:
        logger.error(f"✗ {category.capitalize()} files failed")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Smart LAZ enrichment with adaptive parallelization"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Input directory with LAZ files"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for enriched LAZ files"
    )
    parser.add_argument(
        "--k-neighbors", type=int, default=20,
        help="Number of neighbors for feature computation"
    )
    parser.add_argument(
        "--mode", type=str, choices=['core', 'building'], default='building',
        help="Feature mode (default: building)"
    )
    parser.add_argument(
        "--size-threshold", type=float, default=100,
        help="File size threshold in MB (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Get all LAZ files with sizes
    logger.info("Analyzing LAZ files...")
    laz_files = get_laz_file_sizes(args.input_dir)
    if not laz_files:
        logger.error(f"No LAZ files found in {args.input_dir}")
        return 1
    
    # Categorize by size
    categories = categorize_files(laz_files, args.size_threshold)
    
    # Statistics
    total_size_mb = sum(size for _, size in laz_files)
    logger.info("="*70)
    logger.info("File categorization:")
    logger.info(f"  Small  (<{args.size_threshold}MB):  "
                f"{len(categories['small'])} files")
    logger.info(f"  Medium (<{args.size_threshold*2}MB): "
                f"{len(categories['medium'])} files")
    logger.info(f"  Large  (>{args.size_threshold*2}MB):  "
                f"{len(categories['large'])} files")
    logger.info(f"  Total: {len(laz_files)} files ({total_size_mb:.1f} MB)")
    logger.info(f"  Mode: {args.mode.upper()}")
    logger.info("="*70)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Process strategy based on mode
    if args.mode == 'building':
        # Building mode: memory intensive
        small_workers = 4   # Small files in parallel
        medium_workers = 2  # Medium files with caution
        large_workers = 1   # Large files one at a time
    else:
        # Core mode: less memory
        small_workers = 6
        medium_workers = 3
        large_workers = 1
    
    # Process small files first (fastest)
    if categories['small']:
        success = process_files(
            categories['small'],
            args.output,
            args.k_neighbors,
            small_workers,
            args.mode,
            'small'
        )
        if not success:
            logger.warning("Small files failed, continuing...")
    
    # Process medium files
    if categories['medium']:
        success = process_files(
            categories['medium'],
            args.output,
            args.k_neighbors,
            medium_workers,
            args.mode,
            'medium'
        )
        if not success:
            logger.warning("Medium files failed, continuing...")
    
    # Process large files (slowest, most memory)
    if categories['large']:
        success = process_files(
            categories['large'],
            args.output,
            args.k_neighbors,
            large_workers,
            args.mode,
            'large'
        )
        if not success:
            logger.error("Large files failed!")
            return 1
    
    logger.info("")
    logger.info("="*70)
    logger.info("✓ All files processed successfully!")
    logger.info("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
