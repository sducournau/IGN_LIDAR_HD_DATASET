#!/usr/bin/env python3
"""
Script pour enrichir les tuiles LAZ par lots avec gestion mémoire.
Évite les crashes OOM en traitant un nombre limité de fichiers à la fois.
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import List
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_laz_files(input_dir: Path) -> List[Path]:
    """Get all LAZ files recursively."""
    return sorted(input_dir.rglob("*.laz"))


def process_batch(
    laz_files: List[Path],
    output_dir: Path,
    k_neighbors: int,
    num_workers: int,
    mode: str = 'core'
) -> bool:
    """Process a batch of LAZ files."""
    # Create temporary directory for batch
    temp_input = Path("/tmp/enrich_batch_input")
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
    
    logger.info(f"Processing batch of {len(laz_files)} files...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Batch processing failed: {e}")
        return False
    finally:
        # Cleanup symlinks
        for symlink in temp_input.glob("*.laz"):
            symlink.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Enrich LAZ files in batches to avoid OOM"
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
        "--num-workers", type=int, default=2,
        help="Number of parallel workers per batch"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Number of files to process per batch (default: 10)"
    )
    parser.add_argument(
        "--mode", type=str, choices=['core', 'full'], default='core',
        help="Feature mode: core (basic) or building (full) (default: core)"
    )
    
    args = parser.parse_args()
    
    # Get all LAZ files
    laz_files = get_laz_files(args.input_dir)
    if not laz_files:
        logger.error(f"No LAZ files found in {args.input_dir}")
        return 1
    
    logger.info("="*70)
    logger.info(f"Batch enrichment configuration:")
    logger.info(f"  Total files: {len(laz_files)}")
    logger.info(f"  Batch size:  {args.batch_size}")
    logger.info(f"  Workers/batch: {args.num_workers}")
    logger.info(f"  Total batches: {(len(laz_files) + args.batch_size - 1) // args.batch_size}")
    logger.info("="*70)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Process in batches
    total_batches = (len(laz_files) + args.batch_size - 1) // args.batch_size
    for i in range(0, len(laz_files), args.batch_size):
        batch_num = (i // args.batch_size) + 1
        batch = laz_files[i:i + args.batch_size]
        
        logger.info("")
        logger.info(f"Processing batch {batch_num}/{total_batches}")
        logger.info(f"Files in batch: {', '.join(f.name for f in batch)}")
        
        success = process_batch(
            batch,
            args.output,
            args.k_neighbors,
            args.num_workers,
            args.mode
        )
        
        if not success:
            logger.error(f"Batch {batch_num} failed, stopping")
            return 1
        
        logger.info(f"✓ Batch {batch_num}/{total_batches} completed")
    
    logger.info("")
    logger.info("="*70)
    logger.info("✓ All batches processed successfully!")
    logger.info("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
