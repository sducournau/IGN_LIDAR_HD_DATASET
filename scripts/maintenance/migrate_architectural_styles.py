#!/usr/bin/env python3
"""
Migrate tile metadata to support multi-label architectural styles.

This script enriches existing tile metadata files with multi-label
architectural style information, allowing for more accurate representation
of mixed architectural styles in each tile.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

# Import direct pour éviter les dépendances lourdes
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ign_lidar.architectural_styles import (
    infer_multi_styles_from_characteristics
)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def migrate_tile_metadata(metadata_file: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single tile metadata file to multi-label format.
    
    Args:
        metadata_file: Path to metadata JSON file
        dry_run: If True, don't save changes
        
    Returns:
        True if migrated successfully
    """
    try:
        # Load existing metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check if already migrated
        if "architectural_styles" in metadata:
            logger.debug(f"  Already migrated: {metadata_file.name}")
            return True
        
        # Infer multi-styles from characteristics
        characteristics = metadata.get("characteristics", [])
        category = metadata.get("location", {}).get("category")
        
        if not characteristics and not category:
            logger.warning(f"  No characteristics found: {metadata_file.name}")
            return False
        
        # Infer styles
        multi_styles = infer_multi_styles_from_characteristics(characteristics)
        
        # Add to metadata
        metadata["architectural_styles"] = multi_styles
        
        # Set dominant style
        dominant = max(multi_styles, key=lambda x: x.get("weight", 0))
        metadata["dominant_style_id"] = dominant["style_id"]
        
        # Log migration
        style_names = [s["style_name"] for s in multi_styles]
        logger.info(f"  ✓ {metadata_file.stem}: {', '.join(style_names)}")
        
        # Save if not dry run
        if not dry_run:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Error migrating {metadata_file.name}: {e}")
        return False


def migrate_directory(input_dir: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Migrate all tile metadata files in a directory.
    
    Args:
        input_dir: Directory containing metadata JSON files
        dry_run: If True, don't save changes
        
    Returns:
        Dictionary with migration statistics
    """
    stats = {
        "total": 0,
        "migrated": 0,
        "already_migrated": 0,
        "failed": 0
    }
    
    # Find all JSON files
    json_files = list(input_dir.rglob("*.json"))
    
    # Filter out stats.json
    json_files = [f for f in json_files if f.name != "stats.json"]
    
    if not json_files:
        logger.warning(f"No metadata files found in {input_dir}")
        return stats
    
    logger.info(f"Found {len(json_files)} metadata files")
    logger.info("=" * 70)
    
    for json_file in json_files:
        stats["total"] += 1
        
        # Check if already migrated
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            if "architectural_styles" in metadata:
                stats["already_migrated"] += 1
                continue
        except:
            pass
        
        # Migrate
        if migrate_tile_metadata(json_file, dry_run):
            stats["migrated"] += 1
        else:
            stats["failed"] += 1
    
    return stats


def analyze_styles_distribution(input_dir: Path) -> Dict[str, int]:
    """
    Analyze the distribution of architectural styles in metadata files.
    
    Args:
        input_dir: Directory containing metadata JSON files
        
    Returns:
        Dictionary mapping style_name to count
    """
    style_counts = {}
    
    json_files = list(input_dir.rglob("*.json"))
    json_files = [f for f in json_files if f.name != "stats.json"]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            # Check for multi-styles
            styles = metadata.get("architectural_styles", [])
            if not styles:
                # Try to infer
                characteristics = metadata.get("characteristics", [])
                styles = infer_multi_styles_from_characteristics(characteristics)
            
            for style in styles:
                style_name = style.get("style_name", "unknown")
                weight = style.get("weight", 1.0)
                style_counts[style_name] = style_counts.get(style_name, 0) + weight
        except:
            continue
    
    return dict(sorted(style_counts.items(), key=lambda x: x[1], reverse=True))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate tile metadata to multi-label architectural styles",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing tile metadata JSON files'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without saving'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze style distribution instead of migrating'
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    if args.analyze:
        # Analyze distribution
        logger.info("Analyzing architectural style distribution...")
        logger.info("=" * 70)
        
        distribution = analyze_styles_distribution(args.input_dir)
        
        logger.info("\nArchitectural Style Distribution:")
        logger.info("-" * 70)
        
        total_weight = sum(distribution.values())
        for style_name, count in distribution.items():
            percentage = (count / total_weight * 100) if total_weight > 0 else 0
            logger.info(f"  {style_name:20s}: {count:6.1f} ({percentage:5.1f}%)")
        
        logger.info("-" * 70)
        logger.info(f"  Total: {total_weight:.1f}")
        
    else:
        # Migrate
        mode = "DRY RUN" if args.dry_run else "MIGRATE"
        logger.info(f"Migrating tile metadata [{mode}]")
        logger.info(f"Input: {args.input_dir}")
        logger.info("=" * 70)
        
        stats = migrate_directory(args.input_dir, args.dry_run)
        
        logger.info("=" * 70)
        logger.info("Migration Summary:")
        logger.info(f"  Total files: {stats['total']}")
        logger.info(f"  Migrated: {stats['migrated']}")
        logger.info(f"  Already migrated: {stats['already_migrated']}")
        logger.info(f"  Failed: {stats['failed']}")
        
        if args.dry_run:
            logger.info("\n⚠️  DRY RUN: No files were modified")
            logger.info("   Run without --dry-run to apply changes")
        else:
            logger.info("\n✅ Migration complete!")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
