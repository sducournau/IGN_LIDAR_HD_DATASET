#!/usr/bin/env python3
"""
Patch all example configuration files to add missing required sections.

This script adds the missing 'preprocess' and 'stitching' sections to all
example configs that don't have them, preventing runtime crashes.

Usage:
    python scripts/patch_configs.py [--backup] [--verify] [--dry-run]
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Standard sections to add if missing
PREPROCESS_SECTION = """
# ============================================================================
# PREPROCESS
# ============================================================================
preprocess:
  enabled: false
  remove_duplicates: true
  remove_outliers: true
  outlier_std_multiplier: 3.0
"""

STITCHING_SECTION = """
stitching:
  enabled: false
  buffer_size: 10.0
  blend_overlap: true
"""


def has_section(content: str, section_name: str) -> bool:
    """Check if configuration content has a section."""
    # Check for the section at the start of a line (not indented)
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{section_name}:"):
            # Make sure it's not indented (top-level section)
            if not line.startswith(' ') and not line.startswith('\t'):
                return True
    return False


def add_missing_sections(content: str) -> Tuple[str, List[str]]:
    """
    Add missing sections to configuration content.
    
    Returns:
        Tuple of (modified_content, list_of_added_sections)
    """
    added_sections = []
    
    # Check for missing sections
    has_preprocess = has_section(content, 'preprocess')
    has_stitching = has_section(content, 'stitching')
    
    if has_preprocess and has_stitching:
        return content, added_sections
    
    # Find insertion point (before variable_object_filtering or logging)
    lines = content.split('\n')
    insertion_idx = None
    
    # Look for good insertion points
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith('variable_object_filtering:') or
            stripped.startswith('logging:') or
            stripped.startswith('optimizations:')):
            # Found a good spot - insert before this section
            insertion_idx = i
            # Back up to find the comment block before this section
            while insertion_idx > 0 and (lines[insertion_idx - 1].strip().startswith('#') or
                                          lines[insertion_idx - 1].strip() == ''):
                insertion_idx -= 1
            break
    
    if insertion_idx is None:
        # No good insertion point found, append at the end
        # But before the Hydra section if it exists
        for i, line in enumerate(lines):
            if line.strip().startswith('hydra:'):
                insertion_idx = i
                while insertion_idx > 0 and (lines[insertion_idx - 1].strip().startswith('#') or
                                              lines[insertion_idx - 1].strip() == ''):
                    insertion_idx -= 1
                break
        
        if insertion_idx is None:
            insertion_idx = len(lines)
    
    # Add missing sections
    sections_to_add = []
    
    if not has_preprocess:
        sections_to_add.append(PREPROCESS_SECTION)
        added_sections.append('preprocess')
    
    if not has_stitching:
        sections_to_add.append(STITCHING_SECTION)
        added_sections.append('stitching')
    
    if sections_to_add:
        # Join sections with newline
        new_content = '\n'.join(sections_to_add)
        
        # Insert at the appropriate location
        lines.insert(insertion_idx, new_content)
        content = '\n'.join(lines)
    
    return content, added_sections


def patch_config_file(config_path: Path, backup: bool = False, dry_run: bool = False) -> bool:
    """
    Patch a single configuration file.
    
    Args:
        config_path: Path to configuration file
        backup: If True, create backup before modifying
        dry_run: If True, don't actually modify files
    
    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Read original content
        with open(config_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Add missing sections
        modified_content, added_sections = add_missing_sections(original_content)
        
        if not added_sections:
            logger.info(f"✓ {config_path.name}: Already complete")
            return False
        
        logger.info(f"✓ {config_path.name}: Adding {', '.join(added_sections)}")
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would modify {config_path}")
            return True
        
        # Create backup if requested
        if backup:
            backup_path = config_path.with_suffix('.yaml.bak')
            shutil.copy2(config_path, backup_path)
            logger.debug(f"  Created backup: {backup_path}")
        
        # Write modified content
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ {config_path.name}: Error - {e}")
        return False


def verify_config(config_path: Path) -> bool:
    """
    Verify configuration can be loaded and has required sections.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check for required sections (top-level)
        required_sections = ['processor', 'features']
        missing = [s for s in required_sections if s not in config]
        
        if missing:
            logger.error(f"✗ {config_path.name}: Missing sections: {', '.join(missing)}")
            return False
        
        # Check for preprocess and stitching in content
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_preprocess = has_section(content, 'preprocess')
        has_stitching = has_section(content, 'stitching')
        
        if not has_preprocess or not has_stitching:
            missing = []
            if not has_preprocess:
                missing.append('preprocess')
            if not has_stitching:
                missing.append('stitching')
            logger.error(f"✗ {config_path.name}: Missing sections: {', '.join(missing)}")
            return False
        
        logger.info(f"✓ {config_path.name}: Valid")
        return True
        
    except yaml.YAMLError as e:
        logger.error(f"✗ {config_path.name}: YAML error - {e}")
        return False
    except Exception as e:
        logger.error(f"✗ {config_path.name}: Error - {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Patch example configs to add missing required sections'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup files before modifying'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify all configs after patching'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without modifying files'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Find all example configs
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / 'examples'
    
    if not examples_dir.exists():
        logger.error(f"Examples directory not found: {examples_dir}")
        return 1
    
    config_files = sorted(examples_dir.glob('*.yaml'))
    
    if not config_files:
        logger.error(f"No config files found in {examples_dir}")
        return 1
    
    logger.info(f"Found {len(config_files)} configuration files")
    logger.info("=" * 80)
    
    if args.dry_run:
        logger.info("[DRY RUN MODE] - No files will be modified")
        logger.info("=" * 80)
    
    # Patch all configs
    modified_count = 0
    error_count = 0
    
    for config_path in config_files:
        if patch_config_file(config_path, backup=args.backup, dry_run=args.dry_run):
            modified_count += 1
    
    logger.info("=" * 80)
    logger.info(f"Summary: {modified_count} files modified, {len(config_files) - modified_count} already complete")
    
    # Verify if requested
    if args.verify and not args.dry_run:
        logger.info("=" * 80)
        logger.info("Verifying all configurations...")
        logger.info("=" * 80)
        
        valid_count = 0
        for config_path in config_files:
            if verify_config(config_path):
                valid_count += 1
            else:
                error_count += 1
        
        logger.info("=" * 80)
        logger.info(f"Verification: {valid_count}/{len(config_files)} configs valid")
        
        if error_count > 0:
            logger.error(f"❌ {error_count} configs have errors")
            return 1
    
    logger.info("✓ Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
