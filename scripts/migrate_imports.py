#!/usr/bin/env python3
"""
IGN LiDAR HD v2.0 - Import Migration Script

Automatically updates imports from v1.x to v2.0 structure.

Usage:
    python scripts/migrate_imports.py --path /path/to/your/code
    python scripts/migrate_imports.py --file myfile.py
    python scripts/migrate_imports.py --dry-run  # Preview changes without applying
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Migration map: old import -> new import
IMPORT_MIGRATIONS = {
    # Core processing
    "from ign_lidar.processor import": "from ign_lidar.core.processor import",
    "from ign_lidar.tile_stitcher import": "from ign_lidar.core.tile_stitcher import",
    "from ign_lidar.pipeline_config import": "from ign_lidar.core.pipeline_config import",
    
    # Features
    "from ign_lidar.features import": "from ign_lidar.features.features import",
    "from ign_lidar.features_gpu import": "from ign_lidar.features.features_gpu import",
    "from ign_lidar.features_gpu_chunked import": "from ign_lidar.features.features_gpu_chunked import",
    "from ign_lidar.features_boundary import": "from ign_lidar.features.features_boundary import",
    
    # Preprocessing
    "from ign_lidar.preprocessing import": "from ign_lidar.preprocessing.preprocessing import",
    "from ign_lidar.rgb_augmentation import": "from ign_lidar.preprocessing.rgb_augmentation import",
    "from ign_lidar.infrared_augmentation import": "from ign_lidar.preprocessing.infrared_augmentation import",
    
    # Formatters
    "from ign_lidar.formatters.": "from ign_lidar.io.formatters.",
}

# Alternative imports (using module-level imports)
ALTERNATIVE_IMPORTS = {
    "from ign_lidar.features.features import": "from ign_lidar.features import",
    "from ign_lidar.preprocessing.preprocessing import": "from ign_lidar.preprocessing import",
}


def find_python_files(path: Path) -> List[Path]:
    """Find all Python files in a directory."""
    if path.is_file():
        return [path] if path.suffix == '.py' else []
    
    return list(path.rglob("*.py"))


def migrate_imports(content: str, use_alternative: bool = False) -> Tuple[str, int]:
    """
    Migrate imports in file content.
    
    Returns:
        Tuple of (migrated_content, number_of_changes)
    """
    migrated = content
    num_changes = 0
    
    # Apply migrations
    migrations = ALTERNATIVE_IMPORTS if use_alternative else IMPORT_MIGRATIONS
    
    for old_import, new_import in migrations.items():
        if old_import in migrated:
            migrated = migrated.replace(old_import, new_import)
            num_changes += migrated.count(new_import) - content.count(old_import)
    
    return migrated, num_changes


def migrate_file(file_path: Path, dry_run: bool = False, use_alternative: bool = False) -> Tuple[bool, int]:
    """
    Migrate imports in a single file.
    
    Returns:
        Tuple of (success, number_of_changes)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        migrated, num_changes = migrate_imports(content, use_alternative)
        
        if num_changes > 0:
            if not dry_run:
                file_path.write_text(migrated, encoding='utf-8')
                print(f"✓ Migrated {file_path} ({num_changes} changes)")
            else:
                print(f"[DRY RUN] Would migrate {file_path} ({num_changes} changes)")
            return True, num_changes
        
        return True, 0
    
    except Exception as e:
        print(f"✗ Error migrating {file_path}: {e}", file=sys.stderr)
        return False, 0


def show_diff(old_content: str, new_content: str):
    """Show a simple diff of changes."""
    old_lines = old_content.split('\n')
    new_lines = new_content.split('\n')
    
    print("\n" + "="*80)
    print("CHANGES:")
    print("="*80)
    
    for i, (old, new) in enumerate(zip(old_lines, new_lines), 1):
        if old != new:
            print(f"\nLine {i}:")
            print(f"  - {old}")
            print(f"  + {new}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate IGN LiDAR HD imports from v1.x to v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate all files in a directory
  python scripts/migrate_imports.py --path /path/to/your/code
  
  # Migrate a single file
  python scripts/migrate_imports.py --file myfile.py
  
  # Preview changes without applying
  python scripts/migrate_imports.py --path /path/to/code --dry-run
  
  # Use module-level imports (cleaner)
  python scripts/migrate_imports.py --path /path/to/code --alternative
  
  # Show detailed diff
  python scripts/migrate_imports.py --file myfile.py --diff
        """
    )
    
    parser.add_argument(
        '--path',
        type=Path,
        help='Path to directory containing Python files to migrate'
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Path to single Python file to migrate'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )
    parser.add_argument(
        '--alternative',
        action='store_true',
        help='Use module-level imports (e.g., from ign_lidar.features import ...)'
    )
    parser.add_argument(
        '--diff',
        action='store_true',
        help='Show detailed diff of changes'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.path and not args.file:
        parser.error("Either --path or --file must be specified")
    
    if args.path and args.file:
        parser.error("Cannot specify both --path and --file")
    
    # Find files to migrate
    if args.file:
        files = [args.file]
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
    else:
        files = find_python_files(args.path)
        if not files:
            print(f"Error: No Python files found in {args.path}", file=sys.stderr)
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"IGN LiDAR HD v2.0 - Import Migration")
    print(f"{'='*80}")
    print(f"Found {len(files)} Python file(s) to process")
    print(f"Mode: {'DRY RUN (no changes will be applied)' if args.dry_run else 'APPLY CHANGES'}")
    print(f"Import style: {'Module-level (cleaner)' if args.alternative else 'Direct (explicit)'}")
    print(f"{'='*80}\n")
    
    # Migrate files
    total_changes = 0
    success_count = 0
    
    for file_path in files:
        if args.diff:
            content = file_path.read_text(encoding='utf-8')
            migrated, num_changes = migrate_imports(content, args.alternative)
            if num_changes > 0:
                show_diff(content, migrated)
        
        success, num_changes = migrate_file(file_path, args.dry_run, args.alternative)
        if success:
            success_count += 1
            total_changes += num_changes
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed: {success_count}/{len(files)}")
    print(f"Total changes: {total_changes}")
    
    if args.dry_run and total_changes > 0:
        print(f"\n💡 Run without --dry-run to apply changes")
    elif total_changes > 0:
        print(f"\n✅ Migration complete!")
    else:
        print(f"\n✓ No changes needed (already migrated or no matching imports)")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
