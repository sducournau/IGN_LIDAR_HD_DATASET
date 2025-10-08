#!/usr/bin/env python3
"""
IGN LiDAR HD v2.0 - Cleanup Script

Safely removes old files from root that have been moved to new modular structure.
Creates a backup before deletion for safety.

Usage:
    python scripts/cleanup_old_files.py --dry-run  # Preview what will be removed
    python scripts/cleanup_old_files.py            # Actually remove files
    python scripts/cleanup_old_files.py --backup   # Create backup before removing
"""

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Files that have been moved to new modules and can be removed
FILES_TO_REMOVE = [
    # Core processing (moved to core/)
    "ign_lidar/processor.py",
    "ign_lidar/tile_stitcher.py",
    "ign_lidar/pipeline_config.py",
    
    # Features (moved to features/)
    "ign_lidar/features.py",
    "ign_lidar/features_gpu.py",
    "ign_lidar/features_gpu_chunked.py",
    "ign_lidar/features_boundary.py",
    
    # Preprocessing (moved to preprocessing/)
    "ign_lidar/preprocessing.py",
    "ign_lidar/rgb_augmentation.py",
    "ign_lidar/infrared_augmentation.py",
    
    # Formatters (moved to io/formatters/)
    "ign_lidar/formatters/",
]

# Files to keep at root level (NOT moved to modules)
FILES_TO_KEEP = [
    "ign_lidar/__init__.py",
    "ign_lidar/__pycache__/",
    "ign_lidar/.mypy_cache/",
    
    # New modular directories
    "ign_lidar/core/",
    "ign_lidar/features/",
    "ign_lidar/preprocessing/",
    "ign_lidar/io/",
    "ign_lidar/config/",
    "ign_lidar/cli/",
    "ign_lidar/datasets/",
    
    # Root level modules (kept at root)
    "ign_lidar/metadata.py",
    "ign_lidar/classes.py",
    "ign_lidar/downloader.py",
    "ign_lidar/qgis_converter.py",
    "ign_lidar/tile_list.py",
    
    # Moved modules (now in subdirectories)
    "ign_lidar/core/memory_utils.py",
    "ign_lidar/core/memory_manager.py",
    "ign_lidar/core/performance_monitor.py",
    "ign_lidar/core/verification.py",
    "ign_lidar/features/architectural_styles.py",
    "ign_lidar/preprocessing/utils.py",
    "ign_lidar/preprocessing/tile_analyzer.py",
    "ign_lidar/datasets/strategic_locations.py",
    "ign_lidar/error_handler.py",
    "ign_lidar/config.py",
    
    # Legacy CLI (will be deprecated later)
    "ign_lidar/cli.py",
    "ign_lidar/cli_config.py",
    "ign_lidar/cli_utils.py",
    "ign_lidar/cli_commands/",
]


def create_backup(files: list[Path], backup_dir: Path) -> bool:
    """Create backup of files before deletion."""
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating backup in {backup_dir}...")
        
        for file_path in files:
            if file_path.exists():
                if file_path.is_dir():
                    backup_path = backup_dir / file_path.name
                    shutil.copytree(file_path, backup_path, dirs_exist_ok=True)
                    print(f"  Backed up directory: {file_path.name}")
                else:
                    backup_path = backup_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                    print(f"  Backed up file: {file_path.name}")
        
        print(f"✓ Backup complete: {backup_dir}")
        return True
    
    except Exception as e:
        print(f"✗ Error creating backup: {e}", file=sys.stderr)
        return False


def remove_file_or_dir(path: Path, dry_run: bool = False) -> bool:
    """Remove a file or directory."""
    try:
        if not path.exists():
            print(f"  ⚠ Already removed: {path.name}")
            return True
        
        if dry_run:
            if path.is_dir():
                print(f"  [DRY RUN] Would remove directory: {path.name}")
            else:
                print(f"  [DRY RUN] Would remove file: {path.name}")
            return True
        
        if path.is_dir():
            shutil.rmtree(path)
            print(f"  ✓ Removed directory: {path.name}")
        else:
            path.unlink()
            print(f"  ✓ Removed file: {path.name}")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error removing {path.name}: {e}", file=sys.stderr)
        return False


def verify_new_structure(base_dir: Path) -> bool:
    """Verify that new modular structure exists before cleaning."""
    required_dirs = [
        base_dir / "ign_lidar/core",
        base_dir / "ign_lidar/features",
        base_dir / "ign_lidar/preprocessing",
        base_dir / "ign_lidar/io",
    ]
    
    missing = [d for d in required_dirs if not d.exists()]
    
    if missing:
        print("✗ ERROR: New modular structure not complete!", file=sys.stderr)
        print("Missing directories:", file=sys.stderr)
        for d in missing:
            print(f"  - {d}", file=sys.stderr)
        return False
    
    print("✓ New modular structure verified")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old files after reorganization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be removed
  python scripts/cleanup_old_files.py --dry-run
  
  # Remove files (no backup)
  python scripts/cleanup_old_files.py
  
  # Remove files with backup
  python scripts/cleanup_old_files.py --backup
  
  # Show what files will be kept
  python scripts/cleanup_old_files.py --show-keep
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without actually removing files'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup before removing files'
    )
    parser.add_argument(
        '--show-keep',
        action='store_true',
        help='Show which files will be kept (not removed)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip verification and proceed with cleanup'
    )
    
    args = parser.parse_args()
    
    # Get base directory (project root)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    print(f"\n{'='*80}")
    print(f"IGN LiDAR HD v2.0 - Cleanup Old Files")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Mode: {'DRY RUN (no changes will be applied)' if args.dry_run else 'REMOVE FILES'}")
    print(f"Backup: {'YES' if args.backup else 'NO'}")
    print(f"{'='*80}\n")
    
    # Show files to keep if requested
    if args.show_keep:
        print("Files that will be KEPT (not removed):")
        print("-" * 80)
        for file_path in sorted(FILES_TO_KEEP):
            full_path = base_dir / file_path
            exists = "✓" if full_path.exists() else "✗"
            print(f"  {exists} {file_path}")
        print()
        return
    
    # Verify new structure exists
    if not args.force:
        if not verify_new_structure(base_dir):
            print("\nAborting cleanup. Run with --force to skip verification.")
            sys.exit(1)
        print()
    
    # Build list of files to remove
    files_to_remove = []
    for file_path in FILES_TO_REMOVE:
        full_path = base_dir / file_path
        if full_path.exists():
            files_to_remove.append(full_path)
    
    if not files_to_remove:
        print("✓ No old files found to remove. Cleanup already complete!")
        return
    
    # Show what will be removed
    print(f"Found {len(files_to_remove)} old files/directories to remove:")
    print("-" * 80)
    for file_path in files_to_remove:
        rel_path = file_path.relative_to(base_dir)
        file_type = "DIR " if file_path.is_dir() else "FILE"
        print(f"  [{file_type}] {rel_path}")
    print()
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = base_dir / f"backup_v1_files_{timestamp}"
        
        if not create_backup(files_to_remove, backup_dir):
            print("\n✗ Backup failed. Aborting cleanup.")
            sys.exit(1)
        print()
    
    # Confirm deletion (unless dry-run)
    if not args.dry_run:
        print("⚠️  WARNING: This will permanently delete the old files!")
        print("   The new modular structure has copies of these files.")
        print()
        response = input("Continue with deletion? [y/N]: ")
        if response.lower() != 'y':
            print("Cleanup cancelled.")
            return
        print()
    
    # Remove files
    print("Removing old files...")
    print("-" * 80)
    
    success_count = 0
    for file_path in files_to_remove:
        if remove_file_or_dir(file_path, args.dry_run):
            success_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed: {success_count}/{len(files_to_remove)}")
    
    if args.dry_run:
        print(f"\n💡 Run without --dry-run to actually remove files")
        print(f"💡 Use --backup to create a backup before removing")
    elif success_count == len(files_to_remove):
        print(f"\n✅ Cleanup complete!")
        if args.backup:
            print(f"✅ Backup saved to: backup_v1_files_{timestamp}/")
    else:
        print(f"\n⚠️  Some files could not be removed")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
