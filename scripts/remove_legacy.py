#!/usr/bin/env python3
"""
IGN LiDAR HD v2.0 - Legacy Cleanup Script

Removes legacy functions, files, and CLI components after reorganization.
Creates deprecation warnings and maintains minimal backward compatibility.

Usage:
    python scripts/remove_legacy.py --dry-run     # Preview what will be removed
    python scripts/remove_legacy.py --backup     # Create backup before removal
    python scripts/remove_legacy.py              # Remove legacy components
"""

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Legacy files to remove (after deprecation period)
LEGACY_FILES_TO_REMOVE = [
    # Legacy CLI files
    "ign_lidar/cli.py",                    # 1863 lines - replaced by hydra_main.py
    "ign_lidar/cli_config.py",             # CLI configuration - replaced by Hydra
    "ign_lidar/cli_utils.py",              # CLI utilities - integrated into new CLI
    
    # Legacy config system
    "ign_lidar/config.py",                 # Old config system - replaced by Hydra
]

# Legacy files to deprecate (keep with warnings)
LEGACY_FILES_TO_DEPRECATE = [
    # Keep for compatibility but add deprecation warnings
    "ign_lidar/pipeline_config.py",        # May still be used - move to core/
]

# New deprecation wrapper files to create
DEPRECATION_WRAPPERS = {
    "ign_lidar/cli.py": {
        "replacement": "ign_lidar.cli.hydra_main",
        "message": "The legacy CLI is deprecated. Use 'ign-lidar-hd-v2' or 'python -m ign_lidar.cli.hydra_main' instead.",
        "functions_to_wrap": ["main"],
    },
    "ign_lidar/config.py": {
        "replacement": "ign_lidar.config.schema", 
        "message": "Old configuration system is deprecated. Use Hydra configuration instead.",
        "functions_to_wrap": ["load_config", "get_default_config"],
    },
}

# Update pyproject.toml script entries
SCRIPT_UPDATES = {
    # Remove legacy entry, keep v2
    "remove": ["ign-lidar-hd"],  # Remove legacy CLI entry
    "keep": ["ign-lidar-hd-v2", "ign-lidar-qgis"],  # Keep new CLI and QGIS
    "add": {
        "ign-lidar-hd": "ign_lidar.cli.hydra_main:process",  # Redirect to v2
    }
}


def create_deprecation_wrapper(file_path: Path, config: Dict) -> str:
    """Create a deprecation wrapper file."""
    
    wrapper_content = f'''"""
DEPRECATED: {file_path.name}

{config['message']}

This file will be removed in v3.0.0.
"""

import warnings
from typing import Any

# Issue deprecation warning
warnings.warn(
    "{config['message']}",
    DeprecationWarning,
    stacklevel=2
)

# Import replacement if available
try:
    from {config['replacement']} import *
    __all__ = []  # Don't export anything to encourage migration
except ImportError:
    pass


def main(*args, **kwargs) -> Any:
    """
    DEPRECATED: Legacy main function.
    
    {config['message']}
    """
    warnings.warn(
        "Legacy CLI is deprecated. Use 'ign-lidar-hd-v2' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Try to redirect to new implementation
    try:
        from ign_lidar.cli.hydra_main import process
        return process()
    except ImportError:
        print("ERROR: New CLI not available. Please install hydra-core and omegaconf.")
        print("Run: pip install hydra-core omegaconf")
        return 1


# Deprecated function aliases (for backward compatibility)
'''
    
    if 'functions_to_wrap' in config:
        for func_name in config['functions_to_wrap']:
            if func_name != 'main':  # main is already defined above
                wrapper_content += f'''
def {func_name}(*args, **kwargs) -> Any:
    """DEPRECATED: Use new implementation."""
    warnings.warn(
        "Function '{func_name}' is deprecated. {config['message']}",
        DeprecationWarning,
        stacklevel=2
    )
    # Return None or raise NotImplementedError
    raise NotImplementedError(f"Legacy function '{func_name}' has been removed. {config['message']}")
'''
    
    return wrapper_content


def backup_files(files: List[Path], backup_dir: Path) -> bool:
    """Create backup of files before removal."""
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating legacy backup in {backup_dir}...")
        
        for file_path in files:
            if file_path.exists():
                backup_path = backup_dir / file_path.name
                if file_path.is_dir():
                    shutil.copytree(file_path, backup_path, dirs_exist_ok=True)
                    print(f"  Backed up directory: {file_path.name}")
                else:
                    shutil.copy2(file_path, backup_path)
                    print(f"  Backed up file: {file_path.name}")
        
        print(f"✓ Legacy backup complete: {backup_dir}")
        return True
    
    except Exception as e:
        print(f"✗ Error creating backup: {e}", file=sys.stderr)
        return False


def update_pyproject_toml(base_dir: Path, dry_run: bool = False) -> bool:
    """Update pyproject.toml to remove legacy script entries."""
    
    pyproject_path = base_dir / "pyproject.toml"
    
    if not pyproject_path.exists():
        print("⚠ pyproject.toml not found")
        return True
    
    try:
        content = pyproject_path.read_text()
        lines = content.split('\\n')
        
        new_lines = []
        in_scripts_section = False
        
        for line in lines:
            if line.strip() == "[project.scripts]":
                in_scripts_section = True
                new_lines.append(line)
                continue
            
            if in_scripts_section:
                # Check if we're entering a new section
                if line.startswith('[') and line != "[project.scripts]":
                    in_scripts_section = False
                    new_lines.append(line)
                    continue
                
                # Process script entries
                if '=' in line:
                    script_name = line.split('=')[0].strip().strip('"')
                    
                    if script_name in SCRIPT_UPDATES["remove"]:
                        if dry_run:
                            print(f"  [DRY RUN] Would remove script: {script_name}")
                        else:
                            print(f"  Removed script: {script_name}")
                        continue  # Skip this line (remove it)
                    
                new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Add new script entries
        if not dry_run:
            # Find the scripts section and add new entries
            scripts_idx = None
            for i, line in enumerate(new_lines):
                if line.strip() == "[project.scripts]":
                    scripts_idx = i
                    break
            
            if scripts_idx is not None:
                for script_name, entry_point in SCRIPT_UPDATES["add"].items():
                    new_entry = f'{script_name} = "{entry_point}"'
                    # Check if it already exists
                    entry_exists = any(script_name in line for line in new_lines[scripts_idx:scripts_idx+10])
                    if not entry_exists:
                        # Insert after the [project.scripts] line
                        new_lines.insert(scripts_idx + 1, new_entry)
                        print(f"  Added script: {script_name}")
        
        new_content = '\\n'.join(new_lines)
        
        if not dry_run:
            pyproject_path.write_text(new_content)
            print("✓ Updated pyproject.toml")
        
        return True
    
    except Exception as e:
        print(f"✗ Error updating pyproject.toml: {e}", file=sys.stderr)
        return False


def remove_legacy_file(file_path: Path, dry_run: bool = False) -> bool:
    """Remove a legacy file."""
    try:
        if not file_path.exists():
            print(f"  ⚠ Already removed: {file_path.name}")
            return True
        
        if dry_run:
            print(f"  [DRY RUN] Would remove: {file_path.name}")
            return True
        
        file_path.unlink()
        print(f"  ✓ Removed: {file_path.name}")
        return True
    
    except Exception as e:
        print(f"  ✗ Error removing {file_path.name}: {e}", file=sys.stderr)
        return False


def create_deprecation_wrappers(base_dir: Path, dry_run: bool = False) -> bool:
    """Create deprecation wrapper files."""
    try:
        for file_rel_path, config in DEPRECATION_WRAPPERS.items():
            file_path = base_dir / file_rel_path
            
            if dry_run:
                print(f"  [DRY RUN] Would create deprecation wrapper: {file_path.name}")
                continue
            
            wrapper_content = create_deprecation_wrapper(file_path, config)
            file_path.write_text(wrapper_content)
            print(f"  ✓ Created deprecation wrapper: {file_path.name}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error creating deprecation wrappers: {e}", file=sys.stderr)
        return False


def verify_new_structure(base_dir: Path) -> bool:
    """Verify new v2.0 structure exists before removing legacy."""
    required_files = [
        base_dir / "ign_lidar/cli/hydra_main.py",
        base_dir / "ign_lidar/config/schema.py",
        base_dir / "ign_lidar/core/processor.py",
        base_dir / "ign_lidar/features/features.py",
        base_dir / "ign_lidar/preprocessing/preprocessing.py",
    ]
    
    missing = [f for f in required_files if not f.exists()]
    
    if missing:
        print("✗ ERROR: New v2.0 structure incomplete!", file=sys.stderr)
        print("Missing files:", file=sys.stderr)
        for f in missing:
            print(f"  - {f.relative_to(base_dir)}", file=sys.stderr)
        return False
    
    print("✓ New v2.0 structure verified")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove legacy files and functions from IGN LiDAR HD v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be removed
  python scripts/remove_legacy.py --dry-run
  
  # Remove with backup (recommended)
  python scripts/remove_legacy.py --backup
  
  # Remove without backup (dangerous)
  python scripts/remove_legacy.py --force
  
  # Create only deprecation warnings (gradual migration)
  python scripts/remove_legacy.py --deprecate-only
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup before removing files'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Remove files without backup or confirmation'
    )
    parser.add_argument(
        '--deprecate-only',
        action='store_true',
        help='Create deprecation wrappers but keep original files'
    )
    
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    print(f"\\n{'='*80}")
    print(f"IGN LiDAR HD v2.0 - Legacy Cleanup")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'REMOVE FILES'}")
    print(f"Backup: {'YES' if args.backup else 'NO'}")
    print(f"Deprecate only: {'YES' if args.deprecate_only else 'NO'}")
    print(f"{'='*80}\\n")
    
    # Verify new structure exists
    if not verify_new_structure(base_dir):
        print("\\nAborting cleanup. New v2.0 structure must be complete first.")
        sys.exit(1)
    
    # Get list of files to process
    files_to_remove = []
    for file_rel_path in LEGACY_FILES_TO_REMOVE:
        file_path = base_dir / file_rel_path
        if file_path.exists():
            files_to_remove.append(file_path)
    
    if not files_to_remove and not args.deprecate_only:
        print("✓ No legacy files found to remove. Cleanup already complete!")
        return
    
    # Show what will be processed
    if files_to_remove:
        print(f"Legacy files to remove ({len(files_to_remove)}):")
        print("-" * 80)
        for file_path in files_to_remove:
            rel_path = file_path.relative_to(base_dir)
            size = f"({file_path.stat().st_size} bytes)" if file_path.exists() else ""
            print(f"  📄 {rel_path} {size}")
        print()
    
    if DEPRECATION_WRAPPERS:
        print(f"Deprecation wrappers to create ({len(DEPRECATION_WRAPPERS)}):")
        print("-" * 80)
        for file_path, config in DEPRECATION_WRAPPERS.items():
            print(f"  ⚠️  {file_path} → {config['replacement']}")
        print()
    
    # Create backup if requested
    if args.backup and not args.dry_run and files_to_remove:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = base_dir / f"backup_legacy_files_{timestamp}"
        
        if not backup_files(files_to_remove, backup_dir):
            print("\\n✗ Backup failed. Aborting cleanup.")
            sys.exit(1)
        print()
    
    # Confirm removal (unless dry-run or force)
    if not args.dry_run and not args.force and files_to_remove:
        print("⚠️  WARNING: This will permanently remove legacy files!")
        print("   The new v2.0 system provides equivalent functionality.")
        print()
        response = input("Continue with removal? [y/N]: ")
        if response.lower() != 'y':
            print("Legacy cleanup cancelled.")
            return
        print()
    
    # Process files
    success_count = 0
    
    # 1. Update pyproject.toml
    print("Updating pyproject.toml...")
    print("-" * 80)
    if update_pyproject_toml(base_dir, args.dry_run):
        success_count += 1
    print()
    
    # 2. Create deprecation wrappers
    if DEPRECATION_WRAPPERS:
        print("Creating deprecation wrappers...")
        print("-" * 80)
        if create_deprecation_wrappers(base_dir, args.dry_run):
            success_count += 1
        print()
    
    # 3. Remove legacy files (unless deprecate-only)
    if files_to_remove and not args.deprecate_only:
        print("Removing legacy files...")
        print("-" * 80)
        for file_path in files_to_remove:
            if remove_legacy_file(file_path, args.dry_run):
                success_count += 1
        print()
    
    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    if args.dry_run:
        print(f"Files that would be processed: {len(files_to_remove)}")
        print(f"\\n💡 Run without --dry-run to apply changes")
        print(f"💡 Use --backup to create backup before removal")
        print(f"💡 Use --deprecate-only for gradual migration")
    elif args.deprecate_only:
        print(f"Deprecation wrappers created: {len(DEPRECATION_WRAPPERS)}")
        print(f"\\n✅ Gradual migration setup complete!")
        print(f"💡 Legacy files kept but will show deprecation warnings")
        print(f"💡 Run without --deprecate-only to fully remove legacy files")
    else:
        print(f"Files removed: {len(files_to_remove)}")
        print(f"Updates applied: {success_count}")
        if args.backup:
            print(f"\\n✅ Legacy cleanup complete with backup!")
        else:
            print(f"\\n✅ Legacy cleanup complete!")
    
    print(f"{'='*80}\\n")


if __name__ == "__main__":
    main()