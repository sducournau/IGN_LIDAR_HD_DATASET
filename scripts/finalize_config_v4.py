#!/usr/bin/env python3
"""
Configuration v4.0 Finalizer - Removes v3.x backward compatibility

This script removes v3.x configuration support as part of v4.0.0 release.

Actions:
1. DELETE ign_lidar/config/schema.py (415 lines) - Old v3.1 Hydra config
2. DELETE ign_lidar/config/schema_simplified.py (~300 lines) - Interim config
3. REMOVE backward compatibility from ign_lidar/config/config.py
4. UPDATE imports in __init__.py files
5. UPDATE tests to use v4.0 structure only

Usage:
    python scripts/finalize_config_v4.py --dry-run  # Preview changes
    python scripts/finalize_config_v4.py --execute  # Apply changes
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import re


class ConfigV4Finalizer:
    """Finalizes configuration v4.0 by removing v3.x support"""
    
    def __init__(self, root_dir: Path, dry_run: bool = True):
        self.root_dir = root_dir
        self.dry_run = dry_run
        self.actions_taken: List[str] = []
        self.files_to_delete: List[Path] = []
        self.files_to_modify: List[Tuple[Path, str]] = []
    
    def plan_deletions(self) -> None:
        """Plan which files to delete"""
        # Old config files to delete
        files = [
            'ign_lidar/config/schema.py',
            'ign_lidar/config/schema_simplified.py',
        ]
        
        for file_path in files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                self.files_to_delete.append(full_path)
                print(f"📋 Plan to DELETE: {file_path}")
            else:
                print(f"⚠️  File not found (already deleted?): {file_path}")
    
    def plan_config_modifications(self) -> None:
        """Plan modifications to config.py"""
        config_file = self.root_dir / 'ign_lidar' / 'config' / 'config.py'
        
        if not config_file.exists():
            print(f"⚠️  Config file not found: {config_file}")
            return
        
        print(f"📋 Plan to MODIFY: ign_lidar/config/config.py")
        print("   - Remove from_legacy_schema() method")
        print("   - Remove processor parameter backward compatibility")
        print("   - Remove deprecated warnings")
        
        self.files_to_modify.append((config_file, 'remove_legacy_config'))
    
    def plan_init_modifications(self) -> None:
        """Plan modifications to __init__.py files"""
        init_files = [
            'ign_lidar/__init__.py',
            'ign_lidar/config/__init__.py',
        ]
        
        for init_path in init_files:
            full_path = self.root_dir / init_path
            if full_path.exists():
                print(f"📋 Plan to MODIFY: {init_path}")
                print("   - Remove schema imports")
                print("   - Remove backward compatibility imports")
                self.files_to_modify.append((full_path, 'remove_schema_imports'))
    
    def execute_deletions(self) -> None:
        """Execute file deletions"""
        for file_path in self.files_to_delete:
            if self.dry_run:
                print(f"🔵 DRY RUN - Would delete: {file_path}")
            else:
                try:
                    # Backup first
                    backup_path = file_path.with_suffix('.py.v3backup')
                    shutil.copy2(file_path, backup_path)
                    
                    file_path.unlink()
                    self.actions_taken.append(f"DELETED: {file_path}")
                    print(f"✅ Deleted: {file_path}")
                    print(f"   Backup saved to: {backup_path}")
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {e}")
    
    def remove_legacy_from_config(self, file_path: Path) -> None:
        """Remove legacy methods from config.py"""
        if self.dry_run:
            print(f"🔵 DRY RUN - Would modify: {file_path}")
            print("   Changes:")
            print("   - Remove from_legacy_schema() method")
            print("   - Remove 'processor' parameter handling")
            print("   - Remove deprecation warnings in __post_init__")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Backup
            backup_path = file_path.with_suffix('.py.v3backup')
            shutil.copy2(file_path, backup_path)
            
            # Remove from_legacy_schema method (find and remove entire method)
            # This is a placeholder - actual implementation would need careful parsing
            print(f"⚠️  Manual removal required for {file_path}")
            print("   Please manually remove:")
            print("   1. from_legacy_schema() classmethod")
            print("   2. Backward compatibility warnings in __post_init__")
            print("   3. 'processor' parameter handling")
            
            self.actions_taken.append(f"FLAGGED FOR MANUAL EDIT: {file_path}")
            
        except Exception as e:
            print(f"❌ Error modifying {file_path}: {e}")
    
    def remove_schema_imports(self, file_path: Path) -> None:
        """Remove schema imports from __init__.py"""
        if self.dry_run:
            print(f"🔵 DRY RUN - Would modify: {file_path}")
            print("   Changes:")
            print("   - Remove schema.py imports")
            print("   - Remove schema_simplified.py imports")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Backup
            backup_path = file_path.with_suffix('.py.v3backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            # Filter out schema imports
            new_lines = []
            skip_block = False
            
            for line in lines:
                # Check if this is a schema import
                if 'from .config.schema import' in line or 'from .schema import' in line:
                    print(f"   Removing: {line.strip()}")
                    continue
                
                # Check for try/except blocks around schema imports
                if 'IGNLiDARConfig' in line or 'ProcessorConfig' in line:
                    # Skip lines related to old config
                    continue
                
                new_lines.append(line)
            
            # Write modified content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            self.actions_taken.append(f"MODIFIED: {file_path}")
            print(f"✅ Modified: {file_path}")
            
        except Exception as e:
            print(f"❌ Error modifying {file_path}: {e}")
    
    def run(self) -> None:
        """Execute the finalization process"""
        print("\n" + "="*80)
        print("Configuration v4.0 Finalization")
        print("="*80)
        
        if self.dry_run:
            print("\n🔵 DRY RUN MODE - No changes will be made")
        else:
            print("\n⚠️  EXECUTE MODE - Changes will be applied!")
            print("Press Ctrl+C to cancel...")
            import time
            time.sleep(3)
        
        print("\n📋 Planning changes...")
        self.plan_deletions()
        self.plan_config_modifications()
        self.plan_init_modifications()
        
        print(f"\n📊 Summary:")
        print(f"   Files to delete: {len(self.files_to_delete)}")
        print(f"   Files to modify: {len(self.files_to_modify)}")
        
        if not self.dry_run:
            print("\n🔧 Executing changes...")
            self.execute_deletions()
            
            for file_path, action_type in self.files_to_modify:
                if action_type == 'remove_legacy_config':
                    self.remove_legacy_from_config(file_path)
                elif action_type == 'remove_schema_imports':
                    self.remove_schema_imports(file_path)
        
        print("\n" + "="*80)
        print("Summary of actions:")
        if self.actions_taken:
            for action in self.actions_taken:
                print(f"  ✅ {action}")
        else:
            print("  🔵 No actions taken (dry run mode)")
        print("="*80 + "\n")
        
        if not self.dry_run:
            print("⚠️  IMPORTANT NEXT STEPS:")
            print("1. Review backup files (*.v3backup)")
            print("2. Run tests: pytest tests/ -v")
            print("3. Fix any import errors")
            print("4. Update documentation")
            print("5. Commit changes to v4.0-dev branch")


def main():
    parser = argparse.ArgumentParser(
        description='Finalize configuration v4.0 by removing v3.x support'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute changes (creates backups first)'
    )
    parser.add_argument(
        '--root',
        default='.',
        help='Root directory of project'
    )
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("❌ Error: Must specify either --dry-run or --execute")
        parser.print_help()
        return 1
    
    root_dir = Path(args.root).resolve()
    finalizer = ConfigV4Finalizer(root_dir, dry_run=args.dry_run)
    finalizer.run()
    
    return 0


if __name__ == '__main__':
    exit(main())
