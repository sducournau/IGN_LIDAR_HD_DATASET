#!/usr/bin/env python3
"""
Version Bumper for IGN LiDAR HD

Updates version numbers across all files in the project.

Usage:
    # Bump to v3.7.0 (transitional release)
    python scripts/bump_version.py 3.7.0 --type minor

    # Bump to v4.0.0-alpha.1
    python scripts/bump_version.py 4.0.0-alpha.1 --type major

    # Bump to v4.0.0 (final)
    python scripts/bump_version.py 4.0.0 --type major
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple
import json


class VersionBumper:
    """Updates version numbers across project files"""
    
    def __init__(self, new_version: str, root_dir: Path):
        self.new_version = new_version
        self.root_dir = root_dir
        self.files_updated: List[Tuple[Path, str, str]] = []
        
        # Files to update with their patterns
        self.version_files = {
            'pyproject.toml': [
                (r'version\s*=\s*"[^"]+"', f'version = "{new_version}"'),
            ],
            'ign_lidar/__init__.py': [
                (r'__version__\s*=\s*"[^"]+"', f'__version__ = "{new_version}"'),
            ],
            'docs/docusaurus.config.ts': [
                (r'version:\s*["\'][^"\']+["\']', f'version: "{new_version}"'),
            ],
            'CHANGELOG.md': [
                # Add new version section at top
                ('# Changelog', 
                 f'# Changelog\n\n## [{new_version}] - TBD\n\n**Status:** In Development\n\n### Added\n\n### Changed\n\n### Fixed\n\n### Removed\n\n---\n'),
            ],
        }
    
    def get_current_version(self) -> str:
        """Get current version from pyproject.toml"""
        pyproject = self.root_dir / 'pyproject.toml'
        if pyproject.exists():
            content = pyproject.read_text()
            match = re.search(r'version\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
        return "unknown"
    
    def update_file(self, rel_path: str, patterns: List[Tuple[str, str]]) -> bool:
        """Update a single file with version patterns"""
        file_path = self.root_dir / rel_path
        
        if not file_path.exists():
            print(f"⚠️  File not found: {rel_path}")
            return False
        
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            for pattern, replacement in patterns:
                if isinstance(pattern, str) and pattern in content:
                    # Simple string replacement
                    content = content.replace(pattern, replacement, 1)
                else:
                    # Regex replacement
                    content = re.sub(pattern, replacement, content, count=1)
            
            if content != original_content:
                # Create backup
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                backup_path.write_text(original_content, encoding='utf-8')
                
                # Write new content
                file_path.write_text(content, encoding='utf-8')
                
                old_ver = self.get_current_version()
                self.files_updated.append((file_path, old_ver, self.new_version))
                print(f"✅ Updated: {rel_path}")
                return True
            else:
                print(f"⚠️  No changes made to: {rel_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error updating {rel_path}: {e}")
            return False
    
    def update_all(self) -> None:
        """Update all version files"""
        print(f"\n🔄 Updating version to {self.new_version}...")
        print(f"📁 Root directory: {self.root_dir}")
        
        current_version = self.get_current_version()
        print(f"📊 Current version: {current_version}")
        print(f"📊 New version: {self.new_version}\n")
        
        for rel_path, patterns in self.version_files.items():
            self.update_file(rel_path, patterns)
        
        print(f"\n✅ Updated {len(self.files_updated)} files")
        
        if self.files_updated:
            print("\n📋 Summary of changes:")
            for file_path, old_ver, new_ver in self.files_updated:
                print(f"  • {file_path.relative_to(self.root_dir)}")
                print(f"    {old_ver} → {new_ver}")
        
        print("\n⚠️  Next steps:")
        print("1. Review changes")
        print("2. Update CHANGELOG.md with actual changes")
        print("3. Run tests: pytest tests/ -v")
        print("4. Commit: git add -A && git commit -m 'Bump version to {}'".format(self.new_version))
        print("5. Tag: git tag v{}".format(self.new_version))
    
    def restore_backups(self) -> None:
        """Restore all backup files"""
        print("\n🔄 Restoring backups...")
        
        for rel_path in self.version_files.keys():
            file_path = self.root_dir / rel_path
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            
            if backup_path.exists():
                backup_content = backup_path.read_text(encoding='utf-8')
                file_path.write_text(backup_content, encoding='utf-8')
                backup_path.unlink()
                print(f"✅ Restored: {rel_path}")
        
        print("✅ All backups restored")


def validate_version(version: str) -> bool:
    """Validate version string format"""
    # Support semantic versioning with optional pre-release tags
    pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$'
    return bool(re.match(pattern, version))


def main():
    parser = argparse.ArgumentParser(
        description='Bump version across all project files'
    )
    parser.add_argument(
        'version',
        help='New version number (e.g., 3.7.0, 4.0.0-alpha.1, 4.0.0)'
    )
    parser.add_argument(
        '--type',
        choices=['major', 'minor', 'patch'],
        help='Type of version bump'
    )
    parser.add_argument(
        '--root',
        default='.',
        help='Root directory of project'
    )
    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore from backups'
    )
    
    args = parser.parse_args()
    
    if not validate_version(args.version):
        print(f"❌ Invalid version format: {args.version}")
        print("Expected format: X.Y.Z or X.Y.Z-prerelease")
        return 1
    
    root_dir = Path(args.root).resolve()
    bumper = VersionBumper(args.version, root_dir)
    
    if args.restore:
        bumper.restore_backups()
    else:
        bumper.update_all()
    
    return 0


if __name__ == '__main__':
    exit(main())
