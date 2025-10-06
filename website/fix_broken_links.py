#!/usr/bin/env python3
"""
Automated Link Fixer for Docusaurus
Fixes common broken link patterns in documentation files.
"""

import re
from pathlib import Path
import shutil
from datetime import datetime

class LinkFixer:
    def __init__(self):
        self.docs_dir = Path("docs")
        self.blog_dir = Path("blog")
        self.backup_dir = Path("link_fixes_backup") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.fixes_applied = []
        self.dry_run = True
    
    def fix_file(self, file_path):
        """Fix links in a single file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = []
        
        # Fix 1: Remove /docs/ prefix from internal links
        # Pattern: [text](/docs/path) -> [text](/path)
        pattern1 = r'\[([^\]]+)\]\(/docs/([^\)]+)\)'
        matches = re.findall(pattern1, content)
        if matches:
            for text, path in matches:
                old = f"[{text}](/docs/{path})"
                new = f"[{text}](/{path})"
                content = content.replace(old, new)
                changes.append(f"  ✓ /docs/{path} → /{path}")
        
        # Fix 2: Remove .md extension from relative links
        # Pattern: [text](./path.md) -> [text](./path)
        # Pattern: [text](../path.md) -> [text](../path)
        pattern2 = r'\[([^\]]+)\]\((\.\.?/[^\)]+)\.md(\)|#[^\)]*\))'
        matches = re.findall(pattern2, content)
        if matches:
            for text, path, ending in matches:
                old = f"[{text}]({path}.md{ending}"
                new = f"[{text}]({path}{ending}"
                content = content.replace(old, new)
                changes.append(f"  ✓ {path}.md → {path}")
        
        # Fix 3: Convert root project file links to GitHub links
        # Pattern: ../../../FILE.md -> GitHub link
        pattern3 = r'\[([^\]]+)\]\(((?:\.\./){3,}([^\)]+)\.md)\)'
        matches = re.findall(pattern3, content)
        if matches:
            for text, old_path, filename in matches:
                new_url = f"https://github.com/sducournau/IGN_LIDAR_HD_DATASET/blob/main/{filename}.md"
                old = f"[{text}]({old_path})"
                new = f"[{text}]({new_url})"
                content = content.replace(old, new)
                changes.append(f"  ✓ {old_path} → GitHub link")
        
        # Fix 4: Fix malformed GitHub URLs (double https://)
        pattern4 = r'https://([^/]+)/https://([^\)]+)'
        if re.search(pattern4, content):
            content = re.sub(pattern4, r'https://\2', content)
            changes.append(f"  ✓ Fixed malformed GitHub URLs")
        
        # Apply changes if not dry run
        if content != original_content and changes:
            if not self.dry_run:
                # Backup original
                self.backup_file(file_path)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Get relative path properly
            try:
                if self.docs_dir in file_path.parents:
                    rel_path = file_path.relative_to(self.docs_dir.parent)
                elif self.blog_dir in file_path.parents:
                    rel_path = file_path.relative_to(self.blog_dir.parent)
                else:
                    rel_path = file_path
            except ValueError:
                rel_path = file_path
            
            self.fixes_applied.append({
                'file': str(rel_path),
                'changes': changes
            })
            
            return len(changes)
        
        return 0
    
    def backup_file(self, file_path):
        """Backup file before modification."""
        # Determine relative path from docs or blog
        if self.docs_dir in file_path.parents:
            rel_path = file_path.relative_to(self.docs_dir.parent)
        elif self.blog_dir in file_path.parents:
            rel_path = file_path.relative_to(self.blog_dir.parent)
        else:
            rel_path = file_path.relative_to(Path.cwd())
        
        backup_path = self.backup_dir / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
    
    def fix_all(self, directories=None, dry_run=True):
        """Fix links in all markdown files."""
        self.dry_run = dry_run
        
        if directories is None:
            directories = [self.docs_dir, self.blog_dir]
        
        print(f"\n{'='*70}")
        print(f"🔧 {'DRY RUN - ' if dry_run else ''}FIXING BROKEN LINKS")
        print(f"{'='*70}\n")
        
        total_files = 0
        total_changes = 0
        
        for directory in directories:
            if not directory.exists():
                continue
            
            md_files = list(directory.rglob("*.md"))
            
            for md_file in md_files:
                changes = self.fix_file(md_file)
                if changes > 0:
                    total_files += 1
                    total_changes += changes
        
        return total_files, total_changes
    
    def print_report(self):
        """Print fix report."""
        if not self.fixes_applied:
            print("\n✅ No broken links found!")
            return
        
        print(f"\n{'='*70}")
        print(f"📊 LINK FIX REPORT")
        print(f"{'='*70}\n")
        
        for fix in self.fixes_applied:
            print(f"📄 {fix['file']}")
            for change in fix['changes']:
                print(change)
            print()
        
        print(f"{'='*70}")
        print(f"📁 Files modified: {len(self.fixes_applied)}")
        print(f"🔗 Total fixes: {sum(len(f['changes']) for f in self.fixes_applied)}")
        
        if not self.dry_run and self.backup_dir.exists():
            print(f"💾 Backups saved to: {self.backup_dir}")
        
        print(f"{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix broken links in Docusaurus documentation"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply fixes (default is dry-run mode)"
    )
    parser.add_argument(
        "--dir",
        help="Specific directory to fix (default: docs and blog)"
    )
    
    args = parser.parse_args()
    
    fixer = LinkFixer()
    
    directories = None
    if args.dir:
        directories = [Path(args.dir)]
    
    # Run fixer
    files_modified, total_changes = fixer.fix_all(
        directories=directories,
        dry_run=not args.apply
    )
    
    # Print report
    fixer.print_report()
    
    # Summary
    if args.apply:
        print("✅ Fixes applied successfully!")
    else:
        print("ℹ️  DRY RUN MODE - No files were modified")
        print("   Run with --apply to apply these fixes")
    
    return 0


if __name__ == "__main__":
    exit(main())
