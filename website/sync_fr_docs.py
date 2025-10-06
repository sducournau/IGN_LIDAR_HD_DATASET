#!/usr/bin/env python3
"""
Comprehensive French Documentation Synchronizer
Syncs French documentation with English version and provides translation templates.
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime

class DocusaurusFrenchSync:
    def __init__(self):
        self.docs_dir = Path("docs")
        self.fr_dir = Path("i18n/fr/docusaurus-plugin-content-docs/current")
        self.backup_dir = Path("i18n/fr/backup") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats = {
            "created": [],
            "updated": [],
            "skipped": [],
            "needs_translation": []
        }
    
    def extract_frontmatter(self, content):
        """Extract frontmatter from markdown content."""
        frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
        if frontmatter_match:
            return frontmatter_match.group(1), frontmatter_match.group(2)
        return None, content
    
    def translate_frontmatter(self, frontmatter):
        """Translate frontmatter titles and descriptions to French placeholders."""
        if not frontmatter:
            return frontmatter
        
        # Add comment about translation needed
        translated = "# 🇫🇷 Traduisez les champs title et description ci-dessous\n" + frontmatter
        return translated
    
    def check_if_translated(self, content):
        """Check if content is actually translated or just English copy."""
        # Common English technical terms that should be translated
        english_indicators = [
            'processing', 'download', 'installation', 'usage', 'example',
            'configuration', 'setup', 'tutorial', 'guide', 'feature'
        ]
        
        # Common French translations
        french_indicators = [
            'traitement', 'téléchargement', 'installation', 'utilisation', 'exemple',
            'configuration', 'paramétrage', 'tutoriel', 'guide', 'fonctionnalité'
        ]
        
        content_lower = content.lower()
        english_count = sum(1 for word in english_indicators if word in content_lower)
        french_count = sum(1 for word in french_indicators if word in content_lower)
        
        # If more English words than French, needs translation
        if english_count > french_count * 1.5:
            return False
        
        # Check for translation markers
        if "TRADUCTION" in content or "À TRADUIRE" in content:
            return False
            
        return True
    
    def create_translation_template(self, en_path, fr_path, force=False):
        """Create or update French translation template."""
        # Read English content
        with open(en_path, 'r', encoding='utf-8') as f:
            en_content = f.read()
        
        # Check if French file exists
        fr_exists = fr_path.exists()
        
        if fr_exists and not force:
            # Check if it needs translation
            with open(fr_path, 'r', encoding='utf-8') as f:
                fr_content = f.read()
            
            if self.check_if_translated(fr_content):
                self.stats["skipped"].append(str(fr_path.relative_to(self.fr_dir)))
                return False
            else:
                # Backup existing file
                self.backup_file(fr_path)
                self.stats["needs_translation"].append(str(fr_path.relative_to(self.fr_dir)))
        
        # Extract frontmatter
        en_frontmatter, en_body = self.extract_frontmatter(en_content)
        
        # Build French template
        translation_notice = """
<!-- 
🇫🇷 TRADUCTION FRANÇAISE REQUISE
Ce document doit être traduit de l'anglais vers le français.
Veuillez traduire les titres, descriptions et texte principal.
Conservez tous les blocs de code, commandes et exemples techniques tels quels.
-->

"""
        
        if en_frontmatter:
            fr_frontmatter = self.translate_frontmatter(en_frontmatter)
            fr_content = f"---\n{fr_frontmatter}\n---\n{translation_notice}{en_body}"
        else:
            fr_content = translation_notice + en_content
        
        # Ensure directory exists
        fr_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write French file
        with open(fr_path, 'w', encoding='utf-8') as f:
            f.write(fr_content)
        
        if fr_exists:
            self.stats["updated"].append(str(fr_path.relative_to(self.fr_dir)))
            return "updated"
        else:
            self.stats["created"].append(str(fr_path.relative_to(self.fr_dir)))
            return "created"
    
    def backup_file(self, fr_path):
        """Backup French file before updating."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        rel_path = fr_path.relative_to(self.fr_dir)
        backup_path = self.backup_dir / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fr_path, backup_path)
    
    def sync_all(self, force=False):
        """Sync all English docs to French."""
        print("🔄 Starting French Documentation Sync...\n")
        
        # Find all English markdown files
        en_files = sorted(self.docs_dir.rglob("*.md"))
        
        # Skip archive directory
        en_files = [f for f in en_files if "archive" not in f.parts]
        
        for en_file in en_files:
            rel_path = en_file.relative_to(self.docs_dir)
            fr_file = self.fr_dir / rel_path
            
            result = self.create_translation_template(en_file, fr_file, force)
            
            if result == "created":
                print(f"✅ Created: {rel_path}")
            elif result == "updated":
                print(f"🔄 Updated: {rel_path}")
    
    def print_report(self):
        """Print summary report."""
        print("\n" + "="*70)
        print("📊 FRENCH DOCUMENTATION SYNC REPORT")
        print("="*70)
        
        print(f"\n✅ Created: {len(self.stats['created'])} files")
        for f in self.stats['created']:
            print(f"   - {f}")
        
        print(f"\n🔄 Updated: {len(self.stats['updated'])} files")
        for f in self.stats['updated']:
            print(f"   - {f}")
        
        print(f"\n⚠️  Needs Translation: {len(self.stats['needs_translation'])} files")
        for f in self.stats['needs_translation']:
            print(f"   - {f}")
        
        print(f"\n⏭️  Skipped (already translated): {len(self.stats['skipped'])} files")
        
        print(f"\n📁 Total processed: {sum(len(v) for v in self.stats.values())} files")
        
        if self.backup_dir.exists():
            print(f"\n💾 Backups saved to: {self.backup_dir}")
        
        print("\n" + "="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sync French documentation with English version"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update all files, even if already translated"
    )
    parser.add_argument(
        "--file",
        help="Sync specific file only (relative path from docs/)"
    )
    
    args = parser.parse_args()
    
    syncer = DocusaurusFrenchSync()
    
    if args.file:
        # Sync single file
        en_file = syncer.docs_dir / args.file
        fr_file = syncer.fr_dir / args.file
        
        if not en_file.exists():
            print(f"❌ English file not found: {en_file}")
            return 1
        
        result = syncer.create_translation_template(en_file, fr_file, args.force)
        if result:
            print(f"✅ Synced: {args.file}")
        else:
            print(f"⏭️  Skipped: {args.file} (already translated)")
    else:
        # Sync all files
        syncer.sync_all(args.force)
        syncer.print_report()
    
    return 0

if __name__ == "__main__":
    exit(main())
