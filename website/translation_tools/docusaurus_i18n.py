#!/usr/bin/env python3
"""
Docusaurus i18n Translation Tool
=================================

Consolidated tool for managing Docusaurus French documentation translation.
Combines functionality from multiple legacy scripts into a single, clean interface.

Usage:
    python docusaurus_i18n.py sync              # Synchronize EN → FR structure
    python docusaurus_i18n.py status            # Check translation status
    python docusaurus_i18n.py validate          # Validate all links
    python docusaurus_i18n.py fix-links         # Fix broken links automatically
    python docusaurus_i18n.py report            # Generate comprehensive report
    python docusaurus_i18n.py all               # Run complete workflow

Author: GitHub Copilot
Date: October 6, 2025
"""

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import sys


class DocusaurusI18N:
    """Main class for Docusaurus i18n operations."""
    
    def __init__(self, website_root: Optional[Path] = None):
        """Initialize with website root directory."""
        self.website_root = website_root or Path(__file__).parent.parent
        self.en_docs_path = self.website_root / "docs"
        self.fr_docs_path = self.website_root / "i18n" / "fr" / "docusaurus-plugin-content-docs" / "current"
        self.backup_dir = self.website_root / "i18n" / "fr" / "backup"
        
        # Technical glossary for consistent translations
        self.glossary = {
            "Quick Start": "Démarrage Rapide",
            "Getting Started": "Premiers Pas",
            "Installation": "Installation",
            "Configuration": "Configuration",
            "Usage": "Utilisation",
            "Examples": "Exemples",
            "API Reference": "Référence API",
            "Tutorial": "Tutoriel",
            "Guide": "Guide",
            "Features": "Fonctionnalités",
            "Advanced": "Avancé",
            "Performance": "Performance",
            "Troubleshooting": "Dépannage",
            "Building": "Bâtiment",
            "Point Cloud": "Nuage de Points",
            "LiDAR": "LiDAR",
            "GPU": "GPU",
        }
        
    # ============================================================
    # 1. SYNCHRONIZATION
    # ============================================================
    
    def sync_structure(self, create_backups: bool = True) -> Dict[str, int]:
        """
        Synchronize French documentation structure with English.
        
        Args:
            create_backups: Create timestamped backups before changes
            
        Returns:
            Dict with statistics: {'created': N, 'updated': N, 'backed_up': N}
        """
        print("🔄 Synchronizing FR documentation structure with EN...")
        
        stats = {'created': 0, 'updated': 0, 'backed_up': 0}
        
        # Create backup if requested
        if create_backups:
            backup_path = self._create_backup()
            if backup_path:
                stats['backed_up'] = len(list(backup_path.glob("**/*.md")))
                print(f"✅ Created backup: {backup_path}")
        
        # Walk through English docs
        for en_file in self.en_docs_path.rglob("*.md"):
            # Get relative path
            rel_path = en_file.relative_to(self.en_docs_path)
            fr_file = self.fr_docs_path / rel_path
            
            # Skip if French file exists and is translated
            if fr_file.exists() and self._is_translated(fr_file):
                continue
            
            # Create parent directories
            fr_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create or update template
            if not fr_file.exists():
                self._create_translation_template(en_file, fr_file)
                stats['created'] += 1
                print(f"✨ Created: {rel_path}")
            else:
                self._update_translation_template(en_file, fr_file)
                stats['updated'] += 1
                print(f"🔄 Updated: {rel_path}")
        
        print(f"\n📊 Sync complete: {stats['created']} created, {stats['updated']} updated")
        return stats
    
    def _create_backup(self) -> Optional[Path]:
        """Create timestamped backup of French docs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / timestamp
        
        if self.fr_docs_path.exists():
            shutil.copytree(self.fr_docs_path, backup_path, dirs_exist_ok=True)
            return backup_path
        return None
    
    def _create_translation_template(self, en_file: Path, fr_file: Path) -> None:
        """Create French translation template from English file."""
        content = en_file.read_text(encoding='utf-8')
        
        # Extract and translate frontmatter
        frontmatter, body = self._extract_frontmatter(content)
        if frontmatter:
            frontmatter = self._translate_frontmatter(frontmatter)
        
        # Add translation notice
        notice = """
<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->
<!-- Ce fichier est un modèle qui nécessite une traduction manuelle. -->
<!-- Veuillez traduire le contenu ci-dessous en conservant : -->
<!-- - Le frontmatter (métadonnées en haut) -->
<!-- - Les blocs de code (traduire uniquement les commentaires) -->
<!-- - Les liens et chemins de fichiers -->
<!-- - La structure Markdown -->

"""
        
        # Combine and write
        if frontmatter:
            new_content = frontmatter + "\n" + notice + body
        else:
            new_content = notice + body
        
        fr_file.write_text(new_content, encoding='utf-8')
    
    def _update_translation_template(self, en_file: Path, fr_file: Path) -> None:
        """Update existing French template if needed."""
        # For now, just ensure it has the translation marker
        content = fr_file.read_text(encoding='utf-8')
        if "🇫🇷 TRADUCTION FRANÇAISE" not in content:
            self._create_translation_template(en_file, fr_file)
    
    # ============================================================
    # 2. STATUS CHECKING
    # ============================================================
    
    def check_status(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Check translation status of all French documentation.
        
        Args:
            detailed: Include per-file details
            
        Returns:
            Dict with status information
        """
        print("📊 Checking translation status...\n")
        
        status = {
            'total_files': 0,
            'translated': 0,
            'needs_translation': 0,
            'missing': 0,
            'files': []
        }
        
        # Check all English files
        for en_file in self.en_docs_path.rglob("*.md"):
            rel_path = en_file.relative_to(self.en_docs_path)
            fr_file = self.fr_docs_path / rel_path
            
            status['total_files'] += 1
            
            file_status = {
                'path': str(rel_path),
                'status': 'unknown',
                'size': en_file.stat().st_size
            }
            
            if not fr_file.exists():
                file_status['status'] = 'missing'
                status['missing'] += 1
            elif self._is_translated(fr_file):
                file_status['status'] = 'translated'
                status['translated'] += 1
            else:
                file_status['status'] = 'needs_translation'
                status['needs_translation'] += 1
            
            if detailed:
                status['files'].append(file_status)
        
        # Print summary
        total = status['total_files']
        translated = status['translated']
        percentage = (translated / total * 100) if total > 0 else 0
        
        print(f"📁 Total files: {total}")
        print(f"✅ Translated: {translated} ({percentage:.1f}%)")
        print(f"⏳ Needs translation: {status['needs_translation']}")
        print(f"❌ Missing: {status['missing']}")
        
        if status['needs_translation'] > 0:
            print(f"\n📝 Files needing translation:")
            for file_info in status['files']:
                if file_info['status'] == 'needs_translation':
                    print(f"   • {file_info['path']}")
        
        return status
    
    def _is_translated(self, fr_file: Path) -> bool:
        """Check if a French file is translated (heuristic)."""
        try:
            content = fr_file.read_text(encoding='utf-8')
            
            # Check for translation markers
            if "🇫🇷 TRADUCTION FRANÇAISE REQUISE" in content:
                return False
            if "TRADUCTION FRANÇAISE" in content:
                return False
            
            # Heuristic: count French vs English words
            french_indicators = ['le', 'la', 'les', 'de', 'du', 'des', 'à', 'au', 'aux', 
                               'ce', 'cette', 'ces', 'pour', 'avec', 'dans', 'sur']
            english_indicators = ['the', 'this', 'that', 'these', 'those', 'for', 'with', 
                                'in', 'on', 'at', 'to', 'from']
            
            words = content.lower().split()
            french_count = sum(1 for word in words if word in french_indicators)
            english_count = sum(1 for word in words if word in english_indicators)
            
            # Consider translated if more French than English indicators
            return french_count > english_count
            
        except Exception:
            return False
    
    # ============================================================
    # 3. LINK VALIDATION & FIXING
    # ============================================================
    
    def validate_links(self, fix: bool = False) -> Dict[str, Any]:
        """
        Validate (and optionally fix) links in all documentation.
        
        Args:
            fix: Automatically fix broken links
            
        Returns:
            Dict with validation results
        """
        print(f"🔗 {'Validating and fixing' if fix else 'Validating'} links...\n")
        
        results = {
            'total_files': 0,
            'total_links': 0,
            'broken_links': 0,
            'fixed_links': 0,
            'issues': []
        }
        
        # Patterns to fix
        patterns = [
            (r'\]\(/docs/', r'](/', 'Remove /docs/ prefix'),
            (r'\]\(([^h][^t][^t][^p][^:][^/][^/].*?)\.md\)', r'](\1)', 'Remove .md extension'),
        ]
        
        # Check all markdown files
        for doc_dir in [self.en_docs_path, self.fr_docs_path]:
            if not doc_dir.exists():
                continue
                
            for md_file in doc_dir.rglob("*.md"):
                results['total_files'] += 1
                content = md_file.read_text(encoding='utf-8')
                original_content = content
                
                # Find all links
                links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
                results['total_links'] += len(links)
                
                # Apply fixes if requested
                if fix:
                    for pattern, replacement, description in patterns:
                        matches = len(re.findall(pattern, content))
                        if matches > 0:
                            content = re.sub(pattern, replacement, content)
                            results['fixed_links'] += matches
                    
                    # Write back if changed
                    if content != original_content:
                        # Create backup
                        backup_file = md_file.with_suffix('.md.bak')
                        shutil.copy2(md_file, backup_file)
                        
                        md_file.write_text(content, encoding='utf-8')
                        rel_path = md_file.relative_to(self.website_root)
                        print(f"🔧 Fixed links in: {rel_path}")
                
                # Check for broken links
                for text, url in links:
                    if url.startswith('/docs/') or url.endswith('.md'):
                        results['broken_links'] += 1
                        if not fix:
                            results['issues'].append({
                                'file': str(md_file.relative_to(self.website_root)),
                                'text': text,
                                'url': url,
                                'fix': 'Remove /docs/ prefix or .md extension'
                            })
        
        # Print summary
        print(f"\n📊 Validation results:")
        print(f"   Files checked: {results['total_files']}")
        print(f"   Total links: {results['total_links']}")
        print(f"   Broken links: {results['broken_links']}")
        if fix:
            print(f"   Fixed links: {results['fixed_links']}")
        
        return results
    
    # ============================================================
    # 4. REPORTING
    # ============================================================
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate comprehensive status report."""
        print("📄 Generating comprehensive report...\n")
        
        # Gather all information
        status = self.check_status(detailed=True)
        links = self.validate_links(fix=False)
        
        # Build report
        report_lines = [
            "=" * 70,
            "DOCUSAURUS FRENCH TRANSLATION STATUS REPORT",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "TRANSLATION STATUS",
            "-" * 70,
            f"Total documentation files: {status['total_files']}",
            f"Translated: {status['translated']} ({status['translated']/status['total_files']*100:.1f}%)",
            f"Needs translation: {status['needs_translation']}",
            f"Missing: {status['missing']}",
            "",
            "LINK VALIDATION",
            "-" * 70,
            f"Total files checked: {links['total_files']}",
            f"Total links found: {links['total_links']}",
            f"Broken links: {links['broken_links']}",
            "",
        ]
        
        if status['needs_translation'] > 0:
            report_lines.extend([
                "FILES NEEDING TRANSLATION",
                "-" * 70,
            ])
            for file_info in status['files']:
                if file_info['status'] == 'needs_translation':
                    report_lines.append(f"  • {file_info['path']}")
            report_lines.append("")
        
        if links['issues']:
            report_lines.extend([
                "LINK ISSUES (Top 10)",
                "-" * 70,
            ])
            for issue in links['issues'][:10]:
                report_lines.append(f"  File: {issue['file']}")
                report_lines.append(f"    Link: [{issue['text']}]({issue['url']})")
                report_lines.append(f"    Fix: {issue['fix']}")
                report_lines.append("")
        
        report_lines.append("=" * 70)
        
        report = "\n".join(report_lines)
        
        # Save if output file specified
        if output_file:
            output_file = Path(output_file)
            output_file.write_text(report, encoding='utf-8')
            print(f"✅ Report saved to: {output_file}")
        
        print(report)
        return report
    
    # ============================================================
    # 5. HELPER METHODS
    # ============================================================
    
    def _extract_frontmatter(self, content: str) -> Tuple[Optional[str], str]:
        """Extract frontmatter from markdown content."""
        if not content.startswith('---'):
            return None, content
        
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter = f"---{parts[1]}---"
            body = parts[2]
            return frontmatter, body
        
        return None, content
    
    def _translate_frontmatter(self, frontmatter: str) -> str:
        """Translate common frontmatter fields."""
        for en, fr in self.glossary.items():
            # Translate title and description
            frontmatter = re.sub(
                f'(title|description):\\s*["\']?{re.escape(en)}["\']?',
                f'\\1: "{fr}"',
                frontmatter,
                flags=re.IGNORECASE
            )
        return frontmatter


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Docusaurus i18n Translation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sync              # Synchronize structure
  %(prog)s status            # Check translation status
  %(prog)s validate          # Validate links
  %(prog)s fix-links         # Fix broken links
  %(prog)s report            # Generate report
  %(prog)s all               # Run complete workflow
        """
    )
    
    parser.add_argument(
        'command',
        choices=['sync', 'status', 'validate', 'fix-links', 'report', 'all'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backups (use with caution)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed information'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for reports'
    )
    
    args = parser.parse_args()
    
    # Initialize tool
    tool = DocusaurusI18N()
    
    # Execute command
    try:
        if args.command == 'sync':
            tool.sync_structure(create_backups=not args.no_backup)
            
        elif args.command == 'status':
            tool.check_status(detailed=args.detailed)
            
        elif args.command == 'validate':
            tool.validate_links(fix=False)
            
        elif args.command == 'fix-links':
            tool.validate_links(fix=True)
            
        elif args.command == 'report':
            output = args.output or Path('translation_report.txt')
            tool.generate_report(output_file=output)
            
        elif args.command == 'all':
            print("🚀 Running complete workflow...\n")
            tool.sync_structure(create_backups=not args.no_backup)
            print("\n")
            tool.check_status(detailed=args.detailed)
            print("\n")
            tool.validate_links(fix=True)
            print("\n")
            output = args.output or Path('translation_report.txt')
            tool.generate_report(output_file=output)
            print("\n✅ Complete workflow finished!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
