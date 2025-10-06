#!/usr/bin/env python3
"""
Comprehensive Documentation Update Script
Updates French documentation to match English structure and provides translation templates.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and report results."""
    print(f"\n{'='*70}")
    print(f"📋 {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️  {result.stderr}")
    return result.returncode == 0

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║        IGN LiDAR HD - Docusaurus French Documentation Update      ║
║                    Comprehensive Synchronization                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    # Change to website directory
    website_dir = Path(__file__).parent
    
    steps = [
        {
            "cmd": "python check_translations.py",
            "desc": "Step 1: Check current translation status"
        },
        {
            "cmd": "python sync_fr_docs.py",
            "desc": "Step 2: Synchronize French docs structure"
        },
        {
            "cmd": "python check_translations.py",
            "desc": "Step 3: Verify updated translation status"
        }
    ]
    
    for step in steps:
        if not run_command(step["cmd"], step["desc"]):
            print(f"\n❌ Failed at: {step['desc']}")
            return 1
    
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    print("""
✅ French documentation structure synchronized with English version
✅ All files updated with translation templates
✅ Backup created for modified files

📝 NEXT STEPS:
1. Review files marked with 🇫🇷 TRADUCTION FRANÇAISE
2. Translate titles, descriptions, and main content
3. Keep all code blocks, commands, and technical examples as-is
4. Use the translation markers <!-- 🔄 À traduire --> as guides

🔧 MANUAL TRANSLATION NEEDED FOR:
   - features/axonometry.md
   - features/format-preferences.md
   - features/lod3-classification.md
   - guides/auto-params.md
   - guides/performance.md
   - guides/visualization.md
   - mermaid-reference.md
   - reference/architectural-styles.md
   - reference/cli-download.md
   - reference/historical-analysis.md
   - release-notes/v1.6.2.md
   - release-notes/v1.7.1.md
   - tutorials/custom-features.md

📚 DOCUMENTATION:
   - English docs: website/docs/
   - French docs: website/i18n/fr/docusaurus-plugin-content-docs/current/
   - Backups: website/i18n/fr/backup/

🚀 TO BUILD AND TEST:
   cd website && npm run build
   cd website && npm start
""")
    
    return 0

if __name__ == "__main__":
    exit(main())
