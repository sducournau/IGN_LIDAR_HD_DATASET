#!/usr/bin/env python3
"""Check translation status of French documentation files."""

import re
from pathlib import Path

def check_translation_status(fr_file):
    """Check if a French file is actually translated or just a copy."""
    with open(fr_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for translation notice
    has_notice = "🇫🇷 TRADUCTION FRANÇAISE" in content or "TRADUCTION" in content
    
    # Count French-specific words vs English-specific words
    french_indicators = ['traitement', 'fichier', 'données', 'configuration', 'utilisation', 'exemple']
    english_indicators = ['processing', 'file', 'data', 'configuration', 'usage', 'example']
    
    french_count = sum(content.lower().count(word) for word in french_indicators)
    english_count = sum(content.lower().count(word) for word in english_indicators)
    
    # Simple heuristic: if English words dominate, it's not translated
    if english_count > french_count * 2:
        return "needs_translation"
    elif has_notice:
        return "partial_translation"  
    else:
        return "translated"

def main():
    fr_dir = Path("i18n/fr/docusaurus-plugin-content-docs/current")
    
    print("=== French Documentation Translation Status ===\n")
    
    status_counts = {
        "translated": 0,
        "partial_translation": 0,
        "needs_translation": 0
    }
    
    files_by_status = {
        "translated": [],
        "partial_translation": [],
        "needs_translation": []
    }
    
    for fr_file in sorted(fr_dir.rglob("*.md")):
        rel_path = str(fr_file.relative_to(fr_dir))
        status = check_translation_status(fr_file)
        status_counts[status] += 1
        files_by_status[status].append(rel_path)
    
    print(f"📊 Summary:")
    print(f"   ✅ Fully Translated: {status_counts['translated']}")
    print(f"   🔄 Partial/Needs Work: {status_counts['partial_translation']}")
    print(f"   ❌ Needs Translation: {status_counts['needs_translation']}")
    print(f"   📁 Total Files: {sum(status_counts.values())}\n")
    
    if files_by_status['needs_translation']:
        print("❌ Files Needing Translation:")
        for file in files_by_status['needs_translation']:
            print(f"   - {file}")
        print()
    
    if files_by_status['partial_translation']:
        print("🔄 Files with Translation Notices (Partial):")
        for file in files_by_status['partial_translation']:
            print(f"   - {file}")
        print()
    
    print("✅ Fully Translated Files:")
    for file in files_by_status['translated'][:10]:  # Show first 10
        print(f"   - {file}")
    if len(files_by_status['translated']) > 10:
        print(f"   ... and {len(files_by_status['translated']) - 10} more")

if __name__ == "__main__":
    main()
