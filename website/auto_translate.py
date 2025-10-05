#!/usr/bin/env python3
"""
Automated French translation marker generator for remaining documentation files.
Creates placeholder French files with key sections marked for translation.
"""

import os
from pathlib import Path
import re

def create_translation_template(en_file, fr_file):
    """Create a French translation template from English file."""
    
    with open(en_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse frontmatter
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if frontmatter_match:
        frontmatter = frontmatter_match.group(0)
        main_content = content[len(frontmatter):]
        
        # Translate common frontmatter terms
        frontmatter = frontmatter.replace('title:', 'title:')  # Keep for now
        frontmatter = frontmatter.replace('description:', 'description:')
    else:
        frontmatter = ""
        main_content = content
    
    # Add translation notice at the top
    translation_notice = """

<!-- 
🇫🇷 TRADUCTION FRANÇAISE
Ce fichier nécessite une traduction complète de l'anglais vers le français.
Conservez tous les blocs de code tels quels.
-->

"""
    
    # Create the French file
    fr_content = frontmatter + translation_notice + main_content
    
    # Ensure directory exists
    fr_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write French file
    with open(fr_file, 'w', encoding='utf-8') as f:
        f.write(fr_content)
    
    print(f"✅ Created: {fr_file}")

def main():
    docs_dir = Path("docs")
    fr_dir = Path("i18n/fr/docusaurus-plugin-content-docs/current")
    
    # List of files to translate (excluding already done: cli.md, custom-features.md)
    files_to_translate = [
        "api/configuration.md",
        "api/gpu-api.md",
        "features/axonometry.md",
        "guides/getting-started.md",
        "installation/gpu-setup.md",
        "reference/architectural-styles.md",
    ]
    
    print("🚀 Generating French translation templates...\n")
    
    for file_path in files_to_translate:
        en_file = docs_dir / file_path
        fr_file = fr_dir / file_path
        
        if en_file.exists():
            if not fr_file.exists():
                create_translation_template(en_file, fr_file)
            else:
                print(f"⏭️  Skipped (already exists): {fr_file}")
        else:
            print(f"❌ English file not found: {en_file}")
    
    print("\n✨ Translation templates generated!")
    print("\n📝 Next steps:")
    print("   1. Review each French file")
    print("   2. Translate text content (keep code blocks unchanged)")
    print("   3. Update frontmatter (title, description)")
    print("   4. Test with: npm run build")

if __name__ == "__main__":
    main()
