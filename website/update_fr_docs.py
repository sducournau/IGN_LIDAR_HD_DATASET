#!/usr/bin/env python3
"""
Automated French documentation updater.
Copies English files to French directory with translation markers.
"""

import os
from pathlib import Path
import shutil
import re

# Translation mapping for common terms
TRANSLATION_MAP = {
    # Headers and common terms
    "Overview": "Vue d'ensemble",
    "Features": "Fonctionnalités",
    "Installation": "Installation",
    "Configuration": "Configuration",
    "Usage": "Utilisation",
    "Examples": "Exemples",
    "API Reference": "Référence API",
    "Getting Started": "Démarrage",
    "Quick Start": "Démarrage rapide",
    "Prerequisites": "Prérequis",
    "Requirements": "Exigences",
    "Parameters": "Paramètres",
    "Returns": "Retourne",
    "Example": "Exemple",
    "Note": "Note",
    "Warning": "Attention",
    "Important": "Important",
    "Tip": "Conseil",
    "See also": "Voir aussi",
    "Next steps": "Étapes suivantes",
    "Performance": "Performance",
    "Troubleshooting": "Dépannage",
    "Advanced": "Avancé",
    "Basic": "Basique",
    
    # Technical terms
    "Processing": "Traitement",
    "Feature extraction": "Extraction de caractéristiques",
    "GPU acceleration": "Accélération GPU",
    "Memory management": "Gestion de la mémoire",
    "Data augmentation": "Augmentation de données",
    "RGB integration": "Intégration RGB",
    "Point cloud": "Nuage de points",
    "Building detection": "Détection de bâtiments",
    "Classification": "Classification",
    "Workflow": "Flux de travail",
    "Pipeline": "Pipeline",
    "Output": "Sortie",
    "Input": "Entrée",
    "File": "Fichier",
    "Directory": "Répertoire",
    "Path": "Chemin",
    "Command": "Commande",
    "Option": "Option",
    "Flag": "Indicateur",
}

def add_translation_notice(content, file_path):
    """Add translation notice after frontmatter."""
    
    # Parse frontmatter
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    
    translation_notice = f"""
<!-- 
🇫🇷 VERSION FRANÇAISE - TRADUCTION REQUISE
Ce fichier provient de: {file_path}
Traduit automatiquement - nécessite une révision humaine.
Conservez tous les blocs de code, commandes et noms techniques identiques.
-->

"""
    
    if frontmatter_match:
        frontmatter = frontmatter_match.group(0)
        main_content = content[len(frontmatter):]
        return frontmatter + translation_notice + main_content
    else:
        return translation_notice + content

def auto_translate_simple_terms(content):
    """Auto-translate simple common terms in markdown headers and callouts."""
    
    # Translate markdown headers
    for en_term, fr_term in TRANSLATION_MAP.items():
        # Only translate in headers
        content = re.sub(
            rf'^(#+\s+){re.escape(en_term)}(\s|$)',
            rf'\1{fr_term}\2',
            content,
            flags=re.MULTILINE
        )
        
        # Translate in callout blocks (:::note, :::tip, etc.)
        content = re.sub(
            rf'(:::.*?\s+){re.escape(en_term)}(\s)',
            rf'\1{fr_term}\2',
            content,
            flags=re.MULTILINE
        )
    
    return content

def translate_frontmatter(content):
    """Translate common frontmatter fields."""
    
    # Extract and translate title
    content = re.sub(
        r'title:\s*["\']?Overview["\']?',
        'title: "Vue d\'ensemble"',
        content
    )
    content = re.sub(
        r'title:\s*["\']?Features["\']?',
        'title: "Fonctionnalités"',
        content
    )
    content = re.sub(
        r'title:\s*["\']?Getting Started["\']?',
        'title: "Démarrage"',
        content
    )
    content = re.sub(
        r'title:\s*["\']?Installation["\']?',
        'title: "Installation"',
        content
    )
    content = re.sub(
        r'title:\s*["\']?Configuration["\']?',
        'title: "Configuration"',
        content
    )
    
    # Translate description field
    content = re.sub(
        r'description:\s*["\']?Learn how to',
        'description: "Apprenez à',
        content
    )
    content = re.sub(
        r'description:\s*["\']?How to',
        'description: "Comment',
        content
    )
    
    return content

def copy_and_prepare_translation(en_file, fr_file):
    """Copy English file to French and prepare for translation."""
    
    # Read English content
    with open(en_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add translation notice
    rel_path = str(en_file.relative_to(Path("docs")))
    content = add_translation_notice(content, rel_path)
    
    # Auto-translate simple terms
    content = auto_translate_simple_terms(content)
    
    # Translate frontmatter
    content = translate_frontmatter(content)
    
    # Ensure directory exists
    fr_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write French file
    with open(fr_file, 'w', encoding='utf-8') as f:
        f.write(content)

def update_french_docs(files_to_update, force=False):
    """Update French documentation files."""
    
    en_dir = Path("docs")
    fr_dir = Path("i18n/fr/docusaurus-plugin-content-docs/current")
    
    print("🚀 Updating French documentation...\n")
    
    updated = 0
    skipped = 0
    errors = 0
    
    for file_path in files_to_update:
        en_file = en_dir / file_path
        fr_file = fr_dir / file_path
        
        if not en_file.exists():
            print(f"❌ English file not found: {file_path}")
            errors += 1
            continue
        
        if fr_file.exists() and not force:
            print(f"⏭️  Skipped (already exists): {file_path}")
            skipped += 1
            continue
        
        try:
            copy_and_prepare_translation(en_file, fr_file)
            print(f"✅ Updated: {file_path}")
            updated += 1
        except Exception as e:
            print(f"❌ Error updating {file_path}: {e}")
            errors += 1
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Updated: {updated}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   ❌ Errors: {errors}")
    print(f"   📁 Total: {len(files_to_update)}")

def main():
    import sys
    
    # Files that need translation based on analysis
    priority_files = [
        # High priority - core documentation
        "api/features.md",
        "api/gpu-api.md",
        "gpu/features.md",
        "gpu/overview.md",
        "gpu/rgb-augmentation.md",
        "workflows.md",
        
        # Medium priority - guides and features
        "guides/auto-params.md",
        "guides/performance.md",
        "features/format-preferences.md",
        "features/lod3-classification.md",
        "features/axonometry.md",
        
        # Lower priority - reference and release notes
        "reference/cli-download.md",
        "reference/architectural-styles.md",
        "reference/historical-analysis.md",
        "tutorials/custom-features.md",
        "mermaid-reference.md",
        "release-notes/v1.6.2.md",
        "release-notes/v1.7.1.md",
    ]
    
    force = "--force" in sys.argv
    
    print("=" * 80)
    print("📚 FRENCH DOCUMENTATION UPDATER")
    print("=" * 80)
    print()
    
    if force:
        print("⚠️  Force mode enabled - will overwrite existing files\n")
    
    update_french_docs(priority_files, force=force)
    
    print("\n" + "=" * 80)
    print("📝 NEXT STEPS:")
    print("=" * 80)
    print("1. Review each updated file in i18n/fr/docusaurus-plugin-content-docs/current/")
    print("2. Translate the content (keep code blocks unchanged)")
    print("3. Update frontmatter titles and descriptions")
    print("4. Remove translation notice once fully translated")
    print("5. Test with: npm run build")
    print()
    print("💡 Use --force flag to overwrite existing French files")

if __name__ == "__main__":
    main()
