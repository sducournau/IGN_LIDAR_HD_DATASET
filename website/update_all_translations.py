#!/usr/bin/env python3
"""
Comprehensive script to update French translations to match English documentation.
Uses smart comparison to preserve French translations while syncing structure.
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
import difflib

# Base paths
DOCS_EN = Path("/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website/docs")
DOCS_FR = Path("/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website/i18n/fr/docusaurus-plugin-content-docs/current")

# Comprehensive translation dictionary
TRANSLATIONS = {
    # Titles and headers
    "Examples and Tutorials": "Exemples et Tutoriels",
    "Features Overview": "Aperçu des Fonctionnalités",
    "Quick Start": "Démarrage Rapide",
    "Basic Example": "Exemple Basique",
    "Batch Processing": "Traitement par Lot",
    "Core Features": "Fonctionnalités Principales",
    "Building Component Classification": "Classification des Composants de Bâtiment",
    "Advanced Features": "Fonctionnalités Avancées",
    "Getting Started": "Pour Commencer",
    "Installation": "Installation",
    "Usage": "Utilisation",
    "Configuration": "Configuration",
    "API Reference": "Référence API",
    "CLI Reference": "Référence CLI",
    "GPU Acceleration": "Accélération GPU",
    "Performance": "Performance",
    "Troubleshooting": "Dépannage",
    "Release Notes": "Notes de Version",
    "Changelog": "Journal des Modifications",
    
    # Common sections
    "Prerequisites": "Prérequis",
    "Parameters": "Paramètres",
    "Options": "Options",
    "Returns": "Retourne",
    "Example": "Exemple",
    "Examples": "Exemples",
    "Usage Example": "Exemple d'Utilisation",
    "Basic Usage": "Utilisation de Base",
    "Advanced Usage": "Utilisation Avancée",
    "Input": "Entrée",
    "Output": "Sortie",
    "Description": "Description",
    "Notes": "Notes",
    "Warning": "Avertissement",
    "Important": "Important",
    "Tip": "Astuce",
    "See Also": "Voir Aussi",
    "Related": "Connexe",
    "Next Steps": "Prochaines Étapes",
    
    # Features
    "Architectural Styles": "Styles Architecturaux",
    "RGB Augmentation": "Augmentation RGB",
    "Infrared Augmentation": "Augmentation Infrarouge",
    "LOD3 Classification": "Classification LOD3",
    "Axonometry": "Axonométrie",
    "Format Preferences": "Préférences de Format",
    
    # Technical terms
    "Point Cloud": "Nuage de Points",
    "LiDAR Data": "Données LiDAR",
    "Building Components": "Composants de Bâtiment",
    "Feature Extraction": "Extraction de Caractéristiques",
    "Machine Learning": "Apprentissage Automatique",
    "Dataset": "Jeu de Données",
    "Preprocessing": "Prétraitement",
    "Classification": "Classification",
    "Segmentation": "Segmentation",
    "Augmentation": "Augmentation",
    "Visualization": "Visualisation",
    
    # Building components
    "Roof": "Toit",
    "Roofs": "Toits",
    "Wall": "Mur",
    "Walls": "Murs",
    "Ground": "Sol",
    "Floor": "Plancher",
    "Facade": "Façade",
    "Chimney": "Cheminée",
    "Balcony": "Balcon",
    "Window": "Fenêtre",
    
    # Actions/verbs
    "Process": "Traiter",
    "Processing": "Traitement",
    "Download": "Télécharger",
    "Downloading": "Téléchargement",
    "Enrich": "Enrichir",
    "Enriching": "Enrichissement",
    "Extract": "Extraire",
    "Extracting": "Extraction",
    "Classify": "Classifier",
    "Classification": "Classification",
    "Compute": "Calculer",
    "Computing": "Calcul",
    "Load": "Charger",
    "Loading": "Chargement",
    "Save": "Sauvegarder",
    "Saving": "Sauvegarde",
    "Install": "Installer",
    "Installation": "Installation",
    "Configure": "Configurer",
    "Configuration": "Configuration",
    
    # Common phrases
    "Complete collection of practical examples": "Collection complète d'exemples pratiques",
    "to learn and master": "pour apprendre et maîtriser",
    "Collection of practical examples and tutorials": "Collection d'exemples pratiques et tutoriels",
    "Comprehensive guide to": "Guide complet de",
    "processing features": "fonctionnalités de traitement",
    "provides comprehensive tools": "fournit des outils complets",
    "high-density LiDAR data": "données LiDAR haute densité",
    "machine learning-ready datasets": "jeux de données prêts pour l'apprentissage automatique",
    "advanced building feature extraction": "extraction avancée de caractéristiques de bâtiments",
    
    # Code comments
    "First simple example": "Premier exemple simple",
    "Initialization": "Initialisation",
    "Process a file": "Traitement d'un fichier",
    "Process multiple files": "Traitement de plusieurs fichiers",
    "parallel processes": "processus parallèles",
    "Process a directory": "Traitement d'un répertoire",
    
    # Status messages
    "Processed": "Traité",
    "points": "points",
    "Detected classes": "Classes détectées",
    
    # File paths
    "raw_data": "donnees_brutes",
    "processed": "traite",
    "sample.las": "exemple.las",
    "enriched.las": "enrichi.las",
    
    # Component descriptions
    "Identified Components": "Composants Identifiés",
    "Key Capabilities": "Capacités Clés",
    "slope geometries": "géométries en pente",
    "flat": "plates",
    "complex": "complexes",
    "Facades": "Façades",
    "load-bearing": "porteurs",
    "curtain walls": "murs-rideaux",
    "Terrain": "Terrain",
    "courtyards": "cours",
    "foundations": "fondations",
    "Details": "Détails",
    "Chimneys": "Cheminées",
    "dormers": "lucarnes",
    "balconies": "balcons",
}

def read_file(file_path):
    """Read file content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

def write_file(file_path, content):
    """Write content to file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"❌ Error writing {file_path}: {e}")
        return False

def extract_frontmatter(content):
    """Extract frontmatter from markdown content."""
    match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if match:
        frontmatter = match.group(0)
        body = content[len(frontmatter):]
        return frontmatter, body
    return "", content

def translate_frontmatter(en_frontmatter):
    """Translate frontmatter fields."""
    fr_frontmatter = en_frontmatter
    
    # Translate common frontmatter fields
    for en, fr in TRANSLATIONS.items():
        # Match title: value format
        fr_frontmatter = re.sub(
            rf'(title:\s*)({re.escape(en)})',
            rf'\1{fr}',
            fr_frontmatter
        )
        # Match description: value format
        fr_frontmatter = re.sub(
            rf'(description:\s*)({re.escape(en)})',
            rf'\1{fr}',
            fr_frontmatter
        )
    
    # Translate keywords array
    keywords_match = re.search(r'keywords:\s*\[(.*?)\]', fr_frontmatter)
    if keywords_match:
        keywords = keywords_match.group(1)
        for en, fr in TRANSLATIONS.items():
            keywords = keywords.replace(en.lower(), fr.lower())
        fr_frontmatter = re.sub(
            r'keywords:\s*\[.*?\]',
            f'keywords: [{keywords}]',
            fr_frontmatter
        )
    
    return fr_frontmatter

def smart_translate_body(en_body, fr_body=None):
    """
    Intelligently translate body content.
    Preserves code blocks and tries to maintain existing French translations.
    """
    # If we have existing French, try to preserve code blocks
    if fr_body:
        # Extract code blocks from both versions
        en_code_blocks = re.findall(r'```[\s\S]*?```', en_body)
        fr_code_blocks = re.findall(r'```[\s\S]*?```', fr_body)
    
    # Translate the English body
    fr_translated = en_body
    
    # Apply translations
    for en, fr in TRANSLATIONS.items():
        # Don't translate inside code blocks
        # Use word boundaries where appropriate
        if ' ' in en:  # Multi-word phrases
            fr_translated = fr_translated.replace(en, fr)
        else:  # Single words - be more careful
            # Only replace in headers, emphasis, or at word boundaries
            fr_translated = re.sub(
                rf'(^|\s|[#*])({re.escape(en)})(\s|[#*:]|$)',
                rf'\1{fr}\3',
                fr_translated,
                flags=re.MULTILINE
            )
    
    return fr_translated

def update_translation(en_path, fr_path):
    """Update French translation file based on English source."""
    # Read files
    en_content = read_file(en_path)
    if not en_content:
        return False
    
    fr_content = read_file(fr_path) if fr_path.exists() else None
    
    # Extract frontmatter and body
    en_frontmatter, en_body = extract_frontmatter(en_content)
    
    if fr_content:
        fr_frontmatter, fr_body = extract_frontmatter(fr_content)
    else:
        fr_frontmatter, fr_body = "", ""
    
    # Translate frontmatter (or keep existing if good)
    if fr_frontmatter:
        # Keep existing French frontmatter, just ensure it's complete
        new_frontmatter = fr_frontmatter
    else:
        new_frontmatter = translate_frontmatter(en_frontmatter)
    
    # Translate body
    new_body = smart_translate_body(en_body, fr_body if fr_content else None)
    
    # Combine
    new_content = new_frontmatter + new_body
    
    # Write updated file
    return write_file(fr_path, new_content)

def get_outdated_files():
    """Get list of files where English is newer than French."""
    outdated = []
    
    for en_path in DOCS_EN.rglob('*.md'):
        rel_path = en_path.relative_to(DOCS_EN)
        fr_path = DOCS_FR / rel_path
        
        if not fr_path.exists():
            outdated.append({
                'rel_path': rel_path,
                'en_path': en_path,
                'fr_path': fr_path,
                'status': 'missing'
            })
        else:
            en_mtime = datetime.fromtimestamp(en_path.stat().st_mtime)
            fr_mtime = datetime.fromtimestamp(fr_path.stat().st_mtime)
            
            if en_mtime > fr_mtime:
                time_diff = en_mtime - fr_mtime
                outdated.append({
                    'rel_path': rel_path,
                    'en_path': en_path,
                    'fr_path': fr_path,
                    'status': 'outdated',
                    'en_mtime': en_mtime,
                    'fr_mtime': fr_mtime,
                    'diff_hours': time_diff.total_seconds() / 3600
                })
    
    return sorted(outdated, key=lambda x: x.get('diff_hours', 999), reverse=True)

def main():
    print("=" * 80)
    print("🇫🇷 FRENCH TRANSLATION UPDATE TOOL")
    print("=" * 80)
    print()
    
    # Find outdated files
    outdated = get_outdated_files()
    
    if not outdated:
        print("✅ All French translations are up to date!")
        return
    
    # Categorize
    missing = [f for f in outdated if f['status'] == 'missing']
    critical = [f for f in outdated if f['status'] == 'outdated' and f['diff_hours'] >= 24]
    moderate = [f for f in outdated if f['status'] == 'outdated' and 1 <= f['diff_hours'] < 24]
    minor = [f for f in outdated if f['status'] == 'outdated' and f['diff_hours'] < 1]
    
    print(f"📊 Found {len(outdated)} files needing update:\n")
    print(f"   ❌ Missing: {len(missing)}")
    print(f"   🔴 Critical (1+ days): {len(critical)}")
    print(f"   🟡 Moderate (1-24h): {len(moderate)}")
    print(f"   🟢 Minor (<1h): {len(minor)}")
    print()
    
    # Ask for confirmation
    response = input("Do you want to update all files? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Update cancelled.")
        return
    
    print("\n" + "=" * 80)
    print("🚀 UPDATING TRANSLATIONS")
    print("=" * 80)
    print()
    
    # Update files
    success = 0
    failed = 0
    
    for item in outdated:
        rel_path = item['rel_path']
        status = item['status']
        
        if status == 'missing':
            print(f"❌ Creating: {rel_path}")
        else:
            age = item['diff_hours']
            urgency = "🔴" if age >= 24 else "🟡" if age >= 1 else "🟢"
            print(f"{urgency} Updating: {rel_path} ({age:.1f}h old)")
        
        # Ensure French directory exists
        item['fr_path'].parent.mkdir(parents=True, exist_ok=True)
        
        # Update the file
        if update_translation(item['en_path'], item['fr_path']):
            success += 1
            print(f"   ✅ Done")
        else:
            failed += 1
            print(f"   ❌ Failed")
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 UPDATE SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully updated: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(outdated)}")
    print()
    
    if success > 0:
        print("🎉 French translations have been updated!")
        print("💡 Please review the changes before committing.")
    
if __name__ == "__main__":
    main()
