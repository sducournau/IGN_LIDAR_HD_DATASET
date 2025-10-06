#!/usr/bin/env python3
"""
Script to update French translations to match English documentation structure
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Translation dictionary for common terms
TRANSLATIONS = {
    # Headers and titles
    "Basic Usage": "Utilisation de base",
    "Overview": "Vue d'ensemble",
    "Download": "Téléchargement",
    "Enrich": "Enrichissement",
    "Process": "Traitement",
    "Parameters": "Paramètres",
    "Output": "Sortie",
    "Example": "Exemple",
    "Complete Workflow": "Workflow complet",
    "Next Steps": "Prochaines étapes",
    "Troubleshooting": "Dépannage",
    "Classification Levels": "Niveaux de classification",
    "Data Loading": "Chargement des données",
    "Memory Considerations": "Considérations sur la mémoire",
    "Smart Skip Detection": "Détection intelligente de saut",
    
    # Common phrases
    "Learn the essential workflows": "Apprenez les workflows essentiels",
    "for processing": "pour traiter",
    "machine learning-ready datasets": "jeux de données prêts pour l'apprentissage automatique",
    "Get LiDAR tiles from IGN servers": "Obtenir les tuiles LiDAR depuis les serveurs IGN",
    "Add building component features to points": "Ajouter des caractéristiques de composants de bâtiment aux points",
    "Extract patches for machine learning": "Extraire des patches pour l'apprentissage automatique",
    "Download tiles for Paris center": "Télécharger les tuiles pour le centre de Paris",
    "Bounding box as": "Boîte englobante au format",
    "Directory to save": "Répertoire pour sauvegarder",
    "Maximum number of tiles": "Nombre maximum de tuiles",
    "Number of parallel workers": "Nombre de workers parallèles",
    "optional": "optionnel",
    "Each point now has": "Chaque point dispose maintenant de",
    "geometric features": "caractéristiques géométriques",
    "for building component classification": "pour la classification des composants de bâtiment",
    
    # Technical terms
    "Point Cloud": "Nuage de points",
    "Geometric Features": "Caractéristiques géométriques",
    "Building Components": "Composants de bâtiment",
    "Classification": "Classification",
    "Patches": "Patches",
    "Raw Data": "Données brutes",
    "Enriched Data": "Données enrichies",
    "ML Dataset": "Jeu de données ML",
    "Training Patches": "Patches d'entraînement",
    "Labels": "Labels",
    
    # File paths
    "/path/to/": "/chemin/vers/",
    "raw_tiles": "tuiles_brutes",
    "enriched_tiles": "tuiles_enrichies",
    "patches": "patches",
    
    # Step labels
    "Step": "Étape",
    "Input": "Entrée",
    "Web Service": "Service Web",
    "Query WFS Service": "Requête service WFS",
    "Download LAZ Tiles": "Téléchargement tuiles LAZ",
    "Validate Files": "Validation fichiers",
    "Load Point Cloud": "Chargement nuage de points",
    "Compute Geometric Features": "Calcul caractéristiques géométriques",
    "Classify Building Components": "Classification composants bâtiment",
    "Save Enriched LAZ": "Sauvegarde LAZ enrichi",
    "Extract Patches": "Extraction patches",
    "Apply Augmentations": "Application augmentations",
    "Assign LOD Labels": "Attribution labels LOD",
    "Save NPZ Files": "Sauvegarde fichiers NPZ",
    "ML-Ready Dataset": "Jeu de données ML",
    "NPZ Patches": "Patches NPZ",
}

def translate_text(text: str) -> str:
    """Translate English text to French using translation dictionary"""
    result = text
    for en, fr in TRANSLATIONS.items():
        result = result.replace(en, fr)
    return result

def update_file(en_file: Path, fr_file: Path) -> bool:
    """Update French file to match English file structure"""
    try:
        # Read English content
        with open(en_file, 'r', encoding='utf-8') as f:
            en_content = f.read()
        
        # Read current French content for frontmatter
        with open(fr_file, 'r', encoding='utf-8') as f:
            fr_content = f.read()
        
        # Extract French frontmatter
        fr_frontmatter_match = re.match(r'^---\n(.*?)\n---', fr_content, re.DOTALL)
        if fr_frontmatter_match:
            fr_frontmatter = fr_frontmatter_match.group(0)
        else:
            # Extract from English and translate
            en_frontmatter_match = re.match(r'^---\n(.*?)\n---', en_content, re.DOTALL)
            if en_frontmatter_match:
                fr_frontmatter = translate_text(en_frontmatter_match.group(0))
            else:
                fr_frontmatter = ""
        
        # Remove frontmatter from English content
        en_body = re.sub(r'^---\n.*?\n---\n\n?', '', en_content, flags=re.DOTALL)
        
        # Translate English body
        fr_body = translate_text(en_body)
        
        # Combine frontmatter and body
        if fr_frontmatter:
            new_content = f"{fr_frontmatter}\n\n{fr_body}"
        else:
            new_content = fr_body
        
        # Write updated content
        with open(fr_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True
    except Exception as e:
        print(f"Error updating {fr_file}: {e}")
        return False

def main():
    """Main function to update translations"""
    en_dir = Path('docs')
    fr_dir = Path('i18n/fr/docusaurus-plugin-content-docs/current')
    
    # Find files needing updates from the JSON report
    import json
    with open('translation_update_needed.json', 'r') as f:
        needs_update = json.load(f)
    
    print(f"Updating {len(needs_update)} files...")
    print("=" * 80)
    
    updated = 0
    failed = 0
    
    for item in needs_update:
        rel_path = item['file']
        en_file = en_dir / rel_path
        fr_file = fr_dir / rel_path
        
        if not en_file.exists():
            print(f"⚠️  English file not found: {rel_path}")
            failed += 1
            continue
        
        if not fr_file.exists():
            print(f"⚠️  French file not found: {rel_path}")
            failed += 1
            continue
        
        print(f"Updating: {rel_path}")
        if update_file(en_file, fr_file):
            updated += 1
            print(f"  ✅ Updated")
        else:
            failed += 1
            print(f"  ❌ Failed")
    
    print("=" * 80)
    print(f"\nResults:")
    print(f"  ✅ Updated: {updated}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Total: {len(needs_update)}")

if __name__ == "__main__":
    main()
