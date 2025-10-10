#!/usr/bin/env python3
"""
Quick translation helper for IGN LiDAR HD documentation
Applies consistent terminology and removes translation markers
"""

import sys
from pathlib import Path
import re

# Technical glossary for consistent translations
GLOSSARY = {
    "Point Cloud": "Nuage de Points",
    "point cloud": "nuage de points",
    "point clouds": "nuages de points",
    "Building": "Bâtiment",
    "building": "bâtiment",
    "buildings": "bâtiments",
    "GPU Acceleration": "Accélération GPU",
    "gpu acceleration": "accélération GPU",
    "Quick Start": "Démarrage Rapide",
    "quick start": "démarrage rapide",
    "Getting Started": "Premiers Pas",
    "getting started": "premiers pas",
    "Installation": "Installation",
    "installation": "installation",
    "Troubleshooting": "Dépannage",
    "troubleshooting": "dépannage",
    "Processing Pipeline": "Pipeline de Traitement",
    "processing pipeline": "pipeline de traitement",
    "Tile": "Dalle",
    "tile": "dalle",
    "tiles": "dalles",
    "Feature": "Caractéristique",
    "feature": "caractéristique",
    "features": "caractéristiques",
    "Classification": "Classification",
    "classification": "classification",
    "Neighborhood": "Voisinage",
    "neighborhood": "voisinage",
    "RGB Augmentation": "Augmentation RGB",
    "rgb augmentation": "augmentation RGB",
    "Preprocessing": "Prétraitement",
    "preprocessing": "prétraitement",
    "Download": "Téléchargement",
    "download": "téléchargement",
    "Upload": "Téléversement",
    "upload": "téléversement",
    "Workflow": "Flux de travail",
    "workflow": "flux de travail",
    "Dataset": "Jeu de données",
    "dataset": "jeu de données",
    "Output": "Sortie",
    "output": "sortie",
    "Input": "Entrée",
    "input": "entrée",
    "Configuration": "Configuration",
    "configuration": "configuration",
    "Parameter": "Paramètre",
    "parameter": "paramètre",
    "parameters": "paramètres",
    "Performance": "Performance",
    "performance": "performance",
    "Memory": "Mémoire",
    "memory": "mémoire",
    "Processing": "Traitement",
    "processing": "traitement",
    "Guide": "Guide",
    "guide": "guide",
}

def remove_translation_markers(content: str) -> str:
    """Remove French translation requirement markers"""
    # Remove the main marker
    content = content.replace('<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->\n', '')
    
    # Remove the explanation block
    pattern = r'<!-- Ce fichier est un modèle.*?-->\n'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    return content

def apply_glossary(text: str) -> str:
    """Apply glossary substitutions (outside code blocks)"""
    # Split by code blocks to avoid translating code
    parts = re.split(r'(```[\s\S]*?```|`[^`]+`)', text)
    
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Not a code block
            for en, fr in GLOSSARY.items():
                part = part.replace(en, fr)
        result.append(part)
    
    return ''.join(result)

def check_file(file_path: Path) -> dict:
    """Check file status"""
    if not file_path.exists():
        return {'exists': False}
    
    content = file_path.read_text(encoding='utf-8')
    needs_translation = '🇫🇷 TRADUCTION FRANÇAISE REQUISE' in content
    
    return {
        'exists': True,
        'needs_translation': needs_translation,
        'lines': len(content.split('\n')),
        'size': len(content)
    }

def prepare_file(file_path: Path, auto_apply_glossary: bool = False):
    """Prepare a file for translation"""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    content = file_path.read_text(encoding='utf-8')
    
    # Check if already prepared
    if '🇫🇷 TRADUCTION FRANÇAISE REQUISE' not in content:
        print(f"✅ Already prepared: {file_path.name}")
        return True
    
    # Remove markers
    content = remove_translation_markers(content)
    
    # Optionally apply glossary
    if auto_apply_glossary:
        print(f"🔄 Applying glossary to: {file_path.name}")
        content = apply_glossary(content)
    
    # Save
    file_path.write_text(content, encoding='utf-8')
    print(f"✅ Prepared: {file_path.name}")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 prepare_translation.py <file.md>          # Remove markers only")
        print("  python3 prepare_translation.py <file.md> --auto   # Remove markers + apply glossary")
        print("  python3 prepare_translation.py --check <file.md>  # Check file status")
        sys.exit(1)
    
    if sys.argv[1] == '--check':
        file_path = Path(sys.argv[2])
        info = check_file(file_path)
        if info['exists']:
            status = "⏳ Needs translation" if info['needs_translation'] else "✅ Translated"
            print(f"{status} | {info['lines']} lines | {file_path.name}")
        else:
            print(f"❌ File not found: {file_path}")
    else:
        file_path = Path(sys.argv[1])
        auto_apply = '--auto' in sys.argv
        prepare_file(file_path, auto_apply_glossary=auto_apply)

if __name__ == '__main__':
    main()
