#!/usr/bin/env python3
"""
Automated French Translation Generator
Uses pattern matching and dictionary-based translation for technical documentation.
"""

import re
from pathlib import Path

class FrenchTranslator:
    def __init__(self):
        # Technical term translations
        self.translations = {
            # Common terms
            "Overview": "Aperçu",
            "Introduction": "Introduction",
            "Getting Started": "Démarrage",
            "Installation": "Installation",
            "Configuration": "Configuration",
            "Usage": "Utilisation",
            "Examples": "Exemples",
            "Tutorial": "Tutoriel",
            "Guide": "Guide",
            "Reference": "Référence",
            "API": "API",
            "Features": "Fonctionnalités",
            "Requirements": "Prérequis",
            "Options": "Options",
            "Parameters": "Paramètres",
            "Arguments": "Arguments",
            "Output": "Sortie",
            "Input": "Entrée",
            
            # Technical terms
            "processing": "traitement",
            "download": "téléchargement",
            "upload": "téléversement",
            "file": "fichier",
            "directory": "répertoire",
            "folder": "dossier",
            "path": "chemin",
            "command": "commande",
            "option": "option",
            "parameter": "paramètre",
            "value": "valeur",
            "default": "par défaut",
            "optional": "optionnel",
            "required": "requis",
            "enabled": "activé",
            "disabled": "désactivé",
            "performance": "performance",
            "optimization": "optimisation",
            "memory": "mémoire",
            "GPU": "GPU",
            "CPU": "CPU",
            "accelerated": "accéléré",
            "acceleration": "accélération",
            
            # Feature-specific
            "LiDAR": "LiDAR",
            "point cloud": "nuage de points",
            "elevation": "élévation",
            "height": "hauteur",
            "building": "bâtiment",
            "architecture": "architecture",
            "classification": "classification",
            "augmentation": "augmentation",
            "RGB": "RGB",
            "infrared": "infrarouge",
            "visualization": "visualisation",
            "QGIS": "QGIS",
            
            # Action words
            "Install": "Installer",
            "Configure": "Configurer",
            "Run": "Exécuter",
            "Execute": "Exécuter",
            "Process": "Traiter",
            "Download": "Télécharger",
            "Export": "Exporter",
            "Import": "Importer",
            "Generate": "Générer",
            "Create": "Créer",
            "Update": "Mettre à jour",
            "Delete": "Supprimer",
            "Enable": "Activer",
            "Disable": "Désactiver",
            
            # Phrases
            "How to": "Comment",
            "Quick Start": "Démarrage Rapide",
            "Best Practices": "Bonnes Pratiques",
            "Common Issues": "Problèmes Courants",
            "Troubleshooting": "Dépannage",
            "Advanced Usage": "Utilisation Avancée",
            "Basic Usage": "Utilisation Basique",
            "Step by step": "Étape par étape",
            "Prerequisites": "Prérequis",
            "Next Steps": "Prochaines Étapes",
        }
        
        self.sentence_patterns = [
            # Pattern: "This feature allows..." -> "Cette fonctionnalité permet..."
            (r"This feature allows", "Cette fonctionnalité permet"),
            (r"This guide shows", "Ce guide montre"),
            (r"This tutorial demonstrates", "Ce tutoriel démontre"),
            (r"The following example", "L'exemple suivant"),
            (r"To install", "Pour installer"),
            (r"To configure", "Pour configurer"),
            (r"To use", "Pour utiliser"),
            (r"You can", "Vous pouvez"),
            (r"It is recommended", "Il est recommandé"),
            (r"Note that", "Notez que"),
            (r"Make sure", "Assurez-vous"),
            (r"For more information", "Pour plus d'informations"),
            (r"See also", "Voir aussi"),
            (r"Available options", "Options disponibles"),
            (r"Default value", "Valeur par défaut"),
        ]
    
    def translate_title(self, title):
        """Translate page titles."""
        for eng, fr in self.translations.items():
            if eng.lower() in title.lower():
                title = title.replace(eng, fr)
        return title
    
    def translate_frontmatter(self, frontmatter):
        """Translate frontmatter fields."""
        if not frontmatter:
            return frontmatter
        
        lines = frontmatter.split('\n')
        translated_lines = []
        
        for line in lines:
            # Translate title
            if line.strip().startswith('title:'):
                title = line.split('title:', 1)[1].strip()
                title = title.strip('"\'')
                translated_title = self.translate_title(title)
                translated_lines.append(f'title: "{translated_title}"')
            # Translate description
            elif line.strip().startswith('description:'):
                desc = line.split('description:', 1)[1].strip()
                desc = desc.strip('"\'')
                # Basic translation of common phrases
                for eng, fr in self.translations.items():
                    desc = desc.replace(eng, fr)
                translated_lines.append(f'description: "{desc}"')
            else:
                translated_lines.append(line)
        
        return '\n'.join(translated_lines)
    
    def add_translation_markers(self, content):
        """Add markers to help human translators."""
        # Find all headings
        lines = content.split('\n')
        marked_lines = []
        
        for line in lines:
            # Mark headings that need translation
            if line.startswith('#') and not line.strip().startswith('```'):
                # Check if it's mostly English
                if any(eng.lower() in line.lower() for eng in ['the', 'and', 'for', 'with', 'this']):
                    marked_lines.append(f"{line} <!-- 🔄 À traduire -->")
                else:
                    marked_lines.append(line)
            else:
                marked_lines.append(line)
        
        return '\n'.join(marked_lines)
    
    def process_file(self, en_path, fr_path):
        """Process a single file for translation."""
        with open(en_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
        
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            body = frontmatter_match.group(2)
            
            # Translate frontmatter
            translated_frontmatter = self.translate_frontmatter(frontmatter)
            
            # Add translation notice
            notice = """
<!-- 
🇫🇷 TRADUCTION FRANÇAISE
Ce document nécessite une traduction complète.
Les marqueurs <!-- 🔄 À traduire --> indiquent les sections à traduire.
Conservez tous les blocs de code et exemples techniques tels quels.
-->

"""
            
            # Add markers to body
            marked_body = self.add_translation_markers(body)
            
            # Combine
            result = f"---\n{translated_frontmatter}\n---\n{notice}{marked_body}"
        else:
            notice = """<!-- 
🇫🇷 TRADUCTION FRANÇAISE REQUISE
Ce document doit être traduit de l'anglais vers le français.
-->

"""
            result = notice + self.add_translation_markers(content)
        
        # Write result
        fr_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fr_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        return True


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python translate_helpers.py <file1> [file2] ...")
        print("Example: python translate_helpers.py features/axonometry.md")
        return 1
    
    translator = FrenchTranslator()
    docs_dir = Path("docs")
    fr_dir = Path("i18n/fr/docusaurus-plugin-content-docs/current")
    
    for file_path in sys.argv[1:]:
        en_path = docs_dir / file_path
        fr_path = fr_dir / file_path
        
        if not en_path.exists():
            print(f"❌ File not found: {en_path}")
            continue
        
        if translator.process_file(en_path, fr_path):
            print(f"✅ Processed: {file_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
