#!/usr/bin/env python3
"""
Test rapide de la configuration
Vérifie que tous les modules sont bien installés
"""

import sys

def test_imports():
    """Test que tous les modules nécessaires sont disponibles."""
    print("🔍 Vérification des dépendances...")
    print()
    
    modules = {
        'laspy': 'Lecture/écriture LAZ',
        'numpy': 'Calculs numériques',
        'scipy': 'Features géométriques',
        'requests': 'Téléchargement',
        'tqdm': 'Barres de progression'
    }
    
    missing = []
    
    for module, description in modules.items():
        try:
            __import__(module)
            print(f"✅ {module:12} - {description}")
        except ImportError:
            print(f"❌ {module:12} - {description} (MANQUANT)")
            missing.append(module)
    
    print()
    
    if missing:
        print("⚠️  Modules manquants détectés!")
        print()
        print("Pour installer:")
        print(f"  pip install {' '.join(missing)}")
        print()
        print("Ou installer toutes les dépendances:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("✅ Toutes les dépendances sont installées!")
        return True


def test_ign_lidar():
    """Test que le module ign_lidar est accessible."""
    print()
    print("🔍 Vérification du module ign_lidar...")
    print()
    
    try:
        from ign_lidar import downloader, processor, strategic_locations
        print("✅ ign_lidar.downloader")
        print("✅ ign_lidar.processor")
        print("✅ ign_lidar.strategic_locations")
        print()
        
        # Compter les localisations urbaines
        urban_count = sum(
            1 for loc in strategic_locations.STRATEGIC_LOCATIONS.values()
            if 'urban' in loc.get('category', '')
        )
        print(f"📍 {urban_count} localisations urbaines disponibles")
        print()
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print()
        return False


def main():
    """Test principal."""
    print("=" * 70)
    print("🧪 TEST DE CONFIGURATION - TÉLÉCHARGEMENT TUILES URBAINES")
    print("=" * 70)
    print()
    
    # Test 1: Dépendances
    deps_ok = test_imports()
    
    # Test 2: Module ign_lidar
    module_ok = test_ign_lidar()
    
    # Résumé
    print("=" * 70)
    if deps_ok and module_ok:
        print("✅ CONFIGURATION OK - PRÊT À TÉLÉCHARGER!")
        print("=" * 70)
        print()
        print("🚀 Prochaine étape:")
        print()
        print("  # Lancer le téléchargement")
        print("  ./quick_download_urban.sh")
        print()
        print("  # Ou avec Python")
        print("  python download_urban_training_dataset.py")
        print()
        print("  # Ou tester avec l'exemple simple")
        print("  python example_urban_simple.py")
        print()
        return 0
    else:
        print("❌ CONFIGURATION INCOMPLÈTE")
        print("=" * 70)
        print()
        print("Veuillez corriger les erreurs ci-dessus avant de continuer.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
