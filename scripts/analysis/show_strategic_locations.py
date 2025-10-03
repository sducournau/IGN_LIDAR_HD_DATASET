#!/usr/bin/env python3
"""
Script pour afficher un résumé détaillé des localisations stratégiques.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importer directement le module sans passer par __init__
import importlib.util
spec = importlib.util.spec_from_file_location(
    "strategic_locations",
    Path(__file__).parent.parent / "ign_lidar" / "strategic_locations.py"
)
strategic_locations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategic_locations)
STRATEGIC_LOCATIONS = strategic_locations.STRATEGIC_LOCATIONS

def main():
    """Afficher les statistiques des localisations stratégiques."""
    
    # Statistiques par catégorie
    categories = {}
    for name, config in STRATEGIC_LOCATIONS.items():
        cat = config['category']
        if cat not in categories:
            categories[cat] = {
                'count': 0,
                'tiles': 0,
                'priority_1': 0,
                'locations': []
            }
        categories[cat]['count'] += 1
        categories[cat]['tiles'] += config['target_tiles']
        if config['priority'] == 1:
            categories[cat]['priority_1'] += 1
        categories[cat]['locations'].append(name)
    
    print('=' * 80)
    print('📊 RÉSUMÉ DES LOCALISATIONS STRATÉGIQUES - DATASET SEGMENTATION 3D')
    print('=' * 80)
    print()
    
    print(f'✅ TOTAL: {len(STRATEGIC_LOCATIONS)} localisations')
    print(f'🎯 TUILES CIBLES: {sum(c["tiles"] for c in categories.values())} tuiles')
    print()
    
    print('📋 PAR CATÉGORIE:')
    print('-' * 80)
    for cat in sorted(categories.keys()):
        stats = categories[cat]
        print(
            f'  {cat:35s} | {stats["count"]:3d} locs | '
            f'{stats["tiles"]:4d} tuiles | P1: {stats["priority_1"]:2d}'
        )
    
    print()
    print('🏛️  MONUMENTS & PATRIMOINE (LOD3):')
    print('-' * 80)
    for name, config in sorted(STRATEGIC_LOCATIONS.items()):
        if 'heritage' in config['category']:
            chars = ', '.join(config['characteristics'][:3])
            print(
                f'  - {name:30s} ({config["category"]:22s}) '
                f'[{config["target_tiles"]} tuiles]'
            )
            print(f'    Caractéristiques: {chars}')
    
    print()
    print('🏙️  DIVERSITÉ URBAINE:')
    print('-' * 80)
    urban_cats = [k for k in categories.keys() if 'urban' in k or 'suburban' in k]
    for cat in sorted(urban_cats):
        print(f'  - {cat:35s}: {categories[cat]["count"]:2d} localisations')
    
    print()
    print('🏔️  DIVERSITÉ GÉOGRAPHIQUE:')
    print('-' * 80)
    geo_cats = [
        k for k in categories.keys()
        if any(x in k for x in ['coastal', 'mountain', 'rural', 'traditional'])
    ]
    for cat in sorted(geo_cats):
        print(f'  - {cat:35s}: {categories[cat]["count"]:2d} localisations')
    
    print()
    print('🏗️  INFRASTRUCTURES:')
    print('-' * 80)
    infra_cats = [k for k in categories.keys() if 'infrastructure' in k or 'campus' in k]
    for cat in sorted(infra_cats):
        print(f'  - {cat:35s}: {categories[cat]["count"]:2d} localisations')
    
    print()
    print('=' * 80)
    print('📍 COUVERTURE TERRITORIALE:')
    print('=' * 80)
    
    regions = {
        'Île-de-France': ['paris', 'versailles', 'saclay', 'orly', 'cdg'],
        'Auvergne-Rhône-Alpes': ['lyon', 'alpes', 'chamonix', 'megeve'],
        'Provence-Alpes-Côte d\'Azur': ['marseille', 'nice', 'cannes', 'avignon', 'provence'],
        'Grand Est': ['strasbourg', 'alsace', 'vosges'],
        'Bretagne': ['bretagne', 'saint_malo', 'rennes'],
        'Normandie': ['normandie', 'deauville'],
        'Occitanie': ['toulouse', 'montpellier', 'carcassonne', 'pyrenees'],
        'Nouvelle-Aquitaine': ['bordeaux', 'arcachon', 'biarritz', 'perigord'],
        'Hauts-de-France': ['lille'],
        'Pays de la Loire': ['nantes'],
        'Bourgogne-Franche-Comté': ['dijon', 'jura'],
    }
    
    for region, keywords in regions.items():
        count = sum(
            1 for name in STRATEGIC_LOCATIONS.keys()
            if any(kw in name for kw in keywords)
        )
        if count > 0:
            print(f'  {region:30s}: {count:2d} localisations')
    
    print()
    print('=' * 80)
    print('🎯 OBJECTIFS DU DATASET:')
    print('=' * 80)
    print('  ✓ Segmentation sémantique 3D de bâtiments')
    print('  ✓ Extraction LOD2 (toitures détaillées)')
    print('  ✓ Extraction LOD3 (façades + détails architecturaux)')
    print('  ✓ Diversité architecturale complète (France entière)')
    print('  ✓ Tous types de bâtiments:')
    print('    - Patrimoine historique (châteaux, monuments)')
    print('    - Urbain dense (centres-villes, Haussmann)')
    print('    - Urbain moderne (tours, gratte-ciels)')
    print('    - Banlieues (HLM, pavillons)')
    print('    - Rural (fermes, villages traditionnels)')
    print('    - Côtier (villas balnéaires)')
    print('    - Montagne (chalets, stations)')
    print('    - Infrastructures (aéroports, gares, ports)')
    print('=' * 80)


if __name__ == '__main__':
    main()
