#!/usr/bin/env python3
"""
Script de validation des localisations stratégiques améliorées.

Vérifie que les bbox ciblées contiennent bien des tuiles disponibles
et préférentiellement avec des bâtiments.

Usage:
    python scripts/validate_improved_locations.py
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.downloader import IGNLiDARDownloader
from ign_lidar.strategic_locations import STRATEGIC_LOCATIONS
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_improved_locations():
    """Valider les nouvelles localisations améliorées."""
    
    logger.info("=" * 80)
    logger.info("🔍 VALIDATION DES LOCALISATIONS AMÉLIORÉES")
    logger.info("=" * 80)
    
    # Initialiser le downloader
    output_dir = Path("temp_validation")
    downloader = IGNLiDARDownloader(output_dir=output_dir)
    
    # Catégories à vérifier en priorité (celles qui étaient problématiques)
    priority_categories = [
        "rural_traditional",
        "mountain_traditional", 
        "mountain_resort",
        "coastal_traditional"
    ]
    
    results = {
        "total": 0,
        "available": 0,
        "empty": 0,
        "by_category": {}
    }
    
    # Trier par catégorie prioritaire d'abord
    sorted_locations = sorted(
        STRATEGIC_LOCATIONS.items(),
        key=lambda x: (
            0 if x[1]['category'] in priority_categories else 1,
            x[0]
        )
    )
    
    for location_name, config in sorted_locations:
        category = config['category']
        
        if category not in results['by_category']:
            results['by_category'][category] = {
                'total': 0,
                'available': 0,
                'empty': 0,
                'locations': []
            }
        
        results['total'] += 1
        results['by_category'][category]['total'] += 1
        
        # Marquer si c'est une catégorie prioritaire
        priority_mark = "🎯" if category in priority_categories else "  "
        
        logger.info(f"\n{priority_mark} {location_name}")
        logger.info(f"   Catégorie: {category}")
        logger.info(f"   BBox: {config['bbox']}")
        
        # Calculer la taille de la bbox
        bbox = config['bbox']
        width_km = (bbox[2] - bbox[0]) * 111  # Approximation 1° ≈ 111 km
        height_km = (bbox[3] - bbox[1]) * 111
        area_km2 = width_km * height_km
        
        logger.info(f"   Taille: {width_km:.2f} × {height_km:.2f} km (~{area_km2:.2f} km²)")
        
        try:
            # Vérifier la disponibilité
            tiles_data = downloader.fetch_available_tiles(bbox=config['bbox'])
            tile_count = len(tiles_data.get('features', [])) if tiles_data else 0
            
            if tile_count > 0:
                results['available'] += 1
                results['by_category'][category]['available'] += 1
                results['by_category'][category]['locations'].append({
                    'name': location_name,
                    'tiles': tile_count,
                    'area_km2': area_km2
                })
                
                status = "✅ EXCELLENT" if tile_count >= config['target_tiles'] else "✅ OK"
                logger.info(f"   {status}: {tile_count} tuiles disponibles (target: {config['target_tiles']})")
                
                # Vérifier si la zone est bien ciblée (< 3 km²)
                if area_km2 > 3.0:
                    logger.warning(f"   ⚠️  Zone large ({area_km2:.2f} km²) - considérer réduction")
                elif area_km2 < 1.5:
                    logger.info(f"   👍 Zone bien ciblée ({area_km2:.2f} km²)")
                
            else:
                results['empty'] += 1
                results['by_category'][category]['empty'] += 1
                logger.warning(f"   ❌ VIDE: Aucune tuile disponible")
                
            time.sleep(0.5)  # Respecter l'API
            
        except Exception as e:
            logger.error(f"   ❌ ERREUR: {e}")
            results['empty'] += 1
            results['by_category'][category]['empty'] += 1
    
    # Résumé global
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ GLOBAL")
    logger.info("=" * 80)
    logger.info(f"Total localisations: {results['total']}")
    logger.info(f"✅ Disponibles: {results['available']} ({results['available']/results['total']*100:.1f}%)")
    logger.info(f"❌ Vides: {results['empty']} ({results['empty']/results['total']*100:.1f}%)")
    
    # Résumé par catégorie prioritaire
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ PAR CATÉGORIE PRIORITAIRE")
    logger.info("=" * 80)
    
    for category in priority_categories:
        if category in results['by_category']:
            cat_data = results['by_category'][category]
            total = cat_data['total']
            available = cat_data['available']
            empty = cat_data['empty']
            
            logger.info(f"\n🎯 {category}")
            logger.info(f"   Total: {total} localisations")
            logger.info(f"   ✅ Disponibles: {available} ({available/total*100:.1f}%)")
            logger.info(f"   ❌ Vides: {empty} ({empty/total*100:.1f}%)")
            
            if cat_data['locations']:
                logger.info(f"   📍 Top localisations:")
                sorted_locs = sorted(
                    cat_data['locations'],
                    key=lambda x: x['tiles'],
                    reverse=True
                )[:5]
                for loc in sorted_locs:
                    logger.info(
                        f"      • {loc['name']}: {loc['tiles']} tuiles "
                        f"({loc['area_km2']:.2f} km²)"
                    )
    
    # Recommandations
    logger.info("\n" + "=" * 80)
    logger.info("💡 RECOMMANDATIONS")
    logger.info("=" * 80)
    
    improvement_rate = results['available'] / results['total'] * 100
    
    if improvement_rate >= 90:
        logger.info("✅ EXCELLENT: Les localisations sont très bien ciblées!")
    elif improvement_rate >= 80:
        logger.info("✅ BON: La plupart des localisations sont valides")
        logger.info("   → Vérifier les localisations vides et ajuster les bbox si nécessaire")
    else:
        logger.warning("⚠️  ATTENTION: Beaucoup de localisations vides")
        logger.warning("   → Réviser les coordonnées des zones vides")
    
    # Sauvegarder les résultats
    import json
    results_file = Path(__file__).parent.parent / "data" / "validation" / "validation_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n📄 Résultats sauvegardés: {results_file}")
    
    return results


if __name__ == '__main__':
    try:
        results = validate_improved_locations()
        
        # Code de sortie basé sur le taux de succès
        success_rate = results['available'] / results['total']
        if success_rate >= 0.9:
            sys.exit(0)  # Excellent
        elif success_rate >= 0.8:
            sys.exit(0)  # Bon
        else:
            sys.exit(1)  # Nécessite amélioration
            
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Validation interrompue par l'utilisateur")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}", exc_info=True)
        sys.exit(1)
