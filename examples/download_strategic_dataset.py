#!/usr/bin/env python3
"""
Script de téléchargement rapide de tuiles stratégiques pour dataset de segmentation 3D.

Usage:
    python download_strategic_dataset.py --max-tiles 100 --output raw_tiles/
    python download_strategic_dataset.py --category heritage_palace --output monuments/
    python download_strategic_dataset.py --priority 1 --output priority_tiles/
"""

import argparse
import logging
import sys
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description='Télécharger des tuiles LIDAR HD depuis localisations stratégiques'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('raw_tiles'),
        help='Répertoire de sortie (défaut: raw_tiles/)'
    )
    
    parser.add_argument(
        '--max-tiles',
        type=int,
        default=60,
        help='Nombre maximum de tuiles à télécharger (défaut: 60)'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        help='Télécharger uniquement une catégorie spécifique'
    )
    
    parser.add_argument(
        '--priority',
        type=int,
        choices=[1, 2],
        help='Filtrer par priorité (1 ou 2)'
    )
    
    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='Afficher la liste des catégories disponibles'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Valider la disponibilité sans télécharger'
    )
    
    args = parser.parse_args()
    
    # Import après parsing pour éviter erreurs d'import si --help
    try:
        from ign_lidar.downloader import IGNLiDARDownloader
        from ign_lidar.strategic_locations import (
            STRATEGIC_LOCATIONS,
            validate_locations_via_wfs,
            download_diverse_tiles,
            get_categories
        )
    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        logger.error("Assurez-vous d'avoir installé les dépendances:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Liste des catégories
    if args.list_categories:
        categories = get_categories()
        print("\n📋 CATÉGORIES DISPONIBLES:")
        print("=" * 60)
        for cat in categories:
            locs = [
                name for name, cfg in STRATEGIC_LOCATIONS.items()
                if cfg['category'] == cat
            ]
            tiles = sum(
                cfg['target_tiles']
                for name, cfg in STRATEGIC_LOCATIONS.items()
                if cfg['category'] == cat
            )
            print(f"  {cat:35s} | {len(locs):2d} locs | {tiles:3d} tuiles")
        print("=" * 60)
        return
    
    # Créer le downloader
    logger.info("🚀 Initialisation du téléchargeur IGN LIDAR HD")
    downloader = IGNLiDARDownloader()
    
    # Filtrer les localisations si nécessaire
    locations_to_check = STRATEGIC_LOCATIONS.copy()
    
    if args.category:
        locations_to_check = {
            name: cfg for name, cfg in locations_to_check.items()
            if cfg['category'] == args.category
        }
        logger.info(f"📍 Filtrage par catégorie: {args.category}")
        logger.info(f"   {len(locations_to_check)} localisations sélectionnées")
    
    if args.priority:
        locations_to_check = {
            name: cfg for name, cfg in locations_to_check.items()
            if cfg['priority'] == args.priority
        }
        logger.info(f"🎯 Filtrage par priorité: {args.priority}")
        logger.info(f"   {len(locations_to_check)} localisations sélectionnées")
    
    if not locations_to_check:
        logger.error("❌ Aucune localisation ne correspond aux filtres")
        return
    
    # Validation
    logger.info(f"\n🔍 Validation de {len(locations_to_check)} localisations via WFS IGN")
    
    # Créer une version temporaire de STRATEGIC_LOCATIONS pour la validation
    import ign_lidar.strategic_locations as sl_module
    original_locations = sl_module.STRATEGIC_LOCATIONS
    sl_module.STRATEGIC_LOCATIONS = locations_to_check
    
    try:
        validation_results = validate_locations_via_wfs(downloader)
    finally:
        sl_module.STRATEGIC_LOCATIONS = original_locations
    
    # Stats validation
    valid_count = sum(
        1 for r in validation_results.values()
        if r['available'] and r['tile_count'] > 0
    )
    total_tiles = sum(
        r['tile_count'] for r in validation_results.values()
        if r['available']
    )
    
    logger.info(f"\n✅ Validation terminée:")
    logger.info(f"   {valid_count}/{len(validation_results)} localisations valides")
    logger.info(f"   {total_tiles} tuiles disponibles au total")
    
    if args.validate_only:
        logger.info("\n✓ Mode validation uniquement, arrêt.")
        return
    
    # Téléchargement
    if valid_count == 0:
        logger.warning("❌ Aucune tuile disponible pour téléchargement")
        return
    
    logger.info(f"\n📥 Téléchargement de {args.max_tiles} tuiles maximum")
    logger.info(f"   Destination: {args.output}")
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    downloaded = download_diverse_tiles(
        downloader,
        validation_results,
        args.output,
        max_total_tiles=args.max_tiles
    )
    
    logger.info(f"\n✅ TERMINÉ!")
    logger.info(f"   {len(downloaded)} fichiers LAZ téléchargés")
    logger.info(f"   Répertoire: {args.output}")
    
    # Statistiques par catégorie
    cat_stats = {}
    for laz_file in downloaded:
        # La catégorie est dans le nom du parent directory
        category = laz_file.parent.name
        if category not in cat_stats:
            cat_stats[category] = []
        cat_stats[category].append(laz_file)
    
    logger.info(f"\n📊 Répartition par catégorie:")
    for cat in sorted(cat_stats.keys()):
        logger.info(f"   {cat:35s}: {len(cat_stats[cat]):3d} tuiles")


if __name__ == '__main__':
    main()
