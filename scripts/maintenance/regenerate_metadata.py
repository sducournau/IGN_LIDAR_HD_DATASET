#!/usr/bin/env python3
"""
Script simple pour régénérer les métadonnées JSON et stats.json
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import juste les localisations, pas tout le module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ign_lidar.strategic_locations import STRATEGIC_LOCATIONS


def create_tile_metadata(filename, location_name, category, characteristics,
                        description, architectural_style, bbox):
    """Crée les métadonnées d'une tuile."""
    return {
        "filename": filename,
        "downloaded_at": datetime.now().isoformat(),
        "location": {
            "name": location_name,
            "category": category
        },
        "characteristics": characteristics or [],
        "description": description,
        "architectural_style": architectural_style,
        "bbox": bbox
    }


def regenerate_metadata(directory):
    """Régénère les métadonnées pour un répertoire."""
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Répertoire introuvable: {directory}")
        return 0
    
    laz_files = list(directory.rglob("*.laz"))
    logger.info(f"📂 {len(laz_files)} tuiles LAZ trouvées dans {directory.name}")
    
    tiles_info = []
    metadata_count = 0
    
    for laz_file in laz_files:
        category = laz_file.parent.name
        location_name = None
        location_config = None
        
        # Trouver la config correspondante
        for loc_name, loc_cfg in STRATEGIC_LOCATIONS.items():
            if loc_cfg['category'] == category:
                location_name = loc_name
                location_config = loc_cfg
                break
        
        if location_config:
            # Créer le fichier JSON de métadonnées
            metadata = create_tile_metadata(
                filename=laz_file.name,
                location_name=location_name,
                category=category,
                characteristics=location_config.get('characteristics', []),
                description=location_config.get('description'),
                architectural_style=location_config.get('architectural_style'),
                bbox=location_config.get('bbox')
            )
            
            # Sauvegarder le JSON à côté de la tuile
            json_file = laz_file.parent / f"{laz_file.stem}.json"
            with open(json_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            metadata_count += 1
        
        tiles_info.append({
            "filename": laz_file.name,
            "category": category,
            "location": location_name or "Unknown"
        })
    
    # Créer stats.json
    stats = {
        "created_at": datetime.now().isoformat(),
        "type": directory.name,
        "tiles": {
            "total": len(laz_files)
        },
        "tiles_list": tiles_info
    }
    
    stats_file = directory / "stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✅ {metadata_count} métadonnées + stats.json créés")
    return metadata_count


def main():
    logger.info("=" * 80)
    logger.info("🔄 RÉGÉNÉRATION DES MÉTADONNÉES")
    logger.info("=" * 80)
    
    # Chemins
    raw_tiles = Path("/mnt/c/Users/Simon/ign/raw_tiles")
    pre_tiles = Path("/mnt/c/Users/Simon/ign/pre_tiles")
    
    # Régénérer
    logger.info("\n📁 RAW TILES")
    raw_count = regenerate_metadata(raw_tiles)
    
    logger.info("\n📁 PRE TILES")
    pre_count = regenerate_metadata(pre_tiles)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ TERMINÉ: {raw_count + pre_count} métadonnées créées")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
