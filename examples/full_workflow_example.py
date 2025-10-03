#!/usr/bin/env python3
"""
Script de téléchargement et préparation de tuiles urbaines pour LOD2/LOD3
Workflow complet :
1. Téléchargement tuiles urbaines (liste stratégique)
2. Calcul features géométriques  
3. Sauvegarde LAZ enrichi intermédiaire
4. Création patches 150m² pour entraînement
"""

import argparse
import logging
from pathlib import Path
import json
import time
from typing import List, Dict, Any

from ign_lidar.downloader import IGNLiDARDownloader
from ign_lidar.processor import LiDARProcessor
from ign_lidar.strategic_locations import STRATEGIC_LOCATIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_urban_tiles(num_tiles: int = 30) -> List[Dict[str, Any]]:
    """
    Récupère une liste de tuiles urbaines stratégiques.
    
    Args:
        num_tiles: Nombre de tuiles à sélectionner
        
    Returns:
        Liste de dictionnaires avec infos sur les tuiles
    """
    urban_locations = []
    
    # Filtrer les localisations urbaines (urban_dense et urban_modern)
    for location_name, location_data in STRATEGIC_LOCATIONS.items():
        category = location_data.get('category', '')
        if 'urban' in category:
            urban_locations.append({
                'name': location_name,
                'bbox': location_data['bbox'],
                'category': category,
                'priority': location_data.get('priority', 2),
                'target_tiles': location_data.get('target_tiles', 2)
            })
    
    # Trier par priorité
    urban_locations.sort(key=lambda x: x['priority'])
    
    logger.info(f"🏙️  Trouvé {len(urban_locations)} localisations urbaines stratégiques")
    
    return urban_locations[:num_tiles]


def download_urban_tiles(
    output_dir: Path,
    num_locations: int = 10,
    max_tiles_per_location: int = 3
) -> List[Path]:
    """
    Télécharge des tuiles urbaines.
    
    Args:
        output_dir: Répertoire de sortie
        num_locations: Nombre de localisations à utiliser
        max_tiles_per_location: Nombre max de tuiles par localisation
        
    Returns:
        Liste des fichiers LAZ téléchargés
    """
    tiles_dir = output_dir / "raw_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Récupérer les localisations urbaines
    urban_locations = get_urban_tiles(num_locations)
    
    downloaded_files = []
    
    logger.info("="*70)
    logger.info("📥 TÉLÉCHARGEMENT DES TUILES URBAINES")
    logger.info("="*70)
    
    # Initialiser le downloader
    downloader = IGNLiDARDownloader(tiles_dir)
    
    for idx, location in enumerate(urban_locations, 1):
        loc_name = location['name']
        loc_cat = location['category']
        logger.info(
            f"\n[{idx}/{len(urban_locations)}] {loc_name} ({loc_cat})"
        )
        logger.info(f"  BBox: {location['bbox']}")
        
        # Créer sous-dossier pour la catégorie
        category_dir = tiles_dir / location['category']
        category_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Récupérer les tuiles disponibles dans la bbox
            bbox_wgs84 = location['bbox']
            tiles_data = downloader.fetch_available_tiles(bbox_wgs84)
            
            features = tiles_data.get('features', [])[:max_tiles_per_location]
            
            if not features:
                logger.warning("  ⚠️  Aucune tuile trouvée")
                continue
                
            # Télécharger chaque tuile
            for feature in features:
                props = feature.get('properties', {})
                tile_name = props.get('name', '')
                
                if tile_name:
                    # Télécharger dans le dossier catégorie
                    downloader.output_dir = category_dir
                    if downloader.download_tile(tile_name):
                        tile_path = category_dir / tile_name
                        downloaded_files.append(tile_path)
            
            logger.info(f"  ✓ Téléchargé {len(features)} tuiles")
                
        except Exception as e:
            logger.error(f"  ❌ Erreur: {e}")
            continue
        
        time.sleep(1)  # Pause entre localisations
    
    logger.info("\n" + "="*70)
    logger.info(f"✓ Téléchargement terminé: {len(downloaded_files)} tuiles")
    logger.info("="*70)
    
    return downloaded_files


def enrich_laz_files(
    laz_files: List[Path],
    output_dir: Path,
    k_neighbors: int = 20
) -> List[Path]:
    """
    Enrichit les fichiers LAZ avec features géométriques et sauvegarde.
    
    Note: Dans cette version simplifiée, on copie simplement les LAZ
    dans le dossier enriched_laz. Le calcul des features se fait
    lors de la création des patches.
    
    Args:
        laz_files: Liste des fichiers LAZ à enrichir
        output_dir: Répertoire de sortie pour LAZ enrichis
        k_neighbors: Nombre de voisins pour calcul features
        
    Returns:
        Liste des fichiers LAZ enrichis (copiés)
    """
    enriched_dir = output_dir / "enriched_laz"
    enriched_dir.mkdir(parents=True, exist_ok=True)
    
    enriched_files = []
    
    logger.info("\n" + "="*70)
    logger.info("🔧 PRÉPARATION DES LAZ POUR ENRICHISSEMENT")
    logger.info("="*70)
    logger.info(f"K-neighbors: {k_neighbors}")
    logger.info(
        "Features seront calculées lors de la création des patches"
    )
    logger.info("")
    
    import shutil
    
    for idx, laz_file in enumerate(laz_files, 1):
        logger.info(f"[{idx}/{len(laz_files)}] {laz_file.name}")
        
        try:
            # Créer sous-dossier par catégorie
            category = laz_file.parent.name
            category_dir = enriched_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Copier le fichier
            enriched_path = category_dir / laz_file.name
            shutil.copy2(laz_file, enriched_path)
            
            enriched_files.append(enriched_path)
            logger.info(f"  ✓ Copié: {enriched_path.name}")
            
        except Exception as e:
            logger.error(f"  ❌ Erreur: {e}")
            continue
    
    logger.info("\n" + "="*70)
    msg = f"✓ Préparation terminée: {len(enriched_files)} LAZ prêts"
    logger.info(msg)
    logger.info("="*70)
    
    return enriched_files


def create_training_patches(
    enriched_laz_files: List[Path],
    output_dir: Path,
    lod_level: str = "LOD2",
    patch_size: float = 150.0,
    no_rgb: bool = True,
    augment: bool = True,
    num_augmentations: int = 3
) -> int:
    """
    Crée les patches d'entraînement depuis les LAZ enrichis.
    
    Args:
        enriched_laz_files: Liste des LAZ enrichis
        output_dir: Répertoire de sortie
        lod_level: Niveau LOD (LOD2 ou LOD3)
        patch_size: Taille des patches en mètres
        no_rgb: Si True, pas de couleur RGB (recommandé)
        augment: Activer l'augmentation de données
        num_augmentations: Nombre d'augmentations par patch
        
    Returns:
        Nombre total de patches créés
    """
    patches_dir = output_dir / f"patches_{lod_level.lower()}"
    patches_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info(f"🎯 CRÉATION DES PATCHES D'ENTRAÎNEMENT {lod_level}")
    logger.info("="*70)
    area_ha = patch_size**2/10000
    logger.info(
        f"Taille patch: {patch_size}m × {patch_size}m (≈ {area_ha:.1f} ha)"
    )
    logger.info(f"RGB: {'Non' if no_rgb else 'Oui'}")
    logger.info(f"Augmentation: {'Oui' if augment else 'Non'}")
    if augment:
        logger.info(f"Nombre d'augmentations: {num_augmentations}")
    logger.info("")
    
    # Initialiser le processor
    processor = LiDARProcessor(
        lod_level=lod_level,
        patch_size=patch_size,
        augment=augment,
        num_augmentations=num_augmentations
        # Note: RGB n'est pas utilisé dans cette version
    )
    
    total_patches = 0
    
    for idx, laz_file in enumerate(enriched_laz_files, 1):
        logger.info(f"[{idx}/{len(enriched_laz_files)}] {laz_file.name}")
        
        try:
            # Créer sous-dossier par catégorie
            category = laz_file.parent.name
            category_output = patches_dir / "train" / category
            category_output.mkdir(parents=True, exist_ok=True)
            
            # Traiter la tuile
            num_patches = processor.process_tile(
                str(laz_file),
                str(category_output)
            )
            
            total_patches += num_patches
            logger.info(f"  ✓ Créé {num_patches} patches")
            
        except Exception as e:
            logger.error(f"  ❌ Erreur: {e}")
            continue
    
    logger.info("\n" + "="*70)
    logger.info(f"✓ Création patches terminée: {total_patches:,} patches")
    logger.info(f"  Localisation: {patches_dir}")
    logger.info("="*70)
    
    return total_patches


def save_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    stats: Dict[str, Any]
):
    """Sauvegarde les métadonnées du workflow."""
    metadata = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': config,
        'statistics': stats
    }
    
    metadata_file = output_dir / "workflow_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📄 Métadonnées sauvegardées: {metadata_file}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Télécharger tuiles urbaines et créer dataset LOD2/LOD3"
    )
    
    # Paramètres de sortie
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('urban_training_dataset'),
        help='Répertoire de sortie (défaut: urban_training_dataset)'
    )
    
    # Paramètres de téléchargement
    parser.add_argument(
        '--num-locations',
        type=int,
        default=10,
        help='Nombre de localisations urbaines (défaut: 10)'
    )
    parser.add_argument(
        '--tiles-per-location',
        type=int,
        default=3,
        help='Tuiles max par localisation (défaut: 3)'
    )
    
    # Paramètres d'enrichissement
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=20,
        help='Voisins pour calcul features (défaut: 20)'
    )
    
    # Paramètres de patches
    parser.add_argument(
        '--lod-level',
        type=str,
        choices=['LOD2', 'LOD3'],
        default='LOD2',
        help='Niveau de détail (défaut: LOD2)'
    )
    parser.add_argument(
        '--patch-size',
        type=float,
        default=150.0,
        help='Taille des patches en mètres (défaut: 150.0)'
    )
    parser.add_argument(
        '--no-rgb',
        action='store_true',
        help='Ne pas inclure les couleurs RGB'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        default=True,
        help='Activer l\'augmentation de données'
    )
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=3,
        help='Nombre d\'augmentations par patch (défaut: 3)'
    )
    
    # Workflow partiel
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip le téléchargement (utiliser LAZ existants)'
    )
    parser.add_argument(
        '--skip-enrichment',
        action='store_true',
        help='Skip l\'enrichissement (utiliser LAZ enrichis existants)'
    )
    parser.add_argument(
        '--skip-patches',
        action='store_true',
        help='Skip la création de patches'
    )
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("🚀 WORKFLOW: TUILES URBAINES → PATCHES LOD2/LOD3")
    logger.info("="*70)
    logger.info(f"Répertoire de sortie: {args.output_dir}")
    logger.info(f"LOD Level: {args.lod_level}")
    logger.info(f"Patch size: {args.patch_size}m × {args.patch_size}m")
    logger.info(f"RGB: {'Non' if args.no_rgb else 'Oui'}")
    logger.info("="*70)
    
    start_time = time.time()
    
    # Statistiques
    stats = {
        'downloaded_tiles': 0,
        'enriched_tiles': 0,
        'total_patches': 0
    }
    
    # ÉTAPE 1: Téléchargement
    downloaded_files = []
    if not args.skip_download:
        downloaded_files = download_urban_tiles(
            args.output_dir,
            args.num_locations,
            args.tiles_per_location
        )
        stats['downloaded_tiles'] = len(downloaded_files)
    else:
        logger.info("\n⏭️  Skip téléchargement (--skip-download)")
        # Récupérer les fichiers existants
        tiles_dir = args.output_dir / "raw_tiles"
        if tiles_dir.exists():
            downloaded_files = list(tiles_dir.rglob("*.laz"))
            logger.info(f"  Trouvé {len(downloaded_files)} LAZ existants")
    
    if not downloaded_files:
        logger.error("❌ Aucun fichier LAZ disponible!")
        return
    
    # ÉTAPE 2: Enrichissement
    enriched_files = []
    if not args.skip_enrichment:
        enriched_files = enrich_laz_files(
            downloaded_files,
            args.output_dir,
            args.k_neighbors
        )
        stats['enriched_tiles'] = len(enriched_files)
    else:
        logger.info("\n⏭️  Skip enrichissement (--skip-enrichment)")
        # Récupérer les LAZ enrichis existants
        enriched_dir = args.output_dir / "enriched_laz"
        if enriched_dir.exists():
            enriched_files = list(enriched_dir.rglob("*_enriched.laz"))
            logger.info(f"  Trouvé {len(enriched_files)} LAZ enrichis")
    
    if not enriched_files:
        logger.error("❌ Aucun fichier LAZ enrichi disponible!")
        return
    
    # ÉTAPE 3: Création des patches
    if not args.skip_patches:
        total_patches = create_training_patches(
            enriched_files,
            args.output_dir,
            args.lod_level,
            args.patch_size,
            args.no_rgb,
            args.augment,
            args.num_augmentations
        )
        stats['total_patches'] = total_patches
    else:
        logger.info("\n⏭️  Skip création patches (--skip-patches)")
    
    # Sauvegarder les métadonnées
    elapsed_time = time.time() - start_time
    config = {
        'num_locations': args.num_locations,
        'tiles_per_location': args.tiles_per_location,
        'k_neighbors': args.k_neighbors,
        'lod_level': args.lod_level,
        'patch_size': args.patch_size,
        'no_rgb': args.no_rgb,
        'augment': args.augment,
        'num_augmentations': args.num_augmentations
    }
    
    stats['elapsed_time_seconds'] = elapsed_time
    stats['elapsed_time_formatted'] = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    
    save_metadata(args.output_dir, config, stats)
    
    # Résumé final
    logger.info("\n" + "="*70)
    logger.info("✅ WORKFLOW TERMINÉ")
    logger.info("="*70)
    logger.info(f"📥 Tuiles téléchargées: {stats['downloaded_tiles']}")
    logger.info(f"🔧 Tuiles enrichies: {stats['enriched_tiles']}")
    logger.info(f"🎯 Patches créés: {stats['total_patches']:,}")
    logger.info(f"⏱️  Temps écoulé: {stats['elapsed_time_formatted']}")
    logger.info(f"📁 Dataset: {args.output_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
