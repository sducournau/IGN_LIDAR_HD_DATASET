#!/usr/bin/env python3
"""
Workflow complet : Téléchargement + Préparation + Génération de patches
Avec styles architecturaux, LOD2/LOD3, et toutes les features géométriques.

Ce script effectue :
1. Téléchargement de 100 tuiles LiDAR stratégiques
2. Préparation des fichiers LAZ (validation)
3. Génération de patches 150m×150m avec k-neighbors=20 et style architectural
"""

import argparse
import logging
import sys
from pathlib import Path
import time
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Point d'entrée principal."""
    
    # Configuration des chemins Windows (convertis automatiquement sous WSL)
    RAW_TILES_DIR = Path(r"C:\Users\Simon\ign\raw_tiles")
    PRE_TILES_DIR = Path(r"C:\Users\Simon\ign\pre_tiles")
    DATASET_DIR = Path(r"C:\Users\Simon\ign\dataset\ign_150")
    
    # Convertir les chemins Windows en chemins WSL si nécessaire
    def convert_to_wsl_path(windows_path: Path) -> Path:
        """Convertit un chemin Windows en chemin WSL si on est sous WSL."""
        path_str = str(windows_path)
        if path_str.startswith("C:\\"):
            # Convertir C:\ en /mnt/c/
            wsl_path = path_str.replace("C:\\", "/mnt/c/").replace("\\", "/")
            return Path(wsl_path)
        return windows_path
    
    raw_tiles = convert_to_wsl_path(RAW_TILES_DIR)
    pre_tiles = convert_to_wsl_path(PRE_TILES_DIR)
    dataset = convert_to_wsl_path(DATASET_DIR)
    
    logger.info("=" * 80)
    logger.info("🚀 WORKFLOW COMPLET IGN LIDAR HD")
    logger.info("=" * 80)
    logger.info(f"📂 Tuiles brutes    : {raw_tiles}")
    logger.info(f"📂 Tuiles préparées : {pre_tiles}")
    logger.info(f"📂 Dataset patches  : {dataset}")
    logger.info(f"")
    logger.info("⚙️  Configuration:")
    logger.info(f"   • Nombre de tuiles : 100")
    logger.info(f"   • LOD Level        : LOD2 + LOD3")
    logger.info(f"   • K-neighbors      : 20")
    logger.info(f"   • Patch size       : 150m × 150m")
    logger.info(f"   • Features         : Toutes (géométriques + bâtiments)")
    logger.info(f"   • Style archi.     : Activé")
    logger.info("=" * 80)
    
    # Import des modules nécessaires
    try:
        from ign_lidar.downloader import IGNLiDARDownloader
        from ign_lidar.strategic_locations import (
            validate_locations_via_wfs,
            download_diverse_tiles
        )
        from ign_lidar.processor import LiDARProcessor
    except ImportError as e:
        logger.error(f"❌ Erreur d'import: {e}")
        logger.error("Installez les dépendances : pip install -r requirements.txt")
        sys.exit(1)
    
    workflow_start = time.time()
    
    # ========================================================================
    # ÉTAPE 1 : TÉLÉCHARGEMENT DES 100 TUILES
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("📥 ÉTAPE 1/3 : TÉLÉCHARGEMENT DES TUILES LIDAR")
    logger.info("=" * 80)
    
    raw_tiles.mkdir(parents=True, exist_ok=True)
    
    # Vérifier si des tuiles existent déjà
    existing_laz = list(raw_tiles.rglob("*.laz"))
    
    # Initialiser le downloader
    downloader = IGNLiDARDownloader()
    
    # Toujours valider les localisations pour régénérer les métadonnées
    logger.info("🔍 Validation des localisations stratégiques via WFS IGN...")
    validation_results = validate_locations_via_wfs(downloader)
    
    valid_count = sum(
        1 for r in validation_results.values()
        if r['available'] and r['tile_count'] > 0
    )
    total_tiles_available = sum(
        r['tile_count'] for r in validation_results.values()
        if r['available']
    )
    
    logger.info(f"✅ {valid_count} localisations valides, {total_tiles_available} tuiles disponibles")
    
    if len(existing_laz) >= 100:
        logger.info(f"✅ {len(existing_laz)} tuiles LAZ déjà présentes dans {raw_tiles}")
        logger.info("🔄 Régénération des métadonnées pour les tuiles existantes...")
        
        # Régénérer stats.json et métadonnées des tuiles
        from ign_lidar.metadata import MetadataManager
        from ign_lidar.strategic_locations import STRATEGIC_LOCATIONS
        
        metadata_mgr = MetadataManager(raw_tiles)
        
        # Créer la liste des infos de tuiles pour stats.json
        tiles_info = []
        metadata_count = 0
        
        for laz_file in existing_laz:
            # Trouver la localisation correspondante depuis le nom du dossier parent
            category = laz_file.parent.name
            location_name = None
            location_config = None
            
            # Chercher dans STRATEGIC_LOCATIONS
            for loc_name, loc_cfg in STRATEGIC_LOCATIONS.items():
                if loc_cfg['category'] == category:
                    location_name = loc_name
                    location_config = loc_cfg
                    break
            
            # Créer metadata pour la tuile
            if location_config:
                tile_metadata = metadata_mgr.create_tile_metadata(
                    filename=laz_file.name,
                    location_name=location_name,
                    category=category,
                    characteristics=location_config.get('characteristics', []),
                    description=location_config.get('description'),
                    architectural_style=location_config.get('architectural_style'),
                    bbox=location_config.get('bbox')
                )
                metadata_mgr.save_tile_metadata(tile_metadata, subdirectory=category)
                metadata_count += 1
            
            # Ajouter aux tiles_info
            tiles_info.append({
                "filename": laz_file.name,
                "category": category,
                "location": location_name or "Unknown"
            })
        
        # Créer et sauvegarder stats.json
        stats = metadata_mgr.create_download_stats(tiles_info)
        stats["regenerated_at"] = datetime.now().isoformat()
        metadata_mgr.save_stats(stats)
        
        logger.info(f"✅ Métadonnées régénérées : {metadata_count} tuiles, stats.json mis à jour")
    else:
        logger.info(f"🔍 {len(existing_laz)} tuiles existantes, téléchargement de {100 - len(existing_laz)} supplémentaires")
        
        # Téléchargement
        downloaded = download_diverse_tiles(
            downloader,
            validation_results,
            raw_tiles,
            max_total_tiles=100
        )
        
        logger.info(f"✅ {len(downloaded)} tuiles téléchargées")
        existing_laz = list(raw_tiles.rglob("*.laz"))
    
    download_time = time.time() - workflow_start
    logger.info(f"⏱️  Temps de téléchargement : {download_time:.1f}s")
    
    # ========================================================================
    # ÉTAPE 2 : PRÉPARATION DES TUILES
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("🔧 ÉTAPE 2/3 : PRÉPARATION DES TUILES LAZ")
    logger.info("=" * 80)
    
    prep_start = time.time()
    pre_tiles.mkdir(parents=True, exist_ok=True)
    
    # Copier/vérifier les tuiles LAZ
    import shutil
    from ign_lidar.metadata import MetadataManager
    from ign_lidar.strategic_locations import STRATEGIC_LOCATIONS
    
    prepared_count = 0
    
    for laz_file in existing_laz:
        # Conserver la structure des sous-dossiers (catégories)
        relative_path = laz_file.relative_to(raw_tiles)
        dest_file = pre_tiles / relative_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not dest_file.exists():
            shutil.copy2(laz_file, dest_file)
            prepared_count += 1
            if prepared_count % 10 == 0:
                logger.info(f"   Préparé {prepared_count} tuiles...")
    
    logger.info(f"✅ {len(existing_laz)} tuiles LAZ préparées dans {pre_tiles}")
    
    # Régénérer les métadonnées pour pre_tiles
    logger.info("🔄 Régénération des métadonnées dans pre_tiles...")
    metadata_mgr_pre = MetadataManager(pre_tiles)
    
    prepared_laz = list(pre_tiles.rglob("*.laz"))
    tiles_info_pre = []
    metadata_count_pre = 0
    
    for laz_file in prepared_laz:
        category = laz_file.parent.name
        location_name = None
        location_config = None
        
        # Chercher dans STRATEGIC_LOCATIONS
        for loc_name, loc_cfg in STRATEGIC_LOCATIONS.items():
            if loc_cfg['category'] == category:
                location_name = loc_name
                location_config = loc_cfg
                break
        
        # Créer et sauvegarder metadata pour la tuile
        if location_config:
            tile_metadata = metadata_mgr_pre.create_tile_metadata(
                filename=laz_file.name,
                location_name=location_name,
                category=category,
                characteristics=location_config.get('characteristics', []),
                description=location_config.get('description'),
                architectural_style=location_config.get('architectural_style'),
                bbox=location_config.get('bbox')
            )
            metadata_mgr_pre.save_tile_metadata(
                tile_metadata,
                subdirectory=category
            )
            metadata_count_pre += 1
        
        tiles_info_pre.append({
            "filename": laz_file.name,
            "category": category,
            "location": location_name or "Unknown"
        })
    
    # Créer et sauvegarder stats.json pour pre_tiles
    stats_pre = metadata_mgr_pre.create_download_stats(tiles_info_pre)
    stats_pre["type"] = "prepared_tiles"
    stats_pre["source_directory"] = str(raw_tiles)
    stats_pre["prepared_at"] = datetime.now().isoformat()
    metadata_mgr_pre.save_stats(stats_pre)
    
    logger.info(f"✅ Métadonnées créées : {metadata_count_pre} tuiles")
    prep_time = time.time() - prep_start
    logger.info(f"⏱️  Temps de préparation : {prep_time:.1f}s")
    
    # ========================================================================
    # ÉTAPE 3 : GÉNÉRATION DES PATCHES AVEC FEATURES
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("🎯 ÉTAPE 3/3 : GÉNÉRATION DES PATCHES 150m×150m")
    logger.info("=" * 80)
    logger.info("⚙️  Configuration des features:")
    logger.info("   • K-neighbors = 20")
    logger.info("   • Features géométriques complètes")
    logger.info("   • Style architectural activé")
    logger.info("   • LOD2 + LOD3 pour bâtiments")
    
    patch_start = time.time()
    
    # Préparer les répertoires pour LOD2 et LOD3
    lod2_dir = dataset / "lod2"
    lod3_dir = dataset / "lod3"
    lod2_dir.mkdir(parents=True, exist_ok=True)
    lod3_dir.mkdir(parents=True, exist_ok=True)
    
    prepared_laz = list(pre_tiles.rglob("*.laz"))
    logger.info(f"📂 {len(prepared_laz)} tuiles LAZ à traiter")
    
    # ========================================================================
    # GÉNÉRATION LOD2
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("🏢 Génération des patches LOD2...")
    logger.info("-" * 80)
    
    processor_lod2 = LiDARProcessor(
        lod_level='LOD2',
        augment=False,  # Pas d'augmentation pour le moment
        patch_size=150.0,
        patch_overlap=0.1,
        include_extra_features=True,  # Toutes les features géométriques
        k_neighbors=20,
        include_architectural_style=True  # Style architectural
    )
    
    total_patches_lod2 = 0
    for idx, laz_file in enumerate(prepared_laz, 1):
        try:
            num_patches = processor_lod2.process_tile(
                laz_file,
                lod2_dir,
                tile_idx=idx,
                total_tiles=len(prepared_laz)
            )
            total_patches_lod2 += num_patches
            
            if idx % 10 == 0:
                logger.info(f"   Progression LOD2: {idx}/{len(prepared_laz)} tuiles, {total_patches_lod2} patches")
                
        except Exception as e:
            logger.error(f"❌ Erreur sur {laz_file.name}: {e}")
            continue
    
    logger.info(f"✅ LOD2: {total_patches_lod2} patches générés")
    
    # Générer stats.json pour LOD2
    logger.info("📝 Création des métadonnées LOD2...")
    metadata_mgr_lod2 = MetadataManager(lod2_dir)
    stats_lod2 = metadata_mgr_lod2.create_processing_stats(
        input_dir=pre_tiles,
        num_tiles=len(prepared_laz),
        num_patches=total_patches_lod2,
        lod_level="LOD2",
        k_neighbors=20,
        patch_size=150.0,
        augmentation=False,
        num_augmentations=0
    )
    stats_lod2["features"] = {
        "geometric": True,
        "building_extraction": True,
        "architectural_style": True
    }
    metadata_mgr_lod2.save_stats(stats_lod2)
    logger.info(f"✅ Métadonnées LOD2 sauvegardées")
    
    # ========================================================================
    # GÉNÉRATION LOD3
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("🏛️  Génération des patches LOD3...")
    logger.info("-" * 80)
    
    processor_lod3 = LiDARProcessor(
        lod_level='LOD3',
        augment=False,
        patch_size=150.0,
        patch_overlap=0.1,
        include_extra_features=True,
        k_neighbors=20,
        include_architectural_style=True
    )
    
    total_patches_lod3 = 0
    for idx, laz_file in enumerate(prepared_laz, 1):
        try:
            num_patches = processor_lod3.process_tile(
                laz_file,
                lod3_dir,
                tile_idx=idx,
                total_tiles=len(prepared_laz)
            )
            total_patches_lod3 += num_patches
            
            if idx % 10 == 0:
                logger.info(f"   Progression LOD3: {idx}/{len(prepared_laz)} tuiles, {total_patches_lod3} patches")
                
        except Exception as e:
            logger.error(f"❌ Erreur sur {laz_file.name}: {e}")
            continue
    
    logger.info(f"✅ LOD3: {total_patches_lod3} patches générés")
    
    # Générer stats.json pour LOD3
    logger.info("📝 Création des métadonnées LOD3...")
    metadata_mgr_lod3 = MetadataManager(lod3_dir)
    stats_lod3 = metadata_mgr_lod3.create_processing_stats(
        input_dir=pre_tiles,
        num_tiles=len(prepared_laz),
        num_patches=total_patches_lod3,
        lod_level="LOD3",
        k_neighbors=20,
        patch_size=150.0,
        augmentation=False,
        num_augmentations=0
    )
    stats_lod3["features"] = {
        "geometric": True,
        "building_extraction": True,
        "architectural_style": True
    }
    metadata_mgr_lod3.save_stats(stats_lod3)
    logger.info(f"✅ Métadonnées LOD3 sauvegardées")
    
    patch_time = time.time() - patch_start
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    total_time = time.time() - workflow_start
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ WORKFLOW TERMINÉ AVEC SUCCÈS!")
    logger.info("=" * 80)
    logger.info(f"")
    logger.info(f"📊 RÉSUMÉ:")
    logger.info(f"   • Tuiles téléchargées  : {len(existing_laz)}")
    logger.info(f"   • Tuiles préparées     : {len(prepared_laz)}")
    logger.info(f"   • Patches LOD2         : {total_patches_lod2}")
    logger.info(f"   • Patches LOD3         : {total_patches_lod3}")
    logger.info(f"   • Total patches        : {total_patches_lod2 + total_patches_lod3}")
    logger.info(f"")
    logger.info(f"⏱️  TEMPS:")
    logger.info(f"   • Téléchargement       : {download_time:.1f}s ({download_time/60:.1f} min)")
    logger.info(f"   • Préparation          : {prep_time:.1f}s")
    logger.info(f"   • Génération patches   : {patch_time:.1f}s ({patch_time/60:.1f} min)")
    logger.info(f"   • TOTAL                : {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"")
    logger.info(f"📂 RÉPERTOIRES:")
    logger.info(f"   • Tuiles brutes    : {raw_tiles}")
    logger.info(f"   • Tuiles préparées : {pre_tiles}")
    logger.info(f"   • Dataset LOD2     : {lod2_dir}")
    logger.info(f"   • Dataset LOD3     : {lod3_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
