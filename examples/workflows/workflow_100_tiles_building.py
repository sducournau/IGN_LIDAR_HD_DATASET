#!/usr/bin/env python3
"""
Workflow complet: télécharger et processer 100 tuiles pour l'entraînement.

Configuration:
- 100 tuiles stratégiques diversifiées
- Patchs de 150m × 150m (22,500 m²)
- Mode building (toutes features géométriques)
- Features: normales, courbure, hauteur, densité, planarity, verticality, etc.

Usage:
    python workflow_100_tiles_building.py
    python workflow_100_tiles_building.py --skip-download  # Si déjà téléchargé
"""
#!/usr/bin/env python3
"""
DEPRECATED: This script has been moved to examples for reference.

⚠️  This script is deprecated and kept only for reference.
    Please use the unified CLI instead:
    
    Instead of: python workflow_100_tiles_building.py
    Use: ign-lidar-hd enrich [OPTIONS]
    
    See: ign-lidar-hd --help
    Or: https://github.com/your-username/ign-lidar-hd#readme

Original script below:
"""


import argparse
import logging
import sys
from pathlib import Path
import json
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_tiles(output_dir: Path, max_tiles: int = 100) -> list:
    """
    Télécharger les tuiles stratégiques.
    
    Args:
        output_dir: Répertoire de sortie
        max_tiles: Nombre maximum de tuiles
        
    Returns:
        Liste des fichiers téléchargés
    """
    logger.info("=" * 80)
    logger.info("ÉTAPE 1/3: TÉLÉCHARGEMENT DES TUILES")
    logger.info("=" * 80)
    
    try:
        from ign_lidar.downloader import IGNLiDARDownloader
        from ign_lidar.strategic_locations import (
            validate_locations_via_wfs,
            download_diverse_tiles
        )
    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        logger.error("Installez les dépendances: pip install -r requirements.txt")
        sys.exit(1)
    
    # Créer le downloader
    logger.info(f"🚀 Initialisation du téléchargeur IGN LIDAR HD")
    downloader = IGNLiDARDownloader(output_dir=output_dir)
    
    # Validation des localisations
    logger.info(f"🔍 Validation des localisations stratégiques via WFS IGN")
    validation_results = validate_locations_via_wfs(downloader)
    
    valid_count = sum(
        1 for r in validation_results.values()
        if r['available'] and r['tile_count'] > 0
    )
    total_tiles = sum(
        r['tile_count'] for r in validation_results.values()
        if r['available']
    )
    
    logger.info(f"✅ Validation:")
    logger.info(f"   {valid_count} localisations valides")
    logger.info(f"   {total_tiles} tuiles disponibles au total")
    
    if valid_count == 0:
        logger.error("❌ Aucune tuile disponible")
        sys.exit(1)
    
    # Téléchargement
    logger.info(f"\n📥 Téléchargement de {max_tiles} tuiles maximum")
    logger.info(f"   Destination: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = download_diverse_tiles(
        downloader,
        validation_results,
        output_dir,
        max_total_tiles=max_tiles
    )
    
    logger.info(f"\n✅ Téléchargement terminé: {len(downloaded)} tuiles")
    
    return downloaded


def enrich_tiles(input_dir: Path, output_dir: Path, num_workers: int = 1) -> int:
    """
    Enrichir les tuiles avec toutes les features géométriques.
    
    Args:
        input_dir: Répertoire des tuiles brutes
        output_dir: Répertoire des tuiles enrichies
        num_workers: Nombre de workers parallèles
        
    Returns:
        Nombre de fichiers traités avec succès
    """
    logger.info("\n" + "=" * 80)
    logger.info("ÉTAPE 2/3: ENRICHISSEMENT DES TUILES (MODE BUILDING)")
    logger.info("=" * 80)
    
    import numpy as np
    import laspy
    from ign_lidar.features import compute_all_features_optimized
    import multiprocessing as mp
    from functools import partial
    
    # Trouver les fichiers LAZ
    laz_files = list(input_dir.rglob("*.laz")) + list(input_dir.rglob("*.LAZ"))
    
    if not laz_files:
        logger.error(f"❌ Aucun fichier LAZ trouvé dans {input_dir}")
        return 0
    
    logger.info(f"🔍 Trouvé {len(laz_files)} fichiers LAZ")
    logger.info(f"📊 Mode: BUILDING (toutes features géométriques)")
    logger.info(f"⚙️  Workers: {num_workers}")
    logger.info(f"💾 Sortie: {output_dir}")
    logger.info("")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def _process_single_tile(laz_file: Path) -> tuple:
        """Worker pour traiter une tuile."""
        try:
            # 1. Charger le fichier LAZ
            las = laspy.read(str(laz_file))
            
            # 2. Extraire les données
            points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
            classification = np.array(las.classification, dtype=np.uint8)
            
            # 3. Calculer les features
            patch_center = np.mean(points, axis=0)
            
            normals, curvature, height, geo_features = \
                compute_all_features_optimized(
                    points=points,
                    classification=classification,
                    auto_k=True,
                    include_extra=True,
                    patch_center=patch_center
                )
            
            # 4. Créer le fichier LAZ enrichi
            header = laspy.LasHeader(
                point_format=las.header.point_format,
                version=las.header.version
            )
            header.offsets = las.header.offsets
            header.scales = las.header.scales
            
            # Dimensions supplémentaires
            extra_dims = [
                ("normal_x", "f4"), ("normal_y", "f4"), ("normal_z", "f4"),
                ("curvature", "f4"), ("height_above_ground", "f4"),
                ("density", "f4"), ("planarity", "f4"), ("linearity", "f4"),
                ("sphericity", "f4"), ("anisotropy", "f4"), ("roughness", "f4"),
                ("z_normalized", "f4"), ("z_from_ground", "f4"),
                ("verticality", "f4"), ("vertical_std", "f4"),
                ("height_extent_ratio", "f4"),
            ]
            
            las_out = laspy.LasData(header)
            las_out.points = las.points
            
            for dim_name, dim_type in extra_dims:
                las_out.add_extra_dim(laspy.ExtraBytesParams(
                    name=dim_name, type=dim_type
                ))
            
            # Remplir les données
            las_out.normal_x = normals[:, 0].astype(np.float32)
            las_out.normal_y = normals[:, 1].astype(np.float32)
            las_out.normal_z = normals[:, 2].astype(np.float32)
            las_out.curvature = curvature.astype(np.float32)
            las_out.height_above_ground = height.astype(np.float32)
            las_out.density = geo_features['density']
            las_out.planarity = geo_features['planarity']
            las_out.linearity = geo_features['linearity']
            las_out.sphericity = geo_features['sphericity']
            las_out.anisotropy = geo_features['anisotropy']
            las_out.roughness = geo_features['roughness']
            las_out.z_normalized = geo_features['z_normalized']
            las_out.z_from_ground = geo_features['z_from_ground']
            las_out.verticality = geo_features['verticality']
            las_out.vertical_std = geo_features['vertical_std']
            las_out.height_extent_ratio = geo_features['height_extent_ratio']
            
            # Sauvegarder (avec compression LAZ pour compatibilité QGIS)
            output_file = output_dir / laz_file.name
            las_out.write(str(output_file), do_compress=True)
            
            return (True, laz_file.name, len(points))
            
        except Exception as e:
            return (False, laz_file.name, str(e))
    
    # Traitement parallèle ou séquentiel
    if num_workers > 1:
        logger.info(f"🚀 Traitement parallèle avec {num_workers} workers")
        from tqdm import tqdm
        
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_single_tile, laz_files),
                total=len(laz_files),
                desc="Enrichissement"
            ))
        
        success_count = sum(1 for r in results if r[0])
        failed_files = [r[1] for r in results if not r[0]]
    else:
        logger.info("� Traitement séquentiel")
        success_count = 0
        failed_files = []
        
        for i, laz_file in enumerate(laz_files, 1):
            logger.info(f"[{i}/{len(laz_files)}] {laz_file.name}")
            success, name, info = _process_single_tile(laz_file)
            
            if success:
                logger.info(f"   ✓ {info:,} points traités")
                success_count += 1
            else:
                logger.error(f"   ✗ Erreur: {info}")
                failed_files.append(name)
    
    logger.info(f"\n✅ Enrichissement terminé:")
    logger.info(f"   {success_count}/{len(laz_files)} fichiers traités avec succès")
    
    if failed_files:
        logger.warning(f"   ⚠️  {len(failed_files)} fichiers échoués:")
        for f in failed_files[:5]:
            logger.warning(f"      - {f}")
        if len(failed_files) > 5:
            logger.warning(f"      ... et {len(failed_files) - 5} autres")
    
    return success_count


def create_patches(input_dir: Path, output_dir: Path, 
                  patch_size: float = 150.0, num_workers: int = 1) -> int:
    """
    Créer des patchs de 150m x 150m.
    
    Args:
        input_dir: Répertoire des tuiles enrichies
        output_dir: Répertoire des patchs
        patch_size: Taille du patch en mètres (défaut: 150.0m pour 150m × 150m)
        num_workers: Nombre de workers parallèles
        
    Returns:
        Nombre de patchs créés
    """
    logger.info("\n" + "=" * 80)
    logger.info("ÉTAPE 3/3: CRÉATION DES PATCHS 150m × 150m")
    logger.info("=" * 80)
    
    import numpy as np
    import laspy
    import multiprocessing as mp
    from functools import partial
    
    laz_files = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.LAZ"))
    
    if not laz_files:
        logger.error(f"❌ Aucun fichier LAZ trouvé dans {input_dir}")
        return 0
    
    logger.info(f"🔍 Trouvé {len(laz_files)} fichiers LAZ enrichis")
    logger.info(f"📏 Taille: {patch_size}m x {patch_size}m ≈ {patch_size**2:.1f}m²")
    logger.info(f"⚙️  Workers: {num_workers}")
    logger.info(f"💾 Sortie: {output_dir}")
    logger.info("")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_patches_for_tile(laz_file: Path) -> int:
        """Worker pour créer des patchs d'une tuile."""
        try:
            las = laspy.read(str(laz_file))
            x = np.array(las.x)
            y = np.array(las.y)
            
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            
            nx = int(np.ceil((x_max - x_min) / patch_size))
            ny = int(np.ceil((y_max - y_min) / patch_size))
            
            patch_count = 0
            
            for ix in range(nx):
                for iy in range(ny):
                    px_min = x_min + ix * patch_size
                    px_max = px_min + patch_size
                    py_min = y_min + iy * patch_size
                    py_max = py_min + patch_size
                    
                    mask = ((x >= px_min) & (x < px_max) & 
                           (y >= py_min) & (y < py_max))
                    
                    if not mask.any() or mask.sum() < 1000:
                        continue
                    
                    patch_las = laspy.LasData(las.header)
                    patch_las.points = las.points[mask]
                    
                    patch_name = f"{laz_file.stem}_patch_{ix:02d}_{iy:02d}.laz"
                    patch_path = output_dir / patch_name
                    
                    patch_las.write(str(patch_path))
                    patch_count += 1
            
            return patch_count
            
        except Exception as e:
            logger.error(f"Erreur {laz_file.name}: {e}")
            return 0
    
    # Traitement parallèle ou séquentiel
    if num_workers > 1:
        logger.info(f"🚀 Traitement parallèle avec {num_workers} workers")
        from tqdm import tqdm
        
        with mp.Pool(num_workers) as pool:
            patch_counts = list(tqdm(
                pool.imap(_create_patches_for_tile, laz_files),
                total=len(laz_files),
                desc="Création patchs"
            ))
        
        total_patches = sum(patch_counts)
    else:
        logger.info("📝 Traitement séquentiel")
        total_patches = 0
        
        for i, laz_file in enumerate(laz_files, 1):
            logger.info(f"[{i}/{len(laz_files)}] {laz_file.name}")
            count = _create_patches_for_tile(laz_file)
            logger.info(f"   ✓ {count} patchs créés")
            total_patches += count
    
    logger.info(f"\n✅ Création des patchs terminée:")
    logger.info(f"   {total_patches} patchs créés au total")
    
    return total_patches


def save_metadata(output_base: Path, stats: dict):
    """Sauvegarder les métadonnées du workflow."""
    metadata = {
        "workflow": "100_tiles_building_training",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "max_tiles": 100,
            "patch_size_m": 150.0,
            "patch_area_m2": 22500,
            "mode": "building",
            "features": [
                "normal_x", "normal_y", "normal_z",
                "curvature", "height_above_ground",
                "density", "planarity", "linearity",
                "sphericity", "anisotropy", "roughness",
                "z_normalized", "z_from_ground",
                "verticality", "vertical_std", "height_extent_ratio"
            ]
        },
        "statistics": stats
    }
    
    metadata_file = output_base / "workflow_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n📄 Métadonnées sauvegardées: {metadata_file}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description='Workflow complet: télécharger et processer 100 tuiles en mode building'
    )
    
    parser.add_argument(
        '--output-base',
        type=Path,
        default=Path('urban_training_dataset'),
        help='Répertoire de base (défaut: urban_training_dataset/)'
    )
    
    parser.add_argument(
        '--max-tiles',
        type=int,
        default=100,
        help='Nombre maximum de tuiles à télécharger (défaut: 100)'
    )
    
    parser.add_argument(
        '--patch-size',
        type=float,
        default=150.0,
        help='Taille des patchs en mètres (défaut: 150.0 pour 150m × 150m)'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Sauter le téléchargement (utiliser les tuiles existantes)'
    )
    
    parser.add_argument(
        '--skip-enrich',
        action='store_true',
        help='Sauter l\'enrichissement (utiliser les tuiles enrichies existantes)'
    )
    
    parser.add_argument(
        '--skip-patches',
        action='store_true',
        help='Sauter la création des patchs'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Nombre de workers parallèles pour traitement (défaut: 1)'
    )
    
    parser.add_argument(
        '--download-workers',
        type=int,
        default=3,
        help='Nombre de workers pour téléchargement parallèle (défaut: 3)'
    )
    
    args = parser.parse_args()
    
    # Créer la structure de répertoires
    raw_dir = args.output_base / "raw_tiles"
    enriched_dir = args.output_base / "enriched_laz_building"
    patches_dir = args.output_base / "patches_150x150m"
    
    logger.info("=" * 80)
    logger.info("WORKFLOW: TÉLÉCHARGEMENT ET TRAITEMENT - 100 TUILES MODE BUILDING")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - Tuiles: {args.max_tiles}")
    logger.info(f"  - Patchs: {args.patch_size}m x {args.patch_size}m "
                f"≈ {args.patch_size**2:.1f}m²")
    logger.info(f"  - Mode: BUILDING (15 features géométriques)")
    logger.info(f"  - Workers traitement: {args.workers}")
    logger.info(f"  - Workers téléchargement: {args.download_workers}")
    logger.info(f"  - Base: {args.output_base}")
    logger.info("")
    
    stats = {}
    
    # ÉTAPE 1: Téléchargement
    if not args.skip_download:
        downloaded = download_tiles(raw_dir, args.max_tiles)
        stats['tiles_downloaded'] = len(downloaded)
    else:
        logger.info("⏭️  Étape 1/3: Téléchargement sauté")
        laz_files = list(raw_dir.rglob("*.laz")) + list(raw_dir.rglob("*.LAZ"))
        stats['tiles_downloaded'] = len(laz_files)
    
    # ÉTAPE 2: Enrichissement
    if not args.skip_enrich:
        enriched_count = enrich_tiles(raw_dir, enriched_dir, args.workers)
        stats['tiles_enriched'] = enriched_count
    else:
        logger.info("⏭️  Étape 2/3: Enrichissement sauté")
        laz_files = (list(enriched_dir.glob("*.laz")) + 
                    list(enriched_dir.glob("*.LAZ")))
        stats['tiles_enriched'] = len(laz_files)
    
    # ÉTAPE 3: Création des patchs
    if not args.skip_patches:
        patches_count = create_patches(
            enriched_dir, patches_dir, args.patch_size, args.workers
        )
        stats['patches_created'] = patches_count
    else:
        logger.info("⏭️  Étape 3/3: Création des patchs sautée")
        laz_files = (list(patches_dir.glob("*.laz")) + 
                    list(patches_dir.glob("*.LAZ")))
        stats['patches_created'] = len(laz_files)
    
    # Sauvegarder les métadonnées
    save_metadata(args.output_base, stats)
    
    # Résumé final
    logger.info("\n" + "=" * 80)
    logger.info("✅ WORKFLOW TERMINÉ!")
    logger.info("=" * 80)
    logger.info(f"📊 Statistiques:")
    logger.info(f"   - Tuiles téléchargées: {stats.get('tiles_downloaded', 'N/A')}")
    logger.info(f"   - Tuiles enrichies: {stats.get('tiles_enriched', 'N/A')}")
    logger.info(f"   - Patchs créés: {stats.get('patches_created', 'N/A')}")
    logger.info(f"\n📁 Répertoires:")
    logger.info(f"   - Tuiles brutes: {raw_dir}")
    logger.info(f"   - Tuiles enrichies: {enriched_dir}")
    logger.info(f"   - Patchs 150m × 150m: {patches_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
