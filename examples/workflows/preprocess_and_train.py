#!/usr/bin/env python3
"""
Script de préprocessing et création du dataset d'entraînement.

Ce script:
1. Enrichit les tuiles brutes avec les features géométriques
2. Crée des patchs de 150m × 150m pour l'entraînement

Usage:
    python preprocess_and_train.py
    python preprocess_and_train.py --workers 4
    python preprocess_and_train.py --skip-enrich  # Si déjà enrichi
"""
#!/usr/bin/env python3
"""
DEPRECATED: This script has been moved to examples for reference.

⚠️  This script is deprecated and kept only for reference.
    Please use the unified CLI instead:
    
    Instead of: python preprocess_and_train.py
    Use: ign-lidar-process enrich [OPTIONS]
    
    See: ign-lidar-process --help
    Or: https://github.com/your-username/ign-lidar-hd#readme

Original script below:
"""


import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _process_single_tile(args):
    """Worker pour traiter une tuile (niveau module pour pickle)."""
    laz_file, output_dir = args
    
    try:
        import numpy as np
        import laspy
        from ign_lidar.features import compute_all_features_optimized
        
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
            ("sphericity", "f4"), ("anisotropy", "f4"),
            ("roughness", "f4"),
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


def _create_patches_for_tile(args):
    """Worker pour créer des patchs d'une tuile (niveau module pour pickle)."""
    laz_file, output_dir, patch_size = args
    
    try:
        import numpy as np
        import laspy
        
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
                
                patch_name = (f"{laz_file.stem}_"
                            f"patch_{ix:02d}_{iy:02d}.laz")
                patch_path = output_dir / patch_name
                
                patch_las.write(str(patch_path))
                patch_count += 1
        
        return patch_count
        
    except Exception as e:
        logger.error(f"Erreur {laz_file.name}: {e}")
        return 0


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
    logger.info("=" * 80)
    logger.info("ÉTAPE 1/2: ENRICHISSEMENT DES TUILES (MODE BUILDING)")
    logger.info("=" * 80)
    
    import numpy as np
    import laspy
    from ign_lidar.features import compute_all_features_optimized
    import multiprocessing as mp
    from tqdm import tqdm
    
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
    
    # Préparer les arguments pour les workers
    worker_args = [(laz_file, output_dir) for laz_file in laz_files]
    
    # Traitement parallèle ou séquentiel
    if num_workers > 1:
        logger.info(f"🚀 Traitement parallèle avec {num_workers} workers")
        
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_single_tile, worker_args),
                total=len(laz_files),
                desc="Enrichissement"
            ))
        
        success_count = sum(1 for r in results if r[0])
        failed_files = [r[1] for r in results if not r[0]]
    else:
        logger.info("📝 Traitement séquentiel")
        success_count = 0
        failed_files = []
        
        for i, args in enumerate(worker_args, 1):
            laz_file, _ = args
            logger.info(f"[{i}/{len(laz_files)}] {laz_file.name}")
            success, name, info = _process_single_tile(args)
            
            if success:
                logger.info(f"   ✓ {info:,} points traités")
                success_count += 1
            else:
                logger.error(f"   ✗ Erreur: {info}")
                failed_files.append(name)
    
    logger.info(f"\n✅ Enrichissement terminé:")
    logger.info(f"   {success_count}/{len(laz_files)} fichiers traités")
    
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
        patch_size: Taille du patch en mètres
        num_workers: Nombre de workers parallèles
        
    Returns:
        Nombre de patchs créés
    """
    logger.info("\n" + "=" * 80)
    logger.info("ÉTAPE 2/2: CRÉATION DES PATCHS 150m × 150m")
    logger.info("=" * 80)
    
    import numpy as np
    import laspy
    import multiprocessing as mp
    from tqdm import tqdm
    
    laz_files = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.LAZ"))
    
    if not laz_files:
        logger.error(f"❌ Aucun fichier LAZ trouvé dans {input_dir}")
        return 0
    
    logger.info(f"🔍 Trouvé {len(laz_files)} fichiers LAZ enrichis")
    logger.info(f"📏 Taille: {patch_size}m x {patch_size}m "
                f"≈ {patch_size**2:.1f}m²")
    logger.info(f"⚙️  Workers: {num_workers}")
    logger.info(f"💾 Sortie: {output_dir}")
    logger.info("")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Préparer les arguments pour les workers
    worker_args = [(laz_file, output_dir, patch_size) for laz_file in laz_files]
    
    # Traitement parallèle ou séquentiel
    if num_workers > 1:
        logger.info(f"🚀 Traitement parallèle avec {num_workers} workers")
        
        with mp.Pool(num_workers) as pool:
            patch_counts = list(tqdm(
                pool.imap(_create_patches_for_tile, worker_args),
                total=len(laz_files),
                desc="Création patchs"
            ))
        
        total_patches = sum(patch_counts)
    else:
        logger.info("📝 Traitement séquentiel")
        total_patches = 0
        
        for i, args in enumerate(worker_args, 1):
            laz_file, _, _ = args
            logger.info(f"[{i}/{len(laz_files)}] {laz_file.name}")
            count = _create_patches_for_tile(args)
            logger.info(f"   ✓ {count} patchs créés")
            total_patches += count
    
    logger.info(f"\n✅ Création des patchs terminée:")
    logger.info(f"   {total_patches} patchs créés au total")
    
    return total_patches


def save_metadata(output_dir: Path, stats: dict):
    """Sauvegarder les métadonnées du traitement."""
    metadata = {
        "workflow": "preprocess_and_train",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
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
    
    metadata_file = output_dir / "processing_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n📄 Métadonnées sauvegardées: {metadata_file}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description='Préprocesser les tuiles et créer le dataset entraînement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Traitement complet avec 4 workers
  python preprocess_and_train.py --workers 4
  
  # Sauter l'enrichissement si déjà fait
  python preprocess_and_train.py --skip-enrich --workers 4
  
  # Sauter la création des patchs
  python preprocess_and_train.py --skip-patches --workers 4
        """
    )
    
    parser.add_argument(
        '--raw-tiles',
        type=Path,
        default=Path('/mnt/c/Users/Simon/ign/raw_tiles'),
        help='Répertoire des tuiles brutes (défaut: /mnt/c/Users/Simon/ign/raw_tiles)'
    )
    
    parser.add_argument(
        '--preprocessed',
        type=Path,
        default=Path('/mnt/c/Users/Simon/ign/preprocessed_tiles'),
        help='Répertoire des tuiles enrichies (défaut: /mnt/c/Users/Simon/ign/preprocessed_tiles)'
    )
    
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('dataset/ign_150'),
        help='Répertoire du dataset d\'entraînement (défaut: dataset/ign_150)'
    )
    
    parser.add_argument(
        '--patch-size',
        type=float,
        default=150.0,
        help='Taille des patchs en mètres (défaut: 150.0)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Nombre de workers parallèles (défaut: 1)'
    )
    
    parser.add_argument(
        '--skip-enrich',
        action='store_true',
        help='Sauter l\'enrichissement (utiliser tuiles déjà enrichies)'
    )
    
    parser.add_argument(
        '--skip-patches',
        action='store_true',
        help='Sauter la création des patchs'
    )
    
    args = parser.parse_args()
    
    # Vérifier que le répertoire source existe
    if not args.raw_tiles.exists():
        logger.error(f"❌ Répertoire introuvable: {args.raw_tiles}")
        logger.info("Créez le répertoire ou spécifiez un autre chemin avec --raw-tiles")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("PRÉPROCESSING ET CRÉATION DU DATASET D'ENTRAÎNEMENT")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - Tuiles brutes:  {args.raw_tiles}")
    logger.info(f"  - Tuiles enrichies: {args.preprocessed}")
    logger.info(f"  - Dataset:         {args.dataset}")
    logger.info(f"  - Patch size:      {args.patch_size}m × {args.patch_size}m")
    logger.info(f"  - Workers:         {args.workers}")
    logger.info("")
    
    stats = {}
    
    # ÉTAPE 1: Enrichissement
    if not args.skip_enrich:
        enriched_count = enrich_tiles(
            args.raw_tiles,
            args.preprocessed,
            args.workers
        )
        stats['tiles_enriched'] = enriched_count
        
        if enriched_count == 0:
            logger.error("❌ Aucune tuile enrichie, arrêt du traitement")
            sys.exit(1)
    else:
        logger.info("⏭️  Étape 1/2: Enrichissement sauté")
        laz_files = (list(args.preprocessed.glob("*.laz")) +
                    list(args.preprocessed.glob("*.LAZ")))
        stats['tiles_enriched'] = len(laz_files)
        logger.info(f"   {stats['tiles_enriched']} tuiles enrichies trouvées")
    
    # ÉTAPE 2: Création des patchs
    if not args.skip_patches:
        patches_count = create_patches(
            args.preprocessed,
            args.dataset,
            args.patch_size,
            args.workers
        )
        stats['patches_created'] = patches_count
        
        if patches_count == 0:
            logger.error("❌ Aucun patch créé")
            sys.exit(1)
    else:
        logger.info("⏭️  Étape 2/2: Création des patchs sautée")
        laz_files = (list(args.dataset.glob("*.laz")) +
                    list(args.dataset.glob("*.LAZ")))
        stats['patches_created'] = len(laz_files)
    
    # Sauvegarder les métadonnées
    save_metadata(args.dataset, stats)
    
    # Résumé final
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAITEMENT TERMINÉ!")
    logger.info("=" * 80)
    logger.info(f"📊 Statistiques:")
    logger.info(f"   - Tuiles enrichies: {stats.get('tiles_enriched', 'N/A')}")
    logger.info(f"   - Patchs créés:     {stats.get('patches_created', 'N/A')}")
    logger.info(f"\n📁 Répertoires:")
    logger.info(f"   - Tuiles brutes:    {args.raw_tiles}")
    logger.info(f"   - Tuiles enrichies: {args.preprocessed}")
    logger.info(f"   - Dataset training: {args.dataset}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Prochaines étapes:")
    logger.info("  1. Vérifier les patchs créés:")
    logger.info(f"     ls -lh {args.dataset}/*.laz | head")
    logger.info("  2. Utiliser le dataset pour l'entraînement:")
    logger.info(f"     python train.py --data {args.dataset}")


if __name__ == '__main__':
    main()
