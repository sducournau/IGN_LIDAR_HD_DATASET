"""
Exemple d'utilisation du classificateur avancé avec CLI

Ce script montre comment intégrer le nouveau classificateur avancé
dans le workflow de traitement existant.
"""

from pathlib import Path
import numpy as np
import laspy
import logging

from ign_lidar.core.modules.advanced_classification import classify_with_all_features
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.features.geometric import compute_geometric_features
from ign_lidar.preprocessing.rgb_augmentation import IGNOrthophotoFetcher
from ign_lidar.preprocessing.infrared_augmentation import IGNInfraredFetcher
from ign_lidar.core.modules.enrichment import compute_ndvi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_laz_file_advanced(
    input_path: Path,
    output_path: Path,
    cache_dir: Path = None,
    fetch_rgb_nir: bool = True,
    compute_geometric: bool = True,
    k_neighbors: int = 20,
    **classifier_kwargs
):
    """
    Classifier un fichier LAZ avec toutes les features disponibles.
    
    Args:
        input_path: Chemin du fichier LAZ d'entrée
        output_path: Chemin du fichier LAZ de sortie
        cache_dir: Répertoire de cache
        fetch_rgb_nir: Récupérer RGB/NIR si non présent
        compute_geometric: Calculer features géométriques
        k_neighbors: Nombre de voisins pour features géométriques
        **classifier_kwargs: Arguments pour AdvancedClassifier
    """
    logger.info(f"📂 Loading: {input_path}")
    las = laspy.read(str(input_path))
    
    # Extraire les points
    points = np.vstack([las.x, las.y, las.z]).T
    n_points = len(points)
    logger.info(f"  Loaded {n_points:,} points")
    
    # Calculer bounding box
    bbox = (
        float(las.header.x_min),
        float(las.header.y_min),
        float(las.header.x_max),
        float(las.header.y_max)
    )
    logger.info(f"  Bounding box: {bbox}")
    
    # ========================================================================
    # 1. Préparer les features géométriques
    # ========================================================================
    height = None
    normals = None
    planarity = None
    curvature = None
    
    if compute_geometric:
        logger.info("🔧 Computing geometric features...")
        
        # Calculer hauteur (Z relatif au sol)
        if hasattr(las, 'classification'):
            ground_mask = las.classification == 2
            if np.any(ground_mask):
                ground_points = points[ground_mask]
                # Simple: min Z dans voisinage
                from scipy.spatial import cKDTree
                tree = cKDTree(ground_points[:, :2])
                distances, _ = tree.query(points[:, :2], k=1)
                ground_z = tree.data[tree.query(points[:, :2], k=1)[1], 2]
                height = points[:, 2] - ground_z
                logger.info(f"  ✓ Height computed (range: {height.min():.1f}m to {height.max():.1f}m)")
        
        if height is None:
            # Fallback: hauteur relative au min Z local
            height = points[:, 2] - points[:, 2].min()
            logger.info(f"  ⚠️  Using relative height (no ground classification)")
        
        # Calculer autres features géométriques
        try:
            features = compute_geometric_features(
                points=points,
                k_neighbors=k_neighbors,
                compute_normals=True,
                compute_planarity=True,
                compute_curvature=True
            )
            normals = features.get('normals')
            planarity = features.get('planarity')
            curvature = features.get('curvature')
            logger.info(f"  ✓ Geometric features computed")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to compute geometric features: {e}")
    
    # ========================================================================
    # 2. Préparer NDVI
    # ========================================================================
    ndvi = None
    
    # Essayer de lire NDVI existant
    if hasattr(las, 'ndvi'):
        ndvi = np.array(las.ndvi)
        logger.info(f"📊 Using existing NDVI from LAZ (range: {ndvi.min():.3f} to {ndvi.max():.3f})")
    
    # Calculer NDVI depuis RGB/NIR
    elif hasattr(las, 'red') and hasattr(las, 'green'):
        nir = None
        for field in ['nir', 'infrared', 'near_infrared']:
            if hasattr(las, field):
                nir = np.array(getattr(las, field))
                break
        
        if nir is not None:
            logger.info("📊 Computing NDVI from LAZ RGB/NIR...")
            # Normaliser
            rgb = np.vstack([
                np.array(las.red) / 65535.0,
                np.array(las.green) / 65535.0,
                np.array(las.blue) / 65535.0
            ]).T
            nir = nir / 65535.0 if nir.max() > 1.0 else nir
            
            ndvi = compute_ndvi(rgb, nir)
            logger.info(f"  ✓ NDVI computed (range: {ndvi.min():.3f} to {ndvi.max():.3f})")
    
    # Récupérer RGB/NIR si demandé
    elif fetch_rgb_nir:
        logger.info("🌍 Fetching RGB/NIR from IGN orthophotos...")
        try:
            rgb_fetcher = IGNOrthophotoFetcher(cache_dir=cache_dir)
            rgb = rgb_fetcher.augment_points_with_rgb(points)
            if rgb is not None:
                rgb = rgb.astype(np.float32) / 255.0
                logger.info(f"  ✓ RGB fetched")
            
            nir_fetcher = IGNInfraredFetcher(cache_dir=cache_dir)
            nir = nir_fetcher.augment_points_with_infrared(points)
            if nir is not None:
                nir = nir.astype(np.float32) / 255.0
                logger.info(f"  ✓ NIR fetched")
            
            if rgb is not None and nir is not None:
                ndvi = compute_ndvi(rgb, nir)
                logger.info(f"  ✓ NDVI computed from fetched data (range: {ndvi.min():.3f} to {ndvi.max():.3f})")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to fetch RGB/NIR: {e}")
    
    # ========================================================================
    # 3. Récupérer Ground Truth et BD Forêt®
    # ========================================================================
    logger.info("🗺️  Fetching ground truth from IGN BD TOPO®...")
    fetcher = IGNGroundTruthFetcher(cache_dir=cache_dir)
    
    logger.info("🌲 Initializing BD Forêt® fetcher...")
    from ign_lidar.io.bd_foret import BDForetFetcher
    forest_fetcher = BDForetFetcher(cache_dir=cache_dir)
    
    # ========================================================================
    # 4. Classification Avancée avec Railways et BD Forêt®
    # ========================================================================
    logger.info("🎯 Starting advanced classification with railways and forest types...")
    labels, forest_attributes = classify_with_all_features(
        points=points,
        ground_truth_fetcher=fetcher,
        bd_foret_fetcher=forest_fetcher,
        bbox=bbox,
        ndvi=ndvi,
        height=height,
        normals=normals,
        planarity=planarity,
        curvature=curvature,
        intensity=np.array(las.intensity) / 65535.0 if hasattr(las, 'intensity') else None,
        return_number=np.array(las.return_number) if hasattr(las, 'return_number') else None,
        include_railways=True,   # 🚂 Include railway classification
        include_forest=True,     # 🌲 Include BD Forêt® forest types
        **classifier_kwargs
    )
    
    # ========================================================================
    # 5. Analyser les résultats de BD Forêt®
    # ========================================================================
    if forest_attributes:
        logger.info("🌲 Forest classification results:")
        
        # Compter les points avec type forestier
        veg_mask = np.isin(labels, [3, 4, 5])  # ASPRS vegetation codes
        n_veg = veg_mask.sum()
        
        forest_types = forest_attributes.get('forest_type', [])
        forest_labeled = sum(1 for t in forest_types if t and t != 'unknown')
        
        if forest_labeled > 0:
            pct = 100.0 * forest_labeled / n_veg if n_veg > 0 else 0
            logger.info(f"  {forest_labeled:,} / {n_veg:,} vegetation points labeled ({pct:.1f}%)")
            
            # Distribution des types
            from collections import Counter
            type_counts = Counter(t for t in forest_types if t and t != 'unknown')
            logger.info(f"  Forest type distribution:")
            for ftype, count in type_counts.most_common():
                pct = 100.0 * count / forest_labeled
                logger.info(f"    {ftype:20s}: {count:8,} ({pct:5.1f}%)")
            
            # Top essences
            species = forest_attributes.get('primary_species', [])
            species_counts = Counter(s for s in species if s and s != 'unknown')
            if species_counts:
                logger.info(f"  Top 5 tree species:")
                for species_name, count in species_counts.most_common(5):
                    logger.info(f"    {species_name:20s}: {count:8,} points")
    
    # ========================================================================
    # 6. Sauvegarder
    # ========================================================================
    logger.info(f"💾 Saving to: {output_path}")
    
    # Comparer avec classification originale
    if hasattr(las, 'classification'):
        original = np.array(las.classification)
        changes = (original != labels).sum()
        change_pct = 100.0 * changes / n_points
        logger.info(f"  Classification changes: {changes:,} ({change_pct:.1f}%)")
        
        # Statistiques par code ASPRS
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info(f"  Final distribution:")
        asprs_names = {
            1: 'Unclassified', 2: 'Ground', 3: 'Low Veg', 4: 'Medium Veg',
            5: 'High Veg', 6: 'Building', 9: 'Water', 10: 'Rail', 11: 'Road'
        }
        for label, count in zip(unique_labels, counts):
            name = asprs_names.get(label, f'Class {label}')
            pct = 100.0 * count / n_points
            logger.info(f"    {name:15s}: {count:10,} ({pct:5.1f}%)")
    
    # Mettre à jour la classification
    las.classification = labels
    
    # Sauvegarder les features calculées comme extra dimensions
    if height is not None:
        try:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name='height',
                type=np.float32,
                description='Height above ground'
            ))
            las.height = height.astype(np.float32)
        except:
            pass
    
    if ndvi is not None:
        try:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name='ndvi',
                type=np.float32,
                description='NDVI vegetation index'
            ))
            las.ndvi = ndvi.astype(np.float32)
        except:
            pass
    
    # Sauvegarder attributs forestiers si disponibles
    if forest_attributes:
        # Sauvegarder estimated_height comme extra dimension
        est_height = forest_attributes.get('estimated_height')
        if est_height is not None:
            try:
                las.add_extra_dim(laspy.ExtraBytesParams(
                    name='forest_height',
                    type=np.float32,
                    description='BD Forêt estimated tree height'
                ))
                las.forest_height = np.array(est_height, dtype=np.float32)
            except:
                pass
    
    # Écrire le fichier
    output_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(output_path))
    
    logger.info(f"✅ Done! Output: {output_path}")


if __name__ == '__main__':
    # Exemple d'utilisation
    classify_laz_file_advanced(
        input_path=Path('data/input/tile_0500_6275.laz'),
        output_path=Path('data/output/tile_0500_6275_classified.laz'),
        cache_dir=Path('cache'),
        fetch_rgb_nir=True,
        compute_geometric=True,
        k_neighbors=20,
        # Paramètres du classificateur
        road_buffer_tolerance=0.5,
        ndvi_veg_threshold=0.35,
        ndvi_building_threshold=0.15,
        height_low_veg_threshold=0.5,
        height_medium_veg_threshold=2.0,
        planarity_road_threshold=0.80,
        planarity_building_threshold=0.70
    )
