"""
Script d'exemple : Classification Complète avec Toutes les Sources de Données

Ce script montre comment utiliser le système de classification avec :
- BD TOPO® V3 (infrastructure complète)
- BD Forêt® V2 (types de forêts)
- RPG (parcelles agricoles et cultures)
- BD PARCELLAIRE (cadastre)

Author: Data Integration Team
Date: October 15, 2025
"""

from pathlib import Path
import numpy as np
import laspy
import logging
from collections import Counter

# Import du fetcher de données
from ign_lidar.io.data_fetcher import (
    DataFetcher,
    DataFetchConfig,
    create_full_fetcher
)

# Import de la classification avancée
from ign_lidar.core.modules.advanced_classification import (
    classify_with_all_features,
    AdvancedClassifier
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def classify_with_full_system(
    input_path: Path,
    output_path: Path,
    cache_dir: Path,
    config: DataFetchConfig = None,
    save_parcel_stats: bool = True
):
    """
    Classifier un fichier LAZ avec le système complet.
    
    Args:
        input_path: Chemin du fichier LAZ d'entrée
        output_path: Chemin du fichier LAZ de sortie
        cache_dir: Répertoire de cache
        config: Configuration du fetcher (None = toutes features activées)
        save_parcel_stats: Sauvegarder statistiques par parcelle
    """
    logger.info("="*80)
    logger.info("Classification Unifiée IGN LiDAR HD")
    logger.info("="*80)
    
    # ========================================================================
    # 1. Charger le fichier LAZ
    # ========================================================================
    logger.info(f"📂 Loading: {input_path}")
    las = laspy.read(str(input_path))
    
    points = np.vstack([las.x, las.y, las.z]).T
    n_points = len(points)
    logger.info(f"  Loaded {n_points:,} points")
    
    # Bounding box
    bbox = (
        float(las.header.x_min),
        float(las.header.y_min),
        float(las.header.x_max),
        float(las.header.y_max)
    )
    logger.info(f"  Bounding box: {bbox}")
    
    # ========================================================================
    # 2. Calculer/récupérer features
    # ========================================================================
    logger.info("🔧 Preparing features...")
    
    # NDVI
    ndvi = None
    if hasattr(las, 'red') and hasattr(las, 'nir'):
        logger.info("  Computing NDVI from RGB/NIR...")
        rgb = np.vstack([
            np.array(las.red) / 65535.0,
            np.array(las.green) / 65535.0,
            np.array(las.blue) / 65535.0
        ]).T
        nir = np.array(las.nir) / 65535.0
        
        from ign_lidar.core.modules.enrichment import compute_ndvi
        ndvi = compute_ndvi(rgb, nir)
        logger.info(f"    NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")
    
    # Height (si disponible)
    height = None
    if hasattr(las, 'height'):
        height = np.array(las.height)
        logger.info(f"  Height available: {height.min():.1f}m to {height.max():.1f}m")
    
    # ========================================================================
    # 3. Initialiser le système de données
    # ========================================================================
    logger.info("🌍 Initializing data fetcher...")
    
    if config is None:
        # Configuration complète par défaut
        config = DataFetchConfig(
            # BD TOPO® complet
            include_buildings=True,
            include_roads=True,
            include_railways=True,
            include_water=True,
            include_vegetation=True,
            include_bridges=True,
            include_parking=True,
            include_cemeteries=True,
            include_power_lines=True,
            include_sports=True,
            # Autres sources
            include_forest=True,
            include_agriculture=True,
            include_cadastre=True,
            group_by_parcel=True
        )
    
    fetcher = DataFetcher(
        cache_dir=cache_dir,
        config=config
    )
    
    # ========================================================================
    # 4. Récupérer toutes les données
    # ========================================================================
    logger.info("📥 Fetching all geographic data...")
    data = fetcher.fetch_all(bbox=bbox, use_cache=True)
    
    # ========================================================================
    # 5. Classification avec toutes les sources
    # ========================================================================
    logger.info("🎯 Starting unified classification...")
    
    labels, forest_attrs, rpg_attrs = classify_with_all_features(
        points=points,
        ground_truth_fetcher=fetcher.ground_truth_fetcher,
        bd_foret_fetcher=fetcher.forest_fetcher,
        rpg_fetcher=fetcher.rpg_fetcher,
        bbox=bbox,
        ndvi=ndvi,
        height=height,
        # Activer toutes les features
        include_railways=config.include_railways,
        include_forest=config.include_forest,
        include_agriculture=config.include_agriculture,
        include_bridges=config.include_bridges,
        include_parking=config.include_parking,
        include_sports=config.include_sports
    )
    
    # ========================================================================
    # 6. Traiter le cadastre
    # ========================================================================
    cadastre_groups = None
    cadastre_labels = None
    
    if config.include_cadastre and data['cadastre'] is not None:
        logger.info("🗺️  Processing cadastral parcels...")
        
        # Grouper par parcelle
        if config.group_by_parcel:
            cadastre_groups = fetcher.cadastre_fetcher.group_points_by_parcel(
                points=points,
                parcels_gdf=data['cadastre'],
                labels=labels
            )
        
        # Labelliser avec ID parcelle
        cadastre_labels = fetcher.cadastre_fetcher.label_points_with_parcel_id(
            points=points,
            parcels_gdf=data['cadastre']
        )
    
    # ========================================================================
    # 7. Analyser les résultats
    # ========================================================================
    logger.info("📊 Analysis Results:")
    logger.info("="*80)
    
    # Distribution des classes ASPRS
    unique, counts = np.unique(labels, return_counts=True)
    class_names = {
        1: 'Unclassified',
        2: 'Ground',
        3: 'Low Vegetation',
        4: 'Medium Vegetation',
        5: 'High Vegetation',
        6: 'Building',
        9: 'Water',
        10: 'Rail',
        11: 'Road',
        17: 'Bridge',
        40: 'Parking',
        41: 'Sports',
        42: 'Cemetery',
        43: 'Power Line',
        44: 'Agriculture'
    }
    
    logger.info("Classification distribution:")
    for code, count in zip(unique, counts):
        name = class_names.get(code, f'Class {code}')
        pct = 100.0 * count / n_points
        logger.info(f"  {name:20s}: {count:10,} ({pct:5.1f}%)")
    
    # Statistiques forêts
    if forest_attrs:
        logger.info("\n🌲 Forest Statistics:")
        forest_types = [ft for ft in forest_attrs.get('forest_type', []) if ft != 'unknown']
        if forest_types:
            type_counts = Counter(forest_types)
            for ftype, count in type_counts.most_common(5):
                pct = 100.0 * count / len(forest_types)
                logger.info(f"  {ftype:20s}: {count:8,} ({pct:5.1f}%)")
    
    # Statistiques agriculture
    if rpg_attrs:
        logger.info("\n🌾 Agriculture Statistics:")
        n_agri = sum(rpg_attrs.get('is_agricultural', []))
        if n_agri > 0:
            logger.info(f"  Agricultural points: {n_agri:,}")
            
            crop_cats = [c for c in rpg_attrs.get('crop_category', []) if c != 'unknown']
            if crop_cats:
                cat_counts = Counter(crop_cats)
                logger.info(f"  Crop categories:")
                for cat, count in cat_counts.most_common():
                    pct = 100.0 * count / len(crop_cats)
                    logger.info(f"    {cat:20s}: {count:8,} ({pct:5.1f}%)")
            
            n_bio = sum(rpg_attrs.get('is_organic', []))
            if n_bio > 0:
                pct = 100.0 * n_bio / n_agri
                logger.info(f"  Organic farming: {n_bio:,} points ({pct:.1f}%)")
    
    # Statistiques cadastre
    if cadastre_groups:
        logger.info("\n🗺️  Cadastre Statistics:")
        logger.info(f"  Number of parcels: {len(cadastre_groups)}")
        
        # Densité de points par parcelle
        densities = [p['point_density'] for p in cadastre_groups.values()]
        logger.info(f"  Point density range: {min(densities):.1f} - {max(densities):.1f} pts/m²")
        
        # Points par parcelle
        points_per_parcel = [p['n_points'] for p in cadastre_groups.values()]
        logger.info(f"  Points per parcel: {min(points_per_parcel)} - {max(points_per_parcel)}")
        logger.info(f"  Average: {int(np.mean(points_per_parcel))} points/parcel")
    
    if cadastre_labels:
        n_assigned = sum(1 for pl in cadastre_labels if pl != 'unassigned')
        pct = 100.0 * n_assigned / len(cadastre_labels)
        logger.info(f"  Points assigned to parcels: {n_assigned:,} ({pct:.1f}%)")
    
    # ========================================================================
    # 8. Sauvegarder les résultats
    # ========================================================================
    logger.info("\n💾 Saving results...")
    
    # Mettre à jour la classification
    las.classification = labels
    
    # Ajouter attributs comme dimensions extra
    try:
        if ndvi is not None:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name='ndvi',
                type=np.float32,
                description='NDVI vegetation index'
            ))
            las.ndvi = ndvi.astype(np.float32)
    except:
        pass
    
    try:
        if height is not None:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name='height',
                type=np.float32,
                description='Height above ground'
            ))
            las.height = height.astype(np.float32)
    except:
        pass
    
    # Sauvegarder LAZ
    las.write(str(output_path))
    logger.info(f"  ✓ Saved classified LAZ: {output_path}")
    
    # Sauvegarder statistiques par parcelle
    if save_parcel_stats and cadastre_groups and data['cadastre'] is not None:
        stats_path = output_path.parent / f"{output_path.stem}_parcel_stats.geojson"
        
        from ign_lidar.io.cadastre import export_parcel_groups_to_geojson
        export_parcel_groups_to_geojson(
            parcel_groups=cadastre_groups,
            parcels_gdf=data['cadastre'],
            output_path=stats_path
        )
        logger.info(f"  ✓ Saved parcel statistics: {stats_path}")
    
    # Sauvegarder attributs détaillés (CSV)
    if forest_attrs or rpg_attrs or cadastre_labels:
        import pandas as pd
        
        csv_path = output_path.parent / f"{output_path.stem}_attributes.csv"
        
        df_data = {
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2],
            'classification': labels
        }
        
        if forest_attrs:
            df_data['forest_type'] = forest_attrs.get('forest_type', [])
            df_data['primary_species'] = forest_attrs.get('primary_species', [])
        
        if rpg_attrs:
            df_data['crop_code'] = rpg_attrs.get('crop_code', [])
            df_data['crop_category'] = rpg_attrs.get('crop_category', [])
            df_data['is_organic'] = rpg_attrs.get('is_organic', [])
        
        if cadastre_labels:
            df_data['parcel_id'] = cadastre_labels
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"  ✓ Saved detailed attributes: {csv_path}")
    
    logger.info("="*80)
    logger.info("✅ Classification complete!")
    logger.info("="*80)


if __name__ == "__main__":
    # Configuration des chemins
    INPUT_FILE = Path("data/test_tile.laz")
    OUTPUT_FILE = Path("data/test_tile_classified_unified.laz")
    CACHE_DIR = Path("cache")
    
    # Créer répertoires si nécessaire
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Exécuter la classification
    classify_with_full_system(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        cache_dir=CACHE_DIR,
        config=None,  # Configuration complète par défaut
        save_parcel_stats=True
    )
