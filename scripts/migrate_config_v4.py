#!/usr/bin/env python3
"""
Migration Automatique IGN LiDAR HD v2.x/v3.0 → v4.0
==================================================

Script de migration automatique des configurations legacy vers le nouveau
schéma unifié v4.0.

Usage:
    python scripts/migrate_config_v4.py --input configs/config_old.yaml --output configs_v4/migrated.yaml
    python scripts/migrate_config_v4.py --batch configs/ --output-dir configs_v4/migrated/
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Mapping des paramètres v2.x → v4.0
V2_TO_V4_MAPPING = {
    # Processor section
    'processor.use_gpu': 'processing.use_gpu',
    'processor.lod_level': 'processing.lod_level', 
    'processor.processing_mode': 'processing.mode',
    'processor.architecture': 'processing.architecture',
    'processor.num_workers': 'processing.num_workers',
    'processor.batch_size': 'processing.gpu.features_batch_size',
    'processor.skip_existing': 'processing.skip_existing',
    
    # Reclassification
    'processor.reclassification.enabled': 'classification.enabled',
    'processor.reclassification.acceleration_mode': 'processing.gpu.reclassification_mode',
    'processor.reclassification.chunk_size': 'processing.gpu.reclassification_chunk_size',
    'processor.reclassification.gpu_chunk_size': 'processing.gpu.reclassification_chunk_size',
    'processor.reclassification.use_geometric_rules': 'classification.post_processing.use_geometric_rules',
    
    # Features section
    'features.use_gpu': 'processing.use_gpu',
    'features.gpu_batch_size': 'processing.gpu.features_batch_size',
    'features.vram_utilization_target': 'processing.gpu.vram_target',
    'features.num_cuda_streams': 'processing.gpu.cuda_streams',
    'features.k_neighbors': 'features.k_neighbors',
    'features.search_radius': 'features.search_radius',
    'features.compute_normals': 'features.compute_normals',
    'features.compute_planarity': 'features.compute_planarity',
    'features.compute_curvature': 'features.compute_curvature',
    'features.compute_linearity': 'features.compute_linearity',
    'features.compute_verticality': 'features.compute_verticality',
    'features.compute_height_above_ground': 'features.compute_height_above_ground',
    'features.use_rgb': 'features.use_rgb',
    'features.use_nir': 'features.use_nir',
    'features.compute_ndvi': 'features.compute_ndvi',
    'features.compute_architectural_features': 'features.compute_architectural_features',
    'features.normalize_xyz': 'features.normalize_xyz',
    'features.normalize_features': 'features.normalize_features',
    
    # Ground truth
    'ground_truth.enabled': 'classification.enabled',
    'ground_truth.update_classification': 'classification.enabled',
    'ground_truth.apply_reclassification': 'classification.enabled',
    'ground_truth.use_ndvi': 'features.compute_ndvi',
    'ground_truth.optimization.force_method': 'processing.gpu.ground_truth_method',
    'ground_truth.optimization.gpu_chunk_size': 'processing.gpu.ground_truth_chunk_size',
    
    # Préprocessing
    'preprocess.enabled': 'preprocess.enabled',
    'preprocess.sor_enabled': 'preprocess.statistical_outlier_removal.enabled',
    'preprocess.ror_enabled': 'preprocess.radius_outlier_removal.enabled',
    
    # Stitching
    'stitching.enabled': 'stitching.enabled',
    'stitching.buffer_size': 'stitching.buffer_size',
}

# Mapping des paramètres v3.0 → v4.0
V3_TO_V4_MAPPING = {
    # Processing section (v3.0 structure closer to v4.0)
    'processing.mode': 'processing.mode',
    'processing.lod_level': 'processing.lod_level',
    'processing.architecture': 'processing.architecture', 
    'processing.use_gpu': 'processing.use_gpu',
    'processing.num_workers': 'processing.num_workers',
    
    # Data sources (v3.0 → v4.0 data_sources)
    'data_sources.bd_topo_enabled': 'data_sources.bd_topo_enabled',
    'data_sources.bd_topo_buildings': 'data_sources.bd_topo_buildings',
    'data_sources.bd_topo_roads': 'data_sources.bd_topo_roads',
    'data_sources.bd_topo_water': 'data_sources.bd_topo_water',
    'data_sources.bd_topo_vegetation': 'data_sources.bd_topo_vegetation',
    'data_sources.cadastre_enabled': 'data_sources.cadastre_enabled',
    'data_sources.bd_foret_enabled': 'data_sources.bd_foret_enabled',
    'data_sources.rpg_enabled': 'data_sources.rpg_enabled',
}

# Paramètres obsolètes (ignorés lors de la migration)
DEPRECATED_PARAMS = {
    'processor.augment',
    'processor.num_augmentations', 
    'processor.prefetch_factor',
    'processor.pin_memory',
    'features.gpu_optimization.enable_mixed_precision',  # Déplacé vers processing.gpu
    'features.cpu_optimization',  # Déplacé vers optimization
    'ground_truth.fetch_rgb_nir',  # Obsolète
    'performance',  # Déplacé vers monitoring
}

def load_yaml_safe(file_path: Path) -> Dict[str, Any]:
    """Charge un fichier YAML de manière sécurisée."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            return content if content is not None else {}
    except Exception as e:
        print(f"❌ Erreur lecture {file_path}: {e}")
        return {}

def save_yaml(data: Dict[str, Any], file_path: Path) -> bool:
    """Sauvegarde un dictionnaire en YAML."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
        return True
    except Exception as e:
        print(f"❌ Erreur écriture {file_path}: {e}")
        return False

def set_nested_value(obj: Dict[str, Any], path: str, value: Any) -> None:
    """Définis une valeur imbriquée dans un dictionnaire."""
    keys = path.split('.')
    current = obj
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value

def get_nested_value(obj: Dict[str, Any], path: str) -> Any:
    """Récupère une valeur imbriquée dans un dictionnaire."""
    keys = path.split('.')
    current = obj
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    
    return current

def detect_schema_version(config: Dict[str, Any]) -> str:
    """Détecte la version du schéma de configuration."""
    
    # Check for explicit version
    if 'config_version' in config:
        return config['config_version']
    
    # Detect by structure
    if 'processing' in config and 'data_sources' in config:
        return "v3.0"
    elif 'processor' in config and 'features' in config:
        return "v2.x"
    else:
        return "unknown"

def migrate_v2_to_v4(config: Dict[str, Any]) -> Dict[str, Any]:
    """Migre une configuration v2.x vers v4.0."""
    
    migrated = {
        'config_version': '4.0.0',
        'config_name': 'migrated_from_v2',
        'config_description': f'Configuration migrée de v2.x le {datetime.now().strftime("%Y-%m-%d")}',
    }
    
    # Migration des paramètres mappés
    for old_path, new_path in V2_TO_V4_MAPPING.items():
        value = get_nested_value(config, old_path)
        if value is not None:
            set_nested_value(migrated, new_path, value)
    
    # Paramètres spéciaux pour v2.x
    
    # Mode de processing
    processing_mode = get_nested_value(config, 'processor.processing_mode')
    if processing_mode == 'enriched_only':
        set_nested_value(migrated, 'processing.mode', 'enriched_only')
        set_nested_value(migrated, 'output.save_patches', False)
    elif processing_mode == 'patches_only':
        set_nested_value(migrated, 'processing.mode', 'patches_only')
        set_nested_value(migrated, 'output.save_enriched', False)
    
    # Correction des valeurs d'accélération
    accel_mode = get_nested_value(config, 'processor.reclassification.acceleration_mode')
    if accel_mode == 'cpu':
        set_nested_value(migrated, 'processing.gpu.reclassification_mode', 'cpu')
    elif accel_mode in ['auto', 'gpu']:
        set_nested_value(migrated, 'processing.gpu.reclassification_mode', 'auto')
    
    # Data sources depuis v2.x structure
    if 'data_sources' in config:
        ds = config['data_sources']
        if isinstance(ds, dict):
            for key, value in ds.items():
                if key.startswith('bd_topo') or key in ['cadastre_enabled', 'bd_foret_enabled', 'rpg_enabled']:
                    set_nested_value(migrated, f'data_sources.{key}', value)
    
    return migrated

def migrate_v3_to_v4(config: Dict[str, Any]) -> Dict[str, Any]:
    """Migre une configuration v3.0 vers v4.0."""
    
    migrated = {
        'config_version': '4.0.0',
        'config_name': 'migrated_from_v3',
        'config_description': f'Configuration migrée de v3.0 le {datetime.now().strftime("%Y-%m-%d")}',
    }
    
    # Migration des paramètres mappés
    for old_path, new_path in V3_TO_V4_MAPPING.items():
        value = get_nested_value(config, old_path)
        if value is not None:
            set_nested_value(migrated, new_path, value)
    
    # Consolider les paramètres GPU en processing.gpu
    gpu_params = {}
    
    # Features GPU → processing.gpu
    features = config.get('features', {})
    if isinstance(features, dict):
        if 'gpu_batch_size' in features:
            gpu_params['features_batch_size'] = features['gpu_batch_size']
        if 'vram_utilization_target' in features:
            gpu_params['vram_target'] = features['vram_utilization_target']
        if 'num_cuda_streams' in features:
            gpu_params['cuda_streams'] = features['num_cuda_streams']
    
    # Ground truth GPU
    gt_opt = get_nested_value(config, 'ground_truth.optimization')
    if gt_opt:
        if 'force_method' in gt_opt:
            gpu_params['ground_truth_method'] = gt_opt['force_method']
        if 'gpu_chunk_size' in gt_opt:
            gpu_params['ground_truth_chunk_size'] = gt_opt['gpu_chunk_size']
    
    # Reclassification GPU
    reclas = get_nested_value(config, 'processor.reclassification')
    if reclas:
        if 'acceleration_mode' in reclas:
            mode = reclas['acceleration_mode']
            gpu_params['reclassification_mode'] = 'cpu' if mode == 'cpu' else 'auto'
        if 'gpu_chunk_size' in reclas:
            gpu_params['reclassification_chunk_size'] = reclas['gpu_chunk_size']
    
    # Appliquer les paramètres GPU
    if gpu_params:
        set_nested_value(migrated, 'processing.gpu', gpu_params)
    
    return migrated

def migrate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Migre une configuration vers v4.0 selon sa version détectée."""
    
    schema_version = detect_schema_version(config)
    
    if schema_version == "v2.x":
        return migrate_v2_to_v4(config)
    elif schema_version == "v3.0":
        return migrate_v3_to_v4(config)
    elif schema_version.startswith("4."):
        print("⚠️  Configuration déjà en v4.0, pas de migration nécessaire")
        return config
    else:
        print(f"⚠️  Version de schéma inconnue: {schema_version}, migration générique")
        return migrate_v2_to_v4(config)  # Tentative migration v2.x par défaut

def add_migration_metadata(migrated: Dict[str, Any], original_file: Path) -> Dict[str, Any]:
    """Ajoute des métadonnées de migration."""
    
    migrated['_migration'] = {
        'original_file': str(original_file),
        'migration_date': datetime.now().isoformat(),
        'migration_tool': 'migrate_config_v4.py',
        'notes': [
            'Configuration migrée automatiquement vers le schéma v4.0',
            'Vérifiez les paramètres GPU et ajustez selon votre hardware',
            'Testez sur un petit dataset avant utilisation en production'
        ]
    }
    
    return migrated

def main():
    parser = argparse.ArgumentParser(description="Migration automatique configurations v2.x/v3.0 → v4.0")
    parser.add_argument('--input', '-i', type=Path, help='Fichier de configuration à migrer')
    parser.add_argument('--output', '-o', type=Path, help='Fichier de sortie')
    parser.add_argument('--batch', type=Path, help='Répertoire de configurations à migrer en lot')
    parser.add_argument('--output-dir', type=Path, help='Répertoire de sortie pour migration en lot')
    parser.add_argument('--dry-run', action='store_true', help='Affichage uniquement, pas de sauvegarde')
    
    args = parser.parse_args()
    
    if args.input and args.output:
        # Migration fichier unique
        print(f"🔄 Migration: {args.input} → {args.output}")
        
        config = load_yaml_safe(args.input)
        if not config:
            print(f"❌ Impossible de charger {args.input}")
            return 1
        
        migrated = migrate_config(config)
        migrated = add_migration_metadata(migrated, args.input)
        
        if args.dry_run:
            print("📋 Configuration migrée (dry-run):")
            print(yaml.dump(migrated, default_flow_style=False, indent=2))
        else:
            if save_yaml(migrated, args.output):
                print(f"✅ Migration réussie: {args.output}")
            else:
                print(f"❌ Échec sauvegarde: {args.output}")
                return 1
    
    elif args.batch and args.output_dir:
        # Migration en lot
        print(f"🔄 Migration en lot: {args.batch} → {args.output_dir}")
        
        config_files = list(args.batch.glob('*.yaml')) + list(args.batch.glob('*.yml'))
        print(f"📁 Fichiers trouvés: {len(config_files)}")
        
        success_count = 0
        
        for config_file in config_files:
            print(f"  📄 {config_file.name}")
            
            config = load_yaml_safe(config_file)
            if not config:
                print(f"    ❌ Erreur chargement")
                continue
            
            migrated = migrate_config(config)
            migrated = add_migration_metadata(migrated, config_file)
            
            output_file = args.output_dir / f"migrated_{config_file.name}"
            
            if args.dry_run:
                print(f"    📋 Dry-run: {output_file}")
            else:
                if save_yaml(migrated, output_file):
                    print(f"    ✅ {output_file}")
                    success_count += 1
                else:
                    print(f"    ❌ Échec: {output_file}")
        
        print(f"🏁 Migration terminée: {success_count}/{len(config_files)} réussies")
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())