#!/usr/bin/env python3
"""
Audit des Configurations IGN LiDAR HD
====================================

Script d'analyse pour identifier la fragmentation des configurations
et préparer la consolidation vers le schéma v4.0 unifié.

Usage:
    python scripts/audit_configs.py --output config_audit.json
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
import argparse

def find_config_files(base_dir: Path) -> List[Path]:
    """Trouve tous les fichiers de configuration YAML dans le projet."""
    config_files = []
    
    # Patterns de recherche
    patterns = ['*.yaml', '*.yml']
    
    for pattern in patterns:
        # Configs du répertoire principal
        config_files.extend(base_dir.glob(f'configs/{pattern}'))
        # Configs dans ign_lidar/configs
        config_files.extend(base_dir.glob(f'ign_lidar/configs/**/{pattern}'))
        # Configs d'exemples
        config_files.extend(base_dir.glob(f'examples/{pattern}'))
    
    return sorted(config_files)

def load_yaml_safe(file_path: Path) -> Dict[str, Any]:
    """Charge un fichier YAML de manière sécurisée."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            return content if content is not None else {}
    except Exception as e:
        print(f"⚠️  Erreur lecture {file_path}: {e}")
        return {}

def extract_config_params(config: Dict[str, Any], prefix: str = "") -> Set[str]:
    """Extrait tous les paramètres d'une configuration de manière récursive."""
    params = set()
    
    def _extract_recursive(obj: Any, path: str):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                params.add(current_path)
                _extract_recursive(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _extract_recursive(item, f"{path}[{i}]" if path else f"[{i}]")
    
    _extract_recursive(config, prefix)
    return params

def detect_schema_version(config: Dict[str, Any]) -> str:
    """Détecte la version du schéma de configuration."""
    
    # Indicateurs de schéma v3.0
    v3_indicators = {
        'processing.mode',
        'processing.architecture', 
        'processing.use_gpu',
        'data_sources.bd_topo_enabled'
    }
    
    # Indicateurs de schéma v2.x
    v2_indicators = {
        'processor.use_gpu',
        'processor.lod_level',
        'processor.processing_mode',
        'features.gpu_batch_size'
    }
    
    params = extract_config_params(config)
    
    v3_score = sum(1 for indicator in v3_indicators if indicator in params)
    v2_score = sum(1 for indicator in v2_indicators if indicator in params)
    
    if v3_score > v2_score:
        return "v3.0"
    elif v2_score > 0:
        return "v2.x"
    else:
        return "unknown"

def analyze_gpu_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyse les paramètres GPU d'une configuration."""
    
    gpu_analysis = {
        "gpu_enabled": False,
        "features_batch_size": None,
        "vram_target": None,
        "acceleration_mode": None,
        "ground_truth_method": None,
        "issues": []
    }
    
    # Extraire les paramètres GPU
    params = extract_config_params(config)
    
    # GPU activé ?
    gpu_indicators = ['processor.use_gpu', 'processing.use_gpu', 'features.use_gpu']
    for indicator in gpu_indicators:
        if indicator in params:
            value = get_nested_value(config, indicator.split('.'))
            if value is True:
                gpu_analysis["gpu_enabled"] = True
                break
    
    # Batch size
    batch_indicators = ['features.gpu_batch_size', 'processor.batch_size']
    for indicator in batch_indicators:
        if indicator in params:
            value = get_nested_value(config, indicator.split('.'))
            if isinstance(value, (int, str)):
                gpu_analysis["features_batch_size"] = str(value).replace('_', '')
                break
    
    # VRAM target
    vram_indicators = ['features.vram_utilization_target', 'features.vram_target']
    for indicator in vram_indicators:
        if indicator in params:
            value = get_nested_value(config, indicator.split('.'))
            gpu_analysis["vram_target"] = value
            break
    
    # Mode d'accélération
    accel_indicators = ['processor.reclassification.acceleration_mode', 'reclassification.acceleration_mode']
    for indicator in accel_indicators:
        if indicator in params:
            value = get_nested_value(config, indicator.split('.'))
            gpu_analysis["acceleration_mode"] = value
            break
    
    # Méthode ground truth
    gt_indicators = ['ground_truth.optimization.force_method', 'processor.optimization.force_method']
    for indicator in gt_indicators:
        if indicator in params:
            value = get_nested_value(config, indicator.split('.'))
            gpu_analysis["ground_truth_method"] = value
            break
    
    # Détecter les problèmes
    if gpu_analysis["gpu_enabled"]:
        if gpu_analysis["acceleration_mode"] == "cpu":
            gpu_analysis["issues"].append("Reclassification forcée en mode CPU")
        
        if gpu_analysis["ground_truth_method"] in ["strtree", "auto"]:
            gpu_analysis["issues"].append("Ground truth en mode CPU fallback")
        
        batch_size = gpu_analysis["features_batch_size"]
        if batch_size and batch_size.isdigit() and int(batch_size) < 4000000:
            gpu_analysis["issues"].append("Batch size GPU sous-optimal (<4M)")
        
        vram = gpu_analysis["vram_target"]
        if vram and vram < 0.8:
            gpu_analysis["issues"].append("Utilisation VRAM conservative (<80%)")
    
    return gpu_analysis

def get_nested_value(obj: Dict[str, Any], keys: List[str]) -> Any:
    """Récupère une valeur imbriquée dans un dictionnaire."""
    current = obj
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def categorize_config_file(file_path: Path) -> str:
    """Catégorise un fichier de configuration."""
    name = file_path.name.lower()
    
    if 'rtx' in name or 'gpu' in name:
        return "gpu_optimized"
    elif 'asprs' in name:
        return "asprs_specific"
    elif 'default' in name:
        return "default"
    elif 'enrichment' in name:
        return "enrichment"
    elif 'classification' in name or 'reclassification' in name:
        return "classification"
    elif 'processing' in name:
        return "processing"
    elif file_path.parent.name in ['presets', 'multiscale']:
        return file_path.parent.name
    else:
        return "other"

def main():
    parser = argparse.ArgumentParser(description="Audit des configurations IGN LiDAR HD")
    parser.add_argument('--output', '-o', default='config_audit.json', 
                       help='Fichier de sortie JSON')
    parser.add_argument('--project-dir', default='.', 
                       help='Répertoire racine du projet')
    
    args = parser.parse_args()
    
    project_dir = Path(args.project_dir).resolve()
    output_file = Path(args.output)
    
    print("🔍 Audit des Configurations IGN LiDAR HD")
    print("=" * 50)
    print(f"Répertoire projet: {project_dir}")
    print(f"Sortie: {output_file}")
    print()
    
    # 1. Trouver tous les fichiers de configuration
    config_files = find_config_files(project_dir)
    print(f"📁 Fichiers trouvés: {len(config_files)}")
    
    # 2. Analyser chaque fichier
    analysis = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "project_dir": str(project_dir),
            "total_files": len(config_files)
        },
        "files": {},
        "summary": {
            "by_category": {},
            "by_schema": {},
            "gpu_issues": [],
            "duplicated_params": {},
            "recommendations": []
        }
    }
    
    all_params = {}  # param -> [files]
    
    for file_path in config_files:
        print(f"  📄 {file_path.relative_to(project_dir)}")
        
        # Charger et analyser
        config = load_yaml_safe(file_path)
        params = extract_config_params(config)
        schema_version = detect_schema_version(config)
        gpu_analysis = analyze_gpu_settings(config)
        category = categorize_config_file(file_path)
        
        # Stocker l'analyse
        rel_path = str(file_path.relative_to(project_dir))
        analysis["files"][rel_path] = {
            "category": category,
            "schema_version": schema_version,
            "param_count": len(params),
            "parameters": sorted(list(params)),
            "gpu_analysis": gpu_analysis,
            "file_size": file_path.stat().st_size if file_path.exists() else 0
        }
        
        # Collecter les paramètres pour détecter la duplication
        for param in params:
            if param not in all_params:
                all_params[param] = []
            all_params[param].append(rel_path)
        
        # Collecter les statistiques
        if category not in analysis["summary"]["by_category"]:
            analysis["summary"]["by_category"][category] = 0
        analysis["summary"]["by_category"][category] += 1
        
        if schema_version not in analysis["summary"]["by_schema"]:
            analysis["summary"]["by_schema"][schema_version] = 0
        analysis["summary"]["by_schema"][schema_version] += 1
        
        # Collecter les problèmes GPU
        if gpu_analysis["issues"]:
            analysis["summary"]["gpu_issues"].extend([
                f"{rel_path}: {issue}" for issue in gpu_analysis["issues"]
            ])
    
    # 3. Analyser la duplication
    duplicated_params = {
        param: files for param, files in all_params.items() 
        if len(files) > 3  # Paramètre dans plus de 3 fichiers
    }
    analysis["summary"]["duplicated_params"] = duplicated_params
    
    # 4. Générer des recommandations
    recommendations = []
    
    # Schémas multiples
    schema_count = len(analysis["summary"]["by_schema"])
    if schema_count > 1:
        recommendations.append(f"URGENT: {schema_count} versions de schéma différentes détectées")
    
    # Problèmes GPU
    gpu_issue_count = len(analysis["summary"]["gpu_issues"])
    if gpu_issue_count > 0:
        recommendations.append(f"PERFORMANCE: {gpu_issue_count} problèmes GPU détectés")
    
    # Fragmentation
    category_count = len(analysis["summary"]["by_category"])
    if category_count > 6:
        recommendations.append(f"ARCHITECTURE: {category_count} catégories de configs (trop fragmenté)")
    
    # Duplication
    duplicated_count = len(duplicated_params)
    if duplicated_count > 20:
        recommendations.append(f"MAINTENANCE: {duplicated_count} paramètres dupliqués")
    
    analysis["summary"]["recommendations"] = recommendations
    
    # 5. Sauvegarder l'audit
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # 6. Afficher le résumé
    print()
    print("📊 Résumé de l'Audit")
    print("=" * 20)
    print(f"Total fichiers: {len(config_files)}")
    print(f"Catégories: {list(analysis['summary']['by_category'].keys())}")
    print(f"Schémas: {list(analysis['summary']['by_schema'].keys())}")
    print(f"Paramètres dupliqués: {len(duplicated_params)}")
    print(f"Problèmes GPU: {len(analysis['summary']['gpu_issues'])}")
    
    print()
    print("🚨 Recommandations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    if not recommendations:
        print("  ✅ Aucun problème majeur détecté")
    
    print()
    print(f"💾 Audit complet sauvé: {output_file}")
    print("📖 Consultez le fichier JSON pour les détails complets")

if __name__ == "__main__":
    main()