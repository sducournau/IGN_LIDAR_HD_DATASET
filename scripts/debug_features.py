#!/usr/bin/env python3
"""
Script de débogage pour vérifier quelles features sont calculées et sauvegardées.
"""
import laspy
import numpy as np
from pathlib import Path

def analyze_computed_features(enriched_laz_path):
    """Analyser les features dans un fichier LAZ enrichi."""
    
    print("🔍 ANALYSE DES FEATURES CALCULÉES")
    print("=" * 70)
    
    # Charger le fichier
    laz = laspy.read(enriched_laz_path)
    
    # Dimensions standard LAS
    standard_dims = [
        'X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns',
        'synthetic', 'key_point', 'withheld', 'overlap', 'scanner_channel',
        'scan_direction_flag', 'edge_of_flight_line', 'classification',
        'user_data', 'scan_angle', 'point_source_id', 'gps_time',
        'red', 'green', 'blue', 'nir'
    ]
    
    print(f"\n📁 Fichier: {Path(enriched_laz_path).name}")
    print(f"Points: {len(laz.points):,}")
    print(f"Point format: {laz.point_format.id}")
    print()
    
    # Toutes les dimensions
    all_dims = list(laz.point_format.dimension_names)
    extra_dims = [d for d in all_dims if d not in standard_dims]
    
    print(f"📊 Dimensions totales: {len(all_dims)}")
    print(f"   Standard: {len(all_dims) - len(extra_dims)}")
    print(f"   Extra:    {len(extra_dims)}")
    print()
    
    if extra_dims:
        print("✨ Dimensions EXTRA trouvées:")
        print("-" * 70)
        for i, dim in enumerate(sorted(extra_dims), 1):
            try:
                values = getattr(laz, dim)
                non_zero = (values != 0).sum() if isinstance(values, np.ndarray) else 0
                pct = (non_zero / len(laz.points) * 100) if len(laz.points) > 0 else 0
                vmin, vmax = values.min(), values.max()
                print(f"  {i:2}. {dim:30s} - Non-zéro: {non_zero:>10,} ({pct:>5.1f}%) | Range: [{vmin:.3f}, {vmax:.3f}]")
            except Exception as e:
                print(f"  {i:2}. {dim:30s} - Erreur: {e}")
    else:
        print("❌ AUCUNE dimension extra trouvée!")
    
    print()
    print("🔍 Features ATTENDUES (devraient être présentes):")
    print("-" * 70)
    
    expected_features = [
        # Géométriques de base
        'height_above_ground', 'planarity', 'verticality', 'curvature',
        'sphericity', 'linearity', 'anisotropy',
        # Normales
        'normal_x', 'normal_y', 'normal_z',
        # Spectrales
        'ndvi',
        # Building-specific
        'BuildingConfidence', 'IsWall', 'IsRoof', 'DistanceToPolygon',
        'AdaptiveExpanded', 'IntelligentRejected'
    ]
    
    missing = []
    present = []
    
    for feat in expected_features:
        if feat in all_dims:
            present.append(feat)
            print(f"  ✓ {feat}")
        else:
            missing.append(feat)
            print(f"  ❌ {feat} - MANQUANTE!")
    
    print()
    print(f"📈 Résumé: {len(present)}/{len(expected_features)} features présentes")
    
    if missing:
        print(f"\n⚠️  Features manquantes ({len(missing)}):")
        for feat in missing:
            print(f"     - {feat}")
    
    return extra_dims, missing

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        laz_path = sys.argv[1]
    else:
        # Défaut: fichier V2
        laz_path = "/mnt/d/ign/versailles_output_v2/LHD_FXX_0635_6857_PTS_C_LAMB93_IGN69_enriched.laz"
    
    analyze_computed_features(laz_path)
