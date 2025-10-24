#!/usr/bin/env python3
"""
Audit des features pour ASPRS, LOD2 et LOD3
Vérifie que toutes les features définies sont bien calculées
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ign_lidar.features.feature_modes import (
    FeatureMode, 
    ASPRS_FEATURES, 
    LOD2_FEATURES, 
    LOD3_FEATURES,
    FEATURE_DESCRIPTIONS
)

def audit_feature_mode(mode_name: str, feature_set: set, mode_enum: FeatureMode):
    """
    Audit un mode de features pour vérifier la complétude
    """
    print(f"\n{'=' * 80}")
    print(f"🔍 MODE: {mode_name} ({mode_enum.value})")
    print(f"{'=' * 80}")
    
    # Compter les features (xyz = 3)
    total_features = len(feature_set)
    if 'xyz' in feature_set:
        total_features += 2  # xyz compte pour 3
    
    print(f"\n📊 Total déclaré: {total_features} features")
    print(f"   (Set size: {len(feature_set)} items, xyz=3)")
    
    # Catégoriser les features
    categories = {
        'Coordinates': set(),
        'Normals': set(),
        'Curvature': set(),
        'Shape Descriptors': set(),
        'Eigenvalues': set(),
        'Heights': set(),
        'Building Scores': set(),
        'Architectural': set(),
        'Density': set(),
        'Spectral': set(),
        'Unknown': set()
    }
    
    for feat in feature_set:
        if feat == 'xyz':
            categories['Coordinates'].add(feat)
        elif 'normal' in feat.lower():
            categories['Normals'].add(feat)
        elif 'curvature' in feat.lower():
            categories['Curvature'].add(feat)
        elif feat in {'planarity', 'linearity', 'sphericity', 'roughness', 
                      'anisotropy', 'omnivariance'}:
            categories['Shape Descriptors'].add(feat)
        elif 'eigenvalue' in feat or 'eigenentropy' in feat or feat == 'sum_eigenvalues':
            categories['Eigenvalues'].add(feat)
        elif 'height' in feat.lower() or feat == 'vertical_std':
            categories['Heights'].add(feat)
        elif feat in {'verticality', 'horizontality', 'wall_score', 'roof_score'}:
            categories['Building Scores'].add(feat)
        elif feat in {'wall_likelihood', 'roof_likelihood', 'facade_score', 
                      'flat_roof_score', 'sloped_roof_score', 'steep_roof_score',
                      'opening_likelihood', 'structural_element_score',
                      'edge_strength', 'corner_likelihood', 'overhang_indicator',
                      'surface_roughness', 'edge_strength_enhanced'}:
            categories['Architectural'].add(feat)
        elif 'density' in feat or 'num_points' in feat or 'neighborhood' in feat:
            categories['Density'].add(feat)
        elif feat in {'red', 'green', 'blue', 'nir', 'ndvi'}:
            categories['Spectral'].add(feat)
        else:
            categories['Unknown'].add(feat)
    
    # Afficher par catégorie
    print("\n📋 Features par catégorie:")
    for category, features in categories.items():
        if features:
            print(f"\n  {category} ({len(features)}):")
            for feat in sorted(features):
                desc = FEATURE_DESCRIPTIONS.get(feat, "⚠️  Pas de description")
                status = "✅" if feat in FEATURE_DESCRIPTIONS else "❌"
                print(f"    {status} {feat:30s} - {desc}")
    
    # Vérifier les features manquantes dans FEATURE_DESCRIPTIONS
    undefined = feature_set - set(FEATURE_DESCRIPTIONS.keys())
    if undefined:
        print(f"\n⚠️  Features SANS description ({len(undefined)}):")
        for feat in sorted(undefined):
            print(f"    ❌ {feat}")
    
    # Recommandations selon le mode
    print(f"\n💡 Recommandations pour {mode_name}:")
    
    if mode_enum == FeatureMode.ASPRS_CLASSES:
        required_geometric = {'planarity', 'sphericity', 'curvature', 'verticality', 
                             'horizontality', 'density'}
        required_heights = {'height_above_ground'}
        required_normals = {'normal_x', 'normal_y', 'normal_z'}
        required_spectral = {'red', 'green', 'blue', 'nir', 'ndvi'}
        
        missing_geo = required_geometric - feature_set
        missing_height = required_heights - feature_set
        missing_normals = required_normals - feature_set
        missing_spectral = required_spectral - feature_set
        
        if not missing_geo and not missing_height and not missing_normals:
            print("   ✅ Toutes les features géométriques essentielles présentes")
        else:
            if missing_geo:
                print(f"   ⚠️  Features géométriques manquantes: {missing_geo}")
            if missing_height:
                print(f"   ⚠️  Features de hauteur manquantes: {missing_height}")
            if missing_normals:
                print(f"   ⚠️  Normales manquantes: {missing_normals}")
        
        if not missing_spectral:
            print("   ✅ Toutes les features spectrales présentes (RGB+NIR+NDVI)")
        else:
            print(f"   ℹ️  Features spectrales manquantes (optionnelles): {missing_spectral}")
    
    elif mode_enum == FeatureMode.LOD2_SIMPLIFIED:
        required = {'xyz', 'normal_z', 'planarity', 'height_above_ground', 
                   'verticality', 'horizontality', 'wall_likelihood', 'roof_likelihood'}
        missing = required - feature_set
        
        if not missing:
            print("   ✅ Toutes les features essentielles LOD2 présentes")
        else:
            print(f"   ⚠️  Features essentielles manquantes: {missing}")
    
    elif mode_enum == FeatureMode.LOD3_FULL:
        # LOD3 doit avoir tout
        required_categories = {
            'Normals': 3,
            'Shape Descriptors': 6,
            'Eigenvalues': 5,
            'Building Scores': 3,
            'Architectural': 4,
        }
        
        for cat, expected_count in required_categories.items():
            actual_count = len(categories[cat])
            if actual_count >= expected_count:
                print(f"   ✅ {cat}: {actual_count}/{expected_count} features")
            else:
                print(f"   ⚠️  {cat}: {actual_count}/{expected_count} features (INCOMPLET)")
    
    return feature_set


def compare_modes():
    """
    Compare les 3 modes et identifie les différences
    """
    print(f"\n{'=' * 80}")
    print("📊 COMPARAISON DES MODES")
    print(f"{'=' * 80}")
    
    asprs_count = len(ASPRS_FEATURES) + (2 if 'xyz' in ASPRS_FEATURES else 0)
    lod2_count = len(LOD2_FEATURES) + (2 if 'xyz' in LOD2_FEATURES else 0)
    lod3_count = len(LOD3_FEATURES) + (2 if 'xyz' in LOD3_FEATURES else 0)
    
    print(f"\n📈 Nombre de features:")
    print(f"   ASPRS:  {asprs_count} features")
    print(f"   LOD2:   {lod2_count} features")
    print(f"   LOD3:   {lod3_count} features")
    
    # Features communes
    common_all = ASPRS_FEATURES & LOD2_FEATURES & LOD3_FEATURES
    print(f"\n🔗 Features COMMUNES aux 3 modes ({len(common_all)}):")
    for feat in sorted(common_all):
        print(f"   ✅ {feat}")
    
    # Features uniques à LOD3
    lod3_unique = LOD3_FEATURES - ASPRS_FEATURES - LOD2_FEATURES
    if lod3_unique:
        print(f"\n🎯 Features UNIQUEMENT en LOD3 ({len(lod3_unique)}):")
        for feat in sorted(lod3_unique):
            desc = FEATURE_DESCRIPTIONS.get(feat, "Pas de description")
            print(f"   🔹 {feat:30s} - {desc}")
    
    # Features en LOD2 mais pas en ASPRS
    lod2_not_asprs = LOD2_FEATURES - ASPRS_FEATURES
    if lod2_not_asprs:
        print(f"\n📍 Features en LOD2 mais PAS en ASPRS ({len(lod2_not_asprs)}):")
        for feat in sorted(lod2_not_asprs):
            desc = FEATURE_DESCRIPTIONS.get(feat, "Pas de description")
            print(f"   🔸 {feat:30s} - {desc}")
    
    # Features en ASPRS mais pas en LOD2
    asprs_not_lod2 = ASPRS_FEATURES - LOD2_FEATURES
    if asprs_not_lod2:
        print(f"\n🔍 Features en ASPRS mais PAS en LOD2 ({len(asprs_not_lod2)}):")
        for feat in sorted(asprs_not_lod2):
            desc = FEATURE_DESCRIPTIONS.get(feat, "Pas de description")
            print(f"   🔹 {feat:30s} - {desc}")


def check_computation_coverage():
    """
    Vérifie que toutes les features déclarées peuvent être calculées
    """
    print(f"\n{'=' * 80}")
    print("🔧 VÉRIFICATION DE LA COUVERTURE DE CALCUL")
    print(f"{'=' * 80}")
    
    # Features qui doivent être calculées par FeatureComputer
    computed_by_computer = {
        'xyz', 'normal_x', 'normal_y', 'normal_z',
        'curvature', 'change_curvature',
        'planarity', 'linearity', 'sphericity', 'roughness', 
        'anisotropy', 'omnivariance',
        'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
        'sum_eigenvalues', 'eigenentropy',
        'height', 'height_above_ground', 'vertical_std',
        'verticality', 'horizontality',
    }
    
    # Features de densité (compute/density.py)
    computed_by_density = {
        'density', 'num_points_2m', 'neighborhood_extent',
        'height_extent_ratio',  # Computed in compute_extended_density_features
    }
    
    # Features calculées par architectural.py
    computed_by_architectural = {
        'wall_likelihood', 'roof_likelihood', 'facade_score',
        'flat_roof_score', 'sloped_roof_score', 'steep_roof_score',
        'opening_likelihood', 'structural_element_score',
        'edge_strength', 'corner_likelihood', 
        'overhang_indicator', 'surface_roughness',
        'edge_strength_enhanced',
    }
    
    # Features calculées par orchestrator (RGB/NIR)
    computed_by_orchestrator = {
        'red', 'green', 'blue', 'nir', 'ndvi'
    }
    
    # Features de densité (compute/density.py)
    computed_by_density = {
        'density', 'num_points_2m', 'neighborhood_extent',
        'height_extent_ratio',  # Computed in compute_extended_density_features
    }
    
    # Legacy features
    legacy_features = {
        'wall_score', 'roof_score',
        'legacy_edge_strength', 'legacy_corner_likelihood',
        'legacy_overhang_indicator', 'legacy_surface_roughness',
    }
    
    all_computable = (computed_by_computer | computed_by_architectural | 
                      computed_by_orchestrator | computed_by_density | legacy_features)
    
    print(f"\n📊 Features computables:")
    print(f"   Computer (géométrique):     {len(computed_by_computer)} features")
    print(f"   Density (densité/extent):   {len(computed_by_density)} features")
    print(f"   Architectural:              {len(computed_by_architectural)} features")
    print(f"   Orchestrator (RGB/NIR):     {len(computed_by_orchestrator)} features")
    print(f"   Legacy:                     {len(legacy_features)} features")
    print(f"   TOTAL:                      {len(all_computable)} features")
    
    # Vérifier chaque mode
    for mode_name, feature_set, mode_enum in [
        ("ASPRS", ASPRS_FEATURES, FeatureMode.ASPRS_CLASSES),
        ("LOD2", LOD2_FEATURES, FeatureMode.LOD2_SIMPLIFIED),
        ("LOD3", LOD3_FEATURES, FeatureMode.LOD3_FULL),
    ]:
        missing = feature_set - all_computable
        if missing:
            print(f"\n⚠️  {mode_name}: Features SANS implémentation ({len(missing)}):")
            for feat in sorted(missing):
                print(f"    ❌ {feat}")
        else:
            print(f"\n✅ {mode_name}: Toutes les features sont implémentées")


def main():
    """
    Audit complet des 3 modes de features
    """
    print("=" * 80)
    print("🔍 AUDIT DES FEATURES - ASPRS / LOD2 / LOD3")
    print("=" * 80)
    
    # Audit de chaque mode
    audit_feature_mode("ASPRS_CLASSES", ASPRS_FEATURES, FeatureMode.ASPRS_CLASSES)
    audit_feature_mode("LOD2_SIMPLIFIED", LOD2_FEATURES, FeatureMode.LOD2_SIMPLIFIED)
    audit_feature_mode("LOD3_FULL", LOD3_FEATURES, FeatureMode.LOD3_FULL)
    
    # Comparaison
    compare_modes()
    
    # Vérification de couverture
    check_computation_coverage()
    
    print(f"\n{'=' * 80}")
    print("✅ AUDIT TERMINÉ")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
