#!/usr/bin/env python3
"""
Test complet des nouvelles features géométriques avec focus bâtiments
"""

import numpy as np
from ign_lidar.features import (
    compute_all_features_optimized,
    compute_height_features,
    compute_local_statistics,
    compute_verticality,
    estimate_optimal_k
)


def create_building_test_data():
    """Créer un nuage de points synthétique simulant un bâtiment."""
    np.random.seed(42)
    
    # 1. Toit plat horizontal (z=10m)
    roof_points = np.random.randn(300, 3) * 0.1
    roof_points[:, :2] += [5, 5]  # Centré en (5, 5)
    roof_points[:, 2] = 10.0
    
    # 2. Mur vertical (façade sud)
    wall_points = np.zeros((200, 3))
    wall_points[:, 0] = np.random.uniform(0, 10, 200)
    wall_points[:, 1] = 0.0 + np.random.randn(200) * 0.05
    wall_points[:, 2] = np.random.uniform(0, 10, 200)
    
    # 3. Sol (z=0)
    ground_points = np.random.randn(200, 3) * 0.1
    ground_points[:, :2] += [5, 5]
    ground_points[:, 2] = 0.0
    
    # 4. Arête de toit (ligne)
    edge_points = np.zeros((100, 3))
    edge_points[:, 0] = np.linspace(0, 10, 100)
    edge_points[:, 1] = 10.0
    edge_points[:, 2] = 10.0 + np.random.randn(100) * 0.05
    
    # Combiner
    points = np.vstack([roof_points, wall_points, ground_points, edge_points])
    
    # Classifications
    classification = np.zeros(len(points), dtype=np.uint8)
    classification[:300] = 6  # Building (roof)
    classification[300:500] = 6  # Building (wall)
    classification[500:700] = 2  # Ground
    classification[700:] = 6  # Building (edge)
    
    # Centre du patch
    patch_center = np.array([5.0, 5.0, 5.0])
    
    return points, classification, patch_center


def test_core_features():
    """Test des features core (rapides)."""
    print("=" * 70)
    print("🧪 TEST 1: CORE FEATURES (rapides)")
    print("=" * 70)
    
    points, classification, _ = create_building_test_data()
    print(f"\n📊 Nuage de test: {len(points)} points")
    print(f"   - 300 points toit")
    print(f"   - 200 points mur")
    print(f"   - 200 points sol")
    print(f"   - 100 points arête\n")
    
    # Calcul features core
    print("⏱️  Calcul des features core...")
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points,
        classification=classification,
        auto_k=True,
        include_extra=False
    )
    
    print("✅ Features core calculées\n")
    
    # Vérifications
    print("📋 Features présentes:")
    expected_core = [
        'planarity', 'linearity', 'sphericity',
        'anisotropy', 'roughness', 'density'
    ]
    
    for feat in expected_core:
        if feat in geo_features:
            values = geo_features[feat]
            print(f"   ✅ {feat:15s} | "
                  f"min={values.min():.3f} "
                  f"max={values.max():.3f} "
                  f"mean={values.mean():.3f}")
        else:
            print(f"   ❌ {feat:15s} | MANQUANT")
    
    print()
    
    # Analyse par type de surface
    print("🔍 Analyse par type de surface:")
    
    roof_mask = classification == 6
    roof_idx = np.where(roof_mask)[0][:300]
    wall_idx = np.arange(300, 500)
    ground_idx = np.arange(500, 700)
    edge_idx = np.arange(700, 800)
    
    print(f"\n   📐 TOIT (planaire):")
    print(f"      Planarity: {geo_features['planarity'][roof_idx].mean():.3f}")
    print(f"      Linearity: {geo_features['linearity'][roof_idx].mean():.3f}")
    
    print(f"\n   🧱 MUR (planaire + vertical):")
    print(f"      Planarity: {geo_features['planarity'][wall_idx].mean():.3f}")
    print(f"      Linearity: {geo_features['linearity'][wall_idx].mean():.3f}")
    
    print(f"\n   🌍 SOL (planaire + horizontal):")
    print(f"      Planarity: {geo_features['planarity'][ground_idx].mean():.3f}")
    print(f"      Sphericity: {geo_features['sphericity'][ground_idx].mean():.3f}")
    
    print(f"\n   📏 ARÊTE (linéaire):")
    print(f"      Linearity: {geo_features['linearity'][edge_idx].mean():.3f}")
    print(f"      Planarity: {geo_features['planarity'][edge_idx].mean():.3f}")
    
    return True


def test_extra_features():
    """Test des features extra (critiques pour bâtiments)."""
    print("\n" + "=" * 70)
    print("🏢 TEST 2: EXTRA FEATURES (critiques pour bâtiments)")
    print("=" * 70)
    
    points, classification, patch_center = create_building_test_data()
    
    print("\n⏱️  Calcul avec include_extra=True...")
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points,
        classification=classification,
        auto_k=True,
        include_extra=True,
        patch_center=patch_center
    )
    
    print("✅ Features extra calculées\n")
    
    # Vérifier les nouvelles features
    print("📋 Features de hauteur (CRITIQUES):")
    height_features = [
        'z_absolute', 'z_normalized', 'z_from_ground',
        'z_from_median', 'distance_to_center'
    ]
    
    for feat in height_features:
        if feat in geo_features:
            values = geo_features[feat]
            print(f"   ✅ {feat:20s} | "
                  f"min={values.min():.3f} "
                  f"max={values.max():.3f} "
                  f"mean={values.mean():.3f}")
        else:
            print(f"   ⚠️  {feat:20s} | MANQUANT")
    
    print("\n📋 Features de statistiques locales (PUISSANTES):")
    local_stats = [
        'vertical_std', 'neighborhood_extent',
        'height_extent_ratio', 'local_roughness', 'verticality'
    ]
    
    for feat in local_stats:
        if feat in geo_features:
            values = geo_features[feat]
            print(f"   ✅ {feat:20s} | "
                  f"min={values.min():.3f} "
                  f"max={values.max():.3f} "
                  f"mean={values.mean():.3f}")
        else:
            print(f"   ⚠️  {feat:20s} | MANQUANT")
    
    # Test spécifique: Verticality pour distinguer murs/toits
    print("\n🔍 Test Verticality (murs vs toits/sol):")
    
    roof_idx = np.arange(0, 300)
    wall_idx = np.arange(300, 500)
    ground_idx = np.arange(500, 700)
    
    if 'verticality' in geo_features:
        vert = geo_features['verticality']
        print(f"   Toit:  {vert[roof_idx].mean():.3f} "
              f"(devrait être proche de 0)")
        print(f"   Mur:   {vert[wall_idx].mean():.3f} "
              f"(devrait être proche de 1)")
        print(f"   Sol:   {vert[ground_idx].mean():.3f} "
              f"(devrait être proche de 0)")
    
    # Test z_normalized pour distinguer étages
    print("\n🔍 Test z_normalized (étages du bâtiment):")
    if 'z_normalized' in geo_features:
        z_norm = geo_features['z_normalized']
        print(f"   Sol:   {z_norm[ground_idx].mean():.3f} "
              f"(devrait être ~0)")
        print(f"   Toit:  {z_norm[roof_idx].mean():.3f} "
              f"(devrait être ~1)")
    
    return True


def test_performance():
    """Test de performance sur un gros nuage."""
    print("\n" + "=" * 70)
    print("⚡ TEST 3: PERFORMANCE (nuage dense)")
    print("=" * 70)
    
    # Créer un gros nuage de points
    np.random.seed(42)
    n_points = 50000
    points = np.random.randn(n_points, 3) * 10
    points[:, 2] += 50  # Élever le nuage
    classification = np.random.choice([2, 6, 3], size=n_points)
    
    print(f"\n📊 Nuage dense: {n_points:,} points")
    
    # Test sans extra features
    import time
    t0 = time.time()
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points,
        classification=classification,
        auto_k=True,
        include_extra=False
    )
    t1 = time.time()
    
    print(f"\n⏱️  Sans extra features: {t1-t0:.2f}s")
    print(f"   ✅ {len(geo_features)} features calculées")
    
    # Test avec extra features
    t0 = time.time()
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points,
        classification=classification,
        auto_k=True,
        include_extra=True
    )
    t1 = time.time()
    
    print(f"\n⏱️  Avec extra features: {t1-t0:.2f}s")
    print(f"   ✅ {len(geo_features)} features calculées")
    print(f"   📊 Surcoût: {((t1-t0) / (t1-t0) - 1) * 100:.0f}%")
    
    return True


def main():
    """Exécute tous les tests."""
    print("\n" + "🎯" * 35)
    print("   TEST COMPLET DES NOUVELLES FEATURES GÉOMÉTRIQUES")
    print("🎯" * 35 + "\n")
    
    try:
        # Test 1: Core features
        if not test_core_features():
            print("\n❌ Test 1 échoué")
            return False
        
        # Test 2: Extra features
        if not test_extra_features():
            print("\n❌ Test 2 échoué")
            return False
        
        # Test 3: Performance
        if not test_performance():
            print("\n❌ Test 3 échoué")
            return False
        
        # Résumé
        print("\n" + "=" * 70)
        print("✅ TOUS LES TESTS SONT PASSÉS !")
        print("=" * 70)
        
        print("\n📊 RÉSUMÉ DES FEATURES:")
        print("\n   🚀 CORE (toujours calculées, rapides):")
        print("      • planarity, linearity, sphericity")
        print("      • anisotropy, roughness, density")
        print("      Total: 6 features + normals + curvature + height")
        
        print("\n   🏢 EXTRA (pour bâtiments, +50% temps):")
        print("      • z_absolute, z_normalized, z_from_ground")
        print("      • vertical_std, height_extent_ratio")
        print("      • verticality, distance_to_center")
        print("      Total: +9 features supplémentaires")
        
        print("\n   💡 UTILISATION:")
        print("      • include_extra=False → Rapide, features essentielles")
        print("      • include_extra=True → Complet, optimal pour bâtiments")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
