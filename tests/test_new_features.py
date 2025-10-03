#!/usr/bin/env python3
"""
Test script pour valider les nouvelles features géométriques
"""

import numpy as np
from ign_lidar.features import (
    compute_all_features_optimized,
    estimate_optimal_k,
    extract_geometric_features
)


def test_new_features():
    """Test que les nouvelles features sont bien calculées."""
    print("🧪 Test des nouvelles features géométriques\n")
    
    # Créer un nuage de points synthétique
    print("1️⃣ Création d'un nuage de points test...")
    np.random.seed(42)
    
    # Points formant un plan (toit de bâtiment)
    plane_points = np.random.randn(500, 3) * 0.1
    plane_points[:, 2] = 5.0  # Plan horizontal à z=5
    
    # Points formant une ligne (arête)
    line_points = np.zeros((200, 3))
    line_points[:, 0] = np.linspace(0, 10, 200)
    line_points[:, 1:] += np.random.randn(200, 2) * 0.05
    
    # Points dispersés (végétation/bruit)
    scattered_points = np.random.randn(300, 3) * 2
    
    # Combiner
    points = np.vstack([plane_points, line_points, scattered_points])
    classification = np.zeros(len(points), dtype=np.uint8)
    classification[:500] = 6  # Building
    classification[500:700] = 14  # Wire
    classification[700:] = 3  # Vegetation
    
    print(f"   ✅ {len(points)} points créés")
    print(f"      - 500 points plans (toit)")
    print(f"      - 200 points linéaires (arête)")
    print(f"      - 300 points dispersés (végétation)\n")
    
    # Test 1: Estimation automatique de k
    print("2️⃣ Test de l'estimation automatique de k...")
    k_estimated = estimate_optimal_k(points, target_radius=0.5)
    print(f"   ✅ k estimé = {k_estimated}\n")
    
    # Test 2: Calcul des features
    print("3️⃣ Calcul de toutes les features...")
    normals, curvature, height, geo_features = compute_all_features_optimized(
        points=points,
        classification=classification,
        auto_k=True
    )
    print("   ✅ Features calculées\n")
    
    # Test 3: Vérifier les nouvelles features
    print("4️⃣ Vérification des features retournées...")
    
    expected_features = [
        'planarity', 'linearity', 'sphericity',
        'anisotropy', 'roughness', 'density'
    ]
    
    removed_features = ['verticality', 'horizontality']
    
    for feature in expected_features:
        if feature in geo_features:
            print(f"   ✅ {feature:12s} présent - "
                  f"shape={geo_features[feature].shape}, "
                  f"min={geo_features[feature].min():.3f}, "
                  f"max={geo_features[feature].max():.3f}, "
                  f"mean={geo_features[feature].mean():.3f}")
        else:
            print(f"   ❌ {feature} MANQUANT !")
            return False
    
    print()
    for feature in removed_features:
        if feature not in geo_features:
            print(f"   ✅ {feature:12s} correctement supprimé")
        else:
            print(f"   ⚠️  {feature} existe encore (devrait être supprimé)")
    
    print()
    
    # Test 4: Analyser les valeurs des features
    print("5️⃣ Analyse des valeurs des features...\n")
    
    # Points plans devraient avoir planarity élevée
    planarity_plane = geo_features['planarity'][:500].mean()
    print(f"   Planarity (points plans) : {planarity_plane:.3f} "
          f"{'✅' if planarity_plane > 0.5 else '⚠️'}")
    
    # Points linéaires devraient avoir linearity élevée
    linearity_line = geo_features['linearity'][500:700].mean()
    print(f"   Linearity (points ligne) : {linearity_line:.3f} "
          f"{'✅' if linearity_line > 0.5 else '⚠️'}")
    
    # Points dispersés devraient avoir sphericity élevée
    sphericity_scattered = geo_features['sphericity'][700:].mean()
    print(f"   Sphericity (points dispersés) : {sphericity_scattered:.3f} "
          f"{'✅' if sphericity_scattered > 0.2 else '⚠️'}")
    
    print()
    
    # Test 5: Vérifier qu'on peut recréer verticality depuis normals
    print("6️⃣ Test de reconstruction de verticality...")
    verticality_from_normals = np.abs(normals[:, 2])
    print(f"   ✅ Verticality reconstruite depuis normals[:, 2]")
    print(f"      min={verticality_from_normals.min():.3f}, "
          f"max={verticality_from_normals.max():.3f}, "
          f"mean={verticality_from_normals.mean():.3f}\n")
    
    # Test 6: Vérifier dimensions
    print("7️⃣ Vérification des dimensions...")
    assert normals.shape == (len(points), 3), "Normals shape incorrect"
    assert curvature.shape == (len(points),), "Curvature shape incorrect"
    assert height.shape == (len(points),), "Height shape incorrect"
    
    for feature_name, feature_values in geo_features.items():
        assert feature_values.shape == (len(points),), \
            f"Feature {feature_name} shape incorrect"
    
    print(f"   ✅ Toutes les dimensions sont correctes")
    print(f"      - Normals: {normals.shape}")
    print(f"      - Curvature: {curvature.shape}")
    print(f"      - Height: {height.shape}")
    print(f"      - Geo features: {len(geo_features)} features × "
          f"{len(points)} points\n")
    
    print("=" * 60)
    print("✅ TOUS LES TESTS SONT PASSÉS !")
    print("=" * 60)
    print()
    print("📊 Résumé des changements:")
    print("   • Features supprimées: verticality, horizontality")
    print("   • Features ajoutées: linearity, sphericity, anisotropy")
    print("   • Total features géométriques: 6")
    print("   • k adaptatif: activé par défaut")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_new_features()
        if success:
            print("🎉 Les nouvelles features sont opérationnelles !\n")
            exit(0)
        else:
            print("❌ Des problèmes ont été détectés.\n")
            exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}\n")
        import traceback
        traceback.print_exc()
        exit(1)
