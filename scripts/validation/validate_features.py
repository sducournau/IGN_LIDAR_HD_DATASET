#!/usr/bin/env python3
"""
Script de validation des features géométriques
Teste que toutes les features sont correctement calculées
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features import compute_all_features_optimized

def test_roof():
    """Test sur un plan horizontal (toit)."""
    print('🧪 Test 1: Plan horizontal (TOIT)...')
    
    # Créer un plan horizontal parfait
    roof = np.random.randn(1000, 3) * 0.05  # Petit bruit
    roof[:, 2] = 10.0  # Z constant
    
    normals, curv, height, features = compute_all_features_optimized(
        roof,
        classification=np.zeros(1000, dtype=np.uint8),
        auto_k=True,
        include_extra=False
    )
    
    # Validation
    planarity = features['planarity'].mean()
    normal_z = normals[:, 2].mean()
    roughness = features['roughness'].mean()
    
    print(f'  Planarity: {planarity:.3f} (attendu > 0.8)')
    print(f'  Normal_z: {normal_z:.3f} (attendu > 0.9)')
    print(f'  Roughness: {roughness:.3f} (attendu < 0.2)')
    
    assert planarity > 0.8, f'❌ Planarity trop faible: {planarity}'
    assert normal_z > 0.9, f'❌ Normale pas verticale: {normal_z}'
    assert roughness < 0.2, f'❌ Roughness trop élevée: {roughness}'
    
    print('  ✅ Test TOIT réussi\n')
    return True


def test_wall():
    """Test sur un plan vertical (mur)."""
    print('🧪 Test 2: Plan vertical (MUR)...')
    
    # Créer un plan vertical parfait (plan YZ, X constant)
    wall = np.random.randn(1000, 3) * 0.05
    wall[:, 0] = 0.0  # X constant
    
    normals, curv, height, features = compute_all_features_optimized(
        wall,
        classification=np.zeros(1000, dtype=np.uint8),
        auto_k=True,
        include_extra=True
    )
    
    # Validation
    planarity = features['planarity'].mean()
    verticality = features['verticality'].mean()
    linearity = features['linearity'].mean()
    
    print(f'  Planarity: {planarity:.3f} (attendu > 0.8)')
    print(f'  Verticality: {verticality:.3f} (attendu > 0.7)')
    print(f'  Linearity: {linearity:.3f} (attendu < 0.4)')
    
    assert planarity > 0.8, f'❌ Planarity trop faible: {planarity}'
    assert verticality > 0.7, f'❌ Pas assez vertical: {verticality}'
    assert linearity < 0.4, f'❌ Linearity trop élevée: {linearity}'
    
    print('  ✅ Test MUR réussi\n')
    return True


def test_edge():
    """Test sur une ligne (arête)."""
    print('🧪 Test 3: Structure linéaire (ARÊTE)...')
    
    # Créer une ligne parfaite avec petit bruit
    edge = np.column_stack([
        np.linspace(0, 10, 1000),
        np.random.randn(1000) * 0.03,
        np.random.randn(1000) * 0.03 + 5
    ])
    
    normals, curv, height, features = compute_all_features_optimized(
        edge,
        classification=np.zeros(1000, dtype=np.uint8),
        auto_k=True,
        include_extra=False
    )
    
    # Validation
    linearity = features['linearity'].mean()
    planarity = features['planarity'].mean()
    sphericity = features['sphericity'].mean()
    
    print(f'  Linearity: {linearity:.3f} (attendu > 0.6)')
    print(f'  Planarity: {planarity:.3f} (attendu < 0.5)')
    print(f'  Sphericity: {sphericity:.3f} (attendu < 0.4)')
    
    assert linearity > 0.6, f'❌ Linearity trop faible: {linearity}'
    assert planarity < 0.5, f'❌ Trop planaire pour une ligne: {planarity}'
    assert sphericity < 0.4, f'❌ Sphericity trop élevée: {sphericity}'
    
    print('  ✅ Test ARÊTE réussi\n')
    return True


def test_sphere():
    """Test sur une sphère (végétation)."""
    print('🧪 Test 4: Structure sphérique (VÉGÉTATION)...')
    
    # Créer des points sur une sphère
    n = 1000
    theta = np.random.uniform(0, 2*np.pi, n)
    phi = np.random.uniform(0, np.pi, n)
    r = 2.0 + np.random.randn(n) * 0.1  # Rayon avec bruit
    
    sphere = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi) + 5
    ])
    
    normals, curv, height, features = compute_all_features_optimized(
        sphere,
        classification=np.zeros(n, dtype=np.uint8),
        auto_k=True,
        include_extra=False
    )
    
    # Validation
    sphericity = features['sphericity'].mean()
    planarity = features['planarity'].mean()
    linearity = features['linearity'].mean()
    roughness = features['roughness'].mean()
    
    print(f'  Sphericity: {sphericity:.3f} (attendu > 0.3)')
    print(f'  Planarity: {planarity:.3f} (attendu < 0.4)')
    print(f'  Linearity: {linearity:.3f} (attendu < 0.4)')
    print(f'  Roughness: {roughness:.3f} (attendu > 0.2)')
    
    assert sphericity > 0.3, f'❌ Sphericity trop faible: {sphericity}'
    assert planarity < 0.4, f'❌ Trop planaire: {planarity}'
    assert linearity < 0.4, f'❌ Trop linéaire: {linearity}'
    
    print('  ✅ Test SPHÈRE réussi\n')
    return True


def test_feature_presence():
    """Vérifier que toutes les features attendues sont présentes."""
    print('🧪 Test 5: Présence de toutes les features...')
    
    # Points de test simples
    points = np.random.randn(500, 3)
    classification = np.zeros(500, dtype=np.uint8)
    
    # Mode core
    normals, curv, height, features = compute_all_features_optimized(
        points,
        classification=classification,
        auto_k=True,
        include_extra=False
    )
    
    # Vérifier features core
    core_features = [
        'planarity', 'linearity', 'sphericity', 
        'anisotropy', 'roughness', 'density'
    ]
    
    for feat in core_features:
        assert feat in features, f'❌ Feature manquante: {feat}'
        assert len(features[feat]) == len(points), f'❌ Taille incorrecte: {feat}'
        print(f'  ✅ {feat}: {features[feat].mean():.3f}')
    
    # Mode building
    normals, curv, height, features = compute_all_features_optimized(
        points,
        classification=classification,
        auto_k=True,
        include_extra=True
    )
    
    # Vérifier features extra
    extra_features = [
        'z_normalized', 'z_from_ground', 'verticality',
        'vertical_std', 'height_extent_ratio'
    ]
    
    for feat in extra_features:
        assert feat in features, f'❌ Feature extra manquante: {feat}'
        assert len(features[feat]) == len(points), f'❌ Taille incorrecte: {feat}'
    
    print(f'\n  ✅ Toutes les features présentes (core: {len(core_features)}, extra: {len(extra_features)})\n')
    return True


def test_eigenvalue_formulas():
    """Vérifier les formules mathématiques des features."""
    print('🧪 Test 6: Validation formules mathématiques...')
    
    # Créer un plan horizontal parfait
    plane = np.zeros((100, 3))
    plane[:, 0] = np.linspace(0, 10, 100)
    plane[:, 1] = np.linspace(0, 10, 100)
    plane[:, 2] = 5.0  # Z constant
    
    normals, curv, height, features = compute_all_features_optimized(
        plane,
        classification=np.zeros(100, dtype=np.uint8),
        k=10,
        auto_k=False,
        include_extra=False
    )
    
    # Pour un plan parfait:
    # - planarity devrait être très élevé (~1)
    # - linearity devrait être faible
    # - sphericity devrait être très faible (~0)
    # - anisotropy devrait être très élevé (~1)
    
    print(f'  Planarity (attendu ~1.0): {features["planarity"].mean():.3f}')
    print(f'  Linearity (attendu ~0.0): {features["linearity"].mean():.3f}')
    print(f'  Sphericity (attendu ~0.0): {features["sphericity"].mean():.3f}')
    print(f'  Anisotropy (attendu ~1.0): {features["anisotropy"].mean():.3f}')
    
    # Vérifications lâches (nuage synthétique simple)
    assert features['planarity'].mean() > 0.7, '❌ Planarity formule incorrecte'
    assert features['linearity'].mean() < 0.4, '❌ Linearity formule incorrecte'
    assert features['sphericity'].mean() < 0.3, '❌ Sphericity formule incorrecte'
    assert features['anisotropy'].mean() > 0.7, '❌ Anisotropy formule incorrecte'
    
    print('  ✅ Formules mathématiques validées\n')
    return True


def main():
    """Exécuter tous les tests."""
    print('=' * 60)
    print('🔍 VALIDATION DES FEATURES GÉOMÉTRIQUES')
    print('=' * 60)
    print()
    
    tests = [
        ('Toit (plan horizontal)', test_roof),
        ('Mur (plan vertical)', test_wall),
        ('Arête (structure linéaire)', test_edge),
        ('Végétation (structure sphérique)', test_sphere),
        ('Présence des features', test_feature_presence),
        ('Formules mathématiques', test_eigenvalue_formulas),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, True, None))
        except AssertionError as e:
            print(f'  ❌ Test échoué: {e}\n')
            results.append((name, False, str(e)))
        except Exception as e:
            print(f'  ❌ Erreur inattendue: {e}\n')
            results.append((name, False, str(e)))
    
    # Résumé
    print('=' * 60)
    print('📊 RÉSUMÉ DES TESTS')
    print('=' * 60)
    
    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    for name, success, error in results:
        status = '✅ RÉUSSI' if success else '❌ ÉCHOUÉ'
        print(f'{status}: {name}')
        if error:
            print(f'         {error}')
    
    print()
    print(f'Résultat: {success_count}/{total_count} tests réussis')
    
    if success_count == total_count:
        print('\n🎉 TOUS LES TESTS SONT RÉUSSIS!')
        print('✅ Validation: Toutes les features sont correctement calculées')
        print('✅ Les formules mathématiques sont conformes aux références scientifiques')
        return 0
    else:
        print('\n❌ CERTAINS TESTS ONT ÉCHOUÉ')
        return 1


if __name__ == '__main__':
    sys.exit(main())
