#!/usr/bin/env python3
"""
Test des performances des features optimisées
"""

import sys
from pathlib import Path
import time
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))

from ign_lidar.features import (
    compute_normals,
    compute_curvature,
    compute_height_above_ground,
    extract_geometric_features
)

def test_performance():
    """Test sur un fichier LAZ réel."""
    
    # Trouver un fichier LAZ
    laz_files = list(Path("urban_training_dataset/raw_tiles").rglob("*.laz"))
    
    if not laz_files:
        print("❌ Aucun fichier LAZ trouvé")
        return
    
    # Prendre le plus petit fichier pour le test
    laz_files.sort(key=lambda f: f.stat().st_size)
    test_file = laz_files[0]
    
    print(f"📁 Test sur: {test_file.name}")
    print(f"   Taille: {test_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Charger le LAZ
    print("\n⏳ Chargement du LAZ...")
    las = laspy.read(str(test_file))
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    classification = np.array(las.classification, dtype=np.uint8)
    
    print(f"✓ {len(points):,} points chargés")
    
    # Test 1: Normales
    print("\n🧮 Test 1: Calcul des normales...")
    start = time.time()
    normals = compute_normals(points, k=10)
    t_normals = time.time() - start
    print(f"✓ Normales calculées en {t_normals:.2f}s")
    print(f"  Vitesse: {len(points)/t_normals:,.0f} points/s")
    
    # Test 2: Courbure
    print("\n🧮 Test 2: Calcul de la courbure...")
    start = time.time()
    curvature = compute_curvature(points, normals, k=10)
    t_curvature = time.time() - start
    print(f"✓ Courbure calculée en {t_curvature:.2f}s")
    print(f"  Vitesse: {len(points)/t_curvature:,.0f} points/s")
    
    # Test 3: Hauteur
    print("\n🧮 Test 3: Calcul de la hauteur...")
    start = time.time()
    height = compute_height_above_ground(points, classification)
    t_height = time.time() - start
    print(f"✓ Hauteur calculée en {t_height:.2f}s")
    print(f"  Vitesse: {len(points)/t_height:,.0f} points/s")
    
    # Test 4: Features géométriques
    print("\n🧮 Test 4: Extraction features géométriques...")
    start = time.time()
    geo_features = extract_geometric_features(points, normals, k=10)
    t_features = time.time() - start
    print(f"✓ Features extraites en {t_features:.2f}s")
    print(f"  Vitesse: {len(points)/t_features:,.0f} points/s")
    
    # Temps total
    total_time = t_normals + t_curvature + t_height + t_features
    print("\n" + "="*60)
    print("📊 RÉSUMÉ")
    print("="*60)
    print(f"Normales:           {t_normals:>8.2f}s  ({t_normals/total_time*100:>5.1f}%)")
    print(f"Courbure:           {t_curvature:>8.2f}s  ({t_curvature/total_time*100:>5.1f}%)")
    print(f"Hauteur:            {t_height:>8.2f}s  ({t_height/total_time*100:>5.1f}%)")
    print(f"Features géo:       {t_features:>8.2f}s  ({t_features/total_time*100:>5.1f}%)")
    print("-"*60)
    print(f"TOTAL:              {total_time:>8.2f}s")
    print(f"Vitesse globale:    {len(points)/total_time:>8,.0f} points/s")
    print("="*60)
    
    # Stats sur les résultats
    print("\n📈 Statistiques des features:")
    print(f"Normales - range: [{normals.min():.2f}, {normals.max():.2f}]")
    print(f"Courbure - mean: {curvature.mean():.4f}, std: {curvature.std():.4f}")
    print(f"Hauteur - mean: {height.mean():.2f}m, max: {height.max():.2f}m")
    print(f"Planarity - mean: {geo_features['planarity'].mean():.3f}")
    print(f"Verticality - mean: {geo_features['verticality'].mean():.3f}")
    print(f"Density - mean: {geo_features['density'].mean():.2f}")

if __name__ == '__main__':
    test_performance()
