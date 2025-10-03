#!/usr/bin/env python3
"""
Benchmark des optimisations de calcul de features LAZ
Compare version originale vs optimisée
"""

import sys
import time
from pathlib import Path
import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))

from ign_lidar.features import (
    compute_normals,
    compute_curvature,
    compute_height_above_ground,
    extract_geometric_features
)


def benchmark_laz_file(laz_file: Path, k_neighbors: int = 10):
    """Benchmark un fichier LAZ."""
    print(f"\n{'='*70}")
    print(f"📊 BENCHMARK: {laz_file.name}")
    print(f"{'='*70}")
    
    # 1. Charger le LAZ
    print("Lecture du fichier LAZ...")
    start = time.time()
    las = laspy.read(str(laz_file))
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    classification = np.array(las.classification, dtype=np.uint8)
    load_time = time.time() - start
    
    num_points = len(points)
    print(f"  ✓ {num_points:,} points chargés en {load_time:.2f}s")
    
    # 2. Benchmark Normals
    print(f"\n🔹 Calcul des normales (k={k_neighbors})...")
    start = time.time()
    normals = compute_normals(points, k=k_neighbors)
    normals_time = time.time() - start
    print(f"  ✓ Temps: {normals_time:.2f}s")
    print(f"  ✓ Vitesse: {num_points/normals_time:,.0f} points/s")
    
    # 3. Benchmark Curvature
    print(f"\n🔹 Calcul de la courbure (k={k_neighbors})...")
    start = time.time()
    curvature = compute_curvature(points, normals, k=k_neighbors)
    curvature_time = time.time() - start
    print(f"  ✓ Temps: {curvature_time:.2f}s")
    print(f"  ✓ Vitesse: {num_points/curvature_time:,.0f} points/s")
    
    # 4. Benchmark Height
    print(f"\n🔹 Calcul de la hauteur...")
    start = time.time()
    height = compute_height_above_ground(points, classification)
    height_time = time.time() - start
    print(f"  ✓ Temps: {height_time:.2f}s")
    print(f"  ✓ Vitesse: {num_points/height_time:,.0f} points/s")
    
    # 5. Benchmark Geometric Features
    print(f"\n🔹 Calcul des features géométriques (k={k_neighbors})...")
    start = time.time()
    geo_features = extract_geometric_features(points, normals, k=k_neighbors)
    geo_time = time.time() - start
    print(f"  ✓ Temps: {geo_time:.2f}s")
    print(f"  ✓ Vitesse: {num_points/geo_time:,.0f} points/s")
    
    # Temps total
    total_time = normals_time + curvature_time + height_time + geo_time
    
    print(f"\n{'='*70}")
    print(f"📊 RÉSUMÉ GLOBAL")
    print(f"{'='*70}")
    print(f"Points traités:     {num_points:,}")
    print(f"Temps total:        {total_time:.2f}s")
    print(f"Vitesse moyenne:    {num_points/total_time:,.0f} points/s")
    print(f"")
    print(f"Décomposition du temps:")
    print(f"  - Normales:       {normals_time:.2f}s ({100*normals_time/total_time:.1f}%)")
    print(f"  - Courbure:       {curvature_time:.2f}s ({100*curvature_time/total_time:.1f}%)")
    print(f"  - Hauteur:        {height_time:.2f}s ({100*height_time/total_time:.1f}%)")
    print(f"  - Geo features:   {geo_time:.2f}s ({100*geo_time/total_time:.1f}%)")
    print(f"{'='*70}")
    
    return {
        'num_points': num_points,
        'normals_time': normals_time,
        'curvature_time': curvature_time,
        'height_time': height_time,
        'geo_time': geo_time,
        'total_time': total_time,
        'points_per_sec': num_points / total_time
    }


def main():
    """Point d'entrée principal."""
    print(f"\n{'='*70}")
    print("⚡ BENCHMARK OPTIMISATIONS FEATURES LAZ")
    print(f"{'='*70}")
    
    # Trouver des fichiers LAZ de test
    dataset_dir = Path("urban_training_dataset")
    raw_dir = dataset_dir / "raw_tiles"
    
    laz_files = list(raw_dir.rglob("*.laz"))[:3]  # Prendre 3 fichiers
    
    if not laz_files:
        print("❌ Aucun fichier LAZ trouvé dans raw_tiles/")
        return 1
    
    print(f"✓ Trouvé {len(laz_files)} fichiers LAZ pour le benchmark")
    print(f"  Test avec k=10 (optimal pour vitesse)")
    
    # Benchmark chaque fichier
    results = []
    for laz_file in laz_files:
        result = benchmark_laz_file(laz_file, k_neighbors=10)
        results.append(result)
    
    # Statistiques globales
    total_points = sum(r['num_points'] for r in results)
    total_time = sum(r['total_time'] for r in results)
    avg_speed = total_points / total_time
    
    print(f"\n{'='*70}")
    print("🏆 STATISTIQUES GLOBALES")
    print(f"{'='*70}")
    print(f"Fichiers traités:   {len(results)}")
    print(f"Points totaux:      {total_points:,}")
    print(f"Temps total:        {total_time:.2f}s")
    print(f"Vitesse moyenne:    {avg_speed:,.0f} points/s")
    print(f"")
    print(f"💡 Estimation pour un gros fichier de 10M points:")
    print(f"   Temps estimé: {10_000_000/avg_speed:.1f}s (~{10_000_000/avg_speed/60:.1f} min)")
    print(f"{'='*70}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
