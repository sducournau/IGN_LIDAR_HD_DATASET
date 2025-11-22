#!/usr/bin/env python3
"""
Benchmark des différentes implémentations de compute_normals()

Compare les performances des 18+ implémentations de calcul de normales.
Usage: python scripts/benchmark_normals.py [--size SIZE] [--iterations N]

Date: 21 Novembre 2025
"""

import argparse
import time
import numpy as np
from typing import Callable, Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


def generate_test_data(n_points: int = 10000) -> np.ndarray:
    """Génère un nuage de points 3D aléatoire."""
    return np.random.rand(n_points, 3).astype(np.float32)


def benchmark_function(
    func: Callable,
    points: np.ndarray,
    k: int = 30,
    iterations: int = 3
) -> Tuple[float, bool]:
    """
    Benchmark une fonction de calcul de normales.
    
    Returns:
        (avg_time, success)
    """
    times = []
    success = True
    
    for _ in range(iterations):
        try:
            start = time.time()
            result = func(points, k_neighbors=k)
            end = time.time()
            
            # Vérifier résultat
            if isinstance(result, tuple):
                normals = result[0]
            else:
                normals = result
            
            if not isinstance(normals, np.ndarray):
                success = False
                break
            
            if normals.shape != (len(points), 3):
                success = False
                break
            
            times.append(end - start)
        except Exception as e:
            print(f"    ❌ Erreur: {e}")
            success = False
            break
    
    if not success or not times:
        return -1.0, False
    
    return np.mean(times), True


def collect_implementations() -> Dict[str, Callable]:
    """Collecte toutes les implémentations disponibles."""
    implementations = {}
    
    # Implementation 1: features/compute/normals.py
    try:
        from ign_lidar.features.compute.normals import compute_normals
        implementations['compute/normals.py'] = compute_normals
    except ImportError:
        pass
    
    # Implementation 2: features/feature_computer.py
    try:
        from ign_lidar.features.feature_computer import FeatureComputer
        fc = FeatureComputer()
        implementations['FeatureComputer'] = fc.compute_normals
    except ImportError:
        pass
    
    # Implementation 3: features/gpu_processor.py
    try:
        from ign_lidar.features.gpu_processor import GPUProcessor
        gpu = GPUProcessor()
        implementations['GPUProcessor'] = gpu.compute_normals
    except ImportError:
        pass
    
    # Implementation 4: compute_normals_fast
    try:
        from ign_lidar.features.compute.normals import compute_normals_fast
        implementations['compute_normals_fast'] = compute_normals_fast
    except ImportError:
        pass
    
    # Implementation 5: compute_normals_accurate
    try:
        from ign_lidar.features.compute.normals import compute_normals_accurate
        implementations['compute_normals_accurate'] = compute_normals_accurate
    except ImportError:
        pass
    
    # Implementation 6: numba_accelerated
    try:
        from ign_lidar.features.numba_accelerated import compute_normals_from_eigenvectors
        implementations['numba_accelerated'] = compute_normals_from_eigenvectors
    except ImportError:
        pass
    
    return implementations


def print_results(results: List[Tuple[str, float, bool, int]], n_points: int):
    """Affiche les résultats de benchmark."""
    print("\n" + "="*80)
    print(f"📊 RÉSULTATS BENCHMARK - {n_points:,} points")
    print("="*80)
    
    # Trier par temps (succès d'abord)
    results_success = [(name, time, True, size) for name, time, success, size in results if success]
    results_fail = [(name, time, False, size) for name, time, success, size in results if not success]
    
    results_success.sort(key=lambda x: x[1])
    
    print("\n✅ Implémentations fonctionnelles:\n")
    
    if not results_success:
        print("  ❌ Aucune implémentation fonctionnelle!")
        return
    
    # Baseline (plus rapide)
    baseline_time = results_success[0][1]
    
    print(f"{'Rang':<6} {'Implémentation':<35} {'Temps (s)':<12} {'Speedup':<10} {'Taille'}")
    print("-"*80)
    
    for i, (name, avg_time, _, size) in enumerate(results_success, 1):
        speedup = baseline_time / avg_time if avg_time > 0 else 0
        speedup_str = f"{speedup:.2f}x" if i > 1 else "baseline"
        
        # Emoji selon performance
        if i == 1:
            emoji = "🥇"
        elif i == 2:
            emoji = "🥈"
        elif i == 3:
            emoji = "🥉"
        elif speedup < 0.5:
            emoji = "🐌"
        elif speedup < 0.8:
            emoji = "⚠️ "
        else:
            emoji = "  "
        
        print(f"{emoji} #{i:<3} {name:<35} {avg_time:>10.4f}s  {speedup_str:<10} {size} loc")
    
    # Échecs
    if results_fail:
        print("\n❌ Implémentations en échec:\n")
        for name, _, _, size in results_fail:
            print(f"  ❌ {name} - Erreur lors de l'exécution")
    
    # Statistiques
    print("\n" + "="*80)
    print("📈 STATISTIQUES")
    print("="*80)
    
    times = [t for _, t, success, _ in results if success]
    if times:
        print(f"\nPlus rapide:  {min(times):.4f}s ({results_success[0][0]})")
        print(f"Plus lent:    {max(times):.4f}s")
        print(f"Écart:        {max(times)/min(times):.1f}x")
        print(f"Moyenne:      {np.mean(times):.4f}s")
        print(f"Médiane:      {np.median(times):.4f}s")
    
    print(f"\nImplémentations testées: {len(results)}")
    print(f"Succès:                  {len(results_success)}")
    print(f"Échecs:                  {len(results_fail)}")
    
    # Recommandation
    print("\n" + "="*80)
    print("💡 RECOMMANDATION")
    print("="*80)
    
    if len(results_success) >= 3:
        print("\n🔴 PROBLÈME: Duplication massive détectée!")
        print(f"   → {len(results_success)} implémentations différentes")
        print(f"   → Écart de performance: {max(times)/min(times):.1f}x")
        print()
        print("📋 Actions recommandées:")
        print("   1. Créer API unique: features/compute/normals_api.py")
        print("   2. Garder uniquement l'implémentation la plus rapide")
        print("   3. Ajouter strategy pattern (cpu/gpu/faiss-gpu)")
        print("   4. Déprécier autres implémentations")
        print()
        print(f"💾 Gain estimé: -{(len(results_success)-1)*200} lignes de code")
    else:
        print("\n✅ Architecture acceptable")
        print(f"   → {len(results_success)} implémentation(s) seulement")


def estimate_loc(name: str) -> int:
    """Estime les lignes de code d'une implémentation."""
    # Estimation très approximative
    estimates = {
        'compute/normals.py': 150,
        'FeatureComputer': 200,
        'GPUProcessor': 300,
        'compute_normals_fast': 50,
        'compute_normals_accurate': 80,
        'numba_accelerated': 100,
    }
    
    for key, loc in estimates.items():
        if key in name:
            return loc
    
    return 100  # Défaut


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Benchmark des implémentations de compute_normals()"
    )
    parser.add_argument(
        '--size',
        type=int,
        default=10000,
        help="Nombre de points à tester (défaut: 10000)"
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help="Nombre d'itérations par test (défaut: 3)"
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help="Comparer toutes les tailles (1K, 10K, 100K points)"
    )
    
    args = parser.parse_args()
    
    print("🔍 Benchmark compute_normals() - IGN LiDAR HD")
    print("="*80)
    
    # Collecter implémentations
    print("\n📦 Recherche des implémentations...")
    implementations = collect_implementations()
    
    if not implementations:
        print("❌ Aucune implémentation trouvée!")
        print("   Assurez-vous que le package est installé:")
        print("   pip install -e .")
        return
    
    print(f"✅ {len(implementations)} implémentations trouvées:")
    for name in implementations:
        print(f"  - {name}")
    
    # Test simple ou multiple tailles
    if args.compare_all:
        sizes = [1000, 10000, 100000]
        print(f"\n🔍 Test sur {len(sizes)} tailles différentes...")
        
        for size in sizes:
            print(f"\n{'='*80}")
            print(f"📊 Taille: {size:,} points")
            print('='*80)
            
            points = generate_test_data(size)
            results = []
            
            for name, func in implementations.items():
                print(f"\n🔬 Test: {name}...")
                avg_time, success = benchmark_function(
                    func, points, k=30, iterations=args.iterations
                )
                loc = estimate_loc(name)
                results.append((name, avg_time, success, loc))
                
                if success:
                    print(f"  ✅ {avg_time:.4f}s")
                else:
                    print(f"  ❌ Échec")
            
            print_results(results, size)
    
    else:
        # Test sur une seule taille
        print(f"\n🔍 Génération de {args.size:,} points 3D...")
        points = generate_test_data(args.size)
        print(f"✅ Shape: {points.shape}, dtype: {points.dtype}")
        
        print(f"\n🏃 Benchmark ({args.iterations} itérations par test)...")
        print("-"*80)
        
        results = []
        
        for name, func in implementations.items():
            print(f"\n🔬 Test: {name}...")
            avg_time, success = benchmark_function(
                func, points, k=30, iterations=args.iterations
            )
            loc = estimate_loc(name)
            results.append((name, avg_time, success, loc))
            
            if success:
                print(f"  ✅ {avg_time:.4f}s (moyenne sur {args.iterations} runs)")
            else:
                print(f"  ❌ Échec de l'exécution")
        
        print_results(results, args.size)
    
    print("\n💡 Pour plus de détails:")
    print("   python scripts/benchmark_normals.py --compare-all")
    print()


if __name__ == "__main__":
    main()
