#!/usr/bin/env python3
"""
Phase 3 Profiling Script - Identify Optimization Targets

This script profiles the current codebase to identify:
1. Hot path functions (>5% runtime)
2. Loop-heavy operations
3. Memory allocation hotspots
4. I/O bottlenecks

Usage:
    python scripts/profile_phase3_targets.py
    
Output:
    - baseline_profile.stats (cProfile output)
    - profile_report.txt (Human-readable report)
    - hotspots.json (Top optimization targets)
"""

import sys
import cProfile
import pstats
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ign_lidar.features.features import (
    compute_normals,
    compute_curvature,
    compute_eigenvalue_features,
)


def generate_test_data(n_points=100_000):
    """Generate realistic test data for profiling."""
    print(f"📊 Generating test data ({n_points:,} points)...")
    
    # Random point cloud
    points = np.random.rand(n_points, 3).astype(np.float32) * 100
    
    # Add some structure (planes, edges)
    n_plane = n_points // 3
    points[:n_plane, 2] = 10.0  # Ground plane
    
    n_edge = n_points // 10
    points[n_plane:n_plane+n_edge, 2] = points[n_plane:n_plane+n_edge, 0] * 0.5  # Ramp
    
    # Classification
    classification = np.random.randint(0, 10, n_points, dtype=np.uint8)
    
    # Intensity
    intensity = np.random.rand(n_points).astype(np.float32)
    
    print(f"✅ Test data generated: {points.shape}")
    return points, classification, intensity


def profile_feature_computation(points: np.ndarray, k: int = 20) -> Dict:
    """Profile core feature computation functions."""
    print(f"\n🔍 Profiling feature computation (k={k})...")
    
    results = {}
    
    # 1. Normals computation
    print("  - Profiling normals computation...")
    start = time.perf_counter()
    normals = compute_normals(points, k=k)
    normals_time = time.perf_counter() - start
    results['normals'] = {
        'time': normals_time,
        'throughput': len(points) / normals_time,
        'time_per_point': normals_time / len(points) * 1e6  # microseconds
    }
    print(f"    ✓ {normals_time:.3f}s ({results['normals']['throughput']:,.0f} pts/s)")
    
    # 2. Curvature computation
    print("  - Profiling curvature computation...")
    start = time.perf_counter()
    curvature = compute_curvature(points, normals, k=k)
    curvature_time = time.perf_counter() - start
    results['curvature'] = {
        'time': curvature_time,
        'throughput': len(points) / curvature_time,
        'time_per_point': curvature_time / len(points) * 1e6
    }
    print(f"    ✓ {curvature_time:.3f}s ({results['curvature']['throughput']:,.0f} pts/s)")
    
    # 3. Eigenvalue features (compute eigenvalues for this)
    print("  - Computing eigenvalues...")
    from ign_lidar.features.utils import build_kdtree, compute_local_eigenvalues
    tree = build_kdtree(points)
    eigenvalues = compute_local_eigenvalues(points, tree, k=k)
    
    print("  - Profiling eigenvalue features...")
    start = time.perf_counter()
    eig_features = compute_eigenvalue_features(eigenvalues)
    eig_time = time.perf_counter() - start
    results['eigenvalue_features'] = {
        'time': eig_time,
        'throughput': len(points) / eig_time,
        'time_per_point': eig_time / len(points) * 1e6
    }
    print(f"    ✓ {eig_time:.3f}s ({results['eigenvalue_features']['throughput']:,.0f} pts/s)")
    
    # Total time
    total_time = normals_time + curvature_time + eig_time
    results['total'] = {
        'time': total_time,
        'throughput': len(points) / total_time,
        'time_per_point': total_time / len(points) * 1e6
    }
    
    print(f"\n  📈 Total feature computation: {total_time:.3f}s ({results['total']['throughput']:,.0f} pts/s)")
    
    return results


def profile_with_cprofile(func, *args, output_file='profile.stats'):
    """Profile a function with cProfile."""
    print(f"\n🔬 Running cProfile on {func.__name__}...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args)
    
    profiler.disable()
    profiler.dump_stats(output_file)
    
    print(f"  ✓ Profile saved to {output_file}")
    
    return result


def analyze_profile(stats_file: str, top_n: int = 20) -> List[Dict]:
    """Analyze cProfile output and extract top functions."""
    print(f"\n📊 Analyzing profile ({stats_file})...")
    
    stats = pstats.Stats(stats_file)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    # Extract top functions
    hotspots = []
    
    for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:top_n]:
        filename, line, func_name = func
        
        hotspot = {
            'function': func_name,
            'file': filename,
            'line': line,
            'calls': nc,
            'total_time': tt,
            'cumulative_time': ct,
            'time_per_call': ct / nc if nc > 0 else 0,
            'percent_time': (ct / stats.total_tt * 100) if stats.total_tt > 0 else 0
        }
        
        hotspots.append(hotspot)
        
        # Print significant ones (>2% runtime)
        if hotspot['percent_time'] > 2.0:
            print(f"  🔥 {hotspot['function']:40s} {hotspot['percent_time']:6.2f}% "
                  f"({hotspot['cumulative_time']:.3f}s, {hotspot['calls']:,} calls)")
    
    return hotspots


def generate_report(
    feature_results: Dict,
    hotspots: List[Dict],
    output_file: str = 'profile_report.txt'
):
    """Generate human-readable profiling report."""
    print(f"\n📝 Generating report ({output_file})...")
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 3 Profiling Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Feature computation summary
        f.write("1. FEATURE COMPUTATION PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        for feature, metrics in feature_results.items():
            f.write(f"\n{feature.upper()}:\n")
            f.write(f"  Time: {metrics['time']:.3f}s\n")
            f.write(f"  Throughput: {metrics['throughput']:,.0f} points/sec\n")
            f.write(f"  Time per point: {metrics['time_per_point']:.2f} μs\n")
        
        # Hot path functions
        f.write("\n\n2. HOT PATH FUNCTIONS (>2% runtime)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Function':<40} {'%Time':>8} {'Cumulative':>12} {'Calls':>10}\n")
        f.write("-" * 80 + "\n")
        
        for hotspot in hotspots:
            if hotspot['percent_time'] > 2.0:
                f.write(f"{hotspot['function']:<40} "
                       f"{hotspot['percent_time']:>7.2f}% "
                       f"{hotspot['cumulative_time']:>11.3f}s "
                       f"{hotspot['calls']:>10,}\n")
        
        # Optimization recommendations
        f.write("\n\n3. OPTIMIZATION RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Top 5 targets
        top_targets = [h for h in hotspots if h['percent_time'] > 5.0][:5]
        
        for i, target in enumerate(top_targets, 1):
            f.write(f"\nTarget {i}: {target['function']}\n")
            f.write(f"  File: {target['file']}:{target['line']}\n")
            f.write(f"  Impact: {target['percent_time']:.1f}% of total runtime\n")
            f.write(f"  Opportunity: High priority for optimization\n")
            
            # Suggest optimization technique
            if target['calls'] > 100000:
                f.write(f"  Suggestion: Vectorize (called {target['calls']:,} times)\n")
            elif target['cumulative_time'] > 1.0:
                f.write(f"  Suggestion: JIT compile with Numba\n")
            else:
                f.write(f"  Suggestion: Profile deeper or parallelize\n")
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("End of report\n")
        f.write("=" * 80 + "\n")
    
    print(f"  ✓ Report saved to {output_file}")


def main():
    """Main profiling workflow."""
    print("=" * 80)
    print("🚀 Phase 3 Profiling - Identifying Optimization Targets")
    print("=" * 80)
    
    # Generate test data
    n_points = 100_000  # Moderate size for profiling
    points, classification, intensity = generate_test_data(n_points)
    
    # Profile feature computation
    feature_results = profile_feature_computation(points, k=20)
    
    # Deep profile with cProfile
    def run_full_pipeline():
        """Run full feature computation pipeline."""
        normals = compute_normals(points, k=20)
        curvature = compute_curvature(points, normals, k=20)
        from ign_lidar.features.utils import build_kdtree, compute_local_eigenvalues
        tree = build_kdtree(points)
        eigenvalues = compute_local_eigenvalues(points, tree, k=20)
        eig_features = compute_eigenvalue_features(eigenvalues)
        return normals, curvature, eig_features
    
    profile_with_cprofile(
        run_full_pipeline,
        output_file='baseline_profile.stats'
    )
    
    # Analyze profile
    hotspots = analyze_profile('baseline_profile.stats', top_n=30)
    
    # Generate report
    generate_report(feature_results, hotspots, 'profile_report.txt')
    
    # Save hotspots as JSON
    hotspots_file = 'hotspots.json'
    with open(hotspots_file, 'w') as f:
        json.dump({
            'feature_results': feature_results,
            'hotspots': hotspots[:20],
            'metadata': {
                'n_points': n_points,
                'k_neighbors': 20,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }, f, indent=2)
    
    print(f"\n✅ Hotspots saved to {hotspots_file}")
    
    print("\n" + "=" * 80)
    print("✅ Profiling complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review profile_report.txt for optimization targets")
    print("2. Visualize with: snakeviz baseline_profile.stats")
    print("3. Start Sprint 1: NumPy vectorization of top targets")
    print("\nTop optimization opportunities:")
    
    top_5 = [h for h in hotspots if h['percent_time'] > 3.0][:5]
    for i, target in enumerate(top_5, 1):
        print(f"  {i}. {target['function']} ({target['percent_time']:.1f}% runtime)")


if __name__ == '__main__':
    main()
