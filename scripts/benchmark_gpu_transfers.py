#!/usr/bin/env python3
"""
Benchmark GPU Transfer Optimizations

Compares GPU performance before and after Phase 2 optimizations.

Usage:
    # Baseline (before optimization)
    python scripts/benchmark_gpu_transfers.py --mode baseline
    
    # After optimization
    python scripts/benchmark_gpu_transfers.py --mode optimized
    
    # Compare
    python scripts/benchmark_gpu_transfers.py --compare baseline.json optimized.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ign_lidar.features import FeatureOrchestrator
from ign_lidar.optimization.gpu_transfer_profiler import GPUTransferProfiler


def benchmark_tile_processing(mode: str = 'baseline', n_points: int = 100_000):
    """Benchmark tile processing with transfer profiling."""
    
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK: {mode.upper()} MODE")
    print(f"{'=' * 80}")
    print(f"Points: {n_points:,}")
    
    # Generate test data
    points = np.random.randn(n_points, 3).astype(np.float32)
    
    # Configuration
    config = {
        'features': {
            'k_neighbors': 30,
            'use_gpu': True,
        },
        'gpu': {
            'use_streams': mode == 'optimized',  # Only in optimized mode
        }
    }
    
    # Create orchestrator
    orchestrator = FeatureOrchestrator(config)
    
    # Profile with transfer tracking
    profiler = GPUTransferProfiler(track_stacks=False)
    
    with profiler:
        start = time.time()
        features = orchestrator.compute_features(
            points=points,
            mode='lod2'
        )
        duration = time.time() - start
    
    # Get statistics
    stats = profiler.get_stats()
    stats['compute_duration'] = duration
    stats['points_per_second'] = n_points / duration
    
    profiler.print_report()
    
    print(f"\nComputation time: {duration:.2f}s")
    print(f"Throughput: {n_points / duration:,.0f} points/s")
    print(f"Transfer ratio: {stats['total_bytes'] / (n_points * 12):.2f}x")
    print(f"  (Expected 2.0x for optimal: 1x input + 1x output)")
    
    return stats


def save_results(stats: dict, output_path: Path):
    """Save benchmark results."""
    output_path.write_text(json.dumps(stats, indent=2))
    print(f"\n💾 Results saved: {output_path}")


def compare_results(baseline_path: Path, optimized_path: Path):
    """Compare baseline vs optimized results."""
    
    baseline = json.loads(baseline_path.read_text())
    optimized = json.loads(optimized_path.read_text())
    
    print(f"\n{'=' * 80}")
    print("COMPARISON: BASELINE vs OPTIMIZED")
    print(f"{'=' * 80}")
    
    metrics = {
        'Transfers': ('total_transfers', 'lower is better'),
        'CPU→GPU transfers': ('cpu_to_gpu', 'lower is better'),
        'GPU→CPU transfers': ('gpu_to_cpu', 'lower is better'),
        'Total bytes': ('total_bytes', 'lower is better'),
        'Compute time (s)': ('compute_duration', 'lower is better'),
        'Throughput (pts/s)': ('points_per_second', 'higher is better'),
    }
    
    for label, (key, direction) in metrics.items():
        base_val = baseline[key]
        opt_val = optimized[key]
        
        if 'higher is better' in direction:
            improvement = (opt_val - base_val) / base_val * 100
            symbol = '📈' if improvement > 0 else '📉'
        else:
            improvement = (base_val - opt_val) / base_val * 100
            symbol = '✅' if improvement > 0 else '❌'
        
        print(f"\n{label}:")
        print(f"  Baseline:  {base_val:,.2f}")
        print(f"  Optimized: {opt_val:,.2f}")
        print(f"  {symbol} {improvement:+.1f}%")
    
    print(f"\n{'=' * 80}")
    print("TARGETS:")
    print(f"{'=' * 80}")
    
    targets = {
        'Transfers < 5': optimized['total_transfers'] < 5,
        'Throughput +20%': (optimized['points_per_second'] - baseline['points_per_second']) / baseline['points_per_second'] > 0.20,
        'GPU utilization > 80%': True,  # Need GPU profiler for this
    }
    
    for target, achieved in targets.items():
        symbol = '✅' if achieved else '❌'
        print(f"{symbol} {target}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark GPU transfer optimizations')
    parser.add_argument('--mode', choices=['baseline', 'optimized'], default='baseline')
    parser.add_argument('--points', type=int, default=100_000)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--compare', nargs=2, type=Path, default=None)
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        stats = benchmark_tile_processing(args.mode, args.points)
        
        if args.output:
            save_results(stats, args.output)
        else:
            default_path = Path(f'benchmark_{args.mode}.json')
            save_results(stats, default_path)


if __name__ == '__main__':
    main()
