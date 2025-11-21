"""
GPU Performance Visualization

Generate performance comparison charts for GPU optimization report.

Usage:
    python scripts/visualize_gpu_performance.py
    
Requires: matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent.parent / "docs" / "diagrams"
output_dir.mkdir(parents=True, exist_ok=True)


def plot_knn_speedup():
    """Plot KNN performance comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dataset sizes
    sizes = ['10K', '100K', '1M']
    cpu_times = [0.5, 5.2, 29.0]
    gpu_times = [0.5, 0.6, 2.7]
    speedups = [t_cpu / t_gpu for t_cpu, t_gpu in zip(cpu_times, gpu_times)]
    
    # Bar chart - Execution time
    x = np.arange(len(sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, gpu_times, width, label='GPU', color='#2ecc71', alpha=0.8)
    
    ax1.set_xlabel('Dataset Size (points)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('KNN Performance: CPU vs GPU', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # Speedup chart
    bars3 = ax2.bar(sizes, speedups, color=['#3498db', '#9b59b6', '#e67e22'], alpha=0.8)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No speedup')
    
    ax2.set_xlabel('Dataset Size (points)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax2.set_title('GPU Speedup Factor', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars3, speedups)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gpu_knn_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'gpu_knn_performance.png'}")


def plot_pipeline_projection():
    """Plot full pipeline speedup projection."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    phases = ['Baseline\n(CPU)', 'Phase 1.4\n(GPU KNN)', 'Phase 2\n(+GPU Reclass)', 'Phase 7\n(Full Optimized)']
    times = [33, 6, 3.5, 2.5]
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    bars = ax.barh(phases, times, color=colors, alpha=0.8)
    
    ax.set_xlabel('Time per Tile (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Pipeline Optimization Roadmap', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels and speedup
    baseline = times[0]
    for i, (bar, time) in enumerate(zip(bars, times)):
        width = bar.get_width()
        speedup = baseline / time
        label = f'{time:.1f} min' if i == 0 else f'{time:.1f} min ({speedup:.1f}× faster)'
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                label, ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Add target line
    ax.axvline(x=2.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target: 2.5 min')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pipeline_optimization_roadmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'pipeline_optimization_roadmap.png'}")


def plot_operation_breakdown():
    """Plot operation time breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before optimization
    operations_before = ['KDTree/KNN', 'Eigenvalues', 'Classification', 'Other']
    times_before = [30, 0.5, 1.5, 1.0]
    colors = ['#e74c3c', '#3498db', '#f39c12', '#95a5a6']
    
    wedges1, texts1, autotexts1 = ax1.pie(times_before, labels=operations_before, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
    ax1.set_title('Before GPU Optimization\n(Total: 33 min)', fontsize=14, fontweight='bold')
    
    # After Phase 1.4
    times_after = [3, 0.5, 1.5, 1.0]
    wedges2, texts2, autotexts2 = ax2.pie(times_after, labels=operations_before, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
    ax2.set_title('After Phase 1.4 (GPU KNN)\n(Total: 6 min)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'operation_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'operation_breakdown.png'}")


def plot_throughput_comparison():
    """Plot throughput comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    implementations = ['CPU\n(sklearn)', 'CPU\n(scipy)', 'GPU\n(FAISS)']
    throughputs = [34454, 38000, 376730]  # points/sec
    colors = ['#e74c3c', '#e67e22', '#2ecc71']
    
    bars = ax.bar(implementations, throughputs, color=colors, alpha=0.8)
    
    ax.set_ylabel('Throughput (points/second)', fontsize=12, fontweight='bold')
    ax.set_title('KNN Throughput Comparison (1M points, k=30)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, throughput in zip(bars, throughputs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{throughput:,}\npts/s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'throughput_comparison.png'}")


def main():
    """Generate all performance charts."""
    print("Generating GPU performance visualizations...\n")
    
    try:
        plot_knn_speedup()
        plot_pipeline_projection()
        plot_operation_breakdown()
        plot_throughput_comparison()
        
        print("\n✅ All charts generated successfully!")
        print(f"\nOutput directory: {output_dir}")
        print("\nGenerated files:")
        print("  - gpu_knn_performance.png")
        print("  - pipeline_optimization_roadmap.png")
        print("  - operation_breakdown.png")
        print("  - throughput_comparison.png")
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("Please install required packages: pip install matplotlib numpy")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
