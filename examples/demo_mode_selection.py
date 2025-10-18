"""
Demo script for automatic mode selection

Demonstrates the ModeSelector in action with various point cloud sizes
and hardware configurations.

Usage:
    python examples/demo_mode_selection.py

Author: Simon Ducournau / GitHub Copilot
Date: October 18, 2025
"""

import numpy as np
from ign_lidar.features.mode_selector import get_mode_selector, ComputationMode


def print_separator():
    """Print a visual separator."""
    print("\n" + "="*70)


def demo_basic_selection():
    """Demonstrate basic automatic mode selection."""
    print_separator()
    print("DEMO 1: Basic Automatic Mode Selection")
    print_separator()
    
    # Get a mode selector (auto-detects hardware)
    selector = get_mode_selector()
    
    # Test various point cloud sizes
    test_sizes = [
        50_000,      # Small
        500_000,     # Medium
        2_000_000,   # Large
        10_000_000,  # Very large
    ]
    
    print(f"\nHardware Configuration:")
    print(f"  GPU Available: {selector.gpu_available}")
    print(f"  GPU Memory: {selector.gpu_memory_gb:.1f} GB")
    print(f"  CPU Memory: {selector.cpu_memory_gb:.1f} GB")
    
    print(f"\nAutomatic Mode Selection:")
    print(f"  {'Points':>12}  {'Selected Mode':<20}  {'Reason':<40}")
    print(f"  {'-'*12}  {'-'*20}  {'-'*40}")
    
    for num_points in test_sizes:
        mode = selector.select_mode(num_points=num_points)
        
        # Get reason based on mode
        if mode == ComputationMode.CPU:
            reason = "Small cloud or GPU unavailable"
        elif mode == ComputationMode.GPU:
            reason = "Optimal for this size with GPU"
        elif mode == ComputationMode.GPU_CHUNKED:
            reason = "Large cloud, chunked processing"
        else:
            reason = "Boundary computation"
        
        print(f"  {num_points:>12,}  {mode.value:<20}  {reason:<40}")


def demo_recommendations():
    """Demonstrate detailed recommendations."""
    print_separator()
    print("DEMO 2: Detailed Recommendations")
    print_separator()
    
    selector = get_mode_selector()
    
    # Get recommendations for a specific size
    num_points = 2_000_000
    recommendations = selector.get_recommendations(num_points=num_points)
    
    print(f"\nRecommendations for {num_points:,} points:")
    print(f"\n  Recommended Mode: {recommendations['recommended_mode'].upper()}")
    print(f"  Estimated Memory: {recommendations['estimated_memory_gb']:.2f} GB")
    print(f"  Available Memory: {recommendations['available_memory_gb']:.2f} GB")
    print(f"  Memory Utilization: {recommendations['memory_utilization_pct']:.1f}%")
    print(f"  Estimated Time: {recommendations['estimated_time_seconds']:.1f} seconds")
    
    print(f"\n  Alternative Modes:")
    for alt in recommendations['alternative_modes']:
        viable = "✅" if alt['viable'] else "❌"
        print(f"    {viable} {alt['mode']:<15} - {alt['reason']}")
        print(f"       Memory: {alt['estimated_memory_gb']:.2f} GB / "
              f"{alt['available_memory_gb']:.2f} GB available")


def demo_force_modes():
    """Demonstrate forcing specific modes."""
    print_separator()
    print("DEMO 3: Forcing Specific Modes")
    print_separator()
    
    selector = get_mode_selector()
    num_points = 1_000_000
    
    print(f"\nProcessing {num_points:,} points:")
    
    # Automatic selection
    mode_auto = selector.select_mode(num_points=num_points)
    print(f"  Automatic: {mode_auto.value}")
    
    # Force CPU
    mode_cpu = selector.select_mode(num_points=num_points, force_cpu=True)
    print(f"  Force CPU: {mode_cpu.value}")
    
    # Force GPU (if available)
    if selector.gpu_available:
        mode_gpu = selector.select_mode(num_points=num_points, force_gpu=True)
        print(f"  Force GPU: {mode_gpu.value}")
    else:
        print(f"  Force GPU: Not available (no GPU detected)")
    
    # User override
    mode_override = selector.select_mode(
        num_points=num_points,
        user_mode=ComputationMode.CPU
    )
    print(f"  User Override (CPU): {mode_override.value}")


def demo_boundary_mode():
    """Demonstrate boundary mode selection."""
    print_separator()
    print("DEMO 4: Boundary Mode Selection")
    print_separator()
    
    selector = get_mode_selector()
    
    print("\nBoundary mode for cross-tile feature computation:")
    
    test_sizes = [5_000, 50_000, 500_000]
    
    for num_points in test_sizes:
        mode = selector.select_mode(num_points=num_points, boundary_mode=True)
        print(f"  {num_points:>8,} points: {mode.value}")


def demo_memory_constraints():
    """Demonstrate mode selection with memory constraints."""
    print_separator()
    print("DEMO 5: Mode Selection with Memory Constraints")
    print_separator()
    
    # Simulate different hardware configurations
    configs = [
        {"name": "High-End Workstation", "gpu_gb": 24.0, "cpu_gb": 64.0},
        {"name": "Mid-Range Desktop", "gpu_gb": 8.0, "cpu_gb": 32.0},
        {"name": "Laptop (No GPU)", "gpu_gb": 0.0, "cpu_gb": 16.0},
    ]
    
    num_points = 5_000_000
    
    print(f"\nMode selection for {num_points:,} points on different hardware:\n")
    
    for config in configs:
        # Mock the GPU availability based on gpu_gb
        selector = get_mode_selector(
            gpu_memory_gb=config["gpu_gb"],
            cpu_memory_gb=config["cpu_gb"]
        )
        
        # Override GPU availability check
        selector.gpu_available = config["gpu_gb"] > 0
        
        mode = selector.select_mode(num_points=num_points)
        est_mem, avail_mem = selector.estimate_memory_usage(num_points, mode)
        
        print(f"  {config['name']}:")
        print(f"    Hardware: GPU {config['gpu_gb']:.0f}GB / CPU {config['cpu_gb']:.0f}GB")
        print(f"    Selected: {mode.value}")
        print(f"    Memory: {est_mem:.2f} GB / {avail_mem:.2f} GB")
        print()


def demo_performance_estimates():
    """Demonstrate performance time estimates."""
    print_separator()
    print("DEMO 6: Performance Time Estimates")
    print_separator()
    
    selector = get_mode_selector()
    
    # Test various sizes
    test_sizes = [100_000, 1_000_000, 5_000_000, 10_000_000]
    
    print("\nEstimated processing times:\n")
    print(f"  {'Points':>12}  {'Mode':<15}  {'Time (sec)':<12}  {'Throughput':<20}")
    print(f"  {'-'*12}  {'-'*15}  {'-'*12}  {'-'*20}")
    
    for num_points in test_sizes:
        recommendations = selector.get_recommendations(num_points=num_points)
        mode = recommendations['recommended_mode']
        time_sec = recommendations['estimated_time_seconds']
        throughput = num_points / time_sec if time_sec > 0 else 0
        
        print(f"  {num_points:>12,}  {mode:<15}  {time_sec:>12.1f}  "
              f"{throughput:>12,.0f} pts/sec")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("  IGN LIDAR HD - Automatic Mode Selection Demos")
    print("="*70)
    
    try:
        demo_basic_selection()
        demo_recommendations()
        demo_force_modes()
        demo_boundary_mode()
        demo_memory_constraints()
        demo_performance_estimates()
        
        print_separator()
        print("All demos completed successfully!")
        print_separator()
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
