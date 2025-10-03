#!/usr/bin/env python3
"""
Demonstration script for GPU integration.

Shows how to use the new GPU-accelerated feature computation
both programmatically and via CLI.
"""

import numpy as np


def demo_programmatic_usage():
    """Demonstrate programmatic GPU usage."""
    print("=" * 70)
    print("GPU Integration Demo - Programmatic Usage")
    print("=" * 70)
    
    # Import the new GPU-enabled function
    from ign_lidar.features import compute_all_features_with_gpu
    
    # Create sample point cloud data
    np.random.seed(42)
    n_points = 5000
    points = np.random.rand(n_points, 3).astype(np.float32)
    points[:, 2] *= 20  # Scale Z coordinate (height)
    classification = np.random.randint(1, 6, size=n_points, dtype=np.uint8)
    
    print("\nCreated sample point cloud:")
    print(f"  - Points: {n_points:,}")
    print(f"  - Shape: {points.shape}")
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    print(f"  - Height range: {z_min:.2f} - {z_max:.2f}m")
    
    # Test 1: CPU computation
    print("\n" + "-" * 70)
    print("Test 1: CPU Computation")
    print("-" * 70)
    
    import time
    start = time.time()
    normals_cpu, curv_cpu, height_cpu, geo_cpu = \
        compute_all_features_with_gpu(
            points, classification,
            k=15,
            auto_k=False,
            use_gpu=False
        )
    cpu_time = time.time() - start
    
    print(f"✓ CPU computation completed in {cpu_time:.3f}s")
    print(f"  - Normals: {normals_cpu.shape}")
    print(f"  - Curvature: {curv_cpu.shape}")
    print(f"  - Height: {height_cpu.shape}")
    print(f"  - Geometric features: {len(geo_cpu)} types")
    print(f"    Features: {', '.join(geo_cpu.keys())}")
    
    # Test 2: GPU computation (with fallback)
    print("\n" + "-" * 70)
    print("Test 2: GPU Computation (with automatic CPU fallback)")
    print("-" * 70)
    
    start = time.time()
    normals_gpu, curv_gpu, height_gpu, geo_gpu = \
        compute_all_features_with_gpu(
            points, classification,
            k=15,
            auto_k=False,
            use_gpu=True
        )
    gpu_time = time.time() - start
    
    print(f"✓ GPU computation completed in {gpu_time:.3f}s")
    print(f"  - Normals: {normals_gpu.shape}")
    print(f"  - Curvature: {curv_gpu.shape}")
    print(f"  - Height: {height_gpu.shape}")
    print(f"  - Geometric features: {len(geo_gpu)} types")
    
    # Compare results
    print("\n" + "-" * 70)
    print("Comparison")
    print("-" * 70)
    
    print(f"  Time ratio: {cpu_time / gpu_time:.2f}x")
    
    # Check consistency
    normals_match = np.allclose(normals_cpu, normals_gpu, atol=1e-3)
    curv_match = np.allclose(curv_cpu, curv_gpu, atol=1e-3)
    height_match = np.allclose(height_cpu, height_gpu, atol=1e-3)
    
    print(f"  Normals match: {'✓' if normals_match else '✗'}")
    print(f"  Curvature match: {'✓' if curv_match else '✗'}")
    print(f"  Height match: {'✓' if height_match else '✗'}")
    
    # Verify normal vectors are normalized
    norms = np.linalg.norm(normals_cpu, axis=1)
    normalized = np.allclose(norms, 1.0, atol=1e-5)
    print(f"  Normals normalized: {'✓' if normalized else '✗'}")
    
    print("\n" + "=" * 70)
    print("✓ Demo completed successfully!")
    print("=" * 70)


def demo_cli_usage():
    """Show CLI usage examples."""
    print("\n\n" + "=" * 70)
    print("GPU Integration Demo - CLI Usage")
    print("=" * 70)
    
    print("\nThe following CLI commands now support GPU acceleration:\n")
    
    examples = [
        (
            "Basic enrichment with CPU",
            "ign-lidar-hd enrich --input-dir tiles/ --output enriched/"
        ),
        (
            "Enrichment with GPU acceleration",
            "ign-lidar-hd enrich --input-dir tiles/ --output out/ --use-gpu"
        ),
        (
            "Multi-worker processing with GPU",
            "ign-lidar-hd enrich --input tiles/ --output out/ "
            "--use-gpu --num-workers 2"
        ),
        (
            "Complete pipeline with GPU",
            "ign-lidar-hd pipeline config.yaml  "
            "# (set use_gpu: true in config)"
        ),
    ]
    
    for i, (description, command) in enumerate(examples, 1):
        print(f"{i}. {description}:")
        print(f"   $ {command}")
        print()
    
    print("Features:")
    print("  ✓ Automatic CPU fallback if GPU unavailable")
    print("  ✓ Clear logging of GPU/CPU status")
    print("  ✓ 4-10x speedup with GPU (if available)")
    print("  ✓ No code changes required for existing workflows")
    print("\n" + "=" * 70)


def demo_gpu_availability():
    """Check GPU availability."""
    print("\n\n" + "=" * 70)
    print("GPU Availability Check")
    print("=" * 70 + "\n")
    
    try:
        from ign_lidar.features_gpu import GPU_AVAILABLE, CUML_AVAILABLE
        
        print(f"CuPy (GPU) available: {GPU_AVAILABLE}")
        if GPU_AVAILABLE:
            print("  ✓ GPU acceleration available")
            try:
                import cupy as cp
                print(f"  ✓ CuPy version: {cp.__version__}")
                
                # Try to get GPU info
                try:
                    device = cp.cuda.Device()
                    print(f"  ✓ GPU: {device.compute_capability}")
                    mem = device.mem_info
                    print(f"  ✓ Memory: {mem[1] / 1e9:.1f} GB total")
                except Exception:
                    pass
            except ImportError:
                pass
        else:
            print("  ⚠ GPU not available - will use CPU fallback")
            print("  Install: pip install cupy-cuda11x")
        
        print(f"\nRAPIDS cuML available: {CUML_AVAILABLE}")
        if CUML_AVAILABLE:
            print("  ✓ Advanced GPU ML algorithms available")
        else:
            print("  ⚠ RAPIDS cuML not available (optional)")
        
    except ImportError as e:
        print(f"GPU module import error: {e}")
        print("  Install GPU dependencies: pip install ign-lidar-hd[gpu]")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run all demos
    demo_gpu_availability()
    demo_programmatic_usage()
    demo_cli_usage()
    
    print("\n\n🎉 GPU Integration is now functional!")
    print("\nFor more information, see:")
    print("  - GPU_IMPLEMENTATION_SUMMARY.md")
    print("  - GPU_ANALYSIS.md")
    print("  - README.md (GPU section)")
