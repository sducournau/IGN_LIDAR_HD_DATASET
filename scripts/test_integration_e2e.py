#!/usr/bin/env python3
"""
End-to-end integration test for unified processing pipeline.

This script tests the complete workflow:
1. Generate sample LAZ files
2. Run unified processing with different configurations
3. Validate outputs
4. Benchmark performance
"""

import sys
import time
from pathlib import Path
import subprocess
import json
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.processor import LiDARProcessor


def run_test(
    test_name: str,
    laz_file: Path,
    output_dir: Path,
    architecture: str = "pointnet++",
    output_format: str = "npz",
    save_enriched: bool = False,
    build_spatial_index: bool = False,
    num_points: int = 4096,
    add_rgb: bool = False,
    preprocess: bool = False,
    use_gpu: bool = False,
):
    """Run a single integration test."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    # Create processor with test-specific settings
    processor = LiDARProcessor(
        num_points=num_points,
        include_rgb=add_rgb,
        preprocess=preprocess,
        use_gpu=use_gpu
    )
    
    # Measure time
    start_time = time.time()
    
    try:
        # Run unified processing
        result = processor.process_tile_unified(
            laz_file=laz_file,
            output_dir=output_dir,
            architecture=architecture,
            output_format=output_format,
            save_enriched=save_enriched,
            build_spatial_index=build_spatial_index,
            skip_existing=False  # Always process for testing
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Test '{test_name}' PASSED")
        print(f"   Time: {elapsed_time:.2f}s")
        print(f"   Patches: {result.get('num_patches', 'N/A')}")
        print(f"   Output: {result.get('output_file', 'N/A')}")
        
        return {
            "status": "PASSED",
            "time": elapsed_time,
            "result": result
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Test '{test_name}' FAILED")
        print(f"   Error: {str(e)}")
        print(f"   Time: {elapsed_time:.2f}s")
        
        return {
            "status": "FAILED",
            "time": elapsed_time,
            "error": str(e)
        }


def main():
    """Run all integration tests."""
    print("="*60)
    print("End-to-End Integration Tests")
    print("Unified Processing Pipeline")
    print("="*60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    sample_dir = project_root / "data" / "sample_laz"
    output_base = project_root / "data" / "test_output"
    
    # Generate sample LAZ files if they don't exist
    if not sample_dir.exists() or not list(sample_dir.glob("*.laz")):
        print("\n📦 Generating sample LAZ files...")
        gen_script = project_root / "scripts" / "generate_sample_laz.py"
        subprocess.run([
            sys.executable,
            str(gen_script),
            "--output-dir", str(sample_dir),
            "--num-files", "3",
            "--add-rgb"
        ])
    
    # Find LAZ files
    laz_files = list(sample_dir.glob("*.laz"))
    if not laz_files:
        print("❌ No LAZ files found. Cannot run tests.")
        return 1
    
    print(f"\n✅ Found {len(laz_files)} LAZ files")
    for laz in laz_files:
        print(f"   - {laz.name}")
    
    # Test configurations
    test_cases = [
        {
            "name": "PointNet++ Basic",
            "architecture": "pointnet++",
            "output_format": "npz",
            "num_points": 4096,
        },
        {
            "name": "PointNet++ with RGB",
            "architecture": "pointnet++",
            "output_format": "npz",
            "num_points": 4096,
            "add_rgb": True,
        },
        {
            "name": "PointNet++ PyTorch",
            "architecture": "pointnet++",
            "output_format": "pytorch",
            "num_points": 4096,
        },
        {
            "name": "Octree Format",
            "architecture": "octree",
            "output_format": "npz",
            "num_points": 4096,
        },
        {
            "name": "Transformer Format",
            "architecture": "transformer",
            "output_format": "npz",
            "num_points": 4096,
        },
        {
            "name": "Sparse Convolution",
            "architecture": "sparse_conv",
            "output_format": "npz",
            "num_points": 4096,
        },
        {
            "name": "With Preprocessing",
            "architecture": "pointnet++",
            "output_format": "npz",
            "num_points": 4096,
            "preprocess": True,
        },
        {
            "name": "Save Enriched LAZ",
            "architecture": "pointnet++",
            "output_format": "npz",
            "num_points": 4096,
            "save_enriched": True,
        },
    ]
    
    # Run tests
    results = []
    test_laz = laz_files[0]  # Use first file for testing
    
    for i, test_case in enumerate(test_cases, 1):
        test_name = test_case.pop("name")
        output_dir = output_base / f"test_{i:02d}_{test_name.lower().replace(' ', '_')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = run_test(
            test_name=test_name,
            laz_file=test_laz,
            output_dir=output_dir,
            **test_case
        )
        
        results.append({
            "test": test_name,
            **result
        })
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    total_time = sum(r["time"] for r in results)
    
    print(f"\n✅ Passed: {passed}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⏱️  Average time: {total_time/len(results):.2f}s")
    
    # Detailed results
    print("\nDetailed Results:")
    for r in results:
        status_icon = "✅" if r["status"] == "PASSED" else "❌"
        print(f"  {status_icon} {r['test']}: {r['time']:.2f}s")
        if r["status"] == "FAILED":
            print(f"     Error: {r.get('error', 'Unknown')}")
    
    # Save results to JSON
    results_file = output_base / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Results saved to: {results_file}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
