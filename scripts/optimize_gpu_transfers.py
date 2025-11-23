#!/usr/bin/env python3
"""
GPU Transfer Optimization Script - November 2025 Audit

This script analyzes and provides fixes for GPU memory transfer bottlenecks
identified in the codebase audit.

Usage:
    python scripts/optimize_gpu_transfers.py --analyze
    python scripts/optimize_gpu_transfers.py --profile /path/to/tile.laz
    python scripts/optimize_gpu_transfers.py --report

Author: IGN LiDAR HD Development Team
Date: November 23, 2025
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("⚠️  CuPy not available. GPU optimization features disabled.")

import numpy as np

try:
    from ign_lidar.optimization.gpu_transfer_profiler import (
        GPUTransferProfiler,
        enable_automatic_tracking,
    )
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
    GPUTransferProfiler = None
    enable_automatic_tracking = None

logger = logging.getLogger(__name__)


def analyze_codebase() -> Dict[str, Dict[str, int]]:
    """
    Analyze codebase for GPU transfer patterns.
    
    Returns:
        Dictionary with transfer statistics by file
    """
    import re
    from collections import defaultdict
    
    project_root = Path(__file__).parent.parent
    python_files = list(project_root.glob("ign_lidar/**/*.py"))
    
    stats = defaultdict(lambda: {"asarray": 0, "asnumpy": 0, "total": 0})
    
    # Patterns to detect transfers
    asarray_pattern = re.compile(r'\bcp\.asarray\(|cupy\.asarray\(')
    asnumpy_pattern = re.compile(r'\bcp\.asnumpy\(|cupy\.asnumpy\(|\b\.get\(\)')
    
    print("🔍 Analyzing codebase for GPU transfers...\n")
    
    for py_file in python_files:
        try:
            content = py_file.read_text()
            
            asarray_count = len(asarray_pattern.findall(content))
            asnumpy_count = len(asnumpy_pattern.findall(content))
            
            if asarray_count > 0 or asnumpy_count > 0:
                rel_path = py_file.relative_to(project_root)
                stats[str(rel_path)]["asarray"] = asarray_count
                stats[str(rel_path)]["asnumpy"] = asnumpy_count
                stats[str(rel_path)]["total"] = asarray_count + asnumpy_count
        except Exception as e:
            logger.warning(f"Could not analyze {py_file}: {e}")
    
    # Sort by total transfers
    sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True))
    
    print("📊 GPU Transfer Statistics\n" + "="*70)
    print(f"{'File':<50} {'CPU→GPU':<10} {'GPU→CPU':<10} {'Total':<10}")
    print("="*70)
    
    total_asarray = 0
    total_asnumpy = 0
    
    for file_path, counts in list(sorted_stats.items())[:20]:  # Top 20
        print(f"{file_path:<50} {counts['asarray']:<10} {counts['asnumpy']:<10} {counts['total']:<10}")
        total_asarray += counts['asarray']
        total_asnumpy += counts['asnumpy']
    
    print("="*70)
    print(f"{'TOTAL (all files)':<50} {total_asarray:<10} {total_asnumpy:<10} {total_asarray + total_asnumpy:<10}")
    print()
    
    return sorted_stats


def profile_tile_processing(tile_path: str):
    """
    Profile GPU transfers during tile processing.
    
    Args:
        tile_path: Path to LAZ tile to process
    """
    if not HAS_CUPY:
        print("❌ CuPy required for profiling")
        return
    
    from ign_lidar import LiDARProcessor
    
    print(f"🔬 Profiling GPU transfers for: {tile_path}\n")
    
    if not HAS_PROFILER:
        print("❌ GPU Transfer Profiler not available")
        return
    
    # Enable automatic tracking
    enable_automatic_tracking()
    profiler = GPUTransferProfiler()
    profiler.start()
    
    try:
        # Create processor with GPU enabled
        from ign_lidar import LiDARProcessor
        processor = LiDARProcessor(use_gpu=True)
        
        # Process tile (simplified - just feature computation)
        print("Processing tile...")
        # Note: This would need actual tile processing logic
        print("⚠️  Full tile processing not implemented in this script")
        print("    Use gpu_transfer_profiler directly in your pipeline")
        
    finally:
        profiler.stop()
    
    # Print results
    stats = profiler.get_summary()
    print("\n" + "="*70)
    print("GPU TRANSFER REPORT")
    print("="*70)
    print(f"Total CPU→GPU transfers: {stats.get('cpu_to_gpu_count', 0)}")
    print(f"Total GPU→CPU transfers: {stats.get('gpu_to_cpu_count', 0)}")
    print(f"Total bytes transferred: {stats.get('total_bytes', 0):,}")


def generate_optimization_report():
    """Generate detailed optimization recommendations."""
    
    print("📋 GPU Transfer Optimization Recommendations\n")
    print("="*70)
    
    recommendations = [
        {
            "priority": "🔴 CRITICAL",
            "title": "Vectorize Loop Transfers in gpu_kernels.py",
            "file": "ign_lidar/optimization/gpu_kernels.py",
            "line": "948-950",
            "issue": "Loop with N×3 individual transfers",
            "fix": """
# BEFORE:
for i in range(n_points):
    normals[i] = cp.asnumpy(normal)
    eigenvalues[i] = cp.asnumpy(evals)
    curvature[i] = float(cp.asnumpy(curv))

# AFTER:
normals = cp.asnumpy(normals_gpu)
eigenvalues = cp.asnumpy(eigenvalues_gpu)
curvature = cp.asnumpy(curvature_gpu)
            """,
            "impact": "50-100× reduction in transfers"
        },
        {
            "priority": "🔴 CRITICAL",
            "title": "Batch Transfers in ground_truth_classifier.py",
            "file": "ign_lidar/optimization/ground_truth_classifier.py",
            "line": "392-396",
            "issue": "5 separate transfers that could be batched",
            "fix": """
# BEFORE:
height_gpu = cp.asarray(height)
planarity_gpu = cp.asarray(planarity)
intensity_gpu = cp.asarray(intensity)
points_gpu = cp.asarray(chunk_points)

# AFTER:
# Stack features into single array
features = np.stack([height, planarity, intensity], axis=1)
features_gpu = cp.asarray(features)
points_gpu = cp.asarray(chunk_points)
# Now: 2 transfers instead of 5
            """,
            "impact": "2.5× reduction in transfers"
        },
        {
            "priority": "🟡 MODERATE",
            "title": "Keep Data on GPU Pipeline",
            "file": "features/gpu_processor.py",
            "line": "multiple",
            "issue": "Data transferred back to CPU between operations",
            "fix": """
# Create a GPU pipeline that keeps data on GPU:
class GPUPipeline:
    def __init__(self):
        self.cache = {}
    
    def process_features(self, points_gpu):
        # All operations on GPU
        normals_gpu = self.compute_normals_gpu(points_gpu)
        curvature_gpu = self.compute_curvature_gpu(points_gpu, normals_gpu)
        features_gpu = self.compute_all_features_gpu(
            points_gpu, normals_gpu, curvature_gpu
        )
        # Single transfer at the end
        return cp.asnumpy(features_gpu)
            """,
            "impact": "2-3× reduction in transfers"
        },
        {
            "priority": "🟡 MODERATE",
            "title": "Enable Transfer Profiling by Default",
            "file": "ign_lidar/core/processor.py",
            "line": "init",
            "issue": "Profiling not enabled, no visibility",
            "fix": """
# Add to LiDARProcessor.__init__:
if use_gpu and config.get('profile_gpu', False):
    from ign_lidar.optimization import enable_automatic_tracking
    enable_automatic_tracking()
    logger.info("GPU transfer profiling enabled")
            """,
            "impact": "Better visibility for optimization"
        },
        {
            "priority": "🟢 LOW",
            "title": "Add GPU Pipeline Documentation",
            "file": "docs/guides/gpu_optimization.md",
            "line": "new",
            "issue": "No clear guidelines for GPU optimization",
            "fix": """
Create comprehensive GPU optimization guide:
- Pattern: Minimize transfers
- Pattern: Batch operations
- Pattern: Keep data on GPU
- Anti-pattern: Loop transfers
- Anti-pattern: Unnecessary round-trips
            """,
            "impact": "Prevent future bottlenecks"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['priority']} - {rec['title']}")
        print(f"   File: {rec['file']}")
        print(f"   Line: {rec['line']}")
        print(f"   Issue: {rec['issue']}")
        print(f"   Impact: {rec['impact']}")
        print(f"\n   Suggested Fix:")
        for line in rec['fix'].strip().split('\n'):
            print(f"   {line}")
        print("\n" + "-"*70)
    
    print("\n✅ Implementation Priority:")
    print("   1. Fix critical loop transfers (gpu_kernels.py)")
    print("   2. Batch ground truth transfers")
    print("   3. Implement GPU-only pipeline")
    print("   4. Enable profiling")
    print("   5. Document patterns")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="GPU Transfer Optimization Tool"
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze codebase for GPU transfer patterns'
    )
    parser.add_argument(
        '--profile',
        metavar='TILE_PATH',
        help='Profile GPU transfers for a specific tile'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate optimization recommendations'
    )
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.profile, args.report]):
        parser.print_help()
        return
    
    if args.analyze:
        analyze_codebase()
    
    if args.profile:
        profile_tile_processing(args.profile)
    
    if args.report:
        generate_optimization_report()


if __name__ == "__main__":
    main()
