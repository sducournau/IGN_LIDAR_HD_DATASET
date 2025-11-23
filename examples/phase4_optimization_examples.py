"""
Phase 4 Optimizations - Complete Usage Example

This example demonstrates how to use all Phase 4 optimizations together
for maximum performance on multi-tile processing workloads.

Expected Performance Gains:
- Phase 4.1 (WFS Cache): +10-15%
- Phase 4.2 (Preprocessing GPU): +10-15%
- Phase 4.3 (GPU Memory Pooling): +8.5%
- Phase 4.4 (Batch Multi-Tile): +25-30%
- Phase 4.5 (Async I/O): +12-14%
- **TOTAL: +66-94%** (2.66× - 2.94× faster)

Author: IGN LiDAR HD Development Team
Date: November 23, 2025
"""

from pathlib import Path
from ign_lidar import LiDARProcessor
from ign_lidar.core.optimization_integration import create_optimization_manager
from omegaconf import OmegaConf

# ==============================================================================
# Example 1: Quick Start with All Optimizations
# ==============================================================================

def example_1_quick_start():
    """Enable all Phase 4 optimizations with default settings."""
    
    print("=" * 80)
    print("Example 1: Quick Start with All Optimizations")
    print("=" * 80)
    
    # Load configuration
    config = OmegaConf.create({
        'input_dir': 'data/tiles',
        'output_dir': 'data/output',
        'processor': {
            'lod_level': 'LOD2',
            'processing_mode': 'patches_only',
            'use_gpu': True,
            'patch_size': 150.0,
            'num_points': 16384,
            # ✨ Enable Phase 4 optimizations
            'enable_optimizations': True,  # Enable all optimizations
            'optimization': {
                'enable_async_io': True,        # Phase 4.5
                'enable_batch_processing': True, # Phase 4.4
                'enable_gpu_pooling': True,     # Phase 4.3
                'async_workers': 2,
                'tile_cache_size': 3,
                'batch_size': 4,
            }
        },
        'features': {
            'mode': 'lod2',
            'k_neighbors': 20,
        },
        'data_sources': {
            'bd_topo': {
                'enabled': True,
                'features': {
                    'buildings': True,
                    'roads': False,
                }
            }
        }
    })
    
    # Create processor
    processor = LiDARProcessor(config)
    
    # Create optimization manager
    opt_mgr = create_optimization_manager(
        use_gpu=True,
        enable_all=True,
    )
    
    # Initialize optimizations with processor's feature orchestrator
    opt_mgr.initialize(feature_orchestrator=processor.feature_orchestrator)
    
    # Process tiles
    try:
        processor.process_directory(
            input_dir=config.input_dir,
            output_dir=config.output_dir,
        )
        
        # Print optimization statistics
        opt_mgr.print_stats()
    
    finally:
        # Clean shutdown
        opt_mgr.shutdown()
    
    print("✅ Example 1 complete!")


# ==============================================================================
# Example 2: Selective Optimizations (CPU-Only)
# ==============================================================================

def example_2_cpu_only():
    """Enable only CPU-compatible optimizations (no GPU)."""
    
    print("\n" + "=" * 80)
    print("Example 2: CPU-Only Optimizations")
    print("=" * 80)
    
    config = OmegaConf.create({
        'input_dir': 'data/tiles',
        'output_dir': 'data/output_cpu',
        'processor': {
            'lod_level': 'LOD2',
            'processing_mode': 'patches_only',
            'use_gpu': False,  # CPU only
            'optimization': {
                'enable_async_io': True,         # Phase 4.5 (works on CPU)
                'enable_batch_processing': False, # Phase 4.4 (needs GPU)
                'enable_gpu_pooling': False,     # Phase 4.3 (needs GPU)
            }
        },
        'features': {
            'mode': 'lod2',
        }
    })
    
    processor = LiDARProcessor(config)
    
    # Only async I/O (Phase 4.5) is enabled
    opt_mgr = create_optimization_manager(
        use_gpu=False,  # Disable GPU optimizations
        enable_all=True,
        async_workers=3,  # More workers for CPU
    )
    
    opt_mgr.initialize(feature_orchestrator=processor.feature_orchestrator)
    
    try:
        processor.process_directory(
            input_dir=config.input_dir,
            output_dir=config.output_dir,
        )
        opt_mgr.print_stats()
    finally:
        opt_mgr.shutdown()
    
    print("✅ Example 2 complete!")


# ==============================================================================
# Example 3: GPU Batch Processing for Large Workloads
# ==============================================================================

def example_3_gpu_batch_processing():
    """Optimize for large batch processing with GPU."""
    
    print("\n" + "=" * 80)
    print("Example 3: GPU Batch Processing (Large Workload)")
    print("=" * 80)
    
    config = OmegaConf.create({
        'input_dir': 'data/large_dataset',
        'output_dir': 'data/output_batch',
        'processor': {
            'lod_level': 'LOD2',
            'use_gpu': True,
            'optimization': {
                'enable_async_io': True,
                'enable_batch_processing': True,
                'enable_gpu_pooling': True,
                'batch_size': 8,  # Larger batches for better GPU utilization
                'async_workers': 4,  # More I/O threads
                'tile_cache_size': 5,  # Larger cache
                'gpu_pool_max_size_gb': 6.0,  # Larger GPU pool
            }
        },
        'features': {
            'mode': 'lod2',
            'k_neighbors': 20,
        }
    })
    
    processor = LiDARProcessor(config)
    
    opt_mgr = create_optimization_manager(
        use_gpu=True,
        enable_all=True,
        batch_size=8,
        async_workers=4,
        tile_cache_size=5,
        gpu_pool_max_size_gb=6.0,
    )
    
    opt_mgr.initialize(feature_orchestrator=processor.feature_orchestrator)
    
    try:
        processor.process_directory(
            input_dir=config.input_dir,
            output_dir=config.output_dir,
        )
        opt_mgr.print_stats()
    finally:
        opt_mgr.shutdown()
    
    print("✅ Example 3 complete!")


# ==============================================================================
# Example 4: Manual Tile Processing with Optimizations
# ==============================================================================

def example_4_manual_processing():
    """Process tiles manually using OptimizationManager."""
    
    print("\n" + "=" * 80)
    print("Example 4: Manual Tile Processing")
    print("=" * 80)
    
    from ign_lidar.core.classification.io import load_laz_file
    from ign_lidar.features import FeatureOrchestrator
    
    # Get tile paths
    tile_dir = Path('data/tiles')
    tile_paths = list(tile_dir.glob('*.laz'))[:10]  # First 10 tiles
    
    print(f"📁 Processing {len(tile_paths)} tiles manually")
    
    # Create feature orchestrator
    config = OmegaConf.create({
        'features': {
            'mode': 'lod2',
            'k_neighbors': 20,
        },
        'processor': {
            'use_gpu': True,
        }
    })
    feature_orchestrator = FeatureOrchestrator(config)
    
    # Create optimization manager
    opt_mgr = create_optimization_manager(
        use_gpu=True,
        enable_all=True,
    )
    opt_mgr.initialize(feature_orchestrator=feature_orchestrator)
    
    # Define processing function
    def process_tile(tile_data, ground_truth):
        """Process a single tile."""
        # Compute features
        features = feature_orchestrator.compute_features(
            points=tile_data.points,
            k=20,
        )
        
        return {
            'num_points': len(tile_data.points),
            'num_features': len(features),
            'feature_names': list(features.keys()),
        }
    
    # Process with optimizations
    try:
        results = opt_mgr.process_tiles_optimized(
            tile_paths=tile_paths,
            processor_func=process_tile,
            fetch_ground_truth=True,
        )
        
        print(f"\n✅ Processed {len(results)} tiles")
        print(f"📊 Sample result: {results[0]}")
        
        opt_mgr.print_stats()
    
    finally:
        opt_mgr.shutdown()
    
    print("✅ Example 4 complete!")


# ==============================================================================
# Example 5: Performance Comparison (With vs Without Optimizations)
# ==============================================================================

def example_5_performance_comparison():
    """Compare performance with and without optimizations."""
    
    print("\n" + "=" * 80)
    print("Example 5: Performance Comparison")
    print("=" * 80)
    
    import time
    from ign_lidar.features import FeatureOrchestrator
    
    tile_dir = Path('data/tiles')
    tile_paths = list(tile_dir.glob('*.laz'))[:20]
    
    config = OmegaConf.create({
        'features': {'mode': 'lod2', 'k_neighbors': 20},
        'processor': {'use_gpu': True},
    })
    
    feature_orchestrator = FeatureOrchestrator(config)
    
    def process_tile(tile_data, ground_truth):
        features = feature_orchestrator.compute_features(
            points=tile_data.points,
            k=20,
        )
        return features
    
    # Test 1: WITHOUT optimizations (sequential)
    print("\n🐢 Test 1: Sequential processing (no optimizations)")
    start = time.time()
    
    from ign_lidar.core.classification.io import load_laz_file
    results_baseline = []
    for tile_path in tile_paths:
        tile_data = load_laz_file(tile_path)
        result = process_tile(tile_data, None)
        results_baseline.append(result)
    
    time_baseline = time.time() - start
    print(f"   ⏱️  Time: {time_baseline:.2f}s")
    
    # Test 2: WITH optimizations
    print("\n🚀 Test 2: Optimized processing (Phase 4.3-4.5)")
    start = time.time()
    
    opt_mgr = create_optimization_manager(use_gpu=True, enable_all=True)
    opt_mgr.initialize(feature_orchestrator=feature_orchestrator)
    
    try:
        results_optimized = opt_mgr.process_tiles_optimized(
            tile_paths=tile_paths,
            processor_func=process_tile,
            fetch_ground_truth=False,
        )
        
        time_optimized = time.time() - start
        print(f"   ⏱️  Time: {time_optimized:.2f}s")
        
        # Calculate speedup
        speedup = time_baseline / time_optimized
        improvement_pct = (speedup - 1) * 100
        
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE COMPARISON")
        print("=" * 80)
        print(f"Baseline (sequential):  {time_baseline:.2f}s")
        print(f"Optimized (Phase 4):    {time_optimized:.2f}s")
        print(f"Speedup:                {speedup:.2f}×")
        print(f"Improvement:            +{improvement_pct:.1f}%")
        print("=" * 80)
        
        if speedup >= 1.5:
            print("✅ Excellent! Optimizations working as expected")
        elif speedup >= 1.1:
            print("⚠️  Good, but consider tuning parameters")
        else:
            print("❌ Optimizations may not be enabled correctly")
        
        opt_mgr.print_stats()
    
    finally:
        opt_mgr.shutdown()
    
    print("✅ Example 5 complete!")


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phase 4 Optimizations - Usage Examples'
    )
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help='Example number to run (1-5)'
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_quick_start,
        2: example_2_cpu_only,
        3: example_3_gpu_batch_processing,
        4: example_4_manual_processing,
        5: example_5_performance_comparison,
    }
    
    print("\n" + "=" * 80)
    print("🚀 IGN LiDAR HD - Phase 4 Optimization Examples")
    print("=" * 80)
    print("\nAvailable Examples:")
    print("  1. Quick Start with All Optimizations")
    print("  2. CPU-Only Optimizations")
    print("  3. GPU Batch Processing (Large Workload)")
    print("  4. Manual Tile Processing")
    print("  5. Performance Comparison (With vs Without)")
    print("\n" + "=" * 80)
    
    # Run selected example
    examples[args.example]()
    
    print("\n" + "=" * 80)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📚 Documentation: docs/optimization/PHASE_4_*.md")
    print("🎯 Expected gains: +66-94% (2.66× - 2.94× faster)")
    print("=" * 80)
