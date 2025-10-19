#!/usr/bin/env python3
"""
Quick test to verify the chunking fix for large point clouds.
"""
import sys
sys.path.insert(0, '/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET')

from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer

# Test the _should_batch_neighbor_queries method
computer = GPUChunkedFeatureComputer(
    use_gpu=True,
    chunk_size=5_000_000,
    neighbor_query_batch_size=30_000_000,
    vram_limit_gb=14.0
)

print("=" * 80)
print("Testing _should_batch_neighbor_queries with different point cloud sizes")
print("=" * 80)

# Test cases: (N_points, k_neighbors, available_vram_gb)
test_cases = [
    (18_651_688, 20, 13.77, "Typical tile - 18.6M points"),
    (30_000_000, 20, 13.77, "Large tile - 30M points"),
    (5_000_000, 20, 13.77, "Small tile - 5M points"),
    (18_651_688, 20, 8.0, "Typical tile - Limited VRAM (8GB)"),
]

for n_points, k, vram_gb, description in test_cases:
    print(f"\n📊 Test: {description}")
    print(f"   Points: {n_points:,} | k: {k} | Available VRAM: {vram_gb:.2f}GB")
    
    should_batch, batch_size, num_batches = computer._should_batch_neighbor_queries(
        n_points, k, vram_gb
    )
    
    print(f"\n   Result:")
    print(f"   - Should batch: {should_batch}")
    print(f"   - Batch size: {batch_size:,}")
    print(f"   - Num batches: {num_batches}")
    print(f"   - Memory per query: {(n_points * k * 8) / (1024**3):.2f}GB (output only)")
    print()

print("=" * 80)
print("✅ Test complete!")
print("=" * 80)
