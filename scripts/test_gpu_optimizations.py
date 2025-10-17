"""
Test script to verify GPU optimizations and measure performance improvements.

This script benchmarks:
1. CUDA streams vs synchronous transfers
2. Pinned memory vs regular memory
3. GPU array caching effectiveness
4. Overall processing speedup

Usage:
    python test_gpu_optimizations.py
"""

import logging
import time
import numpy as np
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("✓ CuPy available")
except ImportError:
    GPU_AVAILABLE = False
    logger.error("✗ CuPy not available - GPU tests will be skipped")


def benchmark_memory_transfers():
    """Benchmark standard vs pinned memory transfers."""
    if not GPU_AVAILABLE:
        logger.warning("Skipping memory transfer benchmark (no GPU)")
        return
    
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK 1: Memory Transfer Performance")
    logger.info("="*60)
    
    sizes = [1_000_000, 5_000_000, 10_000_000]
    
    for size in sizes:
        # Create test data
        data = np.random.randn(size, 3).astype(np.float32)
        size_mb = data.nbytes / (1024**2)
        
        # Standard transfer
        times_standard = []
        for _ in range(5):
            start = time.time()
            gpu_data = cp.asarray(data)
            cp.cuda.Device().synchronize()
            times_standard.append(time.time() - start)
        
        # Pinned memory transfer
        times_pinned = []
        pinned_mem = cp.cuda.alloc_pinned_memory(data.nbytes)
        pinned_array = np.frombuffer(pinned_mem, dtype=np.float32).reshape(data.shape)
        pinned_array[:] = data
        
        for _ in range(5):
            start = time.time()
            gpu_data = cp.asarray(pinned_array)
            cp.cuda.Device().synchronize()
            times_pinned.append(time.time() - start)
        
        avg_standard = np.mean(times_standard) * 1000
        avg_pinned = np.mean(times_pinned) * 1000
        speedup = avg_standard / avg_pinned
        
        logger.info(f"\nData size: {size:,} points ({size_mb:.1f}MB)")
        logger.info(f"  Standard transfer: {avg_standard:.2f}ms")
        logger.info(f"  Pinned transfer:   {avg_pinned:.2f}ms")
        logger.info(f"  Speedup:           {speedup:.2f}x")
        
        del gpu_data, pinned_mem, pinned_array


def benchmark_cuda_streams():
    """Benchmark CUDA streams for overlapped processing."""
    if not GPU_AVAILABLE:
        logger.warning("Skipping CUDA streams benchmark (no GPU)")
        return
    
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK 2: CUDA Streams Performance")
    logger.info("="*60)
    
    try:
        from ign_lidar.optimization.cuda_streams import create_stream_manager
        
        num_chunks = 10
        chunk_size = 2_000_000
        
        # Generate test data
        chunks = [
            np.random.randn(chunk_size, 3).astype(np.float32)
            for _ in range(num_chunks)
        ]
        
        # Synchronous processing
        start = time.time()
        for chunk in chunks:
            gpu_chunk = cp.asarray(chunk)
            # Simulate computation
            result = gpu_chunk * 2.0
            cpu_result = cp.asnumpy(result)
        cp.cuda.Device().synchronize()
        time_sync = time.time() - start
        
        # Async processing with streams
        manager = create_stream_manager(num_streams=3, enable_pinned=True)
        
        start = time.time()
        for i, chunk in enumerate(chunks):
            stream_idx = i % 3
            gpu_chunk = manager.async_upload(chunk, stream_idx=stream_idx)
            
            with manager.get_stream(stream_idx):
                result = gpu_chunk * 2.0
            
            cpu_result = manager.async_download(result, stream_idx=stream_idx)
        
        manager.synchronize_all()
        time_async = time.time() - start
        
        speedup = time_sync / time_async
        
        logger.info(f"\nProcessing {num_chunks} chunks of {chunk_size:,} points")
        logger.info(f"  Synchronous:  {time_sync:.3f}s")
        logger.info(f"  Async streams: {time_async:.3f}s")
        logger.info(f"  Speedup:       {speedup:.2f}x")
        
        manager.cleanup()
        
    except Exception as e:
        logger.error(f"CUDA streams benchmark failed: {e}")


def benchmark_gpu_caching():
    """Benchmark GPU array caching."""
    if not GPU_AVAILABLE:
        logger.warning("Skipping GPU caching benchmark (no GPU)")
        return
    
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK 3: GPU Array Caching")
    logger.info("="*60)
    
    try:
        from ign_lidar.optimization.gpu_memory import GPUArrayCache
        
        cache = GPUArrayCache(max_size_gb=4.0)
        
        # Create test arrays
        normals = np.random.randn(10_000_000, 3).astype(np.float32)
        size_mb = normals.nbytes / (1024**2)
        
        # Without caching - multiple uploads
        start = time.time()
        for _ in range(10):
            gpu_normals = cp.asarray(normals)
            result = gpu_normals[::100]  # Sample
            del gpu_normals
        cp.cuda.Device().synchronize()
        time_no_cache = time.time() - start
        
        # With caching - single upload
        start = time.time()
        for i in range(10):
            gpu_normals = cache.get_or_upload('normals', normals)
            result = gpu_normals[::100]  # Sample
        cp.cuda.Device().synchronize()
        time_cached = time.time() - start
        
        speedup = time_no_cache / time_cached
        
        logger.info(f"\nArray size: {len(normals):,} points ({size_mb:.1f}MB)")
        logger.info(f"  Without caching: {time_no_cache:.3f}s (10 uploads)")
        logger.info(f"  With caching:    {time_cached:.3f}s (1 upload)")
        logger.info(f"  Speedup:         {speedup:.2f}x")
        
        stats = cache.get_stats()
        logger.info(f"\nCache stats:")
        logger.info(f"  Cached arrays: {stats['num_cached']}")
        logger.info(f"  Total size: {stats['total_size_gb']:.2f}GB")
        logger.info(f"  Utilization: {stats['utilization_pct']:.1f}%")
        
        cache.clear()
        
    except Exception as e:
        logger.error(f"GPU caching benchmark failed: {e}")


def benchmark_chunked_processing():
    """Benchmark full chunked processing with all optimizations."""
    if not GPU_AVAILABLE:
        logger.warning("Skipping chunked processing benchmark (no GPU)")
        return
    
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK 4: Chunked Feature Processing")
    logger.info("="*60)
    
    try:
        from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
        
        # Generate synthetic point cloud
        num_points = 5_000_000
        points = np.random.randn(num_points, 3).astype(np.float32)
        
        logger.info(f"\nTest dataset: {num_points:,} points")
        
        # Without CUDA streams
        logger.info("\nTesting WITHOUT CUDA streams...")
        computer_no_streams = GPUChunkedFeatureComputer(
            chunk_size=2_000_000,
            use_gpu=True,
            show_progress=False,
            use_cuda_streams=False
        )
        
        start = time.time()
        normals_no_streams = computer_no_streams.compute_normals_chunked(points, k=10)
        time_no_streams = time.time() - start
        
        # With CUDA streams
        logger.info("\nTesting WITH CUDA streams...")
        computer_streams = GPUChunkedFeatureComputer(
            chunk_size=2_000_000,
            use_gpu=True,
            show_progress=False,
            use_cuda_streams=True
        )
        
        start = time.time()
        normals_streams = computer_streams.compute_normals_chunked(points, k=10)
        time_streams = time.time() - start
        
        speedup = time_no_streams / time_streams
        
        logger.info(f"\nResults:")
        logger.info(f"  Without streams: {time_no_streams:.2f}s")
        logger.info(f"  With streams:    {time_streams:.2f}s")
        logger.info(f"  Speedup:         {speedup:.2f}x")
        
        # Verify results are similar
        diff = np.abs(normals_no_streams - normals_streams).max()
        logger.info(f"  Max difference:  {diff:.6f} (should be ~0)")
        
    except Exception as e:
        logger.error(f"Chunked processing benchmark failed: {e}")


def main():
    """Run all benchmarks."""
    logger.info("\n" + "="*60)
    logger.info("GPU OPTIMIZATION BENCHMARK SUITE")
    logger.info("="*60)
    
    if not GPU_AVAILABLE:
        logger.error("\n❌ GPU not available - cannot run benchmarks")
        logger.error("Install CuPy with: pip install cupy-cuda11x")
        return
    
    # Get GPU info
    device = cp.cuda.Device()
    device_props = device.attributes
    total_mem = device.mem_info[1] / (1024**3)
    
    logger.info(f"\nGPU Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    logger.info(f"Total VRAM: {total_mem:.1f}GB")
    logger.info(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")
    
    # Run benchmarks
    benchmark_memory_transfers()
    benchmark_cuda_streams()
    benchmark_gpu_caching()
    benchmark_chunked_processing()
    
    logger.info("\n" + "="*60)
    logger.info("✓ All benchmarks completed")
    logger.info("="*60)


if __name__ == '__main__':
    main()
