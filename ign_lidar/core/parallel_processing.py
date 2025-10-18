"""
Parallel tile processing for Phase 3 Sprint 3.

This module provides multi-core parallel processing of LiDAR tiles
to achieve 2-4x speedup on multi-core systems.

Key features:
- Process multiple tiles simultaneously
- Dynamic load balancing
- Memory-aware scheduling
- Progress tracking across workers
- Error handling and recovery
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import multiprocessing as mp
from functools import partial
import time

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)


def process_tile_worker(
    laz_file: Path,
    processor_factory: Callable,
    output_dir: Path,
    tile_idx: int,
    total_tiles: int,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    Worker function to process a single tile.
    
    This runs in a separate process, so it receives a processor_factory
    function that creates a fresh processor instance.
    
    Parameters
    ----------
    laz_file : Path
        Path to LAZ tile
    processor_factory : Callable
        Function that creates a processor instance
    output_dir : Path
        Output directory for patches
    tile_idx : int
        Tile index (for progress)
    total_tiles : int
        Total tiles (for progress)
    skip_existing : bool
        Skip if outputs exist
        
    Returns
    -------
    result : dict
        Processing result with num_patches, errors, etc.
    """
    try:
        # Create processor in worker process
        processor = processor_factory()
        
        # Process the tile
        num_patches = processor.process_tile(
            laz_file=laz_file,
            output_dir=output_dir,
            tile_idx=tile_idx,
            total_tiles=total_tiles,
            skip_existing=skip_existing
        )
        
        return {
            'tile': laz_file.name,
            'num_patches': num_patches,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing {laz_file.name}: {e}")
        return {
            'tile': laz_file.name,
            'num_patches': 0,
            'success': False,
            'error': str(e)
        }


def process_tiles_parallel(
    laz_files: List[Path],
    processor_factory: Callable,
    output_dir: Path,
    n_jobs: int = -1,
    skip_existing: bool = True,
    batch_size: Optional[int] = None,
    prefer: str = 'processes'
) -> Dict[str, Any]:
    """
    Process multiple tiles in parallel using joblib.
    
    Parameters
    ----------
    laz_files : List[Path]
        List of LAZ files to process
    processor_factory : Callable
        Function that creates a processor instance
        Example: lambda: LiDARProcessor(config)
    output_dir : Path
        Output directory for patches
    n_jobs : int
        Number of parallel jobs:
        - -1: Use all CPU cores
        - 1: Sequential (no parallelism)
        - n: Use n cores
    skip_existing : bool
        Skip tiles with existing outputs
    batch_size : int, optional
        Process tiles in batches of this size
        Helps with memory management
    prefer : str
        Backend preference: 'processes' or 'threads'
        - processes: True parallelism (recommended)
        - threads: Lower memory but GIL limited
        
    Returns
    -------
    results : dict
        Summary of processing results
        
    Examples
    --------
    >>> from ign_lidar.core.processor import LiDARProcessor
    >>> from ign_lidar.core.parallel_processing import process_tiles_parallel
    >>> 
    >>> # Define processor factory
    >>> def make_processor():
    >>>     return LiDARProcessor(config)
    >>> 
    >>> # Process in parallel
    >>> results = process_tiles_parallel(
    >>>     laz_files=tile_paths,
    >>>     processor_factory=make_processor,
    >>>     output_dir=output_path,
    >>>     n_jobs=-1  # Use all cores
    >>> )
    >>> 
    >>> print(f"Processed {results['tiles_processed']} tiles")
    >>> print(f"Total patches: {results['total_patches']}")
    """
    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    total_tiles = len(laz_files)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 PARALLEL TILE PROCESSING (Sprint 3 Optimization)")
    logger.info(f"{'='*70}")
    logger.info(f"   Tiles to process: {total_tiles}")
    logger.info(f"   Parallel jobs: {n_jobs}")
    logger.info(f"   CPU cores: {mp.cpu_count()}")
    logger.info(f"   Backend: {prefer}")
    if batch_size:
        logger.info(f"   Batch size: {batch_size}")
    logger.info(f"{'='*70}\n")
    
    # If n_jobs=1, fall back to sequential processing
    if n_jobs == 1:
        logger.info("⚠️  Sequential processing (n_jobs=1)")
        results_list = []
        for idx, laz_file in enumerate(tqdm(laz_files, desc="Processing tiles"), 1):
            result = process_tile_worker(
                laz_file, processor_factory, output_dir,
                idx, total_tiles, skip_existing
            )
            results_list.append(result)
    else:
        # Parallel processing with joblib
        logger.info(f"⚡ Parallel processing with {n_jobs} workers")
        
        # Create worker function with fixed args
        worker_fn = partial(
            process_tile_worker,
            processor_factory=processor_factory,
            output_dir=output_dir,
            total_tiles=total_tiles,
            skip_existing=skip_existing
        )
        
        # Process in parallel
        if batch_size:
            # Batch processing for memory efficiency
            results_list = []
            for batch_start in range(0, total_tiles, batch_size):
                batch_end = min(batch_start + batch_size, total_tiles)
                batch_files = laz_files[batch_start:batch_end]
                batch_indices = list(range(batch_start + 1, batch_end + 1))
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}: "
                           f"tiles {batch_start+1}-{batch_end}")
                
                batch_results = Parallel(
                    n_jobs=n_jobs,
                    prefer=prefer,
                    verbose=10
                )(
                    delayed(worker_fn)(laz_file, idx)
                    for laz_file, idx in zip(batch_files, batch_indices)
                )
                
                results_list.extend(batch_results)
        else:
            # Process all at once
            results_list = Parallel(
                n_jobs=n_jobs,
                prefer=prefer,
                verbose=10
            )(
                delayed(worker_fn)(laz_file, idx)
                for idx, laz_file in enumerate(laz_files, 1)
            )
    
    # Aggregate results
    total_patches = sum(r['num_patches'] for r in results_list)
    tiles_processed = sum(1 for r in results_list if r['num_patches'] > 0)
    tiles_skipped = sum(1 for r in results_list if r['num_patches'] == 0 and r['success'])
    tiles_failed = sum(1 for r in results_list if not r['success'])
    
    # Log summary
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ PARALLEL PROCESSING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"   Tiles processed: {tiles_processed}")
    logger.info(f"   Tiles skipped: {tiles_skipped}")
    logger.info(f"   Tiles failed: {tiles_failed}")
    logger.info(f"   Total patches: {total_patches}")
    logger.info(f"{'='*70}\n")
    
    # Collect errors
    errors = [r for r in results_list if not r['success']]
    if errors:
        logger.warning(f"⚠️  {len(errors)} tiles failed:")
        for err in errors[:5]:  # Show first 5 errors
            logger.warning(f"   - {err['tile']}: {err['error']}")
        if len(errors) > 5:
            logger.warning(f"   ... and {len(errors)-5} more")
    
    return {
        'total_tiles': total_tiles,
        'tiles_processed': tiles_processed,
        'tiles_skipped': tiles_skipped,
        'tiles_failed': tiles_failed,
        'total_patches': total_patches,
        'results': results_list,
        'errors': errors,
        'n_jobs': n_jobs
    }


def estimate_optimal_jobs(
    total_tiles: int,
    avg_tile_size_mb: float = 500,
    available_memory_gb: Optional[float] = None,
    memory_safety_factor: float = 0.7
) -> int:
    """
    Estimate optimal number of parallel jobs based on system resources.
    
    Parameters
    ----------
    total_tiles : int
        Number of tiles to process
    avg_tile_size_mb : float
        Average tile size in MB
    available_memory_gb : float, optional
        Available system memory in GB
        If None, will auto-detect
    memory_safety_factor : float
        Use this fraction of available memory (0.7 = 70%)
        
    Returns
    -------
    n_jobs : int
        Recommended number of parallel jobs
        
    Notes
    -----
    Formula considers:
    - CPU cores available
    - Memory capacity
    - Tile size
    - Safety margin for system stability
    """
    # Get CPU cores
    n_cpus = mp.cpu_count()
    
    # Get available memory
    if available_memory_gb is None:
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_memory_gb = mem.available / (1024**3)
        except ImportError:
            # Conservative fallback
            available_memory_gb = 8.0
            logger.warning("psutil not available, assuming 8GB RAM")
    
    # Estimate memory per job (tile + processing overhead)
    memory_per_job_gb = (avg_tile_size_mb * 3) / 1024  # 3x for processing overhead
    
    # Calculate memory-constrained max jobs
    usable_memory_gb = available_memory_gb * memory_safety_factor
    max_jobs_memory = int(usable_memory_gb / memory_per_job_gb)
    
    # Take minimum of CPU and memory constraints
    optimal_jobs = min(n_cpus, max_jobs_memory, total_tiles)
    
    # Ensure at least 1 job
    optimal_jobs = max(1, optimal_jobs)
    
    logger.info(f"\n📊 Parallel Processing Estimation:")
    logger.info(f"   CPU cores: {n_cpus}")
    logger.info(f"   Available memory: {available_memory_gb:.1f} GB")
    logger.info(f"   Usable memory: {usable_memory_gb:.1f} GB ({memory_safety_factor*100:.0f}%)")
    logger.info(f"   Memory per job: {memory_per_job_gb:.1f} GB")
    logger.info(f"   Memory-limited jobs: {max_jobs_memory}")
    logger.info(f"   Recommended jobs: {optimal_jobs}\n")
    
    return optimal_jobs


def benchmark_parallel_speedup(
    laz_files: List[Path],
    processor_factory: Callable,
    output_dir: Path,
    n_jobs_list: Optional[List[int]] = None,
    skip_existing: bool = True
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark parallel processing with different job counts.
    
    Useful for finding optimal parallelism level for your hardware.
    
    Parameters
    ----------
    laz_files : List[Path]
        Test tiles (recommend 5-10 tiles)
    processor_factory : Callable
        Processor factory function
    output_dir : Path
        Output directory
    n_jobs_list : List[int], optional
        Job counts to test (default: [1, 2, 4, 8, -1])
    skip_existing : bool
        Skip existing outputs
        
    Returns
    -------
    benchmarks : dict
        Mapping of n_jobs to timing results
        
    Examples
    --------
    >>> # Test with small subset
    >>> test_tiles = laz_files[:10]
    >>> benchmarks = benchmark_parallel_speedup(
    >>>     test_tiles, make_processor, output_dir
    >>> )
    >>> 
    >>> # Find best configuration
    >>> best_jobs = min(benchmarks, key=lambda k: benchmarks[k]['time_per_tile'])
    >>> print(f"Best: {best_jobs} jobs")
    """
    if n_jobs_list is None:
        n_jobs_list = [1, 2, 4, 8, -1]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🔬 PARALLEL PROCESSING BENCHMARK")
    logger.info(f"{'='*70}")
    logger.info(f"   Test tiles: {len(laz_files)}")
    logger.info(f"   Configurations: {n_jobs_list}")
    logger.info(f"{'='*70}\n")
    
    benchmarks = {}
    
    for n_jobs in n_jobs_list:
        logger.info(f"\n📊 Testing n_jobs={n_jobs}...")
        
        start_time = time.perf_counter()
        
        results = process_tiles_parallel(
            laz_files=laz_files,
            processor_factory=processor_factory,
            output_dir=output_dir,
            n_jobs=n_jobs,
            skip_existing=skip_existing
        )
        
        elapsed = time.perf_counter() - start_time
        
        benchmarks[n_jobs] = {
            'total_time': elapsed,
            'time_per_tile': elapsed / len(laz_files),
            'tiles_processed': results['tiles_processed'],
            'total_patches': results['total_patches'],
            'actual_jobs': results['n_jobs']
        }
        
        logger.info(f"   Time: {elapsed:.1f}s ({elapsed/len(laz_files):.1f}s per tile)")
    
    # Calculate speedups
    baseline_time = benchmarks[1]['total_time']
    
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 BENCHMARK RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"{'Jobs':<10} {'Time':<12} {'Time/Tile':<15} {'Speedup':<10}")
    logger.info(f"{'-'*70}")
    
    for n_jobs in n_jobs_list:
        bench = benchmarks[n_jobs]
        speedup = baseline_time / bench['total_time']
        logger.info(
            f"{n_jobs:<10} "
            f"{bench['total_time']:>8.1f}s   "
            f"{bench['time_per_tile']:>8.1f}s       "
            f"{speedup:>6.2f}x"
        )
    
    logger.info(f"{'='*70}\n")
    
    return benchmarks


if __name__ == '__main__':
    print("🔧 Parallel processing module for Phase 3 Sprint 3")
    print("   Use process_tiles_parallel() to process tiles in parallel")
    print("   Use estimate_optimal_jobs() to find best parallelism level")
    print("   Use benchmark_parallel_speedup() to test configurations")
