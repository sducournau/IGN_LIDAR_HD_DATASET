"""
I/O Optimization Utilities

Provides optimized I/O strategies for LAZ files including:
- Parallel I/O for multiple files
- Buffer optimization
- Prefetching and caching
- SSD-optimized access patterns

Part of Phase 3 Sprint 4.

Author: Phase 3 Sprint 4 Optimization
Date: October 18, 2025
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger(__name__)


class ParallelLAZReader:
    """
    Parallel LAZ file reader for improved I/O throughput.
    
    Reads multiple LAZ files simultaneously using thread pool,
    which is beneficial for networked storage or SSDs with
    high parallel throughput.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize parallel reader.
        
        Args:
            max_workers: Maximum number of parallel read threads
        """
        self.max_workers = max_workers
        logger.info(f"📚 Parallel LAZ reader initialized ({max_workers} workers)")
    
    def read_multiple(
        self,
        laz_files: List[Path],
        reader_func: Callable[[Path], Any],
        show_progress: bool = True
    ) -> List[Any]:
        """
        Read multiple LAZ files in parallel.
        
        Args:
            laz_files: List of LAZ file paths
            reader_func: Function to read single LAZ file
            show_progress: Show progress during reading
            
        Returns:
            List of read results (in same order as input)
        """
        logger.info(f"📖 Reading {len(laz_files)} files in parallel...")
        
        start_time = time.time()
        results = [None] * len(laz_files)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(reader_func, laz_file): i
                for i, laz_file in enumerate(laz_files)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                    completed += 1
                    
                    if show_progress and completed % max(1, len(laz_files) // 10) == 0:
                        logger.info(f"   📊 Progress: {completed}/{len(laz_files)} files")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to read file {laz_files[idx].name}: {e}")
                    results[idx] = None
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r is not None)
        
        logger.info(f"✅ Read {successful}/{len(laz_files)} files in {elapsed:.2f}s "
                   f"({successful/elapsed:.1f} files/sec)")
        
        return results


class BufferedLAZWriter:
    """
    Buffered LAZ writer for improved write performance.
    
    Accumulates points in memory buffer and writes in larger chunks
    to reduce I/O overhead.
    """
    
    def __init__(
        self,
        output_path: Path,
        header: Any,
        buffer_size: int = 1_000_000
    ):
        """
        Initialize buffered writer.
        
        Args:
            output_path: Path to output LAZ file
            header: LAZ header template
            buffer_size: Number of points to buffer before writing
        """
        self.output_path = output_path
        self.header = header
        self.buffer_size = buffer_size
        
        self.buffer: List[np.ndarray] = []
        self.buffer_points = 0
        self.total_written = 0
        
        import laspy
        self.writer = laspy.open(str(output_path), mode='w', header=header)
        
        logger.debug(f"📝 Buffered writer initialized (buffer={buffer_size:,} points)")
    
    def write_points(self, points: np.ndarray):
        """
        Write points to file (buffered).
        
        Args:
            points: Point data to write
        """
        self.buffer.append(points)
        self.buffer_points += len(points)
        
        # Flush if buffer is full
        if self.buffer_points >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Flush buffer to disk."""
        if self.buffer_points == 0:
            return
        
        # Concatenate buffer
        combined = np.vstack(self.buffer) if len(self.buffer) > 1 else self.buffer[0]
        
        # Write to file
        self.writer.write_points(combined)
        self.total_written += len(combined)
        
        # Clear buffer
        self.buffer.clear()
        self.buffer_points = 0
        
        logger.debug(f"   💾 Flushed {len(combined):,} points (total: {self.total_written:,})")
    
    def close(self):
        """Close writer and flush remaining buffer."""
        self.flush()
        self.writer.close()
        
        logger.info(f"✅ Wrote {self.total_written:,} points to {self.output_path.name}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PrefetchCache:
    """
    Prefetch cache for predictive file loading.
    
    Anticipates which files will be needed next and loads them
    in background threads to hide I/O latency.
    """
    
    def __init__(
        self,
        reader_func: Callable[[Path], Any],
        max_cache_size: int = 5,
        max_workers: int = 2
    ):
        """
        Initialize prefetch cache.
        
        Args:
            reader_func: Function to read files
            max_cache_size: Maximum number of files to cache
            max_workers: Number of background loading threads
        """
        self.reader_func = reader_func
        self.max_cache_size = max_cache_size
        self.max_workers = max_workers
        
        self.cache: Dict[Path, Any] = {}
        self.loading: Dict[Path, Any] = {}  # Future objects
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"🔮 Prefetch cache initialized (size={max_cache_size}, workers={max_workers})")
    
    def prefetch(self, file_paths: List[Path]):
        """
        Start prefetching files in background.
        
        Args:
            file_paths: List of files to prefetch
        """
        for file_path in file_paths[:self.max_cache_size]:
            if file_path not in self.cache and file_path not in self.loading:
                future = self.executor.submit(self.reader_func, file_path)
                self.loading[file_path] = future
                logger.debug(f"🔮 Prefetching: {file_path.name}")
    
    def get(self, file_path: Path) -> Any:
        """
        Get file data (blocks if still loading).
        
        Args:
            file_path: Path to file
            
        Returns:
            File data
        """
        # Check cache first
        if file_path in self.cache:
            logger.debug(f"✅ Cache hit: {file_path.name}")
            return self.cache[file_path]
        
        # Check if loading
        if file_path in self.loading:
            logger.debug(f"⏳ Waiting for prefetch: {file_path.name}")
            future = self.loading[file_path]
            data = future.result()
            del self.loading[file_path]
            
            # Add to cache
            self._add_to_cache(file_path, data)
            return data
        
        # Not cached or loading - load now
        logger.debug(f"❌ Cache miss: {file_path.name}")
        data = self.reader_func(file_path)
        self._add_to_cache(file_path, data)
        return data
    
    def _add_to_cache(self, file_path: Path, data: Any):
        """Add data to cache (with LRU eviction)."""
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_cache_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            logger.debug(f"🗑️  Evicted from cache: {oldest.name}")
        
        self.cache[file_path] = data
    
    def clear(self):
        """Clear cache and cancel pending loads."""
        self.cache.clear()
        
        # Cancel pending futures
        for future in self.loading.values():
            future.cancel()
        self.loading.clear()
        
        logger.info("🗑️  Prefetch cache cleared")
    
    def shutdown(self):
        """Shutdown executor and clear cache."""
        self.clear()
        self.executor.shutdown(wait=False)


def estimate_optimal_buffer_size(
    file_size_mb: float,
    available_memory_mb: float,
    safety_factor: float = 0.5
) -> int:
    """
    Estimate optimal buffer size for I/O operations.
    
    Args:
        file_size_mb: Size of file in MB
        available_memory_mb: Available system memory in MB
        safety_factor: Safety factor (0-1) for memory usage
        
    Returns:
        Optimal buffer size in number of points
    """
    # Target: use up to safety_factor of available memory
    target_memory_mb = available_memory_mb * safety_factor
    
    # Estimate points per MB (assuming 20 bytes per point with attributes)
    bytes_per_point = 20
    points_per_mb = (1024 * 1024) / bytes_per_point
    
    # Calculate buffer size
    if file_size_mb < target_memory_mb:
        # Small file - load entirely
        buffer_size = int(file_size_mb * points_per_mb)
    else:
        # Large file - use target memory
        buffer_size = int(target_memory_mb * points_per_mb)
    
    # Clamp to reasonable range
    min_buffer = 100_000
    max_buffer = 10_000_000
    buffer_size = max(min_buffer, min(buffer_size, max_buffer))
    
    logger.debug(f"📊 Optimal buffer size: {buffer_size:,} points "
                f"(file={file_size_mb:.1f}MB, mem={available_memory_mb:.1f}MB)")
    
    return buffer_size


def benchmark_io_performance(
    laz_file: Path,
    num_runs: int = 5
) -> Dict[str, float]:
    """
    Benchmark I/O performance for a LAZ file.
    
    Args:
        laz_file: Path to LAZ file
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with performance metrics
    """
    import laspy
    
    logger.info(f"⏱️  Benchmarking I/O performance: {laz_file.name}")
    
    # Benchmark standard read
    read_times = []
    for i in range(num_runs):
        start = time.perf_counter()
        las = laspy.read(str(laz_file))
        read_times.append(time.perf_counter() - start)
        del las
    
    # Benchmark chunked read
    chunk_times = []
    chunk_size = 1_000_000
    for i in range(num_runs):
        start = time.perf_counter()
        with laspy.open(str(laz_file)) as laz_reader:
            for chunk in laz_reader.chunk_iterator(chunk_size):
                pass  # Just read, don't process
        chunk_times.append(time.perf_counter() - start)
    
    # Get file size
    file_size_mb = laz_file.stat().st_size / (1024 * 1024)
    
    # Compute statistics
    avg_read_time = np.mean(read_times)
    avg_chunk_time = np.mean(chunk_times)
    read_throughput = file_size_mb / avg_read_time
    chunk_throughput = file_size_mb / avg_chunk_time
    
    results = {
        'file_size_mb': file_size_mb,
        'read_time_s': avg_read_time,
        'chunk_time_s': avg_chunk_time,
        'read_throughput_mbps': read_throughput,
        'chunk_throughput_mbps': chunk_throughput,
        'speedup': avg_read_time / avg_chunk_time
    }
    
    logger.info(f"   📊 Standard read: {avg_read_time:.3f}s ({read_throughput:.1f} MB/s)")
    logger.info(f"   📊 Chunked read: {avg_chunk_time:.3f}s ({chunk_throughput:.1f} MB/s)")
    logger.info(f"   📊 Speedup: {results['speedup']:.2f}x")
    
    return results
