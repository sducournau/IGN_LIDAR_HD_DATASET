"""
Profiling-based Auto-dispatcher for CPU/GPU Strategy Selection.

This module implements intelligent CPU vs GPU selection based on runtime profiling
rather than static rules. It measures:

1. CPU performance baseline on first run
2. GPU transfer overhead and computation time
3. Optimal switching point (dataset size threshold)

Phase 3.3 Optimization: Adaptive backend selection based on profiling

Author: IGN LiDAR HD Development Team
Date: November 25, 2025
Version: 1.0.0
"""

import logging
import time
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# Profile cache location
PROFILE_CACHE_DIR = Path.home() / '.ign_lidar' / 'profiles'
PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_CACHE_FILE = PROFILE_CACHE_DIR / 'dispatch_profile.pkl'


@dataclass
class DispatchProfile:
    """
    Performance profile for CPU vs GPU dispatch decision.

    Attributes:
        cpu_time_per_1m_points: CPU throughput (seconds for 1M points)
        gpu_overhead_ms: GPU transfer overhead (milliseconds)
        gpu_time_per_1m_points: GPU throughput (seconds for 1M points)
        optimal_switching_point: Dataset size where GPU becomes better (points)
        gpu_available: Whether GPU was available during profiling
        cpu_efficiency: CPU computation efficiency (points/second)
        gpu_efficiency: GPU computation efficiency (points/second)
        profile_timestamp: When profile was created
        hardware_name: GPU/CPU name this profile was created on
    """
    cpu_time_per_1m_points: float
    gpu_overhead_ms: float
    gpu_time_per_1m_points: float
    optimal_switching_point: int
    gpu_available: bool
    cpu_efficiency: float
    gpu_efficiency: float
    profile_timestamp: float
    hardware_name: str

    def estimate_time(self, n_points: int, use_gpu: bool) -> float:
        """
        Estimate computation time for given dataset size and backend.

        Args:
            n_points: Number of points to process
            use_gpu: Whether to use GPU

        Returns:
            Estimated time in seconds
        """
        if use_gpu and self.gpu_available:
            # GPU time = overhead + computation
            base_time = (n_points / 1_000_000) * self.gpu_time_per_1m_points
            return self.gpu_overhead_ms / 1000 + base_time
        else:
            # CPU time = computation only
            return (n_points / 1_000_000) * self.cpu_time_per_1m_points

    def recommend_backend(self, n_points: int, force_gpu: Optional[bool] = None) -> str:
        """
        Recommend optimal backend based on profiling data.

        Args:
            n_points: Number of points to process
            force_gpu: Force GPU (True) or CPU (False), or auto (None)

        Returns:
            Recommended backend: 'CPU' or 'GPU'
        """
        if force_gpu is True:
            return 'GPU' if self.gpu_available else 'CPU'
        elif force_gpu is False:
            return 'CPU'

        # Auto: Use profiling data to decide
        if not self.gpu_available:
            return 'CPU'

        if n_points >= self.optimal_switching_point:
            return 'GPU'
        else:
            return 'CPU'

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DispatchProfile("
            f"CPU={self.cpu_efficiency:.0f}pts/s, "
            f"GPU={self.gpu_efficiency:.0f}pts/s, "
            f"switch@{self.optimal_switching_point:,}pts, "
            f"timestamp={self.profile_timestamp})"
        )


class ProfileDispatcher:
    """
    Dispatcher that selects CPU or GPU based on runtime profiling.

    This class:
    1. Profiles CPU and GPU performance on first run
    2. Creates a lookup table for optimal backend selection
    3. Caches profile for future use
    4. Provides recommendations for any dataset size

    Usage:
        >>> dispatcher = ProfileDispatcher()
        >>> dispatcher.run_profiling()  # First run
        >>> backend = dispatcher.recommend_backend(n_points=5_000_000)
        >>> print(f"Use {backend} for 5M points")
        Use GPU for 5M points
    """

    def __init__(self, enable_caching: bool = True, verbose: bool = False):
        """
        Initialize profile dispatcher.

        Args:
            enable_caching: Whether to cache profiles for reuse
            verbose: Enable detailed logging
        """
        self.enable_caching = enable_caching
        self.verbose = verbose
        self._profile: Optional[DispatchProfile] = None
        self._profiled = False

        if verbose:
            logger.setLevel(logging.DEBUG)

        # Try to load cached profile
        if enable_caching:
            self._load_cached_profile()

    def _load_cached_profile(self):
        """Load cached profile if available and recent."""
        try:
            if PROFILE_CACHE_FILE.exists():
                with open(PROFILE_CACHE_FILE, 'rb') as f:
                    self._profile = pickle.load(f)
                    self._profiled = True

                # Check if profile is older than 30 days
                import time
                age_days = (time.time() - self._profile.profile_timestamp) / 86400
                if age_days > 30:
                    logger.warning(
                        f"Cached profile is {age_days:.1f} days old, "
                        "consider re-profiling with run_profiling()"
                    )
                else:
                    logger.debug(f"Loaded cached profile: {self._profile}")
        except Exception as e:
            logger.warning(f"Could not load cached profile: {e}")

    def _save_profile(self):
        """Save current profile to cache."""
        if not self._profile or not self.enable_caching:
            return

        try:
            with open(PROFILE_CACHE_FILE, 'wb') as f:
                pickle.dump(self._profile, f)
                logger.debug(f"Profile cached to {PROFILE_CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Could not cache profile: {e}")

    def run_profiling(
        self,
        test_sizes: Optional[list] = None,
        timeout: float = 60.0,
        force: bool = False
    ):
        """
        Run profiling to measure CPU and GPU performance.

        Args:
            test_sizes: Dataset sizes to test (default: [100k, 500k, 1M])
            timeout: Maximum time per test in seconds
            force: Re-profile even if cached profile exists

        Returns:
            DispatchProfile with measurement results
        """
        if self._profiled and not force:
            logger.info(f"Using cached profile: {self._profile}")
            return self._profile

        if test_sizes is None:
            test_sizes = [100_000, 500_000, 1_000_000]

        logger.info("Starting CPU/GPU profiling...")
        logger.info(f"Test sizes: {[f'{s:,}' for s in test_sizes]}")

        # Test CPU performance
        cpu_times = self._profile_cpu(test_sizes, timeout)
        cpu_time_per_1m = np.mean([t / (s / 1_000_000) for t, s in zip(cpu_times, test_sizes)])
        cpu_efficiency = 1_000_000 / cpu_time_per_1m if cpu_time_per_1m > 0 else 0

        logger.info(f"CPU: {cpu_efficiency:.0f} points/second")

        # Test GPU if available
        gpu_available = self._check_gpu_available()
        if gpu_available:
            gpu_overhead, gpu_times = self._profile_gpu(test_sizes, timeout)
            gpu_time_per_1m = np.mean([t / (s / 1_000_000) for t, s in zip(gpu_times, test_sizes)])
            gpu_efficiency = 1_000_000 / gpu_time_per_1m if gpu_time_per_1m > 0 else 0

            logger.info(f"GPU: {gpu_efficiency:.0f} points/second (overhead: {gpu_overhead:.1f}ms)")

            # Calculate switching point
            # GPU becomes better when: gpu_overhead + gpu_time < cpu_time
            # = gpu_overhead + (n_points / 1M) * gpu_time_per_1m < (n_points / 1M) * cpu_time_per_1m
            # n_points > gpu_overhead_sec / (cpu_time_per_1m - gpu_time_per_1m) * 1M
            overhead_sec = gpu_overhead / 1000
            time_diff = cpu_time_per_1m - gpu_time_per_1m
            if time_diff > 0:
                switching_point = int(overhead_sec / time_diff * 1_000_000)
                switching_point = max(100_000, min(switching_point, 10_000_000))  # Clamp
            else:
                # GPU faster even for small datasets
                switching_point = 100_000
        else:
            gpu_overhead = 0
            gpu_time_per_1m = 0
            gpu_efficiency = 0
            switching_point = float('inf')

        # Create profile
        self._profile = DispatchProfile(
            cpu_time_per_1m_points=cpu_time_per_1m,
            gpu_overhead_ms=gpu_overhead,
            gpu_time_per_1m_points=gpu_time_per_1m,
            optimal_switching_point=switching_point,
            gpu_available=gpu_available,
            cpu_efficiency=cpu_efficiency,
            gpu_efficiency=gpu_efficiency,
            profile_timestamp=time.time(),
            hardware_name=self._get_hardware_name()
        )

        self._profiled = True
        self._save_profile()

        logger.info(f"✅ Profiling complete: {self._profile}")
        return self._profile

    def _profile_cpu(self, test_sizes: list, timeout: float) -> list:
        """Profile CPU performance on test sizes."""
        from .adaptive_chunking import auto_chunk_size

        times = []
        for size in test_sizes:
            try:
                # Generate random points
                points = np.random.rand(size, 3).astype(np.float32)

                # Simple feature computation (covariance + PCA)
                start = time.time()

                # Simulate feature computation
                chunk_size = auto_chunk_size((size, 3), use_gpu=False)
                n_chunks = (size + chunk_size - 1) // chunk_size

                for _ in range(n_chunks):
                    _ = np.cov(points[:min(chunk_size, size)].T)

                elapsed = time.time() - start

                if elapsed > timeout:
                    logger.warning(f"CPU test timed out for {size:,} points")
                    break

                times.append(elapsed)

            except Exception as e:
                logger.warning(f"CPU profiling failed for {size:,} points: {e}")
                times.append(float('inf'))

        return times if times else [1.0]  # Default

    def _profile_gpu(self, test_sizes: list, timeout: float) -> Tuple[float, list]:
        """Profile GPU performance on test sizes."""
        try:
            import cupy as cp
            from ..core.gpu import GPUManager

            gpu_mgr = GPUManager()
            if not gpu_mgr.gpu_available:
                return 0, [float('inf')] * len(test_sizes)

            times = []
            overhead_times = []

            for size in test_sizes:
                try:
                    # Measure transfer overhead
                    points = np.random.rand(size, 3).astype(np.float32)

                    start = time.time()
                    points_gpu = cp.asarray(points)
                    overhead_ms = (time.time() - start) * 1000

                    # Simulate GPU computation
                    start = time.time()
                    cov_gpu = cp.cov(points_gpu.T)
                    _ = cp.linalg.eigh(cov_gpu)
                    cp.cuda.Stream.null.synchronize()
                    elapsed = time.time() - start

                    if elapsed > timeout:
                        logger.warning(f"GPU test timed out for {size:,} points")
                        break

                    times.append(elapsed)
                    overhead_times.append(overhead_ms)

                except Exception as e:
                    logger.warning(f"GPU profiling failed for {size:,} points: {e}")
                    return 0, [float('inf')] * len(test_sizes)

            avg_overhead = np.mean(overhead_times) if overhead_times else 0
            return avg_overhead, times if times else [float('inf')] * len(test_sizes)

        except ImportError:
            logger.warning("GPU not available for profiling")
            return 0, [float('inf')] * len(test_sizes)

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            from ..core.gpu import GPUManager
            gpu_mgr = GPUManager()
            return gpu_mgr.gpu_available
        except Exception:
            return False

    def _get_hardware_name(self) -> str:
        """Get hardware name for profile identification."""
        try:
            import cupy as cp
            gpu = cp.cuda.Device()
            return f"CUDA-{gpu.compute_capability}"
        except Exception:
            return "CPU-only"

    def recommend_backend(
        self,
        n_points: int,
        force_gpu: Optional[bool] = None
    ) -> str:
        """
        Recommend optimal backend for dataset size.

        Args:
            n_points: Number of points to process
            force_gpu: Force GPU (True), CPU (False), or auto (None)

        Returns:
            Recommended backend: 'CPU' or 'GPU'
        """
        if not self._profiled:
            logger.warning(
                "No profiling data available. Run run_profiling() first, "
                "or will use conservative defaults (CPU for <1M, GPU for >1M)"
            )
            if force_gpu is True:
                return 'GPU'
            elif force_gpu is False:
                return 'CPU'
            else:
                return 'GPU' if n_points >= 1_000_000 else 'CPU'

        return self._profile.recommend_backend(n_points, force_gpu)

    def estimate_time(self, n_points: int, use_gpu: bool) -> float:
        """
        Estimate computation time for given configuration.

        Args:
            n_points: Number of points
            use_gpu: Whether to use GPU

        Returns:
            Estimated time in seconds
        """
        if not self._profiled:
            # Rough estimates without profiling
            if use_gpu:
                return (n_points / 1_000_000) * 2.0  # 2 sec/M points estimate
            else:
                return (n_points / 1_000_000) * 10.0  # 10 sec/M points estimate

        return self._profile.estimate_time(n_points, use_gpu)

    def get_profile(self) -> Optional[DispatchProfile]:
        """Get current profile, if available."""
        return self._profile

    def clear_cache(self):
        """Clear cached profile."""
        try:
            if PROFILE_CACHE_FILE.exists():
                PROFILE_CACHE_FILE.unlink()
                logger.info(f"Profile cache cleared: {PROFILE_CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Could not clear cache: {e}")

    def __repr__(self) -> str:
        """String representation."""
        if self._profiled:
            return f"ProfileDispatcher({self._profile})"
        else:
            return "ProfileDispatcher(not profiled)"


# Singleton instance for global use
_dispatcher_instance: Optional[ProfileDispatcher] = None


def get_profile_dispatcher(force_refresh: bool = False) -> ProfileDispatcher:
    """
    Get global profile dispatcher instance (singleton).

    Args:
        force_refresh: Force re-profiling

    Returns:
        ProfileDispatcher instance
    """
    global _dispatcher_instance

    if _dispatcher_instance is None:
        _dispatcher_instance = ProfileDispatcher(enable_caching=True)
        if force_refresh or not _dispatcher_instance._profiled:
            try:
                _dispatcher_instance.run_profiling()
            except Exception as e:
                logger.warning(f"Auto-profiling failed: {e}. Using defaults.")

    return _dispatcher_instance


__all__ = [
    'DispatchProfile',
    'ProfileDispatcher',
    'get_profile_dispatcher',
]
