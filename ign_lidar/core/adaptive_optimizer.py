"""
Adaptive Chunk Size Optimizer for GPU Processing

Automatically calculates optimal chunk sizes for GPU-accelerated feature computation
based on available VRAM, point cloud characteristics, and feature complexity.

Features:
- VRAM-aware chunk size calculation
- Feature complexity estimation
- Safety margins for stable processing
- Hardware capability detection
- Performance profiling and recommendations

Author: Performance Optimization Team
Date: October 16, 2025
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# GPU availability check (centralized via GPUManager)
from .gpu import GPUManager
_gpu_manager = GPUManager()
GPU_AVAILABLE = _gpu_manager.gpu_available

if GPU_AVAILABLE:
    import cupy as cp
else:
    cp = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

__all__ = ['AdaptiveChunkSizeOptimizer', 'ChunkSizeConfig', 'estimate_memory_requirements']


@dataclass
class ChunkSizeConfig:
    """Configuration for chunk size optimization."""
    
    safety_margin: float = 0.75
    """Use only this fraction of available VRAM (default: 75%)"""
    
    min_chunk_size: int = 100_000
    """Minimum chunk size (points)"""
    
    max_chunk_size: int = 20_000_000
    """Maximum chunk size (points)"""
    
    prefer_power_of_2: bool = False
    """Round chunk sizes to powers of 2 for GPU efficiency"""
    
    overhead_factor: float = 2.5
    """Memory overhead multiplier for intermediate calculations"""
    
    enable_profiling: bool = False
    """Enable profiling to refine estimates"""


def estimate_memory_requirements(
    num_points: int,
    num_features: int,
    k_neighbors: int,
    dtype_size: int = 4
) -> Dict[str, int]:
    """
    Estimate memory requirements for feature computation.
    
    Args:
        num_points: Number of points to process
        num_features: Number of features to compute
        k_neighbors: Number of neighbors for KNN
        dtype_size: Size of data type in bytes (4 for float32)
    
    Returns:
        Dictionary with memory estimates in bytes:
            - points: Memory for point coordinates
            - features: Memory for computed features
            - knn_indices: Memory for KNN indices
            - intermediate: Memory for intermediate calculations
            - total: Total estimated memory
    """
    # Point coordinates (x, y, z)
    points_memory = num_points * 3 * dtype_size
    
    # Features output
    features_memory = num_points * num_features * dtype_size
    
    # KNN indices (int32)
    knn_indices_memory = num_points * k_neighbors * 4
    
    # Intermediate calculations:
    # - Neighbor point gathering: num_points * k_neighbors * 3 * dtype_size
    # - Covariance matrices: num_points * 3 * 3 * dtype_size
    # - Eigenvalues/vectors: num_points * 3 * dtype_size * 2
    intermediate_memory = (
        num_points * k_neighbors * 3 * dtype_size +  # Neighbors
        num_points * 9 * dtype_size +                 # Covariance
        num_points * 6 * dtype_size                   # Eigen
    )
    
    total_memory = points_memory + features_memory + knn_indices_memory + intermediate_memory
    
    return {
        'points': points_memory,
        'features': features_memory,
        'knn_indices': knn_indices_memory,
        'intermediate': intermediate_memory,
        'total': total_memory
    }


class AdaptiveChunkSizeOptimizer:
    """
    Automatically calculate optimal chunk sizes for GPU processing.
    
    The optimizer considers:
    1. Available GPU VRAM
    2. Point cloud size
    3. Feature computation complexity
    4. k_neighbors parameter
    5. Safety margins for stability
    
    Example:
        >>> optimizer = AdaptiveChunkSizeOptimizer()
        >>> 
        >>> # Calculate for a tile
        >>> chunk_size = optimizer.calculate_optimal_chunk_size(
        ...     num_points=18_651_688,
        ...     num_features=15,
        ...     k_neighbors=12
        ... )
        >>> print(f"Optimal chunk size: {chunk_size:,} points")
        >>> 
        >>> # Get recommendations
        >>> recommendations = optimizer.get_recommendations()
        >>> print(recommendations)
    """
    
    def __init__(self, config: Optional[ChunkSizeConfig] = None, gpu_device_id: int = 0):
        """
        Initialize optimizer.
        
        Args:
            config: Configuration for chunk size optimization
            gpu_device_id: GPU device ID to use (default: 0)
        """
        self.config = config or ChunkSizeConfig()
        self.gpu_device_id = gpu_device_id
        self.gpu_info = self._get_gpu_info()
        
        if self.gpu_info:
            logger.info(
                f"Adaptive optimizer initialized: "
                f"GPU={self.gpu_info['name']}, "
                f"VRAM={self.gpu_info['total_memory_gb']:.1f}GB"
            )
        else:
            logger.warning("GPU not available - optimizer will use conservative defaults")
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information using pynvml."""
        if not NVML_AVAILABLE:
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_device_id)
            
            # Get device name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get compute capability
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            
            return {
                'name': name,
                'total_memory_gb': mem_info.total / (1024**3),
                'free_memory_gb': mem_info.free / (1024**3),
                'used_memory_gb': mem_info.used / (1024**3),
                'compute_capability': f"{major}.{minor}"
            }
        except Exception as e:
            logger.debug(f"Failed to get GPU info: {e}")
            return None
    
    def calculate_optimal_chunk_size(
        self,
        num_points: int,
        num_features: int = 15,
        k_neighbors: int = 12,
        vram_available_gb: Optional[float] = None
    ) -> int:
        """
        Calculate optimal chunk size.
        
        Args:
            num_points: Total number of points in the point cloud
            num_features: Number of features to compute
            k_neighbors: Number of neighbors for KNN
            vram_available_gb: Available VRAM in GB (auto-detected if None)
        
        Returns:
            Optimal chunk size in number of points
        """
        # Get available VRAM
        if vram_available_gb is None:
            if self.gpu_info:
                vram_available_gb = self.gpu_info['free_memory_gb']
            else:
                # Conservative default for 8GB GPU
                vram_available_gb = 6.0
                logger.warning(
                    f"GPU info not available, using conservative default: "
                    f"{vram_available_gb:.1f}GB"
                )
        
        # Apply safety margin
        usable_vram_gb = vram_available_gb * self.config.safety_margin
        usable_vram_bytes = usable_vram_gb * (1024**3)
        
        # Estimate memory per point
        memory_estimates = estimate_memory_requirements(
            num_points=1_000_000,  # Estimate for 1M points
            num_features=num_features,
            k_neighbors=k_neighbors
        )
        memory_per_point = memory_estimates['total'] / 1_000_000
        
        # Apply overhead factor for intermediate calculations
        memory_per_point *= self.config.overhead_factor
        
        # Calculate chunk size
        chunk_size = int(usable_vram_bytes / memory_per_point)
        
        # Apply constraints
        chunk_size = max(self.config.min_chunk_size, chunk_size)
        chunk_size = min(self.config.max_chunk_size, chunk_size)
        chunk_size = min(chunk_size, num_points)  # Don't exceed total points
        
        # Round to power of 2 if requested
        if self.config.prefer_power_of_2:
            chunk_size = self._round_to_power_of_2(chunk_size)
        else:
            # Round to nearest 100k for readability
            chunk_size = round(chunk_size / 100_000) * 100_000
        
        # Log recommendation
        num_chunks = (num_points + chunk_size - 1) // chunk_size
        memory_estimate_mb = (chunk_size * memory_per_point) / (1024**2)
        
        logger.info(
            f"Adaptive chunk size calculation:\n"
            f"  Total points: {num_points:,}\n"
            f"  Available VRAM: {vram_available_gb:.2f}GB "
            f"(usable: {usable_vram_gb:.2f}GB)\n"
            f"  Estimated memory per point: {memory_per_point:.1f} bytes\n"
            f"  Optimal chunk size: {chunk_size:,} points\n"
            f"  Number of chunks: {num_chunks}\n"
            f"  Estimated VRAM per chunk: {memory_estimate_mb:.1f}MB"
        )
        
        return chunk_size
    
    def _round_to_power_of_2(self, n: int) -> int:
        """Round to nearest lower power of 2."""
        return 2 ** int(np.log2(n))
    
    def calculate_optimal_batch_size_for_gpu(
        self,
        num_points: int,
        k_neighbors: int = 12,
        vram_available_gb: Optional[float] = None
    ) -> int:
        """
        Calculate optimal batch size for basic GPU mode (non-chunked).
        
        This is for the batch size within GPUFeatureComputer, which processes
        smaller batches for KNN queries.
        
        Args:
            num_points: Number of points to process
            k_neighbors: Number of neighbors for KNN
            vram_available_gb: Available VRAM in GB
        
        Returns:
            Optimal batch size for GPU processing
        """
        # For basic GPU mode, we want smaller batches for KNN queries
        # but large enough to keep GPU busy
        
        if vram_available_gb is None:
            if self.gpu_info:
                vram_available_gb = self.gpu_info['free_memory_gb']
            else:
                vram_available_gb = 6.0
        
        # Rule of thumb: batch size ~= VRAM_GB * 500K
        # This balances GPU occupancy with memory constraints
        base_batch_size = int(vram_available_gb * 500_000)
        
        # Apply safety margin
        batch_size = int(base_batch_size * self.config.safety_margin)
        
        # Apply constraints
        batch_size = max(100_000, batch_size)  # Min 100K
        batch_size = min(5_000_000, batch_size)  # Max 5M
        batch_size = min(batch_size, num_points)
        
        # Round to 100K
        batch_size = round(batch_size / 100_000) * 100_000
        
        logger.info(
            f"Recommended GPU batch size: {batch_size:,} points "
            f"(VRAM: {vram_available_gb:.1f}GB)"
        )
        
        return batch_size
    
    def get_recommendations(
        self,
        num_points: int,
        num_features: int = 15,
        k_neighbors: int = 12
    ) -> Dict[str, Any]:
        """
        Get comprehensive recommendations for processing configuration.
        
        Args:
            num_points: Total number of points
            num_features: Number of features to compute
            k_neighbors: Number of neighbors for KNN
        
        Returns:
            Dictionary with recommendations:
                - chunk_size: Optimal chunk size for chunked mode
                - batch_size: Optimal batch size for basic GPU mode
                - num_chunks: Number of chunks needed
                - use_chunked_mode: Whether to use chunked mode
                - estimated_memory_gb: Estimated VRAM usage
                - mode: Recommended processing mode
        """
        # Calculate both sizes
        chunk_size = self.calculate_optimal_chunk_size(
            num_points, num_features, k_neighbors
        )
        batch_size = self.calculate_optimal_batch_size_for_gpu(
            num_points, k_neighbors
        )
        
        # Determine if chunked mode is beneficial
        # Use chunked if we need more than 2 chunks
        num_chunks = (num_points + chunk_size - 1) // chunk_size
        use_chunked = num_chunks >= 2
        
        # Estimate memory usage
        memory_estimates = estimate_memory_requirements(
            num_points=chunk_size if use_chunked else num_points,
            num_features=num_features,
            k_neighbors=k_neighbors
        )
        estimated_memory_gb = (
            memory_estimates['total'] * self.config.overhead_factor / (1024**3)
        )
        
        # Determine mode
        if not GPU_AVAILABLE:
            mode = "cpu"
        elif use_chunked:
            mode = "gpu_chunked"
        else:
            mode = "gpu"
        
        recommendations = {
            'chunk_size': chunk_size,
            'batch_size': batch_size,
            'num_chunks': num_chunks,
            'use_chunked_mode': use_chunked,
            'estimated_memory_gb': estimated_memory_gb,
            'mode': mode,
            'k_neighbors': k_neighbors,
            'vram_info': self.gpu_info
        }
        
        return recommendations
    
    def print_recommendations(
        self,
        num_points: int,
        num_features: int = 15,
        k_neighbors: int = 12
    ):
        """
        Print formatted recommendations.
        
        Args:
            num_points: Total number of points
            num_features: Number of features to compute
            k_neighbors: Number of neighbors for KNN
        """
        recs = self.get_recommendations(num_points, num_features, k_neighbors)
        
        print("=" * 80)
        print("Adaptive Chunk Size Optimizer - Recommendations")
        print("=" * 80)
        print(f"Point cloud: {num_points:,} points")
        print(f"Features: {num_features} | k_neighbors: {k_neighbors}")
        print()
        
        if recs['vram_info']:
            info = recs['vram_info']
            print(f"GPU: {info['name']}")
            print(f"VRAM: {info['total_memory_gb']:.1f}GB total, "
                  f"{info['free_memory_gb']:.1f}GB free")
            print()
        
        print(f"Recommended mode: {recs['mode'].upper()}")
        print()
        
        if recs['use_chunked_mode']:
            print(f"✓ Use GPU Chunked Mode")
            print(f"  - Chunk size: {recs['chunk_size']:,} points")
            print(f"  - Number of chunks: {recs['num_chunks']}")
            print(f"  - Estimated VRAM per chunk: {recs['estimated_memory_gb']:.2f}GB")
        else:
            print(f"✓ Use Basic GPU Mode")
            print(f"  - Batch size: {recs['batch_size']:,} points")
            print(f"  - Estimated VRAM: {recs['estimated_memory_gb']:.2f}GB")
        
        print()
        print("Configuration:")
        print(f"```yaml")
        print(f"features:")
        print(f"  gpu_batch_size: {recs['chunk_size']}")
        print(f"  use_gpu_chunked: {str(recs['use_chunked_mode']).lower()}")
        print(f"  k_neighbors: {k_neighbors}")
        print(f"processor:")
        print(f"  use_gpu: true")
        print(f"```")
        print("=" * 80)


def get_optimal_config_for_hardware(
    num_points: int = 18_000_000,
    num_features: int = 15,
    k_neighbors: int = 12,
    gpu_device_id: int = 0
) -> Dict[str, Any]:
    """
    Get optimal configuration for current hardware.
    
    Convenience function that creates optimizer and returns recommendations.
    
    Args:
        num_points: Expected point cloud size
        num_features: Number of features to compute
        k_neighbors: Number of neighbors for KNN
        gpu_device_id: GPU device ID
    
    Returns:
        Configuration dictionary ready for use in config files
    """
    optimizer = AdaptiveChunkSizeOptimizer(gpu_device_id=gpu_device_id)
    recs = optimizer.get_recommendations(num_points, num_features, k_neighbors)
    
    return {
        'features': {
            'gpu_batch_size': recs['chunk_size'],
            'use_gpu_chunked': recs['use_chunked_mode'],
            'k_neighbors': k_neighbors,
        },
        'processor': {
            'use_gpu': recs['mode'] != 'cpu',
        },
        'metadata': {
            'estimated_memory_gb': recs['estimated_memory_gb'],
            'num_chunks': recs['num_chunks'],
            'optimization_method': 'adaptive',
        }
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Parse command line args
    num_points = int(sys.argv[1]) if len(sys.argv) > 1 else 18_651_688
    
    # Create optimizer
    optimizer = AdaptiveChunkSizeOptimizer()
    
    # Print recommendations
    optimizer.print_recommendations(
        num_points=num_points,
        num_features=15,
        k_neighbors=12
    )
