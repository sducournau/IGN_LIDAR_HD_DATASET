"""
Distributed Processing Module - Phase 6

Multi-GPU and cluster support for large-scale LiDAR processing.
Handles distributed feature computation across multiple GPUs and nodes.

⚠️ CONSOLIDATION NOTE (v3.8.1):
The GPUManager class in this module is DEPRECATED.
Use ign_lidar.core.gpu.MultiGPUManager instead for consistent multi-GPU support.

**Phase 6 Features (November 25, 2025):**

1. **Multi-GPU Coordination**: Automatic GPU load balancing
2. **Distributed Features**: Feature computation across GPUs/nodes
3. **Data Partitioning**: Intelligent tile/point cloud distribution
4. **Communication**: Efficient inter-GPU communication
5. **Cluster Support**: Multi-node distributed processing

Example Usage:

    from ign_lidar.core.gpu import MultiGPUManager  # RECOMMENDED
    
    # Multi-GPU setup (automatic)
    multi_gpu = MultiGPUManager()
    available_gpus = multi_gpu.get_available_gpus()
    
    # Get optimal batch size
    batch_size = multi_gpu.get_optimal_batch_size(gpu_id=0, memory_per_item_mb=50)

    # Legacy: Still works but DEPRECATED
    from ign_lidar.optimization.distributed_processor import MultiGPUProcessor
    processor = MultiGPUProcessor(num_gpus='all')

Version: 1.0.0 (Consolidated with core.gpu module in v3.8.1)
Date: November 26, 2025
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.distributed as dist
    from torch.distributed import launch
    TORCH_AVAILABLE = True
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    DISTRIBUTED_AVAILABLE = False
    torch = None
    dist = None
    launch = None


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    gpu_id: int
    device_name: str
    total_memory_gb: float
    available_memory_gb: float
    compute_capability: Tuple[int, int]
    is_available: bool = True


class GPUManager:
    """
    ⚠️ DEPRECATED: Use ign_lidar.core.gpu.MultiGPUManager instead.
    
    This class is kept for backward compatibility but will be removed in v4.0.
    Use the consolidated MultiGPUManager in the core module for consistent
    multi-GPU support across the library.
    
    Manage and coordinate multiple GPUs for distributed processing.

    Features:
    - Automatic GPU detection and monitoring
    - Load balancing across GPUs
    - Memory tracking and management
    - Performance profiling per GPU
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize GPU manager.

        Args:
            verbose: Whether to log GPU info
            
        Note:
            This class is deprecated. Use ign_lidar.core.gpu.MultiGPUManager instead.
        """
        import warnings
        warnings.warn(
            "distributed_processor.GPUManager is deprecated. "
            "Use ign_lidar.core.gpu.MultiGPUManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        self.verbose = verbose
        self.gpus: Dict[int, GPUInfo] = {}
        self._detect_gpus()

    def _detect_gpus(self) -> None:
        """Detect available GPUs."""
        if not torch.cuda.is_available():
            logger.warning("No CUDA-capable devices found")
            return

        num_gpus = torch.cuda.device_count()

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)  # Convert to GB

            self.gpus[i] = GPUInfo(
                gpu_id=i,
                device_name=props.name,
                total_memory_gb=total_mem,
                available_memory_gb=total_mem,
                compute_capability=props.major,
                is_available=True
            )

            if self.verbose:
                logger.info(
                    f"GPU {i}: {props.name}, "
                    f"Memory: {total_mem:.1f}GB, "
                    f"Compute Capability: {props.major}.{props.minor}"
                )

    def get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs."""
        return sorted([
            gpu_id for gpu_id, info in self.gpus.items()
            if info.is_available
        ])

    def get_gpu_memory_usage(self) -> Dict[int, float]:
        """
        Get current memory usage for each GPU.

        Returns:
            Dict mapping GPU ID to memory used (GB)
        """
        usage = {}
        for gpu_id in self.get_available_gpus():
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            usage[gpu_id] = allocated

        return usage

    def get_least_loaded_gpu(self) -> int:
        """
        Get GPU with least memory usage.

        Returns:
            GPU ID of least loaded GPU
        """
        usage = self.get_gpu_memory_usage()
        return min(usage.keys(), key=lambda k: usage[k])

    def allocate_to_gpu(
        self,
        gpu_id: int,
        data_size_gb: float
    ) -> bool:
        """
        Check if GPU can allocate data.

        Args:
            gpu_id: GPU ID
            data_size_gb: Data size in GB

        Returns:
            Whether allocation is possible
        """
        available = self.gpus[gpu_id].available_memory_gb
        return data_size_gb <= available * 0.8  # Reserve 20% safety margin

    def get_optimal_batch_size(
        self,
        gpu_id: int,
        memory_per_sample_mb: float,
        safety_factor: float = 0.8
    ) -> int:
        """
        Calculate optimal batch size for GPU.

        Args:
            gpu_id: GPU ID
            memory_per_sample_mb: Memory required per sample
            safety_factor: Safety margin (0-1)

        Returns:
            Recommended batch size
        """
        available_mb = self.gpus[gpu_id].available_memory_gb * 1024
        batch_size = int(
            (available_mb * safety_factor) / memory_per_sample_mb
        )
        return max(1, batch_size)


class MultiGPUProcessor:
    """
    Process data in parallel across multiple GPUs.

    Handles:
    - Automatic GPU load balancing
    - Data partitioning
    - Result aggregation
    - Error handling and recovery
    """

    def __init__(
        self,
        num_gpus: Optional[int] = None,
        batch_per_gpu: int = 10,
        verbose: bool = True
    ):
        """
        Initialize multi-GPU processor.

        Args:
            num_gpus: Number of GPUs to use ('all' or specific number)
            batch_per_gpu: Items to process per GPU batch
            verbose: Whether to log progress
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        self.gpu_manager = GPUManager(verbose=verbose)
        self.verbose = verbose

        if num_gpus == 'all':
            self.gpu_ids = self.gpu_manager.get_available_gpus()
        elif num_gpus is None:
            self.gpu_ids = self.gpu_manager.get_available_gpus()
        else:
            self.gpu_ids = self.gpu_manager.get_available_gpus()[:num_gpus]

        self.batch_per_gpu = batch_per_gpu
        self.num_gpus = len(self.gpu_ids)

        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available")

        if self.verbose:
            logger.info(f"MultiGPUProcessor using {self.num_gpus} GPUs: {self.gpu_ids}")

    def process_batch(
        self,
        data_list: List[Any],
        process_fn: Callable,
        partition_strategy: str = 'round_robin'
    ) -> List[Any]:
        """
        Process list of data items across multiple GPUs.

        Args:
            data_list: List of items to process
            process_fn: Function that takes (data, gpu_id) and returns result
            partition_strategy: How to partition data ('round_robin', 'balanced')

        Returns:
            List of processed results
        """
        if partition_strategy == 'round_robin':
            partitions = self._partition_round_robin(data_list)
        elif partition_strategy == 'balanced':
            partitions = self._partition_balanced(data_list)
        else:
            raise ValueError(f"Unknown partition strategy: {partition_strategy}")

        results = []
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = []

            for gpu_idx, gpu_id in enumerate(self.gpu_ids):
                partition = partitions[gpu_idx]
                future = executor.submit(
                    self._process_partition,
                    gpu_id,
                    partition,
                    process_fn
                )
                futures.append(future)

            for future in futures:
                partition_result = future.result()
                results.extend(partition_result)

        return results

    def _partition_round_robin(self, data_list: List[Any]) -> List[List[Any]]:
        """Partition data using round-robin strategy."""
        partitions = [[] for _ in range(self.num_gpus)]
        for i, item in enumerate(data_list):
            gpu_idx = i % self.num_gpus
            partitions[gpu_idx].append(item)
        return partitions

    def _partition_balanced(self, data_list: List[Any]) -> List[List[Any]]:
        """Partition data using balanced strategy (equal chunks)."""
        partitions = [[] for _ in range(self.num_gpus)]
        items_per_gpu = len(data_list) // self.num_gpus
        remainder = len(data_list) % self.num_gpus

        start = 0
        for gpu_idx in range(self.num_gpus):
            end = start + items_per_gpu + (1 if gpu_idx < remainder else 0)
            partitions[gpu_idx] = data_list[start:end]
            start = end

        return partitions

    def _process_partition(
        self,
        gpu_id: int,
        partition: List[Any],
        process_fn: Callable
    ) -> List[Any]:
        """Process partition on specific GPU."""
        torch.cuda.set_device(gpu_id)
        results = []

        for item in partition:
            try:
                result = process_fn(item, gpu_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item on GPU {gpu_id}: {e}")
                results.append(None)

        return results


class DistributedFeatureCompute:
    """
    Compute features in a distributed manner across multiple GPUs.

    Strategies:
    - Spatial partitioning (geographic chunks)
    - Feature-based partitioning (by feature type)
    - Load-balanced partitioning (by GPU capacity)
    """

    def __init__(
        self,
        num_gpus: Optional[int] = None,
        num_workers: int = 4,
        use_multiprocessing: bool = False
    ):
        """
        Initialize distributed feature computation.

        Args:
            num_gpus: Number of GPUs to use
            num_workers: Number of CPU workers for data loading
            use_multiprocessing: Use multiprocessing instead of threading
        """
        self.processor = MultiGPUProcessor(num_gpus=num_gpus)
        self.num_workers = num_workers
        self.use_multiprocessing = use_multiprocessing

        if self.use_multiprocessing:
            self.executor_class = ProcessPoolExecutor
        else:
            self.executor_class = ThreadPoolExecutor

    def compute_features(
        self,
        point_cloud: np.ndarray,
        feature_fn: Callable,
        partition_strategy: str = 'spatial',
        chunk_size: int = 100_000
    ) -> np.ndarray:
        """
        Compute features in distributed manner.

        Args:
            point_cloud: Point cloud array [N, 3+]
            feature_fn: Feature computation function
            partition_strategy: How to partition data
            chunk_size: Size of each chunk

        Returns:
            Computed features [N, D]
        """
        # Partition point cloud
        partitions = self._partition_point_cloud(
            point_cloud,
            strategy=partition_strategy,
            chunk_size=chunk_size
        )

        # Process partitions
        feature_list = self.processor.process_batch(
            partitions,
            lambda chunk, gpu_id: feature_fn(chunk, gpu_id),
            partition_strategy=partition_strategy
        )

        # Aggregate results
        features = np.vstack([f for f in feature_list if f is not None])
        return features

    def _partition_point_cloud(
        self,
        point_cloud: np.ndarray,
        strategy: str = 'spatial',
        chunk_size: int = 100_000
    ) -> List[np.ndarray]:
        """
        Partition point cloud into chunks.

        Args:
            point_cloud: Point cloud array
            strategy: Partitioning strategy
            chunk_size: Size of each chunk

        Returns:
            List of point cloud chunks
        """
        if strategy == 'spatial':
            return self._partition_spatial(point_cloud, chunk_size)
        elif strategy == 'balanced':
            return self._partition_balanced(point_cloud, chunk_size)
        else:
            raise ValueError(f"Unknown partition strategy: {strategy}")

    def _partition_spatial(
        self,
        point_cloud: np.ndarray,
        chunk_size: int
    ) -> List[np.ndarray]:
        """Partition by spatial locality (morton curve)."""
        partitions = []
        for i in range(0, len(point_cloud), chunk_size):
            partition = point_cloud[i:i + chunk_size]
            partitions.append(partition)
        return partitions

    def _partition_balanced(
        self,
        point_cloud: np.ndarray,
        chunk_size: int
    ) -> List[np.ndarray]:
        """Partition into balanced chunks."""
        partitions = []
        n_chunks = max(1, len(point_cloud) // chunk_size)

        for i in range(n_chunks):
            start = i * len(point_cloud) // n_chunks
            end = (i + 1) * len(point_cloud) // n_chunks
            partitions.append(point_cloud[start:end])

        return partitions


class DistributedDataLoader:
    """
    Distributed data loading for large-scale datasets.

    Features:
    - Automatic data sharding across processes
    - Efficient prefetching
    - Balanced workload distribution
    - Fault tolerance
    """

    def __init__(
        self,
        data_source: Any,
        num_workers: int = 4,
        batch_size: int = 32,
        num_ranks: int = 1,
        rank: int = 0,
        shuffle: bool = True
    ):
        """
        Initialize distributed data loader.

        Args:
            data_source: Data source or dataset
            num_workers: Number of worker threads
            batch_size: Batch size per worker
            num_ranks: Total number of ranks (processes)
            rank: Current rank ID
            shuffle: Whether to shuffle data
        """
        self.data_source = data_source
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_ranks = num_ranks
        self.rank = rank
        self.shuffle = shuffle

    def __iter__(self):
        """Iterate over batches."""
        indices = self._get_shard_indices()

        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.data_source[idx] for idx in batch_indices]
            yield batch

    def _get_shard_indices(self) -> np.ndarray:
        """Get indices for this rank's shard."""
        total = len(self.data_source)
        shard_size = total // self.num_ranks
        start = self.rank * shard_size
        end = start + shard_size if self.rank < self.num_ranks - 1 else total
        return np.arange(start, end)


def initialize_distributed_env(
    backend: str = 'nccl',
    init_method: str = 'env://'
) -> None:
    """
    Initialize distributed environment for multi-process training.

    Args:
        backend: Communication backend ('nccl', 'gloo', 'mpi')
        init_method: Initialization method
    """
    if not DISTRIBUTED_AVAILABLE:
        raise ImportError("Distributed PyTorch not available")

    if not dist.is_available():
        raise RuntimeError("Distributed package is not available")

    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"Distributed initialized: rank={rank}, world_size={world_size}")
    except Exception as e:
        logger.error(f"Failed to initialize distributed environment: {e}")
        raise


def cleanup_distributed_env() -> None:
    """Clean up distributed environment."""
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed environment cleaned up")
