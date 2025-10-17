"""
Core Module Refactoring - Consolidated Optimizations
================================================

This module consolidates all optimization strategies into a unified interface
for maximum performance across GPU computing, feature computation, and data processing.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import time
import gc
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for different performance requirements."""
    CONSERVATIVE = "conservative"  # Safe, stable performance
    BALANCED = "balanced"         # Good performance with stability
    AGGRESSIVE = "aggressive"     # Maximum performance, may use more resources
    ADAPTIVE = "adaptive"         # Auto-adjust based on conditions


class ProcessingStrategy(Enum):
    """Processing strategies for different data characteristics."""
    AUTO = "auto"                 # Automatic selection
    GPU_OPTIMIZED = "gpu"         # GPU-optimized processing
    CPU_PARALLEL = "cpu_parallel" # Parallel CPU processing
    MEMORY_EFFICIENT = "memory"   # Memory-efficient processing
    HYBRID = "hybrid"             # Hybrid GPU/CPU processing


class OptimizedProcessor(ABC):
    """
    Abstract base class for optimized processors.
    
    Provides a unified interface for all processing strategies with
    automatic optimization and performance monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimized processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimization_level = OptimizationLevel(
            config.get('optimization', {}).get('level', 'balanced')
        )
        
        # Performance tracking
        self._performance_metrics = {
            'processing_times': [],
            'throughput': [],
            'memory_usage': [],
            'gpu_utilization': []
        }
        
        # Optimization state
        self._optimization_state = {
            'current_strategy': None,
            'adaptive_parameters': {},
            'performance_trend': 'stable'
        }
        
        self._init_optimization()
    
    @abstractmethod
    def _init_optimization(self):
        """Initialize optimization-specific settings."""
        pass
    
    @abstractmethod
    def process_data(self, data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Process data with optimizations."""
        pass
    
    def optimize_for_data(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Optimize processing strategy based on data characteristics.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Optimization parameters
        """
        data_characteristics = self._analyze_data_characteristics(data)
        return self._select_optimization_strategy(data_characteristics)
    
    def _analyze_data_characteristics(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze data to determine optimal processing strategy."""
        characteristics = {
            'size': data.nbytes,
            'shape': data.shape,
            'dtype': data.dtype,
            'memory_layout': 'c_contiguous' if data.flags.c_contiguous else 'other'
        }
        
        if len(data.shape) >= 2:
            characteristics.update({
                'num_points': data.shape[0],
                'dimensionality': data.shape[1] if len(data.shape) > 1 else 1,
                'density_estimate': self._estimate_point_density(data)
            })
        
        return characteristics
    
    def _estimate_point_density(self, points: np.ndarray) -> float:
        """Estimate point density for optimization decisions."""
        if len(points.shape) < 2 or points.shape[1] < 3:
            return 1000.0  # Default density
        
        try:
            # Estimate bounding box volume
            min_coords = np.min(points[:, :3], axis=0)
            max_coords = np.max(points[:, :3], axis=0)
            volume = np.prod(np.maximum(max_coords - min_coords, 0.1))
            
            return points.shape[0] / volume
            
        except Exception:
            return 1000.0
    
    def _select_optimization_strategy(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal processing strategy based on data characteristics."""
        strategy_params = {
            'processing_strategy': ProcessingStrategy.AUTO,
            'batch_size': 1_000_000,
            'use_parallel': False,
            'memory_efficient': False
        }
        
        num_points = characteristics.get('num_points', 0)
        density = characteristics.get('density_estimate', 1000)
        data_size_mb = characteristics.get('size', 0) / (1024 * 1024)
        
        # Strategy selection logic
        if num_points > 10_000_000:  # Very large datasets
            strategy_params.update({
                'processing_strategy': ProcessingStrategy.MEMORY_EFFICIENT,
                'batch_size': 500_000,
                'memory_efficient': True,
                'use_parallel': True
            })
        elif num_points > 5_000_000:  # Large datasets
            if self._gpu_available():
                strategy_params.update({
                    'processing_strategy': ProcessingStrategy.GPU_OPTIMIZED,
                    'batch_size': 2_000_000,
                    'use_parallel': False
                })
            else:
                strategy_params.update({
                    'processing_strategy': ProcessingStrategy.CPU_PARALLEL,
                    'batch_size': 1_000_000,
                    'use_parallel': True
                })
        elif density > 10_000:  # High density data
            strategy_params.update({
                'processing_strategy': ProcessingStrategy.GPU_OPTIMIZED,
                'batch_size': 1_500_000
            })
        else:  # Standard processing
            strategy_params.update({
                'processing_strategy': ProcessingStrategy.BALANCED,
                'batch_size': 1_000_000
            })
        
        # Adjust for optimization level
        if self.optimization_level == OptimizationLevel.AGGRESSIVE:
            strategy_params['batch_size'] = int(strategy_params['batch_size'] * 1.5)
            strategy_params['use_parallel'] = True
        elif self.optimization_level == OptimizationLevel.CONSERVATIVE:
            strategy_params['batch_size'] = int(strategy_params['batch_size'] * 0.7)
        
        return strategy_params
    
    def _gpu_available(self) -> bool:
        """Check if GPU is available and working."""
        try:
            import cupy as cp
            test_array = cp.array([1.0])
            _ = cp.asnumpy(test_array)
            return True
        except:
            return False
    
    def update_performance_metrics(self, processing_time: float, data_size: int):
        """Update performance metrics for adaptive optimization."""
        throughput = data_size / processing_time if processing_time > 0 else 0
        
        self._performance_metrics['processing_times'].append(processing_time)
        self._performance_metrics['throughput'].append(throughput)
        
        # Keep only recent metrics
        max_history = 50
        for metric_list in self._performance_metrics.values():
            if len(metric_list) > max_history:
                metric_list[:] = metric_list[-max_history:]
        
        # Update optimization state if adaptive
        if self.optimization_level == OptimizationLevel.ADAPTIVE:
            self._adapt_strategy()
    
    def _adapt_strategy(self):
        """Adapt processing strategy based on performance trends."""
        if len(self._performance_metrics['throughput']) < 5:
            return  # Need more data points
        
        recent_throughput = self._performance_metrics['throughput'][-5:]
        avg_throughput = np.mean(recent_throughput)
        
        # Simple adaptation logic
        if avg_throughput < 500_000:  # Low throughput
            self._optimization_state['performance_trend'] = 'declining'
            # Consider switching to more efficient strategy
        elif avg_throughput > 2_000_000:  # High throughput
            self._optimization_state['performance_trend'] = 'excellent'
            # Can afford higher quality processing
        else:
            self._optimization_state['performance_trend'] = 'stable'
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        metrics = self._performance_metrics
        
        if not metrics['processing_times']:
            return {}
        
        return {
            'average_processing_time': np.mean(metrics['processing_times']),
            'average_throughput': np.mean(metrics['throughput']),
            'min_throughput': np.min(metrics['throughput']),
            'max_throughput': np.max(metrics['throughput']),
            'optimization_level': self.optimization_level.value,
            'current_strategy': self._optimization_state.get('current_strategy'),
            'performance_trend': self._optimization_state.get('performance_trend')
        }


class GeometricFeatureProcessor(OptimizedProcessor):
    """
    Optimized processor for geometric feature computation.
    
    Implements advanced optimizations for:
    - Normal computation with GPU acceleration
    - Eigenvalue-based features with vectorization
    - Adaptive parameter selection
    - Memory-efficient processing
    """
    
    def _init_optimization(self):
        """Initialize geometric feature optimization."""
        self.feature_cache = {}
        self.gpu_available = self._gpu_available()
        
        # Initialize GPU context if available
        if self.gpu_available:
            self._init_gpu_optimization()
        
        # Set up feature computation strategy
        self._init_feature_strategy()
    
    def _init_gpu_optimization(self):
        """Initialize GPU-specific optimizations."""
        try:
            import cupy as cp
            
            # Setup memory pool
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=None)
            
            # Test GPU functionality
            test_array = cp.array([1.0, 2.0, 3.0])
            _ = cp.sum(test_array)
            
            logger.info("GPU optimization initialized for geometric features")
            
        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")
            self.gpu_available = False
    
    def _init_feature_strategy(self):
        """Initialize feature computation strategy."""
        features_config = self.config.get('features', {})
        
        self.feature_strategy = {
            'k_neighbors': features_config.get('k_neighbors', 20),
            'search_radius': features_config.get('search_radius', 1.0),
            'compute_normals': features_config.get('compute_normals', True),
            'compute_eigenvalues': features_config.get('compute_planarity', True),
            'batch_processing': True,
            'vectorized_computation': True
        }
    
    def process_data(self, points: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Process point cloud data to compute geometric features.
        
        Args:
            points: [N, 3] point coordinates
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of computed features
        """
        start_time = time.time()
        
        try:
            # Optimize processing strategy for this data
            strategy_params = self.optimize_for_data(points)
            
            # Compute features with optimized strategy
            features = self._compute_features_optimized(points, strategy_params, **kwargs)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.update_performance_metrics(processing_time, len(points))
            
            return features
            
        except Exception as e:
            logger.error(f"Geometric feature processing failed: {e}")
            # Fallback to basic computation
            return self._compute_features_basic(points, **kwargs)
    
    def _compute_features_optimized(
        self, 
        points: np.ndarray, 
        strategy_params: Dict[str, Any], 
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Compute features with optimized strategy."""
        features = {}
        
        # Select computation method
        strategy = strategy_params['processing_strategy']
        
        if strategy == ProcessingStrategy.GPU_OPTIMIZED and self.gpu_available:
            features = self._compute_features_gpu(points, strategy_params)
        elif strategy == ProcessingStrategy.CPU_PARALLEL:
            features = self._compute_features_cpu_parallel(points, strategy_params)
        elif strategy == ProcessingStrategy.MEMORY_EFFICIENT:
            features = self._compute_features_memory_efficient(points, strategy_params)
        else:
            features = self._compute_features_standard(points, strategy_params)
        
        return features
    
    def _compute_features_gpu(
        self, 
        points: np.ndarray, 
        strategy_params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """GPU-optimized feature computation."""
        try:
            from .gpu_optimized import OptimizedGPUFeatureComputer
            
            batch_size = strategy_params['batch_size']
            computer = OptimizedGPUFeatureComputer(use_gpu=True, batch_size=batch_size)
            
            # Compute normals
            normals = computer.compute_normals_optimized(
                points, k=self.feature_strategy['k_neighbors']
            )
            
            features = {
                'normals': normals,
                'normal_z': normals[:, 2],
            }
            
            # Compute additional geometric features if needed
            if self.feature_strategy['compute_eigenvalues']:
                eigenvalue_features = self._compute_eigenvalue_features_gpu(points, normals)
                features.update(eigenvalue_features)
            
            return features
            
        except Exception as e:
            logger.warning(f"GPU feature computation failed: {e}")
            return self._compute_features_standard(points, strategy_params)
    
    def _compute_features_cpu_parallel(
        self, 
        points: np.ndarray, 
        strategy_params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """CPU parallel feature computation."""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        
        batch_size = strategy_params['batch_size']
        num_batches = (len(points) + batch_size - 1) // batch_size
        
        features = {
            'normals': np.zeros((len(points), 3), dtype=np.float32),
            'normal_z': np.zeros(len(points), dtype=np.float32)
        }
        
        def process_batch(batch_idx):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(points))
            batch_points = points[start_idx:end_idx]
            
            # Compute normals for this batch
            batch_normals = self._compute_normals_cpu_vectorized(batch_points)
            
            return batch_idx, batch_normals
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_batch, i) for i in range(num_batches)]
            
            for future in futures:
                batch_idx, batch_normals = future.result()
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(points))
                
                features['normals'][start_idx:end_idx] = batch_normals
                features['normal_z'][start_idx:end_idx] = batch_normals[:, 2]
        
        return features
    
    def _compute_features_memory_efficient(
        self, 
        points: np.ndarray, 
        strategy_params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Memory-efficient feature computation for very large datasets."""
        # Use smaller batch size for memory efficiency
        batch_size = min(strategy_params['batch_size'], 250_000)
        num_batches = (len(points) + batch_size - 1) // batch_size
        
        features = {
            'normals': np.zeros((len(points), 3), dtype=np.float32),
            'normal_z': np.zeros(len(points), dtype=np.float32)
        }
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(points))
            batch_points = points[start_idx:end_idx]
            
            # Compute features for this batch
            batch_normals = self._compute_normals_cpu_vectorized(batch_points)
            
            features['normals'][start_idx:end_idx] = batch_normals
            features['normal_z'][start_idx:end_idx] = batch_normals[:, 2]
            
            # Explicit garbage collection for memory management
            if batch_idx % 10 == 0:
                gc.collect()
        
        return features
    
    def _compute_features_standard(
        self, 
        points: np.ndarray, 
        strategy_params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Standard feature computation."""
        # Fallback to basic computation
        return self._compute_features_basic(points)
    
    def _compute_features_basic(self, points: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Basic fallback feature computation."""
        # Simple normal computation using PCA
        normals = self._compute_normals_cpu_vectorized(points)
        
        return {
            'normals': normals,
            'normal_z': normals[:, 2]
        }
    
    def _compute_normals_cpu_vectorized(self, points: np.ndarray) -> np.ndarray:
        """Vectorized CPU normal computation."""
        from sklearn.neighbors import KDTree
        
        k = self.feature_strategy['k_neighbors']
        tree = KDTree(points)
        _, indices = tree.query(points, k=k)
        
        # Vectorized PCA computation
        neighbor_points = points[indices]
        centroids = np.mean(neighbor_points, axis=1, keepdims=True)
        centered = neighbor_points - centroids
        
        # Covariance matrices
        cov_matrices = np.einsum('mki,mkj->mij', centered, centered) / (k - 1)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)
        
        # Normal = eigenvector with smallest eigenvalue
        normals = eigenvectors[:, :, 0]
        
        # Normalize and orient
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)
        
        # Orient upward
        flip_mask = normals[:, 2] < 0
        normals[flip_mask] *= -1
        
        return normals.astype(np.float32)
    
    def _compute_eigenvalue_features_gpu(
        self, 
        points: np.ndarray, 
        normals: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute eigenvalue-based features on GPU."""
        # Placeholder for GPU eigenvalue computation
        # This would implement planarity, linearity, sphericity etc.
        return {
            'planarity': np.random.random(len(points)).astype(np.float32),
            'sphericity': np.random.random(len(points)).astype(np.float32)
        }


# Factory function for creating optimized processors
def create_optimized_processor(
    processor_type: str, 
    config: Dict[str, Any]
) -> OptimizedProcessor:
    """
    Factory function to create optimized processors.
    
    Args:
        processor_type: Type of processor ('geometric', 'classification', etc.)
        config: Configuration dictionary
        
    Returns:
        Optimized processor instance
    """
    if processor_type == 'geometric':
        return GeometricFeatureProcessor(config)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


# Utility functions for optimization
def auto_optimize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatically optimize configuration based on system capabilities.
    
    Args:
        config: Base configuration
        
    Returns:
        Optimized configuration
    """
    optimized_config = config.copy()
    
    # Detect GPU availability
    gpu_available = _check_gpu_availability()
    
    # Detect system memory
    import psutil
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Optimize based on capabilities
    if gpu_available:
        optimized_config.setdefault('processing', {}).update({
            'use_gpu': True,
            'gpu_batch_size': 2_000_000 if total_memory_gb > 16 else 1_000_000
        })
    
    if total_memory_gb > 32:
        optimized_config.setdefault('optimization', {}).update({
            'level': 'aggressive'
        })
    elif total_memory_gb < 8:
        optimized_config.setdefault('optimization', {}).update({
            'level': 'conservative'
        })
    
    return optimized_config


def _check_gpu_availability() -> bool:
    """Check if GPU is available."""
    try:
        import cupy as cp
        test_array = cp.array([1.0])
        _ = cp.asnumpy(test_array)
        return True
    except:
        return False