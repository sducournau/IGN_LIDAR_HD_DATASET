"""
Optimization Factory - Intelligent Strategy Selection
==================================================

This module provides a factory pattern for automatically selecting optimal
processing strategies based on data characteristics, hardware capabilities,
and configuration requirements.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from enum import Enum
import time
import gc

logger = logging.getLogger(__name__)

# GPU detection (centralized via GPUManager)
from .gpu import GPUManager
_gpu_manager = GPUManager()
GPU_AVAILABLE = _gpu_manager.gpu_available

if GPU_AVAILABLE:
    import cupy as cp
    try:
        gpu_memory_gb = cp.cuda.Device().mem_info[1] / (1024**3)
        logger.info(f"✓ GPU detected: {gpu_memory_gb:.1f}GB VRAM available")
    except Exception:
        gpu_memory_gb = 0
else:
    gpu_memory_gb = 0

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - using default system values")


class ProcessingStrategy(Enum):
    """Available processing strategies."""
    AUTO = "auto"
    GPU_OPTIMIZED = "gpu_optimized"
    CPU_PARALLEL = "cpu_parallel"  
    MEMORY_EFFICIENT = "memory_efficient"
    HYBRID = "hybrid"


class DataCharacteristics:
    """Analyze data characteristics to inform strategy selection."""
    
    def __init__(self, points_per_tile: int, num_tiles: int, 
                 feature_complexity: str = "medium", has_rgb: bool = False):
        self.points_per_tile = points_per_tile
        self.num_tiles = num_tiles
        self.total_points = points_per_tile * num_tiles
        self.feature_complexity = feature_complexity
        self.has_rgb = has_rgb
        
    @property
    def size_category(self) -> str:
        """Categorize dataset size."""
        if self.total_points < 1_000_000:
            return "small"
        elif self.total_points < 50_000_000:
            return "medium"
        elif self.total_points < 500_000_000:
            return "large"
        else:
            return "massive"
    
    @property
    def complexity_score(self) -> float:
        """Calculate complexity score (0-1)."""
        base_score = 0.3
        
        # Add complexity based on feature types
        if self.feature_complexity == "high":
            base_score += 0.4
        elif self.feature_complexity == "medium":
            base_score += 0.2
            
        # Add complexity for RGB processing
        if self.has_rgb:
            base_score += 0.3
            
        return min(base_score, 1.0)


class SystemCapabilities:
    """Analyze system capabilities."""
    
    def __init__(self):
        if PSUTIL_AVAILABLE:
            import psutil
            self.cpu_cores = psutil.cpu_count(logical=False) or 4
            self.cpu_threads = psutil.cpu_count(logical=True) or 8
            self.memory_gb = psutil.virtual_memory().total / (1024**3)
        else:
            # Default values when psutil not available
            self.cpu_cores = 4
            self.cpu_threads = 8
            self.memory_gb = 16.0
            
        self.gpu_available = GPU_AVAILABLE
        self.gpu_memory_gb = gpu_memory_gb
        
    @property
    def memory_category(self) -> str:
        """Categorize system memory."""
        if self.memory_gb < 8:
            return "low"
        elif self.memory_gb < 32:
            return "medium"
        elif self.memory_gb < 128:
            return "high"
        else:
            return "massive"
    
    @property
    def gpu_category(self) -> str:
        """Categorize GPU capabilities."""
        if not self.gpu_available:
            return "none"
        elif self.gpu_memory_gb < 4:
            return "low"
        elif self.gpu_memory_gb < 12:
            return "medium"
        elif self.gpu_memory_gb < 24:
            return "high"
        else:
            return "massive"
    
    @property
    def parallel_capability(self) -> int:
        """Estimate optimal parallel processing capability."""
        return min(self.cpu_cores, 8)  # Cap at 8 for stability


class OptimizationFactory:
    """
    Factory for selecting optimal processing strategies.
    
    Analyzes data characteristics and system capabilities to recommend
    the best processing strategy and configuration parameters.
    """
    
    def __init__(self):
        self.system = SystemCapabilities()
        logger.info(f"System capabilities: {self.system.cpu_cores} cores, "
                   f"{self.system.memory_gb:.1f}GB RAM, "
                   f"GPU: {self.system.gpu_category}")
    
    def analyze_data(self, data_info: Dict[str, Any]) -> DataCharacteristics:
        """
        Analyze data characteristics from configuration or metadata.
        
        Args:
            data_info: Dictionary containing data information
            
        Returns:
            DataCharacteristics object
        """
        points_per_tile = data_info.get('points_per_tile', 1_000_000)
        num_tiles = data_info.get('num_tiles', 1)
        
        # Determine feature complexity
        feature_complexity = "medium"
        if data_info.get('ground_truth_enabled', False):
            feature_complexity = "high"
        elif data_info.get('basic_features_only', False):
            feature_complexity = "low"
            
        has_rgb = data_info.get('has_rgb', False) or data_info.get('use_rgb', False)
        
        return DataCharacteristics(
            points_per_tile=points_per_tile,
            num_tiles=num_tiles,
            feature_complexity=feature_complexity,
            has_rgb=has_rgb
        )
    
    def select_strategy(self, data_chars: DataCharacteristics, 
                       preferences: Optional[Dict[str, Any]] = None) -> ProcessingStrategy:
        """
        Select optimal processing strategy.
        
        Args:
            data_chars: Data characteristics
            preferences: User preferences (e.g., force_gpu, prefer_cpu)
            
        Returns:
            Recommended ProcessingStrategy
        """
        preferences = preferences or {}
        
        # Check for explicit preferences
        if preferences.get('force_gpu', False) and self.system.gpu_available:
            return ProcessingStrategy.GPU_OPTIMIZED
        if preferences.get('force_cpu', False):
            return ProcessingStrategy.CPU_PARALLEL
        if preferences.get('memory_limited', False):
            return ProcessingStrategy.MEMORY_EFFICIENT
            
        # Automatic selection based on characteristics
        if not self.system.gpu_available:
            # No GPU available
            if data_chars.size_category in ["large", "massive"]:
                return ProcessingStrategy.CPU_PARALLEL
            else:
                return ProcessingStrategy.MEMORY_EFFICIENT
                
        # GPU available - consider memory requirements
        estimated_gpu_memory = self._estimate_gpu_memory_usage(data_chars)
        
        if estimated_gpu_memory > self.system.gpu_memory_gb * 0.8:
            # GPU memory insufficient
            if data_chars.size_category == "massive":
                return ProcessingStrategy.HYBRID
            else:
                return ProcessingStrategy.CPU_PARALLEL
        
        # GPU memory sufficient
        if data_chars.size_category in ["medium", "large", "massive"]:
            return ProcessingStrategy.GPU_OPTIMIZED
        else:
            # Small datasets might be faster on CPU due to overhead
            return ProcessingStrategy.CPU_PARALLEL
    
    def _estimate_gpu_memory_usage(self, data_chars: DataCharacteristics) -> float:
        """Estimate GPU memory usage in GB."""
        # Base memory per point (coordinates + features)
        memory_per_point = 32  # bytes (rough estimate)
        
        # Increase for RGB data
        if data_chars.has_rgb:
            memory_per_point *= 1.5
            
        # Increase for complex features
        if data_chars.feature_complexity == "high":
            memory_per_point *= 2
        elif data_chars.feature_complexity == "medium":
            memory_per_point *= 1.3
            
        # Estimate total memory with overhead
        total_memory_bytes = data_chars.points_per_tile * memory_per_point * 2  # 2x overhead
        return total_memory_bytes / (1024**3)
    
    def get_optimal_config(self, strategy: ProcessingStrategy, 
                          data_chars: DataCharacteristics) -> Dict[str, Any]:
        """
        Get optimal configuration parameters for the selected strategy.
        
        Args:
            strategy: Selected processing strategy
            data_chars: Data characteristics
            
        Returns:
            Dictionary of optimal configuration parameters
        """
        config = {}
        
        if strategy == ProcessingStrategy.GPU_OPTIMIZED:
            config.update(self._get_gpu_config(data_chars))
        elif strategy == ProcessingStrategy.CPU_PARALLEL:
            config.update(self._get_cpu_config(data_chars))
        elif strategy == ProcessingStrategy.MEMORY_EFFICIENT:
            config.update(self._get_memory_config(data_chars))
        elif strategy == ProcessingStrategy.HYBRID:
            config.update(self._get_hybrid_config(data_chars))
            
        return config
    
    def _get_gpu_config(self, data_chars: DataCharacteristics) -> Dict[str, Any]:
        """Get GPU-optimized configuration."""
        # Determine optimal batch size
        if data_chars.size_category == "small":
            batch_size = 500_000
        elif data_chars.size_category == "medium":
            batch_size = 1_000_000
        elif data_chars.size_category == "large":
            batch_size = 2_000_000
        else:  # massive
            batch_size = 3_000_000
            
        # Adjust based on GPU memory
        if self.system.gpu_memory_gb < 8:
            batch_size = min(batch_size, 500_000)
        elif self.system.gpu_memory_gb < 16:
            batch_size = min(batch_size, 1_500_000)
            
        return {
            'use_gpu': True,
            'gpu_batch_size': batch_size,
            'gpu_memory_target': 0.8,
            'enable_memory_pooling': True,
            'enable_async_transfers': True,
            'num_workers': 2  # Reduce CPU workers when using GPU
        }
    
    def _get_cpu_config(self, data_chars: DataCharacteristics) -> Dict[str, Any]:
        """Get CPU-optimized configuration."""
        optimal_workers = min(self.system.parallel_capability, 
                             max(2, self.system.cpu_cores // 2))
        
        # Adjust chunk size based on memory
        if self.system.memory_category == "low":
            chunk_size = 100_000
        elif self.system.memory_category == "medium":
            chunk_size = 500_000
        else:
            chunk_size = 1_000_000
            
        return {
            'use_gpu': False,
            'num_workers': optimal_workers,
            'chunk_size': chunk_size,
            'enable_parallel_processing': True
        }
    
    def _get_memory_config(self, data_chars: DataCharacteristics) -> Dict[str, Any]:
        """Get memory-efficient configuration."""
        return {
            'use_gpu': False,
            'num_workers': 2,  # Conservative
            'chunk_size': 50_000,  # Small chunks
            'enable_parallel_processing': False,
            'adaptive_chunk_sizing': True,
            'architecture': 'standard'  # Use standard orchestrator
        }
    
    def _get_hybrid_config(self, data_chars: DataCharacteristics) -> Dict[str, Any]:
        """Get hybrid GPU/CPU configuration."""
        return {
            'use_gpu': True,
            'gpu_batch_size': 1_000_000,  # Conservative GPU usage
            'gpu_memory_target': 0.6,     # Leave room for CPU processing
            'num_workers': self.system.parallel_capability // 2,
            'enable_parallel_processing': True
        }
    
    def recommend_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze configuration and recommend optimizations.
        
        Args:
            config: Current configuration
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            'strategy': None,
            'config_updates': {},
            'warnings': [],
            'estimated_performance': None
        }
        
        # Extract data characteristics from config
        data_info = {
            'points_per_tile': config.get('processor', {}).get('points_per_tile', 1_000_000),
            'num_tiles': 1,  # Default for single tile processing
            'has_rgb': config.get('features', {}).get('use_rgb', False),
            'ground_truth_enabled': config.get('processor', {}).get('ground_truth_method') is not None
        }
        
        data_chars = self.analyze_data(data_info)
        
        # Get current strategy
        current_gpu = config.get('processor', {}).get('use_gpu', False)
        current_arch = config.get('processing', {}).get('architecture', 'standard')
        
        # Recommend optimal strategy
        optimal_strategy = self.select_strategy(data_chars)
        optimal_config = self.get_optimal_config(optimal_strategy, data_chars)
        
        recommendations['strategy'] = optimal_strategy
        recommendations['config_updates'] = optimal_config
        
        # Add warnings for potential issues
        if current_gpu and not self.system.gpu_available:
            recommendations['warnings'].append(
                "GPU requested but not available - will fallback to CPU"
            )
            
        if data_chars.size_category == "massive" and self.system.memory_category == "low":
            recommendations['warnings'].append(
                "Large dataset with limited memory - consider processing in smaller batches"
            )
            
        # Estimate performance improvement
        if optimal_strategy == ProcessingStrategy.GPU_OPTIMIZED and not current_gpu:
            recommendations['estimated_performance'] = "2-5x speedup with GPU optimization"
        elif optimal_strategy == ProcessingStrategy.CPU_PARALLEL and current_arch == 'standard':
            recommendations['estimated_performance'] = "1.5-3x speedup with optimized orchestrator"
            
        return recommendations


# Factory instance for easy access
optimization_factory = OptimizationFactory()


def get_optimization_recommendations(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to get optimization recommendations.
    
    Args:
        config: Current configuration
        
    Returns:
        Dictionary with optimization recommendations
    """
    return optimization_factory.recommend_optimization(config)


def auto_optimize_config(config: Dict[str, Any], 
                        preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Automatically optimize configuration based on system capabilities.
    
    Args:
        config: Base configuration
        preferences: User preferences
        
    Returns:
        Optimized configuration
    """
    preferences = preferences or {}
    
    # Extract data info
    data_info = {
        'points_per_tile': config.get('processor', {}).get('points_per_tile', 1_000_000),
        'num_tiles': 1,
        'has_rgb': config.get('features', {}).get('use_rgb', False),
        'ground_truth_enabled': config.get('processor', {}).get('ground_truth_method') is not None
    }
    
    data_chars = optimization_factory.analyze_data(data_info)
    strategy = optimization_factory.select_strategy(data_chars, preferences)
    optimal_config = optimization_factory.get_optimal_config(strategy, data_chars)
    
    # Merge with existing config
    optimized_config = config.copy()
    
    # Update processor settings
    if 'processor' not in optimized_config:
        optimized_config['processor'] = {}
    optimized_config['processor'].update(optimal_config)
    
    # Update processing settings
    if 'processing' not in optimized_config:
        optimized_config['processing'] = {}
    if 'architecture' in optimal_config:
        optimized_config['processing']['architecture'] = optimal_config['architecture']
    
    logger.info(f"Auto-optimization: Selected {strategy.value} strategy")
    
    return optimized_config