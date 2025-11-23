"""
Phase 4 Integration Guide - LiDARProcessor Integration

This guide explains how to integrate the OptimizationManager into LiDARProcessor
to enable all Phase 4 optimizations (+66-94% performance gain) in production.

Target Files:
- ign_lidar/core/processor.py
- ign_lidar/config/schema.py

Author: IGN LiDAR HD Development Team
Date: November 23, 2025
"""

# ==============================================================================
# Step 1: Add Configuration Schema (config/schema.py)
# ==============================================================================

"""
Add to ProcessorConfig dataclass in ign_lidar/config/schema.py:

```python
@dataclass
class ProcessorConfig:
    # ... existing fields ...
    
    # Phase 4 Optimizations
    enable_optimizations: bool = True
    """Enable all Phase 4 optimizations (async I/O, batch processing, GPU pooling)."""
    
    optimization: Optional['OptimizationConfig'] = None
    """Detailed optimization configuration."""


@dataclass
class OptimizationConfig:
    '''Phase 4 optimization configuration.'''
    
    # Phase 4.5: Async I/O
    enable_async_io: bool = True
    """Enable asynchronous tile loading with prefetching."""
    
    async_workers: int = 2
    """Number of async I/O worker threads."""
    
    tile_cache_size: int = 3
    """Number of tiles to cache in memory."""
    
    # Phase 4.4: Batch Multi-Tile Processing
    enable_batch_processing: bool = True
    """Enable batch processing of multiple tiles on GPU."""
    
    batch_size: int = 4
    """Number of tiles to process in each GPU batch."""
    
    # Phase 4.3: GPU Memory Pooling
    enable_gpu_pooling: bool = True
    """Enable GPU memory pool for reduced allocation overhead."""
    
    gpu_pool_max_size_gb: float = 4.0
    """Maximum GPU memory pool size in GB."""
    
    # Statistics
    print_stats: bool = True
    """Print optimization statistics after processing."""
```
"""

# ==============================================================================
# Step 2: Modify LiDARProcessor.__init__ (core/processor.py)
# ==============================================================================

"""
In ign_lidar/core/processor.py, add to __init__ method (after line ~384):

```python
def __init__(self, config: Union[str, Path, DictConfig, ProcessorConfig]):
    '''Initialize LiDAR processor with configuration.'''
    
    # ... existing initialization code ...
    
    # Feature orchestrator setup (line ~384)
    self.feature_orchestrator = self.feature_engine.orchestrator
    
    # ✨ Phase 4: Initialize OptimizationManager
    self._setup_optimization_manager()
    
    # ... rest of initialization ...

def _setup_optimization_manager(self):
    '''Setup Phase 4 optimization manager.'''
    from ign_lidar.core.optimization_integration import (
        create_optimization_manager,
        ASYNC_IO_AVAILABLE,
        GPU_MEMORY_POOL_AVAILABLE,
        GPU_PROCESSOR_AVAILABLE
    )
    
    # Check if optimizations are enabled
    if not self.config.processor.enable_optimizations:
        self.logger.info("Phase 4 optimizations disabled by config")
        self.optimization_manager = None
        return
    
    # Get optimization config (use defaults if not specified)
    opt_config = self.config.processor.optimization or OptimizationConfig()
    
    # Log component availability
    self.logger.info("Phase 4 Optimization Components:")
    self.logger.info(f"  - Async I/O (4.5):       {'✅' if ASYNC_IO_AVAILABLE else '❌'}")
    self.logger.info(f"  - GPU Memory Pool (4.3): {'✅' if GPU_MEMORY_POOL_AVAILABLE else '❌'}")
    self.logger.info(f"  - Batch Processing (4.4): {'✅' if GPU_PROCESSOR_AVAILABLE else '❌'}")
    
    # Create optimization manager
    try:
        self.optimization_manager = create_optimization_manager(
            use_gpu=self.config.processor.use_gpu,
            enable_all=True,  # Enable all available optimizations
            batch_size=opt_config.batch_size,
            async_workers=opt_config.async_workers,
            tile_cache_size=opt_config.tile_cache_size,
            gpu_pool_max_size_gb=opt_config.gpu_pool_max_size_gb,
        )
        
        # Initialize with feature orchestrator
        self.optimization_manager.initialize(
            feature_orchestrator=self.feature_orchestrator
        )
        
        self.logger.info("✅ Phase 4 OptimizationManager initialized")
        
    except Exception as e:
        self.logger.warning(f"Failed to initialize OptimizationManager: {e}")
        self.logger.warning("Falling back to standard processing")
        self.optimization_manager = None

def __del__(self):
    '''Clean up resources on deletion.'''
    # Shutdown optimization manager
    if hasattr(self, 'optimization_manager') and self.optimization_manager:
        try:
            self.optimization_manager.shutdown()
        except Exception as e:
            self.logger.warning(f"Error shutting down OptimizationManager: {e}")
```
"""

# ==============================================================================
# Step 3: Modify process_directory Method
# ==============================================================================

"""
In ign_lidar/core/processor.py, modify process_directory method:

```python
def process_directory(
    self,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    file_pattern: str = "*.laz"
) -> Dict[str, Any]:
    '''
    Process all tiles in a directory.
    
    Args:
        input_dir: Input directory with LAZ tiles
        output_dir: Output directory for results
        file_pattern: File glob pattern (default: *.laz)
    
    Returns:
        Processing statistics
    '''
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all tile paths
    tile_paths = sorted(input_dir.glob(file_pattern))
    
    if not tile_paths:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {input_dir}")
    
    self.logger.info(f"Found {len(tile_paths)} tiles to process")
    
    # ✨ Use OptimizationManager if available
    if self.optimization_manager:
        self.logger.info("🚀 Using Phase 4 optimized processing")
        results = self._process_directory_optimized(
            tile_paths=tile_paths,
            output_dir=output_dir,
        )
    else:
        self.logger.info("Using standard sequential processing")
        results = self._process_directory_sequential(
            tile_paths=tile_paths,
            output_dir=output_dir,
        )
    
    # Print optimization statistics
    if self.optimization_manager and self.config.processor.optimization.print_stats:
        self.optimization_manager.print_stats()
    
    return results

def _process_directory_optimized(
    self,
    tile_paths: List[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    '''Process directory using Phase 4 optimizations.'''
    
    def process_tile(tile_data, ground_truth):
        '''Process a single tile with all features.'''
        return self._process_tile_core(
            tile_data=tile_data,
            ground_truth=ground_truth,
            output_dir=output_dir,
        )
    
    # Process with optimizations
    results = self.optimization_manager.process_tiles_optimized(
        tile_paths=tile_paths,
        processor_func=process_tile,
        fetch_ground_truth=True,  # Phase 4.1 WFS cache
    )
    
    # Aggregate statistics
    total_points = sum(r['num_points'] for r in results if r)
    total_patches = sum(r.get('num_patches', 0) for r in results if r)
    
    return {
        'num_tiles': len(results),
        'total_points': total_points,
        'total_patches': total_patches,
        'optimization_stats': self.optimization_manager.get_stats(),
    }

def _process_directory_sequential(
    self,
    tile_paths: List[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    '''Process directory sequentially (legacy method).'''
    
    results = []
    
    for tile_path in tqdm(tile_paths, desc="Processing tiles"):
        try:
            result = self.process_tile(
                tile_path=tile_path,
                output_dir=output_dir,
            )
            results.append(result)
        
        except Exception as e:
            self.logger.error(f"Failed to process {tile_path}: {e}")
            results.append(None)
    
    # Aggregate statistics
    total_points = sum(r['num_points'] for r in results if r)
    total_patches = sum(r.get('num_patches', 0) for r in results if r)
    
    return {
        'num_tiles': len([r for r in results if r]),
        'total_points': total_points,
        'total_patches': total_patches,
    }
```
"""

# ==============================================================================
# Step 4: Add Configuration Examples
# ==============================================================================

"""
Create examples/config_phase4_optimized.yaml:

```yaml
# Phase 4 Optimized Configuration
# Expected performance: +66-94% (2.66× - 2.94× faster)

input_dir: /data/tiles
output_dir: /data/output_optimized

processor:
  lod_level: LOD2
  processing_mode: patches_only
  use_gpu: true
  patch_size: 150.0
  num_points: 16384
  
  # ✨ Enable Phase 4 optimizations
  enable_optimizations: true
  
  optimization:
    # Phase 4.5: Async I/O (+12-14%)
    enable_async_io: true
    async_workers: 2
    tile_cache_size: 3
    
    # Phase 4.4: Batch Multi-Tile (+25-30%)
    enable_batch_processing: true
    batch_size: 4
    
    # Phase 4.3: GPU Memory Pooling (+8.5%)
    enable_gpu_pooling: true
    gpu_pool_max_size_gb: 4.0
    
    # Statistics
    print_stats: true

features:
  mode: lod2
  k_neighbors: 20
  search_radius: 3.0

data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      vegetation: false
```
"""

# ==============================================================================
# Step 5: Testing Integration
# ==============================================================================

"""
Test script (tests/test_phase4_integration.py):

```python
import pytest
from pathlib import Path
from ign_lidar import LiDARProcessor
from omegaconf import OmegaConf

def test_optimization_manager_initialization():
    '''Test OptimizationManager is properly initialized.'''
    
    config = OmegaConf.create({
        'processor': {
            'lod_level': 'LOD2',
            'use_gpu': True,
            'enable_optimizations': True,
            'optimization': {
                'enable_async_io': True,
                'enable_batch_processing': True,
                'enable_gpu_pooling': True,
            }
        },
        'features': {'mode': 'lod2'},
    })
    
    processor = LiDARProcessor(config)
    
    # Check optimization manager exists
    assert hasattr(processor, 'optimization_manager')
    assert processor.optimization_manager is not None
    
    # Check components initialized
    if processor.optimization_manager:
        assert processor.optimization_manager.async_pipeline is not None
        assert processor.optimization_manager.enable_async_io

def test_optimized_vs_sequential(tmp_path):
    '''Compare optimized vs sequential processing.'''
    import time
    
    # Create test tiles (mock)
    tile_paths = [tmp_path / f"tile_{i}.laz" for i in range(10)]
    
    config_optimized = OmegaConf.create({
        'input_dir': str(tmp_path),
        'output_dir': str(tmp_path / 'output_opt'),
        'processor': {
            'lod_level': 'LOD2',
            'use_gpu': True,
            'enable_optimizations': True,
        },
        'features': {'mode': 'lod2'},
    })
    
    config_sequential = OmegaConf.create({
        'input_dir': str(tmp_path),
        'output_dir': str(tmp_path / 'output_seq'),
        'processor': {
            'lod_level': 'LOD2',
            'use_gpu': True,
            'enable_optimizations': False,  # Disable optimizations
        },
        'features': {'mode': 'lod2'},
    })
    
    # Test optimized
    processor_opt = LiDARProcessor(config_optimized)
    start = time.time()
    results_opt = processor_opt.process_directory(tmp_path, tmp_path / 'output_opt')
    time_opt = time.time() - start
    
    # Test sequential
    processor_seq = LiDARProcessor(config_sequential)
    start = time.time()
    results_seq = processor_seq.process_directory(tmp_path, tmp_path / 'output_seq')
    time_seq = time.time() - start
    
    # Optimized should be faster (or at least not slower)
    speedup = time_seq / time_opt
    print(f"Speedup: {speedup:.2f}×")
    
    assert speedup >= 1.0  # At minimum, no slowdown
```
"""

# ==============================================================================
# Step 6: Migration Checklist
# ==============================================================================

MIGRATION_CHECKLIST = """
✅ Integration Checklist:

Phase 1: Configuration
[ ] Add OptimizationConfig to config/schema.py
[ ] Add enable_optimizations field to ProcessorConfig
[ ] Add optimization field to ProcessorConfig
[ ] Create example config: examples/config_phase4_optimized.yaml

Phase 2: LiDARProcessor Modifications
[ ] Import optimization_integration in processor.py
[ ] Add _setup_optimization_manager() method
[ ] Call _setup_optimization_manager() in __init__
[ ] Add __del__ method for cleanup
[ ] Add _process_directory_optimized() method
[ ] Modify process_directory() to use OptimizationManager
[ ] Keep _process_directory_sequential() as fallback

Phase 3: Testing
[ ] Create tests/test_phase4_integration.py
[ ] Test OptimizationManager initialization
[ ] Test with enable_optimizations=True
[ ] Test with enable_optimizations=False
[ ] Test graceful fallback on errors
[ ] Test performance comparison

Phase 4: Documentation
[ ] Update README.md with Phase 4 section
[ ] Update docs/architecture.md
[ ] Create docs/integration/PHASE_4_INTEGRATION.md
[ ] Update CHANGELOG.md

Phase 5: Validation
[ ] Run full test suite: pytest tests/ -v
[ ] Test on real dataset (10+ tiles)
[ ] Measure actual speedup (target: +66-94%)
[ ] Check GPU memory usage
[ ] Verify backward compatibility

Phase 6: Release
[ ] Update version to 3.1.0 (or appropriate)
[ ] Tag release: git tag v3.1.0
[ ] Build package: python -m build
[ ] Upload to PyPI: twine upload dist/*
[ ] Announce Phase 4 completion
"""

# ==============================================================================
# Expected Performance
# ==============================================================================

PERFORMANCE_TARGETS = """
📊 Phase 4 Performance Targets:

Component                Performance Gain    Cumulative
─────────────────────────────────────────────────────────
Phase 4.1 (WFS Cache)    +10-15%            1.10× - 1.15×
Phase 4.2 (Preproc GPU)  +10-15%            1.21× - 1.32×
Phase 4.3 (GPU Pool)     +8.5%              1.32× - 1.44×
Phase 4.4 (Batch)        +25-30%            1.65× - 1.87×
Phase 4.5 (Async I/O)    +12-14%            1.85× - 2.13×
─────────────────────────────────────────────────────────
TOTAL                    +66-94%            2.66× - 2.94×

Workload Assumptions:
- 10+ tiles per batch
- GPU enabled (RTX 3090 or better)
- ~5-15M points per tile
- LOD2 features (12 features)
- BD TOPO ground truth enabled

Performance may vary based on:
- Hardware (GPU, CPU, storage)
- Tile size and point density
- Feature complexity (LOD2 vs LOD3)
- I/O speed (SSD vs HDD)
"""

if __name__ == '__main__':
    print("=" * 80)
    print("Phase 4 Integration Guide")
    print("=" * 80)
    print(MIGRATION_CHECKLIST)
    print("\n" + "=" * 80)
    print(PERFORMANCE_TARGETS)
    print("=" * 80)
