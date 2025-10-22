---
sidebar_position: 1
---

# Core Module API

The `ign_lidar.core` module provides the main processing classes for the IGN LiDAR HD v2.0+ unified pipeline.

## Overview

The core module contains:

- **`LiDARProcessor`** - Main processing orchestrator
- **`TileStitcher`** - Multi-tile stitching functionality
- **`BoundaryHandler`** - Boundary-aware feature computation
- **`PipelineManager`** - Pipeline state and configuration management

---

## LiDARProcessor

Main class for processing LiDAR tiles through the unified pipeline.

### Class Definition

```python
from ign_lidar.core.processor import LiDARProcessor
from omegaconf import DictConfig

class LiDARProcessor:
    """
    Unified LiDAR processing pipeline.

    Handles RAW LAZ → Enriched LAZ → Patches in a single workflow.
    """

    def __init__(
        self,
        config: DictConfig,
        gpu_enabled: bool = False,
        verbose: bool = True
    ):
        """
        Initialize processor with configuration.

        Args:
            config: Hydra configuration object
            gpu_enabled: Enable GPU acceleration
            verbose: Print progress information
        """
```

### Methods

#### `process_tile()`

Process a single LAZ tile through the complete pipeline.

```python
def process_tile(
    self,
    input_path: str,
    output_path: str,
    save_enriched: bool = False,
    enriched_path: Optional[str] = None
) -> ProcessingResult:
    """
    Process single tile: RAW LAZ → Patches.

    Args:
        input_path: Path to input RAW LAZ file
        output_path: Path for output patches LAZ
        save_enriched: Whether to save enriched LAZ
        enriched_path: Path for enriched LAZ (if save_enriched=True)

    Returns:
        ProcessingResult with statistics

    Raises:
        FileNotFoundError: Input file doesn't exist
        ValueError: Invalid LAZ format
        RuntimeError: Processing error
    """
```

**Example:**

```python
from ign_lidar.core.processor import LiDARProcessor
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("config.yaml")

# Create processor
processor = LiDARProcessor(config, gpu_enabled=True)

# Process tile
result = processor.process_tile(
    input_path="data/raw/tile_001.laz",
    output_path="data/output/tile_001.laz"
)

print(f"Processed {result.num_points:,} points")
print(f"Extracted {result.num_patches} patches")
print(f"Time: {result.elapsed_time:.2f}s")
```

#### `process_directory()`

Process all LAZ files in a directory.

```python
def process_directory(
    self,
    input_dir: str,
    output_dir: str,
    pattern: str = "*.laz",
    parallel: bool = True,
    max_workers: int = 4
) -> List[ProcessingResult]:
    """
    Process all LAZ files in directory.

    Args:
        input_dir: Directory containing RAW LAZ files
        output_dir: Output directory for patches
        pattern: File pattern (glob) to match
        parallel: Process files in parallel
        max_workers: Number of parallel workers

    Returns:
        List of ProcessingResult objects
    """
```

**Example:**

```python
# Process entire directory
results = processor.process_directory(
    input_dir="data/raw",
    output_dir="data/output",
    pattern="*.laz",
    parallel=True,
    max_workers=4
)

# Summary
total_points = sum(r.num_points for r in results)
total_time = sum(r.elapsed_time for r in results)
print(f"Processed {len(results)} tiles")
print(f"Total points: {total_points:,}")
print(f"Total time: {total_time:.2f}s")
```

#### `process_with_stitching()`

Process multiple tiles with stitching.

```python
def process_with_stitching(
    self,
    input_paths: List[str],
    output_path: str,
    stitch_config: Optional[DictConfig] = None
) -> ProcessingResult:
    """
    Process and stitch multiple tiles.

    Args:
        input_paths: List of input LAZ file paths
        output_path: Output path for stitched result
        stitch_config: Optional stitching configuration

    Returns:
        ProcessingResult for stitched output
    """
```

**Example:**

```python
# Stitch 3x3 grid of tiles
tile_paths = [
    "data/raw/tile_001.laz",
    "data/raw/tile_002.laz",
    "data/raw/tile_003.laz",
    # ... more tiles
]

result = processor.process_with_stitching(
    input_paths=tile_paths,
    output_path="data/output/stitched.laz"
)
```

### Properties

```python
@property
def config(self) -> DictConfig:
    """Get current configuration."""

@property
def gpu_available(self) -> bool:
    """Check if GPU is available."""

@property
def stats(self) -> Dict[str, Any]:
    """Get processing statistics."""
```

---

## ProcessingResult

Result object returned by processing methods.

### Class Definition

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ProcessingResult:
    """Results from processing operation."""

    input_path: str           # Input file path
    output_path: str          # Output file path
    num_points: int           # Number of points processed
    num_patches: int          # Number of patches extracted
    elapsed_time: float       # Processing time (seconds)
    memory_peak: float        # Peak memory usage (MB)
    gpu_used: bool           # Whether GPU was used
    features_computed: List[str]  # List of computed features
    metadata: Dict[str, Any] # Additional metadata
    success: bool            # Whether processing succeeded
    error: Optional[str]     # Error message (if failed)
```

### Example Usage

```python
result = processor.process_tile(
    input_path="tile.laz",
    output_path="output.laz"
)

if result.success:
    print(f"✓ Success: {result.num_patches} patches")
    print(f"  Time: {result.elapsed_time:.2f}s")
    print(f"  Memory: {result.memory_peak:.1f} MB")
    print(f"  GPU: {'Yes' if result.gpu_used else 'No'}")
else:
    print(f"✗ Failed: {result.error}")
```

---

## TileStitcher

Handle stitching of multiple LiDAR tiles.

### Class Definition

```python
from ign_lidar.core.stitcher import TileStitcher

class TileStitcher:
    """
    Stitch multiple LiDAR tiles into combined output.
    """

    def __init__(
        self,
        config: DictConfig,
        buffer_size: float = 10.0,
        remove_duplicates: bool = True
    ):
        """
        Initialize tile stitcher.

        Args:
            config: Hydra configuration
            buffer_size: Buffer overlap size (meters)
            remove_duplicates: Remove duplicate points in overlap
        """
```

### Methods

#### `stitch_tiles()`

```python
def stitch_tiles(
    self,
    tile_paths: List[str],
    output_path: str,
    pattern: Optional[str] = None
) -> StitchingResult:
    """
    Stitch multiple tiles together.

    Args:
        tile_paths: List of LAZ file paths to stitch
        output_path: Output path for stitched result
        pattern: Optional grid pattern (e.g., "3x3")

    Returns:
        StitchingResult with statistics
    """
```

**Example:**

```python
from ign_lidar.core.stitcher import TileStitcher

stitcher = TileStitcher(
    config=config,
    buffer_size=10.0,
    remove_duplicates=True
)

result = stitcher.stitch_tiles(
    tile_paths=["tile_001.laz", "tile_002.laz"],
    output_path="stitched.laz",
    pattern="2x1"
)

print(f"Stitched {result.num_tiles} tiles")
print(f"Total points: {result.total_points:,}")
print(f"Duplicates removed: {result.duplicates_removed:,}")
```

#### `detect_neighbors()`

```python
def detect_neighbors(
    self,
    tile_path: str,
    all_tiles: List[str],
    max_distance: float = 1.0
) -> List[str]:
    """
    Detect neighboring tiles for boundary processing.

    Args:
        tile_path: Path to reference tile
        all_tiles: List of all available tiles
        max_distance: Maximum distance to consider as neighbor (meters)

    Returns:
        List of neighboring tile paths
    """
```

---

## BoundaryHandler

Handle boundary-aware feature computation across tile edges.

### Class Definition

```python
from ign_lidar.core.boundary import BoundaryHandler

class BoundaryHandler:
    """
    Handle boundary-aware feature computation.
    """

    def __init__(
        self,
        buffer_size: float = 10.0,
        search_radius: float = 5.0
    ):
        """
        Initialize boundary handler.

        Args:
            buffer_size: Buffer zone size at tile edges (meters)
            search_radius: Radius for neighbor search (meters)
        """
```

### Methods

#### `extract_boundary_buffer()`

```python
def extract_boundary_buffer(
    self,
    points: np.ndarray,
    bounds: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract points in boundary buffer zone.

    Args:
        points: Point cloud array (N, 3+)
        bounds: Tile bounds (xmin, ymin, xmax, ymax)

    Returns:
        Tuple of (interior_points, boundary_points)
    """
```

#### `compute_with_neighbors()`

```python
def compute_with_neighbors(
    self,
    tile_points: np.ndarray,
    neighbor_points: List[np.ndarray],
    feature_func: Callable
) -> np.ndarray:
    """
    Compute features using cross-tile neighbors.

    Args:
        tile_points: Points from current tile
        neighbor_points: Points from neighboring tiles
        feature_func: Function to compute features

    Returns:
        Feature values for tile_points
    """
```

**Example:**

```python
from ign_lidar.core.boundary import BoundaryHandler
from ign_lidar.features.geometric import compute_planarity

handler = BoundaryHandler(
    buffer_size=10.0,
    search_radius=5.0
)

# Load tile and neighbors
tile_pc = load_point_cloud("tile_001.laz")
neighbors = [
    load_point_cloud("tile_002.laz"),
    load_point_cloud("tile_003.laz")
]

# Compute features with boundary handling
features = handler.compute_with_neighbors(
    tile_points=tile_pc,
    neighbor_points=neighbors,
    feature_func=compute_planarity
)
```

---

## PipelineManager

Manage pipeline state and configuration.

### Class Definition

```python
from ign_lidar.core.pipeline import PipelineManager

class PipelineManager:
    """
    Manage pipeline execution and state.
    """

    def __init__(self, config: DictConfig):
        """Initialize pipeline manager."""
```

### Methods

#### `create_processor()`

```python
def create_processor(
    self,
    gpu_enabled: bool = False
) -> LiDARProcessor:
    """
    Create configured processor instance.

    Args:
        gpu_enabled: Enable GPU acceleration

    Returns:
        LiDARProcessor instance
    """
```

#### `validate_config()`

```python
def validate_config(self) -> bool:
    """
    Validate pipeline configuration.

    Returns:
        True if configuration is valid

    Raises:
        ValueError: Invalid configuration
    """
```

---

## Complete Example

### Basic Processing Workflow

```python
from ign_lidar.core.processor import LiDARProcessor
from omegaconf import OmegaConf
from pathlib import Path

# 1. Load configuration
config = OmegaConf.load("config.yaml")

# 2. Create processor
processor = LiDARProcessor(
    config=config,
    gpu_enabled=True,
    verbose=True
)

# 3. Process directory
input_dir = Path("data/raw")
output_dir = Path("data/output")
output_dir.mkdir(exist_ok=True)

results = processor.process_directory(
    input_dir=str(input_dir),
    output_dir=str(output_dir),
    parallel=True,
    max_workers=4
)

# 4. Report results
successful = [r for r in results if r.success]
failed = [r for r in results if not r.success]

print(f"\n{'='*50}")
print(f"Processing Complete")
print(f"{'='*50}")
print(f"Total files: {len(results)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if successful:
    total_points = sum(r.num_points for r in successful)
    total_patches = sum(r.num_patches for r in successful)
    total_time = sum(r.elapsed_time for r in successful)

    print(f"\nStatistics:")
    print(f"  Total points: {total_points:,}")
    print(f"  Total patches: {total_patches:,}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time/tile: {total_time/len(successful):.2f}s")
```

### Advanced: Custom Pipeline

```python
from ign_lidar.core.processor import LiDARProcessor
from ign_lidar.core.boundary import BoundaryHandler
from ign_lidar.preprocessing.enrichment import enrich_point_cloud
from ign_lidar.features.geometric import compute_features

# Custom processing with boundary handling
processor = LiDARProcessor(config)
boundary_handler = BoundaryHandler(buffer_size=10.0)

# Load tiles
tile = load_tile("tile_001.laz")
neighbors = load_neighbors(["tile_002.laz", "tile_003.laz"])

# Enrich with boundary awareness
enriched = enrich_point_cloud(tile, config.preprocessing)

# Compute features using neighbors
features = boundary_handler.compute_with_neighbors(
    tile_points=enriched,
    neighbor_points=neighbors,
    feature_func=lambda pts: compute_features(pts, config.features)
)

# Extract patches
patches = processor.extract_patches(enriched, features)

# Save output
save_patches(patches, "output.laz")
```

---

## Related Documentation

- [Unified Pipeline Guide](../guides/unified-pipeline.md) - Workflow overview
- [Configuration API](configuration.md) - Configuration options
- [Preprocessing Guide](../guides/preprocessing.md) - Preprocessing functions
- [Feature Modes](../features/feature-modes.md) - Feature computation modes

---

## See Also

- [Hydra CLI](../guides/hydra-cli.md) - Command-line usage
- [Tile Stitching](../features/tile-stitching.md) - Multi-tile processing
- [Boundary-Aware Features](../features/boundary-aware.md) - Edge handling
- [GPU Acceleration](../guides/gpu-acceleration.md) - GPU usage
