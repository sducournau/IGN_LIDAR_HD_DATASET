---
sidebar_position: 1
---

# Processor API

Complete API reference for the IGN LiDAR HD processor module.

## Processor Class

The main processor class for handling LiDAR data processing workflows.

### Constructor

```python
from ign_lidar import Processor

processor = Processor(config=None, verbose=True)
```

**Parameters:**

- `config` (Config, optional): Configuration object with processing parameters
- `verbose` (bool): Enable verbose logging output

### Methods

#### process_tile()

Process a single LAS/LAZ tile with feature extraction.

```python
def process_tile(self, input_path: str, output_path: str = None) -> dict:
```

**Parameters:**

- `input_path` (str): Path to input LAS/LAZ file
- `output_path` (str, optional): Path for output file

**Returns:**

- `dict`: Processing results with metrics and status

**Example:**

```python
result = processor.process_tile("input.las", "output_enriched.las")
print(f"Processed {result['points_count']} points")
```

#### batch_process()

Process multiple tiles in batch mode.

```python
def batch_process(self, tile_list: list, output_dir: str) -> list:
```

**Parameters:**

- `tile_list` (list): List of input file paths
- `output_dir` (str): Output directory for processed files

**Returns:**

- `list`: List of processing results for each tile

#### extract_features()

Extract geometric and radiometric features from point cloud.

```python
def extract_features(self, points: numpy.ndarray,
                    feature_types: list = None) -> numpy.ndarray:
```

**Parameters:**

- `points` (numpy.ndarray): Point cloud data
- `feature_types` (list, optional): List of features to extract

**Returns:**

- `numpy.ndarray`: Array with extracted features

**Available feature types:**

- `height_above_ground`: Normalized height
- `intensity_normalized`: Normalized intensity
- `local_density`: Local point density
- `surface_roughness`: Surface roughness measure
- `planarity`: Planarity coefficient
- `sphericity`: Sphericity coefficient

#### classify_points()

Classify points into semantic categories.

```python
def classify_points(self, points: numpy.ndarray,
                   features: numpy.ndarray) -> numpy.ndarray:
```

**Parameters:**

- `points` (numpy.ndarray): Point cloud data
- `features` (numpy.ndarray): Extracted features

**Returns:**

- `numpy.ndarray`: Classification labels

## Configuration Options

### Processing Parameters

```python
from ign_lidar import Config

config = Config(
    chunk_size=1000000,
    overlap_size=50,
    classification_model="random_forest",
    feature_radius=2.0,
    min_points_per_class=100
)
```

### Performance Tuning

```python
config = Config(
    use_gpu=True,
    max_memory_gb=16,
    num_workers=8,
    cache_features=True
)
```

## Error Handling

The processor handles common errors gracefully:

```python
try:
    result = processor.process_tile("input.las")
except FileNotFoundError:
    print("Input file not found")
except MemoryError:
    print("Insufficient memory for processing")
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

## Performance Tips

1. **Batch Processing**: Process multiple tiles together for better efficiency
2. **Memory Management**: Set appropriate chunk sizes for available RAM
3. **GPU Acceleration**: Enable GPU processing for large datasets
4. **Feature Caching**: Cache computed features for repeated use
