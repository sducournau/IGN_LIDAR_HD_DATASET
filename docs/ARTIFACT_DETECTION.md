# Artifact Detection & Quality Control

This document describes the artifact detection system for identifying and managing data quality issues in LiDAR point cloud features.

## Overview

The artifact detection system identifies **scan line artifacts** (dash lines) and other spatial patterns that indicate data quality issues. These artifacts commonly appear in geometric features like planarity, curvature, and linearity due to:

- Scan line patterns from airborne LiDAR acquisition
- Sparse point neighborhoods near tile boundaries
- Noise from outlier contamination
- Degenerate features from irregular point distributions

## Key Features

### 1. **Dash Line Detection**

- Detects parallel stripes perpendicular to flight direction
- Visualizes artifacts as 2D heatmaps with overlay lines
- Computes coefficient of variation (CV) metrics

### 2. **Automatic Severity Classification**

- **Low (CV < 0.10)**: No artifacts, good quality
- **Medium (CV < 0.20)**: Minor artifacts, acceptable
- **High (CV < 0.35)**: Significant artifacts, review recommended
- **Severe (CV ≥ 0.35)**: Major artifacts, drop recommended

### 3. **Field Dropping**

- Automatically identifies features to drop based on CV thresholds
- Prevents artifact-contaminated features from affecting model training
- Configurable thresholds for different quality requirements

### 4. **Comprehensive Visualization**

Each artifact report includes:

- **2D Heatmap**: Spatial distribution with dash line overlays
- **Y-Profile**: Feature values perpendicular to flight lines
- **X-Profile**: Feature values parallel to flight lines (reference)
- **Statistics**: CV metrics, severity, and recommendations

## Usage

### Command-Line Interface

#### Analyze a Single File

```bash
python scripts/detect_artifacts.py --input data/tile.laz
```

#### Analyze Specific Features

```bash
python scripts/detect_artifacts.py \
    --input data/tile.laz \
    --features planarity,roof_score,linearity
```

#### Batch Analysis

```bash
python scripts/detect_artifacts.py \
    --input data/tiles/ \
    --batch \
    --output artifacts_report/
```

#### Custom Thresholds

```bash
python scripts/detect_artifacts.py \
    --input data/tile.laz \
    --threshold 0.30 \
    --review-threshold 0.20
```

#### Fast Batch Processing (No Visualizations)

```bash
python scripts/detect_artifacts.py \
    --input data/tiles/ \
    --batch \
    --no-visualize \
    --output reports/
```

### Python API

#### Single File Analysis

```python
from pathlib import Path
from ign_lidar.preprocessing import ArtifactDetector, ArtifactDetectorConfig

# Configure detector
config = ArtifactDetectorConfig()
config.auto_drop_threshold = 0.40
config.review_threshold = 0.25

detector = ArtifactDetector(config)

# Analyze file
laz_file = Path("data/tile.laz")
results = detector.analyze_file(
    laz_file,
    features_to_check=['planarity', 'roof_score'],
    visualize=True,
    output_dir=Path("artifact_reports")
)

# Get fields to drop
drop_list = detector.get_fields_to_drop(results)
print(f"Fields to drop: {drop_list}")
```

#### Batch Analysis

```python
from pathlib import Path
from ign_lidar.preprocessing import ArtifactDetector

detector = ArtifactDetector()

# Find all LAZ files
laz_files = list(Path("data/tiles").glob("*.laz"))

# Batch analyze
results = detector.batch_analyze(
    laz_files,
    features_to_check=None,  # Check all features
    output_dir=Path("batch_reports")
)

# Generate summary
for file_path, file_results in results.items():
    filename = Path(file_path).name
    for feat_name, metrics in file_results.items():
        if metrics.recommended_action == 'drop':
            print(f"DROP: {filename}:{feat_name} (CV={metrics.cv_y:.3f})")
```

#### Custom Artifact Detection

```python
import numpy as np
from ign_lidar.preprocessing import ArtifactDetector

detector = ArtifactDetector()

# Load your data
coords = np.load("coords.npy")  # [N, 3]
feature_values = np.load("planarity.npy")  # [N]

# Detect artifacts
metrics = detector.detect_spatial_artifacts(
    coords,
    feature_values,
    feature_name="planarity"
)

print(f"CV_Y: {metrics.cv_y:.3f}")
print(f"Severity: {metrics.severity}")
print(f"Action: {metrics.recommended_action}")

# Visualize
detector.visualize_artifacts(
    coords,
    feature_values,
    "planarity",
    metrics,
    output_path=Path("planarity_artifacts.png")
)
```

## Configuration

### In YAML Config File

```yaml
preprocess:
  enabled: true

  # Artifact Detection & Field Dropping
  artifact_detection:
    enabled: true
    auto_drop_artifacts: true
    cv_threshold: 0.40
    review_threshold: 0.25
    visualize: true
    output_dir: /path/to/artifacts/reports
    features_to_check:
      - planarity
      - roof_score
      - linearity
      - curvature
      - verticality
      - surface_variation
```

### Python Configuration

```python
from ign_lidar.preprocessing import ArtifactDetectorConfig

config = ArtifactDetectorConfig()

# Thresholds
config.cv_low_threshold = 0.10
config.cv_medium_threshold = 0.20
config.cv_high_threshold = 0.35
config.auto_drop_threshold = 0.40
config.review_threshold = 0.25

# Grid parameters
config.n_bins_x = 25
config.n_bins_y = 50
config.grid_size = 50

# Visualization
config.show_dash_lines = True
config.plot_dpi = 150

# Default features to check
config.default_features = [
    'planarity', 'roof_score', 'linearity',
    'curvature', 'verticality', 'surface_variation'
]
```

## Understanding Metrics

### Coefficient of Variation (CV)

CV measures spatial variability relative to mean value:

```
CV = std(bin_means) / mean(bin_means)
```

- **Low CV**: Feature values are spatially consistent (good)
- **High CV**: Feature values vary significantly across space (artifacts)

### Spatial Directions

- **Y-direction (CV_Y)**: Perpendicular to flight lines - where scan line artifacts appear
- **X-direction (CV_X)**: Parallel to flight lines - reference/baseline

### Severity Classification

```
if CV < 0.10:
    severity = 'low'        # Good quality
    action = 'keep'
elif CV < 0.20:
    severity = 'medium'     # Acceptable
    action = 'keep'
elif CV < 0.35:
    severity = 'high'       # Review suggested
    action = 'review'
else:
    severity = 'severe'     # Drop recommended
    action = 'drop'
```

## Output Files

### Visualizations

For each analyzed feature:

- `{filename}_artifact_{feature}.png`: Comprehensive 4-panel visualization
  - Top-left: 2D heatmap with dash line overlays
  - Top-right: Y-direction profile (dash detection)
  - Bottom-left: X-direction profile (reference)
  - Bottom-right: Statistics and recommendations

### CSV Report (Batch Mode)

`artifact_analysis_report.csv` contains:

- File name
- Feature name
- CV_X, CV_Y, Max_CV
- Mean, Std
- Severity
- Has_Artifacts (Yes/No)
- Recommended_Action (keep/review/drop)

## Integration with Processing Pipeline

The artifact detector can be integrated into the processing pipeline to automatically drop problematic features:

```python
from ign_lidar.core import LiDARProcessor
from ign_lidar.preprocessing import ArtifactDetector

# Create processor with artifact detection config
config = {
    'processor': {...},
    'preprocess': {
        'artifact_detection': {
            'enabled': True,
            'auto_drop_artifacts': True,
            'cv_threshold': 0.40
        }
    }
}

processor = LiDARProcessor(config)

# Process tiles with automatic artifact detection
processor.process_directory(
    input_dir="input_tiles/",
    output_dir="output_tiles/"
)
```

## Best Practices

### 1. **Always Check Before Training**

Run artifact detection on enriched tiles before creating ML datasets to ensure feature quality.

### 2. **Adjust Thresholds Based on Use Case**

- **Strict (0.30)**: For production models requiring highest quality
- **Moderate (0.40)**: For research and experimentation (default)
- **Relaxed (0.50)**: For exploratory analysis with limited data

### 3. **Batch Analysis First**

For large datasets, run batch analysis without visualization first to get a quick overview, then visualize specific problematic files.

### 4. **Review Borderline Cases**

Features with `action='review'` should be manually inspected before deciding to keep or drop.

### 5. **Document Dropped Features**

Keep a record of which features were dropped and why for reproducibility.

## Common Artifacts Detected

### Scan Line Striping

**Pattern**: Parallel dashes perpendicular to flight direction  
**Affected Features**: Planarity, linearity, curvature  
**Cause**: LiDAR scan pattern, point density variations  
**Action**: Drop if CV > 0.40

### Boundary Effects

**Pattern**: Edge regions with different values  
**Affected Features**: All geometric features  
**Cause**: Sparse neighborhoods near tile edges  
**Action**: Use tile stitching to mitigate

### Noise Amplification

**Pattern**: Random high-frequency variations  
**Affected Features**: Curvature, surface variation  
**Cause**: Measurement noise, outliers  
**Action**: Apply preprocessing (SOR/ROR) before feature computation

## Troubleshooting

### No Features Found

**Problem**: "No features found to analyze"  
**Solution**: Ensure LAZ file has computed features (not just raw XYZ)

### All Features Flagged

**Problem**: All features marked as 'drop'  
**Solution**: Check preprocessing settings, may need stronger outlier removal

### Visualizations Too Large

**Problem**: Figure files are very large  
**Solution**: Reduce `--dpi` or `--grid-size` parameters

### Memory Issues

**Problem**: Out of memory during batch processing  
**Solution**: Use `--no-visualize` flag for large batches

## Example Workflow

```bash
# Step 1: Process tiles with preprocessing
ign-lidar-hd process --config config_asprs_preprocessing.yaml

# Step 2: Detect artifacts in enriched tiles
python scripts/detect_artifacts.py \
    --input /mnt/d/ign/preprocessed/asprs/enriched_tiles/ \
    --batch \
    --output /mnt/d/ign/artifacts/asprs/ \
    --threshold 0.40

# Step 3: Review CSV report
cat /mnt/d/ign/artifacts/asprs/artifact_analysis_report.csv

# Step 4: Update config to drop problematic features
# Edit config_asprs_preprocessing.yaml based on report

# Step 5: Reprocess with updated config
ign-lidar-hd process --config config_asprs_preprocessing.yaml
```

## References

- Scan line artifacts: Common in airborne LiDAR due to acquisition pattern
- Coefficient of Variation: Statistical measure of relative variability
- Feature quality: Critical for ML model performance and generalization
