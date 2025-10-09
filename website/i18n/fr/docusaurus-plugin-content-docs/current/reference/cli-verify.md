---
sidebar_position: 5
title: CLI - Verify
description: Verify features in enriched LAZ files
---

<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->
<!-- Ce fichier est un modèle qui nécessite une traduction manuelle. -->
<!-- Veuillez traduire le contenu ci-dessous en conservant : -->
<!-- - Le frontmatter (métadonnées en haut) -->
<!-- - Les blocs de code (traduire uniquement les commentaires) -->
<!-- - Les liens et chemins de fichiers -->
<!-- - La structure Markdown -->



# Verify Command

The `verify` command validates features in enriched LAZ files, checking RGB, NIR, and geometric features for correctness and anomalies.

## Basic Usage

```bash
# Verify a single file
ign-lidar-hd verify --input enriched/file.laz

# Verify all files in a directory
ign-lidar-hd verify --input-dir enriched/

# Show sample points
ign-lidar-hd verify --input enriched/file.laz --show-samples
```

## Command Options

### Input Options

| Option        | Type | Description                              |
| ------------- | ---- | ---------------------------------------- |
| `--input`     | Path | Single LAZ file to verify                |
| `--input-dir` | Path | Directory containing LAZ files to verify |

:::info
Either `--input` or `--input-dir` must be specified, but not both.
:::

### Output Options

| Option           | Type    | Default | Description                                 |
| ---------------- | ------- | ------- | ------------------------------------------- |
| `--quiet`        | Flag    | False   | Suppress detailed output, show only summary |
| `--show-samples` | Flag    | False   | Display sample points from each file        |
| `--max-files`    | Integer | None    | Maximum number of files to verify           |

## What Gets Verified

### RGB Values

- ✅ Presence of red, green, blue channels
- ✅ Value ranges (0-255 for 8-bit display)
- ✅ Statistical analysis (min, max, mean, std)
- ✅ Unique color combinations
- ⚠️ Warns if too few unique values (fetch failure)
- ⚠️ Warns if majority are default gray (128,128,128)

### NIR (Infrared) Values

- ✅ Presence of NIR channel
- ✅ Value ranges and distribution
- ✅ Unique value count
- ⚠️ Warns if majority have default value (128)

### Geometric Features

- ✅ **Linearity**: shape elongation along principal axis [0-1]
- ✅ **Planarity**: flatness of local neighborhood [0-1]
- ✅ **Sphericity**: 3D compactness [0-1]
- ✅ **Anisotropy**: degree of directional dependence [0-1]
- ✅ **Roughness**: surface irregularity [0-1]

### Quality Checks

- ✅ Values within valid range [0, 1] for normalized features
- ✅ Non-zero point counts
- ✅ Statistical distributions
- ⚠️ Out-of-range value detection
- ⚠️ Missing feature warnings

## Examples

### Basic Verification

```bash
# Verify a single enriched file
ign-lidar-hd verify --input enriched/LHD_FXX_0473_6916.laz
```

**Output:**

```
================================================================================
Analyzing: LHD_FXX_0473_6916.laz
================================================================================
Total points: 6,579,534

1. RGB VALUES CHECK
--------------------------------------------------------------------------------
✓ RGB channels present
  Red:   min= 64, max=255, mean=151.34, std= 39.22
  Green: min= 76, max=255, mean=156.38, std= 27.50
  Blue:  min= 73, max=255, mean=147.91, std= 23.85
  Unique RGB combinations: 81,840
  ✓ RGB values look good

2. NIR (INFRARED) VALUES CHECK
--------------------------------------------------------------------------------
✓ NIR channel present
  Range: 1 - 253
  Mean: 61.31, Std: 45.44
  Unique NIR values: 248
  Most common value: 23 (appears 2.16%)
  ✓ NIR values look good

3. LINEARITY CHECK
--------------------------------------------------------------------------------
✓ Linearity present
  Range: 0.000073 - 0.999326
  Mean: 0.611597, Std: 0.292987
  Non-zero count: 6,579,534 (100.00%)
  ✓ Linearity values in valid range [0, 1]

4. OTHER GEOMETRIC FEATURES CHECK
--------------------------------------------------------------------------------
✓ planarity   : min=0.0002, max=0.4997, mean=0.1899, non-zero=100.0%
✓ sphericity  : min=0.0000, max=0.3001, mean=0.0029, non-zero=100.0%
✓ anisotropy  : min=0.2203, max=1.0000, mean=0.9955, non-zero=100.0%
✗ roughness NOT present

================================================================================
```

### Batch Verification with Samples

```bash
# Verify first 5 files with sample display
ign-lidar-hd verify \
  --input-dir enriched/ \
  --max-files 5 \
  --show-samples
```

Sample output includes random point examples:

```
5. SAMPLE POINTS
--------------------------------------------------------------------------------
Random sample of 10 points:

  Point  2679693: RGB=(141,148,140) NIR= 33 L=0.9950 P=0.0024 S=0.0001
  Point  2155237: RGB=(145,155,147) NIR= 33 L=0.9910 P=0.0042 S=0.0002
  Point  5221160: RGB=(139,149,141) NIR= 31 L=0.3013 P=0.3489 S=0.0003
  ...
```

### Quick Quality Check

```bash
# Quick check of large dataset (first 10 files, summary only)
ign-lidar-hd verify \
  --input-dir enriched/ \
  --max-files 10 \
  --quiet
```

**Summary Output:**

```
================================================================================
VERIFICATION SUMMARY
================================================================================
Files verified: 10

Feature presence:
  ✓ rgb         : 10/10 files (100.0%)
  ✓ nir         : 10/10 files (100.0%)
  ✓ linearity   : 10/10 files (100.0%)
  ✓ planarity   : 10/10 files (100.0%)
  ✓ sphericity  : 10/10 files (100.0%)
  ✓ anisotropy  : 10/10 files (100.0%)
  ⚠️ roughness   : 0/10 files (0.0%)

✓ No warnings detected
================================================================================
```

## Use Cases

### After Enrichment Pipeline

```bash
# Verify enrichment was successful
ign-lidar-hd enrich --input-dir data/ --output enriched/
ign-lidar-hd verify --input-dir enriched/
```

### Quality Assurance

```bash
# Sample verification of large dataset
ign-lidar-hd verify --input-dir /data/tiles --max-files 20 --show-samples
```

### Debugging Issues

```bash
# Detailed analysis of problematic file
ign-lidar-hd verify --input ./problem_tile.laz --show-samples
```

### Automated Testing

```bash
# Quick validation in CI/CD
ign-lidar-hd verify --input-dir ./test_output --max-files 5 --quiet
```

## Python API

You can also use verification programmatically:

```python
from pathlib import Path
from ign_lidar.verifier import verify_laz_files, FeatureVerifier

# Simple verification
results = verify_laz_files(
    input_path=Path("enriched/file.laz"),
    verbose=True,
    show_samples=True
)

# Batch verification
results = verify_laz_files(
    input_dir=Path("enriched/"),
    max_files=10,
    verbose=True
)

# Custom verification with class
verifier = FeatureVerifier(verbose=True, show_samples=True)
result = verifier.verify_file(Path("enriched/file.laz"))

# Print summary
verifier.print_summary(results)
```

## Common Warnings

### RGB Warnings

- **Very few unique RGB values** → RGB fetch failed or no orthophoto available
- **Majority gray (128,128,128)** → Default values, RGB augmentation not applied

### NIR Warnings

- **Majority value is 128** → Default value, NIR augmentation not applied

### Feature Warnings

- **Values exceed 1.0** → Computation error, check feature calculation
- **Values below 0.0** → Computation error, negative eigenvalues
- **All zeros** → Feature not computed or computation failed

## Troubleshooting

### Issue: RGB/NIR channels missing

**Solution:**

```bash
# Re-run enrichment with RGB/NIR augmentation
ign-lidar-hd enrich \
  --input-dir data/ \
  --output enriched/ \
  --add-rgb --rgb-cache-dir cache/rgb \
  --add-infrared --infrared-cache-dir cache/infrared
```

### Issue: Geometric features missing

**Solution:**

```bash
# Re-run enrichment with full mode
ign-lidar-hd enrich \
  --input-dir data/ \
  --output enriched/ \
  --mode full
```

### Issue: Out-of-range feature values

This indicates a computation error. Check:

1. Input data quality
2. Neighborhood parameters (k_neighbors, radius)
3. Point density (too sparse/dense)

Report the issue with sample data if problem persists.

## See Also

- [Enrich Command](./cli-enrich.md) - Feature computation
- [RGB Augmentation](../features/rgb-augmentation.md) - Adding RGB colors
- [Infrared Augmentation](../features/infrared-augmentation.md) - Adding NIR values
- [Pipeline Configuration](../features/pipeline-configuration.md) - Automated workflows
