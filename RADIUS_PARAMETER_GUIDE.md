# Radius Parameter Quick Reference

**Added:** October 3, 2025 (v1.6.0+)  
**Purpose:** Eliminate LIDAR scan line artefacts in geometric features

---

## What is the Radius Parameter?

The `radius` parameter controls the search radius (in meters) used when computing geometric features (linearity, planarity, sphericity, etc.). It replaces the fixed k-nearest neighbor approach to prevent "dash line" artefacts.

### Why Use Radius-Based Search?

**Problem (k-NN):**

- Fixed k neighbors capture LIDAR scan line patterns
- Creates "dash lines" in linearity/planarity visualizations
- Ignores actual point cloud density variations

**Solution (Radius):**

- Searches within a fixed spatial radius
- Adapts to local point density
- Captures true surface geometry
- Eliminates artefacts ✅

---

## Usage

### CLI

```bash
# Auto-radius (recommended, default)
ign-lidar-hd enrich \
  --input-dir data/raw \
  --output data/enriched \
  --mode building

# Manual radius (advanced tuning)
ign-lidar-hd enrich \
  --input-dir data/raw \
  --output data/enriched \
  --mode building \
  --radius 1.5
```

### YAML Configuration

```yaml
enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "building"
  k_neighbors: 10 # Used for normals/curvature
  radius: null # null = auto-estimate (recommended)
  # radius: 1.5         # Or specify manually in meters
```

### Python API

```python
from ign_lidar.features import compute_all_features_with_gpu

# Auto-radius
normals, curvature, height, geo_features = compute_all_features_with_gpu(
    points=points,
    classification=classification,
    k=10,                # For normals/curvature
    auto_k=False,
    use_gpu=True,
    radius=None          # Auto-estimate
)

# Manual radius
normals, curvature, height, geo_features = compute_all_features_with_gpu(
    points=points,
    classification=classification,
    k=10,
    auto_k=False,
    use_gpu=True,
    radius=1.5           # 1.5 meters
)
```

---

## Parameter Guidelines

### Auto-Estimation (Recommended)

**When:** Most cases  
**How:** Set `radius=None` or omit parameter  
**Algorithm:**

```python
# Estimates optimal radius based on point density
# Formula: radius = avg_nn_distance * 20
# Typical values: 0.75-1.5m for IGN LIDAR HD
radius = estimate_optimal_radius_for_features(points, 'geometric')
```

### Manual Tuning

| Point Density           | Recommended Radius | Use Case                        |
| ----------------------- | ------------------ | ------------------------------- |
| Very dense (>40 pts/m²) | 2.0m               | Urban areas, detailed buildings |
| Dense (20-40 pts/m²)    | 1.5m               | Suburban, mixed areas           |
| Moderate (10-20 pts/m²) | 1.0m               | Rural, standard LIDAR HD        |
| Sparse (<10 pts/m²)     | 0.5m               | Minimum for stability           |

### Effects of Radius

| Radius                  | Effect on Features              | Best For           |
| ----------------------- | ------------------------------- | ------------------ |
| **Too small** (< 0.5m)  | Captures noise, artefacts       | ❌ Not recommended |
| **Optimal** (0.75-1.5m) | Clean, smooth features          | ✅ Most cases      |
| **Large** (1.5-2.0m)    | Over-smoothed, may miss details | Dense urban areas  |
| **Too large** (> 2.0m)  | Loses fine geometry             | ❌ Not recommended |

---

## Features Affected by Radius

### ✅ Uses Radius (Artefact-Free)

These features use radius-based search:

- **linearity** - 1D structures (edges, cables)
- **planarity** - 2D structures (roofs, walls)
- **sphericity** - 3D structures (vegetation)
- **anisotropy** - Directionality measure
- **roughness** - Surface roughness
- **density** - Local point density

### ✅ Not Affected by Radius

These features use k-NN (independent of radius):

- **normals** (normal_x, normal_y, normal_z)
- **curvature**
- **height_above_ground**

---

## Performance Impact

| Radius Setting   | Processing Time | Artefact-Free? |
| ---------------- | --------------- | -------------- |
| k-NN only (k=50) | 1.0x (baseline) | ❌ No          |
| Auto-radius      | 1.10-1.15x      | ✅ Yes         |
| Manual radius    | 1.10-1.20x      | ✅ Yes         |

**Trade-off:** ~10-15% slower, but scientifically correct results.

---

## Examples

### Example 1: Dense Urban Area

```bash
# Use larger radius for very dense clouds
ign-lidar-hd enrich \
  --input-dir data/paris_urban \
  --output data/enriched \
  --mode building \
  --radius 2.0  # Larger radius for dense data
```

### Example 2: Rural Area (Auto-Radius)

```bash
# Let algorithm estimate optimal radius
ign-lidar-hd enrich \
  --input-dir data/rural \
  --output data/enriched \
  --mode building
# No --radius = auto-estimate (optimal)
```

### Example 3: Pipeline Configuration

```yaml
# config_enrich.yaml
global:
  num_workers: 4

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "building"
  k_neighbors: 10
  radius: 1.5 # Manual radius for consistency
  use_gpu: true
```

```bash
ign-lidar-hd pipeline --config config_enrich.yaml
```

---

## Troubleshooting

### Problem: Still seeing artefacts

**Solution:** Increase radius

```bash
# Try larger radius
ign-lidar-hd enrich \
  --input-dir data/raw \
  --output data/enriched \
  --mode building \
  --radius 2.0  # Increase from auto-estimated value
```

### Problem: Features too smooth, losing details

**Solution:** Decrease radius

```bash
# Try smaller radius
ign-lidar-hd enrich \
  --input-dir data/raw \
  --output data/enriched \
  --mode building \
  --radius 0.75  # Decrease for finer details
```

### Problem: Processing very slow

**Solution:** Check radius isn't too large

```bash
# Optimal range: 0.5-2.0m
# If radius > 2.0m, consider reducing
ign-lidar-hd enrich \
  --input-dir data/raw \
  --output data/enriched \
  --mode building \
  --radius 1.5  # Reduce if was larger
```

---

## Verification

### Check if Radius is Being Used

Look for this message in the log output:

```
Using radius-based search: r=1.23m (avoids scan line artifacts)
```

### Validate Results

1. **Visualize in CloudCompare:**

   - Open enriched LAZ file
   - Display `linearity` or `planarity` scalar field
   - Should see smooth transitions, no dash patterns

2. **Run tests:**

   ```bash
   python tests/test_feature_fixes.py
   ```

3. **Generate audit visualizations:**
   ```bash
   python scripts/analysis/visualize_artefact_audit.py
   ```

---

## Migration from k-NN

### Old Approach (v1.0.0)

```bash
# Used fixed k-neighbors (artefacts present)
ign-lidar-hd enrich \
  --input-dir data/raw \
  --output data/enriched \
  --k-neighbors 50
```

### New Approach (v1.6.0+)

```bash
# Use radius-based search (artefact-free)
ign-lidar-hd enrich \
  --input-dir data/raw \
  --output data/enriched \
  --mode building
# Radius auto-estimated by default
```

**Note:** `k_neighbors` is still used for normals and curvature computation. Only geometric features (linearity, planarity, sphericity) use radius-based search.

---

## Related Documentation

- `ARTEFACT_AUDIT_REPORT.md` - Full technical audit
- `ARTEFACT_AUDIT_SUMMARY.md` - Quick summary
- `CHANGELOG.md` - Version 1.1.0+ release notes
- `tests/test_feature_fixes.py` - Validation tests

---

## Technical Details

### Formula

Geometric features are computed from eigenvalue decomposition of the local covariance matrix:

```python
# For each point, find neighbors within radius
neighbors = tree.query_radius(points, r=radius)

# Compute covariance matrix
cov = (centered.T @ centered) / (len(neighbors) - 1)

# Eigenvalues λ0 >= λ1 >= λ2
eigenvalues = np.linalg.eigvalsh(cov)

# Features (Weinmann et al., Demantké et al.)
linearity = (λ0 - λ1) / (λ0 + λ1 + λ2)
planarity = (λ1 - λ2) / (λ0 + λ1 + λ2)
sphericity = λ2 / (λ0 + λ1 + λ2)
```

### Auto-Estimation Algorithm

```python
def estimate_optimal_radius_for_features(points, feature_type='geometric'):
    # Sample 1000 points
    sample_points = points[np.random.choice(len(points), 1000)]

    # Build KDTree
    tree = KDTree(sample_points)

    # Find average nearest neighbor distance
    distances, _ = tree.query(sample_points, k=10)
    avg_nn_dist = np.median(distances[:, 1:])

    # For geometric features: use 20x average NN distance
    radius = avg_nn_dist * 20.0
    radius = np.clip(radius, 0.5, 2.0)  # Clamp to safe range

    return radius
```

---

**Last Updated:** October 3, 2025  
**Version:** 1.6.0+
