# ASPRS Classification with BD TOPO & Cadastre - Optimized Configuration

## 🎯 Overview

This configuration provides **high-performance ASPRS LAS 1.4 classification** with integrated ground truth from IGN BD TOPO® and cadastral data, optimized for **RTX 4080** GPUs (or similar hardware).

### Key Features

- ✅ **ASPRS LAS 1.4 compliance** - Standard classification codes
- ✅ **BD TOPO V3 ground truth** - Buildings, roads, water, vegetation
- ✅ **Cadastral integration** - Spatial coherence with parcel boundaries
- ✅ **RTX 4080 optimized** - 30M point batch processing, 90% VRAM usage
- ✅ **STRtree spatial indexing** - 10× faster ground truth classification
- ✅ **NDVI-based refinement** - Improved vegetation classification
- ✅ **Advanced reclassification** - Spectral + geometric + clustering rules

### Performance (RTX 4080, 18.6M point tile)

| Stage                  | Time           | Notes                         |
| ---------------------- | -------------- | ----------------------------- |
| Feature computation    | ~1-2 min       | GPU-accelerated, single batch |
| Ground truth (BD TOPO) | ~2-5 min       | STRtree optimization          |
| Classification         | ~1-2 min       | ASPRS rules + NDVI            |
| I/O (read/write)       | ~2-3 min       | LAZ compression               |
| **Total**              | **~10-15 min** | vs 60-80 min unoptimized      |

**Expected Speedup**: 5-8× faster than default configuration

---

## 📋 Configuration Files

### Primary Configuration

**`config_asprs_bdtopo_cadastre_optimized.yaml`** - Main production config

```yaml
# Inherits from: presets/asprs_rtx4080.yaml
config_version: "5.1.0"
preset_name: "asprs_bdtopo_cadastre_optimized"

# Key settings:
processor:
  lod_level: "ASPRS"
  use_gpu: true
  gpu_batch_size: 30_000_000 # 30M points
  ground_truth_method: "auto" # STRtree on CPU, GPU on CUDA

data_sources:
  bd_topo: { enabled: true, features: { ... } }
  cadastre_enabled: true
  orthophoto_rgb: true
  orthophoto_nir: true
```

### Available Presets (Inheritance)

```
base.yaml
  └─ presets/asprs.yaml
      └─ presets/asprs_rtx4080.yaml
          └─ config_asprs_bdtopo_cadastre_optimized.yaml
```

---

## 🚀 Usage

### Basic Usage

```bash
# Process Versailles tiles
ign-lidar-hd process \
  -c examples/config_asprs_bdtopo_cadastre_optimized.yaml \
  input_dir="/mnt/d/ign/selected_tiles/asprs/tiles" \
  output_dir="/mnt/d/ign/preprocessed_ground_truth_v3"
```

### Override Settings via CLI

```bash
# Disable GPU (use CPU)
ign-lidar-hd process \
  -c examples/config_asprs_bdtopo_cadastre_optimized.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output" \
  processor.use_gpu=false

# Change batch size for smaller GPU
ign-lidar-hd process \
  -c examples/config_asprs_bdtopo_cadastre_optimized.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output" \
  processor.gpu_batch_size=15000000
```

### Batch Processing

```bash
# Process all tiles in directory
find /mnt/d/ign/tiles -name "*.laz" | while read tile; do
  ign-lidar-hd process \
    -c examples/config_asprs_bdtopo_cadastre_optimized.yaml \
    input_dir="$(dirname "$tile")" \
    output_dir="/mnt/d/ign/output_asprs"
done
```

---

## 📊 Output Format

### Enriched LAZ Files

Each output file contains:

- **XYZ coordinates** - Original point positions
- **Classification** - ASPRS LAS 1.4 classes (see below)
- **RGB colors** - Original or fetched from orthophotos
- **NIR channel** - Near-infrared (if available)
- **Extra dimensions**:
  - `NDVI` - Normalized Difference Vegetation Index
  - `height` - Height above ground
  - `planarity` - Planarity score
  - `verticality` - Verticality score
  - `curvature` - Surface curvature

### ASPRS Classification Codes

| Code | Class                  | Source                  |
| ---- | ---------------------- | ----------------------- |
| 1    | Unclassified           | Default                 |
| 2    | Ground                 | Geometric + BD TOPO     |
| 3    | Low Vegetation (<0.5m) | NDVI + height           |
| 4    | Medium Veg (0.5-2m)    | NDVI + height           |
| 5    | High Vegetation (>2m)  | NDVI + height + BD TOPO |
| 6    | Building               | BD TOPO (primary)       |
| 9    | Water                  | BD TOPO                 |
| 11   | Road Surface           | BD TOPO + buffer        |
| 17   | Bridge Deck            | BD TOPO (optional)      |
| 14   | Wire - Conductor       | BD TOPO (optional)      |
| 40   | Parking                | BD TOPO (optional)      |
| 41   | Sports Facility        | BD TOPO (optional)      |
| 42   | Cemetery               | BD TOPO (optional)      |

---

## 🗺️ Data Sources

### BD TOPO V3 (Primary Ground Truth)

**Source**: IGN Géoplateforme WFS  
**URL**: `https://data.geopf.fr/wfs`  
**Layers**:

- `BDTOPO_V3:batiment` - Building footprints
- `BDTOPO_V3:troncon_de_route` - Road segments
- `BDTOPO_V3:surface_hydrographique` - Water bodies
- `BDTOPO_V3:zone_de_vegetation` - Vegetation zones
- `BDTOPO_V3:ouvrage_d_art` - Bridges
- `BDTOPO_V3:ligne_electrique` - Power lines

**Caching**: Automatic caching to `{input_dir}/cache/ground_truth/`

### BD Parcellaire (Cadastre)

**Source**: IGN Géoplateforme WFS  
**Layer**: `CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle`  
**Purpose**: Spatial coherence, parcel-based analysis  
**Caching**: Automatic caching to `{input_dir}/cache/cadastre/`

### Orthophotos (RGB + NIR)

**Source**: IGN Géoplateforme WMTS  
**Layers**:

- `ORTHOIMAGERY.ORTHOPHOTOS` - RGB orthophotos
- `ORTHOIMAGERY.ORTHOPHOTOS.IRC` - Near-infrared

**Usage**: NDVI computation for vegetation classification  
**Caching**: Automatic caching to `{input_dir}/cache/orthophotos/`

---

## ⚙️ Configuration Details

### GPU Optimization (RTX 4080)

```yaml
processor:
  use_gpu: true
  gpu_batch_size: 30_000_000 # 30M points (single chunk for 18M tiles)
  gpu_memory_target: 0.90 # 90% VRAM (14.4GB of 16GB)
  gpu_streams: 8 # 8 concurrent CUDA streams
  vram_limit_gb: 14 # Conservative limit

  # CUDA optimizations
  use_cuda_streams: true
  enable_memory_pooling: true
  enable_pipeline_optimization: true
```

**Memory Breakdown** (30M points):

- Points: ~2.3GB (30M × 3 × 4 bytes)
- KNN indices: ~2.4GB (30M × 20 × 4 bytes)
- Neighbor points: ~7.2GB (30M × 20 × 3 × 4 bytes)
- Features: ~360MB (30M × 3 × 4 bytes)
- **Total**: ~12.3GB (fits in 16GB with headroom)

### Ground Truth Optimization

```yaml
processor:
  ground_truth_method: "auto" # Auto-select optimal method
  ground_truth_chunk_size: 30_000_000 # 30M points
  use_optimized_ground_truth: true # STRtree spatial indexing
```

**Methods**:

- `auto` - Intelligent selection (GPU if available, STRtree otherwise)
- `gpu_chunked` - GPU with memory-efficient chunking
- `strtree` - CPU with STRtree spatial index (10× faster than brute-force)
- `vectorized` - Vectorized CPU operations

### Reclassification Rules

```yaml
processor:
  reclassification:
    enabled: true
    use_geometric_rules: true # Height, planarity, verticality
    use_ndvi_classification: true # NDVI-based vegetation
    use_spectral_rules: true # NIR-based material classification
    use_clustering: true # Spatial clustering (10-100× speedup)

    # Thresholds
    ndvi_vegetation_threshold: 0.3
    ndvi_road_threshold: 0.15
    nir_vegetation_threshold: 0.4
    nir_building_threshold: 0.3

    # Clustering
    spatial_cluster_eps: 0.5 # 50cm clustering radius
    min_cluster_size: 10
```

### Feature Computation

```yaml
features:
  k_neighbors: 20
  search_radius: 1.0

  # GPU batching (RTX 4080)
  neighbor_query_batch_size: 30_000_000 # Single batch = no overhead
  feature_batch_size: 30_000_000 # Single batch = no overhead

  # Features
  compute_normals: true
  compute_curvature: true
  compute_height: true
  compute_geometric: true # Planarity, verticality
  compute_ndvi: true # Requires RGB + NIR
```

---

## 📈 Quality Metrics

### Expected Accuracy (with ground truth)

| Metric             | Value  | Notes                        |
| ------------------ | ------ | ---------------------------- |
| Overall accuracy   | 85-95% | Depends on data quality      |
| Building detection | 90-95% | BD TOPO is highly accurate   |
| Road detection     | 80-90% | Buffer zone affects accuracy |
| Vegetation class   | 85-90% | NDVI improves accuracy       |
| Water detection    | 95-99% | Very reliable                |

### Classification Coverage (typical urban tile)

| Source          | Coverage | Notes              |
| --------------- | -------- | ------------------ |
| BD TOPO         | 60-80%   | Primary source     |
| Geometric rules | 15-25%   | Fill gaps          |
| NDVI refinement | 5-10%    | Improve vegetation |
| Unclassified    | <5%      | Minimal gaps       |

### Class Distribution (typical urban/suburban tile)

- Ground (2): 30-40%
- Vegetation (3-5): 25-35%
- Building (6): 15-25%
- Road (11): 10-15%
- Water (9): 0-5%
- Other: <5%

---

## 🔧 Troubleshooting

### GPU Out of Memory

**Symptom**: CUDA OOM error during processing

**Solutions**:

1. Reduce `gpu_batch_size`:

   ```bash
   processor.gpu_batch_size=20000000  # 20M instead of 30M
   ```

2. Reduce `gpu_memory_target`:

   ```bash
   processor.gpu_memory_target=0.80  # 80% instead of 90%
   ```

3. Reduce `neighbor_query_batch_size`:
   ```bash
   features.neighbor_query_batch_size=20000000
   ```

### Ground Truth Fetching Slow

**Symptom**: WFS queries take >5 min per tile

**Solutions**:

1. Check cache directory exists:

   ```bash
   ls -lah /path/to/input_dir/cache/ground_truth/
   ```

2. Enable global cache:

   ```yaml
   data_sources:
     bd_topo:
       use_global_cache: true
   ground_truth:
     cache_dir: "/path/to/shared/cache"
   ```

3. Pre-fetch ground truth:
   ```bash
   ign-lidar-hd ground-truth fetch \
     --bbox "xmin ymin xmax ymax" \
     --cache-dir "/path/to/cache"
   ```

### Missing NIR Channel

**Symptom**: NDVI computation disabled, warnings about NIR

**Solutions**:

1. Check orthophoto availability:

   ```yaml
   data_sources:
     orthophoto_nir: true
   ```

2. Disable NDVI if NIR unavailable:
   ```yaml
   features:
     compute_ndvi: false
   processor:
     reclassification:
       use_ndvi_classification: false
   ```

### Low Classification Coverage

**Symptom**: >20% unclassified points

**Solutions**:

1. Enable all BD TOPO layers:

   ```yaml
   data_sources:
     bd_topo_bridges: true
     bd_topo_power_lines: true
     bd_topo_sports: true
     bd_topo_cemeteries: true
   ```

2. Lower confidence threshold:

   ```yaml
   processor:
     reclassification:
       min_confidence: 0.6 # Lower from 0.8
   ```

3. Enable geometric fallback:
   ```yaml
   processor:
     reclassification:
       use_geometric_rules: true
   ```

---

## 🔬 Advanced Usage

### Custom NDVI Thresholds

For specific vegetation types or seasons:

```yaml
processor:
  reclassification:
    ndvi_vegetation_threshold: 0.4 # Stricter threshold
    nir_vegetation_threshold: 0.5 # Higher NIR threshold
```

### Road Buffer Tuning

Adjust road classification buffer:

```yaml
data_sources:
  bd_topo:
    road_buffer: 3.0 # Wider buffer for major roads

processor:
  reclassification:
    building_buffer_distance: 1.5 # Smaller buffer
```

### Multi-Tile Parallel Processing

Process multiple tiles in parallel (requires multiple GPUs or CPU workers):

```bash
# GNU Parallel
find /path/to/tiles -name "*.laz" | \
  parallel -j 4 \
    ign-lidar-hd process \
      -c examples/config_asprs_bdtopo_cadastre_optimized.yaml \
      input_dir="{//}" \
      output_dir="/path/to/output"
```

---

## 📚 References

### Documentation

- [Full Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- [Configuration Guide](../docs/guides/CONFIG_GUIDE.md)
- [ASPRS Classification Reference](../docs/docs/reference/asprs-classification.md)
- [Ground Truth Classification](../docs/docs/features/ground-truth-classification.md)

### Data Sources

- [IGN Géoplateforme](https://geoplateforme.ign.fr/)
- [BD TOPO Documentation](https://geoservices.ign.fr/bdtopo)
- [BD Parcellaire Documentation](https://geoservices.ign.fr/bdparcellaire)
- [ASPRS LAS 1.4 Specification](https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities)

### Related Configs

- `presets/asprs.yaml` - Base ASPRS configuration
- `presets/asprs_rtx4080.yaml` - RTX 4080 optimized (parent)
- `presets/asprs_cpu.yaml` - CPU-only multi-worker
- `config_parcel_versailles.yaml` - Experimental parcel classification

---

## 📝 Version History

- **v5.1.0** (2025-10-19) - Initial optimized configuration
  - RTX 4080 optimization (30M batch sizes)
  - BD TOPO + Cadastre integration
  - STRtree spatial indexing
  - Advanced reclassification rules

---

## 📧 Support

For issues, questions, or contributions:

- **GitHub Issues**: [IGN_LIDAR_HD_DATASET/issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- **Documentation**: [sducournau.github.io/IGN_LIDAR_HD_DATASET](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- **Email**: [Contact via GitHub](https://github.com/sducournau)

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) file for details.
