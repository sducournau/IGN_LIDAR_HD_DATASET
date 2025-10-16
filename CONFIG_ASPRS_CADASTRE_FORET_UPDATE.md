# Config Update Summary: config_asprs_cadastre_foret.yaml

**Date:** October 16, 2025  
**Config:** `configs/multiscale/config_asprs_cadastre_foret.yaml`  
**Status:** ✅ UPDATED for GPU + Advanced NDVI Ground Truth

---

## 🎯 **What Was Updated**

### 1. ✅ **GPU OPTIMIZATION**

**Processor Settings:**

```yaml
processor:
  use_gpu: true # ✅ GPU ENABLED for feature computation
  num_workers: 1 # ✅ SINGLE WORKER - GPU works best sequentially
  batch_size: "auto" # Auto-detect optimal batch size for GPU
  pin_memory: true # ✅ Pin memory for faster GPU transfer
```

**Features Settings:**

```yaml
features:
  gpu_batch_size: 2000000 # ✅ INCREASED to 2M points per batch
  use_gpu_chunked: true # ✅ Use GPU chunked processing
```

**Reclassification Settings:**

```yaml
reclassification:
  acceleration_mode: gpu # ✅ GPU ACCELERATION (use 'auto' to fallback)
  gpu_chunk_size: 2000000 # ✅ INCREASED to 2M points per chunk
```

---

### 2. ✅ **ENRICHED-ONLY MODE (NO PATCHES)**

**Confirmed Settings:**

```yaml
processor:
  # ✅ ENRICHED ONLY MODE - NO PATCH GENERATION
  patch_size: null # ❌ NO patches - direct tile enrichment only
  patch_overlap: null # ❌ NO overlap - not generating patches
  num_points: null # ❌ NO sampling - full tiles
  augment: false # ❌ NO augmentation - enrichment mode
  num_augmentations: 0 # ❌ NO augmentations

output:
  processing_mode: enriched_only # ✅ NO PATCH GENERATION - Full tile enrichment only
```

---

### 3. ✅ **ADVANCED NDVI-BASED GROUND TRUTH**

**Enhanced NDVI Thresholds:**

```yaml
ground_truth:
  # ✅ ADVANCED NDVI-BASED VEGETATION CLASSIFICATION
  ndvi_vegetation_threshold: 0.3 # NDVI >= 0.3 = vegetation
  ndvi_low_vegetation_threshold: 0.5 # NDVI >= 0.5 = low veg
  ndvi_high_vegetation_threshold: 0.7 # NDVI >= 0.7 = high veg (trees)

  # ✅ ADVANCED REFINEMENT THRESHOLDS
  ndvi_water_max: 0.0 # NDVI <= 0.0 = water
  ndvi_building_max: 0.2 # NDVI <= 0.2 = building
  ndvi_road_max: 0.15 # NDVI <= 0.15 = road/pavement
  ndvi_bare_soil_max: 0.25 # NDVI <= 0.25 = bare soil

  # ✅ INTELLIGENT DISAMBIGUATION
  use_ndvi_for_building_refinement: true
  use_ndvi_for_road_refinement: true
  use_ndvi_for_ground_refinement: true

  # Roof vegetation detection
  building_roof_ndvi_threshold: 0.4 # NDVI > 0.4 on roofs = vegetation
```

**Enhanced Post-Processing:**

```yaml
post_processing:
  use_ndvi_similarity: true # ✅ NEW: Use NDVI for similarity classification
  ndvi_based_refinement: true # ✅ Use NDVI to classify unclassified points
  ndvi_confidence_threshold: 0.8 # High NDVI confidence threshold

  # Multi-pass refinement with NDVI
  multi_pass_ndvi_refinement: true # ✅ Multiple passes with NDVI checks
  max_ndvi_refinement_passes: 2 # Maximum NDVI refinement iterations
```

**Reclassification NDVI Settings:**

```yaml
reclassification:
  # ✅ NDVI-BASED ADVANCED REFINEMENT
  ndvi_vegetation_threshold: 0.3
  ndvi_road_threshold: 0.15
  ndvi_building_threshold: 0.2 # ✅ NEW: Building roof threshold

  # ✅ ROAD-VEGETATION DISAMBIGUATION
  road_vegetation_height_threshold: 2.0
  use_ndvi_for_road_refinement: true # ✅ NEW: Use NDVI for roads
  ndvi_tree_threshold: 0.6 # ✅ NEW: High confidence trees
```

---

### 4. ✅ **CRITICAL FEATURES ENABLED**

**NDVI Requirements:**

```yaml
features:
  # ✅ SPECTRAL FEATURES - CRITICAL FOR NDVI
  use_rgb: true # ✅ REQUIRED for NDVI computation
  use_infrared: true # ✅ REQUIRED for NDVI computation
  compute_ndvi: true # ✅ REQUIRED for advanced ground truth!

  # ✅ HEIGHT FEATURE - CRITICAL
  compute_height_above_ground: true # ✅ CRITICAL for road filtering & veg classes!
```

---

### 5. ✅ **OUTPUT ENHANCEMENTS**

**Added NDVI to Output:**

```yaml
output:
  extra_dimensions:
    # ... existing dimensions ...
    - name: ndvi
      type: float
      description: "NDVI value (Normalized Difference Vegetation Index)"
      source: computed

  # NEW: Save NDVI statistics
  save_ndvi_statistics: true # ✅ Save NDVI distribution stats
```

---

## 📊 **Key Improvements**

### **Performance:**

- ✅ GPU acceleration: 2M point batches
- ✅ Optimized chunking for GPU processing
- ✅ Single worker for maximum GPU utilization

### **Accuracy:**

- ✅ Advanced NDVI-based classification
- ✅ Multi-threshold NDVI refinement (water, buildings, roads, vegetation)
- ✅ Intelligent disambiguation using NDVI + geometry
- ✅ Roof vegetation detection
- ✅ Multi-pass NDVI refinement

### **Mode:**

- ✅ Enriched-only mode confirmed (NO patches)
- ✅ Full tile processing with all enhancements
- ✅ Direct ground truth integration

---

## 🚀 **Usage**

Run the updated configuration:

```bash
ign-lidar-hd process --config-file configs/multiscale/config_asprs_cadastre_foret.yaml
```

**What it will do:**

1. ✅ Process tiles with GPU acceleration (2M points/batch)
2. ✅ Compute NDVI from RGB + NIR
3. ✅ Apply BD TOPO® ground truth (roads, railways, buildings)
4. ✅ Use NDVI for intelligent refinement:
   - Vegetation vs non-vegetation
   - Road vs vegetation disambiguation
   - Building roof vegetation detection
   - Unclassified point classification
5. ✅ Enrich with cadastre + BD Forêt + RPG data
6. ✅ Output enriched LAZ tiles (NO patches)

---

## 📁 **Output Files**

**Enriched LAZ tiles will include:**

- ASPRS classification (Classes 1-67)
- NDVI values for all points
- Cadastral parcel IDs + cluster IDs
- Forest types + species (where applicable)
- RPG crop codes + categories (where applicable)
- All geometric features (~15 features)

**Statistics files:**

- Class distribution per tile
- NDVI distribution statistics
- Parcel-level metrics
- Forest-level metrics
- Quality reports

---

## ⚙️ **Configuration Summary**

| Setting           | Value         | Purpose                    |
| ----------------- | ------------- | -------------------------- |
| **GPU**           | ✅ Enabled    | 2M point batches           |
| **Workers**       | 1             | Single worker for GPU      |
| **Mode**          | enriched_only | NO patches                 |
| **NDVI**          | ✅ Advanced   | Multi-threshold refinement |
| **Ground Truth**  | ✅ Direct     | BD TOPO® during processing |
| **Height Field**  | ✅ Fixed      | Uses `height_above_ground` |
| **Spatial Index** | ✅ Optimized  | Numpy bbox filtering       |

---

## ✅ **Ready to Process!**

All settings are optimized for:

- 🚀 **GPU performance** (2M point batches)
- 🎯 **Advanced NDVI ground truth** (multi-threshold)
- 📦 **Enriched-only output** (no patches)
- 🗺️ **Multi-source enrichment** (BD TOPO + Cadastre + BD Forêt + RPG)

Your configuration is ready for production use! 🎉
