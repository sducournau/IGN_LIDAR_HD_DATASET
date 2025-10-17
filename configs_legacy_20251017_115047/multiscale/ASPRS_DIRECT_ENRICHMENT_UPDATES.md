# ASPRS Preprocessing Configuration - Full Direct Enrichment Mode

**Updated:** October 16, 2025  
**Configuration:** `config_asprs_preprocessing.yaml`

## 🎯 What Changed

This update enables **FULL RECLASSIFICATION DIRECTLY DURING TILE ENRICHMENT** instead of as a post-processing step. All ground truth integration, geometric rules, and refinements are now applied in a single, optimized pass.

---

## ✅ Key Configuration Changes

### 1. **Processor Configuration** (`processor` section)

#### Before:

```yaml
skip_existing: true # Skip already processed files
```

#### After:

```yaml
skip_existing: false # ❌ Force full reprocessing with complete reclassification
apply_reclassification_inline: true # ✅ Apply during enrichment, not after
ground_truth_integration_mode: "direct" # Apply ground truth immediately
```

**Impact:** Forces complete reprocessing of all tiles with full reclassification applied directly during enrichment.

---

### 2. **Reclassification Settings** (`processor.reclassification` section)

#### Before:

```yaml
reclassification:
  enabled: true
  skip_existing: true # Skip already classified files
  gpu_chunk_size: 500000
```

#### After:

```yaml
reclassification:
  enabled: true # ✅ FULL reclassification during enrichment
  skip_existing: false # ❌ Force full processing of all tiles
  gpu_chunk_size: 1000000 # Doubled for better GPU utilization
  chunk_size: 150000 # Increased for direct mode

  # NEW: Direct enrichment optimization
  apply_during_feature_computation: true # Apply immediately after features
  multi_pass_refinement: true # Multiple passes for higher accuracy
  max_refinement_passes: 3 # Up to 3 iterations
  convergence_threshold: 0.01 # Stop when < 1% points change
```

**Impact:**

- All tiles are fully processed (no skipping)
- Larger GPU chunks for better performance
- Multi-pass refinement for maximum accuracy
- Rules applied immediately after feature computation

---

### 3. **Ground Truth Post-Processing** (`ground_truth.post_processing` section)

#### Before:

```yaml
post_processing:
  enabled: true
  ground_truth_building_priority: medium # Standard priority
```

#### After:

```yaml
post_processing:
  enabled: true # ✅ FULL post-processing during enrichment
  ground_truth_building_priority: high # ✅ HIGH priority for direct mode
  aggressive_mode: true # ✅ NEW: Aggressive reclassification
  max_unclassified_ratio: 0.05 # Target: < 5% unclassified
```

**Impact:**

- Ground truth gets highest priority during enrichment
- Aggressive reclassification of ambiguous points
- Target of <5% unclassified points after processing

---

## 🚀 Processing Pipeline Changes

### **Old Pipeline (2-Stage):**

```
1. Tile Enrichment (features only)
   ↓
2. Reclassification Pass (separate step)
```

### **New Pipeline (1-Stage Direct):**

```
1. Tile Enrichment with FULL Reclassification
   ├─ Load tile
   ├─ Compute features
   ├─ Apply ground truth (BD TOPO®)
   ├─ Apply geometric rules
   ├─ Multi-pass refinement (up to 3 iterations)
   ├─ Post-process unclassified
   └─ Save enriched LAZ
```

---

## 📊 Expected Benefits

### **Performance:**

- ✅ **Single-pass processing** - No need for separate reclassification step
- ✅ **Better GPU utilization** - Larger chunks (1M points vs 500K)
- ✅ **Optimized memory usage** - Features and classification in one pass

### **Accuracy:**

- ✅ **Multi-pass refinement** - Up to 3 iterations for convergence
- ✅ **Aggressive mode** - Reclassifies ambiguous points
- ✅ **Higher completion rate** - Target <5% unclassified points

### **Workflow:**

- ✅ **Simplified pipeline** - One command does everything
- ✅ **No post-processing needed** - Tiles are fully classified on output
- ✅ **Immediate results** - Ready for training/analysis

---

## 🎮 Usage

### Run Full Direct Enrichment:

```bash
ign-lidar-hd process --config-file configs/multiscale/config_asprs_preprocessing.yaml
```

### What Happens:

1. **All tiles processed** (skip_existing=false)
2. **Full reclassification applied** during enrichment
3. **Ground truth integrated** directly (BD TOPO®)
4. **Geometric rules applied** immediately after features
5. **Multi-pass refinement** (up to 3 iterations)
6. **Unclassified points minimized** (<5% target)

---

## ⚙️ Advanced Settings

### **Multi-Pass Refinement:**

```yaml
multi_pass_refinement: true
max_refinement_passes: 3
convergence_threshold: 0.01 # Stop when <1% points change
```

### **Aggressive Mode:**

```yaml
aggressive_mode: true # Reclassify ambiguous points
max_unclassified_ratio: 0.05 # Target <5% unclassified
```

### **GPU Optimization:**

```yaml
gpu_chunk_size: 1000000 # 1M points per GPU chunk
chunk_size: 150000 # 150K points per CPU chunk
```

---

## 📝 Configuration Summary

| Setting                                   | Old Value | New Value | Impact                      |
| ----------------------------------------- | --------- | --------- | --------------------------- |
| `processor.skip_existing`                 | `true`    | `false`   | Force full reprocessing     |
| `processor.apply_reclassification_inline` | N/A       | `true`    | Direct mode enabled         |
| `reclassification.skip_existing`          | `true`    | `false`   | Process all tiles           |
| `reclassification.gpu_chunk_size`         | `500000`  | `1000000` | 2x GPU throughput           |
| `reclassification.multi_pass_refinement`  | N/A       | `true`    | Multiple passes             |
| `reclassification.max_refinement_passes`  | N/A       | `3`       | Up to 3 iterations          |
| `ground_truth_building_priority`          | `medium`  | `high`    | Higher priority             |
| `post_processing.aggressive_mode`         | N/A       | `true`    | Aggressive reclassification |
| `post_processing.max_unclassified_ratio`  | N/A       | `0.05`    | <5% unclassified target     |

---

## 🔍 Monitoring Progress

The configuration includes detailed progress tracking:

```yaml
show_progress: true # Real-time progress bars
logging:
  level: INFO
  progress_bar: true
```

Watch for these indicators during processing:

- ✅ "Applying reclassification inline..."
- ✅ "Multi-pass refinement: iteration 1/3"
- ✅ "Convergence achieved at X% change"
- ✅ "Unclassified ratio: X% (target: 5%)"

---

## 🎯 Quality Targets

With full direct enrichment, you should achieve:

- **Building Classification:** >95% accuracy (ground truth + geometric rules)
- **Road/Railway Classification:** >90% coverage (adaptive buffering + height filtering)
- **Vegetation Classification:** >85% accuracy (NDVI + height-based)
- **Unclassified Points:** <5% (aggressive post-processing)

---

## 📚 Related Configurations

- **Ground Truth:** `data_sources.bd_topo.enabled = true`
- **Transport Enhancement:** `transport_enhancement.adaptive_buffering.enabled = true`
- **Quality Metrics:** `transport_enhancement.quality_metrics.enabled = true`

---

## ⚠️ Important Notes

1. **Processing Time:** First run will take longer (no skip_existing), but subsequent runs benefit from cached ground truth data.

2. **Memory Usage:** Larger GPU chunks require more VRAM. Reduce `gpu_chunk_size` if you encounter OOM errors.

3. **Convergence:** If tiles don't converge after 3 passes, check `convergence_threshold` (lower = stricter).

4. **Unclassified Points:** If >5% remain unclassified, consider:
   - Lowering `min_building_planarity` threshold
   - Increasing `building_buffer_distance`
   - Enabling more aggressive NDVI thresholds

---

## 🚦 Status

✅ **Configuration Updated**  
✅ **Ready for Processing**  
⏳ **Waiting for Execution**

Run the command above to start full direct enrichment!
