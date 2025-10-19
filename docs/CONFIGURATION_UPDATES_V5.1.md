# Configuration Updates for V5.1 Enhancements

## 📋 Summary

All configuration files have been updated to support the new **V5.1 enhancements**:

1. **Building clustering** with centroid attraction and wall buffers
2. **Plane detection** (horizontal/vertical/inclined)
3. **Roof type classification** (flat/gable/hip/complex)
4. **Architectural element detection** (chimneys, dormers, balconies, parapets)

---

## ✅ Updated Configuration Files

### 1. Preset Configurations (Core)

#### `ign_lidar/configs/presets/lod2.yaml`

**Added sections:**

- `processor.building_clustering` - Building clustering configuration
- `processor.plane_detection` - Plane detection (basic roof detection)
  - Horizontal/vertical/inclined plane parameters
  - Roof type classification enabled
  - Architectural elements disabled (LOD2 level)

**Key settings:**

```yaml
building_clustering:
  enabled: true
  wall_buffer: 0.3 # Additional buffer for walls
  detect_near_vertical_walls: true

plane_detection:
  enabled: true
  horizontal_angle_max: 10.0 # ≤10° = horizontal
  vertical_angle_min: 75.0 # ≥75° = vertical
  inclined_angle_min: 15.0 # 15-70° = inclined
  classify_roof_types: true
  detect_architectural_elements: false # LOD2 = basic
```

---

#### `ign_lidar/configs/presets/lod3.yaml`

**Added sections:**

- `processor.building_clustering` - Building clustering configuration
- `processor.plane_detection` - **Enhanced** plane detection (LOD3 detail)
  - Stricter thresholds for higher precision
  - Architectural elements **enabled**
  - Fine-grained element parameters

**Key settings:**

```yaml
building_clustering:
  enabled: true
  wall_buffer: 0.3
  detect_near_vertical_walls: true

plane_detection:
  enabled: true
  horizontal_angle_max: 8.0 # Stricter (LOD3)
  vertical_angle_min: 78.0 # Stricter (LOD3)
  inclined_planarity_min: 0.75 # Higher quality

  # LOD3 specific
  detect_architectural_elements: true # ✅ ENABLED
  balcony_max_area: 20.0
  chimney_min_height: 8.0
  dormer_min_protrusion: 1.0
  parapet_max_height: 2.0
```

---

#### `ign_lidar/configs/presets/asprs.yaml`

**Added sections:**

- `processor.building_clustering` - Building clustering for ASPRS

**Key settings:**

```yaml
building_clustering:
  enabled: true
  use_centroid_attraction: true
  polygon_buffer: 0.5
  wall_buffer: 0.3
  detect_near_vertical_walls: true
```

**Note:** Plane detection not added to ASPRS (not relevant for standard land cover classification)

---

#### `ign_lidar/configs/presets/asprs_rtx4080.yaml`

**Added sections:**

- `processor.building_clustering` - Building clustering for RTX 4080

**Same settings as asprs.yaml** (building clustering only)

---

### 2. Example Configurations

#### `examples/config_asprs_bdtopo_cadastre_optimized.yaml`

**Added sections:**

- `processor.building_clustering` - Building clustering configuration

**Integration:** Complements existing BD TOPO ground truth with building-level clustering

---

#### `examples/config_plane_detection_lod3.yaml` ⭐ **NEW FILE**

**Purpose:** Comprehensive configuration showcasing all plane detection features

**Complete feature set:**

```yaml
processor:
  building_clustering:
    enabled: true
    wall_buffer: 0.3 # Total: 0.5 + 0.3 = 0.8m

  plane_detection:
    enabled: true

    # Horizontal (toits plats, terrasses)
    horizontal_angle_max: 8.0
    horizontal_planarity_min: 0.80

    # Vertical (murs, façades)
    vertical_angle_min: 78.0
    vertical_planarity_min: 0.70

    # Inclined (toits en pente)
    inclined_angle_min: 12.0
    inclined_angle_max: 75.0
    inclined_planarity_min: 0.75

    # Roof classification
    classify_roof_types: true

    # Architectural elements (LOD3)
    detect_architectural_elements: true
    balcony_max_area: 20.0
    chimney_min_height: 8.0
    dormer_min_protrusion: 1.0
    parapet_height_range: [0.5, 2.0]
```

**Use cases:**

- Architectural heritage documentation
- 3D building reconstruction (LOD3)
- Solar panel placement analysis
- Facade orientation studies

---

## 🔧 Configuration Parameters Reference

### Building Clustering Parameters

| Parameter                    | Type  | Default | Description                              |
| ---------------------------- | ----- | ------- | ---------------------------------------- |
| `enabled`                    | bool  | true    | Enable building clustering               |
| `use_centroid_attraction`    | bool  | true    | Use centroids for ambiguous points       |
| `attraction_radius`          | float | 5.0     | Max distance for centroid attraction (m) |
| `min_points_per_building`    | int   | 10      | Min points per building cluster          |
| `adjust_polygons`            | bool  | true    | Adjust polygons to match point cloud     |
| `polygon_buffer`             | float | 0.5     | Base buffer for polygons (m)             |
| `wall_buffer`                | float | 0.3     | Additional buffer for walls (m)          |
| `detect_near_vertical_walls` | bool  | true    | Enable wall detection                    |

**Total effective buffer:** `polygon_buffer + wall_buffer` = **0.8m**

---

### Plane Detection Parameters

#### Horizontal Planes (Toits Plats, Terrasses)

| Parameter                  | Type  | LOD2 | LOD3 | Description                   |
| -------------------------- | ----- | ---- | ---- | ----------------------------- |
| `horizontal_angle_max`     | float | 10.0 | 8.0  | Max angle from horizontal (°) |
| `horizontal_planarity_min` | float | 0.75 | 0.80 | Min planarity (0-1)           |

**Detects:** Flat roofs, terraces, parking decks, balconies

---

#### Vertical Planes (Murs, Façades)

| Parameter                | Type  | LOD2 | LOD3 | Description                   |
| ------------------------ | ----- | ---- | ---- | ----------------------------- |
| `vertical_angle_min`     | float | 75.0 | 78.0 | Min angle from horizontal (°) |
| `vertical_planarity_min` | float | 0.65 | 0.70 | Min planarity (0-1)           |

**Detects:** Walls, facades, parapets, gables

---

#### Inclined Planes (Toits en Pente)

| Parameter                | Type  | LOD2 | LOD3 | Description                   |
| ------------------------ | ----- | ---- | ---- | ----------------------------- |
| `inclined_angle_min`     | float | 15.0 | 12.0 | Min angle from horizontal (°) |
| `inclined_angle_max`     | float | 70.0 | 75.0 | Max angle from horizontal (°) |
| `inclined_planarity_min` | float | 0.70 | 0.75 | Min planarity (0-1)           |

**Detects:** Gable roofs, hip roofs, shed roofs, slopes

---

#### General Parameters

| Parameter               | Type  | LOD2 | LOD3 | Description                 |
| ----------------------- | ----- | ---- | ---- | --------------------------- |
| `min_points_per_plane`  | int   | 50   | 30   | Min points to form plane    |
| `max_plane_distance`    | float | 0.15 | 0.10 | Max distance from plane (m) |
| `use_spatial_coherence` | bool  | true | true | Group spatially connected   |
| `classify_roof_types`   | bool  | true | true | Enable roof classification  |

---

#### Architectural Elements (LOD3 Only)

| Parameter                       | Type  | Default                    | Description                   |
| ------------------------------- | ----- | -------------------------- | ----------------------------- |
| `detect_architectural_elements` | bool  | false (LOD2) / true (LOD3) | Enable element detection      |
| `balcony_max_area`              | float | 20.0                       | Max area for balconies (m²)   |
| `chimney_max_area`              | float | 10.0                       | Max area for chimneys (m²)    |
| `chimney_min_height`            | float | 8.0                        | Min height for chimneys (m)   |
| `dormer_min_protrusion`         | float | 1.0                        | Min protrusion above roof (m) |
| `parapet_max_height`            | float | 2.0                        | Max height for parapets (m)   |

---

## 📖 Usage Examples

### Example 1: LOD2 Building Detection with Plane Detection

```bash
ign-lidar-hd process \
  --preset lod2 \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Features enabled:**

- Building clustering with wall buffers (0.8m total)
- Basic plane detection (horizontal/vertical/inclined)
- Roof type classification
- **No** architectural element detection

---

### Example 2: LOD3 Architectural Analysis

```bash
ign-lidar-hd process \
  --preset lod3 \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Features enabled:**

- Enhanced building clustering
- Strict plane detection thresholds
- Roof type classification
- **Full** architectural element detection (chimneys, dormers, balconies, parapets)

---

### Example 3: Dedicated Plane Detection Configuration

```bash
ign-lidar-hd process \
  -c examples/config_plane_detection_lod3.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Features enabled:**

- All LOD3 features
- Optimized plane detection parameters
- Comprehensive architectural element detection
- Detailed statistics and metadata output

---

### Example 4: ASPRS with Building Clustering

```bash
ign-lidar-hd process \
  --preset asprs_rtx4080 \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Features enabled:**

- ASPRS LAS 1.4 classification
- Building clustering for improved building detection
- BD TOPO ground truth integration
- **No** plane detection (not relevant for ASPRS)

---

## 🎯 Configuration Selection Guide

| Use Case                    | Recommended Config                 | Plane Detection | Elements | Speed     |
| --------------------------- | ---------------------------------- | --------------- | -------- | --------- |
| **Basic building modeling** | `lod2.yaml`                        | ✅ Basic        | ❌       | Fast      |
| **Detailed architecture**   | `lod3.yaml`                        | ✅ Enhanced     | ✅       | Medium    |
| **Heritage documentation**  | `config_plane_detection_lod3.yaml` | ✅ Optimized    | ✅       | Medium    |
| **Standard classification** | `asprs.yaml`                       | ❌              | ❌       | Fast      |
| **ASPRS + buildings**       | `asprs_rtx4080.yaml`               | ❌              | ❌       | Very Fast |

---

## 🔍 Verification

To verify configurations are loaded correctly:

```bash
# Check LOD2 config
ign-lidar-hd config show --preset lod2 | grep -A 10 "building_clustering"

# Check LOD3 config
ign-lidar-hd config show --preset lod3 | grep -A 20 "plane_detection"

# Validate plane detection config
ign-lidar-hd config validate -c examples/config_plane_detection_lod3.yaml
```

---

## 📊 Expected Performance Impact

| Configuration                   | Processing Time | Memory | VRAM | Quality |
| ------------------------------- | --------------- | ------ | ---- | ------- |
| **LOD2** (no plane detection)   | Baseline        | 8GB    | 6GB  | Good    |
| **LOD2** (with plane detection) | +15-25%         | 10GB   | 8GB  | Better  |
| **LOD3** (full features)        | +30-50%         | 12GB   | 10GB | Best    |
| **Plane Detection LOD3**        | +35-55%         | 14GB   | 12GB | Best+   |

**Note:** Times relative to base LOD2 processing

---

## ✅ Migration Checklist

If you have **custom configurations**, add these sections:

- [ ] Add `processor.building_clustering` section
- [ ] Add `processor.plane_detection` section (LOD2/LOD3 only)
- [ ] Set `wall_buffer: 0.3` for wall detection
- [ ] Enable `detect_near_vertical_walls: true`
- [ ] Configure architectural element parameters (LOD3 only)
- [ ] Test with small tile before batch processing

---

## 📚 Documentation References

- **Plane Detection Guide**: `docs/PLANE_DETECTION_GUIDE.md`
- **Wall Detection Guide**: `docs/WALL_DETECTION_GUIDE.md`
- **Classification Enhancements**: `docs/CLASSIFICATION_ENHANCEMENTS_V2.md`
- **Quick Start**: `QUICK_START_ENHANCEMENTS.md`
- **Example Scripts**: `examples/demo_wall_detection.py`

---

**Version:** 5.1.0  
**Date:** October 19, 2025  
**Status:** ✅ All configurations updated and tested
