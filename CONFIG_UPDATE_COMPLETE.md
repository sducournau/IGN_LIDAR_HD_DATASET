# Configuration Files Updated for v1.7.0 ✅

**Date:** October 4, 2025  
**Version:** 1.7.0  
**Summary:** All configuration files updated with preprocessing parameters

---

## 📝 Files Updated

### 1. Version Numbers Updated

| File                           | Old Version | New Version | Status |
| ------------------------------ | ----------- | ----------- | ------ |
| `pyproject.toml`               | 1.6.5       | **1.7.0**   | ✅     |
| `ign_lidar/__init__.py`        | 1.6.4       | **1.7.0**   | ✅     |
| `README.md`                    | 1.6.5       | **1.7.0**   | ✅     |
| `website/docs/intro.md`        | 1.6.4       | **1.7.0**   | ✅     |
| `website/i18n/fr/.../intro.md` | 1.6.4       | **1.7.0**   | ✅     |

### 2. Configuration Files Updated

#### `config_examples/pipeline_enrich.yaml` ✅

**Added preprocessing parameters:**

```yaml
# 🆕 v1.7.0: Point cloud preprocessing for artifact mitigation
preprocess: false

# Statistical Outlier Removal (SOR) parameters
sor_k: 12 # Number of nearest neighbors
sor_std_multiplier: 2.0 # Standard deviation multiplier

# Radius Outlier Removal (ROR) parameters
ror_radius: 1.0 # Search radius in meters
ror_min_neighbors: 4 # Minimum neighbors required

# Voxel downsampling (optional)
# voxel_size: 0.5            # Voxel size in meters (uncomment to enable)

# Preprocessing presets (uncomment one to use):
# Conservative (preserve details):
#   sor_k: 15, sor_std_multiplier: 3.0, ror_radius: 1.5, ror_min_neighbors: 3
# Standard (balanced):
#   sor_k: 12, sor_std_multiplier: 2.0, ror_radius: 1.0, ror_min_neighbors: 4
# Aggressive (maximum artifact removal):
#   sor_k: 10, sor_std_multiplier: 1.5, ror_radius: 0.8, ror_min_neighbors: 5
# Memory-optimized (large datasets):
#   voxel_size: 0.4, sor_k: 10, sor_std_multiplier: 2.0
```

#### `config_examples/pipeline_full.yaml` ✅

**Added preprocessing parameters to enrich stage:**

```yaml
# 🆕 v1.7.0: Point cloud preprocessing for artifact mitigation
preprocess: false

# Statistical Outlier Removal (SOR) parameters
sor_k: 12 # Number of nearest neighbors
sor_std_multiplier: 2.0 # Standard deviation multiplier

# Radius Outlier Removal (ROR) parameters
ror_radius: 1.0 # Search radius in meters
ror_min_neighbors: 4 # Minimum neighbors required

# Voxel downsampling (optional) - uncomment to enable
# voxel_size: 0.5            # Voxel size in meters
```

#### `config_examples/pipeline_patch.yaml` ✅

**No changes needed** - Preprocessing happens during enrich stage, not patch stage.

### 3. CHANGELOG.md Updated ✅

**Added complete v1.7.0 release notes:**

- Point Cloud Preprocessing Pipeline section
- CLI Preprocessing Integration details
- Processor Integration documentation
- Comprehensive Documentation updates (English & French)
- Performance impact metrics
- Validation status

### 4. Documentation Updated ✅

**English Documentation:**

- `website/docs/intro.md`: Updated to v1.7.0 with preprocessing highlights
- Latest Release section updated with preprocessing features
- Code examples added
- Links to preprocessing guide

**French Documentation:**

- `website/i18n/fr/.../intro.md`: Already updated in previous session
- Complete French translation maintained

---

## 🎯 Configuration Parameters Summary

### New Preprocessing Parameters Available

| Parameter            | Type    | Default | Description                        |
| -------------------- | ------- | ------- | ---------------------------------- |
| `preprocess`         | boolean | false   | Enable preprocessing pipeline      |
| `sor_k`              | integer | 12      | SOR: Number of nearest neighbors   |
| `sor_std_multiplier` | float   | 2.0     | SOR: Standard deviation multiplier |
| `ror_radius`         | float   | 1.0     | ROR: Search radius in meters       |
| `ror_min_neighbors`  | integer | 4       | ROR: Minimum neighbors required    |
| `voxel_size`         | float   | None    | Voxel size in meters (optional)    |

### Recommended Presets

1. **Conservative (High-Quality Data)**

   - `sor_k: 15, sor_std_multiplier: 3.0, ror_radius: 1.5, ror_min_neighbors: 3`
   - Minimal point removal, preserves fine details

2. **Standard (General Purpose)**

   - `sor_k: 12, sor_std_multiplier: 2.0, ror_radius: 1.0, ror_min_neighbors: 4`
   - Balanced quality/speed, default parameters

3. **Aggressive (Noisy Data)**

   - `sor_k: 10, sor_std_multiplier: 1.5, ror_radius: 0.8, ror_min_neighbors: 5`
   - Maximum artifact removal, may simplify geometry

4. **Memory-Optimized (Large Datasets)**

   - `voxel_size: 0.4, sor_k: 10, sor_std_multiplier: 2.0`
   - Reduces memory usage significantly

5. **Urban (High-Density Areas)**
   - `sor_k: 12, ror_radius: 0.8, ror_min_neighbors: 5, voxel_size: 0.3`
   - Optimized for dense urban environments

---

## 📊 Impact on Users

### Backward Compatibility ✅

- **Default behavior unchanged**: `preprocess: false` by default
- **No breaking changes**: Existing configurations work without modification
- **Opt-in feature**: Users must explicitly enable preprocessing

### Configuration File Usage

**Before v1.7.0:**

```yaml
enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "building"
  # ... other parameters
```

**After v1.7.0 (Optional):**

```yaml
enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  mode: "building"

  # NEW: Enable preprocessing
  preprocess: true
  sor_k: 12
  sor_std_multiplier: 2.0
  ror_radius: 1.0
  ror_min_neighbors: 4
  # voxel_size: 0.5  # Optional
```

### CLI Usage

**Before v1.7.0:**

```bash
ign-lidar-hd enrich --input-dir data/ --output output/ --mode building
```

**After v1.7.0 (Optional):**

```bash
# With default preprocessing
ign-lidar-hd enrich --input-dir data/ --output output/ --mode building --preprocess

# With custom parameters
ign-lidar-hd enrich --input-dir data/ --output output/ --mode building \
  --preprocess --sor-k 15 --sor-std 3.0 --ror-radius 1.5 --ror-neighbors 3
```

---

## ✅ Validation Checklist

- [x] `pyproject.toml` version updated to 1.7.0
- [x] `ign_lidar/__init__.py` version updated to 1.7.0
- [x] `README.md` version updated to 1.7.0
- [x] `CHANGELOG.md` updated with v1.7.0 release notes
- [x] `config_examples/pipeline_enrich.yaml` updated with preprocessing parameters
- [x] `config_examples/pipeline_full.yaml` updated with preprocessing parameters
- [x] `config_examples/pipeline_patch.yaml` verified (no changes needed)
- [x] `website/docs/intro.md` updated to v1.7.0
- [x] `website/i18n/fr/.../intro.md` already updated to v1.7.0
- [x] All preprocessing parameters documented in config files
- [x] Recommended presets included in comments
- [x] Backward compatibility maintained

---

## 🚀 Ready for Release

All configuration files are now updated and ready for v1.7.0 release:

1. ✅ Version numbers consistent across all files (1.7.0)
2. ✅ Configuration files include all new preprocessing parameters
3. ✅ Presets documented with clear examples
4. ✅ CHANGELOG.md comprehensive release notes
5. ✅ Documentation updated in English and French
6. ✅ Backward compatibility maintained
7. ✅ All files staged for commit

**Next Steps:**

1. Commit all changes
2. Create git tag v1.7.0
3. Push to repository
4. Build and publish to PyPI

---

**Status:** ✅ **ALL CONFIGURATION FILES UPDATED - READY FOR RELEASE**

**Date:** October 4, 2025  
**Version:** 1.7.0
