# Configuration Simplification - Implementation Plan

**Status**: 🚧 In Progress  
**Priority**: 🔴 Critical (Blocks production use)  
**Timeline**: 8 hours total implementation  
**Impact**: 97% config size reduction, immediate validation, zero-config defaults

---

## 🎯 Goals

1. **Fix Critical Bug**: Add missing `preprocess`/`stitching` keys (✅ DONE)
2. **Simplify Configs**: Reduce 650-line configs to 20-line presets
3. **Add Validation**: Catch errors at load time, not runtime
4. **Smart Defaults**: Enable zero-config usage for 80% of users
5. **Maintainability**: Single source of truth for all defaults

---

## 📋 Implementation Phases

### Phase 1: Emergency Fix (✅ COMPLETED - 15 min)

**Status**: ✅ Done

**What was fixed:**

- Added missing `preprocess` and `stitching` sections to `config_asprs_gpu_16gb.yaml`
- Immediate crash resolved

**Files modified:**

- `examples/config_asprs_gpu_16gb.yaml`

**Next Issue**: Need to apply to all 40 example configs

---

### Phase 2: Apply Fix to All Configs (⏳ NEXT - 1 hour)

**Priority**: 🔴 Critical  
**Estimated Time**: 1 hour

#### Tasks

1. **Create Patch Script**

   ```bash
   # scripts/patch_configs.py
   # Auto-add preprocess/stitching to all example configs
   ```

2. **Apply to All Configs**

   - Scan `examples/*.yaml` for missing sections
   - Add standardized `preprocess` and `stitching` blocks
   - Preserve existing formatting and comments

3. **Verify**
   - Test load all configs
   - Ensure no validation errors
   - Create backup before patching

#### Files to Modify (~40 files)

```
examples/
├── config_asprs_bdtopo_cadastre_*.yaml (8 files)
├── config_architectural_*.yaml (3 files)
├── config_versailles_*.yaml (3 files)
├── config_*.yaml (26+ other files)
└── ...
```

#### Acceptance Criteria

- [ ] All example configs load without "Missing key" errors
- [ ] All configs validated successfully
- [ ] Backup created before modifications
- [ ] Git commit with clear message

---

### Phase 3: Create Base Complete Config (⏳ TODO - 2 hours)

**Priority**: 🟡 High  
**Estimated Time**: 2 hours

#### Tasks

1. **Consolidate All Defaults**

   - Merge `base.yaml` + all `base/*.yaml` files
   - Add all required sections with smart defaults
   - Document every parameter inline

2. **Create Single Source of Truth**

   ```yaml
   # ign_lidar/configs/base_complete.yaml
   # ~300 lines with full documentation
   # ALL required keys present
   # Smart defaults for 80% of use cases
   ```

3. **Define Complete Schema**
   - processor (with preprocess, stitching)
   - features
   - data_sources
   - ground_truth
   - output
   - variable_object_filtering
   - logging
   - optimizations
   - validation
   - hardware

#### File Structure

```
ign_lidar/configs/
├── base_complete.yaml (NEW - 300 lines, fully documented)
├── base.yaml (keep for compatibility)
└── base/
    ├── processor.yaml (keep for compatibility)
    ├── features.yaml (keep for compatibility)
    └── ...
```

#### Acceptance Criteria

- [ ] `base_complete.yaml` contains ALL required keys
- [ ] Loads without any validation errors
- [ ] Works as standalone config (with input_dir/output_dir only)
- [ ] Fully documented with inline comments
- [ ] Tested with real data

---

### Phase 4: Create Hardware Profiles (⏳ TODO - 1 hour)

**Priority**: 🟢 Medium  
**Estimated Time**: 1 hour

#### Tasks

1. **Create Profile Directory**

   ```
   ign_lidar/configs/profiles/
   ├── gpu_rtx4090.yaml (24GB VRAM)
   ├── gpu_rtx4080.yaml (16GB VRAM)
   ├── gpu_rtx3080.yaml (12GB VRAM)
   ├── gpu_rtx3060.yaml (8GB VRAM)
   ├── cpu_high_end.yaml (32+ cores, 64GB+ RAM)
   └── cpu_standard.yaml (8-16 cores, 32GB RAM)
   ```

2. **Define Each Profile** (~15-30 lines each)

   ```yaml
   defaults:
     - /base_complete

   config_name: "gpu_rtx4090"

   processor:
     gpu_batch_size: 16_000_000
     gpu_memory_target: 0.90
     gpu_streams: 8
     vram_limit_gb: 22

   features:
     gpu_batch_size: 12_000_000
   ```

3. **Add Auto-Detection Script**
   ```python
   # ign_lidar/config/hardware_detector.py
   def detect_optimal_profile() -> str:
       """Auto-detect best hardware profile."""
   ```

#### Profiles to Create

| Profile      | GPU      | VRAM | Batch Size | Workers |
| ------------ | -------- | ---- | ---------- | ------- |
| gpu_rtx4090  | RTX 4090 | 24GB | 16M        | 1       |
| gpu_rtx4080  | RTX 4080 | 16GB | 8M         | 1       |
| gpu_rtx3080  | RTX 3080 | 12GB | 6M         | 1       |
| gpu_rtx3060  | RTX 3060 | 8GB  | 4M         | 1       |
| cpu_high_end | None     | N/A  | 1M         | 8       |
| cpu_standard | None     | N/A  | 1M         | 4       |

#### Acceptance Criteria

- [ ] All profiles tested on target hardware
- [ ] Auto-detection works correctly
- [ ] Profiles load and merge with base_complete
- [ ] CLI can select profiles by name

---

### Phase 5: Create Task Presets (⏳ TODO - 2 hours)

**Priority**: 🟢 Medium  
**Estimated Time**: 2 hours

#### Tasks

1. **Create Preset Directory**

   ```
   ign_lidar/configs/presets/
   ├── asprs_classification.yaml
   ├── asprs_classification_gpu.yaml
   ├── lod2_buildings.yaml
   ├── lod3_architecture.yaml
   ├── fast_preview.yaml
   ├── high_quality.yaml
   ├── building_fusion.yaml
   └── parcel_classification.yaml
   ```

2. **Define Each Preset** (~20-50 lines each)

   ```yaml
   defaults:
     - /base_complete
     - /profiles/gpu_rtx4080 # Can override with CLI

   config_name: "asprs_classification_gpu"

   processor:
     lod_level: "ASPRS"

   features:
     mode: "asprs_classes"
     k_neighbors: 60

   data_sources:
     bd_topo:
       enabled: true
     cadastre:
       enabled: true
   ```

3. **Document Use Cases**
   - When to use each preset
   - Expected performance
   - Hardware requirements
   - Output characteristics

#### Presets to Create

| Preset                   | Purpose                          | Hardware  | Time/Tile |
| ------------------------ | -------------------------------- | --------- | --------- |
| asprs_classification_gpu | Standard ASPRS with GPU          | GPU 8GB+  | 8-14 min  |
| asprs_classification_cpu | Standard ASPRS with CPU          | CPU 32GB+ | 45-60 min |
| fast_preview             | Quick preview (minimal features) | Any       | 2-4 min   |
| high_quality             | Maximum quality (all features)   | GPU 16GB+ | 20-30 min |
| lod2_buildings           | LOD2 building analysis           | GPU 12GB+ | 12-18 min |
| lod3_architecture        | LOD3 detailed architecture       | GPU 16GB+ | 25-35 min |

#### Acceptance Criteria

- [ ] All presets tested on real data
- [ ] Performance metrics documented
- [ ] Use cases clearly defined
- [ ] Examples in documentation

---

### Phase 6: Add Schema Validation (⏳ TODO - 2 hours)

**Priority**: 🔴 Critical  
**Estimated Time**: 2 hours

#### Tasks

1. **Create Validator Module**

   ```python
   # ign_lidar/config/validator.py
   class ConfigSchemaValidator:
       """Validates config at load time."""

       REQUIRED_SECTIONS = [...]
       REQUIRED_PROCESSOR = [...]
       REQUIRED_FEATURES = [...]

       @classmethod
       def validate(cls, config: Dict) -> List[str]:
           """Validate and return errors."""
   ```

2. **Integrate with Config Loader**

   ```python
   # ign_lidar/cli/hydra_runner.py
   def load_config(...):
       cfg = OmegaConf.load(...)

       # VALIDATE IMMEDIATELY
       errors = ConfigSchemaValidator.validate(cfg)
       if errors:
           raise ConfigValidationError(errors)

       return cfg
   ```

3. **Add Validation Command**
   ```bash
   ign-lidar-hd validate-config -c my_config.yaml
   ```

#### Validation Checks

**Required Sections:**

- processor, features, data_sources, ground_truth
- output, logging, optimizations, validation, hardware

**Required Keys:**

- processor: lod_level, processing_mode, use_gpu, num_workers, preprocess, stitching
- features: mode, k_neighbors, compute_normals, compute_height

**Value Validation:**

- Enums (lod_level, processing_mode, feature mode)
- Ranges (gpu_memory_target: 0-1, k_neighbors: >=1)
- Types (bool, int, float, str)

#### Acceptance Criteria

- [ ] Validation catches missing keys at load time
- [ ] Clear error messages with suggestions
- [ ] Validate command works standalone
- [ ] All example configs pass validation
- [ ] CI/CD includes validation tests

---

### Phase 7: Update CLI & Documentation (⏳ TODO - 1 hour)

**Priority**: 🟡 High  
**Estimated Time**: 1 hour

#### Tasks

1. **Update CLI Commands**

   ```python
   # Default to base_complete if no config specified
   @click.command()
   def process(...):
       if not config:
           config = "base_complete"
   ```

2. **Add List Commands**

   ```bash
   ign-lidar-hd list-profiles    # Show available hardware profiles
   ign-lidar-hd list-presets     # Show available task presets
   ign-lidar-hd show-config <name>  # Display config contents
   ```

3. **Update Documentation**
   - Quick Start Guide (zero-config example)
   - Configuration Guide (3-tier system)
   - Migration Guide (old → new)
   - Preset Reference (all presets documented)

#### CLI Examples

```bash
# Zero-config (uses base_complete + auto-detect)
ign-lidar-hd process input_dir=/data output_dir=/output

# Select profile
ign-lidar-hd process \
  --config-name profiles/gpu_rtx4090 \
  input_dir=/data output_dir=/output

# Use preset
ign-lidar-hd process \
  --config-name presets/asprs_classification_gpu \
  input_dir=/data output_dir=/output

# Preset + overrides
ign-lidar-hd process \
  --config-name presets/asprs_classification_gpu \
  input_dir=/data output_dir=/output \
  processor.gpu_batch_size=16_000_000
```

#### Documentation Files to Create/Update

- [ ] docs/quick_start.md (zero-config example)
- [ ] docs/configuration_guide.md (3-tier system)
- [ ] docs/presets_reference.md (all presets)
- [ ] docs/migration_v5_to_v6.md (upgrade guide)
- [ ] README.md (update examples)

#### Acceptance Criteria

- [ ] CLI supports all new features
- [ ] Documentation complete and tested
- [ ] Examples verified
- [ ] Migration guide clear

---

### Phase 8: Cleanup & Migration (⏳ TODO - 1 hour)

**Priority**: 🟢 Low  
**Estimated Time**: 1 hour

#### Tasks

1. **Deprecate Old Configs**

   ```
   examples/
   ├── deprecated/  (MOVE old configs here)
   │   ├── config_asprs_gpu_16gb.yaml
   │   └── ...
   └── README.md (explain new system)
   ```

2. **Create Migration Tool**

   ```bash
   ign-lidar-hd migrate-config \
     --from examples/config_asprs_gpu_16gb.yaml \
     --to my_custom_preset.yaml
   ```

3. **Add Deprecation Warnings**
   ```python
   if config_file in DEPRECATED_CONFIGS:
       logger.warning(
           "This config format is deprecated. "
           "Use 'ign-lidar-hd migrate-config' to convert."
       )
   ```

#### Acceptance Criteria

- [ ] Old configs moved to deprecated/
- [ ] Migration tool tested
- [ ] Deprecation warnings logged
- [ ] No breaking changes for existing users

---

## 🎯 Success Metrics

### Before (Current State)

| Metric              | Value                      |
| ------------------- | -------------------------- |
| Example config size | 650 lines                  |
| Number of configs   | 40 files                   |
| Total config LOC    | ~24,000 lines              |
| Validation time     | Runtime (crash)            |
| New user config     | Must copy 650 lines        |
| Maintenance         | Update 40 files per change |

### After (Target State)

| Metric             | Value                | Improvement          |
| ------------------ | -------------------- | -------------------- |
| Custom config size | 20 lines             | **97% reduction** ✅ |
| Number of configs  | 20 files             | **50% reduction** ✅ |
| Total config LOC   | ~4,000 lines         | **83% reduction** ✅ |
| Validation time    | Load time (<1s)      | **Immediate** ✅     |
| New user config    | Zero (defaults work) | **Zero config** ✅   |
| Maintenance        | Update 1 base file   | **40× easier** ✅    |

---

## 📅 Timeline

### Week 1 (Current)

- [x] **Day 1**: Audit completed ✅
- [x] **Day 1**: Emergency fix applied ✅
- [ ] **Day 2**: Apply fix to all configs (Phase 2)
- [ ] **Day 2**: Create base_complete.yaml (Phase 3)
- [ ] **Day 3**: Create hardware profiles (Phase 4)

### Week 2

- [ ] **Day 1**: Create task presets (Phase 5)
- [ ] **Day 2**: Add schema validation (Phase 6)
- [ ] **Day 3**: Update CLI & docs (Phase 7)
- [ ] **Day 4**: Cleanup & migration (Phase 8)
- [ ] **Day 5**: Testing & refinement

---

## 🚀 Quick Start (For Users - After Implementation)

### Option 1: Zero Config (Recommended)

```bash
ign-lidar-hd process input_dir=/data/tiles output_dir=/data/output
```

### Option 2: Select Hardware

```bash
ign-lidar-hd process \
  --config-name profiles/gpu_rtx4080 \
  input_dir=/data/tiles output_dir=/data/output
```

### Option 3: Use Preset

```bash
ign-lidar-hd process \
  --config-name presets/asprs_classification_gpu \
  input_dir=/data/tiles output_dir=/data/output
```

### Option 4: Custom Config (20 lines)

```yaml
# my_config.yaml
defaults:
  - /base_complete
  - /profiles/gpu_rtx4080

config_name: "my_custom_config"

processor:
  gpu_batch_size: 10_000_000

features:
  k_neighbors: 40
```

---

## 🔧 Development Commands

### Test Configuration

```bash
# Validate config
ign-lidar-hd validate-config -c my_config.yaml

# Show merged config
ign-lidar-hd process \
  --config-name presets/asprs_classification_gpu \
  --cfg job

# List available configs
ign-lidar-hd list-profiles
ign-lidar-hd list-presets
```

### Apply Patches

```bash
# Patch all example configs
python scripts/patch_configs.py --backup --verify

# Migrate old config to new format
ign-lidar-hd migrate-config \
  --from examples/deprecated/config_old.yaml \
  --to my_new_preset.yaml
```

---

## 📝 Notes

### Critical Decisions

1. **Keep Old Configs?**

   - ✅ Yes, move to `examples/deprecated/`
   - ✅ Add deprecation warnings
   - ✅ Provide migration tool

2. **Break Compatibility?**

   - ❌ No, maintain backward compatibility
   - ✅ Auto-migrate on load (with warning)
   - ✅ Support both old and new formats

3. **Validation Strategy?**
   - ✅ Validate at load time (fail fast)
   - ✅ Provide clear error messages
   - ✅ Suggest fixes automatically

### Risks & Mitigations

| Risk                  | Probability | Impact | Mitigation                              |
| --------------------- | ----------- | ------ | --------------------------------------- |
| Breaking changes      | Medium      | High   | Backward compatibility layer            |
| User confusion        | High        | Medium | Clear migration guide + examples        |
| Validation too strict | Medium      | Medium | Warnings vs. errors, config suggestions |
| Missing edge cases    | Low         | Medium | Comprehensive testing on real data      |

---

## ✅ Current Status

**Phase 1**: ✅ Complete  
**Phase 2**: ⏳ Ready to start  
**Next Action**: Create and run patch script for all example configs

**Ready to proceed with Phase 2?** Just say the word!
