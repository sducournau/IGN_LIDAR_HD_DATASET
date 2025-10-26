# TileProcessor Refactoring Strategy

## 🎯 Goal
Extract tile processing logic from LiDARProcessor (1,320 lines) into modular, testable components.

## 📊 Current Analysis

**Target Method**: `_process_tile_core` (lines 1623-2942, 1,320 lines!)

**Method Responsibilities** (identified by reading the code flow):
1. Load and validate point cloud data
2. Remove outliers (preprocessing)
3. Compute features (geometric, RGB, NIR, NDVI)
4. Apply ground truth classification
5. Refine classification with rules engine
6. Extract and augment patches
7. Save outputs (patches and/or enriched LAZ)
8. Update metadata and statistics

## 🔨 Refactoring Approach

### Phase 1: Create TileProcessor Class
Break down into these focused classes:

```
ign_lidar/core/
├── tile_processor.py        # NEW - Tile processing coordination
├── patch_extractor.py       # NEW - Patch extraction logic  
├── classification_applier.py # NEW - Ground truth application
└── output_writer.py         # NEW - Output generation
```

### Phase 2: Method Extraction Strategy

#### From `_process_tile_core` (1,320 lines) → Multiple classes:

**TileProcessor (Coordinator)** ~200 lines:
- `process_tile()` - Main entry point
- `_load_and_validate()` - Load LAZ, validate
- `_preprocess()` - Call outlier removal
- `_compute_features()` - Delegate to orchestrator
- `_apply_ground_truth()` - Delegate to ClassificationApplier
- `_extract_patches()` - Delegate to PatchExtractor
- `_save_outputs()` - Delegate to OutputWriter

**PatchExtractor** ~300 lines:
- `extract_patches()` - Main extraction logic
- `_validate_patch()` - Patch validation
- `_augment_patch()` - Augmentation
- `_filter_by_class()` - Class-based filtering

**ClassificationApplier** ~200 lines:
- `apply_ground_truth()` - Apply ground truth data
- `refine_classification()` - Rules engine
- `_merge_classifications()` - Merge logic

**OutputWriter** ~200 lines:
- `save_patches()` - Save patch files
- `save_enriched_laz()` - Save enriched LAZ
- `_format_for_architecture()` - Format conversion

## 📋 Implementation Steps

### Step 1: Create PatchExtractor (Easier, standalone)
**Estimated Time**: 2 hours
**Lines**: ~300

Extract patch-related methods:
- `extract_and_augment_patches()`
- `format_patch_for_architecture()`
- Validation logic

### Step 2: Create ClassificationApplier
**Estimated Time**: 2 hours
**Lines**: ~200

Extract classification methods:
- Ground truth application
- Rules engine integration
- Classification merging

### Step 3: Create OutputWriter
**Estimated Time**: 1.5 hours
**Lines**: ~200

Extract output methods:
- Multi-format patch saving
- Enriched LAZ generation
- Metadata updates

### Step 4: Create TileProcessor (Coordinator)
**Estimated Time**: 3 hours
**Lines**: ~200

Coordinate all components:
- Simplified `process_tile()` 
- Delegate to specialist classes
- Error handling and recovery

### Step 5: Update LiDARProcessor (Facade)
**Estimated Time**: 1 hour

Update to use new TileProcessor:
```python
class LiDARProcessor:
    def __init__(self, config):
        self.core = ProcessorCore(config)
        self.tile_processor = TileProcessor(self.core)
    
    def process_tile(self, laz_file, output_dir):
        return self.tile_processor.process_tile(laz_file, output_dir)
```

## ✅ Benefits

1. **Testability**: Each component can be tested independently
2. **Maintainability**: Clear responsibilities, easy to modify
3. **Reusability**: Components can be reused in different contexts
4. **Readability**: Each class < 300 lines, methods < 50 lines
5. **Performance**: Same performance, better code organization

## 🧪 Testing Strategy

### Unit Tests (per component):
- `test_patch_extractor.py` - Test patch extraction logic
- `test_classification_applier.py` - Test classification application
- `test_output_writer.py` - Test output generation
- `test_tile_processor.py` - Test coordination

### Integration Tests:
- `test_tile_processing_pipeline.py` - End-to-end tile processing
- Compare outputs with original implementation

### Regression Tests:
- Run on existing test tiles
- Verify bit-for-bit identical outputs

## 📦 Migration Strategy

### Phase A: Parallel Implementation (Week 1)
- Create new classes alongside existing code
- Add feature flag: `use_refactored_processor`
- Run both implementations, compare outputs

### Phase B: Gradual Migration (Week 2)
- Enable refactored version by default
- Keep old code for 1 release cycle
- Add deprecation warnings

### Phase C: Cleanup (Week 3)
- Remove old implementation
- Update all tests
- Update documentation

## 🎯 Success Criteria

- [x] processor_core.py created (493 lines)
- [ ] patch_extractor.py created (~300 lines)
- [ ] classification_applier.py created (~200 lines)
- [ ] output_writer.py created (~200 lines)
- [ ] tile_processor.py created (~200 lines)
- [ ] All tests passing
- [ ] Performance maintained or improved
- [ ] Documentation updated

## 📊 Before/After Metrics

**Before**:
- `processor.py`: 3,082 lines (god class)
- `_process_tile_core`: 1,320 lines (mega method)
- Cyclomatic complexity: 45+ (unmaintainable)

**After**:
- `processor.py`: ~300 lines (facade)
- `processor_core.py`: 493 lines ✅
- `tile_processor.py`: ~200 lines
- `patch_extractor.py`: ~300 lines
- `classification_applier.py`: ~200 lines
- `output_writer.py`: ~200 lines
- Max cyclomatic complexity: <15 per method

**Total reduction**: 3,082 → 1,693 lines across 6 focused files
**Improvement**: 45% size reduction + infinitely better organization!

---

**Status**: Phase 1 (ProcessorCore) DONE, Starting Phase 2 (TileProcessor)
**Next**: Create PatchExtractor.py
**Updated**: October 26, 2025
