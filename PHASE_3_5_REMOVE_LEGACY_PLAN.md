# Phase 3.5: Remove Legacy Code - Action Plan

**Date:** October 13, 2025  
**Objective:** Remove duplicate/legacy process_tile method  
**Expected Impact:** ~1,536 lines removed (57% reduction!)

---

## 🔍 Discovery

**Current processor.py:** 2,684 lines

**Found:** TWO `process_tile` methods!

### Method 1: Refactored Version ✅ (Line 698)

- **Signature:** `process_tile(self, laz_file, output_dir, tile_idx, total_tiles, skip_existing)`
- **Lines:** ~265 lines
- **Status:** Uses TileLoader and FeatureComputer modules
- **Phase:** Created in Phase 3.4
- **Action:** **KEEP THIS ONE**

### Method 2: Legacy Version ❌ (Line 1148)

- **Signature:** `process_tile(self, laz_file, output_dir, architecture, save_enriched, only_enriched, ...)`
- **Lines:** ~1,536 lines (57% of file!)
- **Status:** Old monolithic implementation
- **Phase:** Pre-consolidation code
- **Action:** **DELETE THIS ONE**

---

## 📊 Impact Analysis

### Current State

```
Total lines: 2,684
- Refactored process_tile: ~265 lines
- Legacy process_tile: ~1,536 lines
- Other methods: ~883 lines
```

### After Cleanup

```
Total lines: ~1,148 lines (57% reduction!)
- Refactored process_tile: ~265 lines
- Legacy process_tile: REMOVED
- Other methods: ~883 lines
```

### Comparison to Original

- **Original processor.py:** ~2,942 lines (pre-Phase 3)
- **After Phase 3.4:** 2,684 lines (9% reduction)
- **After Phase 3.5:** ~1,148 lines (61% reduction from original!)

---

## ✅ Validation Steps

Before deleting, we need to confirm:

### 1. Which method is being called?

Check all call sites to ensure they're using the refactored version

### 2. Are there any references to the legacy version?

Search for any code that specifically expects the old signature

### 3. Run tests after deletion

Ensure nothing breaks when we remove the legacy code

---

## 🎯 Execution Plan

### Step 1: Identify the exact line range (5 min)

Find where legacy process_tile starts and ends

### Step 2: Check for callers (5 min)

```bash
# Find all calls to process_tile
grep -rn "\.process_tile(" ign_lidar/ tests/
```

### Step 3: Backup and delete (2 min)

Remove lines 1148-2684 (legacy method)

### Step 4: Test (10 min)

```bash
# Run integration test
python tests/test_phase_3_4_integration.py

# Run unit tests
pytest tests/test_modules/

# Quick smoke test
python -c "from ign_lidar.core.processor import LiDARProcessor; print('OK')"
```

### Step 5: Validate (5 min)

- Check line count
- Run basic processing
- Confirm no errors

**Total Time:** ~30 minutes

---

## 🚨 Risk Assessment

**Risk Level:** LOW ⚠️

**Why Low Risk:**

- Integration test already passed using refactored version
- Legacy method likely not being called
- Easy to rollback if needed (git)
- Tests will catch any issues

**Mitigation:**

- Check all call sites first
- Run full test suite after deletion
- Keep git commit separate for easy rollback

---

## 📋 Checklist

### Pre-Deletion

- [ ] Identify exact line range of legacy method
- [ ] Find all callers of process_tile
- [ ] Confirm refactored version is being used
- [ ] Create git checkpoint

### Deletion

- [ ] Remove legacy process_tile method
- [ ] Save file
- [ ] Check for syntax errors

### Post-Deletion Validation

- [ ] Count new line total
- [ ] Run integration test
- [ ] Run unit tests
- [ ] Import test (module loads)
- [ ] Quick processing test

### Documentation

- [ ] Update CONSOLIDATION_PROGRESS_UPDATE.md
- [ ] Update Phase 3 status (75% → 95%+)
- [ ] Document line reduction achievement

---

## 🎉 Expected Outcome

**Phase 3.5 Complete:**

- ✅ Legacy code removed
- ✅ 1,536 lines deleted
- ✅ processor.py: 2,684 → ~1,148 lines (57% reduction)
- ✅ Cumulative from original: 61% reduction
- ✅ Clean, maintainable codebase

**Phase 3 Status:**

- Current: 75% complete
- After 3.5: **95% complete**

**Remaining for Phase 3:**

- Maybe one final cleanup pass
- Update documentation
- Declare Phase 3 complete!

---

## 💡 Why This is Great News

1. **Easy Win:** Just delete dead code, no refactoring needed
2. **Big Impact:** 57% reduction in one step
3. **Low Risk:** Code not being used
4. **Quick:** ~30 minutes total
5. **Clean:** Removes confusion about which method to use

---

## 🚀 Ready to Execute?

**Say:** "Remove legacy process_tile" and I'll:

1. Identify exact line range
2. Check for callers
3. Delete the legacy method
4. Run validation tests
5. Update documentation

**This will take us from 75% → 95% Phase 3 completion in 30 minutes!**

---

**What would you like to do?**

- "Remove legacy process_tile" - Execute Phase 3.5
- "Check callers first" - Safety check before deletion
- "Show me the legacy method" - Review before deleting
