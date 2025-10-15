# 🚀 Phase 1 Quick Reference Card

> **Print this page and keep it at your desk during Phase 1 implementation**

---

## ⚡ Quick Commands

```bash
# Verify environment
./scripts/phase1_preflight.sh

# Analyze duplication
python3 scripts/analyze_duplication.py --output report.json

# Create feature branch
git checkout -b refactor/phase1-$(date +%Y%m%d)

# Run tests
pytest tests/ -v
pytest tests/ --cov=ign_lidar --cov-report=html

# Check specific module
pytest tests/features/ -v -k "test_normals"
```

---

## 📋 Phase 1 Tasks (2 Weeks)

### Week 1

- [ ] **Task 1.1** (2h): Fix duplicate `compute_verticality` at line 877
- [ ] **Task 1.2** (16h): Create `features/core/` module
  - [ ] normals.py (4h)
  - [ ] curvature.py (3h)
  - [ ] eigenvalues.py (3h)
  - [ ] density.py (3h)
  - [ ] architectural.py (3h)
- [ ] **Task 1.3** (6h): Consolidate memory modules (3 → 1)

### Week 2

- [ ] **Task 1.4** (12h): Update feature modules to use core
- [ ] **Task 1.5** (4h): Testing & validation

---

## 🎯 Success Criteria

✅ Duplicate function bug fixed  
✅ `features/core/` created with 6 files  
✅ Memory consolidated to 1 file  
✅ All tests passing  
✅ Coverage >= 70%  
✅ LOC reduced by ~6%

---

## 📊 Key Metrics

| Before         | After | Target       |
| -------------- | ----- | ------------ |
| 40,002 LOC     | ?     | 37,602 (-6%) |
| 25 duplicates  | ?     | 12 (-52%)    |
| 65% coverage   | ?     | 70% (+5%)    |
| 1 critical bug | ?     | 0 (fixed)    |

---

## 🐛 Critical Bug

**File**: `ign_lidar/features/features.py`  
**Lines**: 440 and 877  
**Issue**: `compute_verticality()` defined twice  
**Fix**: Rename second to `compute_normal_verticality()`

---

## 📁 New Structure

```
features/
└── core/              ← NEW
    ├── __init__.py
    ├── normals.py
    ├── curvature.py
    ├── eigenvalues.py
    ├── density.py
    ├── architectural.py
    └── utils.py

core/
└── memory.py          ← UNIFIED (was 3 files)
```

---

## 🔗 Documentation Links

- **Full Guide**: `PHASE1_IMPLEMENTATION_GUIDE.md`
- **Visual**: `PHASE1_BEFORE_AFTER.md`
- **Roadmap**: `CONSOLIDATION_ROADMAP.md`
- **Index**: `CONSOLIDATION_INDEX.md`

---

## 🆘 Troubleshooting

### Tests fail after changes

```bash
pytest tests/features/ -v --tb=short
# Check imports and function signatures
```

### Import errors

```bash
pip install -e .  # Reinstall package
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### GPU tests fail

```bash
pytest tests/ -v -m "not gpu"  # Skip GPU tests
```

---

## 📈 Daily Checklist

**Start of Day**:

- [ ] Pull latest changes: `git pull origin main`
- [ ] Check current task in `PHASE1_IMPLEMENTATION_GUIDE.md`
- [ ] Review yesterday's commits

**During Work**:

- [ ] Write code
- [ ] Write/update tests
- [ ] Run tests: `pytest tests/features/ -v`
- [ ] Commit often: `git commit -m "feat: ..."`

**End of Day**:

- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Check coverage: `pytest --cov=ign_lidar --cov-report=term`
- [ ] Push changes: `git push origin <branch>`
- [ ] Update task checklist

---

## 🎯 Phase 1 Timeline

```
Week 1, Day 1-2:  Task 1.1 (Fix bug)
Week 1, Day 3-5:  Task 1.2 (Create core)
Week 2, Day 1-3:  Task 1.3 (Consolidate memory)
Week 2, Day 4-5:  Task 1.4 (Update modules)
Week 2, Day 5:    Task 1.5 (Testing)
```

---

## 🏆 When Done

```bash
# Final checks
pytest tests/ -v
pytest tests/ --cov=ign_lidar --cov-report=html

# Update version
# Edit pyproject.toml: version = "2.5.2"

# Update changelog
# Edit CHANGELOG.md

# Create PR
git push origin refactor/phase1-$(date +%Y%m%d)

# Tag release (after merge)
git tag v2.5.2
git push origin v2.5.2
```

---

## 💡 Pro Tips

1. **Commit early, commit often** - Small commits are easier to review
2. **Run tests after each file** - Catch issues early
3. **Use deprecation warnings** - Maintain backward compatibility
4. **Document as you go** - Update docstrings immediately
5. **Ask for help** - Review `CONSOLIDATION_INDEX.md` for guidance

---

## 📞 Need Help?

- **Implementation details**: `PHASE1_IMPLEMENTATION_GUIDE.md`
- **Code examples**: `CONSOLIDATION_ROADMAP.md`
- **Visual diagrams**: `PHASE1_BEFORE_AFTER.md`
- **Full context**: `CONSOLIDATION_INDEX.md`

---

**Start Date**: ****\_\_****  
**Target Completion**: ****\_\_**** (2 weeks)  
**Actual Completion**: ****\_\_****

**Team Members**: **************\_\_\_**************

---

_Keep this card visible during Phase 1 implementation_  
_Update checkboxes daily to track progress_  
_Print at: 100% scale, landscape orientation recommended_
