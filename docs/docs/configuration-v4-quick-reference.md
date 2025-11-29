# Configuration v4.0 Harmonization - Quick Reference

**For:** Development Team  
**Date:** November 28, 2025  
**Status:** Implementation Ready

---

## 🎯 Quick Overview

**Goal:** Consolidate 3 parallel config systems → 1 unified v4.0 system

**Timeline:** 12 weeks  
**Effort:** ~150-180 hours

---

## 📋 Key Documents

1. **Analysis:** `configuration-harmonization-analysis.md` (original proposal)
2. **Implementation Plan:** `configuration-harmonization-implementation-plan.md` (detailed roadmap)
3. **GitHub Issues:** `configuration-v4-github-issues.md` (25 ready-to-create issues)
4. **This Document:** Quick reference for daily work

---

## 🚀 Getting Started

### Step 1: Set Up Project (Week 0)

```bash
# 1. Create GitHub milestone
# Go to: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/milestones
# Title: "v4.0 Configuration Harmonization"
# Due date: 12 weeks from now
# Description: [copy from implementation plan]

# 2. Create labels
# Go to: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/labels
# Create:
#   - config-harmonization (purple, #9966cc)
#   - breaking-change (red, #d73a4a)
#   - migration (orange, #f9a825)
#   - v4.0 (green, #0e8a16)

# 3. Create v4.0-dev branch
git checkout -b v4.0-dev
git push -u origin v4.0-dev

# 4. Protect branch
# Settings → Branches → Add rule
# Require PR reviews, status checks

# 5. Create GitHub project board
# Projects → New project → "Configuration v4.0"
# Columns: Backlog, In Progress, Review, Testing, Done
```

### Step 2: Create All Issues

Use the templates in `configuration-v4-github-issues.md` to create 25 issues.

**Quick create script:**
```bash
# Option 1: Manual (recommended for first time)
# - Go to Issues tab
# - Click "New issue"
# - Copy/paste from github-issues.md
# - Set labels, milestone, assignee

# Option 2: GitHub CLI (if installed)
gh issue create --title "Add OptimizationsConfig to config.py" \
                --body-file issue-1-body.md \
                --label "config-harmonization,v4.0" \
                --milestone "v4.0"
```

---

## 📅 Week-by-Week Breakdown

### Week 1-2: Phase 1 - Python Config

**Focus:** Consolidate Python config classes

**Issues:**
- #1: Add OptimizationsConfig
- #2: Rename feature_set → mode
- #3: Add from_legacy_schema()
- #4: Add comprehensive docstrings
- #5: Update all imports
- #6: Add deprecation warnings

**Deliverables:**
- ✅ Enhanced `config.py` with v4.0 structure
- ✅ Migration method for legacy configs
- ✅ Deprecation warnings in `schema.py`
- ✅ All imports updated

**Branch:** `feature/python-config-consolidation`

---

### Week 3-4: Phase 2 - YAML Harmonization

**Focus:** Flatten and standardize YAML configs

**Issues:**
- #7: Update base.yaml
- #8: Update all 7 presets
- #9: Update example configs

**Deliverables:**
- ✅ `base.yaml` v4.0 structure
- ✅ All presets updated
- ✅ Examples migrated

**Branch:** `feature/yaml-harmonization`

---

### Week 5-6: Phase 3 - Migration Tooling

**Focus:** Automatic migration CLI

**Issues:**
- #10: Implement ConfigMigrator
- #11: Create migrate-config CLI
- #12: Write migration tests

**Deliverables:**
- ✅ `migration.py` module
- ✅ `migrate-config` command
- ✅ Full test suite

**Branch:** `feature/migration-tooling`

---

### Week 7: Phase 4 - Documentation

**Focus:** Unified documentation

**Issues:**
- #13: Create unified guide
- #14: Create migration guide
- #15: Update inline docs
- #16: Archive old docs

**Deliverables:**
- ✅ New documentation structure
- ✅ Migration guide
- ✅ API documentation
- ✅ Archived old docs

**Branch:** `feature/documentation`

---

### Week 8: Phase 5 - Testing

**Focus:** Comprehensive validation

**Issues:**
- #17: Unit tests
- #18: Integration tests
- #19: Validate presets

**Deliverables:**
- ✅ >95% test coverage
- ✅ All presets validated
- ✅ Integration tests pass

**Branch:** `feature/testing`

---

### Week 9-10: Beta Release

**Focus:** v3.9 pre-release + beta testing

**Issues:**
- #20: Create v3.9 release
- #21: Beta testing
- #22: Prepare release notes

**Deliverables:**
- ✅ v3.9.0 released to PyPI
- ✅ Beta feedback collected
- ✅ Bugs fixed

---

### Week 11-12: Final Release

**Focus:** v4.0.0 stable release

**Issues:**
- #23: Final v4.0 release
- #24: Project board setup
- #25: Progress tracking

**Deliverables:**
- ✅ v4.0.0 released
- ✅ Announced
- ✅ Documentation live

---

## 🔧 Development Workflow

### For Each Issue

```bash
# 1. Create feature branch
git checkout v4.0-dev
git pull origin v4.0-dev
git checkout -b feature/issue-X-short-name

# 2. Implement changes
# ... code ...

# 3. Write tests
pytest tests/... -v

# 4. Update documentation
# Update docstrings, README, etc.

# 5. Run full test suite
pytest tests/ -v --cov=ign_lidar

# 6. Check linting
black ign_lidar/
ruff check ign_lidar/

# 7. Commit with clear message
git add .
git commit -m "feat(config): Add OptimizationsConfig (#1)

- Add OptimizationsConfig dataclass
- Nest in Config class
- Add defaults for all Phase 4 optimizations
- Update tests

Closes #1"

# 8. Push and create PR
git push origin feature/issue-X-short-name
# Create PR on GitHub targeting v4.0-dev
```

### PR Review Checklist

- [ ] Code follows project style
- [ ] Tests added/updated
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] CHANGELOG updated (if needed)
- [ ] Approved by 1+ reviewers

---

## 🎯 Critical Paths

### Must Complete Before v3.9

1. ✅ All Phase 1 issues (#1-6)
2. ✅ All Phase 2 issues (#7-9)
3. ✅ Migration tooling (#10-11)
4. ✅ Migration tests (#12)

### Must Complete Before v4.0

1. ✅ All documentation (#13-16)
2. ✅ All tests (#17-19)
3. ✅ Beta feedback addressed (#21)
4. ✅ Release notes (#22)

---

## 📊 Metrics to Track

### Code Metrics

```bash
# Lines of config code
find ign_lidar/config -name "*.py" -exec wc -l {} + | tail -1

# Test coverage
pytest tests/ --cov=ign_lidar.config --cov-report=term

# Config loading time
python -m timeit -s "from ign_lidar.config import Config" "Config.preset('lod2_buildings')"
```

### Target Metrics

| Metric | Current | Target v4.0 | Status |
|--------|---------|-------------|--------|
| Config code lines | ~2,366 | ~1,651 | ⬜ |
| Test coverage | ~80% | >95% | ⬜ |
| Load time | ~200ms | <50ms | ⬜ |
| Config systems | 3 | 1 | ⬜ |

---

## 🚨 Common Pitfalls to Avoid

### 1. Breaking Backward Compatibility Too Early

⚠️ **Don't:** Remove `schema.py` before v4.0  
✅ **Do:** Keep it in v3.9 with loud deprecation warnings

### 2. Incomplete Migration

⚠️ **Don't:** Forget to update all imports  
✅ **Do:** Use grep to find all references before changing

```bash
# Find all schema.py imports
grep -r "from.*config.schema import" ign_lidar/
grep -r "import.*schema" ign_lidar/
```

### 3. Inconsistent Naming

⚠️ **Don't:** Mix old and new parameter names  
✅ **Do:** Complete rename in one PR

### 4. Missing Documentation

⚠️ **Don't:** Change APIs without updating docs  
✅ **Do:** Update docs in the same PR as code changes

### 5. Skipping Tests

⚠️ **Don't:** "I'll write tests later"  
✅ **Do:** Write tests as you go

---

## 🔍 Code Review Focus Areas

### For Reviewers

#### Python Changes
- [ ] Type hints complete and correct
- [ ] Docstrings comprehensive (Google style)
- [ ] Backward compatibility maintained (v3.9)
- [ ] Deprecation warnings clear
- [ ] Tests cover new code

#### YAML Changes
- [ ] Hydra composition still works
- [ ] All parameters documented
- [ ] Consistent naming (lowercase, underscores)
- [ ] Default values sensible

#### Migration Code
- [ ] All parameter mappings correct
- [ ] Edge cases handled
- [ ] Error messages helpful
- [ ] Validation thorough

#### Documentation
- [ ] Examples work
- [ ] Links valid
- [ ] Clear and concise
- [ ] No typos

---

## 🆘 Getting Help

### Resources

- **Implementation Plan:** See full details in `configuration-harmonization-implementation-plan.md`
- **Analysis:** See original proposal in `configuration-harmonization-analysis.md`
- **GitHub Issues:** All templates in `configuration-v4-github-issues.md`

### Contacts

- **Lead:** [Your Name]
- **Config Expert:** [Expert Name]
- **Documentation:** [Doc Lead]
- **Testing:** [QA Lead]

### Communication

- **Daily:** Slack #config-v4-dev
- **Weekly:** Friday progress updates
- **Blockers:** Tag @lead in GitHub issue

---

## ✅ Definition of Done

### For Each Issue

- [ ] Code implemented and working
- [ ] Tests written and passing (>90% coverage)
- [ ] Documentation updated
- [ ] Code reviewed and approved
- [ ] PR merged to v4.0-dev
- [ ] Issue closed with summary comment

### For Each Phase

- [ ] All phase issues complete
- [ ] Integration tests pass
- [ ] Documentation reviewed
- [ ] Demo/walkthrough complete
- [ ] Phase retrospective done

### For v3.9 Release

- [ ] All Phase 1-3 complete
- [ ] Migration tool tested with real configs
- [ ] Documentation published
- [ ] CHANGELOG updated
- [ ] PyPI package released
- [ ] Announcement sent

### For v4.0 Release

- [ ] All phases complete
- [ ] Beta feedback addressed
- [ ] All tests pass (>95% coverage)
- [ ] Documentation complete
- [ ] Performance validated
- [ ] Release notes approved
- [ ] PyPI package released
- [ ] GitHub release created
- [ ] Announcement sent

---

## 📈 Success Criteria

### Quantitative

- ✅ Config loading time: <50ms (target: -75%)
- ✅ Lines of code: <1,651 (target: -30%)
- ✅ Test coverage: >95% (target: +15%)
- ✅ Migration success rate: >99%
- ✅ Documentation pages: 1 comprehensive guide (from 5+)

### Qualitative

- ✅ Zero "config confusion" reports
- ✅ All presets work out-of-box
- ✅ Migration tool handles all known configs
- ✅ Documentation clear and comprehensive
- ✅ Team confident in v4.0 system

---

## 🎉 Celebration Milestones

- 🎈 Phase 1 complete → Team lunch
- 🎈 v3.9 released → Happy hour
- 🎈 Beta testing complete → Team dinner
- 🎈 v4.0 released → Big celebration! 🎊

---

**Remember:** This is a major refactoring. Take time to do it right. Communication is key!

**Last Updated:** November 28, 2025  
**Version:** 1.0

