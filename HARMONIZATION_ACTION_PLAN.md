# IGN LiDAR HD - Harmonization & Consolidation Action Plan

**Version:** 3.2.1 → 3.3.0
**Date:** October 25, 2025  
**Status:** 🎯 Ready to Execute

---

## 📋 Executive Summary

This document outlines the specific actions needed to complete the harmonization and consolidation of the IGN LiDAR HD codebase. Based on the comprehensive audit (see `CODEBASE_AUDIT_2025-10-25.md`), we have identified clear, actionable steps to improve:

1. **Package organization** and metadata
2. **Configuration system** simplification
3. **Documentation** completeness and accessibility
4. **Changelog** finalization

**Timeline:** 1-2 weeks  
**Risk:** Low (mostly documentation and organization)  
**Breaking Changes:** None

---

## ✅ Completed Actions (Current Session)

- [x] Created comprehensive codebase audit report
- [x] Updated `pyproject.toml` version to 3.2.1
- [x] Updated `CHANGELOG.md` release date to 2025-10-25
- [x] Moved unreleased items to v3.2.1 section
- [x] Created this action plan document

---

## 🎯 Phase 1: Package Metadata & Structure (Priority: HIGH)

### 1.1 Update Package Metadata

**File:** `pyproject.toml`

**Actions:**

1. **Clarify GPU Dependencies** ✅ (Already well-documented)

   ```toml
   # Already good - includes clear comments about manual installation
   gpu = []  # Intentionally empty - see installation docs
   ```

2. **Add Development Dependency Groups**

   ```toml
   [project.optional-dependencies]
   # ... existing ...

   # New groups for development workflows
   test = [
       "pytest>=6.0",
       "pytest-cov>=4.0",
       "pytest-xdist>=2.0",
       "pytest-timeout>=2.0",
   ]

   docs = [
       "sphinx>=4.0",
       "sphinx-rtd-theme>=1.0",
       "myst-parser>=0.18",
       "sphinx-autodoc-typehints>=1.12",
   ]

   lint = [
       "black>=22.0",
       "flake8>=4.0",
       "mypy>=0.910",
       "isort>=5.0",
       "pylint>=2.12",
   ]

   # Complete development environment
   dev = [
       "ign-lidar-hd[test,docs,lint]",
       "pre-commit>=2.20",
       "build>=0.7.0",
       "twine>=4.0",
   ]
   ```

3. **Update Classifiers** (if needed)
   ```toml
   classifiers = [
       "Development Status :: 4 - Beta",  # Or "5 - Production/Stable" if ready
       # ... rest unchanged ...
   ]
   ```

**Estimated Time:** 30 minutes  
**Files Modified:** 1 (`pyproject.toml`)

---

### 1.2 Verify Package Structure

**Actions:**

1. **Ensure MANIFEST.in is complete**
   - Review included/excluded files
   - Verify examples/ and docs/ are properly included
2. **Check **init**.py exports**
   - Verify all public APIs are exported
   - Check for deprecated imports still in use

**Estimated Time:** 1 hour  
**Files to Review:** `MANIFEST.in`, `ign_lidar/__init__.py`

---

## 📚 Phase 2: Documentation Updates (Priority: HIGH)

### 2.1 Update Docusaurus Site

**Directory:** `docs/`

**Actions:**

#### 2.1.1 Update Main Configuration

**File:** `docs/docusaurus.config.ts`

```typescript
// Update version references
const config: Config = {
  title: "IGN LiDAR HD Processing Library",
  tagline: "Process IGN LiDAR data into ML-ready datasets with v3.2.1",
  // ... rest of config
};
```

#### 2.1.2 Create New Documentation Pages

**New Files to Create:**

1. **docs/docs/features/rules-framework.md**

   ```markdown
   ---
   sidebar_position: 5
   ---

   # Rules Framework

   Complete guide to the extensible rule-based classification system
   introduced in v3.2.0.

   ## Overview

   The Rules Framework provides...

   [Continue with comprehensive guide]
   ```

2. **docs/docs/api/rules.md**

   ```markdown
   ---
   sidebar_position: 6
   ---

   # Rules API Reference

   API documentation for rule-based classification.

   ## Core Classes

   ### BaseRule

   [API documentation]
   ```

3. **docs/docs/migration/v3.1-to-v3.2.md**

   ```markdown
   ---
   sidebar_position: 3
   ---

   # Migration Guide: v3.1 → v3.2

   Guide for upgrading from v3.1 to v3.2, covering:

   - Rules framework introduction
   - Classification consolidation completion
   - Ground truth refinement enhancements
   ```

4. **docs/docs/migration/overview.md**

   ```markdown
   ---
   sidebar_position: 1
   ---

   # Migration Guide Overview

   Complete reference for migrating between versions.

   ## By Version

   - [v2.x → v3.0](./v2-to-v3.md)
   - [v3.0 → v3.1](./v3.0-to-v3.1.md)
   - [v3.1 → v3.2](./v3.1-to-v3.2.md)

   ## By Module

   [Links to module-specific guides]
   ```

#### 2.1.3 Update Existing Pages

**Files to Update:**

1. **docs/docs/introduction.md**

   - Add v3.2.1 features summary
   - Update "What's New" section
   - Add rules framework highlight

2. **docs/docs/guides/configuration.md**

   - Update for v5.5 configuration system
   - Add preset examples
   - Update hardware profiles documentation

3. **docs/docs/guides/ground-truth.md**

   - Add refinement features documentation
   - Update with new refinement methods
   - Add examples

4. **docs/docs/api/classification.md**
   - Update for consolidated modules
   - Add deprecation warnings
   - New import paths

#### 2.1.4 Update Navigation

**File:** `docs/sidebars.ts`

```typescript
const sidebars = {
  tutorialSidebar: [
    "introduction",
    {
      type: "category",
      label: "Installation",
      items: ["installation/quick-start", "installation/gpu-setup"],
    },
    {
      type: "category",
      label: "Guides",
      items: [
        "guides/basic-usage",
        "guides/configuration",
        "guides/processing-modes",
        "guides/feature-modes",
        "guides/rules-framework", // NEW
        "guides/ground-truth",
      ],
    },
    {
      type: "category",
      label: "API Reference",
      items: [
        "api/processor",
        "api/features",
        "api/classification",
        "api/rules", // NEW
        "api/cli",
      ],
    },
    {
      type: "category",
      label: "Migration",
      items: [
        "migration/overview", // NEW
        "migration/v2-to-v3",
        "migration/v3.0-to-v3.1",
        "migration/v3.1-to-v3.2", // NEW
      ],
    },
    {
      type: "category",
      label: "Architecture",
      items: [
        "architecture/overview",
        "architecture/classification",
        "architecture/rules-system", // NEW
      ],
    },
  ],
};
```

**Estimated Time:** 6-8 hours  
**Files Created:** 4 new pages  
**Files Modified:** 6-8 existing pages

---

### 2.2 Consolidate Migration Guides

**Directory:** `docs/`

**Actions:**

1. **Create Unified Migration Guide**

   **New File:** `docs/MIGRATION_GUIDE_UNIFIED.md`

   Structure:

   ```markdown
   # IGN LiDAR HD Migration Guide: Complete Reference

   ## Quick Navigation

   ### By Version

   - [v2.x → v3.0](#v2-to-v3)
   - [v3.0 → v3.1](#v3-0-to-v3-1)
   - [v3.1 → v3.2](#v3-1-to-v3-2)

   ### By Module

   - [Classification Module](#classification)
   - [Feature Module](#features)
   - [Configuration System](#configuration)
   - [Rules Framework](#rules)

   ### By Feature

   - [GPU Acceleration](#gpu)
   - [Ground Truth](#ground-truth)
   - [Rules Framework](#rules-framework)

   [Full content with consolidated information]
   ```

2. **Update Existing Migration Guides**

   Add cross-references to unified guide:

   ```markdown
   > **📖 See also:** [Unified Migration Guide](./MIGRATION_GUIDE_UNIFIED.md)
   > for a complete overview of all migration paths.
   ```

**Estimated Time:** 3-4 hours  
**Files Created:** 1 comprehensive guide  
**Files Modified:** 3-4 existing migration docs

---

## ⚙️ Phase 3: Configuration System (Priority: MEDIUM)

### 3.1 Reorganize Example Configs

**Directory:** `examples/`

**Actions:**

1. **Create New Directory Structure**

   ```
   examples/
   ├── README.md                     # NEW: Complete guide
   ├── quickstart/                   # NEW: Getting started
   │   ├── 00_minimal.yaml          # Simplest config
   │   ├── 01_cpu_basic.yaml        # Basic CPU
   │   ├── 02_gpu_basic.yaml        # Basic GPU
   │   └── README.md                 # Quick start guide
   ├── production/                   # NEW: Production workflows
   │   ├── asprs_classification.yaml
   │   ├── lod2_buildings.yaml
   │   ├── lod3_architecture.yaml
   │   └── README.md                 # Production guide
   ├── advanced/                     # NEW: Advanced use cases
   │   ├── multi_scale.yaml
   │   ├── custom_features.yaml
   │   ├── hybrid_processing.yaml
   │   └── README.md                 # Advanced guide
   └── legacy/                       # MOVED: Old configs
       └── [old configs]
   ```

2. **Create Main README**

   **File:** `examples/README.md`

   ```markdown
   # Configuration Examples

   Complete guide to IGN LiDAR HD configurations.

   ## Quick Start

   New to IGN LiDAR HD? Start here:

   - [Minimal Config](./quickstart/00_minimal.yaml) - Simplest possible
   - [CPU Basic](./quickstart/01_cpu_basic.yaml) - CPU processing
   - [GPU Basic](./quickstart/02_gpu_basic.yaml) - GPU acceleration

   ## Production Workflows

   Ready for production? Use these:

   - [ASPRS Classification](./production/asprs_classification.yaml)
   - [LOD2 Buildings](./production/lod2_buildings.yaml)
   - [LOD3 Architecture](./production/lod3_architecture.yaml)

   ## Advanced Use Cases

   Need custom workflows?

   - [Multi-scale Processing](./advanced/multi_scale.yaml)
   - [Custom Features](./advanced/custom_features.yaml)
   - [Hybrid Processing](./advanced/hybrid_processing.yaml)

   ## Configuration System

   Learn about the v5.5 configuration system:

   - [Hardware Profiles](../ign_lidar/configs/hardware/README.md)
   - [Task Presets](../ign_lidar/configs/task/README.md)
   - [Data Sources](../ign_lidar/configs/data_sources/README.md)
   ```

3. **Create Quickstart Configs**

   **File:** `examples/quickstart/00_minimal.yaml`

   ```yaml
   # Minimal configuration - bare essentials
   # Perfect for: Testing, quick experiments

   input_dir: /data/tiles
   output_dir: /data/output

   processor:
     use_feature_computer: true # Automatic mode selection
     lod_level: LOD2
   # That's it! Everything else uses intelligent defaults
   ```

   **File:** `examples/quickstart/01_cpu_basic.yaml`

   ```yaml
   # Basic CPU configuration
   # Perfect for: Workstations without GPU, baseline processing

   defaults:
     - hardware/cpu_standard # 4 workers, 32GB RAM
     - task/lod2_buildings # Fast building classification
     - _self_

   input_dir: /data/tiles
   output_dir: /data/output
   ```

   **File:** `examples/quickstart/02_gpu_basic.yaml`

   ```yaml
   # Basic GPU configuration
   # Perfect for: RTX 3060+, accelerated processing

   defaults:
     - hardware/gpu_rtx4080 # Or gpu_rtx3080
     - task/asprs_classification
     - _self_

   input_dir: /data/tiles
   output_dir: /data/output
   ```

**Estimated Time:** 4-5 hours  
**Files Created:** 10+ config files + READMEs  
**Files Moved:** 10-15 legacy configs

---

### 3.2 Update Preset Documentation

**Directory:** `ign_lidar/configs/`

**Actions:**

1. **Create Main Config README**

   **File:** `ign_lidar/configs/README.md`

   ```markdown
   # IGN LiDAR HD Configuration System

   Complete guide to the v5.5 hierarchical configuration system.

   ## Overview

   The configuration system uses three layers:

   1. **Base Defaults** - `base_complete.yaml` (rarely modified)
   2. **Hardware Profiles** - GPU/CPU optimization
   3. **Task Presets** - Feature sets and workflows

   ## Hardware Profiles

   Located in `hardware/`:

   - `gpu_rtx4080.yaml` - RTX 4080 (16GB VRAM, 5M batch)
   - `gpu_rtx3080.yaml` - RTX 3080 (10GB VRAM, 3M batch)
   - `cpu_high.yaml` - High-end CPU (64GB RAM, 8 workers)
   - `cpu_standard.yaml` - Standard CPU (32GB RAM, 4 workers)

   ## Task Presets

   Located in `task/`:

   - `asprs_classification.yaml` - Full ASPRS with ground truth
   - `lod2_buildings.yaml` - Fast building classification
   - `lod3_architecture.yaml` - Detailed architectural features
   - `quick_enrich.yaml` - Minimal features for testing

   ## Usage Examples

   [Examples of composing configs]
   ```

2. **Update Individual Preset READMEs**

   Add to each subdirectory (`hardware/`, `task/`, `data_sources/`)

**Estimated Time:** 2-3 hours  
**Files Created:** 4 README files  
**Files Modified:** 0

---

## 🧪 Phase 4: Testing Enhancements (Priority: MEDIUM)

### 4.1 Add Rules Framework Tests

**Directory:** `tests/`

**Actions:**

1. **Create Rules Framework Test File**

   **File:** `tests/test_rules_framework.py`

   ```python
   """
   Tests for rules framework (v3.2.0+).

   Tests cover:
   - BaseRule abstract class
   - RuleEngine execution
   - HierarchicalRuleEngine
   - Confidence scoring methods
   - Rule validation
   """

   import pytest
   import numpy as np
   from ign_lidar.core.classification.rules import (
       BaseRule,
       RuleEngine,
       HierarchicalRuleEngine,
       RuleResult,
       RuleType,
       RulePriority,
   )

   @pytest.mark.unit
   class TestBaseRule:
       """Test BaseRule abstract class."""

       def test_rule_creation(self):
           """Test creating custom rule."""
           # Implementation
           pass

       def test_rule_evaluation(self):
           """Test rule evaluation."""
           pass

   @pytest.mark.unit
   class TestRuleEngine:
       """Test RuleEngine execution."""

       def test_engine_execution(self):
           """Test engine executes rules."""
           pass

       def test_conflict_resolution(self):
           """Test conflict resolution strategies."""
           pass

   @pytest.mark.integration
   class TestHierarchicalRules:
       """Test hierarchical classification."""

       def test_multi_level_execution(self):
           """Test multi-level rule execution."""
           pass

       def test_strategy_selection(self):
           """Test execution strategy selection."""
           pass

   @pytest.mark.unit
   class TestConfidenceScoring:
       """Test confidence calculation methods."""

       def test_linear_confidence(self):
           """Test linear confidence scoring."""
           pass

       def test_sigmoid_confidence(self):
           """Test sigmoid confidence scoring."""
           pass

       # More tests for other confidence methods
   ```

2. **Add Integration Tests**

   **File:** `tests/test_integration_rules.py`

   ```python
   """Integration tests for rules framework with full pipeline."""

   @pytest.mark.integration
   @pytest.mark.slow
   def test_rules_with_processor():
       """Test rules framework integrated with LiDARProcessor."""
       pass

   @pytest.mark.integration
   def test_rules_with_ground_truth():
       """Test rules with ground truth refinement."""
       pass
   ```

**Estimated Time:** 6-8 hours  
**Files Created:** 2 test files  
**Test Coverage Added:** ~200-300 lines

---

### 4.2 Update Legacy Tests

**Actions:**

1. **Review test_modules/ Directory**

   - Identify tests needing updates
   - Update for Phase 1-3 consolidation
   - Fix factory pattern references

2. **Add Missing Test Coverage**
   - Transport module tests
   - Building module tests
   - Threshold module tests

**Estimated Time:** 3-4 hours  
**Files Modified:** 5-10 test files

---

## 📊 Phase 5: Final Polish (Priority: LOW)

### 5.1 Generate API Documentation

**Tools:** Sphinx or pdoc

**Actions:**

1. **Setup Sphinx Configuration**

   ```bash
   cd docs
   sphinx-quickstart
   ```

2. **Configure API Documentation**

   ```python
   # docs/conf.py
   extensions = [
       'sphinx.ext.autodoc',
       'sphinx.ext.napoleon',
       'sphinx.ext.viewcode',
       'sphinx_autodoc_typehints',
   ]
   ```

3. **Generate API Docs**
   ```bash
   sphinx-apidoc -o api/ ../ign_lidar/
   make html
   ```

**Estimated Time:** 4-5 hours  
**Output:** Complete API documentation in HTML

---

### 5.2 Code Quality Improvements

**Actions:**

1. **Run Linters**

   ```bash
   black ign_lidar/ tests/
   flake8 ign_lidar/
   mypy ign_lidar/
   pylint ign_lidar/
   ```

2. **Fix Any Issues Found**

3. **Add Pre-commit Hooks**
   ```yaml
   # .pre-commit-config.yaml
   repos:
     - repo: https://github.com/psf/black
       rev: 22.0.0
       hooks:
         - id: black
     - repo: https://github.com/PyCQA/flake8
       rev: 4.0.0
       hooks:
         - id: flake8
   ```

**Estimated Time:** 2-3 hours  
**Files Modified:** Various (formatting/style fixes)

---

## 📅 Timeline & Resource Allocation

### Week 1: High Priority Items

| Day     | Phase     | Tasks                        | Hours |
| ------- | --------- | ---------------------------- | ----- |
| Mon     | Phase 1   | Package metadata, structure  | 1.5   |
| Mon-Tue | Phase 2.1 | Docusaurus updates           | 8     |
| Wed     | Phase 2.2 | Consolidate migration guides | 4     |
| Thu-Fri | Phase 3.1 | Reorganize examples          | 5     |
| Fri     | Phase 3.2 | Update preset docs           | 3     |

**Week 1 Total:** ~21.5 hours

### Week 2: Medium Priority Items

| Day     | Phase     | Tasks                 | Hours |
| ------- | --------- | --------------------- | ----- |
| Mon-Tue | Phase 4.1 | Rules framework tests | 8     |
| Wed     | Phase 4.2 | Update legacy tests   | 4     |
| Thu     | Phase 5.1 | API documentation     | 5     |
| Fri     | Phase 5.2 | Code quality          | 3     |

**Week 2 Total:** ~20 hours

**Total Project Time:** 41.5 hours (~1 week full-time or 2 weeks part-time)

---

## ✅ Success Criteria

### Documentation

- [ ] All v3.2.1 features documented in Docusaurus
- [ ] Unified migration guide created
- [ ] All example configs have README files
- [ ] API documentation generated
- [ ] No broken links in documentation

### Code Quality

- [ ] All linters pass
- [ ] Test coverage >80% for new code
- [ ] No duplicate configuration files
- [ ] Clear preset hierarchy

### User Experience

- [ ] Clear getting started path
- [ ] Easy to find example configs
- [ ] Comprehensive migration guidance
- [ ] Validated configurations

### Package

- [ ] Version numbers correct (3.2.1)
- [ ] Dependencies well-documented
- [ ] CHANGELOG complete and accurate
- [ ] Development workflow documented

---

## 🚀 Deployment Checklist

### Pre-Release

- [ ] All high-priority tasks complete
- [ ] Documentation reviewed and tested
- [ ] Example configs validated
- [ ] Tests passing (>95%)
- [ ] Changelog finalized

### Release v3.3.0

- [ ] Tag release in git: `git tag v3.3.0`
- [ ] Update version in pyproject.toml to 3.3.0
- [ ] Build package: `python -m build`
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Deploy Docusaurus: `cd docs && npm run deploy`
- [ ] Create GitHub release with notes

### Post-Release

- [ ] Announce on GitHub
- [ ] Update README badges
- [ ] Monitor for issues
- [ ] Plan v3.4.0 features

---

## 📝 Notes & Decisions

### Design Decisions

1. **No Breaking Changes:** All changes are additive or organizational
2. **Backward Compatibility:** Legacy configs/imports still work
3. **Progressive Enhancement:** New features opt-in, not mandatory
4. **Clear Migration Path:** Multiple migration guides for different users

### Deferred Items

- Advanced config discovery CLI tools (planned for v3.4.0)
- Complete test coverage (target 90%, currently ~75%)
- Performance benchmarking updates
- Additional GPU optimization documentation

---

## 📞 Support & Questions

For questions about this action plan:

- **GitHub Issues:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Documentation:** https://sducournau.github.io/IGN_LIDAR_HD_DATASET/

---

**Action Plan Created:** October 25, 2025  
**Last Updated:** October 25, 2025  
**Status:** Ready for execution ✅
