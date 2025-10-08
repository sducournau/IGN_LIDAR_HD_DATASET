# Documentation Update Checklist - v2.0.0

Quick reference checklist for updating Docusaurus English documentation.

---

## 🔴 CRITICAL - Week 1

### Version & Release Information

- [ ] **intro.md** - Change version from 1.7.6 to 2.0.0
- [ ] **intro.md** - Update "What's New" section with v2.0.0 features
- [ ] **release-notes/v2.0.0.md** - Create new release notes file
- [ ] **release-notes/v2.0.0.md** - Document architecture overhaul
- [ ] **release-notes/v2.0.0.md** - Document Hydra CLI
- [ ] **release-notes/v2.0.0.md** - Document unified pipeline

### CLI Documentation

- [ ] **guides/hydra-cli.md** - Create new Hydra CLI guide
- [ ] **guides/hydra-cli.md** - Document `ign-lidar-hd` command
- [ ] **guides/hydra-cli.md** - Document `process`, `verify`, `info` subcommands
- [ ] **guides/hydra-cli.md** - Add configuration override examples
- [ ] **api/cli.md** - Add section for Hydra CLI
- [ ] **api/cli.md** - Document dual CLI system (v1 legacy + v2 Hydra)
- [ ] **api/cli.md** - Update command examples

### Architecture Documentation

- [ ] **architecture.md** - Update system architecture diagram
- [ ] **architecture.md** - Document new `core/` module
- [ ] **architecture.md** - Document new `config/` module
- [ ] **architecture.md** - Document new `preprocessing/` module
- [ ] **architecture.md** - Document new `io/` module
- [ ] **architecture.md** - Document restructured `cli/` module
- [ ] **architecture.md** - Update module interaction diagrams

### Migration Guide

- [ ] **guides/migration-v1-to-v2.md** - Create migration guide
- [ ] **guides/migration-v1-to-v2.md** - Document CLI changes
- [ ] **guides/migration-v1-to-v2.md** - Document API import changes
- [ ] **guides/migration-v1-to-v2.md** - Document configuration changes
- [ ] **guides/migration-v1-to-v2.md** - Add before/after examples
- [ ] **guides/migration-v1-to-v2.md** - Document breaking changes

---

## 🟡 HIGH PRIORITY - Week 2

### Configuration System

- [ ] **guides/configuration-system.md** - Create configuration guide
- [ ] **guides/configuration-system.md** - Document Hydra basics
- [ ] **guides/configuration-system.md** - Document config composition
- [ ] **guides/configuration-system.md** - Document config overrides
- [ ] **api/configuration.md** - Update with Hydra YAML structure
- [ ] **api/configuration.md** - Document all preset configs
- [ ] **api/configuration.md** - Document `processor/` presets
- [ ] **api/configuration.md** - Document `features/` presets
- [ ] **api/configuration.md** - Document `stitching/` presets
- [ ] **api/configuration.md** - Document `experiment/` configs

### New Features Documentation

- [ ] **features/boundary-aware.md** - Create boundary features guide
- [ ] **features/boundary-aware.md** - Document cross-tile computation
- [ ] **features/boundary-aware.md** - Add usage examples
- [ ] **features/boundary-aware.md** - Add performance impact
- [ ] **features/tile-stitching.md** - Create stitching guide
- [ ] **features/tile-stitching.md** - Document buffer zones
- [ ] **features/tile-stitching.md** - Document configuration options
- [ ] **features/tile-stitching.md** - Add examples
- [ ] **features/multi-architecture.md** - Create multi-arch guide
- [ ] **features/multi-architecture.md** - Document supported architectures
- [ ] **features/multi-architecture.md** - Document output formats
- [ ] **features/multi-architecture.md** - Add usage examples

### Pipeline & Workflows

- [ ] **workflows.md** - Update with unified pipeline
- [ ] **workflows.md** - Document RAW→Patches single-step
- [ ] **workflows.md** - Remove old multi-step references
- [ ] **workflows.md** - Update workflow diagrams
- [ ] **workflows.md** - Add performance comparisons
- [ ] **guides/unified-pipeline.md** - Create unified pipeline guide
- [ ] **guides/unified-pipeline.md** - Document in-memory processing
- [ ] **guides/unified-pipeline.md** - Document space savings
- [ ] **guides/unified-pipeline.md** - Add examples

---

## 🟢 MEDIUM PRIORITY - Week 3

### API Reference Updates

- [ ] **api/core-module.md** - Create core module API doc
- [ ] **api/core-module.md** - Document `LiDARProcessor`
- [ ] **api/core-module.md** - Document `TileStitcher`
- [ ] **api/core-module.md** - Document `MemoryManager`
- [ ] **api/core-module.md** - Document `PerformanceMonitor`
- [ ] **api/preprocessing-module.md** - Create preprocessing API doc
- [ ] **api/preprocessing-module.md** - Document preprocessing functions
- [ ] **api/preprocessing-module.md** - Document tile analyzer
- [ ] **api/config-module.md** - Create config module API doc
- [ ] **api/config-module.md** - Document config schema
- [ ] **api/config-module.md** - Document config classes

### Update Existing API Docs

- [ ] **api/processor.md** - Update import paths
- [ ] **api/processor.md** - Update `from ign_lidar import` → `from ign_lidar.core import`
- [ ] **api/processor.md** - Document new processor parameters
- [ ] **api/processor.md** - Update examples
- [ ] **api/features.md** - Add boundary-aware features
- [ ] **api/features.md** - Update feature list
- [ ] **api/features.md** - Add feature computation modes
- [ ] **api/gpu-api.md** - Document chunked processing
- [ ] **api/gpu-api.md** - Update GPU memory management
- [ ] **api/gpu-api.md** - Add performance benchmarks

### GPU Documentation

- [ ] **guides/gpu-acceleration.md** - Add v2.0.0 optimizations
- [ ] **guides/gpu-acceleration.md** - Document chunked GPU processing
- [ ] **guides/gpu-acceleration.md** - Update performance benchmarks
- [ ] **guides/gpu-acceleration.md** - Document memory management
- [ ] **guides/gpu-acceleration.md** - Add Hydra GPU config examples

### Quick Start Updates

- [ ] **guides/quick-start.md** - Add note about dual CLI
- [ ] **guides/quick-start.md** - Add Hydra CLI option
- [ ] **guides/quick-start.md** - Update examples for v2.0.0
- [ ] **guides/quick-start.md** - Add configuration examples

---

## 🔵 POLISH - Week 4

### Code Examples

- [ ] **examples/** - Review all code examples
- [ ] **examples/** - Update import statements
- [ ] **examples/** - Add Hydra configuration examples
- [ ] **examples/** - Add multi-architecture examples
- [ ] **examples/** - Test all examples work with v2.0.0

### Cross-References & Links

- [ ] Search all docs for version references
- [ ] Update all internal links
- [ ] Fix any broken references
- [ ] Ensure consistency across docs
- [ ] Update navigation/sidebar if needed

### Diagrams & Visuals

- [ ] Update architecture diagrams
- [ ] Update workflow diagrams
- [ ] Update data flow diagrams
- [ ] Add new diagrams for:
  - [ ] Boundary-aware processing
  - [ ] Tile stitching
  - [ ] Unified pipeline
  - [ ] Configuration hierarchy

### Testing & Validation

- [ ] Test all CLI commands documented
- [ ] Test all code examples
- [ ] Verify all imports work
- [ ] Check all links work
- [ ] Spell check all documents
- [ ] Grammar check all documents

### User Experience

- [ ] Add migration tips throughout docs
- [ ] Add "New in v2.0.0" badges/callouts
- [ ] Add comparison tables (v1 vs v2)
- [ ] Ensure smooth learning path
- [ ] Add troubleshooting for migration issues

---

## 📋 File-by-File Checklist

### Files to UPDATE

- [ ] `intro.md` - Version, features, quick start
- [ ] `architecture.md` - Complete rewrite for v2.0.0
- [ ] `workflows.md` - Unified pipeline
- [ ] `api/cli.md` - Dual CLI system
- [ ] `api/configuration.md` - Hydra system
- [ ] `api/processor.md` - Import paths, new params
- [ ] `api/features.md` - New features list
- [ ] `api/gpu-api.md` - Chunked processing
- [ ] `guides/quick-start.md` - v2.0.0 commands
- [ ] `guides/gpu-acceleration.md` - v2.0.0 optimizations
- [ ] `guides/performance.md` - Updated benchmarks

### Files to CREATE

- [ ] `release-notes/v2.0.0.md`
- [ ] `guides/hydra-cli.md`
- [ ] `guides/configuration-system.md`
- [ ] `guides/migration-v1-to-v2.md`
- [ ] `guides/unified-pipeline.md`
- [ ] `features/boundary-aware.md`
- [ ] `features/tile-stitching.md`
- [ ] `features/multi-architecture.md`
- [ ] `api/core-module.md`
- [ ] `api/preprocessing-module.md`
- [ ] `api/config-module.md`

---

## 🎯 Quick Commands

### Find version references

```bash
grep -r "1.7.6" website/docs/
grep -r "v1\\.7" website/docs/
```

### Find old import patterns

```bash
grep -r "from ign_lidar import" website/docs/
grep -r "ign-lidar-hd enrich" website/docs/
```

### Find CLI command references

```bash
grep -r "ign-lidar-hd" website/docs/ | grep -v "ign-lidar-hd"
```

---

## ✅ Completion Criteria

Documentation update is complete when:

- [ ] All version numbers show 2.0.0
- [ ] All CLI commands reference correct version
- [ ] All import statements use v2.0.0 paths
- [ ] All code examples tested and working
- [ ] All new features documented
- [ ] All diagrams updated
- [ ] All links working
- [ ] Migration guide complete
- [ ] User can complete workflow from docs alone
- [ ] No references to outdated v1.x patterns (except in migration guide)

---

**Last Updated:** October 8, 2025  
**Target Completion:** November 5, 2025 (4 weeks)
