# Docusaurus Documentation Update Plan - V5 Harmonization

**Date**: October 17, 2025  
**Scope**: Complete documentation update for V5 configuration system and new features  
**Status**: 📋 **PLANNING**

---

## 🎯 Executive Summary

This plan outlines the comprehensive update of Docusaurus documentation to reflect:

- **V5 Configuration System** (simplified from V4)
- **New Features**: Ground Truth Classification, Tile Stitching, ASPRS/LOD Classification
- **Harmonization**: Remove duplicates, consolidate scattered documentation
- **Improved Structure**: Logical organization, better navigation

---

## 📊 Current State Analysis

### Documentation Gaps Identified

| Feature                         | Current Status          | Required Action                             |
| ------------------------------- | ----------------------- | ------------------------------------------- |
| **Ground Truth Classification** | Partial (fetching only) | Enhance with classification workflow        |
| **Tile Stitching**              | Exists but incomplete   | Update with V5 config, add examples         |
| **ASPRS Classes**               | Scattered in guides     | Create comprehensive reference              |
| **LOD2/LOD3 Classes**           | LOD3 only documented    | Add LOD2, enhance LOD3                      |
| **V5 Configuration**            | Missing                 | Create migration guide, update all examples |
| **Classification Systems**      | Fragmented              | Consolidate into unified guide              |

### Documentation Structure Issues

#### Current Problems:

1. **Duplicate Content**: Multiple files covering same topics in `/docs/features` and `/docs/guides`
2. **Outdated Examples**: Most examples use V3/V4 configuration syntax
3. **Scattered Information**: Classification info spread across 23+ markdown files
4. **Inconsistent Naming**: Mixed conventions (UPPERCASE.md, lowercase.md, kebab-case.md)
5. **Poor Organization**: Sidebar categories don't reflect current feature set

#### Files to Consolidate/Archive:

```
docs/features/
├── BUILDING_CLASSIFICATION_IMPROVEMENTS.md          # Duplicate
├── BUILDING_CLASSIFICATION_QUICK_REF.md             # Duplicate
├── CLASSIFICATION_GRAMMAR_COMPLETE_SUMMARY.md        # Duplicate
├── CLASSIFICATION_QUICK_START.md                     # Duplicate
├── CLASSIFICATION_REFERENCE.md                       # Duplicate
├── CLASSIFICATION_SUMMARY.md                         # Duplicate
└── ... (17 more classification-related files)        # Consolidate

docs/guides/
├── ASPRS_CLASSIFICATION_GUIDE.md                     # Move to reference
├── BUILDING_CLASSIFICATION_QUICK_REFERENCE.md        # Duplicate
├── UNIFIED_SYSTEM_GUIDE.md                           # Outdated
└── classification_normalization.md                   # Consolidate
```

---

## 🔄 Documentation Update Strategy

### Phase 1: Core Feature Documentation (Priority 1)

#### 1.1 Ground Truth Classification

**File**: `docs/docs/features/ground-truth-classification.md` (NEW)

**Content**:

- What is Ground Truth Classification
- WFS Integration with IGN BD TOPO®
- Classification workflow (fetch → label → refine)
- GPU-accelerated classification
- NDVI refinement for vegetation
- Configuration (V5 syntax)
- Python API examples
- CLI usage

**Consolidates**:

- Current `ground-truth-fetching.md`
- Current `ground-truth-ndvi-refinement.md`
- Scattered WFS documentation

#### 1.2 ASPRS Classification System

**File**: `docs/docs/reference/asprs-classification.md` (NEW)

**Content**:

- ASPRS LAS 1.4 specification overview
- Standard classes (0-31)
- Extended classes (32-255) for BD TOPO®
- Class 67 fix (non-standard IGN class)
- BD TOPO → ASPRS mapping table
- Configuration options
- Examples and best practices

**Consolidates**:

- `/ASPRS_CLASSES_REFERENCE.md` (root)
- `docs/guides/ASPRS_CLASSIFICATION_GUIDE.md`
- `docs/guides/ASPRS_QUICK_REFERENCE.md`
- `docs/guides/BD_TOPO_ASPRS_MAPPING_REFERENCE.md`

#### 1.3 LOD Classification System

**File**: `docs/docs/reference/lod-classification.md` (ENHANCED from lod3-classification.md)

**Content**:

- LOD levels overview (LOD0-LOD4)
- **LOD2 Classes** (15 classes - NEW section)
  - Building-focused taxonomy
  - Wall, roof types, details
  - Context classes
- **LOD3 Classes** (30 classes - ENHANCED section)
  - Extended building taxonomy
  - Architectural details (windows, doors, etc.)
  - Mapping from ASPRS
- Configuration for LOD2/LOD3 modes
- Use cases and applications

**Consolidates**:

- Current `lod3-classification.md`
- `/ign_lidar/classes.py` documentation
- Scattered LOD references

#### 1.4 Tile Stitching (UPDATE)

**File**: `docs/docs/features/tile-stitching.md` (UPDATE EXISTING)

**Updates Needed**:

- Replace V4 config with V5 syntax
- Add configuration reference table
- Enhanced examples with real-world scenarios
- Performance optimization tips
- Integration with boundary-aware features

**Current File**: Mostly complete, needs config update

---

### Phase 2: Configuration Documentation (Priority 1)

#### 2.1 Configuration V5 Guide

**File**: `docs/docs/guides/configuration-v5.md` (NEW)

**Content**:

- V5 overview and benefits (60% complexity reduction)
- Simplified structure (5 base configs vs 14)
- Base configurations explained:
  - `processor.yaml` - Core processing settings
  - `features.yaml` - Feature computation
  - `data_sources.yaml` - BD TOPO, RPG, Cadastre
  - `output.yaml` - Output formats and options
  - `monitoring.yaml` - Logging and metrics
- Preset system (gpu_optimized, asprs_classification, etc.)
- Hardware profiles (RTX 4080, RTX 3080, CPU)
- Examples and best practices

#### 2.2 Migration Guide V4 → V5

**File**: `docs/docs/guides/migration-v4-to-v5.md` (NEW)

**Content**:

- Why migrate to V5
- Breaking changes summary
- Step-by-step migration process
- Parameter mapping table (V4 → V5)
- Common migration issues and solutions
- Migration script usage
- Rollback instructions

**References**: `/MIGRATION_V5_GUIDE.md` (root)

#### 2.3 Update All Configuration Examples

**Files to Update**:

- `docs/docs/reference/config-examples.md` - All examples to V5
- `docs/docs/guides/configuration-system.md` - Update to V5
- `docs/docs/guides/processing-modes.md` - V5 syntax
- All feature docs with config snippets

---

### Phase 3: Reference Documentation (Priority 2)

#### 3.1 Classification Reference Consolidation

**New Structure**:

```
docs/docs/reference/
├── asprs-classification.md          # NEW - ASPRS standard + extensions
├── lod-classification.md            # NEW - LOD2 + LOD3 systems
├── classification-workflow.md       # NEW - End-to-end classification
└── bd-topo-integration.md           # NEW - BD TOPO data sources
```

**Consolidates 23+ scattered files into 4 comprehensive references**

#### 3.2 Classification Workflow Guide

**File**: `docs/docs/reference/classification-workflow.md` (NEW)

**Content**:

- Overview of classification pipeline
- Input → Processing → Output flow
- ASPRS mode workflow
- LOD2 mode workflow
- LOD3 mode workflow
- Ground truth integration
- Quality assurance and validation
- Mermaid diagrams for each workflow

---

### Phase 4: Examples and Tutorials (Priority 2)

#### 4.1 Update Python API Examples

**Files to Update**:

- `docs/docs/api/processor.md` - V5 config examples
- `docs/docs/api/features.md` - V5 feature computation
- `docs/docs/api/configuration.md` - V5 configuration API

#### 4.2 Create New Example Notebooks

**New Files**:

- `docs/docs/examples/ground-truth-classification.md`
- `docs/docs/examples/tile-stitching-workflow.md`
- `docs/docs/examples/asprs-classification.md`
- `docs/docs/examples/lod2-classification.md`
- `docs/docs/examples/lod3-classification.md`

#### 4.3 Update CLI Examples

**Files to Update**:

- `docs/docs/reference/cli-*.md` - All CLI docs with V5 syntax
- `docs/docs/guides/cli-commands.md` - V5 examples

---

### Phase 5: Sidebar Reorganization (Priority 2)

#### New Sidebar Structure

```typescript
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    "intro",

    // 📦 Getting Started
    {
      label: "📦 Getting Started",
      items: [
        "installation/quick-start",
        "installation/gpu-setup",
        "guides/quick-start",
        "guides/getting-started",
      ],
    },

    // ⚙️ Configuration V5 (NEW SECTION)
    {
      label: "⚙️ Configuration V5",
      items: [
        "guides/configuration-v5", // NEW
        "guides/migration-v4-to-v5", // NEW
        "guides/processing-modes",
        "guides/feature-modes-guide",
        "reference/config-examples",
      ],
    },

    // 🚀 Core Features
    {
      label: "🚀 Core Features",
      items: [
        "features/smart-skip",
        "features/format-preferences",
        "features/enriched-laz-only",
        "features/geometric-features",
        "features/feature-modes",
        "features/boundary-aware",
        "features/tile-stitching", // UPDATED
        "features/pipeline-configuration",
      ],
    },

    // 🏗️ Classification Systems (NEW SECTION)
    {
      label: "🏗️ Classification Systems",
      items: [
        "reference/classification-workflow", // NEW
        "reference/asprs-classification", // NEW
        "reference/lod-classification", // NEW
        "features/ground-truth-classification", // NEW
        "reference/bd-topo-integration", // NEW
      ],
    },

    // 🎨 Advanced Features
    {
      label: "🎨 Advanced Features",
      items: [
        "features/rgb-augmentation",
        "features/infrared-augmentation",
        "features/ground-truth-ndvi-refinement", // Keep for NDVI specifics
        "features/axonometry",
        "features/multi-architecture",
        "features/architectural-styles",
      ],
    },

    // ⚡ GPU Acceleration
    {
      label: "⚡ GPU Acceleration",
      items: [
        "gpu/overview",
        "gpu/features",
        "gpu/rgb-augmentation",
        "guides/gpu-acceleration",
      ],
    },

    // 📖 User Guides
    {
      label: "📖 User Guides",
      items: [
        "guides/cli-commands",
        "guides/hydra-cli",
        "guides/auto-params",
        "guides/preprocessing",
        "guides/complete-workflow",
        "guides/regional-processing",
        "guides/performance",
        "guides/troubleshooting",
      ],
    },

    // 📊 Examples & Tutorials (NEW SECTION)
    {
      label: "📊 Examples & Tutorials",
      items: [
        "examples/ground-truth-classification", // NEW
        "examples/tile-stitching-workflow", // NEW
        "examples/asprs-classification", // NEW
        "examples/lod2-classification", // NEW
        "examples/lod3-classification", // NEW
        "tutorials/custom-features",
      ],
    },

    // 🗺️ QGIS & Visualization
    {
      label: "🗺️ QGIS & Visualization",
      items: [
        "guides/qgis-integration",
        "guides/qgis-troubleshooting",
        "guides/visualization",
      ],
    },

    // 💻 CLI Reference
    {
      label: "💻 CLI Reference",
      items: [
        "reference/cli-download",
        "reference/cli-enrich",
        "reference/cli-patch",
        "reference/cli-qgis",
        "reference/cli-verify",
      ],
    },

    // 📚 Technical Reference
    {
      label: "📚 Technical Reference",
      items: [
        "reference/classification-workflow", // NEW
        "reference/asprs-classification", // NEW
        "reference/lod-classification", // NEW
        "reference/bd-topo-integration", // NEW
        "reference/workflow-diagrams",
        "reference/memory-optimization",
        "reference/architectural-styles",
        "reference/historical-analysis",
        "architecture",
        "workflows",
        "mermaid-reference",
      ],
    },

    // 🔌 API Reference
    {
      label: "🔌 API Reference",
      items: [
        "api/processor",
        "api/cli",
        "api/configuration",
        "api/core-module",
        "api/features",
        "api/gpu-api",
        "api/rgb-augmentation",
        "api/architectural-style-api",
        "api/auto-params",
      ],
    },

    // 📝 Release Notes
    {
      label: "📝 Release Notes",
      items: [
        "release-notes/v5.0.0", // NEW
        "release-notes/v3.0.0",
        "release-notes/v1.7.2",
        "release-notes/v1.7.1",
        "release-notes/v1.7.0",
        "release-notes/v1.6.2",
        "release-notes/v1.6.0",
        "release-notes/v1.5.0",
      ],
    },
  ],
};
```

---

### Phase 6: Content Cleanup (Priority 3)

#### 6.1 Archive Deprecated Content

**Move to**: `docs/docs/archive/`

**Files to Archive**:

```
# Duplicate/superseded classification guides (23 files)
docs/features/BUILDING_CLASSIFICATION_*.md
docs/features/CLASSIFICATION_*.md
docs/features/ROADS_RAILWAYS_*.md
docs/features/TRANSPORT_*.md
docs/features/RAILWAYS_*.md
docs/features/POWER_LINE_*.md
docs/features/GRAMMAR_3D_GUIDE.md

# Duplicate guides
docs/guides/BUILDING_CLASSIFICATION_*.md
docs/guides/classification_normalization.md
docs/guides/UNIFIED_SYSTEM_GUIDE.md

# Old migration guides (keep in archive for reference)
docs/guides/migration-v1-to-v2.md
docs/guides/migration-v2-to-v3.md
```

#### 6.2 Consolidate Architecture Docs

**Current**: Scattered in multiple locations
**New**: Single architecture documentation

```
docs/architecture/
├── README.md                              # Overview
├── V5_ARCHITECTURE.md                     # NEW - V5 system design
├── FEATURE_ORCHESTRATOR.md                # Core orchestrator
├── CLASSIFICATION_PIPELINE.md             # NEW - Classification system
└── GPU_OPTIMIZATION.md                    # GPU architecture
```

**Archive**:

```
docs/architecture/CONFIG_*.md              # V4 config docs (archive)
docs/configuration/CONFIG_*.md             # Duplicate (archive)
```

---

### Phase 7: Blog Posts and Release Notes (Priority 3)

#### 7.1 Create V5 Release Blog Post

**File**: `docs/blog/2025-10-17-v5-configuration-release.md` (NEW)

**Content**:

- V5 configuration system highlights
- 60% complexity reduction
- New features overview (Ground Truth, Tile Stitching)
- Classification systems (ASPRS, LOD2, LOD3)
- Migration guide summary
- Performance improvements
- Breaking changes notice

#### 7.2 Update Release Notes

**File**: `docs/docs/release-notes/v5.0.0.md` (NEW)

**Content**:

- Complete changelog from CHANGELOG.md
- V5 configuration system
- New features detailed
- Breaking changes
- Migration instructions
- Known issues
- Upgrade path

---

## 📋 Implementation Checklist

### Phase 1: Core Features (Week 1)

- [ ] Create `ground-truth-classification.md` (consolidate fetching + NDVI)
- [ ] Create `asprs-classification.md` reference
- [ ] Enhance `lod-classification.md` (add LOD2, enhance LOD3)
- [ ] Update `tile-stitching.md` to V5

### Phase 2: Configuration (Week 1)

- [ ] Create `configuration-v5.md` guide
- [ ] Create `migration-v4-to-v5.md` guide
- [ ] Update all config examples to V5 syntax
- [ ] Update processing modes guide

### Phase 3: Reference Docs (Week 2)

- [ ] Create `classification-workflow.md`
- [ ] Create `bd-topo-integration.md`
- [ ] Consolidate scattered classification docs
- [ ] Update API reference docs

### Phase 4: Examples (Week 2)

- [ ] Create 5 new example docs (ground truth, stitching, classifications)
- [ ] Update Python API examples
- [ ] Update CLI examples
- [ ] Create tutorial notebooks

### Phase 5: Sidebar & Structure (Week 3)

- [ ] Update `sidebars.ts` with new structure
- [ ] Reorganize categories
- [ ] Add new sections
- [ ] Update navigation

### Phase 6: Cleanup (Week 3)

- [ ] Archive 30+ duplicate/deprecated files
- [ ] Consolidate architecture docs
- [ ] Remove broken links
- [ ] Clean up directory structure

### Phase 7: Release Content (Week 3)

- [ ] Create V5 release blog post
- [ ] Create V5.0.0 release notes
- [ ] Update main README
- [ ] Update changelog

### Phase 8: Validation (Week 4)

- [ ] Build Docusaurus site locally
- [ ] Check all internal links
- [ ] Validate code examples
- [ ] Test all diagrams render
- [ ] Cross-browser testing
- [ ] Deploy to GitHub Pages

---

## 📊 Impact Assessment

### Before Update

- **Config Complexity**: V4 with 14 base configs, 200+ parameters
- **Documentation Files**: 180+ markdown files (many duplicates)
- **Classification Docs**: Scattered across 23+ files
- **Examples**: Mostly V3/V4 syntax
- **User Experience**: Confusing, hard to find information

### After Update

- **Config Complexity**: V5 with 5 base configs, 80 parameters (60% reduction)
- **Documentation Files**: ~100 well-organized files
- **Classification Docs**: 4 comprehensive references
- **Examples**: All V5 syntax, new feature examples
- **User Experience**: Clear, logical, easy navigation

---

## 🎯 Success Metrics

1. **Documentation Coverage**

   - ✅ 100% of V5 features documented
   - ✅ All new features have examples
   - ✅ Zero broken internal links
   - ✅ All code examples validated

2. **Content Quality**

   - ✅ No duplicate content
   - ✅ Consistent formatting and style
   - ✅ Up-to-date configuration syntax
   - ✅ Comprehensive classification references

3. **User Experience**

   - ✅ Logical sidebar organization
   - ✅ Clear navigation paths
   - ✅ Quick start to advanced flow
   - ✅ Easy-to-find references

4. **Maintainability**
   - ✅ Reduced file count (180 → 100)
   - ✅ Single source of truth for each topic
   - ✅ Archived historical content
   - ✅ Clear documentation structure

---

## 🚀 Timeline

| Week       | Phase                  | Deliverables                        |
| ---------- | ---------------------- | ----------------------------------- |
| **Week 1** | Core Features + Config | 7 new/updated docs, V5 config guide |
| **Week 2** | Reference + Examples   | 4 reference docs, 5 example docs    |
| **Week 3** | Structure + Cleanup    | New sidebar, archived 30+ files     |
| **Week 4** | Release + Validation   | Blog post, release notes, testing   |

**Total Duration**: 4 weeks  
**Start Date**: October 17, 2025  
**Target Completion**: November 14, 2025

---

## 📝 File Naming Conventions

### New Standards

- **Features**: `kebab-case.md` (e.g., `ground-truth-classification.md`)
- **Guides**: `kebab-case.md` (e.g., `configuration-v5.md`)
- **Reference**: `kebab-case.md` (e.g., `asprs-classification.md`)
- **API**: `kebab-case.md` (e.g., `processor-api.md`)
- **Examples**: `kebab-case.md` (e.g., `tile-stitching-workflow.md`)

### Archive Naming

- **Format**: `YYYY-MM-DD_original-name.md`
- **Example**: `2025-10-17_BUILDING_CLASSIFICATION_IMPROVEMENTS.md`

---

## ✅ Validation Checklist

### Pre-Deployment

- [ ] All markdown files render correctly
- [ ] All code blocks have proper syntax highlighting
- [ ] All diagrams (Mermaid) render correctly
- [ ] All internal links work
- [ ] All external links are valid
- [ ] All configuration examples are V5 compliant
- [ ] All Python examples run successfully
- [ ] All CLI examples are correct
- [ ] Sidebar navigation is logical
- [ ] Search functionality works
- [ ] Mobile responsiveness verified

### Post-Deployment

- [ ] GitHub Pages deployment successful
- [ ] All routes accessible
- [ ] Images and assets load correctly
- [ ] No console errors
- [ ] Analytics tracking works
- [ ] User feedback collected
- [ ] Known issues documented

---

## 📚 Reference Materials

### Key Source Files

- `/HARMONIZATION_AUDIT_REPORT_V5.md` - V5 harmonization details
- `/CONFIG_V5_CONSOLIDATION_REPORT.md` - V5 config consolidation
- `/CONFIG_V5_IMPROVEMENTS_SUMMARY.md` - Recent improvements
- `/MIGRATION_V5_GUIDE.md` - V4 → V5 migration
- `/ASPRS_CLASSES_REFERENCE.md` - ASPRS classification
- `/ign_lidar/classes.py` - LOD2/LOD3 classes
- `/ign_lidar/asprs_classes.py` - ASPRS implementation

### Documentation Standards

- Follow Docusaurus best practices
- Use Mermaid for diagrams
- Include runnable code examples
- Add frontmatter to all docs
- Use proper markdown formatting
- Include table of contents for long docs

---

## 🎉 Expected Outcomes

1. **Simplified Documentation**

   - 60% reduction in duplicate content
   - Clear, logical structure
   - Easy to navigate and find information

2. **Comprehensive Coverage**

   - All V5 features documented
   - Complete classification references
   - Extensive examples and tutorials

3. **Improved User Experience**

   - Quick start to advanced learning path
   - Consistent formatting and style
   - Up-to-date examples and references

4. **Better Maintainability**
   - Single source of truth for each topic
   - Archived historical content
   - Clear contribution guidelines

---

**Plan Prepared**: October 17, 2025  
**Next Steps**: Begin Phase 1 implementation  
**Status**: ✅ **READY FOR IMPLEMENTATION**
