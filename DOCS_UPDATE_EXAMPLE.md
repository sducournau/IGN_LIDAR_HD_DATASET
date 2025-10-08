# Documentation Update Example: intro.md

This document shows specific changes needed for `website/docs/intro.md` as an example of the update process.

---

## Current State (v1.7.6)

```markdown
---
slug: /
sidebar_position: 1
title: IGN LiDAR HD Processing Library
---

# IGN LiDAR HD Processing Library

**Version 1.7.6** | Python 3.8+ | MIT License
```

## Updated State (v2.0.0)

```markdown
---
slug: /
sidebar_position: 1
title: IGN LiDAR HD Processing Library
---

# IGN LiDAR HD Processing Library

**Version 2.0.0** | Python 3.8+ | MIT License

:::tip New in v2.0.0
Complete architecture overhaul with Hydra configuration, modular design, and unified pipeline!
See [What's New](#-whats-new-in-v20) and [Migration Guide](/guides/migration-v1-to-v2).
:::
```

---

## Section Updates Needed

### 1. Latest Release Section

**CURRENT:**

```markdown
## 🎉 Latest Release: v1.7.6

### 🚀 MASSIVE Performance Optimization - 100-200x Speedup!

The latest release eliminates a critical bottleneck through **vectorized feature computation**:
```

**SHOULD BE:**

````markdown
## 🎉 Latest Release: v2.0.0

### 🏗️ Complete Architecture Overhaul

Version 2.0.0 brings a fundamental redesign with:

**Major Features:**

- **🎯 Modular Architecture**: Specialized modules for core, features, preprocessing, I/O
- **⚙️ Hydra Configuration**: Hierarchical config management with presets
- **⚡ Unified Pipeline**: Single-step RAW→Patches (35-50% space savings, 2-3x faster)
- **🔗 Boundary-Aware Features**: Cross-tile computation for seamless results
- **🎨 Multi-Architecture Support**: PointNet++, Octree, Transformer, Sparse Conv in one workflow

**New CLI:**

```bash
# Legacy CLI (still supported)
ign-lidar-hd enrich --input-dir data/ --output output/

# New Hydra CLI (recommended)
ign-lidar-hd process input_dir=data/ output_dir=output/
```
````

📖 [Full v2.0.0 Release Notes](/release-notes/v2.0.0) | [Migration Guide](/guides/migration-v1-to-v2)

---

## Previous Releases

### v1.7.6 - Critical Fix & Verification

- 🐛 Fixed verticality computation in GPU chunked processing
- 🔍 Feature verification system with `verify` command

### v1.7.5 - Performance Breakthrough

- 🚀 100-200x faster feature computation through vectorization
- 💯 100% GPU utilization improvements

````

---

### 2. Getting Started Section

**CURRENT:**
```markdown
Process your first tile:

```bash
ign-lidar-hd enrich \
  --input-dir data/raw_tiles \
  --output data/enriched \
  --auto-params \
  --preprocess \
  --add-rgb \
  --add-infrared
````

**SHOULD ADD:**

````markdown
Process your first tile:

**Option 1: Legacy CLI (v1.x compatible)**

```bash
ign-lidar-hd enrich \
  --input-dir data/raw_tiles \
  --output data/enriched \
  --auto-params \
  --preprocess \
  --add-rgb \
  --add-infrared
```
````

**Option 2: New Hydra CLI (v2.0.0 recommended)**

```bash
ign-lidar-hd process \
  input_dir=data/raw_tiles \
  output_dir=data/enriched \
  processor=gpu \
  features=full \
  preprocess=default
```

:::tip Which CLI to use?

- **Legacy CLI** (`ign-lidar-hd`): Compatible with v1.x workflows, stable
- **Hydra CLI** (`ign-lidar-hd`): New features, better configuration, recommended for new projects

See [Hydra CLI Guide](/guides/hydra-cli) for details.
:::

````

---

### 3. Features Section

**ADD THIS SECTION:**
```markdown
### v2.0.0 Architecture

- **🏗️ Modular Design**: Specialized modules (core, features, preprocessing, I/O, config)
- **⚙️ Hydra Configuration**: Hierarchical YAML configs with composition and overrides
- **⚡ Unified Pipeline**: Direct RAW→Patches (eliminates intermediate files)
- **🔗 Boundary-Aware**: Cross-tile feature computation for seamless results
- **🎯 Multi-Architecture**: Support for PointNet++, Octree, Transformer, Sparse Conv
- **📊 Preset Configs**: Pre-configured for common scenarios (fast, GPU, memory-constrained)
- **🔬 Experiment Tracking**: Built-in experiment management with Hydra

### Core Capabilities (Enhanced in v2.0.0)
````

---

### 4. Documentation Structure

**ADD NEW SECTIONS:**

```markdown
📚 **Getting Started**

- [Quick Start](/installation/quick-start) - Get up and running in 5 minutes
- [Hydra CLI Guide](/guides/hydra-cli) - **NEW** - Modern CLI system
- [Migration Guide](/guides/migration-v1-to-v2) - **NEW** - Upgrade from v1.x
- [Configuration System](/guides/configuration-system) - **NEW** - Hydra configs

⚡ **Guides**

- [GPU Acceleration](/guides/gpu-acceleration) - Performance optimization
- [Unified Pipeline](/guides/unified-pipeline) - **NEW** - Single-step workflow
- [Basic Usage](/guides/basic-usage) - Common workflows
- [Advanced Usage](/guides/advanced-usage) - Power user features

🎨 **Features**

- [Boundary-Aware Features](/features/boundary-aware) - **NEW** - Cross-tile processing
- [Tile Stitching](/features/tile-stitching) - **NEW** - Seamless multi-tile
- [Multi-Architecture](/features/multi-architecture) - **NEW** - Multiple ML architectures
- [RGB Augmentation](/features/rgb-augmentation) - Add true color
- [Infrared Augmentation](/features/infrared-augmentation) - NIR and NDVI
- [Auto Parameters](/features/auto-params) - Automatic optimization

🔧 **API Reference**

- [Core Module](/api/core-module) - **NEW** - Main processing engine
- [Config Module](/api/config-module) - **NEW** - Configuration classes
- [Preprocessing Module](/api/preprocessing-module) - **NEW** - Data preparation
- [CLI Commands](/api/cli) - Command-line interface (both CLIs)
- [Python API](/api/features) - Programmatic usage
```

---

## Complete Diff Summary

### Changes Required:

1. **Version badge**: `1.7.6` → `2.0.0`
2. **Add migration banner** at top
3. **Rewrite "Latest Release"** section for v2.0.0
4. **Add "Previous Releases"** subsection
5. **Update "Getting Started"** with dual CLI options
6. **Add v2.0.0 features** section
7. **Update documentation structure** with new guides
8. **Add "New in v2.0.0" badges** throughout
9. **Update all code examples** to show both CLI options
10. **Add links** to new documentation pages

### Estimated Time: 2-3 hours

---

## Testing Checklist

After updating `intro.md`:

- [ ] Version number displays correctly
- [ ] All internal links work
- [ ] Code examples are syntactically correct
- [ ] New sections render properly
- [ ] Callouts/admonitions display correctly
- [ ] Images/diagrams load
- [ ] Mobile view looks good
- [ ] Navigation/sidebar updated if needed

---

## Pattern for Other Files

Use this same approach for other documentation files:

1. **Identify outdated content**
2. **Show current state**
3. **Show updated state**
4. **List specific changes**
5. **Provide testing checklist**

---

**Example Document Purpose:** Demonstrate practical update process  
**Next Files to Update:** architecture.md, api/cli.md, workflows.md
