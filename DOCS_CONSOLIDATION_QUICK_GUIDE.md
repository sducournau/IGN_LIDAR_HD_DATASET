# Documentation Consolidation - Quick Implementation Guide

**For:** Documentation Maintainer  
**Time Required:** 6-8 hours (Week 1 critical path)  
**Date:** October 3, 2025

---

## 🎯 Goal

Consolidate 3 fragmented GPU guides into a single, coherent GPU section with proper navigation.

---

## 📋 Prerequisites

- [x] Git repository access
- [x] Node.js and npm installed
- [x] Familiarity with Markdown
- [x] Understanding of Docusaurus structure

---

## ⚡ Quick Start (15 minutes)

### Step 1: Create Branch & Backup

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET
git checkout -b docs-consolidation-gpu
git push origin docs-consolidation-gpu

# Backup current docs
cp -r website/docs website/docs.backup.$(date +%Y%m%d)
```

### Step 2: Create GPU Directory Structure

```bash
# Create new GPU section
mkdir -p website/docs/gpu

# Placeholder files (we'll populate these next)
touch website/docs/gpu/overview.md
touch website/docs/gpu/features.md

# Move RGB GPU guide to new location
mv website/docs/rgb-gpu-guide.md website/docs/gpu/rgb-augmentation.md
```

### Step 3: Test Structure

```bash
cd website
npm install  # If first time
npm run start
```

Open <http://localhost:3000> - verify site builds (placeholders will be empty).

---

## 📝 Content Consolidation (4-5 hours)

### Task 1: Create `gpu/overview.md`

**Source files:**

- `website/docs/gpu-guide.md` (lines 1-200)
- `website/docs/guides/gpu-acceleration.md` (lines 1-100)

**Template:**

```markdown
---
sidebar_position: 1
title: "GPU Acceleration Overview"
description: "Setup and use GPU acceleration for faster LiDAR processing"
keywords: [gpu, cuda, cupy, performance, acceleration]
---

# GPU Acceleration Overview

**Available in:** v1.3.0+  
**Performance:** 5-10x faster than CPU  
**Requirements:** NVIDIA GPU, CUDA 11.0+

## Overview

[Merge introduction from both files - focus on benefits]

## Requirements

[From gpu-guide.md lines ~15-25]

- Hardware: NVIDIA GPU with CUDA support
- Software: CUDA Toolkit 11.0+
- Python: 3.8+

## Installation

[From gpu-guide.md lines ~26-75]

### Step 1: Check CUDA

### Step 2: Install CUDA Toolkit

### Step 3: Install Python Dependencies

### Step 4: Verify Installation

## Quick Start

[Combine CLI and API examples from both files]

### Command Line

### Python API

## When to Use GPU

[NEW section - decision tree]

✅ Use GPU for:

- Large point clouds (>100K points)
- Batch processing
- Production pipelines

❌ Use CPU for:

- Small point clouds (<10K points)
- One-off tasks
- Systems without NVIDIA GPU

## Configuration

[From guides/gpu-acceleration.md]

## See Also

- [GPU Features](features.md) - Feature computation details
- [RGB GPU](rgb-augmentation.md) - RGB acceleration
- [Architecture](../architecture.md) - System architecture
```

**Estimated time:** 1.5 hours

### Task 2: Create `gpu/features.md`

**Source:** `website/docs/gpu-guide.md` (lines 200-580)

**Template:**

```markdown
---
sidebar_position: 2
title: "GPU Feature Computation"
description: "Technical details of GPU-accelerated feature extraction"
keywords: [gpu, features, performance, cupy, benchmarks]
---

# GPU Feature Computation

**Available in:** v1.3.0+  
**Acceleration:** 5-10x speedup

## Features Accelerated

[From gpu-guide.md lines ~200-250]

### Core Features

- ✅ Surface normals
- ✅ Curvature
- ✅ Height above ground

### Geometric Features

- ✅ Planarity
- ✅ Linearity
- ✅ Sphericity
  [etc.]

## Performance Benchmarks

[From gpu-guide.md lines ~250-320]

### Synthetic Benchmarks

### Real-World Performance

### GPU Models Tested

## API Reference

[From gpu-guide.md lines ~320-400]

### GPUFeatureComputer Class

### Methods

### Parameters

## Troubleshooting

[From gpu-guide.md lines ~400-520]

### Common Issues

### Performance Tips

### Memory Management

## Advanced Topics

[From gpu-guide.md lines ~520-580]

### Batch Processing

### Multi-GPU (Future)

### Mixed Precision (Future)

## See Also

- [GPU Overview](overview.md) - Setup guide
- [RGB GPU](rgb-augmentation.md) - RGB acceleration
```

**Estimated time:** 1.5 hours

### Task 3: Update `gpu/rgb-augmentation.md`

**Source:** Already moved to `website/docs/gpu/rgb-augmentation.md`

**Updates needed:**

```markdown
---
sidebar_position: 3
title: "GPU RGB Augmentation"
description: "24x faster RGB augmentation with GPU acceleration"
keywords: [gpu, rgb, orthophoto, color, performance]
---

# GPU-Accelerated RGB Augmentation

**Available in:** v1.5.0+  
**Performance:** 24x faster than CPU  
**Requirements:** NVIDIA GPU, CuPy

[Keep existing content - just update frontmatter and links]

## See Also

- [GPU Overview](overview.md) - Setup GPU acceleration
- [GPU Features](features.md) - Feature computation
- [RGB Augmentation (CPU)](../features/rgb-augmentation.md) - CPU version
```

**Changes:**

1. Update frontmatter (add `sidebar_position: 3`)
2. Fix relative links:
   - `gpu-guide.md` → `overview.md`
   - `../GPU_PHASE3_PLAN.md` → `overview.md#future-features`
3. Add "See Also" section with cross-references

**Estimated time:** 30 minutes

---

## 🔧 Update Sidebar Configuration (30 minutes)

### Edit `website/sidebars.ts`

**Replace:**

```typescript
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    "intro",
    {
      type: "category",
      label: "Installation",
      items: ["installation/quick-start"],
    },
    {
      type: "category",
      label: "User Guides",
      items: [
        "guides/basic-usage",
        "guides/cli-commands",
        "guides/qgis-integration",
      ],
    },
```

**With:**

```typescript
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    "intro",

    {
      type: "category",
      label: "Installation",
      items: ["installation/quick-start"],
    },

    {
      type: "category",
      label: "User Guides",
      items: ["guides/basic-usage", "guides/cli-commands"],
    },

    // Promote core docs
    "architecture",
    "workflows",

    // NEW: GPU Section
    {
      type: "category",
      label: "GPU Acceleration",
      items: ["gpu/overview", "gpu/features", "gpu/rgb-augmentation"],
    },

    {
      type: "category",
      label: "Features",
      items: [
        "features/smart-skip",
        "features/format-preferences",
        "features/lod3-classification",
        "features/rgb-augmentation",
        "features/pipeline-configuration",
      ],
    },

    {
      type: "category",
      label: "QGIS Integration",
      items: ["guides/qgis-integration", "guides/qgis-troubleshooting"],
    },

    {
      type: "category",
      label: "Technical Reference",
      items: [
        "reference/memory-optimization",
        "mermaid-reference",
        "api/processor",
      ],
    },

    {
      type: "category",
      label: "Release Notes",
      items: ["release-notes/v1.5.0"],
    },
  ],
};
```

---

## 🔍 Update Cross-References (1 hour)

### Files to Update

#### 1. `architecture.md`

Find:

```markdown
### GPU Acceleration (New in v1.5.0)
```

Add after:

```markdown
See [GPU Acceleration Guide](gpu/overview.md) for complete setup instructions.
```

#### 2. `workflows.md`

Find:

```markdown
See [GPU Guide](gpu-guide.md) for detailed setup instructions.
```

Replace with:

```markdown
See [GPU Overview](gpu/overview.md) for detailed setup instructions.
```

#### 3. `features/rgb-augmentation.md`

Add at end:

```markdown
## GPU Acceleration

For 24x faster RGB augmentation, see [GPU RGB Guide](../gpu/rgb-augmentation.md).
```

---

## ✅ Testing Checklist (1 hour)

### 1. Local Development

```bash
cd website
npm run start
```

**Test:**

- [ ] Site builds without errors
- [ ] All pages accessible
- [ ] GPU section appears in sidebar
- [ ] Navigation works (click through all links)
- [ ] No broken internal links
- [ ] Images load correctly
- [ ] Mermaid diagrams render

### 2. Production Build

```bash
npm run build
npm run serve
```

**Test:**

- [ ] Production build completes
- [ ] Site works on <http://localhost:3000>
- [ ] Search works (if enabled)
- [ ] Mobile responsive

### 3. Link Validation

```bash
# Install link checker
npm install -g markdown-link-check

# Check all markdown files
find website/docs -name "*.md" -exec markdown-link-check {} \;
```

**Fix any broken links**

---

## 🚀 Deployment (15 minutes)

### Commit Changes

```bash
git add website/docs/gpu/
git add website/sidebars.ts
git add website/docs/architecture.md
git add website/docs/workflows.md
git add website/docs/features/rgb-augmentation.md

git commit -m "docs: consolidate GPU documentation into unified section

- Create new gpu/ directory with 3 consolidated guides
- Merge gpu-guide.md + guides/gpu-acceleration.md → gpu/overview.md
- Extract feature details → gpu/features.md
- Move rgb-gpu-guide.md → gpu/rgb-augmentation.md
- Update sidebar with GPU section
- Add cross-references between docs
- Fix duplicate sidebar positions

Closes #XXX"
```

### Archive Old Files

```bash
mkdir -p website/docs/.archive
mv website/docs/gpu-guide.md website/docs/.archive/
mv website/docs/guides/gpu-acceleration.md website/docs/.archive/

git add .
git commit -m "docs: archive old GPU guides (content consolidated)"
```

### Deploy to GitHub Pages

```bash
cd website
npm run deploy
```

### Verify Production

1. Go to <https://sducournau.github.io/IGN_LIDAR_HD_DATASET/>
2. Check GPU section appears
3. Click through all GPU docs
4. Verify no 404 errors

---

## 🎉 Success Criteria

- ✅ GPU section appears in sidebar
- ✅ All 3 GPU guides accessible
- ✅ No broken links
- ✅ Cross-references work
- ✅ Old guides archived (not deleted)
- ✅ Site builds and deploys successfully
- ✅ Mobile responsive
- ✅ Search works (if enabled)

---

## 🆘 Troubleshooting

### Issue: Broken Links

**Solution:** Use relative paths from file location

```markdown
<!-- From gpu/overview.md -->

[Architecture](../architecture.md) ✅
[Architecture](architecture.md) ❌
```

### Issue: Sidebar Not Updating

**Solution:** Restart dev server

```bash
# Stop server (Ctrl+C)
npm run start  # Restart
```

### Issue: Mermaid Diagrams Not Rendering

**Solution:** Check `docusaurus.config.ts` has Mermaid enabled

```typescript
markdown: {
  mermaid: true,
},
themes: ['@docusaurus/theme-mermaid'],
```

### Issue: Deployment Fails

**Solution:** Check build logs

```bash
npm run build  # Should show errors
```

Common issues:

- Broken links (fix them)
- Invalid frontmatter (check YAML syntax)
- Missing images (check file paths)

---

## 📚 Reference Files

- **Detailed Analysis:** `DOCUSAURUS_ANALYSIS.md`
- **Executive Summary:** `DOCS_CONSOLIDATION_SUMMARY.md`
- **This Guide:** `DOCS_CONSOLIDATION_QUICK_GUIDE.md`

---

## ⏱️ Time Tracking

| Task                    | Estimated  | Actual |
| ----------------------- | ---------- | ------ |
| Setup & Backup          | 15 min     |        |
| Create overview.md      | 1.5 hours  |        |
| Create features.md      | 1.5 hours  |        |
| Update rgb-augmentation | 30 min     |        |
| Update sidebars.ts      | 30 min     |        |
| Update cross-references | 1 hour     |        |
| Testing                 | 1 hour     |        |
| Deployment              | 15 min     |        |
| **Total**               | **6-8 hr** |        |

---

## ✅ Completion Checklist

- [ ] Branch created
- [ ] Backup created
- [ ] GPU directory created
- [ ] overview.md written
- [ ] features.md written
- [ ] rgb-augmentation.md updated
- [ ] sidebars.ts updated
- [ ] Cross-references added
- [ ] Local testing passed
- [ ] Production build successful
- [ ] Deployed to GitHub Pages
- [ ] Verified in production
- [ ] Old files archived
- [ ] PR created (if using PRs)
- [ ] Documentation reviewed

---

**Good luck! 🚀**
