import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    "intro",

    // Getting Started Section
    {
      type: "category",
      label: "📦 Getting Started",
      items: [
        "installation/quick-start",
        "installation/gpu-setup",
        "guides/quick-start",
        "guides/getting-started",
        "guides/basic-usage",
      ],
    },

    // User Guides (Moved earlier for better UX)
    {
      type: "category",
      label: "📖 User Guides",
      items: [
        "guides/rge-alti-integration",
        "guides/plane-detection",
        "guides/wall-detection",
        "guides/variable-object-filtering",
        "guides/cli-commands",
        "guides/hydra-cli",
        "guides/auto-params",
        "guides/preprocessing",
        "guides/complete-workflow",
        "guides/unified-pipeline",
        "guides/regional-processing",
        "guides/performance",
        "guides/troubleshooting",
      ],
    },

    // Configuration V5 (NEW SECTION)
    {
      type: "category",
      label: "⚙️ Configuration V5",
      items: [
        "guides/configuration-v5",
        "guides/migration-v4-to-v5",
        "guides/processing-modes",
        "guides/feature-modes-guide",
        "reference/config-examples",
      ],
    },

    // Core Features
    {
      type: "category",
      label: "🎯 Core Features",
      items: [
        "features/adaptive-classification",
        "features/building-analysis",
        "features/road-classification",
        "features/rules-framework", // NEW: Rules Framework
        "features/smart-skip",
        "features/format-preferences",
        "features/enriched-laz-only",
        "features/geometric-features",
        "features/feature-modes",
        "features/boundary-aware",
        "features/tile-stitching",
        "features/pipeline-configuration",
      ],
    },

    // Classification Systems (NEW SECTION)
    {
      type: "category",
      label: "🏗️ Classification Systems",
      items: [
        "reference/classification-workflow",
        "reference/asprs-classification",
        "reference/lod-classification",
        "features/ground-truth-classification",
        "reference/bd-topo-integration",
      ],
    },

    // Advanced Features
    {
      type: "category",
      label: "🎨 Advanced Features",
      items: [
        "features/rgb-augmentation",
        "features/infrared-augmentation",
        "features/ground-truth-ndvi-refinement",
        "features/axonometry",
        "features/multi-architecture",
        "features/architectural-styles",
      ],
    },

    // GPU Acceleration
    {
      type: "category",
      label: "⚡ GPU Acceleration",
      items: [
        "gpu/overview",
        "gpu/features",
        "gpu/rgb-augmentation",
        "guides/gpu-acceleration",
      ],
    },

    // Examples & Tutorials (NEW SECTION)
    {
      type: "category",
      label: "📊 Examples & Tutorials",
      items: [
        "examples/ground-truth-classification-example",
        "examples/tile-stitching-example",
        "examples/asprs-classification-example",
        "examples/lod2-classification-example",
        "tutorials/custom-features",
      ],
    },

    // User Guides
    {
      type: "category",
      label: "📖 User Guides",
      items: [
        "guides/cli-commands",
        "guides/hydra-cli",
        "guides/auto-params",
        "guides/preprocessing",
        "guides/complete-workflow",
        "guides/unified-pipeline",
        "guides/regional-processing",
        "guides/performance",
        "guides/troubleshooting",
      ],
    },

    // QGIS Integration
    {
      type: "category",
      label: "🗺️ QGIS & Visualization",
      items: [
        "guides/qgis-integration",
        "guides/qgis-troubleshooting",
        "guides/visualization",
      ],
    },

    // CLI Reference
    {
      type: "category",
      label: "💻 CLI Reference",
      items: [
        "reference/cli-download",
        "reference/cli-enrich",
        "reference/cli-patch",
        "reference/cli-qgis",
        "reference/cli-verify",
      ],
    },

    // Technical Reference
    {
      type: "category",
      label: "📚 Technical Reference",
      items: [
        "reference/dtm-processing",
        "architecture",
        "workflows",
        "reference/workflow-diagrams",
        "reference/memory-optimization",
        "reference/classification-taxonomy",
        "reference/architectural-styles",
        "reference/historical-analysis",
        "guides/data-sources",
        "mermaid-reference",
      ],
    },

    // API Reference
    {
      type: "category",
      label: "🔌 API Reference",
      items: [
        "api/processor",
        "api/cli",
        "api/configuration",
        "api/core-module",
        "api/features",
        "api/rules", // NEW: Rules API
        "api/gpu-api",
        "api/rgb-augmentation",
        "api/architectural-style-api",
      ],
    },

    // Legacy & Migration
    {
      type: "category",
      label: "🔄 Legacy & Migration",
      items: [
        "guides/configuration-system",
        "features/lod3-classification",
        "features/ground-truth-fetching",
      ],
    },

    // Release Notes
    {
      type: "category",
      label: "📝 Release Notes",
      items: [
        "release-notes/v3.3.5", // Maintenance release
        "release-notes/v3.3.4", // CRITICAL: Bug fix + unified filtering
        "release-notes/v3.3.3", // Performance optimizations
        "release-notes/v3.2.1", // Rules framework
        "release-notes/v3.1.0", // Unified feature filtering
        "release-notes/v3.0.6", // Planarity filtering
        "release-notes/v3.0.0", // Major refactor
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

export default sidebars;
