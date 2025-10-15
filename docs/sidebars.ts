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

    // Core Documentation
    {
      type: "category",
      label: "📖 Core Concepts",
      items: [
        "architecture",
        "workflows",
        "guides/configuration-system",
        "guides/processing-modes",
      ],
    },

    // User Guides
    {
      type: "category",
      label: "🔧 User Guides",
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
        "guides/migration-v1-to-v2",
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

    // Features
    {
      type: "category",
      label: "🚀 Features",
      items: [
        "features/smart-skip",
        "features/format-preferences",
        "features/lod3-classification",
        "features/rgb-augmentation",
        "features/infrared-augmentation",
        "features/pipeline-configuration",
        "features/enriched-laz-only",
        "features/geometric-features",
        "features/feature-modes",
        "features/boundary-aware",
        "features/tile-stitching",
        "features/ground-truth-fetching",
        "features/ground-truth-ndvi-refinement",
        "features/axonometry",
        "features/multi-architecture",
        "features/architectural-styles",
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

    // Tutorials
    {
      type: "category",
      label: "📖 Tutorials",
      items: ["tutorials/custom-features"],
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
        "reference/config-examples",
        "reference/workflow-diagrams",
        "reference/memory-optimization",
        "reference/classification-taxonomy",
        "reference/architectural-styles",
        "reference/historical-analysis",
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
        "api/gpu-api",
        "api/rgb-augmentation",
        "api/architectural-style-api",
        "api/auto-params",
      ],
    },

    // Release Notes
    {
      type: "category",
      label: "📝 Release Notes",
      items: [
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
