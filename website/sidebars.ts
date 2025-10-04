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
        "guides/quick-start",
        "guides/basic-usage",
      ],
    },

    // Core Documentation
    {
      type: "category",
      label: "📖 Core Concepts",
      items: ["architecture", "workflows"],
    },

    // User Guides
    {
      type: "category",
      label: "🔧 User Guides",
      items: [
        "guides/cli-commands",
        "guides/auto-params",
        "guides/preprocessing",
      ],
    },

    // GPU Acceleration
    {
      type: "category",
      label: "⚡ GPU Acceleration",
      items: ["gpu/overview", "gpu/features", "gpu/rgb-augmentation"],
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
        "features/pipeline-configuration",
      ],
    },

    // QGIS Integration
    {
      type: "category",
      label: "🗺️ QGIS Integration",
      items: ["guides/qgis-integration", "guides/qgis-troubleshooting"],
    },

    // Technical Reference
    {
      type: "category",
      label: "📚 Technical Reference",
      items: [
        "reference/memory-optimization",
        "mermaid-reference",
        "api/processor",
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
