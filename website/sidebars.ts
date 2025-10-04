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
        "guides/auto-params",
        "guides/preprocessing",
      ],
    },

    // Promote core documentation
    "architecture",
    "workflows",

    // NEW: GPU Acceleration Section
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
      items: [
        "release-notes/v1.7.1",
        "release-notes/v1.6.2",
        "release-notes/v1.6.0",
        "release-notes/v1.5.0",
      ],
    },
  ],
};

export default sidebars;
