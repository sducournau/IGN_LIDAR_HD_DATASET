import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

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
    'intro',
    {
      type: 'category',
      label: 'Installation',
      items: ['installation/quick-start'],
    },
    {
      type: 'category',
      label: 'User Guides',
      items: [
        'guides/basic-usage',
        'guides/cli-commands',
        'guides/qgis-integration',
        'guides/qgis-troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Features',
      items: ['features/smart-skip', 'features/format-preferences'],
    },
    {
      type: 'category',
      label: 'Technical Reference',
      items: ['reference/memory-optimization'],
    },
  ],
};

export default sidebars;
