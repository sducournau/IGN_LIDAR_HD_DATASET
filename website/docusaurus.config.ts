import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: "IGN LiDAR HD Processing Library",
  tagline: "Process IGN LiDAR data into ML-ready datasets",
  favicon: "img/favicon.ico",

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: "https://sducournau.github.io",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/IGN_LIDAR_HD_DATASET/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "sducournau", // Usually your GitHub org/user name.
  projectName: "IGN_LIDAR_HD_DATASET", // Usually your repo name.

  onBrokenLinks: "warn",

  // GitHub Pages deployment config
  trailingSlash: false,
  deploymentBranch: "gh-pages",

  // Markdown configuration
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
    mermaid: true,
  },

  // Internationalization (English + French)
  i18n: {
    defaultLocale: "en",
    locales: ["en", "fr"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          routeBasePath: "/", // Serve the docs at the root
          sidebarPath: "./sidebars.ts",
          editUrl:
            "https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/website/",
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: {
          routeBasePath: "/blog", // Serve the blog at /blog
          showReadingTime: true,
          feedOptions: {
            type: ["rss", "atom"],
            xslt: true,
          },
          editUrl:
            "https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/website/",
          // Useful options to enforce blogging best practices
          onInlineTags: "warn",
          onInlineAuthors: "warn",
          onUntruncatedBlogPosts: "warn",
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: ["@docusaurus/theme-mermaid"],

  themeConfig: {
    // Replace with your project's social card
    image: "img/lidar-visualization-thumbnail.jpg",
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "IGN LiDAR HD",
      logo: {
        alt: "IGN LiDAR Logo",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Documentation",
        },
        { to: "/blog", label: "📢 First Release!", position: "left" },
        {
          type: "localeDropdown",
          position: "right",
        },
        {
          href: "https://github.com/sducournau/IGN_LIDAR_HD_DATASET",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            {
              label: "Getting Started",
              to: "/",
            },
            {
              label: "Installation",
              to: "/installation/quick-start",
            },
            {
              label: "Basic Usage",
              to: "/guides/basic-usage",
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/sducournau/IGN_LIDAR_HD_DATASET",
            },
            {
              label: "Issues",
              href: "https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "Blog",
              to: "/blog",
            },
            {
              label: "GitHub",
              href: "https://github.com/sducournau/IGN_LIDAR_HD_DATASET",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} IGN LiDAR HD. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["python", "bash"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
