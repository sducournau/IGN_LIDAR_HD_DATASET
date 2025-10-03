# Quick Start: Implementing Docusaurus Documentation

This guide provides step-by-step instructions to quickly set up Docusaurus documentation for the IGN LiDAR HD Processing Library.

## Prerequisites

- Node.js 18.0 or higher
- npm or yarn package manager
- Git

## Step 1: Initialize Docusaurus (5 minutes)

```bash
# Navigate to project root
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_downloader

# Create Docusaurus site
npx create-docusaurus@latest website classic

# Navigate to website directory
cd website

# Install dependencies
npm install
```

## Step 2: Basic Configuration (10 minutes)

Edit `website/docusaurus.config.js`:

```javascript
const config = {
  title: "IGN LiDAR HD Processing Library",
  tagline: "Process IGN LiDAR data into ML-ready datasets",
  favicon: "img/favicon.ico",

  // GitHub Pages configuration
  url: "https://yourusername.github.io",
  baseUrl: "/IGN_LIDAR_HD_downloader/",
  organizationName: "yourusername",
  projectName: "IGN_LIDAR_HD_downloader",

  // Deployment branch
  deploymentBranch: "gh-pages",
  trailingSlash: false,

  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",

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
          sidebarPath: require.resolve("./sidebars.js"),
          editUrl:
            "https://github.com/yourusername/IGN_LIDAR_HD_downloader/tree/main/website/",
        },
        blog: {
          showReadingTime: true,
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],

  themeConfig: {
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
        { to: "/blog", label: "Blog", position: "left" },
        {
          type: "localeDropdown",
          position: "right",
        },
        {
          href: "https://github.com/yourusername/IGN_LIDAR_HD_downloader",
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
              to: "/docs/intro",
            },
            {
              label: "User Guides",
              to: "/docs/category/user-guides",
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/yourusername/IGN_LIDAR_HD_downloader",
            },
            {
              label: "Issues",
              href: "https://github.com/yourusername/IGN_LIDAR_HD_downloader/issues",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} IGN LiDAR HD. Built with Docusaurus.`,
    },
    prism: {
      theme: require("prism-react-renderer/themes/github"),
      darkTheme: require("prism-react-renderer/themes/dracula"),
      additionalLanguages: ["python", "bash"],
    },
  },
};

module.exports = config;
```

## Step 3: Test Locally (2 minutes)

```bash
# From website/ directory
npm start

# Open browser at http://localhost:3000
```

## Step 4: Migrate Essential Content (30 minutes)

### Create intro.md

````bash
# In website/docs/
cat > intro.md << 'EOF'
---
sidebar_position: 1
---

# Getting Started

Welcome to the IGN LiDAR HD Processing Library documentation!

## What is IGN LiDAR HD?

The IGN LiDAR HD Processing Library is a Python toolkit for processing high-density LiDAR data from the French National Institute of Geographic and Forest Information (IGN) into machine learning-ready datasets.

## Key Features

- 🎯 **LiDAR-Only Processing** - No RGB dependency
- ⚡ **Smart Skip Detection** - Resume interrupted workflows
- 🏗️ **Multi-Level Classification** - LOD2 and LOD3 support
- 🚀 **GPU Acceleration** - Optional CUDA support
- 🔄 **Parallel Processing** - Multi-worker batch processing
- 📊 **Rich Features** - Comprehensive geometric feature extraction

## Quick Installation

```bash
pip install ign-lidar-hd
````

## Quick Example

```python
from ign_lidar import LiDARProcessor

# Initialize processor
processor = LiDARProcessor(lod_level="LOD2")

# Process a single tile
patches = processor.process_tile("data.laz", "output/")
```

## Next Steps

- 📖 Read the [Installation Guide](installation/quick-start.md)
- 🎓 Follow a [Tutorial](category/tutorials)
- 🔍 Explore [Features](category/features)
- 📚 Check the [API Reference](category/api-reference)

EOF

````

### Create Installation Guide

```bash
# Create directory
mkdir -p docs/installation

# Create quick-start.md
cat > docs/installation/quick-start.md << 'EOF'
---
sidebar_position: 1
---

# Quick Start Installation

## Requirements

- Python 3.8 or higher
- pip package manager

## Install from PyPI

```bash
pip install ign-lidar-hd
````

## Verify Installation

```bash
ign-lidar-process --version
```

## Optional: GPU Support

For GPU-accelerated feature computation:

```bash
pip install ign-lidar-hd[gpu]
```

Requires:

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0 or higher
- cupy-cuda11x package

## Next Steps

- Try the [Basic Usage Guide](../guides/basic-usage.md)
- Explore [CLI Commands](../guides/cli-commands.md)
- Learn about [Smart Skip Features](../features/smart-skip.md)

EOF

````

## Step 5: Copy Existing Documentation (15 minutes)

```bash
# From project root
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_downloader

# Create categories in Docusaurus
mkdir -p website/docs/{guides,features,reference,tutorials,api}

# Copy and convert existing docs
# Guides
cp docs/guides/QUICK_START_QGIS.md website/docs/guides/qgis-integration.md
cp docs/guides/QGIS_TROUBLESHOOTING.md website/docs/guides/qgis-troubleshooting.md

# Features
cp docs/features/SMART_SKIP_SUMMARY.md website/docs/features/smart-skip.md
cp docs/features/OUTPUT_FORMAT_PREFERENCES.md website/docs/features/format-preferences.md

# Reference
cp docs/reference/MEMORY_OPTIMIZATION.md website/docs/reference/memory-optimization.md
````

### Add Frontmatter to Copied Files

Add this to the top of each copied file:

```markdown
---
sidebar_position: 1
title: Document Title
description: Brief description
---
```

Example for `smart-skip.md`:

```markdown
---
sidebar_position: 1
title: Smart Skip Detection
description: Automatically skip existing downloads, enriched files, and patches
keywords: [skip, idempotent, resume, workflow]
---

# Smart Skip Detection

[Rest of content...]
```

## Step 6: Configure Sidebar (10 minutes)

Edit `website/sidebars.js`:

```javascript
const sidebars = {
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
        "guides/qgis-troubleshooting",
      ],
    },
    {
      type: "category",
      label: "Features",
      items: ["features/smart-skip", "features/format-preferences"],
    },
    {
      type: "category",
      label: "Technical Reference",
      items: ["reference/memory-optimization"],
    },
  ],
};

module.exports = sidebars;
```

## Step 7: Customize Home Page (20 minutes)

Edit `website/src/pages/index.js`:

```javascript
import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import styles from "./index.module.css";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx("hero hero--primary", styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro"
          >
            Get Started - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Process IGN LiDAR data into ML-ready datasets"
    >
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
```

Edit `website/src/components/HomepageFeatures/index.js`:

```javascript
import React from "react";
import clsx from "clsx";
import styles from "./styles.module.css";

const FeatureList = [
  {
    title: "⚡ Smart Skip Detection",
    description: (
      <>
        Automatically skip existing downloads, enriched files, and patches.
        Resume interrupted workflows without reprocessing existing data.
      </>
    ),
  },
  {
    title: "🏗️ Multi-Level Classification",
    description: (
      <>
        Support for LOD2 (15 classes) and LOD3 (30 classes) building
        classification taxonomies with rich geometric features.
      </>
    ),
  },
  {
    title: "🚀 GPU Acceleration",
    description: (
      <>
        Optional GPU support for faster feature computation using CUDA and cupy
        for large-scale LiDAR processing.
      </>
    ),
  },
];

function Feature({ title, description }) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
```

## Step 8: Build and Test (5 minutes)

```bash
# Build for production
cd website
npm run build

# Test production build locally
npm run serve

# Open browser at http://localhost:3000
```

## Step 9: Deploy to GitHub Pages (10 minutes)

### Configure GitHub Pages

1. Go to your GitHub repository settings
2. Navigate to Pages section
3. Set source to "gh-pages" branch

### Deploy

```bash
# Set your GitHub username
export GIT_USER=yourusername

# Deploy (from website/ directory)
npm run deploy
```

Or add to `package.json`:

```json
{
  "scripts": {
    "deploy": "GIT_USER=yourusername docusaurus deploy"
  }
}
```

## Step 10: Setup CI/CD (Optional, 15 minutes)

Create `.github/workflows/deploy-docs.yml`:

```yaml
name: Deploy Docusaurus

on:
  push:
    branches: [main]
    paths:
      - "website/**"
      - ".github/workflows/deploy-docs.yml"

jobs:
  deploy:
    name: Deploy to GitHub Pages
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-node@v3
        with:
          node-version: 18
          cache: npm
          cache-dependency-path: website/package-lock.json

      - name: Install dependencies
        run: cd website && npm ci

      - name: Build website
        run: cd website && npm run build

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./website/build
          user_name: github-actions[bot]
          user_email: github-actions[bot]@users.noreply.github.com
```

## Timeline Summary

| Step                         | Duration     | Status |
| ---------------------------- | ------------ | ------ |
| 1. Initialize Docusaurus     | 5 min        | ⏳     |
| 2. Basic Configuration       | 10 min       | ⏳     |
| 3. Test Locally              | 2 min        | ⏳     |
| 4. Migrate Essential Content | 30 min       | ⏳     |
| 5. Copy Existing Docs        | 15 min       | ⏳     |
| 6. Configure Sidebar         | 10 min       | ⏳     |
| 7. Customize Home Page       | 20 min       | ⏳     |
| 8. Build and Test            | 5 min        | ⏳     |
| 9. Deploy to GitHub Pages    | 10 min       | ⏳     |
| 10. Setup CI/CD              | 15 min       | ⏳     |
| **Total**                    | **~2 hours** |        |

## Next Steps

After basic setup is complete:

1. **Add More Content**

   - Create API reference pages
   - Write comprehensive tutorials
   - Add code examples

2. **Enable French Translation**

   - Run `npm run write-translations -- --locale fr`
   - Translate key pages

3. **Add Search**

   - Configure Algolia DocSearch
   - Or use local search plugin

4. **Enhance Design**

   - Customize colors in `custom.css`
   - Add logo and favicon
   - Create custom components

5. **Add Blog Posts**
   - Announce new features
   - Share tutorials
   - Document release notes

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use different port
npm start -- --port 3001
```

### Build Errors

```bash
# Clear cache
npm run clear
npm run build
```

### Deployment Issues

```bash
# Ensure gh-pages branch exists
git checkout -b gh-pages
git push origin gh-pages

# Then retry deployment
```

## Resources

- [Docusaurus Documentation](https://docusaurus.io/)
- [Docusaurus GitHub](https://github.com/facebook/docusaurus)
- [Markdown Features](https://docusaurus.io/docs/markdown-features)
- [Deployment Guide](https://docusaurus.io/docs/deployment)

---

**Estimated Time**: 2 hours for basic setup
**Result**: Professional documentation website with multi-language support
