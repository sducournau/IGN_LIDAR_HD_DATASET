# Docusaurus Documentation Plan

## 📋 Overview

This document outlines the complete plan for creating professional Docusaurus documentation for the IGN LiDAR HD Processing Library.

## 🎯 Goals

1. **Professional Documentation**: Create a modern, searchable documentation website
2. **Multi-language Support**: English and French documentation
3. **Easy Navigation**: Clear structure with sidebar navigation
4. **Interactive Examples**: Code examples with syntax highlighting
5. **API Reference**: Auto-generated API documentation
6. **Search Functionality**: Built-in search for easy discovery
7. **Mobile Responsive**: Works on all devices
8. **Version Control**: Support for multiple documentation versions

## 📁 Proposed Site Structure

```
website/
├── docs/                           # Documentation content
│   ├── intro.md                    # Getting started
│   ├── installation/
│   │   ├── quick-start.md
│   │   ├── from-source.md
│   │   ├── dependencies.md
│   │   └── troubleshooting.md
│   ├── guides/
│   │   ├── basic-usage.md
│   │   ├── cli-commands.md
│   │   ├── python-api.md
│   │   ├── qgis-integration.md
│   │   ├── batch-processing.md
│   │   └── gpu-acceleration.md
│   ├── features/
│   │   ├── smart-skip.md
│   │   ├── format-preferences.md
│   │   ├── lod-classification.md
│   │   ├── feature-extraction.md
│   │   ├── data-augmentation.md
│   │   └── parallel-processing.md
│   ├── tutorials/
│   │   ├── urban-processing.md
│   │   ├── rural-processing.md
│   │   ├── full-workflow.md
│   │   ├── ml-training.md
│   │   └── custom-features.md
│   ├── api/
│   │   ├── processor.md
│   │   ├── downloader.md
│   │   ├── features.md
│   │   ├── classes.md
│   │   └── utils.md
│   ├── reference/
│   │   ├── memory-optimization.md
│   │   ├── performance-tuning.md
│   │   ├── configuration.md
│   │   ├── data-formats.md
│   │   └── architecture.md
│   └── migration/
│       ├── v1-to-v2.md
│       └── changelog.md
├── i18n/
│   └── fr/                         # French translations
│       └── docusaurus-plugin-content-docs/
│           └── current/
│               ├── intro.md
│               ├── installation/
│               ├── guides/
│               └── ...
├── blog/                           # Optional blog for updates
│   ├── 2025-01-03-smart-skip.md
│   └── 2025-01-01-v1.1.0-release.md
├── src/
│   ├── components/                 # Custom React components
│   │   ├── HomepageFeatures.js
│   │   └── CodeExample.js
│   ├── css/
│   │   └── custom.css
│   └── pages/
│       ├── index.js                # Landing page
│       └── help.js                 # Help page
├── static/
│   ├── img/
│   │   ├── logo.svg
│   │   ├── favicon.ico
│   │   └── examples/               # Screenshots, diagrams
│   └── downloads/                  # Sample data, configs
├── docusaurus.config.js            # Main configuration
├── sidebars.js                     # Sidebar structure
├── package.json
└── README.md
```

## 🚀 Implementation Phases

### Phase 1: Setup and Configuration (Week 1)

#### Step 1.1: Initialize Docusaurus

```bash
cd /path/to/IGN_LIDAR_HD_downloader
npx create-docusaurus@latest website classic
cd website
npm install
```

#### Step 1.2: Configure docusaurus.config.js

```javascript
module.exports = {
  title: "IGN LiDAR HD Processing Library",
  tagline: "Process IGN LiDAR data into ML-ready datasets",
  url: "https://yourusername.github.io",
  baseUrl: "/IGN_LIDAR_HD_downloader/",
  organizationName: "yourusername",
  projectName: "IGN_LIDAR_HD_downloader",

  i18n: {
    defaultLocale: "en",
    locales: ["en", "fr"],
    localeConfigs: {
      en: { label: "English" },
      fr: { label: "Français" },
    },
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          editUrl:
            "https://github.com/yourusername/IGN_LIDAR_HD_downloader/tree/main/website/",
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: {
          showReadingTime: true,
          editUrl:
            "https://github.com/yourusername/IGN_LIDAR_HD_downloader/tree/main/website/",
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
          type: "doc",
          docId: "intro",
          position: "left",
          label: "Docs",
        },
        {
          to: "/blog",
          label: "Blog",
          position: "left",
        },
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
              label: "API Reference",
              to: "/docs/api/processor",
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "GitHub Issues",
              href: "https://github.com/yourusername/IGN_LIDAR_HD_downloader/issues",
            },
            {
              label: "Discussions",
              href: "https://github.com/yourusername/IGN_LIDAR_HD_downloader/discussions",
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
              href: "https://github.com/yourusername/IGN_LIDAR_HD_downloader",
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
    algolia: {
      // Algolia DocSearch configuration (optional, for search)
      appId: "YOUR_APP_ID",
      apiKey: "YOUR_API_KEY",
      indexName: "ign-lidar",
    },
  },
};
```

#### Step 1.3: Configure sidebars.js

```javascript
module.exports = {
  docs: [
    "intro",
    {
      type: "category",
      label: "Installation",
      items: [
        "installation/quick-start",
        "installation/from-source",
        "installation/dependencies",
        "installation/troubleshooting",
      ],
    },
    {
      type: "category",
      label: "User Guides",
      items: [
        "guides/basic-usage",
        "guides/cli-commands",
        "guides/python-api",
        "guides/qgis-integration",
        "guides/batch-processing",
        "guides/gpu-acceleration",
      ],
    },
    {
      type: "category",
      label: "Features",
      items: [
        "features/smart-skip",
        "features/format-preferences",
        "features/lod-classification",
        "features/feature-extraction",
        "features/data-augmentation",
        "features/parallel-processing",
      ],
    },
    {
      type: "category",
      label: "Tutorials",
      items: [
        "tutorials/urban-processing",
        "tutorials/rural-processing",
        "tutorials/full-workflow",
        "tutorials/ml-training",
        "tutorials/custom-features",
      ],
    },
    {
      type: "category",
      label: "API Reference",
      items: [
        "api/processor",
        "api/downloader",
        "api/features",
        "api/classes",
        "api/utils",
      ],
    },
    {
      type: "category",
      label: "Technical Reference",
      items: [
        "reference/memory-optimization",
        "reference/performance-tuning",
        "reference/configuration",
        "reference/data-formats",
        "reference/architecture",
      ],
    },
    {
      type: "category",
      label: "Migration",
      items: ["migration/v1-to-v2", "migration/changelog"],
    },
  ],
};
```

### Phase 2: Content Migration (Week 2)

#### Step 2.1: Convert Existing Documentation

- Migrate content from `docs/` folder to Docusaurus format
- Update internal links to use Docusaurus format
- Add frontmatter to all markdown files
- Create new content where gaps exist

#### Step 2.2: Frontmatter Format

```markdown
---
id: smart-skip
title: Smart Skip Detection
sidebar_label: Smart Skip
description: Automatically skip existing downloads, enriched files, and patches
keywords: [skip, idempotent, resume, optimization]
---

# Smart Skip Detection

Content here...
```

#### Step 2.3: Content Mapping from Current Docs

**From docs/guides/** → **website/docs/guides/**

- `QUICK_START_QGIS.md` → `qgis-integration.md`
- `QGIS_COMPATIBILITY.md` → Add to `qgis-integration.md`
- `QGIS_TROUBLESHOOTING.md` → `installation/troubleshooting.md`

**From docs/features/** → **website/docs/features/**

- `SMART_SKIP_SUMMARY.md` → `smart-skip.md`
- `SKIP_EXISTING_TILES.md` → Merge into `smart-skip.md`
- `SKIP_EXISTING_PATCHES.md` → Merge into `smart-skip.md`
- `SKIP_EXISTING_ENRICHED.md` → Merge into `smart-skip.md`
- `OUTPUT_FORMAT_PREFERENCES.md` → `format-preferences.md`

**From docs/reference/** → **website/docs/reference/**

- `MEMORY_OPTIMIZATION.md` → `memory-optimization.md`
- `QUICK_REFERENCE_MEMORY.md` → Merge into `memory-optimization.md`

**From README.md** → **Multiple pages**

- Quick Start section → `intro.md`
- Features section → Landing page features
- CLI Commands → `guides/cli-commands.md`
- API Reference → `api/*.md`

### Phase 3: Enhanced Content (Week 3)

#### Step 3.1: Create New Content

1. **Comprehensive Tutorials**

   - Step-by-step urban area processing
   - Full ML pipeline example
   - Custom feature engineering guide

2. **API Documentation**

   - Auto-generate from docstrings using sphinx-to-md
   - Code examples for each class/function
   - Parameter descriptions and return values

3. **Visual Assets**

   - Architecture diagrams
   - Workflow flowcharts
   - Example outputs (point clouds, patches)
   - Performance benchmarks graphs

4. **Interactive Examples**
   - Live code snippets
   - Parameter configuration examples
   - Output format visualizations

#### Step 3.2: Create Landing Page

```jsx
// src/pages/index.js
import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./index.module.css";
import HomepageFeatures from "@site/src/components/HomepageFeatures";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link className="button button--primary button--lg" to="/docs/intro">
            Get Started - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  return (
    <Layout
      title="IGN LiDAR HD Processing"
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

#### Step 3.3: Create Feature Showcase Component

```jsx
// src/components/HomepageFeatures.js
const FeatureList = [
  {
    title: "LiDAR-Only Processing",
    Svg: require("@site/static/img/lidar.svg").default,
    description: (
      <>
        No RGB dependency. Works purely with geometric data for robust building
        component classification.
      </>
    ),
  },
  {
    title: "Smart Skip Detection",
    Svg: require("@site/static/img/skip.svg").default,
    description: (
      <>
        Automatically skip existing downloads, enriched files, and patches.
        Resume interrupted workflows without reprocessing.
      </>
    ),
  },
  {
    title: "Multi-Level Classification",
    Svg: require("@site/static/img/lod.svg").default,
    description: (
      <>
        Support for LOD2 (15 classes) and LOD3 (30 classes) building
        classification taxonomies.
      </>
    ),
  },
  {
    title: "GPU Acceleration",
    Svg: require("@site/static/img/gpu.svg").default,
    description: (
      <>
        Optional GPU support for faster feature computation using CUDA and cupy
        for large-scale processing.
      </>
    ),
  },
  {
    title: "Parallel Processing",
    Svg: require("@site/static/img/parallel.svg").default,
    description: (
      <>
        Multi-worker support for efficient batch processing of large LiDAR
        datasets.
      </>
    ),
  },
  {
    title: "Rich Features",
    Svg: require("@site/static/img/features.svg").default,
    description: (
      <>
        Extract comprehensive geometric features: normals, curvature, planarity,
        verticality, density, and more.
      </>
    ),
  },
];
```

### Phase 4: French Translation (Week 4)

#### Step 4.1: Translate Core Documentation

```bash
# Generate translation files
npm run write-translations -- --locale fr

# Translate key pages
- intro.md
- installation/quick-start.md
- guides/basic-usage.md
- guides/cli-commands.md
- features/smart-skip.md
```

#### Step 4.2: Use README_FR.md Content

- Migrate French README content
- Ensure terminology consistency
- Add French-specific examples

### Phase 5: API Documentation Generation (Week 5)

#### Step 5.1: Auto-generate API Docs

```bash
# Install sphinx and extensions
pip install sphinx sphinx-markdown-builder sphinx-autodoc-typehints

# Generate API docs
cd docs
sphinx-apidoc -o api_source ../ign_lidar
sphinx-build -b markdown api_source api_output

# Move to Docusaurus
mv api_output/* ../website/docs/api/
```

#### Step 5.2: Enhance API Docs

- Add code examples for each function
- Include parameter descriptions
- Add usage notes and warnings
- Link to related tutorials

### Phase 6: Testing and Deployment (Week 6)

#### Step 6.1: Local Testing

```bash
cd website
npm start  # Test locally at http://localhost:3000
npm run build  # Test production build
npm run serve  # Test production build locally
```

#### Step 6.2: GitHub Pages Deployment

```bash
# Configure GitHub Pages
# Add to docusaurus.config.js:
# url: 'https://yourusername.github.io'
# baseUrl: '/IGN_LIDAR_HD_downloader/'

# Deploy
GIT_USER=yourusername npm run deploy
```

#### Step 6.3: CI/CD Setup

Create `.github/workflows/deploy-docs.yml`:

```yaml
name: Deploy Docusaurus

on:
  push:
    branches: [main]
    paths:
      - "website/**"
      - "docs/**"
      - ".github/workflows/deploy-docs.yml"

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
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
```

## 📚 Content Checklist

### Essential Pages

- [ ] Introduction / Getting Started
- [ ] Installation Guide
- [ ] Quick Start Tutorial
- [ ] CLI Command Reference
- [ ] Python API Guide
- [ ] Feature Extraction Guide
- [ ] Smart Skip Documentation
- [ ] LOD Classification Reference
- [ ] QGIS Integration Guide
- [ ] Memory Optimization Guide
- [ ] Configuration Reference
- [ ] Troubleshooting Guide
- [ ] Migration Guide
- [ ] Changelog

### API Documentation

- [ ] LiDARProcessor class
- [ ] IGNLiDARDownloader class
- [ ] Feature extraction functions
- [ ] Classification schemas
- [ ] Utility functions

### Tutorials

- [ ] Basic processing workflow
- [ ] Urban area processing
- [ ] Batch processing with skip detection
- [ ] GPU-accelerated processing
- [ ] Custom feature engineering
- [ ] ML model training pipeline

### Reference

- [ ] Data format specifications
- [ ] Configuration options
- [ ] Performance benchmarks
- [ ] Architecture overview
- [ ] Memory management strategies

## 🎨 Design Considerations

### Theme Customization

```css
/* src/css/custom.css */
:root {
  --ifm-color-primary: #2e8555;
  --ifm-color-primary-dark: #29784c;
  --ifm-color-primary-darker: #277148;
  --ifm-color-primary-darkest: #205d3b;
  --ifm-color-primary-light: #33925d;
  --ifm-color-primary-lighter: #359962;
  --ifm-color-primary-lightest: #3cad6e;
  --ifm-code-font-size: 95%;
  --ifm-font-family-base: "Inter", system-ui, -apple-system, sans-serif;
}

.hero__title {
  font-size: 3rem;
  font-weight: 700;
}

.hero__subtitle {
  font-size: 1.5rem;
  margin-bottom: 2rem;
}
```

### Logo and Branding

- Create logo.svg for navbar
- Create favicon.ico
- Add social media preview image (og:image)

## 🔍 SEO Optimization

### Meta Tags

```javascript
// docusaurus.config.js
metadata: [
  {name: 'keywords', content: 'lidar, ign, machine learning, building classification, lod, python'},
  {name: 'description', content: 'Process IGN LiDAR HD data into ML-ready datasets for building LOD classification'},
],
```

### Sitemap

- Automatically generated by Docusaurus
- Submit to Google Search Console

## 📊 Analytics

### Google Analytics

```javascript
// docusaurus.config.js
gtag: {
  trackingID: 'G-XXXXXXXXXX',
  anonymizeIP: true,
},
```

## 🚀 Launch Checklist

### Pre-Launch

- [ ] All essential pages completed
- [ ] French translations for key pages
- [ ] API documentation generated
- [ ] Search functionality configured
- [ ] Mobile responsiveness tested
- [ ] All links verified
- [ ] Images optimized
- [ ] SEO metadata added
- [ ] Analytics configured

### Launch

- [ ] Deploy to GitHub Pages
- [ ] Update README.md with docs link
- [ ] Announce in project README
- [ ] Share with community
- [ ] Submit to relevant directories

### Post-Launch

- [ ] Monitor analytics
- [ ] Collect user feedback
- [ ] Fix broken links
- [ ] Add missing content
- [ ] Keep synchronized with code updates

## 🔄 Maintenance Plan

### Regular Updates

- **Weekly**: Review and fix issues
- **Monthly**: Update changelog and migration guides
- **Per Release**: Update API docs and version selector

### Content Review

- Review outdated content quarterly
- Update examples with new features
- Refresh screenshots and diagrams
- Verify all external links

## 📦 Dependencies

### Required npm Packages

```json
{
  "dependencies": {
    "@docusaurus/core": "^3.0.0",
    "@docusaurus/preset-classic": "^3.0.0",
    "@docusaurus/theme-mermaid": "^3.0.0",
    "@mdx-js/react": "^3.0.0",
    "clsx": "^2.0.0",
    "prism-react-renderer": "^2.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@docusaurus/module-type-aliases": "^3.0.0",
    "@docusaurus/types": "^3.0.0"
  }
}
```

## 🎯 Success Metrics

### Target Metrics (6 months)

- 1000+ monthly page views
- <2s average page load time
- > 90% mobile usability score
- 80%+ documentation coverage
- <5% bounce rate on docs pages

## 📝 Notes

- Keep documentation synchronized with code
- Use version selector for multiple versions
- Maintain consistency in terminology
- Include runnable code examples
- Add "Edit this page" links for community contributions

---

**Estimated Timeline**: 6 weeks
**Estimated Effort**: 80-100 hours
**Priority**: High - Improves user experience and project professionalism
