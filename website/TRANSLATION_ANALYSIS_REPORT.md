# Docusaurus Translation Analysis & Update Plan

**Date:** October 6, 2025  
**Repository:** IGN_LIDAR_HD_DATASET  
**Documentation System:** Docusaurus with i18n (English + French)

## Executive Summary

### Current Status

- ✅ **English files:** 57 documentation files
- ✅ **French files:** 59 documentation files (2 extra French-only files)
- ⚠️ **Missing translations:** 0 files
- 🔄 **Outdated translations:** 21 files need updating
- ✅ **Up-to-date translations:** 36 files (63%)

### Key Findings

1. **Good News:** All English files have French translations - no missing files!
2. **Issue:** 21 files (37%) have outdated French translations where the English version has been modified more recently
3. **Extra Content:** 2 French-specific files exist that don't have English equivalents

## Detailed Analysis

### Files Needing Critical Updates (1+ days old)

These files have significant time gaps and should be prioritized:

1. **features/architectural-styles.md** (1 day old)

   - EN: 8,612 bytes | FR: 9,056 bytes | Diff: -444 bytes
   - Impact: Medium - Regional building classification documentation

2. **guides/features/overview.md** (1 day old)

   - EN: 8,956 bytes | FR: 10,027 bytes | Diff: -1,071 bytes
   - Impact: High - Main features overview page

3. **reference/cli-enrich.md** (1 day old)

   - EN: 9,247 bytes | FR: 9,888 bytes | Diff: -641 bytes
   - Impact: High - CLI command reference

4. **reference/cli-patch.md** (1 day old)

   - EN: 1,628 bytes | FR: 1,792 bytes | Diff: -164 bytes
   - Impact: Medium - CLI command reference

5. **release-notes/v1.5.0.md** (1 day old)
   - EN: 10,988 bytes | FR: 11,583 bytes | Diff: -595 bytes
   - Impact: Low - Historical release notes

### Files Needing Moderate Updates (< 1 day old)

16 additional files were modified today:

- api/features.md
- api/gpu-api.md
- features/axonometry.md
- features/format-preferences.md
- features/infrared-augmentation.md
- features/lod3-classification.md
- features/rgb-augmentation.md
- gpu/features.md
- guides/auto-params.md
- guides/performance.md
- installation/gpu-setup.md
- intro.md
- reference/architectural-styles.md
- reference/historical-analysis.md
- release-notes/v1.6.0.md
- release-notes/v1.7.1.md

### Extra French Files

Two files exist in French but not in English:

1. **examples/index.md** (13,490 bytes)

   - Modified: 2025-10-04
   - Decision needed: Keep French-only or create English version?

2. **guides/visualization.md** (12,590 bytes)
   - Modified: 2025-10-06 (today!)
   - Decision needed: Keep French-only or create English version?

## Translation Update Strategy

### Phase 1: Critical Updates (Priority 1)

Update the 5 files that are 1+ days old:

```bash
# Files to update immediately
1. features/architectural-styles.md
2. guides/features/overview.md
3. reference/cli-enrich.md
4. reference/cli-patch.md
5. release-notes/v1.5.0.md
```

**Action:** Manual review and translation update for each file

### Phase 2: Same-Day Updates (Priority 2)

Update the 16 files modified today. These likely have minor changes:

**Action:**

- Compare diffs to identify changes
- Update French translations for modified sections
- Maintain existing French translations for unchanged sections

### Phase 3: French-Only Content Review (Priority 3)

Review the 2 French-only files:

**Options:**

1. **Keep as French-only:** If content is France-specific
2. **Create English versions:** For consistency and international audience
3. **Remove:** If content is outdated or redundant

## Translation Workflow

### Recommended Process

1. **Identify changes** in English file

   ```bash
   diff -u fr_file en_file
   ```

2. **Extract changed sections** from English

3. **Translate new/modified content** to French

4. **Update French file** maintaining:

   - Same document structure
   - Same frontmatter metadata (translated)
   - Same code blocks (untranslated)
   - Translated prose and headers

5. **Verify** formatting and links

6. **Test build**
   ```bash
   npm run build
   ```

### Translation Guidelines

**DO Translate:**

- Headers and titles
- Body text and descriptions
- Image alt text
- Link descriptions
- Admonition content (notes, tips, warnings)

**DO NOT Translate:**

- Code blocks
- File paths
- Command names
- Python/JavaScript variable names
- URLs
- Version numbers
- Technical acronyms (API, CLI, GPU, etc.)

**Frontmatter Translation:**

```yaml
# English
---
title: Getting Started
description: Quick start guide
keywords: [lidar, processing, python]
---
# French
---
title: Démarrage Rapide
description: Guide de démarrage rapide
keywords: [lidar, traitement, python]
---
```

## Technical Implementation

### Docusaurus i18n Structure

```
website/
├── docs/                          # English (default)
│   ├── intro.md
│   ├── guides/
│   │   └── basic-usage.md
│   └── ...
└── i18n/
    └── fr/                        # French
        └── docusaurus-plugin-content-docs/
            └── current/           # Mirrors docs/ structure
                ├── intro.md
                ├── guides/
                │   └── basic-usage.md
                └── ...
```

### Configuration Check

The `docusaurus.config.ts` correctly configures:

```typescript
i18n: {
  defaultLocale: "en",
  locales: ["en", "fr"],
}
```

## Quality Assurance

### Checklist for Each Updated File

- [ ] English changes identified
- [ ] French translation updated
- [ ] Frontmatter metadata translated
- [ ] Code blocks preserved (untranslated)
- [ ] Links verified
- [ ] Build successful
- [ ] Visual check in browser
- [ ] No broken links

### Build & Deploy

```bash
# Local build test
cd website
npm run build

# Start local server
npm run start -- --locale fr

# Deploy (if all looks good)
npm run deploy
```

## Recommendations

### Immediate Actions

1. **Update critical files first** (5 files, 1+ days old)
2. **Set up automated checks** for translation drift
3. **Document translation process** for future maintainers

### Long-term Solutions

1. **Translation Management System**
   - Consider tools like Crowdin or Lokalise
   - Automate detection of changed files
2. **CI/CD Integration**
   - Add checks to detect translation drift
   - Alert when English files are modified without French updates
3. **Contribution Guidelines**

   - Require French translations for PR merges
   - Provide translation templates

4. **Automated Translation Assistance**
   - Use AI-assisted translation for first draft
   - Manual review and refinement

## Priority Files List

### Immediate (This Week)

1. guides/features/overview.md - Main overview page
2. reference/cli-enrich.md - Primary CLI command
3. features/architectural-styles.md - Core feature
4. intro.md - Homepage (minor update needed)
5. installation/gpu-setup.md - Installation guide

### Soon (Next Week)

6. features/rgb-augmentation.md
7. features/infrared-augmentation.md
8. gpu/features.md
9. api/gpu-api.md
10. guides/performance.md

### Later (As Needed)

- Release notes (historical)
- Reference documentation (stable)
- Minor updates (< 300 bytes difference)

## Conclusion

The Docusaurus documentation is well-structured with good French translation coverage (63% up-to-date). The main task is to update 21 files where the English version has been modified more recently. Priority should be given to user-facing guides and feature documentation over historical release notes.

**Estimated Effort:**

- Critical updates: 2-3 hours
- All updates: 8-10 hours
- With automation: 4-6 hours

**Next Steps:**

1. Start with the 5 critical files
2. Set up translation workflow
3. Batch process same-day updates
4. Establish ongoing maintenance process
