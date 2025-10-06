# Translation Workflow Tools

This directory contains tools for managing French translations of the Docusaurus documentation.

## Quick Status Check

```bash
python3 check_translations.py
```

Shows a quick overview of translation status (translated, partial, needs work).

## Comprehensive Analysis

```bash
python3 analyze_translations.py
```

Provides detailed analysis including:

- Missing files in French
- Extra files in French (not in English)
- Word count comparisons
- Translation status heuristics
- Priority-ordered action items

## Update French Documentation

```bash
# Update only missing French files
python3 update_fr_docs.py

# Force update all priority files (overwrites existing)
python3 update_fr_docs.py --force
```

This script:

- Copies English content to French directory
- Adds translation notice markers
- Auto-translates common terms in headers
- Preserves all code blocks and technical terms

## Generate Reports

```bash
python3 generate_report.py
```

Creates:

- `translation_report.json` - Machine-readable status
- `TRANSLATION_STATUS.md` - Human-readable markdown report

## Current Status (October 6, 2025)

- ✅ **100% Coverage**: All 57 English files have French versions
- 🔄 **18 Files Updated**: Recently synchronized with translation markers
- 📊 **59 Total French Files**: Includes 2 French-only files

## Files Recently Updated

1. `api/features.md`, `api/gpu-api.md`
2. `gpu/features.md`, `gpu/overview.md`, `gpu/rgb-augmentation.md`
3. `workflows.md`
4. `guides/auto-params.md`, `guides/performance.md`
5. `features/format-preferences.md`, `features/lod3-classification.md`, `features/axonometry.md`
6. `reference/cli-download.md`, `reference/architectural-styles.md`, `reference/historical-analysis.md`
7. `tutorials/custom-features.md`
8. `mermaid-reference.md`
9. `release-notes/v1.6.2.md`, `release-notes/v1.7.1.md`

## Translation Guidelines

### Keep Unchanged

- Code blocks (Python, YAML, Bash)
- Command examples
- API names and signatures
- File paths and URLs
- Technical acronyms (GPU, LiDAR, RGB, etc.)

### Translate

- Paragraph text
- Headers and titles
- Frontmatter descriptions
- User-facing explanations

## Workflow

1. **Check Status**: `python3 check_translations.py`
2. **Analyze**: `python3 analyze_translations.py` for details
3. **Update**: `python3 update_fr_docs.py --force` to sync files
4. **Translate**: Manually translate content in `i18n/fr/docusaurus-plugin-content-docs/current/`
5. **Remove Notices**: Delete translation markers after complete translation
6. **Test**: `npm run build && npm run serve`
7. **Verify**: Check both English and French sites

## Directory Structure

```
website/
├── docs/                     # English (source)
│   ├── api/
│   ├── features/
│   ├── gpu/
│   └── ...
├── i18n/fr/                  # French translations
│   └── docusaurus-plugin-content-docs/
│       └── current/          # Mirrors docs/ structure
└── Translation tools:
    ├── analyze_translations.py
    ├── check_translations.py
    ├── update_fr_docs.py
    ├── generate_report.py
    └── README_TRANSLATION.md (this file)
```

## Testing

```bash
# Build site
npm run build

# Serve locally
npm run serve

# Visit http://localhost:3000/IGN_LIDAR_HD_DATASET/
# Use language selector to switch between English/French
```

## Maintenance

Run `analyze_translations.py` periodically to check for:

- New English files without French versions
- Outdated French translations
- Files with similar word counts (might be untranslated copies)

## See Also

- [DOCUSAURUS_UPDATE_SUMMARY.md](DOCUSAURUS_UPDATE_SUMMARY.md) - Complete update summary
- [TRANSLATION_STATUS.md](TRANSLATION_STATUS.md) - Current status report
- [translation_report.json](translation_report.json) - Machine-readable data
