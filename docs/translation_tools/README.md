# Docusaurus i18n Translation Tools

**Clean, consolidated translation management for Docusaurus French documentation.**

## 🎯 Quick Start

```bash
# Check translation status
python translation_tools/docusaurus_i18n.py status

# Synchronize EN → FR structure
python translation_tools/docusaurus_i18n.py sync

# Fix broken links automatically
python translation_tools/docusaurus_i18n.py fix-links

# Generate comprehensive report
python translation_tools/docusaurus_i18n.py report

# Run complete workflow
python translation_tools/docusaurus_i18n.py all
```

## 📚 Main Tool: `docusaurus_i18n.py`

**Single consolidated tool** that replaces all legacy scripts with clean, unified interface.

### Commands

#### `sync` - Synchronize Structure

Synchronizes French documentation structure with English, creating translation templates for new or untranslated files.

```bash
python docusaurus_i18n.py sync
python docusaurus_i18n.py sync --no-backup  # Skip backup (not recommended)
```

**What it does:**

- ✅ Creates missing French files from English sources
- ✅ Updates templates for untranslated files
- ✅ Adds translation markers (`🇫🇷 TRADUCTION FRANÇAISE REQUISE`)
- ✅ Preserves existing translations
- ✅ Creates automatic timestamped backups

#### `status` - Check Translation Status

Analyzes all documentation and reports translation coverage.

```bash
python docusaurus_i18n.py status
python docusaurus_i18n.py status --detailed  # Show per-file details
```

**Output:**

```
📊 Checking translation status...

📁 Total files: 59
✅ Translated: 45 (76.3%)
⏳ Needs translation: 14
❌ Missing: 0

📝 Files needing translation:
   • guides/performance.md
   • guides/auto-params.md
   ...
```

#### `validate` - Validate Links

Checks all links in documentation for common issues.

```bash
python docusaurus_i18n.py validate
```

**Detects:**

- `/docs/` prefixes (Docusaurus incompatibility)
- `.md` extensions in links
- Broken relative paths
- Malformed URLs

#### `fix-links` - Fix Broken Links

Automatically fixes common link issues in all documentation.

```bash
python docusaurus_i18n.py fix-links
```

**Fixes applied:**

1. Removes `/docs/` prefix from internal links
2. Removes `.md` extensions from relative links
3. Creates backup (`.bak`) before modifying

#### `report` - Generate Report

Creates comprehensive status report with all metrics.

```bash
python docusaurus_i18n.py report
python docusaurus_i18n.py report --output my_report.txt
```

**Includes:**

- Translation statistics
- Link validation results
- List of files needing translation
- Top link issues with suggested fixes

#### `all` - Complete Workflow

Runs entire workflow in sequence: sync → status → fix-links → report.

```bash
python docusaurus_i18n.py all
python docusaurus_i18n.py all --output workflow_report.txt
```

## 🏗️ Architecture

### Directory Structure

```
website/
├── docs/                       # English documentation (source)
├── i18n/
│   └── fr/
│       ├── docusaurus-plugin-content-docs/
│       │   └── current/        # French documentation (target)
│       └── backup/             # Automatic backups (timestamped)
└── translation_tools/
    ├── docusaurus_i18n.py      # Main consolidated tool
    ├── README.md               # This file
    └── archive/                # Legacy scripts (deprecated)
```

### How It Works

1. **Structure Synchronization**

   - Scans `docs/` for all `.md` files
   - Creates corresponding files in `i18n/fr/.../current/`
   - Uses heuristic detection to identify translated vs template files
   - Preserves existing translations

2. **Translation Detection** (Heuristic)

   - Checks for translation markers (`🇫🇷 TRADUCTION FRANÇAISE`)
   - Counts French vs English word indicators
   - Considers translated if French indicators > English indicators

3. **Link Fixing**

   - Pattern matching with regex
   - Creates backups before modification
   - Supports dry-run mode (validate without fixing)

4. **Backup System**
   - Timestamped directories: `backup/YYYYMMDD_HHMMSS/`
   - Complete copy of French docs before changes
   - Automatic in sync operations

## 📝 Translation Workflow

### For Manual Translation

1. **Check what needs translation:**

   ```bash
   python docusaurus_i18n.py status --detailed
   ```

2. **Open a template file:**

   ```bash
   code i18n/fr/docusaurus-plugin-content-docs/current/guides/performance.md
   ```

3. **Translate the content:**

   - Keep frontmatter intact (translate title/description)
   - Translate narrative text
   - Keep code blocks in English (translate comments if needed)
   - Keep markdown structure
   - Remove `🇫🇷 TRADUCTION FRANÇAISE` markers when done

4. **Test your translation:**

   ```bash
   npm run build      # Build both EN and FR
   npm start          # Preview locally
   ```

5. **Verify links:**
   ```bash
   python docusaurus_i18n.py validate
   ```

### Technical Glossary

Built-in glossary for consistent translations:

| English         | French           |
| --------------- | ---------------- |
| Quick Start     | Démarrage Rapide |
| Getting Started | Premiers Pas     |
| Installation    | Installation     |
| Configuration   | Configuration    |
| Performance     | Performance      |
| Point Cloud     | Nuage de Points  |
| Building        | Bâtiment         |
| GPU             | GPU              |

## 🔧 Advanced Usage

### Python API

```python
from pathlib import Path
from docusaurus_i18n import DocusaurusI18N

# Initialize
tool = DocusaurusI18N(website_root=Path("/path/to/website"))

# Synchronize
stats = tool.sync_structure(create_backups=True)
print(f"Created: {stats['created']}, Updated: {stats['updated']}")

# Check status
status = tool.check_status(detailed=True)
for file_info in status['files']:
    if file_info['status'] == 'needs_translation':
        print(f"Translate: {file_info['path']}")

# Validate and fix links
results = tool.validate_links(fix=True)
print(f"Fixed {results['fixed_links']} links")

# Generate report
report = tool.generate_report(output_file=Path("report.txt"))
```

### Integration with CI/CD

```yaml
# .github/workflows/docs-check.yml
name: Check Documentation
on: [push, pull_request]

jobs:
  check-translations:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Check translation status
        run: |
          cd website
          python translation_tools/docusaurus_i18n.py status
          python translation_tools/docusaurus_i18n.py validate
```

## 📊 Current Status

**As of October 6, 2025:**

- 📁 Total files: 59
- ✅ Translated: 45 (76%)
- ⏳ Needs translation: 14
- 🔗 Links fixed: 201
- 📦 Backups created: Multiple timestamped

**Ready for Phase 2:** Manual translation of remaining 14 files (~28-56 hours)

## 🗂️ Legacy Scripts (Archived)

The following scripts have been **consolidated** into `docusaurus_i18n.py` and moved to `archive/`:

- ❌ `sync_fr_docs.py` → Use `docusaurus_i18n.py sync`
- ❌ `check_translations.py` → Use `docusaurus_i18n.py status`
- ❌ `check_translation_status.py` → Use `docusaurus_i18n.py status --detailed`
- ❌ `fix_broken_links.py` → Use `docusaurus_i18n.py fix-links`
- ❌ `validate_links.py` → Use `docusaurus_i18n.py validate`
- ❌ `update_docs_comprehensive.py` → Use `docusaurus_i18n.py all`
- ❌ `generate_final_report.py` → Use `docusaurus_i18n.py report`
- ❌ `generate_report.py` → Use `docusaurus_i18n.py report`
- ❌ `update_fr_docs.py` → Use `docusaurus_i18n.py sync`
- ❌ `auto_translate.py` → Not needed (manual translation preferred)
- ❌ `translate_helpers.py` → Functionality integrated
- ❌ `analyze_translations.py` → Use `docusaurus_i18n.py status`
- ❌ `generate_missing_fr.py` → Use `docusaurus_i18n.py sync`

**Why consolidate?**

- ✅ Single tool with consistent interface
- ✅ Reduced complexity (13 scripts → 1 tool)
- ✅ Better maintainability
- ✅ Cleaner codebase
- ✅ Unified documentation

## 🚀 Quick Commands Reference

```bash
# Daily workflow
python docusaurus_i18n.py status              # Check progress
python docusaurus_i18n.py sync                # Sync any new files
python docusaurus_i18n.py fix-links           # Fix link issues

# After translating files
npm run build                                 # Build & test
python docusaurus_i18n.py validate            # Check links

# Generate reports
python docusaurus_i18n.py report              # Status report
python docusaurus_i18n.py all                 # Full workflow

# Deploy
npm run deploy                                # Deploy to GitHub Pages
```

## 📖 Related Documentation

- **START_HERE.md** - Project overview and quick start
- **PHASE_2_ACTION_PLAN.md** - Detailed translation plan
- **COMPLETION_SUMMARY.txt** - Project completion report

## 🆘 Troubleshooting

### "No module named 'docusaurus_i18n'"

Run from website root:

```bash
cd website
python translation_tools/docusaurus_i18n.py <command>
```

### "Build fails with broken links"

Fix links automatically:

```bash
python translation_tools/docusaurus_i18n.py fix-links
npm run build
```

### "French file exists but shows as needs_translation"

File still contains translation markers. Remove `🇫🇷 TRADUCTION FRANÇAISE` after translating.

### "Want to restore from backup"

Backups are in `i18n/fr/backup/YYYYMMDD_HHMMSS/`. Copy files back manually:

```bash
cp -r i18n/fr/backup/20251006_140000/* i18n/fr/docusaurus-plugin-content-docs/current/
```

---

**Last Updated:** October 6, 2025  
**Version:** 1.0.0 (Consolidated)  
**Status:** ✅ Production Ready
