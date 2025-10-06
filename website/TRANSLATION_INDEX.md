# Translation Analysis Reports - Index

Welcome! This directory contains comprehensive analysis of the Docusaurus French translation status.

## 🎯 Start Here

**New to this analysis?** Start with one of these:

1. **[TRANSLATION_QUICK_REFERENCE.md](TRANSLATION_QUICK_REFERENCE.md)** - Best starting point! Quick overview and action items
2. **[TRANSLATION_SUMMARY.txt](TRANSLATION_SUMMARY.txt)** - Plain text summary for quick reading

## 📚 All Reports

### Executive Summaries

| File                                                             | Description                             | Format   | Size     |
| ---------------------------------------------------------------- | --------------------------------------- | -------- | -------- |
| [TRANSLATION_QUICK_REFERENCE.md](TRANSLATION_QUICK_REFERENCE.md) | Quick reference guide with action items | Markdown | Detailed |
| [TRANSLATION_SUMMARY.txt](TRANSLATION_SUMMARY.txt)               | Plain text summary                      | Text     | 6.3 KB   |

### Detailed Analysis

| File                                                                       | Description                   | Format   | Size     |
| -------------------------------------------------------------------------- | ----------------------------- | -------- | -------- |
| [FRENCH_TRANSLATION_UPDATE_REPORT.md](FRENCH_TRANSLATION_UPDATE_REPORT.md) | Comprehensive analysis report | Markdown | Detailed |

### Data & Metrics

| File                                                             | Description                | Format | Size   |
| ---------------------------------------------------------------- | -------------------------- | ------ | ------ |
| [TRANSLATION_STATUS_REPORT.json](TRANSLATION_STATUS_REPORT.json) | Categorized priority lists | JSON   | 4.8 KB |
| [translation_update_needed.json](translation_update_needed.json) | Raw analysis metrics       | JSON   | 6.2 KB |
| [translation_report.json](translation_report.json)               | Basic comparison data      | JSON   | 272 B  |

### Tools & Scripts

| File                                                           | Description                  | Format | Size   |
| -------------------------------------------------------------- | ---------------------------- | ------ | ------ |
| [update_french_translations.py](update_french_translations.py) | Automated translation helper | Python | 6.5 KB |

## 🎯 By Use Case

### "I want a quick overview"

→ Start with **TRANSLATION_QUICK_REFERENCE.md**

### "I need to know what files to update"

→ Check the priority sections in **TRANSLATION_SUMMARY.txt**

### "I want detailed metrics for a specific file"

→ Use **translation_update_needed.json** and search for the filename

### "I need to understand the full methodology"

→ Read **FRENCH_TRANSLATION_UPDATE_REPORT.md**

### "I want to automate updates"

→ Use **update_french_translations.py**

### "I need machine-readable data"

→ Parse **TRANSLATION_STATUS_REPORT.json** or **translation_update_needed.json**

## 📊 Key Findings Summary

- **57 English files** vs **59 French files** (+2 extra)
- **32 files (56%)** need updates
- **25 files (44%)** are up-to-date
- **1 file** has been updated (guides/basic-usage.md)

### Priority Breakdown

| Priority  | Count | Threshold    | Timeline  |
| --------- | ----- | ------------ | --------- |
| 🔴 High   | 3     | >100% diff   | Immediate |
| 🟡 Medium | 8     | 40-100% diff | 2 weeks   |
| 🟢 Low    | 21    | 20-40% diff  | 1 month   |

## 🔴 High Priority Files (Update First!)

1. **api/configuration.md** (577% difference!)
2. **guides/qgis-troubleshooting.md** (168% difference)
3. **guides/preprocessing.md** (103% difference)

## ✅ Completed Updates

- ✓ **guides/basic-usage.md** - Fully updated to match English structure

## 🚀 Quick Commands

```bash
# View text summary
cat TRANSLATION_SUMMARY.txt

# View detailed report
cat TRANSLATION_QUICK_REFERENCE.md

# Check specific file metrics (requires jq)
cat translation_update_needed.json | jq '.[] | select(.file == "api/configuration.md")'

# View high priority files
cat TRANSLATION_STATUS_REPORT.json | jq '.high_priority'

# Run automation script
python3 update_french_translations.py

# Test Docusaurus build
npm run build

# Preview French site
npm start -- --locale fr
```

## 📋 Next Steps

1. **This Week:** Update 3 high priority files
2. **Next 2 Weeks:** Update 8 medium priority files
3. **Next Month:** Batch update 21 low priority files
4. **Ongoing:** Establish translation maintenance workflow

## 🛠️ Translation Guidelines

### DO Translate

- ✓ All user-facing text
- ✓ Comments in code
- ✓ Mermaid diagram labels
- ✓ File paths (raw_tiles → tuiles_brutes)

### DON'T Translate

- ✗ Code commands/variables
- ✗ "Patches", "Workflow" (keep English)
- ✗ Diagram node IDs

### Key Terminology

```
Point Cloud        → Nuage de points
Building Components → Composants de bâtiment
Geometric Features → Caractéristiques géométriques
Dataset            → Jeu de données
Training           → Entraînement
```

## 📞 Need Help?

- **For quick questions:** Check TRANSLATION_QUICK_REFERENCE.md
- **For detailed info:** Read FRENCH_TRANSLATION_UPDATE_REPORT.md
- **For metrics:** Parse translation_update_needed.json
- **For automation:** Use update_french_translations.py

## 📝 Report Information

- **Generated:** October 6, 2025
- **Analyzer:** GitHub Copilot
- **Method:** Content comparison (words, sections, structure)
- **Threshold:** Files with >20% difference flagged for update

---

**Ready to start?** Open [TRANSLATION_QUICK_REFERENCE.md](TRANSLATION_QUICK_REFERENCE.md) for the complete guide!
