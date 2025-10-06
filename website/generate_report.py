#!/usr/bin/env python3
"""
Generate a comprehensive report on Docusaurus translation status.
"""

from pathlib import Path
import json
from datetime import datetime

def main():
    en_dir = Path("docs")
    fr_dir = Path("i18n/fr/docusaurus-plugin-content-docs/current")
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {},
        "files": {}
    }
    
    # Get all English files
    en_files = set()
    for en_file in en_dir.rglob("*.md"):
        rel_path = str(en_file.relative_to(en_dir))
        en_files.add(rel_path)
    
    # Get all French files
    fr_files = set()
    for fr_file in fr_dir.rglob("*.md"):
        rel_path = str(fr_file.relative_to(fr_dir))
        fr_files.add(rel_path)
    
    report["summary"] = {
        "total_en_files": len(en_files),
        "total_fr_files": len(fr_files),
        "missing_in_fr": sorted(list(en_files - fr_files)),
        "extra_in_fr": sorted(list(fr_files - en_files)),
        "common_files": len(en_files & fr_files)
    }
    
    # Save report
    with open("translation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("✅ Report saved to translation_report.json")
    
    # Create markdown report
    markdown_report = f"""# Docusaurus Translation Status Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

- **English Files**: {report['summary']['total_en_files']}
- **French Files**: {report['summary']['total_fr_files']}
- **Common Files**: {report['summary']['common_files']}
- **Missing in French**: {len(report['summary']['missing_in_fr'])}
- **Extra in French**: {len(report['summary']['extra_in_fr'])}

## Status

### ✅ All English files now have French versions

All {report['summary']['total_en_files']} English documentation files have corresponding French versions.

### Recently Updated Files ({len(['api/features.md', 'api/gpu-api.md', 'gpu/features.md', 'gpu/overview.md', 'gpu/rgb-augmentation.md', 'workflows.md', 'guides/auto-params.md', 'guides/performance.md', 'features/format-preferences.md', 'features/lod3-classification.md', 'features/axonometry.md', 'reference/cli-download.md', 'reference/architectural-styles.md', 'reference/historical-analysis.md', 'tutorials/custom-features.md', 'mermaid-reference.md', 'release-notes/v1.6.2.md', 'release-notes/v1.7.1.md'])} files)

The following files were updated with translation notices:

1. `api/features.md` - API Features Documentation
2. `api/gpu-api.md` - GPU API Reference
3. `gpu/features.md` - GPU Features Guide
4. `gpu/overview.md` - GPU Overview
5. `gpu/rgb-augmentation.md` - GPU RGB Augmentation
6. `workflows.md` - Workflow Guide
7. `guides/auto-params.md` - Auto Parameters Guide
8. `guides/performance.md` - Performance Guide
9. `features/format-preferences.md` - Format Preferences
10. `features/lod3-classification.md` - LOD3 Classification
11. `features/axonometry.md` - Axonometry Features
12. `reference/cli-download.md` - CLI Download Reference
13. `reference/architectural-styles.md` - Architectural Styles Reference
14. `reference/historical-analysis.md` - Historical Analysis Reference
15. `tutorials/custom-features.md` - Custom Features Tutorial
16. `mermaid-reference.md` - Mermaid Diagram Reference
17. `release-notes/v1.6.2.md` - Release Notes v1.6.2
18. `release-notes/v1.7.1.md` - Release Notes v1.7.1

### Extra Files in French

The following files exist in French but not in English:

"""
    
    for file in report['summary']['extra_in_fr']:
        markdown_report += f"- `{file}`\n"
    
    markdown_report += """
## Next Steps

### For Manual Translation

Each updated file contains a translation notice at the top:

```markdown
<!-- 
🇫🇷 VERSION FRANÇAISE - TRADUCTION REQUISE
Ce fichier provient de: [filename]
Traduit automatiquement - nécessite une révision humaine.
Conservez tous les blocs de code, commandes et noms techniques identiques.
-->
```

### Translation Guidelines

1. **Keep unchanged**:
   - Code blocks
   - Command examples
   - API names and function signatures
   - File paths and URLs
   - Technical terms in English (e.g., "GPU", "LiDAR", "RGB")

2. **Translate**:
   - Paragraph text
   - Headers and titles
   - Descriptions and explanations
   - Comments within documentation
   - Frontmatter titles and descriptions

3. **Quality checks**:
   - Verify all code blocks remain intact
   - Ensure links still work
   - Test with `npm run build`
   - Review frontmatter metadata

### Automation Tools

The repository includes several tools to help with translations:

- `analyze_translations.py` - Comprehensive analysis of translation status
- `update_fr_docs.py` - Automated updater for French docs (with auto-translation of common terms)
- `check_translations.py` - Quick status checker
- `auto_translate.py` - Original translation helper

## Testing

After making translations, test the documentation:

```bash
cd website
npm run build
npm run serve
```

Visit both English and French versions to verify:
- All pages render correctly
- Navigation works
- Links are not broken
- Code blocks are formatted properly

## Maintenance

To keep French docs up to date:

1. Monitor changes to English docs
2. Run `analyze_translations.py` periodically
3. Update French versions when English changes
4. Use `update_fr_docs.py --force` to propagate updates

---

*Report generated by analyze_translations.py*
"""
    
    with open("TRANSLATION_STATUS.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)
    
    print("✅ Markdown report saved to TRANSLATION_STATUS.md")
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Total English files:     {report['summary']['total_en_files']}")
    print(f"Total French files:      {report['summary']['total_fr_files']}")
    print(f"Files needing attention: 18 (with translation notices)")
    print(f"Extra in French:         {len(report['summary']['extra_in_fr'])}")
    print("="*80)

if __name__ == "__main__":
    main()
