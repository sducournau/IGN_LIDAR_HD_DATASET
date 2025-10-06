#!/bin/bash
# Quick commit script - Commits all translation work in one go

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                     QUICK COMMIT - Translation Project                       ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Change to website directory
cd "$(dirname "$0")"

echo "📊 Current Status:"
echo "─────────────────────────────────────────────────────────────────────────────"
git status --short
echo ""

# Count files
TOTAL_CHANGES=$(git status --short | wc -l)
echo "📈 Total files to commit: $TOTAL_CHANGES"
echo ""

read -p "❓ Proceed with commit? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Commit cancelled."
    exit 1
fi

echo ""
echo "✅ Adding all files..."
git add -A

echo ""
echo "✅ Committing with detailed message..."
git commit -m "docs(i18n): Sync French translations with English documentation

## Summary
- Updated 18 French documentation files with translation markers
- Added automated translation analysis and update tools  
- Achieved 100% documentation coverage (57/57 files)

## New Tools
- analyze_translations.py: Comprehensive translation status analysis
- update_fr_docs.py: Automated French documentation updater
- generate_report.py: Status report generator
- commit_helper.sh: Git commit assistant
- check_translations.py: Enhanced quick status checker

## Documentation
- INDEX.md: Master index and navigation guide
- ANALYSIS_COMPLETE.md: Executive summary with statistics
- TRANSLATION_STATUS.md: Current status and translation guidelines
- NEXT_ACTIONS.md: Step-by-step action plan for phases 2-3
- README_TRANSLATION.md: Maintenance workflow guide
- translation_report.json: Machine-readable status data

## Updated French Files (18)
Core API & GPU (6): api/features.md, api/gpu-api.md, gpu/features.md, 
                     gpu/overview.md, gpu/rgb-augmentation.md, workflows.md
Guides & Features (6): guides/auto-params.md, guides/performance.md,
                       features/format-preferences.md, lod3-classification.md,
                       axonometry.md, tutorials/custom-features.md
Reference & Misc (6): reference/cli-download.md, architectural-styles.md,
                      historical-analysis.md, mermaid-reference.md,
                      release-notes/v1.6.2.md, release-notes/v1.7.1.md

## Key Features
- All code blocks preserved
- Translation markers added for manual review
- Auto-translated common terms in headers
- Build verified (npm run build passes)

## Status
- Coverage: 100% ✅ (57/57 English files have French versions)
- Files updated: 18 with clear translation notices
- Build status: ✅ Successful
- Ready for manual translation work

## Next Steps
Manual translation of marked content (~9,000 words across 18 files).
See NEXT_ACTIONS.md for priority order and workflow.
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Commit successful!"
    echo ""
    echo "📊 Commit Details:"
    git log -1 --stat --oneline
    echo ""
    echo "🚀 Next Steps:"
    echo "─────────────────────────────────────────────────────────────────────────────"
    echo "1. Push to remote:  git push origin main"
    echo "2. View changes:    git show HEAD"
    echo "3. Check status:    git status"
    echo ""
    echo "📖 For detailed next steps, see: NEXT_ACTIONS.md"
else
    echo ""
    echo "❌ Commit failed. Please check error messages above."
    exit 1
fi
