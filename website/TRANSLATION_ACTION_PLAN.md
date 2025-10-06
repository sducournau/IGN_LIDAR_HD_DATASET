# French Translation Action Plan

**Created:** October 6, 2025  
**Status:** ✅ Analysis Complete | 🔄 Ready for Updates  
**Build Status:** ✅ Passing

---

## 🎯 Mission

Update 32 French documentation files to match the current English structure and content.

## 📊 Current Status

- **Total Files:** 57 English / 59 French
- **Up-to-date:** 25 files (44%)
- **Need Updates:** 32 files (56%)
- **Completed Today:** 1 file (guides/basic-usage.md)
- **Build Status:** ✅ Passing

---

## Week 1: High Priority Files (3 files)

### 🔴 CRITICAL - Complete This Week

#### [ ] 1. api/configuration.md
- **Difference:** 577% (French is 6x longer!)
- **Issue:** French version contains obsolete content
- **Action:** Complete rewrite to match English structure
- **EN:** 152 words | **FR:** 1,029 words
- **Time Estimate:** 2-3 hours

**Steps:**
1. Read English version: `docs/api/configuration.md`
2. Compare with French: `i18n/fr/.../api/configuration.md`
3. Rewrite French to match English structure
4. Test: `npm run build`

---

#### [ ] 2. guides/qgis-troubleshooting.md
- **Difference:** 168%
- **Issue:** French version has extra outdated content
- **Action:** Remove obsolete sections, align with English
- **EN:** 142 words | **FR:** 380 words
- **Time Estimate:** 1-2 hours

**Steps:**
1. Read English version
2. Identify obsolete French sections
3. Update French to match current English
4. Test build

---

#### [ ] 3. guides/preprocessing.md
- **Difference:** 103%
- **Issue:** French version is double the size
- **Action:** Major cleanup and restructuring
- **EN:** 1,056 words | **FR:** 2,147 words
- **Time Estimate:** 3-4 hours

**Steps:**
1. Read English version carefully
2. Identify extra French content
3. Restructure to match English
4. Update examples and code snippets
5. Test build

---

### Week 1 Checklist
- [ ] Update api/configuration.md
- [ ] Update guides/qgis-troubleshooting.md
- [ ] Update guides/preprocessing.md
- [ ] Run `npm run build` after each update
- [ ] Preview French site: `npm start -- --locale fr`
- [ ] Commit changes with clear messages

**Estimated Time:** 6-9 hours total

---

## Week 2-3: Medium Priority Files (8 files)

### 🟡 IMPORTANT - Complete in 2 Weeks

#### [ ] 4. reference/workflow-diagrams.md (88.5% diff)
- **Time:** 30 min

#### [ ] 5. api/rgb-augmentation.md (81.5% diff)
- **Time:** 1 hour

#### [ ] 6. reference/config-examples.md (71.1% diff)
- **Time:** 45 min

#### [ ] 7. guides/regional-processing.md (70.6% diff)
- **Time:** 1.5 hours

#### [ ] 8. release-notes/v1.7.4.md (57.6% diff)
- **Time:** 1 hour

#### [ ] 9. guides/qgis-integration.md (50.5% diff)
- **Time:** 2 hours

#### [ ] 10. guides/quick-start.md (42.5% diff)
- **Time:** 1.5 hours

#### [ ] 11. features/auto-params.md (40.6% diff)
- **Time:** 45 min

**Estimated Time:** 9-10 hours total

---

## Week 4+: Low Priority Files (21 files)

### 🟢 MAINTENANCE - Complete in 1 Month

<details>
<summary>Click to expand full list (21 files)</summary>

#### [ ] 12. guides/getting-started.md (35.3% diff)
#### [ ] 13. reference/memory-optimization.md (35.2% diff)
#### [ ] 14. installation/quick-start.md (34.8% diff)
#### [ ] 15. release-notes/v1.7.5.md (33.5% diff)
#### [ ] 16. features/smart-skip.md (35.6% diff)
#### [ ] 17. reference/cli-qgis.md (36.1% diff)
#### [ ] 18. guides/cli-commands.md (25.6% diff)
#### [ ] 19. release-notes/v1.7.3.md (27.9% diff)
#### [ ] 20. api/processor.md
#### [ ] 21. reference/cli-patch.md
#### [ ] 22. api/cli.md
#### [ ] 23. features/architectural-styles.md
#### [ ] 24. api/features.md
#### [ ] 25. features/pipeline-configuration.md
#### [ ] 26. release-notes/v1.7.2.md
#### [ ] 27. gpu/overview.md
#### [ ] 28. gpu/features.md
#### [ ] 29. reference/cli-enrich.md
#### [ ] 30. release-notes/v1.6.2.md
#### [ ] 31. gpu/rgb-augmentation.md
#### [ ] 32. guides/performance.md

</details>

**Estimated Time:** 12-15 hours total

---

## Extra Tasks

### Remove Extra French Files

#### [ ] Review examples/index.md
- **Status:** Exists in French only
- **Action:** Determine if needed, remove or create English version

#### [ ] Review guides/visualization.md
- **Status:** Exists in French only
- **Action:** Determine if needed, remove or create English version

---

## 🛠️ Tools & Resources

### Reference Documents
- **Start Here:** [TRANSLATION_INDEX.md](TRANSLATION_INDEX.md)
- **Quick Guide:** [TRANSLATION_QUICK_REFERENCE.md](TRANSLATION_QUICK_REFERENCE.md)
- **Full Report:** [FRENCH_TRANSLATION_UPDATE_REPORT.md](FRENCH_TRANSLATION_UPDATE_REPORT.md)

### Data Files
- **Metrics:** translation_update_needed.json
- **Priorities:** TRANSLATION_STATUS_REPORT.json

### Commands
```bash
# Check specific file metrics
cat translation_update_needed.json | jq '.[] | select(.file == "api/configuration.md")'

# Build and test
npm run build

# Preview French site
npm start -- --locale fr

# Use automation helper
python3 update_french_translations.py
```

---

## 📝 Translation Guidelines

### DO Translate
✅ All user-facing text  
✅ Comments in code  
✅ Mermaid diagram labels  
✅ File paths (raw_tiles → tuiles_brutes)

### DON'T Translate
❌ Code commands/variables  
❌ "Patches", "Workflow" (keep English)  
❌ Diagram node IDs

### Key Terminology
```
Point Cloud        → Nuage de points
Building Components → Composants de bâtiment
Geometric Features → Caractéristiques géométriques
Dataset            → Jeu de données
Training           → Entraînement
Classification     → Classification
```

---

## ✅ Quality Checklist

After updating each file:

- [ ] Content matches English structure
- [ ] All sections present
- [ ] Code examples updated
- [ ] Diagrams translated
- [ ] Links work correctly
- [ ] Build passes: `npm run build`
- [ ] French site previews correctly
- [ ] No broken links or images

---

## 📈 Progress Tracking

### Overall Progress
```
Week 1: [░░░░░░░░░░] 0/3 High Priority
Week 2: [░░░░░░░░░░] 0/8 Medium Priority  
Week 4: [░░░░░░░░░░] 0/21 Low Priority
Total:  [█░░░░░░░░░] 1/32 (3.1%)
```

### Update This After Each Completion
- ✅ guides/basic-usage.md (Week 0)
- ⏳ api/configuration.md (Week 1)
- ⏳ guides/qgis-troubleshooting.md (Week 1)
- ⏳ guides/preprocessing.md (Week 1)

---

## 🎯 Success Criteria

### Week 1 Success
- ✅ 3 high priority files updated
- ✅ Build passes
- ✅ French site works correctly

### Week 3 Success
- ✅ 11 total files updated (high + medium)
- ✅ Build passes
- ✅ All updated sections tested

### Final Success
- ✅ All 32 files updated
- ✅ Extra files removed/migrated
- ✅ Translation workflow documented
- ✅ Build passes consistently
- ✅ French site fully functional

---

## 📞 Need Help?

### Resources
- **Quick Start:** TRANSLATION_INDEX.md
- **Detailed Guide:** FRENCH_TRANSLATION_UPDATE_REPORT.md
- **Automation:** update_french_translations.py

### Common Issues
1. **Build fails:** Check markdown syntax and frontmatter
2. **Links broken:** Verify relative paths
3. **Diagrams not rendering:** Check Mermaid syntax

---

## 📅 Timeline Summary

| Week | Tasks | Files | Hours | Priority |
|------|-------|-------|-------|----------|
| 1 | High Priority | 3 | 6-9 | 🔴 Critical |
| 2-3 | Medium Priority | 8 | 9-10 | 🟡 Important |
| 4+ | Low Priority | 21 | 12-15 | 🟢 Maintenance |
| **Total** | **All Updates** | **32** | **27-34** | |

**Target Completion:** End of October 2025

---

**Last Updated:** October 6, 2025  
**Next Review:** After completing Week 1 tasks

---

🚀 **Ready to start?** Begin with `api/configuration.md` - the highest priority file!
