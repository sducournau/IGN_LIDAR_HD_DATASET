# 🎯 Quick Start Guide - Translating French Documentation

**Current Status:** 20/73 files (27.4%) | **Goal:** Complete Tier 1 (8 files)

---

## ✅ Phase 1 Complete!

- [x] Structure synchronized (73 FR files created)
- [x] YAML frontmatter fixed (12 files)
- [x] Build validated (SUCCESS for EN + FR)
- [x] Backup created

---

## 🚀 Phase 2: Priority Files (START HERE)

### **Tier 1 - Critical Path (8 files)** ⭐⭐⭐

These 8 files cover the essential user journey:

| #   | File                             | Priority | Status          |
| --- | -------------------------------- | -------- | --------------- |
| 1   | `installation/quick-start.md`    | ⭐⭐⭐   | ⏳ To translate |
| 2   | `guides/quick-start.md`          | ⭐⭐⭐   | ⏳ To translate |
| 3   | `guides/getting-started.md`      | ⭐⭐⭐   | ⏳ To translate |
| 4   | `guides/cli-commands.md`         | ⭐⭐⭐   | ⏳ To translate |
| 5   | `architecture.md`                | ⭐⭐⭐   | ⏳ To translate |
| 6   | `guides/troubleshooting.md`      | ⭐⭐⭐   | ⏳ To translate |
| 7   | `guides/qgis-troubleshooting.md` | ⭐⭐⭐   | ⏳ To translate |
| 8   | `features/multi-architecture.md` | ⭐⭐⭐   | ⏳ To translate |

---

## 🔧 Translation Workflow

### **Step-by-Step Process:**

1. **Open file** to translate:

   ```bash
   code i18n/fr/docusaurus-plugin-content-docs/current/FILE.md
   ```

2. **Translate frontmatter** (top section between `---`):

   ```yaml
   ---
   title: Translated Title Here
   description: Translated description here
   ---
   ```

3. **Remove translation marker** - Delete these lines:

   ```html
   <!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->
   <!-- Ce fichier est un modèle... -->
   ```

4. **Translate content**:

   - ✅ Translate all text
   - ✅ Translate comments in code
   - ✅ Translate image alt text
   - ❌ DON'T translate code itself
   - ❌ DON'T change IDs, slugs, links

5. **Test build**:

   ```bash
   npm run build
   ```

6. **Preview locally**:
   ```bash
   npm run start -- --locale fr
   ```

---

## 📚 Translation Glossary

Use these consistently:

| English             | French                 |
| ------------------- | ---------------------- |
| Point Cloud         | Nuage de Points        |
| Building            | Bâtiment               |
| GPU Acceleration    | Accélération GPU       |
| Quick Start         | Démarrage Rapide       |
| Getting Started     | Premiers Pas           |
| Installation        | Installation           |
| Troubleshooting     | Dépannage              |
| Processing Pipeline | Pipeline de Traitement |
| Tile                | Dalle                  |
| Feature             | Caractéristique        |
| Classification      | Classification         |
| Neighborhood        | Voisinage              |
| RGB Augmentation    | Augmentation RGB       |
| Preprocessing       | Prétraitement          |
| Download            | Téléchargement         |
| Upload              | Téléversement          |
| Workflow            | Flux de travail        |
| Dataset             | Jeu de données         |
| Output              | Sortie                 |
| Input               | Entrée                 |
| Configuration       | Configuration          |
| Parameter           | Paramètre              |

---

## 🎯 Translation Options

### **Option A: Use AI Assistant (Fastest)**

1. Copy file content
2. Ask ChatGPT/Claude:
   > "Translate this Docusaurus markdown documentation to French. Preserve all markdown formatting, code blocks (don't translate code), YAML frontmatter structure, and use these terms: 'nuage de points' for point cloud, 'accélération GPU' for GPU acceleration, etc."
3. Paste result back
4. Review and adjust

### **Option B: Use DeepL (High Quality)**

1. Get API key: https://www.deepl.com/pro-api
2. Use translation script (see PHASE2_ACTION_PLAN.md)
3. Review output

### **Option C: Manual Translation**

Best for short files or when you want full control.

---

## ⚠️ Common Pitfalls

1. **Don't translate code blocks**

   ```python
   # ✅ Translate this comment: Télécharger les données
   data = download_lidar()  # ❌ DON'T translate function names
   ```

2. **Keep links without /docs/ or .md**

   - ❌ `[Guide](/docs/guides/quick-start.md)`
   - ✅ `[Guide](/guides/quick-start)`

3. **Preserve YAML structure**

   ```yaml
   # ✅ Good
   title: Guide de Démarrage Rapide

   # ❌ Bad (unescaped quotes)
   title: "Guide" de "Démarrage Rapide"
   ```

4. **Keep emoji and icons**
   ```markdown
   ## ⚡ GPU Acceleration → ## ⚡ Accélération GPU
   ```

---

## 📊 Progress Tracking

### Check status anytime:

```bash
cd website
python3 translation_tools/docusaurus_i18n.py status
```

### Count remaining:

```bash
grep -r '🇫🇷 TRADUCTION FRANÇAISE REQUISE' i18n/fr/docusaurus-plugin-content-docs/current --include='*.md' | wc -l
```

### Current:

- **Total:** 73 files
- **Translated:** 20 files (27.4%)
- **Remaining:** 53 files (72.6%)

---

## 🎯 Today's Goal

**Target:** Translate 3-5 Tier 1 files

Start with the easiest:

1. `installation/quick-start.md` (shorter, mostly instructions)
2. `guides/quick-start.md` (concise examples)
3. `guides/troubleshooting.md` (structured Q&A format)

---

## 🔥 Quick Commands Reference

```bash
# Navigate to website
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website

# Check status
python3 translation_tools/docusaurus_i18n.py status

# Build (test after each file)
npm run build

# Preview French site
npm run start -- --locale fr
# Then open: http://localhost:3000/IGN_LIDAR_HD_DATASET/fr/

# Fix links automatically
python3 translation_tools/docusaurus_i18n.py fix-links

# Generate report
python3 translation_tools/docusaurus_i18n.py report
```

---

## 💡 Pro Tips

1. **Translate in batches of 2-3 files**, then test build
2. **Start with shorter files** to build momentum
3. **Use "Find & Replace"** for consistent terminology
4. **Keep original open** in split screen for reference
5. **Test locally** before committing large batches

---

## 📝 File Sizes (Estimated Translation Time)

| File                             | Lines | Time   | Difficulty |
| -------------------------------- | ----- | ------ | ---------- |
| `installation/quick-start.md`    | ~215  | 30 min | Easy       |
| `guides/quick-start.md`          | ~419  | 60 min | Medium     |
| `guides/troubleshooting.md`      | ~467  | 60 min | Easy       |
| `guides/cli-commands.md`         | ~500  | 90 min | Medium     |
| `guides/getting-started.md`      | ~600  | 90 min | Medium     |
| `architecture.md`                | ~400  | 60 min | Hard       |
| `guides/qgis-troubleshooting.md` | ~300  | 45 min | Easy       |
| `features/multi-architecture.md` | ~200  | 30 min | Easy       |

**Total Tier 1:** ~6-7 hours

---

## ✅ Success Checklist

After translating each file:

- [ ] Translation marker removed
- [ ] Frontmatter translated
- [ ] All text content translated
- [ ] Code blocks preserved (not translated)
- [ ] Comments in code translated
- [ ] Links checked (no .md, no /docs/)
- [ ] Build successful: `npm run build`
- [ ] Preview checked: `npm run start -- --locale fr`

---

## 🚀 Ready? Let's Go!

**Start with:** `installation/quick-start.md`

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website
code i18n/fr/docusaurus-plugin-content-docs/current/installation/quick-start.md
```

Good luck! 🎉
