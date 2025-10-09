# 🚀 Translation Workflow - Step by Step Guide

**Last Updated:** October 9, 2025  
**Status:** Phase 2 - Ready to translate Tier 1 files

---

## 📊 Current Status

```
Progress: ████░░░░░░░░░░░░░░░░ 27.4%

Tier 1 Files (8):  ⏳⏳⏳⏳⏳⏳⏳⏳ (0/8 complete)
Total Progress:    20/73 files (27.4%)
```

---

## 🎯 Strategy: Start with Shortest Files First

Build momentum by completing easier files first:

| Priority | File                             | Lines | Est. Time | Difficulty    |
| -------- | -------------------------------- | ----- | --------- | ------------- |
| 1️⃣       | `guides/qgis-troubleshooting.md` | 86    | 15 min    | ⭐ Easy       |
| 2️⃣       | `installation/quick-start.md`    | 213   | 30 min    | ⭐⭐ Easy     |
| 3️⃣       | `guides/quick-start.md`          | 419   | 60 min    | ⭐⭐ Medium   |
| 4️⃣       | `guides/troubleshooting.md`      | 465   | 60 min    | ⭐ Easy       |
| 5️⃣       | `features/multi-architecture.md` | 523   | 60 min    | ⭐⭐ Medium   |
| 6️⃣       | `guides/getting-started.md`      | 595   | 90 min    | ⭐⭐⭐ Medium |
| 7️⃣       | `guides/cli-commands.md`         | 722   | 90 min    | ⭐⭐⭐ Hard   |
| 8️⃣       | `architecture.md`                | 805   | 120 min   | ⭐⭐⭐ Hard   |

**Total:** ~8.5 hours

---

## 🔧 Translation Methods

### **Method 1: AI-Assisted (Recommended - Fastest)**

#### Using ChatGPT/Claude:

1. **Open the file:**

   ```bash
   code i18n/fr/docusaurus-plugin-content-docs/current/FILE.md
   ```

2. **Copy the entire content**

3. **Paste into ChatGPT/Claude with this prompt:**

   ```
   Translate this Docusaurus technical documentation from English to French.

   Requirements:
   - Preserve ALL markdown formatting (headings, lists, tables, links)
   - DO NOT translate code blocks or command examples
   - DO translate comments inside code blocks
   - Use these technical terms:
     * "point cloud" → "nuage de points"
     * "GPU acceleration" → "accélération GPU"
     * "building" → "bâtiment"
     * "preprocessing" → "prétraitement"
     * "troubleshooting" → "dépannage"
     * "quick start" → "démarrage rapide"
   - Keep all URLs unchanged
   - Keep all YAML frontmatter keys (title, description, etc.)
   - Translate YAML frontmatter values
   - Remove the translation marker: <!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->

   Here's the content:
   [PASTE CONTENT HERE]
   ```

4. **Copy the translated result back to the file**

5. **Test build:**

   ```bash
   npm run build
   ```

6. **Preview:**
   ```bash
   npm run start -- --locale fr
   ```

---

### **Method 2: DeepL API (High Quality)**

If you have a DeepL Pro API key:

```bash
cd website

# Install DeepL
pip install deepl

# Use the API (you'll need to create a script)
# See PHASE2_ACTION_PLAN.md for full script
```

---

### **Method 3: Semi-Manual with Helper**

Use the helper script to remove markers and apply glossary:

```bash
cd website

# Prepare file (removes markers, applies glossary to non-code text)
python3 prepare_translation.py i18n/fr/docusaurus-plugin-content-docs/current/FILE.md --auto

# Then manually review and complete translation
code i18n/fr/docusaurus-plugin-content-docs/current/FILE.md
```

---

## ✅ Translation Checklist (For Each File)

### Before Translation:

- [ ] File opened in editor
- [ ] English version open for reference
- [ ] Glossary handy (see below)

### During Translation:

- [ ] Frontmatter title translated
- [ ] Frontmatter description translated
- [ ] Translation marker removed
- [ ] All headings translated
- [ ] All paragraphs translated
- [ ] Code blocks preserved (NOT translated)
- [ ] Comments in code translated
- [ ] Image alt text translated
- [ ] Table headers translated
- [ ] Links checked (no `.md`, no `/docs/`)

### After Translation:

- [ ] Build test: `npm run build`
- [ ] Visual check: `npm run start -- --locale fr`
- [ ] Links work correctly
- [ ] No broken formatting
- [ ] Commit with message: `feat(docs): translate FILE.md to French`

---

## 📚 Quick Reference Glossary

| English             | French                 | Context         |
| ------------------- | ---------------------- | --------------- |
| Point Cloud         | Nuage de Points        | Data type       |
| Building            | Bâtiment               | Structure       |
| GPU Acceleration    | Accélération GPU       | Feature         |
| Quick Start         | Démarrage Rapide       | Guide type      |
| Getting Started     | Premiers Pas           | Onboarding      |
| Installation        | Installation           | Process         |
| Troubleshooting     | Dépannage              | Support         |
| Processing Pipeline | Pipeline de Traitement | Architecture    |
| Tile                | Dalle                  | LiDAR unit      |
| Feature             | Caractéristique        | Data attribute  |
| Classification      | Classification         | ML task         |
| Neighborhood        | Voisinage              | Spatial context |
| Preprocessing       | Prétraitement          | Data prep       |
| Download            | Téléchargement         | Action          |
| Workflow            | Flux de travail        | Process         |
| Dataset             | Jeu de données         | Collection      |
| Configuration       | Configuration          | Settings        |
| Parameter           | Paramètre              | Variable        |

---

## 🎯 Today's Goal: First 3 Files (2-3 hours)

### Session 1: Quick Wins (1 hour)

**File 1: `guides/qgis-troubleshooting.md` (15 min)**

```bash
code i18n/fr/docusaurus-plugin-content-docs/current/guides/qgis-troubleshooting.md
```

- Shortest file (86 lines)
- Structured Q&A format
- Build confidence!

**File 2: `installation/quick-start.md` (30 min)**

```bash
code i18n/fr/docusaurus-plugin-content-docs/current/installation/quick-start.md
```

- Installation instructions
- Mostly commands (easy)
- Critical for users

**File 3: `guides/quick-start.md` (60 min)**

```bash
code i18n/fr/docusaurus-plugin-content-docs/current/guides/quick-start.md
```

- Usage examples
- Concrete code samples
- Very useful for beginners

---

## 🔥 Quick Commands

```bash
# Navigate to website directory
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website

# Check status
python3 translation_tools/docusaurus_i18n.py status

# Prepare a file (remove markers + apply glossary)
python3 prepare_translation.py i18n/fr/docusaurus-plugin-content-docs/current/FILE.md --auto

# Build
npm run build

# Preview French site (then open http://localhost:3000/IGN_LIDAR_HD_DATASET/fr/)
npm run start -- --locale fr

# Fix links
python3 translation_tools/docusaurus_i18n.py fix-links

# Generate report
python3 translation_tools/docusaurus_i18n.py report
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: YAML Frontmatter Error

```
Error: can not read a block mapping entry
```

**Solution:** Remove quotes around title/description or escape them properly

```yaml
# ❌ Bad
title: "Guide" de "Démarrage"

# ✅ Good
title: Guide de Démarrage
```

### Issue 2: Broken Links

```
Error: Broken link
```

**Solution:** Remove `.md` extension and `/docs/` prefix

```markdown
❌ [Guide](/docs/guides/quick-start.md)
✅ [Guide](/guides/quick-start)
```

### Issue 3: Code Block Formatting

**Solution:** Preserve code blocks exactly, translate only comments

```python
# ❌ Don't do this
données = télécharger_lidar()

# ✅ Do this
# Télécharger les données LiDAR
data = download_lidar()
```

---

## 📊 Progress Tracking

After each file, run:

```bash
python3 << 'EOF'
from pathlib import Path
fr_docs = Path('i18n/fr/docusaurus-plugin-content-docs/current')
all_files = list(fr_docs.rglob('*.md'))
remaining = sum(1 for f in all_files if '🇫🇷 TRADUCTION FRANÇAISE' in f.read_text(encoding='utf-8'))
done = len(all_files) - remaining
print(f"Progress: {done}/{len(all_files)} ({done/len(all_files)*100:.1f}%)")
print(f"Remaining: {remaining}")
EOF
```

---

## 🎉 Motivation Tracker

- [ ] 🥉 **Bronze:** Translate 3 files (reach 30%)
- [ ] 🥈 **Silver:** Translate 8 files (complete Tier 1)
- [ ] 🥇 **Gold:** Translate 25 files (reach 50%)
- [ ] 💎 **Diamond:** Translate all 73 files (100%)

---

## 🚀 Ready to Start?

**Recommended first file:**

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website
code i18n/fr/docusaurus-plugin-content-docs/current/guides/qgis-troubleshooting.md
```

Use AI assistance (ChatGPT) with the prompt above for fastest results!

**After translating 3 files today:**

- Progress will be 31.5% (23/73)
- Tier 1 will be 37.5% complete (3/8)
- You'll have translated ~718 lines
- Build momentum for tomorrow! 💪

---

Good luck! You've got this! 🎯
