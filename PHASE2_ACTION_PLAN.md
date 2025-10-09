# 🚀 Phase 2 Action Plan - Critical Files Translation

**Date:** October 9, 2025  
**Current Status:** 20/73 files translated (27.4%)  
**Remaining:** 53 files to translate

---

## 🎯 Priority Translation Order

### **Tier 1: Critical User Journey** (9 files) ⭐⭐⭐

Must translate first - these cover 90% of user interactions:

1. **`installation/quick-start.md`** - Installation guide
2. **`guides/quick-start.md`** - First 5 minutes
3. **`guides/getting-started.md`** - Complete onboarding
4. **`guides/cli-commands.md`** - CLI reference
5. **`architecture.md`** - System overview
6. **`guides/troubleshooting.md`** - Problem solving
7. **`guides/qgis-troubleshooting.md`** - QGIS issues
8. **`features/multi-architecture.md`** - Multi-arch support

### **Tier 2: Advanced Guides** (14 files) ⭐⭐

For users going deeper:

9. **`guides/auto-params.md`**
10. **`guides/complete-workflow.md`**
11. **`guides/configuration-system.md`**
12. **`guides/gpu-acceleration.md`**
13. **`guides/hydra-cli.md`**
14. **`guides/migration-v1-to-v2.md`**
15. **`guides/performance.md`**
16. **`guides/preprocessing.md`**
17. **`guides/qgis-integration.md`**
18. **`guides/regional-processing.md`**
19. **`guides/unified-pipeline.md`**
20. `gpu/overview.md`
21. `gpu/rgb-augmentation.md`

### **Tier 3: Features & API** (16 files) ⭐

Technical documentation:

22-26. API docs (5 files)
27-32. Features (8 files)
33-37. Reference docs (5 files)

### **Tier 4: Release Notes** (13 files) ⭐

Version documentation:

38-50. Release notes (13 files)

### **Tier 5: Supplementary** (2 files)

51. `mermaid-reference.md`
52. `tutorials/custom-features.md`

---

## 🤖 Translation Automation Options

### **Option 1: DeepL API (Recommended)**

Best quality for French technical translation.

#### Setup:

```bash
pip install deepl
```

#### Script:

```python
import deepl
from pathlib import Path

auth_key = "YOUR_DEEPL_API_KEY"  # Get from https://www.deepl.com/pro-api
translator = deepl.Translator(auth_key)

def translate_file(md_file):
    content = md_file.read_text(encoding='utf-8')

    # Extract frontmatter
    if content.startswith('---'):
        parts = content.split('---', 2)
        frontmatter = parts[1]
        body = parts[2]

        # Translate body (preserve code blocks)
        # Implementation here...

        # Combine and save
        # ...
```

**Cost:** €20/month for 500K characters

### **Option 2: Google Cloud Translation**

```bash
pip install google-cloud-translate
```

#### Script:

```python
from google.cloud import translate_v2 as translate

client = translate.Client()

def translate_text(text, target='fr'):
    result = client.translate(text, target_language=target)
    return result['translatedText']
```

**Cost:** $20 per 1M characters

### **Option 3: Azure Translator**

```bash
pip install azure-ai-translation-text
```

Good integration if you're already using Azure.

### **Option 4: Manual + AI Assistant**

Use ChatGPT/Claude for translation:

1. Copy markdown file content
2. Prompt: "Translate this technical documentation to French, preserving all markdown formatting, code blocks, and technical terms. Use 'nuage de points' for 'point cloud', 'accélération GPU' for 'GPU acceleration', etc."
3. Review and paste back
4. Test build

---

## 🛠️ Semi-Automated Translation Script

Here's a script to help with batch translation:

````python
#!/usr/bin/env python3
"""
translate_batch.py - Semi-automated translation helper
"""
import re
from pathlib import Path
from typing import List, Tuple

# Technical glossary
GLOSSARY = {
    "Point Cloud": "Nuage de Points",
    "point cloud": "nuage de points",
    "Building": "Bâtiment",
    "building": "bâtiment",
    "LiDAR": "LiDAR",
    "GPU Acceleration": "Accélération GPU",
    "gpu acceleration": "accélération GPU",
    "Quick Start": "Démarrage Rapide",
    "Getting Started": "Premiers Pas",
    "Installation": "Installation",
    "Troubleshooting": "Dépannage",
    "Processing Pipeline": "Pipeline de Traitement",
    "Tile": "Dalle",
    "Feature": "Caractéristique",
    "Classification": "Classification",
    "Neighborhood": "Voisinage",
    "RGB Augmentation": "Augmentation RGB",
    "Preprocessing": "Prétraitement",
}

def preserve_code_blocks(text: str) -> Tuple[str, List[str]]:
    """Extract code blocks and replace with placeholders."""
    code_blocks = []
    pattern = r'```[\s\S]*?```'

    def replacer(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"

    text = re.sub(pattern, replacer, text)
    return text, code_blocks

def restore_code_blocks(text: str, code_blocks: List[str]) -> str:
    """Restore code blocks from placeholders."""
    for i, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)
    return text

def apply_glossary(text: str) -> str:
    """Apply glossary substitutions."""
    for en, fr in GLOSSARY.items():
        text = text.replace(en, fr)
    return text

def translate_frontmatter(frontmatter: str) -> str:
    """Translate frontmatter fields."""
    lines = []
    for line in frontmatter.split('\n'):
        if line.startswith('title:') or line.startswith('description:'):
            # Extract and translate
            key, value = line.split(':', 1)
            value = apply_glossary(value.strip())
            lines.append(f"{key}: {value}")
        else:
            lines.append(line)
    return '\n'.join(lines)

def process_file(file_path: Path, use_deepl: bool = False):
    """Process a single markdown file."""
    content = file_path.read_text(encoding='utf-8')

    # Check if already translated
    if '🇫🇷 TRADUCTION FRANÇAISE REQUISE' not in content:
        print(f"⏭️  Skip: {file_path.name} (already translated)")
        return False

    # Extract parts
    if content.startswith('---'):
        parts = content.split('---', 2)
        frontmatter = parts[1]
        body = parts[2]
    else:
        frontmatter = ""
        body = content

    # Remove translation marker
    body = body.replace('<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->\n', '')
    body = re.sub(r'<!-- Ce fichier.*?-->\n', '', body, flags=re.DOTALL)

    # Preserve code blocks
    body, code_blocks = preserve_code_blocks(body)

    # Apply glossary
    body = apply_glossary(body)

    if use_deepl:
        # Use DeepL API (requires setup)
        try:
            import deepl
            translator = deepl.Translator("YOUR_API_KEY")
            result = translator.translate_text(body, target_lang="FR")
            body = result.text
        except ImportError:
            print("⚠️  DeepL not available, using glossary only")

    # Restore code blocks
    body = restore_code_blocks(body, code_blocks)

    # Translate frontmatter
    if frontmatter:
        frontmatter = translate_frontmatter(frontmatter)
        new_content = f"---{frontmatter}---{body}"
    else:
        new_content = body

    # Save
    file_path.write_text(new_content, encoding='utf-8')
    print(f"✅ Translated: {file_path.name}")
    return True

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch translate markdown files")
    parser.add_argument('files', nargs='+', help='Files to translate')
    parser.add_argument('--deepl', action='store_true', help='Use DeepL API')

    args = parser.parse_args()

    for file_path in args.files:
        path = Path(file_path)
        if path.exists():
            process_file(path, use_deepl=args.deepl)

if __name__ == '__main__':
    main()
````

---

## 📋 Quick Translation Workflow

### For Each File:

1. **Open the file** in your editor
2. **Check frontmatter** - ensure title/description are in French
3. **Remove translation marker** (`<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->`)
4. **Translate content**:
   - Use glossary for technical terms
   - Preserve all code blocks
   - Translate comments in code
   - Translate image alt text
5. **Validate**:
   ```bash
   npm run build
   ```
6. **Preview**:
   ```bash
   npm run start -- --locale fr
   ```

---

## 🎯 Today's Goal: Translate Tier 1 (9 files)

Focus on the critical path files. Estimated: 4-6 hours

### Checklist:

- [ ] `installation/quick-start.md`
- [ ] `guides/quick-start.md`
- [ ] `guides/getting-started.md`
- [ ] `guides/cli-commands.md`
- [ ] `architecture.md`
- [ ] `guides/troubleshooting.md`
- [ ] `guides/qgis-troubleshooting.md`
- [ ] `features/multi-architecture.md`

---

## 💡 Pro Tips

1. **Test frequently**: Run `npm run build` after every 2-3 files
2. **Use glossary**: Keep terminology consistent
3. **Preserve structure**: Don't change heading levels, IDs, or slugs
4. **Check links**: Make sure internal links work
5. **Review formatting**: Ensure tables, lists, and callouts render correctly

---

## 🔧 Useful Commands

```bash
# Check translation status
python3 translation_tools/docusaurus_i18n.py status

# Fix links automatically
python3 translation_tools/docusaurus_i18n.py fix-links

# Build and check for errors
npm run build

# Preview French version
npm run start -- --locale fr

# Generate progress report
python3 translation_tools/docusaurus_i18n.py report --output progress.txt
```

---

## 📊 Track Your Progress

Create a simple tracker:

```bash
# Count remaining files
echo "Remaining: $(grep -r '🇫🇷 TRADUCTION FRANÇAISE REQUISE' i18n/fr/docusaurus-plugin-content-docs/current --include='*.md' | wc -l)"

# Count completed today
echo "Completed today: $((53 - $(grep -r '🇫🇷 TRADUCTION FRANÇAISE REQUISE' i18n/fr/docusaurus-plugin-content-docs/current --include='*.md' | wc -l)))"
```

---

Ready to start? Pick the first file and let's translate! 🚀
