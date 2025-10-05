#!/usr/bin/env python3
"""Script to identify missing French translation files and their characteristics."""

import os
from pathlib import Path

DOCS_DIR = Path("docs")
FR_DIR = Path("i18n/fr/docusaurus-plugin-content-docs/current")

missing_files = [
    "api/cli.md",
    "api/configuration.md",
    "api/gpu-api.md",
    "features/axonometry.md",
    "guides/getting-started.md",
    "installation/gpu-setup.md",
    "reference/architectural-styles.md",
    "tutorials/custom-features.md"
]

print("=== Missing French Translation Files ===\n")
for file_path in missing_files:
    en_path = DOCS_DIR / file_path
    fr_path = FR_DIR / file_path
    
    if en_path.exists():
        # Count lines in English version
        with open(en_path, 'r', encoding='utf-8') as f:
            lines = len(f.readlines())
        
        # Get first few lines for metadata
        with open(en_path, 'r', encoding='utf-8') as f:
            first_lines = ''.join([f.readline() for _ in range(10)])
        
        print(f"File: {file_path}")
        print(f"  Lines: {lines}")
        print(f"  French target: {fr_path}")
        print(f"  Directory exists: {fr_path.parent.exists()}")
        print()

print("\n=== Files to Create ===")
for file_path in missing_files:
    fr_path = FR_DIR / file_path
    if not fr_path.parent.exists():
        print(f"  mkdir -p {fr_path.parent}")
    print(f"  touch {fr_path}")
