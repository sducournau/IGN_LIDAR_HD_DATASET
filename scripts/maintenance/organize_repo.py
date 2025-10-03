#!/usr/bin/env python3
"""
Script to organize the repository structure
Moves documentation and scripts to appropriate directories
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict

# Couleurs
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_section(text: str):
    print(f"\n{BOLD}{CYAN}{text}{RESET}")
    print(f"{CYAN}{'-'*70}{RESET}")

def print_success(text: str):
    print(f"  {GREEN}✓{RESET} {text}")

def print_info(text: str):
    print(f"  {BLUE}ℹ{RESET}  {text}")

def main():
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{'📂 ORGANISATION DU REPOSITORY':^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")
    
    root = Path(".")
    
    # ========== 1. Documentation à archiver (obsolète/old) ==========
    print_section("📦 1. Archivage documentation obsolète")
    
    docs_to_archive = [
        "AMÉLIORATION_VITESSE_LAZ.md",
        "DONE_OLD.md",
        "OPTIMISATIONS_CALCUL_LAZ.md",
        "OPTIMISATIONS_LOD2_LOD3.md",
        "OPTIMISATIONS_NORMALES.md",
        "PERFORMANCES_OPTIMISATIONS.md",
        "RESUME_OPTIMISATIONS.md",
        "CLEANUP_SUMMARY.md",
        "ORGANIZATION.md",
        "STATUS.md",
        "STRUCTURE.md",
    ]
    
    archive_dir = Path("docs/archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    for doc in docs_to_archive:
        src = root / doc
        if src.exists():
            dst = archive_dir / doc
            shutil.move(str(src), str(dst))
            print_success(f"{doc} → docs/archive/")
    
    # ========== 2. Documentation de configuration/setup ==========
    print_section("📦 2. Organisation documentation setup")
    
    setup_docs = [
        ("GPU_SETUP.md", "docs/setup/"),
        ("INSTALL_RAPIDS.md", "docs/setup/"),
        ("RAPIDS_OPTIONS.md", "docs/setup/"),
        ("SETUP_VENV.md", "docs/setup/"),
    ]
    
    setup_dir = Path("docs/setup")
    setup_dir.mkdir(parents=True, exist_ok=True)
    
    for doc, dest in setup_docs:
        src = root / doc
        if src.exists():
            dst = Path(dest) / doc
            shutil.move(str(src), str(dst))
            print_success(f"{doc} → {dest}")
    
    # ========== 3. Guides utilisateur ==========
    print_section("📦 3. Organisation guides utilisateur")
    
    user_guides = [
        ("QUICKSTART.md", "docs/guides/"),
        ("QUICKSTART_GPU.md", "docs/guides/"),
        ("QUICKSTART_NEW.md", "docs/guides/"),
        ("GUIDE_DEMARRAGE_RAPIDE.md", "docs/guides/"),
        ("START_HERE.md", "docs/guides/"),
        ("START_URBAN_DOWNLOAD.md", "docs/guides/"),
        ("README_AI_DATASET.md", "docs/guides/"),
        ("README_URBAN_DOWNLOAD.md", "docs/guides/"),
        ("WORKFLOW_LAZ_ENRICHED.md", "docs/guides/"),
    ]
    
    guides_dir = Path("docs/guides")
    guides_dir.mkdir(parents=True, exist_ok=True)
    
    for doc, dest in user_guides:
        src = root / doc
        if src.exists():
            dst = Path(dest) / doc
            shutil.move(str(src), str(dst))
            print_success(f"{doc} → {dest}")
    
    # ========== 4. Scripts shell utilitaires ==========
    print_section("📦 4. Organisation scripts shell")
    
    shell_scripts = [
        ("build_package.sh", "scripts/"),
        ("clean_repo.sh", "scripts/"),
        ("quick_download_urban.sh", "scripts/"),
        ("setup_dev.sh", "scripts/"),
        ("show_info.sh", "scripts/"),
    ]
    
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    for script, dest in shell_scripts:
        src = root / script
        if src.exists():
            dst = Path(dest) / script
            shutil.move(str(src), str(dst))
            # Garder les permissions d'exécution
            dst.chmod(dst.stat().st_mode | 0o111)
            print_success(f"{script} → {dest}")
    
    # ========== 5. Scripts Python utilitaires ==========
    print_section("📦 5. Organisation scripts Python")
    
    python_scripts = [
        ("clean_repo_interactive.py", "scripts/"),
        ("SYNTHESE.py", "scripts/"),
        ("test_consolidation.py", "tests/"),
    ]
    
    for script, dest in python_scripts:
        src = root / script
        if src.exists():
            dst_dir = Path(dest)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / script
            shutil.move(str(src), str(dst))
            print_success(f"{script} → {dest}")
    
    # ========== 6. Créer un README consolidé dans docs/ ==========
    print_section("📦 6. Création index documentation")
    
    docs_index = Path("docs/README.md")
    docs_index.write_text("""# Documentation

## 📚 Structure de la Documentation

### Guides Utilisateur (`guides/`)
- **START_HERE.md** - Point de départ pour les nouveaux utilisateurs
- **QUICKSTART.md** - Démarrage rapide (CPU)
- **QUICKSTART_GPU.md** - Démarrage rapide (GPU)
- **GUIDE_DEMARRAGE_RAPIDE.md** - Guide francophone
- **README_AI_DATASET.md** - Création de datasets IA
- **README_URBAN_DOWNLOAD.md** - Téléchargement zones urbaines
- **WORKFLOW_LAZ_ENRICHED.md** - Workflow d'enrichissement LAZ

### Configuration & Installation (`setup/`)
- **GPU_SETUP.md** - Configuration GPU
- **INSTALL_RAPIDS.md** - Installation RAPIDS
- **RAPIDS_OPTIONS.md** - Options RAPIDS
- **SETUP_VENV.md** - Configuration environnement virtuel

### Archives (`archive/`)
Documentation historique et notes de développement

## 🚀 Démarrage Rapide

1. **Nouveaux utilisateurs**: Commencez par `guides/START_HERE.md`
2. **Installation GPU**: Consultez `setup/GPU_SETUP.md`
3. **Exemples**: Voir le dossier `/examples`

## 📖 Autres Ressources

- **README.md** (racine) - Vue d'ensemble du projet
- **README_FR.md** (racine) - Version française
- **CHANGELOG.md** (racine) - Historique des versions
""")
    print_success("Créé: docs/README.md")
    
    # ========== 7. Créer README pour scripts/ ==========
    scripts_readme = Path("scripts/README.md")
    scripts_readme.write_text("""# Scripts Utilitaires

## Scripts Shell

- **build_package.sh** - Construit le package Python
- **clean_repo.sh** - Nettoyage automatique du repository
- **quick_download_urban.sh** - Téléchargement rapide zones urbaines
- **setup_dev.sh** - Configuration environnement de développement
- **show_info.sh** - Affiche informations système

## Scripts Python

- **clean_repo_interactive.py** - Nettoyage interactif du repository
- **SYNTHESE.py** - Génération de synthèses

## Utilisation

```bash
# Rendre un script exécutable
chmod +x scripts/nom_du_script.sh

# Exécuter
./scripts/nom_du_script.sh
```

Pour les scripts Python:
```bash
python scripts/nom_du_script.py
```
""")
    print_success("Créé: scripts/README.md")
    
    # ========== RÉSUMÉ ==========
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{'✅ ORGANISATION TERMINÉE':^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")
    
    print_info("Structure finale:")
    print(f"""
    {CYAN}Racine/{RESET}
      {GREEN}├── README.md, README_FR.md{RESET} (documentation principale)
      {GREEN}├── CHANGELOG.md, DONE.md{RESET} (historique)
      {GREEN}├── pyproject.toml, requirements.txt{RESET} (configuration)
      {GREEN}├── *.py{RESET} (scripts principaux)
      {GREEN}├── docs/{RESET}
      {GREEN}│   ├── README.md{RESET} (index)
      {GREEN}│   ├── guides/{RESET} (guides utilisateur)
      {GREEN}│   ├── setup/{RESET} (installation)
      {GREEN}│   └── archive/{RESET} (docs obsolètes)
      {GREEN}├── scripts/{RESET} (utilitaires)
      {GREEN}├── examples/{RESET} (exemples)
      {GREEN}├── tests/{RESET} (tests)
      {GREEN}└── ign_lidar/{RESET} (code source)
    """)
    
    print(f"\n{GREEN}{BOLD}✨ Repository organisé avec succès!{RESET}\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'\033[91m'}❌ Erreur: {e}{RESET}")
        raise
