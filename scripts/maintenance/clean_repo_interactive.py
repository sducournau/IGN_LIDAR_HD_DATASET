#!/usr/bin/env python3
"""
Guide interactif de nettoyage du repository
Permet de choisir ce qui doit être supprimé
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

# Couleurs
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text: str):
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text:^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def print_section(text: str):
    print(f"\n{BOLD}{CYAN}{text}{RESET}")
    print(f"{CYAN}{'-'*70}{RESET}")

def print_success(text: str):
    print(f"  {GREEN}✓{RESET} {text}")

def print_warning(text: str):
    print(f"  {YELLOW}⚠{RESET}  {text}")

def print_info(text: str):
    print(f"  {BLUE}ℹ{RESET}  {text}")

def ask_yes_no(question: str, default: bool = False) -> bool:
    """Pose une question oui/non."""
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        response = input(f"{CYAN}?{RESET} {question}{suffix}: ").strip().lower()
        if not response:
            return default
        if response in ['y', 'yes', 'oui', 'o']:
            return True
        if response in ['n', 'no', 'non']:
            return False
        print(f"{RED}Réponse invalide. Utilisez y/n{RESET}")

def get_file_size_mb(path: Path) -> float:
    """Retourne la taille d'un fichier en MB."""
    try:
        return path.stat().st_size / 1024 / 1024
    except:
        return 0

def main():
    print_header("🧹 NETTOYAGE INTERACTIF DU REPOSITORY")
    
    print(f"{BOLD}Ce script va vous guider à travers le nettoyage du repository.{RESET}")
    print(f"Vous pourrez choisir ce qui doit être supprimé.\n")
    
    if not ask_yes_no("Continuer?", default=True):
        print(f"\n{YELLOW}Annulé{RESET}")
        return 0
    
    deleted_files = []
    kept_files = []
    total_size_saved = 0
    
    # ========== 1. Scripts d'enrichissement obsolètes ==========
    print_section("📦 1. Scripts d'enrichissement obsolètes")
    
    enrich_scripts = [
        ("enrich_laz_optimized.py", "Version parallèle (3 workers max)"),
        ("enrich_laz_safe.py", "Version séquentielle safe"),
        ("enrich_laz_sequential.py", "Version séquentielle basique"),
        ("enrich_laz_smart.py", "Version smart (expérimentale)"),
        ("enrich_laz_ultra.py", "Version ultra avec KDTree unique"),
        ("enrich_laz_with_features.py", "Version originale CPU"),
    ]
    
    print_info(f"Script consolidé: {GREEN}enrich_laz.py{RESET} (remplace tous)")
    print()
    
    for script, description in enrich_scripts:
        path = Path(script)
        if path.exists():
            size_mb = get_file_size_mb(path)
            print(f"  • {script:40} - {description} ({size_mb:.1f} KB)")
    
    print()
    if ask_yes_no("Supprimer ces 6 scripts?", default=True):
        for script, _ in enrich_scripts:
            path = Path(script)
            if path.exists():
                size_mb = get_file_size_mb(path)
                path.unlink()
                deleted_files.append(script)
                total_size_saved += size_mb
                print_success(f"Supprimé: {script}")
    else:
        print_warning("Scripts conservés")
        kept_files.extend([s for s, _ in enrich_scripts if Path(s).exists()])
    
    # ========== 2. Module features redondant ==========
    print_section("📦 2. Module features redondant")
    
    features_ultra = Path("ign_lidar/features_ultra.py")
    if features_ultra.exists():
        size_mb = get_file_size_mb(features_ultra)
        print_info(f"features_ultra.py fusionné dans features.py ({size_mb:.1f} KB)")
        
        if ask_yes_no("Supprimer features_ultra.py?", default=True):
            features_ultra.unlink()
            deleted_files.append("ign_lidar/features_ultra.py")
            total_size_saved += size_mb
            print_success("Supprimé: ign_lidar/features_ultra.py")
        else:
            print_warning("Conservé")
            kept_files.append("ign_lidar/features_ultra.py")
    else:
        print_info("features_ultra.py déjà supprimé")
    
    # ========== 3. Scripts shell obsolètes ==========
    print_section("📦 3. Scripts shell obsolètes")
    
    shell_scripts = [
        "activate_and_test.sh",
        "benchmark_gpu.sh",
        "build_dataset_quick.sh",
        "install_rapids.sh",
        "monitor_enrichment.sh",
        "monitor_progress.sh",
        "quick_enrich.sh",
        "run_enrichment_optimized.sh",
        "run_enrichment_pipeline.sh",
        "run_enrichment_vectorized.sh",
    ]
    
    existing_shell = [s for s in shell_scripts if Path(s).exists()]
    if existing_shell:
        print_info(f"{len(existing_shell)} scripts shell trouvés:")
        for script in existing_shell:
            print(f"  • {script}")
        
        print()
        if ask_yes_no(f"Supprimer ces {len(existing_shell)} scripts?", default=True):
            for script in existing_shell:
                path = Path(script)
                size_mb = get_file_size_mb(path)
                path.unlink()
                deleted_files.append(script)
                total_size_saved += size_mb
                print_success(f"Supprimé: {script}")
        else:
            print_warning("Scripts conservés")
            kept_files.extend(existing_shell)
    else:
        print_info("Aucun script shell obsolète trouvé")
    
    # ========== 4. Fichiers log ==========
    print_section("📦 4. Fichiers log")
    
    log_files = list(Path(".").glob("*.log"))
    if log_files:
        total_log_size = sum(get_file_size_mb(f) for f in log_files)
        print_info(f"{len(log_files)} fichiers log ({total_log_size:.1f} MB):")
        for log in log_files[:5]:
            size_mb = get_file_size_mb(log)
            print(f"  • {log.name:40} ({size_mb:.1f} MB)")
        if len(log_files) > 5:
            print(f"  ... et {len(log_files)-5} autres")
        
        print()
        if ask_yes_no(f"Supprimer tous les fichiers log?", default=True):
            for log in log_files:
                size_mb = get_file_size_mb(log)
                log.unlink()
                deleted_files.append(str(log))
                total_size_saved += size_mb
            print_success(f"Supprimé: {len(log_files)} fichiers log")
        else:
            print_warning("Logs conservés")
            kept_files.extend([str(f) for f in log_files])
    else:
        print_info("Aucun fichier log trouvé")
    
    # ========== 5. Documentation temporaire ==========
    print_section("📦 5. Documentation temporaire (archivage)")
    
    docs_to_archive = [
        "DÉPANNAGE_ENRICHISSEMENT.md",
        "ENRICHISSEMENT_EN_COURS.md",
        "FICHIERS_CRÉÉS.txt",
        "GPU_STATUS.md",
        "GUIDE_ENRICHISSEMENT_OPTIMISÉ.md",
        "INSTALLATION_COMPLETE.md",
        "RÉSUMÉ_CONFIGURATION.md",
        "TRAITEMENT_EN_COURS.md",
    ]
    
    existing_docs = [d for d in docs_to_archive if Path(d).exists()]
    if existing_docs:
        print_info(f"{len(existing_docs)} documents temporaires:")
        for doc in existing_docs:
            print(f"  • {doc}")
        
        print()
        if ask_yes_no(f"Archiver dans docs/archive/?", default=True):
            archive_dir = Path("docs/archive")
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            for doc in existing_docs:
                src = Path(doc)
                dst = archive_dir / doc
                shutil.move(str(src), str(dst))
                deleted_files.append(f"{doc} → docs/archive/")
                print_success(f"Archivé: {doc}")
        else:
            print_warning("Documents conservés")
            kept_files.extend(existing_docs)
    else:
        print_info("Aucun document temporaire trouvé")
    
    # ========== 6. Tests à réorganiser ==========
    print_section("📦 6. Réorganisation des tests")
    
    test_files = [
        "test_config.py",
        "test_config_gpu.py",
        "test_optimized_features.py",
        "benchmark_optimization.py",
    ]
    
    existing_tests = [t for t in test_files if Path(t).exists()]
    if existing_tests:
        print_info(f"{len(existing_tests)} fichiers de test à déplacer:")
        for test in existing_tests:
            print(f"  • {test} → tests/")
        
        print()
        if ask_yes_no("Déplacer vers tests/?", default=True):
            tests_dir = Path("tests")
            tests_dir.mkdir(exist_ok=True)
            
            for test in existing_tests:
                src = Path(test)
                # Renommer pour convention pytest
                if test == "test_config.py":
                    dst_name = "test_configuration.py"
                elif test == "test_optimized_features.py":
                    dst_name = "test_features.py"
                else:
                    dst_name = test
                
                dst = tests_dir / dst_name
                shutil.move(str(src), str(dst))
                deleted_files.append(f"{test} → tests/{dst_name}")
                print_success(f"Déplacé: {test} → tests/{dst_name}")
        else:
            print_warning("Tests conservés à la racine")
            kept_files.extend(existing_tests)
    else:
        print_info("Aucun test à réorganiser")
    
    # ========== RÉSUMÉ ==========
    print_header("📊 RÉSUMÉ DU NETTOYAGE")
    
    print(f"\n{BOLD}Fichiers supprimés/déplacés:{RESET} {len(deleted_files)}")
    if deleted_files:
        for i, file in enumerate(deleted_files[:10], 1):
            print(f"  {i:2}. {file}")
        if len(deleted_files) > 10:
            print(f"  ... et {len(deleted_files)-10} autres")
    
    print(f"\n{BOLD}Fichiers conservés:{RESET} {len(kept_files)}")
    if kept_files:
        for file in kept_files[:5]:
            print(f"  • {file}")
        if len(kept_files) > 5:
            print(f"  ... et {len(kept_files)-5} autres")
    
    print(f"\n{BOLD}Espace libéré:{RESET} ~{total_size_saved:.1f} MB")
    
    # ========== PROCHAINES ÉTAPES ==========
    print_section("🎯 Prochaines Étapes")
    
    print_info("1. Tester le nouveau workflow:")
    print(f"     {CYAN}python enrich_laz.py --help{RESET}")
    print(f"     {CYAN}python enrich_laz.py --sequential{RESET}")
    
    print()
    print_info("2. Vérifier que tout fonctionne:")
    print(f"     {CYAN}python test_consolidation.py{RESET}")
    
    print()
    print_info("3. Commit les changements:")
    print(f"     {CYAN}git add -A{RESET}")
    print(f"     {CYAN}git commit -m 'chore: clean and consolidate repository'{RESET}")
    
    print()
    print_info("4. Lire la documentation:")
    print(f"     • {CYAN}DONE.md{RESET} - Résumé rapide")
    print(f"     • {CYAN}CLEANUP_SUMMARY.md{RESET} - Résumé détaillé")
    print(f"     • {CYAN}ORGANIZATION.md{RESET} - Guide complet")
    
    print(f"\n{GREEN}{BOLD}✅ Nettoyage terminé avec succès!{RESET}\n")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}⚠️  Annulé par l'utilisateur{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}❌ Erreur: {e}{RESET}")
        sys.exit(1)
