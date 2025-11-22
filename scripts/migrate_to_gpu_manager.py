#!/usr/bin/env python3
"""
Script de migration vers GPUManager centralisé

Ce script aide à migrer les accès GPU dispersés vers l'API GPUManager unifiée.
Usage: python scripts/migrate_to_gpu_manager.py [--dry-run] [--file FILE]

Date: 21 Novembre 2025
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple


# Patterns à remplacer
PATTERNS = [
    # Pattern 1: cp.cuda.Device().mem_info
    (
        re.compile(r'(\s*)free_mem,\s*total_mem\s*=\s*cp\.cuda\.Device\(\)\.mem_info'),
        r'\1from ign_lidar.core.gpu import GPUManager\n\1free_gb, total_gb = GPUManager.get_memory_info()'
    ),
    
    # Pattern 2: cp.cuda.runtime.memGetInfo()
    (
        re.compile(r'(\s*)free_mem,\s*total_mem\s*=\s*cp\.cuda\.runtime\.memGetInfo\(\)'),
        r'\1from ign_lidar.core.gpu import GPUManager\n\1free_gb, total_gb = GPUManager.get_memory_info()'
    ),
    
    # Pattern 3: device.mem_info
    (
        re.compile(r'(\s*)device\s*=\s*cp\.cuda\.Device\(\)\s*\n\s*free_mem,\s*total_mem\s*=\s*device\.mem_info'),
        r'\1from ign_lidar.core.gpu import GPUManager\n\1free_gb, total_gb = GPUManager.get_memory_info()'
    ),
    
    # Pattern 4: cp.cuda.Stream.null.synchronize()
    (
        re.compile(r'(\s*)cp\.cuda\.Stream\.null\.synchronize\(\)'),
        r'\1from ign_lidar.core.gpu import GPUManager\n\1GPUManager.synchronize()'
    ),
]


def analyze_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Analyse un fichier et trouve les patterns à remplacer.
    
    Returns:
        Liste de (line_number, old_pattern, new_pattern)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️  Erreur lecture {file_path}: {e}")
        return []
    
    matches = []
    
    for pattern, replacement in PATTERNS:
        for match in pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            matches.append((line_num, match.group(0), replacement))
    
    return matches


def migrate_file(file_path: Path, dry_run: bool = True) -> bool:
    """
    Migre un fichier vers GPUManager.
    
    Returns:
        True si modifications effectuées
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Erreur lecture {file_path}: {e}")
        return False
    
    original_content = content
    modified = False
    
    # Vérifier si GPUManager déjà importé
    has_gpu_manager_import = 'from ign_lidar.core.gpu import GPUManager' in content
    
    # Appliquer tous les patterns
    for pattern, replacement in PATTERNS:
        if pattern.search(content):
            content = pattern.sub(replacement, content)
            modified = True
    
    if not modified:
        return False
    
    # Ajouter import si nécessaire
    if not has_gpu_manager_import and modified:
        # Trouver position d'import (après les autres imports)
        import_section = re.search(r'(import\s+\w+.*?\n)+', content)
        if import_section:
            insert_pos = import_section.end()
            content = (
                content[:insert_pos] + 
                'from ign_lidar.core.gpu import GPUManager\n' +
                content[insert_pos:]
            )
    
    if dry_run:
        print(f"\n📝 Modifications pour {file_path}:")
        print("─" * 60)
        
        # Afficher diff simple
        original_lines = original_content.split('\n')
        new_lines = content.split('\n')
        
        for i, (old, new) in enumerate(zip(original_lines, new_lines)):
            if old != new:
                print(f"  Ligne {i+1}:")
                print(f"  - {old}")
                print(f"  + {new}")
        
        return True
    else:
        # Écrire le fichier modifié
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Modifié: {file_path}")
            return True
        except Exception as e:
            print(f"❌ Erreur écriture {file_path}: {e}")
            return False


def find_gpu_access_files(root_dir: Path = Path("ign_lidar")) -> List[Path]:
    """Trouve tous les fichiers avec accès GPU direct."""
    files_with_gpu = []
    
    patterns_to_search = [
        r'cp\.cuda\.Device\(\)',
        r'cp\.cuda\.runtime\.memGetInfo',
        r'device\.mem_info',
    ]
    
    for py_file in root_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in patterns_to_search:
                if re.search(pattern, content):
                    files_with_gpu.append(py_file)
                    break
        except:
            continue
    
    return files_with_gpu


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Migre les accès GPU vers GPUManager centralisé"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Affiche les modifications sans les appliquer"
    )
    parser.add_argument(
        '--file',
        type=str,
        help="Migrer un fichier spécifique"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help="Migrer tous les fichiers trouvés"
    )
    
    args = parser.parse_args()
    
    print("🔍 Migration vers GPUManager centralisé")
    print("=" * 60)
    
    if args.file:
        # Migrer un fichier spécifique
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Fichier introuvable: {file_path}")
            return
        
        matches = analyze_file(file_path)
        if matches:
            print(f"\n📊 {len(matches)} patterns trouvés dans {file_path}")
            for line_num, old, new in matches:
                print(f"  Ligne {line_num}: {old.strip()}")
            
            if args.dry_run:
                print("\n🔍 Mode DRY-RUN - Aucune modification")
                migrate_file(file_path, dry_run=True)
            else:
                migrate_file(file_path, dry_run=False)
        else:
            print(f"✅ Aucun pattern trouvé dans {file_path}")
    
    elif args.all:
        # Trouver et migrer tous les fichiers
        print("\n🔍 Recherche fichiers avec accès GPU direct...")
        files = find_gpu_access_files()
        
        print(f"\n📊 {len(files)} fichiers trouvés:")
        for f in files:
            print(f"  - {f}")
        
        if args.dry_run:
            print("\n🔍 Mode DRY-RUN - Analyse des modifications...")
            modified_count = 0
            for file_path in files:
                if migrate_file(file_path, dry_run=True):
                    modified_count += 1
            
            print("\n" + "=" * 60)
            print(f"📊 Résumé: {modified_count}/{len(files)} fichiers à modifier")
            print("\n💡 Exécutez sans --dry-run pour appliquer les modifications")
        else:
            print("\n⚠️  ATTENTION: Modifications réelles!")
            response = input("Continuer? (y/N): ")
            if response.lower() != 'y':
                print("❌ Annulé")
                return
            
            modified_count = 0
            for file_path in files:
                if migrate_file(file_path, dry_run=False):
                    modified_count += 1
            
            print("\n" + "=" * 60)
            print(f"✅ Migration terminée: {modified_count}/{len(files)} fichiers modifiés")
            print("\n💡 N'oubliez pas de:")
            print("  1. Tester les modifications")
            print("  2. Committer les changements")
            print("  3. Mettre à jour les tests")
    
    else:
        # Mode analyse seulement
        print("\n🔍 Recherche fichiers avec accès GPU direct...")
        files = find_gpu_access_files()
        
        print(f"\n📊 {len(files)} fichiers trouvés avec accès GPU direct:")
        for f in files:
            matches = analyze_file(f)
            if matches:
                print(f"\n  📄 {f} ({len(matches)} patterns)")
                for line_num, old, _ in matches[:3]:  # Max 3 exemples
                    print(f"    Ligne {line_num}: {old.strip()[:60]}...")
                if len(matches) > 3:
                    print(f"    ... et {len(matches)-3} autres")
        
        print("\n" + "=" * 60)
        print("💡 Usage:")
        print("  --dry-run --all     : Voir toutes les modifications")
        print("  --all               : Appliquer toutes les modifications")
        print("  --file FICHIER      : Migrer un fichier spécifique")


if __name__ == "__main__":
    main()
