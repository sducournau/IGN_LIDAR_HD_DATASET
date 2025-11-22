#!/usr/bin/env python3
"""
Script d'analyse de duplication - IGN LiDAR HD Dataset

Génère un rapport détaillé des fonctions dupliquées dans le codebase.
Usage: python scripts/analyze_duplication.py

Date: 21 Novembre 2025
"""

import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class DuplicationAnalyzer(ast.NodeVisitor):
    """Analyseur de duplication de code basé sur l'AST."""
    
    def __init__(self):
        self.functions: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.classes: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.current_file = ""
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visite une définition de fonction."""
        self.functions[node.name].append((self.current_file, node.lineno))
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visite une définition de classe."""
        self.classes[node.name].append((self.current_file, node.lineno))
        self.generic_visit(node)


def analyze_file(file_path: Path, analyzer: DuplicationAnalyzer) -> None:
    """Analyse un fichier Python."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        analyzer.current_file = str(file_path)
        tree = ast.parse(content, filename=str(file_path))
        analyzer.visit(tree)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"⚠️  Erreur parsing {file_path}: {e}")


def find_duplications(root_dir: Path = Path("ign_lidar")) -> DuplicationAnalyzer:
    """Trouve toutes les duplications dans le répertoire."""
    analyzer = DuplicationAnalyzer()
    
    for py_file in root_dir.rglob("*.py"):
        if "__pycache__" not in str(py_file):
            analyze_file(py_file, analyzer)
    
    return analyzer


def print_report(analyzer: DuplicationAnalyzer, min_duplicates: int = 2) -> None:
    """Affiche le rapport de duplication."""
    print("\n" + "="*80)
    print("📊 RAPPORT DE DUPLICATION - IGN LIDAR HD DATASET")
    print("="*80 + "\n")
    
    # Fonctions dupliquées
    print("🔴 FONCTIONS DUPLIQUÉES\n")
    
    # Priorité: compute_normals et variantes
    normals_funcs = {
        name: locations 
        for name, locations in analyzer.functions.items() 
        if "normals" in name.lower() and len(locations) >= min_duplicates
    }
    
    if normals_funcs:
        print("┌─ Calcul de Normales (CRITIQUE) ─────────────────────────────┐")
        total_normals = sum(len(locs) for locs in normals_funcs.values())
        print(f"│  Total implémentations: {total_normals}")
        print("│")
        
        for func_name, locations in sorted(normals_funcs.items(), 
                                           key=lambda x: len(x[1]), 
                                           reverse=True):
            print(f"│  {func_name}() : {len(locations)} implémentations")
            for file_path, lineno in locations:
                # Raccourcir le chemin
                short_path = str(file_path).replace("ign_lidar/", "")
                print(f"│    ├─ {short_path}:{lineno}")
        
        print("└─────────────────────────────────────────────────────────────┘\n")
    
    # KNN et KDTree
    knn_funcs = {
        name: locations 
        for name, locations in analyzer.functions.items() 
        if any(kw in name.lower() for kw in ["knn", "kdtree", "neighbor"]) 
        and len(locations) >= min_duplicates
    }
    
    if knn_funcs:
        print("┌─ KNN / KDTree ──────────────────────────────────────────────┐")
        total_knn = sum(len(locs) for locs in knn_funcs.values())
        print(f"│  Total implémentations: {total_knn}")
        print("│")
        
        for func_name, locations in sorted(knn_funcs.items(), 
                                           key=lambda x: len(x[1]), 
                                           reverse=True):
            print(f"│  {func_name}() : {len(locations)} implémentations")
            for file_path, lineno in locations[:3]:  # Max 3 pour lisibilité
                short_path = str(file_path).replace("ign_lidar/", "")
                print(f"│    ├─ {short_path}:{lineno}")
            if len(locations) > 3:
                print(f"│    └─ ... et {len(locations)-3} autres")
        
        print("└─────────────────────────────────────────────────────────────┘\n")
    
    # Autres duplications
    other_duplicates = {
        name: locations 
        for name, locations in analyzer.functions.items() 
        if len(locations) >= 3  # 3+ duplications
        and name not in normals_funcs
        and name not in knn_funcs
        and not name.startswith("_")  # Ignorer méthodes privées
    }
    
    if other_duplicates:
        print("┌─ Autres Duplications (3+) ──────────────────────────────────┐")
        print(f"│  Total fonctions: {len(other_duplicates)}")
        print("│")
        
        for func_name, locations in sorted(other_duplicates.items(), 
                                           key=lambda x: len(x[1]), 
                                           reverse=True)[:10]:  # Top 10
            print(f"│  {func_name}() : {len(locations)} implémentations")
        
        print("└─────────────────────────────────────────────────────────────┘\n")
    
    # Classes avec noms similaires
    print("🟡 CLASSES AVEC NOMS SIMILAIRES\n")
    
    processor_classes = {
        name: locations 
        for name, locations in analyzer.classes.items() 
        if any(kw in name for kw in ["Processor", "Computer", "Engine", "Orchestrator", "Manager"])
    }
    
    if processor_classes:
        print("┌─ Processors / Computers / Engines ──────────────────────────┐")
        print(f"│  Total classes: {len(processor_classes)}")
        print("│")
        
        for class_name, locations in sorted(processor_classes.items()):
            if len(locations) > 1:
                print(f"│  ⚠️  {class_name} : {len(locations)} définitions (DUPLIQUÉ!)")
            else:
                print(f"│  {class_name}")
            for file_path, lineno in locations:
                short_path = str(file_path).replace("ign_lidar/", "")
                print(f"│    └─ {short_path}:{lineno}")
        
        print("└─────────────────────────────────────────────────────────────┘\n")
    
    # Statistiques globales
    print("📈 STATISTIQUES GLOBALES\n")
    
    total_functions = len(analyzer.functions)
    duplicated_functions = sum(1 for locs in analyzer.functions.values() if len(locs) >= 2)
    duplicate_instances = sum(len(locs) - 1 for locs in analyzer.functions.values() if len(locs) >= 2)
    
    print(f"Fonctions totales:        {total_functions}")
    print(f"Fonctions dupliquées:     {duplicated_functions} ({100*duplicated_functions/total_functions:.1f}%)")
    print(f"Instances dupliquées:     {duplicate_instances}")
    print(f"Classes totales:          {len(analyzer.classes)}")
    print(f"Classes dupliquées:       {sum(1 for locs in analyzer.classes.values() if len(locs) >= 2)}")
    
    # Estimation lignes dupliquées (très approximatif)
    avg_lines_per_function = 50  # Estimation conservative
    estimated_duplicate_lines = duplicate_instances * avg_lines_per_function
    
    print(f"\n💡 Estimation lignes dupliquées: ~{estimated_duplicate_lines:,}")
    print(f"   (Basé sur {avg_lines_per_function} lignes/fonction en moyenne)")
    
    print("\n" + "="*80)
    print("📝 RECOMMANDATIONS")
    print("="*80 + "\n")
    
    print("1. 🔴 URGENT: Unifier compute_normals() - 18+ implémentations")
    print("   → Créer: ign_lidar/features/compute/normals_api.py")
    print()
    print("2. 🔴 URGENT: Centraliser accès GPU - 25+ fichiers")
    print("   → Utiliser: ign_lidar.core.gpu.GPUManager")
    print()
    print("3. 🟡 Migrer vers KNNEngine partout")
    print("   → Déprécier: build_kdtree(), faiss_knn.py")
    print()
    print("4. 🟡 Nettoyer classes Processor/Computer/Engine")
    print("   → Auditer utilisation, supprimer si redondant")
    print()
    
    print("Voir: docs/audit_reports/CODEBASE_AUDIT_NOV_2025.md")
    print()


def main():
    """Point d'entrée principal."""
    print("🔍 Analyse de duplication en cours...")
    
    root = Path("ign_lidar")
    if not root.exists():
        print(f"❌ Erreur: Répertoire {root} introuvable")
        print("   Exécutez ce script depuis la racine du projet")
        return
    
    analyzer = find_duplications(root)
    print_report(analyzer, min_duplicates=2)


if __name__ == "__main__":
    main()
