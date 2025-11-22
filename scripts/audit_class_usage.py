#!/usr/bin/env python3
"""
Script d'audit d'utilisation des classes

Vérifie quelles classes Processor/Manager/Engine sont réellement utilisées.
Usage: python scripts/audit_class_usage.py

Date: 21 Novembre 2025
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class ClassUsageAnalyzer(ast.NodeVisitor):
    """Analyseur d'utilisation de classes."""
    
    def __init__(self):
        self.class_definitions: Dict[str, List[str]] = defaultdict(list)
        self.class_imports: Dict[str, List[str]] = defaultdict(list)
        self.class_instantiations: Dict[str, List[str]] = defaultdict(list)
        self.current_file = ""
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visite une définition de classe."""
        self.class_definitions[node.name].append(self.current_file)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visite un import from."""
        if node.module:
            for alias in node.names:
                self.class_imports[alias.name].append(
                    f"{self.current_file}:{node.lineno}"
                )
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Visite un appel de fonction/classe."""
        if isinstance(node.func, ast.Name):
            # Instanciation directe: ClassName()
            self.class_instantiations[node.func.id].append(
                f"{self.current_file}:{node.lineno}"
            )
        self.generic_visit(node)


def analyze_codebase(root_dir: Path = Path("ign_lidar")) -> ClassUsageAnalyzer:
    """Analyse tout le codebase."""
    analyzer = ClassUsageAnalyzer()
    
    for py_file in root_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analyzer.current_file = str(py_file)
            tree = ast.parse(content, filename=str(py_file))
            analyzer.visit(tree)
        except:
            continue
    
    return analyzer


def find_suspect_classes() -> List[str]:
    """Retourne la liste des classes suspectes à auditer."""
    return [
        'ProcessorCore',
        'OptimizedProcessor',
        'GeometricFeatureProcessor',
        'AsyncGPUProcessor',
        'StreamingTileProcessor',
        'GPUMemoryManager',
        'CUDAStreamManager',
    ]


def is_used(class_name: str, analyzer: ClassUsageAnalyzer) -> Tuple[bool, Dict[str, int]]:
    """
    Vérifie si une classe est utilisée.
    
    Returns:
        (is_used, stats_dict)
    """
    stats = {
        'definitions': len(analyzer.class_definitions.get(class_name, [])),
        'imports': len(analyzer.class_imports.get(class_name, [])),
        'instantiations': len(analyzer.class_instantiations.get(class_name, [])),
    }
    
    # Une classe est "utilisée" si elle est importée OU instanciée
    # (hors fichier de définition)
    definition_files = set(analyzer.class_definitions.get(class_name, []))
    import_files = set(
        loc.split(':')[0] 
        for loc in analyzer.class_imports.get(class_name, [])
    )
    instantiation_files = set(
        loc.split(':')[0] 
        for loc in analyzer.class_instantiations.get(class_name, [])
    )
    
    # Retirer les fichiers de définition
    usage_files = (import_files | instantiation_files) - definition_files
    
    is_used = len(usage_files) > 0
    stats['usage_files'] = len(usage_files)
    
    return is_used, stats


def print_class_report(class_name: str, analyzer: ClassUsageAnalyzer):
    """Affiche un rapport détaillé pour une classe."""
    is_used_flag, stats = is_used(class_name, analyzer)
    
    # Header
    status = "✅ UTILISÉE" if is_used_flag else "⚠️  NON UTILISÉE"
    print(f"\n{'='*60}")
    print(f"📦 {class_name} - {status}")
    print('='*60)
    
    # Définitions
    definitions = analyzer.class_definitions.get(class_name, [])
    if definitions:
        print(f"\n📝 Définition ({len(definitions)}):")
        for def_file in definitions:
            print(f"  └─ {def_file}")
    else:
        print("\n⚠️  Aucune définition trouvée")
        return
    
    # Imports
    imports = analyzer.class_imports.get(class_name, [])
    if imports:
        print(f"\n📥 Imports ({len(imports)}):")
        for imp in imports[:5]:  # Max 5
            print(f"  └─ {imp}")
        if len(imports) > 5:
            print(f"  └─ ... et {len(imports)-5} autres")
    else:
        print("\n📥 Imports: aucun")
    
    # Instantiations
    instantiations = analyzer.class_instantiations.get(class_name, [])
    if instantiations:
        print(f"\n🏗️  Instantiations ({len(instantiations)}):")
        for inst in instantiations[:5]:  # Max 5
            print(f"  └─ {inst}")
        if len(instantiations) > 5:
            print(f"  └─ ... et {len(instantiations)-5} autres")
    else:
        print("\n🏗️  Instantiations: aucune")
    
    # Recommandation
    print("\n💡 Recommandation:")
    if not is_used_flag:
        print("  🔴 CANDIDAT À LA SUPPRESSION")
        print("  → Ajouter @deprecated puis supprimer dans v3.2.0")
    elif stats['imports'] == 0 and stats['instantiations'] == 0:
        print("  🟡 UTILISATION DOUTEUSE")
        print("  → Vérifier manuellement l'usage réel")
    elif stats['usage_files'] < 3:
        print("  🟡 UTILISATION LIMITÉE")
        print(f"  → Utilisé dans seulement {stats['usage_files']} fichier(s)")
        print("  → Envisager refactoring ou fusion")
    else:
        print("  ✅ Classe bien utilisée")
        print(f"  → {stats['usage_files']} fichiers utilisateurs")


def generate_deprecation_code(class_name: str, replacement: str = None):
    """Génère le code de dépréciation."""
    print(f"\n```python")
    print("import warnings")
    print()
    print("@deprecated(")
    print('    version="3.1.0",')
    if replacement:
        print(f'    reason="Use {replacement} instead"')
    else:
        print('    reason="No longer used"')
    print(")")
    print(f"class {class_name}:")
    print('    """')
    print(f'    Deprecated: This class is no longer maintained.')
    if replacement:
        print(f'    Use {replacement} instead.')
    print('    Will be removed in v3.2.0')
    print('    """')
    print("    def __init__(self, *args, **kwargs):")
    print("        warnings.warn(")
    print(f'            "{class_name} is deprecated. "')
    if replacement:
        print(f'            "Use {replacement} instead. "')
    print('            "Will be removed in v3.2.0",')
    print("            DeprecationWarning,")
    print("            stacklevel=2")
    print("        )")
    print("```")


def main():
    """Point d'entrée principal."""
    print("🔍 Audit d'utilisation des classes")
    print("="*60)
    
    print("\n🔍 Analyse du codebase...")
    analyzer = analyze_codebase()
    
    print(f"✅ Analyse terminée:")
    print(f"  - {len(analyzer.class_definitions)} classes définies")
    print(f"  - {sum(len(v) for v in analyzer.class_imports.values())} imports")
    print(f"  - {sum(len(v) for v in analyzer.class_instantiations.values())} instantiations")
    
    # Analyser les classes suspectes
    suspect_classes = find_suspect_classes()
    
    print(f"\n📊 Audit des {len(suspect_classes)} classes suspectes:")
    print("─"*60)
    
    unused_classes = []
    limited_use_classes = []
    used_classes = []
    
    for class_name in suspect_classes:
        is_used_flag, stats = is_used(class_name, analyzer)
        
        if not is_used_flag:
            unused_classes.append(class_name)
        elif stats['usage_files'] < 3:
            limited_use_classes.append((class_name, stats['usage_files']))
        else:
            used_classes.append((class_name, stats['usage_files']))
    
    # Résumé
    print(f"\n{'='*60}")
    print("📈 RÉSUMÉ")
    print('='*60)
    
    if unused_classes:
        print(f"\n🔴 Classes NON utilisées ({len(unused_classes)}):")
        for cls in unused_classes:
            print(f"  ❌ {cls}")
    
    if limited_use_classes:
        print(f"\n🟡 Classes à utilisation LIMITÉE ({len(limited_use_classes)}):")
        for cls, count in limited_use_classes:
            print(f"  ⚠️  {cls} ({count} fichier(s))")
    
    if used_classes:
        print(f"\n✅ Classes bien utilisées ({len(used_classes)}):")
        for cls, count in used_classes:
            print(f"  ✓ {cls} ({count} fichier(s))")
    
    # Rapports détaillés
    print(f"\n{'='*60}")
    print("📋 RAPPORTS DÉTAILLÉS")
    print('='*60)
    
    for class_name in suspect_classes:
        print_class_report(class_name, analyzer)
    
    # Recommandations actions
    print(f"\n{'='*60}")
    print("🎯 ACTIONS RECOMMANDÉES")
    print('='*60)
    
    if unused_classes:
        print("\n🔴 Classes à déprécier immédiatement:")
        for cls in unused_classes:
            print(f"\n  {cls}:")
            print(f"    1. Ajouter @deprecated dans le fichier de définition")
            print(f"    2. Mettre à jour CHANGELOG.md")
            print(f"    3. Supprimer dans v3.2.0")
            
            # Chercher remplacement potentiel
            replacements = {
                'GPUMemoryManager': 'GPUManager',
                'ProcessorCore': 'LiDARProcessor',
                'OptimizedProcessor': 'TileProcessor',
            }
            
            if cls in replacements:
                print(f"\n  Code de dépréciation:")
                generate_deprecation_code(cls, replacements[cls])
    
    if limited_use_classes:
        print("\n🟡 Classes à auditer manuellement:")
        for cls, count in limited_use_classes:
            print(f"\n  {cls} ({count} usage(s)):")
            print(f"    → Vérifier si utilisation essentielle")
            print(f"    → Envisager fusion avec classe similaire")
            print(f"    → Documenter responsabilité claire")
    
    print("\n" + "="*60)
    print("✅ Audit terminé")
    print("\nVoir: docs/audit_reports/CODEBASE_AUDIT_NOV_2025.md")


if __name__ == "__main__":
    main()
