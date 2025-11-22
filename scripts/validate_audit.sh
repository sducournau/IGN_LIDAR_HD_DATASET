#!/bin/bash
# Script de vérification et validation de l'audit
# Usage: bash scripts/validate_audit.sh

echo "════════════════════════════════════════════════════════════════"
echo "🔍 VALIDATION AUDIT - IGN LiDAR HD Dataset"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Compteurs
total_checks=0
passed_checks=0

# Fonction de vérification
check_file() {
    total_checks=$((total_checks + 1))
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $2: OK"
        passed_checks=$((passed_checks + 1))
        return 0
    else
        echo -e "${RED}❌${NC} $2: MANQUANT"
        return 1
    fi
}

check_executable() {
    total_checks=$((total_checks + 1))
    if [ -x "$1" ] || [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $2: OK"
        passed_checks=$((passed_checks + 1))
        return 0
    else
        echo -e "${RED}❌${NC} $2: MANQUANT"
        return 1
    fi
}

echo "📚 Vérification documentation..."
echo "────────────────────────────────────────────────────────────────"
check_file "docs/audit_reports/EXECUTIVE_SUMMARY.md" "Rapport exécutif"
check_file "docs/audit_reports/CODEBASE_AUDIT_NOV_2025.md" "Audit complet"
check_file "docs/audit_reports/QUICK_FIX_GUIDE.md" "Guide rapide"
check_file "docs/audit_reports/SUMMARY_VISUAL.md" "Résumé visuel"
check_file "docs/audit_reports/INDEX.md" "Index"
check_file "docs/audit_reports/DELIVERABLE.md" "Livrable final"
check_file "docs/audit_reports/FILES_CREATED.txt" "Liste fichiers"

echo ""
echo "🛠️  Vérification outils..."
echo "────────────────────────────────────────────────────────────────"
check_executable "scripts/analyze_duplication.py" "Analyse duplication"
check_executable "scripts/audit_class_usage.py" "Audit classes"
check_executable "scripts/migrate_to_gpu_manager.py" "Migration GPU"
check_executable "scripts/benchmark_normals.py" "Benchmark normales"

echo ""
echo "🧪 Tests rapides des outils..."
echo "────────────────────────────────────────────────────────────────"

# Test analyze_duplication.py
total_checks=$((total_checks + 1))
if python scripts/analyze_duplication.py > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} analyze_duplication.py: Exécutable"
    passed_checks=$((passed_checks + 1))
else
    echo -e "${YELLOW}⚠️${NC}  analyze_duplication.py: Erreur d'exécution"
fi

# Test audit_class_usage.py
total_checks=$((total_checks + 1))
if python scripts/audit_class_usage.py > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} audit_class_usage.py: Exécutable"
    passed_checks=$((passed_checks + 1))
else
    echo -e "${YELLOW}⚠️${NC}  audit_class_usage.py: Erreur d'exécution"
fi

echo ""
echo "📊 Statistiques documentation..."
echo "────────────────────────────────────────────────────────────────"

# Compter lignes
if command -v wc &> /dev/null; then
    doc_lines=$(find docs/audit_reports -name "*.md" -exec wc -l {} + | tail -1 | awk '{print $1}')
    echo "Lignes documentation: $doc_lines"
fi

# Compter lignes code
if command -v wc &> /dev/null; then
    script_lines=$(wc -l scripts/analyze_duplication.py scripts/audit_class_usage.py scripts/migrate_to_gpu_manager.py scripts/benchmark_normals.py 2>/dev/null | tail -1 | awk '{print $1}')
    echo "Lignes scripts:       $script_lines"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📈 RÉSULTAT VALIDATION"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Tests réussis: $passed_checks/$total_checks"

percentage=$((passed_checks * 100 / total_checks))
if [ $percentage -eq 100 ]; then
    echo -e "${GREEN}✅ VALIDATION COMPLÈTE (100%)${NC}"
    echo ""
    echo "🎉 Tous les livrables sont présents et fonctionnels!"
    exit_code=0
elif [ $percentage -ge 80 ]; then
    echo -e "${YELLOW}⚠️  VALIDATION PARTIELLE ($percentage%)${NC}"
    echo ""
    echo "Certains éléments sont manquants, mais l'essentiel est là."
    exit_code=1
else
    echo -e "${RED}❌ VALIDATION ÉCHOUÉE ($percentage%)${NC}"
    echo ""
    echo "Trop d'éléments manquants. Vérifiez l'installation."
    exit_code=2
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "💡 PROCHAINES ÉTAPES"
echo "────────────────────────────────────────────────────────────────"
echo ""
echo "1. Lire la documentation:"
echo "   → docs/audit_reports/DELIVERABLE.md (commencer ici)"
echo "   → docs/audit_reports/EXECUTIVE_SUMMARY.md"
echo ""
echo "2. Exécuter les outils:"
echo "   → python scripts/analyze_duplication.py"
echo "   → python scripts/audit_class_usage.py"
echo ""
echo "3. Planifier implémentation:"
echo "   → Créer 4 issues GitHub (Phases 1-4)"
echo "   → Sprint planning Phase 1"
echo ""
echo "════════════════════════════════════════════════════════════════"

exit $exit_code
