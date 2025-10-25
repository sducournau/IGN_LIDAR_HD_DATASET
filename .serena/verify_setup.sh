#!/bin/bash
# Script de vérification de l'installation Serena MCP
# Pour IGN_LIDAR_HD_DATASET

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Vérification Serena MCP pour IGN_LIDAR_HD_DATASET"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Couleurs
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour vérifier un fichier
check_file() {
    if [ -f "$1" ]; then
        echo -e "  ${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "  ${RED}✗${NC} $2 (manquant)"
        return 1
    fi
}

# Fonction pour vérifier un dossier
check_dir() {
    if [ -d "$1" ]; then
        echo -e "  ${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "  ${RED}✗${NC} $2 (manquant)"
        return 1
    fi
}

# 1. Vérifier les fichiers de configuration
echo "1. Configuration VS Code:"
check_file ".vscode/settings.json" "settings.json"
check_file ".vscode/mcp-config.json" "mcp-config.json"
check_file ".vscode/extensions.json" "extensions.json"
echo ""

# 2. Vérifier les fichiers Serena
echo "2. Fichiers de contexte Serena:"
check_file ".serena/project_active.json" "project_active.json"
check_file ".serena/project_overview.md" "project_overview.md"
check_file ".serena/coding_patterns.md" "coding_patterns.md"
check_file ".serena/QUICK_REFERENCE.md" "QUICK_REFERENCE.md"
check_file ".serena/ACTIVATION_STATUS.md" "ACTIVATION_STATUS.md"
echo ""

# 3. Vérifier les instructions Copilot
echo "3. Instructions GitHub Copilot:"
check_file ".github/copilot-instructions.md" "copilot-instructions.md"
echo ""

# 4. Vérifier le token GitHub
echo "4. Token GitHub:"
if [ -n "$GITHUB_TOKEN" ]; then
    echo -e "  ${GREEN}✓${NC} GITHUB_TOKEN défini (${#GITHUB_TOKEN} caractères)"
else
    echo -e "  ${YELLOW}⚠${NC} GITHUB_TOKEN non défini (serveur MCP GitHub non disponible)"
fi
echo ""

# 5. Vérifier Node.js (pour npx)
echo "5. Dépendances système:"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "  ${GREEN}✓${NC} Node.js installé ($NODE_VERSION)"
else
    echo -e "  ${RED}✗${NC} Node.js non installé (requis pour MCP servers)"
fi

if command -v npx &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} npx disponible"
else
    echo -e "  ${RED}✗${NC} npx non disponible"
fi
echo ""

# 6. Vérifier Python
echo "6. Environnement Python:"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "  ${GREEN}✓${NC} Python installé ($PYTHON_VERSION)"
else
    echo -e "  ${RED}✗${NC} Python non installé"
fi

if command -v pytest &> /dev/null; then
    PYTEST_VERSION=$(pytest --version | head -n1)
    echo -e "  ${GREEN}✓${NC} pytest installé ($PYTEST_VERSION)"
else
    echo -e "  ${YELLOW}⚠${NC} pytest non installé"
fi
echo ""

# 7. Lire le statut du projet
echo "7. Statut du projet actif:"
if [ -f ".serena/project_active.json" ]; then
    PROJECT_NAME=$(grep -o '"project_name": "[^"]*"' .serena/project_active.json | cut -d'"' -f4)
    ACTIVATED=$(grep -o '"activated": [^,]*' .serena/project_active.json | awk '{print $2}')
    MCP_ENABLED=$(grep -o '"mcp_enabled": [^,]*' .serena/project_active.json | awk '{print $2}')
    
    echo -e "  Nom du projet: ${GREEN}$PROJECT_NAME${NC}"
    echo -e "  Activé: ${GREEN}$ACTIVATED${NC}"
    echo -e "  MCP activé: ${GREEN}$MCP_ENABLED${NC}"
else
    echo -e "  ${RED}✗${NC} Fichier project_active.json non trouvé"
fi
echo ""

# 8. Résumé
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Résumé"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Compter les éléments configurés
total=0
configured=0

# Config VS Code (3 fichiers)
for file in ".vscode/settings.json" ".vscode/mcp-config.json" ".vscode/extensions.json"; do
    total=$((total + 1))
    [ -f "$file" ] && configured=$((configured + 1))
done

# Fichiers Serena (5 fichiers)
for file in ".serena/project_active.json" ".serena/project_overview.md" ".serena/coding_patterns.md" ".serena/QUICK_REFERENCE.md" ".serena/ACTIVATION_STATUS.md"; do
    total=$((total + 1))
    [ -f "$file" ] && configured=$((configured + 1))
done

# Instructions Copilot (1 fichier)
total=$((total + 1))
[ -f ".github/copilot-instructions.md" ] && configured=$((configured + 1))

percentage=$((configured * 100 / total))

echo ""
echo "Configuration complète à: ${GREEN}$percentage%${NC} ($configured/$total éléments)"

if [ $percentage -eq 100 ]; then
    echo -e "${GREEN}✅ Installation complète!${NC}"
    echo ""
    echo "🚀 Votre workspace est prêt pour:"
    echo "   • GitHub Copilot avec contexte projet"
    echo "   • Serveurs MCP (filesystem, github, memory)"
    echo "   • Suggestions de code intelligentes"
    echo "   • Respect des patterns et standards"
elif [ $percentage -ge 80 ]; then
    echo -e "${YELLOW}⚠ Installation presque complète${NC}"
    echo "Quelques fichiers manquent, mais l'essentiel est configuré."
else
    echo -e "${RED}❌ Installation incomplète${NC}"
    echo "Plusieurs fichiers de configuration manquent."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 Documentation:"
echo "   • Status: .serena/ACTIVATION_STATUS.md"
echo "   • Setup: .serena/MCP_SETUP_COMPLETE.md"
echo "   • Quick Reference: .serena/QUICK_REFERENCE.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
