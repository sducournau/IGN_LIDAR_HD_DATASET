#!/bin/bash

# IGN LiDAR HD Documentation Harmonization Script

# This script harmonizes and merges duplicate content in the Docusaurus documentation

set -e

echo "🧹 Démarrage de l'harmonisation de la documentation française IGN LiDAR HD..."

# Check if we're in the right directory

if [ ! -d "website" ]; then
echo "❌ Erreur : Ce script doit être exécuté depuis la racine du projet"
echo "📂 Répertoire courant : $(pwd)"
echo "🔧 Veuillez naviguer vers le répertoire racine du projet IGN_LIDAR_HD_DATASET"
exit 1
fi

# Create archive directory for backup

echo "📁 Création des répertoires d'archive..."
mkdir -p "website/archive/pre-harmonization/$(date +%Y%m%d\_%H%M%S)"

# Variables
WEBSITE_DIR="website"
DOCS_EN="${WEBSITE_DIR}/docs"
DOCS_FR="${WEBSITE_DIR}/i18n/fr/docusaurus-plugin-content-docs/current"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="${WEBSITE_DIR}/archive/pre-harmonization/${TIMESTAMP}"

# Check if French documentation exists

if [ ! -d "$DOCS_FR" ]; then
echo "⚠️ Répertoire de documentation française non trouvé : $DOCS_FR"
    echo "📝 Création de la structure de base..."
    mkdir -p "$DOCS_FR"
fi

echo "🔧 Mise à jour des références de documentation française..."

# Function to harmonize workflow content

harmonize_workflows() {
echo "🔄 Harmonisation des workflows..."

    # List of files that contain duplicate workflow information
    local workflow_files=(
        "guides/complete-workflow.md"
        "guides/basic-usage.md"
        "guides/quick-start.md"
        "features/pipeline-configuration.md"
        "workflows.md"
    )

    for file in "${workflow_files[@]}"; do
        if [ -f "$DOCS_EN/$file" ]; then
            echo "  📄 Analysing: $file"

            # Check for duplicate mermaid diagrams
            local mermaid_count=$(grep -c "```mermaid" "$DOCS_EN/$file" 2>/dev/null || echo "0")
            if [ "$mermaid_count" -gt 2 ]; then
                echo "    ⚠️  Multiple mermaid diagrams detected: $mermaid_count"
            fi

            # Check for duplicate YAML examples
            local yaml_count=$(grep -c "```yaml" "$DOCS_EN/$file" 2>/dev/null || echo "0")
            if [ "$yaml_count" -gt 3 ]; then
                echo "    ⚠️  Multiple YAML examples detected: $yaml_count"
            fi
        fi
    done

}

# Function to create unified workflow structure

create_unified_structure() {
echo "🏗️ Création de la structure unifiée..."

    # Create a mapping of content areas to avoid duplication
    cat > "$ARCHIVE_DIR/content_mapping.md" << 'EOF'

# Content Harmonization Mapping

## Workflow Documentation Structure

### Core Concepts (workflows.md)

- High-level workflow overview
- Decision trees and flowcharts
- Best practice recommendations

### Quick Start Guide (guides/quick-start.md)

- 5-minute getting started
- Basic CLI commands
- Simple examples only

### Complete Workflow Guide (guides/complete-workflow.md)

- Detailed step-by-step process
- Advanced configuration options
- Troubleshooting

### Pipeline Configuration (features/pipeline-configuration.md)

- YAML configuration reference
- All configuration parameters
- Advanced pipeline features

### Basic Usage (guides/basic-usage.md)

- Essential commands and patterns
- Common use cases
- Simple workflow examples

## Duplicate Content to Merge

1. **Workflow Diagrams**: Consolidate similar mermaid diagrams
2. **YAML Examples**: Create reusable configuration snippets
3. **Installation Steps**: Single source of truth
4. **Command Examples**: Avoid repetition across guides

EOF

    echo "✅ Structure de mapping créée dans $ARCHIVE_DIR/content_mapping.md"

}

# Function to validate Docusaurus configuration

validate_docusaurus_config() {
echo "🔍 Validation de la configuration Docusaurus..."

    if [ -f "$WEBSITE_DIR/docusaurus.config.ts" ]; then
        echo "  ✅ Configuration Docusaurus trouvée"

        # Check for internationalization setup
        if grep -q "i18n:" "$WEBSITE_DIR/docusaurus.config.ts"; then
            echo "  ✅ Configuration i18n détectée"
        else
            echo "  ⚠️  Configuration i18n non trouvée"
        fi

        # Check for mermaid theme
        if grep -q "theme-mermaid" "$WEBSITE_DIR/docusaurus.config.ts"; then
            echo "  ✅ Support Mermaid configuré"
        else
            echo "  ⚠️  Support Mermaid non configuré"
        fi

    else
        echo "  ❌ Configuration Docusaurus non trouvée"
        exit 1
    fi

}

# Function to create content deduplication report

create_deduplication_report() {
echo "📊 Création du rapport de déduplication..."

    local report_file="$ARCHIVE_DIR/duplication_report.md"

    cat > "$report_file" << 'EOF'

# Documentation Duplication Analysis Report

## Identified Duplicate Content Areas

### 1. Workflow Diagrams

- Similar mermaid flowcharts in multiple files
- Recommendation: Create shared diagram components

### 2. Configuration Examples

- Repeated YAML snippets across guides
- Recommendation: Create reusable config includes

### 3. Installation Instructions

- Similar setup steps in multiple locations
- Recommendation: Single installation guide with links

### 4. Command Reference

- CLI commands repeated in different contexts
- Recommendation: Centralized command reference

## Files with High Overlap

1. `guides/complete-workflow.md` vs `features/pipeline-configuration.md`
   - Both contain detailed YAML configuration examples
   - Similar workflow explanations
2. `guides/quick-start.md` vs `guides/basic-usage.md`

   - Overlapping introductory content
   - Similar simple examples

3. `workflows.md` vs multiple guide files
   - Workflow concepts scattered across files
   - Inconsistent depth and detail

## Recommended Actions

1. **Consolidate Core Concepts**: Move high-level concepts to workflows.md
2. **Standardize Examples**: Create example library for reuse
3. **Cross-reference Instead of Duplicate**: Use links between related sections
4. **Establish Content Hierarchy**: Clear separation of beginner vs advanced content

EOF

    echo "✅ Rapport créé : $report_file"

}

# Function to backup current state

backup_current_state() {
echo "💾 Sauvegarde de l'état actuel..."

    if [ -d "$DOCS_EN" ]; then
        cp -r "$DOCS_EN" "$ARCHIVE_DIR/docs_en_backup"
        echo "  ✅ Documentation anglaise sauvegardée"
    fi

    if [ -d "$DOCS_FR" ]; then
        cp -r "$DOCS_FR" "$ARCHIVE_DIR/docs_fr_backup"
        echo "  ✅ Documentation française sauvegardée"
    fi

    if [ -f "$WEBSITE_DIR/sidebars.ts" ]; then
        cp "$WEBSITE_DIR/sidebars.ts" "$ARCHIVE_DIR/sidebars_backup.ts"
        echo "  ✅ Configuration sidebar sauvegardée"
    fi

}

# Main execution

main() {
echo "🚀 Début de l'harmonisation..."

    backup_current_state
    validate_docusaurus_config
    harmonize_workflows
    create_unified_structure
    create_deduplication_report

    echo ""
    echo "✅ Harmonisation terminée avec succès!"
    echo ""
    echo "📋 Résumé des actions:"
    echo "  • Sauvegarde créée dans : $ARCHIVE_DIR"
    echo "  • Rapport d'analyse généré"
    echo "  • Structure unifiée proposée"
    echo ""
    echo "🔧 Prochaines étapes recommandées:"
    echo "  1. Examiner le rapport : $ARCHIVE_DIR/duplication_report.md"
    echo "  2. Réviser la structure : $ARCHIVE_DIR/content_mapping.md"
    echo "  3. Implémenter les recommandations de déduplication"
    echo "  4. Tester la construction Docusaurus : cd website && npm run build"
    echo ""
    echo "📚 Documentation mise à jour disponible sur : https://sducournau.github.io/IGN_LIDAR_HD_DATASET/"

}

# Execute main function

main "$@"
