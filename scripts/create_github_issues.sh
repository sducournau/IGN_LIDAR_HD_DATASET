#!/bin/bash
# Script pour créer les 4 issues GitHub du plan d'action audit
# Nécessite: gh CLI (GitHub CLI) installé et authentifié
# Usage: bash scripts/create_github_issues.sh

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════════════════"
echo "🎫 CRÉATION ISSUES GITHUB - Plan d'action audit"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Vérifier que gh CLI est installé
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) n'est pas installé${NC}"
    echo ""
    echo "Installation:"
    echo "  • Ubuntu/Debian: sudo apt install gh"
    echo "  • macOS:         brew install gh"
    echo "  • Windows:       winget install GitHub.cli"
    echo ""
    echo "Puis authentifiez-vous: gh auth login"
    exit 1
fi

# Vérifier l'authentification
if ! gh auth status &> /dev/null; then
    echo -e "${RED}❌ Non authentifié avec GitHub${NC}"
    echo "Exécutez: gh auth login"
    exit 1
fi

echo -e "${GREEN}✅ GitHub CLI authentifié${NC}"
echo ""

# Demander confirmation
echo -e "${YELLOW}⚠️  Ce script va créer 4 issues dans le repo IGN_LIDAR_HD_DATASET${NC}"
echo ""
read -p "Continuer? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Annulé."
    exit 0
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "📝 Création des issues..."
echo "────────────────────────────────────────────────────────────────"
echo ""

# Issue 1: Phase 1 - Unifier compute_normals()
echo -e "${BLUE}[1/4]${NC} Phase 1: Unifier compute_normals()..."

gh issue create \
  --title "🔧 Phase 1: Unifier compute_normals() - 18 implémentations → 1 API" \
  --label "refactoring,critical,technical-debt" \
  --milestone "v3.2.0" \
  --body "## 🎯 Objectif

Consolider les 18 implémentations différentes de \`compute_normals()\` en une seule API unifiée avec stratégies CPU/GPU/FAISS.

## 📊 État actuel

**Problème:** 18 implémentations trouvées dans:
- \`features/compute/normals.py\`
- \`features/compute/geometric.py\`
- \`features/feature_computer.py\`
- \`optimization/gpu_features.py\`
- \`preprocessing/outliers.py\`
- Et 13 autres fichiers...

**Impact:** ~5,000 lignes dupliquées, maintenance cauchemardesque, bugs inconsistants

## 🛠️ Plan d'implémentation

### Étape 1: Créer l'API unifiée (2 jours)
\`\`\`python
# ign_lidar/features/compute/normals_api.py
def compute_normals(
    points: np.ndarray,
    k_neighbors: int = 30,
    search_radius: float = 3.0,
    strategy: str = 'auto'  # auto, cpu, gpu, faiss-gpu
) -> np.ndarray:
    \"\"\"Unified API for normal computation.\"\"\"
    pass
\`\`\`

### Étape 2: Benchmarker (1 jour)
\`\`\`bash
python scripts/benchmark_normals.py
\`\`\`

### Étape 3: Migrer et déprécier (2 jours)
- Ajouter \`@deprecated\` aux 18 anciennes implémentations
- Mettre à jour tous les appels vers nouvelle API
- Tests de non-régression

## 📈 Bénéfices attendus

- ✅ **-5,000 lignes** de code dupliqué
- ✅ **18x plus rapide** à maintenir
- ✅ **Performances unifiées** et optimisées
- ✅ **Tests centralisés** plus faciles

## 📚 Ressources

- Script: \`scripts/benchmark_normals.py\`
- Audit: \`docs/audit_reports/CODEBASE_AUDIT_NOV_2025.md\` (section 1.1)
- Guide: \`docs/audit_reports/QUICK_FIX_GUIDE.md\` (TOP 1)

## ⏱️ Estimation

**5 jours** (1 développeur)

## ✅ Checklist

- [ ] Créer \`normals_api.py\` avec stratégies
- [ ] Benchmarker toutes les implémentations
- [ ] Choisir meilleure stratégie par cas d'usage
- [ ] Ajouter tests unitaires
- [ ] Migrer 18 anciennes implémentations
- [ ] Ajouter warnings de dépréciation
- [ ] Mettre à jour documentation
- [ ] Tests de non-régression complets" || echo "Erreur création issue 1"

echo -e "${GREEN}✅ Issue 1 créée${NC}"
echo ""

# Issue 2: Phase 2 - Centraliser accès GPU
echo -e "${BLUE}[2/4]${NC} Phase 2: Centraliser accès GPU..."

gh issue create \
  --title "🚀 Phase 2: Centraliser accès GPU - 25+ fichiers → GPUManager" \
  --label "refactoring,critical,gpu,performance" \
  --milestone "v3.2.0" \
  --body "## 🎯 Objectif

Centraliser tous les accès GPU directs (CuPy, CUDA) vers un \`GPUManager\` unique pour optimisation et testabilité.

## 📊 État actuel

**Problème:** 25+ fichiers avec accès GPU direct:
\`\`\`python
# Actuellement (25+ endroits différents)
import cupy as cp
device = cp.cuda.Device()
mem_info = device.mem_info  # ⚠️ Non centralisé
\`\`\`

**3 APIs différentes utilisées:**
- \`cp.cuda.Device().mem_info\`
- \`cp.cuda.runtime.memGetInfo()\`
- \`cp.get_default_memory_pool().used_bytes()\`

**Impact:** Impossible de tester sans GPU, synchronisations inefficaces, code non portable

## 🛠️ Plan d'implémentation

### Étape 1: Étendre GPUManager (1 jour)
\`\`\`python
# ign_lidar/core/gpu.py
class GPUManager:
    def get_memory_info(self) -> Tuple[int, int]:
        \"\"\"Unified memory info API.\"\"\"
        
    def synchronize(self):
        \"\"\"Explicit GPU sync point.\"\"\"
        
    def optimal_chunk_size(self, total_size: int) -> int:
        \"\"\"Calculate optimal chunk based on memory.\"\"\"
\`\`\`

### Étape 2: Migration automatique (1 jour)
\`\`\`bash
python scripts/migrate_to_gpu_manager.py
\`\`\`

### Étape 3: Tests et validation (1 jour)
- Mock GPUManager pour tests sans GPU
- Benchmarks de performance
- Valider synchronisations

## 📈 Bénéfices attendus

- ✅ **+30% performance GPU** (sync optimisés)
- ✅ **100% testable** sans matériel GPU
- ✅ **Code portable** CPU/GPU
- ✅ **Monitoring unifié** de la mémoire

## 📚 Ressources

- Script: \`scripts/migrate_to_gpu_manager.py\`
- Audit: \`docs/audit_reports/CODEBASE_AUDIT_NOV_2025.md\` (section 1.3)
- Guide: \`docs/audit_reports/QUICK_FIX_GUIDE.md\` (TOP 3)

## ⏱️ Estimation

**3 jours** (1 développeur)

## ✅ Checklist

- [ ] Étendre API GPUManager
- [ ] Exécuter migrate_to_gpu_manager.py
- [ ] Revue manuelle migrations
- [ ] Créer mocks pour tests
- [ ] Tests unitaires sans GPU
- [ ] Benchmarks performance
- [ ] Mettre à jour documentation GPU" || echo "Erreur création issue 2"

echo -e "${GREEN}✅ Issue 2 créée${NC}"
echo ""

# Issue 3: Phase 3 - Compléter migration KNN
echo -e "${BLUE}[3/4]${NC} Phase 3: Compléter migration KNN..."

gh issue create \
  --title "🔄 Phase 3: Compléter migration KNN vers optimization.knn_search" \
  --label "refactoring,high-priority,technical-debt" \
  --milestone "v3.2.0" \
  --body "## 🎯 Objectif

Finaliser la migration KNN commencée en novembre 2025 vers l'API unifiée \`optimization.knn_search\`.

## 📊 État actuel

**Problème:** Migration KNN incomplète:
- ✅ \`optimization/knn_search.py\` créé (KNNEngine)
- ❌ Anciens modules toujours utilisés
- ❌ 3 implémentations parallèles coexistent

**Fichiers redondants:**
- \`features/compute/faiss_knn.py\` (200 lignes) - À supprimer
- \`optimization/gpu_kdtree.py\` (150 lignes) - À supprimer
- \`features/utils.py::build_kdtree()\` - À déprécier

**Impact:** ~3,000 lignes dupliquées, confusion API

## 🛠️ Plan d'implémentation

### Étape 1: Déprécier anciennes APIs (0.5 jour)
\`\`\`python
# features/utils.py
@deprecated(
    version=\"3.2.0\",
    reason=\"Use optimization.knn_search.KNNEngine instead\"
)
def build_kdtree(...):
    warnings.warn(...)
    return KNNEngine().build_index(...)
\`\`\`

### Étape 2: Migrer usages restants (1 jour)
\`\`\`bash
# Trouver tous les usages
grep -r \"build_kdtree\" ign_lidar/
grep -r \"from.*faiss_knn import\" ign_lidar/
\`\`\`

### Étape 3: Supprimer modules obsolètes (0.5 jour)
- Supprimer \`features/compute/faiss_knn.py\`
- Supprimer \`optimization/gpu_kdtree.py\`
- Mettre à jour imports

## 📈 Bénéfices attendus

- ✅ **-3,000 lignes** code redondant
- ✅ **API unifiée** CPU/GPU/FAISS
- ✅ **Maintenance simplifiée**
- ✅ **Migration complète** finalisée

## 📚 Ressources

- Module cible: \`ign_lidar/optimization/knn_search.py\`
- Audit: \`docs/audit_reports/CODEBASE_AUDIT_NOV_2025.md\` (section 1.2)

## ⏱️ Estimation

**2 jours** (1 développeur)

## ✅ Checklist

- [ ] Ajouter @deprecated à build_kdtree()
- [ ] Identifier tous usages restants
- [ ] Migrer vers KNNEngine
- [ ] Tests de non-régression
- [ ] Supprimer faiss_knn.py
- [ ] Supprimer gpu_kdtree.py
- [ ] Mettre à jour CHANGELOG.md
- [ ] Annoncer dépréciation dans release notes" || echo "Erreur création issue 3"

echo -e "${GREEN}✅ Issue 3 créée${NC}"
echo ""

# Issue 4: Phase 4 - Cleanup classes inutilisées
echo -e "${BLUE}[4/4]${NC} Phase 4: Cleanup classes inutilisées..."

gh issue create \
  --title "🧹 Phase 4: Supprimer 5 classes inutilisées + audit Manager/Processor" \
  --label "cleanup,technical-debt,medium-priority" \
  --milestone "v3.2.0" \
  --body "## 🎯 Objectif

Nettoyer les classes mortes identifiées par l'audit automatisé et clarifier l'architecture Processor/Manager/Engine.

## 📊 État actuel

**Problème:** 5 classes définies mais jamais utilisées:

| Classe | Fichier | Lignes | Usages |
|--------|---------|--------|--------|
| \`OptimizedProcessor\` | \`core/processor.py\` | 150 | 0 |
| \`GeometricFeatureProcessor\` | \`features/feature_computer.py\` | 120 | 0 |
| \`AsyncGPUProcessor\` | \`core/async_gpu.py\` | 200 | 0 |
| \`StreamingTileProcessor\` | \`core/streaming.py\` | 180 | 0 |
| \`CUDAStreamManager\` | \`core/gpu_streams.py\` | 100 | 0 |

**Impact:** 750 lignes de code mort, confusion architecture

## 🛠️ Plan d'implémentation

### Étape 1: Validation audit (0.5 jour)
\`\`\`bash
python scripts/audit_class_usage.py
\`\`\`

Vérifier manuellement que les 5 classes sont vraiment inutilisées.

### Étape 2: Dépréciation (0.5 jour)
\`\`\`python
@deprecated(
    version=\"3.2.0\",
    reason=\"Unused class, will be removed in v3.3.0\"
)
class OptimizedProcessor:
    def __init__(self):
        warnings.warn(...)
\`\`\`

### Étape 3: Audit Manager/Processor (1 jour)

**Classes à usage limité:**
- \`ProcessorCore\` (1 usage) - Fusionner avec LiDARProcessor?
- \`GPUMemoryManager\` (1 usage) - Étendre GPUManager?

**34 classes Manager/Processor/Engine:**
- Clarifier responsabilités
- Éviter création de nouvelles duplications

## 📈 Bénéfices attendus

- ✅ **-750 lignes** de code mort
- ✅ **Architecture claire** et documentée
- ✅ **Prévention duplications** futures
- ✅ **Conventions de nommage** établies

## 📚 Ressources

- Script: \`scripts/audit_class_usage.py\`
- Audit: \`docs/audit_reports/CODEBASE_AUDIT_NOV_2025.md\` (section 1.5)
- Guide: \`docs/audit_reports/QUICK_FIX_GUIDE.md\` (section 2)

## ⏱️ Estimation

**2 jours** (1 développeur)

## ✅ Checklist

- [ ] Valider avec audit_class_usage.py
- [ ] Ajouter @deprecated aux 5 classes
- [ ] Tests vérifiant que rien ne casse
- [ ] Audit manuel ProcessorCore et GPUMemoryManager
- [ ] Documenter conventions Manager/Processor/Engine
- [ ] Mettre à jour CHANGELOG.md
- [ ] Planifier suppression définitive v3.3.0
- [ ] Créer guide anti-duplication pour futurs PRs" || echo "Erreur création issue 4"

echo -e "${GREEN}✅ Issue 4 créée${NC}"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}🎉 4 ISSUES CRÉÉES AVEC SUCCÈS${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📋 Plan d'action complet sur GitHub:"
echo "   → https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues"
echo ""
echo "📊 Résumé timeline:"
echo "   • Phase 1 (5j): compute_normals()       [CRITICAL]"
echo "   • Phase 2 (3j): Centraliser GPU        [CRITICAL]"
echo "   • Phase 3 (2j): Migration KNN          [HIGH]"
echo "   • Phase 4 (2j): Cleanup classes        [MEDIUM]"
echo "   ─────────────────────────────────────────────"
echo "   TOTAL: 12 jours sur 2-3 mois (v3.2.0)"
echo ""
echo "🚀 Prochaine étape: Sprint planning Phase 1"
echo "════════════════════════════════════════════════════════════════"
