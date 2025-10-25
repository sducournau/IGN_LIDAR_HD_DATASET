# 📁 Répertoire .serena/

Ce répertoire contient les fichiers de configuration et de contexte pour **Serena MCP**, qui améliore GitHub Copilot avec le contexte spécifique de votre projet.

## 📋 Fichiers Disponibles

### Configuration et État

| Fichier                | Description                                 |
| ---------------------- | ------------------------------------------- |
| `project_active.json`  | ✅ État d'activation du projet Serena MCP   |
| `ACTIVATION_STATUS.md` | 📊 Statut détaillé de la configuration MCP  |
| `GUIDE_UTILISATION.md` | 📖 Guide complet d'utilisation de Serena    |
| `verify_setup.sh`      | 🔍 Script de vérification de l'installation |

### Documentation Contextuelle

| Fichier               | Description                        |
| --------------------- | ---------------------------------- |
| `project_overview.md` | 🏗️ Architecture complète du projet |
| `coding_patterns.md`  | 🎨 Patterns de code spécifiques    |
| `QUICK_REFERENCE.md`  | ⚡ Référence rapide des commandes  |

### Documentation Additionnelle

| Fichier                    | Description                       |
| -------------------------- | --------------------------------- |
| `MCP_SETUP_COMPLETE.md`    | Installation et configuration MCP |
| `MCP_TEST_GUIDE.md`        | Guide de test des fonctionnalités |
| `CONFIGURATION_SUMMARY.md` | Résumé de la configuration        |
| `INSTALLATION_SUMMARY.md`  | Résumé de l'installation          |

## 🚀 Démarrage Rapide

### 1. Vérifier l'Installation

```bash
.serena/verify_setup.sh
```

**Attendu:** `✅ Installation complète! Configuration complète à: 100%`

### 2. Ouvrir le Guide d'Utilisation

```bash
cat .serena/GUIDE_UTILISATION.md
# ou ouvrir dans VS Code
code .serena/GUIDE_UTILISATION.md
```

### 3. Utiliser avec GitHub Copilot

Ouvrez Copilot Chat (`Ctrl+Shift+I`) et posez des questions :

```text
@workspace Comment fonctionne le FeatureOrchestrator?
```

```text
Montre-moi le pattern GPU/CPU fallback du projet
```

```text
Comment ajouter une nouvelle feature géométrique?
```

## 📊 Statut Actuel

```text
Projet: IGN_LIDAR_HD_DATASET v3.0.0
Status: 🟢 ACTIF ET CONFIGURÉ (100%)
MCP Servers: filesystem, github, memory (3/3)
Extensions: Copilot MCP, GitHub Copilot, GitHub Copilot Chat
```

## 🔧 Configuration MCP

Les serveurs MCP sont configurés dans `.vscode/settings.json` :

1. **Filesystem Server** - Accès aux fichiers du projet
2. **GitHub Server** - Intégration GitHub (repos, PRs, issues)
3. **Memory Server** - Mémoire persistante entre sessions

## 📚 Documentation Détaillée

### Pour Commencer

1. **Lire:** `GUIDE_UTILISATION.md` - Guide complet d'utilisation
2. **Vérifier:** `ACTIVATION_STATUS.md` - État de la configuration
3. **Exécuter:** `verify_setup.sh` - Vérification rapide

### Pour le Développement

1. **Architecture:** `project_overview.md` - Comprendre la structure du projet
2. **Patterns:** `coding_patterns.md` - Patterns de code à suivre
3. **Référence:** `QUICK_REFERENCE.md` - Commandes et configurations rapides

### Pour le Dépannage

1. **Tests:** `MCP_TEST_GUIDE.md` - Tester les fonctionnalités MCP
2. **Setup:** `MCP_SETUP_COMPLETE.md` - Détails d'installation
3. **Vérification:** `verify_setup.sh` - Diagnostic automatique

## 🎯 Fonctionnalités Serena MCP

### ✅ Activées

- GitHub Copilot avec contexte projet
- Suggestions de code intelligentes
- Complétion respectant les patterns
- Documentation automatique (docstrings)
- Tests générés automatiquement
- Accès aux fichiers via MCP Filesystem
- Mémoire persistante via MCP Memory

### ⚠️ Optionnelles

- GitHub MCP Server (nécessite `GITHUB_TOKEN`)

## 🔍 Vérification Rapide

```bash
# Installation complète?
.serena/verify_setup.sh

# Token GitHub configuré?
echo $GITHUB_TOKEN

# Node.js disponible? (pour MCP)
node --version

# Python et pytest OK?
python3 --version
pytest --version
```

## 💡 Astuces

### Utilisation de Copilot

1. **Utilisez @workspace** pour le contexte global
2. **Soyez spécifique** dans vos questions
3. **Référencez les fichiers** du projet
4. **Demandez des exemples** de code

### Maintenance

1. **Exécutez** `verify_setup.sh` régulièrement
2. **Mettez à jour** les fichiers de contexte si l'architecture change
3. **Rechargez VS Code** après modification de configuration

## 🆘 Support

### Problèmes Courants

**Copilot ne suggère pas les bons patterns:**

- Vérifiez que les fichiers `.serena/` existent
- Redémarrez VS Code
- Utilisez `@workspace` dans vos questions

**Serveur MCP ne démarre pas:**

- Vérifiez Node.js : `node --version` (v18+)
- Vérifiez les logs : View → Output → "Copilot MCP"
- Rechargez VS Code : Ctrl+Shift+P → "Reload Window"

**Token GitHub non reconnu:**

- Vérifiez : `echo $GITHUB_TOKEN`
- Scopes requis : `repo`, `read:org`, `read:user`
- Rechargez le shell : `source ~/.zshrc`

### Ressources

- Model Context Protocol: <https://modelcontextprotocol.io/>
- GitHub Copilot Docs: <https://docs.github.com/en/copilot>
- Projet GitHub: <https://github.com/sducournau/IGN_LIDAR_HD_DATASET>

## 🎉 Prêt à l'Emploi!

Votre workspace est configuré et prêt pour le développement avec GitHub Copilot et Serena MCP!

**Commencez maintenant:**

```bash
# Ouvrez Copilot Chat
# Ctrl+Shift+I (Linux/Windows) ou Cmd+Shift+I (Mac)

# Posez votre première question
@workspace Quelle est l'architecture du projet?
```

---

**Dernière mise à jour:** 2025-10-25  
**Version:** 1.0.0  
**Statut:** 🟢 Actif et Optimisé
