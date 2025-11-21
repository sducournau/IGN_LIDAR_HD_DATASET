# LiDAR Trainer Agent - Résumé Technique

## 🎯 Vue d'Ensemble

**LiDAR Trainer Agent** est un agent IA expert spécialisé dans le développement et l'entraînement de modèles de Deep Learning pour la segmentation sémantique de nuages de points 3D LiDAR.

### Caractéristiques Principales

✅ **Expertise Deep Learning 3D**

- PointNet++, Point Transformer, KPConv
- Architectures hiérarchiques et attention mechanisms
- Optimisation GPU (CuPy, RAPIDS, chunking)

✅ **Basé sur Florent Poux**

- Compilation de 5+ articles de référence
- Best practices académiques + industrielles
- Validation sur datasets réels (IGN LiDAR HD)

✅ **Intégration Projet IGN**

- Compatible avec `ign_lidar/` library
- Respect des conventions (PEP 8, type hints)
- Configuration Hydra/OmegaConf

✅ **Optimisé Copilot & Claude**

- Prompts structurés et testés
- Serena MCP integration
- Code exécutable clé-en-main

---

## 📁 Structure des Fichiers

```
.github/agents/
├── lidarTrainer.agent.md       # Définition complète de l'agent
├── KNOWLEDGE_BASE.md           # Synthèse techniques (PointNet++, features, etc.)
├── README.md                   # Guide d'utilisation complet
├── PROMPT_EXAMPLES.md          # 30+ exemples de prompts optimisés
├── config_template.yaml        # Template configuration entraînement
└── QUICKSTART.md              # ← Ce fichier
```

---

## 🚀 Démarrage Ultra-Rapide

### 1. Activation de l'Agent

```
@lidarTrainer [votre demande]
```

### 2. Exemple Minimal

```
@lidarTrainer Entraîne un PointNet++ SSG pour classifier
mes données LiDAR en 3 classes (ground, vegetation, buildings).
Dataset : 3.4M points, target mIoU > 0.85.
```

### 3. Réponse Attendue

L'agent va :

1. ✅ Analyser votre contexte (dataset, GPU, etc.)
2. ✅ Proposer une architecture justifiée
3. ✅ Générer la configuration complète
4. ✅ Fournir le code d'entraînement
5. ✅ Estimer le temps et les performances

---

## 💡 Cas d'Usage Principaux

| Besoin                     | Prompt Type               | Temps Réponse |
| -------------------------- | ------------------------- | ------------- |
| **Nouveau projet**         | Template complet          | 2-3 min       |
| **Optimiser features**     | Analyse + recommandations | 1-2 min       |
| **Débugger overfitting**   | Diagnostic + solutions    | 1-2 min       |
| **Comparer architectures** | Benchmark tableau         | 2-3 min       |
| **Transfer learning**      | Config fine-tuning        | 1-2 min       |
| **Optimiser GPU**          | Chunking strategy         | 2-3 min       |

---

## 📊 Performances Typiques

### Baseline Random Forest (CPU)

```
Features : XYZ + RGB
Train time : 2 min
Val mIoU : 0.54
```

### Optimisé Random Forest (CPU)

```
Features : XYZ + RGB + Geometric (10 features)
Train time : 5 min
Val mIoU : 0.85 (+31%)
```

### PointNet++ MSG (GPU)

```
Features : XYZ + RGB + Geometric
Train time : 3h (RTX 3090)
Val mIoU : 0.97 (+43% vs baseline)
```

---

## 🔧 Prérequis Techniques

### Environnement

```bash
# OBLIGATOIRE pour GPU
conda activate ign_gpu

# Vérifier GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Libraries Clés

- PyTorch ≥ 2.0
- Open3D ≥ 0.16
- CuPy (GPU)
- RAPIDS cuML (GPU)
- ign_lidar (ce projet)

### Dataset Format

```
Recommandé : .las ou .laz avec :
- Coordonnées XYZ
- Colors RGB (optionnel mais recommandé)
- Classification ASPRS
- Features géométriques (optionnel, l'agent peut générer)
```

---

## 🎓 Concepts Clés à Maîtriser

### 1. Feature Engineering

```
Features géométriques > Architecture complexe

Essentiels :
- Verticality (façades)
- Planarity (sol, toits)
- Omnivariance (complexité locale)
- Normal change rate (discontinuités)
```

### 2. Validation 3-Datasets

```
Train (60%) : Ajustement poids
Test (30%) : Tuning hyperparamètres
Validation (externe) : Généralisation réelle

Exemple : Train Louhans, Val Manosque
```

### 3. Classes Déséquilibrées

```
Solution : Weighted CrossEntropy

weights = [1/freq_class_i for i in classes]
criterion = nn.CrossEntropyLoss(weight=weights)
```

### 4. GPU Strategies

```
Full GPU : Dataset < GPU RAM
Chunked GPU : Dataset > GPU RAM
CPU : Fallback ou petit dataset

Sélection automatique dans ign_lidar
```

---

## 🐛 Troubleshooting Rapide

### Agent ne répond pas

```bash
# Vérifier fichier agent
cat .github/agents/lidarTrainer.agent.md | head -5

# Syntaxe TOON correcte ?
description: Agent IA spécialisé...
```

### GPU non utilisé

```bash
# Toujours utiliser :
conda run -n ign_gpu python script.py

# Jamais :
python script.py  # ❌ mauvais env
```

### Import errors

```bash
# Réinstaller projet
cd /path/to/IGN_LIDAR_HD_DATASET
pip install -e .
```

### Performances décevantes

```
❓ Features normalisées ? (MinMaxScaler)
❓ Classes équilibrées ? (weighted loss)
❓ Augmentation activée ? (rotation, jitter)
❓ Validation sur distribution différente ? (généralisation)
```

---

## 📚 Ressources par Niveau

### 🥉 Débutant

1. Lire `README.md` complet
2. Tester prompts de `PROMPT_EXAMPLES.md`
3. Utiliser `config_template.yaml`

### 🥈 Intermédiaire

1. Explorer `KNOWLEDGE_BASE.md`
2. Lire articles Florent Poux (`.github/articles/`)
3. Personnaliser architectures

### 🥇 Avancé

1. Modifier `lidarTrainer.agent.md`
2. Contribuer à `KNOWLEDGE_BASE.md`
3. Créer prompts optimisés custom

---

## 🔗 Liens Utiles

### Documentation

- [IGN LiDAR HD Docs](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- [GitHub Repository](https://github.com/sducournau/IGN_LIDAR_HD_DATASET)

### Papers Fondamentaux

- [PointNet++](https://arxiv.org/abs/1706.02413)
- [Point Transformer](https://arxiv.org/abs/2012.09164)

### Formations

- [3D Geodata Academy](https://learngeodata.eu)

### Outils

- [Open3D](http://www.open3d.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

---

## 📞 Support

**Questions ?** Ouvrir une issue GitHub avec :

```
- OS et versions libraries (torch, open3d, cupy)
- Dataset caractéristiques (taille, classes, features)
- GPU model et RAM
- Message d'erreur complet si applicable
- Prompt utilisé et réponse agent
```

**Contribuer ?** Pull requests bienvenues :

- Nouveaux prompts optimisés
- Corrections KNOWLEDGE_BASE.md
- Améliorations agent

---

## 🎯 Checklist Projet Type

```
☐ 1. Environnement configuré
   ☐ conda env ign_gpu activé
   ☐ GPU détecté (torch.cuda.is_available())
   ☐ ign_lidar installé (pip install -e .)

☐ 2. Dataset préparé
   ☐ Format .las/.laz avec classification
   ☐ Features calculées ou liste à générer
   ☐ Distribution classes connue

☐ 3. Objectifs définis
   ☐ mIoU target : [X]
   ☐ F1 per class : [X, Y, Z]
   ☐ Temps entraînement : < [X]h

☐ 4. Agent consulté
   ☐ Prompt structuré avec contexte complet
   ☐ Architecture proposée validée
   ☐ Configuration générée

☐ 5. Entraînement lancé
   ☐ conda run -n ign_gpu python train.py
   ☐ Monitoring actif (TensorBoard)
   ☐ Checkpoints sauvegardés

☐ 6. Évaluation complète
   ☐ Métriques sur test set
   ☐ Validation sur distribution externe
   ☐ Analyse erreurs (confusion matrix)

☐ 7. Itération / Production
   ☐ Améliorations identifiées
   ☐ Optimisation inference
   ☐ Documentation résultats
```

---

## 🏆 Métriques de Succès

**Good** (Prototypage)

```
mIoU : 0.75-0.80
F1 per class : 0.70-0.85
Gap train/val : < 15%
```

**Excellent** (Production)

```
mIoU : 0.85-0.90
F1 per class : 0.85-0.95
Gap train/val : < 10%
```

**State-of-art** (Recherche)

```
mIoU : > 0.90
F1 per class : > 0.90
Gap train/val : < 5%
Généralisation validée sur 3+ datasets
```

---

**Version** : 1.0  
**Dernière mise à jour** : Novembre 2025  
**Maintenu par** : Simon Ducournau  
**Basé sur** : Florent Poux, Ph.D. research

---

**Ready to start? 🚀**

```
@lidarTrainer Je commence un nouveau projet [description]
```
