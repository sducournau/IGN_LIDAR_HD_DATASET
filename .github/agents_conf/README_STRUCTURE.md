# Documentation LiDAR Trainer Agent

> 📁 Répertoire `agents_conf/` - Documentation complète de l'agent

## 📂 Structure

```
.github/
├── agents/
│   └── lidarTrainer.agent.md          # ⭐ Définition agent (fichier principal)
│
└── agents_conf/                        # 📚 Documentation complète
    ├── README_STRUCTURE.md             # Ce fichier
    ├── QUICKSTART.md                   # 🚀 Démarrage rapide (3 min)
    ├── README.md                       # 📘 Guide complet (10 min)
    ├── KNOWLEDGE_BASE.md               # 🧠 Base connaissances fondamentale (20 min)
    ├── KNOWLEDGE_BASE_EXTENDED.md      # 🚀 Techniques avancées 2025 (25 min)
    ├── PROMPT_EXAMPLES.md              # 💡 30+ exemples de prompts
    ├── INDEX.md                        # 🗺️ Navigation complète
    ├── UPDATE_SUMMARY.md               # ✨ Nouveautés v1.1
    ├── CHANGELOG_AGENT.md              # 📋 Historique versions
    └── config_template.yaml            # ⚙️ Template configuration
```

## 🎯 Utilisation

### Pour Utiliser l'Agent

**Fichier principal** : `../agents/lidarTrainer.agent.md`

```
@lidarTrainer [votre demande]
```

### Pour Consulter la Documentation

**Navigation rapide** :

1. **Nouveau ?** → [QUICKSTART.md](QUICKSTART.md)
2. **Apprendre ?** → [README.md](README.md)
3. **Approfondir ?** → [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md)
4. **Techniques avancées ?** → [KNOWLEDGE_BASE_EXTENDED.md](KNOWLEDGE_BASE_EXTENDED.md)
5. **Exemples ?** → [PROMPT_EXAMPLES.md](PROMPT_EXAMPLES.md)
6. **Navigation complète ?** → [INDEX.md](INDEX.md)

## 📊 Contenu

### Guides d'Utilisation

| Fichier                                  | Description                  | Temps lecture |
| ---------------------------------------- | ---------------------------- | ------------- |
| [QUICKSTART.md](QUICKSTART.md)           | Démarrage ultra-rapide       | 3 min         |
| [README.md](README.md)                   | Guide complet d'utilisation  | 10 min        |
| [PROMPT_EXAMPLES.md](PROMPT_EXAMPLES.md) | 30+ exemples de prompts      | 15 min        |
| [INDEX.md](INDEX.md)                     | Navigation dans toute la doc | Variable      |

### Base de Connaissances

| Fichier                                                  | Description                   | Temps lecture |
| -------------------------------------------------------- | ----------------------------- | ------------- |
| [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md)                   | Fondamentaux Deep Learning 3D | 20 min        |
| [KNOWLEDGE_BASE_EXTENDED.md](KNOWLEDGE_BASE_EXTENDED.md) | Techniques avancées 2024-2025 | 25 min        |

**Contenu KNOWLEDGE_BASE.md** :

- Architecture PointNet++
- Pipeline ML 3D complet
- Feature Engineering géométrique
- Optimisation GPU
- Cas d'usage IGN LiDAR HD

**Contenu KNOWLEDGE_BASE_EXTENDED.md** (NOUVEAU v1.1) :

- 🧩 Clustering avec Graph Theory
- 🔍 Segment Anything 3D (SAM)
- 🌳 Scene Graphs pour LLMs
- 🔄 Change Detection 3D (C2C, M3C2)
- 📊 Métriques avancées

### Nouveautés & Versions

| Fichier                                  | Description                 | Temps lecture |
| ---------------------------------------- | --------------------------- | ------------- |
| [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md)   | Résumé nouveautés v1.1      | 5 min         |
| [CHANGELOG_AGENT.md](CHANGELOG_AGENT.md) | Historique complet versions | 10 min        |

### Configuration

| Fichier                                      | Description                   | Usage          |
| -------------------------------------------- | ----------------------------- | -------------- |
| [config_template.yaml](config_template.yaml) | Template configuration modèle | Copier/adapter |

## 🚀 Quick Start

### 1. Découvrir l'agent (5 min)

```bash
# Lire démarrage rapide
cat QUICKSTART.md
```

### 2. Premier usage

```
@lidarTrainer Je veux entraîner un PointNet++ pour classifier
              mes données LiDAR IGN en 3 classes
```

### 3. Approfondir

```bash
# Guide complet
cat README.md

# Base de connaissances
cat KNOWLEDGE_BASE.md

# Techniques avancées (NEW v1.1)
cat KNOWLEDGE_BASE_EXTENDED.md
```

## 📚 Ressources Externes

### Articles Sources

**23 articles** de Florent Poux, Ph.D. dans `../.github/articles/`

Liste complète : [INDEX.md](INDEX.md#-articles-sources-florent-poux)

### Code Source

**Projet IGN LiDAR HD** : `../../ign_lidar/`

## 🆕 Nouveautés Version 1.1

### +18 Nouveaux Articles

Base de connaissances étendue de 5 à **23 articles** :

- Articles fondamentaux (2020-2023) : 5
- Nouveaux articles avancés (2024-2025) : 18

### 4 Nouveaux Domaines

1. **Clustering avancé** : Graph Theory, connectivité
2. **SAM 3D** : Segment Anything adapté aux nuages 3D
3. **Scene Graphs** : Relations spatiales pour LLMs
4. **Change Detection** : Monitoring temporel (C2C, M3C2)

### Nouveau Fichier

**KNOWLEDGE_BASE_EXTENDED.md** avec :

- 🧩 Clustering & Segmentation Non-Supervisée
- 🔍 Segment Anything 3D
- 🌳 Scene Graphs pour Spatial AI
- 🔄 Change Detection 3D

Détails complets : [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md)

## 📖 Parcours de Lecture

### 🥉 Débutant (Nouveau sur DL 3D)

```
1. QUICKSTART.md (3 min)
2. README.md - Exemples 1-2 (10 min)
3. config_template.yaml (5 min)
4. PROMPT_EXAMPLES.md - Architecture (10 min)
5. KNOWLEDGE_BASE.md - PointNet++ (15 min)
```

### 🥈 Intermédiaire (6+ mois DL 3D)

```
1. README.md complet (10 min)
2. KNOWLEDGE_BASE.md complet (30 min)
3. KNOWLEDGE_BASE_EXTENDED.md (50 min)
4. PROMPT_EXAMPLES.md tous (20 min)
5. INDEX.md (5 min)
```

### 🥇 Avancé (18+ mois DL 3D)

```
1. INDEX.md (overview)
2. KNOWLEDGE_BASE.md + KNOWLEDGE_BASE_EXTENDED.md
3. Articles sources complets (4-5h)
4. Code source ign_lidar/
5. Contribuer documentation
```

## 🔗 Navigation

### Liens Internes

Tous les fichiers utilisent des liens relatifs :

- `../agents/lidarTrainer.agent.md` → Définition agent
- `./KNOWLEDGE_BASE.md` → Base de connaissances
- `../../ign_lidar/` → Code source

### Fichier Principal

**Retour vers l'agent** : [../agents/lidarTrainer.agent.md](../agents/lidarTrainer.agent.md)

## 💡 Conseils

### Pour Apprendre Efficacement

1. **Commencer petit** : QUICKSTART → README → KNOWLEDGE_BASE
2. **Pratiquer** : Tester avec @lidarTrainer après chaque section
3. **Approfondir** : KNOWLEDGE_BASE_EXTENDED quand à l'aise
4. **Référencer** : INDEX.md comme table des matières

### Pour Trouver Rapidement

1. **Besoin spécifique** : INDEX.md > "Par Besoin"
2. **Thème précis** : INDEX.md > "Par Thématique"
3. **Niveau expertise** : INDEX.md > "Par Niveau d'Expertise"

## 📞 Support

### Questions sur la Documentation

- Consulter [INDEX.md](INDEX.md) pour navigation complète
- Vérifier [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md) pour nouveautés

### Questions sur l'Agent

```
@lidarTrainer [votre question]
```

### Issues GitHub

Ouvrir issue avec tag `[lidar-trainer-agent]`

---

**Version** : 1.1  
**Dernière mise à jour** : Novembre 2025  
**Maintenu par** : Simon Ducournau

📚 **Documentation complète** : Ce dossier  
⭐ **Agent principal** : [../agents/lidarTrainer.agent.md](../agents/lidarTrainer.agent.md)
