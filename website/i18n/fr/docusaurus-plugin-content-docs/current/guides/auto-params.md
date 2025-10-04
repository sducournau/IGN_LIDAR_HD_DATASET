---
sidebar_position: 8
title: Guide Auto-Params (Français)
description: Optimisation automatique des paramètres pour un traitement LiDAR de qualité optimale
keywords: [auto-params, optimisation, paramètres, qualité, automatisation]
---

## Guide Auto-Paramètres (v1.7.1+)

**🎯 Optimisation Automatique des Paramètres**  
**🔧 Aucun Réglage Manuel**  
**📊 Qualité Optimale Garantie**  
**⚡ Analyse Intelligente**

---

## 🚀 Vue d'ensemble

Auto-Paramètres (Auto-Params) est un système intelligent qui analyse automatiquement vos tuiles LiDAR et sélectionne les paramètres de traitement optimaux. Introduite dans la **v1.7.1**, cette fonctionnalité élimine le besoin de réglage manuel des paramètres et garantit des résultats cohérents et de haute qualité sur des jeux de données divers.

### Pourquoi Auto-Params ?

**Avant v1.7.1 (Réglage Manuel) :**

```bash
# Sélection manuelle des paramètres - expertise requise
ign-lidar-hd enrich input.laz output.laz \
  --k-neighbors 15 \
  --radius 2.5 \
  --sor-k 20 \
  --sor-std 1.8 \
  --patch-size 32
# ❌ Nécessite une expertise LiDAR
# ❌ Processus d'essais-erreurs
# ❌ Résultats sous-optimaux
# ❌ Qualité incohérente
```

**Avec v1.7.1 (Auto-Params) :**

```bash
# Optimisation automatique - fonctionne pour tous
ign-lidar-hd enrich input.laz output.laz --auto-params
# ✅ Aucune expertise requise
# ✅ Optimisation instantanée
# ✅ Résultats optimaux garantis
# ✅ Qualité cohérente
```

---

## 📊 Comment ça marche

### 1. Analyse des caractéristiques

Le système analyse automatiquement :

- **Densité de points** : Nombre de points par m²
- **Distribution spatiale** : Régularité du maillage
- **Complexité géométrique** : Présence de structures complexes
- **Bruit** : Niveau de bruit dans les données
- **Couverture** : Uniformité de la couverture

### 2. Sélection adaptative

Basé sur l'analyse, Auto-Params sélectionne :

```python
# Paramètres optimisés automatiquement
optimal_params = {
    'k_neighbors': auto_select_k(point_density),
    'radius': auto_calculate_radius(spatial_distribution),
    'sor_parameters': auto_tune_outlier_removal(noise_level),
    'patch_size': auto_optimize_patch(complexity),
    'chunk_size': auto_size_chunks(memory_available)
}
```

### 3. Validation et ajustement

- Tests de performance en temps réel
- Ajustements dynamiques si nécessaire
- Garantie de qualité minimale

---

## 🎯 Utilisation

### Activation simple

```bash
# Activation d'Auto-Params
ign-lidar-hd enrich input.laz output.laz --auto-params

# Avec verbose pour voir les paramètres sélectionnés
ign-lidar-hd enrich input.laz output.laz --auto-params --verbose
```

### Configuration avancée

```bash
# Auto-Params avec contraintes
ign-lidar-hd enrich input.laz output.laz \
  --auto-params \
  --quality-target high \
  --speed-preference balanced \
  --memory-limit 8GB
```

### Mode batch

```bash
# Optimisation automatique pour plusieurs fichiers
ign-lidar-hd batch-enrich data/ output/ \
  --auto-params \
  --adaptive-per-tile  # Paramètres uniques par tuile
```

---

## ⚙️ Niveaux de qualité

### Quality Target

```bash
# Qualité économique (rapide)
--quality-target economy
# Paramètres: vitesse privilégiée, qualité acceptable

# Qualité équilibrée (par défaut)
--quality-target balanced
# Paramètres: compromis vitesse/qualité optimal

# Haute qualité (précis)
--quality-target high
# Paramètres: qualité maximale, traitement plus long

# Qualité premium (recherche)
--quality-target premium
# Paramètres: qualité recherche, temps de calcul étendu
```

### Profils de vitesse

```bash
# Privilégier la vitesse
--speed-preference fast

# Équilibre vitesse/qualité
--speed-preference balanced

# Privilégier la qualité
--speed-preference quality
```

---

## 🔍 Analyse et feedback

### Mode verbose

```bash
ign-lidar-hd enrich input.laz output.laz --auto-params --verbose
```

**Sortie exemple :**

```
[AUTO-PARAMS] Analyse des caractéristiques de la tuile...
[AUTO-PARAMS] Densité détectée: 12.4 pts/m²
[AUTO-PARAMS] Complexité géométrique: Moyenne
[AUTO-PARAMS] Niveau de bruit: Faible
[AUTO-PARAMS]
[AUTO-PARAMS] Paramètres sélectionnés:
[AUTO-PARAMS]   k-neighbors: 12
[AUTO-PARAMS]   radius: 2.1m
[AUTO-PARAMS]   sor-k: 18
[AUTO-PARAMS]   sor-std: 1.6
[AUTO-PARAMS]   patch-size: 28
[AUTO-PARAMS]
[AUTO-PARAMS] Temps estimé: 3.2 minutes
[AUTO-PARAMS] Qualité attendue: 94.2%
```

### Rapport de performance

```bash
# Génération d'un rapport détaillé
ign-lidar-hd enrich input.laz output.laz \
  --auto-params \
  --performance-report report.json
```

---

## 📈 Types de données supportés

### Données urbaines

```bash
# Optimisé pour les environnements urbains
ign-lidar-hd enrich urban_tile.laz output.laz \
  --auto-params \
  --data-type urban
```

**Optimisations urbaines :**

- Détection de bâtiments renforcée
- Filtrage du bruit routier
- Gestion des surfaces réfléchissantes

### Données forestières

```bash
# Optimisé pour les environnements forestiers
ign-lidar-hd enrich forest_tile.laz output.laz \
  --auto-params \
  --data-type forest
```

**Optimisations forestières :**

- Pénétration de canopée
- Détection du sous-bois
- Classification multi-strates

### Données côtières

```bash
# Optimisé pour les zones côtières
ign-lidar-hd enrich coastal_tile.laz output.laz \
  --auto-params \
  --data-type coastal
```

**Optimisations côtières :**

- Gestion des surfaces d'eau
- Filtrage des embruns
- Détection des structures côtières

---

## 🎛️ Configuration personnalisée

### Fichier de configuration

```yaml
# config/auto_params.yaml
auto_params:
  quality_target: "balanced"
  speed_preference: "quality"

  constraints:
    max_processing_time: "30min"
    memory_limit: "16GB"
    min_quality_score: 0.90

  advanced:
    adaptive_chunking: true
    dynamic_adjustment: true
    quality_monitoring: true
```

```bash
# Utilisation avec configuration
ign-lidar-hd enrich input.laz output.laz \
  --auto-params \
  --config config/auto_params.yaml
```

### API Python

```python
from ign_lidar import AutoParamsProcessor

# Configuration avancée
processor = AutoParamsProcessor(
    quality_target='high',
    speed_preference='balanced',
    adaptive_per_region=True
)

# Analyse préliminaire
analysis = processor.analyze_tile("input.laz")
print(f"Paramètres recommandés: {analysis.recommended_params}")

# Traitement avec auto-optimisation
result = processor.process_with_auto_params("input.laz", "output.laz")
print(f"Qualité atteinte: {result.quality_score}")
```

---

## 🔬 Cas d'usage avancés

### Traitement adaptatif par région

```bash
# Auto-Params avec adaptation régionale
ign-lidar-hd enrich large_dataset/ output/ \
  --auto-params \
  --regional-adaptation \
  --region-size 1km
```

### Optimisation pour GPU

```bash
# Auto-Params optimisé GPU
ign-lidar-hd enrich input.laz output.laz \
  --auto-params \
  --use-gpu \
  --gpu-optimization auto
```

### Mode recherche

```bash
# Mode recherche avec journalisation complète
ign-lidar-hd enrich input.laz output.laz \
  --auto-params \
  --research-mode \
  --log-all-decisions \
  --export-metadata
```

---

## 📊 Comparaison des performances

### Résultats typiques

| Méthode              | Temps setup | Qualité  | Consistance | Expertise requise |
| -------------------- | ----------- | -------- | ----------- | ----------------- |
| Manuel traditionnel  | 2-4 heures  | Variable | Faible      | Élevée            |
| Auto-Params Economy  | 0 minutes   | 85-90%   | Élevée      | Aucune            |
| Auto-Params Balanced | 0 minutes   | 90-95%   | Élevée      | Aucune            |
| Auto-Params High     | 0 minutes   | 95-98%   | Élevée      | Aucune            |
| Auto-Params Premium  | 0 minutes   | 98-99%   | Élevée      | Aucune            |

### Gains de productivité

```bash
# Benchmark comparatif
ign-lidar-hd benchmark \
  --compare-methods manual,auto-params \
  --dataset test_tiles/ \
  --output benchmark_results.json
```

---

## 🔧 Dépannage

### Problèmes courants

**Auto-Params ne s'active pas :**

```bash
# Vérification de la version
ign-lidar-hd --version  # Doit être >= 1.7.1

# Mise à jour si nécessaire
pip install --upgrade ign-lidar-hd
```

**Qualité insuffisante :**

```bash
# Forcer un niveau de qualité supérieur
ign-lidar-hd enrich input.laz output.laz \
  --auto-params \
  --quality-target high \
  --force-premium-algorithms
```

**Traitement trop lent :**

```bash
# Privilégier la vitesse
ign-lidar-hd enrich input.laz output.laz \
  --auto-params \
  --speed-preference fast \
  --quality-target economy
```

### Mode diagnostic

```bash
# Diagnostic Auto-Params
ign-lidar-hd diagnostic \
  --auto-params-test \
  --input sample.laz \
  --report diagnostic_report.html
```

---

## 🎯 Meilleures pratiques

### Recommandations générales

1. **Première utilisation** : Commencer avec `--quality-target balanced`
2. **Production** : Utiliser `--auto-params` avec les paramètres par défaut
3. **Recherche** : Utiliser `--quality-target premium` avec `--research-mode`
4. **Lots importants** : Activer `--adaptive-per-tile`

### Optimisation workflow

```bash
# Workflow de production optimisé
ign-lidar-hd batch-enrich input_dir/ output_dir/ \
  --auto-params \
  --quality-target balanced \
  --adaptive-per-tile \
  --progress-bar \
  --resume-on-error
```

---

## 🔗 Ressources supplémentaires

- [Guide de Performance](./performance.md)
- [Dépannage](./troubleshooting.md)
- [API Auto-Params](../api/auto-params.md)
- [Notes de version v1.7.1](../release-notes/v1.7.1.md)

**🎉 Auto-Params révolutionne le traitement LiDAR en rendant l'optimisation accessible à tous, sans compromis sur la qualité !**
