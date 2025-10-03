---
sidebar_position: 3
title: Intégration QGIS
description: Guide pour utiliser les fichiers LAZ enrichis dans QGIS pour la visualisation et l'analyse
keywords: [qgis, visualisation, laz, nuage de points, sig]
---

# Guide d'Intégration QGIS

Apprenez à visualiser et analyser les fichiers LiDAR enrichis dans QGIS avec des caractéristiques géométriques pour l'analyse des composants de bâtiment.

## Vue d'Ensemble

Les fichiers LAZ enrichis de cette bibliothèque sont entièrement compatibles avec QGIS et incluent plus de 30 caractéristiques géométriques parfaites pour :

- Visualisation des composants de bâtiment
- Analyse de surface (murs, toits, sol)
- Détection de contours et analyse de linéarité
- Cartographie de densité et de rugosité

## Prérequis

### Exigences QGIS

- **QGIS 3.10+** avec support des nuages de points
- **Plugin lecteur LAZ/LAS** (habituellement inclus)

### Vérifier l'Installation

```bash
# Vérifier si QGIS peut lire les nuages de points
# Ouvrir QGIS et chercher : Couche > Ajouter une couche > Ajouter une couche de nuages de points
```

## Étape 1 : Enrichir les Fichiers LAZ

D'abord, créer des fichiers LAZ enrichis avec des caractéristiques géométriques :

```bash
# Enrichir les tuiles avec les caractéristiques de bâtiment
ign-lidar-hd enrich \
  --input-dir /chemin/vers/tuiles_brutes/ \
  --output /chemin/vers/tuiles_enrichies/ \
  --mode building \
  --num-workers 4
```

Cela ajoute plus de 30 attributs à chaque point incluant :

- **Propriétés de surface** : planarité, sphéricité, linéarité
- **Géométrie locale** : courbure, verticalité, normalité
- **Contexte spatial** : densité, rugosité, homogénéité

## Étape 2 : Charger dans QGIS

### Ajouter une Couche de Nuages de Points

1. **Ouvrir QGIS** et créer un nouveau projet
2. **Aller à** : `Couche > Ajouter une couche > Ajouter une couche de nuages de points`
3. **Sélectionner** votre fichier LAZ enrichi
4. **Cliquer** sur Ajouter

### Configuration Initiale

```text
Type de fichier : LAS/LAZ Point Cloud
Système de coordonnées : EPSG:2154 (RGF93 / Lambert-93)
Encodage : UTF-8
```

## Étape 3 : Visualisation des Caractéristiques

### Propriétés de la Couche

1. **Clic droit** sur la couche → `Propriétés`
2. **Aller à** l'onglet `Symbologie`
3. **Choisir** le mode de rendu

### Modes de Rendu Recommandés

#### Classification par Attribut

```text
Type de rendu : Classifié par valeurs d'attribut
Attribut : building_component (ou autre attribut enrichi)
Palette de couleurs : Spectral ou Custom
```

#### Visualisation par Intensité

```text
Type de rendu : Par valeur d'attribut
Attribut : planarity (planarité)
Plage : 0.0 - 1.0
Couleurs : Bleu (faible) → Rouge (forte)
```

### Attributs Utiles pour la Visualisation

| Attribut             | Plage   | Description                   |
| -------------------- | ------- | ----------------------------- |
| `building_component` | 0-14    | Classification des composants |
| `planarity`          | 0.0-1.0 | Degré de planarité            |
| `sphericity`         | 0.0-1.0 | Degré de sphéricité           |
| `verticality`        | 0.0-1.0 | Degré de verticalité          |
| `curvature`          | 0.0-1.0 | Courbure de surface           |
| `roughness`          | 0.0-1.0 | Rugosité locale               |

## Étape 4 : Analyse Avancée

### Filtrage par Attributs

1. **Ouvrir** la table d'attributs
2. **Utiliser** l'expression de filtrage :

```sql
-- Filtrer seulement les murs (composant 1)
"building_component" = 1

-- Points très planaires (surfaces lisses)
"planarity" > 0.8

-- Combinaison : murs planaires
"building_component" = 1 AND "planarity" > 0.7
```

### Styles Personnalisés

#### Style Composants de Bâtiment

```text
Règles de style :
- Sol (0) : Brun (#8B4513)
- Mur (1) : Rouge (#FF0000)
- Toit (2) : Bleu (#0000FF)
- Végétation (3) : Vert (#00FF00)
- Autre (4) : Gris (#808080)
```

#### Style Qualité de Surface

```text
Planarity > 0.9 : Vert (surface très lisse)
Planarity 0.7-0.9 : Jaune (surface lisse)
Planarity 0.5-0.7 : Orange (surface rugueuse)
Planarity < 0.5 : Rouge (surface très rugueuse)
```

## Étape 5 : Exportation et Partage

### Export en Formats Standards

```bash
# Exporter vers d'autres formats depuis QGIS
Couche → Exporter → Exporter vers un fichier
```

**Formats supportés :**

- **Shapefile** (avec attributs)
- **GeoPackage** (recommandé)
- **CSV** (points + attributs)
- **PLY** (nuage de points)

### Création de Cartes

1. **Basculer** vers le composeur d'impression
2. **Ajouter** des éléments de carte
3. **Configurer** les légendes pour les attributs enrichis
4. **Exporter** en PDF/PNG

## Optimisation des Performances

### Pour de Gros Fichiers

```text
Paramètres de rendu :
- Taille max de points : 1M
- Niveau de détail : Moyen
- Cache : Activé
```

### Indexation Spatiale

```bash
# Créer un index spatial pour de meilleures performances
# Outils → Traitement → Créer un index spatial
```

## Dépannage

### Problèmes Courants

**"Impossible de charger la couche"**

- Vérifier le chemin du fichier
- Confirmer que le fichier LAZ n'est pas corrompu
- Vérifier les permissions de fichier

**"Attributs non visibles"**

- S'assurer que le fichier a été enrichi
- Vérifier la table d'attributs de la couche
- Recalculer les statistiques de la couche

**"Rendu lent"**

- Réduire la taille d'affichage des points
- Utiliser le niveau de détail adaptatif
- Filtrer les points non nécessaires

## Exemples d'Analyse

### Détection de Bâtiments

```sql
-- Identifier les structures verticales (murs potentiels)
"verticality" > 0.8 AND "planarity" > 0.6

-- Structures horizontales (toits/sols potentiels)
"verticality" < 0.2 AND "planarity" > 0.7
```

### Analyse de Qualité

```sql
-- Points de bonne qualité pour l'entraînement
"planarity" > 0.5 AND "roughness" < 0.3

-- Points ambigus nécessitant une vérification
"planarity" < 0.3 OR "sphericity" > 0.8
```

## Voir Aussi

- [Guide d'Utilisation de Base](basic-usage.md) - Premiers pas avec la bibliothèque
- [Optimisation Mémoire](../reference/memory-optimization.md) - Traiter de gros fichiers
- [Commandes CLI](cli-commands.md) - Référence complète des commandes
