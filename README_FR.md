# IGN LiDAR - Bibliothèque de Classification LOD des Bâtiments

Une bibliothèque Python pour traiter les données IGN (Institut National de l'Information Géographique et Forestière) LiDAR HD et les convertir en jeux de données prêts pour l'apprentissage automatique pour les tâches de classification des composants de bâtiments par niveau de détail (LOD).

## Vue d'ensemble

La bibliothèque `ign-lidar` traite les tuiles LAZ (LAS compressé) de l'IGN et les convertit en patches d'entraînement structurés adaptés aux modèles d'apprentissage automatique qui classifient les composants de bâtiments à différents niveaux de détail (LOD2 et LOD3).

### Fonctionnalités clés

- **Traitement LiDAR uniquement** : Aucune dépendance RGB, fonctionne exclusivement avec des données géométriques
- **Classification multi-niveaux** : Supporte les taxonomies de bâtiments LOD2 (15 classes) et LOD3 (30 classes)
- **Extraction riche de caractéristiques** : Calcule les caractéristiques géométriques incluant normales, courbure, planarité, verticalité, densité et rugosité
- **Traitement par patches** : Extrait des patches de 150m × 150m avec 10% de recouvrement pour des données d'entraînement cohérentes
- **Augmentation de données** : Rotations, perturbations, mise à l'échelle et dropout optionnels
- **Traitement parallèle** : Support multi-worker pour un traitement efficace à grande échelle

### Caractéristiques extraites

Pour chaque point dans les données LiDAR, les caractéristiques suivantes sont calculées :

1. **Attributs LiDAR de base** :

   - Coordonnées XYZ
   - Intensité (normalisée à [0,1])
   - Numéro de retour
   - Hauteur au-dessus du sol

2. **Caractéristiques géométriques** :
   - Normales de surface (vecteurs 3D)
   - Courbure principale
   - Planarité (basée sur les valeurs propres)
   - Verticalité (angle avec l'axe Z)
   - Horizontalité (complément de la verticalité)
   - Densité locale de points
   - Rugosité de surface

## Schémas de classification

### Classes de bâtiments LOD2 (15 classes)

Représentation simplifiée des bâtiments adaptée à l'urbanisme :

- **Structurel** : mur
- **Toits** : toit_plat, toit_pignon, toit_croupe
- **Détails de toit** : cheminée, lucarne
- **Façades** : balcon, surplomb
- **Fondation** : fondation
- **Contexte** : sol, végétation_basse, végétation_haute, eau, véhicule, autre

### Classes de bâtiments LOD3 (30 classes)

Représentation détaillée des bâtiments avec éléments architecturaux :

- **Murs** : mur_simple, mur_avec_fenêtres, mur_avec_porte
- **Toits détaillés** : toit_plat, toit_pignon, toit_croupe, toit_mansarde, toit_gambrel
- **Éléments de toit** : cheminée, lucarne_pignon, lucarne_appentis, vélux, arête_toit
- **Ouvertures** : fenêtre, porte, porte_garage
- **Détails de façade** : balcon, balustrade, surplomb, pilier, corniche
- **Fondation** : fondation, soupirail
- **Contexte** : sol, végétation_basse, végétation_haute, eau, véhicule, mobilier_urbain, autre

## Installation

### Prérequis

```bash
pip install numpy laspy scikit-learn tqdm
```

### Dépendances

- `numpy` : Calculs numériques
- `laspy` : Lecture des fichiers LAZ/LAS
- `scikit-learn` : PCA et KDTree pour les calculs géométriques
- `tqdm` : Barres de progression

## Utilisation

### Traiter une seule tuile

```bash
lidar-hd-process \
    --input /chemin/vers/tuile_ign.laz \
    --output /chemin/vers/dossier_sortie \
    --lod-level LOD2
```

### Traiter un répertoire de tuiles

```bash
lidar-hd-process \
    --input-dir /chemin/vers/tuiles_ign/ \
    --output /chemin/vers/dossier_sortie \
    --lod-level LOD3 \
    --num-workers 4
```

### Arguments en ligne de commande

- `--input` : Chemin vers un fichier LAZ unique
- `--input-dir` : Répertoire contenant des fichiers LAZ
- `--output` : Répertoire de sortie pour les patches traités (requis)
- `--lod-level` : Niveau de classification (`LOD2` ou `LOD3`, par défaut : `LOD2`)
- `--no-augment` : Désactiver l'augmentation de données
- `--num-augmentations` : Nombre de versions augmentées par patch (par défaut : 3)
- `--num-workers` : Nombre de workers parallèles (par défaut : 1)

## Format de sortie

Le script génère des patches d'entraînement au format NPZ avec la structure suivante :

```python
{
    'points': np.ndarray,      # [N, 3] coordonnées XYZ
    'normals': np.ndarray,     # [N, 3] normales de surface
    'curvature': np.ndarray,   # [N] courbure principale
    'intensity': np.ndarray,   # [N] intensité normalisée
    'return_number': np.ndarray, # [N] numéro de retour
    'height': np.ndarray,      # [N] hauteur au-dessus du sol
    'planarity': np.ndarray,   # [N] mesure de planarité
    'verticality': np.ndarray, # [N] mesure de verticalité
    'horizontality': np.ndarray, # [N] mesure d'horizontalité
    'density': np.ndarray,     # [N] densité locale de points
    'roughness': np.ndarray,   # [N] rugosité de surface
    'labels': np.ndarray,      # [N] étiquettes de classe
    'lod_level': str          # LOD2 ou LOD3
}
```

## Détails techniques

### Extraction de patches

- Taille de patch : 150m × 150m
- Recouvrement : 10% entre patches adjacents
- Points minimum par patch : 10 000
- Extraction basée sur grille avec indexation spatiale

### Calcul géométrique

- Estimation des normales : ACP sur k plus proches voisins (k=20)
- Courbure : Courbure principale par analyse des valeurs propres
- Calcul des caractéristiques : Analyse de voisinage local avec KDTree

### Augmentation de données

- **Rotation** : Rotation aléatoire autour de l'axe Z
- **Perturbation** : Petit déplacement aléatoire (±0,1m)
- **Échelle** : Mise à l'échelle aléatoire (0,95-1,05)
- **Dropout** : Suppression aléatoire de points (5-15%)

### Correspondance de classification ASPRS

Le script fait automatiquement correspondre les codes de classification LAS ASPRS standard aux classes LOD centrées sur les bâtiments :

- Sol (classe 2) → sol
- Végétation (classes 3,4,5) → végétation_basse/haute
- Bâtiment (classe 6) → mur (nécessite un raffinement ultérieur)
- Eau (classe 9) → eau
- Autres classes → classes de contexte appropriées

## Enrichissement de fichiers LAZ

Vous pouvez enrichir vos fichiers LAZ avec des caractéristiques géométriques pour visualisation dans QGIS ou analyse :

```bash
# Enrichissement standard (normales, courbure, hauteur)
ign-lidar enrich --input input.laz --output enriched/ --mode core

# Enrichissement pour bâtiments (+ verticalité, scores murs/toits)
ign-lidar enrich --input input.laz --output enriched/ --mode building
```

### Compatibilité QGIS

Les fichiers LAZ enrichis sont **entièrement compatibles avec QGIS** grâce à la compression LAZ standard (LASzip). Vous pouvez :

1. **Charger dans QGIS** :

   - Menu : Couche > Ajouter une couche > Ajouter une couche nuage de points
   - Sélectionner votre fichier enrichi `.laz`

2. **Visualiser les caractéristiques enrichies** :

   - Clic droit sur la couche > Propriétés
   - Onglet "Symbologie" > Choisir "Attribut"
   - Sélectionner une dimension enrichie :
     - `curvature` : Courbure de surface (détection d'arêtes)
     - `height_above_ground` : Hauteur normalisée
     - `normal_x/y/z` : Composantes des normales
     - `verticality` : Score de verticalité (murs)
     - `wall_score` / `roof_score` : Scores de détection (mode building)

3. **Tester la compatibilité** :
   ```bash
   python scripts/validation/test_qgis_compatibility.py fichier_enriched.laz
   ```

### Format de sortie

Les fichiers enrichis conservent :

- Toutes les données originales (XYZ, intensité, classification, etc.)
- Nouvelles dimensions supplémentaires (extra dimensions LAZ)
- Compression LAZ pour taille minimale
- Compatibilité avec LAStools, PDAL, CloudCompare

## Considérations de performance

- **Utilisation mémoire** : Les grandes tuiles sont traitées par patches pour gérer la mémoire
- **Traitement parallèle** : Utilisez `--num-workers` pour le traitement multi-cœurs
- **Calcul des caractéristiques** : Étape la plus intensive en calcul (O(N²) pour les voisinages)
- **E/S disque** : Les patches de sortie sont des fichiers NPZ compressés
- **Compression LAZ** : Les fichiers enrichis utilisent LASzip (backend standard)

## Intégration avec les pipelines ML

Les patches générés sont prêts pour :

- Frameworks d'apprentissage profond sur nuages de points (PointNet++, DGCNN, etc.)
- Classificateurs ML traditionnels avec caractéristiques géométriques
- Flux de travail de modélisation des informations du bâtiment (BIM)
- Applications d'urbanisme

## 📚 Documentation

### Documentation Actuelle

Pour une documentation complète, consultez le [Hub de Documentation](docs/README.md):

- 📖 **[Guides Utilisateur](docs/guides/)** - Guides de démarrage rapide et tutoriels
- ⚡ **[Fonctionnalités](docs/features/)** - Détection de saut intelligente, préférences de format
- 🔧 **[Référence Technique](docs/reference/)** - Optimisation mémoire, détails API
- 📦 **[Archive](docs/archive/)** - Corrections de bugs historiques et notes de version

### Liens Rapides

- **[Fonctionnalités de Saut Intelligent](docs/features/SMART_SKIP_SUMMARY.md)** - Éviter les téléchargements, enrichissements et traitements redondants
- **[Démarrage Rapide QGIS](docs/guides/QUICK_START_QGIS.md)** - Démarrer avec l'intégration QGIS
- **[Optimisation Mémoire](docs/reference/MEMORY_OPTIMIZATION.md)** - Guide d'optimisation des performances
- **[Préférences de Format de Sortie](docs/features/OUTPUT_FORMAT_PREFERENCES.md)** - LAZ 1.4 vs formats QGIS

### 🚀 Prochainement : Documentation Interactive

Nous travaillons sur un [site de documentation Docusaurus](DOCUSAURUS_PLAN.md) complet qui comprendra :

- 🌐 Support multilingue (Anglais & Français)
- 🔍 Recherche en texte intégral
- 📱 Design responsive mobile
- 📖 Tutoriels interactifs
- 🔗 Référence API auto-générée
- 💡 Exemples de code en direct

Voir le [Plan Docusaurus](DOCUSAURUS_PLAN.md) pour plus de détails.

## Licence

MIT License - voir le fichier [LICENSE](LICENSE) pour les détails.

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à soumettre une Pull Request.

## 📧 Support

Pour les problèmes et questions, veuillez utiliser la page [GitHub Issues](https://github.com/your-username/ign-lidar-hd-downloader/issues).
