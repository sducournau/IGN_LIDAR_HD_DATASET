# Guide Rapide : Fichiers LAZ enrichis compatibles QGIS

## 🎯 Objectif

Créer des fichiers LAZ enrichis avec des caractéristiques géométriques (normales, courbure, planarity, etc.) qui sont :

- ✅ Lisibles dans QGIS
- ✅ Sans artefacts de lignes pointillées
- ✅ Prêts pour l'analyse et la visualisation

## 📋 Prérequis

```bash
# Installer les dépendances
pip install laspy laszip numpy scikit-learn

# Vérifier l'installation
python -c "import laspy; print('laspy OK')"
```

## 🚀 Utilisation

### 1. Enrichir un fichier LAZ

```bash
# Enrichissement simple (normales, courbure, hauteur)
ign-lidar enrich \
  --input /path/to/tile.laz \
  --output /path/to/enriched/ \
  --mode core

# Enrichissement pour bâtiments (+ verticality, wall_score, roof_score)
ign-lidar enrich \
  --input /path/to/tiles/ \
  --output /path/to/enriched/ \
  --mode building \
  --num-workers 4
```

**Note :** Le rayon de recherche pour les caractéristiques géométriques est **calculé automatiquement** pour éviter les artefacts !

### 2. Charger dans QGIS

1. **Ouvrir QGIS**
2. **Menu :** Couche → Ajouter une couche → Ajouter une couche nuage de points
3. **Sélectionner :** Votre fichier `*_enriched.laz`
4. Le nuage de points apparaît dans le panneau

### 3. Visualiser les caractéristiques

1. **Clic droit** sur la couche → Propriétés
2. **Onglet Symbologie**
3. **Rendu par :** Attribut
4. **Attribut :** Sélectionner une dimension :

#### Caractéristiques disponibles

**Mode CORE :**

- `normal_x`, `normal_y`, `normal_z` - Composantes des normales
- `curvature` - Courbure de surface (détection d'arêtes)
- `height_above_ground` - Hauteur normalisée
- `planarity` - Score de planarité (murs, toits) [0-1]
- `linearity` - Score de linéarité (arêtes) [0-1]
- `sphericity` - Score de sphéricité (végétation) [0-1]
- `anisotropy` - Anisotropie générale
- `roughness` - Rugosité de surface
- `density` - Densité locale de points

**Mode BUILDING (inclut tout CORE +) :**

- `verticality` - Score de verticalité (détection murs)
- `wall_score` - Probabilité de mur [0-1]
- `roof_score` - Probabilité de toit [0-1]
- `num_points_2m` - Nombre de points dans rayon de 2m
- `vertical_std` - Écart-type vertical
- `neighborhood_extent` - Étendue du voisinage
- `height_extent_ratio` - Ratio hauteur/étendue

### 4. Rampes de couleur recommandées

#### Pour `planarity` (surfaces)

- **Rampe :** Viridis ou Plasma
- **Min :** 0.0
- **Max :** 1.0
- **Interprétation :**
  - Bleu/violet (0-0.3) : Non-planaire (végétation, arêtes)
  - Jaune/vert (0.7-1.0) : Très planaire (murs, toits, sol)

#### Pour `linearity` (arêtes)

- **Rampe :** Inferno ou Hot
- **Min :** 0.0
- **Max :** 1.0
- **Interprétation :**
  - Noir/bleu (0-0.3) : Surfaces
  - Jaune/blanc (0.7-1.0) : Arêtes (coins de bâtiments, câbles)

#### Pour `verticality` (murs)

- **Rampe :** RdYlGn (inversée)
- **Min :** 0.0
- **Max :** 1.0
- **Interprétation :**
  - Rouge (0-0.3) : Horizontal (sol, toits plats)
  - Vert (0.7-1.0) : Vertical (murs)

## ✅ Vérification

### Test de compatibilité

```bash
# Tester un fichier enrichi
python scripts/validation/test_qgis_compatibility.py fichier_enriched.laz
```

**Attendu :**

```
✓ Fichier lu avec succès
✓ Dimensions supplémentaires trouvées:
    - curvature
    - height_above_ground
    - normal_x
    - normal_y
    - normal_z
    - planarity
    - linearity
    - ...
✓ Extension LAZ correcte
✓ En-tête LAZ valide

✅ RÉSULTAT: Fichier compatible QGIS
```

### Test k-neighbors vs radius

```bash
# Comparer ancienne méthode (artefacts) vs nouvelle (propre)
python scripts/validation/test_radius_vs_k.py fichier.laz
```

**Attendu :**

```
TEST 1: Fixed k-neighbors (k=50)
  Linearity:  Mean: 0.820  (HIGH = scan artifacts!)
  Planarity:  Mean: 0.320  (LOW = fragmented)

TEST 2: Radius-based search (r=0.75m)
  Linearity:  Mean: 0.250  (LOW = correct!)
  Planarity:  Mean: 0.680  (HIGH = correct!)

✅ Radius-based search is CLEARLY BETTER!
```

## 🔧 Résolution de problèmes

### Problème : Fichier ne s'ouvre pas dans QGIS

**Cause possible :** Backend laszip non installé

**Solution :**

```bash
pip install laszip
# ou
conda install -c conda-forge laszip
```

### Problème : Lignes pointillées dans planarity/linearity

**Cause :** Utilisation d'ancienne version du code (k-neighbors fixe)

**Solution :**

1. Mettre à jour le code (avec corrections du 3 oct 2025)
2. Régénérer les fichiers enrichis
3. Le rayon adaptatif élimine ces artefacts

### Problème : Dimensions enrichies non visibles

**Cause possible :** Fichier COPC mal converti

**Solution :**
Le code gère automatiquement la conversion COPC → LAZ standard.
Vérifier avec :

```bash
python -c "import laspy; las = laspy.read('file.laz'); print(las.point_format.extra_dimension_names)"
```

## 📊 Exemples de visualisation

### Détection de bâtiments

**Combinaison recommandée :**

1. Charger le nuage enrichi
2. Filtrer : `wall_score > 0.7` (murs probables)
3. Colorier par : `height_above_ground` (hauteur des étages)
4. Résultat : Murs de bâtiments colorés par hauteur

### Extraction de toits

**Combinaison recommandée :**

1. Filtrer : `planarity > 0.7 AND verticality < 0.3`
2. Colorier par : `roof_score`
3. Résultat : Surfaces de toit détectées

### Arêtes de bâtiments

**Combinaison recommandée :**

1. Filtrer : `linearity > 0.6`
2. Colorier par : `height_above_ground`
3. Résultat : Arêtes de bâtiments (coins, rives de toit)

## 📚 Documentation complète

- **Compatibilité QGIS :** `docs/QGIS_COMPATIBILITY_FIX.md`
- **Artefacts radius vs k :** `docs/RADIUS_BASED_FEATURES_FIX.md`
- **Formules géométriques :** `docs/GEOMETRIC_FEATURES_FIX.md`
- **Résumé complet :** `CORRECTIONS_SUMMARY.md`

## 🆘 Support

Si vous rencontrez des problèmes :

1. Consulter la documentation dans `docs/`
2. Exécuter les scripts de test
3. Vérifier que les fichiers ont bien l'extension `.laz`
4. Vérifier que QGIS supporte les nuages de points (v3.x+)

---

**Dernière mise à jour :** 3 octobre 2025
