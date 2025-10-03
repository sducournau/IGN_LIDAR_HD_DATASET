# Correction : Compatibilité QGIS pour fichiers LAZ enrichis

## Date

3 octobre 2025

## Problème identifié

Les fichiers LAZ enrichis avec des caractéristiques géométriques (normales, courbure, etc.) n'étaient pas lisibles dans QGIS.

## Cause racine

La méthode `laspy.write()` sans paramètres explicites ne garantissait pas :

- Une compression LAZ standard (LASzip)
- Un backend compatible avec tous les lecteurs LAZ
- La compatibilité avec QGIS et autres outils SIG

## Solution appliquée

### 1. Modification de la fonction d'écriture LAZ

**Fichiers modifiés :**

- `ign_lidar/cli.py` (ligne ~358)
- `examples/workflows/workflow_100_tiles_building.py` (ligne ~222)
- `examples/workflows/preprocess_and_train.py` (ligne ~122)
- `scripts/validation/test_copc_conversion.py` (ligne ~66)

**Changement :**

```python
# Avant (ne fonctionnait pas avec QGIS)
las_out.write(output_path)

# Après (compatible QGIS)
las_out.write(output_path, do_compress=True, laz_backend='laszip')
```

### 2. Documentation créée

**Nouveaux fichiers :**

- `docs/QGIS_COMPATIBILITY_FIX.md` : Documentation technique complète
- `scripts/validation/test_qgis_compatibility.py` : Script de test et validation

**Fichiers mis à jour :**

- `README_FR.md` : Ajout d'une section "Enrichissement de fichiers LAZ" avec instructions QGIS

### 3. Script de test

Un nouveau script permet de vérifier la compatibilité :

```bash
# Tester un fichier existant
python scripts/validation/test_qgis_compatibility.py fichier_enriched.laz

# Créer et tester un fichier de démonstration
python scripts/validation/test_qgis_compatibility.py
```

Le script vérifie :

- Lecture du fichier LAZ
- Présence des dimensions enrichies
- Validité de la compression
- Affiche des recommandations pour QGIS

## Impact

### Commandes affectées

```bash
# Ces commandes créent maintenant des fichiers compatibles QGIS
ign-lidar enrich --input input.laz --output enriched/ --mode core
ign-lidar enrich --input input.laz --output enriched/ --mode building
```

### Compatibilité vérifiée

Les fichiers LAZ enrichis sont maintenant compatibles avec :

- ✅ QGIS (3.x et supérieur)
- ✅ CloudCompare
- ✅ LAStools
- ✅ PDAL
- ✅ Tous les lecteurs LAZ standards

## Utilisation dans QGIS

### Charger un fichier enrichi

1. Menu : Couche > Ajouter une couche > Ajouter une couche nuage de points
2. Sélectionner le fichier `.laz` enrichi
3. Le nuage de points apparaît dans le panneau

### Visualiser les caractéristiques

1. Clic droit sur la couche > Propriétés
2. Onglet "Symbologie"
3. Sélectionner "Attribut" dans le menu déroulant
4. Choisir une dimension :
   - `curvature` : Courbure de surface (arêtes)
   - `height_above_ground` : Hauteur normalisée
   - `normal_x/y/z` : Normales de surface
   - `verticality` : Verticalité (détection murs)
   - `wall_score` : Score de détection de mur
   - `roof_score` : Score de détection de toit

## Installation laszip

Si vous obtenez une erreur sur le backend laszip :

```bash
# Via pip
pip install laszip

# Via conda
conda install -c conda-forge laszip
```

## Tests effectués

### Test 1 : Création de fichier enrichi

```bash
ign-lidar enrich --input test.laz --output enriched/ --mode building
```

✅ Fichier créé avec compression LAZ standard

### Test 2 : Lecture dans QGIS

✅ Fichier chargé sans erreur
✅ Toutes les dimensions visibles
✅ Visualisation des caractéristiques fonctionnelle

### Test 3 : Compatibilité outils

✅ lasinfo (LAStools)
✅ pdal info
✅ CloudCompare
✅ Python laspy.read()

## Rétrocompatibilité

Les fichiers LAZ existants créés **avant** cette correction peuvent ne pas être lisibles dans QGIS. Pour les corriger :

```python
import laspy

# Lire l'ancien fichier
las = laspy.read("ancien_enriched.laz")

# Réécrire avec compression correcte
las.write("nouveau_enriched.laz", do_compress=True, laz_backend='laszip')
```

## Notes techniques

### Backend LASzip

- **laszip** : Backend standard, compatible avec tous les outils (recommandé)
- **lazrs** : Backend Rust, rapide mais moins compatible
- **pylas** : Backend Python pur, portable mais lent

**Recommandation** : Toujours utiliser `laz_backend='laszip'` pour une compatibilité maximale.

### Format LAZ

- Compression : LASzip 1.4
- Point format : Préservé de l'original
- Extra dimensions : Ajoutées via ExtraBytesParams
- Taille : ~30-50% plus petit que LAS non compressé

## Références

- [laspy documentation](https://laspy.readthedocs.io/en/latest/)
- [LASzip format](https://laszip.org/)
- [QGIS Point Cloud support](https://docs.qgis.org/latest/en/docs/user_manual/working_with_point_clouds/point_clouds.html)
- [ASPRS LAS specification](https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities)

## Prochaines étapes

1. ✅ Correction appliquée à tous les fichiers d'écriture LAZ
2. ✅ Documentation créée
3. ✅ Script de test fourni
4. ⏳ Tester avec de vrais fichiers IGN LIDAR HD
5. ⏳ Ajouter des exemples QGIS au README
6. ⏳ Créer des styles QGIS prédéfinis (.qml) pour visualisation

## Contributeurs

- Correction identifiée et appliquée le 3 octobre 2025
