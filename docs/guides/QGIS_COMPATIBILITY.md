# Fix: Fichiers LAZ enrichis non lisibles dans QGIS

## Problème

Les fichiers LAZ enrichis (avec des dimensions supplémentaires comme normals, curvature, etc.) n'étaient pas lisibles dans QGIS, même s'ils contenaient des données valides.

## Cause

Le problème venait de la méthode d'écriture des fichiers LAZ. Par défaut, `laspy.write()` sans paramètres peut créer des fichiers avec :

- Une compression non spécifiée ou incorrecte
- Un backend par défaut qui n'est pas compatible avec tous les lecteurs LAZ
- Un format qui n'est pas reconnu par QGIS (même si valide selon la spécification LAS)

## Solution

Utiliser explicitement les paramètres de compression LAZ lors de l'écriture :

```python
# Avant (ne fonctionne pas avec QGIS)
las_out.write(output_path)

# Après (compatible QGIS)
las_out.write(output_path, do_compress=True, laz_backend='laszip')
```

### Paramètres importants

- **`do_compress=True`** : Force la compression LAZ (LASzip)
- **`laz_backend='laszip'`** : Utilise le backend laszip qui est le standard de facto et compatible avec :
  - QGIS
  - CloudCompare
  - LAStools
  - PDAL
  - Tous les lecteurs LAZ standards

## Alternatives

Si `laszip` n'est pas disponible, `laspy` peut utiliser d'autres backends :

```python
# Backend lazrs (Rust, rapide mais moins compatible)
las_out.write(output_path, do_compress=True, laz_backend='lazrs')

# Backend pylas (Python pur, lent mais portable)
las_out.write(output_path, do_compress=True, laz_backend='pylas')
```

**Recommandation** : Toujours utiliser `laz_backend='laszip'` pour une compatibilité maximale.

## Vérification

Pour vérifier qu'un fichier LAZ est correctement formaté :

```bash
# Avec lasinfo (LAStools)
lasinfo fichier_enriched.laz

# Avec pdal
pdal info fichier_enriched.laz

# Dans Python
import laspy
las = laspy.read("fichier_enriched.laz")
print(f"Points: {len(las.points)}")
print(f"Extra dims: {list(las.point_format.extra_dimension_names)}")
```

## Installation de laszip

Si vous obtenez une erreur sur le backend laszip :

```bash
# Via pip
pip install laszip

# Via conda
conda install -c conda-forge laszip
```

## Impact

Cette correction s'applique à la commande `enrich` du CLI :

```bash
# Les fichiers enrichis seront maintenant lisibles dans QGIS
ign-lidar enrich --input input.laz --output enriched/ --mode building
```

## Fichiers modifiés

- `ign_lidar/cli.py` : Ligne ~356, fonction `_process_single_laz_enrich()`
  - Ajout de `do_compress=True, laz_backend='laszip'`

## Références

- [laspy documentation](https://laspy.readthedocs.io/en/latest/)
- [LASzip compression format](https://laszip.org/)
- [QGIS Point Cloud support](https://docs.qgis.org/latest/en/docs/user_manual/working_with_point_clouds/point_clouds.html)
