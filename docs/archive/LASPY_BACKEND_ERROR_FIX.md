# Erreur laspy : 'str' object has no attribute 'is_available'

## Problème

Lors de l'enrichissement de fichiers LAZ avec la commande `ign-lidar enrich`, l'erreur suivante apparaît :

```
ERROR - 'str' object has no attribute 'is_available'
ERROR - No LazBackend could be initialized: 'str' object has no attribute 'is_available'
```

## Cause

Le paramètre `laz_backend='laszip'` passe une **chaîne de caractères** à `laspy.write()`, mais selon la version de `laspy`, celui-ci peut attendre soit :

- Une chaîne de caractères (anciennes versions)
- Un objet `LazBackend` de l'enum (versions récentes)

Cette incompatibilité de version crée l'erreur.

## Solution

**Supprimer complètement le paramètre `laz_backend`** et laisser `laspy` choisir automatiquement le meilleur backend disponible :

### Avant (causait l'erreur)

```python
las_out.write(output_path, do_compress=True, laz_backend='laszip')
```

### Après (fonctionne avec toutes les versions)

```python
las_out.write(output_path, do_compress=True)
```

## Explication

Quand on spécifie seulement `do_compress=True`, `laspy` :

1. Détecte automatiquement les backends LAZ disponibles
2. Choisit le meilleur dans cet ordre de priorité :
   - **laszip** (préféré, le plus compatible)
   - **lazrs** (Rust, rapide)
   - **pylas** (Python pur, portable)
3. Utilise celui qui est installé

Résultat : **Même comportement**, mais **compatible avec toutes les versions de laspy** !

## Fichiers corrigés

- `ign_lidar/cli.py`
- `examples/workflows/workflow_100_tiles_building.py`
- `examples/workflows/preprocess_and_train.py`
- `scripts/validation/test_copc_conversion.py`
- `scripts/validation/test_qgis_compatibility.py`

## Vérification

Pour vérifier quel backend est utilisé :

```python
import laspy

# Voir les backends disponibles
print("Available backends:", laspy.LazBackend.detect_available())

# Écrire un fichier (laspy choisit automatiquement)
las.write("output.laz", do_compress=True)
```

## Compatibilité QGIS

Cette modification **n'affecte pas** la compatibilité QGIS :

- ✅ Les fichiers restent en format LAZ standard
- ✅ La compression est toujours appliquée (`do_compress=True`)
- ✅ QGIS peut toujours les lire sans problème

Le backend utilisé (laszip, lazrs, ou pylas) produit tous des fichiers LAZ standards compatibles.

## Installation des backends

Si vous voulez forcer l'utilisation de laszip (recommandé) :

```bash
# Via pip
pip install laszip

# Via conda
conda install -c conda-forge laszip
```

Mais ce n'est **pas obligatoire** - `laspy` peut utiliser d'autres backends si laszip n'est pas installé.

## Références

- [laspy documentation](https://laspy.readthedocs.io/en/latest/basic.html#laz-backend)
- [Issue GitHub laspy](https://github.com/laspy/laspy/issues/) - Recherche "laz_backend"

---

**Date de la correction :** 3 octobre 2025
