# Fichiers LAZ enrichis non lisibles dans QGIS - Solutions

## Diagnostic

Vos fichiers LAZ enrichis sont **techniquement corrects** :

- ✅ Format LAZ 1.4 valide
- ✅ Compression LAZ standard (lazrs)
- ✅ 15 dimensions enrichies présentes
- ✅ Non-COPC (format standard)
- ✅ Lisibles par laspy et autres librairies

**Le problème vient de QGIS, pas des fichiers !**

## Causes possibles

### 1. Version de QGIS trop ancienne

**Solution :** QGIS 3.18+ requis pour le support complet des nuages de points

```bash
# Vérifier version
qgis --version

# Mettre à jour si < 3.18
# https://qgis.org/fr/site/forusers/download.html
```

### 2. Extension Point Cloud désactivée

**Solution :** Activer l'extension intégrée

1. Menu : **Extensions** > **Gérer et installer les extensions**
2. Onglet **Installées**
3. Chercher "Point Cloud" ou "pdal"
4. Cocher la case si désactivé

### 3. Bibliothèque PDAL manquante

QGIS utilise PDAL pour lire les LAZ. Si PDAL n'est pas installé :

```bash
# Ubuntu/Debian
sudo apt install libpdal-plugin-python

# Windows - réinstaller QGIS avec option "PDAL"
# Ou installer PDAL séparément
```

### 4. Format Point format 6 avec extra bytes

Certaines versions de QGIS ont des problèmes avec :

- Point format 6 (LAS 1.4)
- Extra bytes (dimensions personnalisées)

**Solution 1 : Convertir en format plus compatible**

```python
# Script de conversion
import laspy

# Charger fichier enrichi
las = laspy.read("fichier_enriched.laz")

# Créer nouveau fichier avec format 3 (plus compatible)
from laspy import LasHeader
header = LasHeader(version="1.2", point_format=3)
header.scales = las.header.scales
header.offsets = las.header.offsets

las_out = laspy.LasData(header)
las_out.x = las.x
las_out.y = las.y
las_out.z = las.z
las_out.classification = las.classification
las_out.intensity = las.intensity

# Ajouter seulement quelques dimensions clés
las_out.add_extra_dim(laspy.ExtraBytesParams(name='planarity', type='f4'))
las_out.planarity = las.planarity

las_out.add_extra_dim(laspy.ExtraBytesParams(name='height_above_ground', type='f4'))
las_out.height_above_ground = las.height_above_ground

# Écrire
las_out.write("fichier_compatible.laz", do_compress=True)
```

**Solution 2 : Convertir en LAS non-compressé**

```bash
# Plus simple mais fichier plus gros
python -c "
import laspy
las = laspy.read('enriched.laz')
las.write('enriched.las')  # Non compressé
"
```

### 5. Problème de chemin Windows/WSL

Si vous utilisez WSL et que QGIS est sous Windows :

```bash
# Copier le fichier vers un chemin Windows natif
cp /mnt/c/Users/Simon/ign/pre_tiles/infrastructure_port/*.laz /mnt/c/Users/Simon/Desktop/

# Puis ouvrir dans QGIS : C:\Users\Simon\Desktop\fichier.laz
```

### 6. Fichier trop volumineux

Fichier de 192 MB peut être trop lourd pour QGIS selon la RAM disponible.

**Solution : Créer un sous-échantillon**

```python
import laspy
import numpy as np

las = laspy.read("fichier_enriched.laz")

# Garder 1 point sur 10
n_points = len(las.points)
keep_indices = np.random.choice(n_points, n_points // 10, replace=False)
keep_indices = np.sort(keep_indices)

# Créer fichier réduit
las_small = laspy.LasData(las.header)
las_small.points = las.points[keep_indices]

# Copier extra dimensions
for dim in ['planarity', 'linearity', 'height_above_ground', 'verticality']:
    if hasattr(las, dim):
        setattr(las_small, dim, getattr(las, dim)[keep_indices])

las_small.write("fichier_petit.laz", do_compress=True)
```

## Solutions alternatives

### Option A : Utiliser CloudCompare

**CloudCompare** est un logiciel gratuit qui lit TOUS les formats LAZ :

1. Télécharger : https://www.danielgm.net/cc/
2. Ouvrir le fichier LAZ
3. Visualiser les dimensions enrichies (scalar fields)
4. Exporter si besoin en autre format

### Option B : Utiliser PDAL

```bash
# Installer PDAL
conda install -c conda-forge pdal python-pdal

# Convertir le fichier
pdal translate enriched.laz output.laz

# Info sur le fichier
pdal info enriched.laz
```

### Option C : Visualisation web avec Potree

```bash
# Convertir en Potree (visualisation web)
pip install py4dgeo[potree]

# Générer visualisation
potree-converter enriched.laz -o potree_output/
```

### Option D : Créer une version simplifiée pour QGIS

Script automatique de conversion compatible QGIS :

```python
#!/usr/bin/env python3
"""
Convertir fichier LAZ enrichi en version compatible QGIS.
Réduit à 3 dimensions clés pour maximiser compatibilité.
"""

import laspy
import sys
from pathlib import Path

def simplify_for_qgis(input_file, output_file=None):
    """Simplifie un LAZ enrichi pour QGIS."""

    if output_file is None:
        output_file = str(Path(input_file).stem) + "_qgis.laz"

    print(f"Chargement: {input_file}")
    las = laspy.read(input_file)

    # Créer header simple (format 3, version 1.2)
    from laspy import LasHeader
    header = LasHeader(version="1.2", point_format=3)
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0, 0, 0]

    las_out = laspy.LasData(header)
    las_out.x = las.x
    las_out.y = las.y
    las_out.z = las.z
    las_out.classification = las.classification

    if hasattr(las, 'intensity'):
        las_out.intensity = las.intensity

    # Ajouter 3 dimensions clés seulement
    if hasattr(las, 'height_above_ground'):
        las_out.add_extra_dim(laspy.ExtraBytesParams(name='height', type='f4'))
        las_out.height = las.height_above_ground

    if hasattr(las, 'planarity'):
        las_out.add_extra_dim(laspy.ExtraBytesParams(name='planar', type='f4'))
        las_out.planar = las.planarity

    if hasattr(las, 'verticality'):
        las_out.add_extra_dim(laspy.ExtraBytesParams(name='vertical', type='f4'))
        las_out.vertical = las.verticality

    print(f"Écriture: {output_file}")
    las_out.write(output_file, do_compress=True)

    print(f"✓ Fichier simplifié créé: {output_file}")
    print(f"  Format: 1.2, Point format 3")
    print(f"  Dimensions: height, planar, vertical")
    print(f"  Essayez maintenant dans QGIS!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simplify_for_qgis.py fichier_enriched.laz")
        sys.exit(1)

    simplify_for_qgis(sys.argv[1])
```

## Test rapide

**Essayez d'abord ceci :**

```bash
# 1. Créer version simplifiée
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_downloader
python scripts/validation/simplify_for_qgis.py /mnt/c/Users/Simon/ign/pre_tiles/infrastructure_port/LHD_FXX_0889_6252_PTS_C_LAMB93_IGN69.laz

# 2. Ouvrir dans QGIS
# Le fichier *_qgis.laz devrait s'ouvrir

# 3. Si ça marche, convertir tous les fichiers
find /mnt/c/Users/Simon/ign/pre_tiles/ -name "*.laz" -exec python scripts/validation/simplify_for_qgis.py {} \;
```

## Informations système

Pour mieux vous aider, vérifiez :

```bash
# Version QGIS
qgis --version

# Plugins QGIS installés
ls ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/

# Version PDAL (si installé)
pdal --version
```

## Support

Si rien ne fonctionne :

1. Indiquez votre **version de QGIS**
2. Testez avec **CloudCompare** pour confirmer que c'est un problème QGIS
3. Essayez la **version simplifiée** (format 1.2)
4. Vérifiez les **logs QGIS** : Menu > Paramètres > Options > Journal

---

**Résumé :**

- Vos fichiers sont corrects
- QGIS a parfois des problèmes avec LAZ 1.4 + extra bytes
- Solutions : version simplifiée OU CloudCompare OU mise à jour QGIS
