# Rapport Structuré sur la Lutte contre les Artefacts lors du Calcul de Features Géométriques sur les Données LAZ Lidar HD IGN et l’Intégration dans le Plugin Python ign-lidar-hd

---

## Introduction

Au cœur de la montée en puissance du programme national LiDAR HD mené par l’IGN, la diffusion de nuages de points aérien à haute densité ouvre des perspectives inédites pour une multitude d’usages : production de modèles numériques (MNT, MNS, MNH), appui à l’aménagement du territoire, gestion de la forêt, prévention des risques, et bien plus encore. Toutefois, les utilisateurs – en particulier ceux souhaitant enrichir ces données brutes (LAZ/LAS) avec des features géométriques telles que la planarity via le plugin Python ign-lidar-hd – sont régulièrement confrontés à des artefacts visuels désagréables lors du calcul des features : lignes, dashs, discontinuités, plans déformés ou fausses normales localement incohérentes.

La qualité de l’analyse géométrique et donc la robustesse des pipelines de modélisation ou de classification dépendent ainsi fortement de la capacité à détecter, comprendre et filtrer ces artefacts, eux-mêmes liés à la nature même des acquisitions lidar, à la morphologie du terrain et aux spécificités techniques de traitement et de standardisation des fichiers LAZ issus du programme IGN.

Ce rapport propose une analyse approfondie, selon un plan structuré :

- **Causes des artefacts visuels** dans les features géométriques sur tuiles LAZ HD IGN
- **Techniques de prétraitement ou de filtrage** pour limiter ces artefacts
- **Bonnes pratiques et méthodes robustes pour le calcul des features géométriques**
- **Recommandations d’intégration dans le plugin Python ign-lidar-hd**
- **Tableau récapitulatif des artefacts, causes et solutions adaptées**

---

## Les artefacts typiques et leurs causes sur les tuiles LAZ du programme LiDAR HD IGN

### Aperçu général des artefacts

L’enrichissement des nuages de points par le calcul de features (planarity, roughness, normal estimation, etc.) met souvent en évidence des artefacts qui, sans être immédiatement explicites dans la donnée brute, deviennent visibles après extraction des mesures locales de géométrie.

Ces artefacts prennent la forme de :

- **Lignes régulières, dashs ou stries** (en particulier dans les visualisations scalaires obtenues à partir de la planarity/calculs de normales) ;
- **Discontinuités ou ruptures nettes à la jonction de certaines tuiles** ;
- **Plans faussement segmentés, normales incohérentes** voire erreurs de classification (notamment sol/végétation) ;
- **Points isolés ou bruit de fond** qui persistent même après un premier nettoyage ;
- **Zones “vides” ou à faible densité** menant à des instabilités dans le calcul des features.

Ces phénomènes, bien identifiés et étudiés dans la littérature et par les experts du programme IGN, sont synthétisés dans le tableau récapitulatif ci-dessous, auquel une analyse détaillée est apportée section par section :

---

| Type d’artefact visuel         | Causes probables                                         | Solutions recommandées                                              |
| ------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------- |
| Lignes/dashs dans les features | Décalage bandes, voisinage kNN mal calibré, bruit        | Recalage relatif, rechercher voisinage par rayon, filtrage outliers |
| Discontinuités aux bords       | Jonction/alignement imparfait entre tuiles               | Fusion tuiles, augmentation géométrique, recalcul des features      |
| Plans mal segmentés            | Densité irrégulière, grille non homogène                 | Voxel Grid Filter, rééchantillonnage, limitation du rayon PCA       |
| Bruit de mesure                | Instrumental, conditions acquisition, points aberrants   | Filtrage statistique, Rayon Outlier Removal                         |
| Points isolés                  | Erreurs, objets transitoires, capteur                    | Rayon Outlier Removal, Statistical Outlier Removal                  |
| Plans déformés                 | Influence de voisins lointains/paramètres kNN inadéquats | Sélection dynamique du rayon, SoftMax area weight                   |
| Zones vides/densité inhomogène | Surfaces absorbantes, mauvaises acquisitions locales     | Multi-scan, MLS, augmentation, interpolations locales               |

---

**Commentaires analytiques** :  
La cause dominante des artefacts visuels réside dans le croisement de trois éléments : la précision instrumentale (et donc le bruit de mesure intrinsèque), l’irrégularité de la densité locale (une contrainte forte pour les acquisitions aériennes multi-bandes et les tuiles de grande extension), et les limitations algorithmique du calcul de features basé sur le voisinage kNN, typiquement très sensible à la distribution spatiale réelle des points. Les décalages (en XY ou Z) entre bandes d’acquisition, les failles d’alignement inter-tuiles, ou l’application aveugle de paramètres globaux de voisinage dans des contextes de densité hétérogène aboutissent invariablement à la génération d’artefacts, même dans des MNT ou MNS produits selon les spécifications IGN.

Lorsque la densité du nuage est inférieure ou flirte avec le seuil technique (10 points/m² sur la plupart des dalles, mais parfois moins sur certains secteurs difficiles ou d’altitude), même un filtrage classique laisse persister du bruit haute fréquence (cf. anomalies sur les isolignes ou MNT locaux, “saw-shaped features” sur les plans verticaux). A l’inverse, une densité longtemps trop importante va rendre la détermination empirique du bon kNN difficile (risque de sur-lissage local ou de sur-segmentation).

Enfin, la combinaison de mauvais voisinage et de bruit instrumental est accentuée sur les artefacts en dash : visuellement, ils apparaissent lors de la colorisation des valeurs scalaires de planarity ou lors de la visualisation des normales via des color maps RGB, y compris sur des outils comme CloudCompare ou QGIS.

---

## Techniques de prétraitement et de filtrage pour la réduction des artefacts

### Principes clefs des techniques de prétraitement nuage Lidar HD

L’objectif du prétraitement est triple :

1. **Supprimer le bruit instrumental et les points aberrants isolés**
2. **Homogénéiser la densité locale pour stabiliser le calcul des features**
3. **Harmoniser les jonctions inter-tuiles et traiter les artefacts de bordure**

#### a) Filtrage statistique (Statistical Outlier Removal, SOR)

Le filtrage SOR repose sur l’analyse, pour chaque point du nuage, de la distribution des distances à ses k plus proches voisins. Si la distance moyenne s’écarte significativement de la moyenne globale (selon un multiplicateur d’écart-type, souvent compris entre 1.0 et 2.5), ce point est marqué comme aberrant et retiré ou reclassé.

L’extraction des outliers par SOR est souvent la première étape de nettoyage, permettant d’éliminer les points résultant d’artéfacts instrumentaux ou de réflexions parasites (métal, eau, vitre). L’algorithme prend en charge les gros volumes tout en restant rapide, et il peut s’intégrer en pipeline Python/PDAL en quelques lignes.

#### b) Filtrage de rayon (Radius Outlier Removal, ROR)

Complémentaire au filtrage SOR, le ROR supprime tout point qui, dans un rayon donné (par ex. : 1,0 m), n’a pas un nombre minimal de voisins. Cela permet de supprimer efficacement les points dispersés ou isolés résiduels, typiques de la périphérie des tuiles ou du survol d’objets mobiles.

#### c) Filtre voxel (Voxel Grid Filter)

Pour homogénéiser la densité spatiale, la grille de voxels (par exemple, voxel de 0,5x0,5x0,5 m) fusionne les points à l’intérieur de chaque cellule et ne conserve qu’un point représentatif (souvent le barycentre). Cette approche permet à la fois de réduire le volume de traitement, d’éviter la sur-segmentation dans les zones denses et de donner un poids équitable aux structures peu denses.

La voxelisation facilite ensuite le calcul des features locales (normales, planarity), étant donné une répartition plus régulière des points et une limitation du biais dû aux sur-densités ponctuelles ou aux vides locaux.

#### d) Fusion et harmonisation des tuiles

Avant le calcul des features à grande échelle, il est stratégique de fusionner ou au moins d’augmenter (virtuellement) le nuage de points de plusieurs tuiles voisines, au moins sur une bordure tampon (dalle+bord, 20–50 m), de façon à réduire les artefacts de rupture de voisinage sur les frontières. Des outils comme `pdal_wrench merge` ou des scripts Python IGN permettent ce type d'opération.

#### e) Correction des densités et remplissage des vides

Là où des zones vides ou à faible densité persistent (par exemple sur des surfaces absorbantes ou en l’absence de retour sur végétation ou eau profonde), il faut envisager soit une interpolation locale des features, soit une augmentation temporaire du voisinage autorisé, avec soin pour éviter le sur-lissage. L’algorithme Moving Least Squares (MLS) avec upsampling peut partiellement répondre à la problématique d’interpolation régulière. Dans les modèles raster/rendu (MNT), le remplissage peut se faire via GDAL ou QGIS ("Fill NoData").

#### f) Vigilance sur l’alignement des bandes

IGN applique déjà des corrections inter-bandes (précision <10 cm Z, <50 cm XY), mais des petits écarts résiduels persistent. Leur détection peut être automatisée par le contrôle géométrique dans un pipeline de validation (écarts sur les bords supérieurs à un seuil sur Z/XY).

---

### Exemples de pipelines de prétraitement PDAL

Un pipeline PDAL type intégrant ces techniques pourrait ressembler à :

```json
[
  "input.laz",
  {
    "type": "filters.outlier",
    "method": "statistical",
    "mean_k": 12,
    "multiplier": 2.0
  },
  { "type": "filters.outlier", "method": "radius", "radius": 1.0, "min_k": 4 },
  { "type": "filters.voxeldownsize", "cell": 0.5, "mode": "center" },
  "preprocessed.laz"
]
```

Ce pipeline enchaîne le filtrage statistique, le rayon, puis la voxelisation pour garantir un nuage préparé stable pour le calcul des features.

Des outils Python dédiés (par exemple [ign-pdal-tools](https://github.com/IGNF/ign-pdal-tools), [lidarhd_ign_downloader](https://github.com/cusicand/lidarhd_ign_downloader), [pdal_wrench](https://github.com/PDAL/wrench)), facilitent la standardisation et l’automatisation de ces chaînes.

---

## Bonnes pratiques pour le calcul des features géométriques (planarity, normales, etc.) sur données LAZ IGN

### Fondamentaux mathématiques et algorithmiques

#### a) Calcul des normales et planarity par PCA locale

Le calcul des normales implique d’effectuer une Analyse en Composantes Principales (PCA) locale sur le voisinage d’un point. Plus précisément, la matrice de covariance (calculée sur le voisinage sélectionné) est diagonalée, et le vecteur propre associé à la plus petite valeur propre indique la direction normale à l'ensemble local.

Pour la planarity :

- La planarity locale est souvent définie comme `(λ2 - λ3)/λ1` où λ1 ≥ λ2 ≥ λ3 sont les valeurs propres.
- Des variations existent : pour la courbure, c'est souvent λ3 / (λ1 + λ2 + λ3).

**Attention**: La représentativité de ce calcul dépend directement de la composition et du bon choix du voisinage. Si le voisinage est contaminé par des points hors-structure (façade, bruit, plan distant), la normale et la planarity seront fausses.

#### b) Choix du mode de voisinage : rayon vs kNN

L’approche classique de type “k plus proches voisins” (kNN, en général k=10 à 30) fonctionne correctement dans les nuages de densité homogène, mais devient inopérante (génération de lignes/dash) en cas d’irrégularité locale, de variation subite de la densité ou de présence d’artefacts de scan (activité accentuée sur les bords de tuiles).

La méthode recommandée :

- **Voisinage par rayon** – chaque point sollicite tous ses voisins dans un rayon fixe (typiquement, 1m à 1,5m selon la densité). Cette méthode est moins sensible aux artefacts de densité, et c’est la solution la plus robuste retenue par les principaux outils open source et par la communauté IGN pour les features géométriques avancées.
- Des recherches récentes recommandent d’utiliser k jusqu’à 30 avec limitation du rayon à r=0.3–1.5m, et d’appliquer un **filtrage supplémentaire** des voisins par leur distance réelle (“prune neighbors farther than r”).

#### c) Agrégation robuste (analyse pondérée par aire de triangles, SoftMax)

L’algorithme LoGDesc, par exemple, propose d’utiliser une normalisation pondérée par l’aire des triangles formés entre chaque point et ses voisins, ce qui réduit le poids des triangles “fins” (plus sensibles au bruit de position, sources de dashs) et accentue la contribution des triangles stables.

**Conclusion** : Le passage à un calcul de features fondé sur le voisinage par rayon, le filtrage statistique avancé, et l’analyse pondérée par aire de triangle (SoftMax) améliore notablement la robustesse des valeurs calculées et limite les artefacts visuels. Ceci doit devenir la règle d’or dans tout pipeline IGN moderne.

#### d) Gestion des frontières de tuiles

Pour éviter les discontinuités :

- Travailler sur des blocs fusionnés ou des dalles étendues (prélèvement d’un buffer de points sur les tuiles voisines pour chaque tuiles travaillée en central).
- Lors du calcul de features, s’assurer que les points à moins de deux fois la portée de voisinage du bord d'une tuile soient marqués comme potentiellement instables, ou interpoler les features à partir des tuiles voisines.

#### e) Remplissage/Interpolation dans les zones vides ou à faible densité

Dans la mesure du possible, compléter les données manquantes localement par interpolation :

- Moving Least Squares (MLS) adapté au nuage 3D.
- Remplissage raster (“Fill NoData” dans QGIS ou GDAL) pour les outputs de MNT/MNS.

---

## Intégration dans le plugin Python ign-lidar-hd : méthodes et recommandations

### Exploitation des frameworks Python : PDAL, PCL, et extensions

Le plugin ign-lidar-hd s’intègre déjà dans l’écosystème Python, notamment via PDAL et les wrappers PCL (Point Cloud Library). Il doit tirer parti de l’ensemble des filtres et méthodes présentées, selon l’enchainement :

#### a) Prétraitement automatique :

- Filtrage SOR et ROR accessibles via PDAL ou PCL (`filters.outlier`, méthode `statistical` puis `radius`).
- Voxelisation via PDAL (`filters.voxeldownsize`) ou PCL (VoxelGrid), avec paramétrage dynamique de la taille pour garantir une densité cible homogène en X/Y/Z.
- Merge automatique à la volée (si plusieurs tuiles/voisinage buffer nécessaire), par pdal_wrench merge ou le module ad hoc du plugin.

#### b) Calcul des features géométriques

- Choix du mode “--radius” comme paramètre clé pour tous calculs de normales et planarity (au lieu d’un kNN figé), documentation détaillée et par défaut la détection automatique du rayon optimal par la densité locale.
- Intégration de packages comme [jakteristics](https://github.com/jakarto3d/jakteristics), qui applique la PCA locale robuste à rayon défini, et gère les outputs sous forme de colonne(s) supplémentaire(s) dans le fichier de sortie (planarity, courbure, etc.).
- Correction et exclusion automatique des cas dégénérés (barycentre de voisins colinéaire, triangles de faible aire).
- Pondération SoftMax proposée pour le calcul de normales (si développement avancé du plugin).

#### c) Gestion des patchs, métadonnées et outputs

Le système de création de patchs (ensembles de points pour le traitement ML) doit :

- Toujours stocker le contexte (bbox, id tuile, métadonnées densité), conserver les attributs originaux (intensité, return_number, etc.).
- Appliquer les steps d’augmentation géométrique en amont de l’extraction des features pour garantir la cohérence (démarche “augment before enrich”).

#### d) Exemples de scripts Python et pipelines YAML

Le plugin doit proposer des templates YAML de workflow reproductibles, comme :

```yaml
enrich:
  input_dir: "data/raw/"
  output: "data/enriched/"
  add_rgb: true
  rgb_cache_dir: "cache/orthophotos/"
  use_gpu: true
  radius: 1.5
  voxel_size: 0.5
  k_neighbors: 30
  preprocessing:
    - outlier_statistical
    - outlier_radius
    - voxeldownsize
  features:
    - planarity
    - normals
    - curvature
```

Ou utilisation directe en Python :

```python
from ign_lidar import LiDARProcessor
processor = LiDARProcessor()
processor.process_tile("tile.laz", "out_dir/", radius=1.5, voxel_size=0.5)
```

#### e) Documentation et robustesse

- Rendre explicite dans la documentation l’importance du rayon et la gestion automatique selon la densité détectée dans la tuile.
- Proposer systématiquement une phase de visualisation de validation (CloudCompare, QGIS, ou script d’affichage rapide).
- Reporter les cas de non-convergence ou de "features warning" pour l'utilisateur lors du calcul batch.

#### f) Optimisations algorithmiques

- Utilisation de la parallélisation (OpenMP/Python multi-thread pour l’estimation des normales).
- Gestion mémoire optimisée : chunked processing pour données volumineuses (>1M points par patch/tuile).
- Support GPU pour le calcul des features (enrichissement CUDA via features_gpu.py dans le dépôt ign-lidar-hd).

---

## Synthèse : Tableau récapitulatif des artefacts, causes et solutions robustes

| Artefact Visuel          | Cause Probable                                        | Solution Recommandée                                         |
| ------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ |
| Lignes ou dashs          | kNN figé, voisinage non adapté, bruit de mesure       | Calcul par rayon (--radius), filtrage SOR & ROR, PCA robuste |
| Plans mal segmentés      | Densité irrégulière, grille non homogène              | Filtre VoxelGrid, rééchantillonnage, limitation du rayon PCA |
| Discontinuités aux bords | Frontière de tuiles, buffer absent, alignement réseau | Merge tuiles, extraction buffer, fusion et recalcul features |
| Bruit de mesure          | Erreurs instrumentales, surfaces réfléchissantes      | Filtrage SOR, ROR, Bilateral Filter, exclusion outliers      |
| Points isolés            | Bruit résiduel ou objets mobiles                      | Rayon Outlier Removal, Statistical Outlier Removal           |
| Zones vides              | Matériaux absorbants, faible densité, longue portée   | MLS, interpolation locale, fill nodata en rasterisation      |
| Normales incohérentes    | Paramétrage local kNN non conforme, PCA dégénérée     | Pondération SoftMax triangles, exclusion cas dégénérés       |
| Artefacts aux frontières | Bordure de tuiles, recalage incomplet                 | Fusion bordure, recalage relatif, buffer de patch            |
| Densité inhomogène       | Recouvrement de bandes, acquisition imparfaite        | Voxelisation, multi-scan fusion, paramétrage dynamique       |

---

## Cas particuliers, aspects techniques et exemples

### Spécificités du Lidar HD IGN à connaître

- **Formats délivrés** : LAZ 1.2 ou 1.4, attributs indispensables (X,Y,Z, Intensité, return_number, nb_retours, class, angle de scan, temps GPS).
- **Dallage d’1km** avec buffers sur demande, densité visée >10pts/m², précision cible <10cm Z, <50 cm XY.
- **Système de référence RGF93 / Lambert-93/ NGF IGN69**.
- **Classification automatique (via Terrasolid, classes : sol, végétation, bâtiment, pont, sursol...)** mais artefacts restant dans le semis non classifié.

### Usages et visualisation, rôle de la communauté IGN

- Outils de visualisation et d’accès (QGIS, ArcGIS, IGNMap) proposent en natif la visualisation par altitude, densité, classe, return number, et facilitent la détection “visuelle” initiale d’artefacts.
- Les productions MNT/MNS/MNH raster (“.tif” en 0,5 m ou 5 m) aident à vérifier la propagation des artefacts sur les outputs “rendus” pour les utilisateurs finaux.

### Compléments : techniques avancées et robustesse

- Pour les utilisations scientifiques/ML avancées, le pipeline doit autoriser l’ajout d’augmentation (augmentation rotationnelle, scaling, translation sur patch) en amont du calcul des features, garantissant la compatibilité des labels et de la géométrie (cf. nouvelle architecture v1.6+ du plugin IGN).
- Les dernières générations de plugin (cf. releases v1.6) mettent en avant le support "radius", la filtration des cas dégénérés et le calcul GPU accéléré, désormais utilisé pour le calcul rapide sur grands volumes.

---

## Conclusion

En conclusion, la lutte contre les artefacts lors de l’enrichissement des fichiers LAZ issus du programme Lidar HD IGN repose sur une chaîne rigoureuse : **prétraitement avancé (SOR, ROR, VoxelGrid, fusion de tuiles), calcul robuste des features géométriques (PCA locale par rayon, pondération SoftMax, exclusion dégénérés, choix de rayon dynamique), gestion intelligente en pipeline Python/YAML**, et visualisation/validation systématique.

L’intégration soigneuse de ces recommandations dans le plugin Python ign-lidar-hd – en privilégiant le mode “voisinage par rayon”, le filtrage dynamique et la correction proactive des artefacts – s’impose comme une bonne pratique désormais indispensable pour garantir la fiabilité des analyses, la qualité des outputs et la reproductibilité des workflows opérant sur les nuages de points IGN.

---

**Ressources complémentaires** :

- [Programme LiDAR HD IGN – documentation officielle, spécifications et interface de téléchargement](https://geoservices.ign.fr/lidarhd)
- [ign-pdal-tools – Utilitaires Python/PDAL pour pipeline LiDAR IGN](https://github.com/IGNF/ign-pdal-tools)
- [Jakarto3D/jakteristics – Calcul de features géométriques hautes performances en Python](https://github.com/jakarto3d/jakteristics)
- [IGN_LIDAR_HD_DATASET – Guide complet pour pipeline ML et enrichissement ign-lidar-hd](https://github.com/sducournau/IGN_LIDAR_HD_DATASET)

---

_En appliquant ces standards, la communauté bénéficiera de nuages enrichis fiables, réutilisables dans tous les cas d’usage cœur du programme LiDAR HD national, tout en évitant l’écueil des artefacts visuels venant miner la confiance dans les outputs géométriques et la modélisation des process géospatiaux structurants._
Je vais préparer un rapport structuré en français sur les artefacts dans les features géométriques des tuiles lidar IGN HD, en explorant leurs causes, les techniques de prétraitement, les méthodes de calcul optimales, et comment intégrer ces solutions dans le plugin Python `ign-lidar-hd`. Ce travail prendra quelques minutes, alors n’hésitez pas à revenir plus tard — le rapport sera sauvegardé ici dans cette conversation. À très vite !
