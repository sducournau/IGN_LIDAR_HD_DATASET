# Guide de Prétraitement

Le prétraitement des nuages de points est une étape critique pour atténuer les artefacts LiDAR et améliorer la qualité de l'extraction des caractéristiques. Ce guide couvre les techniques de prétraitement disponibles dans IGN Lidar HD.

## Vue d'Ensemble

### Le Problème

Les données LiDAR aéroportées contiennent souvent des artefacts causés par :

- **Artefacts de lignes de balayage** : Patterns visibles de chevauchement entre les lignes de scan
- **Variations de densité de points** : Densité de points irrégulière créant des artefacts géométriques
- **Bruit de mesure** : Points aberrants dus aux erreurs de capteur ou au retour atmosphérique
- **Points isolés** : Faux positifs depuis les oiseaux, les véhicules ou les erreurs de mesure

Ces artefacts dégradent les caractéristiques géométriques, en particulier les normales de surface et les propriétés de courbure.

### La Solution

IGN Lidar HD propose un pipeline de prétraitement en trois étapes :

1. **SOR (Suppression Statistique des Valeurs Aberrantes)** : Supprime le bruit statistique et les erreurs de mesure
2. **ROR (Suppression des Valeurs Aberrantes par Rayon)** : Élimine les artefacts de lignes de balayage et les points isolés
3. **Sous-échantillonnage par Voxel** : Homogénéise la densité de points (optionnel)

## Démarrage Rapide

### Exemple 1 : Prétraitement de Base

```bash
# Activer le prétraitement avec les paramètres par défaut
ign-lidar-hd enrich \
  --input-dir /data/raw_tiles/ \
  --output /data/enriched/ \
  --mode full \
  --preprocess
```

**Impact :** ~70% de réduction des artefacts de lignes, ~50% de normales plus lisses

### Exemple 2 : Prétraitement Conservateur (Données Haute Qualité)

```bash
# Paramètres plus relaxés pour préserver les détails
ign-lidar-hd enrich \
  --input-dir /data/raw_tiles/ \
  --output /data/enriched/ \
  --mode full \
  --preprocess \
  --sor-k 15 \
  --sor-std 3.0 \
  --ror-radius 1.5 \
  --ror-neighbors 3
```

**Cas d'Utilisation :** Données LiDAR haute qualité, besoin de préserver les détails fins

### Exemple 3 : Prétraitement Agressif (Données Bruitées)

```bash
# Paramètres stricts pour supprimer les artefacts maximum
ign-lidar-hd enrich \
  --input-dir /data/raw_tiles/ \
  --output /data/enriched/ \
  --mode full \
  --preprocess \
  --sor-k 10 \
  --sor-std 1.5 \
  --ror-radius 0.8 \
  --ror-neighbors 5 \
  --voxel-size 0.3
```

**Cas d'Utilisation :** Données bruitées, artefacts de lignes de balayage sévères, besoin de réduire la mémoire

## Détails des Techniques

### 1. Suppression Statistique des Valeurs Aberrantes (SOR)

#### Principe

SOR supprime les points avec des distances anormales à leurs k plus proches voisins :

1. Pour chaque point, calculer la distance moyenne à ses k voisins
2. Calculer la moyenne globale (μ) et l'écart-type (σ) des distances
3. Supprimer les points où `distance > μ + (std_multiplier × σ)`

#### Paramètres

| Paramètre   | Type  | Défaut | Description                                 |
| ----------- | ----- | ------ | ------------------------------------------- |
| `--sor-k`   | int   | 12     | Nombre de plus proches voisins à considérer |
| `--sor-std` | float | 2.0    | Multiplicateur d'écart-type pour le seuil   |

#### Guide d'Ajustement

**Augmenter `--sor-k`** (par ex., 15-20) :

- ✅ Suppression du bruit plus conservatrice
- ✅ Préserve les structures fines
- ❌ Temps de traitement plus lent
- 🎯 **Utilisez pour :** Données haute qualité, géométries détaillées

**Diminuer `--sor-k`** (par ex., 8-10) :

- ✅ Traitement plus rapide
- ✅ Suppression du bruit plus agressive
- ❌ Peut supprimer les structures fines
- 🎯 **Utilisez pour :** Données bruitées, traitement rapide

**Augmenter `--sor-std`** (par ex., 2.5-3.0) :

- ✅ Moins de points supprimés (conservateur)
- ✅ Préserve les bordures et les coins
- ❌ Peut garder plus de bruit
- 🎯 **Utilisez pour :** Préserver les détails des bâtiments

**Diminuer `--sor-std`** (par ex., 1.0-1.5) :

- ✅ Suppression du bruit plus agressive
- ✅ Artefacts plus lisses
- ❌ Peut sur-simplifier la géométrie
- 🎯 **Utilisez pour :** Artefacts sévères, données très bruitées

#### Algorithme

```python
def statistical_outlier_removal(points, k=12, std_multiplier=2.0):
    """
    Supprimer les valeurs statistiques aberrantes basées sur la distribution
    des distances aux k plus proches voisins.
    """
    # Construire l'arbre KD pour recherche efficace des voisins
    tree = KDTree(points)

    # Calculer les distances moyennes aux k voisins
    distances = []
    for point in points:
        dists, _ = tree.query(point, k=k+1)  # +1 pour exclure le point lui-même
        mean_dist = np.mean(dists[1:])  # Exclure distance 0 au point lui-même
        distances.append(mean_dist)

    # Calculer les statistiques globales
    mean = np.mean(distances)
    std = np.std(distances)
    threshold = mean + (std_multiplier * std)

    # Filtrer les points en dessous du seuil
    mask = np.array(distances) <= threshold
    return points[mask], mask
```

### 2. Suppression des Valeurs Aberrantes par Rayon (ROR)

#### Principe

ROR supprime les points isolés sans suffisamment de voisins dans un rayon spécifié :

1. Pour chaque point, compter les voisins dans le rayon de recherche
2. Supprimer les points avec moins de `min_neighbors` voisins

#### Paramètres

| Paramètre         | Type  | Défaut | Description                      |
| ----------------- | ----- | ------ | -------------------------------- |
| `--ror-radius`    | float | 1.0    | Rayon de recherche en mètres     |
| `--ror-neighbors` | int   | 4      | Nombre minimum de voisins requis |

#### Guide d'Ajustement

**Augmenter `--ror-radius`** (par ex., 1.5-2.0) :

- ✅ Recherche de voisins plus large
- ✅ Moins de points supprimés (conservateur)
- ❌ Peut manquer des points isolés serrés
- 🎯 **Utilisez pour :** Densité de points faible, préserver les petits objets

**Diminuer `--ror-radius`** (par ex., 0.5-0.8) :

- ✅ Détection plus stricte des points isolés
- ✅ Supprime mieux les artefacts de lignes de balayage
- ❌ Peut supprimer les structures fines
- 🎯 **Utilisez pour :** Densité de points haute, artefacts agressifs

**Augmenter `--ror-neighbors`** (par ex., 5-8) :

- ✅ Suppression du bruit plus agressive
- ✅ Nécessite des groupes de points plus denses
- ❌ Peut supprimer les bords légitimes
- 🎯 **Utilisez pour :** Artefacts de lignes de balayage sévères

**Diminuer `--ror-neighbors`** (par ex., 2-3) :

- ✅ Plus conservateur, préserve les points de bord
- ✅ Supprime seulement les points vraiment isolés
- ❌ Peut garder plus de bruit
- 🎯 **Utilisez pour :** Préserver les détails de géométrie

#### Algorithme

```python
def radius_outlier_removal(points, radius=1.0, min_neighbors=4):
    """
    Supprimer les valeurs aberrantes basées sur le comptage des voisins
    dans un rayon spécifié.
    """
    # Construire l'arbre KD pour recherche efficace par rayon
    tree = KDTree(points)

    # Compter les voisins pour chaque point
    mask = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        # Requête les voisins dans le rayon
        indices = tree.query_ball_point(point, radius)
        # Exclure le point lui-même du comptage
        neighbor_count = len(indices) - 1
        # Garder le point s'il a suffisamment de voisins
        mask[i] = neighbor_count >= min_neighbors

    return points[mask], mask
```

### 3. Sous-échantillonnage par Voxel (Optionnel)

#### Principe

Le sous-échantillonnage par voxel réduit la densité de points tout en préservant la géométrie :

1. Diviser l'espace en grille 3D de voxels (cubes de `voxel_size`)
2. Pour chaque voxel contenant des points, calculer le centroïde
3. Remplacer tous les points du voxel par un point unique au centroïde

#### Paramètres

| Paramètre      | Type  | Défaut | Description               |
| -------------- | ----- | ------ | ------------------------- |
| `--voxel-size` | float | aucun  | Taille du voxel en mètres |

#### Guide d'Ajustement

**Petit Voxel** (0.1-0.3m) :

- ✅ Préserve bien les détails
- ✅ Réduit modérément la mémoire (30-50%)
- ❌ Avantages de vitesse limités
- 🎯 **Utilisez pour :** Bâtiments détaillés, petits objets

**Voxel Moyen** (0.3-0.5m) :

- ✅ Bon équilibre qualité/vitesse
- ✅ Réduit considérablement la mémoire (50-70%)
- ❌ Peut lisser les petits détails
- 🎯 **Utilisez pour :** Usage général, grands ensembles de données

**Grand Voxel** (0.5-1.0m) :

- ✅ Réduction maximale de la mémoire (70-90%)
- ✅ Traitement le plus rapide
- ❌ Perd des détails fins
- 🎯 **Utilisez pour :** Mémoire très limitée, formes de bâtiments uniquement

#### Algorithme

```python
def voxel_downsampling(points, voxel_size=0.5):
    """
    Sous-échantillonner les points en moyennant les points dans chaque voxel.
    """
    # Calculer les indices de voxel pour chaque point
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Créer une clé unique pour chaque voxel
    voxel_keys = {}
    for i, voxel_idx in enumerate(voxel_indices):
        key = tuple(voxel_idx)
        if key not in voxel_keys:
            voxel_keys[key] = []
        voxel_keys[key].append(i)

    # Calculer le centroïde pour chaque voxel
    downsampled_points = []
    for indices in voxel_keys.values():
        centroid = np.mean(points[indices], axis=0)
        downsampled_points.append(centroid)

    return np.array(downsampled_points)
```

## Préréglages Recommandés

### 1. Conservateur (Données Haute Qualité)

```bash
--preprocess --sor-k 15 --sor-std 3.0 --ror-radius 1.5 --ror-neighbors 3
```

**Caractéristiques :**

- Suppression du bruit minimale
- Préserve la géométrie détaillée
- Adapté aux données haute qualité
- ~5-10% de points supprimés

**Utilisez Quand :**

- Données LiDAR haute qualité (>10 pts/m²)
- Besoin de préserver les détails fins
- Géométries de bâtiments complexes
- Artefacts minimes

### 2. Standard/Équilibré (Usage Général)

```bash
--preprocess
```

**Caractéristiques :**

- Paramètres par défaut équilibrés
- Bon rapport qualité/vitesse
- Adapté à la plupart des ensembles de données
- ~10-20% de points supprimés

**Utilisez Quand :**

- Qualité de données moyenne
- Usage général
- Pas sûr de quel préréglage utiliser
- Artefacts modérés

### 3. Agressif (Données Bruitées)

```bash
--preprocess --sor-k 10 --sor-std 1.5 --ror-radius 0.8 --ror-neighbors 5
```

**Caractéristiques :**

- Suppression des artefacts maximale
- Normales plus lisses
- Peut sur-simplifier la géométrie
- ~20-30% de points supprimés

**Utilisez Quand :**

- Données bruitées ou ancienne acquisition
- Artefacts de lignes de balayage sévères
- La géométrie grossière est suffisante
- Artefacts sévères

### 4. Optimisé Mémoire (Grands Ensembles de Données)

```bash
--preprocess --voxel-size 0.4 --sor-k 10 --sor-std 2.0
```

**Caractéristiques :**

- Réduit significativement l'utilisation de la mémoire
- Traitement plus rapide
- Bon pour les grands ensembles de données
- ~40-60% de points supprimés

**Utilisez Quand :**

- Erreurs de mémoire insuffisante
- Traitement de grandes tuiles
- Ressources système limitées
- Besoin de vitesse

### 5. Urbain (Zones de Haute Densité)

```bash
--preprocess --sor-k 12 --sor-std 2.0 --ror-radius 0.8 --ror-neighbors 5 --voxel-size 0.3
```

**Caractéristiques :**

- Optimisé pour environnements urbains denses
- Homogénéise les zones à haute densité
- Équilibre détails et vitesse
- ~30-40% de points supprimés

**Utilisez Quand :**

- Centres-villes, zones urbaines denses
- Bâtiments avec géométrie complexe
- Haute densité de points (>15 pts/m²)
- Artefacts urbains

## API Python

### Utilisation de Base

```python
from ign_lidar.preprocessing import (
    statistical_outlier_removal,
    radius_outlier_removal,
    voxel_downsampling
)

# Charger les points (N, 3)
points = load_laz_file("input.laz")

# Appliquer SOR
filtered_points, sor_mask = statistical_outlier_removal(
    points, k=12, std_multiplier=2.0
)

# Appliquer ROR
filtered_points, ror_mask = radius_outlier_removal(
    filtered_points, radius=1.0, min_neighbors=4
)

# Optionnel : sous-échantillonnage
downsampled = voxel_downsampling(filtered_points, voxel_size=0.5)
```

### Pipeline Complet

```python
from ign_lidar.preprocessing import preprocess_point_cloud

# Prétraitement avec tous les paramètres
result = preprocess_point_cloud(
    points=points,
    apply_sor=True,
    sor_k=12,
    sor_std_multiplier=2.0,
    apply_ror=True,
    ror_radius=1.0,
    ror_min_neighbors=4,
    voxel_size=None,  # Optionnel
    intensity=intensity,  # Optionnel
    classification=classification  # Optionnel
)

# Extraire les résultats
filtered_points = result["points"]
filtered_intensity = result.get("intensity")
filtered_classification = result.get("classification")
statistics = result["statistics"]

print(f"Points originaux: {statistics['original_points']}")
print(f"Points après SOR: {statistics['after_sor']}")
print(f"Points après ROR: {statistics['after_ror']}")
print(f"Points finaux: {statistics['final_points']}")
print(f"Réduction totale: {statistics['total_reduction_percent']:.1f}%")
```

### Intégration avec Processor

```python
from ign_lidar.processor import LidarProcessor

# Créer le processeur avec prétraitement
processor = LidarProcessor(
    preprocess=True,
    preprocess_config={
        'sor_k': 12,
        'sor_std_multiplier': 2.0,
        'ror_radius': 1.0,
        'ror_min_neighbors': 4,
        'voxel_size': 0.5  # Optionnel
    }
)

# Traiter la tuile
features = processor.process_tile("input.laz")
```

## Considérations de Performance

### Impact sur le Temps de Traitement

| Configuration       | Temps (relatif) | Réduction | Qualité    |
| ------------------- | --------------- | --------- | ---------- |
| Aucun prétraitement | 1.0x (base)     | 0%        | Baseline   |
| SOR uniquement      | 1.15x           | ~10%      | Modérée    |
| ROR uniquement      | 1.20x           | ~15%      | Bonne      |
| SOR + ROR           | 1.30x           | ~20%      | Excellente |
| SOR + ROR + Voxel   | 1.10x           | ~50%      | Très Bonne |

### Impact sur l'Utilisation de la Mémoire

| Configuration       | Mémoire (pic) | Réduction |
| ------------------- | ------------- | --------- |
| Aucun prétraitement | 100% (base)   | 0%        |
| SOR + ROR           | 85-90%        | ~20%      |
| + Voxel 0.5m        | 40-50%        | ~50%      |
| + Voxel 0.3m        | 50-60%        | ~40%      |
| + Voxel 0.1m        | 70-80%        | ~25%      |

### Amélioration de la Qualité des Caractéristiques

| Métrique                        | Sans Prétraitement | Avec Prétraitement | Amélioration |
| ------------------------------- | ------------------ | ------------------ | ------------ |
| Bruit des normales (σ)          | 0.24               | 0.10               | 58% ↓        |
| Erreur de courbure              | 0.18               | 0.08               | 56% ↓        |
| Cohérence du voisinage          | 0.72               | 0.91               | 26% ↑        |
| Artefacts de lignes de balayage | Sévères            | Minimes            | ~80% ↓       |

## Meilleures Pratiques

### 1. Commencez Conservateur

Toujours commencer avec des paramètres conservateurs et augmenter la rigueur seulement si nécessaire :

```bash
# Première tentative - conservateur
--preprocess --sor-k 15 --sor-std 3.0

# Si les artefacts persistent - standard
--preprocess

# Si encore des problèmes - agressif
--preprocess --sor-k 10 --sor-std 1.5 --ror-radius 0.8
```

### 2. Inspectez Visuellement les Résultats

Utilisez des outils de visualisation (CloudCompare, QGIS) pour inspecter :

- Artefacts de lignes de balayage (motifs de chevauchement)
- Qualité des normales de surface (lissage uniforme vs. en bloc)
- Préservation des bords (coins de bâtiments, bordures de toit)
- Suppression des points aberrants (bruit vs. petits objets)

### 3. Considérez les Caractéristiques des Données

Ajustez les paramètres selon les propriétés de votre ensemble de données :

**Haute Densité (>15 pts/m²):**

- Utilisez des valeurs `sor-k` plus élevées (15-20)
- Utilisez un rayon `ror-radius` plus petit (0.5-0.8m)
- Considérez le sous-échantillonnage par voxel (0.2-0.3m)

**Faible Densité (moins de 5 pts/m²):**

- Utilisez des valeurs `sor-k` plus faibles (8-10)
- Utilisez un rayon `ror-radius` plus grand (1.5-2.0m)
- Évitez le sous-échantillonnage par voxel ou utilisez de grands voxels (>0.5m)

**Données Bruitées (anciennes ou météo défavorable):**

- Utilisez des paramètres plus agressifs
- `sor-std` inférieur (1.5-2.0)
- `ror-neighbors` supérieur (5-8)

### 4. Équilibrez Qualité et Vitesse

Pour les grands ensembles de données, trouvez le bon équilibre :

```bash
# Priorité à la vitesse
--preprocess --voxel-size 0.5 --sor-k 10

# Priorité à la qualité
--preprocess --sor-k 15 --sor-std 3.0 --ror-radius 1.5
```

### 5. Surveillez les Statistiques de Réduction

Gardez un œil sur le pourcentage de points supprimés dans les logs :

```
INFO: Preprocessing: 1,234,567 points → 987,654 points (20.0% reduction)
```

**Lignes Directrices :**

- **5-15%** : Réduction normale, bonne préservation
- **15-30%** : Réduction modérée, attendue avec le nettoyage agressif
- **30-50%** : Réduction élevée, vérifiez que les détails sont préservés
- **>50%** : Réduction excessive, paramètres peut-être trop agressifs

### 6. Utiliser les Préréglages comme Points de Départ

Les préréglages recommandés sont des points de départ éprouvés, mais ajustez selon les besoins :

```bash
# Commencez avec un préréglage
--preprocess  # Standard

# Ajustez un paramètre spécifique
--preprocess --sor-std 2.5  # Plus conservateur que la norme

# Combinez avec voxel pour la vitesse
--preprocess --voxel-size 0.3
```

## Dépannage

### Problème 1 : Trop de Points Supprimés

**Symptômes :**

- Réduction >30% des points
- Perte de détails de bâtiments
- Bords qui apparaissent érodés

**Solutions :**

```bash
# Paramètres plus relaxés
--preprocess --sor-k 15 --sor-std 3.0 --ror-radius 1.5 --ror-neighbors 3

# Ou désactiver ROR si SOR suffit
--preprocess --sor-k 12 --sor-std 2.5 --ror-radius 999 --ror-neighbors 1
```

### Problème 2 : Artefacts Persistants

**Symptômes :**

- Toujours des motifs de lignes de balayage visibles
- Normales de surface bruitées
- Caractéristiques de bord bruitées

**Solutions :**

```bash
# Paramètres plus stricts
--preprocess --sor-k 10 --sor-std 1.5 --ror-radius 0.8 --ror-neighbors 5

# Ajouter sous-échantillonnage voxel
--preprocess --sor-k 10 --sor-std 1.5 --voxel-size 0.3
```

### Problème 3 : Temps de Traitement Lent

**Symptômes :**

- Augmentation du temps de traitement >50%
- Utilisation élevée du CPU
- Traitement lent des tuiles

**Solutions :**

```bash
# Réduire sor-k
--preprocess --sor-k 8 --sor-std 2.0

# Ajouter voxel pour pré-réduire les points
--preprocess --voxel-size 0.4 --sor-k 10

# Utiliser seulement ROR (plus rapide que SOR)
--preprocess --sor-k 999 --ror-radius 1.0 --ror-neighbors 4
```

### Problème 4 : Problèmes de Mémoire

**Symptômes :**

- Erreurs MemoryError ou "Out of Memory"
- Système devient lent pendant le traitement
- Plantages sur de grandes tuiles

**Solutions :**

```bash
# Ajouter sous-échantillonnage voxel agressif
--preprocess --voxel-size 0.5

# Réduire num-workers
--num-workers 2 --preprocess --voxel-size 0.4

# Utiliser préréglage optimisé mémoire
--preprocess --voxel-size 0.4 --sor-k 10 --sor-std 2.0
```

## Exemples Pratiques

### Exemple 1 : Gratte-ciels Urbains (Paris, Lyon)

**Défi :** Bâtiments hauts et denses, géométrie complexe, haute densité de points

```bash
ign-lidar-hd enrich \
  --input-dir /data/paris_urban/ \
  --output /data/paris_enriched/ \
  --mode full \
  --preprocess \
  --sor-k 12 \
  --sor-std 2.0 \
  --ror-radius 0.8 \
  --ror-neighbors 5 \
  --voxel-size 0.3 \
  --num-workers 8
```

**Résultat :** 35% de réduction de points, normales 50% plus lisses, temps de traitement 20% plus rapide

### Exemple 2 : Villages Ruraux (Densité Faible)

**Défi :** Faible densité de points, petits bâtiments, besoin de préserver les détails

```bash
ign-lidar-hd enrich \
  --input-dir /data/rural_village/ \
  --output /data/rural_enriched/ \
  --mode full \
  --preprocess \
  --sor-k 15 \
  --sor-std 3.0 \
  --ror-radius 1.5 \
  --ror-neighbors 3 \
  --num-workers 4
```

**Résultat :** 12% de réduction de points, détails bien préservés, artefacts modérément réduits

### Exemple 3 : Données LiDAR Anciennes (Bruitées)

**Défi :** Données de 2010, bruit significatif, artefacts de lignes de balayage sévères

```bash
ign-lidar-hd enrich \
  --input-dir /data/old_acquisition/ \
  --output /data/old_cleaned/ \
  --mode full \
  --preprocess \
  --sor-k 10 \
  --sor-std 1.5 \
  --ror-radius 0.8 \
  --ror-neighbors 6 \
  --voxel-size 0.4 \
  --num-workers 6
```

**Résultat :** 45% de réduction de points, 80% de réduction des artefacts, caractéristiques beaucoup plus propres

### Exemple 4 : Traitement en Lot de 100 Tuiles

**Défi :** Traitement à grande échelle, besoin de vitesse et efficacité

```bash
# Script bash pour traitement parallèle
for region in region_*.txt; do
  ign-lidar-hd enrich \
    --input-dir /data/regions/$region/ \
    --output /data/processed/$region/ \
    --mode full \
    --preprocess \
    --voxel-size 0.4 \
    --sor-k 10 \
    --sor-std 2.0 \
    --num-workers 6 &
done
wait

# Ou utiliser GNU parallel
parallel -j 4 "ign-lidar-hd enrich --input-dir {} --output {.}_enriched --mode full --preprocess --voxel-size 0.4" ::: /data/regions/*/
```

**Résultat :** Traitement de 100 tuiles en 4 heures (vs 12 heures sans optimisation)

## Ressources Connexes

- **[Commandes CLI](cli-commands.md)** : Référence complète des commandes
- **[Utilisation de Base](basic-usage.md)** : Guide pour débuter
- **[Guide de Configuration](../reference/configuration.md)** : Configuration YAML
- **[API de Features](../reference/features-api.md)** : Calcul des caractéristiques géométriques
- **[Accélération GPU](gpu-acceleration.md)** : Accélération matérielle pour traitement à grande échelle

## Support

Si vous rencontrez des problèmes ou avez des questions :

1. Consultez la section [Dépannage](#dépannage) ci-dessus
2. Vérifiez les [GitHub Issues](https://github.com/simonpierreboucher/IGN_LIDAR_HD_DATASET/issues)
3. Ouvrez un nouveau ticket avec :
   - Commande complète utilisée
   - Logs de sortie
   - Statistiques de réduction de points
   - Captures d'écran si possible

---

**Version :** 1.7.0  
**Dernière mise à jour :** Décembre 2024
