# Résumé Exécutif : Plan de Réduction des Artefacts

**Date** : 4 Octobre 2025  
**Version** : 1.0  
**Document détaillé** : [ARTIFACT_MITIGATION_PLAN.md](ARTIFACT_MITIGATION_PLAN.md)

---

## 🎯 Objectifs

Réduire les artefacts visuels (lignes, dashs, discontinuités) dans le calcul de features géométriques sur données LiDAR HD IGN en implémentant :

1. **Prétraitement automatique** : Filtrage outliers (SOR/ROR) + homogénéisation densité
2. **Gestion bordures** : Buffer automatique entre tuiles pour continuité
3. **Métriques qualité** : Diagnostic et monitoring automatique des artefacts
4. **Documentation** : Guides utilisateur et bonnes pratiques

---

## 📊 État des Lieux

### ✅ Déjà Implémenté (Points Forts)

- **Recherche par rayon** : `estimate_optimal_radius_for_features()` adapte le rayon selon densité
- **Filtrage features dégénérées** : Validation eigenvalues + masquage invalides
- **Courbure robuste MAD** : Résistance aux outliers via Median Absolute Deviation
- **Support GPU** : Accélération calculs volumétriques (`features_gpu.py`)

### ❌ Manquants (Lacunes)

- **Prétraitement outliers** : Pas de SOR/ROR avant calcul features
- **Gestion bordures** : Discontinuités aux jonctions de tuiles
- **Voxelisation** : Densité hétérogène non homogénéisée
- **Métriques qualité** : Pas de diagnostic automatique artefacts

---

## 🗺 Plan d'Implémentation (4 Sprints)

### Sprint 1 : Module Prétraitement (Priorité HAUTE 🔴)

**Objectif** : Créer `ign_lidar/preprocessing.py` avec filtres robustes

**Fonctionnalités** :

- `statistical_outlier_removal()` : Filtrage SOR (k=12, std=2.0)
- `radius_outlier_removal()` : Filtrage ROR (r=1.0m, min_neighbors=4)
- `voxel_downsample()` : Homogénéisation densité (optionnel)
- `preprocess_point_cloud()` : Pipeline complet configurable

**Livrables** :

- Module `ign_lidar/preprocessing.py` (300 lignes)
- Tests unitaires `tests/test_preprocessing.py`
- Intégration dans `processor.py`

**Durée** : 1 semaine

---

### Sprint 2 : Gestion Bordures (Priorité HAUTE 🔴)

**Objectif** : Éviter discontinuités aux jonctions de tuiles

**Fonctionnalités** :

- `extract_tile_with_buffer()` : Chargement tuile + buffer 50m
- `find_neighbor_tiles()` : Détection auto des 8 tuiles voisines
- Marquage points de bordure (`is_border` mask)

**Livrables** :

- Module `ign_lidar/tile_borders.py` (200 lignes)
- Tests sur grille 3×3 tuiles
- Option CLI `--buffer-distance`

**Durée** : 1 semaine

---

### Sprint 3 : CLI & Configuration (Priorité MOYENNE 🟡)

**Objectif** : Rendre prétraitement accessible et configurable

**Fonctionnalités** :

- Arguments CLI : `--preprocess`, `--sor-k`, `--ror-radius`, `--buffer-distance`
- Config YAML avancée avec section `preprocessing`
- Documentation inline et exemples

**Livrables** :

- CLI étendu dans `cli.py`
- Config exemple `config_examples/pipeline_enrich_advanced.yaml`
- Guide utilisateur

**Durée** : 1 semaine

---

### Sprint 4 : Qualité & Diagnostic (Priorité MOYENNE 🟡)

**Objectif** : Détecter et quantifier artefacts automatiquement

**Fonctionnalités** :

- `compute_feature_quality_metrics()` : Score qualité 0-100
- `detect_scan_line_artifacts()` : Détection patterns via FFT 2D
- Script CLI `diagnose_artifacts.py`
- Monitoring automatique dans pipeline

**Livrables** :

- Module `ign_lidar/quality_metrics.py` (250 lignes)
- Script `scripts/diagnose_artifacts.py`
- Guide troubleshooting

**Durée** : 1 semaine

---

## 🔧 Solutions Techniques par Type d'Artefact

| Artefact                    | Cause                      | Solution                    | Phase |
| --------------------------- | -------------------------- | --------------------------- | ----- |
| **Lignes/dashs**            | kNN sur densité hétérogène | Radius-based (✅) + SOR/ROR | 1     |
| **Discontinuités bordures** | Pas de buffer entre tuiles | Buffer auto 50m             | 2     |
| **Plans mal segmentés**     | Densité irrégulière        | Voxelisation (optionnel)    | 1     |
| **Points isolés**           | Bruit instrumental         | ROR (min_neighbors=4)       | 1     |
| **Normales incohérentes**   | Outliers dans voisinage    | SOR avant PCA               | 1     |
| **Features dégénérées**     | Eigenvalues nulles         | Filtrage existant ✅        | -     |
| **Zones vides**             | Faible couverture scan     | Détection + warning         | 4     |

---

## 📈 Métriques de Succès

### Objectifs Quantitatifs

- **Réduction artefacts** : <5% features dégénérées (vs ~10-15% actuellement)
- **Continuité bordures** : Écart <10cm aux jonctions
- **Qualité globale** : Score moyen >70/100
- **Performance** : Overhead <20% temps calcul

### Tests de Validation

- Dataset test : 100 tuiles diversifiées (urbain/rural/forêt)
- Visualisation : CloudCompare + color map planarity
- Comparaison avant/après LAZ

---

## 💻 Exemples d'Usage

### CLI Simple (Par Défaut)

```bash
# Prétraitement activé par défaut
ign-lidar-hd enrich \
  --input-dir raw/ \
  --output enriched/ \
  --buffer-distance 50
```

### CLI Avancé

```bash
# Configuration personnalisée
ign-lidar-hd enrich \
  --input-dir raw/ \
  --output enriched/ \
  --preprocess \
  --sor-k 12 \
  --sor-std 2.0 \
  --ror-radius 1.0 \
  --ror-min-neighbors 4 \
  --buffer-distance 50 \
  --voxel-size 0.5
```

### Configuration YAML

```yaml
enrich:
  input_dir: "data/raw"
  output: "data/enriched"

  preprocessing:
    enable: true
    statistical_outlier:
      k_neighbors: 12
      std_multiplier: 2.0
    radius_outlier:
      radius: 1.0
      min_neighbors: 4
    voxel_downsampling:
      enable: false
      voxel_size: 0.5

  buffer_distance: 50.0
  quality_check: true
```

### API Python

```python
from ign_lidar import LiDARProcessor

processor = LiDARProcessor(
    enable_preprocessing=True,
    preprocessing_config={
        'sor': {'enable': True, 'k': 12, 'std_multiplier': 2.0},
        'ror': {'enable': True, 'radius': 1.0, 'min_neighbors': 4}
    },
    buffer_distance=50.0,
    use_gpu=True
)

processor.process_tile('tile.laz', 'enriched/')
```

---

## 🚀 Roadmap

### Version 1.7.0 (Q4 2025)

- ✅ Prétraitement SOR/ROR/Voxel
- ✅ Buffer tuiles automatique
- ✅ Métriques qualité

### Version 1.8.0 (Q1 2026)

- Intégration PDAL native
- Support COPC optimisé
- Voxelisation GPU (CuPy)

### Version 2.0.0 (Q2 2026)

- Recalage fin inter-tuiles (ICP)
- Détection ML artefacts
- Dashboard qualité interactif

---

## 📚 Ressources

### Documentation

- **Plan détaillé** : [ARTIFACT_MITIGATION_PLAN.md](ARTIFACT_MITIGATION_PLAN.md)
- **Analyse théorique** : [artifacts.md](artifacts.md)
- **Guide utilisateur** : `website/docs/guides/artifact-mitigation.md` (à créer)

### Outils et Bibliothèques

- [PDAL Filters](https://pdal.io/stages/filters.html)
- [ign-pdal-tools](https://github.com/IGNF/ign-pdal-tools)
- [jakteristics](https://github.com/jakarto3d/jakteristics)

---

## ✅ Prochaines Actions Immédiates

1. **Revue du plan** : Validation approche par équipe
2. **Sprint 1 kickoff** : Début implémentation `preprocessing.py`
3. **Setup tests** : Dataset test 100 tuiles
4. **Documentation** : Template guide utilisateur

---

**Contact** : GitHub Copilot  
**Dernière mise à jour** : 4 Octobre 2025  
**Statut** : ✅ Prêt pour implémentation
