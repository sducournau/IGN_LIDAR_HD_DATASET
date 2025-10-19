# 🛣️ Améliorations Classification Routes - Résumé

## Problèmes Corrigés

### ❌ Avant

- Végétation (arbres) classée comme route
- Parties de bâtiments classées comme route
- Débordement des polygons BD TOPO

### ✅ Après

- Filtrage NDVI : exclut végétation (NDVI > 0.20)
- Filtrage courbure : exclut surfaces complexes (> 0.05)
- Filtrage verticalité : exclut murs (> 0.30)
- Hauteur réduite : 1.5m max (exclut arbres)

## Nouveaux Filtres

### 1. NDVI (Végétation)

```python
ROAD_NDVI_MAX = 0.20  # Végétation au-dessus
```

### 2. Courbure (Surface)

```python
ROAD_CURVATURE_MAX = 0.05  # Feuillage au-dessus
```

### 3. Verticalité (Structure)

```python
ROAD_VERTICALITY_MAX = 0.30  # Murs au-dessus
```

### 4. Hauteur (Élévation)

```python
ROAD_HEIGHT_MAX = 1.5m  # Réduit de 2.0m
```

### 5. Planarite (Surface)

```python
ROAD_PLANARITY_MIN = 0.7  # Augmenté de 0.6
```

## Ordre des Filtres

1. Protection classifications existantes
2. Filtre NDVI (végétation)
3. Filtre courbure (complexité)
4. Filtre verticalité (orientation)
5. Filtre hauteur (élévation)
6. Validation géométrique

## Fichiers Modifiés

- `classification_thresholds.py` - nouveaux seuils
- `classification_refinement.py` - logique de filtrage

## Test

Commande pour tester:

```bash
ign-lidar-hd process \
  -c "examples/config_asprs_bdtopo_cadastre_optimized.yaml" \
  input_dir="/mnt/d/ign/versailles/" \
  output_dir="/mnt/d/ign/versailles"
```

## Résultats Attendus

- 🌳 Arbres → Classe végétation (pas route)
- 🏠 Bâtiments → Classe bâtiment (pas route)
- 🛣️ Routes → Surfaces planes horizontales uniquement
