---
sidebar_position: 1
---

# Fonctionnalités Personnalisées

Apprenez à créer et intégrer des extracteurs de fonctionnalités personnalisés pour des tâches spécialisées de traitement LiDAR.

## Vue d'Ensemble

La bibliothèque IGN LiDAR HD fournit un framework flexible pour implémenter des algorithmes d'extraction de fonctionnalités personnalisés. Ce tutoriel montre comment créer vos propres extracteurs de fonctionnalités.

## Création de Fonctionnalités Personnalisées

### Extracteur de Fonctionnalités de Base

Créez une classe de fonctionnalité personnalisée en héritant de la classe de base FeatureExtractor :

```python
from ign_lidar.features import FeatureExtractor
import numpy as np

class CustomFeature(FeatureExtractor):
    def __init__(self, radius=2.0):
        super().__init__(name="custom_feature", radius=radius)

    def extract(self, points, neighborhoods):
        """Extraire une fonctionnalité personnalisée des voisinages de points."""
        features = []

        for neighbors in neighborhoods:
            # Implémentez votre calcul de fonctionnalité personnalisée
            feature_value = self._calculate_feature(neighbors)
            features.append(feature_value)

        return np.array(features)

    def _calculate_feature(self, neighbors):
        """Implémentez votre calcul personnalisé ici."""
        # Exemple : variance de distance
        if len(neighbors) < 3:
            return 0.0

        distances = np.linalg.norm(neighbors - neighbors.mean(axis=0), axis=1)
        return np.var(distances)
```

### Fonctionnalité Avancée avec Paramètres

Créez des fonctionnalités plus sophistiquées avec des paramètres configurables :

```python
class AdvancedFeature(FeatureExtractor):
    def __init__(self, radius=2.0, min_points=10, weight_function="inverse"):
        super().__init__(name="advanced_feature", radius=radius)
        self.min_points = min_points
        self.weight_function = weight_function

    def extract(self, points, neighborhoods):
        features = []

        for i, neighbors in enumerate(neighborhoods):
            if len(neighbors) < self.min_points:
                features.append(0.0)
                continue

            # Appliquer une pondération basée sur la distance
            center = points[i]
            distances = np.linalg.norm(neighbors - center, axis=1)
            weights = self._compute_weights(distances)

            # Calcul de fonctionnalité pondérée
            weighted_values = self._compute_weighted_values(neighbors, weights)
            features.append(weighted_values)

        return np.array(features)

    def _compute_weights(self, distances):
        if self.weight_function == "inverse":
            return 1.0 / (distances + 1e-6)
        elif self.weight_function == "gaussian":
            sigma = self.radius / 3.0
            return np.exp(-distances**2 / (2 * sigma**2))
        else:
            return np.ones_like(distances)
```

## Enregistrement de Fonctionnalités Personnalisées

### Enregistrement de Fonctionnalité Unique

Enregistrez votre fonctionnalité personnalisée avec le processeur :

```python
from ign_lidar import Processor, Config

# Créer un processeur avec des fonctionnalités personnalisées
processor = Processor()

# Enregistrer une fonctionnalité personnalisée
custom_feature = CustomFeature(radius=3.0)
processor.register_feature(custom_feature)

# Utiliser dans le traitement
config = Config(
    feature_types=["height_above_ground", "custom_feature"],
    feature_radius=3.0
)

result = processor.process_tile("input.las", config=config)
```

### Enregistrement de Fonctionnalités par Lots

Enregistrez plusieurs fonctionnalités personnalisées :

```python
# Créer plusieurs fonctionnalités personnalisées
features = [
    CustomFeature(radius=2.0),
    AdvancedFeature(radius=3.0, min_points=15),
    CustomFeature(radius=1.0)  # Paramètres différents
]

# Enregistrer toutes les fonctionnalités
for feature in features:
    processor.register_feature(feature)

# Configurer le traitement
config = Config(
    feature_types=["custom_feature", "advanced_feature"],
    enable_gpu=True  # Accélération GPU pour les fonctionnalités personnalisées
)
```

## Combinaison de Fonctionnalités

### Combiner Plusieurs Fonctionnalités

Créez des fonctionnalités composites qui combinent plusieurs calculs :

```python
class CompositeFeature(FeatureExtractor):
    def __init__(self, radius=2.0):
        super().__init__(name="composite_feature", radius=radius)

        # Initialiser les sous-fonctionnalités
        self.geometric_feature = CustomFeature(radius)
        self.intensity_feature = IntensityFeature(radius)

    def extract(self, points, neighborhoods):
        # Extraire les fonctionnalités individuelles
        geometric = self.geometric_feature.extract(points, neighborhoods)
        intensity = self.intensity_feature.extract(points, neighborhoods)

        # Combiner les fonctionnalités (exemple : somme pondérée)
        weights = [0.7, 0.3]
        combined = weights[0] * geometric + weights[1] * intensity

        return combined
```

## Optimisation des Performances

### Fonctionnalités Accélérées par GPU

Implémentez l'accélération GPU pour les fonctionnalités personnalisées :

```python
try:
    import cupy as cp

    class GPUFeature(FeatureExtractor):
        def __init__(self, radius=2.0):
            super().__init__(name="gpu_feature", radius=radius)
            self.use_gpu = cp.cuda.is_available()

        def extract(self, points, neighborhoods):
            if self.use_gpu:
                return self._extract_gpu(points, neighborhoods)
            else:
                return self._extract_cpu(points, neighborhoods)

        def _extract_gpu(self, points, neighborhoods):
            # Implémentation GPU utilisant CuPy
            gpu_points = cp.asarray(points)
            # ... calculs accélérés par GPU
            return cp.asnumpy(result)

        def _extract_cpu(self, points, neighborhoods):
            # Implémentation CPU de secours
            pass

except ImportError:
    print("Accélération GPU non disponible")
```

## Test des Fonctionnalités Personnalisées

### Tests Unitaires

Créez des tests pour vos fonctionnalités personnalisées :

```python
import unittest
from ign_lidar.test_utils import generate_test_points

class TestCustomFeature(unittest.TestCase):
    def setUp(self):
        self.feature = CustomFeature(radius=2.0)
        self.test_points = generate_test_points(1000)

    def test_feature_shape(self):
        """Tester que la sortie de la fonctionnalité a la forme correcte."""
        neighborhoods = self.feature.get_neighborhoods(self.test_points)
        features = self.feature.extract(self.test_points, neighborhoods)

        self.assertEqual(len(features), len(self.test_points))

    def test_feature_range(self):
        """Tester que les valeurs de fonctionnalité sont dans la plage attendue."""
        neighborhoods = self.feature.get_neighborhoods(self.test_points)
        features = self.feature.extract(self.test_points, neighborhoods)

        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(np.isfinite(features)))
```

## Bonnes Pratiques

1. **Normalisation** : Toujours normaliser les valeurs de fonctionnalités dans la plage [0,1] ou [-1,1]
2. **Gestion des Erreurs** : Gérer les cas limites (voisinages vides, valeurs NaN)
3. **Documentation** : Fournir des docstrings claires expliquant la signification des fonctionnalités
4. **Tests** : Écrire des tests unitaires complets
5. **Performance** : Considérer l'implémentation GPU pour les fonctionnalités intensives en calcul
