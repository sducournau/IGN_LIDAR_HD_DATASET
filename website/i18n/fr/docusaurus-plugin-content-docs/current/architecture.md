---
sidebar_position: 3
---

# Architecture du Système

Comprendre l'architecture de la bibliothèque vous aide à tirer le meilleur parti de ses capacités et à la personnaliser selon vos besoins spécifiques.

## 🏗️ Architecture de Base

```mermaid
graph TB
    subgraph "Couche Interface Utilisateur"
        CLI[Interface Ligne de Commande]
        API[API Python]
    end

    subgraph "Noyau de Traitement"
        PROC[Processeur LiDAR]
        FEAT[Moteur de Caractéristiques]
        GPU[Accélérateur GPU]
    end

    subgraph "Gestion des Données"
        DOWN[Téléchargeur IGN]
        TILE[Gestionnaire de Dalles]
        META[Stockage Métadonnées]
    end

    subgraph "Couche Classification"
        LOD2[Schéma LOD2<br/>15 Classes]
        LOD3[Schéma LOD3<br/>30+ Classes]
        ARCH[Styles Architecturaux]
    end

    subgraph "Formats de Sortie"
        NPZ[Patches NPZ]
        LAZ[LAZ Enrichi]
        QGIS[Compatible QGIS]
    end

    CLI --> PROC
    API --> PROC
    PROC --> FEAT
    PROC --> DOWN
    FEAT --> GPU
    FEAT --> LOD2
    FEAT --> LOD3
    FEAT --> ARCH
    DOWN --> TILE
    DOWN --> META
    PROC --> NPZ
    PROC --> LAZ
    PROC --> QGIS

    style CLI fill:#e3f2fd
    style API fill:#e3f2fd
    style PROC fill:#fff3e0
    style FEAT fill:#e8f5e8
    style GPU fill:#ffebee
```

## 🔄 Architecture de Flux de Données

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant C as CLI/API
    participant D as Téléchargeur
    participant F as Moteur Caractéristiques
    participant P as Processeur
    participant S as Stockage

    U->>C: Demande de traitement
    C->>D: Télécharger dalles
    D->>D: Vérifier fichiers existants
    D->>S: Stocker LAZ brut
    D-->>C: Dalles disponibles

    C->>F: Enrichir avec caractéristiques
    F->>F: Calculer normales
    F->>F: Extraire courbure
    F->>F: Analyser géométrie
    F->>S: Stocker LAZ enrichi
    F-->>C: Caractéristiques prêtes

    C->>P: Créer patches
    P->>P: Extraire patches
    P->>P: Appliquer augmentation
    P->>P: Assigner labels LOD
    P->>S: Stocker patches NPZ
    P-->>C: Dataset prêt
    C-->>U: Traitement terminé
```

## 🧩 Détails des Composants

### Processeur Principal

La classe `LiDARProcessor` orchestre l'ensemble du pipeline :

- Gère l'exécution du workflow
- Gère le traitement parallèle
- Coordonne la détection de saut intelligent
- Applique l'augmentation de données

### Moteur de Caractéristiques

Analyse géométrique avancée :

- Calcul des normales de surface
- Calcul de la courbure principale
- Mesures de planarité et verticalité
- Estimation de la densité locale
- Inférence du style architectural

### Système de Saut Intelligent

Reprise intelligente du workflow :

- Vérification de l'existence des fichiers
- Validation des métadonnées
- Comparaison des horodatages
- Suivi de la progression

### Accélération GPU (Nouveau en v1.5.0, Corrigé en v1.6.2)

Accélération CUDA optionnelle pour :

- Recherches k plus proches voisins
- Opérations matricielles
- Calculs de caractéristiques (formules corrigées en v1.6.2)
- **Interpolation de couleurs RGB (24x plus rapide)** 🆕
- **Mise en cache mémoire GPU pour dalles RGB** 🆕
- Traitement de grands jeux de données

:::tip En Savoir Plus
Voir le [Guide d'Accélération GPU](gpu/overview.md) pour les instructions complètes et le [Guide RGB GPU](gpu/rgb-augmentation.md) pour les détails spécifiques RGB.
:::

#### Pipeline RGB GPU

```mermaid
flowchart LR
    A[Points] --> B[Transfert GPU]
    B --> C[Caractéristiques GPU]
    C --> D[Cache RGB GPU]
    D --> E[Interpolation Couleur GPU]
    E --> F[Résultats Combinés]
    F --> G[Transfert CPU]

    style B fill:#c8e6c9
    style C fill:#c8e6c9
    style D fill:#c8e6c9
    style E fill:#c8e6c9
    style F fill:#c8e6c9
```

**Performance :** Accélération 24x pour l'augmentation RGB (v1.5.0)

## 📊 Caractéristiques de Performance

```mermaid
graph LR
    subgraph "Vitesse de Traitement"
        CPU[Mode CPU<br/>~1-2 dalles/min]
        GPU_ACC[Mode GPU<br/>~5-10 dalles/min]
    end

    subgraph "Utilisation Mémoire"
        SMALL[Petites Dalles<br/>~512MB RAM]
        LARGE[Grandes Dalles<br/>~2-4GB RAM]
    end

    subgraph "Taille de Sortie"
        INPUT[LAZ Brut<br/>~50-200MB]
        OUTPUT[LAZ Enrichi<br/>~80-300MB]
        PATCHES[Patches NPZ<br/>~10-50MB chacun]
    end

    style GPU_ACC fill:#e8f5e8
    style LARGE fill:#fff3e0
    style OUTPUT fill:#e3f2fd
```

## 🔧 Système de Configuration

La bibliothèque utilise une approche de configuration hiérarchique :

1. **Paramètres par Défaut** - Valeurs optimales intégrées
2. **Fichiers de Configuration** - Paramètres spécifiques au projet
3. **Variables d'Environnement** - Remplacements à l'exécution
4. **Arguments de Commande** - Paramètres immédiats

### Options de Configuration Clés

| Catégorie   | Options                          | Impact                         |
| ----------- | -------------------------------- | ------------------------------ |
| Performance | `num_workers`, `use_gpu`         | Vitesse de traitement          |
| Qualité     | `k_neighbors`, `patch_size`      | Précision des caractéristiques |
| Sortie      | `lod_level`, `format_preference` | Caractéristiques du dataset    |
| Workflow    | `skip_existing`, `force`         | Comportement de reprise        |

## 🚀 Points d'Extension

L'architecture supporte la personnalisation via :

- **Extracteurs de Caractéristiques Personnalisés** - Ajouter des caractéristiques spécifiques au domaine
- **Schémas de Classification** - Définir de nouveaux niveaux LOD
- **Formats de Sortie** - Supporter des formats de fichiers supplémentaires
- **Hooks de Traitement** - Insérer des étapes de traitement personnalisées
- **Règles de Validation** - Ajouter des vérifications de qualité

Cette conception modulaire garantit que la bibliothèque peut s'adapter à diverses exigences de recherche et de production tout en maintenant performance et fiabilité.
