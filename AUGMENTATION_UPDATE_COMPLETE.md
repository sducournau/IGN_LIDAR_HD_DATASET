# Mise à Jour Complète : Augmentation Activée par Défaut

## Vue d'Ensemble

Mise à jour complète du code et de la documentation (anglais et français) pour refléter que **l'augmentation de données est maintenant ACTIVÉE PAR DÉFAUT** dans la commande `ign-lidar-hd enrich`.

## ✅ Modifications du Code

### CLI (`ign_lidar/cli.py`)

- ✅ Ajout du drapeau `--no-augment` pour désactivation explicite
- ✅ Texte d'aide mis à jour : "(default: enabled)"
- ✅ Les deux drapeaux disponibles : `--augment` et `--no-augment`

## ✅ Documentation Anglaise

### README.md

- ✅ Exemples de commandes mis à jour avec note sur le comportement par défaut
- ✅ Exemple `--no-augment` ajouté
- ✅ Note v1.6.0 mise à jour avec détails du comportement par défaut
- ✅ Section caractéristiques mise à jour

### Docusaurus (Anglais)

- ✅ `website/docs/intro.md` - Ajout "(enabled by default)"
- ✅ `website/docs/guides/quick-start.md` - Bloc d'info proéminent + exemples mis à jour
- ✅ `website/docs/release-notes/v1.6.0.md` - Emphase "ENABLED BY DEFAULT"

### Documentation d'Implémentation

- ✅ `AUGMENTATION_IMPLEMENTATION.md` - Notice de comportement par défaut
- ✅ `AUGMENTATION_DEFAULT_ENABLED.md` - Document récapitulatif créé

## ✅ Documentation Française

### Docusaurus (Français)

- ✅ `website/i18n/fr/.../intro.md` - Caractéristique ajoutée
- ✅ `website/i18n/fr/.../guides/quick-start.md` - Bloc d'info + exemples
- ✅ `website/i18n/fr/.../release-notes/v1.6.0.md` - "ACTIVÉE PAR DÉFAUT"

### Documentation Récapitulative

- ✅ `DOCUMENTATION_FR_UPDATE.md` - Résumé des modifications françaises

## 📋 Comportement par Défaut

### Commande

```bash
ign-lidar-hd enrich --input-dir raw/ --output enriched/ --mode building
```

### Sortie

```
enriched/
├── tile_name.laz          # Original
├── tile_name_aug1.laz     # Augmentée 1
├── tile_name_aug2.laz     # Augmentée 2
└── tile_name_aug3.laz     # Augmentée 3
```

### Transformations Appliquées

Chaque version augmentée reçoit :

1. **Rotation aléatoire** (0-360° autour de l'axe Z)
2. **Bruit** (Gaussien, σ=0.1m)
3. **Mise à l'échelle** (0.95-1.05)
4. **Suppression de points** (5-15%)

## 🎮 Contrôle Utilisateur

| Action                              | Commande                                                  |
| ----------------------------------- | --------------------------------------------------------- |
| **Par défaut** (3 augmentations)    | `ign-lidar-hd enrich --input-dir raw/ --output enriched/` |
| **Désactiver**                      | `... --no-augment`                                        |
| **Personnaliser** (5 augmentations) | `... --num-augmentations 5`                               |

## 📚 Fichiers Modifiés

### Code

- `ign_lidar/cli.py`

### Documentation Anglaise

- `README.md`
- `AUGMENTATION_IMPLEMENTATION.md`
- `AUGMENTATION_DEFAULT_ENABLED.md` (nouveau)
- `website/docs/intro.md`
- `website/docs/guides/quick-start.md`
- `website/docs/release-notes/v1.6.0.md`

### Documentation Française

- `DOCUMENTATION_FR_UPDATE.md` (nouveau)
- `website/i18n/fr/docusaurus-plugin-content-docs/current/intro.md`
- `website/i18n/fr/docusaurus-plugin-content-docs/current/guides/quick-start.md`
- `website/i18n/fr/docusaurus-plugin-content-docs/current/release-notes/v1.6.0.md`

## 🎯 Messages Clés

### Anglais

> **Augmentation is ENABLED by default** - each tile produces 1 original + 3 augmented versions (configurable with `--num-augmentations` or disable with `--no-augment`)

### Français

> **L'augmentation est ACTIVÉE PAR DÉFAUT** - chaque dalle crée automatiquement 1 original + 3 versions augmentées (configurable avec `--num-augmentations` ou désactiver avec `--no-augment`)

## ✅ Vérification CLI

```bash
$ ign-lidar-hd enrich --help | grep -A3 augment
  --augment             Enable geometric data augmentation (default: enabled)
  --no-augment          Disable geometric data augmentation
  --num-augmentations NUM_AUGMENTATIONS
                        Number of augmented versions per tile (default: 3)
```

## 💡 Avantages

1. **Meilleur entraînement ML** - Plus de données diversifiées par défaut
2. **Cohérence des caractéristiques** - Calculées sur géométrie augmentée
3. **Commodité** - Pas besoin de se rappeler d'ajouter `--augment`
4. **Bonnes pratiques** - Encourage l'utilisation de l'augmentation

## 📊 Impact

### Avant

- Utilisateurs pouvaient oublier `--augment`
- Pas d'indication claire sur le comportement par défaut
- Documentation incohérente

### Après

- ✅ Augmentation activée automatiquement
- ✅ Documentation claire et cohérente (EN + FR)
- ✅ Options de contrôle explicites (`--no-augment`)
- ✅ Exemples mis à jour dans tous les guides

---

**Statut :** ✅ Complet  
**Date :** 4 octobre 2025  
**Langues :** 🇬🇧 Anglais + 🇫🇷 Français  
**Impact :** Code + Documentation complète
