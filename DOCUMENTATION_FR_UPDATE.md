# Documentation Française Mise à Jour - Augmentation Activée par Défaut

## Résumé

Mise à jour de la documentation française Docusaurus pour refléter que **l'augmentation de données est maintenant ACTIVÉE PAR DÉFAUT** dans la commande `ign-lidar-hd enrich`.

## Modifications Effectuées

### 1. **Notes de Version v1.6.0** (`website/i18n/fr/.../release-notes/v1.6.0.md`)

**Avant :**

> L'augmentation des données se produit maintenant pendant la phase ENRICH

**Après :**

> L'augmentation des données se produit maintenant pendant la phase ENRICH. **L'augmentation est maintenant ACTIVÉE PAR DÉFAUT** - chaque dalle crée automatiquement 1 original + 3 versions augmentées pendant le processus d'enrichissement.

### 2. **Guide de Démarrage Rapide** (`website/i18n/fr/.../guides/quick-start.md`)

#### Étape 2 : Enrichissement

**Ajouté :**

- Note que l'augmentation crée des versions augmentées (activée par défaut)
- Bloc d'information proéminent expliquant :
  - 4 versions créées par dalle (1 original + 3 augmentées)
  - Suffixes de fichiers : `nom_dalle.laz`, `nom_dalle_aug1.laz`, etc.
  - Comment désactiver avec `--no-augment`
  - Comment personnaliser avec `--num-augmentations N`

#### Étape 3 : Création de Patches

**Mis à jour :**

- Supprimé les flags `--augment --num-augmentations` de l'exemple
- Ajouté note : "L'augmentation se fait pendant la phase ENRICH (activée par défaut)"
- Expliqué que les dalles enrichies contiennent déjà les versions augmentées

### 3. **Introduction** (`website/i18n/fr/.../intro.md`)

**Ajouté dans les Caractéristiques Principales :**

```markdown
🔄 **Augmentation de Données** - Activée par défaut : transformations géométriques
avant calcul des caractéristiques (v1.6.0+)
```

## Comportement par Défaut (Français)

Lors de l'exécution de :

```bash
ign-lidar-hd enrich --input-dir brut/ --output enrichi/ --mode building
```

**Produit :**

- `nom_dalle.laz` (original)
- `nom_dalle_aug1.laz` (version augmentée 1)
- `nom_dalle_aug2.laz` (version augmentée 2)
- `nom_dalle_aug3.laz` (version augmentée 3)

## Options pour l'Utilisateur

1. **Par défaut (3 augmentées) :** Exécuter sans drapeaux
2. **Désactiver :** Ajouter `--no-augment`
3. **Personnaliser :** Ajouter `--num-augmentations N`

## Fichiers Modifiés

✅ `website/i18n/fr/docusaurus-plugin-content-docs/current/release-notes/v1.6.0.md`
✅ `website/i18n/fr/docusaurus-plugin-content-docs/current/guides/quick-start.md`
✅ `website/i18n/fr/docusaurus-plugin-content-docs/current/intro.md`

## Cohérence de la Documentation

Toute la documentation (anglaise et française) mentionne maintenant de manière cohérente :

- ✅ L'augmentation est **activée par défaut**
- ✅ Crée **4 versions** par dalle (1 original + 3 augmentées)
- ✅ Comment **désactiver** : `--no-augment`
- ✅ Comment **personnaliser** : `--num-augmentations N`

## Messages Clés (Français)

### Bloc d'Information dans le Guide

```markdown
:::info Augmentation de Données (Activée par Défaut)
Par défaut, la commande enrich crée **4 versions** de chaque dalle :

- `nom_dalle.laz` (original)
- `nom_dalle_aug1.laz` (version augmentée 1)
- `nom_dalle_aug2.laz` (version augmentée 2)
- `nom_dalle_aug3.laz` (version augmentée 3)

Chaque version augmentée applique rotation aléatoire, bruit, mise à l'échelle
et suppression de points avant le calcul des caractéristiques.

Pour désactiver : ajoutez `--no-augment`  
Pour changer le nombre : ajoutez `--num-augmentations N`
:::
```

## Impact Utilisateur

Les utilisateurs francophones sont maintenant clairement informés que :

- ✅ L'augmentation est **activée par défaut**
- ✅ Chaque dalle crée **4 fichiers**
- ✅ Comment contrôler ce comportement
- ✅ Pourquoi c'est bénéfique (caractéristiques cohérentes)

---

**Statut :** ✅ Complet  
**Date :** 4 octobre 2025  
**Langues :** 🇬🇧 Anglais + 🇫🇷 Français
