# Extensions Knowledge Base - Nouveaux Articles 2025

> Sections supplémentaires basées sur les nouveaux articles de Florent Poux

## 🧩 Clustering & Segmentation Non-Supervisée

### Théorie des Graphes pour Clustering 3D

#### Workflow Complet (Graph Theory)

```
Point Cloud → Graph Generation → Cluster Identification
```

**Étapes clés**

1. **Construction du graphe de connectivité**

   ```python
   import networkx as nx
   from scipy.spatial import cKDTree

   # Construire KD-tree pour recherche de voisins
   tree = cKDTree(points)

   # Créer graphe avec arêtes entre voisins proches
   G = nx.Graph()
   for i, point in enumerate(points):
       neighbors = tree.query_ball_point(point, r=radius)
       for neighbor in neighbors:
           if i != neighbor:
               G.add_edge(i, neighbor)
   ```

2. **Identification des composantes connexes**

   ```python
   # Extraire clusters (composantes connexes)
   clusters = list(nx.connected_components(G))

   # Chaque cluster = ensemble d'indices de points
   for cluster_id, cluster_indices in enumerate(clusters):
       cluster_points = points[list(cluster_indices)]
       # Traiter chaque cluster...
   ```

3. **Filtrage et post-traitement**
   ```python
   # Supprimer petits clusters (bruit)
   min_cluster_size = 50
   filtered_clusters = [c for c in clusters if len(c) >= min_cluster_size]
   ```

#### Comparaison Méthodes Clustering

| Méthode          | Type         | Complexité | Avantages                     | Inconvénients             |
| ---------------- | ------------ | ---------- | ----------------------------- | ------------------------- |
| **K-Means**      | Centroid     | O(n·k·i)   | Rapide, simple                | K fixe, formes sphériques |
| **DBSCAN**       | Density      | O(n log n) | Détecte outliers, K adaptatif | Sensible aux paramètres   |
| **Graph-based**  | Connectivity | O(n log n) | Précis, topologie             | Plus lent                 |
| **Hierarchical** | Tree         | O(n²)      | Pas de K, dendrogramme        | Lent, mémoire             |

#### Applications Pratiques

**Segmentation d'objets en intérieur**

```python
# Scénario : Bibliothèque avec meubles
# Objectif : Isoler chaises, tables, lampes

# 1. Supprimer le sol (plan RANSAC)
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.02,
    ransac_n=3,
    num_iterations=1000
)
pcd_no_floor = pcd.select_by_index(inliers, invert=True)

# 2. Clustering euclidien avec Graph Theory
from scipy.spatial import cKDTree
import networkx as nx

# Construire graphe de connectivité
tree = cKDTree(np.asarray(pcd_no_floor.points))
G = nx.Graph()
for i in range(len(pcd_no_floor.points)):
    neighbors = tree.query_ball_point(pcd_no_floor.points[i], r=0.05)
    for neighbor in neighbors:
        if i < neighbor:  # Éviter doublons
            G.add_edge(i, neighbor)

# Extraire composantes connexes
clusters = list(nx.connected_components(G))

# 3. Analyse par cluster
for cluster_id, cluster_indices in enumerate(clusters):
    cluster_points = np.asarray(pcd_no_floor.points)[list(cluster_indices)]

    # Bounding box
    min_bound = cluster_points.min(axis=0)
    max_bound = cluster_points.max(axis=0)
    extent = max_bound - min_bound

    volume = np.prod(extent)
    height = extent[2]

    # Classification basée sur géométrie
    if height < 0.5:
        label = "table"
    elif height > 1.0 and volume < 0.2:
        label = "lamp"
    else:
        label = "chair"

    print(f"Cluster {cluster_id}: {label} (volume={volume:.2f}m³, height={height:.2f}m)")
```

---

## 🔍 Segment Anything 3D (SAM 3D)

### Adaptation de SAM pour Nuages de Points

#### Principe

**SAM (Segment Anything Model)** est conçu pour images 2D. Pour l'appliquer aux nuages 3D :

1. **Projection 3D → 2D**
2. **Segmentation 2D avec SAM**
3. **Remontée 2D → 3D** (back-projection)

#### Pipeline Complet

```python
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import open3d as o3d

# 1. Charger point cloud
pcd = o3d.io.read_point_cloud("scene.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 2. Projection orthographique (vue de dessus)
def project_to_2d(points, colors, resolution=0.01):
    """
    Projette nuage 3D en image 2D (vue de dessus)

    Args:
        points: Array [N, 3] de coordonnées XYZ
        colors: Array [N, 3] de couleurs RGB [0, 1]
        resolution: Taille pixel en mètres

    Returns:
        image: Image 2D [H, W, 3]
        depth_map: Carte de profondeur [H, W]
        mapping: Correspondance pixel → indices points 3D
    """
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    grid_x = int((max_bound[0] - min_bound[0]) / resolution) + 1
    grid_y = int((max_bound[1] - min_bound[1]) / resolution) + 1

    image = np.zeros((grid_y, grid_x, 3))
    depth_map = np.full((grid_y, grid_x), -np.inf)
    mapping = {}  # (y, x) → point_id

    for i, (point, color) in enumerate(zip(points, colors)):
        x_idx = int((point[0] - min_bound[0]) / resolution)
        y_idx = int((point[1] - min_bound[1]) / resolution)

        if 0 <= x_idx < grid_x and 0 <= y_idx < grid_y:
            # Garder point le plus haut (Z max)
            if point[2] > depth_map[y_idx, x_idx]:
                depth_map[y_idx, x_idx] = point[2]
                image[y_idx, x_idx] = color
                mapping[(y_idx, x_idx)] = i

    return image, depth_map, mapping

image_2d, depth_map, pixel_to_point = project_to_2d(points, colors)

# 3. Appliquer SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

predictor.set_image((image_2d * 255).astype(np.uint8))

# Segmentation automatique
masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    multimask_output=True
)

# 4. Back-projection 3D
def backproject_masks_to_3d(masks, points, pixel_to_point, min_bound, resolution):
    """
    Remonte masques 2D vers labels 3D
    """
    labels_3d = np.zeros(len(points), dtype=int)

    for i, point in enumerate(points):
        x_idx = int((point[0] - min_bound[0]) / resolution)
        y_idx = int((point[1] - min_bound[1]) / resolution)

        # Trouver masque actif pour ce pixel
        for mask_id, mask in enumerate(masks):
            if 0 <= y_idx < mask.shape[0] and 0 <= x_idx < mask.shape[1]:
                if mask[y_idx, x_idx]:
                    labels_3d[i] = mask_id + 1
                    break

    return labels_3d

labels = backproject_masks_to_3d(
    masks, points, pixel_to_point,
    points.min(axis=0), resolution=0.01
)

# Visualiser résultats
pcd.colors = o3d.utility.Vector3dVector(
    plt.cm.tab20(labels % 20)[:, :3]  # Colormap
)
o3d.visualization.draw_geometries([pcd])
```

#### Avantages et Limites

**✅ Avantages**

- Utilise modèle pré-entraîné puissant (ViT)
- Zero-shot segmentation (pas de réentraînement)
- Segmentation interactive possible
- Rapide sur GPU

**❌ Limites**

- Perte d'information lors projection 2D
- Occlusions non gérées
- Objets verticaux mal segmentés (vue de dessus)
- Multi-vues nécessaires pour scènes complexes

**Solution : Multi-vues avec fusion**

```python
def sam_3d_multiview(points, colors, views=['top', 'front', 'side']):
    """
    SAM 3D avec fusion multi-vues
    """
    all_labels = []

    for view in views:
        # Rotation points selon vue
        if view == 'top':
            pts_view = points.copy()
        elif view == 'front':
            R = rotation_matrix_x(np.pi/2)
            pts_view = points @ R.T
        elif view == 'side':
            R = rotation_matrix_y(np.pi/2)
            pts_view = points @ R.T

        # Projection et segmentation
        image_2d, depth_map, mapping = project_to_2d(pts_view, colors)
        masks = sam_segment(image_2d)
        labels_3d = backproject_masks(masks, pts_view, mapping)

        all_labels.append(labels_3d)

    # Fusion par vote majoritaire
    final_labels = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=np.vstack(all_labels)
    )

    return final_labels
```

---

## 🌳 Scene Graphs pour Spatial AI

### Graphes de Scène 3D pour LLMs

#### Motivation

Les **Scene Graphs** formalisent les relations spatiales pour permettre aux LLMs de comprendre les scènes 3D.

```
Scene Graph = {Objects, Relationships, Attributes}
```

**Exemple de représentation**

```
Table (brown, wooden, 1.2m x 0.8m)
  ├─ supports → Laptop (silver, Dell)
  ├─ supports → Cup (white, ceramic)
  └─ near → Chair (black, office)

Floor (concrete, grey)
  ├─ supports → Table
  └─ supports → Chair

Wall (white, plaster)
  └─ adjacent_to → Floor
```

#### Construction Automatique avec NetworkX

```python
import networkx as nx
import numpy as np

def build_scene_graph_from_clusters(clusters, labels):
    """
    Construit scene graph à partir de clusters segmentés

    Args:
        clusters: Liste de point clouds (clusters)
        labels: Liste de labels d'objets

    Returns:
        G: NetworkX DiGraph (scene graph)
    """
    G = nx.DiGraph()

    # 1. Ajouter objets (nœuds)
    for i, (cluster, label) in enumerate(zip(clusters, labels)):
        # Calculer attributs géométriques
        points = np.asarray(cluster.points)
        bbox = cluster.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()

        # Couleur dominante
        colors = np.asarray(cluster.colors)
        dominant_color = colors.mean(axis=0)
        color_name = rgb_to_color_name(dominant_color)

        # Ajouter nœud avec attributs
        G.add_node(
            f"{label}_{i}",
            type=label,
            color=color_name,
            volume=bbox.volume(),
            height=extent[2],
            centroid=bbox.get_center().tolist()
        )

    # 2. Calculer relations spatiales
    nodes = list(G.nodes())
    for i, node_i in enumerate(nodes):
        attrs_i = G.nodes[node_i]
        centroid_i = np.array(attrs_i['centroid'])

        for j, node_j in enumerate(nodes):
            if i == j:
                continue

            attrs_j = G.nodes[node_j]
            centroid_j = np.array(attrs_j['centroid'])

            # Relation "supports" (objet i sous objet j)
            if (centroid_j[2] > centroid_i[2] + attrs_i['height'] * 0.9 and
                np.linalg.norm(centroid_i[:2] - centroid_j[:2]) < 0.5):
                G.add_edge(node_i, node_j, relation="supports", confidence=0.95)

            # Relation "on" (objet i sur objet j)
            elif (centroid_i[2] > centroid_j[2] + attrs_j['height'] * 0.9 and
                  np.linalg.norm(centroid_i[:2] - centroid_j[:2]) < 0.5):
                G.add_edge(node_i, node_j, relation="on", confidence=0.95)

            # Relation "near" (proximité horizontale)
            elif np.linalg.norm(centroid_i[:2] - centroid_j[:2]) < 1.5:
                G.add_edge(node_i, node_j, relation="near", confidence=0.8)

            # Relation "adjacent_to" (murs, plans verticaux)
            elif attrs_i['type'] in ['wall', 'door'] and attrs_j['type'] in ['wall', 'floor']:
                G.add_edge(node_i, node_j, relation="adjacent_to", confidence=0.9)

    return G

# Utilisation
scene_graph = build_scene_graph_from_clusters(segmented_clusters, object_labels)
```

#### Export OpenUSD pour Visualisation

```python
from pxr import Usd, UsdGeom, Sdf

def export_scene_graph_to_usd(scene_graph, output_path="scene.usda"):
    """
    Export scene graph vers format OpenUSD
    """
    # Créer stage USD
    stage = Usd.Stage.CreateNew(output_path)

    # Définir unités (mètres)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Créer racine de scène
    scene_root = UsdGeom.Xform.Define(stage, "/Scene")

    # Ajouter chaque objet
    for node in scene_graph.nodes():
        attrs = scene_graph.nodes[node]

        # Créer primitive USD
        prim_path = f"/Scene/{node}"
        xform = UsdGeom.Xform.Define(stage, prim_path)

        # Position (centroid)
        centroid = attrs['centroid']
        xform.AddTranslateOp().Set(tuple(centroid))

        # Ajouter attributs custom
        prim = stage.GetPrimAtPath(prim_path)
        prim.SetCustomDataByKey("object_type", attrs['type'])
        prim.SetCustomDataByKey("color", attrs['color'])
        prim.SetCustomDataByKey("volume", attrs['volume'])
        prim.SetCustomDataByKey("height", attrs['height'])

        # Ajouter relations
        relationships = []
        for _, target, edge_data in scene_graph.out_edges(node, data=True):
            relationships.append({
                "target": target,
                "relation": edge_data['relation'],
                "confidence": edge_data['confidence']
            })

        if relationships:
            prim.SetCustomDataByKey("relationships", str(relationships))

    # Sauvegarder
    stage.Save()
    print(f"Scene graph exported to {output_path}")

# Export
export_scene_graph_to_usd(scene_graph, "my_scene.usda")
```

#### Intégration LLM pour Requêtes Spatiales

```python
from openai import OpenAI

def scene_graph_to_natural_language(G):
    """
    Convertit scene graph en description textuelle
    """
    description = "# Scene Description\n\n## Objects:\n"

    for node in G.nodes():
        attrs = G.nodes[node]
        description += f"- **{node}**: a {attrs['color']} {attrs['type']}"
        description += f" ({attrs['height']:.2f}m high, volume {attrs['volume']:.2f}m³)\n"

    description += "\n## Spatial Relationships:\n"
    for u, v, data in G.edges(data=True):
        relation = data['relation']
        confidence = data.get('confidence', 1.0)
        description += f"- {u} **{relation}** {v} (confidence: {confidence:.2%})\n"

    return description

def query_scene_with_llm(scene_graph, question, model="gpt-4"):
    """
    Interroge scène 3D via LLM
    """
    client = OpenAI()

    scene_description = scene_graph_to_natural_language(scene_graph)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an AI that understands 3D spatial scenes. "
                          "Answer questions based on the scene graph provided."
            },
            {
                "role": "user",
                "content": f"{scene_description}\n\nQuestion: {question}"
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# Exemples de requêtes
questions = [
    "Where is the laptop?",
    "What objects are on the table?",
    "How many chairs are in the scene?",
    "Is there anything near the door?"
]

for q in questions:
    answer = query_scene_with_llm(scene_graph, q)
    print(f"Q: {q}\nA: {answer}\n")
```

**Exemple de sortie**

```
Q: Where is the laptop?
A: The laptop (silver) is on the brown wooden table.

Q: What objects are on the table?
A: The table supports two objects: a silver laptop and a white ceramic cup.

Q: How many chairs are in the scene?
A: There is 1 chair in the scene (a black office chair near the table).
```

---

## 🔄 Change Detection 3D

### Détection de Changements dans Nuages de Points

#### Scénarios d'Application

1. **Surveillance d'infrastructure** : Détecter éléments manquants (poutres, tuyaux, câbles)
2. **Chantiers BIM** : Vérifier conformité as-built vs as-designed
3. **Monitoring environnemental** : Évolution de végétation, érosion, glissements de terrain
4. **Sécurité** : Détecter intrusions, modifications non autorisées

#### Méthode Cloud-to-Cloud (C2C)

**Principe** : Calculer distance entre chaque point et son plus proche voisin dans l'autre nuage.

```python
import open3d as o3d
import numpy as np

def cloud_to_cloud_change_detection(pcd_ref, pcd_new, threshold=0.10):
    """
    Change detection par distances C2C

    Args:
        pcd_ref: Point cloud de référence (temps t0)
        pcd_new: Point cloud nouveau (temps t1)
        threshold: Seuil de détection (en mètres)

    Returns:
        distances: Distances C2C pour chaque point
        changed_indices: Indices des points ayant changé
    """
    # 1. Alignement ICP (si nécessaire)
    threshold_icp = 0.02
    trans_init = np.eye(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_new, pcd_ref, threshold_icp, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    pcd_new_aligned = pcd_new.transform(reg_p2p.transformation)
    print(f"ICP fitness: {reg_p2p.fitness:.4f}")

    # 2. Calcul distances C2C
    tree_ref = o3d.geometry.KDTreeFlann(pcd_ref)
    distances = []

    for point in pcd_new_aligned.points:
        [_, idx, dist] = tree_ref.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(dist[0]))

    distances = np.array(distances)

    # 3. Détection changements
    changed_indices = np.where(distances > threshold)[0]

    print(f"Points changed: {len(changed_indices)} / {len(distances)} "
          f"({len(changed_indices)/len(distances)*100:.1f}%)")

    return distances, changed_indices, pcd_new_aligned

# Utilisation
pcd_t0 = o3d.io.read_point_cloud("scan_time0.ply")
pcd_t1 = o3d.io.read_point_cloud("scan_time1.ply")

distances, changed_idx, pcd_aligned = cloud_to_cloud_change_detection(
    pcd_t0, pcd_t1, threshold=0.10
)

# Visualisation avec colormap
import matplotlib.pyplot as plt

colors = plt.cm.RdYlGn_r(np.clip(distances / 0.5, 0, 1))[:, :3]
pcd_aligned.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd_aligned], window_name="C2C Distances")
```

#### Méthode Multi-Scale Model-to-Model (M3C2)

**Plus robuste** : Prend en compte la géométrie locale (normales) et utilise projection cylindrique.

```python
def m3c2_change_detection(pcd_ref, pcd_new, normal_scale=0.5, projection_scale=2.0, threshold=0.15):
    """
    M3C2 : Multi-scale Model to Model Cloud Comparison

    Plus robuste que C2C pour surfaces complexes

    Args:
        normal_scale: Échelle pour calcul normales (rayon voisinage)
        projection_scale: Rayon cylindre de projection
        threshold: Seuil de détection changement significatif
    """
    # 1. Calculer normales à échelle D
    pcd_ref.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamRadius(normal_scale)
    )

    # 2. Pour chaque point core (sous-échantillonné)
    # Sous-échantillonner pour accélérer
    pcd_ref_sampled = pcd_ref.voxel_down_sample(voxel_size=0.05)

    distances_m3c2 = []
    tree_ref = o3d.geometry.KDTreeFlann(pcd_ref)
    tree_new = o3d.geometry.KDTreeFlann(pcd_new)

    for i, point_core in enumerate(pcd_ref_sampled.points):
        normal = pcd_ref_sampled.normals[i]

        # 3. Trouver points dans cylindre (ref)
        [_, idx_ref, _] = tree_ref.search_radius_vector_3d(point_core, projection_scale)
        cylinder_pts_ref = np.asarray(pcd_ref.points)[idx_ref]

        # Projeter sur axe normal
        proj_ref = np.dot(cylinder_pts_ref - point_core, normal)
        mean_proj_ref = proj_ref.mean()

        # 4. Trouver points dans cylindre (new)
        [_, idx_new, _] = tree_new.search_radius_vector_3d(point_core, projection_scale)

        if len(idx_new) < 5:  # Pas assez de points → changement majeur
            distances_m3c2.append(999.0)  # Valeur sentinelle
            continue

        cylinder_pts_new = np.asarray(pcd_new.points)[idx_new]
        proj_new = np.dot(cylinder_pts_new - point_core, normal)
        mean_proj_new = proj_new.mean()

        # 5. Distance M3C2 (signée)
        distance_m3c2 = mean_proj_new - mean_proj_ref
        distances_m3c2.append(distance_m3c2)

    distances_m3c2 = np.array(distances_m3c2)

    # Détection changements significatifs
    significant_changes = np.abs(distances_m3c2) > threshold

    print(f"M3C2 significant changes: {significant_changes.sum()} / {len(distances_m3c2)}")

    return distances_m3c2, significant_changes, pcd_ref_sampled

# Utilisation
distances_m3c2, changes, pcd_cores = m3c2_change_detection(
    pcd_t0, pcd_t1,
    normal_scale=0.5,
    projection_scale=2.0,
    threshold=0.15
)

# Visualisation résultats M3C2
colors_m3c2 = plt.cm.RdBu(np.clip((distances_m3c2 + 0.5) / 1.0, 0, 1))[:, :3]
pcd_cores.colors = o3d.utility.Vector3dVector(colors_m3c2)
o3d.visualization.draw_geometries([pcd_cores], window_name="M3C2 Distances")
```

#### Change Clustering & Semantic Analysis

Après détection → Grouper changements en objets sémantiques

```python
def analyze_changes_semantically(pcd_changes, labels_semantic=None):
    """
    Analyse sémantique des zones de changement

    Args:
        pcd_changes: Point cloud des changements détectés
        labels_semantic: Labels sémantiques optionnels (building, road, etc.)
    """
    # 1. Clustering des zones de changement
    labels_clusters = np.array(pcd_changes.cluster_dbscan(
        eps=0.3, min_points=50
    ))

    n_clusters = labels_clusters.max() + 1
    print(f"Detected {n_clusters} change clusters")

    # 2. Analyse par cluster
    change_report = []

    for cluster_id in range(n_clusters):
        cluster_mask = (labels_clusters == cluster_id)
        cluster_pcd = pcd_changes.select_by_index(np.where(cluster_mask)[0])

        # Caractérisation géométrique
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        volume = bbox.volume()
        extent = bbox.get_extent()
        centroid = bbox.get_center()

        # Classification du type de changement
        if extent[2] < 0.5:  # Faible hauteur
            change_type = "Surface change (erosion/accumulation)"
        elif volume > 5.0:  # Grand volume
            change_type = "Major structural change (building/demolition)"
        elif extent[2] > 2.0:  # Grande hauteur
            change_type = "Vertical structure (wall/pole added/removed)"
        else:
            change_type = "Component change (furniture/equipment)"

        # Classification sémantique (si disponible)
        semantic_label = "Unknown"
        if labels_semantic is not None:
            # Majorité vote
            cluster_semantics = labels_semantic[cluster_mask]
            semantic_label = np.bincount(cluster_semantics).argmax()

        change_report.append({
            "cluster_id": cluster_id,
            "type": change_type,
            "semantic_class": semantic_label,
            "volume_m3": volume,
            "height_m": extent[2],
            "centroid": centroid.tolist(),
            "num_points": len(cluster_pcd.points)
        })

        print(f"\nCluster {cluster_id}:")
        print(f"  Type: {change_type}")
        print(f"  Volume: {volume:.2f} m³")
        print(f"  Height: {extent[2]:.2f} m")
        print(f"  Points: {len(cluster_pcd.points)}")

    return change_report, labels_clusters

# Utilisation
changed_pcd = pcd_aligned.select_by_index(changed_idx)
change_analysis, cluster_labels = analyze_changes_semantically(changed_pcd)

# Export rapport JSON
import json
with open("change_detection_report.json", "w") as f:
    json.dump(change_analysis, f, indent=2)
```

#### Comparaison Méthodes

| Méthode         | Vitesse       | Précision  | Robustesse Bruit | Usage                            |
| --------------- | ------------- | ---------- | ---------------- | -------------------------------- |
| **C2C**         | ⭐⭐⭐ Rapide | Bonne      | ❌ Sensible      | Changements globaux, screening   |
| **M3C2**        | ⭐⭐ Moyen    | Excellente | ✅ Robuste       | Surfaces complexes, analyse fine |
| **Voxel-based** | ⭐⭐⭐ Rapide | Moyenne    | ✅ Robuste       | Gros volumes, temps réel         |

---

## 📊 Métriques Avancées pour Évaluation

### Beyond Accuracy : Métriques Spécialisées 3D

#### Métriques Géométriques

```python
def compute_geometric_metrics(y_true, y_pred, points):
    """
    Métriques tenant compte de la géométrie 3D
    """
    from sklearn.metrics import confusion_matrix

    # 1. Boundary IoU (frontières objets)
    boundary_mask = detect_boundary_points(points, k=30)
    boundary_iou = compute_iou(
        y_true[boundary_mask],
        y_pred[boundary_mask],
        num_classes
    )

    # 2. Surface Coverage (% surface bien classifiée)
    surface_areas = compute_surface_per_class(points, y_true)
    surface_coverage = {}
    for cls in range(num_classes):
        mask_true = (y_true == cls)
        mask_correct = (y_true == cls) & (y_pred == cls)
        surface_coverage[cls] = mask_correct.sum() / mask_true.sum()

    # 3. Topological Correctness
    # Nombre de composantes connexes préservées
    topo_score = compute_topology_preservation(y_true, y_pred, points)

    return {
        "boundary_iou": boundary_iou,
        "surface_coverage": surface_coverage,
        "topology_score": topo_score
    }
```

---

**Dernière mise à jour** : Novembre 2025  
**Basé sur** : 23 articles de Florent Poux, Ph.D. (2020-2025)
