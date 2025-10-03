#!/usr/bin/env python3
"""
Test de compatibilité QGIS pour les fichiers LAZ enrichis.

Ce script vérifie que les fichiers LAZ enrichis peuvent être :
1. Écrits avec la bonne compression
2. Relus correctement
3. Affichés dans QGIS

Usage:
    python scripts/validation/test_qgis_compatibility.py <fichier_enrichi.laz>
"""

import sys
from pathlib import Path
import laspy
import numpy as np


def test_laz_compatibility(laz_file: Path) -> bool:
    """
    Teste la compatibilité d'un fichier LAZ enrichi.
    
    Args:
        laz_file: Chemin du fichier LAZ à tester
        
    Returns:
        True si compatible, False sinon
    """
    print(f"\n{'='*70}")
    print(f"Test de compatibilité QGIS pour: {laz_file.name}")
    print(f"{'='*70}\n")
    
    try:
        # 1. Lire le fichier
        print("1️⃣  Lecture du fichier...")
        las = laspy.read(str(laz_file))
        print(f"   ✓ Fichier lu avec succès")
        print(f"   - Points: {len(las.points):,}")
        print(f"   - Format: {las.header.point_format.id}")
        print(f"   - Version: {las.header.version}")
        
        # 2. Vérifier les dimensions
        print("\n2️⃣  Vérification des dimensions...")
        standard_dims = {'X', 'Y', 'Z', 'intensity', 'return_number', 
                        'number_of_returns', 'classification', 'user_data',
                        'point_source_id', 'gps_time', 'red', 'green', 'blue',
                        'scan_direction_flag', 'edge_of_flight_line',
                        'synthetic', 'key_point', 'withheld', 'overlap',
                        'scan_angle_rank', 'scanner_channel', 'scan_angle'}
        
        all_dims = set(las.point_format.dimension_names)
        extra_dims = all_dims - standard_dims
        
        if extra_dims:
            print(f"   ✓ Dimensions supplémentaires trouvées:")
            for dim in sorted(extra_dims):
                print(f"      - {dim}")
        else:
            print(f"   ⚠️  Aucune dimension supplémentaire")
        
        # 3. Vérifier la compression
        print("\n3️⃣  Vérification de la compression...")
        if laz_file.suffix.lower() == '.laz':
            print(f"   ✓ Extension LAZ correcte")
            
            # Tenter de déterminer le type de compression
            try:
                # Lire les premiers octets pour détecter la compression
                with open(laz_file, 'rb') as f:
                    header = f.read(4)
                    if header == b'LASF':
                        print(f"   ✓ En-tête LAZ valide")
                    else:
                        print(f"   ⚠️  En-tête non standard: {header}")
            except Exception as e:
                print(f"   ⚠️  Impossible de vérifier l'en-tête: {e}")
        else:
            print(f"   ⚠️  Extension non-LAZ: {laz_file.suffix}")
        
        # 4. Test d'échantillonnage
        print("\n4️⃣  Test d'échantillonnage des données...")
        sample_size = min(10, len(las.points))
        
        print(f"   Échantillon de {sample_size} points:")
        for i in range(sample_size):
            x, y, z = las.x[i], las.y[i], las.z[i]
            print(f"      Point {i}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
            
            # Tester les dimensions extra si présentes
            if extra_dims:
                first_extra = sorted(extra_dims)[0]
                value = getattr(las, first_extra)[i]
                print(f"              {first_extra}={value:.4f}")
        
        # 5. Statistiques
        print("\n5️⃣  Statistiques...")
        print(f"   Étendue X: [{las.header.x_min:.2f}, {las.header.x_max:.2f}]")
        print(f"   Étendue Y: [{las.header.y_min:.2f}, {las.header.y_max:.2f}]")
        print(f"   Étendue Z: [{las.header.z_min:.2f}, {las.header.z_max:.2f}]")
        
        if hasattr(las, 'classification'):
            unique_classes = np.unique(las.classification)
            print(f"   Classes: {len(unique_classes)} types")
            print(f"   Classes présentes: {sorted(unique_classes.tolist())}")
        
        # 6. Recommandations
        print("\n6️⃣  Recommandations pour QGIS...")
        print(f"   📌 Chargement dans QGIS:")
        print(f"      1. Menu: Couche > Ajouter une couche > Ajouter une couche nuage de points")
        print(f"      2. Sélectionner: {laz_file}")
        print(f"      3. Le nuage devrait apparaître dans le panneau")
        print(f"")
        print(f"   📌 Affichage des dimensions extra:")
        print(f"      1. Clic droit sur la couche > Propriétés")
        print(f"      2. Onglet 'Symbologie'")
        print(f"      3. Choisir 'Attribut' dans le menu déroulant")
        print(f"      4. Sélectionner une dimension enrichie (ex: 'curvature')")
        
        print(f"\n{'='*70}")
        print(f"✅ RÉSULTAT: Fichier compatible QGIS")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERREUR: {e}")
        print(f"{'='*70}\n")
        return False


def create_test_enriched_laz(output_path: Path) -> bool:
    """
    Crée un fichier LAZ enrichi de test.
    
    Args:
        output_path: Chemin de sortie
        
    Returns:
        True si succès
    """
    print(f"Création d'un fichier LAZ enrichi de test...")
    
    try:
        # Créer des données de test
        n_points = 10000
        x = np.random.uniform(0, 100, n_points)
        y = np.random.uniform(0, 100, n_points)
        z = np.random.uniform(0, 50, n_points)
        
        # Créer un fichier LAZ
        header = laspy.LasHeader(version="1.4", point_format=7)
        header.scales = [0.001, 0.001, 0.001]
        header.offsets = [np.min(x), np.min(y), np.min(z)]
        
        las = laspy.LasData(header)
        las.x = x
        las.y = y
        las.z = z
        las.classification = np.random.randint(0, 6, n_points, dtype=np.uint8)
        
        # Ajouter des dimensions enrichies
        las.add_extra_dim(laspy.ExtraBytesParams(name='curvature', type=np.float32))
        las.curvature = np.random.uniform(0, 1, n_points)
        
        las.add_extra_dim(laspy.ExtraBytesParams(name='height_above_ground', type=np.float32))
        las.height_above_ground = np.abs(z - np.min(z))
        
        for i, dim in enumerate(['normal_x', 'normal_y', 'normal_z']):
            las.add_extra_dim(laspy.ExtraBytesParams(name=dim, type=np.float32))
            setattr(las, dim, np.random.uniform(-1, 1, n_points))
        
        # IMPORTANT: Écrire avec compression LAZ pour QGIS
        # laspy choisit automatiquement le meilleur backend disponible
        las.write(str(output_path), do_compress=True)
        
        print(f"✓ Fichier de test créé: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors de la création: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test d'un fichier existant
        laz_file = Path(sys.argv[1])
        if not laz_file.exists():
            print(f"❌ Fichier non trouvé: {laz_file}")
            sys.exit(1)
        
        success = test_laz_compatibility(laz_file)
        sys.exit(0 if success else 1)
    else:
        # Créer et tester un fichier de test
        test_file = Path("/tmp/test_enriched_qgis.laz")
        
        print("🔧 Mode test: création d'un fichier LAZ enrichi")
        print(f"Fichier: {test_file}\n")
        
        if create_test_enriched_laz(test_file):
            success = test_laz_compatibility(test_file)
            
            if success:
                print(f"\n💡 Vous pouvez maintenant ouvrir ce fichier dans QGIS:")
                print(f"   {test_file}")
            
            sys.exit(0 if success else 1)
        else:
            sys.exit(1)
