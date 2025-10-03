#!/usr/bin/env python3
"""
Exemple d'utilisation de la conversion QGIS.

Montre comment convertir des fichiers LAZ enrichis pour QGIS,
en mode simple ou batch.
"""

from ign_lidar import simplify_for_qgis
from pathlib import Path

# Exemple 1: Conversion simple
print("="*70)
print("EXEMPLE 1: Conversion simple")
print("="*70)

input_file = "/chemin/vers/fichier_enrichi.laz"
output_file = simplify_for_qgis(input_file)
print(f"\n✅ Fichier converti: {output_file}")


# Exemple 2: Conversion avec sortie personnalisée
print("\n\n" + "="*70)
print("EXEMPLE 2: Sortie personnalisée")
print("="*70)

input_file = "/chemin/vers/fichier_enrichi.laz"
output_file = "/chemin/vers/sortie_custom.laz"
simplify_for_qgis(input_file, output_file)


# Exemple 3: Conversion batch (plusieurs fichiers)
print("\n\n" + "="*70)
print("EXEMPLE 3: Conversion batch")
print("="*70)

input_dir = Path("/chemin/vers/dossier_enrichis")
enriched_files = list(input_dir.glob("*_enriched.laz"))

print(f"\n🔄 Conversion de {len(enriched_files)} fichiers...\n")

for i, file_path in enumerate(enriched_files, 1):
    print(f"\n[{i}/{len(enriched_files)}] {file_path.name}")
    output_file = simplify_for_qgis(str(file_path), verbose=False)
    print(f"  ✓ Converti: {Path(output_file).name}")

print(f"\n✅ Tous les fichiers convertis!")


# Exemple 4: Utilisation en ligne de commande
print("\n\n" + "="*70)
print("EXEMPLE 4: Ligne de commande")
print("="*70)

print("""
# Conversion simple
ign-lidar-qgis enriched.laz

# Avec sortie personnalisée
ign-lidar-qgis enriched.laz output.laz

# Mode silencieux
ign-lidar-qgis enriched.laz -q

# Batch (plusieurs fichiers)
ign-lidar-qgis -b file1.laz -b file2.laz -b file3.laz

# Batch avec find
find /path/to/files/ -name "*_enriched.laz" | while read f; do
    ign-lidar-qgis "$f" -q
done
""")
