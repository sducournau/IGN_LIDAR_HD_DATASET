#!/usr/bin/env python3
"""
Diagnostic script to validate building classification features.

This script checks if features are computed correctly and identifies
why buildings may not be classified properly.

Usage:
    python scripts/diagnose_classification.py <las_file>

Author: Classification Quality Audit
Date: October 24, 2025
"""
import sys
import numpy as np

try:
    import laspy
except ImportError:
    print("❌ Error: laspy not installed. Run: pip install laspy")
    sys.exit(1)


def diagnose_classification_features(las_path: str):
    """Check if features are computed correctly."""
    
    print("🔍 Building Classification Diagnostic")
    print("=" * 60)
    
    # Load point cloud
    try:
        las = laspy.read(las_path)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)
    
    points = np.vstack([las.x, las.y, las.z]).T
    labels = las.classification
    
    print(f"\n📊 Point Cloud Statistics:")
    print(f"  Total points: {len(points):,}")
    print(f"  Bounds: X=[{np.min(points[:, 0]):.2f}, {np.max(points[:, 0]):.2f}], "
          f"Y=[{np.min(points[:, 1]):.2f}, {np.max(points[:, 1]):.2f}], "
          f"Z=[{np.min(points[:, 2]):.2f}, {np.max(points[:, 2]):.2f}]")
    
    print(f"\n📊 Classification Distribution:")
    class_names = {
        0: "Never classified",
        1: "Unclassified",
        2: "Ground",
        3: "Low Vegetation",
        4: "Medium Vegetation",
        5: "High Vegetation",
        6: "Building",
        7: "Low Point (noise)",
        8: "Reserved",
        9: "Water",
        10: "Rail",
        11: "Road Surface",
        12: "Reserved",
        13: "Wire - Guard",
        14: "Wire - Conductor",
        15: "Transmission Tower",
        16: "Wire-Struct Connector",
        17: "Bridge Deck",
        18: "High Noise",
    }
    
    unique_classes = np.unique(labels)
    for cls in sorted(unique_classes):
        count = np.sum(labels == cls)
        pct = count / len(labels) * 100
        name = class_names.get(cls, f"Unknown ({cls})")
        icon = "🏢" if cls == 6 else "  "
        print(f"  {icon} Class {cls:2d} ({name:20s}): {count:8,} ({pct:5.2f}%)")
    
    # Critical check: Building classification
    building_mask = labels == 6
    building_count = np.sum(building_mask)
    building_pct = building_count / len(labels) * 100
    
    print(f"\n{'='*60}")
    if building_pct < 1.0:
        print(f"🔴 CRITICAL: Only {building_pct:.2f}% classified as buildings")
        print(f"   Expected: >5-10% for typical urban/suburban areas")
    elif building_pct < 5.0:
        print(f"⚠️  WARNING: Only {building_pct:.2f}% classified as buildings")
        print(f"   Expected: >5-10% for typical urban/suburban areas")
    else:
        print(f"✅ Building classification: {building_pct:.2f}% (reasonable)")
    print(f"{'='*60}")
    
    # Check if extra dimensions exist
    print(f"\n📝 Extra Dimensions (Features):")
    extra_dims = [dim.name for dim in las.point_format.extra_dimensions]
    
    if not extra_dims:
        print("  ❌ No extra dimensions found")
        print("     This may be an unprocessed/unenriched point cloud")
    else:
        print(f"  Found {len(extra_dims)} extra dimensions")
    
    critical_features = [
        'height_above_ground',
        'planarity',
        'verticality',
        'curvature',
        'ndvi',
        'BuildingConfidence',
        'IsWall',
        'IsRoof',
        'DistanceToPolygon',
    ]
    
    available_features = {}
    for feature in critical_features:
        if feature in extra_dims:
            data = las[feature]
            available_features[feature] = data
            print(f"  ✅ {feature:25s}: "
                  f"min={np.nanmin(data):7.3f}, "
                  f"mean={np.nanmean(data):7.3f}, "
                  f"max={np.nanmax(data):7.3f}")
        else:
            print(f"  ❌ {feature:25s}: MISSING")
    
    # Analyze building points
    if building_count > 0:
        print(f"\n🏢 Building Points Analysis ({building_count:,} points):")
        
        if 'height_above_ground' in available_features:
            hag = available_features['height_above_ground'][building_mask]
            print(f"  Height above ground: "
                  f"min={np.nanmin(hag):6.2f}m, "
                  f"mean={np.nanmean(hag):6.2f}m, "
                  f"max={np.nanmax(hag):6.2f}m")
            
            if np.nanmean(hag) < 2.5:
                print(f"  ⚠️  WARNING: Mean building height < 2.5m (threshold)")
                print(f"     This suggests incorrect HAG computation or ground points")
            
            # Check distribution
            below_threshold = np.sum(hag < 2.5)
            if below_threshold > building_count * 0.3:
                print(f"  ⚠️  {below_threshold:,} building points ({below_threshold/building_count*100:.1f}%) "
                      f"have HAG < 2.5m")
        
        if 'planarity' in available_features:
            plan = available_features['planarity'][building_mask]
            print(f"  Planarity (roofs):   "
                  f"min={np.nanmin(plan):6.3f}, "
                  f"mean={np.nanmean(plan):6.3f}, "
                  f"max={np.nanmax(plan):6.3f}")
            
            if np.nanmean(plan) < 0.65:
                print(f"  ⚠️  WARNING: Low mean planarity - roof detection may be difficult")
            
            high_planarity = np.sum(plan > 0.70)
            print(f"     {high_planarity:,} points ({high_planarity/building_count*100:.1f}%) "
                  f"have planarity > 0.70 (roof threshold)")
        
        if 'verticality' in available_features:
            vert = available_features['verticality'][building_mask]
            print(f"  Verticality (walls): "
                  f"min={np.nanmin(vert):6.3f}, "
                  f"mean={np.nanmean(vert):6.3f}, "
                  f"max={np.nanmax(vert):6.3f}")
            
            high_vert = np.sum(vert > 0.65)
            print(f"     {high_vert:,} points ({high_vert/building_count*100:.1f}%) "
                  f"have verticality > 0.65 (wall threshold)")
        
        if 'ndvi' in available_features:
            ndvi = available_features['ndvi'][building_mask]
            print(f"  NDVI (vegetation):   "
                  f"min={np.nanmin(ndvi):6.3f}, "
                  f"mean={np.nanmean(ndvi):6.3f}, "
                  f"max={np.nanmax(ndvi):6.3f}")
            
            if np.nanmean(ndvi) > 0.25:
                print(f"  ⚠️  WARNING: High NDVI - may indicate vegetation confusion")
            
            high_ndvi = np.sum(ndvi > 0.28)
            if high_ndvi > 0:
                print(f"     {high_ndvi:,} points ({high_ndvi/building_count*100:.1f}%) "
                      f"have NDVI > 0.28 (building threshold)")
        
        if 'BuildingConfidence' in available_features:
            conf = available_features['BuildingConfidence'][building_mask]
            print(f"  Classification confidence: "
                  f"min={np.nanmin(conf):6.3f}, "
                  f"mean={np.nanmean(conf):6.3f}, "
                  f"max={np.nanmax(conf):6.3f}")
            
            low_conf = np.sum(conf < 0.5)
            print(f"     {low_conf:,} points ({low_conf/building_count*100:.1f}%) "
                  f"have confidence < 0.5")
            
            high_conf = np.sum(conf > 0.7)
            print(f"     {high_conf:,} points ({high_conf/building_count*100:.1f}%) "
                  f"have confidence > 0.7 (high quality)")
        
        if 'DistanceToPolygon' in available_features:
            dist = available_features['DistanceToPolygon'][building_mask]
            inside = np.sum(dist <= 0)
            outside = np.sum(dist > 0)
            print(f"  Distance to polygon: "
                  f"min={np.nanmin(dist):6.2f}m, "
                  f"mean={np.nanmean(dist):6.2f}m, "
                  f"max={np.nanmax(dist):6.2f}m")
            print(f"     Inside polygon: {inside:,} ({inside/building_count*100:.1f}%)")
            print(f"     Outside polygon: {outside:,} ({outside/building_count*100:.1f}%)")
            
            if outside > inside:
                print(f"  ⚠️  WARNING: More building points outside than inside polygons")
                print(f"     This suggests severe polygon misalignment")
    else:
        print(f"\n❌ NO BUILDING POINTS FOUND")
        print(f"   This is the PRIMARY ISSUE - buildings not being classified")
    
    # Check unclassified points that might be buildings
    unclass_mask = labels == 1
    unclass_count = np.sum(unclass_mask)
    
    if unclass_count > 0:
        print(f"\n⚠️  Unclassified Points Analysis ({unclass_count:,} points, "
              f"{unclass_count/len(labels)*100:.1f}%):")
        
        if 'height_above_ground' in available_features:
            unclass_hag = available_features['height_above_ground'][unclass_mask]
            elevated = np.sum(unclass_hag > 2.5)
            
            print(f"  Elevated (HAG > 2.5m): {elevated:,} "
                  f"({elevated/unclass_count*100:.1f}% of unclassified)")
            
            if elevated > 1000:
                print(f"  🔴 CRITICAL: Many elevated points unclassified")
                print(f"     These are likely MISSED BUILDINGS")
                
                # Check features of elevated unclassified points
                elevated_mask = unclass_mask & (available_features['height_above_ground'] > 2.5)
                
                if 'planarity' in available_features:
                    elev_plan = available_features['planarity'][elevated_mask]
                    high_plan = np.sum(elev_plan > 0.65)
                    print(f"     {high_plan:,} elevated points have planarity > 0.65 (roof-like)")
                
                if 'verticality' in available_features:
                    elev_vert = available_features['verticality'][elevated_mask]
                    high_vert = np.sum(elev_vert > 0.65)
                    print(f"     {high_vert:,} elevated points have verticality > 0.65 (wall-like)")
                
                if 'ndvi' in available_features:
                    elev_ndvi = available_features['ndvi'][elevated_mask]
                    low_ndvi = np.sum(elev_ndvi < 0.25)
                    print(f"     {low_ndvi:,} elevated points have NDVI < 0.25 (non-vegetation)")
    
    # Summary and recommendations
    print(f"\n{'='*60}")
    print(f"📋 DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    
    issues = []
    
    if building_pct < 1.0:
        issues.append("🔴 CRITICAL: Very low building classification rate")
    elif building_pct < 5.0:
        issues.append("⚠️  Low building classification rate")
    
    if 'height_above_ground' not in available_features:
        issues.append("❌ Height above ground not computed")
    elif building_count > 0 and np.nanmean(available_features['height_above_ground'][building_mask]) < 2.5:
        issues.append("⚠️  Building heights suspiciously low")
    
    if 'BuildingConfidence' not in available_features:
        issues.append("❌ Building confidence scores not available")
    
    if 'DistanceToPolygon' in available_features and building_count > 0:
        dist = available_features['DistanceToPolygon'][building_mask]
        outside = np.sum(dist > 0)
        if outside > building_count * 0.5:
            issues.append("⚠️  Most building points are outside ground truth polygons")
    
    if unclass_count > len(labels) * 0.3:
        issues.append(f"⚠️  High unclassified rate ({unclass_count/len(labels)*100:.1f}%)")
    
    if issues:
        print("\n❌ Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n💡 Recommended Actions:")
        if building_pct < 5.0:
            print("  1. Check ground truth polygon alignment:")
            print("     - Increase max_translation to 12.0m")
            print("     - Enable polygon rotation")
            print("     - Increase fuzzy_boundary_outer to 5.0m")
        
        if 'height_above_ground' not in available_features:
            print("  2. Enable RGE ALTI DTM:")
            print("     - Set rge_alti.enabled = true")
            print("     - Set features.use_rge_alti_for_height = true")
        elif building_count > 0 and np.nanmean(available_features['height_above_ground'][building_mask]) < 2.5:
            print("  2. Improve DTM quality:")
            print("     - Reduce augmentation_spacing to 2.0m")
            print("     - Add 'buildings' to augmentation_areas")
        
        if building_count > 0:
            print("  3. Relax classification thresholds:")
            print("     - Reduce min_classification_confidence to 0.45")
            print("     - Reduce roof_planarity_min to 0.65")
            print("     - Increase roof_curvature_max to 0.15")
    else:
        print("\n✅ No major issues detected!")
        print("   Classification appears to be working correctly.")
    
    print(f"\n{'='*60}")
    print("📝 For detailed recommendations, see:")
    print("   docs/CLASSIFICATION_QUALITY_AUDIT_2025.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_classification.py <las_file>")
        print("\nExample:")
        print("  python scripts/diagnose_classification.py output/tile_enriched.laz")
        sys.exit(1)
    
    diagnose_classification_features(sys.argv[1])
