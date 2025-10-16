# Classification Normalization

## Overview

The IGN LiDAR HD tiles sometimes contain **non-standard classification codes** that don't conform to ASPRS LAS 1.4 specifications. This module automatically detects and normalizes these codes during tile loading.

## Known Non-Standard Classes

### Class 67 (Unknown)

- **Found in**: Various IGN LiDAR HD tiles
- **Typical percentage**: 0.02-0.05% of points
- **Default behavior**: Remapped to **Class 1 (Unclassified)**
- **Reason**: Non-standard code, possibly artifact from IGN processing

## Configuration

### Basic Configuration

```yaml
preprocess:
  # Automatically fix non-standard classification codes
  normalize_classification: true # Recommended: true

  # Strict mode: remap ALL unknown codes to Unclassified
  strict_class_normalization: false # Recommended: false
```

### Configuration Options

| Option                       | Default | Description                                   |
| ---------------------------- | ------- | --------------------------------------------- |
| `normalize_classification`   | `true`  | Enable classification normalization           |
| `strict_class_normalization` | `false` | If `true`, remap ALL unknown codes to Class 1 |

## Behavior

### Normal Mode (`strict_class_normalization: false`)

- Only remaps **known problematic classes** (e.g., Class 67)
- Unknown extended classes (32+) are left unchanged
- Safe for most use cases

### Strict Mode (`strict_class_normalization: true`)

- Remaps **ALL** non-ASPRS-standard codes to Class 1
- Use if you need strict ASPRS compliance
- May remap valid extended classes

## Example Output

```
⚠️  Found 1 non-standard classification codes: [67]
   Class 67 → 1: 4,380 points (0.02%) [known IGN class]
✓ Remapped 4,380 points (0.02%) from 1 classes
```

## Standard ASPRS Classes (0-22)

| Code | Name                      |
| ---- | ------------------------- |
| 0    | Created, Never Classified |
| 1    | Unclassified              |
| 2    | Ground                    |
| 3    | Low Vegetation            |
| 4    | Medium Vegetation         |
| 5    | High Vegetation           |
| 6    | Building                  |
| 7    | Low Point (Noise)         |
| 9    | Water                     |
| 10   | Rail                      |
| 11   | Road Surface              |
| 17   | Bridge Deck               |
| 18   | High Noise                |

## Extended ASPRS Classes (32+)

Used by this project for BD TOPO® enrichment:

| Code | Name            | Source   |
| ---- | --------------- | -------- |
| 40   | Parking         | BD TOPO® |
| 41   | Sports Facility | BD TOPO® |
| 42   | Cemetery        | BD TOPO® |
| 43   | Power Line      | BD TOPO® |
| 44   | Agriculture     | RPG      |

## API Usage

```python
from ign_lidar.preprocessing.class_normalization import normalize_classification

# Normalize classification codes
classification_normalized, stats = normalize_classification(
    classification,
    strict_mode=False,
    report_unknown=True
)

# Check statistics
print(f"Remapped {stats['remapped_count']} points")
print(f"Unknown classes: {stats['unknown_classes']}")
print(f"Remapping details: {stats['remapping_details']}")
```

## Validation

```python
from ign_lidar.preprocessing.class_normalization import validate_classification

# Validate classification codes
report = validate_classification(
    classification,
    allow_extended=True
)

if not report['is_valid']:
    print(f"Invalid classes found: {report['invalid_classes']}")
    print(f"Warnings: {report['warnings']}")
```

## Troubleshooting

### Issue: "Found non-standard classification codes"

**Solution**: This is normal for some IGN tiles. The system automatically remaps these codes.

### Issue: "Too many unknown classes"

**Solution**:

1. Check if you're processing non-IGN data
2. Enable strict mode: `strict_class_normalization: true`
3. Review the remapping log to understand which classes are being changed

### Issue: "Extended classes being removed in strict mode"

**Solution**:

- Use normal mode (`strict_class_normalization: false`)
- Or update the `VALID_ASPRS_CLASSES` set in `class_normalization.py`

## Adding New Remapping Rules

If you discover additional non-standard IGN classes:

1. Edit `ign_lidar/preprocessing/class_normalization.py`
2. Update the `IGN_CLASS_REMAPPING` dictionary:

```python
IGN_CLASS_REMAPPING = {
    67: 1,   # Unknown class 67 -> Unclassified
    # Add new mappings here:
    # 123: 1,  # Example: Class 123 -> Unclassified
}
```

3. Document the new class in this README

## References

- [ASPRS LAS 1.4 Specification](https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf)
- [IGN LiDAR HD Documentation](https://geoservices.ign.fr/lidarhd)
