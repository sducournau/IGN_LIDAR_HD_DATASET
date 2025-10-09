---
sidebar_position: 4
---

# QGIS Dépannage

Common issues and solutions when working with QGIS and the IGN LiDAR HD library.

## Common Issues

### Installation Problems

**Issue**: QGIS plugin not loading
**Solution**: Check Python path and dependencies:

```bash
# Verify QGIS Python environment
qgis --version
```

**Issue**: Missing dependencies
**Solution**: Install required packages in QGIS Python environment:

```bash
pip install laspy numpy
```

### Chargement des données Issues

**Issue**: LAS files not displaying in QGIS
**Solution**: Use the Nuage de points plugin or convert to compatible format:

```python
# Convert LAS to compatible format
from ign_lidar import QGISConverter
converter = QGISConverter()
converter.las_to_shapefile("input.las", "output.shp")
```

**Issue**: Large files causing memory issues
**Solution**: Enable chunked processing:

```python
config = Config(
    chunk_size=1000000,  # Traitement 1M points at a time
    memory_limit=8.0     # Limit to 8GB RAM
)
```

### Performance Issues

**Issue**: Slow processing in QGIS
**Solutions**:

- Reduce point density for visualization
- Use spatial indexing
- Enable GPU acceleration if available

### Projection Issues

**Issue**: Coordinate system misalignment
**Solution**: Verify and set correct CRS:

```python
# Set correct coordinate reference system
converter.set_crs("EPSG:2154")  # RGF93 / Lambert-93
```

## Getting Help

For additional support:

- Check the [QGIS documentation](https://qgis.org/documentation/)
- Visit the [IGN LiDAR HD GitHub repository](https://github.com/sducournau/IGN_LIDAR_HD_DATASET)
- Report issues in the project's issue tracker
