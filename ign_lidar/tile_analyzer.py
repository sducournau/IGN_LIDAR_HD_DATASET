"""
Tile Analysis Module

Analyzes LiDAR tiles to determine optimal processing parameters.
Automatically detects:
- Point density
- Noise levels
- Scan line spacing
- Optimal radius for feature computation
- Recommended preprocessing parameters
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import laspy


def analyze_tile(
    laz_path: Path,
    sample_size: int = 50000
) -> Dict[str, float]:
    """
    Analyze a LiDAR tile to determine optimal processing parameters.
    
    Analyzes:
    - Point density (points per m²)
    - Average nearest neighbor distance
    - Noise level (outlier percentage estimate)
    - Optimal radius for feature computation
    - Recommended preprocessing parameters
    
    Args:
        laz_path: Path to LAZ file
        sample_size: Number of points to sample for analysis
        
    Returns:
        Dictionary with recommended parameters:
        - point_density: points per square meter
        - avg_nn_distance: average nearest neighbor distance (m)
        - optimal_radius: recommended radius for features (m)
        - sor_k: recommended SOR k neighbors
        - sor_std: recommended SOR std multiplier
        - ror_radius: recommended ROR radius (m)
        - ror_neighbors: recommended ROR min neighbors
        - noise_level: estimated noise percentage
    """
    # Read tile
    las = laspy.read(laz_path)
    points = np.vstack([las.x, las.y, las.z]).T
    n_points = len(points)
    
    # Sample if needed
    if n_points > sample_size:
        indices = np.random.choice(n_points, sample_size, replace=False)
        sample_points = points[indices]
    else:
        sample_points = points
    
    # Compute bounding box and point density
    x_range = las.x.max() - las.x.min()
    y_range = las.y.max() - las.y.min()
    area_m2 = x_range * y_range
    point_density = n_points / area_m2 if area_m2 > 0 else 0
    
    # Estimate nearest neighbor distances
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(11, len(sample_points)))
    nbrs.fit(sample_points)
    distances, _ = nbrs.kneighbors(sample_points)
    
    # Average nearest neighbor distance (exclude self at index 0)
    avg_nn_distance = np.median(distances[:, 1])
    
    # Estimate noise level from distance distribution
    # Points with very large nearest neighbor distances are likely outliers
    distances_1nn = distances[:, 1]
    distance_threshold = np.percentile(distances_1nn, 95)
    noise_estimate = np.sum(distances_1nn > distance_threshold * 2) / len(
        distances_1nn
    )
    
    # Optimal radius for feature computation
    # Use 15-20x the average nearest neighbor distance
    # This captures true surface geometry, not scan lines
    optimal_radius = avg_nn_distance * 18.0
    
    # Clamp radius to reasonable range
    optimal_radius = np.clip(optimal_radius, 0.5, 3.0)
    
    # Determine preprocessing parameters based on density and noise
    if point_density > 20:  # Very dense (>20 pts/m²)
        sor_k = 15
        sor_std = 2.5
        ror_radius = 1.0
        ror_neighbors = 6
    elif point_density > 10:  # Dense (10-20 pts/m²)
        sor_k = 12
        sor_std = 2.0
        ror_radius = 1.0
        ror_neighbors = 4
    elif point_density > 5:  # Medium (5-10 pts/m²)
        sor_k = 10
        sor_std = 2.0
        ror_radius = 1.5
        ror_neighbors = 3
    else:  # Sparse (<5 pts/m²)
        sor_k = 8
        sor_std = 1.5
        ror_radius = 2.0
        ror_neighbors = 2
    
    # Adjust for high noise
    if noise_estimate > 0.05:  # >5% noise
        sor_std *= 0.8  # More aggressive filtering
        ror_neighbors += 1
    
    return {
        'point_density': float(point_density),
        'avg_nn_distance': float(avg_nn_distance),
        'optimal_radius': float(optimal_radius),
        'sor_k': int(sor_k),
        'sor_std': float(sor_std),
        'ror_radius': float(ror_radius),
        'ror_neighbors': int(ror_neighbors),
        'noise_level': float(noise_estimate * 100),  # as percentage
        'tile_size_mb': laz_path.stat().st_size / (1024 * 1024),
        'n_points': n_points
    }


def format_analysis_report(analysis: Dict[str, float]) -> str:
    """
    Format analysis results as a human-readable report.
    
    Args:
        analysis: Dictionary from analyze_tile()
        
    Returns:
        Formatted string report
    """
    lines = [
        "📊 Tile Analysis Report",
        "=" * 60,
        f"Total points:        {analysis['n_points']:>12,}",
        f"File size:           {analysis['tile_size_mb']:>12.1f} MB",
        f"Point density:       {analysis['point_density']:>12.1f} pts/m²",
        f"Avg NN distance:     {analysis['avg_nn_distance']:>12.3f} m",
        f"Noise estimate:      {analysis['noise_level']:>12.1f}%",
        "",
        "💡 Recommended Parameters",
        "-" * 60,
        f"Feature radius:      {analysis['optimal_radius']:>12.2f} m",
        f"SOR k-neighbors:     {analysis['sor_k']:>12d}",
        f"SOR std multiplier:  {analysis['sor_std']:>12.1f}",
        f"ROR radius:          {analysis['ror_radius']:>12.1f} m",
        f"ROR min neighbors:   {analysis['ror_neighbors']:>12d}",
        "=" * 60,
    ]
    return "\n".join(lines)


def should_use_chunked_processing(
    analysis: Dict[str, float],
    mode: str = 'full'
) -> Tuple[bool, int]:
    """
    Determine if chunked processing should be used and chunk size.
    
    Args:
        analysis: Dictionary from analyze_tile()
        mode: 'core' or 'full'
        
    Returns:
        (use_chunking, chunk_size)
    """
    n_points = analysis['n_points']
    
    # Building mode is more memory intensive
    if mode == 'full':
        if n_points > 15_000_000:
            return True, 10_000_000
        elif n_points > 10_000_000:
            return True, 15_000_000
        elif n_points > 5_000_000:
            return True, 20_000_000
        else:
            return False, 0
    else:  # core mode
        if n_points > 20_000_000:
            return True, 15_000_000
        elif n_points > 10_000_000:
            return True, 20_000_000
        else:
            return False, 0
    
    return False, 0
