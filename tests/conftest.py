#!/usr/bin/env python3
"""
Test Configuration and Fixtures for IGN LiDAR HD Library

This module provides common test fixtures and utilities.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_point_cloud():
    """Generate a simple synthetic point cloud for testing."""
    # Create a 100x100 grid of points
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    
    # Add some elevation variation
    z = 10 + 2 * np.sin(xx/10) + np.cos(yy/10) + np.random.normal(0, 0.1, xx.shape)
    
    # Flatten to point cloud format
    points = np.column_stack([
        xx.flatten(),
        yy.flatten(), 
        z.flatten()
    ])
    
    # Add some classification (mostly ground and building)
    classification = np.random.choice([2, 6], size=len(points), p=[0.3, 0.7])
    
    return {
        'xyz': points,
        'classification': classification,
        'num_points': len(points)
    }


@pytest.fixture
def sample_tile_info():
    """Create sample tile information for testing."""
    from ign_lidar.tile_list import TileInfo
    
    return TileInfo(
        filename="TEST_TILE_100_200.laz",
        tile_x=100,
        tile_y=200,
        location="Test Location",
        environment="urban",
        description="Test tile for unit testing",
        recommended_lod="LOD2",
        coordinates_lambert93=(100500, 200500),
        coordinates_gps=(48.8566, 2.3522),  # Paris coordinates
        priority=1
    )


class MockLASFile:
    """Mock LAS file for testing without actual file I/O."""
    
    def __init__(self, point_cloud_data):
        self.xyz = point_cloud_data['xyz']
        self.classification = point_cloud_data['classification']
        self.header = MockHeader()
    
    @property
    def x(self):
        return self.xyz[:, 0]
    
    @property  
    def y(self):
        return self.xyz[:, 1]
        
    @property
    def z(self):
        return self.xyz[:, 2]


class MockHeader:
    """Mock LAS header for testing."""
    
    def __init__(self):
        self.min = [0, 0, 0]
        self.max = [100, 100, 20]
        self.point_count = 2500


# Test constants
TEST_TOLERANCE = 1e-6
SAMPLE_BBOX = (10, 10, 90, 90)