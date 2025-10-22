import numpy as np
from ign_lidar.core.classification.building import BuildingClusterer


def test_optimize_bbox_for_building_translation():
    # Create synthetic points: a building cluster elevated at z=5 around (10, 10)
    rng = np.random.RandomState(0)
    n_build = 200
    build_xy = rng.normal(loc=(10.0, 10.0), scale=1.0, size=(n_build, 2))
    build_z = np.full((n_build, 1), 5.0)
    build_pts = np.hstack([build_xy, build_z])

    # Ground points near origin (0,0) at z=0
    n_ground = 300
    ground_xy = rng.uniform(low=(-5, -5), high=(5, 5), size=(n_ground, 2))
    ground_z = np.zeros((n_ground, 1))
    ground_pts = np.hstack([ground_xy, ground_z])

    points = np.vstack([build_pts, ground_pts])
    heights = points[:, 2]

    # Initial bbox centered wrongly near origin
    initial_bbox = (-4.0, -4.0, 4.0, 4.0)

    clusterer = BuildingClusterer()
    best_shift, best_bbox = clusterer.optimize_bbox_for_building(
        points=points,
        heights=heights,
        initial_bbox=initial_bbox,
        max_shift=15.0,
        step=1.0,
        height_threshold=1.0,
        ground_penalty=1.0,
        non_ground_reward=1.0
    )

    dx, dy = best_shift

    # Expect a positive shift towards the building cluster at (10,10)
    assert dx > 0 or dy > 0

    # Ensure the optimized bbox includes many building points (z>1)
    xmin, ymin, xmax, ymax = best_bbox
    xs = points[:, 0]
    ys = points[:, 1]
    mask = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
    n_build_in = np.sum(mask & (heights > 1.0))
    n_ground_in = np.sum(mask & (heights <= 1.0))

    # Many more building points than ground points inside optimized bbox
    assert n_build_in > 50
    assert n_build_in > n_ground_in
