"""Descriptive outer-convex-hull boundary definition, independent of labels."""

import numpy as np
from scipy.spatial import ConvexHull


def compute_boundary_mask(coords, sections, percentile=15.0):
    if not 0 < percentile < 100:
        raise ValueError("Boundary percentile must be between 0 and 100")
    result = np.zeros(len(coords), dtype=bool)
    for section in np.unique(sections):
        mask = sections == section
        points = coords[mask, :2]
        hull = ConvexHull(points)
        distances = np.full(len(points), np.inf)
        for simplex in hull.simplices:
            a, b = points[simplex]
            ab = b - a
            t = np.clip(((points - a) @ ab) / (ab @ ab), 0, 1)
            distances = np.minimum(distances, np.linalg.norm(points - a - t[:, None] * ab, axis=1))
        result[mask] = distances <= np.percentile(distances, percentile)
    return result
