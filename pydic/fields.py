"""Regular subset-center grids, ROI masks, and 4-neighbor connectivity used
to drive reliability-guided propagation."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def polygon_to_mask(image_shape: Tuple[int, int], polygon: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Rasterize a polygon ROI (list of (x, y)) to a boolean mask."""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def generate_grid(
    image_shape: Tuple[int, int],
    step: int,
    subset_radius: int,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """Generate a regular grid of subset centers, `step` pixels apart, staying
    `subset_radius` away from the image border and inside `mask` if given.

    Returns (points[N,2] float64 in (x,y), index_of[(row,col)] -> point index)
    used to build 4-neighbor connectivity for RG-DIC.
    """
    h, w = image_shape
    xs = np.arange(subset_radius, w - subset_radius, step)
    ys = np.arange(subset_radius, h - subset_radius, step)

    points = []
    index_of: Dict[Tuple[int, int], int] = {}
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            if mask is not None and not mask[int(y), int(x)]:
                continue
            index_of[(row, col)] = len(points)
            points.append((float(x), float(y)))
    return np.array(points, dtype=np.float64), index_of


def build_neighbors(index_of: Dict[Tuple[int, int], int]) -> List[List[int]]:
    """4-connected neighbor lists aligned with generate_grid's row/col indices."""
    n = len(index_of)
    neighbors: List[List[int]] = [[] for _ in range(n)]
    for (row, col), idx in index_of.items():
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nkey = (row + drow, col + dcol)
            if nkey in index_of:
                neighbors[idx].append(index_of[nkey])
    return neighbors


def nearest_point_index(points: np.ndarray, xy: Tuple[float, float]) -> int:
    d2 = np.sum((points - np.asarray(xy)) ** 2, axis=1)
    return int(np.argmin(d2))


def khop_neighbors(neighbors: Sequence[Sequence[int]], idx: int, k: int) -> List[int]:
    """Grid indices within `k` graph hops of `idx` (inclusive), via BFS.

    Used to define the local strain window directly from grid topology,
    consistent with the physical spacing between subset centers.
    """
    visited = {idx}
    frontier = [idx]
    for _ in range(k):
        next_frontier = []
        for node in frontier:
            for nb in neighbors[node]:
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier
        if not frontier:
            break
    return list(visited)
