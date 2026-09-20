"""Matplotlib plotting helpers for displacement/strain fields.

Kept deliberately simple (scatter-based, works on the scattered/irregular
subset grid without needing a triangulation) so a GUI layer can call these
directly or reuse the same data prep for its own canvas widgets.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def plot_scalar_field(
    points: np.ndarray,
    values: np.ndarray,
    valid: Optional[np.ndarray] = None,
    title: str = "",
    cmap: str = "jet",
    point_size: float = 20.0,
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    mask = np.isfinite(values) if valid is None else (valid & np.isfinite(values))
    sc = ax.scatter(points[mask, 0], points[mask, 1], c=values[mask], cmap=cmap, s=point_size)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax)
    return ax


def plot_vector_field(
    points: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
    title: str = "displacement",
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    mask = np.ones(len(points), dtype=bool) if valid is None else valid
    ax.quiver(points[mask, 0], points[mask, 1], u[mask], v[mask], scale=scale, angles="xy")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def plot_points_3d(
    points3d: np.ndarray,
    values: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    title: str = "",
    cmap: str = "jet",
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    mask = np.all(np.isfinite(points3d), axis=1)
    if valid is not None:
        mask &= valid
    c = values[mask] if values is not None else None
    sc = ax.scatter(points3d[mask, 0], points3d[mask, 1], points3d[mask, 2], c=c, cmap=cmap)
    ax.set_title(title)
    if c is not None:
        plt.colorbar(sc, ax=ax)
    return ax
