"""Strain from displacement/point-cloud fields via local least-squares fitting.

Fitting a plane to a small neighborhood of points (rather than differentiating
adjacent points directly) is the standard way DIC software gets a smooth,
noise-tolerant strain field out of noisy per-point displacements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .fields import khop_neighbors


@dataclass
class StrainField:
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    valid: np.ndarray

    @property
    def major_principal(self) -> np.ndarray:
        avg = 0.5 * (self.exx + self.eyy)
        rad = np.sqrt((0.5 * (self.exx - self.eyy)) ** 2 + self.exy ** 2)
        return avg + rad

    @property
    def minor_principal(self) -> np.ndarray:
        avg = 0.5 * (self.exx + self.eyy)
        rad = np.sqrt((0.5 * (self.exx - self.eyy)) ** 2 + self.exy ** 2)
        return avg - rad

    @property
    def von_mises(self) -> np.ndarray:
        return np.sqrt(self.exx ** 2 - self.exx * self.eyy + self.eyy ** 2 + 3 * self.exy ** 2)


def _gradient_to_strain(dudx, dudy, dvdx, dvdy, kind: str):
    if kind == "engineering":
        exx = dudx
        eyy = dvdy
        exy = 0.5 * (dudy + dvdx)
    elif kind == "green-lagrange":
        exx = dudx + 0.5 * (dudx ** 2 + dvdx ** 2)
        eyy = dvdy + 0.5 * (dudy ** 2 + dvdy ** 2)
        exy = 0.5 * (dudy + dvdx) + 0.5 * (dudx * dudy + dvdx * dvdy)
    else:
        raise ValueError(f"unknown strain kind: {kind}")
    return exx, eyy, exy


def compute_strain_2d(
    points: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    neighbors: Sequence[Sequence[int]],
    window_hops: int = 2,
    kind: str = "green-lagrange",
    min_points: int = 5,
) -> StrainField:
    """Strain from a planar displacement field defined on a subset grid.

    points: (N, 2) reference (x, y) subset centers.
    u, v: (N,) displacement components (e.g. from DicPointResult).
    neighbors: grid adjacency (fields.build_neighbors), used to gather a local
        window of `window_hops` grid rings around each point.
    """
    n = len(points)
    exx = np.full(n, np.nan)
    eyy = np.full(n, np.nan)
    exy = np.full(n, np.nan)
    out_valid = np.zeros(n, dtype=bool)

    for i in range(n):
        if not valid[i]:
            continue
        idxs = [j for j in khop_neighbors(neighbors, i, window_hops) if valid[j]]
        if len(idxs) < min_points:
            continue
        idxs = np.array(idxs)
        rel = points[idxs] - points[i]
        A = np.column_stack([np.ones(len(idxs)), rel[:, 0], rel[:, 1]])
        au, *_ = np.linalg.lstsq(A, u[idxs], rcond=None)
        av, *_ = np.linalg.lstsq(A, v[idxs], rcond=None)
        dudx, dudy = au[1], au[2]
        dvdx, dvdy = av[1], av[2]
        e = _gradient_to_strain(dudx, dudy, dvdx, dvdy, kind)
        exx[i], eyy[i], exy[i] = e
        out_valid[i] = True

    return StrainField(exx, eyy, exy, out_valid)


def compute_strain_3d_surface(
    ref_points_3d: np.ndarray,
    cur_points_3d: np.ndarray,
    valid: np.ndarray,
    neighbors: Sequence[Sequence[int]],
    window_hops: int = 2,
    kind: str = "green-lagrange",
    min_points: int = 5,
) -> StrainField:
    """Surface (Lagrangian) strain on a triangulated 3D point cloud.

    For each point, a local tangent-plane *normal* is estimated from the
    reference (undeformed) neighborhood via PCA (the least-variance
    direction). The in-plane axes are then taken as the global X/Y axes
    projected onto that tangent plane (falling back to Y/Z if the normal is
    nearly parallel to X), rather than the PCA's own in-plane singular
    vectors: PCA leaves the in-plane rotation, and the sign of each axis,
    arbitrary per point, which would make exx/eyy/exy meaningless to compare
    or plot across the field even though rotation-invariant quantities
    (principal strains, von Mises) come out correct either way. Projecting a
    fixed global direction instead gives a basis that varies smoothly across
    a gently curved surface, so exx/eyy/exy stay physically comparable
    point-to-point.
    """
    n = len(ref_points_3d)
    exx = np.full(n, np.nan)
    eyy = np.full(n, np.nan)
    exy = np.full(n, np.nan)
    out_valid = np.zeros(n, dtype=bool)

    global_x = np.array([1.0, 0.0, 0.0])
    global_y = np.array([0.0, 1.0, 0.0])

    for i in range(n):
        if not valid[i]:
            continue
        idxs = [j for j in khop_neighbors(neighbors, i, window_hops) if valid[j]]
        if len(idxs) < min_points:
            continue
        idxs = np.array(idxs)

        ref_rel = ref_points_3d[idxs] - ref_points_3d[i]
        _, _, vt = np.linalg.svd(ref_rel, full_matrices=False)
        normal = vt[2] if vt.shape[0] > 2 else np.cross(vt[0], vt[1])

        # Project global X and Y onto the tangent plane and Gram-Schmidt them
        # into an orthonormal in-plane basis. Unlike cross(normal, e1), this
        # is invariant to the sign of `normal` (PCA/SVD only determines it up
        # to +/-), so the basis orientation stays consistent across every
        # point without needing to resolve that sign first.
        e1 = global_x - np.dot(global_x, normal) * normal
        if np.linalg.norm(e1) < 0.2:
            e1 = global_y - np.dot(global_y, normal) * normal
        e1 /= np.linalg.norm(e1)
        e2 = global_y - np.dot(global_y, e1) * e1 - np.dot(global_y, normal) * normal
        if np.linalg.norm(e2) < 1e-6:
            e2 = global_x - np.dot(global_x, e1) * e1 - np.dot(global_x, normal) * normal
        e2 /= np.linalg.norm(e2)

        ref_2d = np.column_stack([ref_rel @ e1, ref_rel @ e2])

        cur_rel = cur_points_3d[idxs] - cur_points_3d[i]
        cur_2d = np.column_stack([cur_rel @ e1, cur_rel @ e2])

        A = np.column_stack([np.ones(len(idxs)), ref_2d[:, 0], ref_2d[:, 1]])
        cx, *_ = np.linalg.lstsq(A, cur_2d[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(A, cur_2d[:, 1], rcond=None)

        # deformation gradient of in-plane coords: F = d(cur_2d)/d(ref_2d)
        f11, f12 = cx[1], cx[2]
        f21, f22 = cy[1], cy[2]
        F = np.array([[f11, f12], [f21, f22]])

        if kind == "green-lagrange":
            E = 0.5 * (F.T @ F - np.eye(2))
        elif kind == "engineering":
            E = 0.5 * (F + F.T) - np.eye(2)
        else:
            raise ValueError(f"unknown strain kind: {kind}")

        exx[i], eyy[i], exy[i] = E[0, 0], E[1, 1], E[0, 1]
        out_valid[i] = True

    return StrainField(exx, eyy, exy, out_valid)
