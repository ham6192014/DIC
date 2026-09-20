"""Subset (facet) representation and first-order affine shape function.

Warp parameters p = [u, ux, uy, v, vx, vy] map a point (dx, dy) relative to the
subset center (x0, y0) in the reference image to a point in the deformed image:

    x' = x0 + dx + u + ux*dx + uy*dy
    y' = y0 + dy + v + vx*dx + vy*dy

Represented as a 3x3 affine matrix acting on homogeneous local coordinates
[dx, dy, 1]^T so that composition (needed for the inverse-compositional
Gauss-Newton update) is a plain matrix product.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

N_PARAMS = 6


def params_to_matrix(p: np.ndarray) -> np.ndarray:
    u, ux, uy, v, vx, vy = p
    return np.array(
        [
            [1.0 + ux, uy, u],
            [vx, 1.0 + vy, v],
            [0.0, 0.0, 1.0],
        ]
    )


def matrix_to_params(M: np.ndarray) -> np.ndarray:
    return np.array(
        [M[0, 2], M[0, 0] - 1.0, M[0, 1], M[1, 2], M[1, 0], M[1, 1] - 1.0]
    )


def compose_inverse(p: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """Inverse-compositional update: W(.;p) <- W(.;p) o W(.;dp)^-1."""
    M = params_to_matrix(p)
    dM = params_to_matrix(dp)
    M_new = M @ np.linalg.inv(dM)
    return matrix_to_params(M_new)


@dataclass
class Subset:
    """A square subset (facet) centered at (x0, y0) in the reference image."""

    x0: float
    y0: float
    radius: int  # half window size; subset is (2r+1) x (2r+1)
    dx: np.ndarray  # flattened local x offsets
    dy: np.ndarray  # flattened local y offsets
    f: np.ndarray  # reference intensities, flattened, float64
    fx: np.ndarray  # reference gradient d f / dx, flattened
    fy: np.ndarray  # reference gradient d f / dy, flattened
    f_mean: float
    f_tilde: np.ndarray  # f - f_mean
    delta_f: float  # sqrt(sum(f_tilde^2)), 0 if flat patch
    steepest_descent: np.ndarray  # (N, 6)
    hessian_inv: np.ndarray  # (6, 6)
    valid: bool  # False if patch has (near) zero contrast or falls outside image


def build_subset(
    ref_image_gray: np.ndarray,
    x0: float,
    y0: float,
    radius: int,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
) -> Subset:
    """Precompute everything IC-GN needs for a subset centered at (x0, y0).

    grad_x / grad_y are the precomputed gradient images of the *whole*
    reference image (see correlation.image_gradients), reused across subsets.
    """
    h, w = ref_image_gray.shape
    if not (np.isfinite(x0) and np.isfinite(y0)):
        return Subset(
            x0, y0, radius,
            *([np.zeros(0)] * 5), 0.0, np.zeros(0), 0.0,
            np.zeros((0, N_PARAMS)), np.zeros((N_PARAMS, N_PARAMS)), False,
        )
    xi = int(round(x0))
    yi = int(round(y0))
    x_lo, x_hi = xi - radius, xi + radius
    y_lo, y_hi = yi - radius, yi + radius

    if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
        return Subset(
            x0, y0, radius,
            *([np.zeros(0)] * 5), 0.0, np.zeros(0), 0.0,
            np.zeros((0, N_PARAMS)), np.zeros((N_PARAMS, N_PARAMS)), False,
        )

    dy_grid, dx_grid = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dx = dx_grid.astype(np.float64).ravel()
    dy = dy_grid.astype(np.float64).ravel()

    patch = ref_image_gray[y_lo:y_hi + 1, x_lo:x_hi + 1].astype(np.float64)
    fx_patch = grad_x[y_lo:y_hi + 1, x_lo:x_hi + 1].astype(np.float64)
    fy_patch = grad_y[y_lo:y_hi + 1, x_lo:x_hi + 1].astype(np.float64)

    f = patch.ravel()
    fx = fx_patch.ravel()
    fy = fy_patch.ravel()

    f_mean = f.mean()
    f_tilde = f - f_mean
    delta_f = float(np.sqrt(np.sum(f_tilde ** 2)))

    if delta_f < 1e-8:
        return Subset(
            x0, y0, radius, dx, dy, f, fx, fy, f_mean, f_tilde, 0.0,
            np.zeros((f.size, N_PARAMS)), np.zeros((N_PARAMS, N_PARAMS)), False,
        )

    # Steepest descent images for affine warp: SD = grad(f) . dW/dp
    sd = np.stack(
        [fx, fx * dx, fx * dy, fy, fy * dx, fy * dy], axis=1
    )  # (N, 6)
    hessian = sd.T @ sd
    try:
        hessian_inv = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return Subset(
            x0, y0, radius, dx, dy, f, fx, fy, f_mean, f_tilde, delta_f,
            sd, np.zeros((N_PARAMS, N_PARAMS)), False,
        )

    return Subset(x0, y0, radius, dx, dy, f, fx, fy, f_mean, f_tilde, delta_f, sd, hessian_inv, True)
