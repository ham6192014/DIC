"""Subpixel image correlation: bicubic interpolation + inverse-compositional
Gauss-Newton (IC-GN) optimization of the affine subset warp.

Reference: Pan, Li, Xie, "Fast, robust and accurate digital image correlation
calculation without redundant computations", Exp. Mech. 2013 (IC-GN algorithm
for DIC); Baker & Matthews, "Lucas-Kanade 20 Years On" (IC framework).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .subset import N_PARAMS, Subset, compose_inverse, params_to_matrix


def image_gradients(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference-quality gradients via a 5x5 Scharr-like Sobel kernel."""
    img = gray.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5, scale=1.0 / 60.0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5, scale=1.0 / 60.0)
    return gx, gy


class ImageInterpolator:
    """Bicubic spline over a full grayscale image, built once and reused for
    every subset evaluated against that (deformed) frame."""

    def __init__(self, gray: np.ndarray):
        h, w = gray.shape
        ys = np.arange(h, dtype=np.float64)
        xs = np.arange(w, dtype=np.float64)
        self._spline = RectBivariateSpline(ys, xs, gray.astype(np.float64), kx=3, ky=3)
        self.shape = (h, w)

    def eval(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._spline.ev(y, x)

    def in_bounds(self, x: np.ndarray, y: np.ndarray, margin: float = 2.0) -> np.ndarray:
        h, w = self.shape
        return (
            (x >= margin) & (x <= w - 1 - margin) & (y >= margin) & (y <= h - 1 - margin)
        )


@dataclass
class CorrelationResult:
    p: np.ndarray
    zncc: float
    converged: bool
    n_iter: int


def icgn_correlate(
    subset: Subset,
    interpolator: ImageInterpolator,
    p_init: np.ndarray,
    max_iter: int = 40,
    tol: float = 1e-4,
    zncc_threshold: float = 0.5,
) -> CorrelationResult:
    """Refine warp parameters p_init against `interpolator`'s frame using IC-GN.

    tol is on the incremental-warp-parameter norm on the subset boundary
    (~ subpixel displacement change per iteration).
    """
    if not subset.valid:
        return CorrelationResult(p_init.copy(), -1.0, False, 0)

    p = p_init.copy()
    n_iter = 0
    converged = False
    zncc = -1.0

    for n_iter in range(1, max_iter + 1):
        M = params_to_matrix(p)
        xw = subset.x0 + M[0, 0] * subset.dx + M[0, 1] * subset.dy + M[0, 2]
        yw = subset.y0 + M[1, 0] * subset.dx + M[1, 1] * subset.dy + M[1, 2]

        if not np.all(interpolator.in_bounds(xw, yw)):
            return CorrelationResult(p, -1.0, False, n_iter)

        g = interpolator.eval(xw, yw)
        g_mean = g.mean()
        g_tilde = g - g_mean
        delta_g = float(np.sqrt(np.sum(g_tilde ** 2)))
        if delta_g < 1e-8:
            return CorrelationResult(p, -1.0, False, n_iter)

        zncc = float(np.sum(subset.f_tilde * g_tilde) / (subset.delta_f * delta_g))

        # ZNSSD-consistent error image (accounts for linear brightness/contrast change).
        # Sign follows the compositional criterion C(dp) = || f(W(.;dp)) - g(W(.;p)) ||^2
        # (in zero-mean-normalized form), i.e. error ~ (g - f), not (f - g).
        error = (subset.delta_f / delta_g) * g_tilde - subset.f_tilde

        rhs = subset.steepest_descent.T @ error
        dp = subset.hessian_inv @ rhs

        p = compose_inverse(p, dp)

        # convergence: max corner displacement induced by dp
        r = subset.radius
        corner_shift = np.abs(dp[0]) + np.abs(dp[1]) * r + np.abs(dp[2]) * r
        corner_shift = max(
            corner_shift,
            np.abs(dp[3]) + np.abs(dp[4]) * r + np.abs(dp[5]) * r,
        )
        if corner_shift < tol:
            converged = True
            break

    converged = converged and zncc >= zncc_threshold
    return CorrelationResult(p, zncc, converged, n_iter)


def initial_guess_template_match(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    center: Tuple[float, float],
    half_win: int,
    search_radius: int,
) -> Tuple[float, float, float]:
    """Integer-pixel initial displacement guess via normalized cross-correlation.

    Returns (u0, v0, score). score is the NCC peak value in [-1, 1].
    """
    x0, y0 = center
    xi, yi = int(round(x0)), int(round(y0))
    h, w = ref_gray.shape

    tx0, tx1 = xi - half_win, xi + half_win + 1
    ty0, ty1 = yi - half_win, yi + half_win + 1
    if tx0 < 0 or ty0 < 0 or tx1 > w or ty1 > h:
        return 0.0, 0.0, -1.0
    template = ref_gray[ty0:ty1, tx0:tx1].astype(np.float32)

    sx0 = max(0, xi - half_win - search_radius)
    sy0 = max(0, yi - half_win - search_radius)
    sx1 = min(w, xi + half_win + search_radius + 1)
    sy1 = min(h, yi + half_win + search_radius + 1)
    search = cur_gray[sy0:sy1, sx0:sx1].astype(np.float32)

    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return 0.0, 0.0, -1.0

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    matched_x0 = sx0 + max_loc[0]
    matched_y0 = sy0 + max_loc[1]
    u0 = (matched_x0 + half_win) - x0
    v0 = (matched_y0 + half_win) - y0
    return float(u0), float(v0), float(max_val)
