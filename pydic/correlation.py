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
    """Bicubic-spline sampler over a grayscale image.

    Builds small *local* splines on demand (see `local_patch`) rather than
    one spline over the whole frame: scipy's RectBivariateSpline evaluation
    cost scales with the spline's total knot count, i.e. with the *whole
    image's* resolution, regardless of how few points you evaluate — so on
    a multi-megapixel frame, evaluating a 31x31 subset window thousands of
    times against one global spline is dramatically (5-10x+) slower than
    building a small spline just for the neighborhood each subset actually
    needs. Bicubic interpolation with s=0 is exact and has local support, so
    a local crop reproduces the same values as the global spline away from
    its own edges — `local_patch` keeps evaluation points a safe margin from
    the crop boundary and re-crops if the warp drifts past it.
    """

    def __init__(self, gray: np.ndarray):
        self.image = np.ascontiguousarray(gray, dtype=np.float64)
        self.shape = self.image.shape

    def in_bounds(self, x: np.ndarray, y: np.ndarray, margin: float = 2.0) -> np.ndarray:
        h, w = self.shape
        return (
            (x >= margin) & (x <= w - 1 - margin) & (y >= margin) & (y <= h - 1 - margin)
        )

    def _build_local_spline(self, x_center: float, y_center: float, half_size: int):
        h, w = self.shape
        xi, yi = int(round(x_center)), int(round(y_center))
        x0, x1 = max(0, xi - half_size), min(w, xi + half_size + 1)
        y0, y1 = max(0, yi - half_size), min(h, yi + half_size + 1)
        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        spline = RectBivariateSpline(ys, xs, self.image[y0:y1, x0:x1], kx=3, ky=3)
        return spline, (x0, x1, y0, y1)

    def local_patch(self, x_center: float, y_center: float, half_size: int = 40) -> "LocalPatch":
        """A small cached spline valid near (x_center, y_center), reused
        across an IC-GN point's iterations instead of rebuilding every time."""
        spline, bounds = self._build_local_spline(x_center, y_center, half_size)
        return LocalPatch(self, spline, bounds, half_size)


class LocalPatch:
    """A local spline crop, auto-recentering if queried outside its margin."""

    _EDGE_MARGIN = 3.0

    def __init__(self, parent: ImageInterpolator, spline, bounds, half_size: int):
        self._parent = parent
        self._spline = spline
        self._bounds = bounds
        self._half_size = half_size

    def eval(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1 = self._bounds
        m = self._EDGE_MARGIN
        if np.any(x < x0 + m) or np.any(x > x1 - 1 - m) or np.any(y < y0 + m) or np.any(y > y1 - 1 - m):
            self._spline, self._bounds = self._parent._build_local_spline(
                float(np.mean(x)), float(np.mean(y)), self._half_size
            )
        return self._spline.ev(y, x)


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

    # A local spline sized to comfortably contain the subset plus some drift
    # across iterations is far cheaper to build+query than repeatedly
    # evaluating one spline fit over the whole frame (see ImageInterpolator).
    patch = interpolator.local_patch(
        subset.x0 + p_init[0], subset.y0 + p_init[3], half_size=subset.radius + 25
    )

    for n_iter in range(1, max_iter + 1):
        M = params_to_matrix(p)
        xw = subset.x0 + M[0, 0] * subset.dx + M[0, 1] * subset.dy + M[0, 2]
        yw = subset.y0 + M[1, 0] * subset.dx + M[1, 1] * subset.dy + M[1, 2]

        if not np.all(interpolator.in_bounds(xw, yw)):
            return CorrelationResult(p, -1.0, False, n_iter)

        g = patch.eval(xw, yw)
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
