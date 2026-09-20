"""Helpers to synthesize speckle images and known warps, for self-validation
of the correlation/strain math without needing real camera images."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline


def make_speckle_image(
    size: Tuple[int, int] = (512, 512),
    n_speckles: int = 4000,
    radius_range: Tuple[float, float] = (2.0, 4.5),
    seed: int = 0,
) -> np.ndarray:
    """Render a synthetic random-speckle pattern (as used to validate DIC)."""
    h, w = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 40.0, dtype=np.float64)

    ss = 4  # supersample for smoother sub-pixel speckle edges
    big = np.full((h * ss, w * ss), 40.0, dtype=np.float64)
    xs = rng.uniform(0, w * ss, n_speckles)
    ys = rng.uniform(0, h * ss, n_speckles)
    rs = rng.uniform(radius_range[0] * ss, radius_range[1] * ss, n_speckles)
    for x, y, r in zip(xs, ys, rs):
        cv2.circle(big, (int(x), int(y)), int(r), 220.0, -1, lineType=cv2.LINE_AA)
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    noise = rng.normal(0, 2.0, img.shape)
    img = np.clip(img + noise, 0, 255)
    return img.astype(np.uint8)


def warp_affine_global(
    img: np.ndarray, p: np.ndarray, center: Tuple[float, float]
) -> np.ndarray:
    """Apply the SAME first-order affine warp used by pydic.subset everywhere
    in the image (i.e. a homogeneous deformation field), via resampling with a
    bicubic spline so the ground truth is continuous/subpixel accurate."""
    u, ux, uy, v, vx, vy = p
    h, w = img.shape
    x0, y0 = center
    ys = np.arange(h, dtype=np.float64)
    xs = np.arange(w, dtype=np.float64)
    spline = RectBivariateSpline(ys, xs, img.astype(np.float64), kx=3, ky=3)

    # For output pixel (X, Y) we need the *reference* location that warps to it,
    # i.e. invert the forward warp X = x0 + dx + u + ux dx + uy dy (dx=x-x0 etc).
    Xg, Yg = np.meshgrid(xs, ys)
    A = np.array([[1 + ux, uy], [vx, 1 + vy]])
    A_inv = np.linalg.inv(A)
    dX = Xg - x0 - u
    dY = Yg - y0 - v
    dx = A_inv[0, 0] * dX + A_inv[0, 1] * dY
    dy = A_inv[1, 0] * dX + A_inv[1, 1] * dY
    src_x = x0 + dx
    src_y = y0 + dy

    out = spline.ev(src_y.ravel(), src_x.ravel()).reshape(h, w)
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)
