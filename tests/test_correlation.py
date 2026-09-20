import time

import numpy as np
from scipy.interpolate import RectBivariateSpline

from pydic.correlation import ImageInterpolator, icgn_correlate, image_gradients
from pydic.subset import build_subset
from tests.synthetic import make_speckle_image, warp_affine_global


def test_icgn_recovers_subpixel_affine_warp():
    ref = make_speckle_image(size=(300, 300), seed=1)
    true_p = np.array([3.37, 0.01, -0.005, -2.14, 0.004, 0.02])
    cur = warp_affine_global(ref, true_p, center=(150, 150))

    gx, gy = image_gradients(ref)
    subset = build_subset(ref, 150, 150, radius=20, grad_x=gx, grad_y=gy)
    assert subset.valid

    interp = ImageInterpolator(cur)
    res = icgn_correlate(subset, interp, p_init=np.zeros(6))

    assert res.converged
    assert res.zncc > 0.99
    assert np.max(np.abs(res.p - true_p)) < 1e-2


def test_icgn_rejects_flat_region():
    ref = np.full((100, 100), 128, dtype=np.uint8)
    cur = ref.copy()
    gx, gy = image_gradients(ref)
    subset = build_subset(ref, 50, 50, radius=15, grad_x=gx, grad_y=gy)
    assert not subset.valid


def test_local_interpolation_much_faster_than_global_spline_on_large_image():
    # Regression guard for a real bug: ImageInterpolator used to build one
    # bicubic spline over the *whole* frame and call .ev() on it per subset
    # per IC-GN iteration. scipy's RectBivariateSpline evaluation cost
    # scales with the spline's total knot count (i.e. with the whole
    # image's resolution), not with how many points are queried, so on a
    # multi-megapixel photo (common for real cameras) this made correlation
    # take minutes, matching a real report of the GUI "running for a long
    # time without detecting anything". Local, per-subset interpolation
    # must stay dramatically faster regardless of the surrounding image's
    # size. An absolute wall-clock bound would be flaky across machines, so
    # this compares the fix directly against the old approach on identical
    # queries instead.
    h, w = 3000, 4000
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, (h, w)).astype(np.float64)

    n_points = 800
    radius = 15
    centers_x = rng.uniform(500, w - 500, n_points)
    centers_y = rng.uniform(500, h - 500, n_points)
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dx, dy = dx.astype(np.float64).ravel(), dy.astype(np.float64).ravel()

    # Old approach: one spline over the whole image, queried per subset.
    t0 = time.time()
    global_spline = RectBivariateSpline(
        np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), image, kx=3, ky=3
    )
    for cx, cy in zip(centers_x, centers_y):
        global_spline.ev(cy + dy, cx + dx)
    global_time = time.time() - t0

    # Current approach: ImageInterpolator's local patches.
    interp = ImageInterpolator(image)
    t0 = time.time()
    for cx, cy in zip(centers_x, centers_y):
        patch = interp.local_patch(cx, cy, half_size=radius + 25)
        patch.eval(cx + dx, cy + dy)
    local_time = time.time() - t0

    assert local_time * 3 < global_time, (
        f"local interpolation ({local_time:.2f}s) should be much faster than "
        f"the whole-image spline ({global_time:.2f}s) on a large image"
    )
