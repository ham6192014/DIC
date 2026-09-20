import numpy as np

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
