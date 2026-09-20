import numpy as np

from pydic.fields import build_neighbors, generate_grid, nearest_point_index
from pydic.strain import compute_strain_2d
from pydic.tracking2d import track_reliability_guided
from tests.synthetic import make_speckle_image, warp_affine_global


def test_rgdic_field_and_strain_match_known_affine_deformation():
    ref = make_speckle_image(size=(400, 400), n_speckles=3000, seed=2)
    true_p = np.array([2.5, 0.02, -0.01, -1.2, 0.015, 0.03])
    cur = warp_affine_global(ref, true_p, center=(200, 200))

    points, index_of = generate_grid(ref.shape, step=20, subset_radius=15)
    neighbors = build_neighbors(index_of)
    seed_idx = nearest_point_index(points, (200, 200))

    results = track_reliability_guided(
        ref, cur, points, neighbors, subset_radius=15,
        seed_indices=[seed_idx], seed_p_init=[np.zeros(6)],
    )
    converged = np.array([r.converged for r in results])
    assert converged.mean() > 0.8

    u = np.array([r.u for r in results])
    v = np.array([r.v for r in results])
    strain = compute_strain_2d(points, u, v, converged, neighbors, window_hops=2)

    ux, uy, vx, vy = true_p[1], true_p[2], true_p[4], true_p[5]
    exx_true = ux + 0.5 * (ux ** 2 + vx ** 2)
    eyy_true = vy + 0.5 * (uy ** 2 + vy ** 2)
    exy_true = 0.5 * (uy + vx) + 0.5 * (ux * uy + vx * vy)

    assert abs(np.nanmean(strain.exx) - exx_true) < 1e-3
    assert abs(np.nanmean(strain.eyy) - eyy_true) < 1e-3
    assert abs(np.nanmean(strain.exy) - exy_true) < 1e-3
