"""Validation of the stereo-matching pipeline against a synthetic rig with
*real lens distortion* and a *wide baseline* (disparity well beyond the old
hardcoded +/-150px default) — the two conditions most likely to explain a
real-world report of near-total matching failure (few valid points).

Camera views are rendered from a continuous speckle texture via the true
(nonlinear) distorted projection, pixel by pixel, so both cameras see a
realistic, richly-textured pattern with the SAME lens distortion a real
industrial-camera stereo rig would have. Ground truth correspondence for
arbitrary query points is computed independently via the same camera model
(ray back-projection through the known plane, then forward projection with
distortion into the other camera), so matching accuracy is checked directly
against it, not just "did something match".
"""
from __future__ import annotations

import numpy as np
import cv2

from pydic.calibration import CameraCalibration, StereoCalibration
from pydic.stereo import (
    estimate_disparity_range,
    stereo_match_reference,
    triangulate_points,
)
from tests.synthetic import make_speckle_image


# A fairly typical industrial-lens distortion (visible barrel distortion,
# not extreme) and a wide-baseline rig where true disparity is ~470px --
# more than 3x the old fixed +/-150px search range.
_K = np.array([[2500.0, 0, 800.0], [0, 2500.0, 600.0], [0, 0, 1]])
_D = np.array([-0.18, 0.06, 0.0005, -0.0003, 0.0])
_IMG_SIZE = (1600, 1200)  # w, h
_BASELINE_MM = 150.0
_WORKING_DISTANCE_MM = 800.0
# world-plane texture: texture pixel -> mm, and mm origin at texture center
_TEX_SCALE_MM_PER_PX = 0.5
_TEX_SIZE = (2400, 2000)  # covers +/-600 x +/-500 mm of world plane


def _make_rig():
    R2 = cv2.Rodrigues(np.array([0.0, -0.06, 0.0]))[0]  # slight vergence
    T2 = np.array([-_BASELINE_MM, 0.0, 0.0])
    Tx = np.array([[0, -T2[2], T2[1]], [T2[2], 0, -T2[0]], [-T2[1], T2[0], 0]])
    E = Tx @ R2
    F = np.linalg.inv(_K).T @ E @ np.linalg.inv(_K)
    cam1 = CameraCalibration(_K, _D, _IMG_SIZE, 0.1)
    cam2 = CameraCalibration(_K, _D, _IMG_SIZE, 0.1)
    return StereoCalibration(cam1, cam2, R2, T2, E, F, 0.1)


def _pixels_to_world_on_plane(pixels_xy, K, D, R_cam, T_cam, plane_z_cam1):
    """Back-project camera pixels (accounting for distortion) to their
    intersection with the world plane Z=plane_z_cam1 in camera-1's frame.
    R_cam, T_cam: this camera's pose relative to camera 1 (identity/zero for
    camera 1 itself)."""
    pixels_xy = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pixels_xy, K, D).reshape(-1, 2)  # (x_n, y_n), z=1 ray in this camera's frame
    rays_cam = np.column_stack([normalized, np.ones(len(normalized))])  # (N,3)

    R_cam_to_1 = R_cam.T
    ray_dirs_1 = (R_cam_to_1 @ rays_cam.T).T  # ray directions in camera-1 frame
    origin_1 = (-R_cam_to_1 @ T_cam).reshape(3)  # this camera's center in camera-1 frame

    t = (plane_z_cam1 - origin_1[2]) / ray_dirs_1[:, 2]
    world = origin_1[None, :] + t[:, None] * ray_dirs_1
    return world  # (N,3), world[:,2] == plane_z_cam1


def _world_to_texture_px(world_xy):
    cx, cy = _TEX_SIZE[0] / 2.0, _TEX_SIZE[1] / 2.0
    tx = cx + world_xy[..., 0] / _TEX_SCALE_MM_PER_PX
    ty = cy + world_xy[..., 1] / _TEX_SCALE_MM_PER_PX
    return tx, ty


def _render_distorted_view(texture, K, D, R_cam, T_cam):
    """Render this camera's full view of the world-plane texture, with true
    lens distortion, via per-pixel inverse mapping (undistort each output
    pixel to a ray, intersect the world plane, sample the texture there)."""
    w, h = _IMG_SIZE
    ys, xs = np.mgrid[0:h, 0:w]
    pixels = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    world = _pixels_to_world_on_plane(pixels, K, D, R_cam, T_cam, _WORKING_DISTANCE_MM)
    tx, ty = _world_to_texture_px(world[:, :2])
    map_x = tx.reshape(h, w).astype(np.float32)
    map_y = ty.reshape(h, w).astype(np.float32)
    img = cv2.remap(texture, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=40)
    return img


def _ground_truth_cam2_pixel(px1, py1, calib):
    world = _pixels_to_world_on_plane(
        np.array([[px1, py1]]), calib.cam1.camera_matrix, calib.cam1.dist_coeffs,
        np.eye(3), np.zeros(3), _WORKING_DISTANCE_MM,
    )
    rvec2, _ = cv2.Rodrigues(calib.R)
    proj2, _ = cv2.projectPoints(world, rvec2, calib.T, calib.cam2.camera_matrix, calib.cam2.dist_coeffs)
    return proj2.reshape(2)


def _render_pair():
    calib = _make_rig()
    texture = make_speckle_image(size=(_TEX_SIZE[1], _TEX_SIZE[0]), n_speckles=45000, seed=3)
    img1 = _render_distorted_view(texture, calib.cam1.camera_matrix, calib.cam1.dist_coeffs, np.eye(3), np.zeros(3))
    img2 = _render_distorted_view(texture, calib.cam2.camera_matrix, calib.cam2.dist_coeffs, calib.R, calib.T)
    return calib, img1, img2


def test_expected_disparity_matches_ground_truth_geometry():
    calib = _make_rig()
    lo, hi = calib.expected_disparity_range_px(_WORKING_DISTANCE_MM, _WORKING_DISTANCE_MM)
    expected = _K[0, 0] * _BASELINE_MM / _WORKING_DISTANCE_MM
    assert abs(lo - expected) < 1e-6 and abs(hi - expected) < 1e-6
    # and, the key point this whole test file is about: it's nowhere near +/-150px
    assert expected > 150.0


def test_old_fixed_range_would_miss_the_true_disparity():
    # Documents *why* a fixed +/-150px assumption fails here: sanity-check
    # that the true disparity for this (realistic) wide-baseline rig falls
    # outside it entirely, before showing that the auto-ranging fix finds it.
    calib = _make_rig()
    true_disparity = _K[0, 0] * _BASELINE_MM / _WORKING_DISTANCE_MM
    assert not (-150.0 <= true_disparity <= 150.0)


def test_auto_disparity_range_recovers_true_wide_baseline_disparity():
    calib, img1, img2 = _render_pair()
    u1 = calib.cam1.undistort_image(img1)
    u2 = calib.cam2.undistort_image(img2)
    match_calib = calib.undistorted_for_matching()

    rng = np.random.default_rng(2)
    xs = rng.uniform(100, _IMG_SIZE[0] - 100, 150)
    ys = rng.uniform(100, _IMG_SIZE[1] - 100, 150)
    points1_distorted = np.column_stack([xs, ys])
    points1_undist = cv2.undistortPoints(
        points1_distorted.reshape(-1, 1, 2), calib.cam1.camera_matrix, calib.cam1.dist_coeffs,
        P=calib.cam1.camera_matrix,
    ).reshape(-1, 2)

    est_range = estimate_disparity_range(u1, u2, points1_undist, match_calib, subset_radius=15)
    assert est_range is not None, "auto disparity-range estimation found no confident matches at all"
    true_disparity = _K[0, 0] * _BASELINE_MM / _WORKING_DISTANCE_MM
    assert est_range[0] <= true_disparity <= est_range[1], (
        f"estimated range {est_range} does not cover the true disparity {true_disparity:.1f}px"
    )


def test_stereo_matching_succeeds_with_undistortion_and_auto_range_but_not_with_old_defaults():
    calib, img1, img2 = _render_pair()

    rng = np.random.default_rng(4)
    xs = rng.uniform(100, _IMG_SIZE[0] - 100, 150)
    ys = rng.uniform(100, _IMG_SIZE[1] - 100, 150)
    points1_distorted = np.column_stack([xs, ys])
    true_points2_distorted = np.array([_ground_truth_cam2_pixel(x, y, calib) for x, y in points1_distorted])

    # --- "before": old behavior -- raw distorted images, fixed +/-150px range ---
    _, valid_before, _, diag_before = stereo_match_reference(
        img1, img2, points1_distorted, calib, subset_radius=15,
        disparity_range=(-150.0, 150.0), auto_disparity_range=False,
    )
    before_fraction = valid_before.mean()

    # --- "after": undistort first, auto-derive the disparity range ---
    u1 = calib.cam1.undistort_image(img1)
    u2 = calib.cam2.undistort_image(img2)
    match_calib = calib.undistorted_for_matching()
    points1_undist = cv2.undistortPoints(
        points1_distorted.reshape(-1, 1, 2), calib.cam1.camera_matrix, calib.cam1.dist_coeffs,
        P=calib.cam1.camera_matrix,
    ).reshape(-1, 2)

    points2_after, valid_after, zncc_after, diag_after = stereo_match_reference(
        u1, u2, points1_undist, match_calib, subset_radius=15,
    )
    after_fraction = valid_after.mean()

    print(f"before (raw images, fixed +/-150px range): {before_fraction*100:.1f}% valid")
    print(f"after  (undistorted, auto disparity range): {after_fraction*100:.1f}% valid")
    print(f"rejection reasons (before): {diag_before.rejection_counts}")
    print(f"rejection reasons (after):  {diag_after.rejection_counts}")

    # Thresholds are deliberately conservative: this synthetic renderer's
    # own undistort/distort round-trip introduces small non-affine residual
    # warping that occasionally slows IC-GN convergence within max_iter, so
    # the "after" fraction here is a lower bound, not a ceiling -- a real,
    # well-applied speckle pattern (this test's pattern is generated, not
    # hand-optimized) should do at least as well. The point of this test is
    # the size of the gap, which mirrors a real report of ~1.3% valid points.
    assert before_fraction < 0.1, "sanity check: old approach should indeed fail badly on this rig"
    assert after_fraction > 0.5, "fixed pipeline should recover the majority of points"
    assert after_fraction > before_fraction + 0.3, "fix should be a large, not marginal, improvement"

    # accuracy against known ground truth, for the points that did match
    true_points2_undist = cv2.undistortPoints(
        true_points2_distorted.reshape(-1, 1, 2), calib.cam2.camera_matrix, calib.cam2.dist_coeffs,
        P=calib.cam2.camera_matrix,
    ).reshape(-1, 2)
    ok = valid_after
    err = np.linalg.norm(points2_after[ok] - true_points2_undist[ok], axis=1)
    assert np.mean(err) < 0.5, f"mean matching error {np.mean(err):.3f}px too high vs ground truth"
    print(f"mean subpixel matching error vs ground truth: {np.mean(err):.4f}px (n={ok.sum()})")


def test_epipolar_and_lr_consistency_reject_forced_bad_match(monkeypatch):
    # Force the coarse NCC step to always propose a match far from the true
    # epipolar line (simulating an aliasing/repetitive-texture false peak)
    # and confirm the explicit epipolar-distance + LR-consistency checks
    # reject it rather than accepting a geometrically-inconsistent point.
    calib, img1, img2 = _render_pair()
    u1 = calib.cam1.undistort_image(img1)
    u2 = calib.cam2.undistort_image(img2)
    match_calib = calib.undistorted_for_matching()

    rng = np.random.default_rng(5)
    xs = rng.uniform(100, _IMG_SIZE[0] - 100, 5)
    ys = rng.uniform(100, _IMG_SIZE[1] - 100, 5)
    points1_undist = cv2.undistortPoints(
        np.column_stack([xs, ys]).reshape(-1, 1, 2), calib.cam1.camera_matrix, calib.cam1.dist_coeffs,
        P=calib.cam1.camera_matrix,
    ).reshape(-1, 2)

    import pydic.stereo as stereo_mod
    real_ncc_peak = stereo_mod._ncc_peak

    def bad_peak(template, search):
        real = real_ncc_peak(template, search)
        if real is None:
            return None
        dx, dy, score = real
        return dx + 80.0, dy + 40.0, max(score, 0.9)  # force far off epipolar line

    monkeypatch.setattr(stereo_mod, "_ncc_peak", bad_peak)

    _, valid, _, diag = stereo_match_reference(
        u1, u2, points1_undist, match_calib, subset_radius=15, epipolar_band=6.0,
    )
    assert not np.any(valid), "forced off-epipolar match should have been rejected, not accepted"
    reasons = diag.rejection_counts
    assert any("epipolar" in r or "converge" in r or "consistency" in r for r in reasons)


def test_stereo_match_diagnostics_counts_are_consistent():
    calib, img1, img2 = _render_pair()
    u1 = calib.cam1.undistort_image(img1)
    u2 = calib.cam2.undistort_image(img2)
    match_calib = calib.undistorted_for_matching()

    rng = np.random.default_rng(6)
    xs = rng.uniform(100, _IMG_SIZE[0] - 100, 80)
    ys = rng.uniform(100, _IMG_SIZE[1] - 100, 80)
    points1_undist = cv2.undistortPoints(
        np.column_stack([xs, ys]).reshape(-1, 1, 2), calib.cam1.camera_matrix, calib.cam1.dist_coeffs,
        P=calib.cam1.camera_matrix,
    ).reshape(-1, 2)

    _, valid, _, diag = stereo_match_reference(u1, u2, points1_undist, match_calib, subset_radius=15)
    assert diag.n_total == len(points1_undist)
    assert diag.n_matched == int(valid.sum())
    assert abs(diag.valid_fraction - valid.mean()) < 1e-9
    assert sum(diag.rejection_counts.values()) == diag.n_total - diag.n_matched
