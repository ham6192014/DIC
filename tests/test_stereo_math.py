import cv2
import numpy as np

from pydic.calibration import CameraCalibration, StereoCalibration
from pydic.fields import build_neighbors, generate_grid
from pydic.stereo import _epipolar_line, triangulate_points
from pydic.strain import compute_strain_3d_surface


def _make_stereo_calib():
    K = np.array([[1000.0, 0, 320], [0, 1000.0, 240], [0, 0, 1]])
    D = np.zeros(5)
    cam1 = CameraCalibration(K, D, (640, 480), 0.1)
    cam2 = CameraCalibration(K, D, (640, 480), 0.1)
    R = cv2.Rodrigues(np.array([0.01, 0.03, 0.0]))[0]
    T = np.array([[-100.0], [2.0], [3.0]])
    Tx = np.array([[0, -T[2, 0], T[1, 0]], [T[2, 0], 0, -T[0, 0]], [-T[1, 0], T[0, 0], 0]])
    E = Tx @ R
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    return StereoCalibration(cam1, cam2, R, T, E, F, 0.1)


def test_triangulation_round_trip():
    calib = _make_stereo_calib()
    rng = np.random.default_rng(1)
    X = rng.uniform(-200, 200, (20, 3)) + np.array([0, 0, 1000])
    Xh = np.hstack([X, np.ones((20, 1))])
    x1 = (calib.P1 @ Xh.T).T
    x1 = x1[:, :2] / x1[:, 2:3]
    x2 = (calib.P2 @ Xh.T).T
    x2 = x2[:, :2] / x2[:, 2:3]

    Xrec = triangulate_points(calib, x1, x2)
    assert np.max(np.abs(Xrec - X)) < 1e-6


def test_epipolar_constraint_satisfied_by_construction():
    calib = _make_stereo_calib()
    rng = np.random.default_rng(2)
    X = rng.uniform(-200, 200, (10, 3)) + np.array([0, 0, 1000])
    Xh = np.hstack([X, np.ones((10, 1))])
    x1 = (calib.P1 @ Xh.T).T
    x1 = x1[:, :2] / x1[:, 2:3]
    x2 = (calib.P2 @ Xh.T).T
    x2 = x2[:, :2] / x2[:, 2:3]

    for i in range(len(x1)):
        line = _epipolar_line(calib.F, x1[i])
        residual = line[0] * x2[i, 0] + line[1] * x2[i, 1] + line[2]
        assert abs(residual) < 1e-8


def test_strain_3d_surface_matches_known_planar_stretch():
    points2d, index_of = generate_grid((400, 400), step=20, subset_radius=15)
    neighbors = build_neighbors(index_of)
    n = len(points2d)
    ref3d = np.column_stack([points2d[:, 0], points2d[:, 1], np.zeros(n)])

    F = np.array([[1.03, 0.01], [0.02, 0.98]])
    center = points2d.mean(axis=0)
    cur2d = (F @ (points2d - center).T).T + center
    cur3d = np.column_stack([cur2d[:, 0], cur2d[:, 1], np.zeros(n)])

    valid = np.ones(n, dtype=bool)
    strain = compute_strain_3d_surface(ref3d, cur3d, valid, neighbors, window_hops=2)

    E_true = 0.5 * (F.T @ F - np.eye(2))
    assert abs(np.nanmean(strain.exx) - E_true[0, 0]) < 1e-6
    assert abs(np.nanmean(strain.eyy) - E_true[1, 1]) < 1e-6
    assert abs(np.nanmean(strain.exy) - E_true[0, 1]) < 1e-6
