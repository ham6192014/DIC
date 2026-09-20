"""End-to-end synthetic validation of the stereo-DIC pipeline: two virtual
pinhole cameras view a flat speckle plane through a homography; the plane is
then given a known in-plane affine stretch and re-rendered. We run full
stereo matching + triangulation + temporal tracking + 3D strain and compare
against the analytically known Green-Lagrange strain of the applied stretch.

Run directly: python3 tests/test_stereo_end_to_end.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from pydic.calibration import CameraCalibration, StereoCalibration
from pydic.fields import build_neighbors, generate_grid
from pydic.pipeline import StereoDic
from tests.synthetic import make_speckle_image


def render_view(texture, tex_to_world, cam_matrix, R, T, world_z, out_size):
    # tex(px) -> world(X,Y) -> camera pixel, all as 3x3 homographies (plane at world_z)
    M_plane = np.array([[1, 0, 0], [0, 1, 0], [0, 0, world_z], [0, 0, 1]])
    P = cam_matrix @ np.hstack([R, T.reshape(3, 1)])
    H_world_to_cam = P @ M_plane  # 3x3
    H = H_world_to_cam @ tex_to_world
    H = H / H[2, 2]
    return cv2.warpPerspective(texture, H, out_size, flags=cv2.INTER_CUBIC, borderValue=40)


def test_full_stereo_pipeline_matches_known_planar_strain():
    rng_seed = 42
    texture = make_speckle_image(size=(1600, 1600), n_speckles=25000, seed=rng_seed)
    tex_scale = 0.5  # mm per texture pixel
    tex_center = np.array([800, 800])
    tex_to_world = np.array(
        [[tex_scale, 0, -tex_center[0] * tex_scale],
         [0, tex_scale, -tex_center[1] * tex_scale],
         [0, 0, 1]]
    )

    K = np.array([[3000.0, 0, 640], [0, 3000.0, 480], [0, 0, 1]])
    D = np.zeros(5)
    img_size = (1280, 960)
    world_z = 800.0  # mm, plane distance from camera 1

    R1, T1 = np.eye(3), np.zeros(3)
    R2 = cv2.Rodrigues(np.array([0.0, 0.25, 0.0]))[0]
    T2 = np.array([-120.0, 0.0, 0.0])

    calib = StereoCalibration(
        cam1=CameraCalibration(K, D, img_size, 0.05),
        cam2=CameraCalibration(K, D, img_size, 0.05),
        R=R2, T=T2,
        E=np.zeros((3, 3)), F=np.zeros((3, 3)),  # not used by triangulation/pipeline math
        rms_error=0.05,
    )
    # F IS used by stereo matching for the epipolar search band -> compute properly.
    Tx = np.array([[0, -T2[2], T2[1]], [T2[2], 0, -T2[0]], [-T2[1], T2[0], 0]])
    E = Tx @ R2
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    calib.F = F

    ref1 = render_view(texture, tex_to_world, K, R1, T1, world_z, img_size)
    ref2 = render_view(texture, tex_to_world, K, R2, T2, world_z, img_size)

    # ground-truth in-plane affine stretch of the physical plane (world mm coords)
    Faff = np.array([[1.02, 0.008], [0.005, 0.985]])
    center_world = np.array([0.0, 0.0])

    def deformed_tex_to_world(t):
        w = (tex_to_world @ np.array([t[0], t[1], 1.0]))[:2]
        w_def = Faff @ (w - center_world) + center_world
        return w_def

    # deformed_tex_to_world is affine in tex coords too -> build as 3x3 homography directly
    A = tex_to_world[:2, :2]
    b = tex_to_world[:2, 2]
    A_def = Faff @ A
    b_def = Faff @ (b - center_world) + center_world
    tex_to_world_def = np.array(
        [[A_def[0, 0], A_def[0, 1], b_def[0]],
         [A_def[1, 0], A_def[1, 1], b_def[1]],
         [0, 0, 1]]
    )

    cur1 = render_view(texture, tex_to_world_def, K, R1, T1, world_z, img_size)
    cur2 = render_view(texture, tex_to_world_def, K, R2, T2, world_z, img_size)

    dic = StereoDic(
        calib, subset_radius=21, grid_step=40, zncc_threshold=0.6,
        disparity_range=(-700.0, 700.0), epipolar_band=10.0,
    )
    # ROI: stay comfortably inside the rendered/valid region of camera 1
    roi = [(300, 200), (980, 200), (980, 760), (300, 760)]
    dic.set_reference(ref1, ref2, roi_polygon=roi)
    print(f"grid points: {len(dic.points1)}")

    frame_results = dic.run_sequence([cur1], [cur2])
    fr = frame_results[0]
    print(f"valid fraction: {fr.valid.mean():.3f}")

    strain = dic.compute_strain(fr, window_hops=2)
    E_true = 0.5 * (Faff.T @ Faff - np.eye(2))
    print("true  Exx,Eyy,Exy:", E_true[0, 0], E_true[1, 1], E_true[0, 1])
    print(
        "recovered exx mean/std:", np.nanmean(strain.exx), np.nanstd(strain.exx[strain.valid])
    )
    print(
        "recovered eyy mean/std:", np.nanmean(strain.eyy), np.nanstd(strain.eyy[strain.valid])
    )
    print(
        "recovered exy mean/std:", np.nanmean(strain.exy), np.nanstd(strain.exy[strain.valid])
    )

    assert fr.valid.mean() > 0.8, "too many points failed to track/triangulate"
    assert abs(np.nanmean(strain.exx) - E_true[0, 0]) < 2e-3
    assert abs(np.nanmean(strain.eyy) - E_true[1, 1]) < 2e-3
    assert abs(np.nanmean(strain.exy) - E_true[0, 1]) < 2e-3
    print("OK: stereo-DIC end-to-end strain matches ground truth within tolerance")


if __name__ == "__main__":
    test_full_stereo_pipeline_matches_known_planar_strain()
