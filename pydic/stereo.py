"""Stereo correspondence at the reference frame, triangulation, and temporal
3D tracking for two-camera (stereo) DIC.

Correspondence strategy: for a point in camera 1, restrict the search in
camera 2 to a band around its epipolar line (from the fundamental matrix
computed at calibration time), get an integer-pixel peak via normalized
cross-correlation, then refine to subpixel with the same IC-GN affine
optimizer used for temporal tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .calibration import StereoCalibration
from .correlation import ImageInterpolator, icgn_correlate, image_gradients
from .subset import build_subset
from .tracking2d import DicPointResult, track_sequence


def _epipolar_line(F: np.ndarray, pt: Tuple[float, float]) -> np.ndarray:
    x, y = pt
    return F @ np.array([x, y, 1.0])  # [a, b, c], a*x + b*y + c = 0


def _epipolar_search_window(
    line: np.ndarray,
    x_range: Tuple[float, float],
    band: float,
) -> Tuple[float, float, float, float]:
    a, b, c = line
    x0, x1 = x_range
    if abs(b) < 1e-9:
        # near-vertical epipolar line: fall back to a fixed vertical band
        x_c = -c / a if abs(a) > 1e-9 else 0.5 * (x0 + x1)
        return x_c - band, x_c + band, -1e9, 1e9
    y0 = -(a * x0 + c) / b
    y1 = -(a * x1 + c) / b
    y_lo, y_hi = min(y0, y1) - band, max(y0, y1) + band
    return min(x0, x1), max(x0, x1), y_lo, y_hi


def stereo_match_reference(
    cam1_gray: np.ndarray,
    cam2_gray: np.ndarray,
    points1: np.ndarray,
    calib: StereoCalibration,
    subset_radius: int = 15,
    disparity_range: Tuple[float, float] = (-150.0, 150.0),
    epipolar_band: float = 6.0,
    max_iter: int = 40,
    tol: float = 1e-4,
    zncc_threshold: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find, for every point1[i] in camera 1's reference image, the matching
    subpixel location in camera 2's reference image.

    Returns (points2 [N,2], valid [N] bool, zncc [N]).
    """
    n = len(points1)
    points2 = np.zeros((n, 2))
    valid = np.zeros(n, dtype=bool)
    zncc = np.full(n, -1.0)

    gx, gy = image_gradients(cam1_gray)
    interp2 = ImageInterpolator(cam2_gray)
    h2, w2 = cam2_gray.shape

    for i in range(n):
        x1, y1 = points1[i]
        line = _epipolar_line(calib.F, (x1, y1))
        x_lo, x_hi, y_lo, y_hi = _epipolar_search_window(
            line, (x1 + disparity_range[0], x1 + disparity_range[1]), epipolar_band
        )
        sx0 = max(0, int(np.floor(x_lo - subset_radius)))
        sy0 = max(0, int(np.floor(y_lo - subset_radius)))
        sx1 = min(w2, int(np.ceil(x_hi + subset_radius)) + 1)
        sy1 = min(h2, int(np.ceil(y_hi + subset_radius)) + 1)

        xi1, yi1 = int(round(x1)), int(round(y1))
        h1, w1 = cam1_gray.shape
        tx0, tx1 = xi1 - subset_radius, xi1 + subset_radius + 1
        ty0, ty1 = yi1 - subset_radius, yi1 + subset_radius + 1
        if tx0 < 0 or ty0 < 0 or tx1 > w1 or ty1 > h1:
            continue
        template = cam1_gray[ty0:ty1, tx0:tx1].astype(np.float32)

        if sx1 - sx0 < template.shape[1] or sy1 - sy0 < template.shape[0]:
            continue
        search = cam2_gray[sy0:sy1, sx0:sx1].astype(np.float32)
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.2:
            continue
        x2_guess = sx0 + max_loc[0] + subset_radius
        y2_guess = sy0 + max_loc[1] + subset_radius

        subset = build_subset(cam1_gray, x1, y1, subset_radius, gx, gy)
        if not subset.valid:
            continue
        p_init = np.array([x2_guess - x1, 0.0, 0.0, y2_guess - y1, 0.0, 0.0])
        res = icgn_correlate(
            subset, interp2, p_init, max_iter=max_iter, tol=tol, zncc_threshold=zncc_threshold
        )
        if not res.converged:
            continue
        points2[i] = (x1 + res.p[0], y1 + res.p[3])
        valid[i] = True
        zncc[i] = res.zncc

    return points2, valid, zncc


def triangulate_points(
    calib: StereoCalibration, pts1: np.ndarray, pts2: np.ndarray
) -> np.ndarray:
    """Triangulate matched pixel points (raw, distorted pixel coordinates) into
    3D points in camera 1's coordinate frame. NaN rows are passed through as
    NaN points.

    Points are undistorted to *normalized* camera coordinates (no P passed to
    undistortPoints) and triangulated with K-free projection matrices
    [I|0] / [R|T]: cv2.undistortPoints' P argument assumes a rectifying
    rotation R was also supplied (the stereoRectify convention), which does
    not hold for the raw, unrectified P1/P2 stored on StereoCalibration.
    """
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 1, 2)

    valid = np.all(np.isfinite(pts1[:, 0, :]), axis=1) & np.all(np.isfinite(pts2[:, 0, :]), axis=1)
    out = np.full((pts1.shape[0], 3), np.nan)
    if not np.any(valid):
        return out

    n1 = cv2.undistortPoints(pts1[valid], calib.cam1.camera_matrix, calib.cam1.dist_coeffs)
    n2 = cv2.undistortPoints(pts2[valid], calib.cam2.camera_matrix, calib.cam2.dist_coeffs)

    P1n = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2n = np.hstack([calib.R, calib.T.reshape(3, 1)])
    homo = cv2.triangulatePoints(P1n, P2n, n1.reshape(-1, 2).T, n2.reshape(-1, 2).T)
    xyz = (homo[:3] / homo[3]).T
    out[valid] = xyz
    return out


@dataclass
class StereoFrameResult:
    points_ref_3d: np.ndarray  # (N, 3) reference 3D positions
    points_cur_3d: np.ndarray  # (N, 3) current 3D positions
    displacement_3d: np.ndarray  # (N, 3) = cur - ref
    valid: np.ndarray
    cam1_results: List[DicPointResult]
    cam2_results: List[DicPointResult]


def track_stereo_sequence(
    calib: StereoCalibration,
    ref_img1: np.ndarray,
    ref_img2: np.ndarray,
    frames1: Sequence[np.ndarray],
    frames2: Sequence[np.ndarray],
    points1: np.ndarray,
    neighbors: Sequence[Sequence[int]],
    subset_radius: int = 15,
    stereo_match_kwargs: Optional[dict] = None,
    track_kwargs: Optional[dict] = None,
    progress_callback: Optional[Callable[[str, int, int, int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[StereoFrameResult]]:
    """Full stereo-DIC pipeline: match cameras at reference, then track and
    triangulate every frame of a synchronized (frames1[t], frames2[t]) pair.

    points1 / neighbors define the measurement grid and its 4-neighbor
    connectivity in camera 1's reference image (see fields.generate_grid /
    fields.build_neighbors); the same connectivity is reused for camera 2's
    tracking since points2[i] corresponds 1:1 to points1[i].

    progress_callback: optional callback(stage, frame_idx, n_frames, n_done,
        n_total_points) where stage is "camera1" or "camera2" — tracking two
        full sequences on a large grid can take a while, so a GUI can use
        this instead of a bare spinner.
    """
    stereo_match_kwargs = stereo_match_kwargs or {}
    track_kwargs = track_kwargs or {}

    points2, stereo_valid, stereo_zncc = stereo_match_reference(
        ref_img1, ref_img2, points1, calib, subset_radius=subset_radius, **stereo_match_kwargs
    )
    ref_points_3d = triangulate_points(calib, points1, points2)
    ref_points_3d[~stereo_valid] = np.nan

    cb1 = (lambda fi, nf, nd, nt: progress_callback("camera1", fi, nf, nd, nt)) if progress_callback else None
    cb2 = (lambda fi, nf, nd, nt: progress_callback("camera2", fi, nf, nd, nt)) if progress_callback else None
    results1 = track_sequence(
        ref_img1, frames1, points1, neighbors, subset_radius, frame_progress_callback=cb1, **track_kwargs
    )
    results2 = track_sequence(
        ref_img2, frames2, points2, neighbors, subset_radius, frame_progress_callback=cb2, **track_kwargs
    )

    frame_results: List[StereoFrameResult] = []
    n = len(points1)
    for r1, r2 in zip(results1, results2):
        cur1 = np.array([[points1[i, 0] + r1[i].u, points1[i, 1] + r1[i].v] for i in range(n)])
        cur2 = np.array([[points2[i, 0] + r2[i].u, points2[i, 1] + r2[i].v] for i in range(n)])

        frame_valid = stereo_valid & np.array([r.converged for r in r1]) & np.array([r.converged for r in r2])
        cur1_masked = np.where(frame_valid[:, None], cur1, np.nan)
        cur2_masked = np.where(frame_valid[:, None], cur2, np.nan)

        cur_points_3d = triangulate_points(calib, cur1_masked, cur2_masked)
        disp = cur_points_3d - ref_points_3d

        frame_results.append(
            StereoFrameResult(
                points_ref_3d=ref_points_3d,
                points_cur_3d=cur_points_3d,
                displacement_3d=disp,
                valid=frame_valid & np.all(np.isfinite(cur_points_3d), axis=1),
                cam1_results=r1,
                cam2_results=r2,
            )
        )

    return points2, stereo_valid, frame_results
