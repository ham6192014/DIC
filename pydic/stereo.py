"""Stereo correspondence at the reference frame, triangulation, and temporal
3D tracking for two-camera (stereo) DIC.

Matching pipeline, in order, for every point1[i] in camera 1's reference
image:
  1. Restrict the search in camera 2 to a band around its epipolar line
     (from the fundamental matrix computed at calibration time) and an
     x-disparity range that is *derived from the data* by default (see
     estimate_disparity_range), not assumed.
  2. Coarse integer-pixel peak via normalized cross-correlation
     (cv2.matchTemplate, TM_CCOEFF_NORMED) within that window.
  3. Subpixel refinement via the same inverse-compositional Gauss-Newton
     (IC-GN) optimizer used for temporal tracking.
  4. Explicit acceptance criteria: distance from the *refined* match to the
     epipolar line (not just the coarse search band), and a right-to-left
     back-match consistency check — both are hard requirements, not just a
     ZNCC threshold, so a locally-plausible but geometrically or physically
     wrong correspondence gets rejected rather than silently accepted.

Every point carries a PointMatchDiagnostic recording exactly what happened
to it (matched, or rejected and why) via StereoMatchDiagnostics, so a run
with few valid points is reported as unreliable rather than presented as a
full-field measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .calibration import StereoCalibration
from .correlation import ImageInterpolator, icgn_correlate, image_gradients
from .subset import build_subset
from .tracking2d import DicPointResult, track_sequence


def _epipolar_line(F: np.ndarray, pt: Tuple[float, float]) -> np.ndarray:
    x, y = pt
    return F @ np.array([x, y, 1.0])  # [a, b, c], a*x + b*y + c = 0


def _epipolar_distance(line: np.ndarray, x: float, y: float) -> float:
    a, b, c = line
    denom = np.hypot(a, b)
    return float(abs(a * x + b * y + c) / denom) if denom > 1e-12 else float("inf")


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


def _ncc_peak(
    template: np.ndarray, search: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """cv2.matchTemplate peak, returning (dx, dy, score) of the peak's
    top-left corner within `search`, or None if search is too small."""
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return float(max_loc[0]), float(max_loc[1]), float(max_val)


def estimate_disparity_range(
    cam1_gray: np.ndarray,
    cam2_gray: np.ndarray,
    points1: np.ndarray,
    calib: StereoCalibration,
    subset_radius: int = 15,
    epipolar_band: float = 6.0,
    n_samples: int = 40,
    min_ncc: float = 0.5,
    margin_fraction: float = 0.2,
    min_margin_px: float = 20.0,
    seed: int = 0,
) -> Optional[Tuple[float, float]]:
    """Data-driven disparity search range: coarse-search a random subsample
    of points across (nearly) the *entire* width of the epipolar band,
    rather than assuming any fixed range, and derive [min, max] from the
    2nd-98th percentile of the disparities that actually matched well.

    Returns None if too few of the sampled points found a confident match
    (e.g. the specimen has very little texture, or the epipolar geometry
    itself is wrong) — callers should fall back to a wide default range and
    flag the run as unreliable rather than trust a range derived from noise.
    """
    n = len(points1)
    if n == 0:
        return None
    h2, w2 = cam2_gray.shape
    h1, w1 = cam1_gray.shape
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=min(n_samples, n), replace=False)

    disparities = []
    for i in sample_idx:
        x1, y1 = points1[i]
        xi1, yi1 = int(round(x1)), int(round(y1))
        tx0, tx1 = xi1 - subset_radius, xi1 + subset_radius + 1
        ty0, ty1 = yi1 - subset_radius, yi1 + subset_radius + 1
        if tx0 < 0 or ty0 < 0 or tx1 > w1 or ty1 > h1:
            continue
        template = cam1_gray[ty0:ty1, tx0:tx1].astype(np.float32)

        line = _epipolar_line(calib.F, (x1, y1))
        # search essentially the full row: we don't know the disparity yet,
        # that's the point of this pass.
        x_lo, x_hi, y_lo, y_hi = _epipolar_search_window(line, (0.0, float(w2)), epipolar_band)
        sx0 = max(0, int(np.floor(x_lo - subset_radius)))
        sy0 = max(0, int(np.floor(y_lo - subset_radius)))
        sx1 = min(w2, int(np.ceil(x_hi + subset_radius)) + 1)
        sy1 = min(h2, int(np.ceil(y_hi + subset_radius)) + 1)
        search = cam2_gray[sy0:sy1, sx0:sx1].astype(np.float32)

        peak = _ncc_peak(template, search)
        if peak is None or peak[2] < min_ncc:
            continue
        dx, dy, _ = peak
        x2 = sx0 + dx + subset_radius
        disparities.append(x2 - x1)

    if len(disparities) < max(5, len(sample_idx) // 4):
        return None

    disparities = np.asarray(disparities)
    lo, hi = np.percentile(disparities, [2, 98])
    margin = max(margin_fraction * (hi - lo), min_margin_px)
    return float(lo - margin), float(hi + margin)


@dataclass
class PointMatchDiagnostic:
    """What happened to one candidate point during stereo matching."""

    index: int
    x1: float
    y1: float
    x2: Optional[float] = None
    y2: Optional[float] = None
    zncc: float = -1.0
    disparity_x: Optional[float] = None
    disparity_y: Optional[float] = None
    epipolar_error_px: Optional[float] = None
    lr_consistency_error_px: Optional[float] = None
    depth_mm: Optional[float] = None
    accepted: bool = False
    reject_reason: Optional[str] = None


@dataclass
class StereoMatchDiagnostics:
    """Full audit trail for one stereo_match_reference() call."""

    points: List[PointMatchDiagnostic]
    disparity_range_used: Tuple[float, float]
    expected_disparity_hint: Optional[Tuple[float, float]] = None

    @property
    def n_total(self) -> int:
        return len(self.points)

    @property
    def n_matched(self) -> int:
        return sum(1 for p in self.points if p.accepted)

    @property
    def valid_fraction(self) -> float:
        return self.n_matched / self.n_total if self.n_total else 0.0

    @property
    def rejection_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in self.points:
            if not p.accepted:
                reason = p.reject_reason or "unknown"
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    def array(self, field_name: str) -> np.ndarray:
        """Convenience: e.g. .array('zncc'), .array('epipolar_error_px')."""
        return np.array(
            [getattr(p, field_name) if getattr(p, field_name) is not None else np.nan for p in self.points]
        )


def stereo_match_reference(
    cam1_gray: np.ndarray,
    cam2_gray: np.ndarray,
    points1: np.ndarray,
    calib: StereoCalibration,
    subset_radius: int = 15,
    disparity_range: Optional[Tuple[float, float]] = None,
    auto_disparity_range: bool = True,
    epipolar_band: float = 6.0,
    max_iter: int = 40,
    tol: float = 1e-4,
    zncc_threshold: float = 0.6,
    lr_consistency_px: float = 1.0,
    max_epipolar_error_px: float = 2.0,
    depth_range_mm: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StereoMatchDiagnostics]:
    """Find, for every point1[i] in camera 1's reference image, the matching
    subpixel location in camera 2's reference image.

    disparity_range: (min, max) offset in x from a cam1 point to search for
        its cam2 correspondence, added to the epipolar-line search band. If
        None (default) and auto_disparity_range is True, this is estimated
        from the data itself (see estimate_disparity_range) instead of
        assuming any fixed window — a fixed range covering the wrong part
        of the image is a common reason stereo matching fails almost
        completely while still looking like it "ran".
    lr_consistency_px: max allowed round-trip error (cam1 -> cam2 -> cam1)
        for a match to be accepted; catches locally-plausible but wrong
        correspondences (repetitive texture, aliasing) that a one-way ZNCC
        check alone would miss.
    max_epipolar_error_px: max allowed distance from the *refined* subpixel
        match to the epipolar line; catches correspondences that drifted
        away from a geometrically consistent solution during IC-GN even
        though the coarse search started on the line.
    depth_range_mm: optional (min, max) plausible triangulated depth (cam1
        Z) — points triangulating outside this range are rejected. Off by
        default (None) since it requires knowing the working distance.

    Returns (points2 [N,2], valid [N] bool, zncc [N], diagnostics).
    """
    n = len(points1)
    points2 = np.full((n, 2), np.nan)
    valid = np.zeros(n, dtype=bool)
    zncc_out = np.full(n, -1.0)
    diags = [PointMatchDiagnostic(index=i, x1=float(points1[i, 0]), y1=float(points1[i, 1])) for i in range(n)]

    expected_hint = None
    try:
        expected_hint = calib.expected_disparity_range_px(50.0, 5000.0)
    except Exception:
        expected_hint = None

    if disparity_range is None:
        if auto_disparity_range:
            disparity_range = estimate_disparity_range(
                cam1_gray, cam2_gray, points1, calib, subset_radius, epipolar_band
            )
        if disparity_range is None:
            # Fall back to (nearly) the whole image width rather than a
            # narrow guess: slower, but won't silently miss correspondences
            # outside an assumed window.
            w2 = cam2_gray.shape[1]
            disparity_range = (-0.9 * w2, 0.9 * w2)

    gx1, gy1 = image_gradients(cam1_gray)
    gx2, gy2 = image_gradients(cam2_gray)
    interp1 = ImageInterpolator(cam1_gray)
    interp2 = ImageInterpolator(cam2_gray)
    h1, w1 = cam1_gray.shape
    h2, w2 = cam2_gray.shape

    for i in range(n):
        x1, y1 = points1[i]
        d = diags[i]
        line = _epipolar_line(calib.F, (x1, y1))
        x_lo, x_hi, y_lo, y_hi = _epipolar_search_window(
            line, (x1 + disparity_range[0], x1 + disparity_range[1]), epipolar_band
        )
        sx0 = max(0, int(np.floor(x_lo - subset_radius)))
        sy0 = max(0, int(np.floor(y_lo - subset_radius)))
        sx1 = min(w2, int(np.ceil(x_hi + subset_radius)) + 1)
        sy1 = min(h2, int(np.ceil(y_hi + subset_radius)) + 1)

        xi1, yi1 = int(round(x1)), int(round(y1))
        tx0, tx1 = xi1 - subset_radius, xi1 + subset_radius + 1
        ty0, ty1 = yi1 - subset_radius, yi1 + subset_radius + 1
        if tx0 < 0 or ty0 < 0 or tx1 > w1 or ty1 > h1:
            d.reject_reason = "subset extends outside camera-1 image"
            continue
        template = cam1_gray[ty0:ty1, tx0:tx1].astype(np.float32)

        search = cam2_gray[sy0:sy1, sx0:sx1].astype(np.float32)
        peak = _ncc_peak(template, search)
        if peak is None:
            d.reject_reason = "epipolar search window too small (near image border)"
            continue
        dx, dy, max_val = peak
        if max_val < 0.2:
            d.reject_reason = f"no plausible initial match (peak NCC={max_val:.2f})"
            continue
        x2_guess = sx0 + dx + subset_radius
        y2_guess = sy0 + dy + subset_radius

        subset1 = build_subset(cam1_gray, x1, y1, subset_radius, gx1, gy1)
        if not subset1.valid:
            d.reject_reason = "low-texture subset in camera 1 (flat/featureless region)"
            continue
        p_init = np.array([x2_guess - x1, 0.0, 0.0, y2_guess - y1, 0.0, 0.0])
        res = icgn_correlate(
            subset1, interp2, p_init, max_iter=max_iter, tol=tol, zncc_threshold=zncc_threshold
        )
        d.zncc = res.zncc
        if not res.converged:
            d.reject_reason = f"IC-GN did not converge (ZNCC={res.zncc:.3f})"
            continue

        x2 = x1 + res.p[0]
        y2 = y1 + res.p[3]
        d.x2, d.y2 = x2, y2
        d.disparity_x, d.disparity_y = x2 - x1, y2 - y1

        epi_err = _epipolar_distance(line, x2, y2)
        d.epipolar_error_px = epi_err
        if epi_err > max_epipolar_error_px:
            d.reject_reason = f"epipolar residual {epi_err:.2f}px exceeds {max_epipolar_error_px}px"
            continue

        subset2 = build_subset(cam2_gray, x2, y2, subset_radius, gx2, gy2)
        if not subset2.valid:
            d.reject_reason = "low-texture subset in camera 2 (flat/featureless region)"
            continue
        p_back_init = np.array([x1 - x2, 0.0, 0.0, y1 - y2, 0.0, 0.0])
        res_back = icgn_correlate(
            subset2, interp1, p_back_init, max_iter=max_iter, tol=tol, zncc_threshold=zncc_threshold
        )
        if not res_back.converged:
            d.reject_reason = "left-right consistency check did not converge"
            continue
        x1_back, y1_back = x2 + res_back.p[0], y2 + res_back.p[3]
        lr_err = float(np.hypot(x1_back - x1, y1_back - y1))
        d.lr_consistency_error_px = lr_err
        if lr_err > lr_consistency_px:
            d.reject_reason = f"left-right consistency error {lr_err:.2f}px exceeds {lr_consistency_px}px"
            continue

        if depth_range_mm is not None:
            xyz = triangulate_points(calib, np.array([[x1, y1]]), np.array([[x2, y2]]))[0]
            depth = float(xyz[2])
            d.depth_mm = depth
            if not np.isfinite(depth) or not (depth_range_mm[0] <= depth <= depth_range_mm[1]):
                d.reject_reason = f"triangulated depth {depth:.1f}mm outside plausible range {depth_range_mm}"
                continue

        points2[i] = (x2, y2)
        valid[i] = True
        zncc_out[i] = res.zncc
        d.accepted = True

    diagnostics = StereoMatchDiagnostics(
        points=diags, disparity_range_used=disparity_range, expected_disparity_hint=expected_hint
    )
    return points2, valid, zncc_out, diagnostics


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

    If the images being matched were already undistorted (see
    CameraCalibration.undistort_image / StereoDic's default behavior), pass
    a `calib` whose dist_coeffs are zero (StereoCalibration.undistorted_for_matching())
    so this step doesn't try to remove distortion twice.
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
) -> Tuple[np.ndarray, np.ndarray, List[StereoFrameResult], StereoMatchDiagnostics]:
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

    Returns (points2, stereo_valid, frame_results, match_diagnostics).
    """
    stereo_match_kwargs = stereo_match_kwargs or {}
    track_kwargs = track_kwargs or {}

    points2, stereo_valid, stereo_zncc, match_diag = stereo_match_reference(
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

    return points2, stereo_valid, frame_results, match_diag
