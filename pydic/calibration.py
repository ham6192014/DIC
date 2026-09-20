"""Chessboard-based single and stereo camera calibration."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class CalibrationQualityError(RuntimeError):
    """Raised when a stereo calibration fails a minimum quality bar.

    The rejected StereoCalibration (with its full per-pair diagnostics
    report) is still attached as `.calibration`, so a caller — a GUI in
    particular — can show *why* it failed rather than just that it did.
    Deliberately a RuntimeError subclass: existing `except RuntimeError`
    call sites (e.g. min-views checks) keep working unchanged.
    """

    def __init__(
        self,
        message: str,
        calibration: "StereoCalibration" = None,
        report: "StereoCalibrationReport" = None,
    ):
        super().__init__(message)
        self.calibration = calibration
        self.report = report if report is not None else (calibration.report if calibration else None)


def find_chessboard_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    refine_win: Tuple[int, int] = (11, 11),
    max_detect_dim: int = 1600,
) -> Tuple[bool, Optional[np.ndarray], Optional[bool]]:
    """Locate inner chessboard corners in a grayscale/BGR image.

    pattern_size is (n_cols, n_rows) of *inner* corners (squares - 1).
    Returns (found, corners, was_flipped): corners has shape (N, 1, 2)
    float32, subpixel refined on the full-resolution image; was_flipped
    reports whether corner-order canonicalization (see
    _canonicalize_corner_order) reversed the detector's raw order — surfaced
    for per-pair diagnostics, not needed for normal use.

    Detection itself runs on a downscaled copy (`max_detect_dim` on the long
    side): OpenCV's chessboard detectors are tuned for, and much faster and
    more reliable on, moderate resolutions — very large source photos (tens
    of megapixels) can make the classical detector miss the board entirely
    or take a very long time. The modern findChessboardCornersSB detector is
    tried first (noticeably more robust to uneven lighting/blur/perspective
    than the classical one) with a fallback to it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    scale = min(1.0, max_detect_dim / max(h, w))
    small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray

    found, corners = cv2.findChessboardCornersSB(
        small, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(small, pattern_size, flags=flags)
    if not found:
        return False, None, None

    corners = (corners / scale).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, refine_win, (-1, -1), criteria)
    corners, was_flipped = _canonicalize_corner_order(gray, corners, pattern_size)
    return True, corners, was_flipped


def _square_sample(gray: np.ndarray, corner_a, corner_b, corner_c, corner_d) -> float:
    """Mean intensity of a small patch at the centroid of 4 corners bounding
    one checkerboard square."""
    center = (corner_a + corner_b + corner_c + corner_d) / 4.0
    x, y = int(round(center[0])), int(round(center[1]))
    h, w = gray.shape
    x0, x1 = max(0, x - 3), min(w, x + 4)
    y0, y1 = max(0, y - 3), min(h, y + 4)
    patch = gray[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size > 0 else 128.0


def _canonicalize_corner_order(
    gray: np.ndarray, corners: np.ndarray, pattern_size: Tuple[int, int]
) -> Tuple[np.ndarray, bool]:
    """Fix the classic checkerboard 180-degree labeling ambiguity.

    Returns (corners, was_flipped).

    A plain checkerboard looks identical rotated 180 degrees, so the
    detector's choice of "corner index 0" is only fixed relative to how the
    board happened to appear in *that* image — nothing stops it from picking
    the diagonally opposite physical corner in another image of the same
    pose (this is exactly what caused a real stereo calibration to fail with
    a huge RMS despite both individual cameras calibrating fine: reprojection
    error within one camera doesn't care about labeling, but stereoCalibrate
    needs corner k in the left image and corner k in the right image to be
    the *same physical point*, for every pose).

    Fixed with a property that doesn't depend on viewpoint at all: the two
    checkerboard squares immediately adjacent to corner 0 (in the +col and
    +row directions) always alternate black/white — a fact about the
    physical board, not the photo. We compare *those two samples against
    each other* rather than against a fixed absolute brightness threshold:
    real camera captures (different exposure per shot, per-camera gain,
    uneven lighting across a stereo rig) can easily put a "black" square
    above 128 in one photo and a "white" square below it in another, which
    would silently break an absolute-threshold test. Whichever of the two
    adjacent squares is *darker relative to the other, in that same photo*
    is treated as the reference "black" square, so the decision is immune
    to the overall brightness of any individual image.
    """
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)

    # Square immediately right of corner (0,0): cell (row 0, col 0).
    sample_a = _square_sample(gray, grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1])
    if cols >= 3:
        # Neighboring cell (row 0, col 1) — opposite color by construction.
        sample_b = _square_sample(gray, grid[0, 1], grid[0, 2], grid[1, 1], grid[1, 2])
    elif rows >= 3:
        # Board too narrow for a column neighbor; use the row neighbor instead.
        sample_b = _square_sample(gray, grid[1, 0], grid[1, 1], grid[2, 0], grid[2, 1])
    else:
        # Degenerate (2x2 corners): no in-image neighbor to compare against;
        # fall back to a plain absolute-brightness heuristic.
        sample_b = 255.0 - sample_a
    is_a_dark = sample_a < sample_b
    flipped = not is_a_dark
    if flipped:
        grid = grid[::-1, ::-1, :]
    return grid.reshape(rows * cols, 1, 2).astype(np.float32), flipped


def preview_chessboard_detection(
    image_paths: Sequence[str], pattern_size: Tuple[int, int]
) -> List[dict]:
    """Run detection on each image without calibrating, for UI diagnostics.

    Returns one dict per path: {path, found, image (BGR np.ndarray or None
    if unreadable), corners (or None)}. Draw with cv2.drawChessboardCorners
    to show the user exactly what was (or wasn't) detected.
    """
    results = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            results.append({"path": path, "found": False, "image": None, "corners": None, "flipped": None})
            continue
        found, corners, flipped = find_chessboard_corners(img, pattern_size)
        results.append({"path": path, "found": found, "image": img, "corners": corners, "flipped": flipped})
    return results


def _object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


def _extract_numeric_id(path: str) -> Optional[int]:
    """Pull the sequence number out of a filename like 'L10.png' -> 10.

    Takes the *last* run of digits in the filename stem, which matches
    common conventions (a trailing frame/pose index) even when the
    filename has other digits earlier (e.g. a camera or rig ID).
    """
    digits = re.findall(r"\d+", Path(path).stem)
    return int(digits[-1]) if digits else None


@dataclass
class PairMatch:
    """Result of matching left/right chessboard images by filename number."""

    pairs: List[Tuple[int, str, str]]  # (numeric_id, left_path, right_path), sorted by id
    unmatched_left: List[str]
    unmatched_right: List[str]
    duplicate_left_ids: Dict[int, List[str]]
    duplicate_right_ids: Dict[int, List[str]]
    unparsed_left: List[str]  # no numeric id could be extracted
    unparsed_right: List[str]

    def to_dict(self) -> dict:
        return {
            "pairs": [{"id": i, "left": l, "right": r} for i, l, r in self.pairs],
            "unmatched_left": self.unmatched_left,
            "unmatched_right": self.unmatched_right,
            "duplicate_left_ids": {str(k): v for k, v in self.duplicate_left_ids.items()},
            "duplicate_right_ids": {str(k): v for k, v in self.duplicate_right_ids.items()},
            "unparsed_left": self.unparsed_left,
            "unparsed_right": self.unparsed_right,
        }


def match_stereo_pairs_by_filename(
    left_paths: Sequence[str], right_paths: Sequence[str]
) -> PairMatch:
    """Pair left/right chessboard images by the numeric ID in their filename
    (L10.png <-> R10.png) instead of trusting list position/order.

    This is the correctness-critical step for stereo calibration: if the
    two lists were ever built independently (different upload order,
    different glob results, a missing file on one side), silently zipping
    them by index pairs the wrong images together, and every downstream
    number — reprojection error, baseline, everything — becomes meaningless
    without ever raising an error.
    """
    left_map: Dict[int, List[str]] = defaultdict(list)
    right_map: Dict[int, List[str]] = defaultdict(list)
    unparsed_left, unparsed_right = [], []

    for p in left_paths:
        nid = _extract_numeric_id(p)
        (unparsed_left if nid is None else left_map[nid]).append(p)
    for p in right_paths:
        nid = _extract_numeric_id(p)
        (unparsed_right if nid is None else right_map[nid]).append(p)

    duplicate_left = {k: v for k, v in left_map.items() if len(v) > 1}
    duplicate_right = {k: v for k, v in right_map.items() if len(v) > 1}

    common_ids = sorted(set(left_map) & set(right_map))
    pairs = [
        (nid, left_map[nid][0], right_map[nid][0])
        for nid in common_ids
        if len(left_map[nid]) == 1 and len(right_map[nid]) == 1
    ]

    unmatched_left = sorted(
        p for nid, paths in left_map.items() if nid not in right_map for p in paths
    )
    unmatched_right = sorted(
        p for nid, paths in right_map.items() if nid not in left_map for p in paths
    )

    return PairMatch(
        pairs=pairs,
        unmatched_left=unmatched_left,
        unmatched_right=unmatched_right,
        duplicate_left_ids=duplicate_left,
        duplicate_right_ids=duplicate_right,
        unparsed_left=unparsed_left,
        unparsed_right=unparsed_right,
    )


@dataclass
class PairDiagnostic:
    """Per-stereo-pair calibration diagnostics: what was detected, whether
    corner order needed flipping, and (once available) reprojection error."""

    numeric_id: Optional[int]
    left_path: str
    right_path: str
    left_found: bool
    right_found: bool
    left_flipped: Optional[bool] = None
    right_flipped: Optional[bool] = None
    used: bool = False
    reject_reason: Optional[str] = None
    left_reproj_error_px: Optional[float] = None
    right_reproj_error_px: Optional[float] = None

    def to_dict(self) -> dict:
        return dict(
            numeric_id=self.numeric_id, left_path=self.left_path, right_path=self.right_path,
            left_found=self.left_found, right_found=self.right_found,
            left_flipped=self.left_flipped, right_flipped=self.right_flipped,
            used=self.used, reject_reason=self.reject_reason,
            left_reproj_error_px=self.left_reproj_error_px,
            right_reproj_error_px=self.right_reproj_error_px,
        )

    @staticmethod
    def from_dict(d: dict) -> "PairDiagnostic":
        return PairDiagnostic(**d)


@dataclass
class StereoCalibrationReport:
    """Full audit trail for one calibrate_stereo() run."""

    pairs: List[PairDiagnostic]
    unmatched_left: List[str]
    unmatched_right: List[str]
    duplicate_left_ids: Dict[int, List[str]]
    duplicate_right_ids: Dict[int, List[str]]
    unparsed_left: List[str]
    unparsed_right: List[str]

    @property
    def used_pairs(self) -> List[PairDiagnostic]:
        return [p for p in self.pairs if p.used]

    @property
    def rejected_pairs(self) -> List[PairDiagnostic]:
        return [p for p in self.pairs if not p.used]

    def to_dict(self) -> dict:
        return dict(
            pairs=[p.to_dict() for p in self.pairs],
            unmatched_left=self.unmatched_left, unmatched_right=self.unmatched_right,
            duplicate_left_ids={str(k): v for k, v in self.duplicate_left_ids.items()},
            duplicate_right_ids={str(k): v for k, v in self.duplicate_right_ids.items()},
            unparsed_left=self.unparsed_left, unparsed_right=self.unparsed_right,
        )

    @staticmethod
    def from_dict(d: dict) -> "StereoCalibrationReport":
        return StereoCalibrationReport(
            pairs=[PairDiagnostic.from_dict(p) for p in d["pairs"]],
            unmatched_left=d["unmatched_left"], unmatched_right=d["unmatched_right"],
            duplicate_left_ids={int(k): v for k, v in d["duplicate_left_ids"].items()},
            duplicate_right_ids={int(k): v for k, v in d["duplicate_right_ids"].items()},
            unparsed_left=d["unparsed_left"], unparsed_right=d["unparsed_right"],
        )


def _stereo_reprojection_errors(
    objp: np.ndarray,
    img_points_l: Sequence[np.ndarray],
    img_points_r: Sequence[np.ndarray],
    K1: np.ndarray, D1: np.ndarray, K2: np.ndarray, D2: np.ndarray,
    R: np.ndarray, T: np.ndarray,
) -> List[Tuple[float, float]]:
    """Per-view (left_rms_px, right_rms_px) reprojection error.

    cv2.stereoCalibrate only returns one aggregate RMS over every point of
    every view, which hides a single bad pair inside a good average. Per
    view error is computed independently here via solvePnP (using the
    view's own left-camera pose) plus projectPoints, composing the right
    camera's pose from the fixed stereo (R, T) rather than re-solving it —
    exactly what "reprojection error for this specific pair" should mean.
    """
    errors = []
    for pts_l, pts_r in zip(img_points_l, img_points_r):
        ok_l, rvec_l, tvec_l = cv2.solvePnP(objp, pts_l, K1, D1)
        if not ok_l:
            errors.append((float("nan"), float("nan")))
            continue
        proj_l, _ = cv2.projectPoints(objp, rvec_l, tvec_l, K1, D1)
        err_l = float(np.sqrt(np.mean(np.sum((proj_l.reshape(-1, 2) - pts_l.reshape(-1, 2)) ** 2, axis=1))))

        R_l, _ = cv2.Rodrigues(rvec_l)
        R_r = R @ R_l
        t_r = R @ tvec_l.reshape(3, 1) + T.reshape(3, 1)
        rvec_r, _ = cv2.Rodrigues(R_r)
        proj_r, _ = cv2.projectPoints(objp, rvec_r, t_r, K2, D2)
        err_r = float(np.sqrt(np.mean(np.sum((proj_r.reshape(-1, 2) - pts_r.reshape(-1, 2)) ** 2, axis=1))))
        errors.append((err_l, err_r))
    return errors


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    rms_error: float
    rvecs: List[np.ndarray] = field(default_factory=list)
    tvecs: List[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size),
            "rms_error": self.rms_error,
        }

    @staticmethod
    def from_dict(d: dict) -> "CameraCalibration":
        return CameraCalibration(
            camera_matrix=np.array(d["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(d["dist_coeffs"], dtype=np.float64),
            image_size=tuple(d["image_size"]),
            rms_error=d["rms_error"],
        )


def calibrate_single_camera(
    image_paths: Sequence[str],
    pattern_size: Tuple[int, int],
    square_size: float,
    min_valid_views: int = 5,
) -> CameraCalibration:
    """Calibrate one camera from a list of chessboard image paths."""
    objp = _object_points(pattern_size, square_size)
    obj_points, img_points = [], []
    image_size = None
    used = 0
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        found, corners, _flipped = find_chessboard_corners(img, pattern_size)
        if not found:
            continue
        obj_points.append(objp)
        img_points.append(corners)
        used += 1
    if used < min_valid_views:
        raise RuntimeError(
            f"Only {used} valid chessboard views found (need >= {min_valid_views})."
        )
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return CameraCalibration(K, dist, image_size, rms, list(rvecs), list(tvecs))


@dataclass
class StereoCalibration:
    cam1: CameraCalibration
    cam2: CameraCalibration
    R: np.ndarray  # rotation cam1 -> cam2
    T: np.ndarray  # translation cam1 -> cam2 (3,1)
    E: np.ndarray
    F: np.ndarray
    rms_error: float
    P1: np.ndarray = None
    P2: np.ndarray = None
    quality_ok: bool = True
    quality_issues: List[str] = field(default_factory=list)
    report: Optional[StereoCalibrationReport] = None

    def __post_init__(self):
        if self.P1 is None:
            self.P1 = self.cam1.camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
        if self.P2 is None:
            self.P2 = self.cam2.camera_matrix @ np.hstack([self.R, self.T.reshape(3, 1)])

    @property
    def baseline_mm(self) -> float:
        """Magnitude of the translation between the two camera centers."""
        return float(np.linalg.norm(self.T))

    def save(self, path: str) -> None:
        d = {
            "cam1": self.cam1.to_dict(),
            "cam2": self.cam2.to_dict(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "E": self.E.tolist(),
            "F": self.F.tolist(),
            "rms_error": self.rms_error,
            "quality_ok": self.quality_ok,
            "quality_issues": self.quality_issues,
            "report": self.report.to_dict() if self.report is not None else None,
        }
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def load(path: str) -> "StereoCalibration":
        d = json.loads(Path(path).read_text())
        report = d.get("report")
        return StereoCalibration(
            cam1=CameraCalibration.from_dict(d["cam1"]),
            cam2=CameraCalibration.from_dict(d["cam2"]),
            R=np.array(d["R"], dtype=np.float64),
            T=np.array(d["T"], dtype=np.float64),
            E=np.array(d["E"], dtype=np.float64),
            F=np.array(d["F"], dtype=np.float64),
            rms_error=d["rms_error"],
            quality_ok=d.get("quality_ok", True),
            quality_issues=d.get("quality_issues", []),
            report=StereoCalibrationReport.from_dict(report) if report else None,
        )


def calibrate_stereo(
    left_image_paths: Sequence[str],
    right_image_paths: Sequence[str],
    pattern_size: Tuple[int, int],
    square_size: float,
    min_valid_views: int = 5,
    fix_intrinsics: bool = True,
    max_rms_error: float = 1.0,
    max_pair_reproj_error: float = 2.0,
    raise_on_poor_quality: bool = True,
) -> StereoCalibration:
    """Calibrate a stereo pair from chessboard image paths.

    Left/right images are matched by the numeric ID in their filename
    (L10.png <-> R10.png), not by list position/order — see
    match_stereo_pairs_by_filename. If no filenames carry a parseable
    numeric ID, falls back to positional pairing (requires equal-length
    lists) so arbitrary naming schemes still work, just without that
    verification.

    Returns a StereoCalibration whose `.report` holds full per-pair
    diagnostics (corner-orientation flips, per-pair reprojection error,
    rejected images, filename-matching issues). By default, a calibration
    that fails the quality gate (too few valid pairs, RMS or any per-pair
    error above threshold) raises CalibrationQualityError rather than
    silently returning something a 3D-DIC run could use — the rejected
    calibration and its report are still attached to the exception
    (`.calibration`, `.report`) for inspection. Pass
    raise_on_poor_quality=False to get the object back unconditionally and
    check `.quality_ok` / `.quality_issues` yourself.
    """
    match = match_stereo_pairs_by_filename(left_image_paths, right_image_paths)
    if match.pairs:
        ordered = match.pairs
    else:
        if len(left_image_paths) != len(right_image_paths):
            raise ValueError(
                "Could not match left/right images by a numeric filename (e.g. "
                "L03.png / R03.png), and the lists have different lengths so "
                "positional pairing isn't safe either."
            )
        ordered = [(None, lp, rp) for lp, rp in zip(left_image_paths, right_image_paths)]

    objp = _object_points(pattern_size, square_size)
    obj_points, img_points_l, img_points_r = [], [], []
    image_size = None
    pair_diags: List[PairDiagnostic] = []

    for nid, lp, rp in ordered:
        img_l = cv2.imread(str(lp))
        img_r = cv2.imread(str(rp))
        if img_l is None or img_r is None:
            pair_diags.append(PairDiagnostic(
                numeric_id=nid, left_path=str(lp), right_path=str(rp),
                left_found=img_l is not None, right_found=img_r is not None,
                used=False, reject_reason="could not read one or both image files",
            ))
            continue
        if image_size is None:
            image_size = (img_l.shape[1], img_l.shape[0])

        found_l, corners_l, flipped_l = find_chessboard_corners(img_l, pattern_size)
        found_r, corners_r, flipped_r = find_chessboard_corners(img_r, pattern_size)

        if not (found_l and found_r):
            reasons = []
            if not found_l:
                reasons.append("left: chessboard not detected")
            if not found_r:
                reasons.append("right: chessboard not detected")
            pair_diags.append(PairDiagnostic(
                numeric_id=nid, left_path=str(lp), right_path=str(rp),
                left_found=found_l, right_found=found_r,
                left_flipped=flipped_l, right_flipped=flipped_r,
                used=False, reject_reason="; ".join(reasons),
            ))
            continue

        obj_points.append(objp)
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)
        pair_diags.append(PairDiagnostic(
            numeric_id=nid, left_path=str(lp), right_path=str(rp),
            left_found=True, right_found=True,
            left_flipped=flipped_l, right_flipped=flipped_r, used=True,
        ))

    report = StereoCalibrationReport(
        pairs=pair_diags,
        unmatched_left=match.unmatched_left, unmatched_right=match.unmatched_right,
        duplicate_left_ids=match.duplicate_left_ids, duplicate_right_ids=match.duplicate_right_ids,
        unparsed_left=match.unparsed_left, unparsed_right=match.unparsed_right,
    )

    # Below this, cv2.calibrateCamera/stereoCalibrate can't run at all (not
    # enough constraints to solve for intrinsics) — an unconditional hard
    # failure, unlike min_valid_views below which is a *quality* bar and
    # respects raise_on_poor_quality so a caller can still inspect why.
    absolute_min_views = 3
    if len(obj_points) < absolute_min_views:
        raise CalibrationQualityError(
            f"Only {len(obj_points)} valid stereo views found (need >= {absolute_min_views} "
            "just to run calibration at all). See the diagnostics report for per-pair "
            "rejection reasons.",
            report=report,
        )

    # Seed intrinsics from a mono calibration on the accepted views only (more robust
    # than letting stereoCalibrate guess from scratch), then refine jointly.
    rms_l, K1, D1, _, _ = cv2.calibrateCamera(obj_points, img_points_l, image_size, None, None)
    rms_r, K2, D2, _, _ = cv2.calibrateCamera(obj_points, img_points_r, image_size, None, None)

    flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_l,
        img_points_r,
        K1,
        D1,
        K2,
        D2,
        image_size,
        criteria=criteria,
        flags=flags,
    )

    per_view_errors = _stereo_reprojection_errors(objp, img_points_l, img_points_r, K1, D1, K2, D2, R, T)
    used_diags = report.used_pairs
    for diag, (err_l, err_r) in zip(used_diags, per_view_errors):
        diag.left_reproj_error_px = err_l
        diag.right_reproj_error_px = err_r

    cam1 = CameraCalibration(K1, D1, image_size, rms_l)
    cam2 = CameraCalibration(K2, D2, image_size, rms_r)

    quality_issues: List[str] = []
    if len(used_diags) < min_valid_views:
        quality_issues.append(f"only {len(used_diags)} valid pairs used (need >= {min_valid_views})")
    if not np.isfinite(rms) or rms > max_rms_error:
        quality_issues.append(f"stereo reprojection RMS {rms:.4f}px exceeds max_rms_error={max_rms_error}px")
    outliers = [
        d for d in used_diags
        if not (np.isfinite(d.left_reproj_error_px) and np.isfinite(d.right_reproj_error_px))
        or max(d.left_reproj_error_px, d.right_reproj_error_px) > max_pair_reproj_error
    ]
    if outliers:
        labels = [d.numeric_id if d.numeric_id is not None else Path(d.left_path).name for d in outliers]
        quality_issues.append(
            f"{len(outliers)} pair(s) exceed max_pair_reproj_error={max_pair_reproj_error}px: {labels}"
        )

    calib = StereoCalibration(
        cam1=cam1, cam2=cam2, R=R, T=T, E=E, F=F, rms_error=rms,
        quality_ok=not quality_issues, quality_issues=quality_issues, report=report,
    )

    if quality_issues and raise_on_poor_quality:
        raise CalibrationQualityError(
            "Stereo calibration failed quality checks:\n- " + "\n- ".join(quality_issues),
            calibration=calib,
        )
    return calib
