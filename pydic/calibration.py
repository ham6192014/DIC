"""Chessboard-based single and stereo camera calibration."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def find_chessboard_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    refine_win: Tuple[int, int] = (11, 11),
    max_detect_dim: int = 1600,
) -> Tuple[bool, np.ndarray]:
    """Locate inner chessboard corners in a grayscale/BGR image.

    pattern_size is (n_cols, n_rows) of *inner* corners (squares - 1).
    Returns (found, corners) with corners shape (N, 1, 2) float32, subpixel
    refined on the full-resolution image.

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
        return False, None

    corners = (corners / scale).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, refine_win, (-1, -1), criteria)
    corners = _canonicalize_corner_order(gray, corners, pattern_size)
    return True, corners


def _canonicalize_corner_order(
    gray: np.ndarray, corners: np.ndarray, pattern_size: Tuple[int, int]
) -> np.ndarray:
    """Fix the classic checkerboard 180-degree labeling ambiguity.

    A plain checkerboard looks identical rotated 180 degrees, so the
    detector's choice of "corner index 0" is only fixed relative to how the
    board happened to appear in *that* image — nothing stops it from picking
    the diagonally opposite physical corner in another image of the same
    pose (this is exactly what caused a real stereo calibration to fail with
    a huge RMS despite both individual cameras calibrating fine: reprojection
    error within one camera doesn't care about labeling, but stereoCalibrate
    needs corner k in the left image and corner k in the right image to be
    the *same physical point*, for every pose).

    Fixed with a property that doesn't depend on viewpoint at all: the real
    checkerboard square diagonally adjacent to corner 0 is either black or
    white, a fact about the physical board, not the photo. Sampling it and
    enforcing a single reference color (black) for every image, of every
    camera, of every pose makes the labeling agree everywhere automatically.
    """
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)
    c00, c01, c10, c11 = grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]
    center = (c00 + c01 + c10 + c11) / 4.0
    x, y = int(round(center[0])), int(round(center[1]))
    h, w = gray.shape
    x0, x1 = max(0, x - 3), min(w, x + 4)
    y0, y1 = max(0, y - 3), min(h, y + 4)
    patch = gray[y0:y1, x0:x1]
    is_dark = patch.size > 0 and float(patch.mean()) < 128.0
    if not is_dark:
        grid = grid[::-1, ::-1, :]
    return grid.reshape(rows * cols, 1, 2).astype(np.float32)


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
            results.append({"path": path, "found": False, "image": None, "corners": None})
            continue
        found, corners = find_chessboard_corners(img, pattern_size)
        results.append({"path": path, "found": found, "image": img, "corners": corners})
    return results


def _object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


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
        found, corners = find_chessboard_corners(img, pattern_size)
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

    def __post_init__(self):
        if self.P1 is None:
            self.P1 = self.cam1.camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
        if self.P2 is None:
            self.P2 = self.cam2.camera_matrix @ np.hstack([self.R, self.T.reshape(3, 1)])

    def save(self, path: str) -> None:
        d = {
            "cam1": self.cam1.to_dict(),
            "cam2": self.cam2.to_dict(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "E": self.E.tolist(),
            "F": self.F.tolist(),
            "rms_error": self.rms_error,
        }
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def load(path: str) -> "StereoCalibration":
        d = json.loads(Path(path).read_text())
        return StereoCalibration(
            cam1=CameraCalibration.from_dict(d["cam1"]),
            cam2=CameraCalibration.from_dict(d["cam2"]),
            R=np.array(d["R"], dtype=np.float64),
            T=np.array(d["T"], dtype=np.float64),
            E=np.array(d["E"], dtype=np.float64),
            F=np.array(d["F"], dtype=np.float64),
            rms_error=d["rms_error"],
        )


def calibrate_stereo(
    left_image_paths: Sequence[str],
    right_image_paths: Sequence[str],
    pattern_size: Tuple[int, int],
    square_size: float,
    min_valid_views: int = 5,
    fix_intrinsics: bool = True,
) -> StereoCalibration:
    """Calibrate a stereo pair from matched lists of chessboard image paths.

    left_image_paths[i] and right_image_paths[i] must be the *same* pose of the
    board, captured simultaneously by camera 1 (left) and camera 2 (right).
    """
    if len(left_image_paths) != len(right_image_paths):
        raise ValueError("left/right image lists must have the same length")

    objp = _object_points(pattern_size, square_size)
    obj_points, img_points_l, img_points_r = [], [], []
    image_size = None
    for lp, rp in zip(left_image_paths, right_image_paths):
        img_l = cv2.imread(str(lp))
        img_r = cv2.imread(str(rp))
        if img_l is None or img_r is None:
            continue
        if image_size is None:
            image_size = (img_l.shape[1], img_l.shape[0])
        found_l, corners_l = find_chessboard_corners(img_l, pattern_size)
        found_r, corners_r = find_chessboard_corners(img_r, pattern_size)
        if not (found_l and found_r):
            continue
        obj_points.append(objp)
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)

    if len(obj_points) < min_valid_views:
        raise RuntimeError(
            f"Only {len(obj_points)} valid stereo views found (need >= {min_valid_views})."
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

    cam1 = CameraCalibration(K1, D1, image_size, rms_l)
    cam2 = CameraCalibration(K2, D2, image_size, rms_r)
    return StereoCalibration(cam1=cam1, cam2=cam2, R=R, T=T, E=E, F=F, rms_error=rms)
