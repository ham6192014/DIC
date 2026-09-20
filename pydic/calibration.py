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
) -> Tuple[bool, np.ndarray]:
    """Locate inner chessboard corners in a grayscale/BGR image.

    pattern_size is (n_cols, n_rows) of *inner* corners (squares - 1).
    Returns (found, corners) with corners shape (N, 1, 2) float32, subpixel refined.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return False, corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, refine_win, (-1, -1), criteria)
    return True, corners


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
