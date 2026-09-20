"""High-level Dic2D / StereoDic APIs tying the pieces together.

These are the classes a GUI should call into: they hide grid/neighbor
bookkeeping and expose plain arrays (points, displacements, strain) per
frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .calibration import StereoCalibration
from .correlation import initial_guess_template_match
from .fields import build_neighbors, generate_grid, nearest_point_index, polygon_to_mask
from .io_utils import ImageLike, load_gray
from .strain import StrainField, compute_strain_2d, compute_strain_3d_surface
from .stereo import StereoFrameResult, track_stereo_sequence
from .tracking2d import DicPointResult, track_sequence


@dataclass
class Dic2DFrameResult:
    points: np.ndarray  # (N, 2) reference coordinates
    u: np.ndarray
    v: np.ndarray
    zncc: np.ndarray
    valid: np.ndarray


class Dic2D:
    """Single-camera planar DIC: displacement + in-plane strain over a sequence."""

    def __init__(
        self,
        subset_radius: int = 15,
        grid_step: int = 10,
        max_iter: int = 40,
        tol: float = 1e-4,
        zncc_threshold: float = 0.5,
    ):
        self.subset_radius = subset_radius
        self.grid_step = grid_step
        self.track_kwargs = dict(max_iter=max_iter, tol=tol, zncc_threshold=zncc_threshold)

        self.ref_image: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None
        self.neighbors: Optional[List[List[int]]] = None
        self._results: Optional[List[List[DicPointResult]]] = None

    def set_reference(self, image: ImageLike, roi_polygon: Optional[Sequence[Tuple[float, float]]] = None) -> None:
        self.ref_image = load_gray(image)
        mask = polygon_to_mask(self.ref_image.shape, roi_polygon) if roi_polygon else None
        self.points, index_of = generate_grid(self.ref_image.shape, self.grid_step, self.subset_radius, mask)
        self.neighbors = build_neighbors(index_of)
        if len(self.points) == 0:
            raise RuntimeError("no grid points generated: check ROI / subset_radius / grid_step")

    def run_sequence(
        self,
        images: Sequence[ImageLike],
        seed_xy: Optional[Tuple[float, float]] = None,
        seed_search_radius: int = 80,
    ) -> List[Dic2DFrameResult]:
        """Track the whole grid across `images` (deformed frames, in order).

        seed_xy: approximate reference-image location of a point you can
        visually track between the reference and the first deformed image;
        used only to bootstrap a coarse initial guess when the first
        deformed frame already involves large motion. Leave as None for the
        common case where frame 1 is close to the reference.
        """
        if self.ref_image is None:
            raise RuntimeError("call set_reference() first")
        frames = [load_gray(im) for im in images]

        first_seed_indices = None
        first_seed_p = None
        if seed_xy is not None and frames:
            idx = nearest_point_index(self.points, seed_xy)
            u0, v0, score = initial_guess_template_match(
                self.ref_image, frames[0], tuple(self.points[idx]), self.subset_radius, seed_search_radius
            )
            if score > 0.2:
                first_seed_indices = [idx]
                first_seed_p = [np.array([u0, 0, 0, v0, 0, 0])]

        raw = track_sequence(
            self.ref_image, frames, self.points, self.neighbors, self.subset_radius,
            first_frame_seed_indices=first_seed_indices, first_frame_seed_p=first_seed_p,
            **self.track_kwargs,
        )
        self._results = raw
        out = []
        for frame_res in raw:
            u = np.array([r.u for r in frame_res])
            v = np.array([r.v for r in frame_res])
            zncc = np.array([r.zncc for r in frame_res])
            valid = np.array([r.converged for r in frame_res])
            out.append(Dic2DFrameResult(self.points.copy(), u, v, zncc, valid))
        return out

    def compute_strain(
        self, frame_result: Dic2DFrameResult, window_hops: int = 2, kind: str = "green-lagrange"
    ) -> StrainField:
        return compute_strain_2d(
            self.points, frame_result.u, frame_result.v, frame_result.valid,
            self.neighbors, window_hops=window_hops, kind=kind,
        )


class StereoDic:
    """Two-camera (stereo) DIC: 3D displacement + surface strain over a sequence."""

    def __init__(
        self,
        calib: StereoCalibration,
        subset_radius: int = 15,
        grid_step: int = 10,
        max_iter: int = 40,
        tol: float = 1e-4,
        zncc_threshold: float = 0.5,
        epipolar_band: float = 6.0,
        disparity_range: Tuple[float, float] = (-150.0, 150.0),
    ):
        self.calib = calib
        self.subset_radius = subset_radius
        self.grid_step = grid_step
        self.track_kwargs = dict(max_iter=max_iter, tol=tol, zncc_threshold=zncc_threshold)
        self.stereo_match_kwargs = dict(
            epipolar_band=epipolar_band,
            disparity_range=disparity_range,
            max_iter=max_iter,
            tol=tol,
        )

        self.ref_img1: Optional[np.ndarray] = None
        self.ref_img2: Optional[np.ndarray] = None
        self.points1: Optional[np.ndarray] = None
        self.points2: Optional[np.ndarray] = None
        self.stereo_valid: Optional[np.ndarray] = None
        self.neighbors: Optional[List[List[int]]] = None

    def set_reference(
        self,
        image1: ImageLike,
        image2: ImageLike,
        roi_polygon: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> None:
        self.ref_img1 = load_gray(image1)
        self.ref_img2 = load_gray(image2)
        mask = polygon_to_mask(self.ref_img1.shape, roi_polygon) if roi_polygon else None
        self.points1, index_of = generate_grid(self.ref_img1.shape, self.grid_step, self.subset_radius, mask)
        self.neighbors = build_neighbors(index_of)
        if len(self.points1) == 0:
            raise RuntimeError("no grid points generated: check ROI / subset_radius / grid_step")

    def run_sequence(
        self, images1: Sequence[ImageLike], images2: Sequence[ImageLike]
    ) -> List[StereoFrameResult]:
        if self.ref_img1 is None:
            raise RuntimeError("call set_reference() first")
        frames1 = [load_gray(im) for im in images1]
        frames2 = [load_gray(im) for im in images2]

        self.points2, self.stereo_valid, frame_results = track_stereo_sequence(
            self.calib, self.ref_img1, self.ref_img2, frames1, frames2,
            self.points1, self.neighbors, subset_radius=self.subset_radius,
            stereo_match_kwargs=self.stereo_match_kwargs, track_kwargs=self.track_kwargs,
        )
        return frame_results

    def compute_strain(
        self, frame_result: StereoFrameResult, window_hops: int = 2, kind: str = "green-lagrange"
    ) -> StrainField:
        return compute_strain_3d_surface(
            frame_result.points_ref_3d, frame_result.points_cur_3d, frame_result.valid,
            self.neighbors, window_hops=window_hops, kind=kind,
        )
