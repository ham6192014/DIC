"""pydic - subset-based 2D/stereo Digital Image Correlation.

High level entry points live in :mod:`pydic.pipeline`:
    - Dic2D        : single camera, planar displacement/strain
    - StereoDic    : two calibrated cameras, 3D displacement/strain

Everything else is exposed for building custom pipelines / a GUI on top.
"""

from .calibration import (
    CameraCalibration,
    StereoCalibration,
    calibrate_single_camera,
    calibrate_stereo,
)
from .subset import Subset
from .correlation import ImageInterpolator, icgn_correlate, initial_guess_template_match
from .tracking2d import track_reliability_guided, DicPointResult
from .strain import compute_strain_2d, compute_strain_3d_surface
from .stereo import stereo_match_reference, triangulate_points
from .pipeline import Dic2D, StereoDic

__all__ = [
    "CameraCalibration",
    "StereoCalibration",
    "calibrate_single_camera",
    "calibrate_stereo",
    "Subset",
    "ImageInterpolator",
    "icgn_correlate",
    "initial_guess_template_match",
    "track_reliability_guided",
    "DicPointResult",
    "compute_strain_2d",
    "compute_strain_3d_surface",
    "stereo_match_reference",
    "triangulate_points",
    "Dic2D",
    "StereoDic",
]
