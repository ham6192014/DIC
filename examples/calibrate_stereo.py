"""Calibrate a two-camera rig from chessboard image pairs.

Expected layout:
    calib_images/left/*.png   (or .jpg/.tif/...)
    calib_images/right/*.png

Each left/right pair at the same index must show the SAME chessboard pose,
captured simultaneously by camera 1 (left) and camera 2 (right).

Usage:
    python3 examples/calibrate_stereo.py calib_images/left calib_images/right \\
        --cols 9 --rows 6 --square-size 20.0 --out stereo_calib.json
"""
import argparse

from pydic.calibration import calibrate_stereo
from pydic.io_utils import load_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left_dir")
    ap.add_argument("right_dir")
    ap.add_argument("--pattern", default="*.png", help="glob pattern for image files")
    ap.add_argument("--cols", type=int, required=True, help="inner corners per row")
    ap.add_argument("--rows", type=int, required=True, help="inner corners per column")
    ap.add_argument("--square-size", type=float, required=True, help="chessboard square size (mm)")
    ap.add_argument("--out", default="stereo_calib.json")
    args = ap.parse_args()

    left_paths = load_sequence(args.left_dir, args.pattern)
    right_paths = load_sequence(args.right_dir, args.pattern)
    print(f"found {len(left_paths)} left / {len(right_paths)} right images")

    calib = calibrate_stereo(left_paths, right_paths, (args.cols, args.rows), args.square_size)
    print(f"stereo RMS reprojection error: {calib.rms_error:.4f} px")
    print(f"cam1 RMS: {calib.cam1.rms_error:.4f} px, cam2 RMS: {calib.cam2.rms_error:.4f} px")
    print(f"baseline: {calib.T.ravel()} mm")

    calib.save(args.out)
    print(f"saved calibration to {args.out}")


if __name__ == "__main__":
    main()
