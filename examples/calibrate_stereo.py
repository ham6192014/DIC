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
import sys
from pathlib import Path

from pydic.calibration import CalibrationQualityError, calibrate_stereo
from pydic.io_utils import load_sequence


def _print_report(report) -> None:
    if report is None:
        return
    if report.unmatched_left:
        print(f"  left images with no matching right-image number: {report.unmatched_left}")
    if report.unmatched_right:
        print(f"  right images with no matching left-image number: {report.unmatched_right}")
    if report.duplicate_left_ids:
        print(f"  left images sharing the same number: {report.duplicate_left_ids}")
    if report.duplicate_right_ids:
        print(f"  right images sharing the same number: {report.duplicate_right_ids}")
    print(f"  {len(report.used_pairs)} pair(s) used, {len(report.rejected_pairs)} rejected:")
    for p in report.pairs:
        tag = "id" if p.numeric_id is not None else "pair"
        label = p.numeric_id if p.numeric_id is not None else f"{Path(p.left_path).name}/{Path(p.right_path).name}"
        if p.used and p.left_reproj_error_px is not None:
            print(
                f"    [{tag} {label}] OK  left_err={p.left_reproj_error_px:.4f}px "
                f"right_err={p.right_reproj_error_px:.4f}px "
                f"(flipped: left={p.left_flipped} right={p.right_flipped})"
            )
        elif p.used:
            print(f"    [{tag} {label}] detected but calibration did not run far enough to compute reprojection error")
        else:
            print(f"    [{tag} {label}] REJECTED: {p.reject_reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left_dir")
    ap.add_argument("right_dir")
    ap.add_argument("--pattern", default="*.png", help="glob pattern for image files")
    ap.add_argument("--cols", type=int, required=True, help="inner corners per row")
    ap.add_argument("--rows", type=int, required=True, help="inner corners per column")
    ap.add_argument("--square-size", type=float, required=True, help="chessboard square size (mm)")
    ap.add_argument("--out", default="stereo_calib.json")
    ap.add_argument("--max-rms-error", type=float, default=1.0, help="quality gate: max stereo RMS (px)")
    ap.add_argument("--max-pair-error", type=float, default=2.0, help="quality gate: max per-pair reprojection error (px)")
    ap.add_argument(
        "--force", action="store_true",
        help="save the calibration even if it fails the quality gate (not recommended)",
    )
    args = ap.parse_args()

    left_paths = load_sequence(args.left_dir, args.pattern)
    right_paths = load_sequence(args.right_dir, args.pattern)
    print(f"found {len(left_paths)} left / {len(right_paths)} right images")

    try:
        calib = calibrate_stereo(
            left_paths, right_paths, (args.cols, args.rows), args.square_size,
            max_rms_error=args.max_rms_error, max_pair_reproj_error=args.max_pair_error,
            raise_on_poor_quality=not args.force,
        )
    except CalibrationQualityError as e:
        print(f"CALIBRATION FAILED QUALITY CHECKS: {e}")
        _print_report(e.report)
        print("Re-run with --force to save it anyway (not recommended for 3D DIC).")
        sys.exit(1)

    print(f"stereo RMS reprojection error: {calib.rms_error:.4f} px")
    print(f"cam1 RMS: {calib.cam1.rms_error:.4f} px, cam2 RMS: {calib.cam2.rms_error:.4f} px")
    print(f"baseline T: {calib.T.ravel()} mm (magnitude {calib.baseline_mm:.4f} mm)")
    _print_report(calib.report)

    if not calib.quality_ok:
        print("WARNING: saving a calibration that failed quality checks (--force was used).")

    calib.save(args.out)
    print(f"saved calibration to {args.out}")


if __name__ == "__main__":
    main()
