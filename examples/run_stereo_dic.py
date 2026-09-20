"""Run stereo-DIC over an image sequence using a saved calibration.

Expected layout:
    images/left/0000.tif, 0001.tif, ...   (frame 0 = reference / undeformed)
    images/right/0000.tif, 0001.tif, ...

Usage:
    python3 examples/run_stereo_dic.py stereo_calib.json images/left images/right \\
        --pattern "*.tif" --subset-radius 15 --grid-step 10 --out results/
"""
import argparse
import os

import numpy as np

from pydic.calibration import StereoCalibration
from pydic.io_utils import load_sequence, save_points3d_csv
from pydic.pipeline import StereoDic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib_json")
    ap.add_argument("left_dir")
    ap.add_argument("right_dir")
    ap.add_argument("--pattern", default="*.tif")
    ap.add_argument("--subset-radius", type=int, default=15)
    ap.add_argument("--grid-step", type=int, default=10)
    ap.add_argument("--out", default="dic_results")
    args = ap.parse_args()

    calib = StereoCalibration.load(args.calib_json)
    left_paths = load_sequence(args.left_dir, args.pattern)
    right_paths = load_sequence(args.right_dir, args.pattern)
    print(f"{len(left_paths)} frames found")
    if len(left_paths) < 2:
        raise SystemExit("need at least a reference frame + one deformed frame")

    dic = StereoDic(calib, subset_radius=args.subset_radius, grid_step=args.grid_step)
    dic.set_reference(left_paths[0], right_paths[0])
    print(f"{len(dic.points1)} measurement points on the grid")

    frame_results = dic.run_sequence(left_paths[1:], right_paths[1:])

    os.makedirs(args.out, exist_ok=True)
    for t, fr in enumerate(frame_results, start=1):
        strain = dic.compute_strain(fr)
        disp = fr.displacement_3d
        save_points3d_csv(
            os.path.join(args.out, f"frame_{t:04d}.csv"),
            fr.points_ref_3d,
            {
                "dX": disp[:, 0], "dY": disp[:, 1], "dZ": disp[:, 2],
                "exx": strain.exx, "eyy": strain.eyy, "exy": strain.exy,
                "valid": fr.valid.astype(int),
            },
        )
        print(f"frame {t}: {fr.valid.mean()*100:.1f}% points valid, "
              f"max principal strain {np.nanmax(strain.major_principal):.4f}")

    print(f"wrote per-frame CSVs to {args.out}/")


if __name__ == "__main__":
    main()
